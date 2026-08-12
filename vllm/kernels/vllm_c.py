# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from vllm import ir
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

current_platform.import_kernels()

CUDA_ALIKE = current_platform.is_cuda_alike()
"""Most kernels in this file are supported on all CUDA-alike platforms."""
IS_ROCM = current_platform.is_rocm()
"""ROCm needs shape normalization before calling some vLLM C kernels."""
GPGPU_DEVICE = CUDA_ALIKE or current_platform.is_xpu()


rms_no_var_size = lambda x, weight, epsilon, variance_size=None: (
    variance_size is None and (weight is None or weight.dtype == x.dtype)
)
"""vLLM kernel requires no variance_size override and matching input/weight dtype."""


@ir.ops.rms_norm.register_impl(
    "vllm_c", supports_args=rms_no_var_size, supported=GPGPU_DEVICE
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    assert variance_size is None
    # ROCm's vLLM C RMSNorm kernel operates on contiguous 2D tensors.
    # Higher-rank callers still normalize over the last dimension, so flatten
    # all leading dims. reshape handles strided views from q/k/v splits.
    if IS_ROCM and (x.dim() > 2 or not x.is_contiguous()):
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])
        # empty_like preserves the strides of transposed inputs, but the
        # libtorch-stable kernel requires a contiguous output tensor.
        output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        torch.ops._C.rms_norm(output, x, weight, epsilon)
        return output.reshape(original_shape)

    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output


rms_add_no_var_size = lambda x, x_residual, weight, epsilon, variance_size=None: (
    variance_size is None and (weight is None or weight.dtype == x.dtype)
)
"""vLLM Kernel does not support variance_size parameter and requires
matching input/weight dtype."""


@ir.ops.fused_add_rms_norm.register_impl(
    "vllm_c",
    supports_args=rms_add_no_var_size,
    supported=GPGPU_DEVICE,
    inplace=True,
)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    assert variance_size is None
    if IS_ROCM and (not x.is_contiguous() or not x_residual.is_contiguous()):
        output, residual = ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
            x, x_residual, weight, epsilon
        )
        x.copy_(output)
        x_residual.copy_(residual)
        return x, x_residual

    # ROCm's vLLM C RMSNorm kernel operates on contiguous 2D tensors.
    # Higher-rank callers still normalize over the last dimension, so flatten
    # all leading dims.
    if IS_ROCM and x.dim() > 2:
        original_shape = x.shape
        x = x.view(-1, original_shape[-1])
        x_residual = x_residual.view(-1, original_shape[-1])
        torch.ops._C.fused_add_rms_norm(x, x_residual, weight, epsilon)
        return x.view(original_shape), x_residual.view(original_shape)

    torch.ops._C.fused_add_rms_norm(x, x_residual, weight, epsilon)
    return x, x_residual


FP8_DTYPE = torch.float8_e4m3fn

# Fused norm/act + quant kernels operate on 16-bit activations.
_fp16_bf16 = lambda dt: dt in (torch.float16, torch.bfloat16)


@ir.ops.rms_norm_static_fp8_quant.register_impl(
    "vllm_c",
    supports_args=lambda x, weight, scale, epsilon: _fp16_bf16(x.dtype),
    supported=current_platform.is_xpu(),
)
def rms_norm_static_fp8_quant(
    x: Tensor, weight: Tensor, scale: Tensor, epsilon: float
) -> Tensor:
    logger.info_once("Using fused XPU kernel: rms_norm_static_fp8_quant")
    out = torch.empty(x.shape, device=x.device, dtype=FP8_DTYPE)
    torch.ops._C.rms_norm_static_fp8_quant(out, x, weight, scale, epsilon)
    return out


@ir.ops.fused_add_rms_norm_static_fp8_quant.register_impl(
    "vllm_c",
    supports_args=lambda x, residual, weight, scale, epsilon: _fp16_bf16(x.dtype),
    supported=current_platform.is_xpu(),
)
def fused_add_rms_norm_static_fp8_quant(
    x: Tensor, residual: Tensor, weight: Tensor, scale: Tensor, epsilon: float
) -> tuple[Tensor, Tensor]:
    logger.info_once("Using fused XPU kernel: fused_add_rms_norm_static_fp8_quant")
    out = torch.empty(x.shape, device=x.device, dtype=FP8_DTYPE)
    torch.ops._C.fused_add_rms_norm_static_fp8_quant(
        out, x, residual, weight, scale, epsilon
    )
    return out, residual


@ir.ops.silu_and_mul_quant.register_impl(
    "vllm_c",
    supports_args=lambda x, scale: _fp16_bf16(x.dtype),
    supported=current_platform.is_xpu(),
)
def silu_and_mul_quant(x: Tensor, scale: Tensor) -> Tensor:
    logger.info_once("Using fused XPU kernel: silu_and_mul_quant")
    out = torch.empty(
        (*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=FP8_DTYPE
    )
    torch.ops._C.silu_and_mul_quant(out, x, scale)
    return out


@ir.ops.rms_norm_mxfp4_quant.register_impl(
    "vllm_c",
    supports_args=lambda x, weight, epsilon, group_size: x.shape[-1] % group_size == 0,
    supported=current_platform.is_xpu(),
)
def rms_norm_mxfp4_quant(
    x: Tensor, weight: Tensor, epsilon: float, group_size: int
) -> tuple[Tensor, Tensor]:
    logger.info_once("Using fused XPU kernel: rms_norm_mxfp4_quant")
    hidden = x.shape[-1]
    x2d = x.reshape(-1, hidden)
    packed = torch.empty(
        (x2d.shape[0], hidden // 2), device=x.device, dtype=torch.uint8
    )
    scale = torch.empty(
        (x2d.shape[0], hidden // group_size), device=x.device, dtype=torch.float32
    )
    torch.ops._C.rms_norm_mxfp4_quant(
        packed, x2d, weight, scale, epsilon, None, group_size
    )
    return (
        packed.reshape(*x.shape[:-1], hidden // 2),
        scale.reshape(*x.shape[:-1], hidden // group_size),
    )


@ir.ops.silu_and_mul_mxfp4_quant.register_impl(
    "vllm_c",
    supports_args=lambda x, group_size, epsilon=1e-10: (
        (x.shape[-1] // 2) % group_size == 0
    ),
    supported=current_platform.is_xpu(),
)
def silu_and_mul_mxfp4_quant(
    x: Tensor, group_size: int, epsilon: float = 1e-10
) -> tuple[Tensor, Tensor]:
    logger.info_once("Using fused XPU kernel: silu_and_mul_mxfp4_quant")
    d = x.shape[-1] // 2
    x2d = x.reshape(-1, x.shape[-1])
    packed = torch.empty((x2d.shape[0], d // 2), device=x.device, dtype=torch.uint8)
    scale = torch.empty(
        (x2d.shape[0], d // group_size), device=x.device, dtype=torch.float32
    )
    torch.ops._C.silu_and_mul_mxfp4_quant(packed, x2d, scale, group_size, epsilon)
    return (
        packed.reshape(*x.shape[:-1], d // 2),
        scale.reshape(*x.shape[:-1], d // group_size),
    )


def _qk_norm_rope_supported(
    xqkv: Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    epsilon: float,
    q_weight: Tensor,
    k_weight: Tensor,
    cos_sin_cache: Tensor,
    is_neox: bool,
    position_ids: Tensor,
) -> bool:
    return _fp16_bf16(xqkv.dtype) and head_dim in (64, 128)


@ir.ops.fused_qk_norm_rope.register_impl(
    "vllm_c",
    supports_args=_qk_norm_rope_supported,
    supported=current_platform.is_xpu(),
    inplace=True,
)
def fused_qk_norm_rope(
    xqkv: Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    epsilon: float,
    q_weight: Tensor,
    k_weight: Tensor,
    cos_sin_cache: Tensor,
    is_neox: bool,
    position_ids: Tensor,
) -> Tensor:
    logger.info_once("Using fused XPU kernel: fused_qk_norm_rope")
    torch.ops._C.fused_qk_norm_rope(
        xqkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        epsilon,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        position_ids,
    )
    return xqkv
