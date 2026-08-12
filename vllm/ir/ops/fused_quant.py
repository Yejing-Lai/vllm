# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import Tensor

from ..op import register_op

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

_MXFP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MXFP4_MAX = 6.0


def _rms_norm(x: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + epsilon)
    return xf * weight.to(torch.float32)


def _static_fp8_quant(x: Tensor, scale: Tensor) -> Tensor:
    q = x.to(torch.float32) / scale.to(torch.float32)
    return q.clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


def _mxfp4_quant(x: Tensor, group_size: int) -> tuple[Tensor, Tensor]:
    orig_shape = x.shape
    hidden = orig_shape[-1]
    xf = x.reshape(-1, hidden // group_size, group_size).to(torch.float32)

    absmax = xf.abs().amax(dim=-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2((absmax / _MXFP4_MAX).clamp(min=1e-30))))
    scale = torch.where(absmax == 0, torch.ones_like(scale), scale)

    q = xf / scale
    mags = torch.tensor(_MXFP4_MAGNITUDES, device=x.device, dtype=torch.float32)
    idx = (q.abs().unsqueeze(-1) - mags).abs().argmin(dim=-1)
    codes = idx.to(torch.uint8) | ((q < 0).to(torch.uint8) << 3)

    codes = codes.reshape(-1, hidden)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    packed = packed.reshape(*orig_shape[:-1], hidden // 2)
    scale = scale.reshape(*orig_shape[:-1], hidden // group_size)
    return packed, scale


@register_op
def rms_norm_static_fp8_quant(
    x: Tensor, weight: Tensor, scale: Tensor, epsilon: float
) -> Tensor:
    return _static_fp8_quant(_rms_norm(x, weight, epsilon), scale)


@register_op
def fused_add_rms_norm_static_fp8_quant(
    x: Tensor, residual: Tensor, weight: Tensor, scale: Tensor, epsilon: float
) -> tuple[Tensor, Tensor]:
    new_residual = (x.to(torch.float32) + residual.to(torch.float32)).to(residual.dtype)
    normed = _rms_norm(new_residual, weight, epsilon)
    return _static_fp8_quant(normed, scale), new_residual


@register_op
def silu_and_mul_quant(x: Tensor, scale: Tensor) -> Tensor:
    d = x.shape[-1] // 2
    activated = torch.nn.functional.silu(x[..., :d].to(torch.float32)) * x[..., d:].to(
        torch.float32
    )
    return _static_fp8_quant(activated, scale)


@register_op
def rms_norm_mxfp4_quant(
    x: Tensor, weight: Tensor, epsilon: float, group_size: int
) -> tuple[Tensor, Tensor]:
    return _mxfp4_quant(_rms_norm(x, weight, epsilon), group_size)


@register_op
def silu_and_mul_mxfp4_quant(
    x: Tensor, group_size: int, epsilon: float = 1e-10
) -> tuple[Tensor, Tensor]:
    d = x.shape[-1] // 2
    activated = torch.nn.functional.silu(x[..., :d].to(torch.float32)) * x[..., d:].to(
        torch.float32
    )
    return _mxfp4_quant(activated, group_size)


@rms_norm_static_fp8_quant.register_input_generator
def _rms_norm_static_fp8_quant_inputs(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, epsilon: float = 1e-6
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    weight = torch.randn(hidden_size, dtype=dtype)
    scale = torch.tensor(0.5, dtype=torch.float32)
    return x, weight, scale, epsilon


@fused_add_rms_norm_static_fp8_quant.register_input_generator
def _fused_add_rms_norm_static_fp8_quant_inputs(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, epsilon: float = 1e-6
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype)
    weight = torch.randn(hidden_size, dtype=dtype)
    scale = torch.tensor(0.5, dtype=torch.float32)
    return x, residual, weight, scale, epsilon


@silu_and_mul_quant.register_input_generator
def _silu_and_mul_quant_inputs(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, epsilon: float = 1e-6
) -> tuple:
    x = torch.randn(num_tokens, 2 * hidden_size, dtype=dtype)
    scale = torch.tensor(0.5, dtype=torch.float32)
    return x, scale


@rms_norm_mxfp4_quant.register_input_generator
def _rms_norm_mxfp4_quant_inputs(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    epsilon: float = 1e-6,
    group_size: int = 32,
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    weight = torch.randn(hidden_size, dtype=dtype)
    return x, weight, epsilon, group_size


@silu_and_mul_mxfp4_quant.register_input_generator
def _silu_and_mul_mxfp4_quant_inputs(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    epsilon: float = 1e-10,
    group_size: int = 32,
) -> tuple:
    x = torch.randn(num_tokens, 2 * hidden_size, dtype=dtype)
    return x, group_size, epsilon


@register_op(allow_inplace=True, activations=["xqkv"])
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
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim
    v_size = num_heads_v * head_dim
    q, k, v = xqkv.split([q_size, k_size, v_size], dim=-1)

    orig_dtype = xqkv.dtype
    q_by_head = _rms_norm(q.unflatten(-1, (num_heads_q, head_dim)), q_weight, epsilon)
    q = q_by_head.to(orig_dtype).flatten(-2)
    k_by_head = _rms_norm(k.unflatten(-1, (num_heads_k, head_dim)), k_weight, epsilon)
    k = k_by_head.to(orig_dtype).flatten(-2)

    rotary_dim = cos_sin_cache.shape[-1]
    q, k = RotaryEmbedding.forward_static(
        position_ids, q, k, head_dim, rotary_dim, cos_sin_cache, is_neox
    )
    return torch.cat([q, k, v], dim=-1)


@fused_qk_norm_rope.register_input_generator
def _fused_qk_norm_rope_inputs(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    epsilon: float = 1e-6,
    head_dim: int = 128,
    num_heads_q: int = 8,
    num_heads_k: int = 2,
) -> tuple:
    num_heads_v = num_heads_k
    total = (num_heads_q + num_heads_k + num_heads_v) * head_dim
    xqkv = torch.randn(num_tokens, total, dtype=dtype)
    q_weight = torch.randn(head_dim, dtype=dtype)
    k_weight = torch.randn(head_dim, dtype=dtype)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(num_tokens, dtype=torch.long)
    freqs = torch.outer(positions.float(), inv_freq)
    cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(dtype)
    return (
        xqkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        epsilon,
        q_weight,
        k_weight,
        cos_sin_cache,
        True,
        positions,
    )
