# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Eager-mode kernel fusion via load-time module rewriting.

import inspect
import types

import torch
from torch import nn

from vllm import ir
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

logger = init_logger(__name__)

_QK_NORM_ROPE_HEAD_DIMS = (64, 128)


def _impl_available(op) -> bool:
    # True when a non-native, platform-supported impl is registered for ``op``.
    return any(
        provider != "native" and impl.supported for provider, impl in op.impls.items()
    )


def _fused_qk_norm_rope_impl_available() -> bool:
    # True when a non-native, platform-supported fused impl is registered.
    return _impl_available(ir.ops.fused_qk_norm_rope)


def _matches_qk_norm_rope(module: nn.Module) -> bool:
    """Structural match for attention modules.

    The unfused forward these modules run is::

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], -1)
        q = self.q_norm(q.view(..., head_dim)).view(q.shape)
        k = self.k_norm(k.view(..., head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)

    which is exactly what ``fused_qk_norm_rope`` fuses.
    """
    required = (
        "qkv_proj",
        "o_proj",
        "attn",
        "q_norm",
        "k_norm",
        "rotary_emb",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "q_size",
        "kv_size",
    )
    if not all(hasattr(module, name) for name in required):
        return False
    if not isinstance(module.q_norm, RMSNorm) or not isinstance(module.k_norm, RMSNorm):
        return False
    # Only the standard NeoX rotary embedding matches the kernel's layout;
    # subclasses (e.g. YaRN/linear-scaled) fall through to the native path.
    if type(module.rotary_emb) is not RotaryEmbedding:
        return False
    if module.head_dim not in _QK_NORM_ROPE_HEAD_DIMS:
        return False
    # The fused forward assumes the (positions, hidden_states) signature.
    try:
        params = list(inspect.signature(module.forward).parameters)
    except (ValueError, TypeError):
        return False
    return params[:2] == ["positions", "hidden_states"]


def _fused_qk_norm_rope_forward(
    self: nn.Module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    qkv = ir.ops.fused_qk_norm_rope.maybe_inplace(
        qkv,
        self.num_heads,
        self.num_kv_heads,
        self.num_kv_heads,
        self.head_dim,
        self.q_norm.variance_epsilon,
        self.q_norm.weight.data,
        self.k_norm.weight.data,
        self.rotary_emb.cos_sin_cache,
        self.rotary_emb.is_neox_style,
        positions,
    )
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


def apply_eager_fusions(model: nn.Module) -> None:
    _apply_qk_norm_rope_fusion(model)
    _apply_quant_fusion(model)


def _apply_qk_norm_rope_fusion(model: nn.Module) -> None:
    if not _fused_qk_norm_rope_impl_available():
        return

    fused = 0
    for name, module in model.named_modules():
        if _matches_qk_norm_rope(module):
            module.forward = types.MethodType(_fused_qk_norm_rope_forward, module)
            fused += 1
            logger.debug("Eager fusion: rebound %s.forward to fused_qk_norm_rope", name)

    if fused:
        logger.info_once(
            "Applied eager qk_norm_rope fusion to %d attention module(s)", fused
        )


def _fp8_static_input_key(linear: nn.Module):
    """Return the static per-tensor FP8 activation key a linear can consume.

    Returns ``None`` unless ``linear`` is an FP8 W8A8 linear whose scaled-MM
    kernel advertises static per-tensor activation quantization and carries a
    static ``input_scale``. Only this scheme matches the static fused kernels
    (``rms_norm_static_fp8_quant`` / ``silu_and_mul_quant``); dynamic-scale
    linears keep quantizing in-kernel.
    """
    quant_method = getattr(linear, "quant_method", None)
    fp8_linear = getattr(quant_method, "fp8_linear", None)
    if fp8_linear is None or not hasattr(fp8_linear, "input_quant_key"):
        return None
    key = fp8_linear.input_quant_key()
    if key != kFp8StaticTensorSym:
        return None
    if getattr(linear, "input_scale", None) is None:
        return None
    return key


def _fused_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    gate_up, _ = self.gate_up_proj(x)
    down_proj = self.down_proj
    scale = down_proj.input_scale
    data = ir.ops.silu_and_mul_quant(gate_up, scale)
    d = gate_up.shape[-1] // 2
    qact = QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=gate_up.dtype,
        orig_shape=torch.Size((*gate_up.shape[:-1], d)),
        quant_key=self._eager_down_input_key,
    )
    out, _ = down_proj(qact)
    return out


def _make_fused_norm_forward(norm: RMSNorm, consumer: nn.Module, key):
    weight = norm.weight
    epsilon = norm.variance_epsilon

    def forward(x: torch.Tensor, residual: torch.Tensor | None = None):
        scale = consumer.input_scale
        if residual is None:
            data = ir.ops.rms_norm_static_fp8_quant(x, weight.data, scale, epsilon)
            return QuantizedActivation(data, scale, x.dtype, x.shape, key)
        data, new_residual = ir.ops.fused_add_rms_norm_static_fp8_quant(
            x, residual, weight.data, scale, epsilon
        )
        return QuantizedActivation(data, scale, x.dtype, x.shape, key), new_residual

    return forward


def _matches_fused_mlp(module: nn.Module) -> bool:
    """SiLU-gated MLP whose down-projection consumes static FP8 activations."""
    if not all(hasattr(module, n) for n in ("gate_up_proj", "act_fn", "down_proj")):
        return False
    if not isinstance(module.act_fn, SiluAndMul):
        return False
    return _fp8_static_input_key(module.down_proj) is not None


def _apply_quant_fusion(model: nn.Module) -> None:
    rms_ready = _impl_available(ir.ops.rms_norm_static_fp8_quant) and _impl_available(
        ir.ops.fused_add_rms_norm_static_fp8_quant
    )
    act_ready = _impl_available(ir.ops.silu_and_mul_quant)
    if not (rms_ready or act_ready):
        return

    act_fused = 0
    norm_fused = 0
    for module in model.modules():
        if act_ready and _matches_fused_mlp(module):
            module._eager_down_input_key = _fp8_static_input_key(module.down_proj)
            module.forward = types.MethodType(_fused_mlp_forward, module)
            act_fused += 1
        if rms_ready:
            norm_fused += _fuse_norm_into_consumer(module)

    if act_fused:
        logger.info_once(
            "Applied eager silu_and_mul_quant fusion to %d MLP module(s)", act_fused
        )
    if norm_fused:
        logger.info_once(
            "Applied eager rms_norm_static_fp8_quant fusion to %d norm(s)", norm_fused
        )


def _fuse_norm_into_consumer(module: nn.Module) -> int:
    """Rewrite a decoder layer's pre-linear RMSNorms to emit quantized output.

    Pairs ``input_layernorm`` -> attention ``qkv_proj`` and
    ``post_attention_layernorm`` -> MLP ``gate_up_proj`` when the consuming
    linear takes static per-tensor FP8 activations. The normed activation is the
    first op each consumer runs, so a ``QuantizedActivation`` flows straight in.
    """
    pairs = (
        ("input_layernorm", getattr(module, "self_attn", None), "qkv_proj"),
        ("post_attention_layernorm", getattr(module, "mlp", None), "gate_up_proj"),
    )
    fused = 0
    for norm_name, attn_or_mlp, proj_name in pairs:
        norm = getattr(module, norm_name, None)
        consumer = getattr(attn_or_mlp, proj_name, None)
        if not isinstance(norm, RMSNorm) or consumer is None:
            continue
        key = _fp8_static_input_key(consumer)
        if key is None:
            continue
        norm.forward = _make_fused_norm_forward(  # type: ignore[method-assign]
            norm, consumer, key
        )
        fused += 1
    return fused
