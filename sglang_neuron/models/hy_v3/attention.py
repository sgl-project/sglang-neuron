# SPDX-License-Identifier: Apache-2.0
"""Hy3's attention over an FP8 KV cache (``--kv-cache-dtype fp8_e4m3``).

The dense transformer's attention (models/transformer/model.Attention) reads and writes a bf16 cache. Over an FP8 pool
(K packed as [pages, kv_heads, page // 2, d, 2], V [pages, kv_heads, page, d]; mem_cache/memory_pool.fp8_packed_k) the
same kernels take the FP8 arguments instead:

- prefill: nkilib's ``qkv_cte`` projects the chunk, RMSNorms Q and K per head before RoPE, rotates them and writes K
  and V quantized (``fp8(clamp(x / scale))``, the scale it gets being the dequantization scale) into the packed cache;
  the segmented attention reads it, with the K scale folded into the softmax scale and the V scale applied to its
  output;
- decode: nkilib's ``attention_block_tkg`` does the same per layer, quantizing by multiplying (it gets the inverted
  scales), with the K scale folded into the softmax scale and the V scale into the output projection.

The per-layer scales are dequantization scales (``x ~= fp8 * scale``) for the platform's FP8 range, read from a KV
scale profile (``kv_scales.py``). These paths run on a NeuronCore only."""

import nki
import torch
from nkilib.core.qkv.qkv_cte import qkv_cte
from nkilib.core.utils.common_types import NormType, QKNormConfig, QuantizationType
from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg
import nki.language as nl

import sglang_neuron.ops as NF
from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.models.transformer.model import Attention
from sglang_neuron.ops.attention.attention_decode_mask import P_MAX
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from sglang_neuron.utils.hardware_config import get_logical_nc_config


@nki.jit
def _qkv_fp8(hidden, weights, cos, sin, q_norm, k_norm, norm_eps, d_head, num_q_heads, num_kv_heads, k_cache, v_cache,
             k_scale, v_scale, fp8_max, fp8_min, kv_dtype, block_size, slot_mapping):
    qk_norm = QKNormConfig(q_norm=NormType.RMS_NORM, k_norm=NormType.RMS_NORM, eps=norm_eps,
                           q_gamma_norm_weights=q_norm, k_gamma_norm_weights=k_norm)
    return qkv_cte(input=hidden, fused_qkv_weights=weights, bias=None, fused_rope=True, cos_cache=cos, sin_cache=sin,
                   d_head=d_head, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, k_cache=k_cache, v_cache=v_cache,
                   k_scale=k_scale, v_scale=v_scale, fp8_max=fp8_max, fp8_min=fp8_min, kv_dtype=kv_dtype,
                   use_block_kv=True, fp8_packed=True, block_size=block_size, slot_mapping=slot_mapping,
                   qk_norm_pre_rope=qk_norm)


@nki.jit
def _decode_fp8(X, W_qkv, q_norm, k_norm, norm_eps, cos, sin, active_blocks_table, K_cache, V_cache, attention_mask,
                kv_cache_update_idx, pos_ids, max_context_len, softmax_scale, W_out, k_scale, v_scale):
    return attention_block_tkg(
        X=X, X_hidden_dim_actual=None, rmsnorm_X_enabled=False, rmsnorm_X_eps=None, rmsnorm_X_gamma=None,
        W_qkv=W_qkv, bias_qkv=None, quantization_type_qkv=QuantizationType.NONE, weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None, rmsnorm_QK_pre_rope_enabled=True, rmsnorm_QK_pre_rope_eps=norm_eps,
        rmsnorm_QK_pre_rope_W_Q=q_norm, rmsnorm_QK_pre_rope_W_K=k_norm, cos=cos, sin=sin, rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False, rmsnorm_QK_post_rope_eps=0.0, rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None, K_cache_transposed=False, active_blocks_table=active_blocks_table,
        K_cache=K_cache, V_cache=V_cache, attention_mask=attention_mask, sink=None, update_cache=True,
        kv_cache_update_idx=kv_cache_update_idx, W_out=W_out, bias_out=None,
        quantization_type_out=QuantizationType.NONE, weight_dequant_scale_out=None, input_dequant_scale_out=None,
        transposed_out=False, out_in_sb=False, softmax_scale=softmax_scale, pos_ids=pos_ids,
        max_context_len=max_context_len, fp8_packed=True, k_scale=k_scale, v_scale=v_scale,
    )


class Fp8Attention(Attention):
    """The dense transformer's attention over the FP8 pool; ``set_kv_scales`` gives it its layer's scales."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        if self.q_norm is None or self.qkv_bias is not None:
            raise ValueError("Hy3's FP8 KV attention runs with the QK-norm and without QKV biases")
        self.k_scale_float = self.v_scale_float = 1.0

    def set_kv_scales(self, k_scale: float, v_scale: float) -> None:
        self.k_scale_float, self.v_scale_float = float(k_scale), float(v_scale)

    def _scale_tensor(self, value: float, rows: int, device) -> torch.Tensor:
        return torch.full((rows, 1), value, dtype=torch.float32, device=device)

    def prefill(self, x, cos, sin, md):
        page = md["block_size"]
        pages = self.v_cache.shape[0]
        tokens = x.shape[0]
        slots = md["slot_mapping"]
        slots = torch.where((slots < 0) | (slots >= pages * page), torch.zeros_like(slots), slots)
        q, k_cache, v_cache = wrap_nki(_qkv_fp8)[get_logical_nc_config()](
            x.unsqueeze(0), weights=self.qkv_weight,
            cos=torch.cat([cos, cos], dim=-1).unsqueeze(0), sin=torch.cat([sin, sin], dim=-1).unsqueeze(0),
            q_norm=self.q_norm.weight.unsqueeze(0), k_norm=self.k_norm.weight.unsqueeze(0), norm_eps=self.norm_eps,
            d_head=self.head_dim, num_q_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            k_cache=self.k_cache, v_cache=self.v_cache,
            # the prefill kernel quantizes by dividing by the scale it gets: the dequantization scale
            k_scale=self._scale_tensor(self.k_scale_float, P_MAX, x.device),
            v_scale=self._scale_tensor(self.v_scale_float, P_MAX, x.device),
            fp8_max=FP8_CLAMP_MAX, fp8_min=-FP8_CLAMP_MAX, kv_dtype=nl.float8_e4m3fn, block_size=page,
            slot_mapping=slots.to(torch.int32).reshape(1, tokens),
        )
        out = NF.segmented_attention(
            q.view(tokens, self.num_heads, self.head_dim).transpose(0, 1), k_cache=k_cache, v_cache=v_cache,
            block_tables=md["block_table_tensor"], prior_tokens=md["cached_seq_len"], block_size=page,
            kv_segment_size=md["kv_segment_size"], scale=self.scaling * self.k_scale_float, sliding_window=None,
            sink=None,
        )
        out = (out * self.v_scale_float).transpose(0, 1).reshape(tokens, -1) @ self.o_weight
        return neuron_collectives.all_reduce(self.attn_tp_group, out)

    def decode(self, x, cos, sin, inputs):
        out, _, _ = wrap_nki(_decode_fp8)[get_logical_nc_config()](
            x.unsqueeze(1), W_qkv=self.qkv_weight, q_norm=self.q_norm.weight.unsqueeze(0),
            k_norm=self.k_norm.weight.unsqueeze(0), norm_eps=self.norm_eps,
            cos=cos.t().unsqueeze(-1).contiguous(), sin=sin.t().unsqueeze(-1).contiguous(),
            active_blocks_table=inputs.table, K_cache=self.k_cache, V_cache=self.v_cache,
            attention_mask=inputs.active_mask, kv_cache_update_idx=inputs.update_idx, pos_ids=inputs.pos_ids,
            max_context_len=inputs.max_context,
            # the decode kernel quantizes by multiplying: the inverted scales; dequantization folds into the softmax
            # scale (K) and the output projection (V)
            softmax_scale=self.scaling * self.k_scale_float,
            W_out=self.o_weight * self.v_scale_float,
            k_scale=self._scale_tensor(1.0 / self.k_scale_float, 1, x.device).reshape(1, 1),
            v_scale=self._scale_tensor(1.0 / self.v_scale_float, 1, x.device).reshape(1, 1),
        )
        return neuron_collectives.all_reduce(self.attn_tp_group, out)
