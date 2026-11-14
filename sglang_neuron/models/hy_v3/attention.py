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
scale profile (``kv_scales.py``). Under ``--dcp-size D`` (an FP8 cache only) each rank keeps every D-th 128-token block of a sequence
(distributed.decode_context_parallel.DcpLayout) and the attention runs over the DCP group's shards:

- prefill: the chunk's K and V go to their owners' slots (the metadata's localized slot mapping), then nkilib's
  ``attention_kv_parallel_segmented_cte`` all-gathers the group's query heads, attends over this rank's shard of the
  prior and the chunk (round-robin blocks, ``kvp_group_size`` D) and merges the partial softmaxes;
- decode: ``attention_block_tkg`` with ``CP = D`` over the decode mask of this rank's share, the decoded token counted
  on its owner only (the runtime's ``dcp_kv_end`` / ``dcp_active_owner``).

These paths run on a NeuronCore only."""

from dataclasses import dataclass

import nki
import nki.collectives as ncc
import torch
from nkilib.core.attention.attention_kv_parallel_segmented_cte import attention_kv_parallel_segmented_cte
from nkilib.core.qkv.qkv_cte import qkv_cte
from nkilib.core.utils.common_types import NormType, QKNormConfig, QuantizationType
from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg
import nki.language as nl

import sglang_neuron.ops as NF
from sglang_neuron.models.hy_v3 import sparse
from sglang_neuron.ops.attention.attention_decode_mask import gen_attention_decode_mask
from sglang_neuron.backend.stack import resolve_full_mesh, wrap_nki
from sglang_neuron.ops.attention.attention_decode import dcp_collective_mode
from sglang_neuron.models.transformer.model import Attention
from sglang_neuron.ops.attention.attention_decode_mask import P_MAX
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader


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


@nki.jit
def _decode_fp8_dcp(X, W_qkv, q_norm, k_norm, norm_eps, cos, sin, active_blocks_table, K_cache, V_cache, attention_mask,
                    kv_cache_update_idx, softmax_scale, W_out, k_scale, v_scale, dcp_size, dcp_mesh, dcp_mode):
    cells = []
    for cell in dcp_mesh:  # NKI's tracer takes no list comprehension
        cells.append(list(cell))
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
        transposed_out=False, out_in_sb=False, softmax_scale=softmax_scale, fp8_packed=True, k_scale=k_scale,
        v_scale=v_scale, CP=dcp_size, CP_replica_group=ncc.ReplicaGroup(cells), CP_collective_mode=dcp_mode,
    )


@nki.jit
def _prefill_kv_parallel(q, k_cache, v_cache, block_tables, kvp_q_offset, kvp_rank_id, valid_num_prior_tokens, k_scale,
                         v_scale, scale, block_size, seg_size, dcp_size, dcp_mesh):
    cells = []
    for cell in dcp_mesh:
        cells.append(list(cell))
    return attention_kv_parallel_segmented_cte(
        q=q, k_cache=k_cache, v_cache=v_cache, block_tables=block_tables, kvp_q_offset=kvp_q_offset,
        replica_groups=ncc.ReplicaGroup(cells), group_size=dcp_size, block_size=block_size, seg_size=seg_size,
        scale=scale, global_q_offset=0, kvp_rank_id=kvp_rank_id, kvp_group_size=dcp_size, apc_mode=True,
        valid_num_prior_tokens=valid_num_prior_tokens, fp8_packed=True, k_scale=k_scale, v_scale=v_scale,
    )


@dataclass
class DcpDecodeInputs:
    """A DCP decode step's kernel inputs, shared by every layer: ``table`` [B, blocks] int32 virtual pages, ``update_idx``
    [B, 1] uint32 this rank's slot of each row's token (the reserved slot where another rank keeps it) and ``mask`` the
    group's decode mask (models/gpt_oss/attention_masks.decode_attention_masks). For sparse decode (sparse.py):
    ``owner`` [B] the rows whose token this rank keeps, ``local_prior`` [B] the prior tokens it keeps, ``positions``
    [B], ``n_keep`` the pages a sparse step keeps (``table``'s width: dense), and the current selection
    (``sparse_table``, ``sparse_mask``), which a selecting layer sets for the layers after it."""

    table: torch.Tensor
    update_idx: torch.Tensor
    mask: torch.Tensor
    owner: torch.Tensor = None
    local_prior: torch.Tensor = None
    positions: torch.Tensor = None
    n_keep: int = 0
    sparse_table: torch.Tensor = None
    sparse_mask: torch.Tensor = None


def dcp_group():
    """This rank's DCP group, or None without decode context parallelism."""
    group = groups.get_neuron_dcp_group()
    return group if group is not None and group.world_size > 1 else None


class Fp8Attention(Attention):
    """The dense transformer's attention over the FP8 pool; ``set_kv_scales`` gives it its layer's scales."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        if self.q_norm is None or self.qkv_bias is not None:
            raise ValueError("Hy3's FP8 KV attention runs with the QK-norm and without QKV biases")
        self.k_scale_float = self.v_scale_float = 1.0
        self.dcp_group = dcp_group()
        self.dcp_size = 1 if self.dcp_group is None else self.dcp_group.world_size
        if self.dcp_size > 1 and self.num_kv_heads != 1:
            raise ValueError(f"Hy3's DCP attention runs one KV head per rank, not {self.num_kv_heads}")
        self.sparse = (sparse.SparseState(config.hidden_size, self.num_heads, self.head_dim, config.rms_norm_eps,
                                          config.rope_theta)
                       if sparse.enabled() and self.dcp_size > 1 else None)
        if self.sparse is not None:
            rows = self.num_heads * self.head_dim
            set_weight_loader(self.sparse.wq, SafetensorsWeightLoader(
                transform=lambda slices, rank: slices[0][rank * rows:(rank + 1) * rows].t()))

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
        q = q.view(tokens, self.num_heads, self.head_dim).transpose(0, 1)
        if self.dcp_size > 1:
            return self._prefill_dcp(q, k_cache, v_cache, md, page, tokens)
        out = NF.segmented_attention(
            q, k_cache=k_cache, v_cache=v_cache,
            block_tables=md["block_table_tensor"], prior_tokens=md["cached_seq_len"], block_size=page,
            kv_segment_size=md["kv_segment_size"], scale=self.scaling * self.k_scale_float, sliding_window=None,
            sink=None,
        )
        out = (out * self.v_scale_float).transpose(0, 1).reshape(tokens, -1) @ self.o_weight
        return neuron_collectives.all_reduce(self.attn_tp_group, out)

    def _mesh(self) -> tuple:
        return tuple(tuple(int(r) for r in cell) for cell in resolve_full_mesh(self.dcp_group))

    def _prefill_dcp(self, q, k_cache, v_cache, md, page, tokens):
        """The chunk's attention over the DCP group's shards: the chunk starts at global position ``cached`` (a whole
        number of virtual pages), whose first ``cached / D`` tokens this rank keeps."""
        size = self.dcp_size
        cached = md["cached_seq_len"].to(torch.int32).reshape(1, 1)
        table = md["block_table_tensor"].to(torch.int32)
        table = torch.where(table < 0, torch.zeros_like(table), table)
        if self.sparse is not None:
            self._refresh_prefill(k_cache, table[0], cached, md, page, tokens)
        out = wrap_nki(_prefill_kv_parallel)[get_logical_nc_config()](
            q, k_cache=k_cache, v_cache=v_cache, block_tables=table[:1], kvp_q_offset=cached,
            kvp_rank_id=self.dcp_group.rank_in_group_tensor.to(torch.int32).reshape(1, 1),
            valid_num_prior_tokens=torch.div(cached, size, rounding_mode="floor"),
            k_scale=self._scale_tensor(self.k_scale_float, P_MAX, q.device),
            v_scale=self._scale_tensor(self.v_scale_float, P_MAX, q.device),
            scale=self.scaling, block_size=page, seg_size=min(tokens, size * page), dcp_size=size,
            dcp_mesh=self._mesh(),
        )
        out = out.transpose(0, 1).reshape(tokens, -1) @ self.o_weight
        return neuron_collectives.all_reduce(self.attn_tp_group, out)

    def _refresh_prefill(self, k_cache, table, cached, md, page, tokens):
        """The summaries of this rank's pages the chunk wrote: local pages from cached / D on, filled up to this rank's
        share of the request's length so far (the chunk's last real position + 1)."""
        size = self.dcp_size
        rank = self.dcp_group.rank_in_group_tensor
        first = torch.div(cached.reshape(-1)[0].to(torch.int64), size * page, rounding_mode="floor")
        end = sparse.local_count(md["positions_end"], size, rank, page)
        idx = first + torch.arange(tokens // (size * page) + 1, device=table.device, dtype=torch.int64)
        valid = torch.clamp(end - idx * page, min=0, max=page)
        width = table.shape[0]
        blocks = torch.where((idx < width) & (valid > 0), table.to(torch.int64)[torch.clamp(idx, max=width - 1)],
                             torch.full_like(idx, -1))
        self.sparse.refresh(k_cache, blocks, valid)

    def _sparse_inputs(self, x, inputs: DcpDecodeInputs) -> tuple:
        """(table, mask) of this layer's decode: the step's selection, made here on a selecting layer."""
        if self.sparse is None or inputs.n_keep >= inputs.table.shape[1]:
            return inputs.table, inputs.mask
        if self.layer_idx % sparse.SELECT_EVERY == 0 or inputs.sparse_table is None:
            page = self.k_cache.shape[2] * 2
            bound = sparse.scores(self.sparse, x, inputs.positions, inputs.table, self.dcp_group)
            kept, prior = sparse.compact(bound, inputs.table, inputs.local_prior, page, inputs.n_keep)
            rows = kept.shape[0]
            inputs.sparse_table = kept
            inputs.sparse_mask = gen_attention_decode_mask(
                prior.reshape(1, rows).to(torch.float32), bs=rows, q_head=self.num_heads * self.dcp_size, s_active=1,
                s_prior=inputs.n_keep * page, block_len=page, active_gate=inputs.owner)
        return inputs.sparse_table, inputs.sparse_mask

    def _decode_dcp(self, x, cos, sin, inputs: DcpDecodeInputs):
        table, mask = self._sparse_inputs(x, inputs)
        out, k_written, _ = wrap_nki(_decode_fp8_dcp)[get_logical_nc_config()](
            x.unsqueeze(1), W_qkv=self.qkv_weight, q_norm=self.q_norm.weight.unsqueeze(0),
            k_norm=self.k_norm.weight.unsqueeze(0), norm_eps=self.norm_eps,
            cos=cos.t().unsqueeze(-1).contiguous(), sin=sin.t().unsqueeze(-1).contiguous(),
            active_blocks_table=table, K_cache=self.k_cache, V_cache=self.v_cache,
            attention_mask=mask, kv_cache_update_idx=inputs.update_idx,
            softmax_scale=self.scaling * self.k_scale_float, W_out=self.o_weight * self.v_scale_float,
            k_scale=self._scale_tensor(1.0 / self.k_scale_float, 1, x.device).reshape(1, 1),
            v_scale=self._scale_tensor(1.0 / self.v_scale_float, 1, x.device).reshape(1, 1),
            dcp_size=self.dcp_size, dcp_mesh=self._mesh(), dcp_mode=dcp_collective_mode(self.dcp_size),
        )
        if self.sparse is not None:
            # the page the row's token landed in, on its owner (the full table's local page)
            page = self.k_cache.shape[2] * 2
            n = inputs.local_prior.to(torch.int64)
            idx = torch.div(n, page, rounding_mode="floor").view(-1, 1)
            blocks = inputs.table.to(torch.int64).gather(1, torch.clamp(idx, max=inputs.table.shape[1] - 1)).view(-1)
            blocks = torch.where(inputs.owner.view(-1) > 0, blocks, torch.full_like(blocks, -1))
            self.sparse.add_token(k_written, blocks, n - idx.view(-1) * page)
        return neuron_collectives.all_reduce(self.attn_tp_group, out)

    def decode(self, x, cos, sin, inputs):
        if isinstance(inputs, DcpDecodeInputs):
            return self._decode_dcp(x, cos, sin, inputs)
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
