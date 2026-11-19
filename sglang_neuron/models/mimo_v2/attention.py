# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's attention on attention TP, one layer kind at a time (config.AttentionKind): its query heads shard over
the ranks, and with more ranks than KV heads each rank holds one KV head, replicated (4 full / 8 sliding KV heads: 16
/ 8 ranks a head at TP64). Every head runs at config.KV_HEAD_DIM channels (weights.head_layout).

- **Prefill**, one request's chunk: the QKV projection and the partial RoPE in the graph, the chunk's K/V into its
  pages (kv_write.py), then nkilib's segmented attention over the prior and the chunk, with the layer's window and
  sinks; the window kernel loads only the pages a window reaches.
- **Decode**: nkilib's ``attention_block_tkg`` runs the input RMSNorm, the QKV projection, RoPE and the attention
  over the paged prior with the sinks. Its in-kernel output projection and cache write take heads of at most 128
  channels, so it runs with ``W_out=None`` and ``update_cache=False``: the graph writes the new token's K/V it returns
  and projects the heads' first ``v_head_dim`` channels itself. The step's masks are built once per layer kind
  (``decode_masks``).

``SGLANG_NEURON_MIMO_FULL_DECODE=posid`` runs the full-attention layers' decode in the kernel's position mode instead
of over a pre-built mask (no mask of the graph's context to build each step); ``mask`` (the default) is the path the
sliding layers take and the one the layer kernels serve. Neither ends its loop early: nkilib's early exit
(``max_context_len``) is wrong past an 8K prior (sgl-project/sglang-neuron#4).

**FP8 KV cache** (``--kv-cache-dtype fp8_e4m3``): K and V pages in Trn2's e4m3 (at most 240), each value times the
layer's scale (``KV_SCALE_ENV``, 1.0 by default: the checkpoint carries no KV scales), both token-major
([page, 256]; the host declares ``KVLayout(fp8_k_packed=False)``), which is the layout nkilib's decode attention reads
FP8 heads of 256 in. The decode kernels quantize the new token's K/V themselves (``k_scale`` / ``v_scale``) and take
1 / k_scale in the softmax scale and 1 / v_scale in the output projection. nkilib's segmented prefill reads FP8 K only
packed (two tokens a bf16-sized slot), so the prefill packs the pages it reads (``pack_prior``): a sliding layer its
window's few pages, a full layer its whole table.

**DCP** (``--dcp-size D``, on one KV pool: ``--disable-hybrid-swa-memory``): the D consecutive ranks that replicate a KV
head keep its sequence in D-way block-cyclic shards (distributed.decode_context_parallel.DcpLayout), each rank writing
only the tokens it owns (the backend's localized slots).

- decode: ``attention_block_tkg`` with ``CP = D`` over every layer's shard (the runtime's per-row bounds and the decoded
  token's owner), the sliding layers' sinks as the group's (``group_sinks``: counted once in the merge);
- prefill, full layers (no sinks): nkilib's KV-parallel segmented attention over the shards (Hy3's);
- prefill, sliding layers: a chunk starts at a whole virtual page, so the 127 keys before it a window reaches are the
  last block of the previous virtual page, which the group's rank D - 1 keeps. Every rank takes that block from it (an
  all-gather over the group) and attends locally over it and the chunk's own K/V, sinks included: nkilib's KV-parallel
  kernel takes no sinks (decode_context_parallel._PREFILL_KERNEL_GAP)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from sglang.srt.distributed.parallel_state import get_attn_tp_group
from torch import nn

from sglang_neuron.models.mimo_v2 import kv_write, weights
from sglang_neuron.models.mimo_v2.config import KV_HEAD_DIM, AttentionKind, MiMoV2Config
from sglang_neuron.models.mimo_v2.rope import rotate
from sglang_neuron.models.transformer.config import DTYPE
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.neuron_utils import can_run_kernel
from sglang_neuron.utils.weight_loader import set_weight_loader

FULL_DECODE_ENV = "SGLANG_NEURON_MIMO_FULL_DECODE"
# the FP8 KV cache's scale: the cache holds clip(x * scale) in e4m3 (module docstring)
KV_SCALE_ENV = "SGLANG_NEURON_MIMO_KV_SCALE"
FP8_KV_MAX = 240.0
# the decode attention half in one kernel (kernels/attn_half.py: attention, K/V write, output projection): 1 (the
# default) or 0 for nkilib's attention kernel with the write and the projection in the graph
FUSED_ENV = "SGLANG_NEURON_MIMO_DECODE_ATTN_FUSED"


def fused_enabled() -> bool:
    value = os.environ.get(FUSED_ENV, "1").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{FUSED_ENV}={value!r}: set 1 (the decode attention half in one kernel) or 0")
    return value == "1"


def full_decode_mode() -> str:
    value = os.environ.get(FULL_DECODE_ENV, "mask").strip()
    if value not in ("mask", "posid"):
        raise ValueError(f"{FULL_DECODE_ENV}={value!r}: set mask (a pre-built mask, the default) or posid (the "
                         "kernel's position mode)")
    return value


def fp8_kv_cache() -> bool:
    """Whether the launch's KV cache is FP8 (--kv-cache-dtype fp8_e4m3)."""
    from sglang.srt.runtime_context import get_server_args

    return str(getattr(get_server_args(), "kv_cache_dtype", "auto") or "auto").startswith("fp8")


def kv_scale(environ=os.environ) -> float:
    value = environ.get(KV_SCALE_ENV, "1.0").strip()
    try:
        scale = float(value)
    except ValueError:
        scale = -1.0
    if not scale > 0:
        raise ValueError(f"{KV_SCALE_ENV}={value!r}: set a positive scale (the FP8 cache holds K and V times it)")
    return scale


def _fp8_rows_t(k_new: torch.Tensor, d: int, tokens: int) -> torch.Tensor:
    """A kernel's new K [d, tokens] as [tokens, d] rows (FP8 over an FP8 cache, transposed as it is)."""
    return k_new.reshape(d, tokens).t().contiguous()


def pack_prior(k_cache: torch.Tensor, v_cache: torch.Tensor, table: torch.Tensor, first_col=None, n_cols=None):
    """The pages a prefill reads from token-major FP8 caches as nkilib's segmented attention takes FP8 K: (K packed
    [n, kv_heads, page // 2, d, 2], V [n, kv_heads, page, d], the table [R, columns] int32 over them). Every column of
    ``table`` [R, columns], or for a sliding window each row's ``n_cols`` columns from ``first_col`` [R] (the pages its
    window reaches; the other columns name the row's first gathered page, which the window never reads). The gather and
    the transpose run on the FP8 tensors themselves (a uint8 view of them fails the compiler's type check)."""
    t = table.to(torch.int64)
    rows, width = t.shape
    blocks, heads, page, d = k_cache.shape
    j = torch.arange(width, device=t.device).reshape(1, width)
    if n_cols is None:
        n_cols, first = width, torch.zeros((rows, 1), dtype=torch.int64, device=t.device)
    else:
        n_cols = min(int(n_cols), width)
        first = first_col.to(torch.int64).reshape(rows, 1)
    cols = (first + torch.arange(n_cols, device=t.device).reshape(1, n_cols)).clamp(max=width - 1)
    src = t.gather(1, cols).clamp(min=0).reshape(-1)
    k_pages = k_cache.index_select(0, src)
    v_pages = v_cache.index_select(0, src)
    n = src.shape[0]
    k_packed = k_pages.reshape(n, heads, page // 2, 2, d).transpose(-1, -2).contiguous()
    rel = j - first
    inside = (rel >= 0) & (rel < n_cols)
    base = torch.arange(rows, device=t.device).reshape(rows, 1) * n_cols
    new = torch.where(inside, base + rel, base)
    return k_packed, v_pages, torch.where(t >= 0, new, torch.full_like(new, -1)).to(torch.int32)


def _group_sink_loader(heads_per_rank: int, size: int):
    """``attention_sink_bias`` [heads] as attention-TP rank ``rank``'s DCP group's sinks (group_sinks: the DCP group is
    ``size`` consecutive attention-TP ranks)."""
    from sglang_neuron.distributed.decode_context_parallel import group_sinks
    from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader

    return SafetensorsWeightLoader(transform=lambda slices, rank: group_sinks(
        slices[0][...].to(torch.float32), heads_per_rank, rank, size, rank % size))


@dataclass
class DecodeInputs:
    """A decode step's inputs shared by every layer of one kind: the layer kind's mask in the kernel's layout, or in
    position mode the kernel's position inputs (ops/attention/attention_decode_paged.PagedDecodeInputs)."""

    positions: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    paged: object = None


class MiMoAttention(nn.Module):
    def __init__(self, config: MiMoV2Config, layer_idx: int, kind: Optional[AttentionKind] = None):
        """``kind`` overrides the layer's own (the MTP layer, past the target's layers, is a sliding one)."""
        super().__init__()
        kind = kind or config.kind(layer_idx)
        self.kind = kind
        self.layer_idx = layer_idx
        self.name = config.layer_name(layer_idx)
        self.hidden_size = config.hidden_size
        self.sliding_window = kind.sliding_window
        self.tp_group = get_attn_tp_group()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.tp_group.world_size
        if kind.num_heads % tp:
            raise ValueError(f"{kind.num_heads} attention heads do not shard over an attention TP of {tp}")
        if kind.num_kv_heads % tp and tp % kind.num_kv_heads:
            raise ValueError(f"{kind.num_kv_heads} KV heads neither shard over nor replicate across an attention TP "
                             f"of {tp}")
        self.num_heads = kind.num_heads // tp
        self.num_kv_heads = max(1, kind.num_kv_heads // tp)
        self.head_dim, self.v_head_dim = kind.head_dim, kind.v_head_dim
        self.scaling = kind.scaling
        self.eps = config.rms_norm_eps
        width = (self.num_heads + 2 * self.num_kv_heads) * KV_HEAD_DIM
        self.qkv_weight = nn.Parameter(torch.empty(config.hidden_size, width, dtype=DTYPE))
        self.o_weight = nn.Parameter(torch.empty(self.num_heads * self.v_head_dim, config.hidden_size, dtype=DTYPE))
        self.sink = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32)) if kind.sink else None
        # bound by bind_kv_cache: [pages, kv_heads, page, KV_HEAD_DIM] each
        self.k_cache = None
        self.v_cache = None
        self._config = config
        self.fp8_kv = fp8_kv_cache()
        self.kv_scale = kv_scale() if self.fp8_kv else 1.0
        group = groups.get_neuron_dcp_group()
        self.dcp_group = group if group is not None and group.world_size > 1 else None
        self.dcp_size = 1 if self.dcp_group is None else self.dcp_group.world_size
        self.set_qkv_disk_tp(1)
        set_weight_loader(self.o_weight, weights.o_proj_loader(self.num_heads, self.v_head_dim))
        if self.sink is not None:
            set_weight_loader(self.sink, weights.sink_loader(self.num_heads))
        # the DCP decode's sinks: the group's gathered heads', real on the group's rank 0 only (group_sinks)
        self.dcp_sink = None
        if self.sink is not None and self.dcp_size > 1:
            self.dcp_sink = nn.Parameter(torch.empty(self.num_heads * self.dcp_size, dtype=torch.float32))
            set_weight_loader(self.dcp_sink, _group_sink_loader(self.num_heads, self.dcp_size))

    def set_qkv_disk_tp(self, disk_tp: int) -> None:
        """The fused QKV's loader for the checkpoint's shard count (weights.qkv_disk_tp), set before the load."""
        kind = self.kind
        set_weight_loader(self.qkv_weight, weights.fused_qkv_loader(
            num_heads=kind.num_heads, num_kv_heads=kind.num_kv_heads, head_dim=kind.head_dim,
            v_head_dim=kind.v_head_dim, rotary_dim=self._config.rotary_dim, v_scale=self._config.v_scale,
            tp=self.tp_group.world_size, disk_tp=disk_tp))

    def _split(self, qkv: torch.Tensor) -> tuple:
        tokens = qkv.shape[0]
        q, k, v = qkv.split([self.num_heads * KV_HEAD_DIM, self.num_kv_heads * KV_HEAD_DIM,
                             self.num_kv_heads * KV_HEAD_DIM], dim=-1)
        return (q.view(tokens, self.num_heads, KV_HEAD_DIM), k.view(tokens, self.num_kv_heads, KV_HEAD_DIM),
                v.view(tokens, self.num_kv_heads, KV_HEAD_DIM))

    def _project_out(self, heads: torch.Tensor) -> torch.Tensor:
        """heads [T, num_heads, KV_HEAD_DIM] through this rank's output projection, all-reduced over attention TP."""
        tokens = heads.shape[0]
        out = heads[..., :self.v_head_dim].reshape(tokens, self.num_heads * self.v_head_dim) @ self.o_weight
        return neuron_collectives.all_reduce(self.attn_tp_group, out)

    def _scale_tensors(self, device) -> tuple:
        """The decode kernels' (k_scale, v_scale) [1, 1] fp32, (None, None) over a bf16 cache."""
        if not self.fp8_kv:
            return None, None
        scale = torch.full((1, 1), self.kv_scale, dtype=torch.float32, device=device)
        return scale, scale

    def _kernel_o_weight(self) -> torch.Tensor:
        """The output projection a decode kernel takes: over an FP8 cache its heads come out times the V scale."""
        return self.o_weight if self.kv_scale == 1.0 else self.o_weight / self.kv_scale

    def _sink_column(self) -> Optional[torch.Tensor]:
        return None if self.sink is None else self.sink.reshape(self.num_heads, 1)

    # -- prefill ------------------------------------------------------------------------------------------------------

    def prefill(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, md: dict) -> torch.Tensor:
        """The chunk's attention output [T, H] for its normed hidden states ``x`` [T, H]."""
        import sglang_neuron.ops as NF

        q, k, v = self._split(x @ self.qkv_weight)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        if self.fp8_kv:
            k, v = ((t * self.kv_scale).clamp(-FP8_KV_MAX, FP8_KV_MAX) for t in (k, v))
        k_cache, v_cache = kv_write.write_kv(self.k_cache, self.v_cache, k.transpose(0, 1), v.transpose(0, 1),
                                             md["slot_mapping"])
        if self.dcp_size > 1:
            return self._prefill_dcp(q, k, v, k_cache, v_cache, md)
        table, scale = md["block_table_tensor"], self.scaling
        if self.fp8_kv:
            page, tokens = md["block_size"], x.shape[0]
            window = {}
            if self.sliding_window:
                cached = md["cached_seq_len"].to(torch.int64).reshape(-1)
                window = dict(first_col=torch.clamp(cached - self.sliding_window + 1, min=0) // page,
                              n_cols=(tokens + self.sliding_window - 1) // page + 2)
            k_cache, v_cache, table = pack_prior(k_cache, v_cache, table, **window)
            scale = scale / self.kv_scale
        out = NF.segmented_attention(
            q.transpose(0, 1).contiguous(), k_cache=k_cache, v_cache=v_cache,
            block_tables=table, prior_tokens=md["cached_seq_len"], block_size=md["block_size"],
            kv_segment_size=md["kv_segment_size"], scale=scale, sliding_window=self.sliding_window,
            sink=self._sink_column(),
        )
        if self.kv_scale != 1.0:
            out = out / self.kv_scale
        return self._project_out(out.transpose(0, 1))

    def _mesh(self) -> tuple:
        from sglang_neuron.backend.stack import resolve_full_mesh

        return tuple(tuple(int(r) for r in cell) for cell in resolve_full_mesh(self.dcp_group))

    def _prefill_dcp(self, q, k, v, k_cache, v_cache, md) -> torch.Tensor:
        """A chunk's attention under DCP (module docstring); ``k`` / ``v`` [T, 1, d] its K/V as the cache holds them
        (times the FP8 scale), the caches as written."""
        import sglang_neuron.ops as NF
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels import attention_block
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        size, page, tokens = self.dcp_size, md["block_size"], q.shape[0]
        cached = md["cached_seq_len"].to(torch.int32).reshape(1, 1)
        table = md["block_table_tensor"].to(torch.int32)
        heads = q.transpose(0, 1).contiguous()
        if self.sliding_window is None:
            # every column of this rank's shard (FP8: packed for the kernel)
            ks = vs = None
            if self.fp8_kv:
                k_cache, v_cache, table = pack_prior(k_cache, v_cache, table)
                ks = vs = torch.full((128, 1), 1.0 / self.kv_scale, dtype=torch.float32, device=q.device)
            table = torch.where(table < 0, torch.zeros_like(table), table)
            out = wrap_nki(attention_block.prefill_kv_parallel)[get_logical_nc_config()](
                heads, k_cache, v_cache, table[:1], cached, self.dcp_group.rank_in_group_tensor.to(torch.int32).reshape(
                    1, 1), torch.div(cached, size, rounding_mode="floor"), ks, vs, scale=self.scaling,
                block_size=page, seg_size=min(tokens, size * page), dcp_size=size, dcp_mesh=self._mesh(),
                fp8_packed=self.fp8_kv)
            return self._project_out(out.transpose(0, 1))
        # a sliding layer: the window's prior block from the group's last rank, then the chunk, attended locally
        from sglang_neuron.parallel import collectives as neuron_collectives_

        col = torch.clamp(torch.div(cached.reshape(1).to(torch.int64), size * page, rounding_mode="floor") - 1, min=0)
        block = table.to(torch.int64)[0].gather(0, col).clamp(min=0)

        def prior(cache):
            blk = cache.index_select(0, block)
            if self.fp8_kv:
                blk = (blk.to(torch.float32) / self.kv_scale).to(DTYPE)
            return neuron_collectives_.all_gather(self.dcp_group, blk, dim=0)[size - 1:size]

        def chunk(t):
            if self.fp8_kv:  # the chunk as the cache holds it
                t = (t.to(k_cache.dtype).to(torch.float32) / self.kv_scale).to(DTYPE)
            return t.reshape(tokens // page, page, 1, KV_HEAD_DIM).transpose(1, 2)

        k_local = torch.cat([prior(k_cache), chunk(k)], dim=0)
        v_local = torch.cat([prior(v_cache), chunk(v)], dim=0)
        n = k_local.shape[0]
        first = (cached.reshape(1) == 0).to(torch.int32)
        local_table = torch.clamp(torch.arange(n, dtype=torch.int32, device=q.device) + first, max=n - 1).reshape(1, n)
        prior_tokens = torch.where(cached > 0, torch.full_like(cached, page), torch.zeros_like(cached))
        out = NF.segmented_attention(
            heads, k_cache=k_local, v_cache=v_local, block_tables=local_table, prior_tokens=prior_tokens,
            block_size=page, kv_segment_size=md["kv_segment_size"], scale=self.scaling,
            sliding_window=self.sliding_window, sink=self._sink_column())
        return self._project_out(out.transpose(0, 1))

    def _dcp_decode(self, x, norm, cos, sin, md, mask, slots, span: int = 1) -> torch.Tensor:
        """``span`` tokens a row (one outside a NEURON_MTP step) over the DCP group's shards: ``mask`` the group's (the
        runtime's per-row bounds), ``slots`` [B x span] this rank's slot of each token (0 where another rank keeps it);
        the K/V written after the attention, the output all-reduced."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels import attention_block
        from sglang_neuron.ops.attention.attention_decode import dcp_collective_mode
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        tokens = x.shape[0]
        rows = tokens // span
        layout = lambda t: t.reshape(rows, span, -1).permute(2, 0, 1).contiguous()  # noqa: E731
        k_scale, v_scale = self._scale_tensors(x.device)
        sink = None if self.dcp_sink is None else self.dcp_sink.reshape(-1, 1)
        out, k_new, v_new = wrap_nki(attention_block.decode_masked_dcp)[get_logical_nc_config()](
            X=x.view(rows, span, -1), X_hidden_dim_actual=self.hidden_size, rmsnorm_X_eps=norm.eps,
            rmsnorm_X_gamma=norm.weight.unsqueeze(0).to(torch.float32), W_qkv=self.qkv_weight, cos=layout(cos),
            sin=layout(sin), active_blocks_table=md["block_table_tensor"], K_cache=self.k_cache, V_cache=self.v_cache,
            attention_mask=mask, sink=sink, softmax_scale=self.scaling / self.kv_scale, k_scale=k_scale,
            v_scale=v_scale, dcp_size=self.dcp_size, dcp_mesh=self._mesh(), dcp_mode=dcp_collective_mode(self.dcp_size))
        heads = out.reshape(rows, self.num_heads, KV_HEAD_DIM, span).permute(0, 3, 1, 2).reshape(
            tokens, self.num_heads, KV_HEAD_DIM)
        if self.kv_scale != 1.0:
            heads = heads / self.kv_scale
        kv_write.write_kv(self.k_cache, self.v_cache, _fp8_rows_t(k_new, KV_HEAD_DIM, tokens).reshape(
            1, tokens, KV_HEAD_DIM), v_new.reshape(1, tokens, KV_HEAD_DIM), slots)
        return self._project_out(heads)

    # -- decode -------------------------------------------------------------------------------------------------------

    def decode(self, x: torch.Tensor, norm, cos: torch.Tensor, sin: torch.Tensor, md: dict,
               inputs: DecodeInputs) -> torch.Tensor:
        """The attention output [B, H] of one token a row; ``x`` [B, H] is the residual stream (the kernel applies the
        layer's input norm ``norm``), cos / sin [B, KV_HEAD_DIM / 2] the layer kind's tables."""
        if self.num_kv_heads != 1:
            raise ValueError(f"the MiMo-V2 decode attends one KV head per rank, not {self.num_kv_heads}")
        rows = x.shape[0]
        if self.dcp_size > 1:
            return self._dcp_decode(x, norm, cos, sin, md, inputs.mask, md["slot_mapping"].reshape(-1))
        if can_run_kernel(x) and inputs.mask is not None and fused_enabled():
            return self._fused_decode(x, norm, cos, sin, md, inputs)
        if not can_run_kernel(x):
            heads, k_new, v_new = self._torch_decode(x, norm, cos, sin, md, inputs.positions)
        else:
            heads, k_new, v_new = self._kernel_decode(x, norm, cos, sin, md, inputs)
        kv_write.write_kv(self.k_cache, self.v_cache, k_new.reshape(1, rows, KV_HEAD_DIM),
                          v_new.reshape(1, rows, KV_HEAD_DIM), md["slot_mapping"])
        return self._project_out(heads)

    def span_decode(self, x, norm, cos, sin, md: dict, mask, slots, span: int, caches=None) -> tuple:
        """The attention output [B x span, H] of ``span`` tokens a row (a NEURON_MTP step's verify, or a draft pass):
        ``x`` [B x span, H] the residual stream, row-major, cos / sin [B x span, KV_HEAD_DIM / 2], ``mask`` the
        layer kind's span mask (``span_mask``), ``slots`` [B x span] each token's pool slot (0 for a token the step
        drops). The tokens attend the prior the mask opens and each other causally; their K/V are written after the
        attention. ``caches`` (K, V) are the caches to read, if a pass of this step wrote them; returns
        (output, K cache, V cache) as written."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels import attention_block
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        if self.num_kv_heads != 1:
            raise ValueError(f"the MiMo-V2 decode attends one KV head per rank, not {self.num_kv_heads}")
        k_cache, v_cache = caches or (self.k_cache, self.v_cache)
        tokens = x.shape[0]
        rows = tokens // span
        # [B x S, d / 2] -> the kernel's [d / 2, B, S]
        layout = lambda t: t.reshape(rows, span, -1).permute(2, 0, 1).contiguous()  # noqa: E731
        args = dict(
            X=x.view(rows, span, -1), rmsnorm_X_gamma=norm.weight.unsqueeze(0).to(torch.float32),
            rmsnorm_X_eps=norm.eps, X_hidden_dim_actual=self.hidden_size, W_qkv=self.qkv_weight, cos=layout(cos),
            sin=layout(sin), K_cache=k_cache, V_cache=v_cache, sink=self._sink_column(),
            softmax_scale=self.scaling / self.kv_scale, active_blocks_table=md["block_table_tensor"],
            attention_mask=mask)
        if self.fp8_kv:
            k_scale, v_scale = self._scale_tensors(x.device)
            out, k_new, v_new = wrap_nki(attention_block.decode_masked_fp8)[get_logical_nc_config()](
                k_scale=k_scale, v_scale=v_scale, **args)
        else:
            out, k_new, v_new = wrap_nki(attention_block.decode_masked)[get_logical_nc_config()](**args)
        # out [B, heads, d, S]; K [d, B, S] and V [B, 1, S, d] for one KV head (nkilib attention_block_tkg)
        heads = out.reshape(rows, self.num_heads, KV_HEAD_DIM, span).permute(0, 3, 1, 2).reshape(
            tokens, self.num_heads, KV_HEAD_DIM)
        if self.kv_scale != 1.0:
            heads = heads / self.kv_scale
        k_cache, v_cache = kv_write.write_kv(k_cache, v_cache, _fp8_rows_t(k_new, KV_HEAD_DIM, tokens).reshape(
            1, tokens, KV_HEAD_DIM), v_new.reshape(1, tokens, KV_HEAD_DIM), slots)
        return self._project_out(heads), k_cache, v_cache

    def _fused_decode(self, x, norm, cos, sin, md, inputs: DecodeInputs) -> torch.Tensor:
        """The attention half in one kernel (kernels/attn_half.py), its partial all-reduced."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.attn_half import attn_half
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        rows = x.shape[0]
        k_scale, v_scale = self._scale_tensors(x.device)
        out, _, _ = wrap_nki(attn_half)[get_logical_nc_config()](
            x.view(rows, 1, -1), norm.weight.unsqueeze(0).to(torch.float32), self.qkv_weight,
            cos.t().unsqueeze(-1).contiguous(), sin.t().unsqueeze(-1).contiguous(), md["block_table_tensor"],
            self.k_cache, self.v_cache, inputs.mask, self._sink_column(),
            md["slot_mapping"].view(rows, 1).to(torch.int32), self._kernel_o_weight(), k_scale, v_scale,
            rmsnorm_X_eps=norm.eps, softmax_scale=self.scaling / self.kv_scale, v_head_dim=self.v_head_dim)
        return neuron_collectives.all_reduce(self.attn_tp_group, out)

    def _kernel_decode(self, x, norm, cos, sin, md, inputs: DecodeInputs) -> tuple:
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels import attention_block

        rows = x.shape[0]
        args = dict(
            X=x.view(rows, 1, -1), rmsnorm_X_gamma=norm.weight.unsqueeze(0).to(torch.float32), rmsnorm_X_eps=norm.eps,
            X_hidden_dim_actual=self.hidden_size, W_qkv=self.qkv_weight,
            # [B, d / 2] -> the kernel's [d / 2, B, 1]
            cos=cos.t().unsqueeze(-1).contiguous(), sin=sin.t().unsqueeze(-1).contiguous(),
            K_cache=self.k_cache, V_cache=self.v_cache, sink=self._sink_column(),
            softmax_scale=self.scaling / self.kv_scale,
        )
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        if self.fp8_kv:
            if inputs.paged is not None:
                raise ValueError(f"{FULL_DECODE_ENV}=posid is not served with the FP8 KV cache")
            k_scale, v_scale = self._scale_tensors(x.device)
            out, k_new, v_new = wrap_nki(attention_block.decode_masked_fp8)[get_logical_nc_config()](
                active_blocks_table=md["block_table_tensor"], attention_mask=inputs.mask, k_scale=k_scale,
                v_scale=v_scale, **args)
        elif inputs.paged is not None:
            p = inputs.paged
            out, k_new, v_new = wrap_nki(attention_block.decode_posid)[get_logical_nc_config()](
                active_blocks_table=p.table, attention_mask=p.active_mask, pos_ids=p.pos_ids, **args)
        else:
            out, k_new, v_new = wrap_nki(attention_block.decode_masked)[get_logical_nc_config()](
                active_blocks_table=md["block_table_tensor"], attention_mask=inputs.mask, **args)
        # out [B, heads, d, 1]; K [d, B, 1] and V [B, 1, 1, d] for one KV head (nkilib attention_block_tkg)
        heads = out.reshape(rows, self.num_heads, KV_HEAD_DIM, 1)[..., 0]
        if self.kv_scale != 1.0:
            heads = heads / self.kv_scale
        return heads, _fp8_rows_t(k_new, KV_HEAD_DIM, rows), v_new.reshape(rows, KV_HEAD_DIM)

    def _torch_decode(self, x, norm, cos, sin, md, positions) -> tuple:
        """The kernel's decode by plain attention over each row's pages (the CPU reference): the prior is the row's
        positions before its token (those within the window on a sliding layer, in the trimmed table's frame), then the
        token itself, with the sinks in the softmax."""
        rows = x.shape[0]
        q, k, v = self._split(norm(x) @ self.qkv_weight)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        table = md["block_table_tensor"].to(torch.int64)
        page = self.k_cache.shape[2]
        span = table.shape[1] * page
        pages = table.clamp_min(0)
        k_ctx = self.k_cache[pages][:, :, 0].reshape(rows, span, KV_HEAD_DIM).float()
        v_ctx = self.v_cache[pages][:, :, 0].reshape(rows, span, KV_HEAD_DIM).float()
        pos = positions.to(torch.int64).reshape(rows, 1)
        if self.sliding_window is not None:
            pos = (pos - md["swa_kv_pos_offset"].to(torch.int64).reshape(rows, 1)).clamp_min(0)
        slot = torch.arange(span).reshape(1, span)
        valid = (slot < pos) & (table.repeat_interleave(page, dim=1) >= 0)
        if self.sliding_window is not None:
            valid &= slot > pos - self.sliding_window
        keys = torch.cat([k_ctx, k.float().reshape(rows, 1, KV_HEAD_DIM)], dim=1)
        values = torch.cat([v_ctx, v.float().reshape(rows, 1, KV_HEAD_DIM)], dim=1)
        valid = torch.cat([valid, torch.ones(rows, 1, dtype=torch.bool)], dim=1)
        scores = torch.einsum("bhd,bsd->bhs", q.float(), keys) * self.scaling
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        if self.sink is not None:
            scores = torch.cat([scores, self.sink.float().reshape(1, self.num_heads, 1).expand(rows, -1, 1)], dim=-1)
        probs = torch.softmax(scores, dim=-1)
        if self.sink is not None:
            probs = probs[..., :-1]
        heads = torch.einsum("bhs,bsd->bhd", probs, values).to(x.dtype)
        return heads, k.reshape(rows, KV_HEAD_DIM), v.reshape(rows, KV_HEAD_DIM)


def qk_swap_layout(mask: torch.Tensor, *, d_head: int, fp8_kv: bool, fp8_packed: bool, kv_heads: int = 1):
    """``mask`` [s_prior, bs, q_head, s_active] (gen_attention_decode_mask) in the layout attention_block_tkg takes
    for it: as it is, or [bs, q_head, s_active, s_prior] where the kernel takes its transposed-score (QK-swap) path for
    the shape, which nkilib's is_qk_swapped decides (a span of tokens a row, as an MTP verify, can take it)."""
    from nkilib.core.attention.attention_tkg_utils import is_qk_swapped

    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    s_prior, bs, q_head, s_active = (int(d) for d in mask.shape)
    swapped = is_qk_swapped(bs=bs, q_head=q_head, d_head=d_head, s_active=s_active, curr_sprior=s_prior,
                            lnc=get_logical_nc_config(), p_max=128, is_block_kv=True, is_2byte_kv=not fp8_kv,
                            fp8_packed=fp8_packed, fuse_rope=False, kv_heads=kv_heads)
    return mask.permute(1, 2, 3, 0).contiguous() if swapped else mask


def decode_masks(attn_metadata: dict, positions: torch.Tensor, config: MiMoV2Config, num_heads: int) -> dict:
    """Each layer kind's ``DecodeInputs`` for a decode step (``{"swa": ..., "full": ...}``), built once for all of its
    layers. On a CPU none are needed (the torch path masks by position)."""
    swa_layer = next((config.layer_name(i) for i, s in enumerate(config.swa_layers) if s), None)
    full_layer = next((config.layer_name(i) for i, s in enumerate(config.swa_layers) if not s), None)
    if not can_run_kernel(positions):
        return {"swa": DecodeInputs(positions=positions), "full": DecodeInputs(positions=positions)}
    from sglang_neuron.models.gpt_oss.attention_masks import decode_attention_masks

    posid = full_decode_mode() == "posid" and full_layer is not None
    masked_full = None if posid else full_layer
    swa_mask = full_mask = None
    if swa_layer is not None or masked_full is not None:
        swa_mask, full_mask = decode_attention_masks(
            attn_metadata, positions, sliding_window=config.swa.sliding_window or 0, q_heads=num_heads,
            swa_layer=swa_layer, full_layer=masked_full)

    def kernel_layout(mask):
        return qk_swap_layout(mask, d_head=KV_HEAD_DIM, fp8_kv=fp8_kv_cache(), fp8_packed=False)

    out = {}
    if swa_mask is not None:
        out["swa"] = DecodeInputs(positions=positions, mask=kernel_layout(swa_mask))
    if posid:
        from sglang_neuron.ops.attention.attention_decode_paged import paged_decode_inputs

        md = attn_metadata[full_layer]
        paged = paged_decode_inputs(md["block_table_tensor"], md["slot_mapping"], positions, kv_heads=1,
                                    q_heads=num_heads, page=md["block_size"])
        out["full"] = DecodeInputs(positions=positions, paged=paged)
    elif full_mask is not None:
        out["full"] = DecodeInputs(positions=positions, mask=kernel_layout(full_mask))
    return out


def span_mask(md: dict, prior_end: torch.Tensor, positions: torch.Tensor, *, window: Optional[int], num_heads: int):
    """A layer kind's decode mask in the kernel's layout for ``positions`` [B, S] (S tokens a row, which see each
    other causally), whose prior ends (exclusive) at ``prior_end`` [B, S]: the full layers' over the row's whole
    table, the sliding layers' over the trimmed one (``swa_kv_pos_offset``), from each token's own window start."""
    from sglang_neuron.ops.attention.attention_decode_mask import gen_attention_decode_mask

    rows, span = positions.shape
    page = md["block_size"]
    end, start = prior_end.to(torch.int64), None
    if window:
        offset = md["swa_kv_pos_offset"].to(torch.int64).reshape(rows, 1)
        end = (end - offset).clamp(min=0)
        start = (positions.to(torch.int64) - offset - window + 1).clamp(min=0)
        start = torch.minimum(start, end).reshape(1, rows * span).to(torch.float32)
    mask = gen_attention_decode_mask(end.reshape(1, rows * span).to(torch.float32), bs=rows, q_head=num_heads,
                                     s_active=span, s_prior=md["max_blocks_per_seq"] * page, block_len=page,
                                     start_pos=start)
    return qk_swap_layout(mask, d_head=KV_HEAD_DIM, fp8_kv=fp8_kv_cache(), fp8_packed=False)


def span_slots(md: dict, positions: torch.Tensor, valid: torch.Tensor, *, window: Optional[int]) -> torch.Tensor:
    """[B x S] the pool slot of each token at ``positions`` [B, S] from its row's table (the sliding layers' trimmed
    one, in the window pool), 0 (SGLang's reserved slot) where ``valid`` is False or the page is unallocated."""
    table = md["block_table_tensor"].to(torch.int64)
    page = md["block_size"]
    pos = positions.to(torch.int64)
    frame = pos - md["swa_kv_pos_offset"].to(torch.int64).reshape(-1, 1) if window else pos
    raw = frame.div(page, rounding_mode="floor")
    block = table.gather(1, raw.clamp(min=0, max=table.shape[1] - 1))
    slot = block * page + pos % page
    keep = valid & (block >= 0) & (raw >= 0) & (raw < table.shape[1])
    return torch.where(keep, slot, torch.zeros_like(slot)).reshape(-1)
