# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's attention on attention TP, one layer kind at a time (config.AttentionKind): its query heads shard over
the ranks, and with more ranks than KV heads each rank holds one KV head, replicated (4 full / 8 sliding KV heads: 16
/ 8 ranks a head at TP64). Every head runs at config.KV_HEAD_DIM channels (weights.head_layout).

- **Prefill**, one request's chunk: the QKV projection and the partial RoPE in the graph, the chunk's K/V into its
  pages (kv_write.py), then nkilib's segmented attention over the prior and the chunk, with the layer's window and
  sinks; the window kernel loads only the pages a window reaches.
- **Decode**: each layer runs as one kernel (kernels/layer.py, from model.MiMoV2Graph._decode_kernels): the input
  RMSNorm, the QKV projection, RoPE, the K/V write and nkilib's ``attention_block_tkg`` over the paged prior with the
  sinks, the output projection and the all-reduce (kernels/attn_half.py); the step's masks, slots and RoPE tables come
  out of kernels/step.decode_inputs. ``decode`` here is the CPU reference.

No decode ends its loop early: nkilib's early exit (``max_context_len``) is wrong past an 8K prior
(sgl-project/sglang-neuron#4).

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

# the FP8 KV cache's scale: the cache holds clip(x * scale) in e4m3 (module docstring)
KV_SCALE_ENV = "SGLANG_NEURON_MIMO_KV_SCALE"
FP8_KV_MAX = 240.0
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
    """A decode step's inputs shared by every layer of one kind: the layer kind's mask in the kernel's layout."""

    positions: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


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

    def prefill(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, md: dict, norm, post_norm) -> tuple:
        """The chunk's attention block over its residual stream ``x`` [T, H]: (h, hn) [T, H], h = x + attention and hn
        its post-attention RMSNorm ``post_norm`` (the MLP's input); ``norm`` is the layer's input RMSNorm. On a device
        the norm, projection, RoPE and K/V write run as kernels/prefill.prefill_front and the output projection,
        all-reduce, residual and norm as kernels/prefill.prefill_back; the torch below is the CPU reference."""
        heads, scale = self._prefill_heads(x, cos, sin, md, norm)
        if can_run_kernel(x):
            from sglang_neuron.backend.stack import resolve_full_mesh, wrap_nki
            from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_back
            from sglang_neuron.utils.hardware_config import get_logical_nc_config

            mesh = tuple(tuple(int(r) for r in cell) for cell in resolve_full_mesh(self.attn_tp_group))
            return wrap_nki(prefill_back)[get_logical_nc_config()](
                heads, self.o_weight, x, post_norm.weight.view(1, -1), mesh, eps=post_norm.eps,
                v_head_dim=self.v_head_dim, kv_scale=scale)
        if scale != 1.0:
            heads = heads / scale
        h = x + self._project_out(heads.transpose(0, 1))
        return h, post_norm(h)

    def _prefill_heads(self, x, cos, sin, md: dict, norm) -> tuple:
        """(the chunk's attention heads [num_heads, T, d] before the output projection, the FP8 scale they are still
        times: 1 where none)."""
        if can_run_kernel(x):
            from sglang_neuron.backend.stack import wrap_nki
            from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_front
            from sglang_neuron.utils.hardware_config import get_logical_nc_config

            if self.num_kv_heads != 1 or x.shape[0] % 128:
                raise ValueError(f"the prefill front kernel takes one KV head a rank and whole 128-token tiles "
                                 f"({self.num_kv_heads} KV heads, {x.shape[0]} tokens): raise the attention TP size")
            slots = md["slot_mapping"].to(torch.int32).reshape(-1, 1)
            # without DCP the queries come out times the segmented attention's softmax scale (over an FP8 cache, over
            # its scale too), which the attention then takes as 1
            # (under DCP a sliding layer's local attention takes the softmax scale on q too; a full layer's KV-parallel
            # kernel applies it itself)
            if self.dcp_size > 1:
                q_scale = self.scaling if self.sliding_window else 1.0
            else:
                q_scale = self.scaling / (self.kv_scale if self.fp8_kv else 1.0)
            heads, k, v, k_cache, v_cache = wrap_nki(prefill_front)[get_logical_nc_config()](
                x, norm.weight.view(1, -1), self.qkv_weight, cos, sin, slots, self.k_cache, self.v_cache,
                eps=norm.eps, num_heads=self.num_heads, kv_scale=float(self.kv_scale) if self.fp8_kv else 1.0,
                q_scale=float(q_scale))
            if self.dcp_size > 1:
                return self._prefill_dcp_kernels(heads, k, v, k_cache, v_cache, md)
            return self._prefill_attend(heads, k_cache, v_cache, md, x.shape[0], prescaled=True)
        x = norm(x)
        q, k, v = self._split(x @ self.qkv_weight)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        if self.fp8_kv:
            k, v = ((t * self.kv_scale).clamp(-FP8_KV_MAX, FP8_KV_MAX) for t in (k, v))
        k_cache, v_cache = kv_write.write_kv(self.k_cache, self.v_cache, k.transpose(0, 1), v.transpose(0, 1),
                                             md["slot_mapping"])
        if self.dcp_size > 1:
            return self._prefill_dcp(q, k, v, k_cache, v_cache, md)
        return self._prefill_attend(q.transpose(0, 1).contiguous(), k_cache, v_cache, md, x.shape[0])

    def _prefill_attend(self, heads, k_cache, v_cache, md: dict, tokens: int, prescaled: bool = False) -> tuple:
        """The chunk's attention over the written caches, ``heads`` [num_heads, T, d] its rotated queries (times the
        softmax scale already where ``prescaled``: prefill_front's): (its output heads [num_heads, T, d], the FP8 scale
        they are still times)."""
        import sglang_neuron.ops as NF

        table, scale = md["block_table_tensor"], self.scaling
        if self.fp8_kv:
            page = md["block_size"]
            if prescaled:
                # on a device: the prior's pages gathered and K packed in one kernel (kernels/prefill.prefill_pack)
                from sglang_neuron.backend.stack import wrap_nki
                from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_pack
                from sglang_neuron.utils.hardware_config import get_logical_nc_config

                window = int(self.sliding_window or 0)
                n_cols = (tokens + window - 1) // page + 2 if window else table.shape[1]
                n_cols = min(n_cols, table.shape[1])
                k_cache, v_cache, table = wrap_nki(prefill_pack)[get_logical_nc_config()](
                    k_cache, v_cache, table.to(torch.int32), md["cached_seq_len"].to(torch.int32).reshape(1, 1),
                    n_cols=n_cols, window=window)
            else:
                window = {}
                if self.sliding_window:
                    cached = md["cached_seq_len"].to(torch.int64).reshape(-1)
                    window = dict(first_col=torch.clamp(cached - self.sliding_window + 1, min=0) // page,
                                  n_cols=(tokens + self.sliding_window - 1) // page + 2)
                k_cache, v_cache, table = pack_prior(k_cache, v_cache, table, **window)
            scale = scale / self.kv_scale
        if prescaled:
            from sglang_neuron.backend.stack import wrap_nki
            from sglang_neuron.ops.attention.attention_segmented_cte import _attention_segmented_cte
            from sglang_neuron.utils.hardware_config import get_logical_nc_config

            out = wrap_nki(_attention_segmented_cte)[get_logical_nc_config()](
                q=heads, k_cache=k_cache, v_cache=v_cache, block_tables=table, prior_tokens=md["cached_seq_len"],
                block_size=md["block_size"], prior_seg_size=md["kv_segment_size"], scale=1.0,
                sliding_window=self.sliding_window, sink=self._sink_column(), num_q_heads=heads.shape[0],
                fp8_packed=k_cache.dim() == v_cache.dim() + 1)
            return out, (float(self.kv_scale) if self.kv_scale != 1.0 else 1.0)
        out = NF.segmented_attention(
            heads, k_cache=k_cache, v_cache=v_cache,
            block_tables=table, prior_tokens=md["cached_seq_len"], block_size=md["block_size"],
            kv_segment_size=md["kv_segment_size"], scale=scale, sliding_window=self.sliding_window,
            sink=self._sink_column(),
        )
        return out, (float(self.kv_scale) if self.kv_scale != 1.0 else 1.0)

    def _mesh(self) -> tuple:
        from sglang_neuron.backend.stack import resolve_full_mesh

        return tuple(tuple(int(r) for r in cell) for cell in resolve_full_mesh(self.dcp_group))

    def _prefill_dcp_kernels(self, heads, k, v, k_cache, v_cache, md) -> tuple:
        """A chunk's attention under DCP on a device, as kernels: ``heads`` [num_heads, T, d] prefill_front's queries
        (a sliding layer's times the softmax scale), ``k`` / ``v`` [T, d] the chunk's K / V as the cache holds them.
        A full layer: the prior's pages (FP8: gathered and packed, kernels/prefill.prefill_pack), the table's missing
        columns at page 0 and each rank's share (kernels/prefill.dcp_full_inputs), then nkilib's KV-parallel segmented
        attention; a sliding layer: its window's prior page from the group's last rank and the chunk's own pages
        (kernels/prefill.prefill_dcp_window), attended locally. Returns (heads [num_heads, T, d], 1.0)."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels import attention_block
        from sglang_neuron.models.mimo_v2.kernels.prefill import dcp_full_inputs, prefill_dcp_window, prefill_pack
        from sglang_neuron.ops.attention.attention_segmented_cte import _attention_segmented_cte
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        grid = get_logical_nc_config()
        size, page, tokens = self.dcp_size, md["block_size"], heads.shape[1]
        cached = md["cached_seq_len"].to(torch.int32).reshape(1, 1)
        table = md["block_table_tensor"].to(torch.int32)
        scale = float(self.kv_scale) if self.fp8_kv else 1.0
        if self.sliding_window is None:
            if self.fp8_kv:
                k_cache, v_cache, table = wrap_nki(prefill_pack)[grid](k_cache, v_cache, table, cached,
                                                                       n_cols=table.shape[1], window=0)
            table, prior, inv = wrap_nki(dcp_full_inputs)[grid](table, cached, dcp=size, kv_scale=scale)
            ks = vs = inv if self.fp8_kv else None
            out = wrap_nki(attention_block.prefill_kv_parallel)[grid](
                heads, k_cache, v_cache, table, cached, self.dcp_group.rank_in_group_tensor.to(torch.int32).reshape(
                    1, 1), prior, ks, vs, scale=self.scaling, block_size=page, seg_size=min(tokens, size * page),
                dcp_size=size, dcp_mesh=self._mesh(), fp8_packed=self.fp8_kv)
            return out, 1.0
        k_local, v_local, local_table, prior_tokens = wrap_nki(prefill_dcp_window)[grid](
            k_cache, v_cache, table, cached, k, v, self._mesh(), dcp=size, kv_scale=scale)
        out = wrap_nki(_attention_segmented_cte)[grid](
            q=heads, k_cache=k_local, v_cache=v_local, block_tables=local_table, prior_tokens=prior_tokens,
            block_size=page, prior_seg_size=md["kv_segment_size"], scale=1.0, sliding_window=self.sliding_window,
            sink=self._sink_column(), num_q_heads=heads.shape[0], fp8_packed=False)
        return out, 1.0

    def _prefill_dcp(self, q, k, v, k_cache, v_cache, md) -> tuple:
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
            return out, 1.0
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
        return out, 1.0

    # -- decode -------------------------------------------------------------------------------------------------------

    def decode(self, x: torch.Tensor, norm, cos: torch.Tensor, sin: torch.Tensor, md: dict,
               inputs: DecodeInputs) -> torch.Tensor:
        """The CPU reference of a decode step's attention: the output [B, H] of one token a row; ``x`` [B, H] is the
        residual stream (the input norm ``norm`` applied here), cos / sin [B, KV_HEAD_DIM / 2] the layer kind's
        tables."""
        if self.num_kv_heads != 1:
            raise ValueError(f"the MiMo-V2 decode attends one KV head per rank, not {self.num_kv_heads}")
        if can_run_kernel(x) or self.dcp_size > 1:
            raise ValueError("a device decode runs the one-kernel layers (model.MiMoV2Graph._decode_kernels); "
                             "this is the CPU reference, without DCP")
        rows = x.shape[0]
        heads, k_new, v_new = self._torch_decode(x, norm, cos, sin, md, inputs.positions)
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


def decode_masks(positions: torch.Tensor) -> dict:
    """Each layer kind's ``DecodeInputs`` for a CPU decode step (``{"swa": ..., "full": ...}``): the torch path masks
    by position (a device decode builds its masks in kernels/step.decode_inputs)."""
    return {"swa": DecodeInputs(positions=positions), "full": DecodeInputs(positions=positions)}


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
