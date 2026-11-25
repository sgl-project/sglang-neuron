# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash's gated MLA (the reference's HYV4FlashAttention and HYV4FlashCompressedIndexer) on attention TP.

Each rank holds ``heads / tp`` heads of a layer (sparse layers 32, sliding layers 64). A token's KV is one MLA row,
its RMS-normed 512-wide latent then its 64 RoPE'd key channels, shared by every head. The row lives in the platform's
MHA pool as one KV head of 288 channels, its first half in K and its second in V (sglang_patches/hy4_flash_launch.py),
the sliding layers' in SGLang's window pool. Each head scores the latent through its slice of ``kv_b_proj`` folded
into the query (absorbed MLA) and reads its values back out of the attended latent. The softmax carries a learned
sink logit per head; the output is gated by ``sigmoid(gate_proj(x))``, projected
out by the rank's columns of ``o_proj``, all-reduced and closed by ``post_proj_norm``.

**What a query reads.** A sliding layer's query attends to the 512 positions ending at its own. A sparse layer's runs
the IndexPool: every 4-token group's indexer keys pool into one group key (written when the group completes, at the
KV slot of its first token), a query scores every complete group before its own (16 relu heads, weighted), keeps the
511 best (ops.top_count_mask) and attends to their tokens and its own group's up to itself.

**The per-slot state the graphs keep beside the KV pool** (``bind_state``, one row per pool slot, row 0 the
reserved slot SGLang never allocates, so padding writes land there): each sparse layer's RMS-normed token indexer key
and its group keys. A step writes its tokens' latent rows and indexer keys first, then reads every row it attends to
through the block table, its own included.

The prefill reads the whole table (a sparse layer's keys, every position the request may hold) or its window (a
sliding layer's, the 512 positions before the chunk and the chunk); the decode reads its graph's context (a sparse
layer) or the runtime's table trimmed to the window (a sliding layer)."""

from __future__ import annotations

import nki
import nki.language as nl
import torch
from sglang.srt.distributed.parallel_state import get_attn_tp_group
from torch import nn

from sglang_neuron.models.hy_v4_flash import ops, weights
from sglang_neuron.models.hy_v4_flash.config import INDEX_DIM, INDEX_RATIO, Hy4FlashConfig
from sglang_neuron.models.hy_v4_flash.engram import token_slots
from sglang_neuron.models.transformer.config import DTYPE
from sglang_neuron.models.transformer.model import RMSNorm
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.neuron_utils import can_run_kernel
from sglang_neuron.utils.weight_loader import set_weight_loader


# a sparse layer's prefill scores the whole table up to this context (compute-bound matmuls), and above it gathers each
# query's IndexPool tokens (attention._prefill_gathered), in blocks of this many queries
MASKED_PREFILL_CONTEXT = 16 * 1024
PREFILL_QUERY_BLOCK = 256
# the IndexPool scores every head in one matmul up to this many [rows, heads, groups] scores, one head at a time above
# (a prefill chunk's would be its largest tensor); a decode step's batch then is one matmul a row, not 16
ALL_HEADS_SCORES = 1 << 23
# the IndexPool's selection runs nkilib's top-k kernel up to this many rows (a decode), the threshold search above (a
# prefill chunk's 2048 rows take longer in the kernel); the score of an invalid group
KERNEL_TOPK_ROWS = 128
# and from this many (one row's threshold search takes 0.13 ms, the kernel 0.23; at 64 rows 1.18 and 0.29)
KERNEL_TOPK_MIN_ROWS = 4
# and up to this many groups a row (the kernel's tested range is 151936; a 1M-token rung's table has 262144)
KERNEL_TOPK_GROUPS = 128 * 1024
TOPK_EMPTY = 3.0e38
# an FP8 KV pool (--kv-cache-dtype fp8_e4m3): its element type and Trn2's e4m3 range
KV_FP8 = torch.float8_e4m3fn
KV_FP8_MAX = 240.0


def write_rows(cache: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor) -> None:
    """Write ``rows`` [T, D] into ``cache`` [slots, ..., D] at ``slots`` [T] (out of range to the reserved slot 0)."""
    flat = cache.view(cache.shape[0], -1)
    slots = slots.to(torch.int64)
    slots = torch.where((slots < 0) | (slots >= flat.shape[0]), torch.zeros_like(slots), slots)
    flat[slots] = rows.reshape(rows.shape[0], -1).to(flat.dtype)


def read_rows(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """[*slots.shape, D] rows of ``cache`` [slots, ..., D] at ``slots``."""
    flat = cache.view(cache.shape[0], -1)
    return flat.index_select(0, slots.reshape(-1).to(torch.int64)).view(*slots.shape, flat.shape[-1])


def read_pages(cache: torch.Tensor, table: torch.Tensor, block: int) -> torch.Tensor:
    """[R, blocks x block, D] rows of the whole pages of ``table`` [R, blocks] in ``cache`` [pages x block, D], in
    table order: one gather row a page instead of one a token (a row of 576 bytes is one DMA descriptor each)."""
    d = cache.shape[-1]
    t = table.to(torch.int64).clamp(min=0)
    return cache.view(-1, block * d).index_select(0, t.reshape(-1)).view(t.shape[0], t.shape[1] * block, d)


def read_groups(cache: torch.Tensor, first_slots: torch.Tensor, ratio: int) -> torch.Tensor:
    """[R, n x ratio, D] rows of the ``ratio``-slot groups starting at ``first_slots`` [R, n] (multiples of ratio)."""
    d = cache.shape[-1]
    rows = torch.div(first_slots.to(torch.int64), ratio, rounding_mode="floor")
    return cache.view(-1, ratio * d).index_select(0, rows.reshape(-1)).view(rows.shape[0], -1, d)


@nki.jit
def _rotational_topk(scores, k):
    from nkilib.core.topk.rotational_topk import create_rotational_topk_config, rotational_topk
    from nkilib.core.topk.rotational_topk_utils import create_topk_config

    cfg = create_topk_config(inp_shape=scores.shape, inp_dtype=getattr(nl, str(scores.dtype).split(".")[-1]), k=k,
                             sorted=False, num_programs=2)
    return rotational_topk.func(inp=scores, config=create_rotational_topk_config(inp_shape=scores.shape,
                                                                                 topk_config=cfg))


def index_topk(scores: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """[R, k] int64 the groups of each row's ``k`` best valid scores (nkilib's rotational top-k), -1 where a row has
    fewer valid groups."""
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    masked = torch.where(valid, scores, torch.full_like(scores, -TOPK_EMPTY))
    values, idx = wrap_nki(_rotational_topk)[get_logical_nc_config()](masked, k)
    return torch.where(values > -TOPK_EMPTY / 2, idx.to(torch.int64), torch.full_like(idx, -1, dtype=torch.int64))


class IndexPool(nn.Module):
    """A sparse layer's IndexPool, replicated on every rank."""

    def __init__(self, config: Hy4FlashConfig):
        super().__init__()
        self.config = config
        self.heads = config.index_n_heads
        self.group_budget = config.index_topk // INDEX_RATIO - 1
        hidden, q_lora = config.hidden_size, config.q_lora_rank
        self.q_w = nn.Parameter(torch.empty(q_lora, self.heads * INDEX_DIM, dtype=DTYPE), requires_grad=False)
        self.head_w = nn.Parameter(torch.empty(hidden, self.heads, dtype=DTYPE), requires_grad=False)
        self.key_w = nn.Parameter(torch.empty(hidden, INDEX_DIM, dtype=DTYPE), requires_grad=False)
        self.key_norm_in = nn.Parameter(torch.empty(INDEX_DIM, dtype=DTYPE), requires_grad=False)
        self.value_proj = nn.Parameter(torch.empty(INDEX_DIM, INDEX_DIM, dtype=torch.float32), requires_grad=False)
        self.gate_proj = nn.Parameter(torch.empty(INDEX_DIM, INDEX_DIM, dtype=torch.float32), requires_grad=False)
        self.ape = nn.Parameter(torch.empty(INDEX_RATIO, INDEX_DIM, dtype=torch.float32), requires_grad=False)
        # stored fp32, read in bf16 as the reference casts it (it is not one of its fp32 modules)
        self.key_norm = nn.Parameter(torch.empty(INDEX_DIM, dtype=DTYPE), requires_grad=False)
        for p in (self.q_w, self.head_w, self.key_w):
            set_weight_loader(p, weights.replicated())
        set_weight_loader(self.key_norm_in, weights.vector(DTYPE))
        for p in (self.value_proj, self.gate_proj, self.ape):
            set_weight_loader(p, weights.vector(torch.float32))
        set_weight_loader(self.key_norm, weights.vector(DTYPE))
        self.register_buffer("inv_freq", ops.inv_freq(config.sparse.rope_theta, config.sparse.qk_rope_head_dim),
                             persistent=False)
        # bound by bind_state: [slots, 128] bf16 token keys, [slots / 4, 128] group keys
        self.token_keys = None
        self.group_keys = None

    def weight_mappings(self, src: str, dst: str) -> dict:
        c = f"{src}.self_attn.compressed_indexer"
        return {f"{dst}.q_w": f"{c}.query_projection.weight", f"{dst}.head_w": f"{c}.head_weight_projection.weight",
                f"{dst}.key_w": f"{c}.token_key_projection.weight", f"{dst}.key_norm_in": f"{c}.token_key_norm.weight",
                f"{dst}.value_proj": f"{c}.key_value_projection.weight",
                f"{dst}.gate_proj": f"{c}.key_gate_projection.weight", f"{dst}.ape": f"{c}.intra_group_ape",
                f"{dst}.key_norm": f"{c}.key_norm.weight"}

    def _rope_last(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x [..., 128] with its last 64 channels rotated at ``positions`` [...] (broadcast over a heads dim)."""
        rope = self.config.sparse.qk_rope_head_dim
        cos, sin = ops.rope_tables(positions, self.inv_freq)
        if x.dim() == positions.dim() + 2:
            cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        return torch.cat([x[..., :INDEX_DIM - rope], ops.rotate(x[..., INDEX_DIM - rope:], cos, sin)], dim=-1)

    def write(self, x: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor, table: torch.Tensor,
              rows_table, block_size: int) -> None:
        """Store the step's token keys and pool every group a token of the step completes (position % 4 == 3) from
        its 4 members' keys, at the slot of its first token; other rows write the reserved slot."""
        keys = ops.rms_norm(x @ self.key_w, self.key_norm_in, self.config.rms_norm_eps)
        write_rows(self.token_keys, keys, slots)
        pos = positions.to(torch.int64)
        first = pos - (INDEX_RATIO - 1)
        member_pos = first.unsqueeze(1) + torch.arange(INDEX_RATIO, device=pos.device)
        tab = table if rows_table is None else table.index_select(0, rows_table)
        member_slots = token_slots(tab, block_size, member_pos)
        members = read_rows(self.token_keys, member_slots)
        pooled = ops.pool_group_keys(members, self.value_proj, self.gate_proj, self.ape, self.key_norm,
                                     self.config.rms_norm_eps, DTYPE)
        pooled = self._rope_last(pooled, first.clamp(min=0))
        completes = (pos % INDEX_RATIO == INDEX_RATIO - 1) & (first >= 0)
        # a group's first token sits at a slot that is a multiple of 4 (pages are whole groups), so the group keys
        # hold one row per 4 slots
        write_rows(self.group_keys, pooled,
                   torch.where(completes, torch.div(member_slots[:, 0], INDEX_RATIO, rounding_mode="floor"),
                               torch.zeros_like(first)))

    def _group_scores(self, x, q_resid, pos, table, block):
        """(scores [R, G] fp32, valid [R, G]: the complete groups before the query's own, own [R]: its own group)
        over the groups of ``table`` [R, blocks] (or one request's [1, blocks]), group g covering positions
        4g .. 4g + 3."""
        rows = x.shape[0]
        q = (q_resid @ self.q_w).view(rows, self.heads, INDEX_DIM)
        q = self._rope_last(q, pos)
        head_w = (x @ self.head_w).float() * (self.heads ** -0.5 * INDEX_DIM ** -0.5)
        # a page's groups are consecutive rows of the group keys (one row per 4 slots)
        keys = read_pages(self.group_keys, table, block // INDEX_RATIO)
        if keys.shape[0] == 1:
            keys = keys[0]
        groups_n = keys.shape[-2]
        if rows * self.heads * groups_n <= ALL_HEADS_SCORES:
            scores = ops.index_scores(q, head_w, keys)
        else:
            scores = None
            for h in range(self.heads):
                s = ops.index_scores(q[:, h:h + 1], head_w[:, h:h + 1], keys)
                scores = s if scores is None else scores + s
        own = torch.div(pos, INDEX_RATIO, rounding_mode="floor")
        valid = torch.arange(groups_n, device=pos.device).unsqueeze(0) < own.unsqueeze(1)
        return scores, valid, own

    def _chosen_groups(self, x, q_resid, pos, table, block):
        """(chosen [R, G] bool: each query's best complete groups before its own, own [R])."""
        scores, valid, own = self._group_scores(x, q_resid, pos, table, block)
        return ops.top_count_mask(scores, valid, self.group_budget), own

    def selection(self, x: torch.Tensor, q_resid: torch.Tensor, positions: torch.Tensor, table: torch.Tensor,
                  block: int) -> torch.Tensor:
        """[R, K] bool: which of one request's table [1, blocks] positions 0 .. K - 1 each query attends."""
        pos = positions.to(torch.int64)
        chosen, own = self._chosen_groups(x, q_resid, pos, table, block)
        return ops.selected_table_tokens(chosen, own, table.shape[1] * block, pos, INDEX_RATIO)

    def selected_groups(self, x, q_resid, positions, table, block) -> tuple:
        """(groups [R, index_topk / 4] int64, positions [R, index_topk] int64, valid [R, index_topk] bool): the
        groups each query attends, its chosen ones then its own, compacted (ops.compact_groups) so the attention
        gathers index_topk rows, not the context, and their tokens."""
        pos = positions.to(torch.int64)
        groups_n = table.shape[1] * block // INDEX_RATIO
        if (KERNEL_TOPK_MIN_ROWS <= x.shape[0] <= KERNEL_TOPK_ROWS and groups_n <= KERNEL_TOPK_GROUPS
                and can_run_kernel(x)):
            # a decode's few rows: nkilib's top-k hands back the groups themselves (in no order: the attention does
            # not care), which the threshold search and the compaction take four times as long to find
            scores, valid, own = self._group_scores(x, q_resid, pos, table, block)
            picked = index_topk(scores, valid, self.group_budget)
            picked = torch.cat([picked, own.unsqueeze(1)], dim=1)
            tokens = (picked.unsqueeze(-1) * INDEX_RATIO + torch.arange(INDEX_RATIO, device=pos.device)).flatten(1)
            return picked, tokens, (tokens >= 0) & (tokens <= pos.unsqueeze(1))
        chosen, own = self._chosen_groups(x, q_resid, pos, table, block)
        return ops.compact_groups(chosen, own, pos, self.group_budget, INDEX_RATIO)


class Attention(nn.Module):
    def __init__(self, config: Hy4FlashConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.name = config.layer_name(layer_idx)
        self.kind = kind = config.kind(layer_idx)
        self.sliding = kind.sliding_window is not None
        self.tp_group = get_attn_tp_group()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.attn_tp_group.world_size
        if kind.num_heads % tp:
            raise ValueError(f"{kind.num_heads} heads of layer {layer_idx} do not shard over an attention TP of {tp}")
        self.heads = heads = kind.num_heads // tp
        hidden, q_lora, latent = config.hidden_size, config.q_lora_rank, config.kv_lora_rank
        rope, nope, v = kind.qk_rope_head_dim, kind.qk_nope_head_dim, kind.v_head_dim
        # layer 0 is bf16 in the FP8 checkpoint; every later layer's q_b, gate and o projections are FP8 blocks
        self.quantized = bool(config.quant_method) and layer_idx > 0
        self.q_a = nn.Parameter(torch.empty(hidden, q_lora, dtype=DTYPE), requires_grad=False)
        self.q_a_norm = RMSNorm(q_lora, config.rms_norm_eps)
        self.q_b = nn.Parameter(torch.empty(q_lora, heads * kind.qk_head_dim, dtype=DTYPE), requires_grad=False)
        self.kv_a = nn.Parameter(torch.empty(hidden, latent + rope, dtype=DTYPE), requires_grad=False)
        self.kv_a_norm = RMSNorm(latent, config.rms_norm_eps)
        self.w_uk = nn.Parameter(torch.empty(heads, nope, latent, dtype=DTYPE), requires_grad=False)
        self.w_uv = nn.Parameter(torch.empty(heads, v, latent, dtype=DTYPE), requires_grad=False)
        self.gate = nn.Parameter(torch.empty(hidden, heads * v, dtype=DTYPE), requires_grad=False)
        self.o = nn.Parameter(torch.empty(heads * v, hidden, dtype=DTYPE), requires_grad=False)
        self.sinks = nn.Parameter(torch.empty(heads, dtype=torch.float32), requires_grad=False)
        self.post_norm = RMSNorm(hidden, config.rms_norm_eps)
        set_weight_loader(self.q_a, weights.replicated())
        set_weight_loader(self.kv_a, weights.replicated())
        set_weight_loader(self.q_b, weights.head_rows(heads * kind.qk_head_dim, self.quantized))
        set_weight_loader(self.gate, weights.head_rows(heads * v, self.quantized))
        set_weight_loader(self.o, weights.head_cols(heads * v, self.quantized))
        set_weight_loader(self.w_uk, weights.kv_b(heads, nope, v, "k"))
        set_weight_loader(self.w_uv, weights.kv_b(heads, nope, v, "v"))
        set_weight_loader(self.sinks, weights.head_vector(heads))
        self.register_buffer("inv_freq", ops.inv_freq(kind.rope_theta, rope), persistent=False)
        self.indexer = None if self.sliding else IndexPool(config)
        # bound by bind_kv_cache: the layer's K and V pool buffers as rows [slots, 288], each half of an MLA row
        self.k_cache = None
        self.v_cache = None

    def weight_mappings(self, src: str, dst: str) -> dict:
        a = f"{src}.self_attn"
        q = self.quantized
        m = {f"{dst}.q_a": f"{a}.q_a_proj.weight", f"{dst}.q_a_norm.weight": f"{a}.q_a_layernorm.weight",
             f"{dst}.q_b": weights.keys(f"{a}.q_b_proj.weight", q),
             f"{dst}.kv_a": f"{a}.kv_a_proj_with_mqa.weight", f"{dst}.kv_a_norm.weight": f"{a}.kv_a_layernorm.weight",
             f"{dst}.w_uk": f"{a}.kv_b_proj.weight", f"{dst}.w_uv": f"{a}.kv_b_proj.weight",
             f"{dst}.gate": weights.keys(f"{a}.linear_gate.weight", q),
             f"{dst}.o": weights.keys(f"{a}.o_proj.weight", q),
             f"{dst}.sinks": f"{a}.learnable_sink_param", f"{dst}.post_norm.weight": f"{a}.post_proj_norm.weight"}
        if self.indexer is not None:
            m.update(self.indexer.weight_mappings(src, f"{dst}.indexer"))
        return m

    # -----------------------------------------------------------------------------------------------------------------

    def _project(self, x: torch.Tensor, positions: torch.Tensor):
        """(the absorbed query [R, heads, latent], its RoPE part [R, heads, rope], q_resid [R, q_lora], the rows'
        MLA rows [R, latent + rope], the output gate [R, heads, v])."""
        kind, latent = self.kind, self.config.kv_lora_rank
        rows = x.shape[0]
        cos, sin = ops.rope_tables(positions, self.inv_freq)
        q_resid = self.q_a_norm(x @ self.q_a)
        q = (q_resid @ self.q_b).view(rows, self.heads, kind.qk_head_dim)
        q_nope, q_rope = q[..., :kind.qk_nope_head_dim], q[..., kind.qk_nope_head_dim:]
        q_rope = ops.rotate(q_rope, cos.unsqueeze(1), sin.unsqueeze(1))
        ckv = x @ self.kv_a
        k_rope = ops.rotate(ckv[:, latent:], cos, sin)
        compact = torch.cat([self.kv_a_norm(ckv[:, :latent]), k_rope], dim=-1)
        q_latent = torch.einsum("rhd,hdl->rhl", q_nope, self.w_uk)
        gate = torch.sigmoid((x @ self.gate).view(rows, self.heads, kind.v_head_dim).float())
        return q_latent, q_rope, q_resid, compact, gate

    def _output(self, attended: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        values = torch.einsum("rhl,hdl->rhd", attended.to(DTYPE), self.w_uv)
        values = (values.float() * gate).to(DTYPE)
        out = values.reshape(values.shape[0], -1) @ self.o
        return self.post_norm(neuron_collectives.all_reduce(self.attn_tp_group, out))

    def prepare_decode_kernel(self, hc) -> None:
        """A sliding layer's decode attention kernel's derived weights (kernels/attn_swa.py), as buffers: w_uv as
        [heads, latent, v], the RoPE pair swap with its sign, and ``hc``'s (the layer's attn_hc) fn transposed in its
        bf16 part and the bf16 rest."""
        fnT = hc.fn.detach().t().contiguous()
        hi = fnT.to(DTYPE)
        self.register_buffer("k_fn_hi", hi, persistent=False)
        self.register_buffer("k_fn_lo", (fnT - hi.float()).to(DTYPE), persistent=False)
        self.register_buffer("k_w_uvT", self.w_uv.detach().transpose(1, 2).contiguous(), persistent=False)
        rope = self.kind.qk_rope_head_dim
        pm = torch.zeros(rope, rope, dtype=torch.float32, device=self.w_uv.device)
        idx = torch.arange(0, rope, 2, device=pm.device)
        pm[idx + 1, idx] = -1.0
        pm[idx, idx + 1] = 1.0
        self.register_buffer("k_pm", pm.to(DTYPE), persistent=False)

    def decode_half(self, streams: torch.Tensor, positions: torch.Tensor, md: dict, hc, ln) -> tuple:
        """(this rank's partial [T, H] before the all-reduce, iHC post [T, hc] fp32) of a sliding layer's decode: the
        streams' mix, ``ln`` (input_layernorm) and the attention up to its all-reduce in one kernel
        (kernels/attn_swa.py); the rows' MLA rows are written to the pool after it."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.hy_v4_flash.kernels.attn_swa import attn_swa
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        cfg, rows = self.config, streams.shape[0]
        cos, sin = ops.rope_tables(positions, self.inv_freq)
        table = md["block_table_tensor"]
        first = md["swa_kv_pos_offset"].reshape(rows, 1).to(torch.float32)
        kernel = wrap_nki(attn_swa)[get_logical_nc_config()]
        partial, post, mla_rows = kernel(
            streams, self.k_fn_hi, self.k_fn_lo, hc.base.view(1, -1), hc.scale.view(1, -1), ln.weight.view(1, -1),
            self.q_a, self.q_a_norm.weight.view(1, -1), self.q_b, self.kv_a, self.kv_a_norm.weight.view(1, -1),
            self.w_uk, self.k_w_uvT, self.gate, self.o, self.sinks.view(1, -1),
            cos.repeat_interleave(2, dim=-1).contiguous(), sin.repeat_interleave(2, dim=-1).contiguous(), self.k_pm,
            self._read_kv_pages_raw(table, int(md["block_size"])), first, positions.reshape(rows, 1).to(torch.float32),
            eps=float(cfg.rms_norm_eps), hc_eps=float(cfg.hc_eps), hc_mag=float(cfg.hc_magnitude),
            scale=float(self.kind.scaling), window=int(self.kind.sliding_window))
        self._write_kv(mla_rows, md["slot_mapping"])
        return partial, post

    def finish_half(self, partial: torch.Tensor) -> torch.Tensor:
        """The attention's output from decode_half's partial: the all-reduce and post_proj_norm."""
        return self.post_norm(neuron_collectives.all_reduce(self.attn_tp_group, partial))

    def _write_kv(self, rows: torch.Tensor, slots: torch.Tensor) -> None:
        half = self.k_cache.shape[-1]
        if self.k_cache.dtype == KV_FP8:
            # Trn2's e4m3 tops out at 240 (the checkpoint's float8_e4m3fn type is compiled as it)
            rows = rows.float().clamp(-KV_FP8_MAX, KV_FP8_MAX).to(KV_FP8)
        write_rows(self.k_cache, rows[:, :half], slots)
        write_rows(self.v_cache, rows[:, half:], slots)

    @staticmethod
    def _rows(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """The MLA rows of read K and V halves, in bf16 (an FP8 pool's widened: no scale, the latent is normed)."""
        return torch.cat([k, v], dim=-1).to(DTYPE)

    def _read_kv(self, slots: torch.Tensor) -> torch.Tensor:
        return self._rows(read_rows(self.k_cache, slots), read_rows(self.v_cache, slots))

    def _read_kv_pages_raw(self, table: torch.Tensor, block: int) -> torch.Tensor:
        """[R, blocks x block, 576] in the pool's dtype: the K and V halves of the pages' rows side by side."""
        return torch.cat([read_pages(self.k_cache, table, block), read_pages(self.v_cache, table, block)], dim=-1)

    def _read_kv_pages(self, table: torch.Tensor, block: int) -> torch.Tensor:
        return self._rows(read_pages(self.k_cache, table, block), read_pages(self.v_cache, table, block))

    def _read_kv_groups(self, groups: torch.Tensor, table: torch.Tensor, block: int) -> torch.Tensor:
        """The MLA rows of the 4-token ``groups`` [R, n] (an empty slot -1 reads group 0) through ``table``: a
        group's 4 slots are consecutive (pages are whole groups), one gather row a group."""
        first = token_slots(table, block, groups.clamp(min=0) * INDEX_RATIO)
        return self._rows(read_groups(self.k_cache, first, INDEX_RATIO), read_groups(self.v_cache, first, INDEX_RATIO))

    def _attend(self, q_latent, q_rope, keys, allowed):
        return ops.mla_attend(q_latent, q_rope, keys, allowed, self.sinks, self.kind.scaling,
                              self.config.kv_lora_rank)

    # -----------------------------------------------------------------------------------------------------------------

    def prefill(self, x: torch.Tensor, positions: torch.Tensor, md: dict, prior: torch.Tensor) -> torch.Tensor:
        """A prefill chunk's attention output: x [T, H] the normed rows of one request, positions [T] (padding rows
        repeat the last real one), md its metadata (block table [1, blocks], slot mapping), prior [1, 1] its cached
        length."""
        q_latent, q_rope, q_resid, compact, gate = self._project(x, positions)
        table, block = md["block_table_tensor"], int(md["block_size"])
        self._write_kv(compact, md["slot_mapping"])
        pos = positions.to(torch.int64)
        rows = x.shape[0]
        if self.sliding:
            window = self.kind.sliding_window
            start = prior.reshape(()).to(torch.int64) - window
            key_pos = start + torch.arange(window + rows, device=pos.device)
            allowed = (key_pos.unsqueeze(0) <= pos.unsqueeze(1)) & (key_pos.unsqueeze(0) > pos.unsqueeze(1) - window)
            allowed = allowed & (key_pos >= 0).unsqueeze(0)
        else:
            context = table.shape[1] * block
            self.indexer.write(x, pos, md["slot_mapping"], table, None, block)
            if context > MASKED_PREFILL_CONTEXT:
                return self._output(self._prefill_gathered(x, q_latent, q_rope, q_resid, pos, table, block), gate)
            allowed = self.indexer.selection(x, q_resid, pos, table, block)
            return self._output(self._attend(q_latent, q_rope, self._read_kv_pages(table, block)[0], allowed), gate)
        keys = self._read_kv(token_slots(table, block, key_pos.clamp(min=0).unsqueeze(0))[0])
        return self._output(self._attend(q_latent, q_rope, keys, allowed), gate)

    def _prefill_gathered(self, x, q_latent, q_rope, q_resid, pos, table, block):
        """A sparse layer's prefill over a long table: in blocks of PREFILL_QUERY_BLOCK queries, each query's
        IndexPool tokens gathered (its index_topk keys, not the table's), so the work and the transients of a chunk do
        not grow with the context."""
        outs = []
        for lo in range(0, x.shape[0], PREFILL_QUERY_BLOCK):
            hi = min(lo + PREFILL_QUERY_BLOCK, x.shape[0])
            picked, _, allowed = self.indexer.selected_groups(x[lo:hi], q_resid[lo:hi], pos[lo:hi], table, block)
            keys = self._read_kv_groups(picked, table, block)
            outs.append(self._attend(q_latent[lo:hi], q_rope[lo:hi], keys, allowed))
        return torch.cat(outs, dim=0)

    def decode(self, x: torch.Tensor, positions: torch.Tensor, md: dict) -> torch.Tensor:
        """A decode step's attention output: x [B, H], positions [B]; md the layer's metadata (a sliding layer's
        table trimmed to its window, with ``swa_kv_pos_offset`` the position of its first column)."""
        q_latent, q_rope, q_resid, compact, gate = self._project(x, positions)
        table, block = md["block_table_tensor"], int(md["block_size"])
        self._write_kv(compact, md["slot_mapping"])
        pos = positions.to(torch.int64)
        rows = x.shape[0]
        width = table.shape[1] * block
        if self.sliding:
            # the trimmed table's pages hold positions offset .. offset + width - 1 in order
            offset = md["swa_kv_pos_offset"].reshape(rows, 1).to(torch.int64)
            key_pos = offset + torch.arange(width, device=pos.device).unsqueeze(0)
            window = self.kind.sliding_window
            allowed = (key_pos <= pos.unsqueeze(1)) & (key_pos > pos.unsqueeze(1) - window)
            keys = self._read_kv_pages(table, block)
        else:
            self.indexer.write(x, pos, md["slot_mapping"], table, torch.arange(rows, device=pos.device), block)
            # the IndexPool's index_topk tokens a row, gathered: a decode reads them, not its graph's whole context
            picked, _, allowed = self.indexer.selected_groups(x, q_resid, pos, table, block)
            keys = self._read_kv_groups(picked, table, block)
        return self._output(self._attend(q_latent, q_rope, keys, allowed), gate)
