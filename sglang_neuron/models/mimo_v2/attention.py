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
(``max_context_len``) is wrong past an 8K prior (sgl-project/sglang-neuron#4)."""

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
        self.set_qkv_disk_tp(1)
        set_weight_loader(self.o_weight, weights.o_proj_loader(self.num_heads, self.v_head_dim))
        if self.sink is not None:
            set_weight_loader(self.sink, weights.sink_loader(self.num_heads))

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

    def _sink_column(self) -> Optional[torch.Tensor]:
        return None if self.sink is None else self.sink.reshape(self.num_heads, 1)

    # -- prefill ------------------------------------------------------------------------------------------------------

    def prefill(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, md: dict) -> torch.Tensor:
        """The chunk's attention output [T, H] for its normed hidden states ``x`` [T, H]."""
        import sglang_neuron.ops as NF

        q, k, v = self._split(x @ self.qkv_weight)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        k_cache, v_cache = kv_write.write_kv(self.k_cache, self.v_cache, k.transpose(0, 1), v.transpose(0, 1),
                                             md["slot_mapping"])
        out = NF.segmented_attention(
            q.transpose(0, 1).contiguous(), k_cache=k_cache, v_cache=v_cache,
            block_tables=md["block_table_tensor"], prior_tokens=md["cached_seq_len"], block_size=md["block_size"],
            kv_segment_size=md["kv_segment_size"], scale=self.scaling, sliding_window=self.sliding_window,
            sink=self._sink_column(),
        )
        return self._project_out(out.transpose(0, 1))

    # -- decode -------------------------------------------------------------------------------------------------------

    def decode(self, x: torch.Tensor, norm, cos: torch.Tensor, sin: torch.Tensor, md: dict,
               inputs: DecodeInputs) -> torch.Tensor:
        """The attention output [B, H] of one token a row; ``x`` [B, H] is the residual stream (the kernel applies the
        layer's input norm ``norm``), cos / sin [B, KV_HEAD_DIM / 2] the layer kind's tables."""
        if self.num_kv_heads != 1:
            raise ValueError(f"the MiMo-V2 decode attends one KV head per rank, not {self.num_kv_heads}")
        rows = x.shape[0]
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
        out, k_new, v_new = wrap_nki(attention_block.decode_masked)[get_logical_nc_config()](
            X=x.view(rows, span, -1), rmsnorm_X_gamma=norm.weight.unsqueeze(0).to(torch.float32),
            rmsnorm_X_eps=norm.eps, X_hidden_dim_actual=self.hidden_size, W_qkv=self.qkv_weight, cos=layout(cos),
            sin=layout(sin), K_cache=k_cache, V_cache=v_cache, sink=self._sink_column(), softmax_scale=self.scaling,
            active_blocks_table=md["block_table_tensor"], attention_mask=mask)
        # out [B, heads, d, S]; K [d, B, S] and V [B, 1, S, d] for one KV head (nkilib attention_block_tkg)
        heads = out.reshape(rows, self.num_heads, KV_HEAD_DIM, span).permute(0, 3, 1, 2).reshape(
            tokens, self.num_heads, KV_HEAD_DIM)
        k_cache, v_cache = kv_write.write_kv(k_cache, v_cache, k_new.reshape(KV_HEAD_DIM, tokens).t().reshape(
            1, tokens, KV_HEAD_DIM), v_new.reshape(1, tokens, KV_HEAD_DIM), slots)
        return self._project_out(heads), k_cache, v_cache

    def _fused_decode(self, x, norm, cos, sin, md, inputs: DecodeInputs) -> torch.Tensor:
        """The attention half in one kernel (kernels/attn_half.py), its partial all-reduced."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.attn_half import attn_half
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        rows = x.shape[0]
        out, _, _ = wrap_nki(attn_half)[get_logical_nc_config()](
            x.view(rows, 1, -1), norm.weight.unsqueeze(0).to(torch.float32), self.qkv_weight,
            cos.t().unsqueeze(-1).contiguous(), sin.t().unsqueeze(-1).contiguous(), md["block_table_tensor"],
            self.k_cache, self.v_cache, inputs.mask, self._sink_column(),
            md["slot_mapping"].view(rows, 1).to(torch.int32), self.o_weight,
            rmsnorm_X_eps=norm.eps, softmax_scale=self.scaling, v_head_dim=self.v_head_dim)
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
            K_cache=self.k_cache, V_cache=self.v_cache, sink=self._sink_column(), softmax_scale=self.scaling,
        )
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        if inputs.paged is not None:
            p = inputs.paged
            out, k_new, v_new = wrap_nki(attention_block.decode_posid)[get_logical_nc_config()](
                active_blocks_table=p.table, attention_mask=p.active_mask, pos_ids=p.pos_ids, **args)
        else:
            out, k_new, v_new = wrap_nki(attention_block.decode_masked)[get_logical_nc_config()](
                active_blocks_table=md["block_table_tensor"], attention_mask=inputs.mask, **args)
        # out [B, heads, d, 1]; K [d, B, 1] and V [B, 1, 1, d] for one KV head (nkilib attention_block_tkg)
        heads = out.reshape(rows, self.num_heads, KV_HEAD_DIM, 1)[..., 0]
        return heads, k_new.reshape(KV_HEAD_DIM, rows).t(), v_new.reshape(rows, KV_HEAD_DIM)

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


def decode_masks(attn_metadata: dict, positions: torch.Tensor, config: MiMoV2Config, num_heads: int) -> dict:
    """Each layer kind's ``DecodeInputs`` for a decode step (``{"swa": ..., "full": ...}``), built once for all of its
    layers. On a CPU none are needed (the torch path masks by position)."""
    swa_layer = next((config.layer_name(i) for i, s in enumerate(config.swa_layers) if s), None)
    full_layer = next((config.layer_name(i) for i, s in enumerate(config.swa_layers) if not s), None)
    if not can_run_kernel(positions):
        return {"swa": DecodeInputs(positions=positions), "full": DecodeInputs(positions=positions)}
    from sglang_neuron.models.gpt_oss.attention_masks import decode_attention_masks
    from sglang_neuron.ops.attention.attention_decode_mask import qk_swap_layout

    posid = full_decode_mode() == "posid" and full_layer is not None
    masked_full = None if posid else full_layer
    swa_mask = full_mask = None
    if swa_layer is not None or masked_full is not None:
        swa_mask, full_mask = decode_attention_masks(
            attn_metadata, positions, sliding_window=config.swa.sliding_window or 0, q_heads=num_heads,
            swa_layer=swa_layer, full_layer=masked_full)

    def kernel_layout(mask):
        return qk_swap_layout(mask, d_head=KV_HEAD_DIM, fp8_kv=False, fp8_packed=False)

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
    from sglang_neuron.ops.attention.attention_decode_mask import gen_attention_decode_mask, qk_swap_layout

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
    return qk_swap_layout(mask, d_head=KV_HEAD_DIM, fp8_kv=False, fp8_packed=False)


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
