# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 dense hybrid as the runtime's role graphs, over the Qwen3.5 per-layer kernels.

One class serves the prefill graph (one request's chunk per forward) and the decode graph (one token per row). Each
decoder layer is the standard pair of residuals: the input norm, then either the gated delta net or the gated
attention, then the post-attention norm and a SwiGLU MLP. Both attention kinds come from
``models/qwen35/kernels/giga`` and return this rank's partial sum of the layer output, so each is followed by an
all-reduce over the attention TP, as the MLP is. In CPU mode the kernels' own torch references run instead
(``can_run_kernel``), which is how the graph is checked against Hugging Face before anything compiles.

State. The GDN kernels read their recurrent and conv state from contiguous rows, while SGLang hands out arbitrary
slots, so the graph reads the pool's **working bank** (``StateSpec.working_rows``): bank row ``b + 1`` belongs to batch row ``b`` (row 0
is the framework's scratch), and the runtime moves slots to rows before a step and back after it (``runtime.py``). Inside the
graph each layer takes a static slice of the bank and lets the kernel write it in place through its alias output.

Attention KV. The pool is paged (``kv_pool.py``): ``k_cache_t [L_a * NPAGE, d, PG]`` and ``v_cache [L_a * NPAGE,
PG, d]``, attention layer ``a``'s page ``p`` at row ``a * NPAGE + p``, and the kernels read a request's pages through
its row of SGLang's block table, so a prompt is prefilled in page-aligned chunks and the context is the table's
length, not one slot.
"""

from __future__ import annotations

import torch
from sglang.srt.distributed.parallel_state import get_attn_tp_group
from torch import nn

from sglang_neuron.layers import ColumnParallelLinear
from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.mem_cache.kv_cache import KVSpec, LayerSpec
from sglang_neuron.models.qwen35.kernels.giga import gdn_decode_torch as gdn_ref
from sglang_neuron.models.qwen38.config import FULL_ATTENTION, LINEAR_ATTENTION, Qwen38Dims, Qwen38Layout
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.neuron_utils import can_run_kernel

DTYPE = torch.bfloat16


class RMSNorm(nn.Module):
    """The layer and final norms: zero-centred, ``normed * (1 + w)`` in fp32 with ``w`` stored around zero (HF
    ``Qwen3_5RMSNorm``). Only the GDN's gated norm is the plain ``w * normed`` (HF ``Qwen3_5RMSNormGated``), and the
    GDN kernel applies that one itself."""

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size, dtype=DTYPE))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).to(x.dtype)


class MLP(nn.Module):
    """SwiGLU: ``down(silu(gate(x)) * up(x))``, the intermediate dimension sharded over the attention TP."""

    def __init__(self, layout: Qwen38Layout):
        super().__init__()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        rows = layout.intermediate
        hidden = layout.dims.hidden
        self.gate_weight = nn.Parameter(torch.empty(hidden, rows, dtype=DTYPE))
        self.up_weight = nn.Parameter(torch.empty(hidden, rows, dtype=DTYPE))
        self.down_weight = nn.Parameter(torch.empty(rows, hidden, dtype=DTYPE))

    def forward(self, x):
        out = (torch.nn.functional.silu(x @ self.gate_weight) * (x @ self.up_weight)) @ self.down_weight
        return neuron_collectives.all_reduce(self.attn_tp_group, out)


class GatedDeltaNet(nn.Module):
    """One GDN layer: the fused kernel on this rank's heads, or its torch reference in CPU mode.

    ``bank`` is the pool's working bank for this layer, ``[rows, ...]`` per component; row ``b`` is batch row ``b``.
    """

    def __init__(self, dims: Qwen38Dims, layout: Qwen38Layout, layer_idx: int, gdn_index: int):
        super().__init__()
        self.layer_idx, self.gdn_index = layer_idx, gdn_index
        self.name = f"layers.{layer_idx}.linear_attn"
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        self.vh, self.kh = layout.gdn_v_heads, layout.gdn_k_heads
        self.head = dims.head_k_dim
        self.conv_channels = 2 * self.kh * self.head + self.vh * self.head
        width = 2 * self.kh * self.head + 2 * self.vh * self.head + 2 * self.vh
        # the kernels' layouts (weights.py): the projection row-permuted, the rest natural
        self.w_proj_p = nn.Parameter(torch.empty(self.head, dims.hidden // self.head, width, dtype=DTYPE))
        self.conv_w = nn.Parameter(torch.empty(self.conv_channels, dims.conv_kernel, dtype=DTYPE))
        self.w_out = nn.Parameter(torch.empty(self.vh * self.head, dims.hidden, dtype=DTYPE))
        self.a_log = nn.Parameter(torch.empty(self.vh, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.vh, dtype=torch.float32))
        self.norm_w = nn.Parameter(torch.ones(self.head, dtype=torch.float32))

    def _reference_operands(self):
        """The torch reference takes the natural projection layout, so undo the kernels' row permutation."""
        flat = self.w_proj_p.permute(1, 0, 2).reshape(-1, self.w_proj_p.shape[2])
        vh = self.vh
        return {"w_qkvz": flat[:, :flat.shape[1] - 2 * vh], "w_a": flat[:, -2 * vh:-vh], "w_b": flat[:, -vh:]}

    def decode(self, x, conv_bank, rec_bank):
        """One token per row. ``conv_bank`` ``[B, Cc, K-1]`` and ``rec_bank`` ``[B, 128, vh, 128]`` are this layer's
        rows of the working bank; the kernel advances them in place."""
        if not can_run_kernel(x):
            out, new_conv, new_rec = gdn_ref.gdn_decode_torch(
                x, conv_w=self.conv_w, conv_state=conv_bank, a_log=self.a_log, dt_bias=self.dt_bias,
                norm_w=self.norm_w, rec_state=gdn_ref.state_from_kernel(rec_bank), w_out=self.w_out,
                **self._reference_operands())
            conv_bank.copy_(new_conv)
            rec_bank.copy_(gdn_ref.state_to_kernel(new_rec))
        else:
            from sglang_neuron.backend.stack import wrap_nki
            from sglang_neuron.models.qwen35.kernels.giga.gdn_decode import gdn_decode_tkg

            out, _c, _r = wrap_nki(gdn_decode_tkg)[1](
                x=x, w_proj_p=self.w_proj_p, conv_w=self.conv_w, conv_state=conv_bank, a_log=self.a_log,
                dt_bias=self.dt_bias, norm_w=self.norm_w, rec_state=rec_bank, w_out=self.w_out, inplace=True)
        return neuron_collectives.all_reduce(self.attn_tp_group, out)

    def prefill(self, x, conv_bank, rec_bank, valid):
        """One sequence's chunk, ``x`` ``[L, H]``, of which the first ``valid`` rows are real and the rest is the
        runtime's padding to the bucket. Padding must not touch the state (it repeats the last real position), so
        the kernel gets ``valid_len``, which passes the state through those rows unchanged and keeps the conv window
        ending at the last real token."""
        if not can_run_kernel(x):
            # the decode reference stepped over the chunk, each step's state kept only for real rows: traceable,
            # unlike gdn_prefill_torch, whose valid_len is a Python int
            conv, rec = conv_bank, gdn_ref.state_from_kernel(rec_bank)
            outs = []
            for t in range(x.shape[0]):
                out, new_conv, new_rec = gdn_ref.gdn_decode_torch(
                    x[t:t + 1], conv_w=self.conv_w, conv_state=conv, a_log=self.a_log, dt_bias=self.dt_bias,
                    norm_w=self.norm_w, rec_state=rec, w_out=self.w_out, **self._reference_operands())
                real = t < valid
                conv = torch.where(real, new_conv, conv)
                rec = torch.where(real, new_rec, rec)
                outs.append(out)
            conv_bank.copy_(conv)
            rec_bank.copy_(gdn_ref.state_to_kernel(rec))
            out = torch.cat(outs, dim=0)
        else:
            from sglang_neuron.backend.stack import wrap_nki
            from sglang_neuron.models.qwen35.kernels.giga.gdn_prefill import gdn_prefill_tkg

            out, new_conv, new_rec = wrap_nki(gdn_prefill_tkg)[1](
                x=x, w_proj_p=self.w_proj_p, conv_w=self.conv_w, a_log=self.a_log, dt_bias=self.dt_bias,
                norm_w=self.norm_w, w_out=self.w_out, conv_state=conv_bank, rec_state=rec_bank,
                seq_len=x.shape[0], valid_len=valid.view(1, 1).to(torch.int32))
            conv_bank.copy_(new_conv)
            rec_bank.copy_(new_rec)
        return neuron_collectives.all_reduce(self.attn_tp_group, out)


class GatedAttention(nn.Module):
    """One full-attention layer over the paged KV pool: the fused kernels on this rank's heads, or a traceable torch
    path in CPU mode with the same semantics.

    The pool (``kv_pool.py``) holds every attention layer's pages in one layer-major stack: ``k_stack [L_a * NPAGE,
    d, PG]`` (K transposed per page) and ``v_stack [L_a * NPAGE, PG, d]``, layer ``a``'s page ``p`` at row
    ``a * NPAGE + p``. A request's pages are its row of SGLang's block table, so the kernels take page tables:
    ``attn_decode_tkg(block_table=...)`` in decode, ``attn_prefill_tkg(write_pages=..., prior_pages=...,
    prior_len=...)`` in prefill, which writes the chunk's pages and attends the request's earlier pages plus the
    chunk (Qwen3.5's paged mode, models/qwen35/kernels/giga/attn_prefill.py)."""

    def __init__(self, dims: Qwen38Dims, layout: Qwen38Layout, layer_idx: int, attn_index: int):
        super().__init__()
        self.layer_idx, self.attn_index = layer_idx, attn_index
        self.name = f"layers.{layer_idx}.self_attn"
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        # the kernel's head count: the rank's own query heads plus a zero one when odd (config.kernel_q_heads)
        self.nq, self.nkv = layout.kernel_q_heads, layout.kv_heads
        self.head = d = dims.head_dim
        self.rotary_dim = dims.rotary_dim
        self.rope_theta = dims.rope_theta
        self.w_qkv = nn.Parameter(torch.empty(dims.hidden, (self.nq + 2 * self.nkv) * d, dtype=DTYPE))
        self.w_gate_p = nn.Parameter(torch.empty(128, dims.hidden // 128, self.nq * d, dtype=DTYPE))
        self.w_o = nn.Parameter(torch.empty(self.nq * d, dims.hidden, dtype=DTYPE))
        # zero-centred, as HF stores them and as the kernel reads them (it applies 1 + w)
        self.q_norm_w = nn.Parameter(torch.zeros(1, d, dtype=torch.float32))
        self.k_norm_w = nn.Parameter(torch.zeros(1, d, dtype=torch.float32))
        # bound by bind_kv_cache: the layer-major K and V page stacks, the pages per layer, the page size
        self.k_stack = None
        self.v_stack = None
        self.pages_per_layer = 0
        self.page = 0

    def rope(self, positions):
        """cos and sin of the rotary dims, as both the kernel and the reference read them (text-only mRoPE is plain
        RoPE: ``attn_decode_torch.rope_cos_sin``)."""
        inv = 1.0 / (self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32,
                                                      device=positions.device) / self.rotary_dim))
        f = positions.float()[:, None] * inv[None, :]
        emb = torch.cat([f, f], dim=-1)
        return emb.cos(), emb.sin()

    def _gate_natural(self):
        return self.w_gate_p.permute(1, 0, 2).reshape(-1, self.w_gate_p.shape[2])

    def _rows(self, table):
        """Pool rows of this layer's pages: the request's page ids (padding -1 read as page 0, finite and masked)
        offset by the layer's block."""
        return torch.clamp(table, min=0).to(torch.int32) + self.attn_index * self.pages_per_layer

    def _torch_decode(self, x, positions, table, write=None):
        """The kernels' semantics in traceable torch (CPU mode): row ``b`` writes its new K and V at
        ``positions[b]`` of its pages ``table[b]``, then attends positions ``0 .. positions[b]`` of them. Same rounding
        points as ``attn_decode_torch`` (projections in bf16, norms and softmax in fp32)."""
        b, d, nq, pg = x.shape[0], self.head, self.nq, self.page
        q = (x @ self.w_qkv[:, :nq * d]).view(b, nq, d)
        k = (x @ self.w_qkv[:, nq * d:(nq + 1) * d]).view(b, d)
        v = (x @ self.w_qkv[:, (nq + 1) * d:(nq + 2) * d]).view(b, d)
        gate = x @ self._gate_natural()
        q = _rmsnorm_zero_centred(q, self.q_norm_w)
        k = _rmsnorm_zero_centred(k, self.k_norm_w)
        cos, sin = self.rope(positions)
        q = _rope(q, cos[:, None, :], sin[:, None, :], self.rotary_dim)
        k = _rope(k, cos, sin, self.rotary_dim)
        pos = positions.long()
        rows = self._rows(table).long()                                   # [B, MP]
        row = rows.gather(1, (pos // pg).view(-1, 1)).view(-1)            # [B] the page row of the new token
        off = pos % pg
        k_new, v_new = k.to(self.k_stack.dtype), v.to(self.v_stack.dtype)
        if write is not None:
            # a padding row must not overwrite a real position (prefill padding repeats the last real one)
            k_new = torch.where(write[:, None], k_new, self.k_stack[row, :, off])
            v_new = torch.where(write[:, None], v_new, self.v_stack[row, off, :])
        self.k_stack[row, :, off] = k_new
        self.v_stack[row, off, :] = v_new
        mp = rows.shape[1]
        keys = self.k_stack[rows].permute(0, 2, 1, 3).reshape(b, d, mp * pg).float()      # [B, d, MP*PG]
        values = self.v_stack[rows].reshape(b, mp * pg, d).float()                        # [B, MP*PG, d]
        scores = torch.matmul(q.float(), keys) * d ** -0.5
        keep = torch.arange(mp * pg, device=x.device)[None, :] <= pos[:, None]
        scores = scores.masked_fill(~keep[:, None, :], float("-inf"))
        out = torch.matmul(torch.softmax(scores, dim=-1), values).to(DTYPE).view(b, nq * d)
        out = (out.float() * torch.sigmoid(gate.float())).to(DTYPE)
        return out @ self.w_o

    def decode(self, x, positions, table):
        """One token per row; ``table`` ``[B, MP]`` the rows' page ids (SGLang's block table)."""
        if not can_run_kernel(x):
            out = self._torch_decode(x, positions, table)
        else:
            from sglang_neuron.backend.stack import wrap_nki
            from sglang_neuron.models.qwen35.kernels.giga.attn_decode import attn_decode_tkg

            cos, sin = self.rope(positions)
            out, _k, _v = wrap_nki(attn_decode_tkg)[1](
                x=x, w_qkv=self.w_qkv, q_norm_w=self.q_norm_w, k_norm_w=self.k_norm_w, w_gate_p=self.w_gate_p,
                cos=cos[:, :self.rotary_dim // 2], sin=sin[:, :self.rotary_dim // 2], k_cache_t=self.k_stack,
                v_cache=self.v_stack, pos=positions.view(-1, 1).to(torch.int32), w_o=self.w_o,
                block_table=self._rows(table))
        return neuron_collectives.all_reduce(self.attn_tp_group, out)

    def prefill(self, x, positions, md, valid):
        """One request's chunk of ``L`` rows (a whole number of pages), the first ``valid`` real. The chunk starts at
        a page boundary (SGLang's chunks and cached prefixes are whole pages), so its pages are those of its slots
        ``slot_mapping[j * PG]``, padding rows' slot 0 naming page 0; its prior is the request's pages before it."""
        table = md["block_table_tensor"][:1]
        if not can_run_kernel(x):
            # token by token over the static chunk: the decode semantics at each position, padding not written
            outs = [self._torch_decode(x[t:t + 1], positions[t:t + 1], table, write=(t < valid).view(1))
                    for t in range(x.shape[0])]
            return neuron_collectives.all_reduce(self.attn_tp_group, torch.cat(outs, dim=0))
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.qwen35.kernels.giga.attn_prefill import attn_prefill_tkg

        pg = self.page
        slots = md["slot_mapping"]
        write_pages = (slots[0::pg] // pg).to(torch.int32).view(1, -1) + self.attn_index * self.pages_per_layer
        prior_pages = self._rows(table)
        prior_len = md["cached_seq_len"].view(1, 1).to(torch.int32)
        cos, sin = self.rope(positions)
        out, _k, _v = wrap_nki(attn_prefill_tkg)[1](
            x=x, w_qkv=self.w_qkv, q_norm_w=self.q_norm_w, k_norm_w=self.k_norm_w, w_gate_p=self.w_gate_p,
            cos=cos[:, :self.rotary_dim // 2], sin=sin[:, :self.rotary_dim // 2], k_cache_t=self.k_stack,
            v_cache=self.v_stack, slot=torch.zeros(1, 1, dtype=torch.int32, device=x.device), w_o=self.w_o,
            write_pages=write_pages, prior_pages=prior_pages, prior_len=prior_len)
        return neuron_collectives.all_reduce(self.attn_tp_group, out)


def _rmsnorm_zero_centred(x, w, eps=1e-6):
    """``x / rms * (1 + w)`` in fp32, cast back (HF's query and key norms; ``attn_decode_torch._rmsnorm_zc``)."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float().view(-1))).to(x.dtype)


def _rope(t, cos, sin, rotary_dim):
    """HF rotate-half over the first ``rotary_dim`` dims; the rest pass through."""
    tr, tp = t[..., :rotary_dim].float(), t[..., rotary_dim:]
    half = rotary_dim // 2
    rotated = torch.cat([-tr[..., half:], tr[..., :half]], dim=-1)
    return torch.cat([(tr * cos + rotated * sin).to(t.dtype), tp], dim=-1)


class DecoderLayer(nn.Module):
    def __init__(self, dims: Qwen38Dims, layout: Qwen38Layout, layer_idx: int, *, gdn_index=None, attn_index=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(dims.hidden, dims.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(dims.hidden, dims.rms_norm_eps)
        self.mlp = MLP(layout)
        self.is_gdn = gdn_index is not None
        if self.is_gdn:
            self.linear_attn = GatedDeltaNet(dims, layout, layer_idx, gdn_index)
        else:
            self.self_attn = GatedAttention(dims, layout, layer_idx, attn_index)

    @property
    def mixer(self):
        return self.linear_attn if self.is_gdn else self.self_attn

    def forward(self, x, run_mixer):
        x = x + run_mixer(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen38Graph(nn.Module):
    """One role graph (``phase`` "prefill" or "decode")."""

    def __init__(self, dims: Qwen38Dims, layout: Qwen38Layout, phase: str,
                 checkpoint_prefix: str = "model.language_model."):
        super().__init__()
        if phase not in ("prefill", "decode"):
            raise ValueError(f"a role graph is a prefill or a decode, not {phase!r}")
        self.config = dims
        self.dims, self.layout, self.phase, self.prefix = dims, layout, phase, checkpoint_prefix
        self.tp_group = get_attn_tp_group()
        self.lm_head_tp_group = group = groups.get_neuron_attn_tp_group()
        self.embed_tokens = VocabDimShardedEmbedding(dims.vocab, dims.hidden, dtype=DTYPE,
                                                     tp_group=group.device_group)
        gdn_of = {g: i for i, g in enumerate(dims.gdn_layers)}
        attn_of = {a: i for i, a in enumerate(dims.attention_layers)}
        self.layers = nn.ModuleList([
            DecoderLayer(dims, layout, i, gdn_index=gdn_of.get(i),
                         attn_index=attn_of.get(i) if i not in gdn_of else None)
            for i in range(dims.num_layers)])
        self.norm = RMSNorm(dims.hidden, dims.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(dims.hidden, dims.vocab, dtype=DTYPE, gather_output=True,
                                            tp_group=group.device_group)
        # the runtime's per-layer declaration (models/contract.declared_layer_kinds): every layer attends the whole
        # context, and the GDN layers carry recurrent state of the gdn family
        self.layer_names = [layer.mixer.name for layer in self.layers]
        self.sliding_windows = [None] * dims.num_layers
        self.state_families = ["gdn" if layer.is_gdn else None for layer in self.layers]
        # bound by the host: the working bank of the state pool and the paged KV stacks
        self.conv_bank = None
        self.rec_bank = None

    # pools
    def bind_state_bank(self, conv_bank, rec_bank) -> None:
        """The pool's working bank, ``[gdn layers, rows, ...]`` per component."""
        self.conv_bank, self.rec_bank = conv_bank, rec_bank

    def bind_kv_cache(self, k_stack, v_stack, pages_per_layer: int, page: int) -> None:
        for layer in self.layers:
            if not layer.is_gdn:
                layer.self_attn.k_stack = k_stack
                layer.self_attn.v_stack = v_stack
                layer.self_attn.pages_per_layer = pages_per_layer
                layer.self_attn.page = page

    def get_kv_spec(self) -> KVSpec:
        """Only the full-attention layers hold KV; SGLang's hybrid pool maps their global ids onto its own."""
        return KVSpec(layers=[LayerSpec(name=f"layers.{i}.self_attn") for i in self.dims.attention_layers])

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        if self.conv_bank is None or self.rec_bank is None:
            raise RuntimeError("the Qwen3.8 graph has no state bank bound (bind_state_bank)")
        hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        rows = positions.shape[0] if self.phase == "decode" else 1
        # a prefill's real rows: the runtime pads the chunk to its bucket and samples the last real one
        valid = sampling_positions.view(-1)[-1] + 1
        for layer in self.layers:
            if layer.is_gdn:
                g = layer.linear_attn.gdn_index
                # bank row 0 is scratch; batch row b is bank row b + 1 (runtime.py)
                conv = self.conv_bank[g, 1:1 + rows]
                rec = self.rec_bank[g, 1:1 + rows]
                if self.phase == "decode":
                    run = lambda h, m=layer.linear_attn, c=conv, r=rec: m.decode(h, c, r)
                else:
                    run = lambda h, m=layer.linear_attn, c=conv, r=rec: m.prefill(h, c, r, valid)
            elif self.phase == "decode":
                md = attn_metadata[layer.self_attn.name]
                run = lambda h, m=layer.self_attn, t=md["block_table_tensor"]: m.decode(h, positions, t)
            else:
                md = attn_metadata[layer.self_attn.name]
                run = lambda h, m=layer.self_attn, md=md: m.prefill(h, positions, md, valid)
            hidden = layer(hidden, run)
        hidden = self.norm(hidden.index_select(0, sampling_positions))
        return self.lm_head(hidden)
