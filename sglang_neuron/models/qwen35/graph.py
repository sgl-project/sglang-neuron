# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5 giga decode graph: one ``ops.qwen35_giga.qwen35_giga_decode`` call per step.

The graph holds every operand of ``qwen35_hybrid_multistep(per_layer_weights=True)`` as a buffer (``WEIGHTS``, the
packer's per-rank layout: ``kernels/weights/pack_model.py``), the static meshes of its groups, and the four
pools it writes in place, bound once SGLang built them (``bind_pools``: the layout's bound tensors). A step takes the
replica's current tokens (``input_ids [rows]``: host ids or the device sampler's futures) and the host-built row tables
(``inputs.build_decode_rows``); the graph takes this rank's token rows itself (``rank_pos``, the rank's position in its
replica, is data, so every rank runs one NEFF) and returns the greedy tokens: the kernel gives this rank's own rows,
and an all-gather over the replica (when the graph has its group) gives every row of the replica, which every TP rank
of SGLang's replica needs.

``operand_shapes`` lists every operand's per-rank shape and dtype for a layout, which the loader checks the packed
weights against and the compile tool allocates.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from sglang_neuron.ops import qwen35_giga

from .config import P, Qwen35Dims, Qwen35Layout, kv_tm_launch

# int32 carries the MXFP8 x4 blocks (four e4m3 bytes per element), viewed as float8_e4m3fn_x4 in the trace
X4 = torch.int32


def operand_shapes(dims: Qwen35Dims, layout: Qwen35Layout, *, layers: Optional[int] = None,
                   rope_len: Optional[int] = None) -> dict:
    """{operand: (shape, dtype)} of one rank (``layers``: the first layers of the model, a multiple of 4)."""
    L = layers or dims.num_layers
    L_g = sum(1 for i in range(L) if i % 4 != 3)
    L_a = L - L_g
    H, r, world = dims.hidden, layout.replica, layout.world
    vh = dims.num_v_heads // r
    kh = max(1, vh // (dims.num_v_heads // dims.num_k_heads))
    Kd, Vd = kh * dims.head_k_dim, vh * dims.head_v_dim
    nqd = dims.num_q_heads // r * dims.head_dim
    # decode context parallelism: the rank attends with every KV head of its block of each page (one head without it)
    kvh = dims.num_kv_heads if layout.kv_cp > 1 else 1
    E_L, Is, I = dims.num_experts // world, dims.shared_intermediate // world, dims.moe_intermediate
    bf16, f32, i32, u8 = torch.bfloat16, torch.float32, torch.int32, torch.uint8
    rope = rope_len or layout.rope_len
    return {
        "embed_table": ((dims.vocab, H), bf16),
        "in_ln_w": ((L, H), f32),
        "w_proj_p": ((L_g, P, H // P, 2 * Kd + 2 * Vd + 2 * vh), bf16),
        "conv_w": ((L_g, 2 * Kd + Vd, dims.conv_kernel), bf16),
        "a_log": ((L_g, vh), f32),
        "dt_bias": ((L_g, vh), f32),
        "norm_w": ((L_g, dims.head_v_dim), f32),
        "w_out": ((L_g, Vd, H), bf16),
        "w_qkv": ((L_a, H, nqd + 2 * kvh * dims.head_dim), bf16),
        "q_norm_w": ((L_a, 1, dims.head_dim), f32),
        "k_norm_w": ((L_a, 1, dims.head_dim), f32),
        "w_gate_p": ((L_a, P, H // P, nqd), bf16),
        "cos_tab": ((rope, dims.rotary_dim // 2), f32),
        "sin_tab": ((rope, dims.rotary_dim // 2), f32),
        "w_o": ((L_a, nqd, H), bf16),
        "post_ln_w": ((L, H), f32),
        "w_router_p": ((L, P, H // P, dims.num_experts), bf16),
        "sel": ((dims.num_experts, E_L), f32),
        "w_shared_p": ((L, P, H // P, 2 * Is + 1), bf16),
        "w_sd": ((L, Is, H), bf16),
        "gate_up_w": ((L, E_L, P, 2, H // 512, I), X4),
        "gate_up_scale": ((L, E_L, 16, 2, H // 512, I), u8),
        "down_w": ((L, E_L, P, I // 512, H), X4),
        "down_scale": ((L, E_L, 16, I // 512, H), u8),
        "final_ln_w": ((H,), f32),
        "w_lm_p": ((P, H // P, dims.vocab // world), bf16),
        "vbase": ((1, 1), i32),
        "ep_rank": ((1, 1), i32),
    }


# the prefill runs the model in segments of this many layers (one compiled kernel, run once per segment)
SEGMENT = 4
# prefill chunk (micro-batch) tokens: a prompt of L tokens runs L / CHUNK chunks
CHUNK = 1024


def prefill_extra_shapes(dims: Qwen35Dims, layout: Qwen35Layout, *, layers: Optional[int] = None) -> dict:
    """{operand: (shape, dtype)} the prefill takes besides the decode's operands: the per-kind input norms, the whole
    shared expert (the prefill MoE runs it replicated) and the rank's global expert ids."""
    L = layers or dims.num_layers
    L_g = sum(1 for i in range(L) if i % 4 != 3)
    H, Is = dims.hidden, dims.shared_intermediate
    f32, bf16 = torch.float32, torch.bfloat16
    return {
        "in_ln_g": ((L_g, H), f32),
        "in_ln_a": ((L - L_g, H), f32),
        "w_shared_full_p": ((L, P, H // P, 2 * Is + 1), bf16),
        "w_sd_full": ((L, Is, H), bf16),
        "expert_ids": ((1, dims.num_experts // layout.world), f32),
    }


class Qwen35DecodeGraph(nn.Module):
    """The decode graph (module docstring)."""

    qwen35_giga_decode = True
    samples_in_forward = True

    def __init__(self, dims: Qwen35Dims, layout: Qwen35Layout, *, layers: Optional[int] = None, tp_group=None,
                 position: int = 0):
        super().__init__()
        self.dims, self.layout = dims, layout
        self.num_layers = layers or dims.num_layers
        if self.num_layers % 4:
            raise ValueError(f"the giga decode runs whole 3 GDN + 1 attention groups; {self.num_layers} layers")
        self.tp_mesh = layout.tp_mesh()
        self.ep_mesh = layout.ep_mesh()
        self.tp_group = tp_group
        self.pools = None
        self.cp_sink = None
        self.register_buffer("rank_pos", torch.tensor([int(position)], dtype=torch.int32), persistent=False)

    # the model contract's layers (models/contract.declared): a checkpoint's config may leave out layer_types, which
    # the dimensions then take from full_attention_interval, so the platform does not derive them from the config
    @property
    def layer_names(self) -> list:
        kinds = self.dims.layer_types[:self.num_layers]
        return [f"layers.{i}.linear_attn" if k == "linear_attention" else f"layers.{i}.self_attn"
                for i, k in enumerate(kinds)]

    @property
    def sliding_windows(self) -> list:
        return [None] * self.num_layers

    @property
    def state_families(self) -> list:
        return ["gdn" if k == "linear_attention" else None for k in self.dims.layer_types[:self.num_layers]]

    def load_operands(self, operands: dict, device) -> None:
        """Register every operand as a buffer on ``device``, checking its shape and dtype against ``operand_shapes``."""
        want = operand_shapes(self.dims, self.layout, layers=self.num_layers,
                              rope_len=operands["cos_tab"].shape[0] if "cos_tab" in operands else None)
        missing = sorted(set(want) - set(operands))
        if missing:
            raise ValueError(f"the Qwen3.5 giga decode's operands lack {missing}")
        for name, (shape, dtype) in want.items():
            t = operands[name]
            if tuple(t.shape) != tuple(shape) or t.dtype != dtype:
                raise ValueError(f"operand {name}: {tuple(t.shape)} {t.dtype}, the kernel takes {tuple(shape)} {dtype}")
            self.register_buffer(name, t.to(device), persistent=False)

    def bind_pools(self, conv_state, rec_state, k_cache_t, v_cache, cp_sink=None) -> None:
        """The pools the kernel writes in place: the state pool's and the slot KV pool's layer-major tensors.
        ``cp_sink [1, 1] int32``: the CP write sink row of the KV stacks (kv_pool.cp_sink_row), only under decode
        context parallelism (``Qwen35Layout.kv_cp``)."""
        self.pools = (conv_state, rec_state, k_cache_t, v_cache)
        self.cp_sink = cp_sink

    def forward(self, input_ids, pos, kv_slot=None, gdn_slot=None, row_valid=None, block_table=None):
        """``kv_slot [rows, 1]`` (slot pool) or, on a paged pool (``Qwen35Layout.paged``), ``block_table [rows, MP]``:
        each row's pages, padded with page 0."""
        if self.pools is None:
            raise RuntimeError("the Qwen3.5 giga decode graph has no pools bound (bind_pools)")
        if (block_table is None) == (kv_slot is None):
            raise ValueError("a decode step takes the rows' KV slots or their block table, not both or neither")
        conv_state, rec_state, k_cache_t, v_cache = self.pools
        replica = self.layout.replica
        shard = input_ids.shape[0] // replica
        tok_shard = input_ids.to(torch.int32).reshape(replica, shard).index_select(0, self.rank_pos).reshape(shard, 1)
        weights = {name: getattr(self, name) for name in qwen35_giga.WEIGHTS}
        tokens, _conv, _rec, _k, _v = qwen35_giga.qwen35_giga_decode(
            tok_shard, weights=weights, conv_state=conv_state, rec_state=rec_state, k_cache_t=k_cache_t,
            v_cache=v_cache, pos=pos, gdn_slot=gdn_slot, kv_slot=kv_slot, row_valid=row_valid,
            tp_mesh=self.tp_mesh, ep_mesh=self.ep_mesh, num_layers=self.num_layers, num_steps=1,
            block_table=block_table, kv_cp=self.layout.kv_cp, cp_sink=self.cp_sink,
            atp_rank=self.rank_pos.reshape(1, 1))
        if self.tp_group is None:
            return {"token_ids": tokens.reshape(-1)}
        from sglang_neuron.parallel import collectives

        # every replica row's token, in row order (rank position t holds rows t * T_shard ...)
        return {"token_ids": collectives.all_gather(self.tp_group, tokens.reshape(-1), dim=0)}


# the stacked operand each prefill weight is a segment's slice of, and its kind: "g" GDN-indexed, "a" attention-indexed,
# "l" layer-indexed (PREFILL_WEIGHTS / PREFILL_MOE apart from the shared cos / sin / expert_ids)
_SEGMENT_SLICES = {
    "in_ln_g": ("in_ln_g", "g"), "w_proj_p": ("w_proj_p", "g"), "conv_w": ("conv_w", "g"), "a_log": ("a_log", "g"),
    "dt_bias": ("dt_bias", "g"), "norm_w": ("norm_w", "g"), "w_out": ("w_out", "g"),
    "in_ln_a": ("in_ln_a", "a"), "w_qkv": ("w_qkv", "a"), "q_norm_w": ("q_norm_w", "a"),
    "k_norm_w": ("k_norm_w", "a"), "w_gate_p": ("w_gate_p", "a"), "w_o": ("w_o", "a"),
    "post_ln_w": ("post_ln_w", "l"), "w_router_p": ("w_router_p", "l"), "w_shared_p": ("w_shared_full_p", "l"),
    "w_sd": ("w_sd_full", "l"), "gate_up_w": ("gate_up_w", "l"), "gate_up_scale": ("gate_up_scale", "l"),
    "down_w": ("down_w", "l"), "down_scale": ("down_scale", "l"),
}


class Qwen35PrefillSegment(nn.Module):
    """One ``SEGMENT``-layer group of the giga prefill (3 GDN + 1 attention layers): its kernel call and its GDN
    hand-off. Every segment is the same computation of its own inputs, so the prefill family's modular compile builds
    one segment and reuses it for the others (``Qwen35PrefillGraph``):

    - its weights are buffers that view the segment's slices of the decode graph's stacked operands (shared, not
      copied), each a graph input of its own;
    - its first GDN layer and first attention layer are data (``gdn_base`` / ``attn_base``), from which it computes
      its GDN hand-off rows of the prompt's state slot and its attention layers' KV rows of the prompt's page."""

    def __init__(self, index: int):
        super().__init__()
        self.index = int(index)

    def bind(self, graph: "Qwen35PrefillGraph", device) -> None:
        ng, na = SEGMENT - SEGMENT // 4, SEGMENT // 4
        first = {"g": self.index * ng, "a": self.index * na, "l": self.index * SEGMENT}
        width = {"g": ng, "a": na, "l": SEGMENT}
        for name, (stacked, kind) in _SEGMENT_SLICES.items():
            self.register_buffer(name, getattr(graph, stacked)[first[kind]:first[kind] + width[kind]],
                                 persistent=False)
        self.register_buffer("gdn_base", torch.tensor([first["g"]], dtype=torch.int64, device=device),
                             persistent=False)
        self.register_buffer("attn_base", torch.tensor([first["a"]], dtype=torch.int64, device=device),
                             persistent=False)

    def forward(self, x, conv_pool, rec_pool, k_cache_t, v_cache, kv_slot, gdn_slot, shared: dict, *, layers_g: int,
                tp_mesh: tuple, ep_mesh: tuple, seq_len: int, n_chunks: int, paged: Optional[dict] = None):
        ng, na = SEGMENT - SEGMENT // 4, SEGMENT // 4
        sp = conv_pool.shape[0] // layers_g
        # the segment's GDN layers' rows of the prompt's slot in the layer-major pools: (g0 + i) * Sp + slot
        layer_rows = (torch.arange(ng, device=gdn_slot.device, dtype=torch.int64) + self.gdn_base) * sp
        rows = layer_rows + gdn_slot.to(torch.int64)
        # the kernel writes the hand-off rows in place: each segment's buffers are its own rows gathered from the
        # pools (a constant buffer, e.g. zeros, is one tensor the compiler shares between the segments, and every
        # segment's scatter would then read the last segment's states)
        conv_h, rec_h = conv_pool.index_select(0, rows), rec_pool.index_select(0, rows)
        weights = {name: getattr(self, name) for name in _SEGMENT_SLICES}
        weights.update(shared)
        seg = None
        if paged is not None:
            # the segment's attention layers' rows of the chunk's page tables; the hand-off rows are also the initial
            # state (the request's slot: a restored checkpoint, the previous chunk's end, or a cleared slot)
            a_rows = torch.arange(na, device=gdn_slot.device, dtype=torch.int64) + self.attn_base
            seg = {"write_pages": paged["write_pages"].index_select(0, a_rows), "valid_len": paged["valid_len"]}
            if paged.get("prior_pages") is not None:
                seg["prior_pages"] = paged["prior_pages"].index_select(0, a_rows)
                seg["prior_len"] = paged["prior_len"]
            kv = kv_slot
        else:
            kv = kv_slot.index_select(0, torch.arange(na, device=kv_slot.device, dtype=torch.int64) + self.attn_base)
        x, h_last, conv_h, rec_h, _k, _v = qwen35_giga.qwen35_giga_prefill(
            x, weights=weights, conv_state=conv_h, rec_state=rec_h, k_cache_t=k_cache_t, v_cache=v_cache,
            kv_slot=kv, attn_sub=shared["attn_sub"], tp_mesh=tp_mesh, ep_mesh=ep_mesh, seq_len=seq_len,
            num_layers=SEGMENT, n_chunks=n_chunks, k_tm=kv_tm_launch(),
            **({} if seg is None else {"paged": seg}))
        conv_pool.index_copy_(0, rows, conv_h)
        rec_pool.index_copy_(0, rows, rec_h)
        if paged is not None:
            # the state the forward ends with is also the radix checkpoint of its chunk boundary (SGLang's tracking
            # slot, or the pool's sink when the batch asks for none)
            track = layer_rows + paged["track_slot"].to(torch.int64)
            conv_pool.index_copy_(0, track, conv_h)
            rec_pool.index_copy_(0, track, rec_h)
        return x, h_last


class _PrefillStage(nn.Module):
    """A module scope around one of the prefill graph's own steps (its embedding, its first token)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


class Qwen35PrefillGraph(nn.Module):
    """The giga prefill of one exactly-``seq_len``-token prompt per replica (the prompt group is the replica).

    The model runs in segments of ``SEGMENT`` layers, one ``Qwen35PrefillSegment`` each (``segments``: one
    ``ops.qwen35_giga.qwen35_giga_prefill`` call with the segment's slices of the decode graph's stacked operands and
    the prefill's own, ``prefill_extra_shapes``), the residual chained from segment to segment. The segments are
    submodules computing the same function of their own inputs, so the prefill family compiles in neuronx-cc's
    modular flow: the hierarchy splitter finds the repeated segment module and compiles it once
    (``sglang_model.prefill_compiler_args``). The attention KV is written in place into the slot pool at the prompt's
    page; each segment's GDN final states land in hand-off rows that the segment scatters into the prompt's state
    slot. The first token comes from this replica's last rank's last row: an all-gather over the replica, the final
    norm, this rank's vocabulary slice of the LM head and an all-gather of every rank's (max, id) over the world.
    Inputs: the prompt (``input_ids [seq_len]``), the prompt's pool rows of every attention layer
    (``kv_slot [L_a, 1]``, row ``a * Spa + page``) and its state slot (``gdn_slot [1]``)."""

    qwen35_giga_prefill = True
    samples_in_forward = True

    def __init__(self, decode: Qwen35DecodeGraph, *, world_group=None, replica: int = 0):
        super().__init__()
        self.replica = int(replica)
        self.decode = [decode]  # not a submodule: its buffers are the decode family's inputs
        self.dims, self.layout = decode.dims, decode.layout
        self.num_layers = decode.num_layers
        if self.num_layers % SEGMENT:
            raise ValueError(f"the giga prefill runs whole {SEGMENT}-layer segments; {self.num_layers} layers")
        self.tp_group, self.world_group = decode.tp_group, world_group
        self.segments = nn.ModuleList(Qwen35PrefillSegment(s) for s in range(self.num_layers // SEGMENT))
        # the embedding and the first token are submodules of their own too, so the hierarchy splitter gives them
        # their own partitions and every segment's partition is the same module definition
        self.embed = _PrefillStage(self._embed)
        self.head = _PrefillStage(self._first_token)

    # the model contract's layers: the decode graph's (Qwen35DecodeGraph.layer_names)
    @property
    def layer_names(self) -> list:
        return self.decode[0].layer_names

    @property
    def sliding_windows(self) -> list:
        return self.decode[0].sliding_windows

    @property
    def state_families(self) -> list:
        return self.decode[0].state_families

    def load_operands(self, extra: dict, device) -> None:
        """Share the decode graph's operands and register the prefill's own (checked against their shapes), then give
        every segment its views of them."""
        want = prefill_extra_shapes(self.dims, self.layout, layers=self.num_layers)
        for name, (shape, dtype) in want.items():
            t = extra[name]
            if tuple(t.shape) != tuple(shape) or t.dtype != dtype:
                raise ValueError(f"prefill operand {name}: {tuple(t.shape)} {t.dtype}, the kernel takes {tuple(shape)} "
                                 f"{dtype}")
            self.register_buffer(name, t.to(device), persistent=False)
        d = self.decode[0]
        for name in qwen35_giga.WEIGHTS:
            self.register_buffer(name, getattr(d, name), persistent=False)
        self.register_buffer("rank_pos", d.rank_pos, persistent=False)
        self.register_buffer("attn_sub", torch.zeros((1, 1), dtype=torch.int32, device=device), persistent=False)
        # this rank's replica, as data (one NEFF for every rank)
        self.register_buffer("replica_idx", torch.tensor([self.replica], dtype=torch.int64, device=device),
                             persistent=False)
        for seg in self.segments:
            seg.bind(self, device)

    def forward(self, input_ids, kv_slot=None, gdn_slot=None, positions=None, track_slot=None, write_pages=None,
                valid_len=None, prior_pages=None, prior_len=None):
        """A whole prompt into the slot pool (``kv_slot``), or on a paged pool a chunk from the request's state and
        pages (``positions`` [Lb], ``track_slot`` [1], ``write_pages`` [L_a, Lb / PG], ``valid_len`` [1, 1] and,
        after a prior, ``prior_pages`` [L_a, tier / PG] and ``prior_len`` [1, 1]: prefix.chunk_tables)."""
        d = self.decode[0]
        if d.pools is None:
            raise RuntimeError("the Qwen3.5 giga prefill graph has no pools bound (the decode graph's bind_pools)")
        conv_pool, rec_pool, k_cache_t, v_cache = d.pools
        L = input_ids.shape[0]
        paged = None
        if write_pages is not None:
            # one micro-batch: the chunk's attention reads one prior (prefix.chunk_tables)
            nch = 1
            paged = {"write_pages": write_pages, "valid_len": valid_len, "track_slot": track_slot,
                     "prior_pages": prior_pages, "prior_len": prior_len}
            kv_slot = valid_len  # unused by the paged kernel: a [1, 1] int32 placeholder operand
        else:
            nch = max(1, L // CHUNK)
        x, cos, sin = self.embed(input_ids, positions)
        layers_g = sum(1 for i in range(self.num_layers) if i % 4 != 3)
        shared = {"cos": cos, "sin": sin, "expert_ids": self.expert_ids, "attn_sub": self.attn_sub}
        h_last = None
        for seg in self.segments:
            x, h_last = seg(x, conv_pool, rec_pool, k_cache_t, v_cache, kv_slot, gdn_slot, shared, layers_g=layers_g,
                            tp_mesh=d.tp_mesh, ep_mesh=d.ep_mesh, seq_len=L, n_chunks=nch, paged=paged)
        if paged is None:
            return {"token_ids": self.head(h_last)}
        # the chunk's last valid row: row (n - 1) % T_pc of rank (n - 1) // T_pc of the replica
        t_pc = L // self.layout.replica
        last = valid_len.reshape(-1)[:1].to(torch.int64) - 1
        h_rank = x.index_select(0, torch.remainder(last, t_pc))
        return {"token_ids": self.head(h_rank, torch.div(last, t_pc, rounding_mode="floor"))}

    def _embed(self, input_ids, positions=None):
        """This rank's rows of every chunk (token c * Lc + t * T_pc + j) embedded, and the prompt's rotary tables (of
        positions 0 .. L-1, or of ``positions``)."""
        L = input_ids.shape[0]
        r = self.layout.replica
        nch = 1 if positions is not None else max(1, L // CHUNK)
        t_pc = L // nch // r
        ids = input_ids.to(torch.int64).reshape(nch, r, t_pc).index_select(1, self.rank_pos.to(torch.int64))
        if positions is None:
            return self.embed_table.index_select(0, ids.reshape(-1)), self.cos_tab[:L], self.sin_tab[:L]
        pos = positions.to(torch.int64)
        return (self.embed_table.index_select(0, ids.reshape(-1)), self.cos_tab.index_select(0, pos),
                self.sin_tab.index_select(0, pos))

    def _first_token(self, h_last, owner=None):
        """Greedy token after the prompt: the replica's last rank holds its last hidden row, or, for a padded chunk,
        rank ``owner`` of the replica holds its last valid row."""
        from sglang_neuron.parallel import collectives

        r, dp = self.layout.replica, self.layout.dp
        h = h_last.reshape(1, -1)
        if self.tp_group is not None:
            h = collectives.all_gather(self.tp_group, h, dim=0)
            h = h[-1:] if owner is None else h.index_select(0, owner.reshape(-1)[:1])
        # every replica's prompt row on every rank: the LM head is sharded over the world's vocabulary slices
        if self.world_group is not None:
            h = collectives.all_gather(self.world_group, h, dim=0).reshape(dp, r, -1)[:, 0]
        hf = h.float()
        hn = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + self.dims.rms_norm_eps)
              * (1.0 + self.final_ln_w.float())).to(torch.bfloat16)
        w = self.w_lm_p.permute(1, 0, 2).reshape(self.dims.hidden, -1)
        logits = hn.float() @ w.float()  # [dp, V / world]
        best, idx = logits.max(dim=-1)
        cand = torch.stack([best, (idx.to(torch.int32) + self.vbase.reshape(-1)).to(torch.float32)], dim=-1)
        if self.world_group is not None:
            # [world, dp, 2]: this replica's column, every rank's candidate
            cand = collectives.all_gather(self.world_group, cand.unsqueeze(0), dim=0)
            cand = cand.index_select(1, self.replica_idx).reshape(-1, 2)
        winner = cand[:, 0].argmax().reshape(1)
        return cand[:, 1].index_select(0, winner).to(torch.int32)
