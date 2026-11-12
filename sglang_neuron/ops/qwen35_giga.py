# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5 giga decode: every layer of ``num_steps`` greedy decode steps in one NKI kernel call.

``qwen35_hybrid_multistep`` (models/qwen35/kernels/giga/multistep.py, models/qwen35/kernels/PROVENANCE.md "Hosting")
runs the whole hybrid stack (45 GDN + 15 gated attention layers, MoE in every layer) for this rank's token rows, then
the final norm, this rank's vocabulary slice of the LM head and the greedy argmax over every rank's slice. It is hosted
here as a phase graph's one custom call (``backend.stack.wrap_nki``) with:

- the per-layer weights stacked on leading layer axes (``per_layer_weights=True``: GDN index, attention index or layer
  index; ``weights/pack_model.py`` writes them);
- SGLang's pools passed whole and written in place (``state_mode`` 3): the GDN state pool's layer-major
  ``conv_state`` / ``rec_state`` (``models/qwen35/state.Qwen35StatePool.kernel_tensors``), addressed per row by
  ``gdn_slot``, and the slot KV pool's ``k_cache_t`` / ``v_cache`` (``models/qwen35/kv_pool``), addressed per row by
  ``kv_slot``. The kernel returns them after its tokens (``return_state``): those outputs are the aliases that carry
  the in-place writes back to SGLang's tensors (``megakernel.pools.PoolLayout``);
- the replica groups as static meshes (``tp_mesh``, ``ep_mesh``), rebuilt inside the trace (a ``ReplicaGroup`` cannot
  cross the HOP boundary), so every rank compiles one SPMD NEFF and the rank-dependent values (``ep_rank``,
  ``vbase``) are data;
- ``row_valid`` (padding rows routed to no expert, identical on every rank in EP-gathered order).

The router fold (``w_out_ext`` / ``wx``, the low-concurrency path for more than 128 GDN rows per group) is not hosted:
a replica's decode rows stay within 128 (``Qwen35Layout.row_problems``).

The MXFP8 expert weights travel as int32 (four packed e4m3 bytes) and are viewed as ``float8_e4m3fn_x4`` inside the
trace. There is no torch fallback: the graph is only built where the kernel imports.
"""
from __future__ import annotations

import importlib

from sglang_neuron.backend.stack import wrap_nki

KERNEL_PACKAGE = "sglang_neuron.models.qwen35.kernels.giga"
PREFILL_MODULE = "sglang_neuron.models.qwen35.kernels.serve.serve_prefill"

# state_mode 3: pools + per-row slots (multistep.py STATE_SLOT)
STATE_SLOT = 3

# Kernel argument -> the stacked weight the graph holds, in the kernel's positional order (multistep.py).
WEIGHTS = (
    "embed_table", "in_ln_w", "w_proj_p", "conv_w", "a_log", "dt_bias", "norm_w", "w_out", "w_qkv", "q_norm_w",
    "k_norm_w", "w_gate_p", "cos_tab", "sin_tab", "w_o", "post_ln_w", "w_router_p", "sel", "w_shared_p", "w_sd",
    "gate_up_w", "gate_up_scale", "down_w", "down_scale", "final_ln_w", "w_lm_p", "vbase", "ep_rank",
)

try:
    import nki
    import nki.language as nl
    from nki.collectives import ReplicaGroup

    _multistep = importlib.import_module(f"{KERNEL_PACKAGE}.multistep").qwen35_hybrid_multistep
except Exception as _err:  # noqa: BLE001 -- no nki / nkilib here; the decode graph refuses to build
    _multistep = None
    _IMPORT_ERROR = _err
else:
    _IMPORT_ERROR = None


def available() -> bool:
    return _multistep is not None


def require() -> None:
    if not available():
        raise RuntimeError(f"the Qwen3.5 giga decode needs {KERNEL_PACKAGE}.multistep, which does not "
                           f"import here: {_IMPORT_ERROR!r}")


def _trace_multistep(tok_shard, embed_table, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state,
                     w_out, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos_tab, sin_tab, k_cache_t, v_cache, pos, w_o,
                     post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,
                     final_ln_w, w_lm_p, vbase, ep_rank, gdn_slot, kv_slot, row_valid,
                     tp_mesh, ep_mesh, tp_size, ep_size, num_layers, num_steps, paged, kv_cp, cp_sink, atp_rank):
    """Rebuild the replica groups inside the trace (loops: NKI's tracer rejects comprehensions) and view the
    int32-packed MX expert weights as float8_e4m3fn_x4. ``paged``: ``kv_slot`` is the rows' block table [rows, MP] of
    page-pool caches (multistep ``block_table``, PROVENANCE.md edit 0004). ``kv_cp`` / ``cp_sink`` / ``atp_rank``:
    decode context parallelism over the replica (multistep ``kv_cp``)."""
    if gate_up_w.dtype != nl.float8_e4m3fn_x4:
        gate_up_w = gate_up_w.view(nl.float8_e4m3fn_x4)
    if down_w.dtype != nl.float8_e4m3fn_x4:
        down_w = down_w.view(nl.float8_e4m3fn_x4)
    tp_cells = []
    for cell in tp_mesh:
        tp_cells.append(list(cell))
    ep_cells = []
    for cell in ep_mesh:
        ep_cells.append(list(cell))
    tp_group = ReplicaGroup(tp_cells)
    return _multistep(
        tok_shard, embed_table, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out,
        w_qkv, q_norm_w, k_norm_w, w_gate_p, cos_tab, sin_tab, k_cache_t, v_cache, pos, w_o, post_ln_w,
        w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, final_ln_w, w_lm_p,
        vbase, ep_rank,
        # GDN TP = attention TP = the DP replica (models/qwen35/config.Qwen35Layout)
        tp_group, tp_group, ReplicaGroup(ep_cells), tp_size, tp_size, ep_size, num_layers, num_steps,
        state_mode=STATE_SLOT, gdn_slot=gdn_slot, return_state=True,
        kv_slot=None if paged else kv_slot, kv_write="flat", per_layer_weights=True, row_valid=row_valid,
        block_table=kv_slot if paged else None, kv_cp=kv_cp, cp_sink=cp_sink, atp_rank=atp_rank,
    )


if _multistep is not None:

    @nki.jit
    def _multistep_jit(
        tok_shard, embed_table, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out,
        w_qkv, q_norm_w, k_norm_w, w_gate_p, cos_tab, sin_tab, k_cache_t, v_cache, pos, w_o, post_ln_w, w_router_p,
        sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, final_ln_w, w_lm_p, vbase, ep_rank,
        gdn_slot, kv_slot, row_valid,
        tp_mesh: tuple = (),
        ep_mesh: tuple = (),
        tp_size: int = 0,
        ep_size: int = 0,
        num_layers: int = 0,
        num_steps: int = 1,
        paged: bool = False,
    ):
        return _trace_multistep(
            tok_shard, embed_table, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out,
            w_qkv, q_norm_w, k_norm_w, w_gate_p, cos_tab, sin_tab, k_cache_t, v_cache, pos, w_o, post_ln_w,
            w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, final_ln_w, w_lm_p,
            vbase, ep_rank, gdn_slot, kv_slot, row_valid,
            tp_mesh, ep_mesh, tp_size, ep_size, num_layers, num_steps, paged, 1, None, None)

    @nki.jit
    def _multistep_jit_cp(
        tok_shard, embed_table, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out,
        w_qkv, q_norm_w, k_norm_w, w_gate_p, cos_tab, sin_tab, k_cache_t, v_cache, pos, w_o, post_ln_w, w_router_p,
        sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, final_ln_w, w_lm_p, vbase, ep_rank,
        gdn_slot, kv_slot, row_valid, cp_sink, atp_rank,
        tp_mesh: tuple = (),
        ep_mesh: tuple = (),
        tp_size: int = 0,
        ep_size: int = 0,
        num_layers: int = 0,
        num_steps: int = 1,
        kv_cp: int = 1,
    ):
        """The decode context-parallel multistep: ``cp_sink [1, 1] int32`` the KV stacks' CP write sink row and
        ``atp_rank [1, 1] int32`` this rank's position in the replica (its CP rank). Always paged (a CP decode reads
        the shared block table)."""
        return _trace_multistep(
            tok_shard, embed_table, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out,
            w_qkv, q_norm_w, k_norm_w, w_gate_p, cos_tab, sin_tab, k_cache_t, v_cache, pos, w_o, post_ln_w,
            w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, final_ln_w, w_lm_p,
            vbase, ep_rank, gdn_slot, kv_slot, row_valid,
            tp_mesh, ep_mesh, tp_size, ep_size, num_layers, num_steps, True, kv_cp, cp_sink, atp_rank)


def qwen35_giga_decode(tok_shard, *, weights: dict, conv_state, rec_state, k_cache_t, v_cache, pos, gdn_slot,
                       kv_slot, row_valid, tp_mesh: tuple, ep_mesh: tuple, num_layers: int, num_steps: int = 1,
                       block_table=None, kv_cp: int = 1, cp_sink=None, atp_rank=None):
    """Run ``num_steps`` decode steps. ``weights`` holds the stacked operands by the names of ``WEIGHTS``; the four
    pools are written in place. ``block_table`` [rows, MP] int32 (page-pool caches, each row's pages; ``kv_slot``
    None) replaces the per-row KV slot. ``kv_cp``: decode context parallelism over the replica, with ``cp_sink [1, 1]
    int32`` the KV stacks' write sink row (kv_pool.Qwen35SlotKVPool.cp_sink_row) and ``atp_rank [1, 1] int32`` this
    rank's position in the replica (its CP rank). Returns ``(tokens [T_shard, num_steps] int32, conv_state,
    rec_state, k_cache_t, v_cache)``: the pool outputs are the kernel's aliases of its inputs."""
    require()
    missing = [name for name in WEIGHTS if name not in weights]
    if missing:
        raise ValueError(f"the Qwen3.5 giga decode has no {missing}")
    if kv_cp > 1 and (block_table is None or cp_sink is None or atp_rank is None):
        raise ValueError(f"the Qwen3.5 giga decode's KV CP degree {kv_cp} runs the paged pool with the CP write sink "
                         "row bound (kv_pool.Qwen35SlotKVPool.cp_sink_row) and the rank's position in the replica "
                         "(atp_rank)")
    w = weights
    tp_size = len(tp_mesh[0])
    operands = (
        tok_shard, w["embed_table"], w["in_ln_w"], w["w_proj_p"], w["conv_w"], conv_state, w["a_log"], w["dt_bias"],
        w["norm_w"], rec_state, w["w_out"], w["w_qkv"], w["q_norm_w"], w["k_norm_w"], w["w_gate_p"], w["cos_tab"],
        w["sin_tab"], k_cache_t, v_cache, pos, w["w_o"], w["post_ln_w"], w["w_router_p"], w["sel"], w["w_shared_p"],
        w["w_sd"], w["gate_up_w"], w["gate_up_scale"], w["down_w"], w["down_scale"], w["final_ln_w"], w["w_lm_p"],
        w["vbase"], w["ep_rank"], gdn_slot, kv_slot if block_table is None else block_table, row_valid,
    )
    if kv_cp > 1:
        # the CP entry: the write sink row and the rank's position are inputs; always the paged pool
        out = wrap_nki(_multistep_jit_cp)[1](
            *operands, cp_sink, atp_rank,
            tp_mesh=tp_mesh, ep_mesh=ep_mesh, tp_size=tp_size, ep_size=len(ep_mesh[0]), num_layers=num_layers,
            num_steps=num_steps, kv_cp=kv_cp,
        )
    else:
        out = wrap_nki(_multistep_jit)[1](
            *operands,
            tp_mesh=tp_mesh, ep_mesh=ep_mesh, tp_size=tp_size, ep_size=len(ep_mesh[0]), num_layers=num_layers,
            num_steps=num_steps, paged=block_table is not None,
        )
    if not isinstance(out, (tuple, list)) or len(out) != 5:
        raise RuntimeError("the Qwen3.5 giga decode returned no pool aliases (return_state): its in-place state and KV "
                           "writes would not reach SGLang's pools")
    return tuple(out)


# ----------------------------------------------------------------------------------------------------------- prefill

# The prefill segment's per-layer operands (models/qwen35/kernels/serve/serve_prefill.py), in its positional order,
# each the segment's slice of the model's stack: GDN-kind, attention-kind and layer-indexed.
PREFILL_WEIGHTS = (
    "in_ln_g", "w_proj_p", "conv_w", "a_log", "dt_bias", "norm_w", "w_out",
    "in_ln_a", "w_qkv", "q_norm_w", "k_norm_w", "w_gate_p", "w_o",
    "cos", "sin",
)
PREFILL_MOE = ("post_ln_w", "w_router_p", "w_shared_p", "w_sd", "expert_ids", "gate_up_w", "gate_up_scale", "down_w",
               "down_scale")

try:
    _serve_prefill = importlib.import_module(PREFILL_MODULE).qwen35_serve_prefill
except Exception as _perr:  # noqa: BLE001
    _serve_prefill = None
    _PREFILL_ERROR = _perr
else:
    _PREFILL_ERROR = None


def prefill_available() -> bool:
    return _serve_prefill is not None


if _serve_prefill is not None:

    @nki.jit
    def _prefill_jit(
        x_shard, in_ln_g, w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out, in_ln_a, w_qkv, q_norm_w, k_norm_w,
        w_gate_p, w_o, cos, sin, conv_state, rec_state, k_cache_t, v_cache, kv_slot, attn_sub, post_ln_w,
        w_router_p, w_shared_p, w_sd, expert_ids, gate_up_w, gate_up_scale, down_w, down_scale,
        tp_mesh: tuple = (),
        ep_mesh: tuple = (),
        seq_len: int = 0,
        num_layers: int = 4,
        n_chunks: int = 1,
        k_tm: bool = False,
    ):
        """Rebuild the replica groups inside the trace and view the int32-packed MX experts as float8_e4m3fn_x4."""
        if gate_up_w.dtype != nl.float8_e4m3fn_x4:
            gate_up_w = gate_up_w.view(nl.float8_e4m3fn_x4)
        if down_w.dtype != nl.float8_e4m3fn_x4:
            down_w = down_w.view(nl.float8_e4m3fn_x4)
        tp_cells = []
        for cell in tp_mesh:
            tp_cells.append(list(cell))
        ep_cells = []
        for cell in ep_mesh:
            ep_cells.append(list(cell))
        tp_group = ReplicaGroup(tp_cells)
        tp_size = len(tp_mesh[0])
        return _serve_prefill(
            x_shard, in_ln_g, w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out, in_ln_a, w_qkv, q_norm_w, k_norm_w,
            w_gate_p, w_o, cos, sin, conv_state, rec_state, k_cache_t, v_cache, kv_slot, attn_sub, post_ln_w,
            w_router_p, w_shared_p, w_sd, expert_ids, gate_up_w, gate_up_scale, down_w, down_scale,
            gdn_group=tp_group, atp_group=tp_group, ep_group=ReplicaGroup(ep_cells), tp_size=tp_size,
            atp_size=tp_size, ep_size=len(ep_mesh[0]), seq_len=seq_len, num_layers=num_layers, n_chunks=n_chunks,
            k_tm=k_tm,
        )


    def _groups(tp_mesh, ep_mesh):
        tp_cells = []
        for cell in tp_mesh:
            tp_cells.append(list(cell))
        ep_cells = []
        for cell in ep_mesh:
            ep_cells.append(list(cell))
        return ReplicaGroup(tp_cells), len(tp_mesh[0]), ReplicaGroup(ep_cells), len(ep_mesh[0])

    @nki.jit
    def _prefill_paged_jit(
        x_shard, in_ln_g, w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out, in_ln_a, w_qkv, q_norm_w, k_norm_w,
        w_gate_p, w_o, cos, sin, conv_state, rec_state, k_cache_t, v_cache, kv_slot, attn_sub, post_ln_w,
        w_router_p, w_shared_p, w_sd, expert_ids, gate_up_w, gate_up_scale, down_w, down_scale, write_pages,
        valid_len,
        tp_mesh: tuple = (),
        ep_mesh: tuple = (),
        seq_len: int = 0,
        num_layers: int = 4,
        k_tm: bool = False,
    ):
        """The paged prefill of a fresh prompt's chunk (no prior): serve_prefill(paged=True)."""
        if gate_up_w.dtype != nl.float8_e4m3fn_x4:
            gate_up_w = gate_up_w.view(nl.float8_e4m3fn_x4)
        if down_w.dtype != nl.float8_e4m3fn_x4:
            down_w = down_w.view(nl.float8_e4m3fn_x4)
        tp_group, tp_size, ep_group, ep_size = _groups(tp_mesh, ep_mesh)
        return _serve_prefill(
            x_shard, in_ln_g, w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out, in_ln_a, w_qkv, q_norm_w, k_norm_w,
            w_gate_p, w_o, cos, sin, conv_state, rec_state, k_cache_t, v_cache, kv_slot, attn_sub, post_ln_w,
            w_router_p, w_shared_p, w_sd, expert_ids, gate_up_w, gate_up_scale, down_w, down_scale,
            gdn_group=tp_group, atp_group=tp_group, ep_group=ep_group, tp_size=tp_size, atp_size=tp_size,
            ep_size=ep_size, seq_len=seq_len, num_layers=num_layers, n_chunks=1, write_pages=write_pages,
            valid_len=valid_len, paged=True, k_tm=k_tm,
        )

    @nki.jit
    def _prefill_paged_prior_jit(
        x_shard, in_ln_g, w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out, in_ln_a, w_qkv, q_norm_w, k_norm_w,
        w_gate_p, w_o, cos, sin, conv_state, rec_state, k_cache_t, v_cache, kv_slot, attn_sub, post_ln_w,
        w_router_p, w_shared_p, w_sd, expert_ids, gate_up_w, gate_up_scale, down_w, down_scale, write_pages,
        valid_len, prior_pages, prior_len,
        tp_mesh: tuple = (),
        ep_mesh: tuple = (),
        seq_len: int = 0,
        num_layers: int = 4,
        k_tm: bool = False,
    ):
        """The paged prefill of a chunk after a prior of the request's pages: serve_prefill(paged=True)."""
        if gate_up_w.dtype != nl.float8_e4m3fn_x4:
            gate_up_w = gate_up_w.view(nl.float8_e4m3fn_x4)
        if down_w.dtype != nl.float8_e4m3fn_x4:
            down_w = down_w.view(nl.float8_e4m3fn_x4)
        tp_group, tp_size, ep_group, ep_size = _groups(tp_mesh, ep_mesh)
        return _serve_prefill(
            x_shard, in_ln_g, w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out, in_ln_a, w_qkv, q_norm_w, k_norm_w,
            w_gate_p, w_o, cos, sin, conv_state, rec_state, k_cache_t, v_cache, kv_slot, attn_sub, post_ln_w,
            w_router_p, w_shared_p, w_sd, expert_ids, gate_up_w, gate_up_scale, down_w, down_scale,
            gdn_group=tp_group, atp_group=tp_group, ep_group=ep_group, tp_size=tp_size, atp_size=tp_size,
            ep_size=ep_size, seq_len=seq_len, num_layers=num_layers, n_chunks=1, write_pages=write_pages,
            prior_pages=prior_pages, prior_len=prior_len, valid_len=valid_len, paged=True, k_tm=k_tm,
        )


def qwen35_giga_prefill(x_shard, *, weights: dict, conv_state, rec_state, k_cache_t, v_cache, kv_slot, attn_sub,
                        tp_mesh: tuple, ep_mesh: tuple, seq_len: int, num_layers: int, n_chunks: int,
                        paged: dict = None, k_tm: bool = False):
    """One segment of ``num_layers`` prefill layers (``weights``: the segment's slices by the names of
    ``PREFILL_WEIGHTS`` and ``PREFILL_MOE``). Returns ``(out, h_last, conv_state, rec_state, k_cache_t, v_cache)``:
    the residual rows for the next segment, this rank's last row of each prompt, and the in-place outputs.

    ``paged``: the chunked prefill from the request's state and pages (serve_prefill's paged mode), a dict of
    ``write_pages``, ``valid_len`` and, after a prior, ``prior_pages`` and ``prior_len``; ``conv_state`` /
    ``rec_state`` then also hold the initial state and ``kv_slot`` is unused.

    ``k_tm``: the pool's K pages are token-major (models/qwen35/config.KV_TM_ENV); it must match the layout of
    the ``k_cache_t`` stack this call writes."""
    if _serve_prefill is None:
        raise RuntimeError(f"the Qwen3.5 giga prefill does not import here: {_PREFILL_ERROR!r}")
    w = weights
    common = (x_shard, *(w[n] for n in PREFILL_WEIGHTS[:13]), w["cos"], w["sin"], conv_state, rec_state, k_cache_t,
              v_cache, kv_slot, attn_sub, *(w[n] for n in PREFILL_MOE))
    if paged is not None:
        if n_chunks != 1:
            raise ValueError(f"a paged prefill runs its chunk as one micro-batch; n_chunks {n_chunks}")
        if paged.get("prior_pages") is None:
            out = wrap_nki(_prefill_paged_jit)[1](
                *common, paged["write_pages"], paged["valid_len"], tp_mesh=tp_mesh, ep_mesh=ep_mesh,
                seq_len=seq_len, num_layers=num_layers, k_tm=k_tm)
        else:
            out = wrap_nki(_prefill_paged_prior_jit)[1](
                *common, paged["write_pages"], paged["valid_len"], paged["prior_pages"], paged["prior_len"],
                tp_mesh=tp_mesh, ep_mesh=ep_mesh, seq_len=seq_len, num_layers=num_layers, k_tm=k_tm)
    else:
        out = wrap_nki(_prefill_jit)[1](
            *common, tp_mesh=tp_mesh, ep_mesh=ep_mesh, seq_len=seq_len, num_layers=num_layers, n_chunks=n_chunks,
            k_tm=k_tm,
        )
    if not isinstance(out, (tuple, list)) or len(out) != 6:
        raise RuntimeError("the Qwen3.5 giga prefill returned no pool aliases: its state and KV writes would not "
                           "reach the pools")
    return tuple(out)
