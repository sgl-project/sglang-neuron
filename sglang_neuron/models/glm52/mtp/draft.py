# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 MTP draft model: layer 78 run ``num_steps`` times in one launch of the draft chain kernel.

The draft uses the target GK's sharding unchanged (rank ``r`` holds global token rows ``[2r, 2r + 2)``), so the
target's per-rank post-norm hidden state is the draft's ``previous_hidden`` with no permutation.

The model takes fixed-shape, slot-major inputs so one compiled graph serves every step and rank. The target builds
them on the device (``mtp.step.acceptance``):

- ``mtp_tokens [num_slots, spec]``: each row's step-0 input token, already shifted by one;
- ``slot_base``, ``slot_last`` and ``slot_live [num_slots]``: row 0's position, the last valid row, and liveness.

Step 0 scores rows ``0 .. last`` and writes their latent and indexer keys into the MTP caches. Each later step runs
only the draft row ``last`` at the next position and reuses step 0's indexer selection. The routed experts and the
greedy head run on that one row per slot, and the head returns the same tokens on every rank.

The chain runs the rows layout: each rank holds a column shard of ``eh_proj`` over its o_proj block group, and the
draft steps use the token-group query path, so the MTP layer keeps ``wq_b`` and ``q_absorb_w`` over the token group
even when the target shards its query projection (``mtp_query_geometry``).

Example:
    >>> model = Glm52MtpDraftModel(geo, rank_id=0, max_position=geo.context)  # doctest: +SKIP
    >>> model.load_weights("/path/to/store")  # doctest: +SKIP
    >>> drafts = model(mtp_tokens, slot_base, slot_last, slot_live, aux, rope=(cos, sin),
    ...                lm_head_weight_t=lm_head_wt)  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import numpy as np
import torch
import torch.nn as nn

from sglang_neuron.models.glm52.layout.decode.gk_geometry import Glm52GkGeometry, mtp_query_geometry
from sglang_neuron.models.glm52.layout.decode.gk_kv_cache import idx_key_dtype
from sglang_neuron.models.glm52.layout.rope_split import LO_ROWS, rope_rows

logger = logging.getLogger(__name__)

# Draft steps for each launch. The checkpoint has one MTP layer, and the chain
# runs it this many times.
NUM_STEPS = 7

# Zero rows before and after the vocabulary shard of `embed_shard_pad`. The
# kernel constant is EMBED_PAD_ROWS; this module must not import the kernel.
EMBED_PAD_ROWS = 2

# The kernel's per-step sampler row, [top_k, top_p, temperature]: greedy. The
# greedy draft head reads none of it; the kernel checks its step count.
GREEDY_SAMPLING = (64.0, 1.0, 0.0)

# Prefix of the per-rank control buffers, and of the weight buffers.
MTP_CTRL_ATTR_PREFIX = "_mtp_ctrl_"
MTP_WEIGHT_ATTR_PREFIX = "_mtp_w_"

# The kernel argument of the lm_head shard. The target model holds that shard,
# and `load_weights` takes it as an argument.
LM_HEAD_ARG = "lm_head_weight_t"

# The weight arguments of the chain kernel that the MTP weight store holds.
MTP_KERNEL_WEIGHTS = (
    "enorm_w",
    "hnorm_w",
    "eh_proj_w_t",
    "embed_shard_pad",
    "input_norm_w",
    "wq_a",
    "wkv_a",
    "q_norm_w",
    "kv_norm_w",
    "wq_b",
    "q_absorb_w",
    "out_absorb_w",
    "o_proj_w",
    "idx_wq_b",
    "idx_wk",
    "idx_k_norm_w",
    "idx_k_norm_b",
    "idx_weights_proj",
    "gamma",
    "router_weights",
    "router_bias",
    "routed_gate_up_weights",
    "routed_down_weights",
    "shared_gate_up_weights",
    "shared_down_weights",
    "shared_head_norm_w",
)

# The draft's experts: these four weights arrive as uint8 FP8 bits (the store's
# per-tensor FP8 files), with one fp32 factor per weight, [E_L, 3] and [1, 3]
# (gate, up, down).
_FP8_EXPERT_WEIGHTS = (
    "routed_gate_up_weights",
    "routed_down_weights",
    "shared_gate_up_weights",
    "shared_down_weights",
)
_FP8_SCALE_WEIGHTS = ("routed_fp8_scales", "shared_fp8_scales")

# Weights that the kernel reads as bf16. A uint16 store file of one of these
# holds bf16 bits, so the loader reinterprets it.
_BF16_WEIGHTS = frozenset(
    {
        "eh_proj_w_t",
        "embed_shard_pad",
        "wq_a",
        "wkv_a",
        "wq_b",
        "q_absorb_w",
        "out_absorb_w",
        "o_proj_w",
        "idx_wq_b",
        "idx_wk",
        "idx_weights_proj",
        "router_weights",
        LM_HEAD_ARG,
    }
)

# Weights that must stay float32. `e_score_correction_bias` reaches about 15.6,
# and a bf16 cast changes 3 of the 8 selected experts.
_FP32_WEIGHTS = frozenset({"router_bias"})

# The 16-bit integer dtypes that can hold bf16 bits. `torch.uint16` is absent on
# an old torch.
_BITS16_DTYPES = tuple(
    dtype for dtype in (getattr(torch, "uint16", None), torch.int16) if dtype
)

# The scalar arguments of the chain kernel. They are a subset of
# `Glm52GkGeometry.scalar_kwargs`. The chain has one layer, so it takes no layer
# count, no indexer period and no dense-layer count.
MTP_SCALAR_KEYS = (
    "cp_degree",
    "head_shard",
    "token_shard",
    "key_size",
    "num_heads",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "kv_lora_rank",
    "idx_n_heads",
    "idx_rope_dim",
    "index_topk",
    "spec",
    "softmax_scale",
    "kv_scale",
    "rms_norm_eps",
    "latent_norm_eps",
    "idx_norm_eps",
    "ep_size",
    "tp_size",
    "top_k",
    "routed_scaling_factor",
)

# The control tensors that the model builds for each launch.
STEP_CONTROL = (
    "mtp_positions_all",
    "cos_all",
    "sin_all",
    "idx_cos",
    "idx_sin",
    "kv_new_slot_mapping_all",
    "idx_new_slot_mapping",
    "idx_positions",
    "shared_extra_keys",
    "live_rows",
    "draft_rows",
)

# The control tensors of the kernel that do not change between launches.
STATIC_CONTROL = (
    "block_table",
    "positions_all",
    "key_offset",
    "cp_pos_offset",
    "vocab_base",
    "rank_id",
    "sampling_params_all",
)

# Per-rank helper buffers that the step builders read. They are not kernel
# arguments.
HELPER_CONTROL = (
    "my_slot",
    "cp_rank",
    "own_rows",
    "key_rank",
    "spec_rows",
    "slot_row_base",
    "step_index",
    "minus_one",
    "dead_threshold",
    "kv_pad_steps",
)


# geometry helpers
def num_sequence_slots(geo: Glm52GkGeometry) -> int:
    """Return the kernel's sequence slots, ``num_seqs * dp_size``."""
    return geo.num_seqs * geo.dp_size


def pool_block_size(geo: Glm52GkGeometry) -> int:
    """Return the positions in one pool block: ``block_size * cp_degree``."""
    return geo.block_size * geo.cp_degree


def vocab_shard(geo: Glm52GkGeometry) -> int:
    """Return the vocabulary rows of one rank."""
    return geo.vocab_size // geo.ranks


def mtp_scalar_kwargs(geo: Glm52GkGeometry) -> dict[str, object]:
    """Return the chain kernel's rank-invariant scalar arguments. The MTP caches use the target caches' packing and
    ``kv_scale``.
    """
    scalars = geo.scalar_kwargs()
    return {name: scalars[name] for name in MTP_SCALAR_KEYS}


def expected_weight_shapes(geo: Glm52GkGeometry) -> dict[str, tuple[int, ...]]:
    """Return the per-rank shape of each weight argument of the chain kernel.

    Attention and indexer weights have the GK store's per-layer shapes. Rank ``r`` holds experts ``[E_L * r, E_L * (r
    + 1))``, with ``(gate, up)`` on axis 2 of the gate-up weight. ``wq_b`` and ``q_absorb_w`` are token-group shards
    (``mtp_query_geometry``), and the model sets the ``eh_proj_w_t`` shard itself.
    """
    h, e = geo.hidden, geo.num_experts
    e_l, i, i_sh = geo.experts_per_rank, geo.moe_intermediate, geo.shared_intermediate
    v_s = vocab_shard(geo)
    q_geo = mtp_query_geometry(geo)
    return {
        "enorm_w": (1, h),
        "hnorm_w": (1, h),
        "eh_proj_w_t": (2 * h, h),
        "embed_shard_pad": (v_s + EMBED_PAD_ROWS, h),
        "input_norm_w": (1, h),
        "wq_a": (h, geo.q_lora_shard),
        "wkv_a": (h, geo.kv_row_shard),
        "q_norm_w": (1, geo.q_lora_rank),
        "kv_norm_w": (1, geo.kv_lora_rank),
        "wq_b": (q_geo.q_lora_rank, q_geo.wq_b_shard),
        "q_absorb_w": (
            q_geo.q_absorb_heads,
            q_geo.qk_nope_head_dim,
            q_geo.kv_lora_rank,
        ),
        "out_absorb_w": (geo.kv_lora_rank, geo.nh_out, geo.v_head_dim),
        "o_proj_w": (geo.nh_out * geo.v_head_dim, h),
        "idx_wq_b": (geo.q_lora_rank, geo.idx_q_shard),
        "idx_wk": (h, geo.idx_head_dim),
        "idx_k_norm_w": (1, geo.idx_head_dim),
        "idx_k_norm_b": (1, geo.idx_head_dim),
        "idx_weights_proj": (h, geo.idx_n_heads),
        "gamma": (1, h),
        "router_weights": (h, e),
        "router_bias": (1, e),
        "routed_gate_up_weights": (e_l, h, 2, i),
        "routed_down_weights": (e_l, i, h),
        "shared_gate_up_weights": (1, h, 2, i_sh),
        "shared_down_weights": (1, i_sh, h),
        "shared_head_norm_w": (1, h),
        LM_HEAD_ARG: (h, v_s),
    }


def rows_chain_supported(geo: Glm52GkGeometry) -> tuple[bool, str]:
    """Whether the vendored kernel's rows-layout draft chain supports this geometry.

    The chain supports the copy's indexer key counts and key dtypes, one latent stack, and a block-sharded o_proj
    whose block group divides the hidden size and the EP group, fits one partition tile, and gives whole 128-column
    k-tiles.

    Returns:
        ``(supported, reason)``, where ``reason`` names the first unmet condition.
    """
    from sglang_neuron.models.glm52.decode.kernel_limits import DECODE_KERNEL_PACKAGE, kernel_copy

    g = geo.oproj_degree
    copy = kernel_copy(DECODE_KERNEL_PACKAGE)
    max_idx_keys = copy.rows_chain_max_idx_keys
    if geo.idx_keys > max_idx_keys:
        return (
            False,
            f"idx_keys {geo.idx_keys} > {max_idx_keys} of the vendored rows chain "
            "(long context)",
        )
    if geo.idx_fp8_keys and "idx_fp8" not in copy.rows_chain_options:
        return (
            False,
            "the vendored rows chain does not declare fp8 indexer keys "
            "(idx_fp8 in the GK_OPTIONS of its source)",
        )
    if g <= geo.cp_degree or geo.head_shard != 1:
        return False, f"o_proj is not block-sharded (oproj {g}, cp {geo.cp_degree})"
    if geo.hidden % g or geo.ranks % g or g * geo.t_shard > 128:
        return False, f"the block group of {g} ranks cannot shard eh_proj"
    if geo.t_local != geo.spec:
        return False, f"t_local {geo.t_local} is not spec {geo.spec}"
    cols, align = geo.hidden // g, copy.rows_chain_eh_proj_align
    if cols % align:
        return (
            False,
            f"the eh_proj column blocks of {cols} columns (o_proj over {g} ranks) are not a "
            f"multiple of {align}: the vendored rows chain reads its gathered rows in "
            f"whole {align}-column k-tiles (its pick of the gathered rows) and has no SBUF-rows "
            "fallback for a narrower block",
        )
    return True, ""


def rank_constants(
    geo: Glm52GkGeometry, rank_id: int, num_steps: int = NUM_STEPS
) -> dict[str, torch.Tensor]:
    """Return the draft model's per-rank constant tensors, on CPU.

    They are registered as buffers rather than Python constants, so every rank traces the same graph. Factories name
    ``device="cpu"`` because the model runner builds the model under a meta device.
    """
    coords = geo.coords(rank_id)
    num_slots = num_sequence_slots(geo)
    t_shard, spec, s = geo.t_shard, geo.spec, int(num_steps)
    my_slot = coords["dp"] * geo.num_seqs + coords["kk"]
    # The spec rows of this rank inside its sequence, and its global token rows.
    first_row = t_shard * coords["pos_in_group"]
    static_positions = geo.positions(rank_id).astype(np.int32)
    positions_all = np.stack([static_positions + i for i in range(s)]).astype(np.int32)
    sampling = np.tile(
        np.asarray(GREEDY_SAMPLING, dtype=np.float32).reshape(1, 1, 3), (s, t_shard, 1)
    )
    return {
        # kernel arguments
        "block_table": torch.from_numpy(geo.block_table()),
        # The STATIC positions of the GK, advanced by one for each step. They
        # feed only the causal keep test, and sit after every key of the run.
        "positions_all": torch.from_numpy(np.ascontiguousarray(positions_all)),
        "key_offset": torch.from_numpy(geo.key_offset(rank_id)),
        "cp_pos_offset": torch.from_numpy(geo.cp_pos_offset(rank_id)),
        "vocab_base": torch.tensor(
            [[rank_id * vocab_shard(geo)]], dtype=torch.int32, device="cpu"
        ),
        "rank_id": torch.tensor([[rank_id]], dtype=torch.int32, device="cpu"),
        "sampling_params_all": torch.from_numpy(np.ascontiguousarray(sampling)),
        # step-builder helpers
        "my_slot": torch.tensor([my_slot], dtype=torch.int64, device="cpu"),
        "cp_rank": torch.tensor([coords["cp"]], dtype=torch.int32, device="cpu"),
        "own_rows": torch.arange(
            first_row, first_row + t_shard, dtype=torch.int32, device="cpu"
        ),
        "key_rank": torch.tensor([coords["key_rank"]], dtype=torch.int32, device="cpu"),
        "spec_rows": torch.arange(spec, dtype=torch.int32, device="cpu"),
        "slot_row_base": torch.arange(
            0, spec * num_slots, spec, dtype=torch.int32, device="cpu"
        ),
        "step_index": torch.arange(s, dtype=torch.int32, device="cpu"),
        "minus_one": torch.tensor([-1], dtype=torch.int32, device="cpu"),
        # A threshold below every local row, for a slot with no request: the
        # indexer then finds nothing resident. The cut is a TRUE position, so -1
        # is below every row.
        "dead_threshold": torch.tensor([-1], dtype=torch.int32, device="cpu"),
        # KVShare: steps 1 and later write no latent row.
        "kv_pad_steps": torch.full(
            (s - 1, geo.num_seqs_local, spec), -1, dtype=torch.int32, device="cpu"
        ),
    }


# device side: the per-rank kernel inputs
def _rope_lookup(
    rope_cos: torch.Tensor | Mapping[str, torch.Tensor],
    rope_sin: torch.Tensor | None,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cosine and sine rows at `index`: from full tables, or from split tables (`rope_split`)."""
    if isinstance(rope_cos, Mapping):
        return rope_rows(rope_cos, index)
    return rope_cos.index_select(0, index), rope_sin.index_select(0, index)


def step_inputs(
    geo: Glm52GkGeometry,
    consts: Mapping[str, torch.Tensor],
    rope_cos: torch.Tensor | Mapping[str, torch.Tensor],
    rope_sin: torch.Tensor | None,
    slot_base: torch.Tensor,
    slot_last: torch.Tensor,
    slot_live: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return one rank's per-launch control tensors (``STEP_CONTROL``), built with torch ops on the device so one graph
    serves every rank and step.

    ``rope_cos``/``rope_sin`` are the target's RoPE tables, or its two-level tables with ``rope_sin`` None.
    """
    spec, t_local, half = geo.spec, geo.t_local, geo.qk_rope_head_dim // 2
    rows = consts["spec_rows"]
    steps = consts["step_index"]
    num_steps = steps.shape[0]
    minus_one = consts["minus_one"]
    max_position = (
        rope_cos["rope_cos_hi"].shape[0] * LO_ROWS
        if isinstance(rope_cos, Mapping)
        else rope_cos.shape[0]
    )

    # This rank's own slot: its base position, its last valid row, and whether
    # a request holds it.
    my_slot = consts["my_slot"]
    base = slot_base.index_select(0, my_slot).to(torch.int32)
    last = slot_last.index_select(0, my_slot).to(torch.int32)
    live = slot_live.index_select(0, my_slot) != 0
    valid = live & (rows <= last)
    pos = base + rows

    # RoPE of row j at step i is at position base + j + i. Step 0 is also the
    # indexer's table.
    grid = pos.reshape(1, spec) + steps.reshape(num_steps, 1)
    index = torch.clamp(grid.to(torch.int64), 0, max_position - 1).reshape(-1)
    cos_rows, sin_rows = _rope_lookup(rope_cos, rope_sin, index)
    cos_all = cos_rows.reshape(num_steps, t_local, half)
    sin_all = sin_rows.reshape(num_steps, t_local, half)

    # The position-0 mask of the prologue reads only these values. The token
    # group's rows, in group order, are the slot's spec rows.
    own = base + consts["own_rows"]
    mtp_positions_all = (own.reshape(1, -1, 1) + steps.reshape(num_steps, 1, 1)).to(
        torch.int32
    )

    # Step 0 latent rows: only the CP owner of a position writes it. A rank's
    # physical block is its sequence's logical block, so the row is arithmetic.
    logical = torch.div(pos, pool_block_size(geo), rounding_mode="floor")
    owner = torch.div(pos, geo.block_size, rounding_mode="floor") % geo.cp_degree
    mine = (owner == consts["cp_rank"]) & valid & (logical < geo.local_blocks_per_seq)
    latent = logical * geo.block_size + pos % geo.block_size
    step0 = torch.where(mine, latent, minus_one).to(torch.int32)
    kv_new_slot_mapping_all = torch.cat(
        [step0.reshape(1, geo.num_seqs_local, spec), consts["kv_pad_steps"]], dim=0
    )

    # Step 0 indexer rows: the keys are DEALT with MLA's rule. Global block g belongs to key
    # rank g % cp_degree at local block g // cp_degree (Glm52GkGeometry.idx_local_index), and the
    # MTP cache reads through the identity table, so the local row is the physical row.
    bs = geo.idx_block_size
    g = torch.div(pos, bs, rounding_mode="floor")
    owned = (g % geo.idx_shard_degree) == consts["key_rank"]
    local = torch.div(g, geo.idx_shard_degree, rounding_mode="floor") * bs + pos % bs
    inside = owned & (pos >= 0) & (local < geo.idx_keys) & valid
    idx_new_slot_mapping = torch.where(inside, local, minus_one).to(torch.int32)

    # The residency cut: keys before the base position, as a TRUE position. The kernel
    # compares TRUE positions under the deal; the rank term rides key_offset.
    idx_positions = (
        torch.where(live, base, consts["dead_threshold"])
        .to(torch.int32)
        .expand(spec)
        .contiguous()
    )

    # Step 0's own rows, which the later steps attend as prior keys. A row that
    # is not valid is -1, so no later step sees it.
    own_keys = torch.where(valid, pos, minus_one).to(torch.int32)
    shared_extra_keys = own_keys.reshape(1, spec).expand(t_local, spec).contiguous()

    # The draft row of each slot, as a global token row. The value is the same
    # on every rank, because it uses only the replicated slot tensors. An empty
    # slot points at its row 0, which is a valid index.
    clamped = torch.clamp(slot_last.to(torch.int32), 0, spec - 1)
    offset = torch.where(slot_live != 0, clamped, torch.zeros_like(clamped))
    live_rows = (consts["slot_row_base"] + offset).to(torch.int32).reshape(-1, 1)
    # The rows chain's draft row of this rank's sequence, inside its token slice.
    # The slice is the rank's own slot, because the model gives each rank one
    # sequence (t_local == spec).
    draft_rows = offset.index_select(0, my_slot).to(torch.int32).reshape(1, 1)

    return {
        "mtp_positions_all": mtp_positions_all,
        "cos_all": cos_all,
        "sin_all": sin_all,
        "idx_cos": cos_all[0].contiguous(),
        "idx_sin": sin_all[0].contiguous(),
        "kv_new_slot_mapping_all": kv_new_slot_mapping_all,
        "idx_new_slot_mapping": idx_new_slot_mapping,
        "idx_positions": idx_positions,
        "shared_extra_keys": shared_extra_keys,
        "live_rows": live_rows,
        "draft_rows": draft_rows,
    }


def local_embedding_rows(
    mtp_tokens: torch.Tensor,
    embed_shard_pad: torch.Tensor,
    vocab_base: torch.Tensor,
) -> torch.Tensor:
    """Return this rank's part of the step-0 embedding of every token row, ``[num_slots * spec, hidden]``.

    The table is row-sharded with a zero row on each side of the shard; tokens outside the shard clamp onto a zero
    row, so summing over ranks gives each token's embedding exactly.
    """
    v_pad = embed_shard_pad.shape[0]
    index = mtp_tokens.reshape(-1).to(torch.int64) - vocab_base.reshape(1).to(
        torch.int64
    )
    index = torch.clamp(index + 1, 0, v_pad - 1)
    return embed_shard_pad.index_select(0, index)


def _resolve_chain() -> tuple[Callable | None, ImportError | None]:
    """Return ``(glm52_mtp_draft, None)``, or ``(None, ImportError)`` when the kernel cannot be imported.

    Resolved at construction because a compiled forward cannot import. A CPU host can still build the model and load
    weights; only a forward needs the kernel.
    """
    import importlib

    try:
        module = importlib.import_module("sglang_neuron.models.glm52.ops.mtp_draft")
    except ImportError as e:
        logger.info(
            "[glm52 MTP] the functional namespace is not available (%s). The "
            "draft model cannot run the kernel on this host.",
            e,
        )
        return None, e
    return module.glm52_mtp_draft, None


class Glm52MtpDraftModel(nn.Module):
    """One rank's MTP draft model: the chain kernel and the buffers it reads.

    Args:
        geo: The target decode's geometry; the draft uses the same sharding.
        rank_id: Global rank in ``[0, ranks)``.
        max_position: Positions the RoPE tables cover, normally ``geo.context``.
        device: Device of the buffers, weights and caches.
        dtype: Hidden-state dtype (bf16).
        num_steps: Draft steps per launch.
    """

    def __init__(
        self,
        geo: Glm52GkGeometry,
        rank_id: int,
        max_position: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        num_steps: int = NUM_STEPS,
    ) -> None:
        super().__init__()
        geo.validate()
        if geo.num_seqs_local != 1:
            raise ValueError(
                f"the draft model gives each rank one sequence, and the geometry "
                f"gives it {geo.num_seqs_local}. Use cp_degree {geo.cp_degree} with "
                "a token group that leaves one sequence for each rank."
            )
        if geo.vocab_size % geo.ranks:
            raise ValueError(
                f"vocab_size {geo.vocab_size} does not divide by the rank count "
                f"{geo.ranks}. The embedding and lm_head need a row shard."
            )
        if num_steps < 1:
            raise ValueError(f"num_steps is {num_steps}. Use 1 or more steps.")
        self.geo = geo
        self.rank_id = int(rank_id)
        self.max_position = int(max_position)
        self.device_str = str(device)
        self.dtype = dtype
        self.num_steps = int(num_steps)
        # the vendored mtp/ fork's rows chain is the only MTP chain
        supported, reason = rows_chain_supported(geo)
        if not supported:
            raise ValueError(f"the rows chain does not serve this geometry: {reason}")
        self._weight_names = MTP_KERNEL_WEIGHTS + _FP8_SCALE_WEIGHTS
        self.num_slots = num_sequence_slots(geo)
        self._meshes = geo.replica_meshes()
        # The rows chain's eh_proj column shard: this rank's position in its
        # o_proj block group (consecutive ranks) picks the columns.
        self._eh_proj_cols = geo.hidden // geo.oproj_degree
        self._eh_proj_pos = next(
            (g.index(self.rank_id) for g in self._meshes["block"] if self.rank_id in g),
            0,
        )
        self._scalars = mtp_scalar_kwargs(geo)
        self._weights_loaded = False
        self._chain_fn, self._chain_import_error = _resolve_chain()
        self._register_control()
        self._register_caches()
        coords = geo.coords(self.rank_id)
        logger.info(
            "[glm52 MTP] rank %d slot %d (cp=%d key_rank=%d) steps=%d slots=%d",
            self.rank_id,
            coords["dp"] * geo.num_seqs + coords["kk"],
            coords["cp"],
            coords["key_rank"],
            self.num_steps,
            self.num_slots,
        )

    # buffers
    def _register_control(self) -> None:
        """Register the per-rank constants."""
        tensors = rank_constants(self.geo, self.rank_id, self.num_steps)
        for name, tensor in tensors.items():
            self.register_buffer(
                MTP_CTRL_ATTR_PREFIX + name,
                tensor.to(self.device_str),
                persistent=False,
            )

    def _register_caches(self) -> None:
        """Allocate the two MTP caches, zeroed, in the target caches' formats.

        Zeroed rather than empty: an unwritten row is read as a prior key until a step writes it, and a NaN there
        would reach every draft.
        """
        geo = self.geo
        kv_cache = torch.zeros(
            (geo.num_blocks, 1, geo.block_size, geo.row_bytes),
            dtype=torch.uint8,
            device="cpu",
        )
        idx_k_cache = torch.zeros(
            (geo.idx_blocks, 1, geo.idx_block_size, geo.idx_head_dim),
            dtype=idx_key_dtype(geo),
            device="cpu",
        )
        self.bind_caches(kv_cache.to(self.device_str), idx_k_cache.to(self.device_str))

    def bind_caches(self, kv_cache: torch.Tensor, idx_k_cache: torch.Tensor) -> None:
        """Bind MTP caches that a caller allocated.

        Args:
            kv_cache: ``[blocks, 1, block_size, geo.row_bytes]`` uint8 latent cache.
            idx_k_cache: ``[idx_blocks, 1, idx_block_size, idx_head_dim]`` indexer key
                cache in MLA's block form, dealt with MLA's rule, of `idx_key_dtype`:
                bf16, or uint8 fp8 bytes for ``idx_fp8_keys``.

        Raises:
            ValueError: If a shape or a dtype is not the one of this geometry.
        """
        geo = self.geo
        shape = tuple(kv_cache.shape)
        if (
            len(shape) != 4
            or shape[0] < geo.num_blocks
            or shape[1:] != (1, geo.block_size, geo.row_bytes)
            or kv_cache.dtype != torch.uint8
        ):
            raise ValueError(
                f"the MTP latent cache is {shape} {kv_cache.dtype}. This geometry "
                f"needs [{geo.num_blocks}, 1, {geo.block_size}, {geo.row_bytes}] uint8."
            )
        key_dtype = idx_key_dtype(geo)
        idx_shape = (geo.idx_blocks, 1, geo.idx_block_size, geo.idx_head_dim)
        if tuple(idx_k_cache.shape) != idx_shape or idx_k_cache.dtype != key_dtype:
            raise ValueError(
                f"the MTP indexer cache is {tuple(idx_k_cache.shape)} "
                f"{idx_k_cache.dtype}. This geometry needs {list(idx_shape)} {key_dtype}."
            )
        self.register_buffer(
            MTP_CTRL_ATTR_PREFIX + "kv_cache", kv_cache, persistent=False
        )
        self.register_buffer(
            MTP_CTRL_ATTR_PREFIX + "idx_k_cache", idx_k_cache, persistent=False
        )

    @property
    def mtp_kv_cache(self) -> torch.Tensor:
        """The ``[blocks, 1, block_size, geo.row_bytes]`` uint8 MTP latent cache."""
        return getattr(self, MTP_CTRL_ATTR_PREFIX + "kv_cache")

    @property
    def mtp_idx_k_cache(self) -> torch.Tensor:
        """The ``[idx_blocks, 1, idx_block_size, idx_head_dim]`` MTP indexer key cache (`idx_key_dtype`)."""
        return getattr(self, MTP_CTRL_ATTR_PREFIX + "idx_k_cache")

    def control(self, name: str) -> torch.Tensor:
        """Return the per-rank control buffer `name`."""
        return getattr(self, MTP_CTRL_ATTR_PREFIX + name)

    def weight(self, name: str) -> torch.Tensor:
        """Return the weight buffer `name`, keyed by the kernel argument name."""
        return getattr(self, MTP_WEIGHT_ATTR_PREFIX + name)

    # weights
    def load_weights(
        self,
        store_dir: str,
        device: str | torch.device | None = None,
    ) -> None:
        """Load this rank's MTP weights from the weight store.

        Args:
            store_dir: Root of the GK weight store that holds the MTP files.
            device: Where to put the weights. Defaults to the model's device.

        Raises:
            FileNotFoundError: If the store has no MTP files or no per-tensor
                FP8 expert files (``.done_mtp`` / ``.done_mtp_fp8t``).
            ValueError: If the store gave no tensor, or a tensor of the wrong
                shape or dtype, for a kernel argument.
        """
        from sglang_neuron.models.glm52.layout.decode.weight_loaders_mtp import (
            MTP_WEIGHT_ARGS,
            load_mtp_fp8_tensor_expert_weights,
            load_mtp_rank_weights,
        )

        target = str(device or self.device_str)
        # the store holds the token-group query path (`mtp_query_geometry`); the experts are its FP8 files
        loaded = load_mtp_rank_weights(
            store_dir, self.geo, self.rank_id, device=target
        )
        loaded.update(
            load_mtp_fp8_tensor_expert_weights(store_dir, self.geo, self.rank_id, device="cpu")
        )
        unknown = sorted(set(MTP_WEIGHT_ARGS) - set(MTP_KERNEL_WEIGHTS) - {LM_HEAD_ARG})
        if unknown:
            logger.warning(
                "[glm52 MTP] the weight store names %s, which are not kernel "
                "arguments. The draft model does not use them.",
                unknown,
            )
        self.set_weights(loaded, device=target)
        logger.info(
            "[glm52 MTP] rank %d weights loaded from %s (%d tensors)",
            self.rank_id,
            store_dir,
            len(self._weight_names),
        )

    def set_weights(
        self,
        weights: Mapping[str, torch.Tensor],
        device: str | torch.device | None = None,
    ) -> None:
        """Check the MTP weights and register them as buffers.

        Args:
            weights: Tensors keyed by the kernel argument names of
                `MTP_KERNEL_WEIGHTS`, the four experts as uint8 FP8 bits, and
                their fp32 factors ``routed_fp8_scales`` and
                ``shared_fp8_scales``. Other keys are ignored.
            device: Where to put the weights. Defaults to the model's device.

        Raises:
            ValueError: If a tensor is missing, or has the wrong shape or dtype.
        """
        from sglang_neuron.models.glm52.layout.decode.weight_loaders_mtp import mtp_fp8t_store_spec

        target = str(device or self.device_str)
        expected = expected_weight_shapes(self.geo)
        expected.update({name: shape for name, (shape, _) in mtp_fp8t_store_spec(self.geo).items()})
        missing = [name for name in self._weight_names if name not in weights]
        if missing:
            raise ValueError(
                f"the MTP weights have no tensor for {missing}. Build the MTP files "
                "of the weight store again."
            )
        tensors = {name: weights[name] for name in self._weight_names}
        eh_proj = tensors["eh_proj_w_t"]
        full = (2 * self.geo.hidden, self.geo.hidden)
        # [2H, H / G]: the columns of this rank's position in its o_proj block
        # group. Slice on the host: the Neuron device refuses a strided copy of
        # a column slice.
        n, pos = self._eh_proj_cols, self._eh_proj_pos
        expected["eh_proj_w_t"] = (2 * self.geo.hidden, n)
        if tuple(eh_proj.shape) == full:
            tensors["eh_proj_w_t"] = eh_proj.to("cpu")[
                :, n * pos : n * (pos + 1)
            ].contiguous()
        for name, tensor in tensors.items():
            if name in _FP8_EXPERT_WEIGHTS or name in _FP8_SCALE_WEIGHTS:
                want = torch.uint8 if name in _FP8_EXPERT_WEIGHTS else torch.float32
                if tuple(tensor.shape) != expected[name] or tensor.dtype != want:
                    raise ValueError(
                        f"the FP8 MTP weight {name} is {tensor.dtype}{tuple(tensor.shape)}. "
                        f"It must be {want}{expected[name]}. Build the FP8 MTP files again."
                    )
                self.register_buffer(
                    MTP_WEIGHT_ATTR_PREFIX + name, tensor.to(target), persistent=False
                )
                continue
            if name in _BF16_WEIGHTS and tensor.dtype in _BITS16_DTYPES:
                tensor = tensor.view(torch.bfloat16)
            if tuple(tensor.shape) != expected[name]:
                raise ValueError(
                    f"the MTP weight {name} is {tuple(tensor.shape)}. This geometry "
                    f"needs {expected[name]}. Build the MTP files for this geometry."
                )
            if name in _FP32_WEIGHTS and tensor.dtype != torch.float32:
                raise ValueError(
                    f"the MTP weight {name} is {tensor.dtype}. It must be float32, "
                    "because a bf16 cast changes the selected experts."
                )
            if name in _BF16_WEIGHTS and tensor.dtype != torch.bfloat16:
                raise ValueError(
                    f"the MTP weight {name} is {tensor.dtype}. The kernel reads it "
                    "as bfloat16."
                )
            self.register_buffer(
                MTP_WEIGHT_ATTR_PREFIX + name, tensor.to(target), persistent=False
            )
        self.device_str = target
        self._weights_loaded = True

    # the traced step
    def _tp_group(self):  # noqa: ANN202  the plugin's group object, see decode.upstream
        """Return the tensor-parallel group over every rank. Its rank is the kernel rank."""
        from sglang_neuron.models.glm52.decode.upstream import get_tp_group

        return get_tp_group()

    def _rope(
        self,
        rope: tuple[torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor | Mapping[str, torch.Tensor], torch.Tensor | None]:
        """Return the target's RoPE tables, which the draft shares (at 1M the full pair is 256 MiB a rank).

        The target gives its two-level tables (`rope_split`) as a mapping; they
        come back as ``(mapping, None)``.
        """
        if isinstance(rope, Mapping):
            return rope, None
        if rope is None:
            raise ValueError(
                "the MTP draft model has no RoPE tables of its own. Give the "
                "target's tables as rope=(cos, sin)."
            )
        return rope

    def kernel_control(
        self,
        slot_base: torch.Tensor,
        slot_last: torch.Tensor,
        slot_live: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return every control tensor of the kernel for one launch.

        Args:
            slot_base: ``[num_slots]`` int32 position of row 0 of each slot.
            slot_last: ``[num_slots]`` int32 last valid row of each slot.
            slot_live: ``[num_slots]`` int32, 1 for a live slot.
            rope: ``(cos, sin)`` ``[max_position, rope / 2]`` float32 tables
                of the target, or None for this model's own tables.

        Returns:
            The step tensors, the static tensors and the two caches, keyed by
            the kernel argument names.
        """
        consts = {name: self.control(name) for name in HELPER_CONTROL}
        rope_cos, rope_sin = self._rope(rope)
        control = step_inputs(
            self.geo,
            consts,
            rope_cos,
            rope_sin,
            slot_base,
            slot_last,
            slot_live,
        )
        control.update({name: self.control(name) for name in STATIC_CONTROL})
        control["kv_cache"] = self.mtp_kv_cache
        control["idx_k_cache"] = self.mtp_idx_k_cache
        return control

    def _lm_head(self, lm_head_weight_t: torch.Tensor | None) -> torch.Tensor:
        """Return the target's ``[hidden, V_s]`` lm_head shard, which the draft head shares."""
        if lm_head_weight_t is None:
            raise ValueError(
                "the MTP draft model has no lm_head shard of its own. Give the "
                "target's [hidden, V_s] shard as lm_head_weight_t."
            )
        return lm_head_weight_t

    def _embed_table(self) -> torch.Tensor:
        """Return this model's padded embedding shard: the store's, with two pad rows where the target's has one."""
        return self.weight("embed_shard_pad")

    def _run_chain(
        self,
        inputs_embeds: torch.Tensor,
        previous_hidden: torch.Tensor,
        control: dict[str, torch.Tensor],
        embed_shard_pad: torch.Tensor,
        lm_head_weight_t: torch.Tensor,
    ) -> torch.Tensor:
        """Launch the chain kernel. Return ``[num_steps, num_slots, 1]`` int32 tokens.

        The head gives one token for each sequence slot, the same on every
        rank.
        """
        if self._chain_fn is None:
            raise RuntimeError(
                "the GLM-5.2 MTP draft kernel is not available on this host. It "
                "needs the Neuron functional namespace and the vendored kernel."
            ) from self._chain_import_error
        weights = {name: self.weight(name) for name in self._weight_names}
        weights["embed_shard_pad"] = embed_shard_pad
        weights[LM_HEAD_ARG] = lm_head_weight_t
        tokens_all, _kv_cache, _idx_k_cache = self._chain_fn(
            inputs_embeds,
            previous_hidden,
            weights=weights,
            control=control,
            meshes=self._meshes,
            scalars=self._scalars,
            num_steps=self.num_steps,
            idx_shard_degree=self.geo.idx_shard_degree,
            idx_block_size=self.geo.idx_block_size,
        )
        # The caches are aliased inputs, so only the tokens are read here.
        return tokens_all

    @torch.no_grad()
    def forward(
        self,
        mtp_tokens: torch.Tensor,
        slot_base: torch.Tensor,
        slot_last: torch.Tensor,
        slot_live: torch.Tensor,
        aux: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        lm_head_weight_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run ``num_steps`` draft steps for every sequence slot.

        Args:
            mtp_tokens: ``[num_slots, spec]`` int32 step-0 input token of each
                row, already shifted by one (`mtp.step.acceptance`).
            slot_base: ``[num_slots]`` int32 position of row 0 of each slot.
            slot_last: ``[num_slots]`` int32 last valid row of each slot.
            slot_live: ``[num_slots]`` int32, 1 for a live slot.
            aux: ``[t_shard, hidden]`` bf16 post-norm hidden state of the target
                model for this rank's rows.
            rope: The target's ``(cos, sin)`` RoPE tables, or its two-level
                tables as a mapping. Required: the draft has none of its own.
            lm_head_weight_t: The target's ``[hidden, V_s]`` lm_head shard.
                Required: the draft head shares it.

        Returns:
            ``[num_slots, num_steps]`` int32 draft tokens in slot order, the same
            on every rank. The tokens of an empty slot have no meaning.

        Raises:
            RuntimeError: If no weights are loaded, or the kernel is not
                available.
            ValueError: If an input has the wrong shape.
        """
        geo = self.geo
        if not self._weights_loaded:
            raise RuntimeError(
                "the MTP draft model has no weights. Call load_weights before the "
                "first step."
            )
        if tuple(mtp_tokens.shape) != (self.num_slots, geo.spec):
            raise ValueError(
                f"mtp_tokens is {tuple(mtp_tokens.shape)}, and the draft model needs "
                f"[{self.num_slots}, {geo.spec}]. Build it with mtp_step.acceptance."
            )
        if tuple(aux.shape) != (geo.t_shard, geo.hidden):
            raise ValueError(
                f"aux is {tuple(aux.shape)}, and the draft model needs "
                f"[{geo.t_shard}, {geo.hidden}]: the target hidden rows of this rank."
            )
        tp = self._tp_group()
        # Step-0 embeddings: a masked local gather, then ONE reduce-scatter. Rank
        # r receives global rows [t_shard * r, t_shard * (r + 1)), its own rows.
        table = self._embed_table()
        rows = local_embedding_rows(mtp_tokens, table, self.control("vocab_base"))
        inputs_embeds = tp.reduce_scatter(rows, dim=0).to(self.dtype)
        control = self.kernel_control(slot_base, slot_last, slot_live, rope)
        tokens_all = self._run_chain(
            inputs_embeds,
            aux.to(self.dtype),
            control,
            table,
            self._lm_head(lm_head_weight_t),
        )
        return tokens_all.reshape(self.num_steps, self.num_slots).t().contiguous()
