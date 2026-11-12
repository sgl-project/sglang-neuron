# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 decode graph with async MTP (K=7): verify, accept, draft and carry in one NEFF per step.

Everything runs on the device, so the host never waits for a step's tokens before launching the next one:

1. Build each slot's rows from its device carry, or from the host's token for a fresh slot (``mtp.step``).
2. Embed the rows and reduce-scatter them so each rank gets its own spec rows.
3. Build each rank's step tensors (RoPE rows, cache write rows, residency cut) from its slot's base and row count.
4. Run the GK over all 78 layers; its LM-head tail gives the target's token for every row.
5. Accept drafts by token agreement (``mtp.step.acceptance``).
6. Run the MTP draft chain (layer 78, 7 times, in one kernel) on the target's post-norm hidden rows.
7. Emit each batch row's accepted tokens and write the next carry and lengths in place (``mtp.async_step``).

The target's token is greedy, or under ``SGLANG_NEURON_SAMPLER=topk_topp`` a draw keyed by the request seed and the
row's input position (``tail_sampling``). A draft is kept only when it equals that draw, so the emitted tokens have
the same distribution as sampling one token at a time.

At the 1M point the GK also takes the chunked indexer pool's tables, built here on the device
(``IdxChunkPoolTables``), and RoPE rows come from the two-level tables (``layout.rope_split``) because a full table
would take 256 MiB a rank.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from sglang_neuron.models.glm52 import tail_sampling
from sglang_neuron.models.glm52.mtp import step as mtp_step
from sglang_neuron.models.glm52.decode.graph import GlmGkDecodeGraph

logger = logging.getLogger(__name__)

# extra host inputs of an MTP step, and the per-slot request table when sampling
MTP_STEP_INPUTS = ("gk_slot_rows", "gk_row_slots", "gk_mtp_ctrl")
MTP_SAMPLING_INPUTS = ("gk_mtp_sparams", "gk_mtp_seeds")

#: Benchmark only: pin the mean acceptance length (1 to spec) so throughput numbers compare at a fixed acceptance.
#: Drafts are kept at fixed rates whatever their tokens, so the output is not the model's; the launch warns when set.
SYNTHETIC_AL_ENV = "SGLANG_NEURON_GLM52_MTP_SYNTHETIC_AL"


def synthetic_acceptance_length(spec: int = 8):
    """The launch's synthetic acceptance length, or None for real acceptance."""
    raw = os.environ.get(SYNTHETIC_AL_ENV, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not 1.0 <= value <= float(spec):
        raise ValueError(f"{SYNTHETIC_AL_ENV}={raw}: an acceptance length is 1 to {spec} tokens a step (a step of "
                         f"{spec - 1} drafts), or leave it unset for the real acceptance")
    return value


def synthetic_conditional_rates(length: float, num_drafts: int) -> list:
    """Per-position conditional keep rates for a mean acceptance ``length``.

    Uses the minimum-variance schedule: the first ``floor(length - 1)`` drafts are always kept, the next one with
    the fractional part, none after.

    Example:
        >>> [round(r, 6) for r in synthetic_conditional_rates(3.91, 7)]
        [1.0, 1.0, 0.91, 0.0, 0.0, 0.0, 0.0]
    """
    accepted = length - 1.0
    full = int(accepted)
    unconditional = ([1.0] * full + [accepted - full] + [0.0] * num_drafts)[:num_drafts]
    previous = [1.0, *unconditional[:-1]]
    return [p / q if q > 0.0 else 0.0 for p, q in zip(unconditional, previous)]


def rope_tables(geo, positions: int):
    """``(cos, sin)`` tables ``[positions, rope / 2]`` in float32, computed from float64 angles like the host rows."""
    half = geo.qk_rope_head_dim // 2
    inv = geo.rope_theta ** (-np.arange(half, dtype=np.float64) / half)
    angle = np.arange(positions, dtype=np.float64).reshape(-1, 1) * inv.reshape(1, half)
    return (torch.from_numpy(np.cos(angle).astype(np.float32)), torch.from_numpy(np.sin(angle).astype(np.float32)))


def zero_mtp_weights(geo) -> dict:
    """Zero draft weights with the store's shapes and dtypes, for a decode over zero weights."""
    from sglang_neuron.models.glm52.mtp import draft as mtp_draft
    from sglang_neuron.models.glm52.layout.decode.weight_loaders_mtp import mtp_fp8t_store_spec

    fp8 = mtp_fp8t_store_spec(geo)
    out = {name: torch.zeros(shape, dtype=torch.bfloat16 if name in mtp_draft._BF16_WEIGHTS else torch.float32,
                             device="cpu")
           for name, shape in mtp_draft.expected_weight_shapes(geo).items()
           if name != mtp_draft.LM_HEAD_ARG and name not in fp8}
    for name, (shape, dtype) in fp8.items():
        out[name] = torch.zeros(shape, dtype=torch.uint8 if dtype is np.uint8 else torch.float32, device="cpu")
    return out


class GlmGkMtpDecodeGraph(GlmGkDecodeGraph):
    """One rank's MTP decode graph."""

    def __init__(self, config, geo, rank: int):
        super().__init__(config, geo, rank)
        from sglang_neuron.models.glm52.mtp import draft as mtp_draft

        c = self.coords
        if geo.rope_split:
            from sglang_neuron.models.glm52.layout import rope_split

            tables = rope_split.split_rope_tables(geo.rope_theta, geo.qk_rope_head_dim // 2, geo.context)
            for name in rope_split.TABLE_NAMES:
                self.register_buffer(f"_{name}", tables[name], persistent=False)
        else:
            cos, sin = rope_tables(geo, geo.context)
            self.register_buffer("_rope_cos", cos, persistent=False)
            self.register_buffer("_rope_sin", sin, persistent=False)
        if geo.idx_pool_chunk:
            # this rank's pool tables; every rank-dependent value is a buffer
            from sglang_neuron.models.glm52.layout.decode.idx_pool import IdxChunkPool
            from sglang_neuron.models.glm52.layout.decode.idx_pool_torch import IdxChunkPoolTables

            pool = IdxChunkPool.for_rank(geo, rank)
            key_rank, pool_index = pool.pool_of_rank(rank)
            self.pool_tables = IdxChunkPoolTables(pool, key_rank, pool_index, spec=geo.idx_pool_tokens, device="cpu")
        # explicit host device: the model runner may build this under a meta default device
        cpu = "cpu"
        self.register_buffer("_spec_offsets", torch.arange(geo.spec, dtype=torch.int32, device=cpu),
                             persistent=False)
        own = torch.arange(geo.t_shard, dtype=torch.int32, device=cpu) + geo.t_shard * c["cp"]
        self.register_buffer("_own_rows", own, persistent=False)
        if self.samples:
            self.register_buffer("_slot_param_columns", torch.arange(tail_sampling.SLOT_PARAM_COLUMNS,
                                                                     dtype=torch.int64, device=cpu), persistent=False)
        self.register_buffer("_cp_rank", torch.tensor([c["cp"]], dtype=torch.int32, device=cpu), persistent=False)
        self.register_buffer("_key_rank", torch.tensor([c["key_rank"]], dtype=torch.int32, device=cpu),
                             persistent=False)
        # per-slot state that every step reads and updates in place
        self.register_buffer("_mtp_carry_tokens", torch.zeros(self.num_slots, geo.spec, dtype=torch.int32,
                                                              device=cpu), persistent=False)
        self.register_buffer("_mtp_slot_len", torch.zeros(self.num_slots, dtype=torch.int32, device=cpu),
                             persistent=False)
        # traced only when SYNTHETIC_AL_ENV is set
        self.synthetic = synthetic_acceptance_length(geo.spec) is not None
        # the draft shares the target's RoPE tables and lm_head shard; its experts are the store's FP8 files
        self.mtp = mtp_draft.Glm52MtpDraftModel(geo, rank, max_position=geo.context, device="cpu")

    def _rope(self):
        """RoPE tables for the verify rows and the draft: the two-level tables, or the full ``(cos, sin)`` pair."""
        if self.geo.rope_split:
            from sglang_neuron.models.glm52.layout.rope_split import TABLE_NAMES

            return {name: getattr(self, f"_{name}") for name in TABLE_NAMES}
        return self._rope_cos, self._rope_sin

    def _rope_rows(self, rows: torch.Tensor):
        """RoPE rows ``[spec, rope / 2]`` at the verify rows' positions, clamped to ``[0, context)``."""
        if self.geo.rope_split:
            from sglang_neuron.models.glm52.layout.rope_split import rope_rows

            return rope_rows(self._rope(), torch.clamp(rows.to(torch.int64), 0, self.geo.context - 1))
        return mtp_step.rope_rows(self._rope_cos, self._rope_sin, rows)

    @property
    def pd_mtp_caches(self):
        """The draft's ``(latent, keys)`` caches, registered as the MTP layer's PD transfer regions."""
        return self.mtp.mtp_kv_cache, self.mtp.mtp_idx_k_cache

    def in_place_targets(self) -> dict:
        """Buffers the step writes in place: the GK caches, the draft's caches and the carry state."""
        targets = super().in_place_targets()
        targets.update(mtp_kv_cache=self.mtp.mtp_kv_cache, mtp_idx_k_cache=self.mtp.mtp_idx_k_cache,
                       mtp_carry_tokens=self._mtp_carry_tokens, mtp_slot_len=self._mtp_slot_len)
        return targets

    def load_weights(self, path: str, device) -> None:
        """Load the target's weights, then the draft's from the same store (or zeros for a zero-weight decode)."""
        from sglang_neuron.models.glm52.decode import pools as decode_pools

        super().load_weights(path, device)
        store = decode_pools.weight_store(self.geo)
        if store == decode_pools.ZERO_WEIGHTS:
            self.mtp.set_weights(zero_mtp_weights(self.geo), device="cpu")
        else:
            self.mtp.load_weights(store, device="cpu")
        # the draft's buffers move to the device with the rest (models/host.place_rest_on_device)
        self.mtp.device_str = str(device)

    def _aux(self, hidden: torch.Tensor) -> torch.Tensor:
        """This rank's post-norm hidden rows ``[t_shard, hidden]`` in bf16, the draft's ``previous_hidden``."""
        from sglang_neuron.models.glm52.layout.decode.gk_geometry import INPUT_NORM_EPS

        rows = hidden.to(torch.float32)
        scale = torch.rsqrt(rows.pow(2).mean(-1, keepdim=True) + INPUT_NORM_EPS)
        return (rows * scale * self._final_norm_w).to(torch.bfloat16)

    def forward(self, input_ids, positions, attn_metadata=None, sampling_positions=None, rank=None, *,
                gk_slot_rows, gk_row_slots, gk_mtp_ctrl, gk_mtp_sparams=None, gk_mtp_seeds=None):
        slot = mtp_step.slot_inputs(gk_slot_rows, gk_mtp_ctrl, self._mtp_carry_tokens, self._mtp_slot_len,
                                    self.geo.context)
        if self.synthetic:
            slot["synth_kept"] = gk_mtp_ctrl[:, mtp_step.CTRL_SYNTH_KEPT]
        if self.samples:
            if gk_mtp_sparams is None or gk_mtp_seeds is None:
                raise RuntimeError("the topk_topp MTP step needs the per-slot request table (gk_mtp_sparams, "
                                   "gk_mtp_seeds; decode_runtime.run_mtp_step)")
            slot.update(sparams=gk_mtp_sparams, seeds=gk_mtp_seeds)
        hidden, target = self.target_step(slot)
        return {"token_ids": self.accept_draft_carry(slot, target, hidden, gk_row_slots)}

    def target_step(self, slot: dict):
        """Steps 2-4: this rank's post-stack hidden rows and the target's token for every row ``[num_slots * spec]``."""
        from sglang_neuron.parallel import collectives
        from sglang_neuron.parallel.groups import get_neuron_attn_tp_group

        geo = self.geo
        group = get_neuron_attn_tp_group()
        # embed every slot's rows; the reduce-scatter leaves each rank its own rows
        ids = slot["tokens"].reshape(-1).to(torch.int64)
        local = ids - self._vocab_lo
        inside = (local >= 0) & (local < self.vocab_shard)
        rows = self._embed_pad.index_select(0, torch.where(inside, local + 1, torch.zeros_like(local)))
        mine = collectives.reduce_scatter(group, rows, dim=0)
        # this rank's slot
        base = slot["base"].index_select(0, self._my_slot)
        count = slot["count"].index_select(0, self._my_slot)
        live = slot["live"].index_select(0, self._my_slot)
        x_shard = (mine.to(torch.float32) * (self._own_rows < count).to(torch.float32).reshape(-1, 1)).to(
            torch.bfloat16)
        row_pos, valid = mtp_step.row_positions(base, count, self._spec_offsets)
        cos, sin = self._rope_rows(row_pos)
        kv_slots = mtp_step.latent_slots(geo, row_pos, valid, live, self._cp_rank)
        idx_slots = mtp_step.indexer_slots(geo, row_pos, valid, live, self._key_rank)
        idx_positions = mtp_step.threshold(base, live, geo.spec)
        # the target's token for every row, gathered in rank order (float32 holds token ids exactly)
        params = self._sampling_params
        if self.samples:
            request = tail_sampling.slot_params(slot["sparams"], slot["seeds"], self._my_slot,
                                                self._slot_param_columns, self.lm_head_k)
            own_live = ((self._own_rows < count) & live).to(torch.float32).reshape(-1, 1)
            params = tail_sampling.gk_tail_rows_at(row_pos.index_select(0, self._own_rows.to(torch.int64)), own_live,
                                                   *request, self._sampling_params)
        head = {"lm_head_weight_t": self._lm_head_wt, "final_norm_w": self._final_norm_w,
                "sampling_params": params, "k": self.lm_head_k}
        # chunked indexer pool tables (1M point)
        pool = self.pool_tables(slot["base"], slot["live"], slot["count"]) if self.geo.idx_pool_chunk else None
        hidden, tokens = self.gk(x_shard, cos, sin, cos, sin, kv_slots, idx_slots, idx_positions, head=head,
                                 idx_pool=pool)
        table = collectives.all_gather(group, tokens.to(torch.int32).reshape(-1, 1).to(torch.float32), dim=0)
        return hidden, table.reshape(-1).to(torch.int32)

    def _acceptance(self, slot: dict) -> dict:
        """Extra acceptance inputs: the benchmark's kept-draft counts under synthetic acceptance, otherwise none."""
        if not self.synthetic:
            return {}
        return {"synthetic_keep": mtp_step.synthetic_keep(slot["synth_kept"], self.geo.spec)}

    def accept_draft_carry(self, slot: dict, target: torch.Tensor, hidden: torch.Tensor, row_slots: torch.Tensor):
        """Steps 5-7: accept, draft, and return each batch row's tokens ``[rows, spec]``, -1 after the accepted."""
        from sglang_neuron.models.glm52.mtp.async_step import step_outputs

        geo = self.geo
        state = mtp_step.acceptance(slot, target, **self._acceptance(slot))
        drafts = self.mtp(state["mtp_tokens"], state["slot_base"], state["slot_last"], state["slot_live"],
                          self._aux(hidden), rope=self._rope(), lm_head_weight_t=self._lm_head_wt)
        # under synthetic acceptance the kept drafts are emitted as is (they need not match the target)
        sampled, carry = step_outputs(state["target"], state["last"], state["verify"], state["live"],
                                      state["next_token"], drafts, row_slots, self._mtp_carry_tokens, geo.vocab_size,
                                      kept_drafts=state["drafts_in"] if self.synthetic else None)
        lengths = mtp_step.commit(state, slot, self._mtp_slot_len)
        self._mtp_carry_tokens.copy_(carry)
        self._mtp_slot_len.copy_(lengths)
        return sampled
