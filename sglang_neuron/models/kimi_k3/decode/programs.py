# SPDX-License-Identifier: Apache-2.0
"""The device programs of one Kimi-K3 DSpark decode step: the verify program, the draft program, the KDA-params init.

A step runs two programs on every rank (the step splits at one point, after the acceptance):

- verify: the carry of the last step, the step positions, the verify kernel metadata of the 93 layers, the masked
  vocab-shard embedding and its TP all-reduce, one ``kimi_k3_verify_model_2srv`` call (93 layers, final AttnRes, norm
  and lm_head shard, drafter aux states), the full-vocab sampler with the drafter q lists as candidates, the greedy and
  ``block`` acceptance, the forced-count select and the drafter sampling inputs;
- draft: the rider table and slots from the request's ring, one ``kimi_k3_draft_mla_step`` call (context rows, 5 MLA
  drafter layers, lm_head + Markov tail, sampled drafts), the next ids, the row flags of each request, the q lists of
  the next drafts.

The ops between the kernels are those of the served step in the same order (``spec_math``, ``verify_metadata``,
``rider_window_inputs``, the sampler parameter split), so the traced graphs hold the same HLO around the same kernel
calls. The decode serves no structured output: the verify takes no grammar mask and no logit mask. The pools are
module tensors that the kernels write in place; a kernel returns them as aliases, and the programs do not read them
back.

Step inputs (``VERIFY_INPUTS``, ``DRAFT_INPUTS``) carry the names of the host step tensors (``host_state.step_tensors``,
``host_state.null_step_outputs``) and neutral names for the per-group metadata. ``StepOutputs`` holds what the host
reads and what the next step carries (``carry_inputs``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from sglang_neuron.models.kimi_k3.decode import kernel_calls as kc
from sglang_neuron.models.kimi_k3.decode import spec_math as sm
from sglang_neuron.models.kimi_k3.decode.layout import DP8_LAYOUT, DecodeLayout
from sglang_neuron.models.kimi_k3.decode.lowering import DecodeGeometry, kda_page_groups, verify_layers
from sglang_neuron.models.kimi_k3.decode.pools import DecodePools
from sglang_neuron.models.kimi_k3.decode.verify_metadata import VerifyMetadataConfig, spec_verify_metadata

VOCAB = 163_840
HIDDEN = 7168
K3_MLA_LAYERS = tuple(range(3, 92, 4)) + (92,)
"""The 24 MLA layers of the 93 (0-indexed); the other 69 are KDA layers."""
DRAFT_WINDOW = 49_152
DRAFT_TABLE_BLOCKS = 256
"""Kernel columns of the 1M MLA table of a rank (1,048,576 / (32 x 128)); the drafter reads its width only."""
KDA_STATE_GROUPS = 3
SAMPLER_PARAM_COLUMNS = 4
"""``sampling_params [B * Q, 4]``: top_k, top_p, temperature, draw."""

# --- step input names

VERIFY_INPUTS = (
    "input_ids",
    "spec_fresh",
    "spec_prev_row",
    "spec_kda_replay_len",
    "spec_kda_state_position",
    "spec_known",
    "spec_window_base",
    "spec_position_window",
    "spec_prev_row0",
    "spec_prev_accepted",
    "spec_prev_next_ids",
    "spec_draws",
    "spec_prev_q_probs",
    "spec_prev_q_ids",
    "spec_forced_accept",
    "sampling_params",
    "mla_block_table",
    "mla_slot_window",
    "kda_state_table_0",
    "kda_state_table_1",
    "kda_state_table_2",
)
"""The step inputs of the verify program (with ``spec_live_rows`` on the k3 form): the host ids, the ``spec_*`` tensors
of the host state, the sampler rows, the MLA host-page table ``[B, 128]`` and slot window ``[B, 2k + 1]`` int64, the
KDA state table ``[B, 1 + Q]`` of each state group."""
VERIFY_OPTIONAL_INPUTS = ("spec_live_rows",)
DRAFT_INPUTS = ("spec_draft_lo", "spec_rope_window", "spec_draft_rider_ring")
"""The host step inputs of the draft program (the rest come from the verify program)."""
STEP_INPUTS = VERIFY_INPUTS + VERIFY_OPTIONAL_INPUTS + DRAFT_INPUTS


@dataclasses.dataclass(frozen=True)
class DecodeProgramConfig:
    """The static settings of the programs of one served form.

    Attributes:
        form: ``k3`` (Q 4, B 4) or ``k7b2`` (Q 8, B 2).
        layout: the 256-rank decode layout.
        vocab, hidden: the target vocab and hidden size.
        rms_norm_eps: the target RMSNorm eps.
        draft_softmax_scale, draft_mask_token: the drafter's YaRN softmax scale and mask token.
        is_mla: the kind of each of the 93 layers.
        method: the rejection rule of the sampled drafts.
    """

    form: str = "k3"
    layout: DecodeLayout = DP8_LAYOUT
    vocab: int = VOCAB
    hidden: int = HIDDEN
    rms_norm_eps: float = 1e-5
    draft_softmax_scale: float = 0.0
    draft_mask_token: int = kc.DRAFT_MASK_TOKEN
    is_mla: tuple[bool, ...] = tuple(layer in K3_MLA_LAYERS for layer in range(93))
    method: str = "block"

    def __post_init__(self) -> None:
        if self.form not in kc.SERVED_FORMS:
            raise ValueError(f"The served form {self.form!r} is not one of {sorted(kc.SERVED_FORMS)}.")
        if self.draft_softmax_scale <= 0:
            raise ValueError("The programs take the drafter softmax scale (draft_softmax_scale > 0).")

    @property
    def queries(self) -> int:
        return kc.SERVED_FORMS[self.form][1]

    @property
    def batch(self) -> int:
        return kc.FORM_BATCH[self.form]

    @property
    def k(self) -> int:
        return self.queries - 1

    @property
    def tp(self) -> int:
        return self.layout.attention_tp

    @property
    def live_rows(self) -> bool:
        return bool(kc.VERIFY_CALL[self.form]["live_rows"])

    @property
    def geometry(self) -> DecodeGeometry:
        return DecodeGeometry(cp_size=self.tp, queries=self.queries)


# --- the traced glue that is not in spec_math / verify_metadata


def rider_window_inputs(ring: Tensor, row0: Tensor, position_rows: Tensor, mla_slots: Tensor, blocks: int,
                        window: int, cp: int) -> tuple[Tensor, Tensor]:
    """The rider table ``[B, blocks]`` and rider slots ``[Q, B, 1]`` of a step from the ring of each sequence.

    Kernel column ``c`` (``cp * 128`` positions) of a sequence lives in ring entry ``c % len(ring)``; only the columns
    the step reads or writes, ``[row0 + 1 - window, row0 + Q - 1]``, take their ring page, the others the null page 0
    (below ``P - window``, which the kernel mask hides). A verify row takes slot ``page * 128 + p % 128`` of its column
    page when this rank owns its MLA row (MLA slot > 0) and the sequence has a ring, else 0.
    """
    ring = ring.to(torch.int32)
    batch, columns = ring.shape
    queries = position_rows.shape[0]
    column = int(cp) * 128
    row0 = row0.to(torch.int32).reshape(batch, 1)
    index = torch.arange(blocks, dtype=torch.int32, device=ring.device).reshape(1, -1)
    low = torch.div(torch.clamp(row0 + 1 - int(window), min=0), column, rounding_mode="floor")
    high = torch.div(row0 + (queries - 1), column, rounding_mode="floor")
    pages = torch.gather(ring, 1, index.remainder(columns).to(torch.int64).expand(batch, -1))
    table = torch.where((index >= low) & (index <= high), pages, torch.zeros_like(pages))
    positions = position_rows.to(torch.int32).reshape(queries, batch)
    entry = torch.div(positions, column, rounding_mode="floor").remainder(columns)
    page = torch.gather(ring, 1, entry.t().to(torch.int64)).t()
    owned = (mla_slots.reshape(queries, batch) > 0) & (page > 0)
    rows = 128
    slots = torch.where(owned, page * rows + positions.remainder(rows), torch.zeros_like(page))
    return table, slots.reshape(queries, batch, 1)


def _column(values: Tensor) -> Tensor:
    return values.to(torch.int32).reshape(-1, 1)


def sampler_columns(sampling_params: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split ``sampling_params [R, 4]`` into the full-vocab sampler inputs ``(top_k [R, 1] int32, top_p [R, 1],
    temperature [R], draw [R] fp32)``; a ``top_k`` of 1 is greedy (temperature 0, the lowest id on a tie)."""
    if sampling_params.ndim != 2 or sampling_params.shape[-1] != SAMPLER_PARAM_COLUMNS:
        raise ValueError(f"The sampling params are [rows, {SAMPLER_PARAM_COLUMNS}] (top_k, top_p, temperature, draw); "
                         f"the step gives {list(sampling_params.shape)}.")
    top_k = sampling_params[:, 0].to(torch.int32)
    top_p = sampling_params[:, 1]
    temperature = sampling_params[:, 2]
    uniform = sampling_params[:, 3]
    top_k = top_k.unsqueeze(-1)
    top_p = top_p.unsqueeze(-1)
    temperature = torch.where(top_k.reshape(-1) == 1, torch.zeros_like(temperature), temperature)
    return top_k, top_p, temperature, uniform.to(torch.float32)


# --- the NKI cumsum of the residual draw

_CUMSUM: list[Any] = []


def nki_cumsum(tensor: Tensor) -> Tensor:
    """The cumulative sum along the last axis of a 2-D tensor with nkilib's cumsum kernel (grid 1 at LNC1), the form
    the served sampler draw runs on the device."""
    if not _CUMSUM:
        if torch.compiler.is_compiling():
            raise kc.kernel_imports.KernelUnavailable("The cumsum kernel was not built before the trace: call "
                                                      "programs.prepare_kernels() when the programs are built.")
        prepare_kernels()
    return _CUMSUM[0][kc.LNC](x=tensor, axis=-1)


def prepare_kernels() -> None:
    """Build the kernel wrappers of the programs (``kernel_calls.prepare``) and the cumsum kernel; call outside any
    trace."""
    kc.prepare()
    if not _CUMSUM:
        # nkilib's cumsum kernel (the served decode's own, byte for byte); it is an nki.jit kernel already.
        from nkilib.core.cumsum.cumsum import cumsum

        from sglang_neuron.backend import stack

        _CUMSUM.append(stack.wrap_nki(cumsum))


# --- the module that holds one rank's tensors


def _buffer(module: nn.Module, name: str, tensor: Tensor) -> None:
    module.register_buffer(name, tensor, persistent=False)


class DecodeRankTensors(nn.Module):
    """One rank's tensors that the programs read: the weights (``weights.load_decode_rank`` names), the pools and the
    rank tensors. The programs' graphs take them as inputs; nothing is copied."""

    def __init__(self, weights: Mapping[str, Tensor], pools: DecodePools, rank: int, layout: DecodeLayout = DP8_LAYOUT,
                 device=None) -> None:
        super().__init__()
        names = {**{n: n for n in kc.WEIGHT_ARGUMENTS}, **{n: n for n in kc.MX_SCALE_ARGUMENTS},
                 kc.ROUTER_STACK: kc.ROUTER_STACK}
        missing = [n for n in (*kc.WEIGHT_ARGUMENTS, *kc.TAIL_ARGUMENTS, "embed_tokens") if n not in weights]
        missing += [f"draft.{n}" for n in kc.DRAFT_WEIGHT_ARGUMENTS if f"draft.{n}" not in weights]
        if missing:
            raise KeyError(f"The decode programs need the weights {missing}.")
        self.verify_names = tuple(n for n in names if weights.get(n) is not None)
        for name in self.verify_names:
            _buffer(self, f"w_{name}", weights[name])
        for name in kc.TAIL_ARGUMENTS:
            _buffer(self, f"tail_{name}", weights[name])
        _buffer(self, "embed_tokens", weights["embed_tokens"])
        for name in kc.DRAFT_WEIGHT_ARGUMENTS:
            _buffer(self, f"draft_{name}", weights[f"draft.{name}"])
        self.pool_count = len(pools.kv)
        for index, pool in enumerate(pools.kv):
            _buffer(self, f"pool_{index}", pool)
        self.pool_of_layer = tuple(int(i) for i in pools.pool_of_layer)
        _buffer(self, "rider_pool", pools.rider)
        tp, dp, world = layout.tp_rank(rank), layout.dp_rank(rank), rank
        for name, value in (("tp_rank", tp), ("dp_rank", dp), ("world_rank", world)):
            _buffer(self, name, torch.tensor([value], dtype=torch.int32, device=device))

    def verify_weights(self) -> dict[str, Tensor]:
        return {name: getattr(self, f"w_{name}") for name in self.verify_names}

    def tail_weights(self) -> dict[str, Tensor]:
        return {name: getattr(self, f"tail_{name}") for name in kc.TAIL_ARGUMENTS}

    def draft_weights(self) -> dict[str, Tensor]:
        return {name: getattr(self, f"draft_{name}") for name in kc.DRAFT_WEIGHT_ARGUMENTS}

    def pools(self) -> list[Tensor]:
        return [getattr(self, f"pool_{index}") for index in range(self.pool_count)]

    def set_kda_params(self, wide: Tensor) -> None:
        """Bind the wide ``kda_params`` stack (the init program's output) in place of the plain one."""
        _buffer(self, "w_kda_params", wide)


@dataclasses.dataclass
class VerifyOutputs:
    """What the verify program gives the draft program and the host (device tensors).

    ``tokens [B, Q]`` (-1 after the accepted prefix), ``accepted``, ``bonus``, ``row0``, ``offset`` ``[B]`` int32;
    ``aux [5, R, H]`` bf16; ``mla_slot [24, Q, B, 1]``, ``position_rows [Q, B, 1]``, ``rank_info [1, 4]`` int32;
    ``sample_params [B, 3]``, ``u_draft [B, 7]`` fp32; ``row_flags [R]`` int32.
    """

    tokens: Tensor
    accepted: Tensor
    bonus: Tensor
    row0: Tensor
    offset: Tensor
    aux: Tensor
    mla_slot: Tensor
    position_rows: Tensor
    rank_info: Tensor
    sample_params: Tensor
    u_draft: Tensor
    row_flags: Tensor

    FIELDS = ("tokens", "accepted", "bonus", "row0", "offset", "aux", "mla_slot", "position_rows", "rank_info",
              "sample_params", "u_draft", "row_flags")

    def as_tuple(self) -> tuple[Tensor, ...]:
        return tuple(getattr(self, name) for name in self.FIELDS)

    @classmethod
    def from_tuple(cls, values: Sequence[Tensor]) -> VerifyOutputs:
        return cls(*values)


@dataclasses.dataclass
class StepOutputs:
    """The outputs of one step (device tensors): ``tokens [B, Q]`` (-1 after the accepted prefix), ``accepted [B]``,
    ``next_ids [B, Q]`` (the bonus and the next k drafts), ``row0 [B]``, ``flags [B]`` (OR of the row flags,
    ``spec_math.ROW_FLAG_*``) and ``next_q_probs`` / ``next_q_ids [B, k, 256]``."""

    tokens: Tensor
    accepted: Tensor
    next_ids: Tensor
    row0: Tensor
    flags: Tensor
    next_q_probs: Tensor
    next_q_ids: Tensor
    launch_event: Any = None
    """The draft graph's launch event (``backend.device_io.after_launch``; None off the device): it covers the
    verify graph too, which ran before it on the graphs' stream. A readback waits for it only."""

    def carry_inputs(self) -> dict[str, Tensor]:
        """The next step's ``spec_prev_*`` inputs: the device outputs of this step (async carry)."""
        return {
            "spec_prev_row0": self.row0,
            "spec_prev_accepted": self.accepted,
            "spec_prev_next_ids": self.next_ids,
            "spec_prev_q_probs": self.next_q_probs,
            "spec_prev_q_ids": self.next_q_ids,
        }


class VerifyProgram(nn.Module):
    """The verify program of one step (``forward(inputs)`` -> ``VerifyOutputs.as_tuple()``)."""

    def __init__(self, tensors: DecodeRankTensors, config: DecodeProgramConfig,
                 tp_all_reduce: Callable[[Tensor], Tensor], cumsum: Callable[[Tensor], Tensor] = nki_cumsum) -> None:
        super().__init__()
        self.tensors = tensors
        self.config = config
        self.tp_all_reduce = tp_all_reduce
        self.cumsum = cumsum
        self.metadata_config = VerifyMetadataConfig(queries=config.queries, cp_size=config.tp)
        self.call = dict(kc.VERIFY_CALL[config.form])
        self.levers = kc.VERIFY_LEVERS[config.form]
        # The state group of each KDA layer (the table verify_layers gives it): the verify builds the KDA page
        # metadata once for each group, as the served call does (kda_page_groups); without it, once a layer.
        self.kda_page_groups = kda_page_groups(config.is_mla, config.geometry.kda_groups)

    def forward(self, inputs: Mapping[str, Tensor]) -> tuple[Tensor, ...]:
        cfg, t = self.config, self.tensors
        queries, batch, vocab = cfg.queries, cfg.batch, cfg.vocab
        carry = sm.carry_spec_state(
            inputs["spec_fresh"], inputs["spec_prev_row"], inputs["spec_kda_replay_len"],
            inputs["spec_kda_state_position"], inputs["input_ids"], inputs["spec_prev_accepted"],
            inputs["spec_prev_row0"], inputs["spec_prev_next_ids"])
        offset, positions = sm.step_positions(carry.row0, inputs["spec_window_base"], inputs["spec_position_window"],
                                              queries)
        pools = t.pools()
        pages = int(pools[0].shape[0])
        kda_tables = [inputs[f"kda_state_table_{g}"] for g in range(KDA_STATE_GROUPS)]
        layers = verify_layers(cfg.is_mla, inputs["mla_block_table"], inputs["mla_slot_window"], kda_tables, pages,
                               cfg.geometry)
        metadata = spec_verify_metadata(layers, positions, offset, carry.replay, carry.state_position, t.tp_rank,
                                        t.dp_rank, t.world_rank, self.metadata_config)
        input_rows = carry.input_ids.reshape(-1)
        call = self.call
        live = {"live_rows": inputs["spec_live_rows"]} if call["live_rows"] else {}
        embeddings = sm.masked_vocab_embedding(input_rows, t.embed_tokens, t.tp_rank, self.tp_all_reduce)
        outputs = kc.verify_block(
            sm.to_token_major(embeddings, queries),
            t.verify_weights(),
            **metadata,
            pools=pools,
            pool_of_layer=t.pool_of_layer,
            eps=cfg.rms_norm_eps,
            queries=queries,
            tail_weights=t.tail_weights(),
            verify_kwargs=self.levers,
            layout=cfg.layout,
            moe_form=call["moe_form"],
            routed_tp=call["routed_tp"],
            greedy=call["greedy"],
            return_logits=call["return_logits"],
            return_aux=call["return_aux"],
            aux_layers=call["aux_layers"],
            aux_tap=call["aux_tap"],
            mla_fp8_pages=call["mla_fp8_pages"],
            mla_loop_path=call["mla_loop_path"],
            mla_mode=call["mla_mode"],
            lora_norm_eps=call["lora_norm_eps"],
            kda_replay=call["kda_replay"],
            kda_conv_dtype=call["kda_conv_dtype"],
            kda_state_dtype=call["kda_state_dtype"],
            kda_page_groups=self.kda_page_groups,
            **live,
        )
        candidates = sm.spec_verify_candidates(
            inputs["spec_prev_row"], inputs["spec_prev_q_probs"], inputs["spec_prev_q_ids"], inputs["spec_fresh"],
            inputs["spec_draws"], batch, queries, vocab, cfg.tp)
        logits = sm.to_request_major(outputs.logits, queries)
        top_k, top_p, temperature, draw = sampler_columns(inputs["sampling_params"])
        target, flags, probs_at, excl_token = kc.full_vocab_verify(
            logits, top_k, top_p, temperature, draw, t.tp_rank.to(torch.int32), candidates[2], candidates[3],
            layout=cfg.layout, direct=None, **kc.SAMPLER_OPTIONS)
        accept = sm.accept_spec_step(
            target, carry.input_ids, inputs["spec_known"], batch=batch, queries=queries, vocab=vocab,
            probabilistic=True, forced=inputs["spec_forced_accept"],
            sampled_inputs={
                "p_at": probs_at, "excl_token": excl_token, "q_probs": candidates[0], "q_ids": candidates[1],
                "sampling_params": inputs["sampling_params"], "fresh": inputs["spec_fresh"],
                "draws": inputs["spec_draws"],
            },
            method=cfg.method, tp_size=cfg.tp, cumsum=self.cumsum)
        sample_params, u_draft = accept.drafter_sampling
        result = VerifyOutputs(
            tokens=accept.tokens, accepted=accept.accepted, bonus=accept.bonus, row0=carry.row0, offset=offset,
            aux=outputs.aux, mla_slot=metadata["mla_slot"], position_rows=metadata["position_rows"],
            rank_info=metadata["rank_info"], sample_params=sample_params, u_draft=u_draft, row_flags=flags)
        return result.as_tuple() + dumped_logits(self, logits)


class DraftProgram(nn.Module):
    """The draft program of one step: ``forward(<VerifyOutputs fields the drafter reads>, draft_lo, rope_window,
    rider_ring)`` -> ``(next_ids [B, Q], flags [B], next_q_probs, next_q_ids [B, k, 256])``."""

    def __init__(self, tensors: DecodeRankTensors, config: DecodeProgramConfig) -> None:
        super().__init__()
        self.tensors = tensors
        self.config = config
        self.stage = dict(kc.DRAFT_STAGE[config.form])
        self.call = dict(kc.DRAFT_CALL[config.form])
        if not self.call["embed_vocab"] or not self.call["sampled"]:
            raise ValueError("The served drafter takes the vocab shard of the embedding and sampled drafts.")

    def forward(self, aux: Tensor, accepted: Tensor, bonus: Tensor, row0: Tensor, offset: Tensor, mla_slot: Tensor,
                position_rows: Tensor, rank_info: Tensor, sample_params: Tensor, u_draft: Tensor, row_flags: Tensor,
                draft_lo: Tensor, rope_window: Tensor, rider_ring: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg, t, call = self.config, self.tensors, self.call
        queries, batch = cfg.queries, cfg.batch
        # The windowed rider: the kernel takes the rider table and slots of the ring in place of the MLA ones.
        table, rider_slots = rider_window_inputs(rider_ring, row0, position_rows, mla_slot[0], DRAFT_TABLE_BLOCKS,
                                                 call["window"], cfg.tp)
        rider_slots = rider_slots.reshape(queries * batch, 1)
        block, q_probs, q_ids = kc.draft_mla_step(
            aux=aux,
            bonus=_column(bonus),
            accepted=_column(accepted),
            row0=_column(row0),
            draft_lo=_column(draft_lo),
            rider_pool=t.rider_pool,
            block_table=table.to(torch.int32),
            verify_slots=rider_slots.to(torch.int32),
            rope_window=rope_window,
            rope_base=_column(row0.to(torch.int32) - offset.to(torch.int32)),
            weights=t.draft_weights(),
            lm_head_weight=t.tail_lm_head_weight,
            block_embed=None,
            rank_info=rank_info,
            softmax_scale=cfg.draft_softmax_scale,
            layout=cfg.layout,
            window=call["window"],
            mask_token=cfg.draft_mask_token,
            loop_path=call["loop_path"],
            fp8_pages=call["fp8_pages"],
            cp=call["cp"],
            window_read=call["window_read"],
            embed_vocab=t.embed_tokens,
            queries=queries,
            # The windowed rider fails closed: no kernel map of an entry to the null page.
            guard_indices=call["guard_indices"],
            sample_params=sample_params,
            u_draft=u_draft,
            **self.stage,
        )
        next_ids, next_q = sm.draft_next_ids(bonus, block, (q_probs, q_ids), batch, queries, cfg.vocab)
        flags = sm.request_flags(row_flags, batch)
        return next_ids, flags, *next_q


class KdaParamsInit(nn.Module):
    """The KDA-params init program: the wide ``kda_params`` stack ``[L, 128, W + Hl]`` (the ``exp(A_log)`` columns
    last, computed on the device engine the per-step exp uses) of the plain stack ``[L, 128, W]``."""

    def __init__(self, conv_dtype: str = "fp32") -> None:
        super().__init__()
        self.init = kc.KdaAExpInit(conv_dtype)

    def forward(self, kda_params: Tensor) -> Tensor:
        return self.init(kda_params)


def verify_input_shapes(config: DecodeProgramConfig) -> dict[str, tuple[tuple[int, ...], Any]]:
    """The shape and dtype of each step input of a served form (the traced shapes of the two programs)."""
    q, b, k = config.queries, config.batch, config.k
    r, w = q * b, sm.slot_window_width(k)
    shapes: dict[str, tuple[tuple[int, ...], Any]] = {
        "input_ids": ((r,), torch.int32),
        **{f"spec_{n}": ((b,), torch.int32) for n in ("fresh", "prev_row", "kda_replay_len", "kda_state_position",
                                                      "known", "window_base", "prev_row0", "prev_accepted",
                                                      "forced_accept", "draft_lo")},
        "spec_position_window": ((b, w), torch.int32),
        "spec_prev_next_ids": ((b, q), torch.int32),
        "spec_draws": ((b, 2 * k + 2), torch.float32),
        "spec_prev_q_probs": ((b, k, sm.DRAFT_SAMPLE_WIDTH), torch.float32),
        "spec_prev_q_ids": ((b, k, sm.DRAFT_SAMPLE_WIDTH), torch.int32),
        "sampling_params": ((r, SAMPLER_PARAM_COLUMNS), torch.float32),
        "mla_block_table": ((b, config.geometry.table_width), torch.int32),
        "mla_slot_window": ((b, w), torch.int64),
        **{f"kda_state_table_{g}": ((b, 1 + q), torch.int32) for g in range(KDA_STATE_GROUPS)},
        "spec_rope_window": ((b, max(3 * k + 2, 2 * k + 8), 128), torch.float32),
        "spec_draft_rider_ring": ((b, 14), torch.int32),
    }
    if config.live_rows:
        shapes["spec_live_rows"] = ((b, 1), torch.int32)
    return shapes


def check_step_inputs(inputs: Mapping[str, Tensor], config: DecodeProgramConfig) -> None:
    """Refuse a step whose inputs miss a name or have another shape or dtype than the traced ones (a new shape would
    compile a new program)."""
    expected = verify_input_shapes(config)
    problems = []
    for name, (shape, dtype) in expected.items():
        value = inputs.get(name)
        if value is None:
            problems.append(f"{name} missing")
        elif tuple(value.shape) != shape or value.dtype != dtype:
            problems.append(f"{name} {value.dtype} {list(value.shape)} (want {dtype} {list(shape)})")
    if problems:
        raise ValueError(f"The {config.form} step inputs do not fit the programs: {'; '.join(problems)}.")


def dumped_logits(program: nn.Module, logits: Tensor) -> tuple[Tensor, ...]:
    """The verify program's last output under the logits dump (models/kimi_k3/logits_dump.py): the request-major logits
    ``[R, V / tp]`` the sampler reads, when ``program.dump_logits`` is set before the first trace (a program of its
    own); else none. At the end of the file: a line moved above a traced method changes the programs' HLO source
    metadata."""
    return (logits,) if getattr(program, "dump_logits", False) else ()
