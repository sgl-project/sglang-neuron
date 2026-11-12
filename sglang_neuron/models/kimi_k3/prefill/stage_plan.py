# SPDX-License-Identifier: Apache-2.0
"""The static plan of one Kimi-K3 prefill pipeline stage: its layers, their kinds, the AttnRes bank at its seams,
its AttnRes call order and its drafter roles.

The layer ranges follow SGLang's pipeline split (``SGLANG_PP_LAYER_PARTITION``, else SGLang's even split with the
remainder on the last stages), so the plan always names the layers SGLang builds on the stage. The served split
``SERVED_PARTITION`` ends every stage directly after an MLA layer and gives each stage 3 MLA layers.

AttnRes bank: layer ``i`` appends its input stream to the bank when ``i % attn_res_block_size == 0``, so the seam
before layer ``b`` carries ``(b - 1) // block + 1`` entries.

Drafter: the drafter reads the stream after each target layer of its checkpoint's ``target_layer_ids``, which is the
input stream of the next layer (the consumer layer). Each stage adds the fc partials of its consumer layers to an
accumulator that rides the seams; the last stage writes the drafter rows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Sequence

from sglang_neuron.models.kimi_k3.prefill.geometry import KimiK3Geometry

PARTITION_ENV = "SGLANG_PP_LAYER_PARTITION"
SERVED_PARTITION = (12, 12, 12, 12, 12, 12, 12, 9)

ATTN_SITE = "attn"
FFN_SITE = "ffn"
MLA = "mla"
KDA = "kda"
MOE = "moe"
DENSE = "dense"


def pp_partition(num_layers: int, pp_size: int, partition: Sequence[int] | str | None = None,
                 environ: Mapping[str, str] | None = None) -> tuple[int, ...]:
    """The layer count of each stage.

    ``partition`` (a list or a comma string) wins; else ``SGLANG_PP_LAYER_PARTITION`` of ``environ``; else SGLang's
    even split, the remainder layers on the last stages. A single stage owns every layer.

    Raises:
        ValueError: The partition does not give ``pp_size`` positive counts that sum to ``num_layers``.
    """
    if not 1 <= pp_size <= num_layers:
        raise ValueError(f"pp_size {pp_size} is outside 1..{num_layers}")
    if pp_size == 1:
        return (num_layers,)
    if partition is None:
        partition = (os.environ if environ is None else environ).get(PARTITION_ENV)
    if partition is None:
        base, extra = divmod(num_layers, pp_size)
        return tuple(base + (1 if r >= pp_size - extra else 0) for r in range(pp_size))
    if isinstance(partition, str):
        try:
            partition = [int(p) for p in partition.split(",")]
        except ValueError as err:
            raise ValueError(f"{PARTITION_ENV} {partition!r} is not a comma list of layer counts") from err
    counts = tuple(int(p) for p in partition)
    if len(counts) != pp_size or sum(counts) != num_layers or min(counts) < 1:
        raise ValueError(f"The layer partition {list(counts)} must give {pp_size} positive layer counts that sum to "
                         f"{num_layers}")
    return counts


def stage_layers(num_layers: int, pp_size: int, pp_rank: int, partition: Sequence[int] | str | None = None,
                 environ: Mapping[str, str] | None = None) -> range:
    """The layer range of stage ``pp_rank`` (``pp_partition``)."""
    counts = pp_partition(num_layers, pp_size, partition, environ)
    if not 0 <= pp_rank < pp_size:
        raise ValueError(f"pp_rank {pp_rank} is outside 0..{pp_size - 1}")
    start = sum(counts[:pp_rank])
    return range(start, start + counts[pp_rank])


def bank_entries_before(layer: int, block: int) -> int:
    """The AttnRes bank entries that the layers before ``layer`` appended."""
    return 0 if layer <= 0 else (layer - 1) // block + 1


def draft_consumers(target_layer_ids: Sequence[int], num_layers: int) -> tuple[int, ...]:
    """The consumer layers of the drafter's target layers: each target id plus one (its output is their input).

    Raises:
        ValueError: The ids are not increasing, or a consumer falls outside the model.
    """
    ids = [int(i) for i in target_layer_ids]
    if not ids or any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError(f"The drafter target_layer_ids {ids} must be a non-empty increasing list")
    consumers = tuple(i + 1 for i in ids)
    if consumers[0] < 1 or consumers[-1] >= num_layers:
        raise ValueError(f"The drafter target_layer_ids {ids} give consumer layers {list(consumers)}; each must be in "
                         f"1..{num_layers - 1}")
    return consumers


def attn_res_calls(layers: Sequence[int], block: int, bank_in: int) -> tuple[tuple[int, str, int], ...] | None:
    """The AttnRes calls of a stage in call order: ``(layer, site, bank width)``.

    A layer mixes before its attention when the bank is not empty and always before its FFN; a layer that appends
    does so between its two calls. The result is None when a layer after the first appends: that entry is not known
    at the stage start, so one bank-score table built at the start cannot hold its scores.
    """
    calls = []
    width = bank_in
    for n, layer in enumerate(layers):
        if width > 0:
            calls.append((layer, ATTN_SITE, width))
        if layer % block == 0:
            if n > 0:
                return None
            width += 1
        calls.append((layer, FFN_SITE, width))
    return tuple(calls)


@dataclass(frozen=True)
class LayerPlan:
    """One layer of a stage."""

    index: int
    attn: str  # MLA or KDA
    ffn: str  # MOE or DENSE
    appends_bank: bool
    bank_width_in: int  # bank entries at the layer input
    draft_consumer: bool

    @property
    def is_mla(self) -> bool:
        return self.attn == MLA


@dataclass(frozen=True)
class StagePlan:
    """One pipeline stage of the prefill.

    ``bank_in`` / ``bank_out``: bank entries at the stage input / after its last layer. ``seam_out_entries`` is
    ``bank_out`` on a stage with a successor and None on the last stage (its bank goes to the output mix only).
    ``draft_writer``: the stage writes the drafter rows (the last stage, when the drafter is on).
    """

    pp_rank: int
    pp_size: int
    layers: range
    layer_plans: tuple[LayerPlan, ...]
    bank_in: int
    bank_out: int
    draft_consumers: tuple[int, ...]
    all_draft_consumers: tuple[int, ...]
    draft_writer: bool
    calls: tuple[tuple[int, str, int], ...] | None

    @property
    def is_first(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    @property
    def seam_out_entries(self) -> int | None:
        return None if self.is_last else self.bank_out

    @property
    def mla_layers(self) -> tuple[int, ...]:
        return tuple(p.index for p in self.layer_plans if p.attn == MLA)

    @property
    def kda_layers(self) -> tuple[int, ...]:
        return tuple(p.index for p in self.layer_plans if p.attn == KDA)

    @property
    def dense_layers(self) -> tuple[int, ...]:
        return tuple(p.index for p in self.layer_plans if p.ffn == DENSE)

    @property
    def drafter(self) -> bool:
        return bool(self.all_draft_consumers)

    @property
    def last_layer(self) -> int:
        return self.layers[-1]


def stage_plan(geometry: KimiK3Geometry, pp_rank: int, pp_size: int, *,
               partition: Sequence[int] | str | None = None, draft_target_layers: Sequence[int] | None = None,
               environ: Mapping[str, str] | None = None) -> StagePlan:
    """The plan of stage ``pp_rank``.

    Args:
        geometry: The model geometry.
        pp_rank: The stage index.
        pp_size: The stage count.
        partition: The layer count of each stage (``pp_partition``); None reads ``SGLANG_PP_LAYER_PARTITION``.
        draft_target_layers: The drafter checkpoint's ``target_layer_ids``, or None without a drafter.
        environ: The environment ``pp_partition`` reads (default ``os.environ``).
    """
    layers = stage_layers(geometry.num_layers, pp_size, pp_rank, partition, environ)
    block = geometry.attn_res_block_size
    consumers = () if draft_target_layers is None else draft_consumers(draft_target_layers, geometry.num_layers)
    bank_in = bank_entries_before(layers.start, block)
    plans = []
    width = bank_in
    for i in layers:
        plans.append(LayerPlan(index=i, attn=MLA if geometry.is_mla(i) else KDA,
                               ffn=MOE if geometry.is_moe(i) else DENSE, appends_bank=geometry.appends_bank(i),
                               bank_width_in=width, draft_consumer=i in consumers))
        width += int(geometry.appends_bank(i))
    return StagePlan(pp_rank=pp_rank, pp_size=pp_size, layers=layers, layer_plans=tuple(plans), bank_in=bank_in,
                     bank_out=width, draft_consumers=tuple(c for c in consumers if c in layers),
                     all_draft_consumers=consumers, draft_writer=bool(consumers) and pp_rank == pp_size - 1,
                     calls=attn_res_calls(list(layers), block, bank_in))
