# SPDX-License-Identifier: Apache-2.0
"""Layout of one GLM-5.2 prefill pipeline stage on the prefill giga kernel.

A stage holds the decoder layers [start_layer, end_layer) of SGLANG_PP_LAYER_PARTITION=18,20,20,20 on its own 32 ranks,
the kernel's context-parallel group. The layout says how those layers split into kernel launches, which world ranks
the in-kernel collectives name, how many rows a rank keeps of each KV page, and what else the stage does: the first
stage embeds, the last one norms, samples and writes the MTP prompt rows. The KV pool, the weight tree and the stage
graph all size a stage from this one value. It imports no torch or SGLang, so the launching process can read it.

A rank keeps CP_INTERLEAVE rows of each KV page: SGLang sees a page of cp * CP_INTERLEAVE = 1024 tokens, and rank r
keeps tokens [32r, 32r + 32) of it.

Example:
    >>> layout = stage_layout(GlmMoeDsaConfig(), pp_rank=1, pp_size=4)  # doctest: +SKIP
    >>> layout.start_layer, layout.end_layer, len(layout.groups), layout.virtual_page  # doctest: +SKIP
    (18, 38, 5, 1024)
"""

from __future__ import annotations

from dataclasses import dataclass

from sglang_neuron.models.glm52.layout.prefill import pipeline

#: The served split of the 78 decoder layers over the four stages (SGLANG_PP_LAYER_PARTITION at PP4); every stage
#: starts on a full indexer layer.
STAGE_PARTITION = (18, 20, 20, 20)
#: The served split over sixteen stages (SGLANG_PP_LAYER_PARTITION at PP16): stages of five layers, most starting on a shared indexer
#: layer, whose GK stack takes the top-k picks of the stage before (``picks_in``), as a GLM_GK_PP_PICKS stage does.
PP16_PARTITION = (3,) + (5,) * 15
#: Tokens one rank keeps of each virtual page, in order: the kernel's cp_interleave (the 32-token block the measured
#: stages run). Rank r owns the r-th run of CP_INTERLEAVE tokens of every cp x CP_INTERLEAVE.
CP_INTERLEAVE = 32
#: Where the PP4 point's stage 0 cuts its prologue [0, 6): [0, 1), [1, 2), [2, 6), the launches the served stages
#: were validated at.
PROLOGUE_CUTS = (1, 2)


@dataclass(frozen=True)
class PrefillPoint:
    """One served shape of the prefill server: ``pp_size`` stages of ``stage_ranks`` ranks each (a stage's TP, its
    kernel's CP group and its MoE's EP group), the layer split, the largest step (``chunk``, SGLang's
    --chunked-prefill-size) and the smaller steps of a prompt's last piece (``smaller``, --cuda-graph-bs-prefill), and
    the launch cuts of stage 0's prologue. A step's tokens are a whole number of ``stage_ranks`` x CP_INTERLEAVE.

    ``pp4``: the served PP4 prefill (18,20,20,20 x 32 ranks, --chunked-prefill-size 4096 and
    --cuda-graph-bs-prefill 1024 2048, prologue cuts 1,2).
    ``pp16``: the served PP16 prefill (3,5 x 15 x 8 ranks, --chunked-prefill-size 2048 and --cuda-graph-bs-prefill
    1024, no prologue cuts)."""

    name: str
    pp_size: int
    stage_ranks: int
    partition: tuple[int, ...]
    chunk: int
    smaller: tuple[int, ...]
    prologue_cuts: tuple[int, ...]

    @property
    def buckets(self) -> tuple[int, ...]:
        """Every step size a stage graph compiles, ascending."""
        return tuple(sorted({*self.smaller, self.chunk}))

    @property
    def virtual_page(self) -> int:
        """Tokens of one SGLang page: one CP_INTERLEAVE run on every rank of a stage."""
        return self.stage_ranks * CP_INTERLEAVE


PREFILL_POINTS = {
    4: PrefillPoint("pp4", 4, 32, STAGE_PARTITION, 4096, (1024, 2048), PROLOGUE_CUTS),
    16: PrefillPoint("pp16", 16, 8, PP16_PARTITION, 2048, (1024,), ()),
}


def prefill_point(pp_size: int) -> PrefillPoint:
    """The served prefill point of ``pp_size`` stages (PREFILL_POINTS)."""
    point = PREFILL_POINTS.get(int(pp_size))
    if point is None:
        raise ValueError(f"--pp-size {pp_size}: the GLM-5.2 prefill serves {sorted(PREFILL_POINTS)} pipeline stages")
    return point


@dataclass(frozen=True)
class LaunchGroup:
    """One _GkLaunch of a stage and its KV / indexer stack pair: the stage-local layers [lo, hi), those of them with a
    full indexer, and each layer's MLP and indexer kind as the kernel takes them (layer_mlp_types, layer_indexer_types:
    static tuples of its trace)."""

    index: int
    lo: int
    hi: int
    full: tuple[int, ...]
    mlp_types: tuple[str, ...]
    indexer_types: tuple[str, ...]

    @property
    def layers(self) -> int:
        return self.hi - self.lo


@dataclass(frozen=True)
class StageLayout:
    """One stage: its global layers, CP group and launch groups.

    ``model_layers`` is the full model's decoder layer count. ``picks_in``: the stage starts on a shared indexer layer,
    so its first launch takes the previous stage's top-k picks; ``picks_out``: the next stage starts on one, so the
    last launch returns its picks and the boundary carries them."""

    pp_rank: int
    pp_size: int
    start_layer: int
    end_layer: int
    cp: int
    groups: tuple[LaunchGroup, ...]
    embeds: bool
    samples: bool
    mtp_rows: bool
    mtp_layer: int
    model_layers: int
    picks_in: bool = False
    picks_out: bool = False

    @property
    def layers(self) -> int:
        return self.end_layer - self.start_layer

    @property
    def kv_page_rows(self) -> int:
        """Rows one rank keeps of each KV page: its run of CP_INTERLEAVE tokens."""
        return CP_INTERLEAVE

    @property
    def virtual_page(self) -> int:
        """Tokens of one SGLang page: one CP_INTERLEAVE run on every rank of the stage (1024 at CP32)."""
        return self.cp * CP_INTERLEAVE

    @property
    def holds_embedding(self) -> bool:
        """The stage embeds tokens: its input ids, or the token after each row for the MTP prompt rows."""
        return self.embeds or self.mtp_rows

    @property
    def full_layers(self) -> tuple[int, ...]:
        """The global ids of the stage's layers with a full indexer, in order."""
        return tuple(self.start_layer + i for g in self.groups for i in g.full)

    def kv_layer_names(self, mtp: bool = False) -> tuple[str, ...]:
        """Every PD KV region's layer name, in the order a decode pairs them by layer: each MLA cache, then each indexer key cache, in global layer
        order. With ``mtp`` (a prefill for a decode that drafts, pd.placement.PD_MTP_ENV) the last stage's MTP layer
        comes last in each kind, where a drafting decode's MTP entries pair with them; else they stay on the device."""
        mla = [f"layers.{g}.self_attn" for g in range(self.start_layer, self.end_layer)]
        keys = [f"layers.{g}.self_attn.indexer" for g in self.full_layers]
        if mtp and self.mtp_rows:
            mla.append(f"layers.{self.mtp_layer}.self_attn")
            keys.append(f"layers.{self.mtp_layer}.self_attn.indexer")
        return tuple(mla + keys)

    def replica_groups(self, cells) -> tuple[tuple[int, ...], ...]:
        """The world ranks of the kernel's replica groups over ``cells``, lists of stage-local positions (0 .. cp - 1):
        each cell on every stage, stage-major, so stages with the same layers compile one NEFF and every NEFF declares
        the world's whole partition, as its compiler collectives do. A static tuple of ints: the groups are the NEFF's
        collective topology, not a tensor."""
        cells = tuple(tuple(int(p) for p in cell) for cell in cells)
        stray = sorted({p for cell in cells for p in cell if not 0 <= p < self.cp})
        if stray:
            raise ValueError(f"replica cells name positions {stray} outside the stage's {self.cp} ranks")
        return tuple(tuple(p + s * self.cp for p in cell) for s in range(self.pp_size) for cell in cells)


def launch_groups(mlp_types, indexer_types, prologue_cuts=PROLOGUE_CUTS,
                  shared_start: bool = False) -> tuple[LaunchGroup, ...]:
    """Split a stack of layers of these kinds into the served launches.

    The layers split at each full indexer layer. The trailing spans of one (MLP, indexer) pattern stay separate
    launches, one per full layer; every layer before them is one prologue launch, cut at ``prologue_cuts``. GLM-5.2's
    stage 0 runs [0, 1), [1, 2), [2, 6), [6, 10), [10, 14), [14, 18); stages 1-3 run five [full, shared x 3] launches.
    With ``shared_start`` (the stage takes the previous stage's picks) the leading shared layers are the first
    launch."""
    mlp_types, indexer_types = tuple(mlp_types), tuple(indexer_types)
    n = len(indexer_types)
    if len(mlp_types) != n or not n:
        raise ValueError(f"a stack of {n} indexer kinds and {len(mlp_types)} MLP kinds")
    full = [kind == "full" for kind in indexer_types]
    moe = [kind == "sparse" for kind in mlp_types]
    if not full[0] and not shared_start:
        raise ValueError("a stage's stack starts on a full indexer layer unless it takes the picks of the stage before: "
                         "the kernel's first launch picks its own top-k")
    starts = [i for i in range(n) if full[i]]
    if not starts:
        return (LaunchGroup(index=0, lo=0, hi=n, full=(), mlp_types=mlp_types, indexer_types=indexer_types),)
    spans = list(zip(starts, starts[1:] + [n]))

    def sig(lo, hi):
        return tuple((moe[i], full[i]) for i in range(lo, hi))

    k = len(spans) - 1
    while k > 0 and sig(*spans[k - 1]) == sig(*spans[-1]):
        k -= 1
    head = [(0, spans[k][0])] if spans[k][0] > 0 else []
    ranges = _split_prologue(head, full, tuple(prologue_cuts)) + spans[k:]
    return tuple(
        LaunchGroup(index=j, lo=lo, hi=hi, full=tuple(i for i in range(lo, hi) if full[i]),
                    mlp_types=mlp_types[lo:hi], indexer_types=indexer_types[lo:hi])
        for j, (lo, hi) in enumerate(ranges))


def _split_prologue(head: list, full: list, cuts: tuple) -> list:
    """Cut the prologue launch ``head`` ([(0, hi)] or []) at full indexer layers inside it, so
    each new launch starts like every other one, on a full layer with the residual stream of the launch before. A stack
    without a prologue (stages 1-3) is unchanged."""
    if not cuts or not head:
        return head
    lo, hi = head[0]
    for c in cuts:
        if not (lo < c < hi) or not full[c]:
            inside = ",".join(str(i) for i in range(lo + 1, hi) if full[i]) or "none"
            raise ValueError(f"the prologue cut {c} is not a full indexer layer inside the prologue launch [{lo}, "
                             f"{hi}) (its full layers: {inside})")
    edges = [lo, *sorted(set(cuts)), hi]
    return list(zip(edges, edges[1:]))


def stage_layout(config, *, pp_rank: int, pp_size: int, partition=None, cp: int | None = None,
                 prologue_cuts=None) -> StageLayout:
    """The layout of stage ``pp_rank`` of the full-model ``config`` (a GlmMoeDsaConfig with first_layer_index 0).

    ``partition``, ``cp`` and ``prologue_cuts`` default to the served point of ``pp_size`` stages (prefill_point);
    the prologue cuts apply to stage 0 only. A stage may start on a shared indexer layer when a full one precedes it:
    it then takes the picks of the stage before (``picks_in``), and that stage hands its own on (``picks_out``).

    Raises:
        ValueError: A partition that does not cover the model, a stage that starts on a shared indexer layer with no
            full one before it (layout/prefill/pipeline.stage_config's message), or an argument outside its values.
    """
    point = PREFILL_POINTS.get(int(pp_size))
    if partition is None or cp is None or prologue_cuts is None:
        if point is None:
            raise ValueError(f"pp_size {pp_size}: no served prefill point gives the partition, cp and prologue cuts "
                             f"({sorted(PREFILL_POINTS)} do); pass them")
        partition = point.partition if partition is None else partition
        cp = point.stage_ranks if cp is None else cp
        prologue_cuts = point.prologue_cuts if prologue_cuts is None else prologue_cuts
    partition = tuple(int(n) for n in partition)
    total = int(config.num_hidden_layers)
    if len(partition) != pp_size or min(partition) < 1 or sum(partition) != total:
        raise ValueError(f"the stage partition {partition} must give each of {pp_size} stages at least one layer and "
                         f"cover the model's {total} decoder layers")
    if not 0 <= pp_rank < pp_size:
        raise ValueError(f"pp_rank {pp_rank} outside {pp_size} stages")
    if cp < 1:
        raise ValueError(f"cp {cp}: a stage holds at least one rank")
    start = sum(partition[:pp_rank])
    end = start + partition[pp_rank]
    try:
        stage = pipeline.stage_config(config, start, end, shared_start=True)
    except ValueError as error:
        raise ValueError(f"{error} Here the split is SGLANG_PP_LAYER_PARTITION.") from error
    picks_in, picks_out = pipeline.stage_picks(list(config.indexer_types), start, end)
    return StageLayout(
        pp_rank=pp_rank, pp_size=pp_size, start_layer=start, end_layer=end, cp=cp,
        groups=launch_groups(stage.mlp_layer_types, stage.indexer_types,
                             tuple(prologue_cuts) if pp_rank == 0 else (), shared_start=picks_in),
        embeds=pp_rank == 0, samples=pp_rank == pp_size - 1, mtp_rows=pp_rank == pp_size - 1,
        mtp_layer=int(stage.mtp_layer_index), model_layers=total, picks_in=picks_in, picks_out=picks_out)
