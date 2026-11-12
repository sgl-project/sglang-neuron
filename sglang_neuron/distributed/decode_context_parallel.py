# SPDX-License-Identifier: Apache-2.0
"""Decode context parallelism (SGLang's ``--dcp-size D``): KV placement, attention sinks, decode context sizes, and the
launch, model and batch checks.

SGLang builds the paged allocator over D x the pool's tokens in pages of D x its page
(mem_cache/kv_cache_configurator.py), so an allocator page is a virtual page of D blocks, one per rank of the DCP group
(D consecutive attention-TP ranks). Rank ``r`` keeps tokens ``[r * P, (r + 1) * P)`` of each virtual page (P = the
pool's block) in its own block of the same index: SGLang's page-interleaved layout (layers/dcp/layout.py with
``interleave_size = P``). Block tables are the same on every rank and whole blocks stay on one rank, so a PD transfer
still moves whole pages. The decode kernel merges each rank's partial attention across the group.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

# Where a rank writes a token another rank owns: slot 0, in SGLang's reserved page, which no sequence reads.
NULL_SLOT = 0


@dataclass(frozen=True)
class DcpLayout:
    """Where rank ``rank`` of a ``size``-rank DCP group keeps a sequence's KV, in blocks of ``block`` tokens."""

    size: int
    rank: int
    block: int

    def __post_init__(self):
        if self.size < 1 or not 0 <= self.rank < self.size or self.block < 1:
            raise ValueError(f"invalid DCP layout {self}")

    @property
    def page(self) -> int:
        """The allocator's page: one block on each rank of the group."""
        return self.size * self.block

    def owner(self, positions: torch.Tensor) -> torch.Tensor:
        """The DCP rank that keeps each position (or virtual slot: both have the same offset in their virtual page)."""
        return torch.div(positions, self.block, rounding_mode="floor") % self.size

    def local_count(self, lengths: torch.Tensor) -> torch.Tensor:
        """How many of a sequence's first ``lengths`` tokens this rank keeps. Tokens stay in order, so global position
        ``p`` is local token ``local_count(p)`` of its owner."""
        from sglang.srt.layers.dcp.layout import get_dcp_lens

        return get_dcp_lens(lengths, self.size, self.rank, interleave_size=self.block)

    def local_slots(self, slots: torch.Tensor) -> torch.Tensor:
        """SGLang's virtual slots as this rank's KV pool slots; a token another rank owns goes to ``NULL_SLOT``."""
        from sglang.srt.layers.dcp.layout import localize_dcp_indices

        local = localize_dcp_indices(slots, self.size, self.rank, interleave_size=self.block)
        return torch.where(local < 0, torch.full_like(local, NULL_SLOT), local)


def dcp_size(server_args) -> int:
    return int(getattr(server_args, "dcp_size", 1) or 1)


def runtime_layout(block: int) -> Optional[DcpLayout]:
    """This rank's DCP placement for ``block``-token KV blocks, or None without DCP. The plugin's DCP group mirrors
    SGLang's and exists before SGLang builds the KV pool."""
    from sglang_neuron.parallel.groups import get_neuron_dcp_group

    group = get_neuron_dcp_group()
    return None if group is None else DcpLayout(int(group.world_size), int(group.rank_in_group), block)


def decode_context(context: int, size: int, block: int, tile: int) -> int:
    """The decode context (table columns x ``size`` x ``block`` tokens) a DCP decode graph needs for ``context`` tokens.

    Each rank's share of the table must be whole ``tile``-token tiles (``2 x LNC x P_MAX``,
    models/gpt_oss/sglang_model._check_decode_context), the unit at which the kernel reads the packed FP8 K cache
    directly; otherwise every step re-swizzles each full layer's pool and overwrites KV a transfer just landed. The
    kernel keeps the decoded token in the table's last row and never reads it as prior, while ranks before the token's
    owner hold a whole block of its virtual page, so the table must reach one virtual page past the last token: round up
    to whole virtual pages, add one, then round up to whole tiles per rank."""
    if size < 1 or block < 1 or tile < 1:
        raise ValueError(f"a DCP decode context needs positive sizes, not size={size} block={block} tile={tile}")
    if size == 1:
        return context
    page = block * size
    covered = -(-context // page) * page + page
    quantum = math.lcm(tile * size, page)
    return -(-covered // quantum) * quantum


# The sink of a rank that must not count it: exp(IDLE_SINK - m) underflows to 0 against any real row max but stays
# finite, so a row with no local keys gives output 0 (no NaN) and adds nothing to the merge.
IDLE_SINK = -30000.0


def group_sinks(sinks: torch.Tensor, heads_per_rank: int, attn_tp_rank: int, size: int, rank: int) -> torch.Tensor:
    """The attention sinks of the decode kernel's DCP-gathered query heads (gathered in rank order). Every rank adds the
    sink to its partial softmax denominator, so only rank 0 gets the real sinks and the others ``IDLE_SINK``, and the
    merge counts each sink once."""
    start = (attn_tp_rank - rank) * heads_per_rank
    group = sinks[start:start + size * heads_per_rank].to(torch.float32)
    return group if rank == 0 else torch.full_like(group, IDLE_SINK)


# Why a DCP prefill is not served: its prior would be sharded across the group (chunked prefill, prefix-cache hits).
_PREFILL_KERNEL_GAP = ("the KV-parallel segmented prefill kernel (nkilib attention_kv_parallel_segmented_cte) takes no "
                       "attention-sink input, so a prefill over a prior sharded across the DCP group cannot be computed "
                       "for a model with sinks")


def dcp_inside_dp_attention(server_args, spec, role: str, size: int) -> bool:
    """Whether the launch is a PD decode in a decode layout its model declares (``spec``'s decode_layouts, keyed by node
    count and --tp-size as serving_policy.declared_decode_layout selects it) with that layout's DP-attention groups and
    DCP size: its DCP groups are runs of ``size`` ranks inside each attention-TP group (parallel.state.dcp_ranks), and
    the model's decode places and attends over them per group."""
    from sglang_neuron import sglang_tree

    nnodes = int(getattr(server_args, "nnodes", 1) or 1)
    tp = int(getattr(server_args, "tp_size", 1) or 1)
    layout = next((each for each in getattr(spec, "decode_layouts", ())
                   if (each.nnodes, each.tp_size) == (nnodes, tp)), None)
    replicas, attention_dp = sglang_tree.dp_widths(server_args)
    return (layout is not None and role == "decode" and replicas == 1 and attention_dp == layout.dp_size
            and size == layout.dcp_size)


def dcp_launch_problems(server_args, architecture: str, spec) -> list:
    """Why this DCP launch (--dcp-size above 1) cannot run, or []. Each reason names what to change.

    DCP is served on the PD decode role, as one TP group, for a model whose decode attends over DCP shards (``spec``,
    its catalog.ModelSpec, ``decode_context_parallel``). The prefill sends whole pages and each DCP rank keeps the ones
    it owns. A model whose prefill stores and attends over DCP shards (``prefill_context_parallel``) is served DCP on
    the PD prefill role too, pipelined or not: each stage's DCP group is ranks of its own TP group. The decode radix
    cache and several DP groups are refused because no run covers them with DCP."""
    size = dcp_size(server_args)
    problems = []
    role = str(getattr(server_args, "disaggregation_mode", "null") or "null")
    # the PD prefill of a model that declares it is served DCP, pipelined or not; every other launch is checked as a
    # DCP decode
    if not (role == "prefill" and getattr(spec, "prefill_context_parallel", False)):
        if not getattr(spec, "decode_context_parallel", False):
            problems.append(f"{architecture}, whose decode does not attend over decode-context-parallel KV shards")
        if role != "decode":
            problems.append(f"the {'aggregated' if role in ('null', 'none') else role} role: DCP is served on the PD "
                            f"decode role only (and on the PD prefill of a model whose catalog entry sets "
                            f"prefill_context_parallel), because {_PREFILL_KERNEL_GAP}")
        if int(getattr(server_args, "pp_size", 1) or 1) > 1:
            problems.append("--pp-size above 1: a pipelined decode's stages each pair their own pool with the "
                            "prefill's, which a DCP group's page interleave does not")
    backend = getattr(server_args, "dcp_comm_backend", None)
    if backend not in (None, "ag_rs"):
        problems.append(f"--dcp-comm-backend {backend}: the Neuron decode kernel runs its own DCP collectives; launch "
                        "without the flag")
    if role == "decode" and getattr(server_args, "disaggregation_decode_enable_radix_cache", False):
        problems.append("--disaggregation-decode-enable-radix-cache: no run covers the decode radix cache under DCP, "
                        "where SGLang pages the tree at the virtual page and the prefill sends each DCP rank its pages "
                        "after the decode's prefix")
    from sglang_neuron import sglang_tree

    # a DCP group is ranks of one attention TP group (--tp-size over the DP-attention groups); each scheduler checks it
    # again with the model's head split (validate_dcp_heads)
    tp = int(getattr(server_args, "tp_size", 1) or 1)
    replicas, attention_dp = sglang_tree.dp_widths(server_args)
    if (replicas, attention_dp) != (1, 1) and not dcp_inside_dp_attention(server_args, spec, role, size):
        problems.append("--dp-size or DP attention: no run covers DCP on a decode of several DP groups, whose ranks "
                        "would split the DCP groups' collectives and placement by group; the DCP decode is served as "
                        "one TP group")
    attn_tp = tp // attention_dp
    if attn_tp % size:
        where = f"--tp-size {tp}" if attention_dp == 1 else (f"an attention TP size of {attn_tp} (--tp-size {tp} over "
                                                             f"{attention_dp} DP-attention groups)")
        problems.append(f"{where}, which {size} ranks of a DCP group do not divide")
    return [f"--dcp-size {size} with {p}" for p in problems]


# (module, qualified name): md5 of the source, against upstream SGLang 43881a7, checked on a DCP launch. SGLang's owner
# rule sets every rank's mask bounds and write slots (DcpLayout) and must agree with the plugin's page transfer
# (sglang_patches/dcp_page_transfer, k mod D)
_SOURCES_MD5 = {
    ("sglang.srt.layers.dcp.layout", "get_dcp_lens"): "70e32e3647415d786fe8e3fd0a06cf29",
    ("sglang.srt.layers.dcp.layout", "localize_dcp_indices"): "67ace265f857dbcc8509abc60c694381",
}


def validate_dcp_launch(server_args, architecture: str, spec) -> None:
    """Before any scheduler starts, refuse a DCP launch the runtime does not serve (naming every reason), or one on an
    SGLang whose DCP layout helpers changed."""
    if dcp_size(server_args) <= 1:
        return
    problems = dcp_launch_problems(server_args, architecture, spec)
    if problems:
        raise ValueError("; ".join(problems))
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SOURCES_MD5, __name__)


def validate_dcp_heads(name: str, q_heads: int, kv_heads: int, attn_tp_size: int, size: int) -> None:
    """Refuse a head split where a DCP group's ranks would not share one KV head. SGLang replicates a KV head over
    consecutive attention-TP ranks, which is what a DCP group slices, and the kernel gathers the group's query heads
    over it."""
    if size <= 1:
        return
    problems = []
    if attn_tp_size % (size * kv_heads):
        problems.append(f"the attention TP size {attn_tp_size} must be a multiple of dcp_size x KV heads "
                        f"({size} x {kv_heads}), so each rank of a DCP group holds the same single KV head")
    if q_heads % attn_tp_size:
        # with the KV-head rule above, query heads that split over the attention TP also gather evenly over a group
        problems.append(f"its {q_heads} query heads do not split over an attention TP of {attn_tp_size}")
    if problems:
        raise ValueError(f"--dcp-size {size} with {name}: " + "; ".join(problems))


# The decode kernel's DCP path holds a batch's gathered query heads in one SBUF tile of this many partitions.
_TILE_PARTITIONS = 128


def check_dcp_decode_batch(q_heads_per_rank: int, size: int, batch: int) -> None:
    """Refuse a decode batch the kernel's DCP path cannot hold: with more than one query head per rank it transposes the
    batch's gathered query heads (``batch x q_heads_per_rank x size`` rows) through one 128-partition tile."""
    rows = batch * q_heads_per_rank * size
    if q_heads_per_rank > 1 and rows > _TILE_PARTITIONS:
        raise ValueError(
            f"--dcp-size {size}: a decode batch of {batch} rows x {q_heads_per_rank * size} gathered query heads "
            f"({rows}) exceeds the decode kernel's {_TILE_PARTITIONS}-row DCP tile; serve at most "
            f"{_TILE_PARTITIONS // (q_heads_per_rank * size)} requests per decode batch (the largest decode batch size, "
            "--cuda-graph-bs-decode or --max-running-requests)")
