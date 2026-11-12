# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 prefill runtime: the platform runtime (model_executor/graph_runner.py) with the K3 stage graph's step
inputs (graph.KimiK3PrefillGraph).

SGLang's metadata of one prefill chunk (attention.backend.NeuronForwardMetadata) is DCP-placed already: an allocator
page is a virtual page of ``world`` blocks, one per rank of the stage's TP group (= its DCP group), the block table
names the request's virtual pages, and each token's slot is this rank's pool slot or ``NULL_SLOT`` for a token
another rank keeps (distributed.decode_context_parallel.DcpLayout). Rank ``r`` keeps the 128-token stripes ``s`` with
``s % world == r``, stripe ``s`` in its block of virtual page ``s // world``. So per chunk and rank:

* ``mla_block_table``: the virtual page table itself (rank ``r``'s block of virtual page ``v`` is block ``v`` of its
  pool), padded with -1 to the context's columns;
* ``mla_write_slots``: SGLang's slot of each row this rank keeps, -1 for the other rows and the pad rows (the MLA rows
  and the drafter rows share them);
* ``cached_len``: the request's tokens before the chunk, any count: a prompt-cache hit can end at any token, and the
  MLA attention reads the prior of each rank from the positions its stripes hold (prefill/fp8_rows.local_prior);
* ``state_slot``: the request's slot of the recurrent-state pool (SGLang's request table maps it);
* ``state_track_slot``: the slot that also takes the state the chunk ends with. On the chunk-keyed radix tree SGLang's
  batch asks for a checkpoint at a chunk boundary in upstream's tracking slot (the mamba extra buffer:
  ``mamba_track_indices`` where ``mamba_track_mask`` is set); every forward ends on such a boundary, so the graph's
  final state is that checkpoint, which the tree adopts at the request's next insert. Without a request, the pool's
  sink;
* ``sampling_positions``: the chunk's last valid row, whose logits the last stage returns (the prefill samples the
  first token);
* ``draft_rope``: the drafter's rope rows of the positions, on the drafter writer stage.

A warm-up call (the runtime's synthetic metadata) writes no MLA row and runs its state on the pool's sink slot.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from sglang_neuron.distributed.decode_context_parallel import NULL_SLOT, DcpLayout
from sglang_neuron.model_executor.graph_runner import NeuronModelRuntime
from sglang_neuron.models.kimi_k3.prefill.fp8_rows import STRIPE

# the request's state slot and the slot that also takes its final state, on a step's metadata
# (KimiK3PrefillRuntime.run_prefill); a warm-up's metadata has neither
STATE_SLOT_ATTR = "kimi_k3_state_slot"
TRACK_SLOT_ATTR = "kimi_k3_state_track_slot"


def track_slot_of(fb, sink: int, size: int) -> int:
    """The slot that takes the state the forward of ``fb`` (one request's prefill chunk) ends with besides the request's
    own: SGLang's tracking slot when the batch asks for a checkpoint, else ``sink``. A checkpoint the batch asks for
    inside the forward is refused: the graph keeps only the state the forward ends with."""
    from sglang_neuron.mem_cache.state_checkpoint import capture_boundary

    mask = getattr(fb, "mamba_track_mask", None)
    if mask is None or not bool(mask.to("cpu").reshape(-1)[0]):
        return int(sink)
    prefix = int(fb.extend_prefix_lens.to("cpu").reshape(-1)[0])
    extend = int(fb.seq_lens.to("cpu").reshape(-1)[0]) - prefix
    track = int(fb.mamba_track_seqlens.to("cpu").reshape(-1)[0])
    capture_boundary(prefix, extend, track, track_chunk=0)
    slot = int(fb.mamba_track_indices.to("cpu").reshape(-1)[0])
    if not 1 <= slot <= size:
        raise RuntimeError(f"the request's tracking slot {slot} is outside the state pool's slots 1..{size}")
    return slot


def stripe_cycle(layout: DcpLayout) -> int:
    """Tokens of one stripe on every rank of the group: a chunk's start and its bucket are multiples of it."""
    return layout.block * layout.size


def lower_prefill_chunk(md, token_ids: torch.Tensor, bucket: int, *, layout: DcpLayout, table_columns: int,
                        state_slot: int, track_slot: int,
                        draft_rope: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> dict:
    """The host inputs of one request's prefill chunk on rank ``layout.rank`` (module docstring).

    Args:
        md: SGLang's metadata of the chunk (one request; ``slot_mapping`` this rank's DCP slots).
        token_ids: The chunk's token ids ``[n]``.
        bucket: The graph's rows ``T``.
        layout: This rank's place in the stage's DCP group (``DcpLayout(world, rank, 128)``).
        table_columns: The graph's block table width (the context's virtual pages).
        state_slot: The request's recurrent-state slot.
        track_slot: The slot that also takes the chunk's final state (``track_slot_of``).
        draft_rope: The drafter's rope rows of ``positions [T]`` (the writer stage), or None.

    Raises:
        ValueError: The chunk is not one the stage graph computes; the message names why.
    """
    T, n, cached = int(bucket), int(md.num_tokens), int(md.cached_seq_len)
    cycle = stripe_cycle(layout)
    if layout.block != STRIPE:
        raise ValueError(f"the Kimi-K3 MLA rows are kept in {STRIPE}-token stripes, not {layout.block}-token blocks")
    if not 0 < n <= T or T % cycle:
        raise ValueError(f"a Kimi-K3 prefill chunk of {n} tokens in a graph of {T} rows: the rows must hold the chunk "
                         f"and be whole stripe cycles of {cycle} tokens ({STRIPE} x {layout.size} ranks)")
    pos = md.positions[:n].to(torch.int64)
    if not torch.equal(pos, cached + torch.arange(n, dtype=torch.int64)):
        raise ValueError("a Kimi-K3 prefill chunk holds one request's consecutive positions from its cached prefix")
    page = cycle
    pages = -(-(cached + n) // page)
    table = md.block_table.reshape(-1).to(torch.int32)
    if pages > min(table.numel(), table_columns) or bool((table[:pages] < 1).any()):
        raise ValueError(f"the request's {cached + n} tokens span {pages} pages of {page} tokens, and its block table "
                         f"names {int((table[:table_columns] >= 1).sum())} of them within the graph's "
                         f"{table_columns} columns")
    block_table = torch.full((table_columns,), -1, dtype=torch.int32)
    block_table[:pages] = table[:pages]

    rows = torch.arange(T, dtype=torch.int64)
    owned = layout.owner(cached + rows) == layout.rank
    slots = md.slot_mapping[:n].to(torch.int64)
    mine = owned[:n]
    if bool((slots[mine] == NULL_SLOT).any()) or bool((slots[~mine] != NULL_SLOT).any()):
        raise ValueError(f"SGLang's DCP slots of the chunk disagree with the stripe owners of rank {layout.rank} of "
                         f"{layout.size}: a rank keeps stripe s when s % {layout.size} is its rank")
    write_slots = torch.full((T,), -1, dtype=torch.int32)
    write_slots[:n] = torch.where(mine, slots, torch.full_like(slots, -1)).to(torch.int32)

    input_ids = torch.zeros(T, dtype=torch.int64)
    input_ids[:n] = token_ids.reshape(-1)[:n].to(torch.int64)
    positions = torch.full((T,), cached + n - 1, dtype=torch.int64)
    positions[:n] = pos
    out = {
        "input_ids": input_ids, "positions": positions,
        "valid_len": torch.tensor([n], dtype=torch.int64), "cached_len": torch.tensor([cached], dtype=torch.int64),
        "mla_block_table": block_table, "mla_write_slots": write_slots,
        "state_slot": torch.tensor([int(state_slot)], dtype=torch.int64),
        "state_track_slot": torch.tensor([int(track_slot)], dtype=torch.int64),
        "sampling_positions": torch.tensor([n - 1], dtype=torch.int64),
    }
    if draft_rope is not None:
        out["draft_rope"] = draft_rope(positions)
    return out


def synthetic_prefill_inputs(bucket: int, *, table_columns: int, state_slot: int,
                             draft_rope: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> dict:
    """A warm-up call's host inputs, in a real step's layout: every row valid, no prior, no MLA or drafter row written
    (every write slot -1), the state read from and written to ``state_slot`` (the pool's sink) alone."""
    T = int(bucket)
    positions = torch.arange(T, dtype=torch.int64)
    out = {
        "input_ids": torch.ones(T, dtype=torch.int64), "positions": positions,
        "valid_len": torch.tensor([T], dtype=torch.int64), "cached_len": torch.zeros(1, dtype=torch.int64),
        "mla_block_table": torch.zeros(table_columns, dtype=torch.int32),
        "mla_write_slots": torch.full((T,), -1, dtype=torch.int32),
        "state_slot": torch.tensor([int(state_slot)], dtype=torch.int64),
        "state_track_slot": torch.tensor([int(state_slot)], dtype=torch.int64),
        "sampling_positions": torch.tensor([T - 1], dtype=torch.int64),
    }
    if draft_rope is not None:
        out["draft_rope"] = draft_rope(positions)
    return out


class KimiK3PrefillRuntime(NeuronModelRuntime):
    """The platform runtime of a Kimi-K3 prefill stage: its prefill buckets are whole stripe cycles, and its step inputs
    are the stage graph's (``lower_prefill_chunk``)."""

    # the debug logits dump (logits_dump.py) on the last stage when its environment variable names a directory
    logits_dump = None

    def __init__(self, model_runner, sgl_model, phase=None):
        super().__init__(model_runner, sgl_model, phase)
        if self.graphs != "prefill":
            raise ValueError("the Kimi-K3 runtime runs the prefill graph of a PD prefill stage")
        self.layout = self.dcp if self.dcp is not None else DcpLayout(1, 0, self.block_size)
        if self.layout.size != self.model_tp_size:
            raise ValueError(f"--dcp-size {self.layout.size}: the Kimi-K3 MLA rows are striped over the stage's whole "
                             f"attention TP group of {self.model_tp_size} ranks; launch with --dcp-size "
                             f"{self.model_tp_size}")
        self.buckets.prefill_token_buckets = tiled_buckets(self.buckets.prefill_token_buckets,
                                                           stripe_cycle(self.layout))
        from sglang_neuron.mem_cache.state_pool import state_pool_of

        self.state_pool = state_pool_of(model_runner.req_to_token_pool)
        if self.state_pool is None:
            raise ValueError("the Kimi-K3 prefill runs over the Neuron recurrent-state pool, which SGLang did not "
                             "build (KimiK3ForConditionalGeneration.state_pool_class)")
        graph = self.model_graph
        self._draft_rope = None
        if graph.drafter_writer:
            from sglang_neuron.models.kimi_k3.prefill.draft_context import draft_rope_rows

            drafter = graph.drafter
            self._draft_rope = lambda positions: draft_rope_rows(positions, drafter)
        from sglang_neuron.models.kimi_k3 import prefill_dump

        # the debug byte dump (prefill_dump.py), None unless its environment variable names a directory
        self.dump = prefill_dump.from_env()
        from sglang_neuron.models.kimi_k3 import logits_dump

        # the debug logits dump of the last stage (logits_dump.py): the graph also copies its logits row into a buffer
        dump = logits_dump.from_env()
        if dump is not None and graph.plan.is_last:
            self.logits_dump = dump
            from sglang_neuron.model_executor.runtime_init import neuron_device

            graph.logits_dump_buffer = torch.zeros((1, graph.config.vocab_size // self.layout.size),
                                                   dtype=torch.float32, device=neuron_device())

    def request_state_slot(self, fb) -> int:
        """The recurrent-state slot SGLang's request table holds for the forward's one request."""
        indices = fb.req_pool_indices.to("cpu", torch.int64).reshape(-1)[:1]
        slot = int(self.model_runner.req_to_token_pool.get_mamba_indices(indices).reshape(-1)[0])
        if not 1 <= slot <= self.state_pool.size:
            raise RuntimeError(f"the request's state slot {slot} is outside the state pool's slots "
                               f"1..{self.state_pool.size}")
        return slot

    def run_prefill(self, fb, md, bucket=None, pp_proxy=None):
        setattr(md, STATE_SLOT_ATTR, self.request_state_slot(fb))
        setattr(md, TRACK_SLOT_ATTR, track_slot_of(fb, self.state_pool.sink_slot, self.state_pool.size))
        if self.dump is not None:
            self.dump.before_forward(self, fb, md)
        out = super().run_prefill(fb, md, bucket, pp_proxy=pp_proxy)
        if self.dump is not None:
            self.dump.after_forward(self, fb, md)
        if self.logits_dump is not None:
            rids = getattr(fb, "rids", None)
            self.logits_dump.after_prefill(rids[0] if rids else None, self.model_graph.logits_dump_buffer,
                                           rank=self.layout.rank, world=self.layout.size,
                                           tokens=int(md.cached_seq_len) + int(md.num_tokens),
                                           prompt=getattr(fb, "orig_seq_lens", None))
        return out

    def build_model_inputs(self, md, token_ids, bucket, temperatures=None, token_plan=None, seeds=None, filters=None,
                           *, context=None) -> dict:
        from sglang_neuron.backend.device_io import upload_many
        from sglang_neuron.model_executor.runtime_init import neuron_device
        from sglang_neuron.sampling import DRAWING_MODES

        if not md.is_prefill:
            raise RuntimeError("the Kimi-K3 runtime runs prefill chunks only")
        columns = self.buckets.max_blocks()
        slot = getattr(md, STATE_SLOT_ATTR, None)
        if slot is None:
            host = synthetic_prefill_inputs(bucket, table_columns=columns, state_slot=self.state_pool.sink_slot,
                                            draft_rope=self._draft_rope)
        else:
            host = lower_prefill_chunk(md, token_ids, bucket, layout=self.layout, table_columns=columns,
                                       state_slot=slot,
                                       track_slot=getattr(md, TRACK_SLOT_ATTR, self.state_pool.sink_slot),
                                       draft_rope=self._draft_rope)
        dev = neuron_device()
        names = list(host)
        out = dict(zip(names, upload_many([host[name] for name in names], dev)))
        out["rank"] = self.rank_tensor
        if self.sampler_mode in DRAWING_MODES:
            out.update(self._sampling_inputs(md, int(md.num_tokens), 1, temperatures, seeds, dev, filters))
        return out


def tiled_buckets(buckets, cycle: int) -> list:
    """The prefill buckets a K3 stage graph runs: those that are whole stripe cycles. The chunk (the largest) must be
    one."""
    buckets = list(buckets)
    kept = [b for b in buckets if b % cycle == 0]
    if not buckets or not kept or kept[-1] != buckets[-1]:
        raise ValueError(f"--chunked-prefill-size {buckets[-1] if buckets else None}: a Kimi-K3 prefill chunk is whole "
                         f"stripe cycles of {cycle} tokens (128 x the DCP size); launch with a multiple of {cycle}")
    return kept
