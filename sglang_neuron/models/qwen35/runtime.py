# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5 giga decode's runtime: the platform runtime (``model_executor.graph_runner.NeuronModelRuntime``) with
the decode inputs of ``graph.Qwen35DecodeGraph``.

A decode step's keyword arguments are the replica's tokens and the row tables of ``inputs.build_decode_rows``:

- the GDN state slot of each row is SGLang's mamba slot of its request (the request table's
  ``req_index_to_mamba_index_mapping``, a host tensor on this platform), the KV slot its page (``out_cache_loc // S``);
- ``row_valid`` comes from the token counts SGLang's DP attention gathered for the step (``global_num_tokens_cpu``),
  so every rank of the world builds the same table; the planned step's batch (``plan_step``) carries them for an idle
  DP rank's dummy too. A step without gathered counts (one replica) marks only its own rows;
- the tokens are the host ids, or the device sampler's futures cut or zero-padded to the bucket on the device.

The warm-up and an idle DP rank's dummy run every row as padding (position 0 of the reserved page 0, the state pool's
sink slot, no expert). One graph serves every context length (the KV slot is the whole context), so the decode
families compile one graph per seqs bucket. On a paged pool (``Qwen35Layout.paged``) a row's KV is its pages, read
through the block table (``block_table [rows, MP]``, MP = the context's pages, padded with page 0) in place of
``kv_slot``. Refused: decode context parallelism, pipeline stages, hybrid SWA memory,
a sampler other than greedy and the in-graph token reorder.
"""
from __future__ import annotations

import os

from typing import Optional

import torch

from sglang_neuron import env_registry
from sglang_neuron.backend import device_io
from sglang_neuron.model_executor.graph_runner import NeuronModelRuntime
from sglang_neuron.model_executor.runtime_init import neuron_device
from sglang_neuron.model_executor.token_io import FixedDecodeTokenIO

from .inputs import build_decode_rows
from . import prefix

_KEYS = ("input_ids", "pos", "kv_slot", "gdn_slot", "row_valid")
_PAGED_KEYS = ("input_ids", "pos", "block_table", "gdn_slot", "row_valid")

_STATE_DUMP = env_registry.zero_or_one("SGLANG_NEURON_QWEN35_STATE_DUMP")


def decode_counts(fb, dp: int):
    """Every replica's decode rows this step, or None when the step's batch does not tell them: from the DP-attention
    cohort snapshot (``mlp_sync_batch_metadata``: a replica's decode requests when it decodes, else 0, so a step where
    other replicas prefill counts them as idle), else SGLang's gathered token counts (a pure decode server, where every
    participant decodes)."""
    if fb is None:
        return None
    meta = getattr(fb, "mlp_sync_batch_metadata", None)
    if meta is not None and len(meta.forward_modes) == dp:
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        out = []
        for mode, decode in zip(meta.forward_modes, meta.num_decode_requests):
            if not ForwardMode(mode).is_decode():
                out.append(0)
            elif decode is None:
                return None
            else:
                out.append(int(decode))
        return out
    counts = getattr(fb, "original_global_num_tokens_cpu", None) or getattr(fb, "global_num_tokens_cpu", None)
    return None if counts is None else [int(c) for c in counts]


class Qwen35TokenIO(FixedDecodeTokenIO):
    """The decode graph on the platform's fixed-width token I/O (model_executor/token_io.py): its ids are assembled in
    the graph from the previous decode's device token vector and the step's ``token_control`` (row 0 host ids, row 1
    each row's source in that vector, -1 for the host id), and its tokens leave at the fixed width, so a decode never
    waits for the previous step's ids on the host. The graph's rows are the replica's rows in SGLang's batch order
    (``pos`` gives their count, the step's bucket)."""

    def forward(self, input_ids, pos, token_control=None, token_ids_out=None, **kwargs):
        rows = pos.shape[0]
        if not 0 < rows <= self.width:
            raise ValueError("Decode metadata exceeds the token capacity")
        ids = self._assembled(input_ids, token_control)[:rows]
        return self._sampled(self.model(input_ids=ids, pos=pos, **kwargs), rows, token_ids_out)


class Qwen35DecodeRuntime(NeuronModelRuntime):
    """The platform runtime with the giga decode's inputs (module docstring).

    Under async execution the decode runs on the platform's device token relay: the graph takes the previous decode's
    device tokens and the step's token control (Qwen35TokenIO), at the fixed token width of the largest seqs bucket, so
    the next step launches while the previous one runs, its ids never read on the host. An idle replica's call is
    launched and never read."""

    idle_unread = True
    # set by _init_decode_token_io: the relay's fixed token width under async execution, else none
    decode_token_width = 0

    def __init__(self, model_runner, sgl_model, phase=None):
        super().__init__(model_runner, sgl_model, phase=phase)
        self._init_qwen35()

    def _init_decode_token_io(self) -> None:
        # async execution: the platform's device token relay at the largest seqs bucket (Qwen35TokenIO); without it the
        # graph takes the rows' host ids
        on = self.graphs == "decode" and self.async_execution
        self.decode_token_width = int(self.buckets.num_seqs_buckets[-1]) if on else 0
        self.ingraph_reorder = bool(on)
        if on:
            self.execution_model = Qwen35TokenIO(self.execution_model, self.decode_token_width)

    def _init_qwen35(self) -> None:
        refused = [what for what, on in (
            ("decode context parallelism", self.dcp is not None),
            ("pipeline stages", bool(getattr(self, "pp_active", False))),
            ("hybrid SWA memory", self.window_pool is not None),
            (f"the {self.sampler_mode!r} device sampler", self.sampler_mode != "greedy"),
        ) if on]
        if refused:
            raise ValueError(f"the Qwen3.5 giga decode inputs are built for none of: {', '.join(refused)}")
        layout = self.model_graph.layout
        # every seqs bucket is a decode step the kernel runs: whole rows per rank, within its partitions and pools
        pool = self.model_runner.req_to_token_pool.mamba_pool
        kv = getattr(self.model_runner.token_to_kv_pool, "full_kv_pool", None)
        kv_slots = None if layout.paged else getattr(kv, "slots", None)
        problems = [p for b in self.buckets.num_seqs_buckets
                    for p in layout.row_problems(self.model_graph.dims, int(b), state_slots=pool.physical_slots,
                                                 kv_slots=kv_slots)]
        if problems:
            raise ValueError("the Qwen3.5 giga decode's seqs buckets (SGLANG_NEURON_NUM_SEQS_BUCKETS and "
                             f"--max-running-requests) {list(self.buckets.num_seqs_buckets)}: " + "; ".join(problems))
        self._q35_layout = layout
        self._q35_ladder()
        self._q35_ctx = None
        self._q35_dev = neuron_device()
        self._q35_copy_hosts = not device_io.on_device() or self._q35_dev.type == "cpu"
        self._q35_step_batch = None
        # the state pool's slot operations compile at startup, not at a request's first clear or copy
        warm = getattr(getattr(type(pool), "ops", None), "warm", None)
        if warm is not None and self._q35_dev.type != "cpu":
            warm(pool.tensors)

    # the step's batch: its gathered per-replica counts (an idle rank's dummy needs them too)
    def plan_step(self, fb, md=None):
        self._q35_step_batch = fb
        return super().plan_step(fb, md)

    def _replica_tokens(self, n: int, rows: int) -> list:
        counts = decode_counts(self._q35_step_batch, self._q35_layout.dp)
        dp = self._q35_layout.dp
        if counts is None or len(counts) != dp:
            if dp > 1:
                # without the gathered counts no rank may mark another replica's row padding: every row is routed
                return [rows] * dp
            return [n]
        counts = [int(c) for c in counts]
        replica = self.tp_rank // self._q35_layout.replica
        if counts[replica] != n:
            raise RuntimeError(f"replica {replica}: {n} decode rows this step, the gathered counts say "
                               f"{counts[replica]}")
        return counts

    def _state_slots(self, fb, n: int) -> list:
        pool = self.model_runner.req_to_token_pool
        indices = fb.req_pool_indices[:n]
        if indices.device.type != "cpu":
            indices = indices.cpu()
        slots = pool.req_index_to_mamba_index_mapping[indices]
        return (slots.cpu() if slots.device.type != "cpu" else slots).tolist()

    def _sink_slot(self) -> int:
        return int(self.model_runner.req_to_token_pool.mamba_pool.sink_slot)

    def _upload(self, rows, device_token_ids=None, block_table=None) -> dict:
        kv = rows.kv_slot if block_table is None else block_table
        keys = _KEYS if block_table is None else _PAGED_KEYS
        hosts = [torch.from_numpy(rows.input_ids), torch.from_numpy(rows.pos), torch.from_numpy(kv),
                 torch.from_numpy(rows.gdn_slot), torch.from_numpy(rows.row_valid)]
        if self._q35_copy_hosts:
            hosts = [h.clone() for h in hosts]
        if device_token_ids is not None:
            uploaded = device_io.upload_many(hosts[1:], self._q35_dev)
            return dict(zip(keys, [device_token_ids, *uploaded]))
        return dict(zip(keys, device_io.upload_many(hosts, self._q35_dev)))

    def _block_table(self, md, n: int, rows: int, context: Optional[int] = None):
        """The decode rows' pages (the metadata's block table, -1 past a row's pages as page 0), padded to the
        bucket's rows and the context rung's pages; None on a slot pool."""
        layout = self._q35_layout
        if not layout.paged:
            return None
        width = (int(context) // layout.slot_tokens) if context else layout.pages_per_row
        table = md.block_table[:n].numpy() if n else []
        pages = [[int(p) for p in row if p >= 0] for row in table]
        if any(len(p) > width for p in pages):
            raise RuntimeError(f"a decode row holds more KV pages than the step's {context}-token context rung "
                               f"({width} pages): the plan chose a rung below the cohort's longest sequence")
        return prefix.decode_block_table(pages, rows, width)

    def _q35_ladder(self) -> None:
        """On a paged pool a decode reads its block table's pages only: the context rungs (a quarter, a half and the
        whole of the context, in whole pages) let short sequences read a shorter span; the cohort agrees on one rung
        a step (megakernel._cohort_context, the snapshot's longest sequence of every decoding replica). Set before the
        base runtime counts its warmed shapes (the program fence and the direct submit bind one per shape)."""
        layout = self.model_graph.layout
        if layout.paged and not getattr(self, "_q35_laddered", False):
            page, full = layout.slot_tokens, layout.rope_len
            self.buckets.decode_contexts = sorted({max(page, -(-full // q // page) * page) for q in (4, 2, 1)})
            self._q35_laddered = True

    def _warm_shapes(self) -> list:
        self._q35_ladder()
        return super()._warm_shapes()

    def warmup_all(self):
        return with_nki_compile_mode(DECODE_NKI_COMPILE_MODE, super().warmup_all)

    def context_for(self, tokens: int) -> int:
        """The smallest decode context rung that holds ``tokens`` positions (paged pool)."""
        for c in self.buckets.decode_contexts:
            if c >= tokens:
                return int(c)
        return int(self.buckets.decode_contexts[-1])

    def run_idle(self, plan):
        """An idle replica's padding decode of the cohort's context rung (every rank runs one NEFF a step)."""
        self._q35_ctx = plan.context
        return super().run_idle(plan)

    def build_model_inputs(self, md, token_ids: torch.Tensor, bucket: int,
                           temperatures: Optional[torch.Tensor] = None, token_plan=None,
                           seeds: Optional[torch.Tensor] = None, filters: Optional[torch.Tensor] = None, *,
                           context: Optional[int] = None) -> dict:
        if md.is_prefill:
            raise RuntimeError("the Qwen3.5 giga decode runtime builds decode steps only; its prefill is not hosted "
                               "on this runtime")
        n = md.batch_size
        fb = self._q35_step_batch
        if fb is None:
            raise RuntimeError("a Qwen3.5 decode step was not planned (plan_step) before its inputs were built")
        relay = bool(self.decode_token_width)
        rows = build_decode_rows(
            token_ids=[0] * n if relay else token_ids.reshape(-1)[:n].tolist(),
            positions=md.positions.reshape(-1)[:n].tolist(), out_cache_loc=md.slot_mapping.reshape(-1)[:n].tolist(),
            state_slots=self._state_slots(fb, n), rows=bucket, replica=self._q35_layout.replica,
            slot_tokens=self._q35_layout.slot_tokens, sink_slot=self._sink_slot(),
            replica_tokens=self._replica_tokens(n, bucket), paged=self._q35_layout.paged)
        if _STATE_DUMP:
            self._q35_state_dump(md, fb, n)
        return self._relayed(self._upload(rows, None, self._block_table(md, n, bucket, context)), token_plan,
                             token_ids, n)

    def _q35_state_dump(self, md, fb, n: int) -> None:
        """SGLANG_NEURON_QWEN35_STATE_DUMP=1 forensics: at a request's first decode step on this rank, its GDN slot and a
        checksum of every component of every layer of the slot's recurrent state (the bytes the transferred
        initial state landed as), so identical requests' initial states compare run to run."""
        import hashlib
        import logging

        last = getattr(self, "_q35_state_last", None)
        if last is None:
            last = self._q35_state_last = {}
        pool = self.model_runner.req_to_token_pool
        mapping = pool.req_index_to_mamba_index_mapping
        idxs = fb.req_pool_indices[:n]
        if idxs.device.type != "cpu":
            idxs = idxs.cpu()
        positions = md.positions.reshape(-1)[:n].tolist()
        log = logging.getLogger(__name__)
        for j, rp in enumerate(idxs.tolist()):
            slot = int(mapping[rp].reshape(-1)[0])
            key = (rp, slot)
            prev = last.get(key)
            last[key] = positions[j]
            if prev is not None and positions[j] > prev:
                continue  # same request's later step; a new request reusing the slot restarts lower
            try:
                conv, temporal = pool.mamba_pool.get_cpu_copy([slot])
                parts = [hashlib.md5(t.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()[:8]
                         if t.dtype == torch.bfloat16 else
                         hashlib.md5(t.contiguous().numpy().tobytes()).hexdigest()[:8]
                         for t in [temporal, *conv]]
            except Exception as exc:  # never let forensics break a step
                parts = [f"err:{type(exc).__name__}:{exc}"]
            log.warning("Q35STATE rp=%d slot=%d pos=%d parts=%s", rp, slot, positions[j], ",".join(parts))

    def _relayed(self, inputs: dict, token_plan, token_ids, n: int) -> dict:
        """On the device token relay: the step's ``input_ids`` are the previous decode's device token vector (or a
        placeholder) and its ``token_control`` the plan's (the platform's _decode_token_inputs: an unchanged batch
        passes the vector through, a changed one reorders it in the graph, a host id comes in row 0)."""
        if not self.decode_token_width:
            return inputs
        prev, control = self._decode_token_inputs(token_plan, token_ids, n, self.decode_token_width)
        return {**inputs, "input_ids": prev, "token_control": control}

    def _synthetic_inputs(self, kind: str, bucket: int, context: Optional[int] = None) -> dict:
        if kind != "decode":
            raise RuntimeError(f"the Qwen3.5 giga decode runtime has no {kind} graph")
        dp = self._q35_layout.dp
        counts = decode_counts(self._q35_step_batch, dp)
        # the warm-up (no batch) pads every replica; an idle rank's dummy carries the busy replicas' counts
        replica_tokens = [int(c) for c in counts] if counts is not None and len(counts) == dp else [0] * dp
        replica_tokens = [min(c, bucket) for c in replica_tokens]
        rows = build_decode_rows(token_ids=[], positions=[], out_cache_loc=[], state_slots=[], rows=bucket,
                                 replica=self._q35_layout.replica, slot_tokens=self._q35_layout.slot_tokens,
                                 sink_slot=self._sink_slot(), replica_tokens=replica_tokens)
        if context is None:
            context = getattr(self, "_q35_ctx", None)
        return self._relayed(self._upload(rows, block_table=self._block_table(None, 0, bucket, context)), None,
                             None, 0)


# NKI's own linear-scan allocation runs out of PSUM in gdn_prefill, and its fallback trips the backend's verifier: the
# prefill segment compiles with neuronx-cc's allocation and scheduling (nki's "legacy" compile mode)
PREFILL_NKI_COMPILE_MODE = "enable-alloc=legacy"
# The decode megakernel at 16 rows per replica exhausts the Pool engine's registers under NKI's own instruction
# scheduling (NCC_IGCA094 at every context rung); in the legacy mode every bucket compiles
DECODE_NKI_COMPILE_MODE = "enable-alloc=legacy"


def with_nki_compile_mode(mode: str, fn):
    """``fn()`` with ``NKI_COMPILE_MODE`` = ``mode`` (read when a graph's NKI kernels compile, at its warm-up)."""
    import os

    held = os.environ.get("NKI_COMPILE_MODE")
    os.environ["NKI_COMPILE_MODE"] = mode
    try:
        return fn()
    finally:
        if held is None:
            os.environ.pop("NKI_COMPILE_MODE", None)
        else:
            os.environ["NKI_COMPILE_MODE"] = held


def track_slot_of(fb, sink: int, size: int) -> int:
    """The slot that also takes the state the forward of ``fb`` (one request's chunk) ends with: SGLang's tracking slot
    when the batch asks for a checkpoint (upstream's mamba extra buffer: ``mamba_track_indices`` where
    ``mamba_track_mask`` is set), else ``sink``. A checkpoint the batch asks for inside the forward is refused: the
    GDN prefill keeps only the state the forward ends with (``track_chunk 0``).

    A decode batch takes ``sink``: upstream sets ``mamba_track_mask`` when a decode step reaches the tracking grid
    but no seqlens, so the ones the batch carries are its prefill's (the first decode step after a prompt still
    carries the tracking hint of that prompt's prefill). The decode megakernel writes no checkpoint, and the tree never
    keys one past a prompt's last whole chunk (chunk_keyed_tree.insert_cap), so upstream's finish donation takes the
    prefill's or none."""
    from sglang_neuron.mem_cache.state_checkpoint import capture_boundary

    mode = getattr(fb, "forward_mode", None)
    if mode is not None and mode.is_decode():
        return int(sink)
    mask = getattr(fb, "mamba_track_mask", None)
    if mask is None or not bool(mask.to("cpu").reshape(-1)[0]):
        return int(sink)
    start = int(fb.extend_prefix_lens.to("cpu").reshape(-1)[0])
    extend = int(fb.seq_lens.to("cpu").reshape(-1)[0]) - start
    capture_boundary(start, extend, int(fb.mamba_track_seqlens.to("cpu").reshape(-1)[0]), track_chunk=0)
    slot = int(fb.mamba_track_indices.to("cpu").reshape(-1)[0])
    if not 1 <= slot <= size:
        raise RuntimeError(f"the request's tracking slot {slot} is outside the state pool's slots 1 .. {size}")
    return slot


class Qwen35PrefillRuntime(NeuronModelRuntime):
    """The platform runtime with the giga prefill's inputs (``graph.Qwen35PrefillGraph``).

    On a slot pool a prefill step is one request per replica whose prompt is exactly the served prefill length ``L``
    (the one prefill token bucket), from position 0: its tokens, its KV page's row in every attention layer's block of
    the slot pool (``a * Spa + page``, the page from its first cache slot, every token at its position in the page) and
    its GDN state slot. The warm-up and an idle replica's dummy prefill padding: page 0 rows and the state pool's sink.

    On a paged pool (``Qwen35Layout.paged``) a step is one request's chunk per replica, after any cached prefix on the
    page grid (a radix hit, or the previous chunk): the program (bucket, prior tier) is the cohort's
    (``prefix.cohort_program``, set by the step's plan: ``run_prefill`` / ``run_idle``), the inputs are
    ``prefix.chunk_tables``' with the request's state slot (the initial state) and tracking slot (``track_slot_of``).
    The programs warm as (bucket, tier) shapes, the tier in the shape's context."""

    def __init__(self, model_runner, sgl_model, phase=None):
        super().__init__(model_runner, sgl_model, phase=phase)
        refused = [what for what, on in (
            ("decode context parallelism", self.dcp is not None),
            ("pipeline stages", bool(getattr(self, "pp_active", False))),
            ("hybrid SWA memory", self.window_pool is not None),
            (f"the {self.sampler_mode!r} device sampler", self.sampler_mode != "greedy"),
        ) if on]
        if refused:
            raise ValueError(f"the Qwen3.5 giga prefill inputs are built for none of: {', '.join(refused)}")
        layout = self.model_graph.layout
        buckets = list(self.buckets.prefill_token_buckets)
        self._q35_program = None
        if layout.paged:
            self._init_paged(layout, max(buckets))
            return
        if len(buckets) != 1:
            raise ValueError(f"the Qwen3.5 giga prefill serves one prompt length (SGLANG_NEURON_NUM_BATCHED_TOKENS_"
                             f"BUCKETS with one value); the buckets are {buckets}")
        self.seq_len = int(buckets[0])
        from .graph import CHUNK

        if self.seq_len > layout.slot_tokens or (self.seq_len > CHUNK and self.seq_len % CHUNK) \
                or (min(self.seq_len, CHUNK) // layout.replica) * layout.replica != min(self.seq_len, CHUNK):
            raise ValueError(f"prompt length {self.seq_len}: the giga prefill runs whole {CHUNK}-token chunks (or one "
                             f"shorter chunk) split over the {layout.replica}-rank replica within the "
                             f"{layout.slot_tokens}-token KV slot")
        self._q35_layout = layout
        self._q35_dev = neuron_device()
        self._q35_copy_hosts = not device_io.on_device() or self._q35_dev.type == "cpu"
        self._q35_fb = None

    def _init_paged(self, layout, chunk: int) -> None:
        page = layout.slot_tokens
        self.chunk = int(chunk)
        self.prefill_buckets = prefix.prefill_buckets(self.chunk, page)
        self.prior_tiers = prefix.prior_tiers(layout.context, page)
        bad = [b for b in self.prefill_buckets if b % page or b % layout.replica or (b // layout.replica) * layout.world
               % 128]
        if bad:
            raise ValueError(f"prefill buckets {bad}: a chunk bucket must be whole {page}-token KV pages, split over "
                             f"the {layout.replica}-rank replica into rows the {layout.world}-rank EP gathers in 128s "
                             "(--chunked-prefill-size, --page-size)")
        self.seq_len = self.chunk
        self._q35_layout = layout
        self._q35_dev = neuron_device()
        self._q35_copy_hosts = not device_io.on_device() or self._q35_dev.type == "cpu"
        self._q35_fb = None

    # an idle replica's padding prefill is launched and never read under async execution, and a call below its rows
    # (an aggregated request's prefill) is one pending result
    host_input_ids = True

    def program(self, tokens, prefixes) -> "prefix.PrefillProgram":
        """The cohort's program for these prefilling replicas' token counts and cached prefixes (paged pool)."""
        return prefix.cohort_program(tokens, prefixes, self.prefill_buckets, self.prior_tiers)

    def _init_decode_token_io(self) -> None:
        self.decode_token_width = 0
        self.ingraph_reorder = False

    def _warm_shapes(self) -> list:
        if self._q35_layout.paged:
            return [("prefill", b, t) for b in self.prefill_buckets for t in self.prior_tiers]
        return [("prefill", self.seq_len, None)]

    def warmup_all(self):
        return with_nki_compile_mode(PREFILL_NKI_COMPILE_MODE, super().warmup_all)

    def run_prefill(self, fb, md, bucket=None, pp_proxy=None):
        """``bucket``: the step's token bucket, or on a paged pool the cohort's ``prefix.PrefillProgram``."""
        self._q35_fb = fb
        if isinstance(bucket, prefix.PrefillProgram):
            self._q35_program, bucket = bucket, bucket.bucket
        return super().run_prefill(fb, md, bucket, pp_proxy=pp_proxy)

    def run_idle(self, plan):
        """An idle replica's padding prefill of the cohort's program (paged: the plan's context is its prior tier)."""
        if self._q35_layout.paged:
            self._q35_program = prefix.PrefillProgram(int(plan.bucket), int(plan.context or 0))
        return super().run_idle(plan)

    def _pools(self):
        full = getattr(self.model_runner.token_to_kv_pool, "full_kv_pool", None)
        return full, self.model_runner.req_to_token_pool.mamba_pool

    def _upload_chunk(self, ids, tables, state_slot: int, track_slot: int) -> dict:
        names = ["input_ids", "gdn_slot", "positions", "track_slot", "write_pages", "valid_len"]
        hosts = [ids.to(torch.int32).reshape(-1), torch.tensor([state_slot], dtype=torch.int64),
                 torch.from_numpy(tables.positions), torch.tensor([track_slot], dtype=torch.int64),
                 torch.from_numpy(tables.write_pages), torch.tensor([[tables.valid_len]], dtype=torch.int32)]
        if tables.prior_pages is not None:
            names += ["prior_pages", "prior_len"]
            hosts += [torch.from_numpy(tables.prior_pages), torch.tensor([[tables.prior_len]], dtype=torch.int32)]
        if self._q35_copy_hosts:
            hosts = [h.clone() for h in hosts]
        return dict(zip(names, device_io.upload_many(hosts, self._q35_dev)))

    def _chunk_tables(self, pages, start: int, n: int, program) -> "prefix.ChunkTables":
        full, _ = self._pools()
        return prefix.chunk_tables(pages=pages, start=start, n=n, program=program, page=self._q35_layout.slot_tokens,
                                   pages_per_layer=full.slots, attention_layers=full.layer_num,
                                   rope_len=self._q35_layout.rope_len)

    def _build_chunk(self, md, token_ids, bucket: int) -> dict:
        program = self._q35_program
        if program is None or program.bucket != bucket:
            raise RuntimeError(f"a Qwen3.5 paged prefill step of bucket {bucket} without its cohort program "
                               f"({program}); the step's plan sets it")
        if not md.is_prefill or md.batch_size != 1:
            raise RuntimeError(f"the Qwen3.5 giga prefill takes one request's chunk per replica per step; this step "
                               f"has {md.batch_size} request(s)")
        n, start = int(md.num_tokens), int(getattr(md, "cached_seq_len", 0) or 0)
        if n > self.chunk:
            raise RuntimeError(f"a prefill forward of {n} tokens exceeds the {self.chunk}-token chunk "
                               "(--chunked-prefill-size; the chunk grid cuts every forward to it)")
        page = self._q35_layout.slot_tokens
        row = md.block_table[0].numpy()
        pages = [int(p) for p in row[: -(-(start + n) // page)]]
        tables = self._chunk_tables(pages, start, n, program)
        fb = self._q35_fb
        pool = self.model_runner.req_to_token_pool
        req = fb.req_pool_indices[:1].cpu()
        state_slot = int(pool.req_index_to_mamba_index_mapping[req].reshape(-1)[0])
        _, state = self._pools()
        track = track_slot_of(fb, int(state.sink_slot), int(state.size))
        ids = torch.zeros(bucket, dtype=torch.int32)
        ids[:n] = token_ids.reshape(-1)[:n].to(torch.int32)
        return self._upload_chunk(ids, tables, state_slot, track)

    def _upload(self, ids, page: int, state_slot: int) -> dict:
        full, _ = self._pools()
        kv = (torch.arange(full.layer_num, dtype=torch.int32) * full.slots + page).reshape(-1, 1)
        hosts = [ids.to(torch.int32).reshape(-1), kv, torch.tensor([state_slot], dtype=torch.int64)]
        if self._q35_copy_hosts:
            hosts = [h.clone() for h in hosts]
        return dict(zip(("input_ids", "kv_slot", "gdn_slot"), device_io.upload_many(hosts, self._q35_dev)))

    def build_model_inputs(self, md, token_ids: torch.Tensor, bucket: int,
                           temperatures: Optional[torch.Tensor] = None, token_plan=None,
                           seeds: Optional[torch.Tensor] = None, filters: Optional[torch.Tensor] = None, *,
                           context: Optional[int] = None) -> dict:
        if self._q35_layout.paged:
            return self._build_chunk(md, token_ids, bucket)
        L, S = self.seq_len, self._q35_layout.slot_tokens
        if not md.is_prefill or md.batch_size != 1 or md.num_tokens != L:
            raise RuntimeError(f"the Qwen3.5 giga prefill takes one request of exactly {L} tokens per replica per "
                               f"step; this step has {md.batch_size} request(s) and {md.num_tokens} tokens (launch "
                               f"with --chunked-prefill-size {L} x the DP size and prompts of {L} tokens)")
        if int(getattr(md, "cached_seq_len", 0) or 0) != 0:
            raise RuntimeError("the Qwen3.5 giga prefill runs whole prompts from position 0 (no prefix reuse)")
        loc = md.slot_mapping.reshape(-1)[:L].to(torch.int64)
        page = int(loc[0]) // S
        if page == 0 or not torch.equal(loc, page * S + torch.arange(L, dtype=torch.int64, device=loc.device)):
            raise RuntimeError("the Qwen3.5 giga prefill writes a prompt's KV at positions 0 .. L-1 of one page; the "
                               "request's cache slots are not that page's (--page-size = the KV slot)")
        fb = self._q35_fb
        pool = self.model_runner.req_to_token_pool
        req = fb.req_pool_indices[:1].cpu()
        state_slot = int(pool.req_index_to_mamba_index_mapping[req].reshape(-1)[0])
        return self._upload(token_ids.reshape(-1)[:L], page, state_slot)

    def _synthetic_inputs(self, kind: str, bucket: int, context: Optional[int] = None) -> dict:
        if kind != "prefill":
            raise RuntimeError(f"the Qwen3.5 giga prefill runtime has no {kind} graph")
        _, state = self._pools()
        if self._q35_layout.paged:
            # padding: one row on page 0 (read as the prior, written as the chunk), the state pool's sink
            if context is not None:
                tier = int(context)
            else:
                tier = self._q35_program.tier if self._q35_program is not None else 0
            program = prefix.PrefillProgram(int(bucket), tier)
            tables = self._pad_tables(program)
            sink = int(state.sink_slot)
            return self._upload_chunk(torch.zeros(bucket, dtype=torch.int32), tables, sink, sink)
        return self._upload(torch.zeros(self.seq_len, dtype=torch.int32), 0, int(state.sink_slot))

    def _pad_tables(self, program) -> "prefix.ChunkTables":
        """A padding chunk: every page is a layer's page 0, one valid row at position 0 (its state goes to the sink)."""
        import numpy as np

        full, _ = self._pools()
        page = self._q35_layout.slot_tokens
        base = (np.arange(full.layer_num, dtype=np.int64) * full.slots)[:, None]
        write = np.repeat(base, program.bucket // page, axis=1).astype(np.int32)
        prior = np.repeat(base, program.tier // page, axis=1).astype(np.int32) if program.tier else None
        return prefix.ChunkTables(positions=np.arange(program.bucket, dtype=np.int64), write_pages=write,
                                  prior_pages=prior, prior_len=0, valid_len=1)
