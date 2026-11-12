# SPDX-License-Identifier: Apache-2.0
"""Per-rank host timeline of the scheduler loop (SGLANG_NEURON_HOST_TRACE=<dir>), to attribute a launch delay on any
rank to the loop phase or collection that caused it, without a profiler.

Rows of <dir>/host_rank<r>.tsv: kind, start and end (CLOCK_REALTIME ns, the clock of the system profile), and an
argument:
  recv    request receive, including the cross-rank broadcast        arg: requests received
  input   process_input_requests                                     arg: requests
  select  get_next_batch_to_run                                      arg: the chosen batch
  launch  run_batch (inputs, launch; a synchronous step also waits)  arg: the batch, '+a' when launched async
  result  process_batch_result                                       arg: the batch
  wait    inside result: the wait for an async step's outputs
  finish  inside result: finish-state handling summed over the batch arg: requests that finished
  stream  inside result: stream_output to the detokenizer            arg: requests
  gc<g>   one collection of generation g                             arg: objects collected
  idle    on_idle                                                    arg: -
  <A.b>   a SGLANG_NEURON_HOST_TRACE_SPANS target, last two parts    arg: -
  submit  inside launch: the graph call (its NEFF submission)        arg: phase
  seam_claim  PP stage, inside launch: wait for free send slots      arg: phase
  seam_wait   PP stage, inside submit: wait for the boundary's WRITE arg: kind and step (ph3)
  seam_fence  seam worker: wait for the forward that wrote the slots arg: kind and step
  seam_write  seam worker: the boundary's WRITEs to the next stage   arg: kind and step
  seam_xfer   inside seam_write: one WRITE, post to DONE             arg: kind and step
  seam_commit inside seam_write: commits and drain (none with 0009)  arg: kind and step
  seam_seen   PP stage, with seam_wait: first drain holding the notification to first holding
              the commit (an instant if it came first, or with 0009) arg: kind and step
  fence       inside submit: program-switch fence's sync and meeting arg: old->new NEFF
A batch is P<requests>x<tokens> (extend), D<rows> (decode), I (idle) or - (none). Rows are buffered and written right
after an asynchronous launch (managers.host_loop.run_batch), so the trace adds no I/O to the critical path of a step;
a stretch without async launches that buffers HARD_ROWS rows writes them where it is. An idle loop trims its own rows:
the rows of a stretch that received, launched or held no work are dropped, and the rows of a stretch that served work
are written once its loop is idle with nothing pending. SGLANG_NEURON_HOST_TRACE_RANKS (comma-separated TP ranks;
unset: every rank) traces those ranks only. The seam_fence, seam_xfer, seam_commit and seam_write rows come from the PP
seam's worker thread, the rest from the scheduler's."""

from __future__ import annotations

import atexit
import gc
import logging
import os
import threading
import time

from sglang_neuron import envs

logger = logging.getLogger(__name__)

FLUSH_ROWS = 4096
HARD_ROWS = 65536
BUSY_KINDS = frozenset(("launch", "result", "stream", "finish"))  # rows that mean the stretch served work


class HostTrace:
    def __init__(self, directory: str, rank):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"host_rank{rank}.tsv")
        with open(self.path, "w") as f:
            f.write("kind\tstart\tend\targ\n")
        self.rows = []
        self.failed = False
        self.busy = False  # a row since the last write or trim belongs to served work
        # the PP seam's worker adds rows from its own thread. _lock guards the rows (reentrant: a gc callback can run
        # inside add); _write_lock keeps file writes one at a time, outside it
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._gc_t0 = 0
        gc.callbacks.append(self._gc)
        atexit.register(self._flush_at_exit)

    def _gc(self, phase, info):
        if phase == "start":
            self._gc_t0 = time.time_ns()
        elif self._gc_t0:
            self.add(f"gc{info.get('generation', '?')}", self._gc_t0, time.time_ns(), info.get("collected", 0))
            self._gc_t0 = 0

    def add(self, kind, t0, t1, arg) -> None:
        if self.failed:
            return
        with self._lock:
            self.rows.append((kind, t0, t1, arg))
            if kind in BUSY_KINDS or (kind == "recv" and arg):
                self.busy = True
            full = len(self.rows) >= HARD_ROWS
        if full:
            self.flush()

    def at_idle(self, pending: bool) -> None:
        """The scheduler loop found nothing to run. ``pending``: it still holds work (queued, bootstrapping or in-flight
        requests), so the stretch goes on. Otherwise a stretch that served work is written now, off every step's
        critical path, and the rows of a stretch that served none (an idle loop's spin) are dropped."""
        with self._lock:
            if pending:
                self.busy = True
                return
            if not self.busy:
                self.rows.clear()
                return
        self.flush()

    def _flush_at_exit(self) -> None:
        self.flush()

    def flush(self, minimum: int = 1) -> None:
        """Append the buffered rows. A write error (full disk, directory gone) stops the trace on this rank instead of
        raising into the scheduler loop: a dead rank would hang every other rank in its collectives."""
        with self._lock:
            if self.failed or len(self.rows) < minimum:
                return
            rows, self.rows = self.rows, []
            self.busy = False
        try:
            with self._write_lock, open(self.path, "a") as f:
                f.writelines(f"{k}\t{a}\t{b}\t{c}\n" for k, a, b, c in rows)
        except OSError as e:
            self.failed = True
            logger.warning("[neuron] host trace %s stopped: %s", self.path, e)


_trace = None


def trace():
    return _trace


def traced_ranks(environ=os.environ):
    """SGLANG_NEURON_HOST_TRACE_RANKS: the TP ranks to trace (a set of ints), or None for every rank."""
    from sglang_neuron import envs

    return envs.ranks("SGLANG_NEURON_HOST_TRACE_RANKS", "TP", environ)


def start(rank, tp_rank=None) -> None:
    """Open this process's trace (named by ``rank``) if SGLANG_NEURON_HOST_TRACE is set and
    SGLANG_NEURON_HOST_TRACE_RANKS is unset or names ``tp_rank``."""
    global _trace
    directory = envs.host_trace_dir()
    if not directory or _trace is not None:
        return
    ranks = traced_ranks()
    if ranks is not None and tp_rank not in ranks:
        return
    _trace = HostTrace(directory, rank)


def describe(batch) -> str:
    if batch is None:
        return "-"
    mode = getattr(batch, "forward_mode", None)
    if mode is None:
        return "-"
    if mode.is_idle():
        return "I"
    n = len(getattr(batch, "reqs", ()) or ())
    if mode.is_extend():
        return f"P{n}x{int(getattr(batch, 'extend_num_tokens', 0) or 0)}"
    return f"D{n}"
