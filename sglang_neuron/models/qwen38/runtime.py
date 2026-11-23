# SPDX-License-Identifier: Apache-2.0
"""The step runtime: the platform's, with the GDN state moved between SGLang's slots and the working bank.

The GDN kernels advance state in contiguous rows, while SGLang hands each request an arbitrary slot, so the state
pool carries a working bank (``StateSpec.working_rows``) and this runtime moves rows around a step: before the
forward, batch row ``b``'s slot is copied into bank row ``b + 1`` (row 0 is the framework's scratch row); after it,
the advanced rows are copied back. Both are
the framework's own operations (``mem_cache/state_ops``), eager on the device and outside the compiled graph, so the
graph itself only ever reads a static slice of the bank and lets the kernel write it in place.

The slot of request ``r`` is ``req_index_to_mamba_index_mapping[r]``, as the Qwen3.5 runtime reads it.

A move covers a bucket of rows (``bank_moves``): the batch's rows, then rows on the pool's sink slot up to the next
power of two (capped at the bank). The row operations are eager indexed copies, which the device compiles once per
index length, so a batch size first seen while serving cost about 12 s of compiles at its first step (notes: Q15);
the buckets, each warmed when the pool is bound, have none, and a move copies at most twice the batch's rows. The
row programs themselves are ``bank_ops.KernelBankOps``.
"""

from __future__ import annotations

import logging

from sglang_neuron.model_executor.graph_runner import NeuronModelRuntime

logger = logging.getLogger(__name__)


def sink_slot(pool) -> int:
    """The state pool's sink slot: the last of every canonical tensor's slots, which no request holds."""
    return next(t.shape[1] for n, t in pool.tensors.items() if not n.startswith("work.")) - 1


def bank_width(pool) -> int:
    """The bank's rows past scratch row 0: one per request the server runs at once."""
    return next(t.shape[1] for n, t in pool.tensors.items() if n.startswith("work.")) - 1


def bank_buckets(width: int) -> list:
    """The move lengths: the powers of two below the bank's width, then the width."""
    return [1 << i for i in range(max(width - 1, 0).bit_length()) if (1 << i) < width] + [width]


def bank_moves(pool, slots) -> tuple:
    """(slots, rows) of one load or publish: batch row ``b`` in bank row ``b + 1``, padded with the sink slot to the
    smallest bucket that holds the batch."""
    width = bank_width(pool)
    if len(slots) > width:
        raise RuntimeError(f"a batch of {len(slots)} rows exceeds the state bank's {width}")
    size = next(b for b in bank_buckets(width) if b >= len(slots))
    return list(slots) + [sink_slot(pool)] * (size - len(slots)), list(range(1, size + 1))


def warm_bank_moves(pool) -> None:
    """Compile every bucket's load and publish before serving (on the sink slot only, which no request reads)."""
    for size in bank_buckets(bank_width(pool)):
        slots, rows = [sink_slot(pool)] * size, list(range(1, size + 1))
        pool.ops.load_rows(pool.tensors, slots, rows)
        pool.ops.publish_rows(pool.tensors, slots, rows)


class Qwen38Runtime(NeuronModelRuntime):
    """The platform runtime with the state bank moved around each step."""

    def _state_pool(self):
        return getattr(self.model_runner.req_to_token_pool, "mamba_pool", None)

    def _slots(self, fb) -> list:
        """This batch's state slots, in row order."""
        pool = self.model_runner.req_to_token_pool
        mapping = pool.req_index_to_mamba_index_mapping
        indices = fb.req_pool_indices
        return [int(v) for v in mapping[indices].reshape(-1).tolist()]

    def _with_bank(self, fn, fb, *args, **kwargs):
        pool = self._state_pool()
        if pool is None:
            return fn(fb, *args, **kwargs)
        batch = self._slots(fb)
        # bank row 0 is the framework's scratch row; batch row b lives in bank row b + 1
        slots, rows = bank_moves(pool, batch)
        pool.ops.load_rows(pool.tensors, slots, rows)
        # a forward that raises publishes nothing: the canonical slots keep the state before it
        out = fn(fb, *args, **kwargs)
        pool.ops.publish_rows(pool.tensors, slots, rows)
        if fb.forward_mode.is_extend():
            capture_tracks(pool, fb, batch)
        return out

    def run_decode(self, fb, *args, **kwargs):
        return self._with_bank(super().run_decode, fb, *args, **kwargs)

    def run_prefill(self, fb, *args, **kwargs):
        return self._with_bank(super().run_prefill, fb, *args, **kwargs)


def capture_tracks(pool, fb, slots) -> None:
    """The radix cache's state checkpoints of an extend forward (docs/design/recurrent-state.md, "The chunk-keyed
    tree"): where the batch asks for one (upstream's tracking buffer, ``mamba_track_mask``), the state the forward
    ends with, just published to the request's slot, is copied into the request's tracking slot. On the chunk-keyed
    tree every such boundary is a forward's end; one inside a forward is refused (the GDN prefill keeps only its final
    state, ``track_chunk`` 0)."""
    mask = getattr(fb, "mamba_track_mask", None)
    if mask is None:
        return
    host = lambda t: [int(v) for v in t.to("cpu").reshape(-1).tolist()]
    on = [bool(v) for v in mask.to("cpu").reshape(-1).tolist()]
    if not any(on):
        return
    prefix = host(fb.extend_prefix_lens)
    extend = [int(n) - p for n, p in zip(host(fb.seq_lens), prefix)]
    inner = pool.capture_tracks(slots, host(fb.mamba_track_indices), prefix, extend, host(fb.mamba_track_seqlens), on)
    if inner:
        raise RuntimeError(f"state checkpoints inside a forward {inner}: the GDN prefill keeps only the state a "
                           "forward ends with; serve on the chunk-keyed radix tree (--radix-cache-backend neuron-chunk)")
