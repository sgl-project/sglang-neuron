# SPDX-License-Identifier: Apache-2.0
"""Hardware check of a node-wide all_to_all: the runtime's one all_to_all over all ranks against
``sglang_neuron.nki.ep_all_to_all``, under rank skew, on every rank and execution.

Runs under nki-library's collective test harness (github.com/aws-neuron/nki-library, validated at ad2f1b9), from its
checkout, with this repository on PYTHONPATH:

    cp tools/collectives/all_to_all_check.py <nki-library>/test/integration/
    cd <nki-library>
    PYTHONPATH=$PWD/src:$PWD:<this repo> A2A_MODE=ep9 A2A_SKEW=1 A2A_POISON=1 A2A_RUNS=8 \
      python -m pytest -c test/pytest.ini -p test.utils.pytest_plugin test/integration/all_to_all_check.py \
      --platform-target trn3 --test-mode compile-and-infer -rA -s
    python <this repo>/tools/collectives/check_all_to_all.py <the run's infer_result dir> --cols 7168 --runs 8

The harness passes whatever the outputs hold (its reference is a placeholder); ``check_all_to_all.py`` compares them
byte for byte.

Each rank sends a seeded random ``[4096, A2A_COLS]`` uint8 buffer (``A2A_COLS=7168``: 29 MB per rank), one 32-row block
per destination. ``A2A_MODE``:

- ``flat``, ``ep``: one exchange, the runtime's all_to_all or ``ep_all_to_all``;
- ``flat<k>``, ``ep<k>``: ``k`` independent exchanges back to back, one per slice of the columns, each from its own
  copy of that slice of the source and into its own destination, which together deliver what one does.

``A2A_SKEW=1`` has every rank stream 235 MB of HBM through SBUF before it writes its source, so ranks reach the exchange
at different times. ``A2A_POISON=1`` fills every exchange's destination before the first exchange, and each of
``ep_all_to_all``'s staging tensors as it allocates it (the check hands it an allocator that fills), with 0xA5, so a
byte an exchange never wrote shows as 0xA5 instead of the previous execution's identical bytes. The runtime's own
buffers (a relay's channel buffer, a receiver's proxy buffer) are not the kernel's to fill: a stale byte from one of
them shows as the wrong sender's or the wrong exchange's data.

The harness compiles the kernel under NKI's tracer frontend (``NKICompilationMode.tracer``).
"""
import os

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nki.collectives import ReplicaGroup

from sglang_neuron.nki import ep_all_to_all as ep
from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, NKICompilationMode, Platforms
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework

N = 128
DEVICE_RANKS = 8  # LNC1
ROWS, COLS = 4096, int(os.environ.get("A2A_COLS", "7168"))
MODE = os.environ.get("A2A_MODE", "ep")
SKEW = int(os.environ.get("A2A_SKEW", "1"))
POISON = int(os.environ.get("A2A_POISON", "1"))
KIND = MODE.rstrip("0123456789")
SLICES = int(MODE[len(KIND):] or 1)
assert KIND in ("flat", "ep") and 1 <= SLICES <= COLS, MODE
BOUNDS = [COLS * i // SLICES for i in range(SLICES + 1)]  # exchange i carries columns [BOUNDS[i], BOUNDS[i + 1])


def _fill(t):
    if not POISON:
        return
    rows, cols = t.shape
    tile = nl.ndarray((128, cols), dtype=t.dtype, buffer=nl.sbuf)
    nisa.memset(dst=tile, value=0xA5)
    for r0 in range(0, rows, 128):
        n = min(128, rows - r0)
        nisa.dma_copy(dst=t[nl.ds(r0, n), :], src=tile[0:n, :])


def _skew(junk):
    rows, cols = junk.shape
    acc = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for r0 in range(0, rows, 128):
        tile = nl.ndarray((128, cols), dtype=junk.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=junk[nl.ds(r0, 128), :])
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tile, op=nl.add)
    sink = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.private_hbm, name="a2a_skew_sink")
    nisa.dma_copy(dst=sink, src=acc)


def _hbm_copy(dst, src, dst_cols=None, src_cols=None):
    """dst[:, dst_cols] = src[:, src_cols] (all columns by default), HBM to HBM, 128 rows per DMA."""
    rows = src.shape[0]
    dst_cols = dst_cols or slice(0, dst.shape[1])
    src_cols = src_cols or slice(0, src.shape[1])
    for r0 in range(0, rows, 128):
        n = min(128, rows - r0)
        nisa.dma_copy(dst=dst[nl.ds(r0, n), dst_cols], src=src[nl.ds(r0, n), src_cols])


class _FilledStaging:
    """nki.language as ep_all_to_all sees it under A2A_POISON=1: every tensor it allocates comes filled."""

    def __getattr__(self, attr):
        return getattr(nl, attr)

    @staticmethod
    def ndarray(*args, **kwargs):
        tensor = nl.ndarray(*args, **kwargs)
        _fill(tensor)
        return tensor


if POISON:
    ep.nl = _FilledStaging()


@nki.jit
def all_to_all_check(x, junk, world=None):
    rows, cols = x.shape
    if SKEW:
        _skew(junk)
    # each exchange's source staged in kernel HBM by DMA, as a kernel's send buffer is, all before the first exchange
    srcs, dsts = [], []
    for i in range(SLICES):
        width = BOUNDS[i + 1] - BOUNDS[i]
        srcs.append(nl.ndarray((rows, width), dtype=x.dtype, buffer=nl.private_hbm, name=f"a2a_src{i}"))
        dsts.append(nl.ndarray((rows, width), dtype=x.dtype, buffer=nl.private_hbm, name=f"a2a_dst{i}"))
        _hbm_copy(srcs[i], x, src_cols=slice(BOUNDS[i], BOUNDS[i + 1]))
        _fill(dsts[i])
    for i in range(SLICES):
        if KIND == "flat":
            ncc.all_to_all(srcs=[srcs[i]], dsts=[dsts[i]], replica_group=world, collective_dim=0)
        else:
            ep.ep_all_to_all(srcs[i], dsts[i], n_ranks=N, device_ranks=DEVICE_RANKS, name=f"a2a_ep{i}")
    out = nl.ndarray((rows, cols), dtype=x.dtype, buffer=nl.shared_hbm)
    for i in range(SLICES):
        _hbm_copy(out, dsts[i], dst_cols=slice(BOUNDS[i], BOUNDS[i + 1]))
    return out


def create(rank_id):
    junk = np.ones((128 * 112, 4096), np.float32) if SKEW else np.ones((128, 8), np.float32)
    return {"x": np.random.default_rng(1000 + rank_id).integers(0, 256, size=(ROWS, COLS), dtype=np.uint8),
            "junk": junk, "world": ReplicaGroup([list(range(N))])}


def ref(x, junk, world=None):
    return {"out": np.zeros((ROWS, COLS), np.uint8)}  # placeholder: check_all_to_all.py compares the bytes


@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
def test_all_to_all_check(test_manager, platform_target):
    CollectiveUnitTestFramework(test_manager, all_to_all_check, ref, create, N).run_test(
        test_config=None,
        compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1,
                                   nki_compilation_mode=NKICompilationMode.tracer),
        output_keys=["out"], rtol=1.0, atol=1e9,
        inference_args=InferenceArgs(collective_ranks=N, enable_determinism_check=False,
                                     num_runs=int(os.environ.get("A2A_RUNS", "8")), profile_all_runs=True),
    )
