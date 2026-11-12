# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Descriptor generation for the small per-layer parameter loads (RMSNorm gammas, RoPE tables).

These loads read kernel inputs, so the compiler picks the software DGE for them by default: GpSimd generates
each one's descriptors (GpSimd time for each DMA, 1-8 DMAs per load). In a multi-layer trace the next
layer's loads are scheduled into the MoE's tail, on the same in-order GpSimd queue as the routed
ReduceScatter trigger (78L layer 41: the input-norm gamma's two
loads and their waits sit between the last down-projection eviction and tile 0's scatter descriptors, the
q-norm gamma's loads between and after the two scatters). :func:`small_load` moves them to a hardware DGE
queue instead.
"""

import os

import nki.isa as nisa

from .kernel_assert import kernel_assert

SMALL_LOADS_HWDGE = os.environ.get("GLM52K_SMALL_LOADS_HWDGE", "0")
"""Hardware DGE for the small parameter loads that go through :func:`small_load` (the input-norm, q-norm and
kv-norm gammas, the RoPE tables): ``0`` (default) keeps the compiler's choice (software DGE on GpSimd); ``1`` or
``sp`` triggers them on the Sync engine, ``act`` on the Scalar (Activation) engine. Same bytes, same order of
use; only the queue that generates their descriptors changes. A casting load (the MoE layer's bf16 gamma into fp32)
stays on the software DGE: the hardware DGE does not cast."""
_SMALL_LOADS_HWDGE_MODES = ("0", "1", "sp", "act")

def small_load(dst, src):
    """``nisa.dma_copy(dst=dst, src=src)`` for a small parameter tile, on a hardware DGE queue when
    ``SMALL_LOADS_HWDGE`` asks for it.

    Args:
        dst (nl.NkiTensor): SBUF destination tile.
        src (nl.NkiTensor): HBM source (a kernel input; may be a reshape or partition broadcast view).

    Returns:
        The ``dma_copy`` instruction handle.
    """
    kernel_assert(
        SMALL_LOADS_HWDGE in _SMALL_LOADS_HWDGE_MODES,
        f"GLM52K_SMALL_LOADS_HWDGE must be one of {_SMALL_LOADS_HWDGE_MODES}, got {SMALL_LOADS_HWDGE!r}",
    )
    # The hardware DGE takes no casting DMA (the compiler rejects a user-assigned HWDGE on one, NCC_IBIR098): a
    # bf16 gamma loaded into an fp32 tile keeps the software DGE.
    if SMALL_LOADS_HWDGE == "0" or src.dtype != dst.dtype:
        dma = nisa.dma_copy(dst=dst, src=src)
    else:
        engine = nisa.engine.scalar if SMALL_LOADS_HWDGE == "act" else nisa.engine.sync
        dma = nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=engine)
    return dma
