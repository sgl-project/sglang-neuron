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

"""Give the Kimi-K3 kernels the Trn2 / Trn3 differences they branch on.

Call these functions only while a kernel is traced: they read the target of the trace with
``nisa.get_nc_version()``. Trn2 is NeuronCore-v3 (``gen3``); Trn3 is NeuronCore-v4 (``gen4``, Trn3 and
Trn3 pre-production). Other generations are refused.

A kernel keeps its Trn3 code as it is and puts a Trn2 change behind ``is_trn2()``, so that the Trn3
programs do not change. ``dma_priority`` gives the Trn3 call ``priority`` as given, so its keyword arguments do not
change.
"""

import nki.isa as nisa
import nki.language as nl

from .kernel_assert import kernel_assert

TRN3_FP8_E4M3_MAX = 448.0
"""The largest finite value of the Trn3 (OCP) ``float8_e4m3fn``."""

TRN2_FP8_E4M3_MAX = 240.0
"""The largest finite value of the Trn2 ``float8_e4m3``. Codes up to this value are the same in both formats."""


def is_trn2() -> bool:
    """Return True if the kernel is traced for Trn2 (NeuronCore-v3)."""
    version = nisa.get_nc_version()
    kernel_assert(
        version in (nisa.nc_version.gen3, nisa.nc_version.gen4),
        f"The Kimi-K3 kernels support Trn2 and Trn3 only, not {version}.",
    )
    return version == nisa.nc_version.gen3


def is_trn3() -> bool:
    """Return True if the kernel is traced for Trn3 (NeuronCore-v4)."""
    return not is_trn2()


def fp8_e4m3():
    """Return the FP8 e4m3 type of the target: ``float8_e4m3fn`` on Trn3, ``float8_e4m3`` on Trn2."""
    return nl.float8_e4m3 if is_trn2() else nl.float8_e4m3fn


def fp8_e4m3_max() -> float:
    """Return the clamp for values cast to ``fp8_e4m3()``: 448 on Trn3, 240 on Trn2."""
    return TRN2_FP8_E4M3_MAX if is_trn2() else TRN3_FP8_E4M3_MAX


def dma_priority(priority) -> dict:
    """Return the DMA priority keyword arguments for ``nisa.dma_copy``.

    Trn2 has no DMA priority, so it gets no argument. Trn3 gets ``priority`` as given. ``None`` gives no
    argument on both.

    Example:
        >>> nisa.dma_copy(dst=dst, src=src, **dma_priority(3))
    """
    if priority is None or is_trn2():
        return {}
    return {"priority": priority}
