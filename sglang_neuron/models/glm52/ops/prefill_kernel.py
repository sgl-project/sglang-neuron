# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2 prefill giga kernel (GK) as one torch op: every decoder layer of a PP stage and its final norm in ONE kernel
launch, with every collective in the trace.

The kernel is the prefill kernel in ``sglang_neuron/models/glm52/kernels/prefill``: bf16 attention, MXFP4 MoE experts,
a bf16 dense MLP, trn3 at LNC=1, traced at grid ``[1]``. The copy is the plugin's own, so a change to the NKI kernel
library never changes what a stage runs.

A host without the NKI DSL (the CPU unit tests) imports this module with the kernel marked unavailable; any other import
failure of the kernel is an error, so a broken kernel copy never passes for a missing toolchain.

Example:
    >>> normed, kv_caches, idx_k_caches = prefill_giga_kernel(  # doctest: +SKIP
    ...     hidden, cos_h, sin_h, q_pos, slot_mapping, cp_rank, ep_rank,
    ...     wo_group_rank, kv_caches, idx_k_caches, **weights, **config)
"""

import logging
import os

from torch import Tensor

from sglang_neuron.models.glm52 import kernels as _kernels
from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: The digest of the kernel copy: its files and the compile knobs it reads at trace time. The NKI compile cache keys a
#: kernel on its qualname, the source of its entry function and its arguments, so it would miss an edit to a helper the
#: entry calls, or a knob; the digest goes in as an argument instead.
_KERNEL_DIGEST = _kernels.kernel_digest(os.path.join(os.path.dirname(_kernels.__file__), "prefill"))

try:
    import nki
except ModuleNotFoundError as exc:
    if exc.name != "nki":
        raise
    nki = None
    logger.info("the GLM-5.2 prefill giga kernel is unavailable: this host has no NKI DSL")

if nki is not None:
    from sglang_neuron.models.glm52.kernels.prefill.giga_kernel import (
        glm52_prefill_giga_kernel as _glm52_prefill_giga_kernel,
    )

    # The kernel's sub-kernels are undecorated so they inline into this one trace, and wrap_nki reads the argument names
    # from the signature, so the kernel itself is jitted, once at import (wrap_nki inside a traced call would cost Dynamo
    # a signature trace on every call).
    _GK_STACK_KERNEL = wrap_nki(nki.jit(_kernels.with_kernel_digest(_glm52_prefill_giga_kernel)))
else:
    _GK_STACK_KERNEL = None


def prefill_giga_kernel(*args, **kwargs) -> tuple[Tensor, ...]:
    """Run every decoder layer of the stage and the final norm in ONE giga-kernel launch.

    Takes the arguments of the kernel's ``glm52_prefill_giga_kernel``: every per-layer weight and cache is stacked on a
    leading layer axis.

    Returns:
        ``(final_normed, kv_caches, idx_k_caches)``: the final RMSNorm of the stage's last layer output, then the caches
        the kernel wrote in place.

    Raises:
        RuntimeError: the host runs no Neuron device or has no NKI DSL.
    """
    if not can_run_kernel() or _GK_STACK_KERNEL is None:
        raise RuntimeError("the GLM-5.2 prefill giga kernel runs on a Neuron device with the NKI DSL installed")
    return _GK_STACK_KERNEL[1](*args, kernel_digest=_KERNEL_DIGEST, **kwargs)
