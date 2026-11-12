# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2 decode giga kernel (all-expert MXFP4 MoE) that the 128k and 1m points serve. See README.md."""

import os

from sglang_neuron.models.glm52.kernels import kernel_digest as _kernel_digest


def kernel_digest() -> str:
    """Return a digest of this copy's files and of the compile knobs set now.

    See ``sglang_neuron.models.glm52.kernels.kernel_digest``.
    """
    return _kernel_digest(os.path.dirname(os.path.abspath(__file__)))
