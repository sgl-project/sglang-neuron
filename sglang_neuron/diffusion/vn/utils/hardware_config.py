# SPDX-License-Identifier: Apache-2.0
"""The logical NeuronCore size (LNC) that NKI kernel launch grids use."""

from sglang_neuron.diffusion import envs
from sglang_neuron.diffusion.neuron_api import coerce_lnc


def get_logical_nc_config() -> int:
    """``NEURON_LOGICAL_NC_CONFIG`` (1 or 2; 2 when unset), which an NKI launch grid must match.

    Another value raises ``ValueError`` here rather than inside the compiler.
    """
    return coerce_lnc(envs.NEURON_LOGICAL_NC_CONFIG, supported_lnc=(1, 2))
