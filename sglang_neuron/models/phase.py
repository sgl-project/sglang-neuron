# SPDX-License-Identifier: Apache-2.0
"""The role a process serves: "all" on an aggregated server, else its --disaggregation-mode."""


def model_phase() -> str:
    from sglang.srt.runtime_context import get_disagg

    phase = str(get_disagg().disaggregation_mode)
    if phase in ("null", "none", ""):
        # an aggregated server: one process serves both phases ("all"), and
        # every model host builds a prefill and a decode graph for it
        # (models/host.py)
        return "all"
    if phase not in ("prefill", "decode"):
        raise ValueError(
            f"Neuron models serve an aggregated server or a prefill or decode role, not "
            f"--disaggregation-mode {phase}"
        )
    return phase
