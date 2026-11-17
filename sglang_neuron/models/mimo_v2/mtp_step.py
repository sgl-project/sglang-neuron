# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's NEURON_MTP decode step (catalog.ModelSpec.mtp_host; speculative/neuron_mtp.py): Hy3's step
(models/hy_v3/mtp_step.py: the device carry of each request slot, the host's per-row control, the verify and the
accept), on MiMo's attention TP without DCP.

A step decodes ``SPEC = NUM_STEPS + 1`` tokens a row: its next input token and the previous step's drafts, each
attending its row's prior and the span before it (attention.span_mask: the full layers over the whole table, the
sliding ones from each token's own window start). The target's sample at every span token checks the next one; the
row keeps the drafts up to the first mismatch and the target's next token after them, so the output is the target's
own sample run. The MTP layer then drafts the next ``NUM_STEPS`` tokens (model.MiMoV2Graph._mtp_drafts)."""

from __future__ import annotations

from sglang_neuron.models.hy_v3.mtp_step import (  # noqa: F401 - the host interface (speculative/neuron_mtp.py)
    CTRL_BASE, CTRL_COLUMNS, CTRL_FRESH, CTRL_LIVE, CTRL_TOKEN, NUM_STEPS, SPEC, StepHost, accept, host_rows,
    request_rows, slot_of, span_inputs)

PREFILL_SETTING = None
# an aggregated server: the target's prefill graph writes the MTP layer's KV (mtp.py)
AGGREGATED = True


def launch_problems(server_args) -> list:
    problems = []
    if str(getattr(server_args, "disaggregation_mode", "null")) != "null":
        problems.append("a PD role (MiMo's MTP step is served on an aggregated server)")
    return problems


def launch_warnings(server_args) -> list:
    return []
