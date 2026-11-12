# SPDX-License-Identifier: Apache-2.0
"""The MTP decode graph with teacher forcing and raw-logit readout, for the logit test (see logit_probe.py).

Only a decode launched with ``SGLANG_NEURON_GLM52_LOGIT_TEST_DIR`` builds this graph; a served launch never traces it.
It runs the served step with one extra input and two extra outputs:

- ``gk_mtp_forced [num_slots, 2 * spec]``: each forced slot's golden tokens (-1 for the other slots). A forced slot
  keeps every draft and emits the golden tokens; the other slots behave as served.
- ``probe_logits [ranks * spec, vocab_shard]`` and ``probe_tokens [spec]``: the forced slot's full-vocabulary logits
  and the tail's tokens for its rows.
"""

from __future__ import annotations

from sglang_neuron.models.glm52 import logit_probe
from sglang_neuron.models.glm52.mtp import step as mtp_step
from sglang_neuron.models.glm52.mtp.graph import SYNTHETIC_AL_ENV, GlmGkMtpDecodeGraph, synthetic_acceptance_length

#: The extra input of the logit test graph.
FORCED_INPUT = "gk_mtp_forced"


class GlmGkMtpLogitTestGraph(GlmGkMtpDecodeGraph):
    """One rank's MTP decode graph with teacher forcing and the logit readout."""

    def __init__(self, config, geo, rank: int):
        if synthetic_acceptance_length(geo.spec) is not None:
            raise ValueError(f"{logit_probe.LOGIT_TEST_DIR_ENV} forces the golden tokens through the step's kept "
                             f"drafts, and {SYNTHETIC_AL_ENV} pins them to a benchmark's rates: unset one of them")
        super().__init__(config, geo, rank)

    def _acceptance(self, slot: dict) -> dict:
        return {"forced": logit_probe.forced_keep(slot["forced"])}

    def forward(self, input_ids, positions, attn_metadata=None, sampling_positions=None, rank=None, *,
                gk_slot_rows, gk_row_slots, gk_mtp_ctrl, gk_mtp_forced, gk_mtp_sparams=None, gk_mtp_seeds=None):
        from sglang_neuron.models.glm52.layout.decode.gk_geometry import INPUT_NORM_EPS
        from sglang_neuron.parallel.groups import get_neuron_attn_tp_group

        geo = self.geo
        slot = mtp_step.slot_inputs(gk_slot_rows, gk_mtp_ctrl, self._mtp_carry_tokens, self._mtp_slot_len,
                                    geo.context)
        slot["forced"] = gk_mtp_forced
        if self.samples:
            if gk_mtp_sparams is None or gk_mtp_seeds is None:
                raise RuntimeError("the topk_topp MTP step needs the per-slot request table (gk_mtp_sparams, "
                                   "gk_mtp_seeds; decode_runtime.run_mtp_step)")
            slot.update(sparams=gk_mtp_sparams, seeds=gk_mtp_seeds)
        hidden, target = self.target_step(slot)
        probe = logit_probe.probe_rows(hidden, gk_mtp_forced, self._my_slot, self._own_rows, self._spec_offsets,
                                       self._final_norm_w, self._lm_head_wt, get_neuron_attn_tp_group(),
                                       INPUT_NORM_EPS)
        sampled = self.accept_draft_carry(slot, target, hidden, gk_row_slots)
        sampled, carry = logit_probe.forced_outputs(sampled, self._mtp_carry_tokens, slot, gk_mtp_forced,
                                                    gk_row_slots)
        self._mtp_carry_tokens.copy_(carry)
        return {"token_ids": sampled, "probe_logits": probe,
                "probe_tokens": logit_probe.probe_tokens(target, gk_mtp_forced, geo.spec)}
