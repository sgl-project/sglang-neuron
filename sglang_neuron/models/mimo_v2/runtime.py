# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's runtime: Hy3's NEURON_MTP decode runtime (models/hy_v3/runtime.py: ``run_mtp_step`` and the step's
carry slots and control in the decode inputs), with the sliding layers' decode tables trimmed wide enough for a
step's span.

The runtime trims a sliding layer's table from the host's length, which lags the device's by up to a step's tokens,
and an MTP step builds its tables over ``2 x SPEC`` positions past it (HyV3Runtime.run_mtp_step); a row's span then
reads its window from up to ``SPEC`` positions behind the host's length to ``2 x SPEC`` past it, one page more than a
one-token decode's window, which the trimmed table takes."""

from sglang_neuron.models.hy_v3.runtime import HyV3Runtime


class MiMoV2Runtime(HyV3Runtime):
    def _swa_num_blocks(self, sliding_window: int) -> int:
        if not self.mtp_on:
            return super()._swa_num_blocks(sliding_window)
        from sglang_neuron.models.mimo_v2 import mtp_step

        # the window, its tail and the span's reach, rounded up to the base rule's alignment
        return super()._swa_num_blocks(sliding_window + 3 * mtp_step.SPEC + self.block_size)
