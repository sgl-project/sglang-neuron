# SPDX-License-Identifier: Apache-2.0
"""Hy3's runtime: the platform runtime with its DCP decode contexts aligned for the decode kernel's block reads, and
the decode step of NEURON_MTP (mtp_step.py).

nkilib's attention_block_tkg reads a row's ``s_prior`` cache tokens in blocks of ``gcd(block, s_prior / 256)`` (two
LNC2 programs of P_MAX tokens each; attention_tkg_utils' block resize). The platform's DCP context gives each rank a
share of whole 256-token tiles, so a 262144-token context's share (33280 tokens) reads blocks of 2 tokens: every
decode step's KV comes in two-token DMAs. A share of whole ``LOCAL_ALIGN`` tokens reads blocks of 16.

Under NEURON_MTP every decode step is the MTP step: the worker's ``run_mtp_step`` builds the step's tables over the
positions the device may reach (the host's length lags the device's by up to one step's tokens, and a step writes
``SPEC`` more), and the step's carry slots and control ride in the decode inputs (``mtp_row_slots``, ``mtp_ctrl``). The
graph takes its tokens from its carry and returns each row's kept run at the decode width ([width, SPEC]), so the
runtime keeps the width and neither assembles ids nor pads tokens."""

import torch

from sglang_neuron.distributed.decode_context_parallel import decode_context
from sglang_neuron.model_executor.graph_runner import NeuronModelRuntime

# tokens of a rank's decode context share: 16 x 256, which keeps 16-token blocks of the 128-token pages
LOCAL_ALIGN = 4096


def aligned_context(context: int, size: int, block: int, tile: int) -> int:
    """The platform's DCP decode context for ``context`` tokens, rounded up to whole LOCAL_ALIGN-token shares."""
    tokens = decode_context(context, size, block, tile)
    quantum = size * LOCAL_ALIGN
    return -(-tokens // quantum) * quantum


class HyV3Runtime(NeuronModelRuntime):
    _mtp_host = None

    def dcp_decode_context(self, context: int, size: int, block: int, tile: int) -> int:
        return aligned_context(context, size, block, tile)

    @property
    def mtp_on(self) -> bool:
        return self.graphs == "decode" and getattr(self.model_graph, "mtp", None) is not None

    @property
    def mtp_num_slots(self) -> int:
        return self.model_graph.mtp_slots

    def _init_decode_token_io(self) -> None:
        if not self.mtp_on:
            return super()._init_decode_token_io()
        on = self.async_execution and not self.host_input_ids
        self.decode_token_width = int(self.buckets.num_seqs_buckets[-1]) if on else 0
        self.decode_token_assemble = False
        self.model_graph.mtp_width = self.decode_token_width or int(self.buckets.num_seqs_buckets[-1])

    def run_mtp_step(self, batch, slots: list, fresh: list):
        """One MTP step of ``batch``'s requests (their carry ``slots``, and whether each is new to its slot)."""
        from sglang_neuron.attention.backend import NeuronForwardMetadata
        from sglang_neuron.models.hy_v3 import mtp_step

        n = len(batch.reqs)
        bucket = self.buckets.seqs_bucket(n)
        lens = torch.tensor([len(r.origin_input_ids) + len(r.output_ids) for r in batch.reqs], dtype=torch.long)
        reach = lens + 2 * mtp_step.SPEC
        rows = batch.req_pool_indices.to("cpu", torch.long).reshape(-1)[:n]
        md = NeuronForwardMetadata(is_prefill=False, batch_size=n, num_tokens=n,
                                   block_table=self.model_runner.attn_backend._block_table(rows, reach, n),
                                   slot_mapping=torch.zeros(n, dtype=torch.long), seq_lens=reach, positions=lens - 1)
        self._mtp_host = mtp_step.host_rows(batch, slots, fresh, bucket, self.mtp_num_slots)
        try:
            kwargs = self.build_model_inputs(md, token_ids=torch.zeros(n, dtype=torch.int32), bucket=bucket,
                                             context=self.buckets.decode_context(int(reach.max())))
            return self._run(kwargs, n_real=n, phase="decode")
        finally:
            self._mtp_host = None

    def build_model_inputs(self, md, token_ids, bucket, temperatures=None, token_plan=None, seeds=None, filters=None,
                           *, context=None) -> dict:
        out = super().build_model_inputs(md, token_ids, bucket, temperatures, token_plan, seeds, filters,
                                         context=context)
        if md.is_prefill or not self.mtp_on:
            return out
        from sglang_neuron.models.hy_v3 import mtp_step
        from sglang_neuron.model_executor.runtime_init import neuron_device

        width = int(out["input_ids"].shape[0])
        host = self._mtp_host
        if host is None:
            # a warm-up or idle step: no live row, every row on the padding slot
            host = (torch.full((width,), self.mtp_num_slots, dtype=torch.int32),
                    torch.zeros(width, mtp_step.CTRL_COLUMNS, dtype=torch.int32))
        row_slots, ctrl = (t[:width] if t.shape[0] >= width else torch.cat(
            [t, torch.zeros((width - t.shape[0], *t.shape[1:]), dtype=t.dtype) + (
                self.mtp_num_slots if t.dim() == 1 else 0)]) for t in host)
        dev = neuron_device()
        out["mtp_row_slots"], out["mtp_ctrl"] = row_slots.to(dev), ctrl.to(dev)
        # the step samples in its own forward (greedy for now)
        for name in ("sampling_temperatures", "sampling_seeds", "sampling_filters"):
            out.pop(name, None)
        return out
