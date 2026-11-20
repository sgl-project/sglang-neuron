# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's runtime: the platform runtime with its DCP decode contexts aligned for the decode kernel's block reads,
the decode step of NEURON_MTP (mtp_step.py), and the sliding layers' decode tables trimmed wide enough for a step's
span. Written after Hy3's runtime (models/hy_v3/runtime.py) and kept apart from it, so neither model's changes reach
the other.

nkilib's attention_block_tkg reads a row's ``s_prior`` cache tokens in blocks of ``gcd(block, s_prior / 256)`` (two
LNC2 programs of P_MAX tokens each); a DCP share of whole ``LOCAL_ALIGN`` tokens reads blocks of 16.

Under NEURON_MTP every decode step is the MTP step: ``run_mtp_step`` builds the step's tables over the positions the
device may reach (the host's length lags the device's by up to one step's tokens, and a step writes ``SPEC`` more), and
the step's carry slots and control ride in the decode inputs (``mtp_row_slots``, ``mtp_ctrl``). The graph takes its
tokens from its carry and returns each row's kept run at the decode width ([width, SPEC]).

The runtime trims a sliding layer's table from the host's length; a row's span then reads its window from up to
``SPEC`` positions behind the host's length to ``2 x SPEC`` past it, one page more than a one-token decode's window,
which the trimmed table takes."""

import torch

from sglang_neuron.distributed.decode_context_parallel import decode_context
from sglang_neuron.model_executor.graph_runner import NeuronModelRuntime
from sglang_neuron.models.mimo_v2 import mtp_step

# tokens of a rank's decode context share: 16 x 256, which keeps 16-token blocks of the 128-token pages
LOCAL_ALIGN = 4096


def aligned_context(context: int, size: int, block: int, tile: int) -> int:
    """The platform's DCP decode context for ``context`` tokens, rounded up to whole LOCAL_ALIGN-token shares."""
    tokens = decode_context(context, size, block, tile)
    quantum = size * LOCAL_ALIGN
    return -(-tokens // quantum) * quantum


class MiMoTokenIO(torch.nn.Module):
    """The platform's fixed-width decode token relay (model_executor/token_io.FixedDecodeTokenIO) without its graph ops:
    the graph takes the previous decode's device tokens and ``token_control`` as they are and assembles its ids in its
    embedding kernel (kernels/head.embed), and its sampler kernel writes the tokens at the width
    (model.MiMoV2Graph._decode_kernels)."""

    def __init__(self, model, width: int):
        super().__init__()
        self.model = model
        self.width = width

    def forward(self, input_ids, positions, token_control=None, token_ids_out=None, **kwargs):
        if token_ids_out is not None:
            raise ValueError("MiMo-V2 decodes without pipeline stages (no token slot to sample into)")
        if not 0 < positions.shape[0] <= self.width:
            raise ValueError("Decode metadata exceeds the token capacity")
        return self.model(input_ids=input_ids, positions=positions, token_control=token_control,
                          token_width=self.width, **kwargs)


class MiMoV2Runtime(NeuronModelRuntime):
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
            super()._init_decode_token_io()
            from sglang_neuron.model_executor.token_io import FixedDecodeTokenIO

            if isinstance(self.execution_model, FixedDecodeTokenIO):
                # the graph assembles its ids from the relay and leaves its tokens at the width in its own kernels
                self.execution_model = MiMoTokenIO(self.execution_model.model, self.decode_token_width)
            return
        on = self.async_execution and not self.host_input_ids
        self.decode_token_width = int(self.buckets.num_seqs_buckets[-1]) if on else 0
        self.decode_token_assemble = False
        self.model_graph.mtp_width = self.decode_token_width or int(self.buckets.num_seqs_buckets[-1])

    def run_mtp_step(self, batch, slots: list, fresh: list):
        """One MTP step of ``batch``'s requests (their carry ``slots``, and whether each is new to its slot)."""
        from sglang_neuron.attention.backend import NeuronForwardMetadata

        n = len(batch.reqs)
        bucket = self.buckets.seqs_bucket(n)
        lens = torch.tensor([len(r.origin_input_ids) + len(r.output_ids) for r in batch.reqs], dtype=torch.long)
        reach = lens + 2 * mtp_step.SPEC
        rows = batch.req_pool_indices.to("cpu", torch.long).reshape(-1)[:n]
        md = NeuronForwardMetadata(is_prefill=False, batch_size=n, num_tokens=n,
                                   block_table=self._allocated_table(batch, rows, reach, n),
                                   slot_mapping=torch.zeros(n, dtype=torch.long), seq_lens=reach, positions=lens - 1)
        self._mtp_host = mtp_step.host_rows(batch, slots, fresh, bucket, self.mtp_num_slots)
        try:
            kwargs = self.build_model_inputs(md, token_ids=torch.zeros(n, dtype=torch.int32), bucket=bucket,
                                             **self._sampling_kwargs(batch),
                                             context=self.buckets.decode_context(int(reach.max())))
            return self._run(kwargs, n_real=n, phase="decode")
        finally:
            self._mtp_host = None

    def _allocated_table(self, batch, rows, reach, n: int) -> torch.Tensor:
        """The step's block table over ``reach``, -1 past each request's own pages (``kv_allocated_len``). A request
        table row past its allocation still holds the slots of the row's earlier requests, pages that are now another
        request's or the radix tree's; the step writes a token there whenever its position passes the allocation (a
        draft's), so those columns must not name a page."""
        table = self.model_runner.attn_backend._block_table(rows, reach, n)
        page = self.model_runner.attn_backend.page_size
        allocated = torch.tensor([int(r.kv.kv_allocated_len) for r in batch.reqs], dtype=torch.long)
        live = torch.arange(table.shape[1], dtype=torch.long).reshape(1, -1) * page < allocated.reshape(-1, 1)
        return torch.where(live, table, torch.full_like(table, -1))

    def build_model_inputs(self, md, token_ids, bucket, temperatures=None, token_plan=None, seeds=None, filters=None,
                           *, context=None) -> dict:
        out = super().build_model_inputs(md, token_ids, bucket, temperatures, token_plan, seeds, filters,
                                         context=context)
        if md.is_prefill or not self.mtp_on:
            return out
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
        from sglang_neuron.backend.device_io import upload_many

        # on the side stream (device_io.upload_many): a plain copy would queue behind the step still running, and hold
        # the host until it ends
        out["mtp_row_slots"], out["mtp_ctrl"] = upload_many([row_slots, ctrl], neuron_device())
        return out

    def _swa_num_blocks(self, sliding_window: int) -> int:
        if not self.mtp_on:
            return super()._swa_num_blocks(sliding_window)
        # the window, its tail and the span's reach, rounded up to the base rule's alignment
        return super()._swa_num_blocks(sliding_window + 3 * mtp_step.SPEC + self.block_size)
