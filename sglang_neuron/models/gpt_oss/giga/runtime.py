# SPDX-License-Identifier: Apache-2.0
"""The GPT-OSS giga-kernel decode's runtime: the platform runtime (``model_executor.graph_runner.NeuronModelRuntime``)
with a decode input builder of its own.

The 128 scheduler ranks of the giga-kernel decode share their node's CPUs, and every device step waits for the last
rank's launch, so a rank's host work per step is on the critical path. The platform's ``build_model_inputs`` pads a
decode step's metadata to its bucket with a few torch operations per tensor, trims the sliding-window block table with a
dozen more, and uploads each changed table on its own. This builder produces the same keyword arguments in one numpy
pass over host buffers it keeps per decode shape:

- The same inputs: the same tensors with the same dtypes, shapes and values, one attention-metadata dictionary for the
  full layers and one per sliding window under every layer name, with the same keys in the same order and the same
  tensors shared between them. The decode graph, and so its NEFF, are unchanged.
- A block table, a window-trimmed table or a window offset whose value hasn't changed since the last step of its shape
  keeps its device copy, as the platform's ``_staged`` does; a changed one gets a fresh device tensor, never an in-place
  write of one a queued graph may still read.
- The step's host inputs (the slot mapping, the positions, the token control, every changed table) go to the device in
  one ``device_io.upload_many``, one switch to the side stream.
- An idle DP rank's dummy decode (``_synthetic_inputs``) has buffers and device copies of its own, so the dummy steps of
  an idle stretch don't displace the device copies of the real steps around them.

Reusing the host buffers is safe because torch-neuronx copies an ordinary host tensor into its own bounce buffer before
the upload returns (``device_io.upload_many``), so the next step can overwrite the buffer at once; only a tensor from
torch-neuronx's pinned host allocator would be read after the upload returns, and a tensor over a numpy array never is
one. A CPU-mode upload is the host tensor itself, so there the builder uploads copies.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from sglang_neuron.backend import device_io
from sglang_neuron.model_executor.graph_runner import NeuronModelRuntime
from sglang_neuron.model_executor.runtime_init import neuron_device


class _Staged:
    """A decode input that keeps its device copy while its host value is unchanged: the host buffer a step writes (and
    its tensor view), the bytes last uploaded and their device tensor."""

    __slots__ = ("host", "tensor", "kept", "device")

    def __init__(self, shape, dtype):
        self.host = np.zeros(shape, dtype)
        self.tensor = torch.from_numpy(self.host)
        self.kept = None
        self.device = None

    def changed(self) -> bool:
        # one buffer of one dtype and shape: equal bytes are equal values
        return self.device is None or self.host.tobytes() != self.kept


class _Window:
    """One sliding window's trimmed block table (``blocks`` columns from each row's start page) and offsets."""

    __slots__ = ("blocks", "table", "offset", "steps")

    def __init__(self, rows: int, blocks: int):
        self.blocks = blocks
        self.table = _Staged((rows, blocks), np.int32)
        self.offset = _Staged((rows,), np.int32)
        self.steps = np.arange(blocks, dtype=np.int64)[None, :]


class _DecodeShape:
    """The host buffers of one decode shape: ``rows`` (the batch bucket) and ``columns`` (the context's pages)."""

    def __init__(self, rows: int, columns: int, windows):
        self.slots = np.zeros(rows, np.int64)
        self.positions = np.zeros(rows, np.int64)
        self.ctx = np.zeros(rows, np.int64)
        self.slots_t, self.positions_t = torch.from_numpy(self.slots), torch.from_numpy(self.positions)
        self.table = _Staged((rows, columns), np.int32)
        self.flat_table = self.table.host.reshape(-1)
        self.row_base = (np.arange(rows, dtype=np.int64) * columns)[:, None]
        self.windows = [_Window(rows, min(blocks, columns)) for blocks in windows]
        self.staged = [self.table] + [s for w in self.windows for s in (w.table, w.offset)]


class GigaDecodeRuntime(NeuronModelRuntime):
    """The platform runtime with the giga-kernel decode's numpy input builder (the module docstring)."""

    # its decode graph's steps go straight to the bound executor (backend.direct_submit); graph_runner.build_compiled
    # builds the direct submit on a device for a decode family whose runtime declares it, as a megakernel phase program
    # declares it (megakernel.PhaseProgram.direct_submit)
    megakernel_direct_submit = True

    def __init__(self, model_runner, sgl_model, phase=None):
        super().__init__(model_runner, sgl_model, phase=phase)
        self._init_decode_inputs()

    def _init_decode_inputs(self) -> None:
        """The builder's constants. Its per-shape buffers are allocated at a shape's first step: the dummy's at the
        warm-up, the real steps' at the first served step."""
        self._decode_dev = neuron_device()
        # a CPU-mode upload is the host tensor itself, which the next step would overwrite
        self._decode_copy_hosts = not device_io.on_device() or self._decode_dev.type == "cpu"
        windows = list(dict.fromkeys(w for w in self.sliding if w is not None))
        self._decode_window_blocks = [self._swa_num_blocks(w) for w in windows]
        # each layer's metadata: 0 the full layers', 1 + i the i-th window's
        self._decode_layer_md = [0 if w is None else 1 + windows.index(w) for w in self.sliding]
        self._decode_shapes = {}
        self._decode_dummy = False

    def _synthetic_inputs(self, kind: str, bucket: int, context: Optional[int] = None) -> dict:
        if kind != "decode":
            return super()._synthetic_inputs(kind, bucket, context)
        # the dummy's buffers and device copies are its own (_build_decode_inputs)
        self._decode_dummy = True
        try:
            return super()._synthetic_inputs(kind, bucket, context)
        finally:
            self._decode_dummy = False

    def build_model_inputs(self, md, token_ids: torch.Tensor, bucket: int, temperatures=None, token_plan=None,
                           seeds=None, filters=None, *, context: Optional[int] = None) -> dict:
        if md.is_prefill:
            return super().build_model_inputs(md, token_ids, bucket, temperatures, token_plan, seeds, filters,
                                              context=context)
        return self._build_decode_inputs(md, token_ids, bucket, context, token_plan)

    def _build_decode_inputs(self, md, token_ids, bucket: int, context: Optional[int], token_plan) -> dict:
        """The platform's decode inputs (``build_model_inputs``) in one numpy pass and one upload."""
        n = md.batch_size
        width = self.decode_token_width
        columns = self.buckets.max_blocks(context)
        key = (self._decode_dummy, bucket, columns)
        shape = self._decode_shapes.get(key)
        if shape is None:
            shape = self._decode_shapes[key] = _DecodeShape(bucket, columns, self._decode_window_blocks)
        dev = self._decode_dev

        # the step's host inputs, in upload order: slot mapping, positions, then the token control if it is the host's
        hosts = [shape.slots_t, shape.positions_t]
        input_ids, token_control, control = self._decode_token_sources(token_plan, token_ids, n, width)
        if control is not None:
            hosts.append(control)
        positions, slots, ctx, table = shape.positions, shape.slots, shape.ctx, shape.table.host
        positions[:n] = md.positions.numpy()[:n]
        slots[:n] = md.slot_mapping.numpy()[:n]
        table.fill(-1)
        source = md.block_table.numpy()
        w = min(source.shape[1], columns)
        table[:n, :w] = source[:, :w]
        # each row's position, which the sliding-window trim starts from
        ctx[:n] = md.seq_lens.numpy()[:n] - 1
        if n < bucket:
            positions[n:] = 0
            slots[n:] = 0
            # padded rows: position 0 over SGLang's reserved page 0, so every row has exactly one valid key
            table[n:, 0] = 0
            ctx[n:] = 0
        page = self.block_size
        for window in shape.windows:
            start = np.maximum(ctx // page - (window.blocks - 1), 0)
            index = start[:, None] + window.steps
            np.take(shape.flat_table, shape.row_base + np.minimum(index, columns - 1), out=window.table.host,
                    mode="clip")
            # columns past the table carry the -1 sentinel, not a clamped duplicate
            if int(start.max()) + window.blocks > columns:
                window.table.host[index >= columns] = -1
            np.multiply(start, page, out=window.offset.host, casting="unsafe")
        changed = [staged for staged in shape.staged if staged.changed()]
        hosts += [staged.tensor for staged in changed]
        if self._decode_copy_hosts:
            hosts = [host.clone() for host in hosts]
        uploaded = device_io.upload_many(hosts, dev)
        for staged, tensor in zip(changed, uploaded[len(uploaded) - len(changed):]):
            staged.device = tensor
            staged.kept = staged.host.tobytes()
        slot_mapping, positions_d = uploaded[0], uploaded[1]
        if control is not None:
            token_control = uploaded[2]
        if self._decode_cached_zero is None:
            self._decode_cached_zero = device_io.upload(torch.tensor([[0]], dtype=torch.int32), dev)
        sampling_positions = self._decode_sampling_positions.get(bucket)
        if sampling_positions is None:
            sampling_positions = self._decode_sampling_positions[bucket] = device_io.upload(
                torch.arange(bucket, dtype=torch.long), dev)

        full = {
            "block_table_tensor": shape.table.device, "slot_mapping": slot_mapping, "max_query_len": 1,
            "block_size": self.block_size, "max_blocks_per_seq": columns, "decode_token_threshold": 1,
            "cached_seq_len": self._decode_cached_zero, "kv_segment_size": 0,
        }
        metadata = [full]
        for window in shape.windows:
            metadata.append({**full, "block_table_tensor": window.table.device, "max_blocks_per_seq": window.blocks,
                             "swa_kv_pos_offset": window.offset.device})
        attn_metadata = dict(zip(self.layer_names, [metadata[i] for i in self._decode_layer_md]))
        return dict(input_ids=input_ids, positions=positions_d, attn_metadata=attn_metadata,
                    sampling_positions=sampling_positions, rank=self.rank_tensor, token_control=token_control)
