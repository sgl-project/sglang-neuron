# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2-family GK decode's runtime: the platform runtime with the per-rank step tensors of the GK.

A decode step's inputs are the platform's (the batch's tokens, assembled on the device by the decode token I/O, and its
positions) plus, for this rank, the kernel's step tensors of its own sequence slot (models/glm52/decode/graph.py):

- ``gk_slot_rows`` ``[num_slots]``: the batch row of each kernel slot, -1 for an empty slot, and ``gk_row_slots``
  ``[width]``: the slot of each batch row (0 for a padding row), from the batch's request rows (``slot = req_pool_idx
  - 1``, models/glm52/decode/pools.py);
- the RoPE rows at the slot's position, the latent and indexer rows the step writes and the residency threshold, the
  geometry's ``step_tensors`` for this rank at that position, the position being the one
  SGLang already holds on the host. An empty slot runs at position 0 and writes no key (its latent and indexer write
  rows are -1, which the kernel skips): a request whose KV lands in the slot before it runs
  keeps every row of it, and its output is not read.

They go to the device in one ``device_io.upload_many`` with the platform's decode inputs; nothing is read back.

Under async MTP (``--speculative-algorithm NEURON_MTP``, models/glm52/mtp/graph.py) the step tensors are built
on the device from its own per-slot lengths, and a step's host inputs are the slot maps and the step host's control
(``run_mtp_step``; models/glm52/mtp/host.py), each staged: uploaded only when it changes."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from sglang_neuron.backend import device_io
from sglang_neuron.model_executor.graph_runner import NeuronModelRuntime
from sglang_neuron.models.glm52.decode.graph import POOL_TENSORS, STEP_TENSORS
from sglang_neuron.models.glm52.decode.pools import slot_of_request_row


def slot_maps(req_pool_indices, batch: int, width: int, slots: int):
    """(``slot_rows`` [slots], ``row_slots`` [width]) int32 for a batch of ``batch`` real rows, whose request rows are
    ``req_pool_indices`` (None: no request, as an idle or a warm-up step)."""
    slot_rows = np.full(slots, -1, np.int32)
    row_slots = np.zeros(width, np.int32)
    if req_pool_indices is not None:
        for row, index in enumerate(np.asarray(req_pool_indices).reshape(-1)[:batch]):
            slot = slot_of_request_row(int(index), slots)
            slot_rows[slot] = row
            row_slots[row] = slot
    return slot_rows, row_slots


# the step tensors that name the cache rows a step writes; -1 everywhere writes none
_WRITE_ROWS = ("kv_new_slot_mapping", "idx_new_slot_mapping")


def rank_step_tensors(geo, rank: int, slot_rows, positions, pool=None, slot=None) -> dict:
    """This rank's GK step tensors (``gk_geometry.step_tensors``: no entry, which the graph embeds): its slot's
    position, or for an empty slot position 0 with no row written (the module docstring). ``pool`` is the rank's
    chunk-dealt indexer pool where the geometry has one (`chunk_pool_tables`), and ``slot`` the slot the rank serves
    (``geo.slot_of(rank)``); a runtime passes both, built once."""
    from sglang_neuron.models.glm52.layout.decode.gk_geometry import step_tensors

    slot = geo.slot_of(rank) if slot is None else slot
    row = int(slot_rows[slot])
    position = int(positions[row]) if row >= 0 else 0
    step = step_tensors(geo, rank, position)
    out = {name: np.ascontiguousarray(step[name]) for name in STEP_TENSORS}
    if row < 0:
        for name in _WRITE_ROWS:
            out[name] = np.full_like(out[name], -1)
    if geo.idx_pool_chunk:
        out.update(chunk_pool_tables(geo, rank, slot_rows, positions, pool))
    return out


def chunk_pool_tables(geo, rank: int, slot_rows, positions, pool=None) -> dict:
    """This rank's chunk-dealt indexer pool tables of the step (layout/decode/idx_pool.CDP_TABLES;
    layout/decode/idx_pool.IdxChunkPool): its pool's sequences' cuts, the position of each slot's token (None for a slot
    with no request), and one new key each. A slot is a CP group (the pool's ``pool_slots``). ``pool`` is the rank's
    pool (`IdxChunkPool.for_rank`), which the runtime builds once: None builds it here."""
    from sglang_neuron.models.glm52.layout.decode.idx_pool import IdxChunkPool

    pool = IdxChunkPool.for_rank(geo, rank) if pool is None else pool
    key_rank, pool_index = pool.pool_of_rank(rank)
    bases, counts = [], []
    for slot in pool.pool_slots:
        row = int(slot_rows[slot])
        bases.append(int(positions[row]) if row >= 0 else None)
        counts.append(1 if row >= 0 else 0)
    tables = pool.tables(key_rank, pool_index, bases, counts, spec=geo.idx_pool_tokens)
    return {name: np.ascontiguousarray(tables[name]) for name in POOL_TENSORS}


class GlmGkDecodeRuntime(NeuronModelRuntime):
    """The platform runtime with the GK step tensors (the module docstring)."""

    # the decode graph's steps go straight to the bound executor (backend.direct_submit), as the GPT-OSS giga-kernel
    # decode's do: one warmed shape, its inputs and outputs bound once, so the runtime's DMA I/O ring is the cached one
    # on every step
    megakernel_direct_submit = True

    def __init__(self, model_runner, sgl_model, phase=None):
        super().__init__(model_runner, sgl_model, phase=phase)
        from sglang_neuron.models.glm52.mtp.graph import GlmGkMtpDecodeGraph

        graph = sgl_model.model_graphs["decode"]
        self._gk_geo, self._gk_rank, self._gk_slots = graph.geo, graph.rank, graph.num_slots
        # the slot this rank serves, fixed for the runtime's life
        self._gk_slot = graph.slot
        self._gk_request_rows = None
        # async MTP: the step's host control (run_mtp_step), and no token I/O of the platform's (the graph's input
        # tokens are its device carry and the control's fresh rows)
        self._gk_mtp = isinstance(graph, GlmGkMtpDecodeGraph)
        self._gk_mtp_ctrl = None
        self._gk_mtp_sampling = None
        self._gk_mtp_samples = self._gk_mtp and graph.samples
        # the rank's chunk-dealt indexer pool, a function of the geometry and the rank alone: built once, not per step
        # (the MTP graph builds its pool's tables on the device, so it needs none)
        self._gk_pool = None
        if self._gk_geo.idx_pool_chunk and not self._gk_mtp:
            from sglang_neuron.models.glm52.layout.decode.idx_pool import IdxChunkPool

            self._gk_pool = IdxChunkPool.for_rank(self._gk_geo, self._gk_rank)
        # the logit test's graph (models/glm52/logit_probe.py): the step's forced windows and the forced request's rows
        # of logits, which rank 0 writes; None on a served launch
        from sglang_neuron.models.glm52 import logit_probe
        from sglang_neuron.models.glm52.mtp.logit_test_graph import GlmGkMtpLogitTestGraph

        self._gk_logit_test = logit_probe.test_dir() if isinstance(graph, GlmGkMtpLogitTestGraph) else None
        self._gk_mtp_forced = None
        if self._gk_logit_test is not None and self._gk_rank == 0:
            logit_probe.warn_on("the decode runs the logit test graph, which forces the golden tokens of a request "
                                "that names them and, at every step, all-reduces and all-gathers that request's rows "
                                "of logits, which rank 0 reads back")

    def _init_decode_token_io(self) -> None:
        """The platform's fixed-width decode token I/O, except under async MTP: the MTP graph takes its input tokens
        from its device carry and the worker's control, and returns each batch row's accepted run ``[width, spec]`` at
        the decode width already, so it keeps the width (an asynchronous step's readback, the token relay's width) and
        neither assembles ids nor pads tokens."""
        from sglang_neuron.models.glm52.mtp.graph import GlmGkMtpDecodeGraph

        if not isinstance(self.model_graph, GlmGkMtpDecodeGraph):
            return super()._init_decode_token_io()
        on = self.graphs == "decode" and self.async_execution
        self.decode_token_width = int(self.buckets.num_seqs_buckets[-1]) if on else 0
        self.decode_token_assemble = False

    @property
    def gk_num_slots(self) -> int:
        return self._gk_slots

    def run_mtp_step(self, req_pool_indices, ctrl, batch_size: int, sampling=None, forced=None):
        """One async MTP step of ``batch_size`` requests (their request rows ``req_pool_indices``) under the host
        control ``ctrl`` ``[num_slots, CTRL_COLUMNS]`` int32 (mtp_host.StepHost.control): its pending output, each
        batch row's accepted tokens ``[rows, spec]`` (-1 after them), read back on the output thread. The graph reads
        no SGLang metadata, so the step's platform inputs are the warm-up's shapes (``_synthetic_inputs``).
        ``sampling``: a sampling graph's per-slot request table ``(sparams [num_slots, 3] float32, seeds [num_slots]
        int32)`` (tail_sampling.slot_request_row), staged: uploaded only when it changes. ``forced``: the logit test
        graph's ``(windows [num_slots, 2 * spec] int32, probe)`` (logit_probe.ForcedTokens; ``probe`` the forced
        request's ``(rid, first_row, count, slot)`` or None): rank 0 reads the step's rows of logits back, which makes
        the step synchronous, and writes them (logit_probe.write_decode_rows)."""
        if not self._gk_mtp:
            raise RuntimeError("an MTP step reached a GK decode without the MTP graph (--speculative-algorithm "
                               "NEURON_MTP builds it)")
        from sglang_neuron.model_executor.graph_runner import DUMMY_LIVE_CTX
        from sglang_neuron.models.glm52.mtp import step as mtp_step

        if tuple(ctrl.shape) != (self._gk_slots, mtp_step.CTRL_COLUMNS):
            raise ValueError(f"the MTP control is {tuple(ctrl.shape)}, the graph's [{self._gk_slots}, "
                             f"{mtp_step.CTRL_COLUMNS}]")
        if (forced is not None) != (self._gk_logit_test is not None):
            raise RuntimeError("an MTP step's forced windows go to the logit test graph only "
                               f"({'no windows' if forced is None else 'windows'} for a "
                               f"{'served' if self._gk_logit_test is None else 'logit test'} graph)")
        bucket = self.dp_identity_bucket or self.buckets.seqs_bucket(batch_size)
        self._gk_request_rows = req_pool_indices.to("cpu").reshape(-1)[:batch_size].numpy()
        self._gk_mtp_ctrl = ctrl
        self._gk_mtp_sampling = sampling
        self._gk_mtp_forced = None if forced is None else forced[0]
        try:
            md, token_ids = self._mtp_dummy_inputs(batch_size, bucket, DUMMY_LIVE_CTX)
            kwargs = self.build_model_inputs(md, token_ids=token_ids, bucket=bucket)
            res = self._run(kwargs, n_real=batch_size, phase="decode")
        finally:
            self._gk_request_rows = self._gk_mtp_ctrl = self._gk_mtp_sampling = self._gk_mtp_forced = None
        if forced is not None and forced[1] is not None:
            self._write_probe(res, *forced[1])
        return res

    def _write_probe(self, res, rid, first_row: int, count: int, slot: int) -> None:
        """Rank 0's file of a forced step's rows of logits (logit_probe.write_decode_rows)."""
        if self._gk_rank != 0:
            return
        from sglang_neuron.backend import device_io
        from sglang_neuron.models.glm52 import logit_probe

        out = logit_probe.outputs_of(res)
        geo = self._gk_geo
        want = (geo.ranks * geo.spec, geo.vocab_size // geo.ranks)
        if tuple(out["probe_logits"].shape) != want:
            raise RuntimeError(f"the logit test step's probe_logits are {tuple(out['probe_logits'].shape)}, not the "
                               f"[ranks * spec, vocab_shard] = {list(want)} of every rank's rows over its shard (a "
                               "synchronous step slices its outputs to its batch: the logit test runs the async step)")
        pending = getattr(res, "neuron_pending_sample", None)
        event = pending.launch_event if pending is not None else None
        gathered, tokens = device_io.readback_many([out["probe_logits"], out["probe_tokens"]], event)
        logit_probe.write_decode_rows(self._gk_logit_test, rid, first_row, count, gathered, tokens, geo.ranks,
                                      geo.spec, slot)

    def _mtp_dummy_inputs(self, batch_size: int, bucket: int, live_ctx: int):
        """``(metadata, token ids)`` of an MTP step of ``batch_size`` requests, built once per shape: the step's
        platform inputs are the warm-up's, which build_model_inputs only reads."""
        cache = self.__dict__.setdefault("_gk_mtp_dummy", {})
        key = (batch_size, bucket, live_ctx)
        if key not in cache:
            cache[key] = (self._mtp_metadata(batch_size, bucket, live_ctx), torch.zeros(batch_size, dtype=torch.int32))
        return cache[key]

    @staticmethod
    def _mtp_metadata(batch_size: int, bucket: int, live_ctx: int):
        """The platform's decode metadata an MTP step hands build_model_inputs: the warm-up's (graph_runner's
        _synthetic_inputs), since the MTP graph reads none of it (its positions and cache rows are its own)."""
        from sglang_neuron.attention.backend import NeuronForwardMetadata

        return NeuronForwardMetadata(
            is_prefill=False, batch_size=batch_size, num_tokens=batch_size,
            block_table=torch.zeros((batch_size, 1), dtype=torch.int32),
            slot_mapping=torch.zeros(batch_size, dtype=torch.long),
            seq_lens=torch.full((batch_size,), live_ctx, dtype=torch.long),
            positions=torch.full((batch_size,), live_ctx - 1, dtype=torch.long), cached_seq_len=0)

    def run_decode(self, fb, md, bucket: int, context: Optional[int] = None, pp_proxy=None):
        if self._gk_mtp:
            raise RuntimeError("the GK decode's MTP graph runs only the speculative worker's steps (run_mtp_step); a "
                               "plain decode step reached it")
        # the batch's request rows, which the step tensors are keyed by (build_model_inputs reads the metadata only)
        self._gk_request_rows = fb.req_pool_indices.to("cpu").reshape(-1)[: md.batch_size].numpy()
        try:
            return super().run_decode(fb, md, bucket, context, pp_proxy)
        finally:
            self._gk_request_rows = None

    def build_model_inputs(self, md, token_ids: torch.Tensor, bucket: int, temperatures=None, token_plan=None,
                           seeds=None, filters=None, *, context: Optional[int] = None) -> dict:
        out = super().build_model_inputs(md, token_ids, bucket, temperatures, token_plan, seeds, filters,
                                         context=context)
        if md.is_prefill:
            return out
        width = int(out["input_ids"].shape[0])
        slot_rows, row_slots = slot_maps(self._gk_request_rows, md.batch_size, width, self._gk_slots)
        from sglang_neuron.models.glm52.decode.pools import slot_fence

        fence = slot_fence()
        if fence is not None:
            # the slots this step writes, which a freed slot's fence waits out (decode.pools.GkSlotFence)
            fence.note_step(np.flatnonzero(slot_rows >= 0))
        if self._gk_mtp:
            from sglang_neuron.models.glm52.mtp import step as mtp_step

            # a warm-up or idle step has no control: every slot verifies its carry, and none is live
            ctrl = self._gk_mtp_ctrl
            if ctrl is None:
                ctrl = torch.zeros(self._gk_slots, mtp_step.CTRL_COLUMNS, dtype=torch.int32)
            # staged: the slot maps change only with the batch, and the control only for a fresh slot (or the
            # benchmark's synthetic counts), so a steady step uploads none of them
            dev = self._gk_device()
            out["gk_slot_rows"] = self._staged("gk_slot_rows", torch.from_numpy(slot_rows), dev)
            out["gk_row_slots"] = self._staged("gk_row_slots", torch.from_numpy(row_slots), dev)
            out["gk_mtp_ctrl"] = self._staged("gk_mtp_ctrl", ctrl, dev)
            # the platform's per-batch-row sampling inputs are not the MTP graph's: it reads its slots' request table
            for name in ("sampling_temperatures", "sampling_seeds", "sampling_filters"):
                out.pop(name, None)
            if self._gk_mtp_samples:
                from sglang_neuron.models.glm52 import tail_sampling

                sampling = self._gk_mtp_sampling
                if sampling is None:  # warm-up or idle: every slot greedy
                    sampling = (torch.zeros(self._gk_slots, tail_sampling.SLOT_PARAM_COLUMNS, dtype=torch.float32),
                                torch.zeros(self._gk_slots, dtype=torch.int32))
                sparams, seeds = sampling
                out["gk_mtp_sparams"] = self._staged("gk_mtp_sparams", sparams, self._gk_device())
                out["gk_mtp_seeds"] = self._staged("gk_mtp_seeds", seeds, self._gk_device())
            if self._gk_logit_test is not None:
                from sglang_neuron.models.glm52 import logit_probe
                from sglang_neuron.models.glm52.mtp.logit_test_graph import FORCED_INPUT

                forced = self._gk_mtp_forced  # a warm-up or idle step forces no slot
                if forced is None:
                    forced = torch.full((self._gk_slots, 2 * self._gk_geo.spec), logit_probe.UNFORCED,
                                        dtype=torch.int32)
                out[FORCED_INPUT] = device_io.upload_many([forced], self._gk_device())[0]
            return out
        step = rank_step_tensors(self._gk_geo, self._gk_rank, slot_rows, md.positions.numpy(), self._gk_pool,
                                 self._gk_slot)
        tensors = STEP_TENSORS + (POOL_TENSORS if self._gk_geo.idx_pool_chunk else ())
        names = ["gk_slot_rows", "gk_row_slots"] + [f"gk_{name}" for name in tensors]
        hosts = [torch.from_numpy(slot_rows), torch.from_numpy(row_slots)] + [torch.from_numpy(step[n])
                                                                             for n in tensors]
        out.update(zip(names, device_io.upload_many(hosts, self._gk_device())))
        return out

    def _gk_device(self):
        from sglang_neuron.model_executor.runtime_init import neuron_device

        return neuron_device()
