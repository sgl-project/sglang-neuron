# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 prefill graph of one pipeline stage: the stage module (prefill/stage.py) behind the platform's graph
contract.

One graph serves every rank of the stage's TP group (which is also its DCP and EP group): the rank is the ``rank``
input, never a constant. Per call (one chunk of one request, ``T`` rows of the bucket):

* inputs: ``input_ids`` int64 ``[T]`` (the first stage embeds them), ``positions`` int64 ``[T]``, ``valid_len`` and
  ``cached_len`` int64 ``[1]``, ``mla_block_table`` int32 ``[columns]`` (this rank's block of each of the request's
  pages), ``mla_write_slots`` int32 ``[T]`` (the pool row of each row this rank owns, -1 otherwise; the drafter rows
  take the same slots), ``state_slot`` int64 ``[1]`` (the request's state slot), ``state_track_slot`` int64 ``[1]``
  (the slot that also takes the state the chunk ends with: the tracking slot of a prefix-cache checkpoint, else the
  pool's sink; the runtime always passes it), ``sampling_positions`` int64 ``[1]``
  (the sampled row), ``rank`` int32 ``[1]``, ``draft_rope`` fp32 ``[T, 2 x rope]`` on the drafter writer stage, and
  the previous stage's boundary on a later stage;
* state: each MLA layer's FP8 rows (the pool's ``[pages + 1, page, 576]`` uint8 block view, written in place), each
  KDA layer's conv and temporal state over all slots (the request's slot read, then the request's and the tracking
  slot written in place), the drafter
  row pools on the writer stage (``bind_kv_cache``);
* output: on a stage before the last, the boundary written in place into the ``<name>_out`` seam slots it is handed,
  and those slots; on the last stage the logits ``[1, V / world]`` of the sampled row (``[1, V]`` when the LM head
  gathers its output).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from sglang_neuron.models.kimi_k3.prefill.geometry import KimiK3Geometry
from sglang_neuron.models.kimi_k3.prefill.stage_plan import StagePlan

HIDDEN = "pipeline_hidden_states"
BANK = "pipeline_attn_res_bank"
DRAFT_ACC = "pipeline_draft_acc"
# seam kinds of the boundary tensors (distributed.pp_seam.BoundaryKind)
KINDS = {HIDDEN: "ph", BANK: "pb", DRAFT_ACC: "pa"}


def layer_names(geometry: KimiK3Geometry) -> list:
    """The name of every layer of the model, in global order: its attention kind keys it."""
    return [f"layers.{i}.{'self_attn' if geometry.is_mla(i) else 'linear_attn'}" for i in range(geometry.num_layers)]


def mla_layer_name(layer: int) -> str:
    return f"layers.{layer}.self_attn"


class HeadOutput:
    """The LM head setting the model host flips: the device sampler reads each rank's vocabulary shard of the logits
    (``gather_output`` False), the host sampler the gathered logits."""

    gather_output = True


class RankComm:
    """The stage group's collectives with the rank taken from the graph's ``rank`` input."""

    def __init__(self, inner, rank_tensor: torch.Tensor):
        self.inner = inner
        self.rank, self.world = inner.rank, inner.world
        self.rank_tensor = rank_tensor
        self.replica_groups = getattr(inner, "replica_groups", None)

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return self.inner.all_gather(x, dim)

    def reduce_scatter(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return self.inner.reduce_scatter(x, dim)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.all_reduce(x)


@dataclass
class StageCaches:
    """The pool tensors a stage graph reads and writes: per MLA layer its ``[pages + 1, page, 576]`` uint8 block view,
    per KDA layer ``(conv [slots, 3 hc D, K - 1], temporal [slots, hc, D, D])``, and the drafter row pools
    ``[pages + 1, page, 576]`` uint8 (one per drafter layer, the writer stage only)."""

    mla: dict
    kda: dict
    draft: list = field(default_factory=list)
    sink_slot: int = 0


class KimiK3PrefillGraph(nn.Module):
    """The prefill graph of one stage rank (module docstring). ``load_weights`` builds the stage module from the
    checkpoint; ``bind_kv_cache`` hands it the pools."""

    phase = "prefill"

    def __init__(self, geometry: KimiK3Geometry, plan: StagePlan, *, rank: int, world: int, comm_factory,
                 drafter_dir: Optional[str] = None, drafter=None, backend: str = "torch"):
        super().__init__()
        if backend not in ("torch", "kernel"):
            raise ValueError(f"the Kimi-K3 graph backend {backend!r} is neither 'torch' nor 'kernel'")
        if (drafter_dir is None) != (drafter is None):
            raise ValueError("a drafter needs both its checkpoint directory and its geometry")
        self.geometry, self.plan = geometry, plan
        self.rank, self.world, self.backend = rank, world, backend
        self.comm_factory = comm_factory
        self.drafter_dir, self.drafter = drafter_dir, drafter
        self.layer_names = layer_names(geometry)
        self.sliding_windows = [None] * geometry.num_layers
        # each layer's recurrent-state family (models/contract.declared_layer_kinds): the KDA layers carry the "kda"
        # state the model's state spec covers (sglang_model.kda_state_spec), checked against it at runtime build
        self.state_families = [None if geometry.is_mla(i) else "kda" for i in range(geometry.num_layers)]
        self.lm_head = HeadOutput() if plan.is_last else None
        # the sampler's vocabulary size (models.host.SampledExecution)
        self.config = _VocabConfig(geometry.vocab_size)
        self.stage = None
        self.comm = None
        self.caches: Optional[StageCaches] = None

    @property
    def lm_head_tp_group(self):
        """The group the LM head's vocabulary is sharded over: the stage's attention TP group."""
        from sglang_neuron.parallel import groups

        return groups.get_neuron_attn_tp_group()

    @property
    def drafter_writer(self) -> bool:
        return self.drafter is not None and self.plan.draft_writer

    def get_kv_spec(self):
        from sglang_neuron.mem_cache.kv_cache import KVSpec, LayerSpec

        return KVSpec(layers=[LayerSpec(name=mla_layer_name(i)) for i in self.plan.mla_layers])

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        """Build the stage module of this rank from the checkpoint (prefill/weights.py) onto ``device``; the kernel
        backend's attention modules then build their trace shims, outside any traced forward."""
        from sglang_neuron.models.kimi_k3.prefill.stage import KimiK3PrefillStage
        from sglang_neuron.models.kimi_k3.prefill.weights import load_stage_rank, stage_modules

        g, plan = self.geometry, self.plan
        loaded = load_stage_rank(checkpoint_path, g, plan, self.rank, self.world, backend=self.backend,
                                 draft_dir=self.drafter_dir)
        modules = stage_modules(loaded, g, plan, self.rank, self.world, backend=self.backend, drafter=self.drafter)
        del loaded
        if plan.drafter and "draft" not in modules:
            from sglang_neuron.models.kimi_k3.prefill.draft_context import DraftContext

            # a stage without a consumer layer passes the drafter accumulator on unchanged
            modules["draft"] = DraftContext(self.drafter, self.world, (), False)
        attn_res = draft_row_writer = None
        if self.backend == "kernel":
            from sglang_neuron.models.kimi_k3.prefill import page_io
            from sglang_neuron.models.kimi_k3.prefill.attn_res import KernelAttnRes

            attn_res = KernelAttnRes()
            # the row-write kernel writes the drafter pools in place; a torch scatter on a view of a pool makes the
            # compiler copy the whole pool
            draft_row_writer = page_io.kernel_row_writer
        stage = KimiK3PrefillStage(g, plan, modules, attn_res=attn_res, draft_row_writer=draft_row_writer).to(device)
        if stage.final is not None:
            stage.final.gather_output = self.lm_head.gather_output
        if self.backend == "kernel":
            for p in plan.layer_plans:
                modules[f"layers.{p.index}.attn"].prepare_kernel()
            page_io.prepare_slot_kernels()
        self.stage = stage
        self.comm = self.comm_factory(device)

    def bind_kv_cache(self, caches: StageCaches) -> None:
        missing = [i for i in self.plan.mla_layers if i not in caches.mla]
        missing += [i for i in self.plan.kda_layers if i not in caches.kda]
        if missing:
            raise ValueError(f"the Kimi-K3 stage of layers {list(self.plan.layers)} got no pool for layers {missing}")
        if self.drafter_writer and len(caches.draft) != self.drafter.num_layers:
            raise ValueError(f"the drafter writer stage needs {self.drafter.num_layers} drafter row pools, got "
                             f"{len(caches.draft)}")
        self.caches = caches

    @torch.no_grad()
    def forward(self, input_ids, positions, valid_len, cached_len, mla_block_table, mla_write_slots, state_slot,
                sampling_positions, rank, draft_rope=None, state_track_slot=None, pipeline_hidden_states=None,
                pipeline_attn_res_bank=None,
                pipeline_draft_acc=None, pipeline_hidden_states_out=None, pipeline_attn_res_bank_out=None,
                pipeline_draft_acc_out=None):
        from sglang_neuron.models.kimi_k3.prefill.mla import MlaCache
        from sglang_neuron.models.kimi_k3.prefill.stage import SeamPayload, StageStep

        plan, caches = self.plan, self.caches
        if self.stage is None or caches is None:
            raise RuntimeError("the Kimi-K3 stage graph runs after load_weights and bind_kv_cache")
        comm = RankComm(self.comm, rank.reshape(1).to(torch.int64))
        attn = {}
        for layer in plan.mla_layers:
            attn[layer] = MlaCache(pool=caches.mla[layer], block_table=mla_block_table, write_slots=mla_write_slots,
                                   cached_len=cached_len)
        # the new state goes into the request's slot, then the same write into the tracking slot (the pool's sink when
        # the batch asks for no checkpoint; on a warm-up both are the sink). A caller without the input writes no
        # second slot, a program of its own
        state_io = _StateSlots(self, state_slot, (state_slot,) if state_track_slot is None
                               else (state_slot, state_track_slot))
        seam = None
        if not plan.is_first:
            seam = SeamPayload(hidden=pipeline_hidden_states, bank=pipeline_attn_res_bank, entries=plan.bank_in,
                               draft_acc=pipeline_draft_acc)
        writer = self.drafter_writer
        step = StageStep(positions=positions, valid_len=valid_len, input_ids=input_ids if plan.is_first else None,
                         seam=seam, attn=attn, sampled_rows=sampling_positions if plan.is_last else None,
                         draft_rope=draft_rope if writer else None, draft_slots=mla_write_slots if writer else None,
                         draft_pools=caches.draft if writer else None, state_io=state_io)
        out = self.stage(step, comm)
        if plan.is_last:
            return dumped_logits(self, out.logits)
        slots = {HIDDEN: pipeline_hidden_states_out, BANK: pipeline_attn_res_bank_out}
        values = {HIDDEN: out.seam.hidden, BANK: out.seam.bank}
        if self.drafter is not None:
            slots[DRAFT_ACC], values[DRAFT_ACC] = pipeline_draft_acc_out, out.seam.draft_acc
        for name, slot in slots.items():
            value = values[name]
            # copy_ would silently convert the dtype or broadcast the shape
            if slot is None or tuple(slot.shape) != tuple(value.shape) or slot.dtype != value.dtype:
                raise ValueError(f"the Kimi-K3 stage boundary's {name} is {tuple(value.shape)} {value.dtype}, its "
                                 f"seam slot {None if slot is None else (tuple(slot.shape), slot.dtype)}")
            slot.copy_(value)
        return slots

    def _read_slot(self, pool, slot):
        """The request's state ``pool[slot]``. The kernel backend reads it with one DMA at the runtime slot."""
        if self.backend == "kernel":
            from sglang_neuron.models.kimi_k3.prefill.page_io import read_slot_kernel

            return read_slot_kernel(pool, slot)
        return pool.index_select(0, slot)[0]

    def _write_slot(self, pool, slot, value):
        """``pool[slot] = value`` in place. The kernel backend writes it with one DMA at the runtime slot; a torch
        index_copy_ lowers to a scatter of one index per element."""
        if self.backend == "kernel":
            from sglang_neuron.models.kimi_k3.prefill.page_io import write_slot_kernel

            write_slot_kernel(pool, slot, value)
            return
        pool.index_copy_(0, slot, value.unsqueeze(0).to(pool.dtype))


class _StateSlots:
    """The KDA states of one request in the bound state pools, read and written inside the stage call
    (``StageStep.state_io``): ``read`` gives every KDA layer's state of the request's slot, ``write`` puts the new
    states into each of ``write_slots`` in turn."""

    def __init__(self, graph: KimiK3PrefillGraph, read_slot, write_slots):
        self.graph, self.read_slot, self.write_slots = graph, read_slot, write_slots

    def read(self) -> dict:
        from sglang_neuron.models.kimi_k3.prefill.kda import KdaState

        g, out = self.graph, {}
        for layer in g.plan.kda_layers:
            conv, temporal = g.caches.kda[layer]
            out[layer] = KdaState(conv=g._read_slot(conv, self.read_slot), temporal=g._read_slot(temporal,
                                                                                               self.read_slot))
        return out

    def write(self, states) -> None:
        g = self.graph
        for slot in self.write_slots:
            for layer in g.plan.kda_layers:
                conv, temporal = g.caches.kda[layer]
                g._write_slot(conv, slot, states[layer].conv)
                g._write_slot(temporal, slot, states[layer].temporal)


class _VocabConfig:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size


def dumped_logits(graph: nn.Module, logits: torch.Tensor) -> torch.Tensor:
    """The last stage's logits; under the logits dump (logits_dump.py) the graph also copies them into
    ``graph.logits_dump_buffer`` (set before the first trace: a graph of its own). At the end of the file, so the
    traced lines above keep their place."""
    buffer = getattr(graph, "logits_dump_buffer", None)
    if buffer is not None:
        buffer.copy_(logits)
    return logits
