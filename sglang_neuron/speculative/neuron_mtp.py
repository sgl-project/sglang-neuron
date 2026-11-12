# SPDX-License-Identifier: Apache-2.0
"""SGLang speculative decoding for a decode kernel that runs its model's MTP layer itself.

``--speculative-algorithm NEURON_MTP`` is registered through SGLang's plugin API. The worker has no draft model or
draft pool: one device step verifies the previous step's drafts, accepts, drafts the next ones and keeps them on the
device. The worker launches that step and hands SGLang its results:

- ``next_token_ids [bs * (num_steps + 1)]`` (each request's accepted run, padded with -1) and ``accept_lens [bs]``,
  read back on the output thread like any async step;
- ``new_seq_lens`` from the host's own lagged lengths, since the device tracks the true lengths;
- a placeholder ``next_draft_input`` that only keys the next step's FutureMap rows.

A request keeps its kernel slot for life, and the worker tells the step which slots are fresh: a request new to its
slot, whose first step starts from the host's token.

The model supplies the step through its catalog entry (``catalog.ModelSpec.mtp_host``), a module that defines:

- ``NUM_STEPS``: the top-1 drafts its kernel chains a step;
- ``PREFILL_SETTING``: the setting the PD prefill of an MTP decode launches with;
- ``launch_problems(server_args) -> list`` and ``launch_warnings(server_args) -> list``: the model's own reasons the
  launch is not served, and what a served launch should know;
- ``StepHost(target_worker)``, with ``slot(req_pool_idx) -> int`` (a request row's kernel slot) and
  ``step(batch, slots, fresh) -> logits_output``, which launches one step for the batch's requests (their slots, and
  whether each is fresh) without reading it back.

Served on a PD decode, without logprobs: greedy requests, and sampled ones under ``SGLANG_NEURON_SAMPLER=topk_topp``.
"""

from __future__ import annotations

import importlib
import logging

import torch

logger = logging.getLogger(__name__)

ALGORITHM = "NEURON_MTP"
TOPK = 1  # the kernels chain top-1 drafts


def is_neuron_mtp(algorithm) -> bool:
    """Whether SGLang's speculative algorithm (a name, enum value or registered algorithm) is this one."""
    name = getattr(algorithm, "name", algorithm)
    return isinstance(name, str) and name.upper() == ALGORITHM


def model_host(server_args):
    """The launch's model's MTP step host (catalog.ModelSpec.mtp_host).

    Raises:
        ValueError: The model has no in-kernel MTP step.
    """
    from sglang_neuron.models import catalog

    architecture, spec = catalog.model_spec(server_args)
    if not spec.mtp_host:
        raise ValueError(f"--speculative-algorithm {ALGORITHM}: {architecture}'s decode runs no MTP step")
    return importlib.import_module(spec.mtp_host)


def chain_shape(host) -> tuple:
    """(speculative_num_steps, speculative_eagle_topk, speculative_num_draft_tokens) of the model's kernel chain."""
    return host.NUM_STEPS, TOPK, host.NUM_STEPS + 1


def server_args_problems(server_args, host=None) -> list:
    """Launch arguments this algorithm does not serve, one message each."""
    host = host or model_host(server_args)
    problems = []
    names = ("speculative_num_steps", "speculative_eagle_topk", "speculative_num_draft_tokens")
    for name, value in zip(names, chain_shape(host)):
        given = getattr(server_args, name, None)
        if given is not None and int(given) != value:
            problems.append(f"--{name.replace('_', '-')} {given} (the kernel's draft chain is {value})")
    mode = str(getattr(server_args, "disaggregation_mode", "null"))
    if mode == "prefill":
        problems.append("a --disaggregation-mode prefill (launch the prefill of an MTP decode without a speculative "
                        f"algorithm, with {host.PREFILL_SETTING})")
    elif mode != "decode":
        problems.append("a server that is not a --disaggregation-mode decode (the step is the decode kernel's)")
    if getattr(server_args, "speculative_draft_model_path", None):
        problems.append("--speculative-draft-model-path (the draft is the checkpoint's own MTP layer)")
    if getattr(server_args, "disable_overlap_schedule", False):
        problems.append("--disable-overlap-schedule (the step is asynchronous: the host never waits for its tokens)")
    return problems + list(host.launch_problems(server_args))


def validate_server_args(server_args) -> None:
    host = model_host(server_args)
    problems = server_args_problems(server_args, host)
    if problems:
        raise ValueError(f"--speculative-algorithm {ALGORITHM} is not served with " + "; ".join(problems))
    for warning in host.launch_warnings(server_args):
        logger.warning("--speculative-algorithm %s: %s", ALGORITHM, warning)


def _spec_algo_class():
    from sglang.srt.speculative.spec_registry import CustomSpecAlgo

    class NeuronMtpSpecAlgo(CustomSpecAlgo):
        """The registered algorithm: overlap scheduling, no draft hidden states over PD, no top-k relay."""

        def carries_draft_hidden_states(self) -> bool:
            return False

        def need_topk(self) -> bool:
            return False

        def create_future_map(self, device, req_to_token_pool, needs_cpu_seq_lens: bool = True,
                              needs_confidence_relay: bool = False):
            from sglang.srt.managers.overlap_utils import FutureMap

            return FutureMap(device, self, req_to_token_pool, needs_cpu_seq_lens, needs_confidence_relay)

        def handle_server_args(self, server_args) -> None:
            # fill in the kernel's chain shape where the launch left it unset
            names = ("speculative_num_steps", "speculative_eagle_topk", "speculative_num_draft_tokens")
            for name, value in zip(names, chain_shape(model_host(server_args))):
                if getattr(server_args, name, None) is None:
                    setattr(server_args, name, value)

        def build_disagg_draft_input(self, batch, last_tokens_tensor, future_map):
            """Publish newly transferred PD requests to the FutureMap so they join the running batch; the device
            takes their first token from the host on their first (fresh) step.
            """
            spec_info = placeholder_draft_input()
            if batch.enable_overlap:
                spec_info.future_indices = batch.req_pool_indices
                future_map.publish(spec_info.future_indices, batch.seq_lens)
            return spec_info

    return NeuronMtpSpecAlgo


def placeholder_draft_input():
    """The ``next_draft_input`` of a step; it only keys the next step's FutureMap rows."""
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.speculative.eagle_info import EagleDraftInput

    return EagleDraftInput(capture_hidden_mode=CaptureHiddenMode.NULL)


def validate_request(obj, preferred, server_args) -> None:
    """Refuse a sampled request unless the decode samples on the device (``SGLANG_NEURON_SAMPLER=topk_topp``)."""
    if not is_neuron_mtp(getattr(server_args, "speculative_algorithm", None)):
        return
    from sglang_neuron import sampling

    if sampling.sampler_mode() != "topk_topp" and not sampling.request_is_greedy(obj, preferred):
        raise ValueError(f"--speculative-algorithm {ALGORITHM} verifies greedy requests only (temperature 0 or "
                         "top_k 1): send the request greedy, or serve it from a decode without MTP")


def register() -> None:
    """Register ``NEURON_MTP`` with SGLang (in every process)."""
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    try:
        SpeculativeAlgorithm.from_string(ALGORITHM)
        return  # this process registered it already
    except ValueError:
        pass
    SpeculativeAlgorithm.register(ALGORITHM, supports_overlap=True, validate_server_args=validate_server_args,
                                  spec_class=_spec_algo_class())(lambda server_args: worker_class())


_WORKER = {}


def worker_class():
    """The worker class, a BaseSpecWorker subclass built on first use."""
    if "cls" in _WORKER:
        return _WORKER["cls"]
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker as base

    class NeuronMtpWorker(base):
        def __init__(self, server_args, gpu_id, nccl_port, target_worker):
            base.__init__(self)
            self.server_args = server_args
            self.gpu_id = gpu_id
            self._target_worker = target_worker
            self._draft_worker = None
            host = model_host(server_args)
            self.speculative_num_draft_tokens = chain_shape(host)[2]
            self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()
            self.host = host.StepHost(target_worker)
            # kernel slot -> request generation whose state the device holds
            self._slot_owner = {}

        def seat(self, batch) -> tuple:
            """Each request's kernel slot, and whether it is new to that slot (a fresh step) or verifies the device's
            carry."""
            indices = batch.req_pool_indices.to("cpu").reshape(-1).tolist()
            generations = self.req_to_token_pool.req_generation[indices].tolist()
            slots, fresh = [], []
            for index, generation in zip(indices, generations):
                slot = self.host.slot(index)
                slots.append(slot)
                fresh.append(self._slot_owner.get(slot) != int(generation))
                self._slot_owner[slot] = int(generation)
            return slots, fresh

        def forward_batch_generation(self, batch, on_publish=None, grammar_barrier=None, pp_proxy_tensors=None,
                                     **kwargs):
            from sglang.srt.managers.utils import GenerationBatchResult

            mode = batch.forward_mode
            if mode.is_extend() or batch.is_extend_in_batch:
                raise RuntimeError(f"{ALGORITHM} serves a PD decode, which runs no extend batch ({mode})")
            if pp_proxy_tensors is not None:
                raise RuntimeError(f"{ALGORITHM} runs without pipeline stages")
            reqs = batch.reqs
            if mode.is_idle() or not reqs:
                # every rank launches every iteration; an idle step runs the target and emits nothing
                result = self.target_worker.forward_batch_generation(batch, is_verify=True)
                empty = torch.zeros(0, dtype=torch.int64)
                result.next_token_ids, result.accept_lens = empty, empty.to(torch.int32)
                result.speculative_num_draft_tokens = self.speculative_num_draft_tokens
                result.next_draft_input = placeholder_draft_input()
                result.new_seq_lens = empty
                if on_publish is not None:
                    on_publish(empty)
                return result
            logits_output = self.host.step(batch, *self.seat(batch))
            # the host's lengths lag the device's; publish them without waiting for this step
            new_seq_lens = torch.tensor([len(r.origin_input_ids) + len(r.output_ids) for r in reqs],
                                        dtype=torch.int64)
            if on_publish is not None:
                on_publish(new_seq_lens)
            return GenerationBatchResult(logits_output=logits_output, next_token_ids=None, accept_lens=None,
                                         speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                                         next_draft_input=placeholder_draft_input(), new_seq_lens=new_seq_lens,
                                         can_run_cuda_graph=True)

    _WORKER["cls"] = NeuronMtpWorker
    return NeuronMtpWorker
