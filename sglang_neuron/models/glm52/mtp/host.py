# SPDX-License-Identifier: Apache-2.0
"""The host side of the GK decode's async MTP K7 step under ``--speculative-algorithm NEURON_MTP``
(speculative/neuron_mtp.py; the step itself is mtp/graph.py).

Each step gets one host control row a slot (``mtp.step.CTRL_*``): a fresh slot's row names the request's position
and next input token, and every other slot's is zeros (verify the device carry). The decode serves one 128-position
block less than a slot holds, so SGLang's lagged reserve of 2 x 8 tokens a step always fits. Beside the control the
step takes the per-slot sampling table under on-device sampling, and two test or benchmark inputs: synthetic
acceptance (benchmark only) and the logit test's forced windows (logit_probe.py).
"""

from __future__ import annotations

import torch

# the kernel's chain: 7 top-1 drafts, so 8 tokens per request per step
NUM_STEPS = 7
NUM_DRAFT_TOKENS = NUM_STEPS + 1
PREFILL_SETTING = "SGLANG_NEURON_GLM52_PD_MTP=1"


def launch_problems(server_args) -> list:
    """A synthetic acceptance length that is not one, or one beside the logit test (which keeps the golden drafts)."""
    from sglang_neuron.models.glm52 import logit_probe
    from sglang_neuron.models.glm52.mtp.graph import SYNTHETIC_AL_ENV, synthetic_acceptance_length

    try:
        synthetic = synthetic_acceptance_length()
    except ValueError as error:
        return [str(error)]
    if logit_probe.test_dir() is not None and synthetic is not None:
        return [f"both {logit_probe.LOGIT_TEST_DIR_ENV} and {SYNTHETIC_AL_ENV} (the logit test keeps the golden "
                "drafts, the benchmark's rates others)"]
    return []


def launch_warnings(server_args) -> list:
    from sglang_neuron.models.glm52.mtp.graph import SYNTHETIC_AL_ENV, synthetic_acceptance_length

    length = synthetic_acceptance_length()
    if length is None:
        return []
    return [f"{SYNTHETIC_AL_ENV}={length:.2f} is a benchmark-only setting: every step keeps each slot's drafts at "
            "fixed rates whatever their tokens, so no request on this server gets the model's output; run accuracy "
            "checks on a server without it"]


class SyntheticKeep:
    """Host-side draws for synthetic acceptance (benchmark only).

    Each step draws one uniform ``[num_slots, spec]`` sample from a fixed-seed generator; a slot keeps its leading
    drafts whose draw is below the position's conditional rate. Every rank draws the same counts.
    """

    SEED = 0  # fixed, so the kept-draft counts repeat run to run

    def __init__(self, length: float, spec: int):
        from sglang_neuron.models.glm52.mtp.graph import synthetic_conditional_rates

        self.length, self.spec = float(length), int(spec)
        self.rates = torch.tensor(synthetic_conditional_rates(length, spec - 1), dtype=torch.float32)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.SEED)

    @classmethod
    def from_env(cls, spec: int = NUM_DRAFT_TOKENS):  # rows a step scores: the next token and 7 drafts
        from sglang_neuron.models.glm52.mtp.graph import synthetic_acceptance_length

        length = synthetic_acceptance_length(spec)
        return None if length is None else cls(length, spec)

    def step(self, num_slots: int) -> list:
        """This step's kept-draft count for every slot."""
        draw = torch.rand((num_slots, self.spec), generator=self.generator, dtype=torch.float32)[:, : self.spec - 1]
        below = (draw < self.rates.reshape(1, -1)).to(torch.int32)
        return torch.cumprod(below, dim=1).sum(dim=1).tolist()  # the kept prefix ends at the first rejection


def fresh_control(req, slot: int, ctrl: torch.Tensor) -> None:
    """Write a fresh slot's control row: its first step scores the request's next input token alone, like the decode
    without MTP.
    """
    from sglang_neuron.models.glm52.mtp import step as mtp_step

    ids = list(req.origin_input_ids) + list(req.output_ids)
    position = len(ids) - 1  # the next input token's
    ctrl[slot] = 0
    ctrl[slot, [mtp_step.CTRL_FRESH, mtp_step.CTRL_BASE, mtp_step.CTRL_TOKEN]] = torch.tensor(
        (1, position, int(ids[position])), dtype=torch.int32)


class StepHost:
    """One decode scheduler's side of the step: it builds the control and launches ``run_mtp_step``."""

    def __init__(self, target_worker):
        from sglang_neuron.models.glm52 import logit_probe

        self.target_worker = target_worker
        # per-slot sampling table; rewritten only when a request takes a slot, so it is re-uploaded only then
        self._sampling = None
        # synthetic acceptance (benchmark only); every rank advances the same seeded generator
        self._synth = SyntheticKeep.from_env()
        # logit test only
        self._forced = logit_probe.ForcedTokens(NUM_DRAFT_TOKENS) if logit_probe.test_dir() is not None else None

    def _runtime(self):
        model = self.target_worker.model_runner.model
        runtime = getattr(model, "runtime", None)
        if runtime is None or not hasattr(runtime, "run_mtp_step"):
            raise RuntimeError(f"the GLM-5.2 MTP step runs on the GK decode's MTP graph, and this model's runtime is "
                               f"{type(runtime).__name__}")
        return runtime

    @property
    def num_slots(self) -> int:
        return self._runtime().gk_num_slots

    def slot(self, req_pool_idx: int) -> int:
        from sglang_neuron.models.glm52.decode.pools import slot_of_request_row

        return slot_of_request_row(req_pool_idx, self.num_slots)

    def control(self, batch, slots: list, fresh: list) -> torch.Tensor:
        """The step's host control ``[num_slots, CTRL_COLUMNS]``: a fresh row for each request new to its slot, zeros
        (verify the device carry) for every other slot, with the synthetic kept-draft count under the benchmark.
        """
        from sglang_neuron.models.glm52.mtp import step as mtp_step

        num_slots = self.num_slots
        ctrl = torch.zeros(num_slots, mtp_step.CTRL_COLUMNS, dtype=torch.int32)
        kept = self._synth.step(num_slots) if self._synth is not None else None
        for req, slot, new in zip(batch.reqs, slots, fresh):
            if new:
                fresh_control(req, slot, ctrl)
                self._take_slot_params(req, slot, num_slots)
                if self._forced is not None:
                    self._start_forced(req, slot, ctrl)
            elif kept is not None:
                ctrl[slot, mtp_step.CTRL_SYNTH_KEPT] = kept[slot]
        return ctrl

    def step(self, batch, slots: list, fresh: list):
        ctrl = self.control(batch, slots, fresh)
        forced = self.forced_windows(batch, slots, ctrl) if self._forced is not None else None
        return self._runtime().run_mtp_step(batch.req_pool_indices, ctrl, len(batch.reqs), sampling=self._sampling,
                                            **({} if forced is None else {"forced": forced}))

    def _start_forced(self, req, slot: int, ctrl: torch.Tensor) -> None:
        """Logit test only: a forced request's first step starts from golden token 0."""
        from sglang_neuron.models.glm52 import logit_probe
        from sglang_neuron.models.glm52.mtp import step as mtp_step

        vocab = self.target_worker.model_runner.model_config.vocab_size
        tokens = logit_probe.forced_tokens(req.sampling_params, vocab)
        if tokens is None:
            self._forced.stop(slot)
            return
        if len(req.output_ids) != 1:
            raise RuntimeError(f"logit test request {req.rid} reached its first decode step with "
                               f"{len(req.output_ids)} output tokens, not the prefill's one")
        ctrl[slot, mtp_step.CTRL_TOKEN] = int(tokens[0])
        self._forced.start(slot, tokens)

    def forced_windows(self, batch, slots: list, ctrl: torch.Tensor):
        """Logit test only: each forced slot's golden window and the probe ``(rid, first_row, count, slot)``, or None.

        Raises:
            RuntimeError: Two forced requests in one step; a logit test sends its requests one at a time.
        """
        from sglang_neuron.models.glm52 import logit_probe
        from sglang_neuron.models.glm52.mtp import step as mtp_step

        windows = torch.full((ctrl.shape[0], 2 * NUM_DRAFT_TOKENS), logit_probe.UNFORCED, dtype=torch.int32)
        probe = seen = None
        for req, slot in zip(batch.reqs, slots):
            if not self._forced.forced(slot):
                continue
            if seen is not None:
                raise RuntimeError(f"logit test requests {seen} and {req.rid} ran in one step: send a logit test's "
                                   "requests one at a time")
            seen = req.rid
            first, rows, window = self._forced.step(slot, fresh=bool(ctrl[slot, mtp_step.CTRL_FRESH]))
            windows[slot] = torch.tensor(window, dtype=torch.int32)
            # a step launched past the request's last token has no golden row to write
            probe = (req.rid, first, rows, slot) if rows else None
        return windows, probe

    def _take_slot_params(self, req, slot: int, num_slots: int) -> None:
        """Write the request's row of the per-slot sampling table (sampling decodes only)."""
        from sglang_neuron import sampling
        from sglang_neuron.models.glm52 import tail_sampling

        if sampling.sampler_mode() != "topk_topp":
            return
        if self._sampling is None:
            self._sampling = (torch.zeros(num_slots, tail_sampling.SLOT_PARAM_COLUMNS, dtype=torch.float32),
                              torch.zeros(num_slots, dtype=torch.int32))
        params, seed = tail_sampling.slot_request_row(req.sampling_params, sampling.request_seed(req))
        # new tensors rather than in-place writes, so the runtime notices the change and re-uploads
        sparams, seeds = (t.clone() for t in self._sampling)
        sparams[slot] = torch.tensor(params, dtype=torch.float32)
        seeds[slot] = seed
        self._sampling = (sparams, seeds)
