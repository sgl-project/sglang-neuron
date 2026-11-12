# SPDX-License-Identifier: Apache-2.0
"""Wan2.2 A14B (T2V and I2V) on Neuron, independent of the serving framework.

``NeuronWanModel`` holds the UMT5 text encoder TP-sharded over the world, the two TP x CP DiT experts
(``transformer`` = high noise, ``transformer_2`` = low noise), the patch-parallel VAE, and the
UniPC scheduler. ``generate`` runs one prompt (and, for I2V, one first-frame image).

It carries the Wan2.2 pipeline pieces SGLang does not have:

- the expert switch on ``boundary_ratio``, each expert with its own guidance scale;
- the cross-attention K/V cache: the first step of each expert runs a cache-generating DiT graph
  that also returns every block's projected text K/V, later steps reuse them;
- CFG parallel: each CFG replica runs one branch, a compiled all-gather + combine fuses them;
- the I2V condition: the VAE ranks encode the first frame, rank 0 broadcasts the normalized
  condition to every rank over Gloo;
- the small per-step math (dtype casts, the UniPC step, the CFG combine, the latent unscale):
  each runs as a compiled graph on torch-neuronx.

The initial noise is drawn from a CPU generator with the reference pipeline's shape, dtype and
order (one float32 draw per video), so a seed gives the same noise as the goldens.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
import statistics
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from sglang_neuron.diffusion.models.wan2_2.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang_neuron.diffusion.stage_timing import StageMarks

logger = logging.getLogger(__name__)

# The pinned compiler's hierarchy partitions change Wan intermediate precision.
WAN_DISABLE_HIERARCHY_SPLITTER = "--internal-hlo2tensorizer-options=--hierarchy-splitter=false"

# The neuronx-cc flags of the text encoder and the DiT.
WAN_TEXT_ENCODER_COMPILER_ARGS = (
    "--model-type=transformer",
    "--auto-cast=none",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
    WAN_DISABLE_HIERARCHY_SPLITTER,
)
WAN_TRANSFORMER_COMPILER_ARGS = WAN_TEXT_ENCODER_COMPILER_ARGS

# CFG evaluates at most two encoder contexts per expert (the positive prompt and the negative
# prompt), so two live entries per expert is the whole working set. The cap is a backstop: if the
# context tensors are not reused across steps the cache would otherwise grow by one full K/V set
# per step (hundreds of MB each).
_MAX_CROSS_ATTENTION_KV_CACHE_ENTRIES_PER_EXPERT = 2

# The diffusers transformer config keys WanTransformer3DModel takes, plus the stage-config
# overrides it accepts.
_TRANSFORMER_CONFIG_KEYS = (
    "patch_size",
    "num_attention_heads",
    "attention_head_dim",
    "in_channels",
    "out_channels",
    "text_dim",
    "freq_dim",
    "ffn_dim",
    "num_layers",
    "cross_attn_norm",
    "eps",
    "image_dim",
    "added_kv_proj_dim",
    "rope_max_seq_len",
    "pos_embed_seq_len",
    "quantization",
    "modules_to_not_convert",
    "tp_sequence_parallel",
)

# Every BF16 Wan2.2 config sets this model_config for both experts.
DEFAULT_TRANSFORMER_OVERRIDES = {"tp_sequence_parallel": True}


def helper_compiler_args() -> list[str]:
    """Compiler flags of the small per-step graphs: ``NEURON_CC_FLAGS`` plus the splitter flag."""
    args = shlex.split(os.environ.get("NEURON_CC_FLAGS", ""))
    if WAN_DISABLE_HIERARCHY_SPLITTER not in args:
        args.append(WAN_DISABLE_HIERARCHY_SPLITTER)
    return args


def compile_helper(fn: Callable, model_name: str, compiler_args=None, **compile_kwargs) -> Callable:
    """``torch.compile`` one fixed-shape helper under its own NEFF ``model_name``."""
    options = {
        **(compile_kwargs.pop("options", None) or {}),
        "model_name": model_name,
        "compiler_args": list(helper_compiler_args() if compiler_args is None else compiler_args),
    }
    compile_kwargs.setdefault("fullgraph", True)
    compile_kwargs.setdefault("dynamic", False)
    return torch.compile(fn, options=options, **compile_kwargs)


# =========================================================================
# Scheduler
# =========================================================================


class CoefficientRows(NamedTuple):
    """One schedule's per-step coefficient inputs, each its own contiguous device tensor.

    The compiled step graphs take one row per step. Separate tensors (uploaded once per
    schedule) instead of views into the coefficient tables keep every graph input a plain
    fixed-shape tensor: no per-step storage offset, no Python scalar.
    """

    convert: list[torch.Tensor]  # [1] per step: sigma_s0
    predictor: list[torch.Tensor]  # [6] per step
    corrector: list[torch.Tensor]  # [7] per corrector step
    corrector_table: torch.Tensor  # the CPU table, for the order-1 resume rows


def coefficient_rows(
    predictor_coefficients: torch.Tensor, corrector_coefficients: torch.Tensor, device
) -> CoefficientRows:
    """Copy a schedule's coefficient tables to the host once and upload one tensor per row."""
    predictor = predictor_coefficients.detach().to("cpu")
    corrector = corrector_coefficients.detach().to("cpu")

    def own(row: torch.Tensor) -> torch.Tensor:  # a fresh host copy, then the upload
        return row.clone(memory_format=torch.contiguous_format).to(device)

    return CoefficientRows(
        convert=[own(predictor[i, 1:2]) for i in range(predictor.shape[0])],
        predictor=[own(predictor[i]) for i in range(predictor.shape[0])],
        corrector=[own(corrector[i]) for i in range(corrector.shape[0])],
        corrector_table=corrector,
    )


class NeuronFlowUniPCMultistepScheduler(FlowUniPCMultistepScheduler):
    """UniPC scheduler with a device-resident schedule and compiled step math.

    ``compile`` wraps the schedule and the convert / predictor / corrector math as compiled
    graphs. Before ``compile`` (CPU, tests) every method is the upstream one. The compiled
    schedule of each step count is computed once and reused (its outputs are read-only).
    """

    @staticmethod
    def _unipc_coefficients(sigmas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute predictor and corrector scalar inputs for every scheduler step."""
        lambdas = torch.log(1 - sigmas) - torch.log(sigmas)
        h = lambdas[1:] - lambdas[:-1]
        h_phi = torch.expm1(-h)
        ones = torch.ones_like(h)
        zeros = torch.zeros_like(h)

        predictor_rk = (lambdas[:-2] - lambdas[1:-1]) / h[1:]
        predictor_rk = torch.cat((ones[:1], predictor_rk[:-1], ones[:1]))[: h.shape[0]]
        predictor_rho = torch.cat((zeros[:1], torch.full_like(predictor_rk[1:-1], 0.5), zeros[:1]))[
            : h.shape[0]
        ]
        predictor_coefficients = torch.stack(
            (
                sigmas[1:],
                sigmas[:-1],
                1 - sigmas[1:],
                h_phi,
                predictor_rk,
                predictor_rho,
            ),
            dim=1,
        )

        if h.shape[0] == 1:
            corrector_coefficients = torch.zeros(
                (1, 7),
                dtype=sigmas.dtype,
                device=sigmas.device,
            )
        else:
            corrector_h = h[:-1]
            corrector_h_phi = h_phi[:-1]
            corrector_rk = (lambdas[:-3] - lambdas[1:-2]) / corrector_h[1:]
            corrector_rk = torch.cat((ones[:1], corrector_rk))[: corrector_h.shape[0]]
            hh = -corrector_h[1:]
            h_phi_k = corrector_h_phi[1:] / hh - 1
            b0 = h_phi_k / corrector_h_phi[1:]
            h_phi_k = h_phi_k / hh - 0.5
            b1 = h_phi_k * 2 / corrector_h_phi[1:]
            rho_history = (b1 - b0) / (corrector_rk[1:] - 1)
            rho_current = b0 - rho_history
            rho_history = torch.cat((zeros[:1], rho_history))[: corrector_h.shape[0]]
            rho_current = torch.cat((torch.full_like(zeros[:1], 0.5), rho_current))[
                : corrector_h.shape[0]
            ]
            corrector_coefficients = torch.stack(
                (
                    sigmas[1:-1],
                    sigmas[:-2],
                    1 - sigmas[1:-1],
                    corrector_h_phi,
                    corrector_rk,
                    rho_history,
                    rho_current,
                ),
                dim=1,
            )
        return predictor_coefficients, corrector_coefficients

    @staticmethod
    def _unipc_schedule(
        parameters: torch.Tensor, num_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_max, sigma_min, shift, exp_mu, num_train_timesteps, sigma_last, dynamic = (
            parameters.unbind()
        )
        index = torch.arange(num_steps, dtype=torch.float32, device=parameters.device)
        sigmas = sigma_max - (sigma_max - sigma_min) * index / num_steps
        static_sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        dynamic_sigmas = exp_mu / (exp_mu + sigmas.reciprocal() - 1)
        sigmas = torch.where(dynamic != 0, dynamic_sigmas, static_sigmas)
        timesteps = (sigmas * num_train_timesteps).to(torch.int64)
        sigmas = torch.cat((sigmas, sigma_last.unsqueeze(0)))
        predictor_coefficients, corrector_coefficients = (
            NeuronFlowUniPCMultistepScheduler._unipc_coefficients(sigmas)
        )
        return timesteps, sigmas, predictor_coefficients, corrector_coefficients

    @staticmethod
    def _unipc_custom_schedule(
        sigmas: torch.Tensor, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shift, exp_mu, num_train_timesteps, sigma_last, dynamic = parameters.unbind()
        static_sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        dynamic_sigmas = exp_mu / (exp_mu + sigmas.reciprocal() - 1)
        sigmas = torch.where(dynamic != 0, dynamic_sigmas, static_sigmas)
        timesteps = (sigmas * num_train_timesteps).to(torch.int64)
        sigmas = torch.cat((sigmas, sigma_last.unsqueeze(0)))
        predictor_coefficients, corrector_coefficients = (
            NeuronFlowUniPCMultistepScheduler._unipc_coefficients(sigmas)
        )
        return timesteps, sigmas, predictor_coefficients, corrector_coefficients

    @staticmethod
    def _unipc_convert(
        model_output: torch.Tensor,
        sample: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return (sample - coefficients[0] * model_output).to(sample.dtype)

    @staticmethod
    def _unipc_predictor(
        sample: torch.Tensor,
        model_output: torch.Tensor,
        previous_model_output: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        sigma_t, sigma_s0, alpha_t, h_phi, rk, rho = coefficients.unbind()
        rho = rho.to(sample.dtype)
        x_t = sigma_t / sigma_s0 * sample - alpha_t * h_phi * model_output
        history = (previous_model_output - model_output) / rk
        return (x_t - alpha_t * h_phi * (rho * history)).to(sample.dtype)

    @staticmethod
    def _unipc_corrector(
        last_sample: torch.Tensor,
        this_model_output: torch.Tensor,
        previous_model_output: torch.Tensor,
        older_model_output: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        sigma_t, sigma_s0, alpha_t, h_phi, rk, rho_history, rho_current = coefficients.unbind()
        rho_history = rho_history.to(last_sample.dtype)
        rho_current = rho_current.to(last_sample.dtype)
        x_t = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi * previous_model_output
        history = (older_model_output - previous_model_output) / rk
        current = this_model_output - previous_model_output
        return (x_t - alpha_t * h_phi * (rho_history * history + rho_current * current)).to(
            last_sample.dtype
        )

    def compile(self, *args, **kwargs):
        """Wrap the schedule and step math as compiled graphs (``wan_unipc_*`` NEFFs)."""
        if not (
            self.predict_x0
            and self.config.prediction_type == "flow_prediction"
            and not self.config.thresholding
            and self.config.solver_order == 2
            and self.config.solver_type == "bh2"
            and self.config.lower_order_final
            and self.solver_p is None
        ):
            raise NotImplementedError("compiled UniPC only supports the Wan scheduler")

        for attr, helper in (
            ("_compiled_schedule", self._unipc_schedule),
            ("_compiled_custom_schedule", self._unipc_custom_schedule),
            ("_compiled_convert_model_output", self._unipc_convert),
            ("_compiled_predictor", self._unipc_predictor),
            ("_compiled_corrector", self._unipc_corrector),
        ):
            setattr(
                self, attr, compile_helper(helper, f"wan{helper.__name__}", *args, **dict(kwargs))
            )
        self._schedule_cache = {}
        self._rows_cache = {}
        return self

    def _set_timesteps_via_cpu(
        self,
        num_inference_steps: int | None,
        device: str | torch.device | None,
        sigmas: list[float] | None,
        mu: float | None,
        shift: float | None,
    ) -> None:
        """Build the upstream schedule on CPU, then move the timesteps to ``device``.

        Upstream writes them as ``torch.from_numpy(...).to(device=device, dtype=torch.int64)``;
        cast on CPU first and transfer afterwards (same tensor, two steps).
        """
        super().set_timesteps(num_inference_steps, device=None, sigmas=sigmas, mu=mu, shift=shift)
        self._coefficient_rows = None
        if device is not None and torch.device(device).type != "cpu":
            self.timesteps = self.timesteps.to(device)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        shift: float | None = None,
    ) -> None:
        target_device = torch.device(device) if device is not None else None
        if (
            target_device is None
            or target_device.type == "cpu"
            or getattr(self, "_compiled_schedule", None) is None
            or getattr(self, "_compiled_custom_schedule", None) is None
        ):
            return self._set_timesteps_via_cpu(num_inference_steps, device, sigmas, mu, shift)
        return self._set_timesteps_compiled(num_inference_steps, target_device, sigmas, mu, shift)

    def _set_timesteps_compiled(
        self,
        num_inference_steps: int | None,
        target_device: torch.device,
        sigmas: list[float] | None,
        mu: float | None,
        shift: float | None,
    ) -> None:
        """The device schedule: the compiled ``_unipc_schedule`` and per-step coefficients."""
        compiled_schedule = self._compiled_schedule
        compiled_custom_schedule = self._compiled_custom_schedule
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("Must pass a value for `mu` when `use_dynamic_shifting` is True")
        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = self.sigma_min
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0.0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be 'zero' or 'sigma_min', "
                f"got {self.config.final_sigmas_type}"
            )

        dynamic = self.config.use_dynamic_shifting
        shift_value = self.config.shift if shift is None else shift
        exp_mu = math.exp(mu) if dynamic else 1.0
        if sigmas is None:
            if num_inference_steps is None:
                raise ValueError("num_inference_steps is required when sigmas are not provided")
            values = [
                self.sigma_max,
                self.sigma_min,
                shift_value,
                exp_mu,
                self.config.num_train_timesteps,
                sigma_last,
                float(dynamic),
            ]
            key = ("schedule", num_inference_steps, str(target_device), *map(float, values))
            outputs = self._schedule_cache.get(key)
            if outputs is None:
                parameters = torch.tensor(values, dtype=torch.float32, device=target_device)
                outputs = compiled_schedule(parameters, num_inference_steps)
                self._schedule_cache[key] = outputs
            rows = self._rows_cache.get(key)
            if rows is None:
                rows = coefficient_rows(outputs[2], outputs[3], target_device)
                self._rows_cache[key] = rows
        else:
            values = [
                shift_value,
                exp_mu,
                self.config.num_train_timesteps,
                sigma_last,
                float(dynamic),
            ]
            sigma_tensor = torch.tensor(sigmas, dtype=torch.float32, device=target_device)
            parameters = torch.tensor(values, dtype=torch.float32, device=target_device)
            outputs = compiled_custom_schedule(sigma_tensor, parameters)
            rows = coefficient_rows(outputs[2], outputs[3], target_device)

        (
            self.timesteps,
            self.sigmas,
            self._predictor_coefficients,
            self._corrector_coefficients,
        ) = outputs
        self._coefficient_rows = rows
        self.num_inference_steps = self.timesteps.shape[0]
        self.reset_solver_state()
        self._begin_index = None

    def reset_solver_state(self) -> None:
        """Start the next solve on the current schedule: no solver history, back at step 0.

        ``set_timesteps`` resets the same state. A request with several videos sets the timesteps
        once and solves each video on that schedule, so every solve but the first needs this.
        The begin index stays.
        """
        self.model_outputs = [None] * self.config.solver_order
        self.timestep_list = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = self._step_index = None

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if (
            getattr(self, "_compiled_convert_model_output", None) is None
            or getattr(self, "_predictor_coefficients", None) is None
            or sample is None
            or model_output.dtype != sample.dtype
            or args
            or kwargs
        ):
            return super().convert_model_output(model_output, *args, sample=sample, **kwargs)
        assert self.step_index is not None
        rows = getattr(self, "_coefficient_rows", None)
        if rows is not None:
            coefficients = rows.convert[self.step_index]
        else:
            coefficients = self._predictor_coefficients[self.step_index, 1:2]
        return self._compiled_convert_model_output(model_output, sample, coefficients)

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor | None = None,
        order: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        model_outputs = getattr(self, "model_outputs", None)
        current_model_output = model_outputs[-1] if model_outputs else None
        if (
            getattr(self, "_compiled_predictor", None) is None
            or getattr(self, "_predictor_coefficients", None) is None
            or sample is None
            or order not in (1, 2)
            or current_model_output is None
            or current_model_output.dtype != sample.dtype
            or args
            or kwargs
        ):
            return super().multistep_uni_p_bh_update(
                model_output, *args, sample=sample, order=order, **kwargs
            )
        assert self.step_index is not None
        previous_model_output = (
            self.model_outputs[-2]
            if order == 2
            else current_model_output.view_as(current_model_output)
        )
        assert previous_model_output is not None
        rows = getattr(self, "_coefficient_rows", None)
        if rows is not None:
            coefficients = rows.predictor[self.step_index]
        else:
            coefficients = self._predictor_coefficients[self.step_index]
        return self._compiled_predictor(
            sample, current_model_output, previous_model_output, coefficients
        )

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        *args,
        last_sample: torch.Tensor | None = None,
        this_sample: torch.Tensor | None = None,
        order: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        model_outputs = getattr(self, "model_outputs", None)
        previous_model_output = model_outputs[-1] if model_outputs else None
        if (
            getattr(self, "_compiled_corrector", None) is not None
            and getattr(self, "_corrector_coefficients", None) is not None
            and last_sample is not None
            and this_sample is not None
            and order in (1, 2)
            and this_model_output.dtype == last_sample.dtype
            and previous_model_output is not None
            and previous_model_output.dtype == last_sample.dtype
            and not args
            and not kwargs
        ):
            return self._run_compiled_corrector(
                this_model_output=this_model_output, last_sample=last_sample, order=order
            )
        return super().multistep_uni_c_bh_update(
            this_model_output,
            *args,
            last_sample=last_sample,
            this_sample=this_sample,
            order=order,
            **kwargs,
        )

    def _run_compiled_corrector(
        self,
        *,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        assert self.step_index is not None
        previous_model_output = self.model_outputs[-1]
        older_model_output = (
            self.model_outputs[-2]
            if order == 2
            else previous_model_output.view_as(previous_model_output)
        )
        assert previous_model_output is not None and older_model_output is not None
        rows = getattr(self, "_coefficient_rows", None)
        if rows is None:
            coefficients = self._corrector_coefficients[self.step_index - 1]
            if order == 1 and self.step_index != 1:
                coefficients = torch.cat((coefficients[:4], self._corrector_coefficients[0, 4:]))
        elif order == 1 and self.step_index != 1:
            # A resumed first-order corrector (never in a fresh 40-step run): first-order rhos.
            table = rows.corrector_table
            row = torch.cat((table[self.step_index - 1, :4], table[0, 4:]))
            coefficients = row.to(rows.corrector[0].device)
        else:
            coefficients = rows.corrector[self.step_index - 1]
        return self._compiled_corrector(
            last_sample,
            this_model_output,
            previous_model_output,
            older_model_output,
            coefficients,
        )


# =========================================================================
# Pure helpers (CPU-testable)
# =========================================================================


def prompt_clean(text: str) -> str:
    """``text`` stripped, with every whitespace run collapsed to one space."""
    return " ".join(text.strip().split())


def effective_dimensions(
    height: int, width: int, num_frames: int, *, spatial: int, temporal: int, patch: int, i2v: bool
) -> tuple[int, int, int]:
    """Rounded ``(height, width, num_frames)`` as the Wan2.2 pipelines use them.

    T2V rounds height and width down to multiples of ``spatial * patch``
    (``conditioning.effective_wan_dimensions``); I2V keeps them. Both round the frame count to
    ``temporal * k + 1``.
    """
    if not i2v:
        modulus = spatial * patch
        height = (height // modulus) * modulus
        width = (width // modulus) * modulus
    if num_frames % temporal != 1:
        num_frames = num_frames // temporal * temporal + 1
    return height, width, max(num_frames, 1)


def latent_shape(
    height: int, width: int, num_frames: int, channels: int, *, spatial: int, temporal: int
) -> tuple[int, int, int, int, int]:
    """The ``(1, C, T_lat, H/8, W/8)`` noise shape of one video."""
    return (1, channels, (num_frames - 1) // temporal + 1, height // spatial, width // spatial)


def draw_noise(
    shapes_and_generators: list[tuple[tuple[int, ...], torch.Generator]],
) -> list[torch.Tensor]:
    """One float32 CPU draw per video, in order.

    One ``(1, *shape[1:])`` draw per generator in the request's generator list. The reference
    pipeline passes the request's CPU generator to ``torch.randn(device=neuron)``; a CPU generator
    can only feed the CPU stream, so its values are this CPU draw.
    """
    return [
        torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)
        for shape, generator in shapes_and_generators
    ]


def boundary_timestep(boundary_ratio: float | None, num_train_timesteps: int) -> float | None:
    return None if boundary_ratio is None else boundary_ratio * num_train_timesteps


def select_expert(
    t: float,
    boundary: float | None,
    *,
    has_transformer: bool,
    has_transformer_2: bool,
    i2v: bool,
) -> tuple[int, bool]:
    """``(expert, use_guidance_2)`` for timestep ``t``: expert 1 = ``transformer`` (high noise),
    2 = ``transformer_2`` (low noise).

    T2V: below the boundary the low-noise stage uses ``guidance_scale_2`` and ``transformer_2``
    (``transformer`` if only it is loaded); otherwise ``guidance_scale`` and ``transformer``. I2V
    switches expert and guidance only when ``transformer_2`` is loaded.
    """
    low_noise = boundary is not None and t < boundary
    if i2v:
        if low_noise and has_transformer_2:
            return 2, True
        return 1, False
    if low_noise:
        if has_transformer_2:
            return 2, True
        if has_transformer:
            return 1, True
        raise RuntimeError("No transformer available for low-noise stage")
    if has_transformer:
        return 1, False
    if has_transformer_2:
        return 2, False
    raise RuntimeError("No transformer available for high-noise stage")


def resolve_guidance(guidance_scale: float | None, guidance_scale_2: float | None, default: float):
    """``(guidance for transformer, guidance for transformer_2)``; ``guidance_scale_2`` falls
    back to ``guidance_scale``."""
    low = default if guidance_scale is None else float(guidance_scale)
    high = low if guidance_scale_2 is None else float(guidance_scale_2)
    return low, high


def i2v_first_frame_mask(
    latent_condition: torch.Tensor, num_latent_frames: int, temporal: int
) -> torch.Tensor:
    """The I2V condition ``[B, temporal + C, T_lat, h, w]``: mask then VAE latents.

    No ``last_image``: the mask is 1 for the ``temporal`` sub-frames of the first latent frame
    and 0 for the rest.
    """
    batch, _, _, h, w = latent_condition.shape
    dtype, device = latent_condition.dtype, latent_condition.device
    first = torch.ones(batch, temporal, 1, h, w, dtype=dtype, device=device)
    rest = torch.zeros(batch, temporal, num_latent_frames - 1, h, w, dtype=dtype, device=device)
    mask = torch.cat([first, rest], dim=2)
    return torch.cat([mask, latent_condition], dim=1)


def preprocess_i2v_image(image, height: int, width: int, spatial: int) -> torch.Tensor:
    """First-frame image -> ``[1, 3, H, W]`` float32 in [-1, 1] on the CPU.

    A PIL image is resized to ``(width, height)`` with LANCZOS, then goes through
    ``TF.to_tensor`` and ``VideoProcessor.preprocess``.
    """
    import PIL.Image
    from diffusers.video_processor import VideoProcessor

    if isinstance(image, str):
        image = PIL.Image.open(image).convert("RGB")
    if isinstance(image, PIL.Image.Image):
        image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
        # torchvision TF.to_tensor for an 8-bit RGB image
        array = np.array(image, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array).view(image.size[1], image.size[0], 3)
        tensor = tensor.permute((2, 0, 1)).contiguous().to(dtype=torch.float32).div(255)
        tensor = VideoProcessor(vae_scale_factor=spatial).preprocess(
            tensor, height=height, width=width
        )
    else:
        tensor = image.unsqueeze(0) if image.ndim == 3 else image
    return tensor.to(dtype=torch.float32)


def retrieve_latents(encoder_output, sample_mode: str = "argmax"):
    """The encoder output's latents (``argmax`` = the posterior mode)."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def load_model_index(model_path: str) -> dict:
    path = os.path.join(model_path, "model_index.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def has_component(model_index: dict, name: str) -> bool:
    """Whether ``model_index.json`` lists ``name`` with a real class (``[lib, cls]``)."""
    entry = model_index.get(name)
    return isinstance(entry, (list, tuple)) and len(entry) == 2 and entry[1] is not None


def load_component_config(model_path: str, subfolder: str) -> dict:
    path = os.path.join(model_path, subfolder, "config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def transformer_kwargs(config: dict) -> dict:
    """The ``_TRANSFORMER_CONFIG_KEYS`` entries of ``config``; ``patch_size`` as a tuple."""
    kwargs = {}
    for key in _TRANSFORMER_CONFIG_KEYS:
        if key in config:
            value = config[key]
            kwargs[key] = tuple(value) if key == "patch_size" else value
    return kwargs


# =========================================================================
# Model
# =========================================================================


class _CrossAttentionKVCacheEntry(NamedTuple):
    """One expert's projected cross-attention K/V for one encoder context.

    ``context`` is kept both to verify the cache hit by tensor identity and to hold a reference to
    the keyed tensors, so their ``id()`` cannot be recycled by a different tensor while this entry
    is live.
    """

    context: tuple[torch.Tensor, ...]
    kv_cache: tuple[torch.Tensor, ...]


class WanSchedule(NamedTuple):
    """The host copy of one step count's timesteps and their per-step device inputs."""

    timesteps: list[int]
    device_timesteps: list[torch.Tensor]  # [1] int64 per step


class NeuronWanModel(nn.Module):
    """Wan2.2 A14B on Neuron: world-sharded UMT5, two TP/CP DiT experts, patch-parallel VAE."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        vae_patch_parallel_size: int = 16,
        boundary_ratio: float | None = None,
        model_config: dict | None = None,
    ):
        """Build every component on the CPU (DiT weights load in ``load_weights``).

        Call after the TP / CP / CFG groups exist (``runtime.initialize_model_groups``), on every
        rank, under the model dtype as torch's default dtype.

        Args:
            model_path: Local diffusers Wan2.2 A14B checkpoint directory.
            dtype: Component dtype.
            vae_patch_parallel_size: VAE group size (ranks ``0..size-1``).
            boundary_ratio: Expert boundary; ``None`` uses ``model_index.json``'s. 1.0 drops the
                high-noise expert, 0.0 the low-noise one.
            model_config: Transformer config overrides (default ``tp_sequence_parallel=True``).
        """
        nn.Module.__init__(self)
        from transformers import AutoTokenizer

        from sglang_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
            build_wan_vae,
        )
        from sglang_neuron.diffusion.distributed.groups import get_cfg_group, get_world_group
        from sglang_neuron.diffusion.models.umt5_encoder.umt5_encoder import (
            NeuronTextEncoderWrapper,
        )
        from sglang_neuron.diffusion.models.wan2_2.wan2_2_transformer import (
            WanTransformer3DModel,
        )
        from sglang_neuron.diffusion.runtime import neuron_device

        self.model_path = model_path
        self.device = neuron_device()
        self.dtype = dtype
        model_index = load_model_index(model_path)
        if model_index.get("expand_timesteps", False):
            raise NotImplementedError("Wan2.2 TI2V (expand_timesteps) is not supported on Neuron")
        if model_index.get("_class_name") == "WanDMDPipeline":
            raise NotImplementedError("DMD Wan checkpoints are not supported on Neuron")
        self.is_i2v = model_index.get("_class_name") == "WanImageToVideoPipeline"
        if has_component(model_index, "image_encoder"):
            raise NotImplementedError("Wan I2V checkpoints with a CLIP image encoder (Wan2.1)")
        self.boundary_ratio = (
            model_index.get("boundary_ratio") if boundary_ratio is None else boundary_ratio
        )
        has_transformer_2 = has_component(model_index, "transformer_2") and os.path.isdir(
            os.path.join(model_path, "transformer_2")
        )
        load_transformer = self.boundary_ratio != 1.0 if self.boundary_ratio is not None else True
        load_transformer_2 = has_transformer_2 and (
            self.boundary_ratio != 0.0 if self.boundary_ratio is not None else True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, subfolder="tokenizer", local_files_only=True
        )
        self.text_encoder = NeuronTextEncoderWrapper(
            model_path=model_path, dtype=dtype, tp_group=get_world_group().device_group
        )

        vae_config = load_component_config(model_path, "vae")
        self.vae_scale_factor_temporal = vae_config.get("scale_factor_temporal", 4)
        self.vae_scale_factor_spatial = vae_config.get("scale_factor_spatial", 8)
        self.vae, self._vae_group = build_wan_vae(
            model_path, dtype, vae_patch_parallel_size, use_tiling=True, local_files_only=True
        )
        self.is_vae_rank = self.vae is not None

        overrides = dict(DEFAULT_TRANSFORMER_OVERRIDES if model_config is None else model_config)
        self.transformer = None
        self.transformer_2 = None
        if load_transformer:
            config = {**load_component_config(model_path, "transformer"), **overrides}
            self.transformer = WanTransformer3DModel(**transformer_kwargs(config))
        if load_transformer_2:
            config = {**load_component_config(model_path, "transformer_2"), **overrides}
            self.transformer_2 = WanTransformer3DModel(**transformer_kwargs(config))
        active = self.transformer if self.transformer is not None else self.transformer_2
        if active is None:
            raise RuntimeError("No transformer loaded")
        self.transformer_config = active.config

        cfg_group = get_cfg_group()
        self.cfg_size = cfg_group.world_size
        self.cfg_rank = cfg_group.rank_in_group
        self.is_output_rank = dist.get_rank() == 0

        self._schedulers: dict[float, NeuronFlowUniPCMultistepScheduler] = {}
        self._schedules: dict[tuple, WanSchedule] = {}
        self._compile_kwargs: dict | None = None
        self._helpers: dict[str, Callable] = {}
        self._tensor_cache: dict[tuple, torch.Tensor] = {}
        self._cross_attention_kv_cache_expert_id: int | None = None
        self._cross_attention_kv_cache: dict[tuple, _CrossAttentionKVCacheEntry] = {}
        # id(expert) -> its compiled first-step full DiT graph, returning
        # ``(noise_prediction, *cross_attention_kv_cache)``.
        self._compiled_cross_attention_kv_generators: dict[int, Callable[..., tuple]] = {}
        self._warned_unstable_cross_attention_context = False

    # ---- set-up ---------------------------------------------------------------------------

    def experts(self) -> tuple[nn.Module, ...]:
        return tuple(t for t in (self.transformer, self.transformer_2) if t is not None)

    def to(self, *args, **kwargs):
        """Move the text encoder, the transformer(s) and (on VAE ranks) the VAE."""
        for module in (self.text_encoder, self.vae, self.transformer, self.transformer_2):
            if module is not None:
                module.to(*args, **kwargs)
        return self

    def load_weights(self):
        """Load the UMT5 and DiT weights from the model directory (the VAE loads in __init__)."""
        self.text_encoder.load_weights(os.path.join(self.model_path, "text_encoder"))
        for name in ("transformer", "transformer_2"):
            transformer = getattr(self, name)
            if transformer is not None:
                transformer.load_weights(os.path.join(self.model_path, name))

    def compile(self, *args, **kwargs):
        """Wrap every device component and per-step helper with lazy ``torch.compile``.

        ``kwargs`` is ``envs.compile_kwargs()`` (``backend``, ``options``). Each graph compiles to a
        NEFF on its first call.
        """
        self._compile_kwargs = dict(kwargs)

        def site(model_name: str, compiler_args) -> dict:
            site_kwargs = dict(kwargs)
            options = dict(site_kwargs.pop("options", None) or {})
            options.update(model_name=model_name, compiler_args=list(compiler_args))
            return {"options": options, "fullgraph": True, "dynamic": False, **site_kwargs}

        self.text_encoder.compile(*args, **site("wan_text_encoder", WAN_TEXT_ENCODER_COMPILER_ARGS))
        if self.is_vae_rank:
            self.vae.compile(*args, compile_encoder=self.is_i2v, **kwargs)
        for transformer in self.experts():
            # Two full DiT graphs at O1: the steady-state graph consumes the cross-attention K/V;
            # the first-step graph generates them inside the same invocation that first consumes
            # them and returns them with the noise prediction.
            transformer.compile(*args, **site("wan_transformer", WAN_TRANSFORMER_COMPILER_ARGS))
            self._compiled_cross_attention_kv_generators[id(transformer)] = torch.compile(
                transformer.forward_and_cache_cross_attention_kv,
                *args,
                **site("wan_transformer_cache_gen", WAN_TRANSFORMER_COMPILER_ARGS),
            )
        self.compile_helpers(*args, **kwargs)
        return self

    def compile_helpers(self, *args, **kwargs):
        """Compile the small per-step graphs and the schedulers' step math (``compile`` calls it).

        Every graph is static (``dynamic=False``): its tensor inputs keep one shape per request
        size, and every per-step or per-request number (guidance scale, timestep, UniPC
        coefficients) reaches it as a tensor, so a new value never recompiles a graph.
        """
        self._compile_kwargs = dict(kwargs)
        for name, fn in (
            ("cast", self._cast),
            ("cfg_combine", self._cfg_combine),
            ("i2v_model_input", self._i2v_model_input),
            ("vae_input", self._vae_input),
        ):
            self._helpers[name] = compile_helper(fn, f"wan_{name}", *args, **dict(kwargs))
        if self.cfg_size > 1:
            from sglang_neuron.diffusion.distributed.cfg_parallel import (
                _cfg_gather_combine,
            )

            # One fullgraph NEFF with the environment's compiler flags, static, the guidance scale
            # a [1] tensor: the collective stays on the CC fabric, and a new scale (the experts'
            # guidance_scale and guidance_scale_2 differ) reuses the graph.
            gather_kwargs = dict(kwargs)
            options = {
                **(gather_kwargs.pop("options", None) or {}),
                "model_name": "cfg_gather_combine",
            }
            self._helpers["cfg_gather_combine"] = torch.compile(
                _cfg_gather_combine,
                *args,
                options=options,
                fullgraph=True,
                dynamic=False,
                **gather_kwargs,
            )
        for scheduler in self._schedulers.values():
            scheduler.compile(**dict(kwargs))
        return self

    # ---- small per-step graphs ------------------------------------------------------------

    @staticmethod
    def _cast(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return tensor.to(dtype=dtype)

    @staticmethod
    def _cfg_combine(positive: torch.Tensor, negative: torch.Tensor, scale: torch.Tensor):
        return negative + scale[0] * (positive - negative)

    @staticmethod
    def _i2v_model_input(latents: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return torch.cat([latents, condition], dim=1)

    @staticmethod
    def _vae_input(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, dtype
    ) -> torch.Tensor:
        """Cast to the VAE dtype, then undo the latent normalization."""
        latents = latents.to(dtype=dtype)
        return latents / latents_std + latents_mean

    def _helper(self, name: str) -> Callable:
        return self._helpers.get(name) or getattr(self, f"_{name}")

    def cast(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Change a tensor's dtype in a compiled graph, which avoids an eager narrowing copy of a
        Neuron tensor."""
        if tensor.dtype == dtype:
            return tensor
        return self._helper("cast")(tensor, dtype)

    def _device_constant(self, key: tuple, make: Callable[[], torch.Tensor]) -> torch.Tensor:
        tensor = self._tensor_cache.get(key)
        if tensor is None:
            tensor = make().to(self.device)
            self._tensor_cache[key] = tensor
        return tensor

    # ---- scheduler ------------------------------------------------------------------------

    def scheduler_for(self, flow_shift: float) -> NeuronFlowUniPCMultistepScheduler:
        """The UniPC scheduler of one flow shift, built with ``shift=flow_shift``."""
        scheduler = self._schedulers.get(flow_shift)
        if scheduler is None:
            scheduler = NeuronFlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=flow_shift,
                prediction_type="flow_prediction",
                solver_order=2,
            )
            if self._compile_kwargs is not None:
                scheduler.compile(**dict(self._compile_kwargs))
            self._schedulers[flow_shift] = scheduler
        return scheduler

    def prepare_timesteps(self, num_steps: int, flow_shift: float):
        """Set the scheduler's timesteps for this request; returns ``(scheduler, WanSchedule)``.

        The schedule runs on the device (the compiled schedule); its timesteps are copied to
        the host once per step count, for the expert switch, and uploaded as one ``[1]`` tensor
        per step for the DiT.
        """
        scheduler = self.scheduler_for(flow_shift)
        scheduler.set_timesteps(num_steps, device=self.device, shift=flow_shift)
        # Upstream finds step 0 by matching the first timestep (a device sync); it is unique, so
        # its index is 0.
        scheduler.set_begin_index(0)
        key = (num_steps, flow_shift)
        schedule = self._schedules.get(key)
        if schedule is None:
            host = [int(t) for t in scheduler.timesteps.cpu().tolist()]
            device_timesteps = [torch.tensor([t], dtype=torch.int64).to(self.device) for t in host]
            schedule = WanSchedule(host, device_timesteps)
            self._schedules[key] = schedule
        return scheduler, schedule

    # ---- text -----------------------------------------------------------------------------

    def _encode_prompt(self, prompts: list[str], max_sequence_length: int) -> torch.Tensor:
        """Tokenize and encode on Neuron; the graph zeros the padded positions itself."""
        text_inputs = self.tokenizer(
            [prompt_clean(p) for p in prompts],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        embeds = self.text_encoder(ids.to(self.device), mask.to(self.device)).last_hidden_state
        return self.cast(embeds, self.transformer_config_dtype)

    @property
    def transformer_config_dtype(self) -> torch.dtype:
        active = self.transformer if self.transformer is not None else self.transformer_2
        return active.dtype

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str | None,
        *,
        do_classifier_free_guidance: bool,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(prompt_embeds, negative_prompt_embeds)``, ``[1, 512, 4096]`` each, on the device.

        The negative embeddings exist only with CFG; an absent negative prompt encodes ``""``.
        """
        prompt_embeds = self._encode_prompt([prompt], max_sequence_length)
        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt_embeds = self._encode_prompt(
                [negative_prompt or ""], max_sequence_length
            )
        return prompt_embeds, negative_prompt_embeds

    # ---- latents and the I2V condition ----------------------------------------------------

    def latent_channels(self) -> int:
        """Noise channels: T2V ``in_channels``; I2V ``out_channels`` (the rest is the condition)."""
        config = self.transformer_config
        return config.out_channels if self.is_i2v else config.in_channels

    def prepare_latents(
        self, height: int, width: int, num_frames: int, generators: list[torch.Generator]
    ) -> list[torch.Tensor]:
        """Draw each video's noise on the CPU and upload it.

        T2V keeps it float32 (the scheduler state stays float32); I2V casts it to the
        transformer dtype.
        """
        shape = latent_shape(
            height,
            width,
            num_frames,
            self.latent_channels(),
            spatial=self.vae_scale_factor_spatial,
            temporal=self.vae_scale_factor_temporal,
        )
        noise = draw_noise([(shape, generator) for generator in generators])
        if self.is_i2v:
            noise = [n.to(self.transformer_config_dtype) for n in noise]
        return [n.contiguous().to(self.device) for n in noise]

    def prepare_i2v_condition(
        self, image_tensor: torch.Tensor | None, num_frames: int, height: int, width: int
    ) -> torch.Tensor:
        """VAE-encode the first frame on the VAE ranks, normalize it, broadcast it from rank 0.

        ``image_tensor``: ``[1, 3, H, W]`` float32 in [-1, 1] (``preprocess_i2v_image``); needed
        on the VAE ranks only. Returns the ``[1, 20, T_lat, h, w]`` condition on the device in the
        transformer dtype, on every rank. The broadcast stages through the CPU over the world's
        Gloo group (torch-neuronx does not service a device broadcast outside a compiled graph).
        """
        from sglang_neuron.diffusion.distributed.groups import get_world_group

        temporal = self.vae_scale_factor_temporal
        num_latent_frames = (num_frames - 1) // temporal + 1
        h, w = height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial
        config = self.transformer_config
        channels = config.in_channels - config.out_channels
        condition = torch.zeros(1, channels, num_latent_frames, h, w, dtype=torch.float32)
        if self.is_vae_rank:
            image = image_tensor.to(self.vae.dtype).to(self.device)
            encoded = self.vae.encode_i2v_condition(
                image.unsqueeze(2), num_frames=num_frames, last_image=None
            )
            if self.is_output_rank:
                latent_condition = retrieve_latents(encoded, sample_mode="argmax").to("cpu")
                vae_config = self.vae.config
                dtype = latent_condition.dtype
                latents_mean = torch.tensor(vae_config.latents_mean, dtype=dtype).view(
                    1, vae_config.z_dim, 1, 1, 1
                )
                latents_std = 1.0 / torch.tensor(vae_config.latents_std, dtype=dtype).view(
                    1, vae_config.z_dim, 1, 1, 1
                )
                latent_condition = (latent_condition - latents_mean) * latents_std
                condition = i2v_first_frame_mask(latent_condition, num_latent_frames, temporal)
                condition = condition.to(torch.float32).contiguous()
        if dist.get_world_size() > 1:
            dist.broadcast(condition, src=0, group=get_world_group().cpu_group)
        return condition.to(self.transformer_config_dtype).to(self.device)

    # ---- denoise --------------------------------------------------------------------------

    def _clear_cross_attention_kv_cache(self) -> None:
        self._cross_attention_kv_cache_expert_id = None
        self._cross_attention_kv_cache.clear()

    def _cross_attention_kv_entry_for(self, current_model, kwargs):
        """Look up per-block K/V for one expert and encoder context.

        Wan 2.2 switches from the high-noise expert to the low-noise expert once, so only one
        expert's cache needs to stay live. Within that expert, sequential CFG uses two stable
        encoder contexts (positive and negative prompts).
        """
        expert_id = id(current_model)
        if self._cross_attention_kv_cache_expert_id != expert_id:
            self._cross_attention_kv_cache.clear()
            self._cross_attention_kv_cache_expert_id = expert_id
        context = tuple(
            tensor
            for tensor in (
                kwargs.get("encoder_hidden_states"),
                kwargs.get("encoder_hidden_states_image"),
            )
            if tensor is not None
        )
        # ``id()`` avoids a device sync; the entry keeps the tensors alive so ids are not reused.
        key = tuple(id(tensor) for tensor in context)
        entry = self._cross_attention_kv_cache.get(key)
        if entry is not None and all(
            cached is live for cached, live in zip(entry.context, context, strict=True)
        ):
            return context, key, entry
        return context, key, None

    def _store_cross_attention_kv(self, context, key, kv_cache) -> None:
        self._cross_attention_kv_cache[key] = _CrossAttentionKVCacheEntry(context, kv_cache)
        if len(self._cross_attention_kv_cache) > _MAX_CROSS_ATTENTION_KV_CACHE_ENTRIES_PER_EXPERT:
            if not self._warned_unstable_cross_attention_context:
                self._warned_unstable_cross_attention_context = True
                logger.warning(
                    "Cross-attention K/V cache exceeded %d entries for one expert, so the encoder "
                    "context tensors are not being reused across denoise steps.",
                    _MAX_CROSS_ATTENTION_KV_CACHE_ENTRIES_PER_EXPERT,
                )
            self._cross_attention_kv_cache.pop(next(iter(self._cross_attention_kv_cache)))

    def predict_noise(self, current_model, **kwargs) -> torch.Tensor:
        """One DiT call: the cache-generating graph on an expert's first step, the steady graph
        after."""
        timestep = kwargs["timestep"]
        if not (timestep.ndim == 1 and timestep.shape[0] == 1):  # one video: already [1]
            kwargs["timestep"] = (
                timestep[0].unsqueeze(0)
                if timestep.ndim == 1
                else torch.stack([timestep[0]] * timestep.shape[0])
            )
        if not self.is_i2v:
            kwargs["hidden_states"] = self.cast(kwargs["hidden_states"], current_model.dtype)
        context, key, entry = self._cross_attention_kv_entry_for(current_model, kwargs)
        if entry is None:
            generator = self._compiled_cross_attention_kv_generators.get(id(current_model))
            if generator is None:  # not compiled (CPU)
                generator = current_model.forward_and_cache_cross_attention_kv
            result = generator(**kwargs)
            if not isinstance(result, tuple):
                raise TypeError("cache-generating DiT must return a tuple")
            expected = current_model.cross_attention_kv_cache_size + 1
            if len(result) != expected:
                raise RuntimeError(
                    f"cache-generating DiT returned {len(result)} outputs, expected {expected} "
                    "(noise prediction plus K/V)"
                )
            noise_pred, *kv_cache = result
            self._store_cross_attention_kv(context, key, tuple(kv_cache))
        else:
            kwargs["cross_attention_kv_cache"] = entry.kv_cache
            noise_pred = current_model(**kwargs)
        while isinstance(noise_pred, tuple):
            if not noise_pred:
                raise RuntimeError("transformer returned an empty output tuple")
            noise_pred = noise_pred[0]
        return noise_pred

    def combine_cfg_noise(self, positive, negative, true_cfg_scale: float) -> torch.Tensor:
        """Sequential CFG: ``neg + scale * (pos - neg)`` (the compiled ``_cfg_combine``)."""
        return self._helper("cfg_combine")(
            positive, negative, self.cfg_scale_tensor(true_cfg_scale)
        )

    def predict_noise_maybe_with_cfg(
        self, do_true_cfg: bool, true_cfg_scale: float, positive_kwargs: dict, negative_kwargs
    ) -> torch.Tensor:
        """Both CFG branches and their combine (``cfg_normalize=False``).

        CFG parallel: CFG rank 0 runs the positive branch, rank 1 the negative one, and a compiled
        all-gather + combine over the CFG group gives every rank the same prediction. One CFG
        replica (or no CFG): the base path, both branches here in sequence.
        """
        if not do_true_cfg:
            return self.predict_noise(**positive_kwargs)
        if negative_kwargs is None:
            raise ValueError("CFG requires negative prediction arguments.")
        if self.cfg_size > 1:
            kwargs = positive_kwargs if self.cfg_rank == 0 else negative_kwargs
            local_pred = self.predict_noise(**kwargs)
            gather_combine = self._helpers.get("cfg_gather_combine")
            if gather_combine is None:
                from sglang_neuron.diffusion.distributed.cfg_parallel import (
                    _cfg_gather_combine,
                )

                gather_combine = _cfg_gather_combine
            return gather_combine(local_pred, self.cfg_scale_tensor(true_cfg_scale))
        positive = self.predict_noise(**positive_kwargs)
        negative = self.predict_noise(**negative_kwargs)
        return self.combine_cfg_noise(positive, negative, true_cfg_scale)

    def cfg_scale_tensor(self, true_cfg_scale: float) -> torch.Tensor:
        """The guidance scale as a ``[1]`` float32 device tensor, uploaded once per value."""
        return self._device_constant(
            ("cfg_scale", float(true_cfg_scale)),
            lambda: torch.tensor([true_cfg_scale], dtype=torch.float32),
        )

    def expert(self, index: int) -> nn.Module:
        return self.transformer if index == 1 else self.transformer_2

    def denoise(
        self,
        latents: torch.Tensor,
        scheduler: NeuronFlowUniPCMultistepScheduler,
        schedule: WanSchedule,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        *,
        guidance_low: float,
        guidance_high: float,
        boundary: float | None,
        condition: torch.Tensor | None = None,
        marks: StageMarks | None = None,
    ) -> tuple[torch.Tensor, list]:
        """The UniPC loop over both experts; returns the final latents and the step marks.

        ``marks`` marks the start and the end of every step (``num_steps + 1`` marks).
        """
        marks = marks or StageMarks(device_events=False)
        step_marks = [marks.mark()]
        for host_t, t in zip(schedule.timesteps, schedule.device_timesteps, strict=True):
            index, use_guidance_2 = select_expert(
                host_t,
                boundary,
                has_transformer=self.transformer is not None,
                has_transformer_2=self.transformer_2 is not None,
                i2v=self.is_i2v,
            )
            current_model = self.expert(index)
            guidance = guidance_high if use_guidance_2 else guidance_low
            if self.is_i2v:
                latent_model_input = self._helper("i2v_model_input")(latents, condition)
            else:
                latent_model_input = latents
            # A [1] tensor per step, uploaded once per step count (prepare_timesteps).
            timestep = t if latents.shape[0] == 1 else t.expand(latents.shape[0])
            do_true_cfg = guidance > 1.0 and negative_prompt_embeds is not None
            positive_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": prompt_embeds,
                "attention_kwargs": {},
                "return_dict": False,
                "current_model": current_model,
            }
            negative_kwargs = None
            if do_true_cfg:
                negative_kwargs = {
                    **positive_kwargs,
                    "encoder_hidden_states": negative_prompt_embeds,
                }
            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg, guidance, positive_kwargs, negative_kwargs
            )
            noise_pred = self.cast(noise_pred, latents.dtype)
            latents = scheduler.step(noise_pred, host_t, latents, return_dict=False)[0]
            step_marks.append(marks.mark())
        # Frees the solver history (it pins solver_order + 1 latents) and rewinds to step 0, so the
        # request's next video starts a fresh solve on the same schedule.
        scheduler.reset_solver_state()
        return latents, step_marks

    # ---- decode ---------------------------------------------------------------------------

    def decode(self, latents: torch.Tensor) -> torch.Tensor | None:
        """Decode on the VAE group; the output rank returns ``[1, 3, F, H, W]`` on the CPU."""
        if not self.is_vae_rank:
            return None
        vae = self.vae
        dtype = vae.dtype
        z_dim = vae.config.z_dim
        latents_mean = self._device_constant(
            ("latents_mean", dtype),
            lambda: torch.tensor(vae.config.latents_mean, dtype=dtype).view(1, z_dim, 1, 1, 1),
        )
        latents_std = self._device_constant(
            ("latents_inv_std", dtype),
            lambda: 1.0 / torch.tensor(vae.config.latents_std, dtype=dtype).view(1, z_dim, 1, 1, 1),
        )
        z = self._helper("vae_input")(latents, latents_mean, latents_std, dtype)
        video = vae.decode(z, return_dict=False)[0]
        if not self.is_output_rank:
            return None
        return video.to("cpu")

    # ---- one request ----------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_steps: int = 40,
        guidance_scale: float | None = None,
        guidance_scale_2: float | None = None,
        flow_shift: float = 5.0,
        boundary_ratio: float | None = None,
        generators: list[torch.Generator] | None = None,
        image=None,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor | None, dict]:
        """Generate one video per generator for one prompt (and, for I2V, one first frame).

        Returns ``(videos, metrics)``: on the output rank a ``[n, 3, F, H, W]`` CPU tensor in
        [-1, 1] in the VAE dtype; ``None`` on the other ranks.
        """
        t_start = time.perf_counter()
        if generators is None:
            generators = [torch.Generator(device="cpu").manual_seed(0)]
        n = len(generators)
        height, width, num_frames = effective_dimensions(
            height,
            width,
            num_frames,
            spatial=self.vae_scale_factor_spatial,
            temporal=self.vae_scale_factor_temporal,
            patch=self.transformer_config.patch_size[1],
            i2v=self.is_i2v,
        )
        guidance_low, guidance_high = resolve_guidance(
            guidance_scale, guidance_scale_2, default=5.0 if self.is_i2v else 4.0
        )
        do_classifier_free_guidance = guidance_low > 1.0 or guidance_high > 1.0
        if boundary_ratio is None:
            boundary_ratio = self.boundary_ratio if self.boundary_ratio is not None else 0.875
        marks = StageMarks(device_events=self.is_output_rank)

        # Host-to-device inputs first, before any graph of this request runs (the order the FLUX
        # model keeps; see NeuronFluxModel.generate).
        scheduler, schedule = self.prepare_timesteps(num_steps, flow_shift)
        boundary = boundary_timestep(boundary_ratio, scheduler.config.num_train_timesteps)
        noise = self.prepare_latents(height, width, num_frames, generators)
        image_tensor = None
        if self.is_i2v:
            if image is None:
                raise ValueError("Wan2.2 I2V needs a first-frame image")
            if self.is_vae_rank:
                image_tensor = preprocess_i2v_image(
                    image, height, width, self.vae_scale_factor_spatial
                )

        self._clear_cross_attention_kv_cache()
        try:
            text_start = marks.mark()
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                max_sequence_length=max_sequence_length,
            )
            condition_start = marks.mark()
            condition = None
            if self.is_i2v:
                condition = self.prepare_i2v_condition(image_tensor, num_frames, height, width)
            condition_end = marks.mark()

            spans = []  # (denoise start, decode start, decode end) per video
            step_marks: list = []
            videos = []
            for i in range(n):
                denoise_start = marks.mark()
                latents, step_marks = self.denoise(
                    noise[i],
                    scheduler,
                    schedule,
                    prompt_embeds,
                    negative_prompt_embeds,
                    guidance_low=guidance_low,
                    guidance_high=guidance_high,
                    boundary=boundary,
                    condition=condition,
                    marks=marks,
                )
                # Free the denoise-only device tensors before the VAE decode, the request's memory
                # peak: the cross-attention K/V (about 100 MiB per context per rank at 1080p; the
                # next video starts on the other expert, which regenerates it anyway), this video's
                # noise and, after the last video, the condition and the prompt embeddings.
                self._clear_cross_attention_kv_cache()
                noise[i] = None
                if i == n - 1:
                    condition = prompt_embeds = negative_prompt_embeds = None
                decode_start = marks.mark()
                video = self.decode(latents)
                spans.append((denoise_start, decode_start, marks.mark()))
                if video is not None:
                    videos.append(video)
        finally:
            self._clear_cross_attention_kv_cache()

        e2e_forward_seconds = time.perf_counter() - t_start
        # After the output rank's video copy: its device events are complete.
        step_seconds = [marks.seconds(a, b) for a, b in zip(step_marks, step_marks[1:])]
        metrics = {
            "text_encode_seconds": marks.seconds(text_start, condition_start),
            "vae_encode_seconds": marks.seconds(condition_start, condition_end),
            "denoise_seconds": sum(marks.seconds(start, end) for start, end, _ in spans),
            "denoise_step_seconds": step_seconds,
            "denoise_step_p50_seconds": statistics.median(step_seconds) if step_seconds else 0.0,
            "vae_decode_seconds": sum(marks.seconds(start, end) for _, start, end in spans),
            "e2e_forward_seconds": e2e_forward_seconds,
            "num_videos": n,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_steps": len(step_seconds),
            "guidance_scale": guidance_low,
            "guidance_scale_2": guidance_high,
            "flow_shift": flow_shift,
            "boundary_ratio": boundary_ratio,
            "cfg_parallel_size": self.cfg_size,
            "task": "i2v" if self.is_i2v else "t2v",
        }
        if not self.is_output_rank:
            return None, metrics
        return torch.cat(videos), metrics


def describe(metrics: dict[str, Any]) -> str:
    """One log line for a request's metrics, in the format the device end-to-end test parses."""
    return (
        "[NeuronWan] {task} {width}x{height}x{num_frames} steps={num_steps} "
        "flow_shift={flow_shift} boundary={boundary_ratio} guidance={guidance_scale}/"
        "{guidance_scale_2} cfg_parallel={cfg_parallel_size} e2e_forward={e2e:.3f}s "
        "text={text:.3f}s vae_encode={enc:.3f}s denoise={den:.3f}s step_p50={p50:.4f}s "
        "vae={dec:.3f}s"
    ).format(
        e2e=metrics["e2e_forward_seconds"],
        text=metrics["text_encode_seconds"],
        enc=metrics["vae_encode_seconds"],
        den=metrics["denoise_seconds"],
        p50=metrics["denoise_step_p50_seconds"],
        dec=metrics["vae_decode_seconds"],
        **metrics,
    )
