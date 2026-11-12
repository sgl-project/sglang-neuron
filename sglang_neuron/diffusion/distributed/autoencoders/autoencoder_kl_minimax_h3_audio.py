# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Neuron port of diffusers ``AutoencoderKLMiniMaxH3Audio``'s decoder: ``dec_in_proj`` and the
# BigVGAN vocoder. The config, module tree (checkpoint keys load 1:1), SnakeBeta activations
# and AMP blocks are upstream's.
#
# Neuron-specific changes
# 1. The audio encoder (``encoder``, ``pre_block``, ``mean_proj``, ``logs_proj``) is dropped:
#    t2va and fl2va never encode a waveform.
# 2. FP32 end to end, as upstream pins it. The model is built under an FP32 default dtype, so
#    the Kaiser filters are FP32 under a BF16 loader default too.
# 3. Weight norms are folded into the convolution weights after loading, so the graphs read
#    plain weights.
# 4. NeuronMiniMaxH3AudioActivation1d: the alias-free activation runs on the two phases of its
#    2x-upsampled signal, which is never built. Upstream's depthwise Kaiser ``conv_transpose1d``
#    overflows SBUF in neuronx-cc (NCC_INLA001), and the phase form also skips a stride-2
#    interleave and read, so an AMP block runs faster. Replicate pads are a ``cat`` of
#    edge slices, which traces where ``F.pad(mode="replicate")`` does not.
# 5. ``decode`` runs fixed windows of ``AUDIO_WINDOW`` latents with ``AUDIO_HALO`` latents of
#    context on each side (``audio_windows``), so one shape serves every clip longer than a
#    window. A window runs one compiled graph per stage piece: upsampler, each AMP block, then
#    the output head. A whole window fails to load (``dlr_kelf_stage: Failed to load
#    subgraph``), and a graph per stage is slower than per-block graphs. A pass runs
#    one window: four stacked on the batch give the last stages 4.5 GB of spill DMA rings.
#    ``decode_window`` lets the pipeline decode the windows in parallel, one per rank.

from __future__ import annotations

import contextlib
import functools

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3AudioActivation1d,
    MiniMaxH3AudioBigVGANDecoder,
)
from diffusers.models.autoencoders.vae import DecoderOutput


# 24 latents of context per side (BigVGAN's receptive field is 30.6 latents) reproduce a
# whole-clip decode to FP32 rounding. A 96-latent window fails neuronx-cc in stage 5's kernel-3
# AMP block (NCC_ITEN406).
AUDIO_WINDOW = 104
AUDIO_HALO = 24
# Named cache_size_limit before torch 2.7.
_RECOMPILE_LIMIT_KEY = (
    "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _replicate_pad_1d(hidden_states: torch.Tensor, before: int, after: int) -> torch.Tensor:
    """``F.pad(hidden_states, (before, after), mode="replicate")`` as a ``cat`` of edge slices."""
    length = hidden_states.shape[-1]
    parts = [hidden_states]
    if before:
        parts.insert(0, hidden_states.narrow(-1, 0, 1).expand(*hidden_states.shape[:-1], before))
    if after:
        parts.append(
            hidden_states.narrow(-1, length - 1, 1).expand(*hidden_states.shape[:-1], after)
        )
    return torch.cat(parts, dim=-1) if len(parts) > 1 else hidden_states


def _correlate(hidden_states: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """Depthwise valid correlation of every channel with the same ``taps``."""
    channels = hidden_states.shape[1]
    return F.conv1d(hidden_states, taps.reshape(1, 1, -1).expand(channels, -1, -1), groups=channels)


class NeuronMiniMaxH3AudioActivation1d(MiniMaxH3AudioActivation1d):
    """Upstream's alias-free activation on the two phases of its 2x-upsampled signal ``u``.

    ``u``'s even and odd samples are each a 6-tap correlation of ``x``, the activation runs on
    each phase, and the stride-2 low-pass is the sum of a 6-tap correlation of each phase.

    The offsets are upstream's for ``ratio=2, kernel_size=12`` (every activation in the
    decoder): the upsampler's ``(5, 5)`` replicate pad and ``(15, 15)`` trim, and the low-pass's
    ``(5, 6)`` replicate pad of ``u``, so both phases are padded with the even phase's first
    and the odd phase's last sample.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up = self.upsample.filter.reshape(-1)
        low = self.downsample.lowpass.filter.reshape(-1)
        ratio = self.upsample.ratio
        even = ratio * _correlate(_replicate_pad_1d(hidden_states, 3, 2), up[1::2].flip(-1))
        odd = ratio * _correlate(_replicate_pad_1d(hidden_states, 2, 3), up[0::2].flip(-1))
        even, odd = self.act(even), self.act(odd)
        length = even.shape[-1]
        first = even.narrow(-1, 0, 1)
        last = odd.narrow(-1, length - 1, 1)
        edges = hidden_states.shape[:-1]
        even = torch.cat([first.expand(*edges, 2), even, last.expand(*edges, 3)], dim=-1)
        odd = torch.cat([first.expand(*edges, 3), odd, last.expand(*edges, 2)], dim=-1)
        return _correlate(even, low[1::2]) + _correlate(odd, low[0::2])


def prepare_for_neuron(module: torch.nn.Module) -> torch.nn.Module:
    """Swap the Neuron activation into an upstream audio VAE (sub)tree in place; it only
    overrides ``forward``, so parameters and buffers are kept."""
    for submodule in module.modules():
        if type(submodule) is MiniMaxH3AudioActivation1d:
            up, low = submodule.upsample, submodule.downsample.lowpass
            if (up.ratio, up.filter.shape[-1], low.stride, low.filter.shape[-1]) != (2, 12, 2, 12):
                raise NotImplementedError(
                    "the polyphase activation is written for ratio 2, kernel 12"
                )
            submodule.__class__ = NeuronMiniMaxH3AudioActivation1d
    return module


def audio_windows(num_latents: int) -> list[tuple[int, int, int]]:
    """``(window_start, core_start, core_end)`` covering ``[0, num_latents)``.

    A clip of at most ``AUDIO_WINDOW`` latents is one window at its own length (8, 37, 65 or 93
    latents for ``17n + 5`` frames). Otherwise every window is ``AUDIO_WINDOW`` latents, placed
    so its core has ``AUDIO_HALO`` latents of context on each side or meets the clip's boundary.
    """
    if num_latents <= AUDIO_WINDOW:
        return [(0, 0, num_latents)]
    core = AUDIO_WINDOW - 2 * AUDIO_HALO
    return [
        (
            min(max(core_start - AUDIO_HALO, 0), num_latents - AUDIO_WINDOW),
            core_start,
            min(core_start + core, num_latents),
        )
        for core_start in range(0, num_latents, core)
    ]


class _UpsampleGraph(torch.nn.Module):
    """A BigVGAN stage's upsampler; in the first stage, after ``dec_in_proj`` and ``conv_pre``."""

    def __init__(self, upsample: torch.nn.Module, *before: torch.nn.Module):
        super().__init__()
        self.before = torch.nn.ModuleList(before)
        self.upsample = upsample

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.before:
            hidden_states = module(hidden_states)
        return self.upsample(hidden_states)


class _BlockGraph(torch.nn.Module):
    """One AMP block of a stage, added to the sum of the stage's earlier blocks; the last block
    turns the sum into upstream's mean."""

    def __init__(self, block: torch.nn.Module, num_kernels: int, last: bool):
        super().__init__()
        self.block = block
        self.num_kernels = num_kernels
        self.last = last

    def forward(self, hidden_states: torch.Tensor, total: torch.Tensor | None) -> torch.Tensor:
        out = self.block(hidden_states)
        if total is not None:
            out = total + out
        return out / self.num_kernels if self.last else out


class _HeadGraph(torch.nn.Module):
    """The decoder's output activation, ``conv_post`` and clamp."""

    def __init__(self, decoder: MiniMaxH3AudioBigVGANDecoder):
        super().__init__()
        self.activation_post = decoder.activation_post
        self.conv_post = decoder.conv_post

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_post(self.activation_post(hidden_states))
        return torch.clamp(hidden_states, min=-1.0, max=1.0)


class NeuronAutoencoderKLMiniMaxH3Audio(AutoencoderKLMiniMaxH3Audio):
    """``AutoencoderKLMiniMaxH3Audio`` decoding on Neuron (see the header).

    Mono, as upstream: the stereo channels are two batch items, ``(2, 32, T)`` latents in and
    ``(2, 1, T * 800)`` FP32 samples out, on the host.
    """

    # Keeps upstream's signature visible to ConfigMixin.from_config.
    @functools.wraps(AutoencoderKLMiniMaxH3Audio.__init__)
    def __init__(self, *args, **kwargs):
        with set_default_torch_dtype(torch.float32):
            super().__init__(*args, **kwargs)
        del self.encoder, self.pre_block, self.mean_proj, self.logs_proj
        prepare_for_neuron(self)
        self.decoder_graphs = None
        self._window_lengths = None
        self._compiled_stages = None
        self._compiled_head = None

    def prepare_decoder_graphs(self) -> None:
        """Fold the weight norms into the loaded weights and build the decoder's graph modules:
        per stage, its upsampler and AMP blocks, then the head."""
        for module in self.modules():
            if hasattr(module, "weight_g"):
                torch.nn.utils.remove_weight_norm(module)
        decoder = self.decoder
        stages = []
        for index in range(decoder.num_upsamples):
            before = (self.dec_in_proj, decoder.conv_pre) if index == 0 else ()
            blocks = [
                _BlockGraph(
                    decoder.resblocks[index * decoder.num_kernels + j],
                    decoder.num_kernels,
                    last=j == decoder.num_kernels - 1,
                )
                for j in range(decoder.num_kernels)
            ]
            stages.append(
                torch.nn.ModuleList([_UpsampleGraph(decoder.ups[index][0], *before)] + blocks)
            )
        self.decoder_graphs = torch.nn.ModuleList(stages + [_HeadGraph(decoder)])

    def compile(self, *args, window_lengths, **compiler_kwargs):
        """Compile the decoder's graphs, named ``<model_name>_stage<i>_upsample``,
        ``<model_name>_stage<i>_block<j>`` and ``<model_name>_head``, for windows of
        ``window_lengths`` latents."""
        self._window_lengths = tuple(window_lengths)
        options = compiler_kwargs.pop("options", None) or {}
        model_name = options.get("model_name", "minimax_h3_audio_vae")

        def compiled(module, name):
            return torch.compile(
                module, *args, **compiler_kwargs, options={**options, "model_name": name}
            )

        *stages, head = self.decoder_graphs
        self._compiled_stages = [
            [compiled(stage[0], f"{model_name}_stage{index}_upsample")]
            + [
                compiled(block, f"{model_name}_stage{index}_block{j}")
                for j, block in enumerate(stage[1:])
            ]
            for index, stage in enumerate(stages)
        ]
        self._compiled_head = compiled(head, f"{model_name}_head")
        return self

    def _run(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode ``latents`` whole, through the compiled graphs, back to the host."""
        hidden_states = latents.to(self.dec_in_proj.weight.device)
        # Dynamo caches every graph instance at every window length under its class's code
        # object; the AMP blocks' holds the most entries.
        limit = len(self.decoder.resblocks) * len(self._window_lengths)
        with torch._dynamo.config.patch(**{_RECOMPILE_LIMIT_KEY: limit}):
            for upsample, *blocks in self._compiled_stages:
                hidden_states = upsample(hidden_states)
                total = None
                for block in blocks:
                    total = block(hidden_states, total)
                hidden_states = total
            hidden_states = self._compiled_head(hidden_states)
        return hidden_states.cpu()

    def decode_window(self, latents: torch.Tensor, window: tuple[int, int, int]) -> torch.Tensor:
        """One ``audio_windows`` window of the host FP32 ``latents``, decoded through the
        compiled graphs and cut to its core."""
        start, core_start, core_end = window
        length = min(AUDIO_WINDOW, latents.shape[-1])
        if length not in self._window_lengths:
            raise ValueError(
                f"the audio VAE is compiled for windows of {list(self._window_lengths)} latents, "
                f"not {length}"
            )
        decoded = self._run(latents.narrow(-1, start, length))
        hop = self.hop_length
        return decoded[..., (core_start - start) * hop : (core_end - start) * hop]

    def decode(self, latents: torch.Tensor, return_dict: bool = True):
        """Upstream ``decode``, windowed through the compiled stages once compiled."""
        if self._compiled_stages is None:
            return super().decode(latents, return_dict=return_dict)
        latents = latents.float().cpu()
        windows = audio_windows(latents.shape[-1])
        decoded = torch.cat([self.decode_window(latents, window) for window in windows], dim=-1)
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def load_weights(self, model_name_or_path: str) -> None:
        """Load the decoder half of the FP32 checkpoint, then build the decoder's graphs.

        Names map 1:1; the Kaiser filters are built, not loaded.
        """
        from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint

        # The loader raises on a checkpoint key it cannot find.
        mappings = {name: name for name, _ in self.named_parameters()}
        state_dict = (
            SafetensorsCheckpoint(model_name_or_path)
            .load_sharded_pipelined(0, 1, self, mappings, torch.device("cpu"))
            .state_dict
        )
        self.load_state_dict(state_dict, strict=False, assign=True)
        self.prepare_decoder_graphs()
