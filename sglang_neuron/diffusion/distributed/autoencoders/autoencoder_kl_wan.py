# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
# Patch-parallel Neuron Wan 2.1 / 2.2 VAE: the compiled decoder (T2V and I2V) and the compiled
# encoder for the I2V image condition, both tiled over the VAE-parallel group.
#
# Neuron-specific changes
# 1. NeuronWanRMSNorm: replaced F.normalize with manual x * rsqrt(mean(x^2)).
#    F.normalize generates a NaN-check (compare + select) that forces extra
#    CROSS_LANE_REDUCE synchronization on GpSimd at each call (~25 calls
#    in the decoder). The manual path eliminates the NaN guard entirely.
#    Not named WanRMS_norm, so a framework patch that rebinds modules' globals of that
#    name leaves it alone.
# 2. WanUpsample: replaced nn.Upsample(nearest-exact) with repeat_interleave.
#    Repeat_interleave maps to bulk tensor copies on the tensor engine. Also
#    eliminates the float32 cast that doubled memory traffic through the upsample.
# 3. Remove all to(cache_x.device) calls — feat_cache tensors stay on the Neuron
#    device throughout; no CPU↔device transfers needed.
# 4. WanResample: added first_chunk param to skip time_conv on first frame.
#    Replaces diffusers' None→"Rep" sentinel-based state machine with an
#    explicit flag, compatible with zero-initialized cache tensors.
# 5. NeuronWanDecoder3d: runs post_quant_conv and the decoder in one graph, with
#    feat_cache as explicit inputs and outputs. The first frame and the later
#    frames compile to two graphs (_compiled_decoder_first / _rest).
# 6. NeuronAutoencoderKLWan._decode: per-frame decode loop that slices z
#    on-device, copies only each output frame to CPU, and keeps feat_cache on
#    Neuron device throughout (no CPU↔device transfer for cache).
# 7. Set feat_idx default value to None and explicitly pass [0] in NeuronWanDecoder3d.
#
# NkiConv2d, its stride-2 encoder downsample and the shared conv3d launcher live in nki_conv.py;
# the shared executor (groups, host gather, compiled pack, gather and blend) in vae_executor.py.
# The patch-parallel tile split, balance and execute, which the shared executor does not carry,
# are in _WanVaeExecutor below.


import logging
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import nki
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.activations import get_activation
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWan,
    AvgDown3D,
    DupUp3D,
    patchify,
    unpatchify,
)
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from nkilib.experimental.conv.conv3d_temporal_unroll import (
    conv3d_temporal_unroll,
    should_use_temporal_unroll,
)

from sglang_neuron.diffusion import compile_backend
from sglang_neuron.diffusion.distributed.autoencoders.nki_conv import (
    _VAE_NKI_CONV_MIN_CHANNELS,
    _VAE_NKI_CONV_PADDING,
    NkiConv2d,
    _pack_nki_filters,
    _vae_conv3d_launch,
    _vae_nki_conv3d,
)
from sglang_neuron.diffusion.distributed.autoencoders.vae_executor import (
    GridSpec,
    NeuronDistributedVaeExecutor,
    decoder_tile_geometry,
    new_vae_parallel_groups,
    vae_nki_attn,
)
from sglang_neuron.diffusion.kernels.kernel_choice import device_reason, record_kernel_choice
from sglang_neuron.diffusion.neuron_api import wrap_nki_kernel_if_supported
from sglang_neuron.utils.hardware_config import get_logical_nc_config

logger = logging.getLogger(__name__)

# The neuronx-cc flags of the Wan VAE: WAN_VAE_COMPILER_ARGS for T2V, WAN_I2V_VAE_COMPILER_ARGS
# for I2V, which also compiles the encoder and does not disable the hierarchy splitter.
WAN_I2V_VAE_COMPILER_ARGS = (
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
)
WAN_VAE_COMPILER_ARGS = (
    *WAN_I2V_VAE_COMPILER_ARGS,
    "--internal-hlo2tensorizer-options=--hierarchy-splitter=false",
)


def _use_vae_nki_attention() -> bool:
    """Mid-block attention on device: the NKI kernel (attention_cte) or the compiler path.

    LNC 2 uses the kernel; LNC 1 takes the compiler path by design (the kernel is validated
    at LNC 2 only). The decision is recorded once through ``record_kernel_choice``.
    """
    if get_logical_nc_config() != 2:
        return record_kernel_choice(
            "vae_attention", False, "LNC 1 not validated for attention_cte", by_design=True
        )
    return record_kernel_choice("vae_attention", True, "LNC 2")


_VAE_ATTN_D_TILE_SIZE = 128
_VAE_ATTN_PAD_HEAD_DIM = 512
_VAE_ATTN_FLASH_THRESHOLD = 10 * 1024


def _should_pad_vae_attn_head_dim(seq_len: int, head_dim: int) -> bool:
    if not (_VAE_ATTN_D_TILE_SIZE < head_dim < _VAE_ATTN_PAD_HEAD_DIM):
        return False

    num_d_tiles = (head_dim + _VAE_ATTN_D_TILE_SIZE - 1) // _VAE_ATTN_D_TILE_SIZE
    flash_threshold = _VAE_ATTN_FLASH_THRESHOLD // num_d_tiles
    return seq_len > flash_threshold


# ---------------------------------------------------------------------------
# NKI conv3d for the VAE's 3x3 convolutions
# ---------------------------------------------------------------------------
#
# Dispatch is unconditional, matching every other NKI kernel in this repo.
# can_run_kernel honours SGLANG_NEURON_DISABLE_NKI_KERNELS (the kill switch for the
# NKI kernels), so anything finer is a code revert.
#
# Why only *some* call sites: measured in isolation, the generic conv3d kernel wins
# on the 3x3x3 resblock convs and is at or below parity on the 2D upsample sites --
# yet those 2D sites still ship, because in-graph they remove a compiler<->NKI
# round-trip: their marginal saving on top of the 3D group exceeds the
# 3D group's own standalone saving. Isolated measurements mispredict in both
# directions, so a site ships only when the full-model A/B says the group wins;
# everything else falls through to the compiler.
#
# conv_out is its own case: generic conv3d is slower there (PE-starved), and the
# small-C_out temporal-unroll variant clears its own gate only when tiling is OFF
# (it needs W_out > 512), so at the shipped tiled shapes conv_out runs on the
# compiler.
#
# Shape note: these convs are *causal*, and WanCausalConv3d.forward prepends 2
# cached frames while reducing pad_d_left to 0. So the convolution the hardware runs
# always has kernel padding (0, 0, 1, 1, 1, 1) (nki_conv._VAE_NKI_CONV_PADDING). That
# single static config covers every eligible site, which is what makes one traced kernel
# enough; the generic launcher is nki_conv's, shared with NkiConv2d.
#
# One gate covers BOTH the 3D sites and the lifted-Conv2d sites, deliberately:
# they are not independently useful. In an isolated 16-rank run the 2D group
# alone gives no saving, while the two together beat the sum of their
# parts. The 2D group's value is contextual -- converting a
# conv that sits BETWEEN existing NKI regions removes a compiler<->NKI round-trip
# instead of adding one -- so no "3D only" or "2D only" configuration can be
# selected.
#
# Encoder downsampling has a separate stride-2 specialization and C=384 gate
# (nki_conv.NkiConv2d). Sites deliberately NOT routed: 1x1 to_qkv/proj (~0.7 GMAC
# combined, no measured saving).

# Eligibility threshold for the generic conv3d kernel's output depth. Deliberately
# not env-tunable -- an unvalidated MIN_D_OUT=1 reachable from the environment is
# unsafe.
#
# MIN_D_OUT=2 rests on measurement, not on a "too little work per call" argument:
# relaxing it to 1 made vae_decode slower end-to-end and gave a
# deterministic allreduce scheduling failure in an isolated run.
# The channel floor (nki_conv._VAE_NKI_CONV_MIN_CHANNELS = 96) excludes conv_in
# (C_in = z_dim = 16); post_quant_conv is already excluded by its 1x1 kernel, before
# channels are considered. C_in only ever takes 16/96/192/384, so no site sits near
# the boundary and the exact value is untested.
_VAE_NKI_CONV_MIN_D_OUT = 2

# TODO: most of the dispatch complexity below works around kernel specialisation
# rather than expressing a real policy. A more generic conv kernel would remove:
#   1. the shape thresholds here (kernel picks its own tiling);
#   2. the (kernel_size, stride, padding) preconditions in the
#      _can_use_nki_conv_kernel predicates -- these are trace-time constants, so
#      each config needs its own @nki.jit wrapper;
#   3. the conv3d vs conv3d_temporal_unroll choice in _maybe_nki_forward;
#   4. NkiConv2d's D=1 lifting, if a native 2D entry point existed.


@nki.jit
def _vae_conv3d_temporal_unroll_kernel(x, filters, bias):
    return conv3d_temporal_unroll(
        x,
        filters,
        bias,
        stride=(1, 1, 1),
        padding=_VAE_NKI_CONV_PADDING,
        dilation=(1, 1, 1),
    )


# Wrapped once, at import (see the launcher note in attention/cp_attention.py). LNC 2 only: None
# at LNC 1, where the conv gates take the compiler path.
_vae_conv3d_temporal_unroll_launch = wrap_nki_kernel_if_supported(
    _vae_conv3d_temporal_unroll_kernel, supported_lnc=(2,)
)


def _vae_nki_conv3d_temporal_unroll(
    x: torch.Tensor, filters: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return _vae_conv3d_temporal_unroll_launch(x, filters, bias)


CACHE_T = 2


def _tail_cache(x):
    """Extract the last CACHE_T temporal frames from x for caching.

    The copy is contiguous whatever the layout of ``x``. After a WanResample, ``x`` is a
    ``[B, T, C, H, W] -> [B, C, T, H, W]`` permute, and a layout-preserving clone would hand the
    next compiled call cache inputs whose strides differ from the call before: Dynamo guards
    input strides, so that recompiles one more full decoder (or encoder) graph per tile shape.
    """
    return x[:, :, -CACHE_T:, :, :].clone(memory_format=torch.contiguous_format)


class WanCausalConv3d(nn.Conv3d):
    r"""
    A custom 3D causal convolution layer with feature caching support.

    This layer extends the standard Conv3D layer by ensuring causality in the time dimension and handling feature
    caching for efficient inference.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0

    Neuron changes:
      - Removed .to(cache_x.device) in forward() to avoid cross-device copy
        errors within compiled graphs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Set up causal padding
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

        # Set by install_nki_dispatch() when this site is NKI-eligible.
        # None => use the compiler path.
        self.register_buffer("_nki_packed_filters", None, persistent=False)

    def _can_use_nki_conv_kernel(self, target_device=None) -> bool:
        """Static (weight-only) half of the NKI dispatch test.

        Checked at pack time, so only the filters that a forward can use are pre-packed.
        The dynamic half -- the post-cache-prepend padding and D_out, which
        depend on the input -- is checked in forward().
        """
        from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel

        # Framework guard first: honours the SGLANG_NEURON_DISABLE_NKI_KERNELS kill
        # switch and refuses CPU weights.
        # Only a can_run_kernel refusal can raise. The other refusals are the measured
        # dispatch policy, so they are recorded as by design.
        device = self.weight if target_device is None else str(target_device)
        if not can_run_kernel(device):
            return record_kernel_choice("vae_conv3d", False, device_reason(device))
        if _vae_conv3d_launch is None:
            return record_kernel_choice(
                "vae_conv3d", False, "the VAE conv kernels run at LNC 2 only", by_design=True
            )
        if tuple(self.kernel_size) != (3, 3, 3):
            return record_kernel_choice(
                "vae_conv3d", False, "kernel size is not 3x3x3", by_design=True
            )
        if tuple(self.stride) != (1, 1, 1):
            return record_kernel_choice("vae_conv3d", False, "stride is not 1", by_design=True)
        if tuple(self.dilation) != (1, 1, 1):
            return record_kernel_choice("vae_conv3d", False, "dilation is not 1", by_design=True)
        if self.groups != 1:
            return record_kernel_choice("vae_conv3d", False, "grouped conv", by_design=True)
        if self.bias is None:
            return record_kernel_choice("vae_conv3d", False, "no bias", by_design=True)
        # Spatial padding must be exactly 1 so the effective kernel padding is
        # the single static config the traced kernels were built for.
        if self._padding[0] != 1 or self._padding[2] != 1:
            return record_kernel_choice(
                "vae_conv3d", False, "spatial padding is not 1", by_design=True
            )
        # Either a fat conv (generic conv3d wins) or a narrow output conv that
        # the temporal-unroll variant can rescue. Everything between is where
        # the generic kernel measured *slower* than the compiler.
        min_ch = _VAE_NKI_CONV_MIN_CHANNELS
        if self.in_channels >= min_ch and self.out_channels >= min_ch:
            return record_kernel_choice("vae_conv3d", True, "eligible (wide)")
        if self.out_channels <= 32:
            return record_kernel_choice("vae_conv3d", True, "eligible (narrow output)")
        return record_kernel_choice(
            "vae_conv3d", False, "channels between the measured limits", by_design=True
        )

    def install_nki_dispatch(self, target_device=None) -> bool:
        """Pack own weight into NKI layout ``[K_d, K_h, K_w, C_in, C_out]``.

        Static half of the gate only; the dynamic half stays in ``forward()``, so a
        packed site can still take the compiler path. Returns True if packed.
        """
        if self._nki_packed_filters is not None:
            return True
        if not self._can_use_nki_conv_kernel(target_device):
            return False
        self._nki_packed_filters = _pack_nki_filters(self.weight, (2, 3, 4, 1, 0))
        return True

    # Set by wan_spatial_shard.install_spatial_shard for spatially-sharded decode / encode.
    _spatial_shard = None
    # Set by the sharded encoder on conv_in: the input block already carries its spatial
    # halo (attached on the host), so only time is padded.
    _input_haloed = False

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]

        if self._input_haloed:
            x = F.pad(x, (0, 0, 0, 0, padding[4], padding[5]))
            return F.conv3d(x, self.weight, self.bias, self.stride)

        if self._spatial_shard is not None:
            # Exact sharded conv: exchange the spatial halo (after the temporal cache
            # prepend, so cached frames' edges travel too), then pad time only.
            x = self._spatial_shard.halo(x, self._padding[0])
            x = F.pad(x, (0, 0, 0, 0, padding[4], padding[5]))
            return F.conv3d(x, self.weight, self.bias, self.stride)

        nki_out = self._maybe_nki_forward(x, padding)
        if nki_out is not None:
            return nki_out

        x = F.pad(x, padding)
        return super().forward(x)

    def _maybe_nki_forward(self, x, padding):
        """Run an NKI conv kernel if this exact convolution is a measured win.

        Returns None to fall through to the compiler. Every condition here is
        resolvable at trace time (module attrs + static shapes), so the branch
        is decided during tracing and only one path is ever compiled.
        """
        if self._nki_packed_filters is None:
            return None
        # The cache prepend must have consumed all temporal padding, leaving the
        # (0, 0, 1, 1, 1, 1) config the kernels were traced with.
        if padding[4] != 0 or padding[5] != 0:
            return None

        d_out = x.shape[2] - self.kernel_size[0] + 1
        w_out = x.shape[4]
        if d_out < _VAE_NKI_CONV_MIN_D_OUT:
            return None

        x_in = x.to(self._nki_packed_filters.dtype)
        bias = self.bias.to(self._nki_packed_filters.dtype)

        if self.out_channels <= 32:
            # Generic conv3d is slower here (PE-starved: C_out sits on the PSUM
            # partition dim). The temporal-unroll variant stacks D_out positions
            # into one matmul and is the only profitable option -- and only when
            # its own gate agrees.
            if not should_use_temporal_unroll(
                self.out_channels, d_out, self.in_channels, self.kernel_size[0], w_out
            ):
                return None
            return _vae_nki_conv3d_temporal_unroll(x_in, self._nki_packed_filters, bias)

        return _vae_nki_conv3d(x_in, self._nki_packed_filters, bias)


class NeuronWanRMSNorm(nn.Module):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
            Default is True.
        images (bool, optional): Whether the input represents image data. Default is True.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.

    Neuron changes:
      - Replaced F.normalize with manual x * rsqrt(mean(x^2)) to eliminate
        NaN-guard CROSS_LANE_REDUCE synchronization at each call.
    """

    def __init__(
        self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        dim = 1 if self.channel_first else -1
        # Manual RMS norm avoids F.normalize's NaN check (compare + select)
        # and its generic L2 path which generates expensive CROSS_LANE_REDUCE.
        # This is mathematically equivalent: F.normalize(x, dim) * scale
        #   = x / ||x||_2 * sqrt(D)
        #   = x / sqrt(mean(x^2))
        #   = x * rsqrt(mean(x^2))
        rms = x.to(torch.float32).pow(2).mean(dim, keepdim=True).clamp(min=1e-8).rsqrt()
        return (x * rms).to(self.gamma.dtype) * self.gamma + self.bias


class WanUpsample(nn.Module):
    r"""
    Perform 2x spatial upsampling via repeat_interleave.

    This avoids nn.Upsample(nearest-exact) which lowers to thousands of
    indirect DMA gather instructions on Neuron. repeat_interleave maps
    to efficient bulk tensor operations instead.

    Args:
        scale_factor: Spatial scale factor (tuple of 2 floats, e.g. (2.0, 2.0)).
        mode: Ignored, kept for API compatibility.

    Neuron changes:
      - Replaced nn.Upsample(nearest-exact) with repeat_interleave.
        nn.Upsample lowers to thousands of indirect DMA gathers on Neuron.
        repeat_interleave maps to efficient bulk tensor copies.
      - Removed the float32 cast (diffusers' WanUpsample does .float() then
        .type_as()) that doubled memory traffic.
    """

    def __init__(self, scale_factor=None, mode=None):
        super().__init__()
        if isinstance(scale_factor, (tuple, list)):
            self.scale_h = int(scale_factor[0])
            self.scale_w = int(scale_factor[1])
        else:
            self.scale_h = int(scale_factor)
            self.scale_w = int(scale_factor)

    def forward(self, x):
        return x.repeat_interleave(self.scale_h, dim=-2).repeat_interleave(self.scale_w, dim=-1)


class WanResample(nn.Module):
    r"""
    A custom resampling module for 2D and 3D data.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'none': No resampling (identity operation).
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
            - 'downsample2d': 2D downsampling with zero-padding and convolution.
            - 'downsample3d': 3D downsampling with zero-padding, convolution, and causal 3D convolution.

    Neuron changes:
      - Added first_chunk param to forward(). For upsample3d mode, when
        first_chunk=True, skips time_conv and leaves zero cache untouched.
        This replaces diffusers' None->"Rep"->tensor state machine which
        is incompatible with torch.compile (no None/string cache values).
      - Replaced "feat_cache[idx] != 'Rep'" check with "not first_chunk".
      - Removed .to(cache_x.device) calls.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # default to dim //2
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                NkiConv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                NkiConv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode in ("downsample2d", "downsample3d"):
            self.resample = nn.Sequential(
                # Keep the checkpoint's convolution at resample.1.
                nn.Identity(),
                NkiConv2d(dim, dim, 3, stride=(2, 2), pad_right_bottom=True),
            )
            if mode == "downsample3d":
                self.time_conv = WanCausalConv3d(
                    dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
                )

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if first_chunk:
                    # Neuron change 4: skip time_conv on first frame to match diffusers'
                    # None→"Rep" path. The zero-initialized cache is left untouched so
                    # that frame 1 sees time_conv(x, zeros) ≡ time_conv(x) (both produce
                    # [0,0,x] as effective conv input via causal zero-padding).
                    feat_idx[0] += 1
                else:
                    cache_x = _tail_cache(x)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and not first_chunk:
                        # prepend the last cached frame, so the cache keeps two frames
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                        # Replaces diffusers' "feat_cache[idx] != 'Rep'" check.
                        # With zero-init cache, first_chunk=False after frame 0 means
                        # the cache holds real data from the previous frame.
                        cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                    # Replaces diffusers' "feat_cache[idx] == 'Rep'" check.
                    if feat_cache[idx] is None:
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if first_chunk:
                    # Neuron change: on the first temporal chunk diffusers stores the
                    # full spatially-downsampled x as the cache and returns x unchanged
                    # (no time_conv). Later chunks only ever read the last frame
                    # (feat_cache[idx][:, :, -1:, :, :]), so store just that — a fixed
                    # [B, C, 1, H', W'] tensor — to keep the cache compile-stable.
                    # Replaces diffusers' "feat_cache[idx] is None" sentinel.
                    feat_cache[idx] = x[:, :, -1:, :, :].clone(
                        memory_format=torch.contiguous_format
                    )
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone(memory_format=torch.contiguous_format)
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class WanResidualBlock(nn.Module):
    r"""
    A custom residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        dropout (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".

    Neuron changes:
      - Removed .to(cache_x.device) calls.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        # layers
        self.norm1 = NeuronWanRMSNorm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = NeuronWanRMSNorm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)

            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)

        # Dropout
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)

            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        # Add residual connection
        return x + h


class WanAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = dim**-0.5
        self.norm = NeuronWanRMSNorm(dim)
        # stay nn.Conv2d: no measured saving in the shipped context,
        # i.e. no benefit. Only ~0.7 GMAC combined -- too little work to amortise
        # a compiler<->NKI graph boundary, even though both sites flank the NKI
        # wan_vae::attention_cte kernel.
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    # Set by wan_spatial_shard.install_spatial_shard for spatially-sharded decode.
    _spatial_shard = None

    def forward(self, x):
        if self._spatial_shard is not None:
            # Global over the frame: gather the full frame, attend, keep this block.
            ctx = self._spatial_shard
            return ctx.select_block(self._forward_full(ctx.gather_full(x)))
        return self._forward_full(x)

    def _forward_full(self, x):
        identity = x

        B, C, T, H, W = x.size()
        S = H * W

        # Diffusers Wan VAE attention layout:
        # apply attention independently per frame over spatial tokens.
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.norm(x)

        qkv = self.to_qkv(x)

        # Single-head layout:
        # q/k/v: (B*T, S, C)
        qkv = qkv.reshape(B * T, 3, C, S)
        qkv = qkv.permute(1, 0, 3, 2).contiguous()
        q, k, v = qkv.unbind(0)

        if x.device.type == "neuron" and not _use_vae_nki_attention():
            # Compiler attention: explicit fp32 scores/softmax (not F.sdpa, which has a
            # known NaN bug at non-power-of-2 sequence lengths on this backend).
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
            out = torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)
        elif x.device.type == "neuron":
            # Keep scale based on the real Wan attention dim C.
            q = (q * self.scale).contiguous()
            k = k.contiguous()
            v = v.contiguous()

            # Existing attention_cte compile workaround:
            # full/no_tiling hits q/k/v = (1, 6240, 384), which fails in the
            # d=384 flash-section path. Padding D to 512 preserves QK math because
            # appended Q/K dims are zero, and the extra V/output dims are sliced away.
            # TODO: Follow up with attention_cte kernel-side fix for long-sequence d=384
            # so this padding workaround can be removed.
            D_orig = q.shape[-1]
            padded_d = False

            if _should_pad_vae_attn_head_dim(q.shape[1], D_orig):
                pad_d = _VAE_ATTN_PAD_HEAD_DIM - D_orig
                q = F.pad(q, (0, pad_d)).contiguous()
                k = F.pad(k, (0, pad_d)).contiguous()
                v = F.pad(v, (0, pad_d)).contiguous()
                padded_d = True

            out = vae_nki_attn(q, k, v)

            if padded_d:
                out = out[..., :D_orig]
        else:
            out = F.scaled_dot_product_attention(
                q.unsqueeze(1),
                k.unsqueeze(1),
                v.unsqueeze(1),
            ).squeeze(1)

        x = out.permute(0, 2, 1).reshape(B * T, C, H, W)
        x = self.proj(x)
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        return x + identity


class WanMidBlock(nn.Module):
    """
    Middle block for WanVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1
    ):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [WanResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanAttentionBlock(dim))
            resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        # First residual block
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
                # TODO: remove clone() after torch-native fix for operator fusion precision loss in attention→resnet path
                # Workaround: prevent compiler from fusing attention output
                # directly into the next resnet. Without this barrier, the
                # torch-native backend's operator fusion keeps the attention
                # output in an internal buffer at reduced precision, causing
                # 2-5x error amplification through the subsequent RMS norm.
                x = x.clone()

            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)

        return x


class WanResidualDownBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, dropout, num_res_blocks, temperal_downsample=False, down_flag=False
    ):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(WanResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.downsampler is not None:
            # Neuron change: thread first_chunk to WanResample so downsample3d can
            # use the explicit first-chunk cache path (replaces None sentinel).
            x = self.downsampler(
                x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
            )

        return x + self.avg_shortcut(x_copy)


class WanEncoder3d(nn.Module):
    r"""
    A 3D encoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_downsample (list of bool): Whether to downsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.

    Neuron changes:
      - Removed .to(cache_x.device) calls.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,  # the Wan 2.2 VAE uses a residual down block
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if is_residual:
                self.down_blocks.append(
                    WanResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=temperal_downsample[i]
                        if i != len(dim_mult) - 1
                        else False,
                        down_flag=i != len(dim_mult) - 1,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(WanResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        self.down_blocks.append(WanAttentionBlock(out_dim))
                    in_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(WanResample(out_dim, mode=mode))
                    scale /= 2.0

        # middle blocks
        self.mid_block = WanMidBlock(out_dim, dropout, non_linearity, num_layers=1)

        # output blocks
        self.norm_out = NeuronWanRMSNorm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # prepend the last cached frame, so the cache keeps two frames
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## downsamples
        # first_chunk drives the temporal-downsample cache path, so it is threaded
        # only to layers that accept it: WanResidualDownBlock (wan2.2 residual path)
        # and WanResample (wan2.1 non-residual downsample3d). Plain WanResidualBlock
        # / WanAttentionBlock layers are frame-local and take no first_chunk arg.
        for layer in self.down_blocks:
            if feat_cache is not None:
                if isinstance(layer, (WanResidualDownBlock, WanResample)):
                    x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
                else:
                    x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        ## middle
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # prepend the last cached frame, so the cache keeps two frames
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)

        return x


class WanResidualUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        temperal_upsample (bool): Whether to upsample on temporal dimension
        up_flag (bool): Whether to upsample or not
        non_linearity (str): Type of non-linearity to use

    Neuron changes:
      - Threads first_chunk to self.upsampler() so WanResample can skip
        time_conv on the first frame. Diffusers does not pass first_chunk
        to the upsampler (relies on None/"Rep" cache state instead).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        # create residual blocks
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)
        else:
            self.upsampler = None

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management

        Returns:
            torch.Tensor: Output tensor
        """
        x_copy = x.clone()

        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = resnet(x)

        if self.upsampler is not None:
            if feat_cache is not None:
                # Neuron change 4: thread first_chunk to WanResample so it can
                # skip time_conv on the first frame (replaces diffusers' None/"Rep" state)
                x = self.upsampler(
                    x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
                )
            else:
                x = self.upsampler(x)

        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)

        return x


class WanUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        upsample_mode (str, optional): Mode for upsampling ('upsample2d' or 'upsample3d')
        non_linearity (str): Type of non-linearity to use

    Neuron changes:
      - Threads first_chunk to self.upsamplers[0]() so WanResample can skip
        time_conv on the first frame.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([WanResample(out_dim, mode=upsample_mode)])

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=None):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management

        Returns:
            torch.Tensor: Output tensor
        """
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = resnet(x)

        if self.upsamplers is not None:
            if feat_cache is not None:
                # Neuron change 4: thread first_chunk to WanResample
                x = self.upsamplers[0](
                    x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
                )
            else:
                x = self.upsamplers[0](x)
        return x


class WanDecoder3d(nn.Module):
    r"""
    A 3D decoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_upsample (list of bool): Whether to upsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.

    Neuron changes:
      - Removed .to(cache_x.device) calls.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        non_linearity: str = "silu",
        out_channels: int = 3,
        is_residual: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.mid_block = WanMidBlock(dims[0], dropout, non_linearity, num_layers=1)

        # upsample blocks
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0 and not is_residual:
                # the Wan 2.1 VAE
                in_dim = in_dim // 2

            # check whether this block upsamples
            up_flag = i != len(dim_mult) - 1
            # select the upsampling mode (None when the block does not upsample)
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
            # Create and add the upsampling block
            if is_residual:
                up_block = WanResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    non_linearity=non_linearity,
                )
            else:
                up_block = WanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            self.up_blocks.append(up_block)

        # output blocks
        self.norm_out = NeuronWanRMSNorm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # prepend the last cached frame, so the cache keeps two frames
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## middle
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        ## upsamples
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # prepend the last cached frame, so the cache keeps two frames
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)

        return x


def _install_nki_conv_dispatch(modules: Iterable[nn.Module], target_device=None) -> int:
    """Pack filters for the selected NKI-capable modules.

    Call after weights load and before ``torch.compile`` traces the model.
    Returns the number of sites packed.
    """
    packed = 0
    for module in modules:
        if hasattr(module, "install_nki_dispatch"):
            packed += module.install_nki_dispatch(target_device)
    return packed


class NeuronWanDecoder3d(nn.Module):
    """Wrapper that runs post_quant_conv + decoder in a single compiled graph.

    Flattens feat_cache into *args for torch.compile compatibility (compiled
    graphs cannot mutate Python list inputs). Returns updated cache as
    explicit outputs.
    """

    def __init__(self, post_quant_conv, decoder):
        super().__init__()
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder

    def install_nki_conv_dispatch(self, target_device=None) -> int:
        """Pack filters for every NKI-capable submodule. Returns the number packed."""
        return _install_nki_conv_dispatch(self.modules(), target_device)

    def forward(self, z_frame, *flat_cache, first_chunk):
        x = self.post_quant_conv(z_frame)
        feat_cache = list(flat_cache)
        feat_idx = [0]  # Explicit reset — do not rely on a mutable default
        out = self.decoder(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
        return out, *feat_cache


class NeuronWanEncoder3d(nn.Module):
    """Wrapper that runs encoder + quant_conv in a single compiled graph.

    Encoder analogue of :class:`NeuronWanDecoder3d`. Flattens the encoder
    feat_cache into ``*args`` for torch.compile (compiled graphs cannot mutate
    Python list inputs) and returns the updated cache as explicit outputs.

    ``quant_conv`` is a 1x1x1 WanCausalConv3d (no temporal receptive field), so
    applying it per temporal chunk and concatenating the results is identical to
    applying it once over the whole latent — the same per-chunk factoring the
    decoder wrapper uses for ``post_quant_conv``.

    The encoder is called on two input temporal sizes (chunk 0 has T_in=1 with
    first_chunk=True; later chunks have T_in=4 with first_chunk=False), so
    torch.compile produces two static graphs per tile shape sharing one fixed-shape
    cache set. That takes ``dynamic=False`` (``NeuronAutoencoderKLWan.compile`` passes it;
    otherwise the second chunk is retraced with a symbolic frame dim, which the neuron
    backend rejects) and contiguous cache outputs (``_tail_cache``; otherwise the third
    chunk recompiles on the cache strides).
    """

    def __init__(self, encoder, quant_conv):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv

    def install_nki_conv_dispatch(self, target_device=None) -> int:
        """Pack the encoder's NKI-eligible spatial downsample filters."""
        modules = (module for module in self.modules() if isinstance(module, NkiConv2d))
        return _install_nki_conv_dispatch(modules, target_device)

    def forward(self, x_chunk, *flat_cache, first_chunk):
        feat_cache = list(flat_cache)
        feat_idx = [0]  # Explicit reset — do not rely on a mutable default
        out = self.encoder(
            x_chunk, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
        )
        out = self.quant_conv(out)
        return out, *feat_cache


@dataclass(frozen=True)
class DecoderTileLayout:
    """The Wan VAE decoder's tile grid for one latent shape (``_prepare_decoder_tiles``)."""

    sample_height: int
    sample_width: int
    tile_latent_height: int
    tile_latent_width: int
    tile_latent_stride_height: int
    tile_latent_stride_width: int
    tile_sample_stride_height: int
    tile_sample_stride_width: int
    output_scale: int
    blend_height: int
    blend_width: int
    row_geometry: tuple[tuple[int, int, int, int], ...]
    column_geometry: tuple[tuple[int, int, int, int], ...]


class NeuronAutoencoderKLWan(AutoencoderKLWan):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Introduced in [Wan 2.1].

    This model inherits from [`ModelMixin`]. Check the superclass documentation for its generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = False
    _group_offload_block_modules = ["quant_conv", "post_quant_conv", "encoder", "decoder"]
    # keys to ignore when AlignDeviceHook moves inputs/outputs between devices
    # these are shared mutable state modified in-place
    _skip_keys = ["feat_cache", "feat_idx"]

    @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        decoder_base_dim: int | None = None,
        z_dim: int = 16,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list[float] = [],
        temperal_downsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: list[float] = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        latents_std: list[float] = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
        is_residual: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        patch_size: int | None = None,
        scale_factor_temporal: int | None = 4,
        scale_factor_spatial: int | None = 8,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        if decoder_base_dim is None:
            decoder_base_dim = base_dim

        self.encoder = WanEncoder3d(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            is_residual=is_residual,
        )
        self.quant_conv = WanCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = WanCausalConv3d(z_dim, z_dim, 1)

        self.decoder = WanDecoder3d(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=self.temperal_upsample,
            dropout=dropout,
            out_channels=out_channels,
            is_residual=is_residual,
        )

        self.spatial_compression_ratio = scale_factor_spatial

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        # NOTE: the pipeline sets this (build_wan_vae(use_tiling=True)), and patch parallelism
        # needs it on.
        self.use_tiling = False

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        # Precompute and cache conv counts for encoder and decoder so clear_cache does not recount them
        self._cached_conv_counts = {
            "decoder": sum(isinstance(m, WanCausalConv3d) for m in self.decoder.modules())
            if self.decoder is not None
            else 0,
            "encoder": sum(isinstance(m, WanCausalConv3d) for m in self.encoder.modules())
            if self.encoder is not None
            else 0,
        }

    def _init_feat_cache(self, z):
        """
        Pre-allocate feat_cache as zero tensors matching the decoder's data flow.
        Replaces diffusers' [None]*N initialization. Zero tensors are
        numerically equivalent to None for WanCausalConv3d (both produce
        the same causal zero-padding), and are required for Neuron
        compilation which needs fixed tensor shapes (no None/string sentinels).
        """
        device, dtype, B = z.device, z.dtype, z.shape[0]
        H, W = z.shape[3], z.shape[4]
        cache = []

        def _add_conv_cache(module, h, w):
            """Add a zero cache tensor for a WanCausalConv3d."""
            cache.append(
                torch.zeros(B, module.in_channels, CACHE_T, h, w, device=device, dtype=dtype)
            )

        def _process_resblock(block, h, w):
            # conv1, conv2, and possibly conv_shortcut
            # conv_shortcut is 1x1 (no temporal padding) but still needs a cache slot
            _add_conv_cache(block.conv1, h, w)
            _add_conv_cache(block.conv2, h, w)

        def _process_resample(resample, h, w):
            if resample.mode == "upsample3d":
                # time_conv runs at pre-upsample resolution
                _add_conv_cache(resample.time_conv, h, w)
                h *= 2
                w *= 2
            elif resample.mode == "upsample2d":
                h *= 2
                w *= 2
            return h, w

        # conv_in
        _add_conv_cache(self.decoder.conv_in, H, W)

        # mid_block: resnet0, (attn + resnet) pairs
        for resnet in self.decoder.mid_block.resnets:
            _process_resblock(resnet, H, W)

        # up_blocks
        for up_block in self.decoder.up_blocks:
            if isinstance(up_block, WanResidualUpBlock):
                for resnet in up_block.resnets:
                    _process_resblock(resnet, H, W)
                if up_block.upsampler is not None:
                    H, W = _process_resample(up_block.upsampler, H, W)
            elif isinstance(up_block, WanUpBlock):
                for resnet in up_block.resnets:
                    _process_resblock(resnet, H, W)
                if up_block.upsamplers is not None:
                    H, W = _process_resample(up_block.upsamplers[0], H, W)

        # conv_out
        _add_conv_cache(self.decoder.conv_out, H, W)

        return cache

    def _prepare_decoder_tiles(self, z: torch.Tensor) -> DecoderTileLayout:
        height, width = z.shape[-2:]
        patch_size = self.config.patch_size
        if patch_size is None:
            patch_size = 1
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")

        scaled_values = {
            "spatial_compression_ratio": self.spatial_compression_ratio,
            "tile_sample_min_height": self.tile_sample_min_height,
            "tile_sample_min_width": self.tile_sample_min_width,
            "tile_sample_stride_height": self.tile_sample_stride_height,
            "tile_sample_stride_width": self.tile_sample_stride_width,
        }
        for name, value in scaled_values.items():
            if value % patch_size:
                raise ValueError(f"{name}={value} must be divisible by patch_size={patch_size}")

        tile_latent_height = min(
            height, self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_width = min(width, self.tile_sample_min_width // self.spatial_compression_ratio)
        tile_latent_stride_height = min(
            tile_latent_height,
            self.tile_sample_stride_height // self.spatial_compression_ratio,
        )
        tile_latent_stride_width = min(
            tile_latent_width,
            self.tile_sample_stride_width // self.spatial_compression_ratio,
        )
        output_scale = self.spatial_compression_ratio // patch_size
        tile_sample_height = tile_latent_height * output_scale
        tile_sample_width = tile_latent_width * output_scale
        tile_sample_stride_height = tile_latent_stride_height * output_scale
        tile_sample_stride_width = tile_latent_stride_width * output_scale

        return DecoderTileLayout(
            sample_height=height * output_scale,
            sample_width=width * output_scale,
            tile_latent_height=tile_latent_height,
            tile_latent_width=tile_latent_width,
            tile_latent_stride_height=tile_latent_stride_height,
            tile_latent_stride_width=tile_latent_stride_width,
            tile_sample_stride_height=tile_sample_stride_height,
            tile_sample_stride_width=tile_sample_stride_width,
            output_scale=output_scale,
            blend_height=tile_sample_height - tile_sample_stride_height,
            blend_width=tile_sample_width - tile_sample_stride_width,
            row_geometry=decoder_tile_geometry(
                height, tile_latent_height, tile_latent_stride_height
            ),
            column_geometry=decoder_tile_geometry(
                width, tile_latent_width, tile_latent_stride_width
            ),
        )

    def _init_enc_feat_cache(self, x):
        """Pre-allocate the ENCODER feat_cache as fixed-shape zero tensors.

        Encoder analogue of :meth:`_init_feat_cache`. Walks conv_in -> down_blocks
        -> mid_block -> conv_out in ``feat_idx`` order, so entry N is the cache for
        the Nth WanCausalConv3d slot the encoder consumes.

        Standard WanCausalConv3d slots get a ``[B, in_ch, CACHE_T, h, w]`` zero
        cache — numerically equivalent to diffusers' ``None`` (both yield causal
        zero-padding), and required for Neuron compilation (no None sentinels).
        The ``downsample3d`` time_conv is special: its reworked forward stores only
        the last post-spatial-downsample frame, so it gets a single-frame
        ``[B, dim, 1, h//2, w//2]`` slot.

        Spatial dims SHRINK by 2 at each spatial/temporal downsample — the
        mirror image of the decoder's ×2 growth.
        """
        device, dtype, B = x.device, x.dtype, x.shape[0]
        H, W = x.shape[3], x.shape[4]
        cache = []

        def _add_conv_cache(module, h, w):
            """Add a CACHE_T-frame zero cache for a standard WanCausalConv3d."""
            cache.append(
                torch.zeros(B, module.in_channels, CACHE_T, h, w, device=device, dtype=dtype)
            )

        def _process_resblock(block, h, w):
            _add_conv_cache(block.conv1, h, w)
            _add_conv_cache(block.conv2, h, w)

        def _process_downsampler(resample, h, w):
            """Add the resample's cache slot (if any) and return the shrunk (h, w).

            Both downsample modes halve the spatial dims. Only downsample3d consumes
            a feat_cache slot and keeps a single post-downsample frame.
            """
            if resample.mode == "downsample3d":
                h //= 2
                w //= 2
                cache.append(
                    torch.zeros(
                        B, resample.time_conv.in_channels, 1, h, w, device=device, dtype=dtype
                    )
                )
            elif resample.mode == "downsample2d":
                h //= 2
                w //= 2
            return h, w

        # conv_in
        _add_conv_cache(self.encoder.conv_in, H, W)

        # down_blocks
        for block in self.encoder.down_blocks:
            if isinstance(block, WanResidualDownBlock):
                for resnet in block.resnets:
                    _process_resblock(resnet, H, W)
                if block.downsampler is not None:
                    H, W = _process_downsampler(block.downsampler, H, W)
            elif isinstance(block, WanResidualBlock):
                _process_resblock(block, H, W)
            elif isinstance(block, WanResample):
                H, W = _process_downsampler(block, H, W)
            # WanAttentionBlock has no WanCausalConv3d -> no cache slot.

        # mid_block: resnet0, then (attn + resnet) pairs (attn adds no conv slot)
        for resnet in self.encoder.mid_block.resnets:
            _process_resblock(resnet, H, W)

        # conv_out
        _add_conv_cache(self.encoder.conv_out, H, W)

        return cache

    def prepare_nki_conv_dispatch(self, target_device) -> int:
        """Pack decoder NKI filters before the model moves to its Neuron device."""
        decoder_wrapper = NeuronWanDecoder3d(self.post_quant_conv, self.decoder)
        return decoder_wrapper.install_nki_conv_dispatch(target_device)

    def compile(self, *args, compile_encoder: bool = False, **compiler_kwargs):
        """Compile the VAE decoder (always) and optionally the encoder.

        The encoder is only needed by image-conditioned pipelines (e.g. I2V,
        which VAE-encodes the conditioning frame). ``compile_encoder`` is off by
        default so the text-to-video path pays no encoder compilation cost.

        Call it after the VAE is on its Neuron device: the NKI conv filters are packed from the
        device weights. Defaults, each overridable: ``backend`` the plugin's torch-neuronx
        backend, ``fullgraph=True``, ``options["model_name"]="wan_vae"`` and
        ``options["compiler_args"]`` = ``WAN_I2V_VAE_COMPILER_ARGS`` with ``compile_encoder``
        (the I2V pipeline's flags), else ``WAN_VAE_COMPILER_ARGS``.
        """
        if "backend" not in compiler_kwargs:
            compiler_kwargs["backend"] = compile_backend.register()
        compiler_kwargs.setdefault("fullgraph", True)
        # torch-neuronx compiles static shapes only: its neuron backend rejects a graph with a
        # symbolic dim. Without this, the encoder's second chunk (4 frames after 1) fails the
        # first chunk's guards and Dynamo's automatic dynamic shapes retrace it with a
        # symbolic frame dim.
        compiler_kwargs.setdefault("dynamic", False)
        base_options = dict(compiler_kwargs.get("options") or {})
        base_options.setdefault("model_name", "wan_vae")
        base_options.setdefault(
            "compiler_args",
            list(WAN_I2V_VAE_COMPILER_ARGS if compile_encoder else WAN_VAE_COMPILER_ARGS),
        )
        compiler_kwargs["options"] = base_options

        decoder_wrapper = NeuronWanDecoder3d(self.post_quant_conv, self.decoder)
        # Pack before tracing so the permute is a graph constant, not a per-call op.
        n_packed = decoder_wrapper.install_nki_conv_dispatch()
        logger.info("packed filters for %d NKI-eligible decoder conv site(s)", n_packed)

        base_model_name = base_options["model_name"]

        first_compiler_kwargs = {
            **compiler_kwargs,
            "options": {
                **base_options,
                "model_name": f"{base_model_name}_first",
            },
        }
        rest_compiler_kwargs = {
            **compiler_kwargs,
            "options": {
                **base_options,
                "model_name": f"{base_model_name}_rest",
            },
        }

        self._compiled_decoder_first = torch.compile(
            decoder_wrapper, *args, **first_compiler_kwargs
        )
        self._compiled_decoder_rest = torch.compile(decoder_wrapper, *args, **rest_compiler_kwargs)

        if compile_encoder:
            encoder_wrapper = NeuronWanEncoder3d(self.encoder, self.quant_conv)
            n_encoder_packed = encoder_wrapper.install_nki_conv_dispatch()
            logger.info(
                "packed filters for %d NKI-eligible encoder downsample site(s)",
                n_encoder_packed,
            )
            encoder_compiler_kwargs = dict(compiler_kwargs)
            if self.use_tiling:
                # Encoder edge tiles remain variable-size.
                encoder_compiler_kwargs["fullgraph"] = False
            self._compiled_encoder = torch.compile(
                encoder_wrapper, *args, **encoder_compiler_kwargs
            )

    def _decode(self, z, return_dict=True):
        """Decode latent frames one at a time through the compiled decoder.
        - z is sliced directly on device (no CPU round-trip for input)
        - feat_map stays on Neuron device throughout the loop
        - Only the output frame is transferred to CPU each iteration
        """
        _, _, num_frame, height, width = z.shape

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        feat_map = self._init_feat_cache(z)

        frames_out = []
        frame_indices = [torch.tensor([i], device=z.device) for i in range(num_frame)]
        for i in range(num_frame):
            frame = torch.index_select(z, 2, frame_indices[i])
            compiled_decoder = (
                self._compiled_decoder_first if i == 0 else self._compiled_decoder_rest
            )
            result = compiled_decoder(frame, *feat_map, first_chunk=(i == 0))
            out_frame = result[0]
            feat_map = list(result[1:])
            frames_out.append(out_frame.cpu())

        out = torch.cat(frames_out, dim=2)

        if self.config.patch_size is not None:
            out = unpatchify(out, patch_size=self.config.patch_size)
        out = torch.clamp(out, min=-1.0, max=1.0)

        if not return_dict:
            return (out,)
        return DecoderOutput(sample=out)

    def tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> DecoderOutput | torch.Tensor:
        r"""
        Decode a batch of images using a tiled decoder.

        When the spatial dimensions of the latent tensor exceed the configured
        tile thresholds (tile_sample_min_height/width ÷ spatial_compression_ratio),
        this method splits the latent into overlapping spatial tiles, decodes each
        tile frame-by-frame through the compiled Neuron decoder, and blends the
        results back together using linear interpolation in the overlap regions.

        Neuron-specific adaptations:
        - Uses torch.index_select instead of Python slicing (z[:, :, :, i:j, k:l])
            to avoid data-dependent slice bounds which may not lower correctly in
            compiled Neuron graphs.
        - Uses full-size boundary windows and crops their decoded outputs.
        - Uses fixed full-tile feat_cache shapes.
        - Output frames are moved to CPU immediately to avoid accumulating device
            memory across tiles and frames.

        Algorithm:
        1. Compute tile/stride/blend sizes in both latent and pixel space.
        2. Shift partial boundary windows inward to keep them full-size.
        3. For each tile position, extract the latent tile and decode all temporal
            frames sequentially (maintaining causal conv cache across frames).
        4. After all tiles are decoded, blend overlapping regions:
            - Vertical blending (blend_v): linear ramp along height overlap.
            - Horizontal blending (blend_h): linear ramp along width overlap.
        5. Crop boundary outputs, blend overlaps, and assemble the result.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors with shape
                [B, C, T, H, W] where H and W are in latent space.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        num_frames = z.shape[2]
        layout = self._prepare_decoder_tiles(z)
        output_heights = tuple(
            layout.tile_sample_stride_height
            if row < len(layout.row_geometry) - 1
            else geometry[3] * layout.output_scale
            for row, geometry in enumerate(layout.row_geometry)
        )
        output_widths = tuple(
            layout.tile_sample_stride_width
            if column < len(layout.column_geometry) - 1
            else geometry[3] * layout.output_scale
            for column, geometry in enumerate(layout.column_geometry)
        )

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for _, input_row, crop_row, valid_height in layout.row_geometry:
            row = []
            for _, input_column, crop_column, valid_width in layout.column_geometry:
                h_indices = torch.arange(
                    input_row, input_row + layout.tile_latent_height, device=z.device
                )
                w_indices = torch.arange(
                    input_column, input_column + layout.tile_latent_width, device=z.device
                )
                tile_z = torch.index_select(z, 3, h_indices)
                tile_z = torch.index_select(tile_z, 4, w_indices)
                feat_map = self._init_feat_cache(tile_z)

                time = []
                frame_indices = [torch.tensor([k], device=z.device) for k in range(num_frames)]
                for k in range(num_frames):
                    frame = torch.index_select(tile_z, 2, frame_indices[k])
                    compiled_decoder = (
                        self._compiled_decoder_first if k == 0 else self._compiled_decoder_rest
                    )
                    result = compiled_decoder(frame, *feat_map, first_chunk=(k == 0))
                    out_frame = result[0]
                    feat_map = list(result[1:])
                    time.append(out_frame.cpu())
                tile = torch.cat(time, dim=2)
                crop_top = crop_row * layout.output_scale
                crop_left = crop_column * layout.output_scale
                crop_bottom = crop_top + valid_height * layout.output_scale
                crop_right = crop_left + valid_width * layout.output_scale
                row.append(tile[..., crop_top:crop_bottom, crop_left:crop_right])
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, layout.blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, layout.blend_width)
                result_row.append(tile[..., : output_heights[i], : output_widths[j]])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=3)[..., : layout.sample_height, : layout.sample_width]

        if self.config.patch_size is not None:
            dec = unpatchify(dec, patch_size=self.config.patch_size)

        dec = torch.clamp(dec, min=-1.0, max=1.0)

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z, return_dict=True):
        self._reject_while_sharded("decoder")
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    # ------------------------------------------------------------------
    # Spatial sharding (wan_spatial_shard.py): the sharded streaming decode and encode run
    # their own graphs over this VAE's modules; its unsharded entry points refuse to run then.
    # ------------------------------------------------------------------
    def _reject_while_sharded(self, side: str) -> None:
        """Refuse an unsharded decode / encode while a spatial shard is installed on ``side``.

        Args:
            side: ``"decoder"`` or ``"encoder"``.

        Raises:
            RuntimeError: ``install_spatial_shard`` has patched that module, so its layers
                expect a rank's block and would exchange halos instead of decoding (encoding)
                the full frame.
        """
        if getattr(getattr(self, side), "_spatial_shard_installed", False):
            wrapper = (
                "SpatialShardedWanDecoder" if side == "decoder" else "SpatialShardedWanEncoder"
            )
            call = "decode_chunk" if side == "decoder" else "encode_chunk"
            raise RuntimeError(
                f"this VAE's {side} is spatially sharded by a {wrapper}; run it through the "
                f"wrapper's {call}"
            )

    # ------------------------------------------------------------------
    # Encoder (image/video -> latent), compiled analogue of the decoder.
    # Mirrors diffusers' AutoencoderKLWan._encode / tiled_encode but routes
    # every temporal chunk through the compiled NeuronWanEncoder3d graph with a
    # fixed-shape feat_cache (see _init_enc_feat_cache), instead of diffusers'
    # None/"Rep" mutable-state machine which torch.compile cannot trace.
    # ------------------------------------------------------------------
    def _encoder_module(self):
        """Return the compiled encoder or an uncompiled wrapper of the same modules."""
        compiled = getattr(self, "_compiled_encoder", None)
        if compiled is not None:
            return compiled
        if getattr(self, "_eager_encoder", None) is None:
            self._eager_encoder = NeuronWanEncoder3d(self.encoder, self.quant_conv)
        return self._eager_encoder

    def _encode(self, x):
        """Encode pixel frames to latent chunks through the Neuron encoder graph.

        Diffusers encodes in temporal chunks: chunk 0 is 1 pixel frame, each
        later chunk is 4 pixel frames, and every chunk emits 1 latent frame. The
        two distinct input temporal sizes (T_in=1, T_in=4) produce two compiled
        specializations that share one fixed-shape feat_cache set.
        """
        _, _, num_frame, height, width = x.shape

        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)

        tile_min_height = self.tile_sample_min_height
        tile_min_width = self.tile_sample_min_width
        if self.use_tiling and (width > tile_min_width or height > tile_min_height):
            return self.tiled_encode(x)

        encoder = self._encoder_module()
        feat_map = self._init_enc_feat_cache(x)

        iter_ = 1 + (num_frame - 1) // 4
        enc_chunks = []
        for i in range(iter_):
            if i == 0:
                idx = torch.arange(0, 1, device=x.device)
            else:
                idx = torch.arange(1 + 4 * (i - 1), 1 + 4 * i, device=x.device)
            chunk = torch.index_select(x, 2, idx)
            result = encoder(chunk, *feat_map, first_chunk=(i == 0))
            enc_chunks.append(result[0])
            feat_map = list(result[1:])

        enc = torch.cat(enc_chunks, dim=2)
        return enc

    @apply_forward_hook
    def encode(self, x, return_dict=True):
        """Encode a batch of images/videos into latents.

        Returns the same AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution)
        shape as diffusers, so callers (e.g. I2V ``retrieve_latents(vae.encode(...),
        "argmax")``) work unchanged.
        """
        self._reject_while_sharded("encoder")
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @staticmethod
    def _validate_i2v_condition_inputs(
        image: torch.Tensor,
        *,
        num_frames: int,
        last_image: torch.Tensor | None,
    ) -> None:
        if image.ndim != 5 or image.shape[2] != 1:
            raise ValueError("image must have shape [B, C, 1, H, W]")
        if last_image is not None and last_image.shape != image.shape:
            raise ValueError("last_image must match image shape")
        if num_frames < 1 or (num_frames - 1) % 4:
            raise ValueError("num_frames must be 1 + a multiple of 4")
        if last_image is not None and num_frames == 1:
            raise ValueError("last_image requires num_frames to be at least 5")

    def encode_i2v_condition(
        self,
        image: torch.Tensor,
        *,
        num_frames: int,
        last_image: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        """Encode I2V endpoints through the general full-video fallback.

        The distributed subclass overrides this method to expand the implicit
        zero frames lazily when tiled patch parallelism is active.
        """
        self._validate_i2v_condition_inputs(
            image,
            num_frames=num_frames,
            last_image=last_image,
        )
        middle_frames = num_frames - (2 if last_image is not None else 1)
        frames = [image]
        if middle_frames:
            frames.append(
                image.new_zeros(
                    *image.shape[:2],
                    middle_frames,
                    *image.shape[3:],
                )
            )
        if last_image is not None:
            frames.append(last_image)
        video_condition = torch.cat(frames, dim=2) if len(frames) > 1 else image
        return self.encode(video_condition, return_dict=return_dict)

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Tiled spatial encode through the compiled encoder.

        Spatial analogue of :meth:`tiled_decode`: split the pixel input into
        overlapping tiles, encode each tile's temporal chunks through the
        compiled encoder with a per-tile fixed-shape feat_cache, then blend
        overlaps on CPU. Returns the encoded latent tensor (the ``h`` fed to the
        DiagonalGaussianDistribution), matching diffusers' ``tiled_encode``.
        """
        _, _, num_frames, height, width = x.shape

        # Latent-space compression ratio (patchify pre-shrinks by patch_size).
        encode_ratio = self.spatial_compression_ratio
        if self.config.patch_size is not None:
            encode_ratio = self.spatial_compression_ratio // self.config.patch_size

        latent_height = height // encode_ratio
        latent_width = width // encode_ratio

        tile_latent_min_height = self.tile_sample_min_height // encode_ratio
        tile_latent_min_width = self.tile_sample_min_width // encode_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // encode_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // encode_ratio
        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        iter_ = 1 + (num_frames - 1) // 4
        encoder = self._encoder_module()

        # Split x into overlapping pixel-space tiles and encode each separately.
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                hi = min(height, i + self.tile_sample_min_height)
                wj = min(width, j + self.tile_sample_min_width)
                h_indices = torch.arange(i, hi, device=x.device)
                w_indices = torch.arange(j, wj, device=x.device)
                tile_x = torch.index_select(x, 3, h_indices)
                tile_x = torch.index_select(tile_x, 4, w_indices)

                feat_map = self._init_enc_feat_cache(tile_x)
                time = []
                for k in range(iter_):
                    if k == 0:
                        f_idx = torch.arange(0, 1, device=x.device)
                    else:
                        f_idx = torch.arange(1 + 4 * (k - 1), 1 + 4 * k, device=x.device)
                    chunk = torch.index_select(tile_x, 2, f_idx)
                    result = encoder(chunk, *feat_map, first_chunk=(k == 0))
                    time.append(result[0].cpu())
                    feat_map = list(result[1:])
                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(
                    tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width]
                )
            result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc


@dataclass
class TileTask:
    """One tile of a patch-parallel decode or encode.

    ``tensor`` is the whole input the tile is cut from: ``exec`` slices its own window, so a rank
    holds only the tiles it runs. ``workload`` weighs the tile in :func:`balance_tile_tasks`.
    """

    tile_id: int
    grid_coord: tuple[int, ...]
    tensor: torch.Tensor
    workload: int | float = 1


@dataclass
class DistributedOperator:
    """The three steps of one patch-parallel VAE call (``_WanVaeExecutor.execute``).

    ``split`` lists the tiles and the grid; ``exec`` runs one tile on its rank; ``merge`` gets this
    rank's packed tiles and the gathered tile metadata, gathers and blends the payload, and returns
    the full output on group rank 0 (anything elsewhere).
    """

    split: Callable[[torch.Tensor], tuple[list[TileTask], GridSpec]]
    exec: Callable[[TileTask], torch.Tensor]
    merge: Callable[..., torch.Tensor | None]


def balance_tile_tasks(tasks: list[TileTask], num_ranks: int) -> list[list[TileTask]]:
    """Assign the tiles to ``num_ranks`` buckets: heaviest first, each to the least-loaded bucket
    (the lowest index on a tie)."""
    workloads = [0] * num_ranks
    assigned: list[list[TileTask]] = [[] for _ in range(num_ranks)]
    for task in sorted(tasks, key=lambda t: t.workload, reverse=True):
        rank = workloads.index(min(workloads))
        assigned[rank].append(task)
        workloads[rank] += task.workload
    return assigned


GATHER_BYTES_ENV = "SGLANG_NEURON_VAE_MAX_DEVICE_GATHER_BYTES"


def gather_bytes_from_env() -> int | None:
    """``SGLANG_NEURON_VAE_MAX_DEVICE_GATHER_BYTES`` as a byte count, or ``None`` when unset."""
    value = os.environ.get(GATHER_BYTES_ENV, "").strip()
    if not value:
        return None
    try:
        budget = int(value)
    except ValueError:
        budget = 0
    if budget <= 0:
        raise ValueError(f"{GATHER_BYTES_ENV}={value!r} must be a positive number of bytes")
    return budget


class _WanVaeExecutor(NeuronDistributedVaeExecutor):
    """The Wan VAE's executor: the patch-parallel tile split and balance on the shared Neuron
    executor, with 128 MiB plane-chunk gathers and one gather in flight per rank.

    ``execute`` runs a decode or an I2V encode: it splits the input into tiles, balances them over
    the group, runs this rank's tiles, pads and packs them (the shared executor's compiled
    ``_pack_local_tiles``), gathers only the small tile metadata up front and hands the packed
    tiles to the operator's ``merge``, which streams the payload gather and blend
    (``gather_and_blend_tiles``). No rank holds the full ``[world, slots, C, T, H, W]`` payload,
    which runs a rank out of memory at 720p.

    At 1080p I2V (VAE patch parallel 32, two decode tiles per rank) the DiT shards, VAE and
    NEFF scratchpads fill about 22.3 of a rank's 24 GB HBM, and the shared 512 MiB gather chunk
    fails to allocate. The allocation fails repeatedly, and a hard OOM then breaks the gather
    collective. So the chunk is 128 MiB here, and every rank has one gather in flight:
    ``_gather_chunk`` waits for each chunk's gather on the ranks other than 0, and rank 0 waits
    for it when it copies the blended chunk to the host. The I2V encode payload (at most 84 MiB)
    stays one chunk; the decode takes 4 / 8 / 16 chunks at 480p / 720p / 1080p.
    ``SGLANG_NEURON_VAE_MAX_DEVICE_GATHER_BYTES``, read when the executor is created, overrides
    the 128 MiB.
    FLUX.1-dev's executor keeps its 512 MiB.
    """

    MAX_DEVICE_GATHER_BYTES = 128 * 1024 * 1024

    def __init__(self):
        super().__init__()
        budget = gather_bytes_from_env()
        if budget is not None:
            self.MAX_DEVICE_GATHER_BYTES = budget

    def execute(
        self,
        z: torch.Tensor,
        operator: DistributedOperator,
        *,
        output_dtype: torch.dtype,
        broadcast_result: bool = True,
    ) -> torch.Tensor:
        """Tiled decode or encode of ``z`` over the group; every rank of the group calls it.

        Returns the merged ``output_dtype`` tensor on group rank 0 and an empty tensor elsewhere,
        or the merged tensor on every rank with ``broadcast_result``.
        """
        pp_size = min(self.parallel_size, self.world_size)

        tiletask_list, grid_spec = operator.split(z)
        tid_coord_map = {task.tile_id: task.grid_coord for task in tiletask_list}

        assigned = balance_tile_tasks(tiletask_list, pp_size)
        local_tasks = assigned[self.rank] if self.rank < pp_size else []
        local_results = [(t.tile_id, operator.exec(t)) for t in local_tasks]

        global_padding_shape = self._compute_global_padding_shape(local_results, z.ndim)
        local_tile_tensor, local_meta_tensor = self._pack_local_tiles(
            local_results, global_padding_shape, grid_spec, z.device, output_dtype
        )

        meta_gather = self.gather_tensors(local_meta_tensor)
        # The merge reads only the packed copy. Free the per-tile outputs (and any tensors the
        # tasks hold) before the payload gather allocates its buffers.
        del local_results, local_tasks, assigned, tiletask_list

        result = operator.merge(local_tile_tensor, meta_gather, grid_spec, tid_coord_map)
        if self.rank != 0:
            result = torch.empty(0, device=z.device)  # Dummy return for non-zero ranks.

        if broadcast_result:
            result = self._sync_final_result(result, z.ndim, z.device, output_dtype)
        return result

    def concat_tile_frames(self, frames):
        """Concatenate a tile's per-chunk decoder (or encoder) outputs along time, on device."""

        def concat(*frame_outputs):
            return torch.cat(frame_outputs, dim=2)

        return self._compile_device_graph("tile_concat", len(frames), concat)(*frames)

    def _raise_if_any_rank_failed(self, local_error, phase):
        """Agree over the Gloo error group on whether any rank failed ``phase``, and raise on every
        rank if one did, so no rank is left waiting in the next collective."""
        failed = torch.tensor([int(local_error is not None)], dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self._error_group)
        if failed.item():
            where = "" if local_error is not None else " on another rank"
            raise RuntimeError(f"Wan VAE {phase} failed{where}") from local_error

    def _compute_global_padding_shape(self, local_results, output_ndim: int) -> list[int]:
        """The slot shape every rank pads its tiles to: ``[slots, *dims]``, the most tiles any rank
        holds (at least one, so a rank without tiles still joins the gathers) and each dim's
        largest size, reduced over the group on the host (Gloo)."""
        error = stat = None
        try:
            local_max = [0] * output_ndim
            for _, tile in local_results:
                for dim, size in enumerate(tile.shape):
                    local_max[dim] = max(local_max[dim], size)
            stat = torch.tensor([len(local_results), *local_max], device="cpu")
        except Exception as exc:  # noqa: BLE001 - reported to every rank, then raised
            error = exc
        self._raise_if_any_rank_failed(error, "shape preparation")
        dist.all_reduce(stat, op=dist.ReduceOp.MAX, group=self.group)
        global_padding_shape = stat.tolist()
        global_padding_shape[0] = max(global_padding_shape[0], 1)
        return global_padding_shape

    def _sync_final_result(self, rank0_result, output_ndim, output_device, output_dtype):
        """Broadcast rank 0's result to every VAE rank through the host.

        Both the shape and the payload are CPU tensors, so they go over the group's Gloo
        backend; the device collectives of the executor all run inside compiled graphs (see
        ``_get_compiled_device_gather``). Only the Wan I2V encode broadcasts (decode passes
        ``broadcast_result=False``), so the payload is one conditioning latent and the host
        round-trip is cheap.

        The bytes move as ``uint8``: Gloo's dtype coverage does not include every float type the
        VAE emits, and a byte view is exact for all of them.
        """
        if self.rank == 0:
            shape_tensor = torch.tensor(tuple(rank0_result.shape), device="cpu", dtype=torch.int64)
        else:
            shape_tensor = torch.empty((output_ndim,), device="cpu", dtype=torch.int64)
        torch.distributed.broadcast(shape_tensor, src=0, group=self.group)

        shape = tuple(shape_tensor.tolist())
        if self.rank == 0:
            flat_bytes = rank0_result.to("cpu").contiguous().reshape(-1).view(torch.uint8)
        else:
            itemsize = torch.empty(0, dtype=output_dtype).element_size()
            flat_bytes = torch.empty(math.prod(shape) * itemsize, device="cpu", dtype=torch.uint8)
        torch.distributed.broadcast(flat_bytes, src=0, group=self.group)

        if self.rank == 0:
            return rank0_result
        return flat_bytes.view(output_dtype).reshape(shape).to(output_device)

    def _gather_chunk(self, compiled_gather, chunk, chunk_index):
        """The shared executor's chunk gather, except that a rank other than 0 waits for it.

        torch-neuronx queues the gather and returns. Rank 0 blends each chunk, which waits for it,
        but the other ranks have nothing to wait on, so they would queue every chunk's gather at
        once, and each queued gather holds an output buffer of the chunk budget until rank 0
        joins it. Copying one element of the output to the host waits for the gather.
        """

        def gather_and_wait(tensor):
            gathered = compiled_gather(tensor)
            if self.rank != 0:
                gathered.reshape(-1)[:1].to("cpu")
            return gathered

        return super()._gather_chunk(gather_and_wait, chunk, chunk_index)

    def _stream_gather_planes(self, local_planes, chunk_planes, planes):
        """The shared executor's plane-chunk gather loop, freeing each local chunk with its gather.

        With two or more tile slots a rank's chunk is a strided copy (``_plane_chunk``); the shared
        loop keeps it until the next chunk replaces it. Keep the rest in step with the shared loop.
        """
        if local_planes.device.type == "cpu":
            yield from super()._stream_gather_planes(local_planes, chunk_planes, planes)
            return

        setup_error = None
        try:
            compiled_gather = self._get_compiled_device_gather()
        except Exception as error:
            setup_error = error
        self._raise_if_device_gather_failed(setup_error, "setup")

        for chunk_index, start in enumerate(range(0, planes, chunk_planes)):
            size = min(chunk_planes, planes - start)
            local_chunk = self._plane_chunk(local_planes, start, size)
            chunk_shape = tuple(local_chunk.shape)
            gathered = self._gather_chunk(compiled_gather, local_chunk.reshape(-1), chunk_index)
            del local_chunk  # a strided chunk is a copy; free it with its gather
            if gathered is None:
                yield None
                continue
            chunk_views = list(gathered.reshape(self.world_size, *chunk_shape).unbind(0))
            del gathered  # drop the alias; chunk_views still pins the storage
            yield chunk_views
            del chunk_views  # on resume, free the buffer before the next gather


class DistributedAutoencoderKLWan(NeuronAutoencoderKLWan):
    r"""
    Patch-parallel Neuron Wan VAE.

    Adds the distributed-VAE executor (``_WanVaeExecutor``: the patch-parallel tile split and
    balance on the shared ``NeuronDistributedVaeExecutor`` of ``vae_executor``) on top
    of :class:`NeuronAutoencoderKLWan`. When patch parallelism is enabled and
    the latent is large enough to tile, the spatial tiles produced by
    ``tile_split`` are distributed across the VAE-parallel process group; each
    rank decodes its assigned tiles and the results are gathered and blended on
    that group's rank 0 by ``tile_merge``.

    Neuron changes vs. the upstream ``DistributedAutoencoderKLWan``:
      - ``tile_exec`` runs each tile's frames through the compiled decoder
        (``self._compiled_decoder_first`` or ``self._compiled_decoder_rest``)
        with an explicit per-tile ``feat_cache`` allocated by
        ``_init_feat_cache`` — the same fixed-shape, no-None/-string cache path
        the non-distributed Neuron ``_decode`` uses — instead of the diffusers
        ``self._feat_map`` / ``self._conv_idx`` mutable-state machine, which
        is incompatible with torch.compile.
      - Per-tile frame slices use ``torch.index_select`` to match the rest of
        the Neuron decode path.
      - ``init_distributed`` accepts an explicit process group (and its Gloo twin for the
        error sync) so the executor's collectives run over just the VAE-parallel ranks
        (a subgroup), rather than the whole DiT group as upstream does.
      - Shape statistics and integer tile metadata stay on CPU so the composite
        process group routes them over Gloo. Decoded tiles are packed, gathered in
        bounded plane chunks by a compiled collective and blended on rank 0's device
        (``NeuronDistributedVaeExecutor.gather_and_blend_tiles``).

    Collective-participation contract:
        When patch parallelism is active (``is_distributed_enabled()`` is True),
        ``tiled_decode`` runs ``gather`` / ``all_reduce`` / ``broadcast`` on the
        executor's process group. **Every** rank in that group MUST call
        ``decode`` in lockstep,
        or the group deadlocks. The pipeline builds this VAE (and calls
        ``decode``) only on the ranks that belong to the VAE-parallel subgroup;
        with ``broadcast_result=False`` only the subgroup's rank 0 receives the
        merged frames.
    """

    def init_distributed(self, group=None, error_group=None):
        """Create the distributed-VAE executor, optionally bound to ``group``.

        ``error_group`` is ``group``'s Gloo twin (``new_vae_parallel_groups``), which the
        executor's error sync all-reduces on; it is required with ``group``.

        ``init_distributed`` is the only place ``distributed_executor`` is created,
        and both ``set_parallel_size`` and ``is_distributed_enabled`` dereference
        it, so it must be called before any decode.

        Upstream's mixin binds the executor to the full DiT group. When
        ``group`` is provided, this rebinds it to that subgroup so the executor's
        collectives (gather/all_reduce/broadcast) run over only the VAE-parallel
        ranks; the remaining ranks never build this VAE and skip decode entirely.
        Sizing the subgroup to exactly the parallel degree also makes the
        executor's ``world_size == parallel_size``, so every participating rank
        is assigned a task bucket (no out-of-range rank) and the merged result
        lands on the subgroup's rank 0.
        """
        self.distributed_executor = _WanVaeExecutor()
        if group is not None:
            self.distributed_executor.bind_group(group, error_group)

    def set_parallel_size(self, pp_size: int, mode: str = "tile") -> None:
        """Override to always use the executor's world_size as parallel size.

        The upstream executor builds an ``assigned`` list with exactly
        ``pp_size`` buckets and indexes it by ``self.rank``. If
        ``pp_size < world_size``, high-rank members get IndexError.
        Always use ``world_size`` (which equals the Neuron-valid subgroup
        size set in init_distributed) so every rank gets a bucket; surplus
        ranks receive no tiles but still join collectives.

        ``mode`` is forwarded to the executor so the parallel mode is preserved.
        """
        self.distributed_executor.set_parallel_size(self.distributed_executor.world_size, mode=mode)

    def is_distributed_enabled(self) -> bool:
        """Whether ``decode`` / ``encode`` run patch parallel rather than the native tiling:
        tiling on, not batch mode, and a parallel degree of at least 2 over the bound group.
        """
        executor = self.distributed_executor
        return (
            executor.parallel_mode != "batch"
            and dist.is_initialized()
            and getattr(self, "use_tiling", False)
            and min(int(executor.parallel_size), int(executor.world_size)) > 1
        )

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        """The decode tile grid. Each task holds the whole latent ``z``: every rank builds the
        full task list before balancing but decodes only its own tiles, so ``tile_exec`` slices
        its tile (slicing every tile and frame here, on every rank, would make at 1080p 60
        tiles x 21 frames of device copies, about 80 MB a rank, alive through the gather)."""
        layout = self._prepare_decoder_tiles(z)

        tiletask_list = []
        for row in range(len(layout.row_geometry)):
            for column in range(len(layout.column_geometry)):
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (row, column),
                        z,
                        workload=layout.tile_latent_height * layout.tile_latent_width,
                    )
                )
        tile_spec = {
            "sample_height": layout.sample_height,
            "sample_width": layout.sample_width,
            "blend_height": layout.blend_height,
            "blend_width": layout.blend_width,
            "tile_sample_stride_height": layout.tile_sample_stride_height,
            "tile_sample_stride_width": layout.tile_sample_stride_width,
            "crop_tops": tuple(
                crop_row * layout.output_scale for _, _, crop_row, _ in layout.row_geometry
            ),
            "crop_lefts": tuple(
                crop_column * layout.output_scale for _, _, crop_column, _ in layout.column_geometry
            ),
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(len(layout.row_geometry), len(layout.column_geometry)),
            tile_spec=tile_spec,
        )
        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        """Decode one full-size latent tile.

        Logical edge crops are applied by the merge graph. Keeping the decoded
        tile full-size here avoids a crop graph followed immediately by padding
        it back to the collective slot shape.
        """
        z = task.tensor
        layout = self._prepare_decoder_tiles(z)
        row, column = task.grid_coord
        input_row = layout.row_geometry[row][1]
        input_column = layout.column_geometry[column][1]
        h_indices = torch.arange(input_row, input_row + layout.tile_latent_height, device=z.device)
        w_indices = torch.arange(
            input_column, input_column + layout.tile_latent_width, device=z.device
        )
        tile_z = torch.index_select(z, 3, h_indices)
        tile_z = torch.index_select(tile_z, 4, w_indices)
        # The compiled decoder consumes one frame at a time (feat_cache carries the temporal
        # state across them).
        feat_map = self._init_feat_cache(tile_z)
        time = []
        for k in range(z.shape[2]):
            frame = torch.index_select(tile_z, 2, torch.tensor([k], device=z.device))
            compiled_decoder = (
                self._compiled_decoder_first if k == 0 else self._compiled_decoder_rest
            )
            result = compiled_decoder(frame, *feat_map, first_chunk=(k == 0))
            time.append(result[0])
            feat_map = list(result[1:])
        # The caches (about 0.3 GB a 256x256 tile) are dead: free them before the concat.
        del feat_map, result, tile_z
        return self.distributed_executor.concat_tile_frames(time)

    def tile_merge(
        self,
        local_tile_tensor: torch.Tensor,
        meta_gather,
        grid_spec: GridSpec,
        tid_coord_map: dict,
    ) -> torch.Tensor:
        """Gather and blend decoded tiles into a full image on the Neuron device."""
        if self.config.patch_size is not None:
            raise NotImplementedError("Device VAE tile merge does not support patchified output")
        ts = grid_spec.tile_spec
        return self.distributed_executor.gather_and_blend_tiles(
            local_tile_tensor,
            meta_gather,
            grid_spec,
            tid_coord_map,
            full_height=ts["sample_height"],
            full_width=ts["sample_width"],
            stride_height=ts["tile_sample_stride_height"],
            stride_width=ts["tile_sample_stride_width"],
            blend_height=ts["blend_height"],
            blend_width=ts["blend_width"],
            clamp=True,
            crop_tops=ts["crop_tops"],
            crop_lefts=ts["crop_lefts"],
        )

    def tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> DecoderOutput | torch.Tensor:
        if not self.is_distributed_enabled():
            return super().tiled_decode(z, return_dict=return_dict)

        result = self.distributed_executor.execute(
            z,
            DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
            output_dtype=self.dtype,
            broadcast_result=False,
        )
        if not return_dict:
            return (result,)

        return DecoderOutput(sample=result)

    # ------------------------------------------------------------------
    # Patch-parallel ENCODE (image/video -> latent), the encoder analogue of the
    # tiled_decode operators above. Reuses the same _WanVaeExecutor
    # (self.distributed_executor, bound to the VAE-parallel subgroup): split the
    # pixel input into overlapping spatial tiles, encode each assigned tile's
    # temporal chunks through the compiled encoder, gather to rank 0, and blend
    # the latent tiles. Without this, encode would run serially on rank 0 while decode
    # shards across the group, and the I2V image encode would be the dominant VAE
    # cost.
    # ------------------------------------------------------------------
    def encode_tile_split(self, x: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        """Split the pixel input into overlapping spatial tiles (encoder side).

        Mirrors the non-distributed :meth:`NeuronAutoencoderKLWan.tiled_encode`
        tiling: iterate the pixel-space grid with ``tile_sample_stride`` and crop
        ``tile_sample_min`` windows. Blend/stride/crop bookkeeping is recorded in
        latent space (the ``merge`` side blends the encoded latent tiles).
        """
        _, _, num_frames, height, width = x.shape

        # Latent-space compression ratio (patchify pre-shrinks by patch_size).
        encode_ratio = self.spatial_compression_ratio
        if self.config.patch_size is not None:
            encode_ratio = self.spatial_compression_ratio // self.config.patch_size

        latent_height = height // encode_ratio
        latent_width = width // encode_ratio
        tile_latent_min_height = self.tile_sample_min_height // encode_ratio
        tile_latent_min_width = self.tile_sample_min_width // encode_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // encode_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // encode_ratio
        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        tiletask_list = []
        for i in range(0, height, self.tile_sample_stride_height):
            for j in range(0, width, self.tile_sample_stride_width):
                hi = min(height, i + self.tile_sample_min_height)
                wj = min(width, j + self.tile_sample_min_width)
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (
                            i // self.tile_sample_stride_height,
                            j // self.tile_sample_stride_width,
                        ),
                        # Keep split metadata-only. Every rank builds the same
                        # task list before balancing, but only executes at most
                        # one of the 15 production tiles. Materializing every
                        # tile here would duplicate all spatial slices on every rank.
                        x,
                        workload=(hi - i) * (wj - j),
                    )
                )
        tile_spec = {
            "latent_height": latent_height,
            "latent_width": latent_width,
            "blend_height": blend_height,
            "blend_width": blend_width,
            "tile_latent_stride_height": tile_latent_stride_height,
            "tile_latent_stride_width": tile_latent_stride_width,
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(
                tiletask_list[-1].grid_coord[0] + 1,
                tiletask_list[-1].grid_coord[1] + 1,
            ),
            tile_spec=tile_spec,
        )
        return tiletask_list, grid_spec

    def encode_tile_exec(self, task: TileTask) -> torch.Tensor:
        """Encode a single pixel tile into a latent tile via the compiled encoder.

        Each tile gets its own feat_cache (sized to the tile's spatial dims by
        _init_enc_feat_cache) so the causal-conv temporal state stays correct
        across the tile's temporal chunks without cross-tile contamination —
        the encoder analogue of :meth:`tile_exec`.
        """
        encoder = self._encoder_module()
        source_x = task.tensor
        row, column = task.grid_coord
        i = row * self.tile_sample_stride_height
        j = column * self.tile_sample_stride_width
        hi = min(source_x.shape[3], i + self.tile_sample_min_height)
        wj = min(source_x.shape[4], j + self.tile_sample_min_width)
        h_indices = torch.arange(i, hi, device=source_x.device)
        w_indices = torch.arange(j, wj, device=source_x.device)
        tile_x = torch.index_select(source_x, 3, h_indices)
        tile_x = torch.index_select(tile_x, 4, w_indices)
        feat_map = self._init_enc_feat_cache(tile_x)

        num_frames = tile_x.shape[2]
        iter_ = 1 + (num_frames - 1) // 4
        time = []
        for k in range(iter_):
            if k == 0:
                f_idx = torch.arange(0, 1, device=tile_x.device)
            else:
                f_idx = torch.arange(1 + 4 * (k - 1), 1 + 4 * k, device=tile_x.device)
            chunk = torch.index_select(tile_x, 2, f_idx)
            result = encoder(chunk, *feat_map, first_chunk=(k == 0))
            time.append(result[0])
            feat_map = list(result[1:])
        del feat_map, result, chunk  # dead caches; free them before the concat
        return self.distributed_executor.concat_tile_frames(time)

    def encode_i2v_tile_exec(
        self,
        task: TileTask,
        *,
        num_frames: int,
        last_image: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode one assigned I2V tile without materializing the full video."""
        image = task.tensor
        row, column = task.grid_coord
        i = row * self.tile_sample_stride_height
        j = column * self.tile_sample_stride_width
        hi = min(image.shape[3], i + self.tile_sample_min_height)
        wj = min(image.shape[4], j + self.tile_sample_min_width)
        h_indices = torch.arange(i, hi, device=image.device)
        w_indices = torch.arange(j, wj, device=image.device)
        first_tile = torch.index_select(image, 3, h_indices)
        first_tile = torch.index_select(first_tile, 4, w_indices)
        last_tile = None
        if last_image is not None:
            last_tile = torch.index_select(last_image, 3, h_indices)
            last_tile = torch.index_select(last_tile, 4, w_indices)

        encoder = self._encoder_module()
        feat_map = self._init_enc_feat_cache(first_tile)
        iter_ = 1 + (num_frames - 1) // 4
        time = []
        for k in range(iter_):
            if k == 0:
                chunk = first_tile
            elif last_tile is not None and k == iter_ - 1:
                chunk = torch.cat(
                    [
                        first_tile.new_zeros(*first_tile.shape[:2], 3, *first_tile.shape[3:]),
                        last_tile,
                    ],
                    dim=2,
                )
            else:
                chunk = first_tile.new_zeros(
                    *first_tile.shape[:2],
                    4,
                    *first_tile.shape[3:],
                )
            result = encoder(chunk, *feat_map, first_chunk=(k == 0))
            time.append(result[0])
            feat_map = list(result[1:])
        del feat_map, result, chunk  # dead caches; free them before the concat
        return self.distributed_executor.concat_tile_frames(time)

    def encode_i2v_condition(
        self,
        image: torch.Tensor,
        *,
        num_frames: int,
        last_image: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        """Encode an I2V condition from endpoint frames with lazy zero expansion."""
        self._reject_while_sharded("encoder")
        if not self.is_distributed_enabled() or not self.use_tiling:
            return super().encode_i2v_condition(
                image,
                num_frames=num_frames,
                last_image=last_image,
                return_dict=return_dict,
            )

        self._validate_i2v_condition_inputs(
            image,
            num_frames=num_frames,
            last_image=last_image,
        )

        if self.config.patch_size is not None:
            image = patchify(image, patch_size=self.config.patch_size)
            if last_image is not None:
                last_image = patchify(last_image, patch_size=self.config.patch_size)

        def exec_i2v_tile(task):
            return self.encode_i2v_tile_exec(
                task,
                num_frames=num_frames,
                last_image=last_image,
            )

        h = self.distributed_executor.execute(
            image,
            DistributedOperator(
                split=self.encode_tile_split,
                exec=exec_i2v_tile,
                merge=self.encode_tile_merge,
            ),
            output_dtype=self.dtype,
            broadcast_result=True,
        )
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def encode_tile_merge(
        self,
        local_tile_tensor: torch.Tensor,
        meta_gather,
        grid_spec: GridSpec,
        tid_coord_map: dict,
    ) -> torch.Tensor:
        """Gather and blend encoded latent tiles into the full latent on Neuron."""
        ts = grid_spec.tile_spec
        return self.distributed_executor.gather_and_blend_tiles(
            local_tile_tensor,
            meta_gather,
            grid_spec,
            tid_coord_map,
            full_height=ts["latent_height"],
            full_width=ts["latent_width"],
            stride_height=ts["tile_latent_stride_height"],
            stride_width=ts["tile_latent_stride_width"],
            blend_height=ts["blend_height"],
            blend_width=ts["blend_width"],
            clamp=False,
        )

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed_enabled():
            return super().tiled_encode(x)

        # broadcast_result=True: every VAE rank calls encode inside the I2V
        # prepare_latents and continues to normalize / retrieve the latent, so
        # all group members need the merged latent (not just rank 0 as in decode).
        return self.distributed_executor.execute(
            x,
            DistributedOperator(
                split=self.encode_tile_split,
                exec=self.encode_tile_exec,
                merge=self.encode_tile_merge,
            ),
            output_dtype=self.dtype,
            broadcast_result=True,
        )


def build_wan_vae(
    model_path: str,
    dtype: torch.dtype,
    vae_patch_parallel_size: int,
    *,
    use_tiling: bool = True,
    local_files_only: bool = True,
) -> tuple[NeuronAutoencoderKLWan | None, dist.ProcessGroup | None]:
    """Create the contiguous rank-0 VAE group on every rank and the Wan VAE on its members.

    Collective: every rank calls it, at the same point relative to its other ``new_group``
    calls. The Wan2.2 VAE setup, in the form of ``build_flux_vae``.

    Args:
        model_path: Diffusers Wan2.2 checkpoint directory or hub id (``vae/`` subfolder).
        dtype: VAE parameter dtype.
        vae_patch_parallel_size: VAE group size, clamped to the world size. 1 keeps the
            unsharded VAE on rank 0; otherwise 2, 4, 8, 16 or a multiple of 32 that divides the
            world (a Neuron collective group). The Wan2.2 configs use T2V 16 / 32 / 64 and I2V
            16 / 32 / 32 at 480p / 720p / 1080p.
        use_tiling: Spatial tiling, which patch parallelism runs on (every Wan2.2 config sets
            it).
        local_files_only: Passed to ``from_pretrained``.

    Returns:
        ``(vae, group)``: a ``DistributedAutoencoderKLWan`` on the group's ranks (a
        ``NeuronAutoencoderKLWan`` on rank 0 when the size is 1, with ``group`` ``None``);
        ``vae`` is ``None`` on the other ranks.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    requested = int(vae_patch_parallel_size or 1)
    size = min(requested, world_size)
    if size != requested:
        logger.info("VAE patch parallel: requested=%d, actual group size=%d", requested, size)
    if size < 1 or (
        size > 1 and (world_size % size or not (size in (2, 4, 8, 16) or size % 32 == 0))
    ):
        raise ValueError(
            f"vae_patch_parallel_size={requested} (group size {size}) must be 1, or divide world "
            f"size {world_size} and be 2, 4, 8, 16 or a multiple of 32"
        )

    group = error_group = None
    if size > 1:
        group, error_group = new_vae_parallel_groups(list(range(size)))
    if rank >= size:
        return None, group

    vae_cls = DistributedAutoencoderKLWan if size > 1 else NeuronAutoencoderKLWan
    vae = vae_cls.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    vae.use_tiling = use_tiling
    if size > 1:
        # Patch parallelism dispatches on tiled_decode(), which requires tiling. Bind the
        # executor to the VAE subgroup and record the parallel degree (== subgroup size).
        vae.init_distributed(group=group, error_group=error_group)
        vae.set_parallel_size(size)
    return vae, group
