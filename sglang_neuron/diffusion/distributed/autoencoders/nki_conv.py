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
# NkiConv2d and its NKI conv3d launchers: the FLUX VAE decoder's and the Wan VAE decoder's 3x3
# stride-1 convolutions, and the Wan VAE encoder's 3x3 stride-2 downsample convolution.

import nki
import torch
import torch.nn as nn
import torch.nn.functional as F
from nkilib.experimental.conv.conv3d import conv3d

from sglang_neuron.diffusion.kernels.kernel_choice import device_reason, record_kernel_choice
from sglang_neuron.diffusion.neuron_api import wrap_nki_kernel_if_supported

# ---------------------------------------------------------------------------
# NKI conv3d for the VAE's 3x3 convolutions
# ---------------------------------------------------------------------------
#
# Dispatch is unconditional, matching every other NKI kernel in this repo.
# can_run_kernel honours SGLANG_NEURON_DISABLE_NKI_KERNELS (the kill switch for the NKI
# kernels), so anything finer is a code revert.
#
# A 2D input is lifted to a depth-1 conv3d (D=1, K_d=1), so the kernel padding is
# (0, 0, 1, 1, 1, 1): no depth padding, and 1 on each side of H and W. That single
# static config covers every eligible site, which is what makes one traced kernel enough. The Wan
# VAE's causal 3x3x3 convs run the same kernel (WanCausalConv3d prepends its 2 cached frames, so
# its depth padding is also 0). NKI specializes stride and padding, so the stride-2 downsample
# has its own wrapper.

# Kernel padding order is (pad_d_l, pad_d_r, pad_h_t, pad_h_b, pad_w_l, pad_w_r).
_VAE_NKI_CONV_PADDING = (0, 0, 1, 1, 1, 1)

# Eligibility threshold for the generic conv3d kernel: below it the contraction dim is too
# underfilled to beat the compiler. Deliberately not env-tunable.
_VAE_NKI_CONV_MIN_CHANNELS = 96
# Wan VAE encoder: only the C=384 stride-2 downsample showed a measurable I2V latency win.
_VAE_NKI_ENCODER_DOWNSAMPLE_MIN_CHANNELS = 384


@nki.jit
def _vae_conv3d_kernel(x, filters, bias):
    return conv3d(
        x,
        filters,
        bias,
        stride=(1, 1, 1),
        padding=_VAE_NKI_CONV_PADDING,
        dilation=(1, 1, 1),
        lnc_shard=True,
    )


@nki.jit
def _vae_conv3d_stride2_kernel(x, filters, bias):
    return conv3d(
        x,
        filters,
        bias,
        stride=(1, 2, 2),
        padding=(0, 0, 0, 1, 0, 1),
        dilation=(1, 1, 1),
        lnc_shard=True,
    )


# Wrapped once, at import (see the launcher note in attention/cp_attention.py). LNC 2 only: None
# at LNC 1, where the conv gates take the compiler path.
_vae_conv3d_launch = wrap_nki_kernel_if_supported(_vae_conv3d_kernel, supported_lnc=(2,))
_vae_conv3d_stride2_launch = wrap_nki_kernel_if_supported(
    _vae_conv3d_stride2_kernel, supported_lnc=(2,)
)


def _vae_nki_conv3d(x: torch.Tensor, filters: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return _vae_conv3d_launch(x, filters, bias)


def _vae_nki_conv3d_stride2(
    x: torch.Tensor, filters: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return _vae_conv3d_stride2_launch(x, filters, bias)


def _pack_nki_filters(weight: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """Permute ``weight`` into the NKI filter layout and materialize it contiguously.

    Packing runs once, at compile time, so the permute and copy run on the CPU and the packed
    result moves back to the original device; no restriding copy runs on the device.
    """
    src = weight.detach()
    device = src.device
    if device.type != "cpu":
        packed = src.cpu().permute(*dims).contiguous().to(src.dtype)
        return packed.to(device)
    return src.permute(*dims).contiguous().to(src.dtype)


class NkiConv2d(nn.Conv2d):
    """``nn.Conv2d`` that can route through the NKI **conv3d** kernel as ``D=1``.

    NKI has no Conv2d entry point, so 2D inputs are lifted to a depth-1 Conv3d.
    Supports the decoders' 3x3 stride-1 convolutions (padding 1) and the Wan encoder's 3x3 stride-2
    downsample, whose right/bottom zero pad (``pad_right_bottom``) runs inside the kernel.
    """

    # A class default, so a decoder conv turned into an NkiConv2d by class swap (the FLUX VAE)
    # takes the plain path.
    pad_right_bottom = False

    def __init__(self, *args, pad_right_bottom: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if pad_right_bottom and (
            tuple(self.kernel_size) != (3, 3)
            or tuple(self.stride) != (2, 2)
            or tuple(self.padding) != (0, 0)
        ):
            raise ValueError("pad_right_bottom requires a 3x3 stride-2 convolution")
        self.pad_right_bottom = pad_right_bottom
        self.register_buffer("_nki_packed_filters", None, persistent=False)

    def _can_use_nki_conv_kernel(self, target_device=None) -> bool:
        """Static (weight-only) dispatch test, checked at pack time."""
        from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel

        # Framework guard first: honours the NKI kill switch and refuses CPU weights.
        # Only a can_run_kernel refusal can raise. The other refusals are the measured
        # dispatch policy, so they are recorded as by design.
        device = self.weight if target_device is None else str(target_device)
        if not can_run_kernel(device):
            return record_kernel_choice("vae_conv2d", False, device_reason(device))
        if _vae_conv3d_launch is None:
            return record_kernel_choice(
                "vae_conv2d", False, "the VAE conv kernels run at LNC 2 only", by_design=True
            )
        if tuple(self.dilation) != (1, 1) or self.groups != 1 or self.bias is None:
            return record_kernel_choice(
                "vae_conv2d", False, "dilation, groups or bias not supported", by_design=True
            )
        if tuple(self.kernel_size) != (3, 3):
            return record_kernel_choice(
                "vae_conv2d", False, "kernel size is not 3x3", by_design=True
            )
        conv_config = (tuple(self.stride), tuple(self.padding))
        if conv_config not in {((1, 1), (1, 1)), ((2, 2), (0, 0))}:
            return record_kernel_choice(
                "vae_conv2d", False, "stride and padding not supported", by_design=True
            )
        if conv_config == ((2, 2), (0, 0)):
            if not self.pad_right_bottom:
                return record_kernel_choice(
                    "vae_conv2d", False, "stride-2 conv without pad_right_bottom", by_design=True
                )
            if (
                self.in_channels < _VAE_NKI_ENCODER_DOWNSAMPLE_MIN_CHANNELS
                or self.out_channels < _VAE_NKI_ENCODER_DOWNSAMPLE_MIN_CHANNELS
            ):
                return record_kernel_choice(
                    "vae_conv2d",
                    False,
                    "downsample channels below the measured limit",
                    by_design=True,
                )
            return record_kernel_choice("vae_conv2d", True, "eligible (stride 2)")
        # PE-starvation floor: C_out sits on the PSUM partition dim, so small-C_out convs
        # run slower than on the compiler.
        if self.out_channels <= 32 or self.in_channels < _VAE_NKI_CONV_MIN_CHANNELS:
            return record_kernel_choice(
                "vae_conv2d", False, "channels below the measured limit", by_design=True
            )
        return record_kernel_choice("vae_conv2d", True, "eligible (stride 1)")

    def install_nki_dispatch(self, target_device=None) -> bool:
        """Pack own weight into NKI layout ``[1, K_h, K_w, C_in, C_out]``, D=1 lifted.

        Returns True if this site was packed.
        """
        if self._nki_packed_filters is not None:
            return True
        if not self._can_use_nki_conv_kernel(target_device):
            return False
        self._nki_packed_filters = _pack_nki_filters(self.weight, (2, 3, 1, 0)).unsqueeze(0)
        return True

    # Set by wan_spatial_shard.install_spatial_shard for spatially-sharded decode.
    _spatial_shard = None

    def forward(self, x):
        if self._spatial_shard is not None:
            if self.pad_right_bottom:
                # Exact sharded stride-2 downsample (encoder): on an even block, output
                # column j reads local inputs 2j..2j+2, so only a right/bottom halo of one
                # pixel is needed -- zeros on the frame border, which is Wan's ZeroPad2d.
                x = self._spatial_shard.halo_after(x, 1)
                return F.conv2d(
                    x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups
                )
            # Exact sharded conv: neighbours' halo replaces the zero padding. The compiler's
            # convolution: an NKI kernel measured slower at the per-rank block sizes.
            x = self._spatial_shard.halo(x, self.padding[0])
            return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)
        packed = self._nki_packed_filters
        if packed is None:
            if self.pad_right_bottom:
                # Matches Wan's original right/bottom ZeroPad2d.
                x = F.pad(x, (0, 1, 0, 1))
            return super().forward(x)
        bias = self.bias.to(packed.dtype)
        # Lift to a degenerate 3D conv: D=1 in, D_out=1 out (K_d=1, d-pad=0).
        x_in = x.unsqueeze(2).to(packed.dtype)
        if self.pad_right_bottom:
            return _vae_nki_conv3d_stride2(x_in, packed, bias).squeeze(2)
        return _vae_nki_conv3d(x_in, packed, bias).squeeze(2)
