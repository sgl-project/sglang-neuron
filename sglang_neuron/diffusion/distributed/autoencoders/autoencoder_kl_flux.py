# SPDX-License-Identifier: Apache-2.0
"""Patch-parallel Neuron decoder for the FLUX.1 2D ``AutoencoderKL``.

Fixed-shape 64-latent tiles are decoded across a contiguous rank-0 process group and
blended on its rank 0 by the shared Neuron distributed-VAE executor. This module must not import
``sglang_neuron.diffusion.models.flux``: that package imports this module.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.upsampling import Upsample2D

from sglang_neuron.diffusion import compile_backend
from sglang_neuron.diffusion.distributed.autoencoders.nki_conv import NkiConv2d
from sglang_neuron.diffusion.distributed.autoencoders.vae_executor import (
    GridSpec,
    NeuronDistributedVaeExecutor,
    decoder_tile_geometry,
    new_vae_parallel_groups,
    vae_nki_attn,
)
from sglang_neuron.diffusion.kernels.kernel_choice import device_reason, record_kernel_choice

VAE_COMPILER_ARGS = (
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
    "--internal-hlo2tensorizer-options=--hierarchy-splitter=false",
)
TILE_LATENT_SIZES = (32, 48, 64)
TILE_LATENT_OVERLAP = 16


def tile_size(latent_height: int, latent_width: int, ranks: int) -> tuple[int, int]:
    """Smallest ``(tile, stride)`` whose tile count fits ``ranks``, else the largest tile."""
    for size in TILE_LATENT_SIZES:
        stride = size - TILE_LATENT_OVERLAP
        count = len(decoder_tile_geometry(latent_height, size, stride))
        count *= len(decoder_tile_geometry(latent_width, size, stride))
        if count <= ranks:
            return size, stride
    return size, stride


class _NeuronGroupNorm(nn.GroupNorm):
    """GroupNorm with explicit fp32 statistics, independent of backend decompositions.

    On Neuron a batch-1 [B, C, H, W] input whose H * W is a multiple of 128 runs the FLUX GroupNorm NKI kernel;
    otherwise normalization and affine are folded into one per-channel scale and shift.
    """

    _silu = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._can_use_kernel(x):
            from sglang_neuron.diffusion.kernels.flux.group_norm import (
                flux_group_norm_fwd,
            )
            from sglang_neuron.diffusion.neuron_api import wrap_nki_kernel

            return wrap_nki_kernel(flux_group_norm_fwd, supported_lnc=(2,))(
                x=x,
                weight=self.weight.float(),
                bias=self.bias.float(),
                num_groups=self.num_groups,
                eps=self.eps,
                silu=self._silu,
            )
        y = self._normalized(x)
        return (F.silu(y) if self._silu else y).to(x.dtype)

    def _can_use_kernel(self, x: torch.Tensor) -> bool:
        """Whether ``x`` runs the FLUX GroupNorm NKI kernel; the decision is reported."""
        from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel

        if not can_run_kernel(x):
            return record_kernel_choice("flux_group_norm", False, device_reason(x))
        if not self.affine or x.ndim != 4:
            return record_kernel_choice(
                "flux_group_norm", False, "not a 4-D affine GroupNorm", by_design=True
            )
        if x.shape[0] != 1 or (x.shape[2] * x.shape[3]) % 128 != 0:
            return record_kernel_choice(
                "flux_group_norm", False, "batch is not 1 or H * W is not a multiple of 128"
            )
        return record_kernel_choice("flux_group_norm", True, "eligible")

    def _normalized(self, x: torch.Tensor) -> torch.Tensor:
        """fp32 ``GroupNorm(x)`` with the affine folded into one per-channel scale and shift."""
        batch, channels = x.shape[:2]
        h = x.float().reshape(batch, self.num_groups, -1)
        mean = h.mean(dim=-1, keepdim=True)
        var = (h - mean).square().mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(var + self.eps).expand(-1, -1, channels // self.num_groups)
        scale = inv.reshape(batch, channels)
        shift = -mean.expand(-1, -1, channels // self.num_groups).reshape(batch, channels) * scale
        if self.affine:
            scale = scale * self.weight.float()
            shift = shift * self.weight.float() + self.bias.float()
        stat_shape = (batch, channels) + (1,) * (x.ndim - 2)
        return x.float() * scale.reshape(stat_shape) + shift.reshape(stat_shape)


class _NeuronGroupNormSiLU(_NeuronGroupNorm):
    """``_NeuronGroupNorm`` followed by SiLU in fp32 before the cast (replaces a GroupNorm + SiLU pair)."""

    _silu = True


class _RepeatUpsample2D(Upsample2D):
    """Nearest x2 upsample as two ``repeat_interleave`` copies, then the 3x3 conv."""

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        return self.conv(hidden_states)


def _as_nki_conv(conv: nn.Conv2d) -> None:
    """Turn a 3x3 decoder ``nn.Conv2d`` into an ``NkiConv2d`` whose filters are packed at compile."""
    conv.__class__ = NkiConv2d
    conv.register_buffer("_nki_packed_filters", None, persistent=False)


class _NeuronVaeAttnProcessor:
    """Single-head mid-block attention as diffusers ``AttnProcessor2_0``, NKI on Neuron."""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        batch, channels, height, width = hidden_states.shape
        hidden_states = attn.group_norm(
            hidden_states.reshape(batch, channels, height * width)
        ).transpose(1, 2)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        if query.device.type == "neuron":
            out = vae_nki_attn(
                (query * attn.scale).contiguous(), key.contiguous(), value.contiguous()
            )
        else:
            out = F.scaled_dot_product_attention(
                query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
            ).squeeze(1)
        out = attn.to_out[1](attn.to_out[0](out.to(query.dtype)))
        out = out.transpose(1, 2).reshape(batch, channels, height, width)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


class _FluxVaeExecutor(NeuronDistributedVaeExecutor):
    """Neuron VAE executor that runs the cross-rank gather checks only on a shape's first decode."""

    def __init__(self):
        super().__init__()
        self.decode_shape = None
        self.checked_shapes = set()

    def _raise_if_device_gather_failed(self, local_error, phase):
        if local_error is None and self.decode_shape in self.checked_shapes:
            return
        super()._raise_if_device_gather_failed(local_error, phase)


class DistributedAutoencoderKLFlux(AutoencoderKL):
    """FLUX.1 ``AutoencoderKL`` whose ``decode`` runs fixed-shape tiles on the VAE group.

    Every rank of the group must call ``decode`` in lockstep; only group rank 0 returns
    the image, the others return an empty tensor.
    """

    def init_distributed(self, group=None, error_group=None) -> None:
        """Create the Flux VAE executor bound to ``group`` (every VAE rank decodes one tile set)."""
        self.distributed_executor = _FluxVaeExecutor()
        if group is not None:
            self.distributed_executor.bind_group(group, error_group)

    def set_parallel_size(self, pp_size: int, mode: str = "tile") -> None:
        """Use the bound group's size as the parallel degree so every rank owns a bucket."""
        self.distributed_executor.set_parallel_size(self.distributed_executor.world_size, mode=mode)

    def is_distributed_enabled(self) -> bool:
        """Whether ``decode`` can run: tile mode over a bound VAE group of at least 2 ranks."""
        executor = getattr(self, "distributed_executor", None)
        return (
            executor is not None
            and dist.is_initialized()
            and getattr(self, "use_tiling", False)
            and executor.parallel_mode == "tile"
            and min(int(executor.parallel_size), int(executor.world_size)) > 1
        )

    def prepare_for_neuron(self) -> None:
        """Swap decoder GroupNorm, Upsample2D and mid-block attention for Neuron forms.

        GroupNorms followed by SiLU (resnet norms, the output norm) apply the SiLU themselves.
        Resnet ``conv2`` and the upsample convs run the LNC2-sharded NKI conv kernel where eligible.
        """
        for module in self.decoder.modules():
            if isinstance(module, nn.GroupNorm):
                module.__class__ = _NeuronGroupNorm
            elif isinstance(module, Upsample2D):
                if (
                    not module.use_conv
                    or module.use_conv_transpose
                    or not module.interpolate
                    or module.norm is not None
                    or module.name != "conv"
                ):
                    raise ValueError("Flux VAE Upsample2D must be a nearest x2 upsample with conv")
                module.__class__ = _RepeatUpsample2D
                _as_nki_conv(module.conv)
        for module in self.decoder.modules():
            if isinstance(module, ResnetBlock2D) and module.time_emb_proj is None:
                module.norm1.__class__ = _NeuronGroupNormSiLU
                module.norm2.__class__ = _NeuronGroupNormSiLU
                module.nonlinearity = nn.Identity()
                _as_nki_conv(module.conv2)
        if isinstance(self.decoder.conv_act, nn.SiLU):
            self.decoder.conv_norm_out.__class__ = _NeuronGroupNormSiLU
            self.decoder.conv_act = nn.Identity()
        attentions = self.decoder.mid_block.attentions
        if (
            len(attentions) != 1
            or attentions[0].heads != 1
            or attentions[0].spatial_norm is not None
        ):
            raise ValueError("Flux VAE mid block must have one single-head attention")
        attentions[0].set_processor(_NeuronVaeAttnProcessor())

    def _decode_tile(self, z: torch.Tensor) -> torch.Tensor:
        """Decode one ``[1, C, t, t]`` latent tile to ``[1, 3, 8t, 8t]``."""
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        return self.decoder(z)

    def compile(self, *args, **kwargs):
        """Lazily compile the fixed-shape tile decoder (NEFF on first call)."""
        options = {
            **(kwargs.pop("options", None) or {}),
            "model_name": "flux_vae_decoder",
            "compiler_args": list(VAE_COMPILER_ARGS),
        }
        kwargs.setdefault("backend", compile_backend.register())
        kwargs.setdefault("fullgraph", True)
        kwargs["dynamic"] = False
        for module in self.decoder.modules():
            if isinstance(module, NkiConv2d):
                module.install_nki_dispatch()
        self._compiled_decode_tile = torch.compile(
            self._decode_tile, *args, options=options, **kwargs
        )

    def _tile_grid(self, z: torch.Tensor):
        """Tile windows ``(row, column, top, left)``, the merge ``GridSpec`` and the tile size."""
        latent_height, latent_width = z.shape[-2:]
        size, stride = tile_size(latent_height, latent_width, self.distributed_executor.world_size)
        if latent_height < size or latent_width < size:
            raise ValueError(f"Flux VAE latent must be at least {size} per side")
        rows = decoder_tile_geometry(latent_height, size, stride)
        columns = decoder_tile_geometry(latent_width, size, stride)
        scale = 2 ** (len(self.config.block_out_channels) - 1)
        windows = [
            (row, column, top, left)
            for row, (_, top, _, _) in enumerate(rows)
            for column, (_, left, _, _) in enumerate(columns)
        ]
        tile_spec = {
            "sample_height": latent_height * scale,
            "sample_width": latent_width * scale,
            "blend_height": TILE_LATENT_OVERLAP * scale,
            "blend_width": TILE_LATENT_OVERLAP * scale,
            "tile_sample_stride_height": stride * scale,
            "tile_sample_stride_width": stride * scale,
            "crop_tops": tuple(crop * scale for _, _, crop, _ in rows),
            "crop_lefts": tuple(crop * scale for _, _, crop, _ in columns),
        }
        grid_spec = GridSpec(
            split_dims=(2, 3),
            grid_shape=(len(rows), len(columns)),
            tile_spec=tile_spec,
        )
        return windows, grid_spec, size

    def tile_merge(
        self,
        local_tile_tensor: torch.Tensor,
        meta_gather,
        grid_spec: GridSpec,
        tid_coord_map: dict,
    ) -> torch.Tensor | None:
        """Gather and blend the decoded tiles into the image on group rank 0."""
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

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        """Decode ``z [1, 16, Hl, Wl]`` across the VAE group; rank 0 gets ``[1, 3, H, W]``.

        Tile ``k`` is decoded by group rank ``k % world`` (the executor's round-robin balance for
        equal tiles), so the tile metadata is known on every rank without a host collective and
        each rank slices only its own windows.
        """
        if not self.is_distributed_enabled():
            raise RuntimeError(
                "DistributedAutoencoderKLFlux decodes only through the distributed executor; "
                "build it with build_flux_vae and a VAE group of at least 2 ranks"
            )
        executor = self.distributed_executor
        executor.decode_shape = tuple(z.shape)
        world, rank = executor.world_size, executor.rank
        windows, grid_spec, size = self._tile_grid(z)
        scale = grid_spec.tile_spec["sample_height"] // z.shape[-2]
        tile_shape = (z.shape[0], self.config.out_channels, size * scale, size * scale)
        slots = -(-len(windows) // world)
        local = []
        for tid in range(rank, len(windows), world):
            _, _, top, left = windows[tid]
            tile = z.index_select(2, torch.arange(top, top + size, device=z.device))
            tile = tile.index_select(3, torch.arange(left, left + size, device=z.device))
            local.append((tid, self._compiled_decode_tile(tile)))
        if slots == 1 and local:
            payload = local[0][1].unsqueeze(0)
        else:
            payload, _ = executor._pack_local_tiles(
                local, [slots, *tile_shape], grid_spec, z.device, self.dtype
            )
        meta_gather = None
        if rank == 0:
            meta_gather = []
            for owner in range(world):
                meta = torch.full((slots, 3), -1, dtype=torch.int64)
                for slot, tid in enumerate(range(owner, len(windows), world)):
                    meta[slot] = torch.tensor([tid, tile_shape[2], tile_shape[3]])
                meta_gather.append(meta)
        tid_coord_map = {tid: (row, column) for tid, (row, column, _, _) in enumerate(windows)}
        result = self.tile_merge(payload, meta_gather, grid_spec, tid_coord_map)
        executor.checked_shapes.add(executor.decode_shape)
        if rank != 0:
            result = torch.empty(0, device=z.device)
        if not return_dict:
            return (result,)
        return DecoderOutput(sample=result)


def build_flux_vae(
    model_path: str, dtype: torch.dtype, vae_patch_parallel_size: int
) -> tuple[DistributedAutoencoderKLFlux | None, dist.ProcessGroup]:
    """Create the contiguous rank-0 VAE group on every rank and the VAE on its members.

    Args:
        model_path: Diffusers FLUX.1 checkpoint directory or hub id (``vae/`` subfolder).
        dtype: VAE parameter dtype.
        vae_patch_parallel_size: VAE group size; divides the world, 2/4/8/16 or a multiple of 32.

    Returns:
        ``(vae, group)``; ``vae`` is ``None`` on ranks outside the group.
    """
    world_size = dist.get_world_size()
    size = vae_patch_parallel_size
    if world_size % size or not (size in (2, 4, 8, 16) or size % 32 == 0):
        raise ValueError(
            f"vae_patch_parallel_size={size} must divide world size {world_size} and be "
            "2, 4, 8, 16 or a multiple of 32"
        )
    group, error_group = new_vae_parallel_groups(list(range(size)))
    if dist.get_rank() >= size:
        return None, group
    vae = DistributedAutoencoderKLFlux.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype
    )
    vae.prepare_for_neuron()
    vae.use_tiling = True
    vae.init_distributed(group=group, error_group=error_group)
    vae.set_parallel_size(size)
    return vae, group
