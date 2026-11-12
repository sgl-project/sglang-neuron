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
# Neuron port of diffusers ``AutoencoderKLMiniMaxH3``: a causal 3-D conv encoder and a
# non-causal ViT decoder. The config, module tree (checkpoint keys load 1:1), encoder, decoder
# blocks, chunk geometry, tile helpers and ``encode`` / ``decode`` are upstream's.
#
# Neuron-specific changes
# 1. NeuronMiniMaxH3VideoAttnProcessor: attention runs the shared VAE ``attention_cte`` NKI
#    kernel (``vae_nki_attn``) on Neuron, SDPA elsewhere.
# 2. NeuronMiniMaxH3VideoViTDecoder3d: the RoPE position grid is built from repeat ops; the
#    Neuron backend rejects reshaping upstream's non-contiguous meshgrid/stack views.
# 3. NeuronMiniMaxH3VideoGroupNorm: group statistics in fp32, as torch's CPU group_norm reduces
#    bf16 inputs; the Neuron native kernel reduces at the input dtype.
# 4. Temporal chunks and spatial tiles are gathered with index_select: the native executor can
#    neither take upstream's strided-slice views as compiled-graph inputs nor make them
#    contiguous.
# 5. post_quant_conv + decoder and encoder + quant_conv each run as one compiled graph
#    (``compile``); ``_tiled`` runs upstream's spatial tiling through it and stitches on the host.
# 6. The encoder half runs in FP32, the decoder half in the model dtype.
# 7. Reflect pads are slices and a concat (``_reflect_pad``): ``F.pad(mode="reflect")`` lowers to
#    per-axis index gathers, and on a 256-pixel encoder tile the downsample's (0, 1, 0, 1)
#    gather fails neuronx-cc with NCC_EBIR033.
# 8. DistributedAutoencoderKLMiniMaxH3: patch-parallel decode and encode over the VAE group
#    (``tiled_vae_executor``). Decode runs every (temporal chunk, tile) in one execute. The tile
#    overlaps differ from tile to tile, which the shared executor's device blend cannot express,
#    so encode stitches on rank 0's host, and decode too without change 9.
# 9. Decode merges on the devices with upstream's ``_blend``: each (chunk, column) goes to a
#    rank's slot (``column_layout``); a rank stitches its columns after an all-gather of the
#    columns' right edges (``stitch_columns``), cross-fades each chunk's head after a second
#    all-gather of the chunks' tails, and only the kept pixels reach rank 0
#    (``gather_columns``). ``frame_fn`` converts them on the device first.

from __future__ import annotations

import functools

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
    AutoencoderKLMiniMaxH3,
    MiniMaxH3VideoCausalConv3d,
    MiniMaxH3VideoDownsample3d,
    MiniMaxH3VideoGroupNorm,
    MiniMaxH3VideoViTDecoder3d,
)
from diffusers.models.modeling_utils import get_parameter_dtype

from sglang_neuron.diffusion.distributed.autoencoders.tiled_vae_executor import (
    DistributedOperator,
    TiledVaeExecutor,
    TileGridSpec,
    TileTask,
)
from sglang_neuron.diffusion.distributed.autoencoders.vae_executor import vae_nki_attn
from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel


class NeuronMiniMaxH3VideoAttnProcessor:
    """``MiniMaxH3VideoAttnProcessor`` with the NKI attention kernel on Neuron."""

    def __call__(self, attn, hidden_states, rotary_emb=None):
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query.float()).to(query.dtype)
        key = attn.norm_k(key.float()).to(key.dtype)

        if rotary_emb is not None:
            cos, sin = rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            rotary_dim = cos.shape[-1]
            query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
            key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
            query_first, query_second = query_rotary.chunk(2, dim=-1)
            key_first, key_second = key_rotary.chunk(2, dim=-1)
            query_rotated = torch.cat([-query_second, query_first], dim=-1)
            key_rotated = torch.cat([-key_second, key_first], dim=-1)
            query = torch.cat([query_rotary * cos + query_rotated * sin, query_pass], dim=-1)
            key = torch.cat([key_rotary * cos + key_rotated * sin, key_pass], dim=-1)

        # (B, S, H, Dh) -> (B, H, S, Dh)
        query, key, value = (t.transpose(1, 2) for t in (query, key, value))
        if can_run_kernel(hidden_states):
            # attention_cte takes (batch, seq, dim) at scale 1.0: heads fold into the batch,
            # and Q carries SDPA's default Dh**-0.5.
            batch, heads, seq, head_dim = query.shape
            query = query * head_dim**-0.5
            query, key, value = (
                t.reshape(batch * heads, seq, head_dim).contiguous() for t in (query, key, value)
            )
            out = vae_nki_attn(query, key, value).reshape(batch, heads, seq, head_dim)
        else:
            out = F.scaled_dot_product_attention(query, key, value)
        return attn.to_out[0](out.transpose(1, 2).flatten(2, 3))


class NeuronMiniMaxH3VideoViTDecoder3d(MiniMaxH3VideoViTDecoder3d):
    """``MiniMaxH3VideoViTDecoder3d`` with the position grid built from repeat ops."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch_size, num_frames * height * width, num_channels
        )
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]

        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        gt, gh, gw = (
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size)
            - 1.0
            for size in (num_frames, height, width)
        )
        # As meshgrid(indexing="ij") + flatten.
        t_ids = gt.repeat_interleave(height * width)
        h_ids = gh.repeat_interleave(width).repeat(num_frames)
        w_ids = gw.repeat(num_frames * height)
        position_ids = torch.stack([t_ids, h_ids, w_ids], dim=-1)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        rotary_emb = self.rope(position_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :num_patches, :]

        patch_size, patch_size_t = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            patch_size_t,
            patch_size,
            patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )


class NeuronMiniMaxH3VideoGroupNorm(MiniMaxH3VideoGroupNorm):
    """``MiniMaxH3VideoGroupNorm`` with explicit fp32 statistics."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        # Frames fold into the batch: statistics are per frame.
        x = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        h = x.float().reshape(batch_size * num_frames, self.num_groups, -1)
        mean = h.mean(dim=-1, keepdim=True)
        var = (h - mean).square().mean(dim=-1, keepdim=True)
        h = ((h - mean) * torch.rsqrt(var + self.eps)).reshape(x.shape)
        if self.affine:
            affine_shape = (1, 1, num_channels, 1, 1)
            h = h * self.weight.float().reshape(affine_shape) + self.bias.float().reshape(
                affine_shape
            )
        return h.to(hidden_states.dtype).permute(0, 2, 1, 3, 4).contiguous()


def _reflect_pad(x: torch.Tensor, pads: tuple[int, ...]) -> torch.Tensor:
    """``F.pad(x, pads, mode="reflect")`` from slices and a concat; see change 7."""
    for i in range(len(pads) // 2):
        left, right = pads[2 * i], pads[2 * i + 1]
        if left == 0 and right == 0:
            continue
        dim = x.ndim - 1 - i
        size = x.shape[dim]
        parts = [x.narrow(dim, k, 1) for k in range(left, 0, -1)]
        parts.append(x)
        parts += [x.narrow(dim, size - 1 - k, 1) for k in range(1, right + 1)]
        x = torch.cat(parts, dim=dim)
    return x


def _pad(x: torch.Tensor, pads: tuple[int, ...], mode: str) -> torch.Tensor:
    return _reflect_pad(x, pads) if mode == "reflect" else F.pad(x, pads, mode=mode)


class NeuronMiniMaxH3VideoCausalConv3d(MiniMaxH3VideoCausalConv3d):
    """``MiniMaxH3VideoCausalConv3d`` with the spatial reflect pad from ``_reflect_pad``."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_padding > 0:
            padding = self.spatial_padding
            hidden_states = _pad(
                hidden_states, (padding, padding, padding, padding), self.spatial_padding_mode
            )
        if self.temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0))
        return F.conv3d(
            hidden_states, self.weight, self.bias, stride=self.stride, dilation=self.dilation
        )


class NeuronMiniMaxH3VideoDownsample3d(MiniMaxH3VideoDownsample3d):
    """``MiniMaxH3VideoDownsample3d`` with the bottom/right reflect pad from ``_reflect_pad``."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_stride == 2:
            hidden_states = _pad(hidden_states, (0, 1, 0, 1), self.spatial_padding_mode)
        return self.conv(hidden_states)


def prepare_for_neuron(module: torch.nn.Module) -> torch.nn.Module:
    """Swap the Neuron classes into an upstream VAE (sub)tree in place.

    They only override ``forward``, so parameters and checkpoint keys are kept.
    """
    for submodule in module.modules():
        if type(submodule) is MiniMaxH3VideoGroupNorm:
            submodule.__class__ = NeuronMiniMaxH3VideoGroupNorm
        elif type(submodule) is MiniMaxH3VideoViTDecoder3d:
            submodule.__class__ = NeuronMiniMaxH3VideoViTDecoder3d
        elif type(submodule) is MiniMaxH3VideoCausalConv3d:
            submodule.__class__ = NeuronMiniMaxH3VideoCausalConv3d
        elif type(submodule) is MiniMaxH3VideoDownsample3d:
            submodule.__class__ = NeuronMiniMaxH3VideoDownsample3d
        if hasattr(submodule, "set_processor"):
            submodule.set_processor(NeuronMiniMaxH3VideoAttnProcessor())
    return module


class _NeuronH3VAEDecoder(torch.nn.Module):
    """``post_quant_conv`` + decoder as one compiled graph."""

    def __init__(self, post_quant_conv: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z))


class _NeuronH3VAEEncoder(torch.nn.Module):
    """Encoder + ``quant_conv`` as one compiled graph; no state carries across chunks."""

    def __init__(self, encoder: torch.nn.Module, quant_conv: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(x))


class NeuronAutoencoderKLMiniMaxH3(AutoencoderKLMiniMaxH3):
    r"""Neuron video VAE for MiniMax-H3 (header changes 1-7); not tensor-parallel.

    t2va only decodes; the encoder serves fl2va's keyframes.
    """

    _group_offload_block_modules = ["quant_conv", "post_quant_conv", "encoder", "decoder"]
    # Upstream pins all four modules to FP32. Only the encoder half, run once per keyframe, stays
    # FP32 here; the decoder runs in the model dtype.
    _keep_in_fp32_modules = ["encoder", "quant_conv"]

    # Keeps upstream's signature visible to ConfigMixin.from_config.
    @functools.wraps(AutoencoderKLMiniMaxH3.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prepare_for_neuron(self)
        self.use_tiling = True
        self._compiled_decoder = None
        self._compiled_encoder = None

    @property
    def dtype(self) -> torch.dtype:
        """The decoder's dtype; ``ModelMixin.dtype`` would report the FP32 encoder's."""
        return get_parameter_dtype(self.decoder)

    @property
    def encoder_dtype(self) -> torch.dtype:
        return get_parameter_dtype(self.encoder)

    def compile(self, *args, **compiler_kwargs):
        """Compile the decode graph and, with ``compile_encoder=True``, the encode graph."""
        compile_encoder = compiler_kwargs.pop("compile_encoder", False)
        options = compiler_kwargs.pop("options", None) or {}
        model_name = options.get("model_name", "minimax_h3_vae")

        def graph_kwargs(suffix):
            return {
                **compiler_kwargs,
                "options": {**options, "model_name": f"{model_name}_{suffix}"},
            }

        self._compiled_decoder = torch.compile(
            _NeuronH3VAEDecoder(self.post_quant_conv, self.decoder),
            *args,
            **graph_kwargs("decoder"),
        )
        if compile_encoder:
            self._compiled_encoder = torch.compile(
                _NeuronH3VAEEncoder(self.encoder, self.quant_conv), *args, **graph_kwargs("encoder")
            )
        return self._compiled_decoder

    def _tiled(self, x: torch.Tensor, graph, input_ratio: int, output_ratio: int) -> torch.Tensor:
        """Upstream's spatial tiling from ``_encode_clip`` / ``_decode_clip``, around ``graph``.

        Tiles are laid out in pixel space: ``input_ratio`` maps them onto ``x``'s grid and
        ``output_ratio`` their overlaps onto the output's. Every tile has one shape, so the grid
        runs one compiled graph.
        """
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            x.shape[-2] * input_ratio,
            self.tile_sample_min_height,
            self.tile_sample_min_overlap_height,
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            x.shape[-1] * input_ratio,
            self.tile_sample_min_width,
            self.tile_sample_min_overlap_width,
        )

        rows = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            row = []
            for j_pos, j_len in zip(x_indices, x_lengths):
                # index_select: change 4.
                h = i_pos // input_ratio
                w = j_pos // input_ratio
                h_idx = torch.arange(h, h + i_len // input_ratio, device=x.device)
                w_idx = torch.arange(w, w + j_len // input_ratio, device=x.device)
                tile = x.index_select(-2, h_idx).index_select(-1, w_idx)
                # Keeping the whole grid on the device segfaults.
                row.append(graph(tile).cpu())
            rows.append(row)

        return self._stitch_tiles(
            rows,
            [overlap // output_ratio for overlap in y_overlaps],
            [overlap // output_ratio for overlap in x_overlaps],
        )

    def _encode_clip(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled_encoder is None:
            return super()._encode_clip(x)
        if self.use_tiling:
            return self._tiled(x, self._compiled_encoder, 1, self.spatial_compression_ratio)
        return self._compiled_encoder(x)

    def _decode_clip(self, z: torch.Tensor) -> torch.Tensor:
        if self._compiled_decoder is None:
            return super()._decode_clip(z)
        if self.use_tiling:
            return self._tiled(z, self._compiled_decoder, self.spatial_compression_ratio, 1)
        return self._compiled_decoder(z)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Upstream ``_decode`` with each temporal chunk gathered by ``index_select``."""
        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.config.token_drop
        temporal_ratio = self.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio

        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        decoded_chunks = []
        overlap = None
        for i in range(num_chunks):
            start = i * tokens_chunk_size
            end = min(start + tokens_chunk_size + self.token_overlap, z.shape[2])
            clip = self._decode_clip(z.index_select(2, torch.arange(start, end, device=z.device)))
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                    decoded_chunks.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded_chunks.append(overlap)

        dec = torch.cat(decoded_chunks, dim=2)

        if pad_tokens > 0:
            intra_tail = self.config.clip_length % temporal_ratio
            num_tokens_before_pad = z.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail
                if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0
                else temporal_ratio
                for k in range(pad_tokens)
            )
            dec = dec[:, :, :-pad_frames]
        return dec

    def load_weights(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Load the FP32 checkpoint into each parameter's dtype.

        Names map 1:1. The loader runs at ``tp_rank=0, tp_size=1`` and casts to the parameter's
        dtype, so the FP32 modules are promoted first.
        """
        from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint

        for name in self._keep_in_fp32_modules:
            getattr(self, name).to(torch.float32)
        # The loader raises on a checkpoint key it cannot find.
        mappings = {name: name for name in self.state_dict().keys()}
        state_dict = (
            SafetensorsCheckpoint(model_name_or_path)
            .load_sharded_pipelined(0, 1, self, mappings, device)
            .state_dict
        )
        self.load_state_dict(state_dict, strict=False, assign=True)


class _MiniMaxH3VaeExecutor(TiledVaeExecutor):
    """Smaller device gathers: at 768p on TP8 x CP8 a rank's DiT, conditioner and VAE weights fill
    22.8 of its 24 GB, and the shared 512 MiB gather chunk fails to allocate."""

    MAX_DEVICE_GATHER_BYTES = 128 * 1024 * 1024


def _channels_last(video: torch.Tensor) -> torch.Tensor:
    return video.permute(0, 2, 3, 4, 1)


class DistributedAutoencoderKLMiniMaxH3(NeuronAutoencoderKLMiniMaxH3):
    """``NeuronAutoencoderKLMiniMaxH3`` with its tiles spread over a VAE group (changes 8, 9).

    Every rank of the group calls ``decode`` and ``encode`` in lockstep. ``decode`` returns the
    video on group rank 0 and an empty tensor elsewhere; ``encode`` returns the latent on every
    rank.
    """

    # Optional: maps a ``(B, C, t, H, W)`` slice of the decoded video to ``(B, t, H, W, C)``
    # frames on the device; ``decode`` then returns the frames.
    frame_fn = None

    def init_distributed(
        self,
        group: dist.ProcessGroup | None = None,
        error_group: dist.ProcessGroup | None = None,
    ) -> None:
        self.distributed_executor = _MiniMaxH3VaeExecutor()
        if group is not None:
            self.distributed_executor.bind_group(group, error_group)

    def set_parallel_size(self, pp_size: int, mode: str = "tile") -> None:
        """Must be the group's size: the executor indexes one task bucket per group rank."""
        group_size = self.distributed_executor.world_size
        if pp_size != group_size:
            raise ValueError(
                f"vae_patch_parallel_size={pp_size} must equal the VAE group size {group_size}"
            )
        self.distributed_executor.set_parallel_size(pp_size, mode=mode)

    def is_distributed_enabled(self) -> bool:
        """Whether ``decode`` / ``encode`` spread their tiles over the VAE group: tile mode,
        tiling on, at least 2 ranks."""
        executor = getattr(self, "distributed_executor", None)
        return (
            executor is not None
            and dist.is_initialized()
            and getattr(self, "use_tiling", False)
            and executor.parallel_mode == "tile"
            and min(int(executor.parallel_size), int(executor.world_size)) > 1
        )

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], TileGridSpec]:
        """Upstream ``_decode``'s chunks times ``_decode_clip``'s tiles. Tasks carry the whole
        latent and indices, so each rank slices only its own tiles."""
        num_latent_frames = z.shape[2]
        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.config.token_drop

        num_tokens = num_latent_frames + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
        chunks = []
        for i in range(num_chunks):
            start = i * tokens_chunk_size
            end = min(start + tokens_chunk_size + self.token_overlap, z.shape[2])
            chunks.append(torch.arange(start, end, device=z.device))

        ratio = self.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            z.shape[-2] * ratio, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            z.shape[-1] * ratio, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )
        rows = [
            torch.arange(i_pos // ratio, i_pos // ratio + i_len // ratio, device=z.device)
            for i_pos, i_len in zip(y_indices, y_lengths)
        ]
        columns = [
            torch.arange(j_pos // ratio, j_pos // ratio + j_len // ratio, device=z.device)
            for j_pos, j_len in zip(x_indices, x_lengths)
        ]

        tiletask_list = []
        for chunk, frames in enumerate(chunks):
            for row, h_indices in enumerate(rows):
                for column, w_indices in enumerate(columns):
                    tiletask_list.append(
                        TileTask(
                            len(tiletask_list),
                            (chunk, row, column),
                            [z, frames, h_indices, w_indices],
                        )
                    )
        tile_spec = {
            "num_latent_frames": num_latent_frames,
            "pad_tokens": pad_tokens,
            "height_overlaps": y_overlaps,
            "width_overlaps": x_overlaps,
        }
        grid_spec = TileGridSpec(
            split_dims=(3, 4),
            grid_shape=(len(chunks), len(rows), len(columns)),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        z, frames, h_indices, w_indices = task.tensor
        tile = z.index_select(2, frames).index_select(-2, h_indices).index_select(-1, w_indices)
        return self._compiled_decoder(tile)

    def tile_merge(
        self,
        local_tile_tensor: torch.Tensor,
        meta_gather,
        grid_spec: TileGridSpec,
        tid_coord_map: dict,
    ) -> torch.Tensor | None:
        executor = self.distributed_executor
        num_chunks, num_rows, num_columns = grid_spec.grid_shape
        if isinstance(executor, _MiniMaxH3VaeExecutor):
            layout = executor.column_layout(num_chunks, num_columns)
            return self._merge_on_device(local_tile_tensor, grid_spec, layout)
        tiles = executor.gather_tiles(local_tile_tensor, meta_gather, grid_spec, tid_coord_map)
        if tiles is None:
            return None
        ts = grid_spec.tile_spec
        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.config.token_drop
        temporal_ratio = self.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio

        decoded_chunks = []
        overlap = None
        for i in range(num_chunks):
            clip = self._stitch_tiles(
                [
                    [tiles[(i, row, column)] for column in range(num_columns)]
                    for row in range(num_rows)
                ],
                ts["height_overlaps"],
                ts["width_overlaps"],
            )
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                    decoded_chunks.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded_chunks.append(overlap)

        dec = torch.cat(decoded_chunks, dim=2)

        if ts["pad_tokens"] > 0:
            dec = dec[:, :, : -self._pad_frames(ts["num_latent_frames"], ts["pad_tokens"])]
        return dec

    def _pad_frames(self, num_latent_frames: int, pad_tokens: int) -> int:
        """The trailing pixel frames upstream ``_decode`` cuts for ``pad_tokens`` repeated
        latent frames."""
        temporal_ratio = self.temporal_compression_ratio
        intra_tail = self.config.clip_length % temporal_ratio
        return sum(
            intra_tail
            if intra_tail and (num_latent_frames + k) % self.tokens_chunk_size == 0
            else temporal_ratio
            for k in range(pad_tokens)
        )

    def decode_tile_ranks(self, latent_shape: tuple[int, ...]) -> set[int]:
        """The ranks that hold tiles of a decode of ``latent_shape``."""
        _, grid_spec = self.tile_split(torch.empty(latent_shape, device="meta"))
        num_chunks, _, num_columns = grid_spec.grid_shape
        return {
            rank for rank, _ in self.distributed_executor.column_layout(num_chunks, num_columns)
        }

    def _merge_on_device(self, local_tile_tensor: torch.Tensor, grid_spec: TileGridSpec, layout):
        """Change 9: the video, or ``frame_fn``'s frames, on rank 0 and None elsewhere.

        ``layout[chunk * columns + column]`` is the ``(rank, slot)`` holding that column's tiles
        of the chunk, top to bottom; a rank without one only joins the collectives.
        """
        executor = self.distributed_executor
        rank = executor.rank
        ts = grid_spec.tile_spec
        num_chunks, num_rows, num_columns = grid_spec.grid_shape
        _, batch, channels, clip_frames, tile_h, tile_w = local_tile_tensor.shape
        device, dtype = local_tile_tensor.device, local_tile_tensor.dtype
        keep_h = [tile_h - overlap for overlap in ts["height_overlaps"]] + [tile_h]
        keep_w = [tile_w - overlap for overlap in ts["width_overlaps"]] + [tile_w]
        height, max_w = sum(keep_h), max(keep_w)

        # Upstream _decode's frames of a clip: its head, and with token_drop the tail that
        # cross-fades into the next clip's head (the last clip's ends the video).
        chunk_frames = self.tokens_chunk_size * self.temporal_compression_ratio
        pre = self.frame_pre_padding
        head = list(range(pre, min(chunk_frames, clip_frames)))
        tail = []
        if self.config.token_drop > 0:
            tail = list(range(chunk_frames + pre, min(2 * chunk_frames, clip_frames)))
        num_frames = num_chunks * len(head) + len(tail)
        num_frames -= self._pad_frames(ts["num_latent_frames"], ts["pad_tokens"])
        out_frames = [
            max(0, min(len(head) + len(tail) * (c == num_chunks - 1), num_frames - c * len(head)))
            for c in range(num_chunks)
        ]
        max_frames = max(out_frames)
        frame_fn = self.frame_fn or _channels_last
        probe = frame_fn(torch.zeros((batch, channels, 1, 1, 1), dtype=dtype))
        out_channels, out_dtype = probe.shape[-1], probe.dtype

        # This rank's (chunk, column)s, in slot order.
        runs = [run for run, (owner, _) in enumerate(layout) if owner == rank]
        num_slots = 1 + max(slot for _, slot in layout)
        chunks = [run // num_columns for run in runs]
        columns = [run % num_columns for run in runs]

        def stack_slots(per_run, dim):
            """The runs' tensors stacked on ``dim``, zero-padded to every rank's slot count."""
            stacked = torch.stack(per_run, dim=dim)
            pad = [0, 0] * (stacked.ndim - 1 - dim) + [0, num_slots - len(per_run)]
            return F.pad(stacked, pad)

        def tails_of(stitched):
            if not tail:
                return ()
            tails = [
                F.pad(column_video[:, :, len(head) :], (0, max_w - keep_w[column]))
                for column_video, column in zip(stitched, columns)
            ]
            return (stack_slots(tails, 0).permute(3, 0, 1, 2, 4, 5).contiguous(),)

        outputs = executor.stitch_columns(
            local_tile_tensor,
            layout,
            num_columns,
            ts["height_overlaps"],
            ts["width_overlaps"],
            self._blend,
            select=head + tail,
            epilogue=tails_of,
            key=(len(head), len(tail), max_w),
        )
        stitched, *tails = outputs if outputs is not None else ((),)

        # The previous chunk's tail, for the cross-fade.
        previous = ()
        if tail:
            if not tails:
                shape = (len(tail), num_slots, batch, channels, height, max_w)
                tails = [torch.zeros(shape, dtype=dtype, device=device)]
            # A run of the first chunk reads its own slot, unused.
            sources = [
                layout[run - num_columns] if run >= num_columns else (rank, 0) for run in runs
            ]
            previous = (executor.gather_slots(tails[0], sources or [(rank, 0)]),)

        if runs:

            def finish(*args):
                stitched, previous = args[: len(runs)], args[len(runs) :]
                blocks = []
                for slot, (chunk, column) in enumerate(zip(chunks, columns)):
                    video = stitched[slot][:, :, : len(head)]
                    if previous and chunk > 0:
                        overlap = previous[0][:, slot].permute(1, 2, 0, 3, 4)[..., : keep_w[column]]
                        video = self._blend(overlap, video, self.frame_overlap, dim=-3)
                    if chunk == num_chunks - 1:
                        video = torch.cat([video, stitched[slot][:, :, len(head) :]], dim=2)
                    frames = frame_fn(video[:, :, : out_frames[chunk]])
                    pad = (0, 0, 0, max_w - keep_w[column], 0, 0, 0, max_frames - out_frames[chunk])
                    blocks.append(F.pad(frames, pad))
                return stack_slots(blocks, 0).permute(2, 0, 1, 3, 4, 5).contiguous()

            # The cached graph keeps its first closure: the key names everything finish reads.
            key = (
                tuple(tuple(column.shape) for column in stitched),
                len(previous),
                tuple(zip(chunks, columns)),
                num_slots,
                tuple(out_frames),
                len(head),
                max_w,
                frame_fn,
            )
            blocks = executor.run_graph("finish", key, finish, *stitched, *previous)
        else:
            shape = (max_frames, num_slots, batch, height, max_w, out_channels)
            blocks = torch.zeros(shape, dtype=out_dtype, device=device)

        video = None
        if rank == 0:
            video = torch.empty(
                (batch, num_frames, height, sum(keep_w), out_channels), dtype=out_dtype
            )
        for start, pieces in executor.gather_columns(blocks, layout, num_columns, keep_w):
            if pieces is None:
                continue
            for c in range(num_chunks):
                end = min(start + pieces.shape[1], out_frames[c])
                if end > start:
                    # Only the real frames leave the device.
                    first = c * len(head) + start
                    video[:, first : first + end - start].copy_(
                        pieces[c, : end - start].transpose(0, 1)
                    )
        if video is None or self.frame_fn is not None:
            return video
        return video.permute(0, 4, 1, 2, 3)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        if self._compiled_decoder is None or not self.is_distributed_enabled():
            return super()._decode(z)
        return self.distributed_executor.execute(
            z,
            DistributedOperator(
                split=self.tile_split,
                exec=self.tile_exec,
                merge=self.tile_merge,
                # Change 9: tiles are dealt out by (chunk, column).
                balance=self.distributed_executor.balance_columns,
            ),
        )

    def encode_tile_split(self, x: torch.Tensor) -> tuple[list[TileTask], TileGridSpec]:
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            x.shape[-2], self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            x.shape[-1], self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )
        rows = [
            torch.arange(i_pos, i_pos + i_len, device=x.device)
            for i_pos, i_len in zip(y_indices, y_lengths)
        ]
        columns = [
            torch.arange(j_pos, j_pos + j_len, device=x.device)
            for j_pos, j_len in zip(x_indices, x_lengths)
        ]

        tiletask_list = []
        for row, h_indices in enumerate(rows):
            for column, w_indices in enumerate(columns):
                tiletask_list.append(
                    TileTask(len(tiletask_list), (row, column), [x, h_indices, w_indices])
                )
        ratio = self.spatial_compression_ratio
        tile_spec = {
            "height_overlaps": [overlap // ratio for overlap in y_overlaps],
            "width_overlaps": [overlap // ratio for overlap in x_overlaps],
        }
        grid_spec = TileGridSpec(
            split_dims=(3, 4),
            grid_shape=(len(rows), len(columns)),
            tile_spec=tile_spec,
            output_dtype=self.encoder_dtype,
        )
        return tiletask_list, grid_spec

    def encode_tile_exec(self, task: TileTask) -> torch.Tensor:
        x, h_indices, w_indices = task.tensor
        return self._compiled_encoder(x.index_select(-2, h_indices).index_select(-1, w_indices))

    def encode_tile_merge(
        self,
        local_tile_tensor: torch.Tensor,
        meta_gather,
        grid_spec: TileGridSpec,
        tid_coord_map: dict,
    ) -> torch.Tensor | None:
        tiles = self.distributed_executor.gather_tiles(
            local_tile_tensor, meta_gather, grid_spec, tid_coord_map
        )
        if tiles is None:
            return None
        ts = grid_spec.tile_spec
        num_rows, num_columns = grid_spec.grid_shape
        return self._stitch_tiles(
            [[tiles[(row, column)] for column in range(num_columns)] for row in range(num_rows)],
            ts["height_overlaps"],
            ts["width_overlaps"],
        )

    def _encode_clip(self, x: torch.Tensor) -> torch.Tensor:
        """The parent's tiled ``_encode_clip``, with the moments on every rank's host."""
        if self._compiled_encoder is None or not self.is_distributed_enabled():
            return super()._encode_clip(x)
        executor = self.distributed_executor
        moments = executor.execute(
            x,
            DistributedOperator(
                split=self.encode_tile_split,
                exec=self.encode_tile_exec,
                merge=self.encode_tile_merge,
            ),
        )
        # Rank 0 stitched the moments on its host, so they reach the other ranks as host tensors
        # over the group's Gloo twin: a device-group broadcast cannot send a host tensor.
        group = getattr(executor, "_error_group", None) or dist.group.WORLD
        src = dist.get_global_rank(group, 0)
        shape = torch.zeros(x.ndim, dtype=torch.int64)
        if executor.rank == 0:
            moments = moments.to(self.encoder_dtype)
            shape.copy_(torch.tensor(moments.shape))
        dist.broadcast(shape, src=src, group=group)
        if executor.rank != 0:
            moments = torch.empty(tuple(shape.tolist()), dtype=self.encoder_dtype)
        dist.broadcast(moments, src=src, group=group)
        return moments
