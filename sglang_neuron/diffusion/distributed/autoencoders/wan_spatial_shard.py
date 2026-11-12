# SPDX-License-Identifier: Apache-2.0
"""Exact spatially-sharded Wan VAE decode and encode with in-graph halo exchange.

The tile-parallel decode (``DistributedAutoencoderKLWan.tiled_decode``) splits a
frame into overlapping 256-px tiles and blends the seams: at 480x832 that is 12
tiles (so 4 of 16 ranks get no tile), redundant compute in the overlaps, a
host-visible gather + blend per chunk, and an output that differs from the
untiled decode at every seam.

This module instead splits the decoder's feature maps into an ``n_h x n_w`` grid
of disjoint blocks, one per rank, and makes every layer that reads across a block
boundary exact:

* a 3x3 spatial convolution (``WanCausalConv3d`` 3x3x3, the upsample ``NkiConv2d``
  3x3) first exchanges a 1-pixel halo with its neighbours in two all-gathers:
  width, then height on the width-padded tensor, so the corners arrive without a
  third exchange. Ranks on the frame border pad with zeros, which is exactly the
  convolution's own zero padding. The decoder and the encoder gather over the
  whole group, so every rank's graph names the same process group;
* the mid-block ``WanAttentionBlock`` (global over one frame's 60x104 latent)
  all-gathers the full frame, runs unchanged, and keeps its own block;
* everything else -- RMSNorm (over channels), SiLU, nearest-neighbour upsampling,
  the (3, 1, 1) temporal convolutions and every 1x1 convolution -- is local.

The result is the untiled decode, not an approximation of it. Every exchange is a
``torch.distributed._functional_collectives.all_gather_tensor`` *inside* the
compiled per-frame graph (the VAE group is registered in torch-neuronx's mesh registry, as
the tile executor's device gather does), so a latent frame is still one graph
launch per rank rather than ~30 graphs joined by eager collectives. Each rank's
causal cache covers only its own block, so per-rank resident state is ~1/N of the
single-device cache. A rank's grid position enters the graph only as a tensor
input (``SpatialShardContext.rank_tensor``), so every rank compiles the same graph.

The encoder (:class:`SpatialShardedWanEncoder`) is split the same way: its 3x3
convolutions and mid-block attention exchange like the decoder's, and its three
stride-2 downsamples (Wan's right/bottom zero pad, then a 3x3 stride-2
convolution) take a one-pixel right/bottom halo only, which is exact on blocks
whose sides are divisible by 8.

The grid is the caller's choice; 2x8 at 480x832 (16 cores at LNC 1) is the validated one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.distributed._functional_collectives as funcol

from sglang_neuron.diffusion.distributed.autoencoders.wan_streaming import (
    WanStreamingDecodeState,
    WanStreamingEncodeState,
)

#: neuronx-cc flags of the sharded decoder's graphs: the Wan VAE's set
#: (``autoencoder_kl_wan.WAN_I2V_VAE_COMPILER_ARGS``). The plugin's compile backend replaces
#: NEURON_CC_FLAGS with a graph's flags, so the -O level and scratchpad page are stated here.
DECODER_COMPILER_ARGS = (
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
)
#: The sharded encoder's: the decoder's set at -O3.
ENCODER_COMPILER_ARGS = (
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O3",
    "--hbm-scratchpad-page-size=2048",
)


def _default_all_gather(group):
    def gather(tensor: torch.Tensor) -> torch.Tensor:
        return funcol.all_gather_tensor(tensor.contiguous(), gather_dim=0, group=group)

    return gather


@dataclass
class SpatialShardContext:
    """One rank's place in an ``n_h x n_w`` block grid over the decoder's feature maps.

    ``rank = row * n_w + col``. ``all_gather(t)`` must return the group's tensors
    concatenated along dim 0 in rank order (``funcol.all_gather_tensor`` semantics);
    it is injectable so the exchange logic can be exercised off-device.

    Inside the compiled graphs the rank enters only as ``rank_tensor``, an ``int32`` tensor of
    shape ``(1,)`` holding ``rank`` (the ``rank_in_group_tensor`` convention of ``groups.py``):
    the halo neighbours, the border zeros and the attention block's own-block keep are one-hot
    masks built from it in the graph (no runtime-indexed gather), so every rank traces the same
    graph and the ranks share one compiled graph per variant. It stays 1-D: ``torch.compile``
    constant-folds a 0-D rank. ``rank`` (a Python int) is for host-side code only. The wrappers
    move ``rank_tensor`` to the VAE's device and hold it as a buffer of their graph module.
    """

    n_h: int
    n_w: int
    rank: int
    all_gather: Callable[[torch.Tensor], torch.Tensor] = field(repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.rank < self.world:
            raise ValueError(f"rank {self.rank} outside a {self.n_h}x{self.n_w} grid")
        if self.rank_tensor is None:
            self.rank_tensor = torch.tensor([self.rank], dtype=torch.int32)
        elif self.rank_tensor.shape != (1,) or self.rank_tensor.dtype != torch.int32:
            raise ValueError(
                "rank_tensor must be an int32 tensor of shape (1,); got "
                f"{self.rank_tensor.dtype} {tuple(self.rank_tensor.shape)}"
            )

    @classmethod
    def for_group(cls, group, n_h: int, n_w: int) -> SpatialShardContext:
        """Context for ``group``.

        A traced all-gather records its process group's name, so every rank exchanges over the
        whole group and the ranks compile one shared graph.

        The grid must span the whole world: the replica-group table registered for torch-neuronx
        covers exactly the grid's ranks.

        Raises:
            ValueError: The world size or ``group``'s size is not ``n_h * n_w``.
        """
        import torch.distributed as dist

        global_world = dist.get_world_size()
        if global_world != n_h * n_w:
            raise ValueError(
                f"a spatially-sharded VAE needs the world to be its {n_h}x{n_w} grid "
                f"({n_h * n_w} ranks); the world has {global_world} ranks"
            )
        world = dist.get_world_size(group)
        if world != n_h * n_w:
            raise ValueError(f"group of {world} ranks cannot form a {n_h}x{n_w} grid")
        return cls(n_h=n_h, n_w=n_w, rank=dist.get_rank(group), all_gather=_default_all_gather(group))

    @property
    def world(self) -> int:
        """Number of ranks in the grid, ``n_h * n_w``."""
        return self.n_h * self.n_w

    @property
    def row(self) -> int:
        """This rank's grid row (row-major rank order)."""
        return self.rank // self.n_w

    @property
    def col(self) -> int:
        """This rank's grid column (row-major rank order)."""
        return self.rank % self.n_w

    # -- geometry --------------------------------------------------------------
    def local_extent(self, height: int, width: int) -> tuple[int, int]:
        """The size of each rank's block of a ``height x width`` map.

        Args:
            height: Full map height.
            width: Full map width.

        Returns:
            ``(height // n_h, width // n_w)``.

        Raises:
            ValueError: The map does not divide evenly into the grid.
        """
        if height % self.n_h or width % self.n_w:
            raise ValueError(
                f"{height}x{width} latent does not divide into a {self.n_h}x{self.n_w} grid"
            )
        return height // self.n_h, width // self.n_w

    def local_slice(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's block of a full ``[..., H, W]`` tensor (host side; in a compiled graph
        use :meth:`select_block`)."""
        h, w = self.local_extent(x.shape[-2], x.shape[-1])
        return x.narrow(-2, self.row * h, h).narrow(-1, self.col * w, w)

    def local_slice_haloed(self, x: torch.Tensor, radius: int = 1) -> torch.Tensor:
        """This rank's block of a full ``[..., H, W]`` tensor plus a ``radius`` halo of the
        neighbouring pixels on every side (zeros beyond the frame), without an exchange."""
        h, w = self.local_extent(x.shape[-2], x.shape[-1])
        r = radius
        padded = torch.nn.functional.pad(x, (r, r, r, r))
        return padded.narrow(-2, self.row * h, h + 2 * r).narrow(-1, self.col * w, w + 2 * r)

    # -- collectives -----------------------------------------------------------
    def _gather_ranks(self, x: torch.Tensor) -> torch.Tensor:
        """``[world, *x.shape]``: every rank's ``x``."""
        return self.all_gather(x.unsqueeze(0)).reshape(self.world, *x.shape)

    def gather_full(self, x: torch.Tensor) -> torch.Tensor:
        """Reassemble the full ``[..., H, W]`` tensor from every rank's block."""
        parts = self._gather_ranks(x)  # [world, ..., h, w], rank (= row-major grid) order
        lead = x.shape[:-2]
        h, w = x.shape[-2], x.shape[-1]
        parts = parts.reshape(self.n_h, self.n_w, *lead, h, w)
        nd = len(lead)
        # [n_h, n_w, *lead, h, w] -> [*lead, n_h, h, n_w, w]
        order = (*range(2, 2 + nd), 0, 2 + nd, 1, 3 + nd)
        return parts.permute(*order).reshape(*lead, self.n_h * h, self.n_w * w)

    # ``rank`` as a graph input; built on the host from ``rank`` when not given.
    rank_tensor: torch.Tensor | None = field(default=None, repr=False)

    # -- in-graph rank arithmetic ----------------------------------------------
    #
    # Selections by rank use a static one-hot mask, never a runtime index: the mask
    # zeroes every other slot and the slots are summed, which returns the selected
    # values exactly (one term, the rest +0). Non-selected NaN / inf never reach the
    # result. The one difference from an indexed copy: a selected -0 comes back as +0.
    def _rank64(self) -> torch.Tensor:
        """``rank_tensor`` as int64, shape (1,)."""
        return self.rank_tensor.to(torch.int64)

    def _grid_coord(self, width: bool) -> torch.Tensor:
        """This rank's column (``width``) or row as a (1,) int64 tensor."""
        r = self._rank64()
        if width:
            return torch.remainder(r, self.n_w)
        return torch.div(r, self.n_w, rounding_mode="floor")

    @staticmethod
    def _masked_select(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        """The slot of ``x`` along ``dim`` where the bool ``mask`` (one entry per slot) is
        true, or +0 where no entry is; summed in at least float32 and cast back (exact)."""
        dim = dim % x.ndim
        mask = mask.reshape(-1, *([1] * (x.ndim - dim - 1)))
        picked = torch.where(mask, x, x.new_zeros(()))
        return picked.sum(dim, dtype=torch.promote_types(x.dtype, torch.float32)).to(x.dtype)

    def select_block(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's block of a full ``[..., H, W]`` tensor, selected by ``rank_tensor``.

        The in-graph form of :meth:`local_slice` (same values, see the one-hot note
        above): ``[..., H, W]`` is viewed as ``[..., n_h, h, n_w, w]`` and the grid axes
        are reduced with this rank's row / column one-hot masks.
        """
        h, w = self.local_extent(x.shape[-2], x.shape[-1])
        lead = x.shape[:-2]
        if self.n_h > 1:
            row = torch.arange(self.n_h, device=x.device) == self._grid_coord(False)
            x = self._masked_select(x.reshape(*lead, self.n_h, h, x.shape[-1]), row, -3)
        if self.n_w > 1:
            col = torch.arange(self.n_w, device=x.device) == self._grid_coord(True)
            x = self._masked_select(x.reshape(*lead, h, self.n_w, w), col, -2)
        return x

    def _neighbour_masks(self, width: bool, device):
        """One-hot masks of the previous / next neighbour along the grid's width or height.

        Over the slots of the gathered tensor's leading dim (the whole group's ranks). A side
        without a neighbour (the frame border) gets an all-false mask, i.e. zeros.
        """
        index = self._grid_coord(width)
        count = self.n_w if width else self.n_h
        base, step = self._rank64(), 1 if width else self.n_w
        slots = torch.arange(self.world, device=device)
        before = (slots == base - step) & (index > 0)
        after = (slots == base + step) & (index < count - 1)
        return before, after

    def halo(self, x: torch.Tensor, radius: int = 1) -> torch.Tensor:
        """Pad the last two dims by ``radius`` with the neighbours' pixels (zeros at the border).

        Two all-gathers: width first, then height on the width-padded tensor (so the
        corner pixels arrive without a third exchange).
        """
        # TODO: the halo exchange is an all-gather because collective_permute does not lower
        # reliably on Neuron yet; switch to a neighbour-only exchange once collective_permute
        # works.
        if self.n_w > 1:
            x = self._exchange(x, radius, dim=-1)
        else:
            x = torch.nn.functional.pad(x, (radius, radius))
        if self.n_h > 1:
            x = self._exchange(x, radius, dim=-2)
        else:
            x = torch.nn.functional.pad(x, (0, 0, radius, radius))
        return x

    def halo_after(self, x: torch.Tensor, radius: int = 1) -> torch.Tensor:
        """Pad only the right and bottom by ``radius`` with the neighbours' pixels.

        The encoder's stride-2 downsample (Wan's right/bottom ``ZeroPad2d`` then a 3x3
        stride-2 convolution) reads one pixel past an even block's right and bottom edges
        and nothing before them. Width first, then height on the width-padded tensor, so
        the bottom-right corner arrives without a third exchange. Zeros on the frame
        border.
        """
        if self.n_w > 1:
            x = self._exchange_after(x, radius, dim=-1)
        else:
            x = torch.nn.functional.pad(x, (0, radius))
        if self.n_h > 1:
            x = self._exchange_after(x, radius, dim=-2)
        else:
            x = torch.nn.functional.pad(x, (0, 0, 0, radius))
        return x

    def _exchange_after(self, x, radius, *, dim):
        """Append the following neighbour's leading edge along ``dim`` (zeros at the border)."""
        edge = x.narrow(dim, 0, radius)
        line = self._gather_ranks(edge)  # [world, ...]
        _, after = self._neighbour_masks(dim == -1, line.device)
        # The next neighbour's leading edge; +0 on the frame border.
        return torch.cat([x, self._masked_select(line, after, 0)], dim=dim)

    def _exchange(self, x, radius, *, dim):
        size = x.shape[dim]
        edges = torch.stack([x.narrow(dim, 0, radius), x.narrow(dim, size - radius, radius)])
        line = self._gather_ranks(edges)  # [world, 2, ...]
        before, after = self._neighbour_masks(dim == -1, line.device)
        # Neighbour before me sends its trailing edge; neighbour after me its leading edge.
        # +0 on the frame border.
        before = self._masked_select(line[:, 1], before, 0)
        after = self._masked_select(line[:, 0], after, 0)
        return torch.cat([before, x, after], dim=dim)


def _place_rank_tensor(ctx: SpatialShardContext, device) -> None:
    """Move ``ctx.rank_tensor`` to ``device`` (the graph's device; a no-op on the CPU path)."""
    ctx.rank_tensor = ctx.rank_tensor.to(device)


def install_spatial_shard(module: torch.nn.Module, ctx: SpatialShardContext) -> None:
    """Point every boundary-crossing layer of a Wan decoder or encoder at ``ctx``.

    Installing records the sharded convolutions' kernel choice: a sharded convolution always
    runs the compiler's convolution on the halo-padded block, whatever its NKI gate decided
    when the filters were packed.

    Installing also marks ``module`` (``_spatial_shard_installed``) so that the VAE's
    unsharded decode / encode entry points refuse to run on it while it is sharded; the
    sharded wrappers' graphs call the module directly.

    Args:
        module: A ``WanDecoder3d`` or ``WanEncoder3d`` (or any module containing its layers).
        ctx: This rank's grid context.
    """
    from sglang_neuron.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        NkiConv2d,
        WanAttentionBlock,
        WanCausalConv3d,
    )
    from sglang_neuron.diffusion.kernels.kernel_choice import record_kernel_choice

    kinds = set()
    for layer in module.modules():
        if isinstance(layer, WanCausalConv3d) and (
            layer.kernel_size[1] > 1 or layer.kernel_size[2] > 1
        ):
            layer._spatial_shard = ctx
            kinds.add("vae_conv3d")
        elif isinstance(layer, NkiConv2d) and tuple(layer.kernel_size) != (1, 1):
            layer._spatial_shard = ctx
            kinds.add("vae_conv2d")
        elif isinstance(layer, WanAttentionBlock):
            layer._spatial_shard = ctx
    module._spatial_shard_installed = True
    for name in sorted(kinds):
        record_kernel_choice(
            name, False, "spatially sharded: compiler conv on the halo-padded block", by_design=True
        )


def register_shard_replica_groups(ctx: SpatialShardContext, group) -> None:
    """Expose a shard group to torch-neuronx's collective lowering (its mesh registry).

    The grid spans the whole world (checked by :meth:`SpatialShardContext.for_group`), so the
    group's table is the single group of every rank.
    """
    from sglang_neuron.diffusion.neuron_api import register_process_group_replica_groups

    register_process_group_replica_groups(group.group_name, [list(range(ctx.world))])


class ShardedWanDecoderFrame(torch.nn.Module):
    """One latent frame of a spatially-sharded decode, as one graph launch per rank.

    Takes the *full* latent frame (every rank has it), decodes this rank's block
    with its own block-sized cache, and ends with an all-gather that returns the
    full decoded frame -- so no separate gather/merge stage runs per chunk.
    """

    def __init__(self, post_quant_conv, decoder, ctx: SpatialShardContext):
        super().__init__()
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder
        self.ctx = ctx
        # The context's rank tensor (the same tensor the sharded layers read), registered so
        # torch.compile lifts it as a graph input rather than a constant.
        self.register_buffer("rank_tensor", ctx.rank_tensor, persistent=False)

    def forward(self, z_frame, *flat_cache, first_chunk):
        x = self.post_quant_conv(self.ctx.select_block(z_frame))
        feat_cache = list(flat_cache)
        out = self.decoder(x, feat_cache=feat_cache, feat_idx=[0], first_chunk=first_chunk)
        return self.ctx.gather_full(out), *feat_cache


class SpatialShardedWanDecoder:
    """Streaming decode of a ``NeuronAutoencoderKLWan`` split over an ``n_h x n_w`` rank grid.

    Every rank of ``group`` calls :meth:`decode_chunk` in lockstep with the same
    full latent chunk; the group's rank 0 returns the full decoded chunk on the host
    (the others get an empty tensor). Session state is a ``WanStreamingDecodeState`` holding this
    rank's block-sized cache.
    """

    def __init__(self, vae, group, n_h: int, n_w: int, *, ctx: SpatialShardContext | None = None):
        self.vae = vae
        # The decoder uses whole-group halo all-gathers (validated on 2x8 and 2x4).
        self.ctx = ctx if ctx is not None else SpatialShardContext.for_group(group, n_h, n_w)
        self.group = group
        _place_rank_tensor(self.ctx, vae.post_quant_conv.weight.device)
        install_spatial_shard(vae.decoder, self.ctx)
        self._frame = ShardedWanDecoderFrame(vae.post_quant_conv, vae.decoder, self.ctx)
        self._graphs: dict[bool, Callable] = {}

    def register_replica_groups(self) -> None:
        """Expose the VAE group to torch-neuronx's collective lowering (as the tile executor does)."""
        register_shard_replica_groups(self.ctx, self.group)

    def compile(self, *args, **compiler_kwargs) -> None:
        """Compile the per-frame graphs (opening frame and steady state).

        The halo-padded convolutions take the compiler's convolution, which is faster than
        the NKI conv3d kernel at the per-rank block sizes.
        """
        self.register_replica_groups()
        base = compiler_kwargs.get("options") or {}
        name = base.get("model_name", "wan_vae_shard")
        for first in (True, False):
            kwargs = {
                **compiler_kwargs,
                "options": {**base, "model_name": f"{name}_{'first' if first else 'rest'}"},
            }
            self._graphs[first] = torch.compile(self._frame, *args, **kwargs)

    def _graph(self, first: bool):
        return self._graphs.get(first, self._frame)

    def new_decode_state(self, session_id: str) -> WanStreamingDecodeState:
        """Create an empty decoder state for a new streaming session.

        Args:
            session_id: Caller-chosen session name.

        Returns:
            A ``WanStreamingDecodeState`` that will hold this rank's block-sized cache.
        """
        return WanStreamingDecodeState(session_id=session_id)

    def decode_chunk(self, latent: torch.Tensor, state):
        """Decode one chunk of latent frames as the continuation of ``state``'s session.

        Lifecycle: ``new_decode_state`` -> ``decode_chunk`` per chunk -> ``release``. Video
        output only; only the group's rank 0 returns pixels, and the state advances only when
        the whole chunk succeeds.

        Collective: every rank of the group must call this in lockstep with the same full
        latent chunk; each latent frame is one launch of the compiled graph (the same graph on
        every rank) whose halo exchanges and final gather are all-gathers over the group.

        A failure on any rank mid-chunk breaks the group's lockstep: each rank commits only its
        own state, and the other ranks block in the next all-gather. Treat a failed chunk as
        ending the session on every rank.

        Args:
            latent: The full latent chunk ``[1, z_dim, T, h, w]`` (already de-normalized),
                identical on every rank.
            state: This rank's ``WanStreamingDecodeState`` for the session; it advances only
                when the whole chunk succeeds.

        Returns:
            The decoded chunk ``[1, 3, T', H, W]`` on the host, clamped to [-1, 1], with
            ``T' = 1 + 4 (T - 1)`` on the session's first chunk and ``4 T`` afterwards, on the
            group's rank 0; an empty tensor on the other ranks.

        Raises:
            ValueError: The latent is not one session's ``[1, z_dim, T, h, w]``, or ``h x w``
                does not divide into the grid.
            NotImplementedError: The VAE is patchified.
        """
        import time

        vae = self.vae
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != vae.config.z_dim:
            raise ValueError(
                f"streaming decode takes one session's latent [1, {vae.config.z_dim}, T, H, W]; "
                f"got {tuple(latent.shape)}."
            )
        if vae.config.patch_size is not None:
            raise NotImplementedError("spatially-sharded decode does not support patchified VAEs")
        height, width = int(latent.shape[3]), int(latent.shape[4])
        local_h, local_w = self.ctx.local_extent(height, width)
        state.check_spatial_shape((height, width))

        start = time.perf_counter()
        # Cast on the host before the upload, not on the device after it.
        latent = latent.to(vae.dtype).contiguous().to(vae.post_quant_conv.weight.device)
        if state.feat_map is None:
            probe = latent.new_zeros(1, latent.shape[1], 1, local_h, local_w)
            feat_map = vae._init_feat_cache(probe)
        else:
            feat_map = state.feat_map
        frames_decoded = state.frames_decoded
        keep = self.ctx.rank == 0
        frames_out = []
        for index in range(latent.shape[2]):
            frame = torch.index_select(latent, 2, torch.tensor([index], device=latent.device))
            first = frames_decoded == 0
            result = self._graph(first)(frame, *feat_map, first_chunk=first)
            feat_map = list(result[1:])
            if keep:
                frames_out.append(result[0].cpu())
            frames_decoded += 1

        out = torch.clamp(torch.cat(frames_out, dim=2), -1.0, 1.0) if keep else torch.empty(0)
        state.feat_map = feat_map
        state.spatial_shape = (height, width)
        state.frames_decoded = frames_decoded
        state.chunks_done += 1
        state.chunk_seconds.append(time.perf_counter() - start)
        return out


# ------------------------------------------------------------------------------ encoder


class ShardedWanEncoderChunk(torch.nn.Module):
    """One temporal step of a spatially-sharded encode (1 or 4 pixel frames), one graph launch.

    Takes this rank's *block* of the pixel frames with a one-pixel halo already attached
    (``local_slice_haloed``: ``conv_in``'s input comes from the host, which holds the full
    frame, so each rank's block is sliced there with its halo and ``conv_in`` needs no
    exchange), encodes it with its own block-sized cache (every other 3x3 convolution exchanges
    a halo, the stride-2 downsamples a right/bottom halo, the mid-block attention gathers the
    frame) and ends with an all-gather that returns the full posterior moments of the latent
    frame on every rank.
    """

    def __init__(self, encoder, quant_conv, ctx: SpatialShardContext):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.ctx = ctx
        # As ShardedWanDecoderFrame: a graph input, not a per-rank constant.
        self.register_buffer("rank_tensor", ctx.rank_tensor, persistent=False)

    def forward(self, x_block, *flat_cache, first_chunk):
        feat_cache = list(flat_cache)
        out = self.encoder(x_block, feat_cache=feat_cache, feat_idx=[0], first_chunk=first_chunk)
        out = self.quant_conv(out)
        return self.ctx.gather_full(out), *feat_cache


class SpatialShardedWanEncoder:
    """Streaming encode of a ``NeuronAutoencoderKLWan`` split over an ``n_h x n_w`` rank grid.

    The encoder analogue of :class:`SpatialShardedWanDecoder`, and exact in the same way: the
    math is the unsharded encode's (every convolution and the channel reductions see the same
    values on a block). Every rank of ``group`` calls
    :meth:`encode_chunk` in lockstep; each passes either the full pixel chunk or (with
    ``block=True``) its own block of it, and gets the full posterior moments
    ``[1, 2 * z_dim, T, h, w]`` back on the device. The session state is a
    ``WanStreamingEncodeState`` holding this rank's block-sized cache.

    Every block side must be divisible by the encoder's spatial compression (8 for Wan 2.1):
    the stride-2 downsamples need even blocks at every level.
    """

    def __init__(
        self,
        vae,
        group,
        n_h: int,
        n_w: int,
        *,
        ctx: SpatialShardContext | None = None,
    ):
        if n_w == 1 and n_h > 1:
            # n_h x 1 strip grids are not validated on Neuron.
            raise ValueError(
                f"a {n_h}x1 encoder grid is not supported on Neuron; split the width too"
            )
        self.vae = vae
        # Whole-group halo all-gathers, as the decoder: one graph shared by every rank.
        self.ctx = ctx if ctx is not None else SpatialShardContext.for_group(group, n_h, n_w)
        self.group = group
        _place_rank_tensor(self.ctx, vae.quant_conv.weight.device)
        install_spatial_shard(vae.encoder, self.ctx)
        # conv_in reads the host-haloed input block instead of exchanging.
        vae.encoder.conv_in._spatial_shard = None
        vae.encoder.conv_in._input_haloed = True
        self._step = ShardedWanEncoderChunk(vae.encoder, vae.quant_conv, self.ctx)
        self._graphs: dict[bool, Callable] = {}

    def compile(self, *args, **compiler_kwargs) -> None:
        """Compile the opening-frame (T_in=1) and continuation (T_in=4) graphs."""
        register_shard_replica_groups(self.ctx, self.group)
        base = compiler_kwargs.get("options") or {}
        name = base.get("model_name", "wan_vae_enc_shard")
        for first in (True, False):
            kwargs = {
                **compiler_kwargs,
                "options": {**base, "model_name": f"{name}_{'first' if first else 'rest'}"},
            }
            self._graphs[first] = torch.compile(self._step, *args, **kwargs)

    def _graph(self, first: bool):
        return self._graphs.get(first, self._step)

    def block_extent(self, height: int, width: int) -> tuple[int, int]:
        """This rank's pixel block size; rejects grids the downsamples cannot split exactly."""
        h, w = self.ctx.local_extent(height, width)
        ratio = int(self.vae.spatial_compression_ratio)
        if h % ratio or w % ratio:
            raise ValueError(
                f"{height}x{width} on a {self.ctx.n_h}x{self.ctx.n_w} grid gives {h}x{w} "
                f"blocks; every block side must be divisible by {ratio}"
            )
        return h, w

    def input_extent(self, height: int, width: int) -> tuple[int, int]:
        """The per-rank input block: this rank's block plus the one-pixel halo."""
        h, w = self.block_extent(height, width)
        return h + 2, w + 2

    def new_encode_state(self, session_id: str) -> WanStreamingEncodeState:
        """Create an empty encoder state for a new streaming session.

        Args:
            session_id: Caller-chosen session name.

        Returns:
            A ``WanStreamingEncodeState`` that will hold this rank's block-sized cache.
        """
        return WanStreamingEncodeState(session_id=session_id)

    def _init_cache(self, x_block):
        """Block-sized zero cache; ``conv_in``'s slot holds haloed input frames."""
        h, w = x_block.shape[-2] - 2, x_block.shape[-1] - 2
        cache = self.vae._init_enc_feat_cache(x_block[..., 1 : 1 + h, 1 : 1 + w])
        cache[0] = x_block.new_zeros(cache[0].shape[:3] + tuple(x_block.shape[-2:]))
        return cache

    def encode_chunk(
        self, x: torch.Tensor, state, *, block: bool = False, full_size=None
    ) -> torch.Tensor:
        """Encode one chunk of pixel frames as the continuation of ``state``'s session.

        Collective: every rank of the group must call this in lockstep; each encoder step
        is one launch of the compiled graph (the same graph on every rank) whose halo
        exchanges and final gather are all-gathers over the group.

        A failure on any rank mid-chunk breaks the group's lockstep: each rank commits only its
        own state, and the other ranks block in the next all-gather. Treat a failed chunk as
        ending the session on every rank.

        Args:
            x: The full pixel chunk ``[1, 3, F, H, W]`` in [-1, 1] (``block=False``, sliced
                here), or this rank's haloed block ``[1, 3, F, H / n_h + 2, W / n_w + 2]``
                (``block=True``, see :meth:`input_extent` and
                ``SpatialShardContext.local_slice_haloed``). ``F`` is ``1 + 4k`` on the
                session's first chunk and ``4k`` afterwards.
            state: This rank's ``WanStreamingEncodeState``; it advances only when the whole
                chunk succeeds.
            block: ``x`` is this rank's haloed block rather than the full frame.
            full_size: ``(H, W)`` of the full frame; required with ``block=True``.

        Returns:
            The full posterior moments ``[1, 2 * z_dim, T, h, w]`` on the device, on every
            rank, with ``T = 1 + k`` on the first chunk and ``k`` afterwards.

        Raises:
            ValueError: Wrong frame count, block shape or grid split.
            NotImplementedError: The VAE is patchified.
        """
        import time

        vae = self.vae
        if x.ndim != 5 or x.shape[0] != 1:
            raise ValueError(f"streaming encode takes [1, 3, F, H, W]; got {tuple(x.shape)}.")
        if vae.config.patch_size is not None:
            raise NotImplementedError("spatially-sharded encode does not support patchified VAEs")
        num_frames = int(x.shape[2])
        first_chunk = state.frames_encoded == 0
        if num_frames == 0 or (num_frames - 1 if first_chunk else num_frames) % 4:
            raise ValueError(
                f"a {'first' if first_chunk else 'continuation'} encode chunk needs "
                f"{'1 + 4k' if first_chunk else '4k'} pixel frames; got {num_frames}."
            )
        if block:
            if full_size is None:
                raise ValueError("block=True needs full_size=(H, W)")
            height, width = (int(v) for v in full_size)
            if tuple(x.shape[-2:]) != self.input_extent(height, width):
                raise ValueError(f"block {tuple(x.shape[-2:])} does not match {height}x{width}")
        else:
            height, width = int(x.shape[3]), int(x.shape[4])
            self.block_extent(height, width)
            x = self.ctx.local_slice_haloed(x, 1)
        state.check_spatial_shape((height, width))

        start = time.perf_counter()
        # Cast on the host before the upload, not on the device after it.
        x = x.to(vae.dtype).contiguous().to(vae.quant_conv.weight.device)
        feat_map = state.feat_map if state.feat_map is not None else self._init_cache(x)
        frames_encoded = state.frames_encoded
        outs = []
        offset = 0
        while offset < num_frames:
            span = 1 if frames_encoded == 0 else 4
            idx = torch.arange(offset, offset + span, device=x.device)
            result = self._graph(frames_encoded == 0)(
                torch.index_select(x, 2, idx), *feat_map, first_chunk=(frames_encoded == 0)
            )
            outs.append(result[0])
            feat_map = list(result[1:])
            frames_encoded += span
            offset += span
        out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]

        state.feat_map = feat_map
        state.spatial_shape = (height, width)
        state.frames_encoded = frames_encoded
        state.chunks_done += 1
        state.chunk_seconds.append(time.perf_counter() - start)
        return out
