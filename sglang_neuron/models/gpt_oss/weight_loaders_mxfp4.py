# SPDX-License-Identifier: Apache-2.0
"""Checkpoint loaders of the GPT-OSS MXFP4 graphs: each reads this rank's shard of a checkpoint tensor and lays it
out as the kernels read it. Most tensors with a hidden dimension hold it padded and shuffled (_shuffle_hidden_dim); the
MXFP4 expert tiles are repacked four FP4 values per uint16 and tiled to the MoE kernels' access pattern."""


import torch

from sglang_neuron.ops.moe.rmsnorm_mx_prefill import swizzle_h_index
from sglang_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    ShardSpec,
    get_shard,
    get_shard_deinterleaved,
    pad_to_shape,
)

PMAX = 128
# the MX quantization block: Q_HEIGHT rows of Q_WIDTH values
Q_WIDTH = 4
Q_HEIGHT = 8


def o_proj_weight_loader(
    shard_size: int,
    expected_shard_shape: tuple[int, ...],
    num_shards: int,
) -> SafetensorsWeightLoader:
    """o_proj: the checkpoint's [H, D] sharded along D and transposed to [D_shard, H_padded], H shuffled."""
    inner_loader = shuffling_weight_loader(
        shuffle_dim=0,
        expected_shard_shape=(
            expected_shard_shape[1],
            expected_shard_shape[0],
        ),  # [H, D_shard]
        shard_spec=ShardSpec(dim=1, size=shard_size, num_shards=num_shards),
    )

    def transform(slices: list, rank: int):
        return inner_loader.load(slices, rank).T

    return SafetensorsWeightLoader(transform=transform)


def shuffling_weight_loader(
    shuffle_dim: int,
    expected_shard_shape: tuple[int],
    shard_spec: ShardSpec | None = None,
    scale: float | None = None,
) -> SafetensorsWeightLoader:
    """Shard (``shard_spec``, none when unset), pad to ``expected_shard_shape`` and shuffle ``shuffle_dim``, divided by
    ``scale`` where set (a bias the ranks' all-reduce would otherwise count ``scale`` times). The shard is taken from
    another dimension than the shuffled one; no GPT-OSS parameter shards its shuffled dimension."""
    if shard_spec and shard_spec.dim == shuffle_dim:
        raise ValueError(f"shuffling_weight_loader shards dimension {shuffle_dim}, the one it shuffles: shard another "
                         "dimension")

    def transform(slices: list, rank: int):
        (slice_obj,) = slices
        if shard_spec:
            tensor = get_shard(slice_obj, shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank)
        else:
            tensor = slice_obj[:]

        if scale is not None:
            tensor = tensor / scale

        tensor = pad_to_shape(tensor, expected_shard_shape)

        tensor = _shuffle_hidden_dim(tensor, dim=shuffle_dim)

        return tensor

    return SafetensorsWeightLoader(transform=transform)


def decode_router_weight_loader(num_experts: int, hidden_size: int) -> SafetensorsWeightLoader:
    """The decode router weight: the checkpoint's [E, H], H shuffled, held as the [H, E] moe_block_tkg reads, so the
    decode graph does not transpose it every step."""
    shuffled = shuffling_weight_loader(shuffle_dim=1, expected_shard_shape=(num_experts, hidden_size))

    def transform(slices: list, rank: int):
        return shuffled.load(slices, rank).t()

    return SafetensorsWeightLoader(transform=transform)


def swizzle_router_weight_loader(
    expected_shard_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """The prefill router weight: the checkpoint's [E, H], H padded and in rmsnorm_mx_prefill's swizzled order, so that
    its transpose is the [H, E] the kernel reads."""

    def transform(slices: list, rank: int):
        (router,) = slices
        tensor = pad_to_shape(router[:], expected_shard_shape)
        return tensor.index_select(1, swizzle_h_index(tensor.shape[1], tensor.device)).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def natural_hidden_weight_loader(
    expected_shard_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """A per-hidden weight padded but in the checkpoint's natural order: the prefill norm's gamma, which
    rmsnorm_mx_prefill indexes by natural hidden index."""

    def transform(slices: list, rank: int):
        (weight,) = slices
        return pad_to_shape(weight[:], expected_shard_shape)

    return SafetensorsWeightLoader(transform=transform)


def _qkv_shards(slices: list, rank: int, q_shard_size: int, kv_shard_size: int, num_kv_replicas: int):
    """This rank's Q, K and V rows. With more ranks than KV heads, ``num_kv_replicas`` consecutive ranks share one."""
    q, k, v = slices
    kv_rank = rank // num_kv_replicas
    return (q[rank * q_shard_size:(rank + 1) * q_shard_size],
            k[kv_rank * kv_shard_size:(kv_rank + 1) * kv_shard_size],
            v[kv_rank * kv_shard_size:(kv_rank + 1) * kv_shard_size])


def fused_qkv_shuffling_weight_loader(q_shard_size: int, kv_shard_size: int, expected_shard_shape: tuple[int, ...],
                                      num_kv_replicas: int) -> SafetensorsWeightLoader:
    """The fused QKV weight: this rank's Q [q, H], K and V [kv, H] rows, transposed and concatenated to
    [H_padded, q + 2 kv] with H shuffled."""

    def transform(slices: list, rank: int):
        q, k, v = _qkv_shards(slices, rank, q_shard_size, kv_shard_size, num_kv_replicas)
        tensor = pad_to_shape(torch.cat([q.T, k.T, v.T], dim=1), expected_shard_shape)
        return _shuffle_hidden_dim(tensor, dim=0)

    return SafetensorsWeightLoader(transform=transform)


def fused_qkv_bias_loader(q_shard_size: int, kv_shard_size: int, num_kv_replicas: int) -> SafetensorsWeightLoader:
    """The fused QKV bias: this rank's Q, K and V entries concatenated."""

    def transform(slices: list, rank: int):
        return torch.cat(_qkv_shards(slices, rank, q_shard_size, kv_shard_size, num_kv_replicas), dim=0)

    return SafetensorsWeightLoader(transform=transform)


def mxfp4_gate_up_blocks_loader(
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """A loader for the gate_up_proj MXFP4 blocks: checkpoint ``[E, 2I, H // 32, 16]`` (two FP4 values per byte) to the
    kernel's ``[E, 128_H, 2, H // 512, TP * I]`` (four per uint16). It shards (``shard_spec``) and pads
    (``expected_shape``), splits the interleaved gate and up rows (even and odd along 2I), transposes H and I, packs FP4
    x2 into x4, tiles for the kernel's access pattern, permutes and concatenates gate and up."""

    def transform(slices: list, rank: int):
        (checkpoint,) = slices

        # Original Checkpoint - [E, 2I, H/32, 16[x2]]

        # Deinterleave + Shard - [E, 2I/TP, H/32, 16[x2]]
        # - Original checkpoint is stored as [gate0, up0, gate1, up1]
        blocks_gate, blocks_up = get_shard_deinterleaved(
            checkpoint, shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank
        )

        # Pad and Concat - [E, Pad(2I/TP), Pad(H/32), 16[x2]]
        per_tensor_shape = [
            expected_shape[0],
            expected_shape[1] // 2,
            expected_shape[2],
            expected_shape[3],
        ]
        blocks_gate = pad_to_shape(blocks_gate, per_tensor_shape)
        blocks_up = pad_to_shape(blocks_up, per_tensor_shape)
        blocks = torch.cat([blocks_gate, blocks_up], dim=1)

        # View (remains contiguous) - [E, Pad(2I/TP), Pad(H/2[x2])]
        blocks = blocks.view(*blocks.shape[:2], -1)

        # Pack - [E, Pad(2I/TP), Pad(H/4[x4])]
        # - the kernels read four FP4 values to a word; the checkpoint packs two
        blocks = _pack_fp4_x4_uint16(blocks)

        # Transpose - [E, Pad(H/4), Pad(2I/TP)[x4]]
        blocks = blocks.transpose(1, 2)

        # Tile Weight - [E, Pad(H/4), Pad(2I/TP)[x4]]
        blocks = _tile_gate_up_blocks(blocks)

        return blocks.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mxfp4_gate_up_scale_loader(
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """A loader for the gate_up_proj MX scales: checkpoint ``[E, 2I, H // 32]``, sharded and padded to
    ``expected_shape`` ``[E, 2I_shard, H // 32]``, to the kernel's ``[E, 16_H, 2, H // 512, I_shard]``."""

    def transform(slices: list, rank: int):
        (checkpoint,) = slices

        # Original Checkpoint - [E, 2I, H/32]

        # Deinterleave + Shard
        scales_gate, scales_up = get_shard_deinterleaved(
            checkpoint, shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank
        )

        # Pad each separately then concat
        per_tensor_shape = (
            expected_shape[0],
            expected_shape[1] // 2,
            expected_shape[2],
        )
        scales_gate = pad_to_shape(scales_gate, per_tensor_shape)
        scales_up = pad_to_shape(scales_up, per_tensor_shape)
        scales = torch.cat([scales_gate, scales_up], dim=1)

        # Transpose - [E, H/32, 2I_shard]
        scales = scales.transpose(1, 2)

        scales = _tile_gate_up_scale(scales)

        return scales.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mxfp4_gate_up_bias_loader(
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
    hidden_act_bias: float = 0.0,
) -> SafetensorsWeightLoader:
    """A loader for the gate_up_proj bias: checkpoint ``[E, 2I]``, sharded and padded to ``expected_shape`` ``[E,
    2I_shard]``, tiled and permuted as the blocks, with ``hidden_act_bias`` added to the up half."""

    def transform(slices: list, rank: int):
        (checkpoint,) = slices

        # Deinterleave + Shard - [E, I_shard] each
        b_gate, b_up = get_shard_deinterleaved(
            checkpoint, shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank
        )

        # Add hidden_act_bias to up_proj before padding, so the padded region stays zero. Out of place: an unpadded
        # shard can be a view of the tensor the loader reads.
        if hidden_act_bias != 0.0:
            b_up = b_up + hidden_act_bias

        # Pad each separately then concat
        per_tensor_shape = (expected_shape[0], expected_shape[1] // 2)
        b_gate = pad_to_shape(b_gate, per_tensor_shape)
        b_up = pad_to_shape(b_up, per_tensor_shape)

        bias = torch.cat([b_gate, b_up], dim=1)  # [E, 2I_shard]

        bias = _tile_gate_up_bias(bias)

        return bias.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mxfp4_down_weight_loader(
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """A loader for the down_proj MXFP4 blocks: checkpoint ``[E, H, I // 32, 16]``, sharded along I (checkpoint dim 2)
    and padded to ``expected_shape`` ``[E, H, I_shard // 32, 16]``, to the kernel's ``[E, 128_I, I_shard // 512, H]``
    (four FP4 values per uint16 along I)."""

    def transform(slices: list, rank: int):
        (checkpoint,) = slices

        # Checkpoint: [E, H, I/32, 16[x2]]
        # Shard along I dim (dim 2)
        blocks = get_shard(
            checkpoint, shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank
        )
        blocks = pad_to_shape(blocks, expected_shape)

        E_size, H_size, _, _ = blocks.shape

        # View: [E, H, I/4, 2[x2]]
        blocks = blocks.view(E_size, H_size, -1, 2)

        # Transpose: [E, I/4, H, 2[x2]]
        blocks = blocks.transpose(1, 2).contiguous()

        # Pack: [E, I/4, H[x4]]
        blocks = _pack_fp4_x4_uint16(blocks)
        blocks = blocks.view(E_size, -1, H_size)

        blocks = _tile_down_blocks(blocks)

        return blocks.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mxfp4_down_scale_loader(
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """A loader for the down_proj MX scales: checkpoint ``[E, H, I // 32]``, sharded along I (checkpoint dim 2) and
    padded to ``expected_shape`` ``[E, H, I_shard // 32]``, to the kernel's ``[E, 16_I, I_shard // 512, H]``."""

    def transform(slices: list, rank: int):
        (checkpoint,) = slices

        # Checkpoint: [E, H, I/32]
        # Shard along I dim (dim 2)
        scales = get_shard(
            checkpoint, shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank
        )
        scales = pad_to_shape(scales, expected_shape)

        # Transpose: [E, I/32, H]
        scales = scales.transpose(1, 2)

        scales = _tile_down_scale(scales)

        return scales.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mxfp4_down_bias_loader(
    expected_shape: tuple[int, ...],
    scale: float | None = None,
) -> SafetensorsWeightLoader:
    """A loader for the down_proj bias: checkpoint ``[E, H]`` padded to ``expected_shape`` ``[E, H_padded]`` with H
    shuffled."""

    def transform(slices: list, rank: int):
        (checkpoint,) = slices

        # Checkpoint: [E, H]
        bias = checkpoint[:]

        # this rank's share: the reduction after the experts sums every expert TP rank's bias
        if scale is not None:
            bias = bias / scale

        bias = pad_to_shape(bias, expected_shape)

        bias = _shuffle_hidden_dim(bias, dim=-1)

        return bias.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _shuffle_hidden_dim(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """``tensor`` with ``dim`` (negative allowed) shuffled: ``[H]`` viewed as ``[H / 4, 4]``, transposed to ``[4, H /
    4]`` and flattened, so H = 8 orders ``[1, 2, 3, 4, 5, 6, 7, 8]`` as ``[1, 5, 2, 6, 3, 7, 4, 8]``."""
    if dim < 0:
        dim = dim + tensor.ndim
    pre_shape, H = tensor.shape[:dim], tensor.shape[dim]
    if dim < tensor.ndim - 1:
        post_shape = tensor.shape[dim + 1 :]
    else:
        post_shape = []
    assert H % 4 == 0, (
        f"H dim (dim={dim=}) in shape {tensor.shape=} must be divisible by 4 to shuffle H dim, got H={H=}"
    )
    tensor = (
        tensor.view(*pre_shape, H // 4, 4, *post_shape)
        .transpose(dim, dim + 1)
        .reshape(*pre_shape, H, *post_shape)
    )
    return tensor


def _pack_fp4_x4_uint16(X: torch.Tensor) -> torch.Tensor:
    """FP4 pairs (uint8) along the last dimension, viewed as quadruples (uint16): [..., D] -> [..., D / 2]."""
    if X.dtype != torch.uint8:
        raise ValueError(f"MXFP4 blocks are uint8 FP4 pairs, not {X.dtype}")
    return X.view(torch.uint16)


def _tile_gate_up_blocks(w_gate_up):
    """
    Tile the fused gate/up MXFP4 blocks into the layout the MoE kernel reads.

    Input - [E, Pad(H/4), Pad(2I/TP)[x4]]
    """

    E_size, H_div_4, _2I_shard = w_gate_up.shape
    num_H_tiles, q_blocks_per_H_tile = _get_h_tiling_shard_i(H_size=H_div_4 * 4)
    num_I_tiles, q_blocks_per_I_tile = _get_i_tiling_shard_i(
        I_shard_size=_2I_shard // 2
    )
    q_packed_H = Q_WIDTH // 4  # four FP4 values per uint16
    _128_H = q_blocks_per_H_tile * Q_HEIGHT

    w_gate_up = w_gate_up.reshape(
        [
            E_size,
            num_H_tiles,
            q_blocks_per_H_tile,
            Q_HEIGHT,
            2,
            num_I_tiles,
            q_blocks_per_I_tile,
            Q_HEIGHT,
            Q_WIDTH,
            q_packed_H,
        ]
    )

    (
        E,
        N_H_TILES,
        Q_BLOCK_H_TILE,
        Q_HEIGHT_H,
        gate_up_concat,
        NUM_I_TILES,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
        Q_WIDTH_I,
        Q_PACKED_H,
    ) = list(range(w_gate_up.ndim))
    w_gate_up = w_gate_up.permute(
        E,
        Q_BLOCK_H_TILE,
        Q_HEIGHT_H,
        gate_up_concat,
        N_H_TILES,
        NUM_I_TILES,
        Q_WIDTH_I,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
        Q_PACKED_H,
    )

    w_gate_up = w_gate_up.reshape(
        E_size, _128_H, 2, num_H_tiles, -1
    )  # [E, 128_H, 2, H // 512, TP * I[x4H]]

    return w_gate_up


def _tile_gate_up_scale(s_gate_up):
    """
    Tile gate_up scales for MXFP4 kernel.

    Input: [E, H/32, 2*I_shard] where 2*I_shard = [gate_shard, up_shard] concatenated
    Output: [E, 16_H, 2, H//512, I_shard]
    """
    E_size, H_div_32, _2I_shard = s_gate_up.shape

    num_H_tiles, q_blocks_per_H_tile = _get_h_tiling_shard_i(H_size=H_div_32 * 32)
    num_I_tiles, q_blocks_per_I_tile = _get_i_tiling_shard_i(
        I_shard_size=_2I_shard // 2
    )

    s_gate_up = s_gate_up.reshape(
        [
            E_size,
            num_H_tiles,
            q_blocks_per_H_tile,
            2,  # gate/up
            num_I_tiles,
            q_blocks_per_I_tile,
            Q_HEIGHT,
            Q_WIDTH,
        ]
    )

    (
        E,
        N_H_TILES,
        Q_BLOCK_H_TILE,
        gate_up_concat,
        NUM_I_TILES,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
        Q_WIDTH_I,
    ) = range(s_gate_up.ndim)

    s_gate_up = s_gate_up.permute(
        E,
        Q_BLOCK_H_TILE,
        gate_up_concat,
        N_H_TILES,
        NUM_I_TILES,
        Q_WIDTH_I,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
    )

    scale = s_gate_up.reshape(
        E_size, q_blocks_per_H_tile, 2, num_H_tiles, -1
    )  # [E, 16_H, 2, H // 512, I_shard]
    return scale


def _tile_gate_up_bias(b_gate_up):
    """
    Tile gate_up bias for MXFP4 kernel.

    Input: [E, 2*I_shard] where first half is gate, second half is up
    Output: [E, 2*I_shard] (tiled/permuted)

    The permutation, of this rank's shard: E, Q_BLOCK_I_TILE, Q_HEIGHT_I, gate_up_concat, NUM_I_TILES, Q_WIDTH_I
    """
    E_size, _2I_shard = b_gate_up.shape
    I_shard = _2I_shard // 2

    num_I_tiles, q_blocks_per_I_tile = _get_i_tiling_shard_i(I_shard_size=I_shard)

    b_gate_up = b_gate_up.reshape(
        [
            E_size,
            2,  # gate_up_concat
            num_I_tiles,
            q_blocks_per_I_tile,
            Q_HEIGHT,
            Q_WIDTH,
        ]
    )

    # Indices: E=0, gate_up_concat=1, NUM_I_TILES=2, Q_BLOCK_I_TILE=3, Q_HEIGHT_I=4, Q_WIDTH_I=5
    # Permute to: E, Q_BLOCK_I_TILE, Q_HEIGHT_I, gate_up_concat, NUM_I_TILES, Q_WIDTH_I
    b_gate_up = b_gate_up.permute(0, 3, 4, 1, 2, 5)

    bias = b_gate_up.reshape(E_size, -1)  # [E, 2I_shard]
    return bias


def _tile_down_blocks(w_down):
    """
    Tile down_proj blocks for MXFP4 kernel (with H dim shuffling).

    Input: [E, I_shard/4, H[x4]]
    Output: [E, 128_I, I_shard//512, H[x4I]]
    """
    E_size, I_div_4, H_size = w_down.shape
    I_shard = I_div_4 * 4

    num_I_tiles, q_blocks_per_I_tile = _get_i_tiling_shard_i(I_shard_size=I_shard)
    q_packed = Q_WIDTH // 4

    w_down = w_down.reshape(
        [
            E_size,
            num_I_tiles,
            q_blocks_per_I_tile,
            Q_HEIGHT,
            H_size // 4,
            4,
            q_packed,
        ]
    )
    # E=0, NUM_I_TILES=1, Q_BLOCK_I_TILE=2, Q_HEIGHT_I=3, NUM_H_BLOCKS=4, H_BLOCK=5, Q_PACKED_I=6
    # Permute to: E, Q_BLOCK_I_TILE, Q_HEIGHT_I, NUM_I_TILES, H_BLOCK, NUM_H_BLOCKS, Q_PACKED_I
    w_down = w_down.permute(0, 2, 3, 1, 5, 4, 6)
    w_down = w_down.reshape(
        E_size, q_blocks_per_I_tile * Q_HEIGHT, num_I_tiles, H_size * q_packed
    )

    return w_down


def _tile_down_scale(s_down):
    """
    Tile down_proj scales for MXFP4 kernel (with H dim shuffling).

    Input: [E, I_shard/32, H]
    Output: [E, 16_I, I_shard//512, H]
    """
    E_size, I_div_32, H_size = s_down.shape
    I_shard = I_div_32 * 32

    num_I_tiles, q_blocks_per_I_tile = _get_i_tiling_shard_i(I_shard_size=I_shard)

    s_down = s_down.reshape(
        [
            E_size,
            num_I_tiles,
            q_blocks_per_I_tile,
            H_size // 4,
            4,
        ]
    )
    # E=0, NUM_I_TILES=1, Q_BLOCK_I_TILE=2, NUM_H_BLOCKS=3, H_BLOCK=4
    # Permute to: E, Q_BLOCK_I_TILE, NUM_I_TILES, H_BLOCK, NUM_H_BLOCKS
    s_down = s_down.permute(0, 2, 1, 4, 3)
    s_down = s_down.reshape(E_size, q_blocks_per_I_tile, num_I_tiles, H_size)

    return s_down


def _get_h_tiling_shard_i(H_size: int):
    """``(num_H_tiles, q_blocks_per_H_tile)`` of the hidden dim's 512-element tiles: ``H_size // (pmax * q_width)``
    tiles of ``512 // (q_width * q_height)`` quantization blocks each."""
    num_H_tiles = H_size // (PMAX * Q_WIDTH)
    q_blocks_per_H_tile = 512 // (Q_WIDTH * Q_HEIGHT)
    return num_H_tiles, q_blocks_per_H_tile


def _get_i_tiling_shard_i(I_shard_size: int):
    """``(num_I_tiles, q_blocks_per_I_tile)`` of this rank's intermediate dim ``I_shard_size``: 512-wide tiles above
    512, one tile at or below it (384 gives ``(1, 12)``, 512 ``(1, 16)``, 1024 ``(2, 16)``). A size that is not a whole
    number of blocks fails its assertion."""
    mx_block_size = Q_WIDTH * Q_HEIGHT
    if I_shard_size > 512:
        assert I_shard_size % 512 == 0, (
            f"Unsupported I_shard_size {I_shard_size} must be divisible by 512"
        )
        num_I_tiles = I_shard_size // 512
        q_blocks_per_I_tile = 512 // mx_block_size
    else:
        assert I_shard_size % mx_block_size == 0, (
            f"Unsupported I_shard_size {I_shard_size}, must be divisible by {mx_block_size}"
        )
        num_I_tiles = 1
        q_blocks_per_I_tile = I_shard_size // mx_block_size
    return num_I_tiles, q_blocks_per_I_tile
