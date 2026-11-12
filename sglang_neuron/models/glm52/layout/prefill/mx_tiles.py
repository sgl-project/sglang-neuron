# SPDX-License-Identifier: Apache-2.0
"""The MX tile layouts of the blockwise MXFP4 MoE kernels, for the GLM-5.2 prefill loaders.

GLM-5.2's prefill experts and router run the same blockwise MX kernels as GPT-OSS's, so these helpers encode a KERNEL
layout, not a model choice: the GPT-OSS loaders (`sglang_neuron.models.gpt_oss.weight_loaders_mxfp4`'s
``_tile_gate_up_blocks`` and ``_tile_gate_up_scale``) and the fused norm (`sglang_neuron.ops.moe.rmsnorm_mx_prefill`'s
``swizzle_h_index``) build the same tensors, byte for byte.
The GLM loaders keep this torch-only copy because both of those modules import the NKI DSL when they load.
"""

import torch

# SBUF partitions. nl.tile_size.pmax has its real value only inside a trace, so the host loaders take the constant.
PMAX = 128
# The MX quantization block: Q_HEIGHT rows of Q_WIDTH values share one scale.
Q_WIDTH = 4
Q_HEIGHT = 8
# Values of one matmul contraction tile.
_TILE = 512


def swizzle_h_index(H: int) -> torch.Tensor:
    """For each slot of the fused norm's swizzled hidden order, its natural index ``512 * t + 4 * p + q``.

    The fused ``rmsnorm_mx_prefill`` kernel matmuls the router weight against swizzle-transposed activations, so
    row ``h`` of the natural weight must sit at the swizzle slot of ``h``: ``W_swz[s] = W_natural[swizzle[s]]``.

    Raises:
        ValueError: If ``H`` is not a whole number of 512-value tiles.
    """
    if H % _TILE:
        raise ValueError(f"the swizzled hidden order takes whole {_TILE}-value tiles, and H is {H}")
    num_h512 = H // _TILE
    h512 = torch.arange(num_h512).reshape(num_h512, 1, 1)
    q = torch.arange(Q_WIDTH).reshape(1, Q_WIDTH, 1)
    p = torch.arange(PMAX).reshape(1, 1, PMAX)
    return (h512 * _TILE + Q_WIDTH * p + q).reshape(-1)


def tile_gate_up_blocks(w_gate_up):
    """Tile the fused gate/up MXFP4 blocks into the layout the blockwise MX MoE kernel reads.

    Args:
        w_gate_up: ``[E, H / 4, 2 * I_shard]`` x4 words (four FP4 values of the H axis each): the gate columns,
            then the up columns, of this rank's intermediate shard.

    Returns:
        ``[E, 128, 2, H / 512, I_shard]``: x4 row ``128 * t + p`` of H tile ``t`` at partition ``p``, gate and up
        at axis 2, and within each intermediate tile of ``W = min(I_shard, 512)`` columns the column
        ``32 * b + 4 * r + w`` (block, row, width) at ``(W / 4) * w + 8 * b + r``.
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
    )  # [E, 128, 2, H // 512, I_shard]

    return w_gate_up


def tile_gate_up_scale(s_gate_up):
    """Tile the fused gate/up MX scales into the layout the blockwise MX MoE kernel reads.

    Args:
        s_gate_up: ``[E, H / 32, 2 * I_shard]`` E8M0 scales: the gate columns, then the up columns.

    Returns:
        ``[E, 16, 2, H / 512, I_shard]``: the 16 scale rows of each H tile, gate and up at axis 2, and the
        intermediate columns in the order of `tile_gate_up_blocks`.
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


def _get_h_tiling_shard_i(H_size: int):
    """``(num_H_tiles, q_blocks_per_H_tile)`` of the hidden axis: ``H_size / 512`` contraction tiles of
    ``512 / (Q_WIDTH * Q_HEIGHT)`` = 16 quantization blocks each."""
    num_H_tiles = H_size // (PMAX * Q_WIDTH)
    q_blocks_per_H_tile = _TILE // (Q_WIDTH * Q_HEIGHT)
    return num_H_tiles, q_blocks_per_H_tile


def _get_i_tiling_shard_i(I_shard_size: int):
    """``(num_I_tiles, q_blocks_per_I_tile)`` of this rank's intermediate shard: 512-column tiles above 512, one tile
    at or below it.

    Examples: 384 gives ``(1, 12)``, 512 gives ``(1, 16)``, 1024 gives ``(2, 16)``.

    Raises:
        ValueError: If a shard above 512 is not a whole number of 512-column tiles, or a smaller one not a whole
            number of 32-value quantization blocks: the kernel has no padded tile.
    """
    mx_block_size = Q_WIDTH * Q_HEIGHT
    if I_shard_size > _TILE:
        if I_shard_size % _TILE:
            raise ValueError(
                f"an intermediate shard of {I_shard_size} above {_TILE} must be whole {_TILE}-column tiles"
            )
        return I_shard_size // _TILE, _TILE // mx_block_size
    if I_shard_size % mx_block_size:
        raise ValueError(
            f"an intermediate shard of {I_shard_size} must be whole {mx_block_size}-value quantization blocks"
        )
    return 1, I_shard_size // mx_block_size
