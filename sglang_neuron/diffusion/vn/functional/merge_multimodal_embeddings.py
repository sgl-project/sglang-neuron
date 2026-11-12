# SPDX-License-Identifier: Apache-2.0
"""Splice encoder embeddings from a cache buffer into this SP rank's token embeddings and
gather the deepstack embeddings: a NKI kernel on device, a torch reference elsewhere.
"""

from __future__ import annotations

import torch

from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode

from sglang_neuron.diffusion.neuron_api import nki_wrap_function
from sglang_neuron.diffusion.vn.utils.hardware_config import get_logical_nc_config
from nkilib.core.utils.kernel_helpers import div_ceil


def wrap_nki(kernel):
    """The runtime's ``wrap_nki``, resolved at call time: importing a runtime at module import
    would start it before the worker claims its NeuronCore."""
    return nki_wrap_function()(kernel)


def scatter_with_dummy_row(
    target: torch.Tensor,
    positions: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """``target`` with ``values`` written at ``positions``.

    A position of ``target.shape[0]`` writes an appended dummy row that is dropped, so a
    traced graph needs no branch on that sentinel.

    Example:
        >>> import torch
        >>> out = scatter_with_dummy_row(
        ...     torch.zeros(3, 2), torch.tensor([0, 3]), torch.ones(2, 2))
        >>> out[0]  # row 0 written; the position-3 write hits the discarded dummy row
        tensor([1., 1.])
    """
    dummy = torch.zeros(1, target.shape[-1], dtype=target.dtype, device=target.device)
    with_dummy = torch.cat([target, dummy], dim=0)
    with_dummy.index_put_((positions,), values.to(target.dtype))
    return with_dummy[: target.shape[0]]


def get_local_encoder_slots_and_positions(
    encoder_slot_mapping: torch.Tensor,
    local_len: int,
    num_slots: int,
    rank: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """This SP rank's gather slots and scatter rows, each ``[local_len]`` int32.

    ``local_encoder_slots[i]`` is the flattened cache-buffer row that local position ``i``
    reads; ``local_encoder_embedding_positions[i]`` is the row of this rank's
    ``hidden_states`` (and of each deepstack level) it writes. A text position (slot
    ``>= num_slots``) reads row ``num_slots - 1`` and writes row ``local_len``, which the
    scatter drops.

    Args:
        encoder_slot_mapping: ``[num_tokens]`` int32 flat cache slot per position.
        local_len: Positions per rank (``num_tokens // world``).
        num_slots: Rows of the flattened cache buffer.
        rank: SP rank. A tensor keeps one graph for all ranks.

    Example:
        Rank 1 holds positions 2 and 3; position 3 is text, so its write (row 2) is dropped.

        >>> import torch
        >>> slots = torch.tensor([9, 5, 2, 9], dtype=torch.int32)  # 9 = text
        >>> get_local_encoder_slots_and_positions(
        ...     slots, local_len=2, num_slots=8, rank=1)
        (tensor([2, 7], dtype=torch.int32), tensor([0, 2], dtype=torch.int32))
    """
    device = encoder_slot_mapping.device
    offsets = torch.arange(local_len, dtype=torch.int32, device=device)
    local_slots = encoder_slot_mapping.index_select(0, offsets + rank * local_len)
    is_encoder = local_slots < num_slots
    local_encoder_slots = torch.where(
        is_encoder, local_slots, torch.full_like(local_slots, num_slots - 1)
    )
    local_encoder_embedding_positions = torch.where(
        is_encoder, offsets, torch.full_like(offsets, local_len)
    )
    return local_encoder_slots, local_encoder_embedding_positions


def merge_multimodal_embeddings(
    hidden_states: torch.Tensor,
    encoder_embedding_buffer: torch.Tensor,
    encoder_slot_mapping: torch.Tensor,
    rank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scatter the main encoder embedding into ``hidden_states`` and gather the deepstack ones.

    Both paths are pure data movement and match exactly.

    Args:
        hidden_states: ``[local_len, hidden_dim]`` token embeddings of this SP rank.
        encoder_embedding_buffer: ``[num_blocks, block_size, hidden_dim * (1 + N_ds)]``, the
            main embedding followed by ``N_ds`` deepstack embeddings.
        encoder_slot_mapping: ``[num_tokens]`` int32 flat slot
            (``block_id * block_size + offset``) per position, or
            ``>= num_blocks * block_size`` for text.
        rank: SP rank.

    Returns:
        ``(hidden_states, deepstack)``: deepstack is ``[N_ds, local_len, hidden_dim]``, zero at
        text rows, or None when ``N_ds == 0``.

    Example:
        >>> import torch
        >>> hidden = torch.zeros(8, 4)                    # [local_len, hidden_dim]
        >>> buffer = torch.arange(16.0).reshape(1, 4, 4)  # one block, no deepstack
        >>> slots = torch.tensor([4, 4, 0, 1, 2, 3, 4, 4], dtype=torch.int32)
        >>> out, deepstack = merge_multimodal_embeddings(hidden, buffer, slots, rank=0)
        >>> deepstack is None
        True
    """
    num_blocks, block_size, _ = encoder_embedding_buffer.shape
    local_encoder_slots, local_encoder_embedding_positions = (
        get_local_encoder_slots_and_positions(
            encoder_slot_mapping, hidden_states.shape[0], num_blocks * block_size, rank
        )
    )
    if _can_use_kernel(hidden_states, encoder_embedding_buffer):
        return _merge_multimodal_embeddings_nki_wrapper(
            hidden_states,
            encoder_embedding_buffer,
            local_encoder_slots,
            local_encoder_embedding_positions,
        )
    return _merge_multimodal_embeddings_torch(
        hidden_states,
        encoder_embedding_buffer,
        local_encoder_slots,
        local_encoder_embedding_positions,
    )


def _num_deepstack_levels(fat_dim: int, hidden_dim: int) -> int:
    return (fat_dim - hidden_dim) // hidden_dim


def _can_use_kernel(
    hidden_states: torch.Tensor, encoder_embedding_buffer: torch.Tensor
) -> bool:
    """Whether the NKI kernel supports these inputs; any row count, block size and N_ds do."""
    if not can_run_kernel(hidden_states):
        return False

    if hidden_states.dtype != torch.bfloat16:
        return False

    # The kernel would drop a partial trailing deepstack level; the torch reference raises.
    hidden_dim = hidden_states.shape[-1]
    fat_dim = encoder_embedding_buffer.shape[-1]
    if (fat_dim - hidden_dim) % hidden_dim != 0:
        return False
    return True


def _merge_multimodal_embeddings_torch(
    hidden_states: torch.Tensor,
    encoder_embedding_buffer: torch.Tensor,
    local_encoder_slots: torch.Tensor,
    local_encoder_embedding_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Torch reference: one ``index_select`` gather, then ``index_put_`` scatters."""
    fat_dim = encoder_embedding_buffer.shape[-1]
    flat_buffer = encoder_embedding_buffer.reshape(-1, fat_dim)
    flat_embeds = flat_buffer.index_select(0, local_encoder_slots)

    hidden_dim = hidden_states.shape[-1]
    local_len = hidden_states.shape[0]

    main_embeds = flat_embeds[:, :hidden_dim]
    hidden_states = scatter_with_dummy_row(
        hidden_states, local_encoder_embedding_positions, main_embeds
    )

    deepstack_embeds = None
    fat_dim = flat_embeds.shape[-1]
    if fat_dim > hidden_dim:
        N_ds = (fat_dim - hidden_dim) // hidden_dim
        ds_embeds = flat_embeds[:, hidden_dim:].reshape(-1, N_ds, hidden_dim)
        deepstack_embeds = torch.zeros(
            N_ds,
            local_len,
            hidden_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for n in range(N_ds):
            deepstack_embeds[n] = scatter_with_dummy_row(
                deepstack_embeds[n],
                local_encoder_embedding_positions,
                ds_embeds[:, n, :],
            )

    return hidden_states, deepstack_embeds


def _merge_multimodal_embeddings_nki_wrapper(
    hidden_states: torch.Tensor,
    encoder_embedding_buffer: torch.Tensor,
    local_encoder_slots: torch.Tensor,
    local_encoder_embedding_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch the kernel. The positions are already local, so it takes no rank."""
    hidden_dim = hidden_states.shape[-1]
    fat_dim = encoder_embedding_buffer.shape[-1]
    n_ds = _num_deepstack_levels(fat_dim, hidden_dim)

    # The indirect DMAs take int32 indices, which the slot mapping already is.
    wrapped = wrap_nki(_merge_multimodal_embeddings_nki)[get_logical_nc_config()]
    # The kernel returns only hidden when n_ds == 0.
    if n_ds > 0:
        out_hidden, out_ds = wrapped(
            hidden_states,
            encoder_embedding_buffer,
            local_encoder_slots,
            local_encoder_embedding_positions,
        )
        return out_hidden, out_ds
    out_hidden = wrapped(
        hidden_states,
        encoder_embedding_buffer,
        local_encoder_slots,
        local_encoder_embedding_positions,
    )
    return out_hidden, None


@nki.jit
def _merge_multimodal_embeddings_nki(
    hidden_states,
    buffer,
    local_encoder_slots,
    local_encoder_embedding_positions,
):
    """Fused gather + scatter in tiles of ``pmax`` rows.

    Each tile gathers its slots' ``[rows, fat_dim]`` with one indirect DMA, then scatters the
    main slice into ``out_hidden`` and each deepstack slice into its level, ``oob_mode.skip``
    dropping text rows (position ``local_len``). No ``[R, fat_dim]`` intermediate is built.
    """
    num_blocks, block_size, fat_dim = buffer.shape
    local_len, hidden_dim = hidden_states.shape
    R = local_encoder_slots.shape[0]
    N_ds = (fat_dim - hidden_dim) // hidden_dim
    pmax = nl.tile_size.pmax
    n_tiles = div_ceil(R, pmax)

    # out_hidden starts as the token embeddings; each deepstack level starts zeroed, since the
    # scatter writes only the encoder rows.
    out_hidden = nl.ndarray(
        hidden_states.shape, hidden_states.dtype, buffer=nl.shared_hbm
    )
    nisa.dma_copy(out_hidden, hidden_states)  # HBM -> HBM

    # Each level is its own HBM tensor: oob_mode.skip bounds against the tensor it writes, so
    # with a slice out_ds[n] of one [N_ds, local_len, hidden_dim] tensor a dropped row (index
    # local_len) would land in row 0 of level n + 1. The levels are copied into out_ds last.
    ds_levels = []
    for n in range(N_ds):
        lvl = nl.ndarray(
            (local_len, hidden_dim), hidden_states.dtype, buffer=nl.shared_hbm
        )
        _zero_hbm(lvl, hidden_dim)
        ds_levels.append(lvl)

    buffer_flat = buffer.reshape((num_blocks * block_size, fat_dim))

    for pt in range(n_tiles):
        row0 = pt * pmax
        rows = min(pmax, R - row0)

        # This tile's slots, one per partition (transpose-on-load).
        src_rows = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            src_rows,
            local_encoder_slots[row0 : row0 + rows].reshape((1, rows)).permute((1, 0)),
        )

        # Gather [rows, fat_dim].
        blk_sb = nl.ndarray((rows, fat_dim), dtype=buffer.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            src=buffer_flat.ap(
                pattern=[[fat_dim, rows], [1, fat_dim]],
                offset=0,
                vector_offset=src_rows,
                indirect_dim=0,
            ),
            dst=blk_sb,
            name=f"merge_multimodal_gather_t{pt}",
        )

        # This tile's destination rows, one per partition.
        pos = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            pos,
            local_encoder_embedding_positions[row0 : row0 + rows]
            .reshape((1, rows))
            .permute((1, 0)),
        )

        # Main embedding: the first hidden_dim columns.
        nisa.dma_copy(
            src=blk_sb[:, 0:hidden_dim],
            dst=out_hidden.ap(
                pattern=[[hidden_dim, rows], [1, hidden_dim]],
                offset=0,
                vector_offset=pos,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip,
            name=f"merge_multimodal_scatter_main_t{pt}",
        )

        # Deepstack level n: the next hidden_dim columns.
        for n in range(N_ds):
            lo = hidden_dim * (1 + n)
            nisa.dma_copy(
                src=blk_sb[:, lo : lo + hidden_dim],
                dst=ds_levels[n].ap(
                    pattern=[[hidden_dim, rows], [1, hidden_dim]],
                    offset=0,
                    vector_offset=pos,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
                name=f"merge_multimodal_scatter_ds{n}_t{pt}",
            )

    if N_ds > 0:
        out_ds = nl.ndarray(
            (N_ds, local_len, hidden_dim), hidden_states.dtype, buffer=nl.shared_hbm
        )
        for n in range(N_ds):
            nisa.dma_copy(out_ds[n], ds_levels[n])
        return out_hidden, out_ds
    return out_hidden


def _zero_hbm(hbm_2d, cols):
    """Zero an HBM ``[rows, cols]`` tensor from one memset ``[pmax, cols]`` SBUF tile.

    It runs before the gather loop, so its SBUF tile is not live alongside ``blk_sb``.
    """
    rows = hbm_2d.shape[0]
    pmax = nl.tile_size.pmax
    zero_sb = nl.ndarray((pmax, cols), dtype=hbm_2d.dtype, buffer=nl.sbuf)
    nisa.memset(zero_sb, 0)
    num_tiles = div_ceil(rows, pmax)
    for tile in range(num_tiles):
        tile_rows = min(pmax, rows - tile * pmax)
        nisa.dma_copy(
            hbm_2d[nl.ds(tile * pmax, tile_rows), :],
            zero_sb[0:tile_rows, :],
        )
