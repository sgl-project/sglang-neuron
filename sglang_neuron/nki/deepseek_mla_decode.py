"""Single-pass direct-paged MLA decode for DeepSeek-V2-Lite.

This module groups explicit slot addresses into full 128-token tiles and
performs score, online softmax, and value accumulation before that tile leaves
SBUF. The kernel consumes physical token slots and is independent of the
allocator page size.

The TP=4 DeepSeek-V2-Lite serving topology has four local query heads, latent
dimension 512, and positional dimension 64. Static contexts are made of
complete 512-token supertiles. The schedule groups four 128-token tiles into
one supertile so max/exp/sum and running-state correction happen once per
supertile rather than once per Tensor Engine tile.
"""

from __future__ import annotations

import torch

from sglang_neuron.nki import logical_cores
from sglang_neuron.nki.views import broadcast_column

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nkilib.core.utils.kernel_assert import kernel_assert
    from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
    from torch_neuronx import wrap_nki

    _NKI_AVAILABLE = True
    _NKI_IMPORT_ERR = None
except Exception as exc:  # noqa: BLE001  # pragma: no cover - Neuron image only
    _NKI_AVAILABLE = False
    _NKI_IMPORT_ERR = exc


_TILE = 128
_HEADS = 4
_LATENT = 512
_ROPE = 64
_COMPACT = _LATENT + _ROPE
_SUPERTILE_TILES = 4
_SUPERTILE = _SUPERTILE_TILES * _TILE
# the context granule the kernel reads, which a caller pads its slots to
SUPERTILE = _SUPERTILE


def _validate_kernel_contract(
    *, batch: int, heads: int, context: int, dtype: torch.dtype
) -> None:
    """Reject structurally invalid inputs without enforcing qualification policy."""
    if batch <= 0:
        raise ValueError(f"paged MLA requires a positive batch; got {batch}")
    if heads != _HEADS:
        raise ValueError(f"paged MLA requires four TP-local heads; got {heads}")
    if context <= 0 or context % _SUPERTILE:
        raise ValueError(
            f"paged MLA context must be a positive multiple of {_SUPERTILE}; "
            f"got {context}"
        )
    if dtype != torch.bfloat16:
        raise ValueError("paged MLA requires bfloat16")


def build_paged_mla_metadata(
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build indirect-DMA slots and a head-aligned online-softmax mask.

    ``block_tables`` already contains physical token slots in logical order,
    so grouping those rows by 128 is independent of allocator page size.
    History excludes the current token,
    which is supplied through ``compact_new``.  The small head-expanded mask
    keeps both operands of the NKI elementwise multiply on the same SBUF
    partitions; broadcasting a single mask row across partition dimension 0
    is not supported by the current Neuron compiler.
    """
    batch, context = block_tables.shape
    if batch <= 0:
        raise ValueError("paged MLA metadata requires a positive batch")
    if context <= 0 or context % _SUPERTILE:
        raise ValueError(
            f"paged MLA metadata context must be a positive multiple of "
            f"{_SUPERTILE}; got {context}"
        )
    positions = torch.arange(context, dtype=torch.int32, device=block_tables.device)
    history_valid = positions.view(1, context) < (
        seq_lens.to(torch.int32).view(batch, 1) - 1
    )
    slots = torch.where(
        history_valid,
        block_tables.to(torch.int32),
        torch.full_like(block_tables, -1, dtype=torch.int32),
    )
    active_slots = slots.reshape(batch, context // _TILE, _TILE, 1).contiguous()
    validity_mask = (
        history_valid[:, None, :]
        .expand(batch, _HEADS, context)
        .to(torch.float32)
        .reshape(batch, _HEADS, context // _SUPERTILE, _SUPERTILE)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return active_slots, validity_mask


if _NKI_AVAILABLE:

    def _prepare_queries(ql_nope, q_pe, batch_id):
        """Load and transpose Q once so every context tile can reuse it."""
        query_tiles = nl.ndarray(
            (_TILE, (_LATENT // _TILE) * _HEADS),
            dtype=ql_nope.dtype,
            buffer=nl.sbuf,
        )
        for latent_tile in nl.affine_range(_LATENT // _TILE):
            query = nl.ndarray((_HEADS, _TILE), dtype=ql_nope.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=query,
                src=ql_nope[batch_id, :, nl.ds(latent_tile * _TILE, _TILE)],
            )
            query_t_psum = nl.ndarray(
                (_TILE, _HEADS), dtype=ql_nope.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(dst=query_t_psum, data=query)
            nisa.tensor_copy(
                dst=query_tiles[:, nl.ds(latent_tile * _HEADS, _HEADS)],
                src=query_t_psum,
            )

        query_pos = nl.ndarray((_HEADS, _ROPE), dtype=q_pe.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=query_pos, src=q_pe[batch_id, :, :])
        query_pos_t_psum = nl.ndarray((_ROPE, _HEADS), dtype=q_pe.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=query_pos_t_psum, data=query_pos)
        query_pos_t = nl.ndarray((_ROPE, _HEADS), dtype=q_pe.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=query_pos_t, src=query_pos_t_psum)
        return query_tiles, query_pos_t

    def _score_tile(page, query_tiles, query_pos_t, scale, scores):
        """Write the tile's scaled FP32 scores [H, 128] into ``scores``. The queries are the stationary operand, so the
        scores come out heads by keys, and the tile's four latent chunks transpose into one PSUM buffer, evicted
        once."""
        latent_t_psum = nl.ndarray((_TILE, _LATENT), dtype=page.dtype, buffer=nl.psum)
        for latent_tile in nl.affine_range(_LATENT // _TILE):
            nisa.nc_transpose(
                dst=latent_t_psum[:, nl.ds(latent_tile * _TILE, _TILE)],
                data=page[:, nl.ds(latent_tile * _TILE, _TILE)],
            )
        latent_t = nl.ndarray((_TILE, _LATENT), dtype=page.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=latent_t, src=latent_t_psum)

        key_pos_t_psum = nl.ndarray((_ROPE, _TILE), dtype=page.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=key_pos_t_psum, data=page[:, nl.ds(_LATENT, _ROPE)])
        key_pos_t = nl.ndarray((_ROPE, _TILE), dtype=page.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=key_pos_t, src=key_pos_t_psum)

        score_psum = nl.ndarray((_HEADS, _TILE), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(score_psum, 0.0)
        for latent_tile in nl.affine_range(_LATENT // _TILE):
            nisa.nc_matmul(
                dst=score_psum,
                stationary=query_tiles[:, nl.ds(latent_tile * _HEADS, _HEADS)],
                moving=latent_t[:, nl.ds(latent_tile * _TILE, _TILE)],
            )
        nisa.nc_matmul(dst=score_psum, stationary=query_pos_t, moving=key_pos_t)
        nisa.tensor_scalar(dst=scores, data=score_psum, op0=nl.multiply, operand0=scale)

    def _online_current_update(
        current_page,
        scores,
        running_max,
        running_sum,
        running_output,
    ):
        """Add the uncached current token as the final online-softmax item."""
        current_score = nl.ndarray((_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=current_score, src=scores[:, nl.ds(0, 1)])
        next_max = nl.ndarray((_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=next_max,
            data1=running_max,
            data2=current_score,
            op=nl.maximum,
        )
        correction = nl.ndarray((_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=correction,
            data1=running_max,
            data2=next_max,
            op=nl.subtract,
        )
        nisa.activation(dst=correction, op=nl.exp, data=correction)
        current_exp = nl.ndarray((_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=current_exp,
            data1=current_score,
            data2=next_max,
            op=nl.subtract,
        )
        nisa.activation(dst=current_exp, op=nl.exp, data=current_exp)

        nisa.tensor_tensor(
            dst=running_sum,
            data1=running_sum,
            data2=correction,
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=running_sum,
            data1=running_sum,
            data2=current_exp,
            op=nl.add,
        )
        nisa.tensor_tensor(
            dst=running_output,
            data1=running_output,
            data2=broadcast_column(correction, _LATENT),
            op=nl.multiply,
        )
        # Use the Tensor Engine for the current-token value contribution too.
        # Elementwise broadcasting the value row from SBUF partition 0 to
        # partitions 1..H fails NKI partition-alignment validation.
        current_probabilities = nl.ndarray(
            (_HEADS, _TILE), dtype=current_page.dtype, buffer=nl.sbuf
        )
        nisa.memset(current_probabilities, 0.0)
        nisa.tensor_copy(dst=current_probabilities[:, nl.ds(0, 1)], src=current_exp)
        current_prob_t_psum = nl.ndarray(
            (_TILE, _HEADS), dtype=current_page.dtype, buffer=nl.psum
        )
        nisa.nc_transpose(dst=current_prob_t_psum, data=current_probabilities)
        current_prob_t = nl.ndarray(
            (_TILE, _HEADS), dtype=current_page.dtype, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=current_prob_t, src=current_prob_t_psum)
        current_contribution_psum = nl.ndarray(
            (_HEADS, _LATENT), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_matmul(
            dst=current_contribution_psum,
            stationary=current_prob_t,
            moving=current_page[:, nl.ds(0, _LATENT)],
        )
        current_contribution = nl.ndarray(
            (_HEADS, _LATENT), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=current_contribution, src=current_contribution_psum)
        nisa.tensor_tensor(
            dst=running_output,
            data1=running_output,
            data2=current_contribution,
            op=nl.add,
        )
        nisa.tensor_copy(dst=running_max, src=next_max)

    def _online_supertile_update(
        pages,
        valid_heads,
        scores,
        running_max,
        running_sum,
        running_output,
    ):
        """Consume one 512-token supertile with one online-softmax update."""
        tile_max = nl.ndarray((_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=tile_max,
            op=nl.maximum,
            data=scores,
            axis=[1],
            keepdims=True,
        )
        next_max = nl.ndarray((_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=next_max,
            data1=running_max,
            data2=tile_max,
            op=nl.maximum,
        )
        correction = nl.ndarray((_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=correction,
            data1=running_max,
            data2=next_max,
            op=nl.subtract,
        )
        nisa.activation(dst=correction, op=nl.exp, data=correction)

        nisa.tensor_tensor(
            dst=scores,
            data1=scores,
            data2=broadcast_column(next_max, _SUPERTILE),
            op=nl.subtract,
        )
        nisa.activation(dst=scores, op=nl.exp, data=scores)
        nisa.tensor_tensor(
            dst=scores,
            data1=scores,
            data2=valid_heads,
            op=nl.multiply,
        )

        tile_sum = nl.ndarray((_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=tile_sum,
            op=nl.add,
            data=scores,
            axis=[1],
            keepdims=True,
        )
        nisa.tensor_tensor(
            dst=running_sum,
            data1=running_sum,
            data2=correction,
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=running_sum,
            data1=running_sum,
            data2=tile_sum,
            op=nl.add,
        )

        tile_output_psum = nl.ndarray(
            (_HEADS, _LATENT), dtype=nl.float32, buffer=nl.psum
        )
        nisa.memset(tile_output_psum, 0.0)
        for tile_id in nl.affine_range(_SUPERTILE_TILES):
            probabilities = nl.ndarray(
                (_HEADS, _TILE), dtype=pages.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(
                dst=probabilities,
                src=scores[:, nl.ds(tile_id * _TILE, _TILE)],
            )
            prob_t_psum = nl.ndarray((_TILE, _HEADS), dtype=pages.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=prob_t_psum, data=probabilities)
            prob_t = nl.ndarray((_TILE, _HEADS), dtype=pages.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=prob_t, src=prob_t_psum)
            page = pages[:, nl.ds(tile_id * _COMPACT, _COMPACT)]
            nisa.nc_matmul(
                dst=tile_output_psum,
                stationary=prob_t,
                moving=page[:, nl.ds(0, _LATENT)],
            )

        tile_output = nl.ndarray((_HEADS, _LATENT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tile_output, src=tile_output_psum)
        nisa.tensor_tensor(
            dst=running_output,
            data1=running_output,
            data2=broadcast_column(correction, _LATENT),
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=running_output,
            data1=running_output,
            data2=tile_output,
            op=nl.add,
        )
        nisa.tensor_copy(dst=running_max, src=next_max)

    @wrap_nki
    @nki.jit
    def _paged_absorbed_mla_kernel(
        ql_nope,
        q_pe,
        compact_new,
        compact_pool,
        active_slots,
        history_mask,
        scale,
    ):
        batch, heads, latent_dim = ql_nope.shape
        _, q_heads, rope_dim = q_pe.shape
        _, compact_dim = compact_new.shape
        _, pool_compact_dim = compact_pool.shape
        slot_batch, num_tiles, indirect_rows, one = active_slots.shape
        mask_batch, mask_supertiles, mask_heads, mask_width = history_mask.shape
        context = num_tiles * _TILE
        kernel_assert(batch > 0, "paged MLA batch must be positive")
        kernel_assert(heads == _HEADS and q_heads == heads, "paged MLA expects H=4")
        kernel_assert(latent_dim == _LATENT, "paged MLA expects latent dim 512")
        kernel_assert(rope_dim == _ROPE, "paged MLA expects RoPE dim 64")
        kernel_assert(
            compact_dim == _COMPACT and pool_compact_dim == _COMPACT,
            "paged MLA expects compact dim 576",
        )
        kernel_assert(
            num_tiles > 0 and num_tiles % _SUPERTILE_TILES == 0,
            "MLA context must contain complete 512-token supertiles",
        )
        kernel_assert(
            indirect_rows == _TILE and one == 1,
            "MLA indirect rows must be [128,1]",
        )
        kernel_assert(slot_batch == batch, "MLA active slot batch mismatch")
        kernel_assert(
            mask_batch == batch
            and mask_supertiles == num_tiles // _SUPERTILE_TILES
            and mask_heads == heads
            and mask_width == _SUPERTILE,
            "MLA history mask mismatch",
        )

        output = nl.ndarray(
            (batch, heads, latent_dim),
            dtype=ql_nope.dtype,
            buffer=nl.shared_hbm,
        )

        # each program of an LNC2 launch attends its share of the rows; a row past the even split runs on every
        # program, which write the same output
        _, programs, program = get_verified_program_sharding_info("paged_absorbed_mla", max_sharding=2)
        share = batch // programs
        rows = [program * share + row for row in range(share)] + list(range(share * programs, batch))
        for batch_id in rows:
            query_tiles, query_pos_t = _prepare_queries(ql_nope, q_pe, batch_id)
            running_max = nl.ndarray((_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
            running_sum = nl.ndarray((_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
            running_output = nl.ndarray(
                (_HEADS, _LATENT), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.memset(running_max, -10000.0)
            nisa.memset(running_sum, 0.0)
            nisa.memset(running_output, 0.0)

            # Device-bounded ``nl.fori_loop`` currently produces asymmetric
            # control flow across the two cores of an LNC2 outer graph
            # (NCC_IXGM002). Keep the context-specialized loop static; the
            # validity mask still handles the request's actual sequence length.
            for supertile_id in nl.affine_range(mask_supertiles):
                pages = nl.ndarray(
                    (_TILE, _SUPERTILE_TILES * _COMPACT),
                    dtype=compact_pool.dtype,
                    buffer=nl.sbuf,
                )
                nisa.memset(pages, 0.0)
                scores = nl.ndarray(
                    (_HEADS, _SUPERTILE), dtype=nl.float32, buffer=nl.sbuf
                )
                valid_heads = nl.ndarray(
                    (_HEADS, _SUPERTILE), dtype=nl.float32, buffer=nl.sbuf
                )
                for local_tile_id in nl.affine_range(_SUPERTILE_TILES):
                    tile_id = supertile_id * _SUPERTILE_TILES + local_tile_id
                    slots = nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=slots,
                        src=active_slots[batch_id, tile_id, :, :],
                    )
                    page = pages[:, nl.ds(local_tile_id * _COMPACT, _COMPACT)]
                    nisa.dma_copy(
                        dst=page,
                        src=compact_pool.ap(
                            [[_COMPACT, _TILE], [1, _COMPACT]],
                            offset=0,
                            vector_offset=slots,
                            indirect_dim=0,
                        ),
                        oob_mode=nisa.oob_mode.skip,
                    )
                    _score_tile(page, query_tiles, query_pos_t, scale, scores[:, nl.ds(local_tile_id * _TILE, _TILE)])
                nisa.dma_copy(
                    dst=valid_heads,
                    src=history_mask[batch_id, supertile_id, :, :],
                )
                _online_supertile_update(
                    pages,
                    valid_heads,
                    scores,
                    running_max,
                    running_sum,
                    running_output,
                )

            current_page = nl.ndarray(
                (_TILE, _COMPACT), dtype=compact_new.dtype, buffer=nl.sbuf
            )
            nisa.memset(current_page, 0.0)
            nisa.dma_copy(
                dst=current_page[nl.ds(0, 1), :],
                src=compact_new[batch_id : batch_id + 1, :],
            )
            current_scores = nl.ndarray((_HEADS, _TILE), dtype=nl.float32, buffer=nl.sbuf)
            _score_tile(current_page, query_tiles, query_pos_t, scale, current_scores)
            _online_current_update(
                current_page,
                current_scores,
                running_max,
                running_sum,
                running_output,
            )

            nisa.reciprocal(dst=running_sum, data=running_sum)
            nisa.tensor_tensor(
                dst=running_output,
                data1=running_output,
                data2=broadcast_column(running_sum, _LATENT),
                op=nl.multiply,
            )
            value = nl.ndarray((_HEADS, _LATENT), dtype=ql_nope.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=value, src=running_output)
            nisa.dma_copy(dst=output[batch_id, :, :], src=value)

        return output


def paged_absorbed_mla(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compact_new: torch.Tensor,
    compact_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    active_slots: torch.Tensor | None = None,
    history_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Read compact MLA slots once and return attended latent output."""
    batch, heads, latent_dim = ql_nope.shape
    context = block_tables.shape[1]
    _validate_kernel_contract(
        batch=batch,
        heads=heads,
        context=context,
        dtype=ql_nope.dtype,
    )
    if latent_dim != _LATENT or q_pe.shape != (batch, heads, _ROPE):
        raise ValueError("paged MLA requires q latent 512 and q RoPE 64")
    if not _NKI_AVAILABLE:
        raise RuntimeError(
            f"DeepSeek decode MLA requires NKI; it failed to import "
            f"({_NKI_IMPORT_ERR!r}). There is no Torch fallback."
        )
    compact_new = compact_new.reshape(batch, _COMPACT)
    compact_pool = compact_pool.reshape(compact_pool.shape[0], _COMPACT)
    if active_slots is None or history_mask is None:
        active_slots, history_mask = build_paged_mla_metadata(
            block_tables, seq_lens
        )
    # The HOP is registered by @wrap_nki at module import. B/T tensor shapes
    # and the scalar scale specialize torch.compile directly, matching the
    # standard MHA wrapper path without a trace-time nki_op factory.
    # Launched at the compiler's logical core count (see logical_cores); the
    # programs split the rows.
    return _paged_absorbed_mla_kernel[logical_cores()](
        ql_nope,
        q_pe,
        compact_new,
        compact_pool,
        active_slots,
        history_mask,
        float(scale),
    )
