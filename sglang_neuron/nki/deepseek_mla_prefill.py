"""Direct compact-KV prefill MLA for DeepSeek-V2-Lite.

Stock ``attention_cte`` requires one shared QK/V dimension and its current
value/output path is limited to 512 channels.  Absorbed MLA instead has
QK=512+64 and V=512.  This CTE-derived kernel keeps the useful execution
structure (128-token tiles, FP32 online softmax, no score spill) while making
that asymmetry explicit.  Prefix pages and the active suffix are merged in
SBUF; no dense compact-KV window is written to HBM.
"""

from __future__ import annotations

import torch

from sglang_neuron.nki import logical_cores
from sglang_neuron.nki.views import broadcast_column

_TILE = 128
_HEADS = 4
_LATENT = 512
_ROPE = 64
_COMPACT = _LATENT + _ROPE
_FLOAT32_MIN = -3.4028235e38
try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nki.isa import reduce_cmd
    from nki.language.opcode import maximum as _maximum
    from nkilib.core.utils.kernel_assert import kernel_assert
    from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
    from torch_neuronx import wrap_nki

    _NKI_AVAILABLE = True
    _NKI_IMPORT_ERR = None
except Exception as exc:  # noqa: BLE001  # pragma: no cover - Neuron image only
    _NKI_AVAILABLE = False
    _NKI_IMPORT_ERR = exc


def build_prefill_mla_metadata(
    block_tables: torch.Tensor,
    prefix_len: torch.Tensor,
    seq_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build O(W) indirect-load and causal-bound metadata.

    ``block_tables`` remains the serving allocator's physical token-slot table,
    so the metadata and NKI tensor contract are independent of page size.
    """
    if block_tables.dim() != 2 or block_tables.shape[0] != 1:
        raise ValueError("prefill MLA-CTE currently requires block_tables [1,W]")
    width = block_tables.shape[1]
    if width <= 0 or width % _TILE:
        raise ValueError(
            f"prefill MLA-CTE width must be a positive multiple of {_TILE}; "
            f"got {width}"
        )
    if prefix_len.numel() != 1 or seq_len.numel() != 1:
        raise ValueError("prefill MLA-CTE requires scalar prefix/sequence lengths")

    device = block_tables.device
    positions = torch.arange(width, dtype=torch.int32, device=device)
    prefix = prefix_len.reshape(1).to(torch.int32)
    length = seq_len.reshape(1).to(torch.int32)
    new_slots = (positions - prefix).clamp(min=0, max=width - 1)
    is_new = ((positions >= prefix) & (positions < length)).to(torch.bfloat16)
    upper_bounds = torch.minimum(positions + prefix, length - 1).to(torch.float32)
    tiles = width // _TILE
    return (
        block_tables.to(torch.int32).reshape(1, tiles, _TILE, 1),
        new_slots.reshape(1, tiles, _TILE, 1),
        is_new.reshape(1, tiles, _TILE, 1),
        upper_bounds.reshape(1, tiles, _TILE, 1),
    )


def _validate_kernel_contract(
    *, heads: int, context: int, dtype: torch.dtype
) -> None:
    if heads != _HEADS:
        raise ValueError(
            f"prefill MLA-CTE requires four TP-local heads; got {heads}"
        )
    if context <= 0 or context % _TILE:
        raise ValueError(
            f"prefill MLA-CTE context must be a positive multiple of {_TILE}; "
            f"got {context}"
        )
    if dtype != torch.bfloat16:
        raise ValueError("prefill MLA-CTE requires BF16")


if _NKI_AVAILABLE:

    def _load_query_tiles(ql_nope, q_pe, head_id, q_tile_id):
        q_start = q_tile_id * _TILE
        q_tiles = nl.ndarray((_TILE, _LATENT), dtype=ql_nope.dtype, buffer=nl.sbuf)
        for latent_tile in nl.affine_range(_LATENT // _TILE):
            query = nl.ndarray((_TILE, _TILE), dtype=ql_nope.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=query,
                src=ql_nope[
                    0,
                    head_id,
                    nl.ds(q_start, _TILE),
                    nl.ds(latent_tile * _TILE, _TILE),
                ],
            )
            query_t_psum = nl.ndarray(
                (_TILE, _TILE), dtype=ql_nope.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(dst=query_t_psum, data=query)
            nisa.tensor_copy(
                dst=q_tiles[:, nl.ds(latent_tile * _TILE, _TILE)],
                src=query_t_psum,
            )

        query_pos = nl.ndarray((_TILE, _ROPE), dtype=q_pe.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=query_pos,
            src=q_pe[0, head_id, nl.ds(q_start, _TILE), :],
        )
        query_pos_t_psum = nl.ndarray((_ROPE, _TILE), dtype=q_pe.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=query_pos_t_psum, data=query_pos)
        query_pos_t = nl.ndarray((_ROPE, _TILE), dtype=q_pe.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=query_pos_t, src=query_pos_t_psum)
        return q_tiles, query_pos_t

    def _score_query_key_tile(page, query_tiles, query_pos_t, scale):
        # Accumulate K @ Q^T as [K,Q], then transpose to [Q,K].
        score_kq = nl.ndarray((_TILE, _TILE), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(score_kq, 0.0)
        for latent_tile in nl.affine_range(_LATENT // _TILE):
            latent_chunk = page[:, nl.ds(latent_tile * _TILE, _TILE)]
            latent_t_psum = nl.ndarray((_TILE, _TILE), dtype=page.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=latent_t_psum, data=latent_chunk)
            latent_t = nl.ndarray((_TILE, _TILE), dtype=page.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=latent_t, src=latent_t_psum)
            nisa.nc_matmul(
                dst=score_kq,
                stationary=latent_t,
                moving=query_tiles[:, nl.ds(latent_tile * _TILE, _TILE)],
            )
        key_pos = page[:, nl.ds(_LATENT, _ROPE)]
        key_pos_t_psum = nl.ndarray((_ROPE, _TILE), dtype=page.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=key_pos_t_psum, data=key_pos)
        key_pos_t = nl.ndarray((_ROPE, _TILE), dtype=page.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=key_pos_t, src=key_pos_t_psum)
        nisa.nc_matmul(dst=score_kq, stationary=key_pos_t, moving=query_pos_t)
        score_kq_sbuf = nl.ndarray((_TILE, _TILE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=score_kq_sbuf, src=score_kq)
        scores_psum = nl.ndarray((_TILE, _TILE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=scores_psum, data=score_kq_sbuf)
        scores = nl.ndarray((_TILE, _TILE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=scores, src=scores_psum)
        nisa.tensor_scalar(dst=scores, data=scores, op0=nl.multiply, operand0=scale)
        return scores

    def _online_stats_update(scores, tile_max, running_max, running_sum):
        """Update FP32 softmax statistics without quantizing value weights."""
        next_max = nl.ndarray((_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=next_max, data1=running_max, data2=tile_max, op=nl.maximum
        )
        correction = nl.ndarray((_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=correction, data1=running_max, data2=next_max, op=nl.subtract
        )
        nisa.activation(dst=correction, op=nl.exp, data=correction)
        nisa.tensor_tensor(
            dst=scores,
            data1=scores,
            data2=broadcast_column(next_max, _TILE),
            op=nl.subtract,
        )
        nisa.activation(dst=scores, op=nl.exp, data=scores)
        tile_sum = nl.ndarray((_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=tile_sum, op=nl.add, data=scores, axis=[1], keepdims=True
        )
        nisa.tensor_tensor(
            dst=running_sum, data1=running_sum, data2=correction, op=nl.multiply
        )
        nisa.tensor_tensor(
            dst=running_sum, data1=running_sum, data2=tile_sum, op=nl.add
        )
        nisa.tensor_copy(dst=running_max, src=next_max)

    @wrap_nki
    @nki.jit
    def _prefill_mla_kernel(
        ql_nope,
        q_pe,
        compact_new,
        compact_pool,
        cache_slots,
        new_slots,
        is_new,
        upper_bounds,
        scale,
    ):
        batch, heads, context, latent_dim = ql_nope.shape
        _, q_heads, q_context, rope_dim = q_pe.shape
        new_rows, compact_dim = compact_new.shape
        _, pool_compact_dim = compact_pool.shape
        meta_batch, num_tiles, tile_rows, one = cache_slots.shape
        kernel_assert(batch == 1 and meta_batch == 1, "prefill MLA-CTE requires B1")
        kernel_assert(
            heads == _HEADS and q_heads == heads, "prefill MLA-CTE expects H4"
        )
        kernel_assert(
            context == q_context and context > 0 and context % _TILE == 0,
            "prefill MLA-CTE context must contain complete 128-token tiles",
        )
        kernel_assert(
            latent_dim == _LATENT and rope_dim == _ROPE, "MLA dimensions mismatch"
        )
        kernel_assert(new_rows == context, "active compact width mismatch")
        kernel_assert(
            compact_dim == _COMPACT and pool_compact_dim == _COMPACT,
            "compact MLA dimension mismatch",
        )
        kernel_assert(num_tiles * _TILE == context, "tile/context mismatch")
        kernel_assert(tile_rows == _TILE and one == 1, "metadata tile mismatch")

        # Merge paged prefix and active suffix once in SBUF.  Every head and Q
        # tile reuses these compact pages without an HBM dense-window spill.
        pages = nl.ndarray(
            (_TILE, num_tiles * _COMPACT),
            dtype=compact_pool.dtype,
            buffer=nl.sbuf,
        )
        for key_tile in nl.affine_range(num_tiles):
            cached_page = nl.ndarray(
                (_TILE, _COMPACT), dtype=compact_pool.dtype, buffer=nl.sbuf
            )
            active_page = nl.ndarray(
                (_TILE, _COMPACT), dtype=compact_pool.dtype, buffer=nl.sbuf
            )
            cache_indices = nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.sbuf)
            active_indices = nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cache_indices, src=cache_slots[0, key_tile, :, :])
            nisa.dma_copy(dst=active_indices, src=new_slots[0, key_tile, :, :])
            # the load skips a -1 slot (past the request's pages): its rows stay zero, not what SBUF last held,
            # since a masked row still enters the scores and P @ V, where 0 x NaN is NaN
            nisa.memset(cached_page, 0.0)
            nisa.dma_copy(
                dst=cached_page,
                src=compact_pool.ap(
                    [[_COMPACT, _TILE], [1, _COMPACT]],
                    vector_offset=cache_indices,
                    indirect_dim=0,
                ),
                oob_mode=nisa.oob_mode.skip,
            )
            nisa.dma_copy(
                dst=active_page,
                src=compact_new.ap(
                    [[_COMPACT, _TILE], [1, _COMPACT]],
                    vector_offset=active_indices,
                    indirect_dim=0,
                ),
                oob_mode=nisa.oob_mode.skip,
            )
            mask = nl.ndarray((_TILE, 1), dtype=is_new.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=mask, src=is_new[0, key_tile, :, :])
            inverse = nl.ndarray((_TILE, 1), dtype=is_new.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=inverse,
                data=mask,
                op0=nl.multiply,
                operand0=-1.0,
                op1=nl.add,
                operand1=1.0,
            )
            cached_selected = nl.ndarray(
                (_TILE, _COMPACT), dtype=compact_pool.dtype, buffer=nl.sbuf
            )
            active_selected = nl.ndarray(
                (_TILE, _COMPACT), dtype=compact_pool.dtype, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=cached_selected,
                data1=cached_page,
                data2=broadcast_column(inverse, _COMPACT),
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=active_selected,
                data1=active_page,
                data2=broadcast_column(mask, _COMPACT),
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=pages[:, nl.ds(key_tile * _COMPACT, _COMPACT)],
                data1=cached_selected,
                data2=active_selected,
                op=nl.add,
            )

        output = nl.ndarray(
            (1, _HEADS, context, _LATENT),
            dtype=ql_nope.dtype,
            buffer=nl.shared_hbm,
        )
        zero_bound = nl.ndarray((_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(zero_bound, 0.0)
        # each program of an LNC2 launch attends its share of the heads
        _, programs, program = get_verified_program_sharding_info("prefill_absorbed_mla_cte", max_sharding=2)
        kernel_assert(_HEADS % programs == 0, "prefill MLA-CTE heads must split evenly over the programs")
        share = _HEADS // programs
        for head in range(share):
            head_id = program * share + head
            for query_tile in nl.affine_range(num_tiles):
                query_tiles, query_pos_t = _load_query_tiles(
                    ql_nope, q_pe, head_id, query_tile
                )
                bounds = nl.ndarray((_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=bounds, src=upper_bounds[0, query_tile, :, :])
                running_max = nl.ndarray((_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                running_sum = nl.ndarray((_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(running_max, -10000.0)
                nisa.memset(running_sum, 0.0)

                # Pass 1 computes the global FP32 max and denominator before
                # probabilities cross the BF16 TensorE input boundary.
                for key_tile in nl.affine_range(num_tiles):
                    page = pages[:, nl.ds(key_tile * _COMPACT, _COMPACT)]
                    scores = _score_query_key_tile(
                        page,
                        query_tiles,
                        query_pos_t,
                        scale,
                    )
                    masked_scores = nl.ndarray(
                        (_TILE, _TILE), dtype=nl.float32, buffer=nl.sbuf
                    )
                    tile_max = nl.ndarray((_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.range_select(
                        masked_scores,
                        on_true_tile=scores,
                        on_false_value=_FLOAT32_MIN,
                        comp_op0=nl.greater_equal,
                        comp_op1=nl.less_equal,
                        bound0=zero_bound,
                        bound1=bounds,
                        reduce_op=_maximum,
                        reduce_res=tile_max,
                        reduce_cmd=reduce_cmd.reset_reduce,
                        range_start=key_tile * _TILE,
                    )
                    _online_stats_update(
                        masked_scores, tile_max, running_max, running_sum
                    )

                nisa.reciprocal(dst=running_sum, data=running_sum)
                # Pass 2 normalizes in FP32 before the BF16 probability input
                # to the TensorE value matmul.
                final_output_psum = nl.ndarray(
                    (_TILE, _LATENT), dtype=nl.float32, buffer=nl.psum
                )
                nisa.memset(final_output_psum, 0.0)
                for key_tile in nl.affine_range(num_tiles):
                    page = pages[:, nl.ds(key_tile * _COMPACT, _COMPACT)]
                    scores = _score_query_key_tile(
                        page,
                        query_tiles,
                        query_pos_t,
                        scale,
                    )
                    masked_scores = nl.ndarray(
                        (_TILE, _TILE), dtype=nl.float32, buffer=nl.sbuf
                    )
                    ignored_max = nl.ndarray(
                        (_TILE, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.range_select(
                        masked_scores,
                        on_true_tile=scores,
                        on_false_value=_FLOAT32_MIN,
                        comp_op0=nl.greater_equal,
                        comp_op1=nl.less_equal,
                        bound0=zero_bound,
                        bound1=bounds,
                        reduce_op=_maximum,
                        reduce_res=ignored_max,
                        reduce_cmd=reduce_cmd.reset_reduce,
                        range_start=key_tile * _TILE,
                    )
                    nisa.tensor_tensor(
                        dst=masked_scores,
                        data1=masked_scores,
                        data2=broadcast_column(running_max, _TILE),
                        op=nl.subtract,
                    )
                    nisa.activation(dst=masked_scores, op=nl.exp, data=masked_scores)
                    nisa.tensor_tensor(
                        dst=masked_scores,
                        data1=masked_scores,
                        data2=broadcast_column(running_sum, _TILE),
                        op=nl.multiply,
                    )
                    probabilities = nl.ndarray(
                        (_TILE, _TILE), dtype=page.dtype, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=probabilities, src=masked_scores)
                    prob_t_psum = nl.ndarray(
                        (_TILE, _TILE), dtype=page.dtype, buffer=nl.psum
                    )
                    nisa.nc_transpose(dst=prob_t_psum, data=probabilities)
                    prob_t = nl.ndarray(
                        (_TILE, _TILE), dtype=page.dtype, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=prob_t, src=prob_t_psum)
                    nisa.nc_matmul(
                        dst=final_output_psum,
                        stationary=prob_t,
                        moving=page[:, nl.ds(0, _LATENT)],
                    )
                value = nl.ndarray(
                    (_TILE, _LATENT), dtype=ql_nope.dtype, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=value, src=final_output_psum)
                nisa.dma_copy(
                    dst=output[
                        0,
                        head_id,
                        nl.ds(query_tile * _TILE, _TILE),
                        :,
                    ],
                    src=value,
                )
        return output


def prefill_absorbed_mla_cte(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compact_new: torch.Tensor,
    compact_pool: torch.Tensor,
    block_tables: torch.Tensor,
    prefix_len: torch.Tensor,
    seq_len: torch.Tensor,
    *,
    scale: float,
    metadata: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
    ) = None,
) -> torch.Tensor:
    if ql_nope.dim() != 4 or ql_nope.shape[:2] != (1, _HEADS):
        raise ValueError(
            f"prefill MLA-CTE q latent must be [1,4,W,512], got {ql_nope.shape}"
        )
    context = ql_nope.shape[2]
    if ql_nope.shape[3] != _LATENT or q_pe.shape != (1, _HEADS, context, _ROPE):
        raise ValueError("prefill MLA-CTE query dimensions do not match absorbed MLA")
    _validate_kernel_contract(
        heads=ql_nope.shape[1],
        context=context,
        dtype=ql_nope.dtype,
    )
    if not _NKI_AVAILABLE:
        raise RuntimeError(
            f"DeepSeek prefill MLA-CTE requires NKI; it failed to import "
            f"({_NKI_IMPORT_ERR!r}). There is no Torch fallback."
        )
    if metadata is None:
        metadata = build_prefill_mla_metadata(
            block_tables, prefix_len, seq_len
        )
    # The HOP is registered by @wrap_nki when this module is imported. Tensor
    # shapes and the scalar scale specialize the surrounding torch.compile
    # graph; no Python operator factory runs during Dynamo tracing.
    # Launched at the compiler's logical core count (see logical_cores); the
    # programs split the heads.
    return _prefill_mla_kernel[logical_cores()](
        ql_nope,
        q_pe,
        compact_new.reshape(context, _COMPACT),
        compact_pool.reshape(compact_pool.shape[0], _COMPACT),
        *metadata,
        float(scale),
    )
