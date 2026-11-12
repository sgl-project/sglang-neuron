# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM-5.2 MTP layer prologue: the masked embedding, the two input norms, and ``eh_proj``.

The reference computes::

    e = inputs_embeds * (positions != 0)      # position 0 has no predecessor to predict from
    h = previous_hidden                       # the target's post-norm hidden state
    x = concat(enorm(e), hnorm(h)) @ eh_proj  # [T, 2H] @ [2H, H] -> [T, H]

and ``x`` is the decoder layer's ``hidden_states`` input. The whole MTP layer ships in bf16, so these are plain bf16
matmuls. ``eh_proj`` is taken transposed, ``[2H, H]``, like every other projection weight here.

A replicated ``eh_proj`` would read 151 MB per rank per draft step, so the draft chain shards it by output column
over a group of ranks and keeps the collectives itself. This module supplies the collective-free pieces:
``mtp_prologue_norms`` writes this rank's normed ``concat(e, h)`` rows, and ``eh_proj_columns`` multiplies the
group's gathered rows by this rank's column shard. ``T`` is read from the inputs (up to 128). LNC=1 only.
"""

import nki.isa as nisa
import nki.language as nl

from ..attention.mla.mla_decode_constants import F_MAX, P_MAX
from ..utils.kernel_assert import kernel_assert
from .attention.mla.mla_proj_primitives import load_gamma_broadcast, rmsnorm_inplace, rmsnorm_into

_ERROR_PREFIX = "[GLM-5.2 MTP Prologue Kernel]"

MAX_TOKENS = P_MAX
"""Token cap. T is the stationary operand's free dimension in every ``eh_proj`` matmul, and the PE
array is 128 wide. Production is 2 (one rank's ``t_shard`` at C32 x spec8 over 128 ranks)."""


def mtp_prologue_norms(
    inputs_embeds: nl.NkiTensor,
    previous_hidden: nl.NkiTensor,
    positions: nl.NkiTensor,
    enorm_weights: nl.NkiTensor,
    hnorm_weights: nl.NkiTensor,
    normed_out: nl.NkiTensor,
    eps: float = 1e-5,
    gamma_after: list = None,
    stores_out: list = None,
):
    """
    Writes this rank's normed ``concat(enorm(e), hnorm(h))`` rows, the input of a sharded ``eh_proj``.

    The reference's position-0 mask and its two norms (the module docstring), each half rounded to bf16 once.
    The caller gathers these rows over its shard group and gives them to :func:`eh_proj_columns` -- or,
    with ``normed_out=None``, keeps the two normed halves in SBUF and hands them to it directly.

    Dimensions:
        T: Token slots this rank owns.
        H: Hidden dimension (6144).

    Args:
        inputs_embeds (nl.NkiTensor): [T, H] @ HBM or SBUF, the embeddings of the draft tokens.
        previous_hidden (nl.NkiTensor): [T, H] @ HBM or SBUF, the target's post-final-norm hidden state.
        positions (nl.NkiTensor): [T, 1] int32 @ HBM, each slot's absolute position (the embedding is
            zeroed at position 0), or None to skip the position-0 mask -- right for every MTP draft step
            after the first, whose positions are the target's plus at least 2.
        enorm_weights (nl.NkiTensor): [1, H] @ HBM. ``enorm.weight``.
        hnorm_weights (nl.NkiTensor): [1, H] @ HBM. ``hnorm.weight``.
        normed_out (nl.NkiTensor): [T, 2H] @ HBM, written here, or None. Columns ``[0, H)`` hold the
            normed embedding and columns ``[H, 2H)`` the normed previous hidden.
        eps (float): RMSNorm epsilon, shared by both norms.
        gamma_after (list): Optional instructions the two gamma loads are ordered after (``depends_on``), so
            they stay out of a critical window that precedes them (the draft chain's previous head).
        stores_out (list): Optional; with ``normed_out``, receives the two stores of the normed halves.

    Returns:
        None, or with ``normed_out=None`` the ``(embeds, hidden)`` normed halves, each [P_MAX, H] @ SBUF
        with rows ``[0, T)`` valid.
    """

    T, H = inputs_embeds.shape
    kernel_assert(
        T <= MAX_TOKENS and H % P_MAX == 0 and (normed_out is None or normed_out.shape == (T, 2 * H)),
        f"{_ERROR_PREFIX} norms need T <= {MAX_TOKENS}, H % {P_MAX} == 0 and normed_out [T, 2H] = "
        f"{(T, 2 * H)}, got T={T}, H={H}, normed_out {None if normed_out is None else normed_out.shape}",
    )
    embeds_sbuf, hidden_sbuf = _normed_halves(
        inputs_embeds, previous_hidden, positions, enorm_weights, hnorm_weights, eps, gamma_after
    )
    if normed_out is None:
        return embeds_sbuf, hidden_sbuf
    stores = [
        nisa.dma_copy(dst=normed_out[0:T, 0:H], src=embeds_sbuf[0:T, 0:H]),
        nisa.dma_copy(dst=normed_out[0:T, H : 2 * H], src=hidden_sbuf[0:T, 0:H]),
    ]
    if stores_out is not None:
        stores_out.extend(stores)


def eh_proj_columns(
    normed_rows: nl.NkiTensor,
    eh_proj_cols: nl.NkiTensor,
    out: nl.NkiTensor,
    k_tiles_per_dma: int = 8,
    pretransposed: tuple = None,
) -> None:
    """
    Computes ``out = normed_rows @ eh_proj_cols`` for one column shard of ``eh_proj``.

    The shard holds all ``2H`` contraction rows and ``N`` of the ``H`` output columns, so this rank
    reads ``2H * N`` weight bytes instead of ``2H * H``. Each output element accumulates the whole
    contraction in increasing k-tile order, in one fp32 PSUM bank, and rounds once on the drain. Against a
    replicated ``eh_proj`` the result is not bit-identical but gives the same draft tokens.

    Three choices fit the narrow shard and the wide token count:

    * The rows arrive concatenated, so one pass transposes all ``2H / 128`` k-tiles. A PSUM tile holds
      ``F_MAX / T_all`` transposed k-tiles, so one copy evicts that many.
    * ``N`` fits one PSUM bank, so the output loop has one pass and the weight is read once.
    * One DMA loads ``k_tiles_per_dma`` k-tiles of the weight, not one. Partition ``p`` of k-tile
      ``kt`` reads row ``128 * kt + p``, which is one contiguous run of ``N`` elements.

    Dimensions:
        T_all: Token rows of the shard group (``T_shard * group size``; 32 in production).
        K: Contraction width, ``2H`` (12288).
        N: Output columns of this shard (``H / group size``; 384 in production).

    Args:
        normed_rows (nl.NkiTensor | tuple): [T_all, K] @ HBM, the group's rows from :func:`mtp_prologue_norms`;
            or a tuple of SBUF tiles whose rows ``[0, T_all)`` hold consecutive column blocks of those rows
            (the ``(embeds, hidden)`` halves :func:`mtp_prologue_norms` returns), which skips the load.
        eh_proj_cols (nl.NkiTensor): [K, N] @ HBM. Columns of ``eh_proj.weight`` TRANSPOSED, the slice
            ``[:, N * p : N * (p + 1)]`` of the ``[2H, H]`` weight for group position ``p``.
        out (nl.NkiTensor): [T_all, N] @ HBM or SBUF, written here.
        k_tiles_per_dma (int): Weight k-tiles in each DMA.
        pretransposed (tuple): Optional ``(row_tiles, k0)`` from :func:`eh_proj_row_tiles`: k-tiles ``[k0, K / 128)``
            are already transposed into ``row_tiles``, and the ``normed_rows`` tuple (empty for ``k0 = 0``) supplies
            k-tiles ``[0, k0)``. The draft chain transposes the normed previous hidden this way while the embedding is
            still in flight; the matmuls and their k order are the same.

    Returns:
        None
    """

    k0 = 0
    if isinstance(normed_rows, (tuple, list)):
        halves = list(normed_rows)
        T_all, K = out.shape[0], sum(h.shape[1] for h in halves)
        if pretransposed is not None:
            k0 = K // P_MAX
            K = eh_proj_cols.shape[0]
    else:
        T_all, K = normed_rows.shape
        rows_sbuf = nl.ndarray((P_MAX, K), dtype=normed_rows.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows_sbuf[0:T_all, 0:K], src=normed_rows[0:T_all, 0:K])
        halves = [rows_sbuf]
    N = eh_proj_cols.shape[1]
    kernel_assert(
        T_all <= P_MAX and all(h.shape[1] % P_MAX == 0 for h in halves) and eh_proj_cols.shape[0] == K and N <= F_MAX,
        f"{_ERROR_PREFIX} eh_proj_columns needs T_all <= {P_MAX}, K % {P_MAX} == 0 per block, eh_proj_cols [K, N] "
        f"with N <= {F_MAX}, got K={K}, cols {eh_proj_cols.shape}",
    )
    kernel_assert(
        out.shape == (T_all, N),
        f"{_ERROR_PREFIX} eh_proj_columns out must be {(T_all, N)}, got {out.shape}",
    )
    n_k_tiles = K // P_MAX

    # ═══════ k on partitions: k-tile kt of the rows lands at columns [kt * T_all, (kt + 1) * T_all) ═══════
    if pretransposed is None:
        row_tiles = eh_proj_row_tiles(halves, T_all, n_k_tiles)
    else:
        row_tiles, pre_k0 = pretransposed
        kernel_assert(
            pre_k0 == k0 and row_tiles.shape == (P_MAX, n_k_tiles * T_all),
            f"{_ERROR_PREFIX} pretransposed k-tiles must start where normed_rows ends ({k0}), got {pre_k0}",
        )
        if halves:
            eh_proj_row_tiles(halves, T_all, n_k_tiles, row_tiles=row_tiles)

    # ═══════ one PSUM bank accumulates the whole contraction ═══════
    acc_psum = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum)
    for c0 in range(0, n_k_tiles, k_tiles_per_dma):
        cn = min(k_tiles_per_dma, n_k_tiles - c0)
        weight_sbuf = nl.ndarray((P_MAX, k_tiles_per_dma, N), dtype=eh_proj_cols.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=weight_sbuf[0:P_MAX, 0:cn, 0:N],
            src=eh_proj_cols.ap(pattern=[[N, P_MAX], [P_MAX * N, cn], [1, N]], offset=c0 * P_MAX * N),
            dge_mode=nisa.dge_mode.none,
        )
        for j in range(cn):
            kt = c0 + j
            nisa.nc_matmul(
                dst=acc_psum[0:T_all, 0:N],
                stationary=row_tiles[0:P_MAX, kt * T_all : (kt + 1) * T_all],
                moving=weight_sbuf[0:P_MAX, j, 0:N],
                accumulate=kt != 0,
            )

    if out.buffer == nl.sbuf:
        nisa.tensor_copy(dst=out[0:T_all, 0:N], src=acc_psum[0:T_all, 0:N])
        return
    out_sbuf = nl.ndarray((P_MAX, N), dtype=out.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_sbuf[0:T_all, 0:N], src=acc_psum[0:T_all, 0:N])
    nisa.dma_copy(dst=out[0:T_all, 0:N], src=out_sbuf[0:T_all, 0:N])


def eh_proj_row_tiles(
    blocks: list,
    T_all: int,
    n_k_tiles: int,
    k0: int = 0,
    row_tiles: nl.NkiTensor = None,
    evict_handles_out: list = None,
) -> nl.NkiTensor:
    """
    Transposes SBUF row blocks into :func:`eh_proj_columns`' stationary layout, k on partitions.

    K-tile ``kt`` of the rows lands at columns ``[kt * T_all, (kt + 1) * T_all)`` of ``row_tiles``; the blocks
    supply consecutive k-tiles from ``k0`` on. A PSUM tile holds ``F_MAX / T_all`` transposed k-tiles, so one copy
    evicts that many; ``k0`` must be a multiple of that, so a group never straddles two calls.

    Args:
        blocks (list): SBUF tiles whose rows ``[0, T_all)`` hold consecutive column blocks of the rows.
        T_all (int): Rows.
        n_k_tiles (int): K-tiles of the whole contraction (the width of ``row_tiles``).
        k0 (int): The first k-tile the blocks supply.
        row_tiles (nl.NkiTensor): [P_MAX, n_k_tiles * T_all] to fill, or None for a fresh tile.
        evict_handles_out (list): Optional; receives each group's PSUM eviction instruction, in k order.

    Returns:
        nl.NkiTensor: ``row_tiles``.
    """
    dtype = blocks[0].dtype
    per_psum = F_MAX // T_all
    kernel_assert(k0 % per_psum == 0, f"{_ERROR_PREFIX} k0={k0} must be a multiple of {per_psum} k-tiles")
    if row_tiles is None:
        row_tiles = nl.ndarray((P_MAX, n_k_tiles * T_all), dtype=dtype, buffer=nl.sbuf)
    # (block, column) of each k-tile across the row blocks.
    k_tile_src = []
    for h in blocks:
        for c in range(h.shape[1] // P_MAX):
            k_tile_src.append((h, c * P_MAX))
    k1 = k0 + len(k_tile_src)
    for g0 in range(k0, k1, per_psum):
        gn = min(per_psum, k1 - g0)
        transposed_psum = nl.ndarray((P_MAX, F_MAX), dtype=dtype, buffer=nl.psum)
        for j in range(gn):
            block, k_start = k_tile_src[g0 - k0 + j]
            nisa.nc_transpose(
                transposed_psum[0:P_MAX, j * T_all : (j + 1) * T_all],
                block[0:T_all, k_start : k_start + P_MAX],
            )
        evict = nisa.tensor_copy(
            dst=row_tiles[0:P_MAX, g0 * T_all : (g0 + gn) * T_all],
            src=transposed_psum[0:P_MAX, 0 : gn * T_all],
        )
        if evict_handles_out is not None:
            evict_handles_out.append(evict)
    return row_tiles


def mtp_prologue_row_norm(
    x_sb: nl.NkiTensor,
    gamma_hbm: nl.NkiTensor,
    eps: float,
    gamma_after: list = None,
    scale_blocks: int = 1,
    between_blocks=None,
) -> nl.NkiTensor:
    """
    One of the two prologue norms on SBUF rows (the draft chain's rows layout), for a caller that runs them apart.

    :func:`_normed_halves` on an SBUF input, instruction for instruction (``load_gamma_broadcast``, then
    ``rmsnorm_into``), with an optional ordering of the gamma load and the elementwise scaling split into
    ``scale_blocks`` column blocks, so the first k-tiles' transposes can start before the last columns are scaled.
    The sum of squares stays one instruction, and each element is scaled by the same instruction arithmetic either
    way: bit-identical.

    Args:
        x_sb (nl.NkiTensor): [T, H] @ SBUF.
        gamma_hbm (nl.NkiTensor): [1, H] @ HBM, the norm's weight.
        eps (float): RMSNorm epsilon.
        gamma_after (list): Optional instructions the gamma load is ordered after.
        scale_blocks (int): Column blocks of the scaling; ``H / scale_blocks`` must be a multiple of 128.
        between_blocks: Optional ``f(b, out)`` called after scaling block ``b`` (the consumer of its columns, traced
            in between); an instruction it returns orders block ``b + 1`` after it on the scaling's engine.

    Returns:
        nl.NkiTensor: [P_MAX, H] @ SBUF, rows ``[0, T)`` valid.
    """
    T, H = x_sb.shape
    kernel_assert(
        H % (scale_blocks * P_MAX) == 0, f"{_ERROR_PREFIX} H={H} must split into {scale_blocks} k-tile column blocks"
    )
    out = nl.ndarray((P_MAX, H), dtype=x_sb.dtype, buffer=nl.sbuf)
    gamma_sb = nl.ndarray((P_MAX, H), dtype=nl.float32, buffer=nl.sbuf)
    gamma_dma = nisa.dma_copy(dst=gamma_sb[0:T, 0:H], src=gamma_hbm[0:1, 0:H].broadcast(0, T))
    if gamma_after:
        gamma_dma.depends_on(*gamma_after)
    squares_discard = nl.ndarray((P_MAX, H), dtype=nl.float32, buffer=nl.sbuf)
    inv_rms = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(
        dst=squares_discard[0:T, 0:H], op=nl.square, data=x_sb[0:T, 0:H], reduce_op=nl.add, reduce_res=inv_rms[0:T, 0:1]
    )
    nisa.activation(inv_rms[0:T, 0:1], op=nl.rsqrt, data=inv_rms[0:T, 0:1], scale=1.0 / H, bias=eps)
    width = H // scale_blocks
    block_after = None
    for b in range(scale_blocks):
        c0, c1 = b * width, (b + 1) * width
        # scalar_tensor_tensor returns the bare traced op, not an NkiInstruction handle.
        scale = nisa.NkiInstruction(
            nisa.scalar_tensor_tensor(
                dst=out[0:T, c0:c1],
                data=x_sb[0:T, c0:c1],
                op0=nl.multiply,
                operand0=inv_rms[0:T, 0:1],
                op1=nl.multiply,
                operand1=gamma_sb[0:T, c0:c1],
            )
        )
        if block_after is not None:
            scale.depends_on(block_after)
        block_after = None if between_blocks is None else between_blocks(b, out)
    return out


def _normed_halves(
    inputs_embeds: nl.NkiTensor,
    previous_hidden: nl.NkiTensor,
    positions: nl.NkiTensor,
    enorm_weights: nl.NkiTensor,
    hnorm_weights: nl.NkiTensor,
    eps: float,
    gamma_after: list = None,
) -> tuple:
    """
    Loads the masked embedding and the previous hidden, and norms each in its own bf16 staging tile.

    An input already in SBUF is normed straight into a fresh tile (the same instructions, out of place),
    so it needs no load and the caller's tile is left as it was. ``gamma_after``: see :func:`mtp_prologue_norms`.

    Returns:
        tuple: ``(embeds_sbuf, hidden_sbuf)``, each [P_MAX, H] @ SBUF with rows ``[0, T)`` valid.
    """

    T, H = inputs_embeds.shape
    io_dtype = inputs_embeds.dtype
    kernel_assert(
        positions is None or inputs_embeds.buffer != nl.sbuf,
        f"{_ERROR_PREFIX} the position-0 mask needs inputs_embeds in HBM",
    )

    def normed(x, gamma_hbm, mask_positions):
        out = nl.ndarray((P_MAX, H), dtype=io_dtype, buffer=nl.sbuf)
        if gamma_after:
            # load_gamma_broadcast's DMA, ordered after gamma_after.
            gamma_sb = nl.ndarray((P_MAX, H), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gamma_sb[0:T, 0:H], src=gamma_hbm[0:1, 0:H].broadcast(0, T)).depends_on(*gamma_after)
        else:
            gamma_sb = load_gamma_broadcast(gamma_hbm, T)
        if x.buffer == nl.sbuf:
            rmsnorm_into(out[0:T, 0:H], x[0:T, 0:H], gamma_sb, T, eps)
            return out
        nisa.dma_copy(dst=out[0:T, 0:H], src=x[0:T, 0:H])
        if mask_positions is not None:
            _mask_position_zero(out, mask_positions, T, H)
        rmsnorm_inplace(out[0:T, 0:H], gamma_sb, T, eps)
        return out

    embeds_sbuf = normed(inputs_embeds, enorm_weights, positions)
    hidden_sbuf = normed(previous_hidden, hnorm_weights, None)
    return embeds_sbuf, hidden_sbuf


def _mask_position_zero(embeds_sbuf: nl.NkiTensor, positions: nl.NkiTensor, T: int, H: int) -> None:
    """
    Zeroes the rows of ``embeds_sbuf`` whose position is 0, in place.

    A token at position 0 has no predecessor, so the reference feeds the projection a zero
    embedding there rather than the token's own. Implemented as a multiply by a per-partition 0/1
    column rather than a predicated copy, because the mask has to survive into the norm: zeroing
    after the norm would give the same answer -- ``rmsnorm(0) == 0`` -- but zeroing first is what
    the reference does and keeps the tapped intermediates comparable.
    """

    positions_int = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=positions_int[0:T, 0:1], src=positions[0:T, 0:1])

    # The compare runs in fp32: every position this model can hold (< 2**31) is exact in fp32, and
    # the result has to be a float anyway to multiply the bf16 embedding.
    positions_float = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=positions_float[0:T, 0:1], src=positions_int[0:T, 0:1])

    keep = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=keep[0:T, 0:1], data=positions_float[0:T, 0:1], op0=nl.not_equal, operand0=0.0)

    nisa.tensor_scalar(dst=embeds_sbuf[0:T, 0:H], data=embeds_sbuf[0:T, 0:H], op0=nl.multiply, operand0=keep[0:T, 0:1])
