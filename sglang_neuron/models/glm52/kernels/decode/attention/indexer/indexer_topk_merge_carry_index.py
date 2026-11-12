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

"""
GLM-5.2 DSA indexer stage-2 MERGE, carry-index: each token's global top-k from the key group's packed local tiles,
published as the selected keys' global positions, so no gather of the positions follows the select.
"""


import nki
import nki.isa as nisa
import nki.language as nl

from sglang_neuron.models.glm52.kernels import topk_select

from ...utils.kernel_assert import kernel_assert
from .indexer_constants import P_MAX, SNAKE_PARTITIONS


# DGE transposes write their destination at this alignment (mirrors prior_index's constraint).
_DGE_DST_ALIGN_BYTES = 32



@nki.jit
def indexer_topk_merge_carry_index_packed(
    gathered: nl.NkiTensor,
    topk: int,
    degree: int,
    block_size: int,
    wide: bool = False,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """The packed CP merge: each token's global top-``topk`` from the key group's gathered packed tiles
    (``indexer_topk_local_packed``). The NKI select (``topk_select.select_cp_deal``) ranks the scores, carries each
    winner's payload code and rebuilds its global position from the code, its shard and the deal; ``DEAD_KEY`` comes out
    -1.

    Args:
        gathered: [degree * T * 16, 2 * (topk // 16)] uint32, the packed local tiles stacked in shard order.
        topk: keys per token, a multiple of 2048.
        degree: the number of key shards.
        block_size: the deal unit, a power of two.
        wide: refused; a chunked shard's tile is its own selection (``indexer_topk_own_selection_packed``).

    Returns:
        ``(positions_tm, positions_km)``: the selection token-major, [T, 16, topk // 16], and the same set key-major,
        [P_MAX, T, topk // P_MAX], the layout ``build_prior_index`` reads with one contiguous DMA per partition. The
        two slot orders differ; every consumer treats the slots as a set.
    """
    k_cols = topk // SNAKE_PARTITIONS  # per-shard columns per snake partition; also the output width
    rows = gathered.shape[0]
    P = rows // degree
    T = P // SNAKE_PARTITIONS
    n_cand = degree * topk
    n_cols = degree * k_cols
    n_tiles = topk // P_MAX
    kernel_assert(
        rows == degree * P and P == P_MAX and gathered.shape[1] == 2 * k_cols,
        f"packed merge expects [{degree}*{P_MAX}, {2 * k_cols}] uint32, got {tuple(gathered.shape)}",
    )
    kernel_assert(k_cols % 2 == 0, f"carry-index merge needs k_cols even (topk multiple of 32); got topk={topk}")
    # This one assert carries BOTH key-major requirements, because the second is IMPLIED by the first:
    # k_cols = topk // 16 being a multiple of P_MAX pins topk to [2048m, 2048m + 15], on which
    # n_tiles = topk // P_MAX is exactly 16m -- so the DGE transpose's per-token dst stride
    # n_tiles * 2 = 32m B is always a multiple of _DGE_DST_ALIGN_BYTES (32). A separate alignment check
    # could therefore never fire, so the requirement is stated here rather than silently dropped.
    kernel_assert(
        k_cols % P_MAX == 0,
        f"key-major output needs topk a multiple of {16 * P_MAX} so each snake row transposes onto whole "
        f"{P_MAX}-partition tiles AND the DGE transpose dst stays {_DGE_DST_ALIGN_BYTES}-byte aligned "
        f"(its per-token stride is {n_tiles * 2} B); got topk={topk}",
    )
    if block_size <= 0:
        raise ValueError(
            f"the packed merge rebuilds each position from its shard's deal, so it needs dealt shards and their deal "
            f"block_size; got block_size={block_size}"
        )
    if wide:
        raise ValueError(
            "the shares of a chunked shard are not merged: each is its CP rank's own selection "
            "(indexer_topk_own_selection_packed), which needs the key shard to be the keys the rank holds"
        )

    # one hardware-DGE DMA per snake straight from the gathered blocks, shard s -> column block s, with explicit
    # access patterns on both sides; nisa.topk reads the score tile as the local stage's topk reads its own
    width16 = 2 * gathered.shape[1]  # 16-bit elements per gathered row
    src_sync = nl.ndarray((P, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
    code_snake = nl.ndarray((P, n_cols), dtype=nl.uint16, buffer=nl.sbuf)
    for dst, dt, off, eng in ((src_sync, nl.bfloat16, 0, nisa.engine.sync),
                              (code_snake, nl.uint16, k_cols, nisa.engine.scalar)):
        nisa.dma_copy(
            dst=dst.ap(pattern=[[n_cols, P], [k_cols, degree], [1, k_cols]]),
            src=gathered.view(dt).ap(pattern=[[width16, P], [P * width16, degree], [1, k_cols]], offset=off),
            dge_mode=nisa.dge_mode.hwdge,
            engine=eng,
        )
    pos_i32 = topk_select.select_cp_deal(src_sync, code_snake, n_cand, topk, block_size=block_size,
                                         shard_degree=degree)
    return _selection_outputs(pos_i32, P, T, k_cols, n_tiles)


def _selection_outputs(pos_i32: nl.NkiTensor, P: int, T: int, k_cols: int, n_tiles: int):
    """A selection's two published forms from its [P, k_cols] int32 snake tile (-1 for an empty slot): token-major
    ``[T, 16, k_cols]`` and key-major ``[P_MAX, T, topk // P_MAX]`` (indexer_topk_merge_carry_index_packed's
    Returns)."""
    # --- token-major output: one order-preserving DMA (the stock convention). ---
    positions_tm = nl.ndarray((T, SNAKE_PARTITIONS, k_cols), dtype=nl.int32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=positions_tm.reshape((P, k_cols)), src=pos_i32[0:P, 0:k_cols])

    # one chunk (topk // 16 == P_MAX): ONE PE transpose of the positions as fp32, not two uint16 halves through HBM
    if k_cols == P_MAX:
        return positions_tm, _key_major_pe_transpose(pos_i32, P, T, n_tiles)

    # --- key-major output. Token t's snake block [16 (q), k_cols (c)] must land as [c % P_MAX, t,
    # (c // P_MAX) * 16 + q]: a hardware transpose of each 16 x P_MAX chunk. nisa.dma_transpose takes
    # 2-byte elements, so the int32 positions are split into low/high uint16 halves, each transposed
    # from an HBM staging copy, and recombined with a shift and an OR (-1 -> 0xFFFF | 0xFFFF<<16 = -1). ---
    chunks = k_cols // P_MAX
    halves_hbm = nl.ndarray((2, P, k_cols), dtype=nl.uint16, buffer=nl.private_hbm)
    for half in range(2):
        part_i32 = nl.ndarray((P, k_cols), dtype=nl.int32, buffer=nl.sbuf)
        if half == 0:
            nisa.tensor_scalar(
                dst=part_i32[0:P, 0:k_cols], data=pos_i32[0:P, 0:k_cols], op0=nl.bitwise_and, operand0=0xFFFF
            )
        else:
            nisa.tensor_scalar(
                dst=part_i32[0:P, 0:k_cols],
                data=pos_i32[0:P, 0:k_cols],
                op0=nl.right_shift,
                operand0=16,
                op1=nl.bitwise_and,
                operand1=0xFFFF,
            )
        part_u16 = nl.ndarray((P, k_cols), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=part_u16[0:P, 0:k_cols], src=part_i32[0:P, 0:k_cols])
        nisa.dma_copy(dst=halves_hbm[half], src=part_u16[0:P, 0:k_cols])
    km_i32 = None
    for half in range(2):
        km_u16 = nl.ndarray((P_MAX, T, n_tiles), dtype=nl.uint16, buffer=nl.sbuf)
        for t in range(T):
            for j in range(chunks):
                nisa.dma_transpose(
                    dst=km_u16[0:P_MAX, t, j * SNAKE_PARTITIONS : (j + 1) * SNAKE_PARTITIONS],
                    src=halves_hbm[
                        half, t * SNAKE_PARTITIONS : (t + 1) * SNAKE_PARTITIONS, j * P_MAX : (j + 1) * P_MAX
                    ],
                    dge_mode=nisa.dge_mode.hwdge,
                )
        part = nl.ndarray((P_MAX, T, n_tiles), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=part[0:P_MAX, 0:T, 0:n_tiles], src=km_u16[0:P_MAX, 0:T, 0:n_tiles])
        if half == 0:
            km_i32 = part
        else:
            nisa.tensor_scalar(
                dst=part[0:P_MAX, 0:T, 0:n_tiles], data=part[0:P_MAX, 0:T, 0:n_tiles], op0=nl.left_shift, operand0=16
            )
            nisa.tensor_tensor(
                dst=km_i32[0:P_MAX, 0:T, 0:n_tiles],
                data1=km_i32[0:P_MAX, 0:T, 0:n_tiles],
                data2=part[0:P_MAX, 0:T, 0:n_tiles],
                op=nl.bitwise_or,
            )
    positions_km = nl.ndarray((P_MAX, T, n_tiles), dtype=nl.int32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=positions_km, src=km_i32[0:P_MAX, 0:T, 0:n_tiles])
    return positions_tm, positions_km


def indexer_topk_own_selection_packed(local: nl.NkiTensor, topk: int):
    """The selection of a CP rank whose key shard is exactly the keys it holds: its own share of the global top-k, with
    no gather of the shares and no merge. Exact for the attention, which resolves only the keys this rank holds, so the
    merged top-k and the share give it the same keys. Slots past the share hold -1.

    Args:
        local: [T * 16, 2 * (topk // 16)] uint32, this rank's packed tile with its share's global positions.
        topk: as in ``indexer_topk_merge_carry_index_packed``.

    Returns:
        ``(positions_tm, positions_km)`` as ``indexer_topk_merge_carry_index_packed``.
    """
    k_cols = topk // SNAKE_PARTITIONS
    P = local.shape[0]
    T = P // SNAKE_PARTITIONS
    n_tiles = topk // P_MAX
    kernel_assert(P == P_MAX and local.shape[1] == 2 * k_cols and k_cols % P_MAX == 0,
                  f"the own-share selection expects [{P_MAX}, {2 * k_cols}] uint32, got {tuple(local.shape)}")
    pos_i32 = nl.ndarray((P, k_cols), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos_i32[0:P, 0:k_cols], src=local[:, nl.ds(k_cols, k_cols)].view(nl.int32))
    return _selection_outputs(pos_i32, P, T, k_cols, n_tiles)


def _key_major_pe_transpose(pos_i32: nl.NkiTensor, P: int, T: int, n_tiles: int) -> nl.NkiTensor:
    """The key-major selection as one PE transpose: ``km[c, t * 16 + q] = pos[t * 16 + q, c]`` when k_cols == P_MAX.

    The transpose runs on fp32 VALUES, not bits: a position is -1 or below 2**24 (2**17 at 128k, 2**19 at 512k), so
    the int32 -> fp32 -> int32 round trip is exact, and the PE transpose is bit-accurate on NeuronCore-v3+. Never
    bitcast: -1 is a NaN pattern in fp32.
    """
    vec = nisa.engine.vector
    pos_f32 = nl.ndarray((P, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pos_f32[0:P, 0:P_MAX], src=pos_i32[0:P, 0:P_MAX], engine=vec)
    km_psum = nl.ndarray((P_MAX, P), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=km_psum[0:P_MAX, 0:P], data=pos_f32[0:P, 0:P_MAX], engine=nisa.engine.tensor)
    km_i32 = nl.ndarray((P_MAX, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=km_i32[0:P_MAX, 0:P], src=km_psum[0:P_MAX, 0:P], engine=vec)
    positions_km = nl.ndarray((P_MAX, T, n_tiles), dtype=nl.int32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=positions_km.reshape((P_MAX, T * n_tiles)), src=km_i32[0:P_MAX, 0:P])
    # GLM52K_PRIOR_INDEX_KM_SBUF: the attention core's prior index may read this tile instead of the HBM copy.
    from ..mla.prior_index import register_positions_km

    register_positions_km(positions_km, km_i32)
    return positions_km
