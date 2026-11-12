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

"""Kimi-K3 routed experts with a dynamic active-expert loop and per-expert token compaction (Trn3, LNC1).

The static EP x TP body runs every local expert on every gathered token: E_L experts x ceil(T / 128) token tiles of
weight passes. Only the routed (token, expert) pairs need work. This module runs each ACTIVE local expert once on a
compact tile of its own routed tokens, so a rank makes about one weight pass for each active expert. Off by default
(``ROUTED_DYNAMIC_DEFAULT``): the served body does not call it.

Two steps for each layer, on one rank, for the T gathered tokens (T <= 512, a multiple of 128, or a multiple of 4 below
128):

1. ``build_dispatch_map`` (the seam of a later GpSimd custom op, which keeps the outputs): from the top-16 global ids
   and gate weights of every token and the rank's ``global_of_local`` table, the dense FP32 local affinities [T, E_L]
   (a one-hot over the local experts: exact, each sum has at most one nonzero term). Then, per local expert, the
   ascending list of its tokens and their gates (``expert_tok`` / ``expert_gate``, one GpSimd ``nonzero_with_count``
   and one gather for each 8 experts), the active experts in ascending local id, and their counts, offsets and tiles.
2. ``routed_dynamic_partial``: zero the routed tokens' rows of the token-major FP32 partial, then for each work item
   (an active expert and one tile of at most ``block_tokens`` of its tokens): gather the item's MXFP8 latent columns,
   stream only that expert's weights, run the expert tile, scale by the gate in FP32, and add the rows into the partial
   with one scatter read-modify-write DMA (``dma_compute``: FP32 adds in the DMA engine, unique rows, out-of-bounds rows
   skipped). Tile 0 of local experts 0 .. ``static_slots`` - 1 runs unrolled (``head`` ``dense``: no loop barriers, so
   the matmuls run back to back, the gate/up of slot s + 1 issued before the down of slot s; the weight loads take
   trace-time expert ids, so they do not wait for the map). The other items (an expert with more than
   ``block_tokens`` tokens, and the experts past ``static_slots``) run in one dynamic loop, one trip for each. With
   ``head`` ``active`` the unrolled slots are the first active experts instead (for a batch with idle experts).

Lossless worst case: every (token, local expert) pair of the routing is computed. The tables hold every token of every
expert; the loop has no cap.

Determinism: for a fixed batch the map, the item order and every sum order are fixed, so two runs of a batch give the
same bits. A token's FP32 partial sums its experts in this order: first the unrolled head (ascending local id), then
the dynamic items (ascending slot, so ascending local id, then tile). This order differs from ascending local id
only for a token at position ``block_tokens`` or later in some expert's list. Across batches, such a token's sum order
can change: an FP32 reorder; a kernel test measures up to 2 BF16 ulp after the round.

With ``pack`` 1 (the default), work items hold one expert each (the I-slice 768 tile). The packed tile of the narrow
slices (several consecutive active experts on one stationary tile of their pairs) takes the CSR form of the map
(``csr``: ``compact_tok``, ``gate_of_pair``, ``tok_offset``), which the map builds with ``csr``; its items are the
packed items of ``pack`` K > 1 (``_packed_partial``).

The FP32 partial (``routed_partial_rows``) gives the BF16 token rows of the rank's routed partial (zero for the tokens
that no local expert takes), the input of the routed reduce.
"""

import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil
from ..moe.routed.moe_routed import _stack_skip_id
from ..tkg_moe import LATENT_SIZE, NUM_EXPERTS, SITU_BETA, SITU_LINEAR_BETA, TOP_K

ROUTED_DYNAMIC_FORMS = (False, True)
"""The values of the routed-dynamic form: False (default) keeps the served static body; True runs this module."""
ROUTED_DYNAMIC_DEFAULT = False
"""The default of the routed-dynamic form: off, so the served body and its bytes do not change."""
HEAD_FORMS = ("dense", "active")
"""The unrolled head of the loop (``routed_dynamic_partial`` ``head``): slot s is local expert s, or the s-th active
expert."""
MAX_TOKENS = 512
"""Most gathered tokens of one map: a GpSimd gather row holds at most 512 elements (one row of an expert's tokens)."""
BLOCK_TOKENS = 128
"""Most tokens of one expert tile: the stationary free size of an MX matmul."""
DISPATCH_MAP_KEYS = (
    "tokens",
    "local_experts",
    "block_tokens",
    "max_tiles",
    "active_count",
    "active_local_id",
    "tile_count",
    "tok_count",
    "tok_offset",
    "expert_tok",
    "expert_gate",
    "routed_rows",
    "compact_tok",
    "gate_of_pair",
    "scatter_tok",
)
"""The fields of a dispatch map (``build_dispatch_map``), by name, so the loop and a custom op agree by name."""

_PMAX = 128
_MX_TILE = 512
_N_D512 = LATENT_SIZE // _MX_TILE
_GPSIMD_STRIDE = 16  # nonzero_with_count reads and writes partitions 0, 16, ..., 112 (one for each GpSimd core)
_CSR_PIECE = 16  # index columns of one CSR gather (2048 rows)

CSR_GATHER_FORMS = ("staged", "rows")
CSR_GATHER_DEFAULT = "staged"
"""The CSR gathers of ``_build_csr`` (``build_dispatch_map`` ``csr_gather``). "staged" (default): one batched gather (a
[128, 16] index) for each 2048-row piece that can hold a pair, into one shared [128, 16] SBUF tile of the call, then a
store; the tile adds 1 DGE table slot a call (MEASURED). "rows": one gather (a [128, 1] index) for each 128 rows that
can hold a pair, straight into the HBM array; no SBUF destination, so no slot a call, but T min(16, E_L) / 64 gathers
(Pool MODEL: more time a layer at T = 128), so only for T <= ``CSR_ROWS_MAX_TOKENS``. The values are the same. A batched
gather cannot write HBM: the 2048 rows of its source are the partitions that an HBM destination must match (the DMA
operand alignment), while its index has 128 partitions that the destination must match too."""

CSR_ROWS_MAX_TOKENS = _PMAX
"""The most gathered tokens of a layer with the "rows" CSR gathers (fail closed above)."""

CSR_STAGED_MAX_LAYERS = 30
"""The most dynamic packed MoE layers of one program with the "staged" CSR gathers (1 DGE table slot each; the served
programs hold ~90 of the ~128 slots before a HWDGE DMA with a dynamic source offset slows down)."""


def is_packed_split(split: dict) -> bool:
    """Return True when the dynamic form of a split runs packed items and builds the CSR arrays (``default_pack`` of
    its I-slice and local experts above 1: I-slice 384, 192 or 96)."""
    return default_pack(split["local_intermediate"], split.get("local_experts", 1)) > 1


def csr_gather_of_layers(packed_layers: int) -> str:
    """Return the CSR form of a program with ``packed_layers`` dynamic packed MoE layers: "staged" up to
    ``CSR_STAGED_MAX_LAYERS`` (1 DGE slot a layer), else "rows" (no slot; T <= ``CSR_ROWS_MAX_TOKENS`` only).

    Example:
        >>> csr_gather_of_layers(0), csr_gather_of_layers(30), csr_gather_of_layers(31)
        ('staged', 'staged', 'rows')
    """
    kernel_assert(isinstance(packed_layers, int) and packed_layers >= 0, f"a layer count, got {packed_layers!r}")
    return CSR_GATHER_DEFAULT if packed_layers <= CSR_STAGED_MAX_LAYERS else "rows"


_ROWS_PER_TILE = _PMAX // _GPSIMD_STRIDE  # experts of one expert-major tile


def _token_tiles(tokens: int) -> tuple:
    """The (first token, rows) of each token tile of ``tokens`` gathered tokens: 128 rows, or all when fewer."""
    return tuple((start, min(_PMAX, tokens - start)) for start in range(0, tokens, _PMAX))


def _check_tokens(tokens: int):
    kernel_assert(
        (tokens % _PMAX == 0 and _PMAX <= tokens <= MAX_TOKENS) or (4 <= tokens < _PMAX and tokens % 4 == 0),
        f"the dynamic routed form takes T gathered tokens, a multiple of {_PMAX} up to {MAX_TOKENS} or a multiple of 4 "
        f"below {_PMAX}, got {tokens}",
    )


def _instruction(handle):
    """Return an ISA call result as an NkiInstruction, for ``depends_on``."""
    return handle if isinstance(handle, nisa.NkiInstruction) else nisa.NkiInstruction(handle)


def _wait_for(later, earlier):
    """Make ``later`` wait for ``earlier`` to complete (a runtime dependency); nothing when ``earlier`` is None."""
    if earlier is not None:
        _instruction(later).depends_on(_instruction(earlier), kind=nisa.dependency.runtime)


def _tile_of(source, start: int, rows: int, cols: int, dtype, name: str) -> nl.NkiTensor:
    """The rows ``[start, start + rows)`` of a token-major ``source``: an SBUF tile of a list of tiles, or one DMA of
    an HBM tensor, converted to ``dtype`` on VectorE when the dtypes differ."""
    if isinstance(source, (list, tuple)):
        tile = source[start // _PMAX]
        kernel_assert(
            tuple(tile.shape) == (rows, cols), f"{name}: expected an SBUF tile [{rows}, {cols}], got {tile.shape}"
        )
    else:
        tile = nl.ndarray((rows, cols), dtype=source.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=source[start : start + rows, :], dge_mode=nisa.dge_mode.none, name=f"{name}_load")
    if tile.dtype == dtype:
        return tile
    out = nl.ndarray((rows, cols), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=tile, engine=nisa.engine.vector, name=f"{name}_cast")
    return out


def global_of_local_hbm(rank_info: nl.NkiTensor, split: dict, name: str) -> nl.NkiTensor:
    """Return the INT32 [1, E_L] private-HBM table of the global ids of this rank's local experts, ascending local id.

    The experts of group G_r are [E_L G_r, E_L G_r + E_L) (the layout of every split); the group of rank g is
    ``(g >> group_shift) & (groups - 1)`` (``split`` from ``moe_routed_layout.routed_split``; ``group_shift``
    0 is the served ``group_minor`` order). ``rank_info``: INT32 HBM [1, >= 1], the world rank in column 0.
    """
    groups, local_experts = split["groups"], split["local_experts"]
    shift = split.get("group_shift", 0)
    kernel_assert(
        groups & (groups - 1) == 0 and groups * local_experts == NUM_EXPERTS,
        f"global_of_local_hbm: {groups} groups of {local_experts} experts must be a power-of-2 split of {NUM_EXPERTS}",
    )
    vector = nisa.engine.vector
    rank = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rank, src=rank_info[0:1, 0:1], dge_mode=nisa.dge_mode.none, name=f"{name}_rank_load")
    group = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=group,
        data=rank,
        op0=nl.right_shift,
        operand0=shift,
        op1=nl.bitwise_and,
        operand1=groups - 1,
        engine=vector,
        name=f"{name}_group",
    )
    base = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=base, data=group, op0=nl.multiply, operand0=float(local_experts), engine=vector, name=f"{name}_base"
    )
    ids_f = nl.ndarray((1, local_experts), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=ids_f, pattern=[[1, local_experts]], offset=0, channel_multiplier=0, name=f"{name}_iota")
    ids = nl.ndarray((1, local_experts), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ids, data=ids_f, op0=nl.add, operand0=base, engine=vector, name=f"{name}_ids")
    table = nl.ndarray((1, local_experts), dtype=nl.int32, buffer=nl.private_hbm, name=f"{name}_global_of_local")
    nisa.dma_copy(dst=table, src=ids, dge_mode=nisa.dge_mode.none, name=f"{name}_store")
    return table


def dense_local_affinities(
    top_ids, gate_weights, global_of_local: nl.NkiTensor, local_experts: int, tokens: int, name: str
) -> list:
    """Return the FP32 [rows, E_L] dense local affinities of each token tile from the top-16 routing.

    Element (t, e) is the gate weight of the pick of token t whose global id is ``global_of_local[e]``, else +0: a
    one-hot over the local experts (one compare, one multiply, one sum of the 16 picks, on VectorE). The 16 ids of a
    token are distinct, so each sum has at most one nonzero term: exact. ``top_ids`` (INT32) and ``gate_weights`` (FP32
    or BF16) are [T, 16] HBM tensors or lists of SBUF tiles [rows, 16]; ``global_of_local`` is INT32 HBM [1, E_L].
    """
    vector = nisa.engine.vector
    fp32 = nl.float32
    rows = min(_PMAX, tokens)
    kernel_assert(
        tuple(global_of_local.shape) == (1, local_experts) and global_of_local.dtype == nl.int32,
        f"global_of_local must be INT32 [1, {local_experts}] (HBM), got {global_of_local.dtype} "
        f"{global_of_local.shape}",
    )
    # The global id of each local expert on every token partition (a broadcast DMA), in FP32 (exact below 2^24).
    ids_local_i = nl.ndarray((rows, local_experts), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=ids_local_i,
        src=global_of_local.ap(pattern=[[0, rows], [1, local_experts]], offset=0),
        dge_mode=nisa.dge_mode.none,
        name=f"{name}_gol_load",
    )
    ids_local = nl.ndarray((rows, local_experts), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ids_local, src=ids_local_i, engine=vector, name=f"{name}_gol_f32")
    tiles = []
    for start, rows_u in _token_tiles(tokens):
        tag = f"{name}_t{start // _PMAX}"
        ids = _tile_of(top_ids, start, rows_u, TOP_K, fp32, f"{tag}_ids")
        weights = _tile_of(gate_weights, start, rows_u, TOP_K, fp32, f"{tag}_w")
        # [t, e, k]: (id_k == global id of e) * w_k, then the sum over the 16 picks k.
        picks = nl.ndarray((rows_u, local_experts, TOP_K), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=picks,
            data1=ids.ap(pattern=[[TOP_K, rows_u], [0, local_experts], [1, TOP_K]], offset=0),
            data2=ids_local.ap(pattern=[[local_experts, rows_u], [1, local_experts], [0, TOP_K]], offset=0),
            op=nl.equal,
            engine=vector,
            name=f"{tag}_onehot",
        )
        nisa.tensor_tensor(
            dst=picks,
            data1=picks,
            data2=weights.ap(pattern=[[TOP_K, rows_u], [0, local_experts], [1, TOP_K]], offset=0),
            op=nl.multiply,
            engine=vector,
            name=f"{tag}_weighted",
        )
        affinities = nl.ndarray((rows_u, local_experts), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=affinities, op=nl.add, data=picks, axis=(2,), name=f"{tag}_affinities")
        tiles.append(affinities)
    return tiles


def dispatch_scratch(tokens: int, local_experts: int, name: str, csr: bool = False) -> dict:
    """Allocate the private-HBM tables of ``build_dispatch_map`` (one set for the layers of a kernel that share T and
    E_L): ``expert_tok`` INT32 and ``expert_gate`` FP32 [E_L T, 1] (row e T + j: entry j of local expert e), and with
    ``csr`` the CSR pair arrays [R, 1]: P = T min(16, E_L) pairs, then padding, R = P + max(T, 128) rounded up to whole
    CSR gather pieces (2048 rows)."""
    _check_tokens(tokens)
    rows = local_experts * tokens
    scratch = {
        "expert_tok": nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.private_hbm, name=f"{name}_expert_tok"),
        "expert_gate": nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.private_hbm, name=f"{name}_expert_gate"),
    }
    if csr:
        # T or 128 rows of padding (the most that one tile read reaches past the pairs), then whole gather pieces.
        piece_rows = _PMAX * _CSR_PIECE
        pairs = div_ceil(tokens * min(TOP_K, local_experts) + max(tokens, _PMAX), piece_rows) * piece_rows
        scratch["compact_tok"] = nl.ndarray(
            (pairs, 1), dtype=nl.int32, buffer=nl.private_hbm, name=f"{name}_compact_tok"
        )
        scratch["gate_of_pair"] = nl.ndarray(
            (pairs, 1), dtype=nl.float32, buffer=nl.private_hbm, name=f"{name}_gate_of_pair"
        )
        # The expert offsets, read back on every partition by the CSR build.
        scratch["csr_offsets"] = nl.ndarray(
            (1, local_experts + 1), dtype=nl.float32, buffer=nl.private_hbm, name=f"{name}_csr_offsets"
        )
    return scratch


def build_dispatch_map(
    top_ids,
    gate_weights,
    global_of_local: nl.NkiTensor,
    local_experts: int,
    block_tokens: int = BLOCK_TOKENS,
    name: str = "k3rdyn_map",
    tokens: int = None,
    dense_affinities: list = None,
    csr: bool = False,
    scratch: dict = None,
    csr_gather: str = CSR_GATHER_DEFAULT,
) -> dict:
    """Build this rank's dispatch map of a layer: its active local experts and the compact token list of each.

    Not ``@nki.jit``: a kernel inlines it. A GpSimd custom op replaces the body later with the same outputs.

    Args:
        top_ids: INT32 [T, 16], the top-16 global expert ids (0 .. 895) of every gathered token: an HBM tensor or a
            list of SBUF tiles [rows, 16] (128 rows, or T when T < 128). Not used with ``dense_affinities``.
        gate_weights: FP32 (or BF16) [T, 16], the gate weight of each pick, in the same forms.
        global_of_local: INT32 HBM [1, E_L], the global id of each local expert, ascending local id
            (``global_of_local_hbm``). Not used with ``dense_affinities``.
        local_experts: E_L, at most 128.
        block_tokens: the tokens of one expert tile (a multiple of 4, at most 128): the loop's static tile size.
        name: a prefix of the instruction and tensor names, unique for each call.
        tokens: T; None takes ``top_ids.shape[0]`` (an HBM tensor).
        dense_affinities: the FP32 [rows, E_L] local affinities of each token tile in place of the top-16 routing
            (for example the 2-server ``_local_affinities``). A nonzero value (NaN too) routes the token to the expert.
        csr: also build the CSR pair arrays (``compact_tok``, ``gate_of_pair``, ``scatter_tok``) of the dispatch
            interface (the packed loop reads them): one batched gather for each array (``_build_csr``). ``"serial"``:
            the same arrays from one DMA for each expert in turn (``_build_csr_serial``, the reference of the twin).
        scratch: the HBM tables (``dispatch_scratch``); None allocates them for the call.
        csr_gather: the form of the CSR gathers (``CSR_GATHER_FORMS``): "staged" (default) or "rows" (T <= 128).

    Returns:
        A dict with the keys of ``DISPATCH_MAP_KEYS``. On partition 0 of SBUF: ``active_count`` INT32 [1, 1] (the
        dynamic loop bound), ``active_local_id`` INT32 [1, E_L] (ascending; E_L past the count), ``tok_count`` and
        ``tile_count`` INT32 [1, E_L] (0 past the count), ``tok_offset`` INT32 [1, E_L + 1] (the exclusive prefix of
        the counts; the total from the count on). In private HBM: ``expert_tok`` INT32 [E_L T, 1] (the ascending
        gathered-token ids of local expert e in rows e T .., then -1) and ``expert_gate`` FP32 [E_L T, 1] (their gates,
        then +0). ``expert_offset``: FP32 [1, E_L + 1] on partition 0, the exclusive prefix of the counts in local id
        order (the CSR offset of each local expert), the total last. ``routed_rows``: for each token tile, INT32 SBUF
        [rows, 1], the token's row when a local expert takes it, else -1. ``compact_tok`` / ``gate_of_pair`` /
        ``scatter_tok``: HBM [R, 1] (``dispatch_scratch``) (None without ``csr``). ``tokens``, ``local_experts``,
        ``block_tokens`` and ``max_tiles`` (T / block_tokens, the most tiles of one expert): ints.
    """
    fp32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
    vector = nisa.engine.vector
    if tokens is None:
        kernel_assert(top_ids is not None and not isinstance(top_ids, (list, tuple)), "give tokens with tile inputs")
        tokens = top_ids.shape[0]
    _check_tokens(tokens)
    e_l = local_experts
    rows = min(_PMAX, tokens)
    kernel_assert(1 <= e_l <= _PMAX, f"build_dispatch_map takes 1 to {_PMAX} local experts, got {e_l}")
    kernel_assert(
        isinstance(block_tokens, int) and 4 <= block_tokens <= BLOCK_TOKENS and block_tokens % 4 == 0,
        f"block_tokens must be a multiple of 4 in [4, {BLOCK_TOKENS}], got {block_tokens}",
    )
    block = min(block_tokens, tokens)
    kernel_assert(tokens % block == 0, f"block_tokens {block} must divide the {tokens} gathered tokens")
    max_tiles = tokens // block
    if scratch is None:
        scratch = dispatch_scratch(tokens, e_l, name, csr=csr)
    kernel_assert(
        tuple(scratch["expert_tok"].shape) == (e_l * tokens, 1) and (not csr or "compact_tok" in scratch),
        f"build_dispatch_map: the scratch holds expert_tok {tuple(scratch['expert_tok'].shape)}, expected "
        f"{(e_l * tokens, 1)}{' and the CSR arrays' if csr else ''}",
    )
    if dense_affinities is None:
        affinity_tiles = dense_local_affinities(top_ids, gate_weights, global_of_local, e_l, tokens, f"{name}_d")
    else:
        affinity_tiles = list(dense_affinities)
        kernel_assert(
            len(affinity_tiles) == len(_token_tiles(tokens))
            and all(tuple(tile.shape) == (rows, e_l) for tile in affinity_tiles),
            f"dense_affinities must be one FP32 [{rows}, {e_l}] tile for each token tile",
        )

    # Token-major: the 0/1 routed flags (VectorE: NaN counts as routed), the counts of each expert on partition 0 (a
    # 0/1 ones matmul, exact), and the routed rows of each token tile.
    ones_col = nl.ndarray((rows, 1), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(dst=ones_col, value=1.0, engine=vector, name=f"{name}_ones_col")
    count_psum = nl.ndarray((1, e_l), dtype=fp32, buffer=nl.psum)
    routed_rows = []
    for tile_idx, (start, rows_u) in enumerate(_token_tiles(tokens)):
        tag = f"{name}_t{tile_idx}"
        hits = nl.ndarray((rows_u, e_l), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=hits, data=affinity_tiles[tile_idx], op0=nl.not_equal, operand0=0.0, engine=vector, name=f"{tag}_hits"
        )
        nisa.nc_matmul(
            dst=count_psum,
            stationary=ones_col[0:rows_u, :],
            moving=hits,
            accumulate=tile_idx > 0,
            name=f"{tag}_count",
        )
        routed = nl.ndarray((rows_u, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=routed, op=nl.maximum, data=hits, axis=(1,), name=f"{tag}_routed")
        # (t + 1) * routed - 1: the token's row, or -1 (an out-of-bounds row that a DMA skips).
        token_plus_one = nl.ndarray((rows_u, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.iota(dst=token_plus_one, pattern=[[0, 1]], offset=start + 1, channel_multiplier=1, name=f"{tag}_token_ids")
        rows_f = nl.ndarray((rows_u, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=rows_f,
            data=token_plus_one,
            op0=nl.multiply,
            operand0=routed,
            op1=nl.add,
            operand1=-1.0,
            engine=vector,
            name=f"{tag}_rows_f",
        )
        rows_i = nl.ndarray((rows_u, 1), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rows_i, src=rows_f, engine=vector, name=f"{tag}_rows")
        routed_rows.append(rows_i)
    # Column E_L is 0: the count of the padding id E_L.
    counts = nl.ndarray((1, e_l + 1), dtype=fp32, buffer=nl.sbuf)
    nisa.memset(dst=counts, value=0.0, engine=vector, name=f"{name}_counts_zero")
    nisa.tensor_copy(dst=counts[0:1, 0:e_l], src=count_psum, engine=vector, name=f"{name}_counts")

    # Expert-major: local expert 8 g + j on partition 16 j of tile g (the partitions that nonzero_with_count reads),
    # its affinities of every token along the free axis. Bit-accurate PE transposes of a tile whose column 16 j holds
    # the expert's affinities.
    expert_tok, expert_gate = scratch["expert_tok"], scratch["expert_gate"]
    table_stores = []
    for group in range(div_ceil(e_l, _ROWS_PER_TILE)):
        first, n_rows = group * _ROWS_PER_TILE, min(_ROWS_PER_TILE, e_l - group * _ROWS_PER_TILE)
        tag = f"{name}_g{group}"
        by_expert = nl.ndarray((_PMAX, tokens), dtype=fp32, buffer=nl.sbuf)
        for tile_idx, (start, rows_u) in enumerate(_token_tiles(tokens)):
            spread = nl.ndarray((rows_u, _PMAX), dtype=fp32, buffer=nl.sbuf)
            nisa.memset(dst=spread, value=0.0, engine=vector, name=f"{tag}_t{tile_idx}_spread_zero")
            nisa.tensor_copy(
                dst=spread.ap(pattern=[[_PMAX, rows_u], [_GPSIMD_STRIDE, n_rows]], offset=0),
                src=affinity_tiles[tile_idx][:, first : first + n_rows],
                engine=vector,
                name=f"{tag}_t{tile_idx}_spread",
            )
            transposed = nl.ndarray((_PMAX, rows_u), dtype=fp32, buffer=nl.psum)
            nisa.nc_transpose(dst=transposed, data=spread, engine=nisa.engine.tensor, name=f"{tag}_t{tile_idx}_tp")
            nisa.tensor_copy(
                dst=by_expert[:, start : start + rows_u], src=transposed, engine=vector, name=f"{tag}_t{tile_idx}_copy"
            )
        # The ascending tokens of each expert, -1 padding, then the count (column T).
        lists = nl.ndarray((_PMAX, tokens + 1), dtype=i32, buffer=nl.sbuf)
        nisa.memset(dst=lists, value=-1, engine=vector, name=f"{tag}_lists_init")
        nisa.nonzero_with_count(dst=lists, src=by_expert, index_offset=0, padding_val=-1, name=f"{tag}_lists")
        # The gate of each list entry: a GpSimd gather along the expert's row (index 0 for the padding), then +0 where
        # the entry is padding.
        gather_index = nl.ndarray((_PMAX, tokens), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=gather_index,
            data=lists[:, 0:tokens],
            op0=nl.maximum,
            operand0=0,
            engine=vector,
            name=f"{tag}_gather_index",
        )
        gates = nl.ndarray((_PMAX, tokens), dtype=fp32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=gates, data=by_expert, indices=gather_index, name=f"{tag}_gates_gather")
        nisa.scalar_tensor_tensor(
            dst=gates,
            data=lists[:, 0:tokens],
            op0=nl.greater_equal,
            operand0=0.0,
            op1=nl.multiply,
            operand1=gates,
            name=f"{tag}_gates",
        )
        last_row = _GPSIMD_STRIDE * (n_rows - 1) + 1
        for table, source, label in ((expert_tok, lists, "tok"), (expert_gate, gates, "gate")):
            table_stores.append(
                nisa.dma_copy(
                    dst=table.ap(pattern=[[tokens, n_rows], [1, tokens]], offset=first * tokens),
                    src=source[0:last_row:_GPSIMD_STRIDE, 0:tokens],
                    dge_mode=nisa.dge_mode.none,
                    name=f"{tag}_{label}_store",
                )
            )

    # The active experts on partition 0: ascending local ids, padding E_L, then the count.
    active = nl.ndarray((1, e_l + 1), dtype=i32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=active, src=counts[0:1, 0:e_l], index_offset=0, padding_val=e_l, name=f"{name}_active")
    active_count = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=active_count, src=active[0:1, e_l : e_l + 1], engine=vector, name=f"{name}_active_count")
    active_local_id = nl.ndarray((1, e_l), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=active_local_id, src=active[0:1, 0:e_l], engine=vector, name=f"{name}_active_ids")
    slot_index = nl.ndarray((1, e_l), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=slot_index, src=active[0:1, 0:e_l], engine=vector, name=f"{name}_slot_index")
    # Per active slot: the count of its expert (0 for the padding id).
    tok_count_f = nl.ndarray((1, e_l), dtype=fp32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=tok_count_f, data=counts, indices=slot_index, name=f"{name}_tok_count_gather")
    tok_count = nl.ndarray((1, e_l), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_count, src=tok_count_f, engine=vector, name=f"{name}_tok_count")
    # The exclusive prefix of the counts in local id order (inactive experts add 0), the total in column E_L.
    ones_row = nl.ndarray((1, e_l), dtype=fp32, buffer=nl.sbuf)
    nisa.memset(dst=ones_row, value=1.0, engine=vector, name=f"{name}_ones_row")
    inclusive = nl.ndarray((1, e_l), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor_scan(
        dst=inclusive,
        data0=ones_row,
        data1=counts[0:1, 0:e_l],
        initial=0.0,
        op0=nl.multiply,
        op1=nl.add,
        name=f"{name}_prefix",
    )
    offsets = nl.ndarray((1, e_l + 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=offsets[0:1, 0:e_l],
        data1=inclusive,
        data2=counts[0:1, 0:e_l],
        op=nl.subtract,
        engine=vector,
        name=f"{name}_offsets",
    )
    nisa.tensor_copy(
        dst=offsets[0:1, e_l : e_l + 1], src=inclusive[0:1, e_l - 1 : e_l], engine=vector, name=f"{name}_total"
    )
    tok_offset_f = nl.ndarray((1, e_l), dtype=fp32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=tok_offset_f, data=offsets, indices=slot_index, name=f"{name}_tok_offset_gather")
    tok_offset = nl.ndarray((1, e_l + 1), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_offset[0:1, 0:e_l], src=tok_offset_f, engine=vector, name=f"{name}_tok_offset")
    nisa.tensor_copy(
        dst=tok_offset[0:1, e_l : e_l + 1], src=offsets[0:1, e_l : e_l + 1], engine=vector, name=f"{name}_tok_total"
    )
    # Tiles of each slot: the count of k < max_tiles with count > block k.
    tile_counts_f = nl.ndarray((1, e_l), dtype=fp32, buffer=nl.sbuf)
    if max_tiles == 1:
        nisa.tensor_scalar(
            dst=tile_counts_f, data=tok_count_f, op0=nl.greater, operand0=0.0, engine=vector, name=f"{name}_tiles"
        )
    else:
        thresholds = nl.ndarray((1, max_tiles), dtype=fp32, buffer=nl.sbuf)
        nisa.iota(dst=thresholds, pattern=[[block, max_tiles]], offset=0, channel_multiplier=0, name=f"{name}_thr")
        above = nl.ndarray((1, e_l, max_tiles), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=above,
            data1=tok_count_f.ap(pattern=[[e_l, 1], [1, e_l], [0, max_tiles]], offset=0),
            data2=thresholds.ap(pattern=[[max_tiles, 1], [0, e_l], [1, max_tiles]], offset=0),
            op=nl.greater,
            engine=vector,
            name=f"{name}_tiles_above",
        )
        nisa.tensor_reduce(dst=tile_counts_f, op=nl.add, data=above, axis=(2,), name=f"{name}_tiles")
    tile_count = nl.ndarray((1, e_l), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tile_count, src=tile_counts_f, engine=vector, name=f"{name}_tile_count")

    dmap = {
        "tokens": tokens,
        "local_experts": e_l,
        "block_tokens": block,
        "max_tiles": max_tiles,
        "active_count": active_count,
        "active_local_id": active_local_id,
        "tile_count": tile_count,
        "tok_count": tok_count,
        "tok_offset": tok_offset,
        "expert_tok": expert_tok,
        "expert_gate": expert_gate,
        "expert_offset": offsets,
        "routed_rows": routed_rows,
        "compact_tok": None,
        "gate_of_pair": None,
        "scatter_tok": None,
        "table_stores": table_stores,
    }
    if csr == "serial":
        _build_csr_serial(dmap, offsets, scratch, name)
    elif csr:
        _build_csr(dmap, offsets, scratch, name, csr_gather)
    return dmap


def _build_csr(dmap: dict, offsets: nl.NkiTensor, scratch: dict, name: str, gather: str = CSR_GATHER_DEFAULT):
    """Build each CSR pair array with indirect gathers from its padded table (``gather``: ``CSR_GATHER_FORMS``).

    Position x < total (the routed pairs) belongs to expert e(x) = the count of experts e with offset(e + 1) <= x, and
    holds entry x - offset(e(x)) of that expert's list: table row e(x) T + x - offset(e(x)). The index of x sits at
    [x % 128, x // 128] of an INT32 [128, C] tile (C = R / 128, R the rows of the array: a multiple of 2048). A
    position past the total takes index -1: out of bounds, so the gather skips it and it keeps the -1 / +0 fill.
    "staged": for each piece q that can hold a pair (2048 q < P, P = T min(16, E_L)), a gather with the [128, 16]
    piece q of the index reads its 2048 rows in column-major index order into the row-major order of the shared
    [128, 16] tile (a memset writes the fill before each gather), and a store moves the tile to rows [2048 q, 2048 q +
    2048) (16 contiguous rows a partition), so position x lands on row x; the rows past the last such piece take a
    static fill (no DGE slot). "rows": see ``_csr_gather_rows``. The values are those of ``_build_csr_serial``: the
    same entries, the same padding (integers below 2^24 in FP32)."""
    tokens, e_l = dmap["tokens"], dmap["local_experts"]
    kernel_assert(gather in CSR_GATHER_FORMS, f"csr_gather must be one of {CSR_GATHER_FORMS}, got {gather!r}")
    kernel_assert(
        gather != "rows" or tokens <= CSR_ROWS_MAX_TOKENS,
        f'the "rows" CSR gathers take at most {CSR_ROWS_MAX_TOKENS} gathered tokens (T min(16, E_L) / 64 gathers a '
        f'layer), got {tokens}: use "staged"',
    )
    vector = nisa.engine.vector
    fp32 = nl.float32
    n = scratch["compact_tok"].shape[0]
    piece_rows = _PMAX * _CSR_PIECE
    kernel_assert(n % piece_rows == 0, f"the CSR arrays hold whole gather pieces of {piece_rows} rows, got {n}")
    cols = n // _PMAX
    # The offsets (the total last) on every partition: a store and a broadcast load.
    store = nisa.dma_copy(
        dst=scratch["csr_offsets"], src=offsets, dge_mode=nisa.dge_mode.none, name=f"{name}_csr_offsets_store"
    )
    bounds = nl.ndarray((_PMAX, e_l + 1), dtype=fp32, buffer=nl.sbuf)
    load = nisa.dma_copy(
        dst=bounds,
        src=scratch["csr_offsets"].ap(pattern=[[0, _PMAX], [1, e_l + 1]], offset=0),
        dge_mode=nisa.dge_mode.none,
        name=f"{name}_csr_offsets_load",
    )
    _wait_for(load, store)
    position = nl.ndarray((_PMAX, cols), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=position, pattern=[[_PMAX, cols]], offset=0, channel_multiplier=1, name=f"{name}_csr_position")
    counts = nl.ndarray((_PMAX, e_l), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=counts,
        data1=bounds[:, 1 : e_l + 1],
        data2=bounds[:, 0:e_l],
        op=nl.subtract,
        engine=vector,
        name=f"{name}_csr_counts",
    )
    # before[., x, e] = (offset(e + 1) <= x): the experts whose pairs end at or before x.
    before = nl.ndarray((_PMAX, cols, e_l), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=before,
        data1=position.ap(pattern=[[cols, _PMAX], [1, cols], [0, e_l]], offset=0),
        data2=bounds.ap(pattern=[[e_l + 1, _PMAX], [0, cols], [1, e_l]], offset=1),
        op=nl.greater_equal,
        engine=vector,
        name=f"{name}_csr_before",
    )
    expert = nl.ndarray((_PMAX, cols), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=expert, op=nl.add, data=before, axis=(2,), name=f"{name}_csr_expert")
    nisa.tensor_tensor(
        dst=before,
        data1=before,
        data2=counts.ap(pattern=[[e_l, _PMAX], [0, cols], [1, e_l]], offset=0),
        op=nl.multiply,
        engine=vector,
        name=f"{name}_csr_before_counts",
    )
    start = nl.ndarray((_PMAX, cols), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=start, op=nl.add, data=before, axis=(2,), name=f"{name}_csr_start")
    routed = nl.ndarray((_PMAX, cols), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=routed,
        data=position,
        op0=nl.less,
        operand0=bounds[:, e_l : e_l + 1],
        engine=vector,
        name=f"{name}_csr_routed",
    )
    # index = (expert T + x - start + 1) routed - 1: the table row, or -1.
    row = nl.ndarray((_PMAX, cols), dtype=fp32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=row,
        data=expert,
        op0=nl.multiply,
        operand0=float(tokens),
        op1=nl.add,
        operand1=position,
        name=f"{name}_csr_row",
    )
    nisa.tensor_tensor(dst=row, data1=row, data2=start, op=nl.subtract, engine=vector, name=f"{name}_csr_row_less")
    nisa.scalar_tensor_tensor(
        dst=row, data=row, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=routed, name=f"{name}_csr_row_routed"
    )
    index = nl.ndarray((_PMAX, cols), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=index, data=row, op0=nl.add, operand0=-1.0, engine=vector, name=f"{name}_csr_index")
    if gather == "rows":
        _csr_gather_rows(dmap, scratch, index, cols, name)
        return
    # "staged": the pieces that can hold a pair (P = T min(16, E_L) pairs at most), one INT32 tile for every gather of
    # the call (the FP32 gates through a view of the same bytes); the rows past those pieces take a static fill.
    pieces = div_ceil(tokens * min(TOP_K, e_l), piece_rows)
    rest = cols - pieces * _CSR_PIECE
    staged = nl.ndarray((_PMAX, _CSR_PIECE), dtype=nl.int32, buffer=nl.sbuf)
    last_store = None
    for key, table_key, fill in (("compact_tok", "expert_tok", -1), ("gate_of_pair", "expert_gate", 0.0)):
        pairs, table = scratch[key], scratch[table_key]
        writes = []
        if rest:
            fill_tile = nl.ndarray((_PMAX, rest), dtype=pairs.dtype, buffer=nl.sbuf)
            nisa.memset(dst=fill_tile, value=fill, engine=vector, name=f"{name}_{key}_rest_fill_tile")
            writes.append(
                nisa.dma_copy(
                    dst=pairs.ap(pattern=[[rest, _PMAX], [1, rest]], offset=pieces * piece_rows),
                    src=fill_tile,
                    dge_mode=nisa.dge_mode.none,
                    name=f"{name}_{key}_rest_fill",
                )
            )
        for piece in range(pieces):
            tag = f"{name}_{key}_p{piece}"
            tile = staged if pairs.dtype == staged.dtype else staged.view(pairs.dtype)
            fill_op = nisa.memset(dst=tile, value=fill, engine=vector, name=f"{tag}_fill")
            _wait_for(fill_op, last_store)
            gather = nisa.dma_copy(
                dst=tile,
                src=table.ap(
                    pattern=[[1, piece_rows], [1, 1]],
                    offset=0,
                    vector_offset=index[:, piece * _CSR_PIECE : (piece + 1) * _CSR_PIECE],
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=nisa.oob_mode.skip,
                name=f"{tag}_gather",
            )
            _wait_for(gather, fill_op)
            for table_store in dmap["table_stores"]:
                _wait_for(gather, table_store)
            # Rows [2048 q, 2048 q + 2048) as [128, 16]: 16 contiguous rows a partition, the tile's row-major order.
            last_store = nisa.dma_copy(
                dst=pairs.ap(pattern=[[_CSR_PIECE, _PMAX], [1, _CSR_PIECE]], offset=piece * piece_rows),
                src=tile,
                dge_mode=nisa.dge_mode.none,
                name=f"{tag}_store",
            )
            writes.append(last_store)
        dmap[key] = pairs
        dmap.setdefault("csr_writes", []).extend(writes)
    dmap["scatter_tok"] = dmap["compact_tok"]


def _csr_gather_rows(dmap: dict, scratch: dict, index: nl.NkiTensor, cols: int, name: str):
    """The "rows" form of ``_build_csr``: -1 / +0 in every row of each pair array (a static DMA), then one gather for
    each column c of the index that can hold a routed pair (128 c < T min(16, E_L)): the [128, 1] index column c into
    rows [128 c, 128 c + 128) of the HBM array (position x = 128 c + p on partition p). A position past the total takes
    index -1 and keeps the fill."""
    tokens, e_l = dmap["tokens"], dmap["local_experts"]
    routed_cols = div_ceil(tokens * min(TOP_K, e_l), _PMAX)
    kernel_assert(routed_cols <= cols, f"the CSR index holds {cols} columns, the pairs need {routed_cols}")
    for key, table_key, fill in (("compact_tok", "expert_tok", -1), ("gate_of_pair", "expert_gate", 0.0)):
        pairs, table = scratch[key], scratch[table_key]
        fill_tile = nl.ndarray((_PMAX, cols), dtype=pairs.dtype, buffer=nl.sbuf)
        nisa.memset(dst=fill_tile, value=fill, engine=nisa.engine.vector, name=f"{name}_{key}_fill_tile")
        prefill = nisa.dma_copy(
            dst=pairs.ap(pattern=[[cols, _PMAX], [1, cols]], offset=0),
            src=fill_tile,
            dge_mode=nisa.dge_mode.none,
            name=f"{name}_{key}_fill",
        )
        writes = []
        for col in range(routed_cols):
            first = col * _PMAX
            gather = nisa.dma_copy(
                dst=pairs[first : first + _PMAX, :],
                src=table.ap(
                    pattern=[[1, _PMAX], [1, 1]], offset=0, vector_offset=index[:, col : col + 1], indirect_dim=0
                ),
                dge_mode=nisa.dge_mode.swdge,
                oob_mode=nisa.oob_mode.skip,
                name=f"{name}_{key}_c{col}_gather",
            )
            _wait_for(gather, prefill)
            for table_store in dmap["table_stores"]:
                _wait_for(gather, table_store)
            writes.append(gather)
        dmap[key] = pairs
        dmap.setdefault("csr_writes", []).extend(writes)
    dmap["scatter_tok"] = dmap["compact_tok"]


def _build_csr_serial(dmap: dict, offsets: nl.NkiTensor, scratch: dict, name: str):
    """Fill the CSR pair arrays from the padded tables: -1 / +0 everywhere, then each expert's padded row of T entries
    at its offset, in ascending local id, so each row overwrites the padding of the one before. 2 E_L DMAs in turn:
    the reference form of the twin (``build_dispatch_map`` ``csr`` ``"serial"``)."""
    tokens, e_l = dmap["tokens"], dmap["local_experts"]
    vector = nisa.engine.vector
    offsets_i = nl.ndarray((1, e_l), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=offsets_i, src=offsets[0:1, 0:e_l], engine=vector, name=f"{name}_csr_offsets")
    for key, table_key, fill in (("compact_tok", "expert_tok", -1), ("gate_of_pair", "expert_gate", 0.0)):
        pairs = scratch[key]
        n = pairs.shape[0]
        cols = div_ceil(n, _PMAX)
        fill_tile = nl.ndarray((_PMAX, cols), dtype=pairs.dtype, buffer=nl.sbuf)
        nisa.memset(dst=fill_tile, value=fill, engine=vector, name=f"{name}_{key}_fill_tile")
        previous = None
        for part, (first, count) in enumerate(((0, (n // cols) * cols), ((n // cols) * cols, n % cols))):
            if count == 0:
                continue
            p_rows = count // cols if part == 0 else 1
            p_cols = cols if part == 0 else count
            previous = nisa.dma_copy(
                dst=pairs.ap(pattern=[[p_cols, p_rows], [1, p_cols]], offset=first),
                src=fill_tile[0:p_rows, 0:p_cols],
                dge_mode=nisa.dge_mode.none,
                name=f"{name}_{key}_fill{part}",
            )
        for expert in range(e_l):
            write = nisa.dma_copy(
                dst=pairs.ap(
                    pattern=[[tokens, 1], [1, tokens]],
                    offset=0,
                    scalar_offset=offsets_i[0:1, expert : expert + 1],
                    indirect_dim=0,
                ),
                src=scratch[table_key].ap(pattern=[[tokens, 1], [1, tokens]], offset=expert * tokens),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{name}_{key}_e{expert}",
            )
            for store in dmap["table_stores"]:
                _wait_for(write, store)
            _wait_for(write, previous)
            previous = write
        dmap[key] = pairs
        dmap.setdefault("csr_writes", []).append(previous)
    dmap["scatter_tok"] = dmap["compact_tok"]


def tile_provider(i_slice: int) -> dict:
    """Return the expert-tile functions of an I-slice (``tkg_moe_routed_tiles``: 768, 384, 192, 96): the
    gate/up + SiTU stage, the activation stage (transposes + MXFP8) and the down stage. ``skips_oob``
    (``LOADS_SKIP_OOB``): a weight load of an expert id past the stacks skips its DMAs, so the buffers keep the expert
    loaded before (False: the loop loads expert 0 for a padding slot, which adds nothing to the partial)."""
    from . import tkg_moe_routed_tiles as tiles

    kernel_assert(i_slice in tiles.TILE_WIDTHS, f"the dynamic routed form takes an I-slice in {tiles.TILE_WIDTHS}")
    return {
        "alloc": tiles.alloc_tile_weights,
        "load": tiles.load_tile_weights,
        "gate_up": tiles.expert_tile_gate_up,
        "activation": tiles.expert_tile_activation,
        "down": tiles.expert_tile_down,
        "skips_oob": bool(getattr(tiles, "LOADS_SKIP_OOB", False)),
    }


# --- The dynamic loop -------------------------------------------------------------------------------------------------


def _slot_rows(dmap: dict, weights: tuple, tiles: dict, name: str) -> dict:
    """Per active slot on partition 0: the load id of its expert (INT32 [1, E_L]; past the count the skip id past the
    stacks when the tile loads skip, else expert 0) and the first row of its entries in the padded tables
    (``active_local_id`` T; E_L T, past the tables, for the padding)."""
    tokens, e_l = dmap["tokens"], dmap["local_experts"]
    vector = nisa.engine.vector
    ids_f = nl.ndarray((1, e_l), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ids_f, src=dmap["active_local_id"], engine=vector, name=f"{name}_ids_f")
    padding = nl.ndarray((1, e_l), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=padding, data=ids_f, op0=nl.equal, operand0=float(e_l), engine=vector, name=f"{name}_pad")
    if tiles["skips_oob"]:
        # id + padding (skip - E_L): the padding id moves past the end of every stack (``_stack_skip_id``).
        skip = _stack_skip_id(weights)
        move = float(skip - e_l)
    else:
        move = float(-e_l)  # the padding id becomes expert 0: a real load, no compute reaches the partial
    load_f = nl.ndarray((1, e_l), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=load_f, data=padding, op0=nl.multiply, operand0=move, op1=nl.add, operand1=ids_f, name=f"{name}_load_f"
    )
    load_ids = nl.ndarray((1, e_l), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=load_ids, src=load_f, engine=vector, name=f"{name}_load_ids")
    first_f = nl.ndarray((1, e_l), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=first_f, data=ids_f, op0=nl.multiply, operand0=float(tokens), engine=vector, name=f"{name}_first_f"
    )
    return {"load_f": load_f, "load_ids": load_ids, "first_f": first_f}


class _ItemTiles:
    """The SBUF tiles of one in-flight work item: the token row of its slots (broadcast to every partition, for the
    latent gathers), its token and gate columns (one slot a partition, for the scatter and the scale), its MXFP8
    latent tile and its FP32 accumulator."""

    def __init__(self, block: int):
        self.block = block
        self.token_row = nl.ndarray((_PMAX, block), dtype=nl.int32, buffer=nl.sbuf)
        self.gather_index = nl.ndarray((_PMAX, block), dtype=nl.uint32, buffer=nl.sbuf)
        self.pad_term = nl.ndarray((_PMAX, block), dtype=nl.float32, buffer=nl.sbuf)
        self.token_col = nl.ndarray((block, 1), dtype=nl.int32, buffer=nl.sbuf)
        self.gate_col = nl.ndarray((block, 1), dtype=nl.float32, buffer=nl.sbuf)
        self.x_mx = nl.ndarray((_PMAX, _N_D512, block), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        self.x_scale = nl.ndarray((_PMAX, _N_D512, block), dtype=nl.uint8, buffer=nl.sbuf)
        self.acc = nl.ndarray((block, LATENT_SIZE), dtype=nl.float32, buffer=nl.sbuf)


def _item_inputs(
    item: _ItemTiles,
    dmap: dict,
    first_row,
    x_mx: nl.NkiTensor,
    x_scale: nl.NkiTensor,
    pad_column: int,
    name: str,
    in_loop: bool = False,
):
    """Load an item's slots from the padded tables at row ``first_row``: a trace-time int (static DMAs), or an INT32
    [1, 1] SBUF tensor read at runtime, -1 / +0 when the row is past the tables (a padding slot: the DMAs skip and keep
    the fill). Then gather its MXFP8 latent columns (a padding entry reads ``pad_column``). Outside a loop body
    (``in_loop`` False) the loads wait for the map's table stores."""
    block = item.block
    vector = nisa.engine.vector
    dynamic = not isinstance(first_row, int)
    if dynamic:
        nisa.memset(dst=item.token_row, value=-1, engine=vector, name=f"{name}_row_fill")
        nisa.memset(dst=item.token_col, value=-1, engine=vector, name=f"{name}_col_fill")
        nisa.memset(dst=item.gate_col, value=0.0, engine=vector, name=f"{name}_gate_fill")
    for dst, table, pattern, label in (
        (item.token_row, dmap["expert_tok"], [[0, _PMAX], [1, block]], "row"),
        (item.token_col, dmap["expert_tok"], [[1, block], [1, 1]], "col"),
        (item.gate_col, dmap["expert_gate"], [[1, block], [1, 1]], "gate"),
    ):
        if dynamic:
            src = table.ap(pattern=pattern, offset=0, scalar_offset=first_row, indirect_dim=0)
            options = {"oob_mode": nisa.oob_mode.skip, "dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}
        else:
            src, options = table.ap(pattern=pattern, offset=first_row), {"dge_mode": nisa.dge_mode.none}
        load = nisa.dma_copy(dst=dst, src=src, name=f"{name}_{label}_load", **options)
        for store in () if in_loop else dmap["table_stores"]:
            _wait_for(load, store)
    # index = token, or pad_column for an entry of -1: token + (token < 0) (pad_column + 1).
    nisa.tensor_scalar(
        dst=item.pad_term,
        data=item.token_row,
        op0=nl.less,
        operand0=0,
        op1=nl.multiply,
        operand1=float(pad_column + 1),
        engine=vector,
        name=f"{name}_pad_term",
    )
    nisa.tensor_tensor(
        dst=item.gather_index, data1=item.pad_term, data2=item.token_row, op=nl.add, engine=vector, name=f"{name}_index"
    )
    for tile_h in range(_N_D512):
        nisa.nc_n_gather(
            dst=item.x_mx.view(nl.uint32)[:, tile_h, :],
            data=x_mx.view(nl.uint32)[:, tile_h, :],
            indices=item.gather_index,
            name=f"{name}_codes_h{tile_h}",
        )
        nisa.nc_n_gather(
            dst=item.x_scale[:, tile_h, :],
            data=x_scale[:, tile_h, :],
            indices=item.gather_index,
            name=f"{name}_scales_h{tile_h}",
        )


def _scatter_add(item: _ItemTiles, partial: nl.NkiTensor, name: str):
    """partial[token] += acc[slot] for every live slot: one scatter read-modify-write DMA (FP32 adds in the DMA
    engine; the slots' tokens are distinct; a slot of -1 is out of bounds and skips)."""
    dst = partial.ap(
        pattern=[[LATENT_SIZE, item.block], [1, LATENT_SIZE]], offset=0, vector_offset=item.token_col, indirect_dim=0
    )
    return nisa.dma_compute(
        dst=dst,
        srcs=[dst, item.acc[0 : item.block, :]],
        reduce_op=nl.add,
        scales=[1.0, 1.0],
        oob_mode=nisa.oob_mode.skip,
        name=f"{name}_scatter_add",
    )


def zero_routed_rows(partial: nl.NkiTensor, dmap: dict, name: str) -> list:
    """Zero the rows of the routed tokens (``routed_rows``) of the FP32 partial: one scatter of a zero tile a token
    tile; the other rows keep their bytes (``routed_partial_rows`` reads 0 for them). Returns the DMAs."""
    rows = min(_PMAX, dmap["tokens"])
    zero = nl.ndarray((rows, LATENT_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero, value=0.0, engine=nisa.engine.gpsimd, name=f"{name}_zero_tile")
    stores = []
    for tile_idx, rows_i in enumerate(dmap["routed_rows"]):
        rows_u = rows_i.shape[0]
        stores.append(
            nisa.dma_copy(
                dst=partial.ap(
                    pattern=[[LATENT_SIZE, rows_u], [1, LATENT_SIZE]], offset=0, vector_offset=rows_i, indirect_dim=0
                ),
                src=zero[0:rows_u, :],
                oob_mode=nisa.oob_mode.skip,
                dge_mode=nisa.dge_mode.swdge,
                name=f"{name}_zero_t{tile_idx}",
            )
        )
    return stores


def routed_dynamic_partial(
    dmap: dict,
    x_mx: nl.NkiTensor,
    x_scale: nl.NkiTensor,
    expert_weights: tuple,
    partial: nl.NkiTensor,
    i_slice: int,
    name: str,
    static_slots: int = None,
    weight_buffers: int = 2,
    tiles: dict = None,
    after=None,
    beta: float = SITU_BETA,
    linear_beta: float = SITU_LINEAR_BETA,
    head: str = "dense",
    pack: int = 1,
    spill_slots: int = 0,
) -> dict:
    """Add this rank's routed experts of the layer into the token-major FP32 ``partial``, active experts only.

    Not ``@nki.jit``: a kernel inlines it.

    Args:
        dmap: the layer's dispatch map (``build_dispatch_map``).
        x_mx, x_scale: the MXFP8 latent of every gathered token, SBUF [128, 7, C] float8_e4m3fn_x4 / uint8 (the
            ``quantized_latent`` layout; token t in column t). C = T + 1 with zero codes and scales in column T, or
            C = T (T = 512: a GpSimd gather row holds 512 elements), when a padding slot reads token 0's finite codes.
            A padding slot's gate is 0 and its row never leaves the tile (its scatter row is out of bounds).
        expert_weights: the rank's HBM expert stacks (gate/up weights, gate/up scales, down weights, down scales),
            [E_L, ...] in the layout of ``tiles``.
        partial: FP32 [T, 3584] HBM. The routed tokens' rows are zeroed, then each (token, expert) pair adds
            gate * expert output (FP32) into its token's row. Other rows are not written.
        i_slice: the intermediate slice of a rank (3072 / routed_tp).
        name: a prefix of the instruction names, unique for each call.
        static_slots: the slots whose tile 0 runs unrolled before the dynamic loop (default E_L). An unrolled slot
            with no tokens adds nothing but takes its PE time: fewer static slots suit a batch with idle experts.
        head: ``HEAD_FORMS``. ``dense`` (default): unrolled slot s is local expert s, its weights and table rows at
            trace-time offsets, so the head's weight stream does not wait for the map (an idle expert reads an empty
            list). ``active``: unrolled slot s is the s-th active expert (``active_local_id``); a slot past the count
            loads no weights (or expert 0, when the tile loads cannot skip). The dynamic loop runs the other items.
        spill_slots: X (even; default 0) items of the dynamic list that run unrolled after the head, at runtime
            offsets and with no loop barrier; the loop runs the items from X on. A spill slot past the item count adds
            nothing but takes one PE pass (and, when the tile loads cannot skip, a load of expert 0).
        pack: 1 (default): one expert an item. K > 1 (an I-slice of ``PACKED_WIDTHS``, K dividing E_L): the packed
            form (``_packed_partial``; the map needs ``csr``).
        weight_buffers: SBUF weight sets of the unrolled slots (the loads of slot s + 2 wait for slot s with 2); the
            dynamic loop takes 2 of them.
        tiles: the tile functions (``tile_provider``); None takes ``tile_provider(i_slice)``.
        after: a TensorE instruction that the first matmul follows, or None.
        beta, linear_beta: the SiTU factors.

    Returns:
        {"zero": the zero DMAs, "scatters": the scatter DMAs of the unrolled slots, "last": the last TensorE
        instruction outside the loop, "loop": True when the dynamic loop was traced}.
    """
    tokens, e_l, block = dmap["tokens"], dmap["local_experts"], dmap["block_tokens"]
    max_tiles = dmap["max_tiles"]
    if pack > 1:
        return _packed_partial(
            dmap,
            x_mx,
            x_scale,
            expert_weights,
            partial,
            i_slice,
            name,
            pack,
            weight_buffers,
            tiles,
            after,
            beta,
            linear_beta,
        )
    tiles = tile_provider(i_slice) if tiles is None else tiles
    static_slots = e_l if static_slots is None else static_slots
    kernel_assert(0 <= static_slots <= e_l, f"static_slots must be in [0, {e_l}], got {static_slots}")
    kernel_assert(head in HEAD_FORMS, f"head must be one of {HEAD_FORMS}, got {head!r}")
    dense = head == HEAD_FORMS[0]
    kernel_assert(weight_buffers >= 1, f"weight_buffers must be at least 1, got {weight_buffers}")
    kernel_assert(
        tuple(partial.shape) == (tokens, LATENT_SIZE) and partial.dtype == nl.float32,
        f"partial must be FP32 [{tokens}, {LATENT_SIZE}], got {partial.dtype} {tuple(partial.shape)}",
    )
    kernel_assert(
        len(expert_weights) == 4 and all(stack.shape[0] == e_l for stack in expert_weights),
        f"expert_weights must be the 4 stacks of {e_l} local experts",
    )
    columns = x_mx.shape[2]
    kernel_assert(
        tuple(x_mx.shape[:2]) == (_PMAX, _N_D512)
        and tuple(x_scale.shape) == tuple(x_mx.shape)
        and columns in (tokens, tokens + 1),
        f"x_mx / x_scale must be [128, {_N_D512}, T or T + 1] for T = {tokens}, got {tuple(x_mx.shape)}",
    )
    pad_column = tokens if columns == tokens + 1 else 0
    rows = _slot_rows(dmap, expert_weights, tiles, f"{name}_slots")
    vector = nisa.engine.vector

    zero = zero_routed_rows(partial, dmap, f"{name}_z")
    pending = list(zero)  # the first scatter waits for every zero DMA, each later one for the scatter before it

    kernel_assert(spill_slots >= 0 and spill_slots % 2 == 0, f"spill_slots must be an even count, got {spill_slots}")
    tail_possible = max_tiles > 1 or static_slots < e_l
    n_sets = max(2 if tail_possible else 1, min(weight_buffers, max(static_slots, 1)))
    weight_sets = [tiles["alloc"](i_slice, f"{name}_w{idx}") for idx in range(n_sets)]
    rings = [_ItemTiles(block) for _ in range(2)]

    # Three stages an item, so TensorE runs the down of one item while the SiTU of the next runs on ScalarE / VectorE:
    # gate_up (gate/up matmuls + SiTU), activation (transposes + MXFP8), finish (down matmuls + scatter).
    def gate_up(item, weights, tag, after_pe):
        return tiles["gate_up"](
            item.x_mx,
            item.x_scale,
            weights["gate_up_w"],
            weights["gate_up_s"],
            block,
            i_slice,
            f"{tag}_gu",
            after=after_pe,
            beta=beta,
            linear_beta=linear_beta,
        )

    def activation(staged, tag, after_pe):
        return tiles["activation"](staged[0], block, i_slice, f"{tag}_act", after=after_pe)

    def start_item(item, weights, load_id, first_row, tag, after_pe):
        _item_inputs(item, dmap, first_row, x_mx, x_scale, pad_column, tag)
        tiles["load"](weights, *expert_weights, load_id, i_slice, f"{tag}_load")
        return gate_up(item, weights, tag, after_pe)

    def finish_item(item, weights, act_tiles, tag, after_pe):
        last = tiles["down"](
            item.acc,
            act_tiles,
            weights["down_w"],
            weights["down_s"],
            block,
            i_slice,
            f"{tag}_down",
            after=after_pe,
            affinity=item.gate_col,
            accumulate=False,
        )
        return last, _scatter_add(item, partial, tag)

    # The static head: tile 0 of slots 0 .. static_slots - 1, unrolled. The front of slot s + 1 goes before the down of
    # slot s on TensorE, so the PE has work while slot s's activation is quantized.
    head_rows, head_loads = [], []
    for slot in range(static_slots):
        if dense:
            head_rows.append(slot * tokens)
            head_loads.append(slot)
            continue
        first_row = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=first_row, src=rows["first_f"][0:1, slot : slot + 1], engine=vector, name=f"{name}_s{slot}_first"
        )
        head_rows.append(first_row)
        head_loads.append(rows["load_ids"][0:1, slot : slot + 1])
    scatters, last_pe, fronts = [], after, {}

    def head_gate_up(slot, after_pe):
        return start_item(
            rings[slot % 2],
            weight_sets[slot % len(weight_sets)],
            head_loads[slot],
            head_rows[slot],
            f"{name}_s{slot}",
            after_pe,
        )

    # TensorE order: gu(0), act(0), then for each slot s: gu(s + 1), down(s), act(s + 1).
    acts = {}
    if static_slots:
        staged = head_gate_up(0, last_pe)
        acts[0] = activation(staged, f"{name}_s0", staged[1])
        last_pe = acts[0][1]
    for slot in range(static_slots):
        down_after = last_pe
        if slot + 1 < static_slots:
            fronts[slot + 1] = head_gate_up(slot + 1, last_pe)
            down_after = fronts[slot + 1][1]
        last_pe, scatter = finish_item(
            rings[slot % 2], weight_sets[slot % len(weight_sets)], acts.pop(slot)[0], f"{name}_s{slot}", down_after
        )
        if slot + 1 < static_slots:
            acts[slot + 1] = activation(fronts.pop(slot + 1), f"{name}_s{slot + 1}", last_pe)
            last_pe = acts[slot + 1][1]
        for earlier in pending:
            _wait_for(scatter, earlier)
        pending = [scatter]
        scatters.append(scatter)

    # The dynamic tail: every other (slot, tile) item, in ascending slot then tile, two items a trip. Item i takes
    # weight set and ring (static_slots + i) % 2 (X for even i, Y for odd i). Each trip issues the inputs and weights of
    # item b before the matmuls of item a, and those of item a + 2 before the matmuls of item b, so every load runs
    # under a matmul and completes inside its trip (the loop barrier). TensorE runs gu(a), gu(b), act(a), down(a),
    # act(b), down(b). An odd item count ends with a one-item trip. A set keeps its expert: an item whose expert is the
    # one its set holds skips its load. Both need tile loads that skip an id past the stacks (LOADS_SKIP_OOB); without
    # them a padding item would load expert 0, so item a's weights load at the start of its own trip (no load ahead).
    if tail_possible:
        sets = (weight_sets[static_slots % len(weight_sets)], weight_sets[(static_slots + 1) % len(weight_sets)])
        tail_rings = (rings[static_slots % 2], rings[(static_slots + 1) % 2])
        previous = (-1, -1)
        if dense:
            # The last head slot of each set (dense: slot s holds expert s).
            previous = tuple(
                max((s for s in range(static_slots) if weight_sets[s % len(weight_sets)] is held), default=-1)
                for held in sets
            )
        item_first, item_load, pairs, odd = _tail_items(
            dmap, rows, static_slots, dense, expert_weights, tiles, previous, f"{name}_tail", spill=spill_slots
        )
        width = item_first.shape[1]
        shifted = nl.ndarray((1, width - 2), dtype=nl.int32, buffer=nl.sbuf)

        ahead = tiles["skips_oob"]

        def stage(index, column, tag, in_loop, weights=True):
            """The inputs (and the weights) of the item in item-row column ``column``, into ring and set ``index``."""
            _item_inputs(
                tail_rings[index],
                dmap,
                item_first[0:1, column : column + 1],
                x_mx,
                x_scale,
                pad_column,
                tag,
                in_loop=in_loop,
            )
            if weights:
                load_id = item_load[0:1, column : column + 1]
                tiles["load"](sets[index], *expert_weights, load_id, i_slice, f"{tag}_load")

        def compute(index, tag, after_pe):
            """One item in ring and set ``index``: gate/up, activation, down, in that order."""
            staged = gate_up(tail_rings[index], sets[index], tag, after_pe)
            act_tiles, act_last = activation(staged, tag, staged[1])
            return finish_item(tail_rings[index], sets[index], act_tiles, tag, after_pe=act_last)

        # The spill slots: items 0 .. X - 1 unrolled, item j + 1 staged before the matmuls of item j (sets by parity, as
        # in the loop, so the load ids keep their reuse).
        for item in range(spill_slots):
            tag = f"{name}_spill{item}"
            if item == 0:
                stage(0, 0, f"{tag}_stage", in_loop=False)
            if item + 1 < spill_slots:
                stage((item + 1) % 2, item + 1, f"{name}_spill{item + 1}_stage", in_loop=False)
            last_pe, scatter = compute(item % 2, tag, last_pe)
            for earlier in pending:
                _wait_for(scatter, earlier)
            pending = [scatter]
            scatters.append(scatter)
        if spill_slots:
            # The loop starts at item X: move the item rows by X.
            moved = nl.ndarray((1, width - spill_slots), dtype=nl.int32, buffer=nl.sbuf)
            for label, row in (("first", item_first), ("load", item_load)):
                nisa.tensor_copy(
                    dst=moved, src=row[0:1, spill_slots:width], engine=vector, name=f"{name}_spill_{label}_next"
                )
                nisa.tensor_copy(
                    dst=row[0:1, 0 : width - spill_slots], src=moved, engine=vector, name=f"{name}_spill_{label}_move"
                )
        # Item 0 of the loop before it, under the last unrolled slot.
        stage(0, 0, f"{name}_tail_first", in_loop=False, weights=ahead)

        def tail_body(_iteration):
            if not ahead:
                tiles["load"](sets[0], *expert_weights, item_load[0:1, 0:1], i_slice, f"{name}_tail_a_load")
            stage(1, 1, f"{name}_tail_b", in_loop=True)
            # TensorE order: gu(a), gu(b), act(a), down(a), act(b), down(b).
            staged_a = gate_up(tail_rings[0], sets[0], f"{name}_tail_a", None)
            staged_b = gate_up(tail_rings[1], sets[1], f"{name}_tail_b", staged_a[1])
            act_a, act_a_last = activation(staged_a, f"{name}_tail_a", staged_b[1])
            last_a, scatter_a = finish_item(tail_rings[0], sets[0], act_a, f"{name}_tail_a", act_a_last)
            stage(0, 2, f"{name}_tail_next", in_loop=True, weights=ahead)
            act_b, act_b_last = activation(staged_b, f"{name}_tail_b", last_a)
            _, scatter_b = finish_item(tail_rings[1], sets[1], act_b, f"{name}_tail_b", act_b_last)
            # Items a and b may share token rows: their read-modify-writes run in order.
            _wait_for(scatter_b, scatter_a)
            # Advance both item rows by two (padding items enter at the end).
            for label, row in (("first", item_first), ("load", item_load)):
                nisa.tensor_copy(dst=shifted, src=row[0:1, 2:width], engine=vector, name=f"{name}_tail_{label}_next")
                nisa.tensor_copy(
                    dst=row[0:1, 0 : width - 2], src=shifted, engine=vector, name=f"{name}_tail_{label}_advance"
                )

        nl.fori_loop(0, nisa.scalar_load(pairs), tail_body)

        def odd_body(_iteration):
            # The last item of an odd count, in ring and set 0 (staged by the last pair trip, or before the loops).
            if not ahead:
                tiles["load"](sets[0], *expert_weights, item_load[0:1, 0:1], i_slice, f"{name}_tail_odd_load")
            compute(0, f"{name}_tail_odd", None)

        # A one-item trip for an odd count, so no padding item takes a PE pass.
        nl.fori_loop(0, nisa.scalar_load(odd), odd_body)
    return {"zero": zero, "scatters": scatters, "last": last_pe, "loop": tail_possible}


def _tail_items(
    dmap: dict,
    rows: dict,
    static_slots: int,
    dense: bool,
    weights: tuple,
    tiles: dict,
    previous: tuple,
    name: str,
    spill: int = 0,
) -> tuple:
    """The items of the dynamic loop, in ascending (slot, tile), on partition 0, with 3 + ``spill`` padding items after
    them (rows past the tables): the first row of each item's entries in the padded tables (the first row of its slot +
    block k) and its weight load id. The head's items are out: tile 0 of the slots below ``static_slots``, or with
    ``dense`` tile 0 of the local experts below it. Item i loads into weight set i % 2, which holds the expert of item
    i - 2 (``previous``: the experts that the two sets hold before item 0, -1 for none). When the tile loads skip an id
    past the stacks, an item of that same expert takes the skip id and keeps the set's bytes. Returns (INT32
    [1, C + 3 + spill] first rows, INT32 [1, C + 3 + spill] load ids, the INT32 [1, 1] pair trips
    floor(max(items - spill, 0) / 2), the INT32 [1, 1] odd trip 0 or 1) for the C = E_L T / block candidates (slot s,
    tile k); ``spill`` items run unrolled before the loop."""
    tokens, e_l, block, max_tiles = (dmap[key] for key in ("tokens", "local_experts", "block_tokens", "max_tiles"))
    vector = nisa.engine.vector
    fp32 = nl.float32
    n_cand = e_l * max_tiles
    by_slot = [[n_cand, 1], [max_tiles, e_l], [1, max_tiles]]  # candidate s * max_tiles + k of a [1, C] tile
    per_slot = [[e_l, 1], [1, e_l], [0, max_tiles]]  # a [1, E_L] row repeated over k
    per_tile = [[max_tiles, 1], [0, e_l], [1, max_tiles]]  # a [1, max_tiles] row repeated over s
    # valid[s, k] = (s < count) (tok_count[s] > block k), and not a head item (k = 0 and s, or with dense the slot's
    # expert, below static_slots).
    count_f = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=count_f, src=dmap["active_count"], engine=vector, name=f"{name}_count_f")
    slots = nl.ndarray((1, e_l), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=slots, pattern=[[1, e_l]], offset=0, channel_multiplier=0, name=f"{name}_slots")
    live = nl.ndarray((1, e_l), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=live, data=slots, op0=nl.less, operand0=count_f, engine=vector, name=f"{name}_live")
    counts_f = nl.ndarray((1, e_l), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=counts_f, src=dmap["tok_count"], engine=vector, name=f"{name}_counts_f")
    thresholds = nl.ndarray((1, max_tiles), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=thresholds, pattern=[[block, max_tiles]], offset=0, channel_multiplier=0, name=f"{name}_thr")
    valid = nl.ndarray((1, n_cand), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=valid.ap(pattern=by_slot, offset=0),
        data1=counts_f.ap(pattern=per_slot, offset=0),
        data2=thresholds.ap(pattern=per_tile, offset=0),
        op=nl.greater,
        engine=vector,
        name=f"{name}_valid_tiles",
    )
    nisa.tensor_tensor(
        dst=valid.ap(pattern=by_slot, offset=0),
        data1=valid.ap(pattern=by_slot, offset=0),
        data2=live.ap(pattern=per_slot, offset=0),
        op=nl.multiply,
        engine=vector,
        name=f"{name}_valid_live",
    )
    if static_slots and dense:
        ids_f = nl.ndarray((1, e_l), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ids_f, src=dmap["active_local_id"], engine=vector, name=f"{name}_ids_f")
        keep = nl.ndarray((1, e_l), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=keep, data=ids_f, op0=nl.greater_equal, operand0=float(static_slots), engine=vector, name=f"{name}_keep"
        )
        first_tiles = valid.ap(pattern=[[n_cand, 1], [max_tiles, e_l]], offset=0)
        nisa.tensor_tensor(
            dst=first_tiles, data1=first_tiles, data2=keep, op=nl.multiply, engine=vector, name=f"{name}_head_out"
        )
    elif static_slots:
        nisa.memset(
            dst=valid.ap(pattern=[[n_cand, 1], [max_tiles, static_slots]], offset=0),
            value=0.0,
            engine=vector,
            name=f"{name}_head_out",
        )
    candidates = nl.ndarray((1, n_cand + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=candidates, src=valid, index_offset=0, padding_val=n_cand, name=f"{name}_candidates")
    item_count = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=item_count, src=candidates[0:1, n_cand : n_cand + 1], engine=vector, name=f"{name}_item_count")
    index = nl.ndarray((1, n_cand), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=index, src=candidates[0:1, 0:n_cand], engine=vector, name=f"{name}_index")
    # The fields of each candidate, the padding candidate (index C) last.
    first_rows = nl.ndarray((1, n_cand + 1), dtype=fp32, buffer=nl.sbuf)
    by_slot_ext = [[n_cand + 1, 1], [max_tiles, e_l], [1, max_tiles]]
    nisa.tensor_tensor(
        dst=first_rows.ap(pattern=by_slot_ext, offset=0),
        data1=rows["first_f"].ap(pattern=per_slot, offset=0),
        data2=thresholds.ap(pattern=per_tile, offset=0),
        op=nl.add,
        engine=vector,
        name=f"{name}_first_rows",
    )
    nisa.memset(
        dst=first_rows[0:1, n_cand : n_cand + 1], value=float(e_l * tokens), engine=vector, name=f"{name}_pad_row"
    )
    load_rows = nl.ndarray((1, n_cand + 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=load_rows.ap(pattern=by_slot_ext, offset=0),
        src=rows["load_f"].ap(pattern=per_slot, offset=0),
        engine=vector,
        name=f"{name}_load_rows",
    )
    skips = tiles["skips_oob"]
    skip_id = float(_stack_skip_id(weights)) if skips else 0.0
    nisa.memset(dst=load_rows[0:1, n_cand : n_cand + 1], value=skip_id, engine=vector, name=f"{name}_pad_load")
    width = n_cand + 3 + spill
    # The candidates' first rows, the padding rows last.
    first_items = nl.ndarray((1, n_cand), dtype=fp32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=first_items, data=first_rows, indices=index, name=f"{name}_first_gather")
    item_first = nl.ndarray((1, width), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=item_first[0:1, 0:n_cand], src=first_items, engine=vector, name=f"{name}_first_items")
    nisa.memset(dst=item_first[0:1, n_cand:width], value=e_l * tokens, engine=vector, name=f"{name}_first_pad")
    # The experts of the items after the two the sets hold: [previous X, previous Y, items .., padding x 3].
    experts = nl.ndarray((1, width + 2), dtype=fp32, buffer=nl.sbuf)
    for column, value in enumerate(previous):
        nisa.memset(
            dst=experts[0:1, column : column + 1], value=float(value), engine=vector, name=f"{name}_previous{column}"
        )
    nisa.nc_n_gather(dst=experts[0:1, 2 : 2 + n_cand], data=load_rows, indices=index, name=f"{name}_load_gather")
    nisa.memset(dst=experts[0:1, 2 + n_cand : width + 2], value=skip_id, engine=vector, name=f"{name}_load_pad")
    item_load = nl.ndarray((1, width), dtype=nl.int32, buffer=nl.sbuf)
    if skips:
        # load = expert, or the skip id when the set already holds it: expert + same (skip - expert).
        same = nl.ndarray((1, width), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=same,
            data1=experts[0:1, 2 : width + 2],
            data2=experts[0:1, 0:width],
            op=nl.equal,
            engine=vector,
            name=f"{name}_same",
        )
        to_skip = nl.ndarray((1, width), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=to_skip,
            data=experts[0:1, 2 : width + 2],
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.add,
            operand1=skip_id,
            engine=vector,
            name=f"{name}_to_skip",
        )
        nisa.tensor_tensor(dst=to_skip, data1=to_skip, data2=same, op=nl.multiply, engine=vector, name=f"{name}_moves")
        nisa.tensor_tensor(
            dst=item_load,
            data1=to_skip,
            data2=experts[0:1, 2 : width + 2],
            op=nl.add,
            engine=vector,
            name=f"{name}_load_items",
        )
    else:
        nisa.tensor_copy(dst=item_load, src=experts[0:1, 2 : width + 2], engine=vector, name=f"{name}_load_items")
    # Pair trips floor((items - spill) / 2) = the count of j with spill + 2 j + 1 < items; odd = ceil - floor (0 or 1).
    half = div_ceil(n_cand, 2)
    count_f2 = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=count_f2, src=item_count, engine=vector, name=f"{name}_items_f")
    counts = []
    for label, first in (("ceil", spill), ("floor", spill + 1)):
        evens = nl.ndarray((1, half), dtype=fp32, buffer=nl.sbuf)
        nisa.iota(dst=evens, pattern=[[2, half]], offset=first, channel_multiplier=0, name=f"{name}_{label}_evens")
        started = nl.ndarray((1, half), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=started, data=evens, op0=nl.less, operand0=count_f2, engine=vector, name=f"{name}_{label}_started"
        )
        count = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=count, op=nl.add, data=started, axis=(1,), name=f"{name}_{label}_count")
        counts.append(count)
    pairs = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pairs, src=counts[1], engine=vector, name=f"{name}_pairs")
    odd = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=odd, data1=counts[0], data2=counts[1], op=nl.subtract, engine=vector, name=f"{name}_odd")
    return item_first, item_load, pairs, odd


def packed_tile_provider() -> dict:
    """Return the packed tile functions (``tkg_moe_routed_tiles``: ``alloc_packed_weights``,
    ``load_packed_weights``, ``routed_expert_tile_packed``) and their limits."""
    from . import tkg_moe_routed_tiles as tiles

    return {
        "alloc": tiles.alloc_packed_weights,
        "load": tiles.load_packed_weights,
        "tile": tiles.routed_expert_tile_packed,
        "widths": tiles.PACKED_WIDTHS,
        "max_rows": tiles.MAX_PACKED_ROWS,
        "skips_oob": False,
    }


def default_pack(i_slice: int, local_experts: int) -> int:
    """The experts of one packed item: the largest K that divides E_L with K I_s / 4 down rows <= 512 (1 for an I-slice
    outside ``PACKED_WIDTHS``).

    Example:
        >>> default_pack(96, 112), default_pack(192, 56), default_pack(384, 28)
        (16, 8, 4)
    """
    if i_slice not in (384, 192, 96):
        return 1
    limit = 4 * _PMAX // (i_slice // 4)
    return max(k for k in range(1, min(limit, local_experts) + 1) if local_experts % k == 0)


class _PackedItem:
    """The SBUF tiles of one packed work item: up to 128 pair rows (one (token, expert) pair a row) of K consecutive
    local experts."""

    def __init__(self, rows: int, experts: int):
        self.block = rows
        self.token_row = nl.ndarray((_PMAX, rows), dtype=nl.int32, buffer=nl.sbuf)
        self.gather_index = nl.ndarray((_PMAX, rows), dtype=nl.uint32, buffer=nl.sbuf)
        self.pad_term = nl.ndarray((_PMAX, rows), dtype=nl.float32, buffer=nl.sbuf)
        self.token_raw = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        self.token_col = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        self.gate_col = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        self.bounds = nl.ndarray((1, experts + 1), dtype=nl.float32, buffer=nl.sbuf)
        self.bounds16 = nl.ndarray((1, experts + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        self.masks = nl.ndarray((rows, experts), dtype=nl.float32, buffer=nl.sbuf)
        self.below = nl.ndarray((rows, experts), dtype=nl.float32, buffer=nl.sbuf)
        self.valid = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        self.x_mx = nl.ndarray((_PMAX, _N_D512, rows), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        self.x_scale = nl.ndarray((_PMAX, _N_D512, rows), dtype=nl.uint8, buffer=nl.sbuf)
        self.acc = nl.ndarray((rows, LATENT_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        # The occurrence of each row's token among the rows before it, and the scatter rows of each occurrence.
        self.scatters = min(experts, TOP_K)
        self.same = nl.ndarray((rows, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
        self.occurrence = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        self.token_f = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        self.pick = nl.ndarray((rows, self.scatters), dtype=nl.float32, buffer=nl.sbuf)
        self.scatter_rows = nl.ndarray((rows, self.scatters), dtype=nl.int32, buffer=nl.sbuf)


def _packed_scatter(item: _PackedItem, consts: dict, partial: nl.NkiTensor, name: str) -> tuple:
    """partial[token] += acc[row] for every row of the pack. A token of several of the pack's experts has a row for
    each in the tile, and a scatter read-modify-write takes distinct rows only. So scatter d takes the rows whose token
    has d rows before it in the tile (the pack's rows run by expert, then token, so d < min(K, 16)), in ascending d: a
    token adds its experts in ascending local id. Returns the (first, last) scatter."""
    rows = item.block
    vector = nisa.engine.vector
    # same[j, i] = (token of row i == token of row j) (i < j): the earlier rows of the token; occurrence = their count.
    nisa.tensor_copy(dst=item.token_f, src=item.token_raw, engine=vector, name=f"{name}_token_raw_f")
    nisa.tensor_scalar(
        dst=item.same,
        data=item.token_row[0:rows, :],
        op0=nl.equal,
        operand0=item.token_f,
        engine=vector,
        name=f"{name}_same",
    )
    nisa.tensor_tensor(
        dst=item.same,
        data1=item.same,
        data2=consts["before"][0:rows, :],
        op=nl.multiply,
        engine=vector,
        name=f"{name}_same_before",
    )
    nisa.tensor_reduce(dst=item.occurrence, op=nl.add, data=item.same, axis=(1,), name=f"{name}_occurrence")
    # (token + 1) (occurrence == d) - 1: the row of scatter d, or -1.
    nisa.tensor_scalar(
        dst=item.token_f, data=item.token_col, op0=nl.add, operand0=1.0, engine=vector, name=f"{name}_token_p1"
    )
    first = previous = None
    for d in range(item.scatters):
        nisa.tensor_scalar(
            dst=item.pick[:, d : d + 1],
            data=item.occurrence,
            op0=nl.equal,
            operand0=float(d),
            op1=nl.multiply,
            operand1=item.token_f,
            engine=vector,
            name=f"{name}_pick{d}",
        )
        nisa.tensor_scalar(
            dst=item.scatter_rows[:, d : d + 1],
            data=item.pick[:, d : d + 1],
            op0=nl.add,
            operand0=-1.0,
            engine=vector,
            name=f"{name}_rows{d}",
        )
        dst = partial.ap(
            pattern=[[LATENT_SIZE, rows], [1, LATENT_SIZE]],
            offset=0,
            vector_offset=item.scatter_rows[:, d : d + 1],
            indirect_dim=0,
        )
        scatter = nisa.dma_compute(
            dst=dst,
            srcs=[dst, item.acc[0:rows, :]],
            reduce_op=nl.add,
            scales=[1.0, 1.0],
            oob_mode=nisa.oob_mode.skip,
            name=f"{name}_scatter_add{d}",
        )
        _wait_for(scatter, previous)
        first = scatter if first is None else first
        previous = scatter
    return first, previous


def _packed_inputs(item, dmap, consts, pack_first, start, x_mx, x_scale, pad_column, name, in_loop=False):
    """Load a packed item: the pair rows [start, start + 128) of the CSR arrays (``start`` INT32 [1, 1] SBUF; -1 / +0
    past the arrays), the row masks of its K experts (row j belongs to expert k when start + j lies in
    [offset(pK + k), offset(pK + k + 1)); ``pack_first`` = pK, a trace-time int or an INT32 [1, 1] SBUF tensor), the
    token of each row (-1 for a row of no expert of the pack) and the rows' MXFP8 latent columns."""
    rows, experts = item.block, item.masks.shape[1]
    vector = nisa.engine.vector
    fp32 = nl.float32
    nisa.memset(dst=item.token_row, value=-1, engine=vector, name=f"{name}_row_fill")
    nisa.memset(dst=item.token_raw, value=-1, engine=vector, name=f"{name}_raw_fill")
    nisa.memset(dst=item.gate_col, value=0.0, engine=vector, name=f"{name}_gate_fill")
    for dst, table, pattern, label in (
        (item.token_row, dmap["compact_tok"], [[0, _PMAX], [1, rows]], "row"),
        (item.token_raw, dmap["compact_tok"], [[1, rows], [1, 1]], "col"),
        (item.gate_col, dmap["gate_of_pair"], [[1, rows], [1, 1]], "gate"),
    ):
        load = nisa.dma_copy(
            dst=dst,
            src=table.ap(pattern=pattern, offset=0, scalar_offset=start, indirect_dim=0),
            oob_mode=nisa.oob_mode.skip,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{name}_{label}_load",
        )
        for store in () if in_loop else dmap["csr_writes"]:
            _wait_for(load, store)
    # The pack's bounds relative to the tile, clamped to [-1, rows + 1] (exact in BF16), on every row (a 0/1 matmul).
    offsets = dmap["expert_offset"]
    e_l = dmap["local_experts"]
    if isinstance(pack_first, int):
        selected = offsets[0:1, pack_first : pack_first + experts + 1]
    else:
        selected = nl.ndarray((1, experts + 1), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=selected,
            src=offsets.ap(
                pattern=[[e_l + 1, 1], [1, experts + 1]], offset=0, scalar_offset=pack_first, indirect_dim=1
            ),
            engine=vector,
            name=f"{name}_bounds_select",
        )
    start_f = nl.ndarray((1, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=start_f, src=start, engine=vector, name=f"{name}_start_f")
    nisa.tensor_scalar(
        dst=item.bounds, data=selected, op0=nl.subtract, operand0=start_f, engine=vector, name=f"{name}_bounds"
    )
    nisa.tensor_scalar(
        dst=item.bounds16,
        data=item.bounds,
        op0=nl.maximum,
        operand0=-1.0,
        op1=nl.minimum,
        operand1=float(rows + 1),
        engine=vector,
        name=f"{name}_bounds_clamp",
    )
    bounds_psum = nl.ndarray((rows, experts + 1), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(
        dst=bounds_psum, stationary=consts["ones_row"][0:1, 0:rows], moving=item.bounds16, name=f"{name}_bounds_bcast"
    )
    row_ids = consts["row_ids"][0:rows, :].ap(pattern=[[1, rows], [0, experts]], offset=0)
    nisa.tensor_tensor(
        dst=item.masks,
        data1=row_ids,
        data2=bounds_psum[:, 0:experts],
        op=nl.greater_equal,
        engine=vector,
        name=f"{name}_masks_lo",
    )
    nisa.tensor_tensor(
        dst=item.below,
        data1=row_ids,
        data2=bounds_psum[:, 1 : experts + 1],
        op=nl.less,
        engine=vector,
        name=f"{name}_masks_hi",
    )
    nisa.tensor_tensor(
        dst=item.masks, data1=item.masks, data2=item.below, op=nl.multiply, engine=vector, name=f"{name}_masks"
    )
    nisa.tensor_reduce(dst=item.valid, op=nl.add, data=item.masks, axis=(1,), name=f"{name}_valid")
    # The scatter row: the pair's token, or -1 for a row of no expert of the pack: (token + 1) valid - 1.
    token_f = nl.ndarray((rows, 1), dtype=fp32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=token_f,
        data=item.token_raw,
        op0=nl.add,
        operand0=1.0,
        op1=nl.multiply,
        operand1=item.valid,
        name=f"{name}_token_f",
    )
    nisa.tensor_scalar(
        dst=item.token_col, data=token_f, op0=nl.add, operand0=-1.0, engine=vector, name=f"{name}_tokens"
    )
    nisa.tensor_scalar(
        dst=item.pad_term,
        data=item.token_row,
        op0=nl.less,
        operand0=0,
        op1=nl.multiply,
        operand1=float(pad_column + 1),
        engine=vector,
        name=f"{name}_pad_term",
    )
    nisa.tensor_tensor(
        dst=item.gather_index,
        data1=item.pad_term,
        data2=item.token_row,
        op=nl.add,
        engine=vector,
        name=f"{name}_index",
    )
    for tile_h in range(_N_D512):
        nisa.nc_n_gather(
            dst=item.x_mx.view(nl.uint32)[:, tile_h, :],
            data=x_mx.view(nl.uint32)[:, tile_h, :],
            indices=item.gather_index,
            name=f"{name}_codes_h{tile_h}",
        )
        nisa.nc_n_gather(
            dst=item.x_scale[:, tile_h, :],
            data=x_scale[:, tile_h, :],
            indices=item.gather_index,
            name=f"{name}_scales_h{tile_h}",
        )


def _packed_partial(
    dmap, x_mx, x_scale, expert_weights, partial, i_slice, name, pack, weight_buffers, tiles, after, beta, linear_beta
) -> dict:
    """``routed_dynamic_partial`` with packed items: item (p, m) runs local experts pK .. pK + K - 1 (pack p) on pair
    rows [offset(pK) + 128 m, + 128) of the CSR arrays (pair rows of the pack's experts in local id order, each
    expert's tokens ascending) with ``routed_expert_tile_packed``. Tile 0 of every pack runs unrolled, with trace-time
    expert ids (every slot loads before each call: the tile reads them all); the other tiles of a pack with more than
    128 pairs run in a dynamic loop, one a trip, in ascending (pack, tile). The scatter adds each pair row into its
    token's row of the partial (``_packed_scatter``). A token sums its experts in ascending local id, except an expert
    in a later tile of a pack, which adds after the unrolled tiles of every pack. The weight sets are allocated with
    zero scales, so a slot that holds no loaded expert reads finite values (the rule of the packed tile)."""
    tokens, e_l = dmap["tokens"], dmap["local_experts"]
    tiles = packed_tile_provider() if tiles is None else tiles
    kernel_assert(i_slice in tiles["widths"], f"the packed form takes an I-slice in {tiles['widths']}, got {i_slice}")
    kernel_assert(
        e_l % pack == 0 and pack * (i_slice // 4) <= tiles["max_rows"],
        f"pack must divide E_L {e_l} with pack x {i_slice // 4} down rows <= {tiles['max_rows']}, got {pack}",
    )
    kernel_assert(dmap.get("compact_tok") is not None, "the packed form reads the CSR arrays: build the map with csr")
    columns = x_mx.shape[2]
    pad_column = tokens if columns == tokens + 1 else 0
    rows = _PMAX
    n_packs = e_l // pack
    max_m = div_ceil(pack * tokens, rows)
    vector = nisa.engine.vector
    consts = {
        "ones_row": nl.ndarray((1, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf),
        "row_ids": nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf),
        "before": nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf),
    }
    nisa.memset(dst=consts["ones_row"], value=1.0, engine=vector, name=f"{name}_ones_row")
    nisa.iota(dst=consts["row_ids"], pattern=[[0, 1]], offset=0, channel_multiplier=1, name=f"{name}_row_ids")
    # before[j, i] = (i < j).
    nisa.iota(dst=consts["before"], pattern=[[1, _PMAX]], offset=0, channel_multiplier=-1, name=f"{name}_ramp")
    nisa.tensor_scalar(
        dst=consts["before"], data=consts["before"], op0=nl.less, operand0=0.0, engine=vector, name=f"{name}_before"
    )
    offsets_i = nl.ndarray((1, e_l + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=offsets_i, src=dmap["expert_offset"], engine=vector, name=f"{name}_offsets_i")
    zero = zero_routed_rows(partial, dmap, f"{name}_z")
    pending = list(zero)
    # One set of the packed weights is large (about 80 KB a partition at I-slice 96, K 16): two only when they fit.
    set_bytes = pack * _N_D512 * 2 * i_slice * 3 + 3 * div_ceil(pack * i_slice // 4, _PMAX) * LATENT_SIZE * 3
    n_sets = max(1, min(weight_buffers, n_packs, 2 if 2 * set_bytes <= 100 * 1024 else 1))
    weight_sets = [tiles["alloc"](i_slice, pack, f"{name}_w{idx}", zero_scales=True) for idx in range(n_sets)]
    rings = [_PackedItem(rows, pack) for _ in range(2)]

    def run(item, weights, pack_first, start, expert_of, tag, after_pe, in_loop=False):
        _packed_inputs(item, dmap, consts, pack_first, start, x_mx, x_scale, pad_column, tag, in_loop=in_loop)
        for slot in range(pack):
            tiles["load"](weights, *expert_weights, slot, expert_of(slot), i_slice, f"{tag}_load{slot}")
        last = tiles["tile"](
            item.acc,
            item.x_mx,
            item.x_scale,
            weights,
            item.masks,
            rows,
            i_slice,
            f"{tag}_tile",
            after=after_pe,
            affinity=item.gate_col,
            accumulate=False,
            beta=beta,
            linear_beta=linear_beta,
        )
        return last, _packed_scatter(item, consts, partial, tag)

    scatters, last_pe = [], after
    for p in range(n_packs):
        first = p * pack
        last_pe, scatter = run(
            rings[p % 2],
            weight_sets[p % n_sets],
            first,
            offsets_i[0:1, first : first + 1],
            lambda slot, first=first: first + slot,
            f"{name}_p{p}",
            last_pe,
        )
        for earlier in pending:
            _wait_for(scatter[0], earlier)
        pending = [scatter[1]]
        scatters.append(scatter[1])
    loop = max_m > 1
    if loop:
        item_start, item_pack, item_count = _packed_tail_items(dmap, pack, n_packs, max_m, rows, f"{name}_tail")
        n_items = item_start.shape[1] - 1
        shifted = nl.ndarray((1, n_items), dtype=nl.int32, buffer=nl.sbuf)
        slot_ids = nl.ndarray((1, pack), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=slot_ids, pattern=[[1, pack]], offset=0, channel_multiplier=0, name=f"{name}_slot_ids")
        pack_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        experts_i = nl.ndarray((1, pack), dtype=nl.int32, buffer=nl.sbuf)

        def tail_body(_iteration):
            # The pack's local experts pK + k (every slot loads: the tile reads them all).
            nisa.tensor_copy(dst=pack_f, src=item_pack[0:1, 0:1], engine=vector, name=f"{name}_tail_pack_f")
            nisa.tensor_scalar(
                dst=experts_i, data=slot_ids, op0=nl.add, operand0=pack_f, engine=vector, name=f"{name}_tail_experts"
            )
            run(
                rings[0],
                weight_sets[0],
                item_pack[0:1, 0:1],
                item_start[0:1, 0:1],
                lambda slot: experts_i[0:1, slot : slot + 1],
                f"{name}_tail_item",
                None,
                in_loop=True,
            )
            for label, row in (("start", item_start), ("pack", item_pack)):
                nisa.tensor_copy(
                    dst=shifted, src=row[0:1, 1 : n_items + 1], engine=vector, name=f"{name}_tail_{label}_next"
                )
                nisa.tensor_copy(
                    dst=row[0:1, 0:n_items], src=shifted, engine=vector, name=f"{name}_tail_{label}_advance"
                )

        nl.fori_loop(0, nisa.scalar_load(item_count), tail_body)
    return {"zero": zero, "scatters": scatters, "last": last_pe, "loop": loop}


def _packed_tail_items(dmap: dict, pack: int, n_packs: int, max_m: int, rows: int, name: str) -> tuple:
    """The overflow items (p, m), m >= 1, of the packs with more than 128 m pairs, in ascending (pack, tile), on
    partition 0, then one padding item (a start past the CSR arrays, pack 0). Returns (INT32 [1, C + 1] CSR starts,
    INT32 [1, C + 1] first local experts pK, the INT32 [1, 1] count) for the C = packs (max_m - 1) candidates."""
    vector = nisa.engine.vector
    fp32 = nl.float32
    e_l = dmap["local_experts"]
    offsets = dmap["expert_offset"]
    later = max_m - 1
    n_cand = n_packs * later
    by_pack = [[e_l + 1, 1], [pack, n_packs]]
    pairs = nl.ndarray((1, n_packs), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=pairs,
        data1=offsets.ap(pattern=by_pack, offset=pack),
        data2=offsets.ap(pattern=by_pack, offset=0),
        op=nl.subtract,
        engine=vector,
        name=f"{name}_pairs",
    )
    thresholds = nl.ndarray((1, later), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(dst=thresholds, pattern=[[rows, later]], offset=rows, channel_multiplier=0, name=f"{name}_thr")
    per_pack = [[n_packs, 1], [1, n_packs], [0, later]]
    per_tile = [[later, 1], [0, n_packs], [1, later]]
    valid = nl.ndarray((1, n_cand), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=valid.ap(pattern=[[n_cand, 1], [later, n_packs], [1, later]], offset=0),
        data1=pairs.ap(pattern=per_pack, offset=0),
        data2=thresholds.ap(pattern=per_tile, offset=0),
        op=nl.greater,
        engine=vector,
        name=f"{name}_valid",
    )
    candidates = nl.ndarray((1, n_cand + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=candidates, src=valid, index_offset=0, padding_val=n_cand, name=f"{name}_candidates")
    item_count = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=item_count, src=candidates[0:1, n_cand : n_cand + 1], engine=vector, name=f"{name}_count")
    index = nl.ndarray((1, n_cand), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=index, src=candidates[0:1, 0:n_cand], engine=vector, name=f"{name}_index")
    by_cand = [[n_cand + 1, 1], [later, n_packs], [1, later]]
    starts = nl.ndarray((1, n_cand + 1), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=starts.ap(pattern=by_cand, offset=0),
        data1=offsets.ap(pattern=[[e_l + 1, 1], [pack, n_packs], [0, later]], offset=0),
        data2=thresholds.ap(pattern=per_tile, offset=0),
        op=nl.add,
        engine=vector,
        name=f"{name}_starts",
    )
    past = float(dmap["compact_tok"].shape[0])
    nisa.memset(dst=starts[0:1, n_cand : n_cand + 1], value=past, engine=vector, name=f"{name}_pad_start")
    firsts = nl.ndarray((1, n_cand + 1), dtype=fp32, buffer=nl.sbuf)
    nisa.iota(
        dst=firsts.ap(pattern=by_cand, offset=0),
        pattern=[[pack, n_packs], [0, later]],
        offset=0,
        channel_multiplier=0,
        name=f"{name}_firsts",
    )
    nisa.memset(dst=firsts[0:1, n_cand : n_cand + 1], value=0.0, engine=vector, name=f"{name}_pad_first")
    outputs = []
    for label, source in (("start", starts), ("pack", firsts)):
        gathered = nl.ndarray((1, n_cand), dtype=fp32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=gathered, data=source, indices=index, name=f"{name}_{label}_gather")
        row = nl.ndarray((1, n_cand + 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=row[0:1, 0:n_cand], src=gathered, engine=vector, name=f"{name}_{label}_items")
        nisa.tensor_copy(
            dst=row[0:1, n_cand : n_cand + 1],
            src=source[0:1, n_cand : n_cand + 1],
            engine=vector,
            name=f"{name}_{label}_pad",
        )
        outputs.append(row)
    return outputs[0], outputs[1], item_count


def routed_partial_tile(
    partial: nl.NkiTensor, dmap: dict, tile_idx: int, name: str, after=None, dtype=nl.bfloat16
) -> nl.NkiTensor:
    """Return the SBUF rows [rows, 3584] of token tile ``tile_idx`` of the rank's routed partial: the FP32 rows of the
    routed tokens, +0 for the others (one gather; an unrouted row is out of bounds and keeps the zero fill), rounded
    once to BF16 on VectorE (``dtype`` BF16) or left FP32. ``after``: DMAs (the last scatters) that the read waits
    for, or None."""
    rows_i = dmap["routed_rows"][tile_idx]
    rows_u = rows_i.shape[0]
    rows_f = nl.ndarray((rows_u, LATENT_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=rows_f, value=0.0, engine=nisa.engine.gpsimd, name=f"{name}_zero")
    read = nisa.dma_copy(
        dst=rows_f,
        src=partial.ap(
            pattern=[[LATENT_SIZE, rows_u], [1, LATENT_SIZE]], offset=0, vector_offset=rows_i, indirect_dim=0
        ),
        oob_mode=nisa.oob_mode.skip,
        dge_mode=nisa.dge_mode.swdge,
        name=f"{name}_read",
    )
    for wait in [] if after is None else list(after) if isinstance(after, (list, tuple)) else [after]:
        _wait_for(read, wait)
    if dtype == nl.float32:
        return rows_f
    rows_out = nl.ndarray((rows_u, LATENT_SIZE), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rows_out, src=rows_f, engine=nisa.engine.vector, name=f"{name}_round")
    return rows_out


def routed_partial_rows(partial: nl.NkiTensor, dmap: dict, name: str, after=None) -> list:
    """Return the BF16 SBUF rows of every token tile of the rank's routed partial (``routed_partial_tile``)."""
    return [
        routed_partial_tile(partial, dmap, tile_idx, f"{name}_t{tile_idx}", after=after)
        for tile_idx in range(len(dmap["routed_rows"]))
    ]


def routed_dynamic_layer(
    hbm: dict,
    affinity_tiles: list,
    code_tiles: list,
    scale_tiles: list,
    expert_weights: tuple,
    i_slice: int,
    name: str,
    static_slots: int = None,
    pack: int = 1,
    csr_gather: str = CSR_GATHER_DEFAULT,
) -> dict:
    """Run one layer's routed experts of a rank in the dynamic form, from the token tiles of the static prep loop.

    ``affinity_tiles``: the FP32 [rows, E_L] local affinities of each token tile (``_local_affinities``);
    ``code_tiles`` / ``scale_tiles``: its MXFP8 latent [128, 7, rows] (``_quantize_natural_latent``). The codes of the
    tiles go into one gather source (GpSimd copies; a zero column T when T < 512), then ``build_dispatch_map`` on the
    affinities and ``routed_dynamic_partial`` into the FP32 partial of ``hbm`` (``hbm["routed_dynamic_partial"]``,
    [T, 3584] private HBM, one for each HBM set of the body). Read each token tile's rows with ``routed_partial_tile``
    (``after`` = the returned ``scatters``). ``pack``: the experts of one packed item (``routed_dynamic_partial``).
    ``csr_gather``: the CSR gathers of a packed layer (``build_dispatch_map``; ``CSR_GATHER_FORMS``).

    Returns:
        {"dmap", "partial", "scatters", "last"}: the dispatch map, the FP32 partial, the scatter DMAs that a read
        waits for, the last TensorE instruction outside the dynamic loop.
    """
    rows = [tile.shape[0] for tile in affinity_tiles]
    tokens = sum(rows)
    _check_tokens(tokens)
    e_l = affinity_tiles[0].shape[1]
    columns = tokens + 1 if tokens < MAX_TOKENS else tokens
    x_mx = nl.ndarray((_PMAX, _N_D512, columns), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    x_scale = nl.ndarray((_PMAX, _N_D512, columns), dtype=nl.uint8, buffer=nl.sbuf)
    pool = nisa.engine.gpsimd
    if columns > tokens:
        # The zero column: code 0 and scale byte 0 (a finite E8M0 scale).
        nisa.memset(dst=x_mx.view(nl.uint32)[:, :, tokens:columns], value=0, engine=pool, name=f"{name}_zero_codes")
        nisa.memset(dst=x_scale[:, :, tokens:columns], value=0, engine=pool, name=f"{name}_zero_scales")
    first = 0
    for tile_idx, (codes, scales) in enumerate(zip(code_tiles, scale_tiles, strict=True)):
        cols = slice(first, first + rows[tile_idx])
        nisa.tensor_copy(
            dst=x_mx.view(nl.uint32)[:, :, cols], src=codes.view(nl.uint32), engine=pool, name=f"{name}_t{tile_idx}_x"
        )
        nisa.tensor_copy(dst=x_scale[:, :, cols], src=scales, engine=pool, name=f"{name}_t{tile_idx}_s")
        first += rows[tile_idx]
    partial = hbm.get("routed_dynamic_partial")
    if partial is None:
        partial = nl.ndarray(
            (tokens, LATENT_SIZE), dtype=nl.float32, buffer=nl.private_hbm, name=f"moe2s_{name}_partial"
        )
        hbm["routed_dynamic_partial"] = partial
    kernel_assert(
        tuple(partial.shape) == (tokens, LATENT_SIZE),
        f"routed_dynamic_layer keeps one FP32 [{tokens}, {LATENT_SIZE}] partial an HBM set, got {tuple(partial.shape)}",
    )
    # The map tables of each (T, E_L, CSR): layers of different splits (a per-layer plan) share an HBM set.
    scratches = hbm.setdefault("routed_dynamic_scratches", {})
    key = (tokens, e_l, pack > 1)
    if key not in scratches:
        scratches[key] = dispatch_scratch(tokens, e_l, f"moe2s_{name}_map", csr=pack > 1)
    dmap = build_dispatch_map(
        None,
        None,
        None,
        e_l,
        name=f"{name}_map",
        tokens=tokens,
        dense_affinities=affinity_tiles,
        csr=pack > 1,
        scratch=scratches[key],
        csr_gather=csr_gather,
    )
    result = routed_dynamic_partial(
        dmap, x_mx, x_scale, expert_weights, partial, i_slice, name, static_slots, pack=pack
    )
    return {"dmap": dmap, "partial": partial, "scatters": result["scatters"], "last": result["last"]}
