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
GLM-5.2 MLA decode: the two column-sharded down-projections that open the layer.

Both projections contract the same hidden states, so they share one transpose of hidden onto
partitions; that is why they are one kernel. Each output is this rank's column slice, and the caller
AllGathers both before the norms run.
"""

import os

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from .mla_decode_constants import P_MAX
from .mla_proj_constants import DownProjShape, get_down_proj_shape
from .mla_proj_primitives import linear, load_gamma_broadcast, rmsnorm_inplace, transpose_into_k_tiles

LAT_DOWNPROJ_CHUNK_LOAD = int(os.environ.get("GLM52K_LAT_DOWNPROJ_CHUNK_LOAD", "0"))
"""Without ``input_norm_w``, load the hidden states in this many column chunks on alternating Sync and ACT HWDGE
queues, so the K-tile transposes start when the first chunk lands rather than after the whole tensor (1.5 MB at
128 tokens). Same bytes, bit-identical. 0 (default) loads it with one DMA."""


@nki.jit
def mla_decode_down_proj(
    hidden_states: nl.NkiTensor,
    wq_a: nl.NkiTensor,
    wkv_a: nl.NkiTensor,
    input_norm_w: nl.NkiTensor | None = None,
    norm_eps: float = 1e-6,
    output: nl.NkiTensor | None = None,
    hidden_tiles_out: list | None = None,
    q_out_extra: nl.NkiTensor | None = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Project hidden states down to this rank's column slice of the query latent and the cache row.

    Usage:
        Tuned for: GLM-5.2 MLA decode, both down-projections column-sharded, LNC=1
        Applicable to: any MLA down-projection pair sharded on its output width

    Neither output is normalized. Both norms reduce over the sharded axis, so norming here would need an
    AllReduce of partial sums of squares, which costs more than gathering the narrow outputs and
    norming after. ``mla_decode_q_kv_up`` asserts complete widths so a forgotten AllGather fails loudly.
    A very narrow shard wastes weight bandwidth once a bf16 row drops under the 64-byte minimum DMA
    transaction.

    Args:
        hidden_states (nl.NkiTensor): [T, hidden] @ HBM, the residual stream with the whole hidden
            width. The shard splits the outputs, not the contraction, so the caller AllGathers the
            token-sharded residual stream first.
        wq_a (nl.NkiTensor): [hidden, q_lora_shard] @ HBM. This rank's columns of the query
            down-projection; its width is the shard (all columns run unsharded).
        wkv_a (nl.NkiTensor): [hidden, kv_row_shard] @ HBM. This rank's columns of the KV
            down-projection.
        input_norm_w (nl.NkiTensor | None): [1, hidden] fp32 @ HBM. When given, the layer's input
            RMSNorm is folded into the hidden load; it reduces over hidden, which is not sharded.
        norm_eps (float): Epsilon for the input RMSNorm.
        output (nl.NkiTensor | None): [T, q_lora_shard + kv_row_shard] @ HBM. When given, both shards
            are written into it side by side (query latent first) and nothing is allocated here. A
            caller that gathers the pair as one tensor passes the collective source, saving a copy.
        hidden_tiles_out (list | None): When given, ``(hidden K-tiles, linear stationary_fn)`` is appended,
            for a caller that projects the same hidden again (the indexer's keys).
        q_out_extra (nl.NkiTensor | None): [T, q_lora_shard] @ HBM, with ``output``. The query shard is
            also stored here from the same eviction (the block AllGather's source), so no HBM copy follows.

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor]:
            - q_latent_shard [T, q_lora_shard] @ HBM, partial: AllGather before use. A column view of
              ``output`` when that is given.
            - kv_row_shard [T, kv_row_shard] @ HBM, partial, likewise.
    """

    shapes = get_down_proj_shape(hidden_states, wq_a, wkv_a)
    T = shapes.num_tokens

    # Hidden onto partitions once, then reused by both projections.
    hidden_tiles_sbuf, stationary_fn = _hidden_k_tiles(hidden_states, input_norm_w, norm_eps, shapes)
    if hidden_tiles_out is not None:
        hidden_tiles_out.append((hidden_tiles_sbuf, stationary_fn))

    q_w, kv_w = shapes.q_lora_shard, shapes.kv_row_shard
    if output is not None:
        # Each evicted chain is stored straight to its columns of ``output``, with no band DMA or second store.
        kernel_assert(
            tuple(output.shape) == (T, q_w + kv_w) and output.dtype == shapes.dtype,
            f"[MLA Decode Down Proj] output must be [{T}, {q_w + kv_w}] {shapes.dtype}, got {output.shape} {output.dtype}",
        )
        linear(hidden_tiles_sbuf, wq_a, T, stationary_fn=stationary_fn, out_hbm=output, out_col0=0, out_hbm2=q_out_extra)
        linear(hidden_tiles_sbuf, wkv_a, T, stationary_fn=stationary_fn, out_hbm=output, out_col0=q_w)
        return output[0:T, 0:q_w], output[0:T, q_w : q_w + kv_w]

    q_latent_sbuf = linear(hidden_tiles_sbuf, wq_a, T, stationary_fn=stationary_fn)
    kv_row_sbuf = linear(hidden_tiles_sbuf, wkv_a, T, stationary_fn=stationary_fn)

    q_latent_out = nl.ndarray((T, q_w), dtype=shapes.dtype, buffer=nl.shared_hbm)
    kv_row_out = nl.ndarray((T, kv_w), dtype=shapes.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=q_latent_out[0:T, 0:q_w], src=q_latent_sbuf[0:T, 0:q_w])
    nisa.dma_copy(dst=kv_row_out[0:T, 0:kv_w], src=kv_row_sbuf[0:T, 0:kv_w])

    return q_latent_out, kv_row_out


def mla_decode_hidden_k_tiles(hidden_states: nl.NkiTensor) -> tuple:
    """The ``linear`` K-tiles of normed hidden states, built exactly as :func:`mla_decode_down_proj` builds them.

    For rows the down-projection does not see: with a latent-sharded down-projection it projects the latent
    group's rows, while the indexer's keys need this rank's own token slice. Same loads, transposes and
    evictions, so the keys come out the same either way.

    Args:
        hidden_states (nl.NkiTensor): [T, hidden] @ HBM, already normed.

    Returns:
        tuple: ``(hidden K-tiles, linear stationary_fn)``, as ``hidden_tiles_out`` receives them.
    """
    T, hidden = hidden_states.shape
    kernel_assert(T <= P_MAX, f"[MLA Decode Down Proj] num_tokens={T} must be <= {P_MAX}, one token per partition")
    shapes = DownProjShape(num_tokens=T, hidden=hidden, q_lora_shard=0, kv_row_shard=0, dtype=hidden_states.dtype)
    return _hidden_k_tiles(hidden_states, None, 0.0, shapes)


def _hidden_k_tiles(
    hidden_states: nl.NkiTensor,
    input_norm_w: nl.NkiTensor | None,
    eps: float,
    shapes: DownProjShape,
) -> tuple:
    """Hidden states onto partitions as ``linear`` K-tiles, returned as ``(tiles, stationary_fn)``.

    Without ``input_norm_w`` the hidden is loaded partition-spread (``reshape((128, T * H / 128))``, one
    contiguous DMA) and K-tiled with ``T * H / 128**2`` transposes and one eviction instead of ``H / 128`` of
    each; ``linear`` reads each k-tile as a stride-``128 / T`` column set. The transposes are pass-through and
    the K order is unchanged, so the result is bit-identical.
    """
    T = shapes.num_tokens
    # At T == P_MAX the spread is a single column group: no gain, and it would hold 6 of the 8 PSUM banks.
    if input_norm_w is None and T < P_MAX and _hidden_spread_fits(T, shapes.hidden):
        return _load_hidden_spread_k_tiles(hidden_states, shapes)
    hidden_sbuf = _load_hidden(hidden_states, input_norm_w, eps, shapes)
    return transpose_into_k_tiles(hidden_sbuf, T), None


def _load_hidden(
    hidden_states: nl.NkiTensor,
    input_norm_w: nl.NkiTensor | None,
    eps: float,
    shapes: DownProjShape,
) -> nl.NkiTensor:
    """Loads hidden states to SBUF, applying the layer's input RMSNorm when ``input_norm_w`` is given."""

    T, hidden = shapes.num_tokens, shapes.hidden

    hidden_sbuf = nl.ndarray((P_MAX, hidden), dtype=shapes.dtype, buffer=nl.sbuf)
    n_chunks = LAT_DOWNPROJ_CHUNK_LOAD if input_norm_w is None else 0
    if n_chunks > 1 and hidden % (n_chunks * P_MAX) == 0:
        # Whole 128-column K-tiles per chunk: each transpose reads one chunk's columns only.
        width = hidden // n_chunks
        engines = (nisa.engine.sync, nisa.engine.scalar)
        for i in range(n_chunks):
            nisa.dma_copy(
                dst=hidden_sbuf[0:T, i * width : (i + 1) * width],
                src=hidden_states[0:T, i * width : (i + 1) * width],
                dge_mode=nisa.dge_mode.hwdge,
                engine=engines[i % 2],
            )
    else:
        nisa.dma_copy(dst=hidden_sbuf[0:T, 0:hidden], src=hidden_states[0:T, 0:hidden])

    if input_norm_w is not None:
        gamma_sbuf = load_gamma_broadcast(input_norm_w, T)
        rmsnorm_inplace(hidden_sbuf[0:P_MAX, 0:hidden], gamma_sbuf, T, eps)

    return hidden_sbuf


def _hidden_spread_fits(T: int, hidden: int) -> bool:
    """Whether ``[T, hidden]`` spreads over ``P_MAX`` partitions as ``P_MAX / T`` whole 128-multiple column groups."""
    if P_MAX % T != 0 or hidden % (P_MAX // T) != 0:
        return False
    return (hidden // (P_MAX // T)) % P_MAX == 0


def _load_hidden_spread_k_tiles(hidden_states: nl.NkiTensor, shapes: DownProjShape):
    """Hidden K-tiled through the partition-spread view, for ``linear(stationary_fn=)``.

    Partition ``p = t * G + j`` (``G = P_MAX / T``) holds ``hidden[t, j * F : (j + 1) * F]``, so transpose ``c``
    of the ``F / 128`` column blocks puts k-tile ``kt = j * (F / 128) + c`` of token ``t`` at
    ``tiles[:, c, t * G + j]``: k-tile ``kt``'s stationary is the stride-``G`` column set starting at ``(c, j)``.

    Returns:
        tuple: the ``[P_MAX, F / 128, P_MAX]`` k-tiles and the map from a k-tile index to its ``[P_MAX, T]``
        stationary.
    """
    T, H = shapes.num_tokens, shapes.hidden
    groups = P_MAX // T
    F = H // groups
    n_sub = F // P_MAX

    hidden_sbuf = nl.ndarray((P_MAX, F), dtype=shapes.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=hidden_sbuf, src=hidden_states.reshape((P_MAX, F)))

    tiles_psum = nl.ndarray((P_MAX, n_sub, P_MAX), dtype=shapes.dtype, buffer=nl.psum)
    for c in range(n_sub):
        nisa.nc_transpose(tiles_psum[0:P_MAX, c, 0:P_MAX], hidden_sbuf[0:P_MAX, c * P_MAX : (c + 1) * P_MAX])
    tiles_sbuf = nl.ndarray((P_MAX, n_sub, P_MAX), dtype=shapes.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tiles_sbuf, src=tiles_psum)

    def stationary(k_index: int) -> nl.NkiTensor:
        j, c = divmod(k_index, n_sub)
        return tiles_sbuf.ap(pattern=[[n_sub * P_MAX, P_MAX], [groups, T]], offset=c * P_MAX + j)

    return tiles_sbuf, stationary
