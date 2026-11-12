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
partitions; that is the reason to fuse them. Each output is this rank's column slice, and the caller
must AllGather both before the norms can run.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ....attention.mla.mla_decode_constants import P_MAX
from ....attention.mla.mla_proj_constants import DownProjShape, get_down_proj_shape
from ....utils.kernel_assert import kernel_assert
from .mla_proj_primitives import linear, load_gamma_broadcast, rmsnorm_inplace, transpose_into_k_tiles


@nki.jit
def mla_decode_down_proj(
    hidden_states: nl.NkiTensor,
    wq_a: nl.NkiTensor,
    wkv_a: nl.NkiTensor | None,
    input_norm_w: nl.NkiTensor | None = None,
    norm_eps: float = 1e-6,
    output: nl.NkiTensor | None = None,
    hidden_tiles_out: list | None = None,
    hidden_tiles_sbuf: nl.NkiTensor | None = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Project hidden states down to this rank's column slice of the query latent and the cache row.

    Usage:
        Tuned for: GLM-5.2 MLA decode, column-shard-64 on both down-projection outputs, LNC=1
        Applicable to: any MLA down-projection pair sharded on its output width

    Args:
        hidden_states (nl.NkiTensor): [T, hidden] @ HBM, with the whole hidden width. The shard splits
            the outputs, not the contraction, so the caller AllGathers the token-sharded residual first.
        wq_a (nl.NkiTensor): [hidden, q_lora_shard] @ HBM, this rank's query down-projection columns.
            Its width is the shard: ``q_lora_rank / 64`` columns for column-shard-64, all to run unsharded.
        wkv_a (nl.NkiTensor | None): [hidden, kv_row_shard] @ HBM, this rank's KV down-projection columns.
            None projects the query latent only (an MTP KVShare draft step writes no cache rows); the
            second result is then None.
        input_norm_w (nl.NkiTensor | None): [1, hidden] fp32 @ HBM, optional. Folds the layer's input
            RMSNorm into the hidden load; safe here, unlike the output norms, because hidden is not sharded.
        norm_eps (float): Epsilon for the input RMSNorm.
        output (nl.NkiTensor | None): [T, q_lora_shard + kv_row_shard] @ HBM, optional. Both shards are
            written side by side into it (query latent first) and nothing is allocated here, so a caller
            that AllGathers the pair as one tensor can pass the collective source and skip a copy.
        hidden_tiles_out (list | None): When given, ``(hidden K-tiles, linear stationary_fn)`` is appended,
            for a caller that projects the same hidden again (the indexer's keys).
        hidden_tiles_sbuf (nl.NkiTensor | None): [P_MAX, hidden / P_MAX * T] @ SBUF, optional: hidden
            already in ``linear``'s K-tile packing. ``hidden_states`` then supplies only shape and dtype,
            and ``input_norm_w`` must be None.

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor]: ``(q_latent_shard [T, q_lora_shard], kv_row_shard
        [T, kv_row_shard])`` @ HBM, both partial (AllGather before use); column views of ``output``
        when it is given.

    Notes:
        Neither output is normalized. Both norms reduce over the sharded axis, so norming here would need
        an AllReduce of partial sums of squares, which costs more than gathering the narrow outputs and
        norming redundantly. The next kernel asserts complete widths, so a missing AllGather fails loudly
        instead of scaling the norm by the shard degree.

        A narrow shard wastes weight DMA: at ``q_lora_shard = 24`` a bf16 row is 48 bytes, under the
        64-byte minimum transaction. That is small here (0.07 MB of 0.295 MB) but is why a very fine
        column shard stops paying off.

        The two matmuls stay separate. Fusing them into one N = 33 pass would save PE time, but both
        are far below the fixed matmul cost, and their total is small.
    """

    q_only = wkv_a is None
    # Without wkv_a only the query columns are checked (the shape helper takes wq_a in both places).
    shapes = get_down_proj_shape(hidden_states, wq_a, wq_a if q_only else wkv_a)
    T = shapes.num_tokens

    # Hidden onto partitions once, then reused by both projections.
    stationary_fn = None
    if hidden_tiles_sbuf is not None:
        kernel_assert(
            input_norm_w is None and tuple(hidden_tiles_sbuf.shape) == (P_MAX, shapes.hidden // P_MAX * T),
            f"[MLA Decode Down Proj] hidden_tiles_sbuf must be [{P_MAX}, {shapes.hidden // P_MAX * T}] with no "
            f"input_norm_w, got {hidden_tiles_sbuf.shape}",
        )
    elif input_norm_w is None and _hidden_spread_fits(T, shapes.hidden):
        # One contiguous partition-spread DMA, then ``T * H / 128**2`` transposes and one eviction instead of
        # ``H / 128`` of each. The transposes are pass-through in the same K order, so the result is bit-identical.
        hidden_tiles_sbuf, stationary_fn = _load_hidden_spread_k_tiles(hidden_states, shapes)
    else:
        hidden_sbuf = _load_hidden(hidden_states, input_norm_w, norm_eps, shapes)
        hidden_tiles_sbuf = transpose_into_k_tiles(hidden_sbuf, T)
    if hidden_tiles_out is not None:
        hidden_tiles_out.append((hidden_tiles_sbuf, stationary_fn))

    q_w, kv_w = shapes.q_lora_shard, 0 if q_only else shapes.kv_row_shard
    if output is not None:
        # Each evicted chain is stored straight to its columns of ``output``: no band hop, no second store.
        kernel_assert(
            tuple(output.shape) == (T, q_w + kv_w) and output.dtype == shapes.dtype,
            f"[MLA Decode Down Proj] output must be [{T}, {q_w + kv_w}] {shapes.dtype}, got {output.shape} {output.dtype}",
        )
        linear(hidden_tiles_sbuf, wq_a, T, stationary_fn=stationary_fn, out_hbm=output, out_col0=0)
        if q_only:
            return output[0:T, 0:q_w], None
        linear(hidden_tiles_sbuf, wkv_a, T, stationary_fn=stationary_fn, out_hbm=output, out_col0=q_w)
        return output[0:T, 0:q_w], output[0:T, q_w : q_w + kv_w]

    kernel_assert(not q_only, "[MLA Decode Down Proj] wkv_a=None needs output")
    q_latent_sbuf = linear(hidden_tiles_sbuf, wq_a, T, stationary_fn=stationary_fn)
    kv_row_sbuf = linear(hidden_tiles_sbuf, wkv_a, T, stationary_fn=stationary_fn)

    q_latent_out = nl.ndarray((T, q_w), dtype=shapes.dtype, buffer=nl.shared_hbm)
    kv_row_out = nl.ndarray((T, kv_w), dtype=shapes.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=q_latent_out[0:T, 0:q_w], src=q_latent_sbuf[0:T, 0:q_w])
    nisa.dma_copy(dst=kv_row_out[0:T, 0:kv_w], src=kv_row_sbuf[0:T, 0:kv_w])

    return q_latent_out, kv_row_out


def _load_hidden(
    hidden_states: nl.NkiTensor,
    input_norm_w: nl.NkiTensor | None,
    eps: float,
    shapes: DownProjShape,
) -> nl.NkiTensor:
    """Loads hidden states to SBUF, applying the layer's input RMSNorm when ``input_norm_w`` is given."""

    T, hidden = shapes.num_tokens, shapes.hidden

    hidden_sbuf = nl.ndarray((P_MAX, hidden), dtype=shapes.dtype, buffer=nl.sbuf)
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

    Partition ``p = t * G + j`` (``G = P_MAX / T`` groups) holds ``hidden[t, j * F : (j + 1) * F]``, so transpose
    ``c`` of the ``F / 128`` column blocks puts k-tile ``kt = j * (F / 128) + c`` of token ``t`` at
    ``tiles[:, c, t * G + j]``: the stationary of ``kt`` is the stride-``G`` column set starting at ``(c, j)``.

    Returns:
        tuple: the ``[P_MAX, F / 128, P_MAX]`` k-tiles and the ``k_index -> [P_MAX, T]`` stationary map.
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
