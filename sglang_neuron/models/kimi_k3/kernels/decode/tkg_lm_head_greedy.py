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

"""Kimi-K3 decode tail: the ``lm_head`` shard of a TP rank and the greedy token of each sequence.

Each of the 32 TP ranks of a DP replica holds the vocab rows ``t * V_s .. (t + 1) * V_s - 1`` of
``lm_head`` (``V_s = 163840 / 32 = 5120``). The tail computes the bf16 logits of the shard, the local
maximum and its first index, and one all-gather of ``(value, index)`` over the TP group. Then each
rank takes the largest value and, among equal values, the smallest vocab id. This is the result of
a greedy sampler over the vocab shards: ``torch.argmax`` takes the first maximum, and a
lower rank holds the lower vocab ids.

The hidden ``x`` is the stationary operand (``B`` columns), and the weight streams as the moving
operand, so the logits come out token-major ``[B, V_s]`` and the TensorE time is about one moving
column per weight column. The weight load (73 MB per rank) sets the time.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from .utils.kernel_assert import kernel_assert
from .utils.kernel_helpers import div_ceil
from .common import P_MAX, PSUM_BANK_FP32
from .tkg_weight_dma_gate import gate_weight_dma
from .tkg_grammar_mask import GrammarMask, check_grammar_mask, grammar_mask_bias, grammar_mask_rows

LM_HEAD_K_TILES_PER_DMA = 2
"""Hidden K-tiles of one ``lm_head`` weight DMA."""
LM_HEAD_BUFFERS = 4
"""SBUF buffers of the ``lm_head`` weight ring."""
LM_HEAD_VOCAB_CHUNK = 5 * PSUM_BANK_FP32
"""Vocab columns of one pass over the hidden: 5 PSUM banks of 512 fp32 columns."""
_NO_TOKEN = float(1 << 22)
"""A vocab id above every real id. fp32 holds the ids and this value exactly."""


def lm_head_greedy_hT(
    x_hT,
    lm_head_weight,
    tp_group,
    tag: str,
    sample: bool = True,
    logits_out=None,
    tp_size: int = 32,
    gather_tiles=None,
    chunk_logits: bool = False,
    grammar: GrammarMask = None,
    grammar_queries: int = 0,
):
    """Return the greedy token of each sequence from the ``lm_head`` shard of this TP rank.

    Args:
        x_hT: SBUF ``[128, B, H/128]`` bf16 normalized hidden (the final norm output).
        lm_head_weight: HBM ``[H, V_s]`` bf16 vocab shard of this TP rank, ``[in, out]`` (the transpose
            of the checkpoint ``lm_head.weight`` rows ``t * V_s ..``).
        tp_group: the TP replica group. The rank order of each group is the vocab shard order.
        tag: unique name prefix.
        sample: compute the greedy tokens.
        logits_out: optional HBM ``[B, V_s]`` bf16 that receives the logits of the shard.
        tp_size: the ranks in each list of ``tp_group``.
        gather_tiles: None, or the SBUF ``[B, 2]`` source and ``[B, 2 tp_size]`` destination of the argmax
            gather (fp32), for example persistent collective tiles of the kernel, and optionally the SBUF
            ``[B, 1]`` int32 tile of the tokens.
        chunk_logits: with ``logits_out`` and no ``sample``, store the logits one vocab chunk at a time.
        grammar: None (default: no grammar mask), or the static grammar mask of the rows
            (``tkg_grammar_mask``): the PSUM -> bf16 pass of each vocab chunk adds the bias row of each row
            (-0.0 allowed, -inf banned), so the logits, their store and the greedy token are those of the masked rows.
        grammar_queries: with ``grammar``, the tokens Q of each sequence (the rows are ``j * (B / Q) + b``).

    Returns:
        SBUF ``[B, 1]`` int32 global vocab id of each sequence in ``[0, tp_size * V_s)``, or None without
        ``sample``. A row with NaN logits also gives an id in the vocab: the last id when no rank matches.

    Example:
        >>> # tokens = lm_head_greedy_hT(h_hT, lm_head_weight, tp_group, "tail")
    """
    _, B, n_tiles = x_hT.shape
    H, V_s = lm_head_weight.shape
    kernel_assert(H == n_tiles * P_MAX, "lm_head_weight must have the hidden size as its first dimension")
    kernel_assert(V_s % P_MAX == 0, f"The vocab shard must be a multiple of {P_MAX}")
    kernel_assert(V_s <= 16384, "max8 reads at most 16384 logits for each sequence")
    kernel_assert(n_tiles % LM_HEAD_K_TILES_PER_DMA == 0, "The hidden tiles must split into whole weight DMAs")

    w_tiles = lm_head_weight.reshape((n_tiles, P_MAX, V_s))
    chunks = [(c0, min(LM_HEAD_VOCAB_CHUNK, V_s - c0)) for c0 in range(0, V_s, LM_HEAD_VOCAB_CHUNK)]
    k_groups = list(range(0, n_tiles, LM_HEAD_K_TILES_PER_DMA))
    loads = [(c0, width, k0) for c0, width in chunks for k0 in k_groups]
    n_bufs = min(LM_HEAD_BUFFERS, len(loads))
    bufs = [
        nl.ndarray((P_MAX, LM_HEAD_K_TILES_PER_DMA, LM_HEAD_VOCAB_CHUNK), dtype=lm_head_weight.dtype, buffer=nl.sbuf)
        for _ in range(n_bufs)
    ]
    engines = (nisa.engine.sync, nisa.engine.scalar)

    def load(i):
        c0, width, k0 = loads[i]
        # The first fill waits for the gate collectives (see tkg_weight_dma_gate), so it uses the SP queue.
        engine = nisa.engine.sync if i < n_bufs else engines[i % len(engines)]
        return nisa.dma_copy(
            dst=bufs[i % n_bufs][0:P_MAX, 0:LM_HEAD_K_TILES_PER_DMA, 0:width],
            src=w_tiles[k0 : k0 + LM_HEAD_K_TILES_PER_DMA, 0:P_MAX, c0 : c0 + width].permute((1, 0, 2)),
            dge_mode=nisa.dge_mode.hwdge,
            engine=engine,
            name=f"{tag}_lm_head_load_{i}",
        )

    for i in range(n_bufs):
        gate_weight_dma(load(i))

    bias_tiles = None
    if grammar is not None:
        # The fold and the bias gathers run on the software DGE queue while the weight stream fills chunk 0.
        check_grammar_mask(grammar, B, grammar_queries, V_s, x_hT.dtype)
        mask_rows = grammar_mask_rows(grammar, grammar_queries, tag)
        bias_tiles = grammar_mask_bias(grammar, mask_rows, chunks, x_hT.dtype, tag)

    def round_bf16(dst, src, chunk: int, col0: int, cols: int):
        """The PSUM -> bf16 rounding of ``cols`` columns from ``col0`` of chunk ``chunk``, with the grammar bias."""
        if bias_tiles is None:
            nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.vector)
        else:
            bias = bias_tiles[chunk][0:B, col0 : col0 + cols]
            nisa.tensor_tensor(dst=dst, data1=src, data2=bias, op=nl.add, engine=nisa.engine.vector)

    chunked = logits_out is not None and not sample and chunk_logits
    logits = None if chunked else nl.ndarray((B, V_s), dtype=x_hT.dtype, buffer=nl.sbuf)
    chunk_bf16 = nl.ndarray((B, LM_HEAD_VOCAB_CHUNK), dtype=x_hT.dtype, buffer=nl.sbuf) if chunked else None
    i = 0
    for chunk, (c0, width) in enumerate(chunks):
        # The last chunk of a shard that is not a multiple of a PSUM bank ends in a partial bank.
        n_banks = div_ceil(width, PSUM_BANK_FP32)
        bank_widths = [min(PSUM_BANK_FP32, width - bank * PSUM_BANK_FP32) for bank in range(n_banks)]
        psum = nl.ndarray((B, n_banks, PSUM_BANK_FP32), dtype=nl.float32, buffer=nl.psum)
        for k0 in k_groups:
            buf = bufs[i % n_bufs]
            for dk in range(LM_HEAD_K_TILES_PER_DMA):
                k = k0 + dk
                for bank in range(n_banks):
                    bw = bank_widths[bank]
                    nisa.nc_matmul(
                        dst=psum[0:B, bank, 0:bw],
                        stationary=x_hT[0:P_MAX, 0:B, k],
                        moving=buf[0:P_MAX, dk, bank * PSUM_BANK_FP32 : bank * PSUM_BANK_FP32 + bw],
                        accumulate=k > 0,
                    )
            if i + n_bufs < len(loads):
                load(i + n_bufs)
            i += 1
        # Round to bf16, as a bf16 lm_head does.
        if chunked:
            round_bf16(chunk_bf16[0:B, 0:width], psum.reshape((B, width)), chunk, 0, width)
            nisa.dma_copy(
                dst=logits_out[0:B, c0 : c0 + width], src=chunk_bf16[0:B, 0:width], name=f"{tag}_logits_store_{c0}"
            )
        elif width % PSUM_BANK_FP32 == 0:
            round_bf16(logits[0:B, c0 : c0 + width], psum.reshape((B, width)), chunk, 0, width)
        else:
            for bank in range(n_banks):
                bw = bank_widths[bank]
                off = bank * PSUM_BANK_FP32
                round_bf16(logits[0:B, c0 + off : c0 + off + bw], psum[0:B, bank, 0:bw], chunk, off, bw)

    if logits_out is not None and not chunked:
        nisa.dma_copy(dst=logits_out, src=logits, name=f"{tag}_logits_store")
    if not sample:
        return None

    # The local maximum and the index of its first occurrence.
    top8 = nl.ndarray((B, 8), dtype=logits.dtype, buffer=nl.sbuf)
    nisa.max8(dst=top8, src=logits)
    idx8 = nl.ndarray((B, 8), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=idx8, data=logits, vals=top8)
    local, gathered, token = (*gather_tiles, None)[:3] if gather_tiles else (None, None, None)
    local = nl.ndarray((B, 2), dtype=nl.float32, buffer=nl.sbuf) if local is None else local
    nisa.tensor_copy(dst=local[0:B, 0:1], src=top8[0:B, 0:1], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=local[0:B, 1:2], src=idx8[0:B, 0:1], engine=nisa.engine.vector)

    # Gather (value, local index) of the 32 ranks along the free axis, in rank order.
    n_ranks = tp_size
    gathered = nl.ndarray((B, n_ranks * 2), dtype=nl.float32, buffer=nl.sbuf) if gathered is None else gathered
    ncc.all_gather(
        srcs=[local], dsts=[gathered], replica_group=tp_group, collective_dim=1, name=f"{tag}_argmax_all_gather"
    )
    if gather_tiles is not None:
        # An engine access of a one-partition view of a persistent tile has an illegal partition step, so copy it.
        gathered_local = nl.ndarray((B, n_ranks * 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gathered_local, src=gathered, name=f"{tag}_argmax_copy")
        gathered = gathered_local
    pairs = gathered.reshape((B, n_ranks, 2))
    values = pairs[0:B, 0:n_ranks, 0]
    # The global id of each candidate: its local index plus the shard start of its rank.
    offsets_i = nl.ndarray((B, n_ranks), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=offsets_i, pattern=[[V_s, n_ranks]], offset=0, channel_multiplier=0)
    ids = nl.ndarray((B, n_ranks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ids, src=offsets_i, engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=ids, data1=ids, data2=pairs[0:B, 0:n_ranks, 1], op=nl.add)

    best = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=best, data=values, op=nl.maximum, axis=(1,))
    # A candidate below the maximum gets an id above every real id, then the minimum id wins.
    is_best = nl.ndarray((B, n_ranks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=is_best, data=values, op0=nl.equal, operand0=best)
    penalty = nl.ndarray((B, n_ranks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=penalty, data=is_best, op0=nl.multiply, operand0=-_NO_TOKEN, op1=nl.add, operand1=_NO_TOKEN)
    nisa.tensor_tensor(dst=ids, data1=ids, data2=penalty, op=nl.add)
    token_f = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=token_f, data=ids, op=nl.minimum, axis=(1,))
    token = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf) if token is None else token
    # A row with no match (NaN logits) has the penalty on every id; the clamp keeps it in the vocab.
    nisa.tensor_scalar(
        dst=token,
        data=token_f,
        op0=nl.minimum,
        operand0=float(n_ranks * V_s - 1),
        op1=nl.maximum,
        operand1=0.0,
        engine=nisa.engine.vector,
    )
    return token
