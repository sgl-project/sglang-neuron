# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's decode step head and tail in NKI kernels, their collectives inside: the token embedding and the greedy
LM head, with no XLA op.

``embed``: each rank holds ``Vr`` consecutive vocabulary rows (layers.embedding.VocabDimShardedEmbedding's shard). A
token's row is gathered by an indirect DMA from the rank's shard; a token outside it is sent past the shard's end, which
the DMA skips (``oob_mode.skip``), so its row stays zero; the sum over the attention-TP ranks is the embedding. Each of
the two programs gathers and all-reduces its own half of H (SBUF to SBUF) and stores it.

``greedy``: the final RMSNorm (fp32 math, bf16 result times the weight, as models/transformer/model.RMSNorm), this
rank's ``Vr`` vocabulary columns against the transposed LM head ``[H, Vr]`` (bf16 x bf16, fp32 accumulation; each
logit rounded to bf16 as the LM head's bf16 output is), the rank's best by ``max8`` / ``nc_find_index8``, the two
programs' halves of the shard crossed by sendrecv, and the ranks' bests all-gathered: the token of the largest logit,
the lowest id among equal ones (torch.argmax's order: program 0's columns and lower ranks come first).

The rows (tokens) are the partitions, at most 128 a call."""

import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode

P = 128
COLS = 512


def _replica_group(mesh):
    cells = []
    for cell in mesh:  # NKI's tracer takes no list comprehension
        cells.append(list(cell))
    return nccl.ReplicaGroup(cells)


def _start(rank, T: int, Vr: int):
    """[128, 1] fp32 this rank's first vocabulary id on rows 0 .. T - 1, from ``rank`` [1] int32 (the attention-TP
    group's rank tensor, a graph input, so every rank traces one graph)."""
    r_i = nl.ndarray((P, 1), dtype=rank.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=r_i[0:T, 0:1], src=rank.ap(pattern=[[0, T], [1, 1]], offset=0))
    st = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=st[0:T, 0:1], src=r_i[0:T, 0:1])
    nisa.tensor_scalar(dst=st[0:T, 0:1], data=st[0:T, 0:1], op0=nl.multiply, operand0=float(Vr))
    return st


@nki.jit
def embed(ids, weight, rank, mesh):
    """ids [T, 1] int32 (T <= 128); weight [Vr, H] bf16, this rank's vocabulary rows; rank [1] int32, this rank's
    place in the group. Returns the embeddings [T, H] bf16."""
    T = ids.shape[0]
    Vr, H = weight.shape
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    HH = H // n_prg
    rg = _replica_group(mesh)

    # the local row of each id, or Vr (past the shard: skipped) for an id another rank holds
    ids_i = nl.ndarray((P, 1), dtype=ids.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ids_i[0:T, 0:1], src=ids.ap(pattern=[[1, T], [1, 1]], offset=0))
    ids_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ids_f[0:T, 0:1], src=ids_i[0:T, 0:1])
    st = _start(rank, T, Vr)
    local = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=local[0:T, 0:1], data1=ids_f[0:T, 0:1], data2=st[0:T, 0:1], op=nl.subtract)
    lo = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lo[0:T, 0:1], data=local[0:T, 0:1], op0=nl.greater_equal, operand0=0.0)
    hi = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hi[0:T, 0:1], data=local[0:T, 0:1], op0=nl.less, operand0=float(Vr))
    ok = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ok[0:T, 0:1], data1=lo[0:T, 0:1], data2=hi[0:T, 0:1], op=nl.multiply)
    # (local - Vr) x ok + Vr: local where it is in the shard, Vr elsewhere
    row_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=row_f[0:T, 0:1], data=local[0:T, 0:1], op0=nl.subtract, operand0=float(Vr))
    nisa.tensor_tensor(dst=row_f[0:T, 0:1], data1=row_f[0:T, 0:1], data2=ok[0:T, 0:1], op=nl.multiply)
    nisa.tensor_scalar(dst=row_f[0:T, 0:1], data=row_f[0:T, 0:1], op0=nl.add, operand0=float(Vr))
    row = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=row[0:T, 0:1], src=row_f[0:T, 0:1])

    # this program's half of H of each row, zero where the DMA skips it
    rows = nl.ndarray((T, HH), dtype=weight.dtype, buffer=nl.sbuf)
    nisa.memset(dst=rows, value=0.0)
    nisa.dma_copy(dst=rows,
                  src=weight.ap(pattern=[[H, T], [1, HH]], offset=pid * HH, vector_offset=row[0:T, 0:1],
                                indirect_dim=0),
                  oob_mode=oob_mode.skip)
    red = nl.ndarray((T, HH), dtype=weight.dtype, buffer=nl.sbuf)
    nccl.all_reduce(dsts=[red], srcs=[rows], op=nl.add, replica_group=rg)
    out = nl.ndarray((T, H), dtype=weight.dtype, buffer=nl.shared_hbm, name="embed_out")
    nisa.dma_copy(dst=out.ap(pattern=[[H, T], [1, HH]], offset=pid * HH), src=red)
    return out


@nki.jit
def greedy(x, norm_w, lm_wT, rank, mesh, eps=1e-6, rows=None):
    """x [T, H] bf16 (T <= 128), the states before the final norm; norm_w [1, H] bf16; lm_wT [H, Vr] bf16, this
    rank's LM head columns; rank [1] int32, this rank's place in the group. Returns the greedy token of each row [T, 1]
    int32."""
    H = x.shape[1]
    T = x.shape[0] if rows is None else rows.shape[0]
    Vr = lm_wT.shape[1]
    H1 = H // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    other = 1 - pid
    VH = Vr // n_prg
    v0 = pid * VH
    rg = _replica_group(mesh)

    # ---- the final RMSNorm, as moe_half's; with ``rows`` [T, 1] int32 the states are x's rows at them ----
    hf = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    if rows is None:
        nisa.dma_copy(dst=hf[0:T, 0:H], src=x.ap(pattern=[[H, T], [1, H]], offset=0))
    else:
        ridx = nl.ndarray((P, 1), dtype=rows.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=ridx[0:T, 0:1], src=rows)
        nisa.dma_copy(dst=hf[0:T, 0:H], src=x.ap(pattern=[[H, T], [1, H]], offset=0, vector_offset=ridx[0:T, 0:1],
                                                 indirect_dim=0))
    gam = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=gam[0:T, 0:H], src=norm_w.ap(pattern=[[0, T], [1, H]], offset=0))
    sq = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq[0:T, 0:H], op=nl.square, data=hf[0:T, 0:H], reduce_op=nl.add,
                    reduce_res=ss[0:T, 0:1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rr[0:T, 0:1], op=nl.rsqrt, data=ss[0:T, 0:1], scale=1.0 / H, bias=eps)
    nb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nb[0:T, 0:H], data=hf[0:T, 0:H], op0=nl.multiply, operand0=rr[0:T, 0:1])
    nrm = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(nrm[0:T, 0:H], nb[0:T, 0:H], gam[0:T, 0:H], nl.multiply)
    nT = nl.ndarray((P, H1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h1 in range(H1):
        tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp[0:P, 0:T], data=nrm[0:T, h1 * P:(h1 + 1) * P])
        nisa.tensor_copy(dst=nT[0:P, h1, 0:T], src=tp[0:P, 0:T])

    # ---- this program's half of the shard's logits, rounded to bf16, back in fp32 ----
    logits = nl.ndarray((P, VH), dtype=nl.float32, buffer=nl.sbuf)
    lb = nl.ndarray((P, COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c0 in range(0, VH, COLS):
        cs = min(COLS, VH - c0)
        ps = nl.ndarray((P, COLS), dtype=nl.float32, buffer=nl.psum)
        for h1 in range(H1):
            w = nl.ndarray((P, COLS), dtype=lm_wT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w[0:P, 0:cs], src=lm_wT.ap(pattern=[[Vr, P], [1, cs]], offset=h1 * P * Vr + v0 + c0), dge_mode=nisa.dge_mode.hwdge)
            nisa.nc_matmul(dst=ps[0:T, 0:cs], stationary=nT[0:P, h1, 0:T], moving=w[0:P, 0:cs])
        nisa.tensor_copy(dst=lb[0:T, 0:cs], src=ps[0:T, 0:cs])
        nisa.tensor_copy(dst=logits[0:T, c0:c0 + cs], src=lb[0:T, 0:cs])

    # ---- this program's best: value and id (the first of equal maxima) ----
    top = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=top[0:T, 0:8], src=logits[0:T, 0:VH])
    at = nl.ndarray((P, 8), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=at[0:T, 0:8], data=logits[0:T, 0:VH], vals=top[0:T, 0:8])
    st = _start(rank, T, Vr)
    best = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)  # (value, global id)
    nisa.tensor_copy(dst=best[0:T, 0:1], src=top[0:T, 0:1])
    nisa.tensor_copy(dst=best[0:T, 1:2], src=at[0:T, 0:1])
    nisa.tensor_scalar(dst=best[0:T, 1:2], data=best[0:T, 1:2], op0=nl.add, operand0=float(v0))
    nisa.tensor_tensor(dst=best[0:T, 1:2], data1=best[0:T, 1:2], data2=st[0:T, 0:1], op=nl.add)

    # ---- the core's best: program 0's columns come first ----
    core = nl.ndarray((T, 2), dtype=nl.float32, buffer=nl.sbuf)
    if n_prg > 1:
        theirs = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(src=best[0:T, 0:2], dst=theirs[0:T, 0:2], send_to_rank=other, recv_from_rank=other, pipe_id=0)
        # p0 = program 0's (value, id), p1 = program 1's
        p0 = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        p1 = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        if pid == 0:
            nisa.tensor_copy(dst=p0[0:T, 0:2], src=best[0:T, 0:2])
            nisa.tensor_copy(dst=p1[0:T, 0:2], src=theirs[0:T, 0:2])
        else:
            nisa.tensor_copy(dst=p0[0:T, 0:2], src=theirs[0:T, 0:2])
            nisa.tensor_copy(dst=p1[0:T, 0:2], src=best[0:T, 0:2])
        # program 1 wins only on a strictly larger value
        win = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=win[0:T, 0:1], data1=p1[0:T, 0:1], data2=p0[0:T, 0:1], op=nl.greater)
        diff = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=diff[0:T, 0:2], data1=p1[0:T, 0:2], data2=p0[0:T, 0:2], op=nl.subtract)
        nisa.tensor_scalar(dst=diff[0:T, 0:2], data=diff[0:T, 0:2], op0=nl.multiply, operand0=win[0:T, 0:1])
        nisa.tensor_tensor(dst=core, data1=p0[0:T, 0:2], data2=diff[0:T, 0:2], op=nl.add)
    else:
        nisa.tensor_copy(dst=core, src=best[0:T, 0:2])

    # ---- the ranks' bests: the largest value, the lowest rank among equal ones ----
    ranks = 1
    for cell in mesh:
        ranks = len(cell)
    every = nl.ndarray((T, 2 * ranks), dtype=nl.float32, buffer=nl.sbuf)
    nccl.all_gather(srcs=[core], dsts=[every], replica_group=rg, collective_dim=1)
    vals = nl.ndarray((P, ranks), dtype=nl.float32, buffer=nl.sbuf)
    ids = nl.ndarray((P, ranks), dtype=nl.float32, buffer=nl.sbuf)
    for r in range(ranks):
        nisa.tensor_copy(dst=vals[0:T, r:r + 1], src=every[0:T, 2 * r:2 * r + 1])
        nisa.tensor_copy(dst=ids[0:T, r:r + 1], src=every[0:T, 2 * r + 1:2 * r + 2])
    gtop = nl.ndarray((P, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=gtop[0:T, 0:8], src=vals[0:T, 0:ranks])
    gat = nl.ndarray((P, 8), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=gat[0:T, 0:8], data=vals[0:T, 0:ranks], vals=gtop[0:T, 0:8])
    gat_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gat_f[0:T, 0:1], src=gat[0:T, 0:1])
    # the winning rank's id: a one-hot over the ranks, summed
    rid = nl.ndarray((P, ranks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(rid[0:T, 0:ranks], pattern=[[1, ranks]], offset=0, channel_multiplier=0)
    oh = nl.ndarray((P, ranks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=oh[0:T, 0:ranks], data=rid[0:T, 0:ranks], op0=nl.equal, operand0=gat_f[0:T, 0:1])
    pick = nl.ndarray((P, ranks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=pick[0:T, 0:ranks], data1=ids[0:T, 0:ranks], data2=oh[0:T, 0:ranks], op=nl.multiply)
    tok_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(tok_f[0:T, 0:1], op=nl.add, data=pick[0:T, 0:ranks], axis=1, keepdims=True)
    tok = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok[0:T, 0:1], src=tok_f[0:T, 0:1])
    out = nl.ndarray((T, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="greedy_out")
    if n_prg == 1 or pid == 0:
        nisa.dma_copy(dst=out, src=tok[0:T, 0:1])
    return out
