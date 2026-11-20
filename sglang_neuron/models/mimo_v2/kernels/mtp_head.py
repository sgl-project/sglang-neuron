# SPDX-License-Identifier: Apache-2.0
"""An MTP layer's input in one NKI kernel (mtp.MiMoMtp.project), its collectives inside:

    e   = embed(next_ids)                     the vocabulary-sharded rows, summed over the ranks
    x   = eh_proj([enorm(e) | hnorm(prev)])   [T, 2H] against this rank's eh_proj rows, all-gathered over the ranks

Each of the two programs holds one half of H of both inputs: it gathers and all-reduces its half of the embedding
rows (as head.embed), takes the RMSNorms' sums of squares over the whole row by exchanging its half's partial sum with
the other program (sendrecv), normalizes its halves, and contracts them against its half of the transposed weight's
rows ``eh_wT [2H, H / ranks]``; the two programs' partials are summed by sendrecv, and the ranks' output columns are
all-gathered into [T, H]. The RMSNorms round as models/transformer/model.RMSNorm (fp32 math, bf16 result, times the
weight)."""

import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode

from sglang_neuron.models.mimo_v2.kernels.head import _replica_group, _start

P = 128


def _half_rms(x_half, w_half, T: int, HH: int, H: int, eps, pid, other, pipe: int):
    """This program's half [T, HH] of an RMSNorm over rows of H, from its half of the row and of the weight; the sum of
    squares crosses the two programs."""
    sq = nl.ndarray((P, HH), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq[0:T, 0:HH], op=nl.square, data=x_half[0:T, 0:HH], reduce_op=nl.add,
                    reduce_res=ss[0:T, 0:1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
    theirs = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=ss[0:T, 0:1], dst=theirs[0:T, 0:1], send_to_rank=other, recv_from_rank=other, pipe_id=pipe)
    tot = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot[0:T, 0:1], data1=ss[0:T, 0:1], data2=theirs[0:T, 0:1], op=nl.add)
    rr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rr[0:T, 0:1], op=nl.rsqrt, data=tot[0:T, 0:1], scale=1.0 / H, bias=eps)
    nb = nl.ndarray((P, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nb[0:T, 0:HH], data=x_half[0:T, 0:HH], op0=nl.multiply, operand0=rr[0:T, 0:1])
    out = nl.ndarray((P, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(out[0:T, 0:HH], nb[0:T, 0:HH], w_half[0:T, 0:HH], nl.multiply)
    return out


@nki.jit
def mtp_project(ids, prev, emb_w, enorm_w, hnorm_w, eh_wT, rank, mesh, eps=1e-6):
    """ids [T, 1] int32 (T <= 128), the next tokens; prev [T, H] bf16, the states they read; emb_w [Vr, H] bf16,
    this rank's embedding rows; enorm_w / hnorm_w [1, H] bf16; eh_wT [2H, H / ranks] bf16, this rank's eh_proj rows
    transposed; rank [1] int32. Returns the MTP layer's input [T, H] bf16. Needs the two programs of LNC2."""
    T = ids.shape[0]
    Vr, H = emb_w.shape
    OR = eh_wT.shape[1]  # this rank's output columns
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    other = 1 - pid
    HH = H // n_prg
    h0 = pid * HH
    rg = _replica_group(mesh)

    # ---- this program's half of the embedding rows (head.embed) ----
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
    row_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=row_f[0:T, 0:1], data=local[0:T, 0:1], op0=nl.subtract, operand0=float(Vr))
    nisa.tensor_tensor(dst=row_f[0:T, 0:1], data1=row_f[0:T, 0:1], data2=ok[0:T, 0:1], op=nl.multiply)
    nisa.tensor_scalar(dst=row_f[0:T, 0:1], data=row_f[0:T, 0:1], op0=nl.add, operand0=float(Vr))
    row = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=row[0:T, 0:1], src=row_f[0:T, 0:1])
    rows = nl.ndarray((T, HH), dtype=emb_w.dtype, buffer=nl.sbuf)
    nisa.memset(dst=rows, value=0.0)
    nisa.dma_copy(dst=rows, src=emb_w.ap(pattern=[[H, T], [1, HH]], offset=h0, vector_offset=row[0:T, 0:1],
                                         indirect_dim=0), oob_mode=oob_mode.skip)
    e_half = nl.ndarray((T, HH), dtype=emb_w.dtype, buffer=nl.sbuf)
    nccl.all_reduce(dsts=[e_half], srcs=[rows], op=nl.add, replica_group=rg)

    # ---- the two RMSNorms over this program's halves ----
    p_half = nl.ndarray((P, HH), dtype=prev.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_half[0:T, 0:HH], src=prev.ap(pattern=[[H, T], [1, HH]], offset=h0))
    ew = nl.ndarray((P, HH), dtype=enorm_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ew[0:T, 0:HH], src=enorm_w.ap(pattern=[[0, T], [1, HH]], offset=h0))
    hw = nl.ndarray((P, HH), dtype=hnorm_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=hw[0:T, 0:HH], src=hnorm_w.ap(pattern=[[0, T], [1, HH]], offset=h0))
    e_n = _half_rms(e_half, ew, T, HH, H, eps, pid, other, 0)
    p_n = _half_rms(p_half, hw, T, HH, H, eps, pid, other, 1)

    # ---- [enorm(e) | hnorm(prev)] against eh_wT: this program's rows h0 .. h0 + HH of each part ----
    K1 = HH // P
    ps = nl.ndarray((P, OR), dtype=nl.float32, buffer=nl.psum)
    for part in range(2):
        src = e_n if part == 0 else p_n
        for k in range(K1):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:T], data=src[0:T, k * P:(k + 1) * P])
            xT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xT[0:P, 0:T], src=tp[0:P, 0:T])
            w = nl.ndarray((P, OR), dtype=eh_wT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w[0:P, 0:OR],
                          src=eh_wT.ap(pattern=[[OR, P], [1, OR]], offset=(part * H + h0 + k * P) * OR))
            nisa.nc_matmul(dst=ps[0:T, 0:OR], stationary=xT[0:P, 0:T], moving=w[0:P, 0:OR])
    mine = nl.ndarray((P, OR), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mine[0:T, 0:OR], src=ps[0:T, 0:OR])
    theirs = nl.ndarray((P, OR), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=mine[0:T, 0:OR], dst=theirs[0:T, 0:OR], send_to_rank=other, recv_from_rank=other, pipe_id=2)
    col = nl.ndarray((T, OR), dtype=prev.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=col, data1=mine[0:T, 0:OR], data2=theirs[0:T, 0:OR], op=nl.add)

    # ---- the ranks' columns: [T, H] ----
    full = nl.ndarray((T, H), dtype=prev.dtype, buffer=nl.sbuf)
    nccl.all_gather(srcs=[col], dsts=[full], replica_group=rg, collective_dim=1)
    out = nl.ndarray((T, H), dtype=prev.dtype, buffer=nl.shared_hbm, name="mtp_project_out")
    nisa.dma_copy(dst=out.ap(pattern=[[H, T], [1, HH]], offset=h0), src=full[0:T, h0:h0 + HH])
    return out
