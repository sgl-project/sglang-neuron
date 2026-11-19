# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's decode sampler in one NKI kernel, its collectives inside: from the states before the final norm to each
row's sampled token, with no XLA op and no logits leaving the device.

    logits  = RMSNorm(x) @ lm_wT          this rank's vocabulary columns, rounded to bf16 as the LM head's output
    pool    = every rank's top 16         per program (max8 / nc_match_replace8), all-gathered: 2048 a row
    cands   = the pool's top 128          sorted descending, by value
    token   = the tail rule over cands    sampling.tail_rule_sample: top_k masked, softmax at the row's temperature,
                                          top_p keeping the crossing candidate, the draw u inverting the kept mass

``u`` is ``sampling.row_draws(sampling.noise_row_keys(seed, position))``, the platform's per-(seed, position) draw,
computed here on int32 lanes (the same 31-bit murmur finalizer), so a seeded request draws the same under any batch
order or MTP acceptance. A temperature below ``GK_TAIL_GREEDY_TEMPERATURE`` takes the best candidate: a greedy row
gets the argmax of its bf16 logits (the lowest pool place among equal values).

The candidate set differs from the XLA ``topk_topp`` sampler's (the global top 256, a Gumbel draw over the filtered
pool): a row's 128 candidates are the best of every rank's 16, which holds a top-p nucleus of any peaked distribution
and caps top_k at 128. The rows (tokens) are the partitions, at most 128 a call; the two programs of LNC2 take one
half of the rank's columns each."""

import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl

from sglang_neuron.models.mimo_v2.kernels.head import _replica_group, _start

P = 128
COLS = 512
K_RANK = 16  # candidates a program sends
K_POOL = 128  # candidates a row samples from
NEG = -3.0e38
TAIL_MASK = -3000.0
GREEDY_TEMPERATURE = 1e-5
LANE = 0x7FFFFFFF
MIX_C1 = 0x05EBCA6B
MIX_C2 = 0x42B2AE35
DRAW_SALT = 0x2545F491


def _mix31(h, T: int):
    """sampling._mix31 on an int32 [128, 1] tile's rows 0 .. T - 1 (lanes in [0, 2**31))."""
    t = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t[0:T, 0:1], data=h[0:T, 0:1], op0=nl.right_shift, operand0=16)
    nisa.tensor_tensor(dst=h[0:T, 0:1], data1=h[0:T, 0:1], data2=t[0:T, 0:1], op=nl.bitwise_xor)
    nisa.tensor_scalar(dst=h[0:T, 0:1], data=h[0:T, 0:1], op0=nl.multiply, operand0=MIX_C1)
    nisa.tensor_scalar(dst=h[0:T, 0:1], data=h[0:T, 0:1], op0=nl.bitwise_and, operand0=LANE)
    nisa.tensor_scalar(dst=t[0:T, 0:1], data=h[0:T, 0:1], op0=nl.right_shift, operand0=13)
    nisa.tensor_tensor(dst=h[0:T, 0:1], data1=h[0:T, 0:1], data2=t[0:T, 0:1], op=nl.bitwise_xor)
    nisa.tensor_scalar(dst=h[0:T, 0:1], data=h[0:T, 0:1], op0=nl.multiply, operand0=MIX_C2)
    nisa.tensor_scalar(dst=h[0:T, 0:1], data=h[0:T, 0:1], op0=nl.bitwise_and, operand0=LANE)
    nisa.tensor_scalar(dst=t[0:T, 0:1], data=h[0:T, 0:1], op0=nl.right_shift, operand0=16)
    nisa.tensor_tensor(dst=h[0:T, 0:1], data1=h[0:T, 0:1], data2=t[0:T, 0:1], op=nl.bitwise_xor)
    return h


def _draws(seeds, positions, T: int):
    """[128, 1] fp32 each row's draw in (0, 1): row_draws(noise_row_keys(seed, position))."""
    k = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=k[0:T, 0:1], data=seeds[0:T, 0:1], op0=nl.bitwise_and, operand0=LANE)
    k = _mix31(k, T)
    p = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p[0:T, 0:1], data=positions[0:T, 0:1], op0=nl.bitwise_and, operand0=LANE)
    nisa.tensor_tensor(dst=k[0:T, 0:1], data1=k[0:T, 0:1], data2=p[0:T, 0:1], op=nl.bitwise_xor)
    k = _mix31(k, T)
    nisa.tensor_scalar(dst=k[0:T, 0:1], data=k[0:T, 0:1], op0=nl.bitwise_and, operand0=LANE, op1=nl.bitwise_xor,
                       operand1=DRAW_SALT)
    k = _mix31(k, T)
    nisa.tensor_scalar(dst=k[0:T, 0:1], data=k[0:T, 0:1], op0=nl.bitwise_and, operand0=0x7FFFFF)
    nisa.tensor_scalar(dst=k[0:T, 0:1], data=k[0:T, 0:1], op0=nl.multiply, operand0=2, op1=nl.add, operand1=1)
    u = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=u[0:T, 0:1], src=k[0:T, 0:1])
    nisa.tensor_scalar(dst=u[0:T, 0:1], data=u[0:T, 0:1], op0=nl.multiply, operand0=1.0 / 16777216.0)
    return u


def _top(work, n: int, k: int, T: int, vals, idx):
    """The top ``k`` (a multiple of 8) of rows 0 .. T - 1 of ``work`` [128, n] fp32, descending: values into ``vals``
    [128, k] and their columns into ``idx`` [128, k] uint32; ``work``'s picked entries are overwritten."""
    for r in range(k // 8):
        nisa.max8(dst=vals[0:T, r * 8:(r + 1) * 8], src=work[0:T, 0:n])
        nisa.nc_match_replace8(dst=work[0:T, 0:n], data=work[0:T, 0:n], vals=vals[0:T, r * 8:(r + 1) * 8], imm=NEG,
                               dst_idx=idx[0:T, r * 8:(r + 1) * 8])


@nki.jit
def sample(x, norm_w, lm_wT, rank, mesh, temperatures, seeds, filters, positions, eps=1e-6, span=1, rows=None):
    """x [T, H] bf16 (T <= 128) the states before the final norm; norm_w [1, H] bf16; lm_wT [H, Vr] bf16 this rank's
    LM head columns; rank [1] int32; per request (T / ``span`` of them, a request's ``span`` rows consecutive):
    temperatures [T / span, 1] fp32, seeds [T / span, 1] int32 (sampling.fold_seeds), filters [T / span, 3] fp32
    (top_k, top_p, min_p normalized; min_p unused); per row positions [T, 1] int32 (the row's input position).
    With ``rows`` [T, 1] int32 the rows are x's and positions' rows at them (a prefill chunk's sampled rows), gathered
    in the kernel. Returns each row's sampled token [T, 1] int32."""
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
    ranks = 1
    for cell in mesh:
        ranks = len(cell)

    # ---- the final RMSNorm (head.greedy's) and this program's half of the shard's logits ----
    hf = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    if rows is None:
        nisa.dma_copy(dst=hf[0:T, 0:H], src=x.ap(pattern=[[H, T], [1, H]], offset=0))
    else:
        ri = nl.ndarray((P, 1), dtype=rows.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=ri[0:T, 0:1], src=rows.ap(pattern=[[1, T], [1, 1]], offset=0))
        nisa.dma_copy(dst=hf[0:T, 0:H], src=x.ap(pattern=[[H, T], [1, H]], offset=0, vector_offset=ri[0:T, 0:1],
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
    logits = nl.ndarray((P, VH), dtype=nl.float32, buffer=nl.sbuf)
    lb = nl.ndarray((P, COLS), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c0 in range(0, VH, COLS):
        cs = min(COLS, VH - c0)
        ps = nl.ndarray((P, COLS), dtype=nl.float32, buffer=nl.psum)
        for h1 in range(H1):
            w = nl.ndarray((P, COLS), dtype=lm_wT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w[0:P, 0:cs], src=lm_wT.ap(pattern=[[Vr, P], [1, cs]], offset=h1 * P * Vr + v0 + c0),
                          dge_mode=nisa.dge_mode.hwdge)
            nisa.nc_matmul(dst=ps[0:T, 0:cs], stationary=nT[0:P, h1, 0:T], moving=w[0:P, 0:cs])
        nisa.tensor_copy(dst=lb[0:T, 0:cs], src=ps[0:T, 0:cs])
        nisa.tensor_copy(dst=logits[0:T, c0:c0 + cs], src=lb[0:T, 0:cs])

    # ---- this program's top K_RANK, with global ids ----
    cv = nl.ndarray((T, K_RANK), dtype=nl.float32, buffer=nl.sbuf)
    ci = nl.ndarray((P, K_RANK), dtype=nl.uint32, buffer=nl.sbuf)
    _top(logits, VH, K_RANK, T, cv, ci)
    cid = nl.ndarray((T, K_RANK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cid[0:T, 0:K_RANK], src=ci[0:T, 0:K_RANK])
    st = _start(rank, T, Vr)
    nisa.tensor_scalar(dst=cid[0:T, 0:K_RANK], data=cid[0:T, 0:K_RANK], op0=nl.add, operand0=st[0:T, 0:1],
                       op1=nl.add, operand1=float(v0))

    # ---- the pool: every rank's candidates, this program's then the other's ----
    n_half = ranks * K_RANK
    pool_v = nl.ndarray((P, n_prg * n_half), dtype=nl.float32, buffer=nl.sbuf)
    pool_i = nl.ndarray((P, n_prg * n_half), dtype=nl.float32, buffer=nl.sbuf)
    gv = nl.ndarray((T, n_half), dtype=nl.float32, buffer=nl.sbuf)
    gi = nl.ndarray((T, n_half), dtype=nl.float32, buffer=nl.sbuf)
    nccl.all_gather(srcs=[cv], dsts=[gv], replica_group=rg, collective_dim=1)
    nccl.all_gather(srcs=[cid], dsts=[gi], replica_group=rg, collective_dim=1)
    nisa.tensor_copy(dst=pool_v[0:T, 0:n_half], src=gv)
    nisa.tensor_copy(dst=pool_i[0:T, 0:n_half], src=gi)
    if n_prg > 1:
        nisa.sendrecv(src=gv, dst=pool_v[0:T, n_half:2 * n_half], send_to_rank=other, recv_from_rank=other, pipe_id=0)
        nisa.sendrecv(src=gi, dst=pool_i[0:T, n_half:2 * n_half], send_to_rank=other, recv_from_rank=other, pipe_id=1)
    n_pool = n_prg * n_half

    # ---- the row's K_POOL best, descending, and their pool places ----
    work = nl.ndarray((P, n_pool), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=work[0:T, 0:n_pool], src=pool_v[0:T, 0:n_pool])
    vals = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    at = nl.ndarray((P, K_POOL), dtype=nl.uint32, buffer=nl.sbuf)
    _top(work, n_pool, K_POOL, T, vals, at)
    at_f = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=at_f[0:T, 0:K_POOL], src=at[0:T, 0:K_POOL])

    # ---- the row's parameters (its request's: request r >> log2(span)) and draw ----
    rq_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(rq_f[0:T, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    rq = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rq[0:T, 0:1], src=rq_f[0:T, 0:1])
    nisa.tensor_scalar(dst=rq[0:T, 0:1], data=rq[0:T, 0:1], op0=nl.right_shift, operand0=(span - 1).bit_length())
    temp = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=temp[0:T, 0:1], src=temperatures.ap(pattern=[[1, T], [1, 1]], offset=0,
                                                          vector_offset=rq[0:T, 0:1], indirect_dim=0))
    flt = nl.ndarray((P, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=flt[0:T, 0:3], src=filters.ap(pattern=[[3, T], [1, 3]], offset=0, vector_offset=rq[0:T, 0:1],
                                                    indirect_dim=0))
    sd = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sd[0:T, 0:1], src=seeds.ap(pattern=[[1, T], [1, 1]], offset=0, vector_offset=rq[0:T, 0:1],
                                                 indirect_dim=0))
    po = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    if rows is None:
        nisa.dma_copy(dst=po[0:T, 0:1], src=positions.ap(pattern=[[1, T], [1, 1]], offset=0))
    else:
        nisa.dma_copy(dst=po[0:T, 0:1], src=positions.ap(pattern=[[1, T], [1, 1]], offset=0, vector_offset=ri[0:T, 0:1],
                                                         indirect_dim=0))
    u = _draws(sd, po, T)

    col = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(col[0:T, 0:K_POOL], pattern=[[1, K_POOL]], offset=0, channel_multiplier=0)
    # the top_k-th value: candidates below it are masked (top_k clamped to [1, K_POOL])
    kth = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=kth[0:T, 0:1], data=flt[0:T, 0:1], op0=nl.minimum, operand0=float(K_POOL), op1=nl.add,
                       operand1=-1.0)
    nisa.tensor_scalar(dst=kth[0:T, 0:1], data=kth[0:T, 0:1], op0=nl.maximum, operand0=0.0)
    oh = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=oh[0:T, 0:K_POOL], data=col[0:T, 0:K_POOL], op0=nl.equal, operand0=kth[0:T, 0:1])
    pick = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=pick[0:T, 0:K_POOL], data1=vals[0:T, 0:K_POOL], data2=oh[0:T, 0:K_POOL], op=nl.multiply)
    thr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(thr[0:T, 0:1], op=nl.add, data=pick[0:T, 0:K_POOL], axis=1, keepdims=True)
    below = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=below[0:T, 0:K_POOL], data=vals[0:T, 0:K_POOL], op0=nl.less, operand0=thr[0:T, 0:1])
    # masked = vals + below * (TAIL_MASK - vals)
    masked = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=masked[0:T, 0:K_POOL], data=vals[0:T, 0:K_POOL], op0=nl.multiply, operand0=-1.0,
                       op1=nl.add, operand1=TAIL_MASK)
    nisa.tensor_tensor(dst=masked[0:T, 0:K_POOL], data1=masked[0:T, 0:K_POOL], data2=below[0:T, 0:K_POOL],
                       op=nl.multiply)
    nisa.tensor_tensor(dst=masked[0:T, 0:K_POOL], data1=masked[0:T, 0:K_POOL], data2=vals[0:T, 0:K_POOL], op=nl.add)
    # 1 / temperature (1 for a greedy row)
    greedy = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=greedy[0:T, 0:1], data=temp[0:T, 0:1], op0=nl.less, operand0=GREEDY_TEMPERATURE)
    t2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t2[0:T, 0:1], data=temp[0:T, 0:1], op0=nl.multiply, operand0=-1.0, op1=nl.add,
                       operand1=1.0)
    nisa.tensor_tensor(dst=t2[0:T, 0:1], data1=t2[0:T, 0:1], data2=greedy[0:T, 0:1], op=nl.multiply)
    nisa.tensor_tensor(dst=t2[0:T, 0:1], data1=t2[0:T, 0:1], data2=temp[0:T, 0:1], op=nl.add)
    inv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv[0:T, 0:1], data=t2[0:T, 0:1])
    # weights = exp((masked - max) / temperature); the max is candidate 0
    nb2 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=nb2[0:T, 0:1], data1=masked[0:T, 0:1], data2=inv[0:T, 0:1], op=nl.multiply)
    nisa.tensor_scalar(dst=nb2[0:T, 0:1], data=nb2[0:T, 0:1], op0=nl.multiply, operand0=-1.0)
    sc = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sc[0:T, 0:K_POOL], data=masked[0:T, 0:K_POOL], op0=nl.multiply, operand0=inv[0:T, 0:1],
                       op1=nl.add, operand1=nb2[0:T, 0:1])
    wts = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=wts[0:T, 0:K_POOL], op=nl.exp, data=sc[0:T, 0:K_POOL])
    # inclusive cumulative sums along the candidates in fp32 on the vector engine (a TensorE matmul against a
    # triangular ones matrix rounds its fp32 operands, which moved draws at the kept mass's boundary)
    def cumsum(src, dst):
        nisa.tensor_scalar_cumulative(dst=dst[0:T, 0:K_POOL], src=src[0:T, 0:K_POOL], op0=nl.multiply, op1=nl.add,
                                      imm0=1.0)

    cum = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    cumsum(wts, cum)
    # top-p: a candidate goes once the mass before it reaches top_p of the total (never candidate 0)
    tot = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot[0:T, 0:1], data1=cum[0:T, K_POOL - 1:K_POOL], data2=flt[0:T, 1:2], op=nl.multiply)
    before = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=before[0:T, 0:K_POOL], data1=cum[0:T, 0:K_POOL], data2=wts[0:T, 0:K_POOL],
                       op=nl.subtract)
    drop = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=drop[0:T, 0:K_POOL], data=before[0:T, 0:K_POOL], op0=nl.greater_equal,
                       operand0=tot[0:T, 0:1])
    notfirst = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=notfirst[0:T, 0:K_POOL], data=col[0:T, 0:K_POOL], op0=nl.greater, operand0=0.0)
    nisa.tensor_tensor(dst=drop[0:T, 0:K_POOL], data1=drop[0:T, 0:K_POOL], data2=notfirst[0:T, 0:K_POOL],
                       op=nl.multiply)
    keep = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=keep[0:T, 0:K_POOL], data=drop[0:T, 0:K_POOL], op0=nl.multiply, operand0=-1.0,
                       op1=nl.add, operand1=1.0)
    kept = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=kept[0:T, 0:K_POOL], data1=wts[0:T, 0:K_POOL], data2=keep[0:T, 0:K_POOL], op=nl.multiply)
    kc = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    cumsum(kept, kc)
    # the draw's place: the kept candidates whose cumulative mass is below u of the kept total
    lim = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=lim[0:T, 0:1], data1=kc[0:T, K_POOL - 1:K_POOL], data2=u[0:T, 0:1], op=nl.multiply)
    under = nl.ndarray((P, K_POOL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=under[0:T, 0:K_POOL], data=kc[0:T, 0:K_POOL], op0=nl.less, operand0=lim[0:T, 0:1])
    place = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(place[0:T, 0:1], op=nl.add, data=under[0:T, 0:K_POOL], axis=1, keepdims=True)
    # a greedy row takes candidate 0: place x (1 - greedy)
    ng = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ng[0:T, 0:1], data=greedy[0:T, 0:1], op0=nl.multiply, operand0=-1.0, op1=nl.add,
                       operand1=1.0)
    nisa.tensor_tensor(dst=place[0:T, 0:1], data1=place[0:T, 0:1], data2=ng[0:T, 0:1], op=nl.multiply)
    nisa.tensor_scalar(dst=place[0:T, 0:1], data=place[0:T, 0:1], op0=nl.minimum, operand0=float(K_POOL - 1))

    # ---- the token: the candidate's pool place, then its id ----
    nisa.tensor_scalar(dst=oh[0:T, 0:K_POOL], data=col[0:T, 0:K_POOL], op0=nl.equal, operand0=place[0:T, 0:1])
    nisa.tensor_tensor(dst=pick[0:T, 0:K_POOL], data1=at_f[0:T, 0:K_POOL], data2=oh[0:T, 0:K_POOL], op=nl.multiply)
    slot = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(slot[0:T, 0:1], op=nl.add, data=pick[0:T, 0:K_POOL], axis=1, keepdims=True)
    pcol = nl.ndarray((P, n_pool), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(pcol[0:T, 0:n_pool], pattern=[[1, n_pool]], offset=0, channel_multiplier=0)
    poh = nl.ndarray((P, n_pool), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=poh[0:T, 0:n_pool], data=pcol[0:T, 0:n_pool], op0=nl.equal, operand0=slot[0:T, 0:1])
    nisa.tensor_tensor(dst=poh[0:T, 0:n_pool], data1=poh[0:T, 0:n_pool], data2=pool_i[0:T, 0:n_pool],
                       op=nl.multiply)
    tok_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(tok_f[0:T, 0:1], op=nl.add, data=poh[0:T, 0:n_pool], axis=1, keepdims=True)
    tok = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok[0:T, 0:1], src=tok_f[0:T, 0:1])
    out = nl.ndarray((T, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="sample_out")
    if n_prg == 1 or pid == 0:
        nisa.dma_copy(dst=out, src=tok[0:T, 0:1])
    return out
