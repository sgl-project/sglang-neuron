# SPDX-License-Identifier: Apache-2.0
"""A NEURON_MTP step's verify, draft rotation and carry commit in NKI kernels (model.MiMoV2Graph._mtp_decode): what
mtp_step.accept, mtp.rotate, the drafts' row gathers and the carry's index_copy_ did in XLA.

Rows w (partitions) hold their span's S tokens in the free dimension:

    ``accept``  kept = the drafts kept: the run of s with tokens[s + 1] == sampled[s], s + 1 valid, the row not fresh;
                out[s] = sampled[s] for s <= kept, -1 after (and on a dead row), padded with -1 rows to the decode
                width; bonus = sampled[kept]; kept_row = w x S + kept, the span token the drafts read
    ``rotate``  each row's tokens one position ahead: tokens[s + 1] before ``at``, ``token`` at ``at``, the rest kept
    ``commit``  carry[slot] = [bonus, drafts], length[slot] = base + kept + 1, written in place at each row's slot

Token ids and counts are integers below 2**24, exact in the fp32 the arithmetic runs in."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


def _load_grid(src, W: int, S: int, dtype):
    """[128, S] rows 0 .. W - 1 of ``src`` [W x S, 1], as fp32."""
    raw = nl.ndarray((P, S), dtype=src.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=raw[0:W, 0:S], src=src.reshape((W, S)))
    out = nl.ndarray((P, S), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out[0:W, 0:S], src=raw[0:W, 0:S])
    return out


@nki.jit
def accept(tokens, sampled, valid, fresh, live, span=4, out_width=0):
    """tokens / sampled / valid / fresh / live [W x S, 1] int32 (step.step_inputs' and the target's draws, row-major,
    S = ``span``). Returns (out [max(W, out_width), S] int32, kept [W, 1], bonus [W, 1], kept_row [W, 1]) int32."""
    R = tokens.shape[0]
    S = span
    W = R // S
    OW = max(W, out_width)
    tok = _load_grid(tokens, W, S, nl.float32)
    smp = _load_grid(sampled, W, S, nl.float32)
    val = _load_grid(valid, W, S, nl.float32)
    frs = _load_grid(fresh, W, S, nl.float32)
    liv = _load_grid(live, W, S, nl.float32)
    # match[s] = tokens[s + 1] == sampled[s] and valid[s + 1] and not fresh
    match = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=match[0:W, 0:S - 1], data1=tok[0:W, 1:S], data2=smp[0:W, 0:S - 1], op=nl.equal)
    nisa.tensor_tensor(dst=match[0:W, 0:S - 1], data1=match[0:W, 0:S - 1], data2=val[0:W, 1:S], op=nl.multiply)
    nf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nf[0:W, 0:1], data=frs[0:W, 0:1], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
    nisa.tensor_scalar(dst=match[0:W, 0:S - 1], data=match[0:W, 0:S - 1], op0=nl.multiply, operand0=nf[0:W, 0:1])
    # kept = the leading run of matches: sum of their running product
    run = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar_cumulative(dst=run[0:W, 0:S - 1], src=match[0:W, 0:S - 1], op0=nl.multiply, op1=nl.multiply,
                                  imm0=1.0)
    kept = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(kept[0:W, 0:1], op=nl.add, data=run[0:W, 0:S - 1], axis=1, keepdims=True)
    col = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(col[0:W, 0:S], pattern=[[1, S]], offset=0, channel_multiplier=0)
    # out = sampled where s <= kept and the row is live, else -1: (sampled + 1) x keep - 1
    keep = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=keep[0:W, 0:S], data=col[0:W, 0:S], op0=nl.less_equal, operand0=kept[0:W, 0:1])
    nisa.tensor_scalar(dst=keep[0:W, 0:S], data=keep[0:W, 0:S], op0=nl.multiply, operand0=liv[0:W, 0:1])
    o_f = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o_f[0:W, 0:S], data=smp[0:W, 0:S], op0=nl.add, operand0=1.0)
    nisa.tensor_tensor(dst=o_f[0:W, 0:S], data1=o_f[0:W, 0:S], data2=keep[0:W, 0:S], op=nl.multiply)
    nisa.tensor_scalar(dst=o_f[0:W, 0:S], data=o_f[0:W, 0:S], op0=nl.add, operand0=-1.0)
    oh = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=oh[0:W, 0:S], data=col[0:W, 0:S], op0=nl.equal, operand0=kept[0:W, 0:1])
    nisa.tensor_tensor(dst=oh[0:W, 0:S], data1=oh[0:W, 0:S], data2=smp[0:W, 0:S], op=nl.multiply)
    bonus = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(bonus[0:W, 0:1], op=nl.add, data=oh[0:W, 0:S], axis=1, keepdims=True)
    krow = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(krow[0:W, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=S)
    nisa.tensor_tensor(dst=krow[0:W, 0:1], data1=krow[0:W, 0:1], data2=kept[0:W, 0:1], op=nl.add)

    out = nl.ndarray((OW, S), dtype=nl.int32, buffer=nl.shared_hbm, name="accept_out")
    o_i = nl.ndarray((P, S), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=o_i[0:OW, 0:S], value=-1)
    nisa.tensor_copy(dst=o_i[0:W, 0:S], src=o_f[0:W, 0:S])
    nisa.dma_copy(dst=out, src=o_i[0:OW, 0:S])
    res = []
    for t, name in ((kept, "kept"), (bonus, "bonus"), (krow, "kept_row")):
        t_i = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=t_i[0:W, 0:1], src=t[0:W, 0:1])
        o = nl.ndarray((W, 1), dtype=nl.int32, buffer=nl.shared_hbm, name=f"accept_{name}")
        nisa.dma_copy(dst=o, src=t_i[0:W, 0:1])
        res.append(o)
    return out, res[0], res[1], res[2]


@nki.jit
def rotate(tokens, at, token):
    """tokens [W x S, 1] int32 (row-major), at / token [W, 1] int32. Returns the rotated tokens [W x S, 1] int32
    (mtp.rotate)."""
    R = tokens.shape[0]
    W = at.shape[0]
    S = R // W
    tok = _load_grid(tokens, W, S, nl.float32)
    a = _load_grid(at, W, 1, nl.float32)
    t = _load_grid(token, W, 1, nl.float32)
    col = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(col[0:W, 0:S], pattern=[[1, S]], offset=0, channel_multiplier=0)
    ahead = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ahead[0:W, 0:S - 1], src=tok[0:W, 1:S])
    nisa.tensor_copy(dst=ahead[0:W, S - 1:S], src=tok[0:W, S - 1:S])
    before = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=before[0:W, 0:S], data=col[0:W, 0:S], op0=nl.less, operand0=a[0:W, 0:1])
    at_m = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=at_m[0:W, 0:S], data=col[0:W, 0:S], op0=nl.equal, operand0=a[0:W, 0:1])
    # out = tok + before x (ahead - tok) + at x (token - tok)
    d = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=d[0:W, 0:S], data1=ahead[0:W, 0:S], data2=tok[0:W, 0:S], op=nl.subtract)
    nisa.tensor_tensor(dst=d[0:W, 0:S], data1=d[0:W, 0:S], data2=before[0:W, 0:S], op=nl.multiply)
    o = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o[0:W, 0:S], data1=tok[0:W, 0:S], data2=d[0:W, 0:S], op=nl.add)
    e = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e[0:W, 0:S], data=tok[0:W, 0:S], op0=nl.multiply, operand0=-1.0, op1=nl.add,
                       operand1=t[0:W, 0:1])
    nisa.tensor_tensor(dst=e[0:W, 0:S], data1=e[0:W, 0:S], data2=at_m[0:W, 0:S], op=nl.multiply)
    nisa.tensor_tensor(dst=o[0:W, 0:S], data1=o[0:W, 0:S], data2=e[0:W, 0:S], op=nl.add)
    o_i = nl.ndarray((P, S), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o_i[0:W, 0:S], src=o[0:W, 0:S])
    out = nl.ndarray((R, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="rotate_out")
    nisa.dma_copy(dst=out.reshape((W, S)), src=o_i[0:W, 0:S])
    return out


@nki.jit
def commit(carry, length, row_slots, bonus, d1, d2, d3, base, kept, out):
    """In place: carry[row_slots[w]] = [bonus[w], drafts[w]] and length[row_slots[w]] = base[w] + kept[w] + 1.
    carry [N, S] / length [N] int32; row_slots [W] int32; bonus / kept [W, 1] int32, base [W, 1] or [W x S', 1] (each token's) int32; d1 .. d3 [W, 1] int32 the three drafts.
    Returns a copy of ``out`` (accept's [OW, S] token ids): the step's result comes out of this call, so the graph
    keeps it, and with it the in-place writes (the caller's graph sees no other use of them)."""
    N, S = carry.shape
    W = row_slots.shape[0]
    OW = out.shape[0]
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    idx = nl.ndarray((P, 1), dtype=row_slots.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx[0:W, 0:1], src=row_slots.reshape((W, 1)))
    row = nl.ndarray((P, S), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=row[0:W, 0:1], src=bonus)
    nisa.dma_copy(dst=row[0:W, 1:2], src=d1)
    nisa.dma_copy(dst=row[0:W, 2:3], src=d2)
    nisa.dma_copy(dst=row[0:W, 3:4], src=d3)
    b = _load_grid(base, W, base.shape[0] // W, nl.float32)  # base per row, or per token (its first)
    k = _load_grid(kept, W, 1, nl.float32)
    nisa.tensor_tensor(dst=b[0:W, 0:1], data1=b[0:W, 0:1], data2=k[0:W, 0:1], op=nl.add)
    nisa.tensor_scalar(dst=b[0:W, 0:1], data=b[0:W, 0:1], op0=nl.add, operand0=1.0)
    ln = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ln[0:W, 0:1], src=b[0:W, 0:1])
    o_sb = nl.ndarray((P, S), dtype=out.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=o_sb[0:OW, 0:S], src=out)
    result = nl.ndarray((OW, S), dtype=out.dtype, buffer=nl.shared_hbm, name="commit_out")
    if n_prg == 1 or pid == 0:
        nisa.dma_copy(dst=carry.ap(pattern=[[S, W], [1, S]], offset=0, vector_offset=idx[0:W, 0:1], indirect_dim=0),
                      src=row[0:W, 0:S])
        nisa.dma_copy(dst=length.reshape((N, 1)).ap(pattern=[[1, W], [1, 1]], offset=0, vector_offset=idx[0:W, 0:1],
                                                    indirect_dim=0), src=ln[0:W, 0:1])
        nisa.dma_copy(dst=result, src=o_sb[0:OW, 0:S])
    return result
