# SPDX-License-Identifier: Apache-2.0
"""A NEURON_MTP step's inputs in one NKI kernel (model.MiMoV2Graph._mtp_decode): what mtp_step.span_inputs,
attention.span_slots, the layers' position masks and the RoPE tables built in XLA, from the device carry and the
host's per-row control.

A row (batch row w, its carry slot ``row_slots[w]``) decodes S tokens; token r = w x S + s is a partition:

    live, fresh       = ctrl[w, LIVE] > 0, ctrl[w, FRESH] > 0 and live
    token             = fresh ? (s == 0 ? ctrl[w, TOKEN] : 0) : carry[slot, s]        0 on a padding row
    base              = fresh ? ctrl[w, BASE] : length[slot]                          0 on a padding row
    count             = fresh ? 1 : S (0 on a padding row);   valid = s < count
    position          = base + s where valid, else base
    full slot         = full_table[w, position / page] x page + position mod page     0 where invalid or unallocated
    sliding slot      = swa_table[w, (position - offset) / page] x page + position mod page, the same way
    full pos          = base                                                          the prior the token reads
    sliding pos       = max(base - offset, 0);  start = min(max(position - offset - window + 1, 0), sliding pos)
    cos / sin         = the RoPE tables of each kind at the position, [d / 2, R] (rope.RotaryTables' channels: the
                        frequencies, then cos 1 / sin 0)
    active mask       = [S, W x heads, S] uint8, key k visible to query q where k <= q: the span's active part, for
                        each head count in ``mask_heads`` (a layer views it as [S, W / kvdp, heads x kvdp, S], the
                        same memory for any kvdp)

Every value is an integer below 2**24, exact in fp32, where the arithmetic runs; the page divisions are shifts."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
CTRL_LIVE, CTRL_FRESH, CTRL_BASE, CTRL_TOKEN = range(4)
TWO_PI = 6.283185307179586
TWO_PI_HI = 6.25
TWO_PI_LO = TWO_PI - TWO_PI_HI
HALF_PI = 1.5707963267948966
PI = 3.141592653589793


def _gather_rows(src, cols: int, idx, R: int, dtype):
    """[128, cols] rows idx[r] of ``src`` [N, cols] (HBM) into SBUF, rows 0 .. R - 1."""
    out = nl.ndarray((P, cols), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=out[0:R, 0:cols],
                  src=src.ap(pattern=[[cols, R], [1, cols]], offset=0, vector_offset=idx[0:R, 0:1], indirect_dim=0))
    return out


def _f(t, R: int):
    """An int32 [128, n] tile's rows as fp32."""
    n = t.shape[1]
    out = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out[0:R, 0:n], src=t[0:R, 0:n])
    return out


def _i(t, R: int):
    n = t.shape[1]
    out = nl.ndarray((P, n), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out[0:R, 0:n], src=t[0:R, 0:n])
    return out


def _slot(table, position_i, frame_i, valid_f, R: int, S: int, page_shift: int):
    """[128, 1] int32 each token's pool slot: table[w, frame >> page_shift] x page + position mod page, 0 where the
    token is invalid, its frame negative or past the table, or the block unallocated (-1)."""
    W, blocks = table.shape
    page = 1 << page_shift
    raw = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=raw[0:R, 0:1], data=frame_i[0:R, 0:1], op0=nl.maximum, operand0=0)
    nisa.tensor_scalar(dst=raw[0:R, 0:1], data=raw[0:R, 0:1], op0=nl.right_shift, operand0=page_shift)
    raw_f = _f(raw, R)
    frame_f = _f(frame_i, R)
    inside = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=inside[0:R, 0:1], data=raw_f[0:R, 0:1], op0=nl.less, operand0=float(blocks))
    nonneg = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nonneg[0:R, 0:1], data=frame_f[0:R, 0:1], op0=nl.greater_equal, operand0=0.0)
    col_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=col_f[0:R, 0:1], data=raw_f[0:R, 0:1], op0=nl.minimum, operand0=float(blocks - 1))
    # element w x blocks + col of the flattened table
    w_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(w_f[0:R, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    w_i = _i(w_f, R)
    nisa.tensor_scalar(dst=w_i[0:R, 0:1], data=w_i[0:R, 0:1], op0=nl.right_shift, operand0=(S - 1).bit_length())
    w_f = _f(w_i, R)
    nisa.tensor_scalar(dst=w_f[0:R, 0:1], data=w_f[0:R, 0:1], op0=nl.multiply, operand0=float(blocks))
    nisa.tensor_tensor(dst=w_f[0:R, 0:1], data1=w_f[0:R, 0:1], data2=col_f[0:R, 0:1], op=nl.add)
    e_i = _i(w_f, R)
    blk = nl.ndarray((P, 1), dtype=table.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=blk[0:R, 0:1],
                  src=table.reshape((W * blocks, 1)).ap(pattern=[[1, R], [1, 1]], offset=0,
                                                         vector_offset=e_i[0:R, 0:1], indirect_dim=0))
    blk_f = _f(blk, R)
    allocated = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=allocated[0:R, 0:1], data=blk_f[0:R, 0:1], op0=nl.greater_equal, operand0=0.0)
    low = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=low[0:R, 0:1], data=position_i[0:R, 0:1], op0=nl.bitwise_and, operand0=page - 1)
    low_f = _f(low, R)
    slot = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=slot[0:R, 0:1], data=blk_f[0:R, 0:1], op0=nl.multiply, operand0=float(page))
    nisa.tensor_tensor(dst=slot[0:R, 0:1], data1=slot[0:R, 0:1], data2=low_f[0:R, 0:1], op=nl.add)
    for gate in (valid_f, inside, nonneg, allocated):
        nisa.tensor_tensor(dst=slot[0:R, 0:1], data1=slot[0:R, 0:1], data2=gate[0:R, 0:1], op=nl.multiply)
    return _i(slot, R)


def _active_mask(S: int, N: int, name: str):
    """[S, N, S] uint8 HBM: element (k, n, q) = 1 where k <= q. Partition k, free n x S + q."""
    F = N * S
    k_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(k_f[0:S, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    idx_f = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(idx_f[0:S, 0:F], pattern=[[1, F]], offset=0, channel_multiplier=0)
    q_i = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_i[0:S, 0:F], src=idx_f[0:S, 0:F])
    nisa.tensor_scalar(dst=q_i[0:S, 0:F], data=q_i[0:S, 0:F], op0=nl.bitwise_and, operand0=S - 1)
    q_f = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_f[0:S, 0:F], src=q_i[0:S, 0:F])
    m_f = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m_f[0:S, 0:F], data=q_f[0:S, 0:F], op0=nl.greater_equal, operand0=k_f[0:S, 0:1])
    m_u = nl.ndarray((P, F), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_copy(dst=m_u[0:S, 0:F], src=m_f[0:S, 0:F])
    out = nl.ndarray((S, N, S), dtype=nl.uint8, buffer=nl.shared_hbm, name=f"step_mask_{name}")
    nisa.dma_copy(dst=out.reshape((S, F)), src=m_u[0:S, 0:F])
    return out


def _rope(position_f, inv_freq, R: int, half: int, name: str):
    """(cos, sin) [half, R] bf16 HBM tables of the positions: rope.RotaryTables' channels (its frequencies, then cos 1
    and sin 0) transposed to the decode kernels' [d / 2, B, S] layout."""
    F = inv_freq.shape[1]
    inv = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=inv[0:R, 0:F], src=inv_freq.ap(pattern=[[0, R], [1, F]], offset=0))
    ang = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ang[0:R, 0:F], data=inv[0:R, 0:F], op0=nl.multiply, operand0=position_f[0:R, 0:1])
    # the angle reduced to about [-pi, pi]: x - k 2pi, k an integer near x / 2pi, 2pi split Cody-Waite style so
    # k x 6.25 is exact in fp32 (k < 2**21); the scalar engine's sin is then accurate, and cos = sin(x + pi / 2)
    k_f = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=k_f[0:R, 0:F], data=ang[0:R, 0:F], op0=nl.multiply, operand0=1.0 / TWO_PI)
    k_i = nl.ndarray((P, F), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=k_i[0:R, 0:F], src=k_f[0:R, 0:F])
    nisa.tensor_copy(dst=k_f[0:R, 0:F], src=k_i[0:R, 0:F])
    red = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=red[0:R, 0:F], data=k_f[0:R, 0:F], op0=nl.multiply, operand0=-TWO_PI_HI,
                              op1=nl.add, operand1=ang[0:R, 0:F])
    nisa.scalar_tensor_tensor(dst=red[0:R, 0:F], data=k_f[0:R, 0:F], op0=nl.multiply, operand0=-TWO_PI_LO,
                              op1=nl.add, operand1=red[0:R, 0:F])
    outs = []
    for which, shift, rest in (("cos", HALF_PI, 1.0), ("sin", 0.0, 0.0)):
        full = nl.ndarray((P, half), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=full[0:R, 0:half], value=rest)
        arg = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=arg[0:R, 0:F], data=red[0:R, 0:F], op0=nl.add, operand0=shift)
        if shift:
            # back into [-pi, pi], where the scalar engine's sin is exact: arg - 2pi where arg > pi
            over = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=over[0:R, 0:F], data=arg[0:R, 0:F], op0=nl.greater, operand0=PI)
            nisa.scalar_tensor_tensor(dst=arg[0:R, 0:F], data=over[0:R, 0:F], op0=nl.multiply, operand0=-TWO_PI,
                                      op1=nl.add, operand1=arg[0:R, 0:F])
        val = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=val[0:R, 0:F], op=nl.sin, data=arg[0:R, 0:F])
        nisa.tensor_copy(dst=full[0:R, 0:F], src=val[0:R, 0:F])
        tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp[0:half, 0:R], data=full[0:R, 0:half])
        t_sb = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=t_sb[0:half, 0:R], src=tp[0:half, 0:R])
        out = nl.ndarray((half, R), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=f"step_{name}_{which}")
        nisa.dma_copy(dst=out, src=t_sb[0:half, 0:R])
        outs.append(out)
    return outs


@nki.jit
def step_inputs(carry, length, ctrl, row_slots, full_table, swa_table, swa_offset, full_inv_freq, swa_inv_freq,
                page_shift=7, window=128, half=128, mask_heads=()):
    """carry [N, S] int32; length [N] int32; ctrl [W, 4] int32; row_slots [W] int32; full_table [W, Bf] /
    swa_table [W, Bs] int32 (-1 past a row's pages); swa_offset [W] int32; the inverse frequencies [1, F] fp32. Returns
    int32 [R, 1] each of (tokens, positions, full slots, sliding slots, valid, fresh, live, base), fp32 [R, 1] each of
    (full pos, sliding pos, sliding start), bf16 [half, R] (full cos, full sin, sliding cos, sliding sin), then a uint8 [S, W x heads, S] active mask for each
    of ``mask_heads``."""
    N, S = carry.shape
    W = ctrl.shape[0]
    R = W * S
    sh = (S - 1).bit_length()  # S a power of two

    # token r: batch row w = r >> sh, span place s = r & (S - 1)
    r_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(r_f[0:R, 0:1], pattern=[[0, 1]], offset=0, channel_multiplier=1)
    r_i = _i(r_f, R)
    w_i = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=w_i[0:R, 0:1], data=r_i[0:R, 0:1], op0=nl.right_shift, operand0=sh)
    s_i = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=s_i[0:R, 0:1], data=r_i[0:R, 0:1], op0=nl.bitwise_and, operand0=S - 1)
    s_f = _f(s_i, R)

    c = _gather_rows(ctrl, 4, w_i, R, ctrl.dtype)
    slot_of_row = _gather_rows(row_slots.reshape((W, 1)), 1, w_i, R, row_slots.dtype)
    off = _gather_rows(swa_offset.reshape((W, 1)), 1, w_i, R, swa_offset.dtype)
    ln = _gather_rows(length.reshape((N, 1)), 1, slot_of_row, R, length.dtype)
    # carry element slot x S + s
    ce = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ce[0:R, 0:1], data=slot_of_row[0:R, 0:1], op0=nl.left_shift, operand0=sh)
    nisa.tensor_tensor(dst=ce[0:R, 0:1], data1=ce[0:R, 0:1], data2=s_i[0:R, 0:1], op=nl.add)
    cv = _gather_rows(carry.reshape((N * S, 1)), 1, ce, R, carry.dtype)

    cf = _f(c, R)
    live = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=live[0:R, 0:1], data=cf[0:R, CTRL_LIVE:CTRL_LIVE + 1], op0=nl.greater, operand0=0.0)
    fresh = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=fresh[0:R, 0:1], data=cf[0:R, CTRL_FRESH:CTRL_FRESH + 1], op0=nl.greater, operand0=0.0)
    nisa.tensor_tensor(dst=fresh[0:R, 0:1], data1=fresh[0:R, 0:1], data2=live[0:R, 0:1], op=nl.multiply)
    nfresh = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nfresh[0:R, 0:1], data=fresh[0:R, 0:1], op0=nl.multiply, operand0=-1.0, op1=nl.add,
                       operand1=1.0)
    first = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=first[0:R, 0:1], data=s_f[0:R, 0:1], op0=nl.equal, operand0=0.0)

    def blend(on_fresh, on_carry):
        # live x (fresh x on_fresh + (1 - fresh) x on_carry)
        out = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        tmp = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=out[0:R, 0:1], data1=on_fresh[0:R, 0:1], data2=fresh[0:R, 0:1], op=nl.multiply)
        nisa.tensor_tensor(dst=tmp[0:R, 0:1], data1=on_carry[0:R, 0:1], data2=nfresh[0:R, 0:1], op=nl.multiply)
        nisa.tensor_tensor(dst=out[0:R, 0:1], data1=out[0:R, 0:1], data2=tmp[0:R, 0:1], op=nl.add)
        nisa.tensor_tensor(dst=out[0:R, 0:1], data1=out[0:R, 0:1], data2=live[0:R, 0:1], op=nl.multiply)
        return out

    host_tok = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=host_tok[0:R, 0:1], data1=cf[0:R, CTRL_TOKEN:CTRL_TOKEN + 1], data2=first[0:R, 0:1],
                       op=nl.multiply)
    tok = blend(host_tok, _f(cv, R))
    base = blend(cf[0:R, CTRL_BASE:CTRL_BASE + 1], _f(ln, R))
    span_n = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=span_n[0:R, 0:1], value=float(S))
    one = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=one[0:R, 0:1], value=1.0)
    count = blend(one, span_n)
    valid = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=valid[0:R, 0:1], data1=s_f[0:R, 0:1], data2=count[0:R, 0:1], op=nl.less)
    pos = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=pos[0:R, 0:1], data1=s_f[0:R, 0:1], data2=valid[0:R, 0:1], op=nl.multiply)
    nisa.tensor_tensor(dst=pos[0:R, 0:1], data1=pos[0:R, 0:1], data2=base[0:R, 0:1], op=nl.add)
    pos_i = _i(pos, R)

    full_slot = _slot(full_table, pos_i, pos_i, valid, R, S, page_shift)
    off_f = _f(off, R)
    frame = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=frame[0:R, 0:1], data1=pos[0:R, 0:1], data2=off_f[0:R, 0:1], op=nl.subtract)
    swa_slot = _slot(swa_table, pos_i, _i(frame, R), valid, R, S, page_shift)

    # the sliding layers' prior end and window start in their trimmed table's frame
    send = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=send[0:R, 0:1], data1=base[0:R, 0:1], data2=off_f[0:R, 0:1], op=nl.subtract)
    nisa.tensor_scalar(dst=send[0:R, 0:1], data=send[0:R, 0:1], op0=nl.maximum, operand0=0.0)
    sstart = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sstart[0:R, 0:1], data=frame[0:R, 0:1], op0=nl.add, operand0=float(1 - window),
                       op1=nl.maximum, operand1=0.0)
    nisa.tensor_tensor(dst=sstart[0:R, 0:1], data1=sstart[0:R, 0:1], data2=send[0:R, 0:1], op=nl.minimum)

    def out_i(t, name):
        o = nl.ndarray((R, 1), dtype=nl.int32, buffer=nl.shared_hbm, name=f"step_{name}")
        nisa.dma_copy(dst=o, src=_i(t, R)[0:R, 0:1] if t.dtype != nl.int32 else t[0:R, 0:1])
        return o

    def out_f(t, name):
        o = nl.ndarray((R, 1), dtype=nl.float32, buffer=nl.shared_hbm, name=f"step_{name}")
        nisa.dma_copy(dst=o, src=t[0:R, 0:1])
        return o

    fc, fs = _rope(pos, full_inv_freq, R, half, "full")
    sc, ss = _rope(pos, swa_inv_freq, R, half, "swa")
    return (out_i(tok, "tokens"), out_i(pos_i, "positions"), out_i(full_slot, "full_slots"),
            out_i(swa_slot, "swa_slots"), out_i(valid, "valid"), out_i(fresh, "fresh"), out_i(live, "live"),
            out_i(base, "base"), out_f(base, "full_pos"), out_f(send, "swa_pos"), out_f(sstart, "swa_start"),
            fc, fs, sc, ss, *(_active_mask(S, W * h, str(h)) for h in mask_heads))
