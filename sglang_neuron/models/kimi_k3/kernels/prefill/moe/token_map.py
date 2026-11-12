# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 prefill MoE token map: scatter the token ids into their block slots.

The blockwise MoE kernel (bwmm) reads ``token_position_to_id``: for each slot
of each expert-major block, the id of the token in the slot. The caller
computes the slot of each routing pick (a prefix count and a block offset).
This kernel writes the token id of each pick to its slot:

    out[slots[t, j]] = t     for each pick j of token t with slots[t, j] < S

A pick of a remote expert has the slot ``S`` (out of bounds), and the DMA
skips it. The slots of the valid picks are unique. The kernel does not write
the other slots (the block pad slots), so their values are not defined: the
caller sets them to the sink id.

The kernel writes each column of picks of a 128-token tile with one
indirect DMA (one row offset for each partition). The native lowering of
``torch.Tensor.scatter_`` issues the same writes as element-wise software
DMAs on GpSimd, a large cost for each MoE layer at T=4096 and k=16 on trn3.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode

P = 128


def _names():
    """Return a name factory that is unique in one kernel trace.

    A name depends only on the order of the buffers in the trace. It does not
    depend on the earlier traces of the process. Thus two processes that
    trace the same call get the same IR, also when one of them gets the
    trace from the NKI trace cache.
    """
    prefix = "ktm"
    count = [0]

    def name(tag):
        count[0] += 1
        return f"{prefix}_{tag}_{count[0]}"

    return name


@nki.jit
def kimi_k3_moe_token_map(slots, num_slots: int):
    """Scatter the token id of each routing pick to its block slot.

    Dimensions:
        T: Token count, a multiple of 128.
        k: Picks of each token (the router top-k).
        S: Slot count of the token map (``num_slots``).

    Args:
        slots (nl.NkiTensor): [T, k] int32 HBM slot of each pick, in
            ``[0, S]``. The value ``S`` marks a pick that has no slot.
        num_slots (int): The slot count ``S``.

    Returns:
        out (nl.NkiTensor): [S, 1] int32 HBM token map. ``out[s] = t`` for
        the pick of token ``t`` in slot ``s``. A slot without a pick is not
        written.

    Example:
        >>> # out = kimi_k3_moe_token_map(slots, num_slots=72448)
    """
    T, k = slots.shape
    assert T % P == 0, f"T ({T}) must be a multiple of {P}"  # noqa: S101
    nm = _names()
    out = nl.ndarray((num_slots, 1), dtype=nl.int32, buffer=nl.shared_hbm, name=nm("out"))
    for tile in range(T // P):
        t0 = tile * P
        sl = nl.ndarray((P, k), dtype=nl.int32, buffer=nl.sbuf, name=nm("slots"))
        nisa.dma_copy(dst=sl[0:P, 0:k], src=slots[t0 : t0 + P, 0:k])
        pos = nl.ndarray((P, k), dtype=nl.uint32, buffer=nl.sbuf, name=nm("pos"))
        nisa.tensor_copy(dst=pos[0:P, 0:k], src=sl[0:P, 0:k])
        # Token ids t0 + p on the partitions.
        tok_row = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf, name=nm("tok_row"))
        nisa.iota(dst=tok_row, pattern=[[1, P]], offset=t0)
        tok_ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum, name=nm("tok_ps"))
        nisa.nc_transpose(dst=tok_ps, data=tok_row)
        tok = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf, name=nm("tok"))
        nisa.tensor_copy(dst=tok, src=tok_ps)
        for j in range(k):
            nisa.dma_copy(
                dst=out.ap(
                    pattern=[[num_slots, P], [1, 1]],
                    offset=0,
                    vector_offset=pos[0:P, j : j + 1],
                    indirect_dim=0,
                ),
                src=tok[0:P, 0:1],
                oob_mode=oob_mode.skip,
            )
    return out


@nki.jit
def kimi_k3_moe_token_runs(sel, start, num_slots: int):
    """Write the token ids of each local expert to its run of slots.

    The picks of expert ``e`` fill the slots ``start[e] ..
    start[e] + n_e - 1`` in token order, where ``n_e`` is the pick count of
    the expert. The kernel writes each run in chunks of 128 slots, from
    ``start[e]``. For each chunk, it builds a 0/1 match of the picks of each
    128-token tile against the 128 slots of the chunk, and sums the token
    ids on TensorE. Then it writes the chunk with one contiguous 512 B DMA.

    No two writes overlap: the chunks of one expert are disjoint, and the
    last chunk ends at ``start[e] + 128 * ceil(n_e / 128)``. That is at most
    ``start[e + 1]``, because each run starts at a multiple of 128 (the
    block size is a multiple of 128). A chunk slot after the last pick gets
    0. The chunk loop of each expert is a dynamic loop with
    ``ceil(n_e / 128)`` trips, so the work follows the pick count.

    Dimensions:
        T: Token count, a multiple of 128, at most 128 * 128.
        E: Local expert count, at most 32.
        S: Slot count of the token map (``num_slots``).

    Args:
        sel (nl.NkiTensor): [T, E] bf16 HBM, 1.0 if token ``t`` picks local
            expert ``e``, else 0.0.
        start (nl.NkiTensor): [1, E] fp32 HBM slot of the first pick of each
            expert: a multiple of 128, in expert order, with
            ``start[e] + 128 * ceil(n_e / 128) <= start[e + 1]``.
        num_slots (int): The slot count ``S``. The last chunk of the last
            expert must end at most at ``S``.

    Returns:
        out (nl.NkiTensor): [S] int32 HBM token map. ``out[start[e] + r]``
        is the token of the ``r``-th pick of expert ``e``. The other slots
        are not defined.

    Example:
        >>> # out = kimi_k3_moe_token_runs(sel, start, num_slots=72448)
    """
    T, E = sel.shape
    assert T % P == 0, f"T ({T}) must be a multiple of {P}"  # noqa: S101
    assert num_slots >= P, f"num_slots ({num_slots}) must be at least {P}"  # noqa: S101
    NT = T // P
    assert NT <= P, f"T ({T}) must be at most {P * P}"  # noqa: S101
    assert E <= 32, f"E ({E}) must be at most 32 (one trip register each)"  # noqa: S101
    nm = _names()
    bf16, f32, i32 = nl.bfloat16, nl.float32, nl.int32
    out = nl.ndarray((num_slots,), dtype=i32, buffer=nl.shared_hbm, name=nm("out"))
    # tri[p', p] = 1 if p' <= p: tri^T @ sel gives the inclusive pick count
    # of each token in its tile. ones^T @ sel gives the tile count of each
    # expert on every partition.
    diff = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf, name=nm("diff"))
    nisa.iota(dst=diff, pattern=[[1, P]], offset=0, channel_multiplier=-1)
    tri = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf, name=nm("tri"))
    nisa.tensor_scalar(dst=tri, data=diff, op0=nl.greater_equal, operand0=0.0)
    ones = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf, name=nm("ones"))
    nisa.memset(dst=ones, value=1.0)
    # col[p, q] = q, the chunk slot of each column.
    col = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf, name=nm("col"))
    nisa.iota(dst=col, pattern=[[1, P]], offset=0)
    # who[p, i, 0] = p and who[p, i, 1] = i: the token t = p + 128 * i of
    # row p of tile i. Both are exact in bf16.
    who = nl.ndarray((P, NT, 2), dtype=bf16, buffer=nl.sbuf, name=nm("who"))
    nisa.iota(dst=who[0:P, 0:NT, 0:1], pattern=[[0, NT], [0, 1]], offset=0, channel_multiplier=1)
    nisa.iota(dst=who[0:P, 0:NT, 1:2], pattern=[[1, NT], [0, 1]], offset=0)
    # comb = [1, 128]: comb^T @ [p-sum; i-sum] gives the token id.
    comb = nl.ndarray((2, 1), dtype=bf16, buffer=nl.sbuf, name=nm("comb"))
    nisa.iota(dst=comb, pattern=[[0, 1]], offset=1, channel_multiplier=P - 1)

    # Pass 1: the rank of each pick in its expert run (-1 for no pick).
    rank = nl.ndarray((P, NT, E), dtype=f32, buffer=nl.sbuf, name=nm("rank"))
    before = nl.ndarray((P, E), dtype=f32, buffer=nl.sbuf, name=nm("before"))
    nisa.memset(dst=before, value=0.0)
    for tile in range(NT):
        t0 = tile * P
        s_sb = nl.ndarray((P, E), dtype=bf16, buffer=nl.sbuf, name=nm("sel"))
        nisa.dma_copy(dst=s_sb, src=sel[t0 : t0 + P, 0:E])
        c_ps = nl.ndarray((P, E), dtype=f32, buffer=nl.psum, name=nm("count"))
        nisa.nc_matmul(dst=c_ps, stationary=tri, moving=s_sb)
        n_ps = nl.ndarray((P, E), dtype=f32, buffer=nl.psum, name=nm("n"))
        nisa.nc_matmul(dst=n_ps, stationary=ones, moving=s_sb)
        # rank = sel * (count + before) - 1.
        r = rank[0:P, tile, 0:E]
        nisa.tensor_tensor(dst=r, data1=c_ps, data2=before, op=nl.add)
        nisa.tensor_tensor(dst=r, data1=r, data2=s_sb, op=nl.multiply)
        nisa.tensor_scalar(dst=r, data=r, op0=nl.add, operand0=-1.0)
        nisa.tensor_tensor(dst=before, data1=before, data2=n_ps, op=nl.add)

    # The chunk count ceil(n_e / 128) and the start slot of each expert.
    chunks = nl.ndarray((1, E), dtype=i32, buffer=nl.sbuf, name=nm("chunks"))
    nisa.tensor_scalar(dst=chunks, data=before[0:1, 0:E], op0=nl.add, operand0=float(P - 1))
    nisa.tensor_scalar(dst=chunks, data=chunks, op0=nl.right_shift, operand0=7)
    first = nl.ndarray((1, E), dtype=f32, buffer=nl.sbuf, name=nm("first"))
    nisa.dma_copy(dst=first, src=start[0:1, 0:E])
    # One trip register for each expert, loaded before the first loop.
    trips = []
    for e in range(E):
        reg = nisa.register_alloc()
        nisa.register_load(reg, chunks[0:1, e : e + 1])
        trips.append(reg)

    # Pass 2: write the chunks of each expert.
    for e in range(E):
        shift = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf, name=nm("shift"))
        nisa.memset(dst=shift, value=0.0)
        off = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf, name=nm("off"))
        nisa.tensor_copy(dst=off, src=first[0:1, e : e + 1])

        def chunk(_, e=e, shift=shift, off=off):
            sums = nl.ndarray((2, P), dtype=f32, buffer=nl.psum, name=nm("sums"))
            for tile in range(NT):
                # match[p, q] = 1 if row p of the tile is pick shift + q.
                match = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf, name=nm("match"))
                nisa.tensor_scalar(dst=match, data=col, op0=nl.add, operand0=shift[0:P, 0:1],
                                   op1=nl.equal, operand1=rank[0:P, tile, e : e + 1])
                nisa.nc_matmul(dst=sums, stationary=who[0:P, tile, 0:2], moving=match,
                               accumulate=tile > 0)
            parts = nl.ndarray((2, P), dtype=bf16, buffer=nl.sbuf, name=nm("parts"))
            nisa.tensor_copy(dst=parts, src=sums)
            ids_ps = nl.ndarray((1, P), dtype=f32, buffer=nl.psum, name=nm("ids_ps"))
            nisa.nc_matmul(dst=ids_ps, stationary=comb, moving=parts)
            ids = nl.ndarray((1, P), dtype=i32, buffer=nl.sbuf, name=nm("ids"))
            nisa.tensor_copy(dst=ids, src=ids_ps)
            # Keep the write in the output for any input. A valid input
            # never needs the clamp. The shape inference of the CPU mode
            # runs the kernel on ones, which is not a valid input.
            dst = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf, name=nm("dst"))
            nisa.tensor_scalar(dst=dst, data=off, op0=nl.minimum, operand0=num_slots - P,
                               op1=nl.maximum, operand1=0)
            nisa.dma_copy(
                dst=out.ap(pattern=[[P, 1], [1, P]], offset=0, scalar_offset=dst[0:1, 0:1], indirect_dim=0),
                src=ids[0:1, 0:P],
                dge_mode=nisa.dge_mode.swdge,
            )
            nisa.tensor_scalar(dst=shift, data=shift, op0=nl.add, operand0=float(P))
            nisa.tensor_scalar(dst=off, data=off, op0=nl.add, operand0=P)

        nl.fori_loop(0, trips[e], chunk)
    return out
