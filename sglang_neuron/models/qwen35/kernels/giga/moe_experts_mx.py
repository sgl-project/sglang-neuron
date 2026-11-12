"""All-local-expert MX MoE for one EP rank, token-stationary on both projections (NKI, LNC=1).

    out[t, :] = sum_e aff[t, e] * down_e( silu(x_t W_gate_e) * (x_t W_up_e) )

Why a new kernel: nkilib's MX all-expert path keeps the gate/up WEIGHTS stationary, so
every 128x512 weight tile goes through LDWEIGHTS_MX (a fixed cost each for MXFP8, ~1000 per
layer at Qwen3.5-397B's EP64 shard), which makes the layer PE-bound well above its DMA
time. Here the quantized tokens are the stationary (8 loads per expert) and the weights
stream through the array as the moving operand (up to 4096 columns per instruction), for
gate/up as well as down.

Weight / scale layouts are nkilib's MX MoE layouts (so its generators and torch reference
apply unchanged):
  gate_up_w     [E, 128, 2, H/512, I]   x4-packed (fp8 or fp4); element h = j*(H/4) + tile*128 + p
  gate_up_scale [E, 16, 2, H/512, I]    uint8 E8M0, 4 rows per SBUF quadrant
  down_w        [E, 128, I/512, H]      x4-packed; element i = tile*512 + j*128 + p
  down_scale    [E, 16, I/512, H]
"""

import os

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

P = 128
Q = 4  # x4 packing / H quarters
_SCALE_ROWS = 4  # scale rows per SBUF quadrant


def _dma(dst, src, mode="auto"):
    if mode == "auto":
        nisa.dma_copy(dst=dst, src=src)
    elif mode.startswith("hwdge"):
        eng = nisa.engine.scalar if mode.endswith("act") else nisa.engine.sync
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=eng)
    else:
        nisa.dma_copy(dst=dst, src=src, dge_mode=getattr(nisa.dge_mode, mode))


def _load_scale(dst, src_rows, mode="auto"):
    """[16, ...] HBM scale rows -> SBUF [128, ...] with rows 4q..4q+3 at partitions 32q..32q+3."""
    for q in range(4):
        _dma(dst[32 * q : 32 * q + _SCALE_ROWS], src_rows[q * _SCALE_ROWS : (q + 1) * _SCALE_ROWS], mode)


# Trace-time FIFO of prefetched expert weight tiles (one entry per upcoming moe_experts_mx call),
# filled by prefetch_expert_weights() earlier in the layer so the expert weight stream overlaps
# the token mixer and the collectives - the GPT-OSS giga kernel's gate/up prefetch, generalised.
_PREFETCH = []
# Trace-time flag: the code being traced is the body of a device loop (multi-step decode). Partly
# written scale tiles must then be memset first (a tile read before it is fully written in an
# iteration counts as loop-carried, and loop-carried tensors must be defined outside the loop).
LOOP_BODY = [False]
# Inside a loop body the chunked loader can instead take its scale tiles from a pool allocated and
# zeroed ONCE outside the loop (open_scale_pool), reused round-robin: no per-chunk memsets (~24 KB per
# partition per expert), at the price of the pool's permanent SBUF and WAR waits between reuses.
_SCALE_POOL = {}  # shape -> [tiles, next index]


def open_scale_pool(gate_up_w, down_w, k_gu, k_dn):
    """Allocate + zero k_gu gate/up and k_dn down scale-chunk tiles (call outside the device loop)."""
    I, n_i = gate_up_w.shape[4], down_w.shape[2]
    _SCALE_POOL.clear()
    for shape, k in (((P, 2, I), k_gu), ((P, n_i, _OCH), k_dn)):
        if k <= 0:
            continue
        tiles = []
        for _ in range(k):
            t = nl.ndarray(shape, dtype=nl.uint8, buffer=nl.sbuf)
            nisa.memset(t, 0)
            tiles.append(t)
        _SCALE_POOL[shape] = [tiles, 0]


def close_scale_pool():
    _SCALE_POOL.clear()


def _scale_tile(shape, zero):
    """Scale tile for one weight chunk: next pool tile if a pool of this shape is open, else a fresh tile
    (memset first when ``zero``)."""
    pool = _SCALE_POOL.get(tuple(shape))
    if pool is not None:
        t = pool[0][pool[1] % len(pool[0])]
        pool[1] += 1
        return t
    t = nl.ndarray(shape, dtype=nl.uint8, buffer=nl.sbuf)
    if zero:
        nisa.memset(t, 0)
    return t
# DMA queue for prefetched weights: a ring separate from the token mixer's loads, so the prefetch
# runs beside them (same-queue DMAs execute in issue order). auto | swdge | hwdge_act | hwdge_sync
_PREFETCH_DGE = os.environ.get("QWEN35_MOE_PREFETCH_DGE", "hwdge_act")
# Queue for the fused block's early expert DMAs (issue_expert_weights): on the default queue they
# sit in front of the token mixer's own weight loads, which then wait for the whole prefetch.
_PRE_DGE = os.environ.get("QWEN35_MOE_PRE_DGE", "auto")
# Activation-quantisation transposes: evacuate every other PSUM tile on ACT instead of DVE. The prefill expert
# passes (128 gathered rows each) are DVE-copy bound with ACT idle on semaphores: on by default there
# (experts_body(act_evac=True); a shorter 8K TP16 stack), decode keeps DVE only. 0 / 1 force it everywhere.
_SWZ_SPLIT = os.environ.get("QWEN35_MOE_SWZ_SPLIT", "")


def _expert_tiles(gate_up_w, gate_up_scale, down_w, down_scale, e, mode="auto", zero_scales=False):
    """Allocate SBUF tiles for expert ``e`` and issue its weight + scale DMAs on queue ``mode``.

    zero_scales: memset the scale tiles first (the DMAs fill 16 of 128 rows; inside a loop body a
    partly written tile would otherwise count as loop-carried)."""
    zero_scales = zero_scales or LOOP_BODY[0]
    n_h, I = gate_up_w.shape[3], gate_up_w.shape[4]
    n_i, H = down_w.shape[2], down_w.shape[3]
    wgu = nl.ndarray((P, 2, n_h, I), dtype=gate_up_w.dtype, buffer=nl.sbuf)
    _dma(wgu, gate_up_w[e], mode)
    wgus = nl.ndarray((P, 2, n_h, I), dtype=nl.uint8, buffer=nl.sbuf)
    if zero_scales:
        nisa.memset(wgus, 0)
    _load_scale(wgus, gate_up_scale[e], mode)
    wd = nl.ndarray((P, n_i, H), dtype=down_w.dtype, buffer=nl.sbuf)
    _dma(wd, down_w[e], mode)
    wds = nl.ndarray((P, n_i, H), dtype=nl.uint8, buffer=nl.sbuf)
    if zero_scales:
        nisa.memset(wds, 0)
    _load_scale(wds, down_scale[e], mode)
    return wgu, wgus, wd, wds


# Chunked weight streaming: load each expert as per-H-tile gate/up chunks (~1 MB) and per-OCH down
# chunks (~2 MB) instead of four ~4-8 MB tensors, so the allocator can recycle small buffers and
# expert e+1's first chunks land while expert e's matmuls run (a whole MXFP8 expert is ~12 MB of
# the 24 MB SBUF, which otherwise leaves the load -> compute chain near-serial).
_CHUNK = os.environ.get("QWEN35_MOE_CHUNK", "1") == "1"
_OCH = 2048  # down-projection output columns per PSUM tile (bf16: 2 banks)


def _expert_chunks(gate_up_w, gate_up_scale, down_w, down_scale, e, mode="auto", zero_scales=False):
    """Per-chunk SBUF tiles for expert ``e``: ([(wgu_th, wgus_th)] * n_h, [(wd_c, wds_c)] * H/OCH)."""
    zero_scales = zero_scales or LOOP_BODY[0]
    n_h, I = gate_up_w.shape[3], gate_up_w.shape[4]
    n_i, H = down_w.shape[2], down_w.shape[3]
    gu = []
    for th in range(n_h):
        w = nl.ndarray((P, 2, I), dtype=gate_up_w.dtype, buffer=nl.sbuf)
        _dma(w, gate_up_w[e, :, :, th, :], mode)
        s = _scale_tile((P, 2, I), zero_scales)
        _load_scale(s, gate_up_scale[e, :, :, th, :], mode)
        gu.append((w, s))
    dn = []
    och = min(_OCH, H)  # H < 2048 only in small test shapes
    for c0 in range(0, H, och):
        w = nl.ndarray((P, n_i, och), dtype=down_w.dtype, buffer=nl.sbuf)
        _dma(w, down_w[e, :, :, c0 : c0 + och], mode)
        s = _scale_tile((P, n_i, och), zero_scales)
        _load_scale(s, down_scale[e, :, :, c0 : c0 + och], mode)
        dn.append((w, s))
    return gu, dn


def issue_expert_weights(gate_up_w, gate_up_scale, down_w, down_scale, n):
    """Start the weight DMAs of the first ``n`` local experts now (chunked or whole, per QWEN35_MOE_CHUNK).

    Returns {expert: tiles} for experts_body(pre=...); issuing these before the collectives / router
    lets the expert weight stream run while the DMA queues would otherwise be idle.
    """
    load = _expert_chunks if _CHUNK else _expert_tiles
    return {
        e: load(gate_up_w, gate_up_scale, down_w, down_scale, e, _PRE_DGE) for e in range(min(n, gate_up_w.shape[0]))
    }


def prefetch_expert_weights(gate_up_w, gate_up_scale, down_w, down_scale, n):
    """Issue the first ``n`` experts' weight DMAs now; the next moe_experts_mx call consumes them."""
    n = min(n, gate_up_w.shape[0])
    _PREFETCH.append(
        {e: _expert_tiles(gate_up_w, gate_up_scale, down_w, down_scale, e, _PREFETCH_DGE) for e in range(n)}
    )


def _swizzle_quantize(src_tok, n_tiles, T, quarter, q_dtype, act_evac=False):
    """Token-major bf16 [T, 4*n_tiles*128] -> MX (data, scale) [128, n_tiles, T] in nkilib's packing.

    x4 slot j of (partition p, tile) holds column ``j*quarter + tile*128 + p`` when ``quarter`` is
    given (hidden dim: H-quarters), else ``tile*512 + j*128 + p`` (intermediate dim, the layout
    nkilib's gate/up output and down weights use).
    """
    swz = nl.ndarray((P, n_tiles, T, Q), dtype=nl.bfloat16, buffer=nl.sbuf)
    for tile in range(n_tiles):
        for j in range(Q):
            c0 = j * quarter + tile * P if quarter else tile * Q * P + j * P
            tp = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=src_tok[:, c0 : c0 + P])
            if act_evac and (tile * Q + j) % 2:  # every other evacuation on ACT
                nisa.activation(dst=swz[:, tile, :, j], op=nl.copy, data=tp)
            else:
                nisa.tensor_copy(dst=swz[:, tile, :, j], src=tp)
    xq = nl.ndarray((P, n_tiles, T), dtype=q_dtype, buffer=nl.sbuf)
    xs = nl.ndarray((P, n_tiles, T), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=xq, src=swz.reshape((P, n_tiles, T * Q)), dst_scale=xs)
    return xq, xs


def _swizzle_quantize_from_xT(xT, n_tiles, T, q_dtype):
    """Feature-major [128, 4*n_tiles, T] (h = h1*128 + p) -> MX tiles in the H-quarter packing.

    Slot j of tile t holds h = j*(H/4) + t*128 + p, i.e. h1 = j*n_tiles + t: a column reorder of
    xT, so no PE transposes are needed when the caller already has the feature-major tokens.
    """
    swz = nl.ndarray((P, n_tiles, T, Q), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(Q):
        nisa.tensor_copy(dst=swz[:, :, :, j], src=xT[:, j * n_tiles : (j + 1) * n_tiles, :])
    xq = nl.ndarray((P, n_tiles, T), dtype=q_dtype, buffer=nl.sbuf)
    xs = nl.ndarray((P, n_tiles, T), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=xq, src=swz.reshape((P, n_tiles, T * Q)), dst_scale=xs)
    return xq, xs


def experts_body(
    x_sb, aff, gate_up_w, gate_up_scale, down_w, down_scale, acc_init=None, pre=None, xT=None, zero_scales=False,
    act_evac=False,
):
    """Local-expert MoE on SBUF inputs; returns the fp32 accumulator [T, H] (SBUF).

    x_sb: bf16 [T, H] (normed tokens), aff: fp32 [T, E] local affinities, both SBUF views at
    base partition 0. acc_init (optional bf16/fp32 [T, H] SBUF): added in (e.g. the shared
    expert), so no separate add pass is needed. pre: {expert: tiles} already in flight.
    More than 128 tokens: pass lists of <= 128-row blocks (x_sb, aff, acc_init); each expert's
    weights are then loaded once and applied to every block, and a list of accumulators returns.
    xT (optional, per block like x_sb): the same tokens feature-major [128, H/128, T], which replaces
    the PE transposes of the activation quantisation with SBUF copies.
    zero_scales: memset the weight-scale tiles before loading them (needed inside an nl.fori_loop body).
    """
    blocked = isinstance(x_sb, (list, tuple))
    xs_l = list(x_sb) if blocked else [x_sb]
    aff_l = list(aff) if blocked else [aff]
    init_l = (list(acc_init) if blocked else [acc_init]) if acc_init is not None else [None] * len(xs_l)
    H = xs_l[0].shape[1]
    E = gate_up_w.shape[0]
    n_h = gate_up_w.shape[3]
    I = gate_up_w.shape[4]
    n_i = down_w.shape[2]
    for xb in xs_l:
        kernel_assert(xb.shape[0] <= P and xb.shape[0] % 2 == 0, f"token blocks must be even and <= 128, got {xb.shape[0]}")
    kernel_assert(n_h * 4 * P == H and n_i * 4 * P == I, "H and I must be multiples of 512")
    kernel_assert(I <= 2048, "gate/up PSUM tiles assume I <= 2048")
    q_dtype = nl.float8_e4m3fn_x4  # activations are always MXFP8; weights may be fp8 or fp4
    OCH = min(_OCH, H)
    pre = pre or {}

    if xT is not None:
        xT_l = list(xT) if blocked else [xT]
        xq_l = [_swizzle_quantize_from_xT(xt, n_h, xb.shape[0], q_dtype) for xt, xb in zip(xT_l, xs_l)]
    else:
        split = act_evac if _SWZ_SPLIT == "" else _SWZ_SPLIT == "1"
        xq_l = [_swizzle_quantize(xb, n_h, xb.shape[0], H // Q, q_dtype, split) for xb in xs_l]
    # Token-partitioned tiles are allocated full-height and viewed [:T], so every two-operand SBUF
    # instruction sees both operands at base partition 0 (the allocator may otherwise place a
    # T<128-row tile in another quadrant, which these instructions reject).
    acc_l = [nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[: xb.shape[0]] for xb in xs_l]

    for e in range(E):
        # ---- weights for expert e (prefetched earlier, or loaded here), shared by all blocks ----
        if _CHUNK:
            gu_ch, dn_ch = (
                pre[e] if e in pre else _expert_chunks(gate_up_w, gate_up_scale, down_w, down_scale, e, "auto", zero_scales)
            )
        elif e in pre:
            wgu, wgus, wd, wds = pre[e]
        else:
            wgu, wgus, wd, wds = _expert_tiles(gate_up_w, gate_up_scale, down_w, down_scale, e, "auto", zero_scales)

        for b, (xq, xs) in enumerate(xq_l):
            T = xs_l[b].shape[0]
            acc, af, acc_init_b = acc_l[b], aff_l[b], init_l[b]
            # ---- gate / up: tokens stationary, weights moving -> token-major [T, I] ----
            ps_g = nl.ndarray((T, I), dtype=nl.float32, buffer=nl.psum)
            ps_u = nl.ndarray((T, I), dtype=nl.float32, buffer=nl.psum)
            for th in range(n_h):
                for gu, ps in ((0, ps_g), (1, ps_u)):
                    if _CHUNK:
                        mw, ms = gu_ch[th][0][:, gu, :], gu_ch[th][1][:, gu, :]
                    else:
                        mw, ms = wgu[:, gu, th, :], wgus[:, gu, th, :]
                    nisa.nc_matmul_mx(
                        dst=ps, stationary=xq[:, th, :], moving=mw,
                        stationary_scale=xs[:, th, :], moving_scale=ms,
                    )
            act = nl.ndarray((P, I), dtype=nl.float32, buffer=nl.sbuf)[:T]
            nisa.activation(dst=act, op=nl.silu, data=ps_g)
            h_bf = nl.ndarray((P, I), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
            nisa.tensor_tensor(dst=h_bf, data1=act, data2=ps_u, op=nl.multiply)
            hq, hs = _swizzle_quantize(h_bf, n_i, T, None, q_dtype)

            # ---- down: activations stationary, weights moving; affinity-scaled accumulate ----
            for c0 in range(0, H, OCH):
                ps_o = nl.ndarray((T, OCH), dtype=nl.bfloat16, buffer=nl.psum)
                for ti in range(n_i):
                    if _CHUNK:
                        mw, ms = dn_ch[c0 // OCH][0][:, ti, :], dn_ch[c0 // OCH][1][:, ti, :]
                    else:
                        mw, ms = wd[:, ti, c0 : c0 + OCH], wds[:, ti, c0 : c0 + OCH]
                    nisa.nc_matmul_mx(
                        dst=ps_o, stationary=hq[:, ti, :], moving=mw,
                        stationary_scale=hs[:, ti, :], moving_scale=ms,
                    )
                if e == 0 and acc_init_b is None:
                    nisa.tensor_scalar(dst=acc[:, c0 : c0 + OCH], data=ps_o, op0=nl.multiply, operand0=af[:, e : e + 1])
                else:
                    base = acc_init_b[:, c0 : c0 + OCH] if e == 0 else acc[:, c0 : c0 + OCH]
                    nisa.scalar_tensor_tensor(
                        dst=acc[:, c0 : c0 + OCH], data=ps_o, op0=nl.multiply, operand0=af[:, e : e + 1],
                        op1=nl.add, operand1=base,
                    )
    return acc_l if blocked else acc_l[0]


@nki.jit
def moe_experts_mx(
    hidden,  # [T, H] bf16
    gate_up_w,  # [E, 128, 2, H/512, I] x4
    gate_up_scale,  # [E, 16, 2, H/512, I] uint8
    down_w,  # [E, 128, I/512, H] x4
    down_scale,  # [E, 16, I/512, H] uint8
    affinities,  # [T, E] (zeros for experts a token is not routed to)
):
    T, H = hidden.shape
    E = gate_up_w.shape[0]
    x_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    nisa.dma_copy(dst=x_sb, src=hidden)
    aff = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)[:T]
    nisa.dma_copy(dst=aff, src=affinities)
    pre = _PREFETCH.pop(0) if _PREFETCH else {}
    acc = experts_body(x_sb, aff, gate_up_w, gate_up_scale, down_w, down_scale, pre=pre)
    out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    o_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:T]
    nisa.tensor_copy(dst=o_sb, src=acc)
    nisa.dma_copy(dst=out, src=o_sb)
    return out
