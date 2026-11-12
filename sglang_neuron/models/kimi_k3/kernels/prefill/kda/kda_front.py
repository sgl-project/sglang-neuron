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

"""KDA front: the projections, short convolutions, scan-input prep and gate projection of one Kimi-K3 KDA layer.

One kernel computes the five fp32 inputs of the KDA scan kernel, the fp32 ``g_proj`` output and the new conv state.
"""

import weakref

import nki
import nki.isa as nisa
import nki.language as nl
from nki.compiler._builtins import get_backend

from ..utils.trn_target import is_trn2

_P_MAX = 128
_F32 = nl.float32
_BF16 = nl.bfloat16
_I32 = nl.int32
_U8 = nl.uint8

_HEADS = 3  # KDA heads per rank (TP32)
_HEAD_DIM = 128
_PROJ = _HEADS * _HEAD_DIM  # 384 columns of q, k, v, g, z
_CONV_TAPS = 4
_CONV_HIST = _CONV_TAPS - 1
_CONV_TILES = 9  # q, k, v x 3 heads; tile c of w_qkvf is conv tile c
_W_TILES = 10  # the 9 conv tiles and f_a
_FA_TILE = 9
_GB_COLS = _PROJ + _HEADS  # g_proj (384) and b_proj (3) columns
# The g|b weight output (free) dim is padded from _GB_COLS (387, odd) to _GB_PAD so the g-phase nc_matmul_mx
# moving/src free dim is even (387 fails the ISA src check 'smx1d3_mm_check_src_tensor'; 388 is the measured
# minimum, 512 is not required); the pad columns are zero and after_g ignores them.
_GB_PAD = 388
_SLOTS = 4  # token tiles of a block: token t0 + slots * p + s sits on partition p, slot s
_BLOCK = _SLOTS * _P_MAX
# W phase order: f_a first (z needs it), v next, then q and k, so the l2norm of q and k ends the block.
_W_ORDER = (9, 6, 7, 8, 0, 1, 2, 3, 4, 5)
_X_GROUP = 8  # hidden chunks of x per staging DMA
_X_BUFS = 2
_W_GROUP = 14  # hidden chunks per weight-tile DMA
_W_BUFS = 4
_GB_GROUP = 4  # hidden chunks per w_gb DMA
_GB_BUFS = 3
_BF_HALVES = 4  # bf16 PSUM transpose tiles of [128, 512] (two banks)
# Matmul bank of G slot s: the slots that the next W tiles and f_b reuse first are evacuated first.
_G_BANK = (2, 3, 0, 1)
_G_EVAC_ORDER = (2, 3, 0, 1)
_RING3 = 3
_RING2 = 2
_L2_EPS = 1e-6  # the l2norm epsilon of the KDA scan's prep
_FP32_MAX = 3.4028234663852886e38
_BF16_MIN = -3.3895313892515355e38  # torch.finfo(torch.bfloat16).min, the pad-row logit
_X_LEAD = 4  # x^T chunks of block 0 built ahead of its first projection
# Estimated PE time (us) of one projection step, for the placement of the deferred work.
_US_PER_COL = 1.0 / 2400.0
_US_PER_TRANSPOSE = 0.06
_DMA_LOAD = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}
_DMA_STORE = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.scalar}
_VEC = nisa.engine.vector

# MXFP8 (opt-in): one nc_matmul_mx contracts 512 hidden rows (128 partitions x 4 x4 lanes), so 4 chunks of the
# bf16 form become one x4 group. f_a and f_b and everything after the projections stay bf16.
_MX_LANES = 4
_MX_GROUP_ROWS = _P_MAX * _MX_LANES  # 512 hidden rows per MX group
_FP8X4 = nl.float8_e4m3fn_x4
_MX_PAIR = 2  # chunks per nc_matmul in FP8 double_row mode (contraction = 128 partitions x 2)
_WMX_BUFS = 2  # conv-weight ring depth: prefetch the next tile while the current tile's matmuls run

# The calls of this kernel in each trace, keyed by the trace context (see _new_trace_id).
_TRACE_CALLS = weakref.WeakKeyDictionary()


@nki.jit
def kimi_k3_kda_front(
    x: nl.NkiTensor,
    w_qkvf: nl.NkiTensor,
    w_gb: nl.NkiTensor,
    w_fb_t: nl.NkiTensor,
    conv_w: nl.NkiTensor,
    conv_state: nl.NkiTensor,
    A_log: nl.NkiTensor,
    dt_bias: nl.NkiTensor,
    ident: nl.NkiTensor,
    valid_len: nl.NkiTensor = None,
    lower_bound: float = -5.0,
    mx: int = 0,
    w_qkv_mx: nl.NkiTensor = None,
    w_qkv_scale: nl.NkiTensor = None,
    w_gb_mx: nl.NkiTensor = None,
    w_gb_scale: nl.NkiTensor = None,
):
    """Compute the front of a Kimi-K3 KDA layer: the projections, short convs, scan-input prep and ``g_proj``.

    Serving shape (Kimi-K3 prefill, TP32): ``T = 4096`` rows of one segment, ``HID = 7168``, 3 local heads of
    128; one call per KDA layer (PE-bound on trn3). Every rounding step is the graph's: bf16 operands with
    fp32 PSUM sums over the hidden chunks in the order 0, 1, ..., bf16 outputs where the layer rounds (q, k, v,
    f_a, z, b), and the conv and prep steps of the KDA scan's prep kernels, so the outputs match the compiled layer
    bit for bit wherever its projections sum in that order. The block kernel (``kda_block_prefill``) runs these
    steps per group of 3 heads, at up to ``T = 8192`` rows (the TP16 form).

    Dimensions:
        T: rows of the segment, a multiple of 128
        HID: hidden size, 56 chunks of 128 at 7168
        H: local KDA heads (3); D: head dimension (128)

    Args:
        x (nl.NkiTensor): [T, HID] bf16 (HBM), the layer input rows.
        w_qkvf (nl.NkiTensor): [10, 128, HID // 128, 128] bf16 (HBM). The q, k, v and f_a weights as 128-column
            stationary tiles: ``w_qkvf[t, p, c, j] = W[128 t + j, 128 c + p]`` for ``W = cat(W_q, W_k, W_v,
            W_fa)`` in ``[out, in]`` layout (tiles 0-2 q, 3-5 k, 6-8 v, 9 f_a).
        w_gb (nl.NkiTensor): [128, HID // 128, 387] bf16 (HBM). ``w_gb[p, c, n] = W_gb[n, 128 c + p]`` for
            ``W_gb = cat(W_g, W_b)``.
        w_fb_t (nl.NkiTensor): [128, 384] bf16 (HBM), ``W_fb`` transposed.
        conv_w (nl.NkiTensor): [3, 384, 4] fp32 (HBM), the q, k and v conv weights.
        conv_state (nl.NkiTensor): [3, 384, 3] bf16 (HBM), the last 3 input rows of the previous segment.
        A_log (nl.NkiTensor): [1, 3] fp32 (HBM), the log gate slopes.
        dt_bias (nl.NkiTensor): [3, 128] fp32 (HBM), the gate bias.
        ident (nl.NkiTensor): [128, 128] fp32 (HBM), the identity (for the TensorE transposes).
        valid_len (nl.NkiTensor): optional [1, 1] int32 (HBM), the count of valid rows. Rows at or after it are
            pad rows: their k is 0 and their z and b logits are the bf16 minimum, as in the layer's graph.
        lower_bound (float): the log-gate lower bound, in [-8, 0).
        mx (int): the projection form of the q, k, v conv tiles and the g|b tile. 0 (default) keeps the bf16 form,
            tracing the same program. 1 and 2 run them as MXFP8 (``nc_matmul_mx``) on Trn3: 1 quantizes x^T with
            ``nisa.quantize_mx`` (per-group E8M0 scales); 2 casts x^T to ``float8_e4m3fn`` with a plain copy and a
            constant scale (no per-group scales). The weights are MXFP8 in the hT_quad layout. f_a and f_b stay bf16.
        w_qkv_mx (nl.NkiTensor): [9, HID // 512, 128, 512] uint8, the MXFP8 q, k, v conv tiles (``kda_front_mx_torch``).
        w_qkv_scale (nl.NkiTensor): [9, 128, HID // 512, 128] uint8, their E8M0 scales.
        w_gb_mx (nl.NkiTensor): [HID // 512, 128, 4 * 387] uint8, the MXFP8 g|b tile.
        w_gb_scale (nl.NkiTensor): [128, HID // 512, 387] uint8, its E8M0 scales.

    Returns:
        qT (nl.NkiTensor): [H, D, T] fp32, ``l2norm(q) * D ** -0.5``.
        kT (nl.NkiTensor): [H, D, T] fp32, ``l2norm(k)``.
        kbT (nl.NkiTensor): [H, D, T] fp32, ``l2norm(k) * beta``.
        vb (nl.NkiTensor): [H, T, D] fp32, ``v * beta``.
        gtok (nl.NkiTensor): [H, T, D] fp32, the log gate ``lower_bound * sigmoid(exp(A_log) * (z + dt_bias))``.
        g (nl.NkiTensor): [T, H * D] fp32, the ``g_proj`` output (the fp32 accumulator, not rounded).
        conv_state_new (nl.NkiTensor): [3, 384, 3] bf16, the last 3 valid input rows of q, k and v.

    Notes:
        - q, k, v are ``silu(causal depthwise conv(x @ W^T))`` per channel, in fp32 from the bf16 projections.
        - The outputs feed the KDA scan kernel (qT, kT, kbT, vb, gtok) and the KDA post kernel (g) unchanged.
        - ``conv_state`` is read last: only the conv of the first 512 rows needs it, and that conv and its prep run
          after the other blocks (from the pre-conv rows kept in HBM). A caller whose conv state comes from a late
          cache-page read thus overlaps the read with the projections instead of waiting for it at the start.
        - SBUF: ~150 KiB per partition. No GpSimd instructions, no DMA transposes, no loops with a runtime trip
          count; all DMAs use the hardware DGE queues of the Sync and Scalar engines.

    Pseudocode:
        for each block of 512 rows (x^T of the block built on TensorE during the previous block):
            for tile in (f_a, v0..2, q0..2, k0..2):     # 56 matmuls each, channel-major out
                acc = sum_c W_tile[c]^T @ x^T[c]         # fp32 PSUM, c = 0, 1, ...
                if tile is a conv tile: y = bf16(silu(conv4(bf16(acc))))
            g|b = sum_c x^T[c]^T @ w_gb[c]              # token-major, 4 token tiles
            z = bf16(f_a) @ W_fb^T
            prep: l2norm(q), l2norm(k) per (row, head) token-major, transposed back; beta = sigmoid(bf16(b));
                  vb = v * beta; kb = kn * beta; gtok = lower_bound * sigmoid(a * (bf16(z) + dt_bias))
    """
    T, HID = x.shape
    KC = HID // _P_MAX
    kernel_assert(T % _P_MAX == 0, f"T={T} must be a multiple of {_P_MAX}")
    kernel_assert(HID % _P_MAX == 0, f"HID={HID} must be a multiple of {_P_MAX}")
    kernel_assert(x.dtype == _BF16, f"x must be bfloat16, got {x.dtype}")
    kernel_assert(
        tuple(w_qkvf.shape) == (_W_TILES, _P_MAX, KC, _P_MAX),
        f"w_qkvf must be [{_W_TILES}, 128, {KC}, 128], got {w_qkvf.shape}",
    )
    kernel_assert(
        tuple(w_gb.shape) == (_P_MAX, KC, _GB_COLS), f"w_gb must be [128, {KC}, {_GB_COLS}], got {w_gb.shape}"
    )
    kernel_assert(tuple(w_fb_t.shape) == (_HEAD_DIM, _PROJ), f"w_fb_t must be [128, {_PROJ}], got {w_fb_t.shape}")
    kernel_assert(
        w_qkvf.dtype == _BF16 and w_gb.dtype == _BF16 and w_fb_t.dtype == _BF16,
        f"the weights must be bfloat16, got {w_qkvf.dtype}, {w_gb.dtype}, {w_fb_t.dtype}",
    )
    kernel_assert(
        tuple(conv_w.shape) == (3, _PROJ, _CONV_TAPS) and conv_w.dtype == _F32,
        f"conv_w must be [3, {_PROJ}, {_CONV_TAPS}] float32, got {conv_w.shape} {conv_w.dtype}",
    )
    kernel_assert(
        tuple(conv_state.shape) == (3, _PROJ, _CONV_HIST) and conv_state.dtype == _BF16,
        f"conv_state must be [3, {_PROJ}, {_CONV_HIST}] bfloat16, got {conv_state.shape} {conv_state.dtype}",
    )
    kernel_assert(
        tuple(A_log.shape) == (1, _HEADS) and A_log.dtype == _F32, f"A_log must be [1, 3] fp32, got {A_log.shape}"
    )
    kernel_assert(
        tuple(dt_bias.shape) == (_HEADS, _HEAD_DIM) and dt_bias.dtype == _F32,
        f"dt_bias must be [3, 128] fp32, got {dt_bias.shape}",
    )
    kernel_assert(
        tuple(ident.shape) == (_P_MAX, _P_MAX) and ident.dtype == _F32,
        f"ident must be [128, 128] fp32, got {ident.shape}",
    )
    kernel_assert(-8.0 <= float(lower_bound) < 0.0, f"lower_bound={lower_bound} must be in [-8, 0)")
    if valid_len is not None:
        kernel_assert(tuple(valid_len.shape) == (1, 1), f"valid_len must be [1, 1], got {valid_len.shape}")
    _check_mx(int(mx), KC, _CONV_TILES, _GB_COLS, w_qkv_mx, w_qkv_scale, w_gb_mx, w_gb_scale)
    front = _KdaFront(
        x,
        w_qkvf,
        w_gb,
        w_fb_t,
        conv_w,
        conv_state,
        A_log,
        dt_bias,
        ident,
        valid_len,
        float(lower_bound),
        mx=int(mx),
        w_qkv_mx=w_qkv_mx,
        w_qkv_scale=w_qkv_scale,
        w_gb_mx=w_gb_mx,
        w_gb_scale=w_gb_scale,
    )
    return front.run()


def _check_mx(mx, KC, n_conv_tiles, gb_cols, w_qkv_mx, w_qkv_scale, w_gb_mx, w_gb_scale):
    """Validate the MX / FP8 option and its weight shapes (called from both front kernels)."""
    if mx == 0:
        return
    kernel_assert(mx in (1, 2, 3), f"mx must be 0, 1, 2 or 3, got {mx}")
    kernel_assert(not is_trn2(), "the FP8 KDA front needs Trn3")
    if mx in (1, 3):
        # mx=1 quantizes x^T with quantize_mx; mx=3 static-casts x^T and uses a constant scale. Same MXFP8 weights.
        kernel_assert(KC % _MX_LANES == 0, f"mx={mx} needs HID % {_MX_GROUP_ROWS} == 0, got HID // 128 = {KC}")
        G = KC // _MX_LANES
        kernel_assert(
            all(t is not None for t in (w_qkv_mx, w_qkv_scale, w_gb_mx, w_gb_scale)),
            f"mx={mx} needs the MXFP8 weight tensors",
        )
        kernel_assert(
            tuple(w_qkv_mx.shape) == (n_conv_tiles, G, _P_MAX, _MX_LANES * _P_MAX),
            f"w_qkv_mx must be [{n_conv_tiles}, {G}, 128, {_MX_LANES * _P_MAX}], got {w_qkv_mx.shape}",
        )
        kernel_assert(
            tuple(w_qkv_scale.shape) == (n_conv_tiles, _P_MAX, G, _P_MAX),
            f"w_qkv_scale must be [{n_conv_tiles}, 128, {G}, 128], got {w_qkv_scale.shape}",
        )
        kernel_assert(
            tuple(w_gb_mx.shape) == (G, _P_MAX, _MX_LANES * gb_cols),
            f"w_gb_mx must be [{G}, 128, {_MX_LANES * gb_cols}], got {w_gb_mx.shape}",
        )
        kernel_assert(
            tuple(w_gb_scale.shape) == (_P_MAX, G, gb_cols),
            f"w_gb_scale must be [128, {G}, {gb_cols}], got {w_gb_scale.shape}",
        )
        return
    # mx == 2 (plain FP8 double_row): the q, k, v conv tiles only; g|b stays bf16, so no w_gb MX tensors.
    kernel_assert(KC % _MX_PAIR == 0, f"mx=2 needs an even HID // 128, got {KC}")
    kernel_assert(w_qkv_mx is not None and w_qkv_scale is not None, "mx=2 needs the FP8 conv weight and its scales")
    kernel_assert(
        tuple(w_qkv_mx.shape) == (n_conv_tiles, _P_MAX, KC, _P_MAX),
        f"w_qkv_mx (mx=2) must be [{n_conv_tiles}, 128, {KC}, 128], got {w_qkv_mx.shape}",
    )
    kernel_assert(
        tuple(w_qkv_scale.shape) == (n_conv_tiles, _P_MAX),
        f"w_qkv_scale (mx=2) must be [{n_conv_tiles}, 128], got {w_qkv_scale.shape}",
    )


def kernel_assert(condition: bool, error_text: str):
    """Raise a kernel input-validation error (a self-contained copy of the NKI library helper)."""
    assert condition, f"Kernel validation exception: {error_text}"  # noqa: S101


def _k_order(kc_count):
    """The order of the hidden chunks in every projection's PSUM accumulation (the compiled graph's order)."""
    return list(range(kc_count))


def _new_trace_id():
    """Number this call among the calls of this kernel in the current trace: 1, 2, ...

    The numbers restart in each trace, so the names of a trace depend only on its own calls, not on what the
    process traced before. The calls that one trace inlines (a program of several layers) still get distinct numbers.
    """
    ctx = get_backend().get_current_context()
    n = _TRACE_CALLS.get(ctx, 0) + 1
    _TRACE_CALLS[ctx] = n
    return n


class _Namer:
    """Give each buffer a name that is unique in one kernel trace: ``kf<call>_<tag>_<n>``."""

    def __init__(self, trace_id):
        self.prefix = f"kf{trace_id}"
        self.n = 0

    def __call__(self, tag):
        self.n += 1
        return f"{self.prefix}_{tag}_{self.n}"


def _pe_transpose(dst, data, ident):
    """Transpose ``data`` [K, n] into the PSUM tile ``dst`` [n, K] on TensorE with a kernel-owned identity.

    Transpose mode is bit exact (each output element takes the input element where the identity is nonzero). A
    kernel-owned identity in ``data``'s dtype replaces NKI's shared one, whose SBUF bytes the allocator may give to
    another call's tensor in a program with several calls.
    """
    k = data.shape[0]
    nisa.nc_matmul(
        dst=dst, stationary=data, moving=ident[0:k, 0:k], is_transpose=True, is_moving_onezero=True, accumulate=False
    )


class _Stream:
    """A ring of SBUF slots fed by DMAs that run ``bufs - 1`` pieces ahead of their use."""

    def __init__(self, ring, loaders):
        self.ring = ring
        self.bufs = ring.shape[1]
        self.loaders = loaders
        self.issued = 0

    def slot(self, n):
        """The ring slot of piece ``n``; issues the DMAs of the pieces up to ``n + bufs - 1``."""
        last = min(n + self.bufs, len(self.loaders))
        while self.issued < last:
            self.loaders[self.issued](self.ring, self.issued % self.bufs)
            self.issued += 1
        return n % self.bufs


def _runs(chunks):
    """Split a list of chunk indices into ``(position, first chunk, count)`` runs of consecutive chunks."""
    out = []
    for i, c in enumerate(chunks):
        if out and c == out[-1][1] + out[-1][2]:
            out[-1] = (out[-1][0], out[-1][1], out[-1][2] + 1)
        else:
            out.append((i, c, 1))
    return out


def _slot_major_lanes(xT, base, ns):
    """The 4 hidden chunks ``base .. base+3`` of x^T as ``[128, ns, 128, 4]`` in slot-major token order.

    Element ``[:, s, p, q]`` is chunk ``base + q`` of token ``ns p + s`` (slot s on partition p), so slot s is the
    contiguous column run ``128 s .. 128 s + 127`` of the x4 x^T that the g phase reads as its stationary.
    """
    return xT[:, base : base + _MX_LANES, 0 : ns * _P_MAX].reshape_dim(2, (_P_MAX, ns)).permute((0, 3, 2, 1))


def _token_order(acc, ns):
    """The ``[128, 128, ns]`` token-order view of a slot-major accumulator ``acc [128, 128 ns]``."""
    return acc.reshape_dim(1, (ns, _P_MAX)).permute((0, 2, 1))


def _copy_rows(rows, acc):
    """VectorE copy of a conv W tile's accumulator into the token-order rows ``rows [128, N]`` of its conv input.

    ``acc`` is ``[128, N]`` in token order, or the ``[128, 128, ns]`` token-order view of a slot-major MX
    accumulator (``_token_order``); only the PSUM read is strided, the SBUF write stays contiguous.
    """
    if len(acc.shape) == 3:
        rows = rows.reshape_dim(1, (acc.shape[1], acc.shape[2]))
    nisa.tensor_copy(dst=rows, src=acc, engine=_VEC)


def _mx_interleave(xT, g, ns, lanes):
    """VectorE copy of the 4 hidden chunks ``4g .. 4g+3`` of x^T into ``lanes [128, N, 4]`` (the x4 pack order),
    the token columns in slot-major order (``_slot_major_lanes``)."""
    N = ns * _P_MAX
    dst = lanes[:, 0:N, :].reshape_dim(1, (ns, _P_MAX))
    nisa.tensor_copy(dst=dst, src=_slot_major_lanes(xT, g * _MX_LANES, ns), engine=_VEC)


class _KdaFront:
    """The kernel: buffers, DMA streams and the per-block pipeline (see ``kimi_k3_kda_front``)."""

    def __init__(
        self,
        x,
        w_qkvf,
        w_gb,
        w_fb_t,
        conv_w,
        conv_state,
        A_log,
        dt_bias,
        ident,
        valid_len,
        lower_bound,
        zf_slots=_RING2,
        mx=0,
        w_qkv_mx=None,
        w_qkv_scale=None,
        w_gb_mx=None,
        w_gb_scale=None,
        out_hbm=None,
        head_group=0,
        p_full=None,
    ):
        self.x, self.w_qkvf, self.w_gb, self.w_fb_t = x, w_qkvf, w_gb, w_fb_t
        self.conv_w, self.conv_state, self.A_log, self.dt_bias = conv_w, conv_state, A_log, dt_bias
        self.ident, self.valid_len, self.lb = ident, valid_len, lower_bound
        self.mx = int(mx)
        self.w_qkv_mx, self.w_qkv_scale, self.w_gb_mx, self.w_gb_scale = w_qkv_mx, w_qkv_scale, w_gb_mx, w_gb_scale
        self.T, self.HID = x.shape
        T = self.T
        self.KC = self.HID // _P_MAX
        self.G = self.KC // _MX_LANES if self.mx else 0
        self.korder = _k_order(self.KC)
        # The W-tile order (the block front sets _W_ORDER_BLOCK before super().__init__).
        self.w_order = getattr(self, "w_order", _W_ORDER)
        self.blocks = []
        for t0 in range(0, T, _BLOCK):
            self.blocks.append((t0, min(_SLOTS, (T - t0) // _P_MAX)))
        self.masked = valid_len is not None
        nm = self.nm = _Namer(_new_trace_id())

        def sb(shape, tag, dtype=_F32):
            return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf, name=nm(tag))

        def ps(shape, tag, dtype=_F32):
            return nl.ndarray(shape, dtype=dtype, buffer=nl.psum, name=nm(tag))

        def hbm(shape, tag, dtype=_F32, buffer=nl.shared_hbm):
            return nl.ndarray(shape, dtype=dtype, buffer=buffer, name=nm(tag))

        H, D = _HEADS, _HEAD_DIM
        # Head-group parameters: group g writes heads 3g .. 3g + 2 of the shared full-width outputs (p_full = hc * D).
        self.out_hbm = out_hbm
        self.hg = int(head_group)
        self.h0 = self.hg * _HEADS
        self.pf = int(p_full) if p_full else _PROJ
        self.ng = self.pf // _PROJ
        self.grouped = self.pf != _PROJ
        self.w_tile_base = self.hg * _CONV_TILES
        self.gb_full = self.ng * _GB_COLS
        self.gb_col = self.hg * _GB_COLS
        self.fb_col = self.hg * _PROJ
        if out_hbm is None:
            self.qT = hbm((H, D, T), "qT")
            self.kT = hbm((H, D, T), "kT")
            self.kbT = hbm((H, D, T), "kbT")
            self.vb = hbm((H, T, D), "vb")
            self.gtok = hbm((H, T, D), "gtok")
            self.g = hbm((T, _PROJ), "g")
            self.cs_out = hbm((3, _PROJ, _CONV_HIST), "conv_state_new", _BF16)
        else:
            self.qT, self.kT, self.kbT = out_hbm["qT"], out_hbm["kT"], out_hbm["kbT"]
            self.vb, self.gtok, self.g = out_hbm["vb"], out_hbm["gtok"], out_hbm["g"]
            self.cs_out = out_hbm["cs_out"]
        # Pre-conv q/k/v rows behind the old state, for the conv state of the last valid rows (per group, 384 wide).
        self.pre = hbm((3, _PROJ, _CONV_HIST + T), "preconv", _BF16)

        # ---- SBUF (per partition, ~150 KiB) ----
        self.xT = sb((_P_MAX, self.KC, _BLOCK), "xT", _BF16)
        self.xst = sb((_P_MAX, _X_BUFS, _SLOTS, _X_GROUP * _P_MAX), "xst", _BF16)
        self.wr = sb((_P_MAX, _W_BUFS, _W_GROUP, _P_MAX), "wring", _BF16)
        self.gbr = sb((_P_MAX, _GB_BUFS, _GB_GROUP, _GB_COLS), "gbring", _BF16)
        self.id_bf = sb((_P_MAX, _P_MAX), "id_bf", _BF16)
        self.id32 = sb((_P_MAX, _P_MAX), "id32")
        self.wfb = sb((_P_MAX, _PROJ), "wfb", _BF16)
        self.cw = sb((_P_MAX, _CONV_TILES, _CONV_TAPS), "cw")
        # Conv input [3 history columns | block rows]; the silu output overwrites the block rows.
        self.cin = sb((_P_MAX, _CONV_TILES, _CONV_HIST + _BLOCK), "cin", _BF16)
        self.cacc = sb((_P_MAX, _RING3, _BLOCK), "cacc")
        self.faT = sb((_P_MAX, _BLOCK), "faT", _BF16)
        self.xb = sb((_P_MAX, 6, _SLOTS, D), "xb", _BF16)  # token-major q/k rows held for the prep
        self.xs = sb((_P_MAX, _RING3, _SLOTS, D), "xs")
        self.ostg = sb((_P_MAX, _RING3, _BLOCK), "ostg")
        self.zb = sb((_P_MAX, _SLOTS, _PROJ), "zb", _BF16)
        # The z-gate tiles: a ring of zf_slots (the block front's after_g rotates over all of them).
        self.zf_slots = int(zf_slots)
        self.zf = sb((_P_MAX, self.zf_slots, _SLOTS, D), "zf")
        self.gst = sb((_P_MAX, _RING2, _PROJ), "gst")
        self.vbs = sb((_P_MAX, _RING2, _SLOTS, D), "vbs")
        # Per-(row, head) prep statistics, two blocks in flight: [mx, rcp, ss, den] x 6 heads (q, k) x slots.
        self.st = sb((_P_MAX, 2, 4, 6, _SLOTS), "stats")
        self.bl = sb((_P_MAX, _SLOTS, H), "blogit", _BF16)
        self.beta = sb((_P_MAX, 2, _SLOTS, H), "beta")
        self.beta0 = sb((_P_MAX, _SLOTS, H), "beta0")  # block 0's beta, for its prep at the end
        self.dtb = sb((_P_MAX, H, D), "dtb")
        self.a_sb = sb((_P_MAX, H), "a")
        if self.masked:
            self.vl_sb = sb((1, 1), "vl", _I32)
            self.vl_f = sb((_P_MAX, 1), "vl_f")
            self.ramp = sb((_P_MAX, _BLOCK), "ramp")
            self.pidx = sb((_P_MAX, 1), "pidx")
            self.keep = sb((_P_MAX, 2, 2, _SLOTS), "keep")  # [block parity, (keep, pad value), slot]
            self.kpred = sb((_P_MAX, 2, _BLOCK), "kpred", _U8)

        # ---- MXFP8 weights (mx == 1 or 3): the conv weights stream, g|b resident ----
        if self.mx in (1, 3):
            G = self.G
            # Two blocks in flight: block b+1's x^T is quantized/cast under block b's matmuls (TensorE stays
            # continuous), so its x_mx must not overwrite block b's. The token columns of x_mx and x_scale are
            # slot-major (column 128 s + p holds token ns p + s): the g phase reads token slot s as the contiguous
            # run 128 s .. 128 s + 127 (nc_matmul_mx needs a unit-stride stationary), and the W phase reads all N
            # columns; its conv input copy puts the rows back in token order (_token_order).
            self.x_mx_fp8 = sb((_P_MAX, 2, G, _BLOCK, _MX_LANES), "x_mx_fp8", nl.float8_e4m3fn)
            self.x_mx = self.x_mx_fp8.view(_FP8X4)
            if self.mx == 1:
                # quantize_mx: an interleave copy then a per-group scale (written through the x4 view).
                self.mx_lanes = sb((_P_MAX, _BLOCK, _MX_LANES), "mx_lanes", _BF16)
                self.x_scale = sb((_P_MAX, 2, G, _BLOCK), "x_scale", _U8)
            else:
                # mx == 3: static e4m3 cast of x^T into the x4 layout (no quantize_mx, no interleave), constant scale.
                self.x_scale_const = sb((_P_MAX, _BLOCK), "x_scale1", _U8)
            # The 9 conv-tile MX weights are ~80 KiB/partition at HID 7168, too big to keep resident, so they
            # stream through a 2-deep per-tile ring (prefetch the next tile while the current tile's matmuls run).
            self.wqtile = sb((_P_MAX, _WMX_BUFS, G, _MX_LANES * _P_MAX), "wqtile", _U8)
            self.wqtiles = sb((_P_MAX, _WMX_BUFS, G, _P_MAX), "wqtiles", _U8)
            # g|b is ~27 KiB and block-invariant: keep it resident (load once), unlike the re-streamed conv tiles.
            self.wgb_d = sb((_P_MAX, G, _MX_LANES * _GB_PAD), "wgb_d", _U8)
            self.wgb_s = sb((_P_MAX, G, _GB_PAD), "wgb_s", _U8)
        # ---- Plain FP8 (mx == 2, double_row): q, k, v conv tiles only; g|b stays bf16 ----
        elif self.mx == 2:
            # Double-buffered so block b+1's cast runs under block b's g-phase matmuls (di/dt), as mx=1.
            self.x_fp8 = sb((_P_MAX, 2, self.KC, _BLOCK), "x_fp8", nl.float8_e4m3fn)
            # The e4m3 conv weights stream through a 2-deep per-tile ring; the per-channel scales are tiny, resident.
            self.wqtile_fp8 = sb((_P_MAX, _WMX_BUFS, self.KC, _P_MAX), "wqtile_fp8", nl.float8_e4m3fn)
            self.wqkv_fp8_sc = sb((_P_MAX, _CONV_TILES), "wqkv_fp8_sc", _F32)

        # ---- PSUM (8 banks) ----
        # A PSUM tensor takes either transposes or matmuls, never both. Four matmul banks: the W tiles take
        # banks 0-1 in turn, f_b banks 2-3, the four G accumulators all of them (_G_BANK); two banks of bf16
        # transposes (four [128, 512] halves) and two of fp32 transposes.
        self.psM = [ps((_P_MAX, 512), f"psM{i}") for i in range(4)]
        self.psB = [ps((_P_MAX, 1024), f"psB{i}", _BF16) for i in range(2)]
        self.psT = [ps((_P_MAX, 512), f"psT{i}") for i in range(2)]
        self.n_t32 = 0
        self.half = 0
        self.ring3 = {"xs": 0, "ostg": 0, "cacc": 0}
        self.ring2 = {"zf": 0, "gst": 0, "vbs": 0}

        # ---- Deferred work on the estimated PE clock: key -> (fire time, sequence, callable) ----
        self.clock = 0.0
        self.pending = {}
        self.seq = 0
        # One W tile of a full block, in estimated PE us; the deferral delays are fractions of it.
        self.tile_us = self.KC * _BLOCK * _US_PER_COL
        self.cacc_owner = [None] * _RING3
        self.last_matmul = None  # the last projection matmul: the conv-state reads are scheduled after it
        self._build_streams()

    # ------------------------------------------------------------------ infrastructure

    def defer(self, frac, key, fn):
        """Emit ``fn`` once the estimated PE clock passes ``now + frac * tile_us``, or when ``key`` is forced.

        The delay only places the work in the program order, so it overlaps the projections. Program order is what
        the dependencies follow, so every reader of a single-buffered tile forces the deferred writers and readers
        of that tile's previous contents first (``force``).
        """
        self.pending[key] = (self.clock + frac * self.tile_us, self.seq, fn)
        self.seq += 1

    def force_tile(self, b, ci, upto):
        """Emit the deferred conv chain of conv tile ``ci`` of block ``b`` (conv -> silu -> ds1) up to ``upto``."""
        for name in ("conv", "silu", "ds1")[: ("conv", "silu", "ds1").index(upto) + 1]:
            self.force((name, b, ci))

    def force(self, key):
        """Emit the deferred work ``key`` now, if it is still pending."""
        item = self.pending.pop(key, None)
        if item is not None:
            item[2]()

    def force_blocks(self, last_block):
        """Emit all deferred work of blocks ``<= last_block``, in time order."""
        while True:
            keys = [k for k in self.pending if k[1] <= last_block]
            if not keys:
                return
            self.force(min(keys, key=lambda k: self.pending[k][:2]))

    def tick(self, us):
        """Advance the estimated PE clock and emit the deferred work that is due, in time order."""
        self.clock += us
        while True:
            due = [k for k, v in self.pending.items() if v[0] <= self.clock]
            if not due:
                return
            self.force(min(due, key=lambda k: self.pending[k][:2]))

    def flush(self):
        """Emit all deferred work, in time order."""
        self.force_blocks(len(self.blocks))

    def next_ring(self, ring, name):
        i = ring[name]
        ring[name] = i + 1
        return i

    def bf_half(self):
        """The next bf16 PSUM transpose tile, [128, 512]."""
        i = self.half % _BF_HALVES
        self.half += 1
        return self.psB[i // 2][:, (i % 2) * 512 : (i % 2 + 1) * 512]

    def _build_streams(self):
        """The DMA streams of x rows, the projection weight tiles and the g|b weights, in use order."""
        korder, KC, HID = self.korder, self.KC, self.HID

        def x_loader(t0, ns, chunks):
            def load(ring, slot):
                for pos, c0, cnt in _runs(chunks):
                    nisa.dma_copy(
                        dst=ring[:, slot, 0:ns, pos * _P_MAX : (pos + cnt) * _P_MAX],
                        src=self.x.ap(
                            pattern=[[HID, _P_MAX], [_P_MAX * HID, ns], [1, cnt * _P_MAX]],
                            offset=t0 * HID + c0 * _P_MAX,
                        ),
                        **_DMA_LOAD,
                    )

            return load

        def w_loader(tile, chunks):
            def load(ring, slot):
                for pos, c0, cnt in _runs(chunks):
                    nisa.dma_copy(
                        dst=ring[:, slot, pos : pos + cnt, :],
                        src=self.w_qkvf.ap(
                            pattern=[[KC * _P_MAX, _P_MAX], [1, cnt * _P_MAX]],
                            offset=tile * _P_MAX * KC * _P_MAX + c0 * _P_MAX,
                        ),
                        **_DMA_LOAD,
                    )

            return load

        def gb_loader(chunks):
            def load(ring, slot):
                for pos, c0, cnt in _runs(chunks):
                    nisa.dma_copy(
                        dst=ring[:, slot, pos : pos + cnt, :],
                        src=self.w_gb.ap(pattern=[[KC * _GB_COLS, _P_MAX], [1, cnt * _GB_COLS]], offset=c0 * _GB_COLS),
                        **_DMA_LOAD,
                    )

            return load

        xl, wl, gl = [], [], []
        for t0, ns in self.blocks:
            for j0 in range(0, KC, _X_GROUP):
                xl.append(x_loader(t0, ns, korder[j0 : j0 + _X_GROUP]))
            for tile in _W_ORDER:
                for j0 in range(0, KC, _W_GROUP):
                    wl.append(w_loader(tile, korder[j0 : j0 + _W_GROUP]))
            for j0 in range(0, KC, _GB_GROUP):
                gl.append(gb_loader(korder[j0 : j0 + _GB_GROUP]))
        self.xs_stream = _Stream(self.xst, xl)
        self.w_stream = _Stream(self.wr, wl)
        self.gb_stream = _Stream(self.gbr, gl)
        self.x_per_block = -(-KC // _X_GROUP)
        self.w_per_tile = -(-KC // _W_GROUP)
        self.gb_per_block = -(-KC // _GB_GROUP)

    # ------------------------------------------------------------------ MXFP8 (mx != 0)

    def build_mx_wstream(self):
        """Build the conv-weight ring: one DMA loader per (block, conv tile), in W-phase order (mx=1 and mx=2)."""
        G, KC = self.G, self.KC
        conv_positions = [pos for pos, tile in enumerate(self.w_order) if tile != _FA_TILE]
        lane_p = _MX_LANES * _P_MAX

        def mx1_loader(tile):
            phys = self.w_tile_base + tile  # the physical conv tile of this group (9 g + tile)

            def load(ring, slot):
                nisa.dma_copy(
                    dst=ring[:, slot],
                    src=self.w_qkv_mx.ap(
                        pattern=[[lane_p, _P_MAX], [_P_MAX * lane_p, G], [1, lane_p]], offset=phys * G * _P_MAX * lane_p
                    ),
                    **_DMA_LOAD,
                )
                nisa.dma_copy(
                    dst=self.wqtiles[:, slot],
                    src=self.w_qkv_scale.ap(
                        pattern=[[G * _P_MAX, _P_MAX], [_P_MAX, G], [1, _P_MAX]], offset=phys * _P_MAX * G * _P_MAX
                    ),
                    **_DMA_LOAD,
                )

            return load

        def mx2_loader(tile):
            def load(ring, slot):
                nisa.dma_copy(
                    dst=ring[:, slot],
                    src=self.w_qkv_mx.ap(
                        pattern=[[KC * _P_MAX, _P_MAX], [_P_MAX, KC], [1, _P_MAX]], offset=tile * _P_MAX * KC * _P_MAX
                    ),
                    **_DMA_LOAD,
                )

            return load

        make = mx2_loader if self.mx == 2 else mx1_loader
        self.wq_index = {}
        loaders = []
        for b in range(len(self.blocks)):
            for pos in conv_positions:
                self.wq_index[(b, pos)] = len(loaders)
                loaders.append(make(self.w_order[pos]))
        self.wq_stream = _Stream(self.wqtile_fp8 if self.mx == 2 else self.wqtile, loaders)

    def load_mx_weights(self):
        """Load the resident FP8 weights (g|b MXFP8 for ``mx=1``/``mx=3``, the per-channel conv scales for ``mx=2``)."""
        self.build_mx_wstream()
        if self.mx == 2:
            return self._load_fp8_weights()
        if self.mx == 3:
            nisa.memset(dst=self.x_scale_const, value=127, engine=_VEC)  # E8M0 127 = 2^0 = 1.0 (static x scale)
        G, gb4 = self.G, _MX_LANES * _GB_COLS
        # Only g|b is resident (block-invariant, ~27 KiB); the conv tiles stream (build_mx_wstream). The weight is
        # loaded into the first _GB_COLS output columns; the pad columns (_GB_COLS .. _GB_PAD) are zeroed.
        if _GB_PAD > _GB_COLS:
            nisa.memset(dst=self.wgb_d[:, :, gb4:], value=0, engine=_VEC)
            nisa.memset(dst=self.wgb_s[:, :, _GB_COLS:], value=0, engine=_VEC)
        if self.grouped:
            # Group-major g|b MXFP8: this group's 4 * 387 packed columns (weight) and 387 scales of each G lane.
            nisa.dma_copy(
                dst=self.wgb_d[:, :, 0:gb4],
                src=self.w_gb_mx.ap(
                    pattern=[[gb4 * self.ng, _P_MAX], [_P_MAX * gb4 * self.ng, G], [1, gb4]], offset=gb4 * self.hg
                ),
                **_DMA_LOAD,
            )
            nisa.dma_copy(
                dst=self.wgb_s[:, :, 0:_GB_COLS],
                src=self.w_gb_scale.ap(
                    pattern=[[G * self.ng * _GB_COLS, _P_MAX], [self.ng * _GB_COLS, G], [1, _GB_COLS]],
                    offset=_GB_COLS * self.hg,
                ),
                **_DMA_LOAD,
            )
        else:
            nisa.dma_copy(
                dst=self.wgb_d[:, :, 0:gb4],
                src=self.w_gb_mx.ap(pattern=[[gb4, _P_MAX], [_P_MAX * gb4, G], [1, gb4]], offset=0),
                **_DMA_LOAD,
            )
            nisa.dma_copy(dst=self.wgb_s[:, :, 0:_GB_COLS], src=self.w_gb_scale, **_DMA_LOAD)
        self.wqtile_x4 = self.wqtile.view(_FP8X4)  # [128, bufs, G, 128]
        self.wgb_x4 = self.wgb_d.view(_FP8X4)  # [128, G, _GB_PAD]

    def _load_fp8_weights(self):
        """Load the resident per-channel conv scales [128, 9]; the e4m3 conv weights stream (build_mx_wstream)."""
        nisa.dma_copy(
            dst=self.wqkv_fp8_sc,
            src=self.w_qkv_scale.ap(pattern=[[1, _P_MAX], [_P_MAX, _CONV_TILES]], offset=0),
            **_DMA_LOAD,
        )

    def quantize_block(self, b):
        """Quantize all of block ``b``'s x^T up front (mx=1), or the plain e4m3 cast of x^T (mx=2).

        The later blocks are quantized group by group under the previous block's g-phase matmuls
        (``_quantize_group``); only block 0 has no previous block to hide under.
        """
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        if self.mx == 2:
            # Plain e4m3 cast, per-tensor scale 1.0 (the post-norm x). Same chunk layout as x^T, buffer parity b%2.
            nisa.activation(dst=self.x_fp8[:, b % 2, :, 0:N], op=nl.copy, data=self.xT[:, :, 0:N])
            return
        for g in range(self.G):
            self._quantize_group(b % 2, g, N)

    def _cast_fp8_next(self, nxt, built_upto, state):
        """Cast block ``nxt``'s newly-built x^T chunks into x_fp8 parity ``nxt % 2`` (mx=2 overlap under b's matmuls)."""
        _, ns = self.blocks[nxt]
        N = ns * _P_MAX
        par = nxt % 2
        ready = self._mx_ready_chunks(built_upto)
        while state[0] < ready:
            c = state[0]
            nisa.activation(dst=self.x_fp8[:, par, c, 0:N], op=nl.copy, data=self.xT[:, c, 0:N])
            state[0] += 1

    def _quantize_group(self, par, g, N):
        """Build hidden group ``g`` of the x4 x^T in buffer parity ``par`` (slot-major token columns): quantize_mx
        (mx=1) or static cast (mx=3)."""
        ns = N // _P_MAX
        if self.mx == 3:
            # Static e4m3 cast fused with the x4 reorder (the 4 hidden chunks 4g..4g+3 -> one x4 element).
            # Known blocker (mx=3 is default off): neuronx-cc rejects this on device (NCC_IBIR221, an activation
            # copy cannot write an MX-tagged buffer); mx=3 needs an MX-aware producer (quantize_mx with a forced
            # constant scale) before it can compile. mx=1 (quantize_mx) is the supported path.
            nisa.activation(
                dst=self.x_mx_fp8[:, par, g, 0:N, :].reshape_dim(1, (ns, _P_MAX)),
                op=nl.copy,
                data=_slot_major_lanes(self.xT, g * _MX_LANES, ns),
            )
            return
        _mx_interleave(self.xT, g, ns, self.mx_lanes)
        nisa.quantize_mx(
            dst=self.x_mx[:, par, g, 0:N], src=self.mx_lanes[:, 0:N, :], dst_scale=self.x_scale[:, par, g, 0:N]
        )

    def _x_scale_w(self, par, g, N):
        """The W-phase moving x scale of group ``g`` (per-group for mx=1, the constant 127 for mx=3)."""
        return self.x_scale_const[:, 0:N] if self.mx == 3 else self.x_scale[:, par, g, 0:N]

    def _mx_ready_chunks(self, built_upto):
        """How many x^T chunks are available after building up to chunk ``built_upto`` (standalone: each is built now)."""
        return built_upto + 1

    def _w_tile_mx(self, b, pos):
        """The G accumulating ``nc_matmul_mx`` of conv W tile ``self.w_order[pos]`` (channel-major output).

        The moving x_mx has slot-major token columns, so the accumulator is slot-major too; the conv input copy reads
        it in token order (``_token_order``). Each column sums the same products in the same order.
        """
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        par = b % 2
        tile = self.w_order[pos]
        acc = self.psM[pos % 2][:, 0:N]
        slot = self.wq_stream.slot(self.wq_index[(b, pos)])  # the streamed conv weights of this tile
        wqd = self.wqtile_x4[:, slot]  # [128, G, 128] x4
        wqs = self.wqtiles[:, slot]  # [128, G, 128]
        step_us = _MX_LANES * N * _US_PER_COL  # the clock of the 4 bf16 chunks this group replaces
        for g in range(self.G):
            nisa.nc_matmul_mx(
                dst=acc,
                stationary=wqd[:, g, :],
                moving=self.x_mx[:, par, g, 0:N],
                stationary_scale=wqs[:, g, :],
                moving_scale=self._x_scale_w(par, g, N),
                accumulate=(g > 0),
            )
            self.tick(step_us)
        self.conv_tile(b, tile, _token_order(acc, ns))

    def _w_tile_fp8(self, b, pos):
        """Conv W tile ``self.w_order[pos]`` as plain FP8 ``double_row``: one matmul per chunk pair, then a per-channel scale."""
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        tile = self.w_order[pos]
        acc = self.psM[pos % 2][:, 0:N]
        wq = self.wqtile_fp8[:, self.wq_stream.slot(self.wq_index[(b, pos)])]  # [128, KC, 128] e4m3 (streamed)
        xf = self.x_fp8[:, b % 2]  # [128, KC, BLOCK] e4m3 of this block
        step_us = _MX_PAIR * N * _US_PER_COL  # the clock of the 2 bf16 chunks this pair replaces
        for pc in range(self.KC // _MX_PAIR):
            nisa.nc_matmul(
                dst=acc,
                stationary=wq[:, _MX_PAIR * pc : _MX_PAIR * pc + _MX_PAIR, :],
                moving=xf[:, _MX_PAIR * pc : _MX_PAIR * pc + _MX_PAIR, 0:N],
                perf_mode=nisa.matmul_perf_mode.double_row,
                accumulate=(pc > 0),
            )
            self.tick(step_us)
        # Per-output-channel weight scale, applied on the channel-major output (channel on the partition dim).
        nisa.tensor_scalar(
            dst=acc, data=acc, op0=nl.multiply, operand0=self.wqkv_fp8_sc[:, tile : tile + 1], engine=_VEC
        )
        self.conv_tile(b, tile, acc)

    def _g_phase_mx(self, b):
        """g|b token-major with ``nc_matmul_mx``; the next block's x^T is built and quantized behind these matmuls.

        Group ``g`` of block ``b + 1`` is quantized into the other MXFP8 buffer right after its x^T chunks are built,
        so block ``b + 1``'s W phase finds its x_mx ready and TensorE does not idle on the quantize.

        The stationary of token slot s (tokens ``ns p + s``) is the contiguous column run ``128 s .. 128 s + 127``
        of the slot-major x_mx and x_scale, so the matmuls read them in place (no re-gather copy).
        """
        t0, ns = self.blocks[b]
        par = b % 2
        nb = len(self.blocks)
        nxt = b + 1 if b + 1 < nb else None
        nxt_ns = self.blocks[nxt][1] if nxt is not None else 0
        for part in range(2):
            self.force(("fb", b, part))
        step_us = _MX_LANES * ns * _GB_COLS * _US_PER_COL
        qg = 0  # the next group of block b+1 to quantize (only once its x^T chunks are built)
        nxt_N = nxt_ns * _P_MAX
        for g in range(self.G):
            for s in range(ns):
                cols = slice(s * _P_MAX, (s + 1) * _P_MAX)
                if self.mx == 1:
                    stat_scale = self.x_scale[:, par, g, cols]
                else:
                    stat_scale = self.x_scale_const[:, 0:_P_MAX]
                self.last_matmul = nisa.nc_matmul_mx(
                    dst=self.psM[_G_BANK[s]][:, 0:_GB_PAD],
                    stationary=self.x_mx[:, par, g, cols],
                    moving=self.wgb_x4[:, g, :],
                    stationary_scale=stat_scale,
                    moving_scale=self.wgb_s[:, g, :],
                    accumulate=(g > 0),
                )
            if nxt is not None:
                for q in range(_MX_LANES):
                    self.xT_chunk(nxt, g * _MX_LANES + q)
                ready = self._mx_ready_chunks(g * _MX_LANES + _MX_LANES - 1)
                while (qg + 1) * _MX_LANES <= ready:
                    self._quantize_group(nxt % 2, qg, nxt_N)
                    qg += 1
            self.tick(step_us)
        self.after_g(b)

    # ------------------------------------------------------------------ setup

    def load_constants(self):
        """Identities, f_b weights, conv weights, gate constants and the pad-row masks (not the conv state)."""
        H, D = _HEADS, _HEAD_DIM
        nisa.dma_copy(dst=self.id32, src=self.ident, **_DMA_LOAD)
        nisa.tensor_copy(dst=self.id_bf, src=self.id32, engine=_VEC)
        if self.grouped:
            # W_fb of this group: columns 384 g .. 384 g + 383 of the full [128, hc * D] tensor.
            nisa.dma_copy(
                dst=self.wfb,
                src=self.w_fb_t.ap(pattern=[[self.pf, _P_MAX], [1, _PROJ]], offset=self.fb_col),
                **_DMA_LOAD,
            )
        else:
            nisa.dma_copy(dst=self.wfb, src=self.w_fb_t, **_DMA_LOAD)
        # cw[p, ci, j] = conv_w[ci // 3, 128 * (ci % 3) + p, j]; tile ci of w_qkvf is conv tile ci.
        if self.grouped:
            # conv_w is [3, hc * D, 4]; this group's 3 heads of each q/k/v are a 384-channel block (per q/k/v DMA).
            for i3 in range(3):
                nisa.dma_copy(
                    dst=self.cw[:, i3 * _HEADS : (i3 + 1) * _HEADS, :],
                    src=self.conv_w.ap(
                        pattern=[[_CONV_TAPS, _P_MAX], [_P_MAX * _CONV_TAPS, _HEADS], [1, _CONV_TAPS]],
                        offset=(i3 * self.pf + self.h0 * _P_MAX) * _CONV_TAPS,
                    ),
                    **_DMA_LOAD,
                )
        else:
            nisa.dma_copy(
                dst=self.cw,
                src=self.conv_w.ap(
                    pattern=[[_CONV_TAPS, _P_MAX], [_P_MAX * _CONV_TAPS, _CONV_TILES], [1, _CONV_TAPS]], offset=0
                ),
                **_DMA_LOAD,
            )
        nisa.dma_copy(
            dst=self.dtb, src=self.dt_bias.ap(pattern=[[0, _P_MAX], [D, H], [1, D]], offset=self.h0 * D), **_DMA_LOAD
        )
        a_raw = nl.ndarray((_P_MAX, H), dtype=_F32, buffer=nl.sbuf, name=self.nm("a_raw"))
        nisa.dma_copy(dst=a_raw, src=self.A_log.ap(pattern=[[0, _P_MAX], [1, H]], offset=self.h0), **_DMA_LOAD)
        # a = min(exp(A_log), FP32_MAX), as the prep kernel computes it.
        nisa.activation(dst=self.a_sb, op=nl.exp, data=a_raw)
        nisa.tensor_scalar(dst=self.a_sb, data=self.a_sb, op0=nl.minimum, operand0=_FP32_MAX, engine=_VEC)
        if self.mx:
            self.load_mx_weights()
        if not self.masked:
            return
        nisa.dma_copy(dst=self.vl_sb, src=self.valid_len, **_DMA_LOAD)
        vl_i = nl.ndarray((_P_MAX, 1), dtype=_I32, buffer=nl.sbuf, name=self.nm("vl_i"))
        nisa.dma_copy(dst=vl_i, src=self.valid_len.ap(pattern=[[0, _P_MAX], [1, 1]], offset=0), **_DMA_LOAD)
        nisa.tensor_copy(dst=self.vl_f, src=vl_i, engine=_VEC)
        # ramp[p, j] = j: a running sum of ones along the free axis (exact small integers).
        zeros = nl.ndarray((_P_MAX, _BLOCK), dtype=_F32, buffer=nl.sbuf, name=self.nm("zeros"))
        nisa.memset(dst=zeros, value=0.0, engine=_VEC)
        nisa.tensor_scalar_cumulative(dst=self.ramp, src=zeros, op0=nl.add, op1=nl.add, imm0=1.0)
        nisa.tensor_scalar(dst=self.ramp, data=self.ramp, op0=nl.subtract, operand0=1.0, engine=_VEC)
        # pidx[p] = p = sum_j ident[p, j] * j.
        prod = nl.ndarray((_P_MAX, _P_MAX), dtype=_F32, buffer=nl.sbuf, name=self.nm("pidx_prod"))
        nisa.tensor_tensor(dst=prod, data1=self.id32, data2=self.ramp[:, 0:_P_MAX], op=nl.multiply, engine=_VEC)
        nisa.tensor_reduce(dst=self.pidx, op=nl.add, data=prod, axis=1)

    def block_masks(self, b):
        """keep / pad value of the block's token-major rows, and the channel-major pad predicate of k."""
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        par = b % 2
        self.force_blocks(b - 2)  # the masks of this parity
        base = nl.ndarray((_P_MAX, 1), dtype=_F32, buffer=nl.sbuf, name=self.nm("mbase"))
        nisa.tensor_scalar(
            dst=base, data=self.pidx, op0=nl.multiply, operand0=float(ns), op1=nl.add, operand1=float(t0), engine=_VEC
        )
        tok = nl.ndarray((_P_MAX, _SLOTS), dtype=_F32, buffer=nl.sbuf, name=self.nm("mtok"))
        nisa.tensor_scalar(dst=tok[:, 0:ns], data=self.ramp[:, 0:ns], op0=nl.add, operand0=base, engine=_VEC)
        self.mtok = tok  # the token of each (partition, slot), for the masks of a subclass
        keep = self.keep[:, par, 0, 0:ns]
        nisa.tensor_scalar(dst=keep, data=tok[:, 0:ns], op0=nl.less, operand0=self.vl_f, engine=_VEC)
        # pad value: 0 on a valid row, the bf16 minimum on a pad row (both exact).
        nisa.tensor_scalar(
            dst=self.keep[:, par, 1, 0:ns],
            data=keep,
            op0=nl.multiply,
            operand0=-_BF16_MIN,
            op1=nl.add,
            operand1=_BF16_MIN,
            engine=_VEC,
        )
        nisa.tensor_scalar(
            dst=self.kpred[:, par, 0:N],
            data=self.ramp[:, 0:N],
            op0=nl.add,
            operand0=float(t0),
            op1=nl.greater_equal,
            operand1=self.vl_f,
            engine=_VEC,
        )

    # ------------------------------------------------------------------ x^T

    def xT_chunk(self, bn, j):
        """Build hidden chunk ``korder[j]`` of block ``bn``'s x^T: 4 bf16 TensorE transposes and a VectorE copy."""
        t0, ns = self.blocks[bn]
        slot = self.xs_stream.slot(bn * self.x_per_block + j // _X_GROUP)
        jj = j % _X_GROUP
        kc = self.korder[j]
        ps = self.bf_half()
        for s in range(ns):
            _pe_transpose(
                ps[:, s * _P_MAX : (s + 1) * _P_MAX], self.xst[:, slot, s, jj * _P_MAX : (jj + 1) * _P_MAX], self.id_bf
            )
        nisa.tensor_copy(dst=self.xT[:, kc, 0 : ns * _P_MAX], src=ps[:, 0 : ns * _P_MAX], engine=_VEC)

    # ------------------------------------------------------------------ W phase

    def w_tile(self, b, pos):
        """The 56 accumulating matmuls of W tile ``_W_ORDER[pos]`` of block ``b`` (channel-major output)."""
        tile = _W_ORDER[pos]
        if self.mx and tile != _FA_TILE:
            return self._w_tile_fp8(b, pos) if self.mx == 2 else self._w_tile_mx(b, pos)
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        acc = self.psM[pos % 2][:, 0:N]
        base = (b * _W_TILES + pos) * self.w_per_tile
        step_us = N * _US_PER_COL
        if pos == _W_TILES - 1 and b + 1 < len(self.blocks):
            # The staging ring is free during the W phase: start the next block's x rows now.
            self.xs_stream.slot((b + 1) * self.x_per_block)
        for j, kc in enumerate(self.korder):
            slot = self.w_stream.slot(base + j // _W_GROUP)
            nisa.nc_matmul(
                dst=acc,
                stationary=self.wr[:, slot, j % _W_GROUP, :],
                moving=self.xT[:, kc, 0:N],
                accumulate=(j > 0),
            )
            if b == 0 and pos == 0 and j + _X_LEAD < self.KC and not self.mx:
                self.xT_chunk(0, j + _X_LEAD)
            self.tick(step_us)
        if tile == _FA_TILE:
            for part in range(2):
                self.force(("fb", b - 1, part))
            # f_a rounds to bf16 (the f_b matmul's moving operand in the graph).
            nisa.tensor_copy(dst=self.faT[:, 0:N], src=acc, engine=_VEC)
            self.defer(0.12, ("fb", b, 0), lambda: self.f_b(b, range(0, min(2, ns))))
            if ns > 2:
                self.defer(0.33, ("fb", b, 1), lambda: self.f_b(b, range(2, ns)))
        else:
            self.conv_tile(b, tile, acc)

    def conv_tile(self, b, ci, acc):
        """Round, store the pre-conv rows, run the causal conv; the silu and its consumers follow later."""
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        i3, h = divmod(ci, 3)
        # The previous block's readers and writer of this conv buffer go first.
        self.force_tile(b - 1, ci, "ds1" if ci < 6 else "silu")
        if ci >= 6:
            self.force(("v", b - 1, ci - 6))
        cin = self.cin[:, ci, :]
        _copy_rows(cin[:, _CONV_HIST : _CONV_HIST + N], acc)
        nisa.dma_copy(
            dst=self.pre.ap(
                pattern=[[_CONV_HIST + self.T, _P_MAX], [1, N]],
                offset=(i3 * _PROJ + h * _P_MAX) * (_CONV_HIST + self.T) + _CONV_HIST + t0,
            ),
            src=cin[:, _CONV_HIST : _CONV_HIST + N],
            **_DMA_STORE,
        )
        if b == 0:
            # Block 0's conv needs the old conv state, the kernel's only input that the caller may produce
            # late (a cache page read). Its conv and prep run at the end, from the pre-conv rows in HBM.
            nisa.tensor_copy(dst=cin[:, 0:_CONV_HIST], src=cin[:, N : N + _CONV_HIST], engine=_VEC)
            return
        self.conv(b, ci)

    def conv(self, b, ci):
        """acc[c, t] = sum_j w[c, j] * xpad[c, t + j] (the KDA short-conv kernel's steps); silu follows later."""
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        cin = self.cin[:, ci, :]
        slot = self.next_ring(self.ring3, "cacc") % _RING3
        if self.cacc_owner[slot] is not None:
            self.force(self.cacc_owner[slot])
        self.cacc_owner[slot] = ("silu", b, ci)
        ca = self.cacc[:, slot, 0:N]
        nisa.tensor_scalar(dst=ca, data=cin[:, 0:N], op0=nl.multiply, operand0=self.cw[:, ci, 0:1], engine=_VEC)
        for j in range(1, _CONV_TAPS):
            nisa.scalar_tensor_tensor(
                dst=ca,
                data=cin[:, j : j + N],
                op0=nl.multiply,
                operand0=self.cw[:, ci, j : j + 1],
                op1=nl.add,
                operand1=ca,
            )
        if b > 0:
            # The next block's history: the last 3 pre-conv rows of this block.
            nisa.tensor_copy(dst=cin[:, 0:_CONV_HIST], src=cin[:, N : N + _CONV_HIST], engine=_VEC)
        self.defer(0.2, ("silu", b, ci), lambda: self.silu_tile(b, ci, ca))

    def block0_prep(self):
        """Block 0's conv and prep, after the other blocks: the old conv state and block 0's pre-conv rows."""
        t0, ns = self.blocks[0]
        N = ns * _P_MAX
        if self.masked:
            self.block_masks(0)
        for ci in range(_CONV_TILES):
            i3, h = divmod(ci, 3)
            row = i3 * _PROJ + h * _P_MAX  # pre scratch row (per group, 384 wide)
            row_in = i3 * self.pf + (self.h0 + h) * _P_MAX  # conv_state input row (full width)
            load = nisa.dma_copy(
                dst=self.cin[:, ci, 0:_CONV_HIST],
                src=self.conv_state.ap(pattern=[[_CONV_HIST, _P_MAX], [1, _CONV_HIST]], offset=row_in * _CONV_HIST),
                **_DMA_LOAD,
            )
            if self.last_matmul is not None and hasattr(load, "depends_on"):
                # Keep the scheduler from hoisting this read: its producer may finish late.
                load.depends_on(self.last_matmul, kind=nisa.dependency.schedule)
            # The old state rows of the scratch, for a new conv state of valid_len < 3 rows.
            nisa.dma_copy(
                dst=self.pre.ap(
                    pattern=[[_CONV_HIST + self.T, _P_MAX], [1, _CONV_HIST]], offset=row * (_CONV_HIST + self.T)
                ),
                src=self.cin[:, ci, 0:_CONV_HIST],
                **_DMA_STORE,
            )
            nisa.dma_copy(
                dst=self.cin[:, ci, _CONV_HIST : _CONV_HIST + N],
                src=self.pre.ap(
                    pattern=[[_CONV_HIST + self.T, _P_MAX], [1, N]], offset=row * (_CONV_HIST + self.T) + _CONV_HIST
                ),
                **_DMA_LOAD,
            )
            self.conv(0, ci)
        self.ds_tail(0)

    def silu_tile(self, b, ci, ca):
        """silu(conv) -> bf16 into the conv buffer; mask k's pad rows; schedule the token-major prep of q and k."""
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        y = self.cin[:, ci, _CONV_HIST : _CONV_HIST + N]
        nisa.activation(dst=y, op=nl.silu, data=ca)
        if self.masked and 3 <= ci < 6:
            nisa.tensor_copy_predicated(dst=y, src=0.0, predicate=self.kpred[:, b % 2, 0:N])
        if ci < 6:
            self.defer(0.2, ("ds1", b, ci), lambda: self.ds1_qk(b, ci))

    def f_b(self, b, slots):
        """z = bf16(f_a) @ W_fb^T for the given token slots, token-major; bf16(z) with the pad rows' value."""
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        for s in slots:
            dst = self.psM[2 + s % 2][:, 0:_PROJ]
            nisa.nc_matmul(dst=dst, stationary=self.faT[:, s:N:ns], moving=self.wfb, accumulate=False)
            if self.masked:
                par = b % 2
                nisa.tensor_scalar(
                    dst=self.zb[:, s, :],
                    data=dst,
                    op0=nl.multiply,
                    operand0=self.keep[:, par, 0, s : s + 1],
                    op1=nl.add,
                    operand1=self.keep[:, par, 1, s : s + 1],
                    engine=_VEC,
                )
            else:
                nisa.tensor_copy(dst=self.zb[:, s, :], src=dst, engine=_VEC)

    # ------------------------------------------------------------------ prep of q and k

    def ds1_qk(self, b, ci):
        """Token-major q or k rows of one head: bf16 transposes, then the l2norm's max scale and sum of squares."""
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        par = b % 2
        # xb[ci] and the statistics of this parity: their previous readers go first.
        self.force(("q" if ci < 3 else "k", b - 1, ci % 3))
        self.force_blocks(b - 2)
        ps = self.bf_half()
        for s in range(ns):
            _pe_transpose(
                ps[:, s * _P_MAX : (s + 1) * _P_MAX], self.cin[:, ci, _CONV_HIST + s : _CONV_HIST + N : ns], self.id_bf
            )
        xt = ps[:, 0:N].reshape_dim(1, (ns, _P_MAX))
        nisa.tensor_copy(dst=self.xb[:, ci, 0:ns, :], src=xt, engine=_VEC)
        mx, rcp, ss = (self.st[:, par, k, ci, 0:ns] for k in range(3))
        # The prep kernel's l2norm: m = max(|abs_max(x)|, 1); xs = x / m; ss = sum(xs * xs).
        if is_trn2():
            # Trn2 has no abs_max reduction: max(x) and -min(x), then the larger of the two.
            mn = nl.ndarray(mx.shape, dtype=_F32, buffer=nl.sbuf, name=self.nm(f"mn{b}_{ci}"))
            nisa.tensor_reduce(dst=mx, op=nl.maximum, data=xt, axis=2)
            nisa.tensor_reduce(dst=mn, op=nl.minimum, data=xt, axis=2)
            nisa.tensor_scalar(dst=mn, data=mn, op0=nl.multiply, operand0=-1.0, engine=_VEC)
            nisa.tensor_tensor(dst=mx, data1=mx, data2=mn, op=nl.maximum, engine=_VEC)
        else:
            nisa.tensor_reduce(dst=mx, op=nl.abs_max, data=xt, axis=2)
        nisa.activation(dst=mx, op=nl.abs, data=mx)
        nisa.tensor_scalar(dst=mx, data=mx, op0=nl.maximum, operand0=1.0, engine=_VEC)
        nisa.reciprocal(dst=rcp, data=mx)
        xs = self.xs[:, self.next_ring(self.ring3, "xs") % _RING3, 0:ns, :]
        nisa.tensor_tensor(
            dst=xs, data1=xt, data2=rcp.expand_dim(2).broadcast(2, _HEAD_DIM), op=nl.multiply, engine=_VEC
        )
        nisa.tensor_tensor(dst=xs, data1=xs, data2=xs, op=nl.multiply, engine=_VEC)
        nisa.tensor_reduce(dst=ss, op=nl.add, data=xs, axis=2)
        if ci == 5:
            self.defer(0.2, ("den", b), lambda: self.den(b))

    def den(self, b):
        """1 / sqrt(ss + eps / m^2) of all q and k rows of the block (one ScalarE sqrt)."""
        t0, ns = self.blocks[b]
        par = b % 2
        for ci in range(6):
            self.force_tile(b, ci, "ds1")
        mx, ss, den = (self.st[:, par, k, :, 0:ns] for k in (0, 2, 3))
        nisa.tensor_tensor(dst=den, data1=mx, data2=mx, op=nl.multiply, engine=_VEC)
        nisa.reciprocal(dst=den, data=den)
        nisa.tensor_scalar(dst=den, data=den, op0=nl.multiply, operand0=_L2_EPS, engine=_VEC)
        nisa.tensor_tensor(dst=den, data1=ss, data2=den, op=nl.add, engine=_VEC)
        nisa.activation(dst=den, op=nl.sqrt, data=den)
        nisa.reciprocal(dst=den, data=den)

    def normalized(self, b, ci):
        """xs * den of head tile ``ci`` (q: also times D ** -0.5) as token-major fp32 rows in an ``xs`` slot."""
        t0, ns = self.blocks[b]
        par = b % 2
        self.force(("den", b))
        rcp, den = self.st[:, par, 1, ci, 0:ns], self.st[:, par, 3, ci, 0:ns]
        xs = self.xs[:, self.next_ring(self.ring3, "xs") % _RING3, 0:ns, :]
        bcast = (rcp.expand_dim(2).broadcast(2, _HEAD_DIM), den.expand_dim(2).broadcast(2, _HEAD_DIM))
        nisa.tensor_tensor(dst=xs, data1=self.xb[:, ci, 0:ns, :], data2=bcast[0], op=nl.multiply, engine=_VEC)
        nisa.tensor_tensor(dst=xs, data1=xs, data2=bcast[1], op=nl.multiply, engine=_VEC)
        if ci < 3:
            nisa.tensor_scalar(dst=xs, data=xs, op0=nl.multiply, operand0=float(_HEAD_DIM) ** -0.5, engine=_VEC)
        return xs

    def store_channel_major(self, b, rows, out, h):
        """Transpose token-major fp32 rows [128, ns, 128] (fp32 TensorE transposes) and store ``out[h, :, block]``."""
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        bank = self.psT[self.n_t32 % 2][:, 0:N]
        evac_on_act = self.n_t32 % 2 == 0  # the two banks' copies alternate ScalarE / VectorE
        self.n_t32 += 1
        for s in range(ns):
            _pe_transpose(bank[:, s * _P_MAX : (s + 1) * _P_MAX], rows[:, s, :], self.id32)
        stg = self.ostg[:, self.next_ring(self.ring3, "ostg") % _RING3, 0:N]
        # Row s of slot p is token ns * p + s of the block.
        dst = stg.reshape_dim(1, (_P_MAX, ns)).permute((0, 2, 1))
        src = bank.reshape_dim(1, (ns, _P_MAX))
        if evac_on_act:
            nisa.activation(dst=dst, op=nl.copy, data=src)
        else:
            nisa.tensor_copy(dst=dst, src=src, engine=_VEC)
        nisa.dma_copy(
            dst=out.ap(pattern=[[self.T, _P_MAX], [1, N]], offset=(self.h0 + h) * _HEAD_DIM * self.T + t0),
            src=stg,
            **_DMA_STORE,
        )

    def beta_of(self, b):
        """[128, slots, H] beta of block ``b`` (block 0's own copy: its prep runs at the end)."""
        return self.beta0 if b == 0 else self.beta[:, b % 2]

    def ds2_q(self, b, h):
        self.store_channel_major(b, self.normalized(b, h), self.qT, h)

    def ds2_k(self, b, h):
        t0, ns = self.blocks[b]
        kn = self.normalized(b, 3 + h)
        self.store_channel_major(b, kn, self.kT, h)
        beta = self.beta_of(b)[:, 0:ns, h]
        nisa.tensor_tensor(
            dst=kn, data1=kn, data2=beta.expand_dim(2).broadcast(2, _HEAD_DIM), op=nl.multiply, engine=_VEC
        )
        self.store_channel_major(b, kn, self.kbT, h)

    def ds1_v(self, b, h):
        """vb = v * beta, token-major: bf16 transposes of the conv output and the product."""
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        self.force_tile(b, 6 + h, "silu")
        ps = self.bf_half()
        for s in range(ns):
            _pe_transpose(
                ps[:, s * _P_MAX : (s + 1) * _P_MAX],
                self.cin[:, 6 + h, _CONV_HIST + s : _CONV_HIST + N : ns],
                self.id_bf,
            )
        vbt = self.vbs[:, self.next_ring(self.ring2, "vbs") % _RING2, 0:ns, :]
        beta = self.beta_of(b)[:, 0:ns, h]
        nisa.tensor_tensor(
            dst=vbt,
            data1=ps[:, 0:N].reshape_dim(1, (ns, _P_MAX)),
            data2=beta.expand_dim(2).broadcast(2, _HEAD_DIM),
            op=nl.multiply,
            engine=_VEC,
        )
        nisa.dma_copy(
            dst=self.vb.ap(
                pattern=[[ns * _HEAD_DIM, _P_MAX], [1, ns * _HEAD_DIM]],
                offset=((self.h0 + h) * self.T + t0) * _HEAD_DIM,
            ),
            src=vbt,
            **_DMA_STORE,
        )

    # ------------------------------------------------------------------ G phase

    def g_phase(self, b):
        """g|b token-major (4 token tiles), with the next block's x^T built chunk by chunk behind it."""
        if self.mx in (1, 3):
            return self._g_phase_mx(b)
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        nb = len(self.blocks)
        nxt = b + 1 if b + 1 < nb else None
        # The G accumulators take all four matmul banks: f_b goes first.
        for part in range(2):
            self.force(("fb", b, part))
        step_us = ns * _GB_COLS * _US_PER_COL + (self.blocks[nxt][1] * _US_PER_TRANSPOSE if nxt is not None else 0.0)
        cast_state = [0]  # mx=2: the next x^T chunk of block nxt to cast to e4m3, under these matmuls
        for j, kc in enumerate(self.korder):
            slot = self.gb_stream.slot(b * self.gb_per_block + j // _GB_GROUP)
            mov = self.gbr[:, slot, j % _GB_GROUP, :]
            for s in range(ns):
                self.last_matmul = nisa.nc_matmul(
                    dst=self.psM[_G_BANK[s]][:, 0:_GB_COLS],
                    stationary=self.xT[:, kc, s:N:ns],
                    moving=mov,
                    accumulate=(j > 0),
                )
            if nxt is not None:
                self.xT_chunk(nxt, j)
                if self.mx == 2:
                    self._cast_fp8_next(nxt, j, cast_state)
            self.tick(step_us)
        self.after_g(b)

    def after_g(self, b):
        """g out; beta = sigmoid(bf16(b)); the gate; then the block's remaining prep (next block's W phase)."""
        t0, ns = self.blocks[b]
        par = b % 2
        H, D = _HEADS, _HEAD_DIM
        self.force_blocks(b - 2)  # beta of this parity
        for s in [s for s in _G_EVAC_ORDER if s < ns]:
            gs = self.gst[:, self.next_ring(self.ring2, "gst") % _RING2, :]
            nisa.activation(dst=gs, op=nl.copy, data=self.psM[_G_BANK[s]][:, 0:_PROJ])
            nisa.dma_copy(
                dst=self.g.ap(pattern=[[ns * _PROJ, _P_MAX], [1, _PROJ]], offset=(t0 + s) * _PROJ), src=gs, **_DMA_STORE
            )
            src = self.psM[_G_BANK[s]][:, _PROJ:_GB_COLS]
            if self.masked:
                nisa.tensor_scalar(
                    dst=self.bl[:, s, :],
                    data=src,
                    op0=nl.multiply,
                    operand0=self.keep[:, par, 0, s : s + 1],
                    op1=nl.add,
                    operand1=self.keep[:, par, 1, s : s + 1],
                    engine=_VEC,
                )
            else:
                nisa.tensor_copy(dst=self.bl[:, s, :], src=src, engine=_VEC)
        nisa.activation(dst=self.beta[:, par, 0:ns, :], op=nl.sigmoid, data=self.bl[:, 0:ns, :])
        if b == 0:
            nisa.tensor_copy(dst=self.beta0[:, 0:ns, :], src=self.beta[:, par, 0:ns, :], engine=_VEC)
        # gtok = lower_bound * sigmoid(a * (z + dt_bias)), per head (the prep kernel's steps).
        for h in range(H):
            zf = self.zf[:, self.next_ring(self.ring2, "zf") % _RING2, 0:ns, :]
            nisa.tensor_tensor(
                dst=zf,
                data1=self.zb[:, 0:ns, h * D : (h + 1) * D],
                data2=self.dtb[:, h : h + 1, :].broadcast(1, ns),
                op=nl.add,
                engine=_VEC,
            )
            nisa.activation(dst=zf, op=nl.sigmoid, data=zf, scale=self.a_sb[:, h : h + 1])
            nisa.tensor_scalar(dst=zf, data=zf, op0=nl.multiply, operand0=self.lb, engine=_VEC)
            nisa.dma_copy(
                dst=self.gtok.ap(pattern=[[ns * D, _P_MAX], [1, ns * D]], offset=(h * self.T + t0) * D),
                src=zf,
                **_DMA_STORE,
            )
        if b > 0:
            self.ds_tail(b)

    def ds_tail(self, b):
        """Schedule the block's prep that needs beta: vb, and the channel-major q, k, kb."""
        for h in range(_HEADS):
            self.defer(0.15 + 0.15 * h, ("v", b, h), lambda h=h: self.ds1_v(b, h))
        for h in range(_HEADS):
            self.defer(0.6 + 0.15 * h, ("q", b, h), lambda h=h: self.ds2_q(b, h))
        for h in range(_HEADS):
            self.defer(1.05 + 0.2 * h, ("k", b, h), lambda h=h: self.ds2_k(b, h))

    # ------------------------------------------------------------------ top level

    def run(self):
        self.load_constants()
        if self.mx:
            # All of block 0's x^T up front: the MX x^T of a block is quantized once, before its W phase.
            for j in range(self.KC):
                self.xT_chunk(0, j)
        else:
            # Block 0's first x^T chunks; the rest are built inside its first projection tile.
            for j in range(min(_X_LEAD, self.KC)):
                self.xT_chunk(0, j)
        for b in range(len(self.blocks)):
            if self.masked:
                self.block_masks(b)
            # Both forms quantize block 0 up front; blocks 1+ are quantized under the previous block's g-phase.
            if self.mx and b == 0:
                self.quantize_block(b)
            for pos in range(_W_TILES):
                self.w_tile(b, pos)
            self.g_phase(b)
        self.flush()
        self.block0_prep()
        self.flush()
        # The new conv state: the 3 pre-conv rows before row valid_len (old state rows when valid_len < 3).
        rows = 3 * _PROJ
        pattern = [[_CONV_HIST + self.T, rows], [1, _CONV_HIST]]
        if self.masked:
            src = self.pre.ap(pattern=pattern, offset=0, scalar_offset=self.vl_sb, indirect_dim=2)
        else:
            src = self.pre.ap(pattern=pattern, offset=self.T)
        nisa.dma_copy(dst=self.cs_out.ap(pattern=[[_CONV_HIST, rows], [1, _CONV_HIST]], offset=0), src=src, **_DMA_LOAD)
        return self.qT, self.kT, self.kbT, self.vb, self.gtok, self.g, self.cs_out
