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

"""The KDA front of the block kernel: the vendored ``kimi_k3_kda_front`` with its x^T from a gathered image.

``_KdaBlockFront`` changes three parts of ``_KdaFront`` and keeps all other steps:

1. x^T. The block kernel all-gathers the hidden-major image of the rows of all ranks
   (``block_cc.gather_rows_feature_major``). The front loads each 128-row tile of a block with one DMA
   (2 KiB runs in HBM, 256 B runs in SBUF) into the x^T buffer. It needs no x row loads,
   no TensorE transposes and no PSUM copies for x.
2. f_a. The block kernel computes f_a on the local rows before the gather, with the same matmul order.
   The front loads the bf16 f_a^T of each block from the image and skips the f_a W tile.
3. Pad rows. Every pad-row mask is a select (``tensor_copy_predicated``), not a multiply. q, k and v get
   0, the z and b logits get the bf16 minimum. A NaN or Inf in a pad row thus cannot reach a valid row or
   the state through a 0 x NaN product.

All other steps, and so every output bit of a valid row, stay those of the vendored kernel.

With ``valid_start`` (a prefix-cache hit inside the segment), the rows before it are pad rows too, and the
conv history of rows vs .. vs + 2 is the old conv state (the pre-conv rows of the cached tokens vs - 3 ..
vs - 1). The kernel puts it in place with static SBUF addresses only: two scalar-offset HBM writes put
the conv state at the columns ``vs % 512`` and ``vs % 512 + 512`` of an HBM scratch, one load makes
it an SBUF tile with the columns of a 512-row block, and each conv tile of each block selects the
window columns from it (a per-block predicate) after the projection and before the conv taps. The
window can end in the last columns of one block and go on in the next: the earlier block selects its
part, and the history copy of the conv carries it into the next block. A head group of a multi-group block
takes its q / k / v channels of the full-width conv state.

With ``lead_outputs``, the front also returns ``conv_lead``, the conv history of row g (the pre-conv rows of
the tokens g - 3 .. g - 1, at column g of the pre-conv scratch): one more scalar-offset read of the scratch,
after the window rows are in it, for g = max(vs, ((valid_len - 1) // 128) * 128), the row of the scan's
``S_lead``.
"""

import nki.isa as nisa
import nki.language as nl

from .kda_front import (
    _BF16,
    _BF16_MIN,
    _BLOCK,
    _CONV_HIST,
    _CONV_TAPS,
    _CONV_TILES,
    _DMA_LOAD,
    _DMA_STORE,
    _F32,
    _FA_TILE,
    _G_BANK,
    _G_EVAC_ORDER,
    _GB_GROUP,
    _HEAD_DIM,
    _HEADS,
    _I32,
    _P_MAX,
    _PROJ,
    _RING3,
    _SLOTS,
    _U8,
    _US_PER_COL,
    _VEC,
    _W_GROUP,
    _W_ORDER,
    _copy_rows,
    _KdaFront,
    _runs,
    _Stream,
)

# The W tiles of the block front: the conv tiles only (f_a comes from the image).
_W_ORDER_BLOCK = tuple(t for t in _W_ORDER if t != _FA_TILE)
# Hidden chunks of x^T per image DMA. One DMA moves cnt * 256 B contiguous on both sides.
_X_GROUP_BLOCK = 8
# Rows of one row half of a 128-row tile (ag_row_halves).
_HALF = _P_MAX // 2
# Columns of a block's conv input (the history and a full block), and of the conv-state window scratch: the state at
# the columns r and r + 512 for r = valid_start % 512, so r + 512 + 2 <= 1025.
_CIN_COLS = _CONV_HIST + _BLOCK
_HIST_COLS = _CONV_HIST + 2 * _BLOCK - 1


def _mod_steps(limit: int, unit: int) -> tuple:
    """The steps of a remainder ladder: ``unit * 2**k`` for every k with ``unit * 2**k < limit``, largest first.

    Subtract each step from v where v >= step. For any integer ``0 <= v < limit`` this leaves ``v % unit``: the
    largest step s is at least ``limit / 2``, so v < 2 s, and after the step of each size v is below that size.

    Example:
        >>> _mod_steps(8192, 512)
        (4096, 2048, 1024, 512)
    """
    steps, step = [], unit
    while step < limit:
        steps.append(step)
        step *= 2
    return tuple(reversed(steps))


class _Shape:
    """A stand-in for the x input of ``_KdaFront``: the front reads only its shape and dtype."""

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class _KdaBlockFront(_KdaFront):
    """The front with x^T and f_a^T from a gathered hidden-major image (see the module docstring).

    Args:
        img: The gathered image pieces, a list of ``(c0, c1, tensor)``. Piece ``(c0, c1, t)`` holds the hidden
            chunks ``c0 .. c1 - 1``: ``t [W * 128, nT * n * 128 (+ Tl)]`` bf16 HBM with ``n = c1 - c0``. Row
            ``r * 128 + p`` holds rank ``r``; column ``(i * n + kc - c0) * 128 + t`` holds
            ``x[r * Tl + i * 128 + t, kc * 128 + p]``. The last piece also holds ``f_a[r * Tl + i * 128 + t, p]``
            (bf16) at column ``nT * n * 128 + i * 128 + t``. With more than one piece, block 0 runs its W tiles
            one piece after the other (a K-pass), and it keeps the partial sums in SBUF between the passes.
        rows: Tl, the rows of one rank.
        world: W.
        hidden: HID.
        x_dma_engine: The engine of the image loads (``nisa.engine.sync`` or ``nisa.engine.scalar``).
        valid_start: Optional [1, 1] int32 HBM tensor. The rows before it are pad rows, and ``conv_state`` is the
            conv history of row ``valid_start`` (see the module docstring). Needs ``valid_len`` and
            ``block0_in_order``, and no ``row_halves``. None gives the program without it.
        lead_outputs: Also return ``conv_lead``, the pre-conv rows of the tokens g - 3 .. g - 1 (the conv history of
            row g), as [3, 384, 3] bf16 (or the group's channels of ``lead_out``), for
            g = max(vs, ((max(valid_len, 1) - 1) // 128) * 128) with vs the clamped valid_start (0 without one): the
            start of the last 128-row chunk with a valid row, the row of the scan's S_lead. Needs ``valid_len``.
            False gives the program without it.
        lead_out: With ``lead_outputs`` and ``out_hbm``: the shared [3, hc * 128, 3] bf16 conv_lead output.
        The other arguments are those of ``_KdaFront``.
    """

    def __init__(
        self,
        img,
        rows,
        world,
        hidden,
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
        x_dma_engine=nisa.engine.sync,
        block0_in_order=True,
        zgate_slots=2,
        row_halves=0,
        halves=None,
        valid_start=None,
        mx=0,
        w_qkv_mx=None,
        w_qkv_scale=None,
        w_gb_mx=None,
        w_gb_scale=None,
        out_hbm=None,
        head_group=0,
        p_full=None,
        lead_outputs=False,
        lead_out=None,
    ):
        self.pieces, self.Tl, self.W = list(img), rows, world
        # Row-half images (kda_block_prefill.gather_input_halves): the first 2 * row_halves blocks project their
        # first-half rows in a pre-pass (pass_a), as block pairs, while the second half's AG moves.
        self.row_halves = int(row_halves)
        self.halves = list(halves) if self.row_halves else None
        # The conv state is an input of the block kernel, ready at its start (the caller reads the state page before
        # the call). So block 0 runs its conv and prep in order, like the other blocks; the vendored front
        # defers them to its end for a late state.
        self.block0_in_order = bool(block0_in_order)
        self.nT = rows // _P_MAX
        self.x_dma = {"dge_mode": nisa.dge_mode.hwdge, "engine": x_dma_engine}
        self.w_order = _W_ORDER_BLOCK
        x = _Shape((world * rows, hidden), _BF16)
        super().__init__(
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
            zgate_slots,
            mx=mx,
            w_qkv_mx=w_qkv_mx,
            w_qkv_scale=w_qkv_scale,
            w_gb_mx=w_gb_mx,
            w_gb_scale=w_gb_scale,
            out_hbm=out_hbm,
            head_group=head_group,
            p_full=p_full,
        )
        # The vendored front keeps its outputs and its pre-conv scratch in shared_hbm (the outputs of a standalone
        # call). Here they are scratch of the block kernel, with different data on each core: keep them private, so
        # NKI_LNC1_SHARED_HBM=1 (one buffer for the two cores of an HBM stack) cannot make the cores race. The
        # new conv state stays shared_hbm: it is an output of the block kernel. With shared full-width outputs only
        # the per-group pre-conv scratch is re-wrapped; the outputs are the shared tensors.
        rewrap = ("pre",) if out_hbm is not None else ("qT", "kT", "kbT", "vb", "gtok", "g", "pre")
        for attr in rewrap:
            t = getattr(self, attr)
            setattr(self, attr, nl.ndarray(t.shape, dtype=t.dtype, buffer=nl.private_hbm, name=self.nm(f"{attr}_p")))
        if self.masked:
            # Token-major pad predicate [128, parity, slot, 384]: row t0 + s + ns * p of the block.
            self.tokpred = nl.ndarray((_P_MAX, 2, _SLOTS, _PROJ), dtype=_U8, buffer=nl.sbuf, name=self.nm("tokpred"))
            self.zero_proj = nl.ndarray((_P_MAX, _PROJ), dtype=_F32, buffer=nl.sbuf, name=self.nm("zproj"))
        if self.row_halves:
            # The first-half pre-conv rows of one pre-pass block, reloaded from the scratch before its convs.
            self.h_a = nl.ndarray(
                (_P_MAX, _CONV_TILES, _SLOTS * _HALF), dtype=_BF16, buffer=nl.sbuf, name=self.nm("h_a")
            )
        self.valid_start = valid_start
        self.vs_on = valid_start is not None
        if self.vs_on:

            def sb(shape, tag, dtype=_F32):
                return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf, name=self.nm(tag))

            self.vs_f = sb((_P_MAX, 1), "vs_f")  # valid_start clamped to [0, valid_len], on every partition
            self.vs_i = sb((1, 1), "vs_i", _I32)  # the same, the offset of the conv_new state write
            self.vs_r = sb((1, 1), "vs_r", _I32)  # valid_start % 512, the window column in its block
            self.ramp_w = sb((_P_MAX, _CIN_COLS), "ramp_w")  # ramp_w[p, c] = c over a block's conv input columns
            # Per block parity: 1 on the conv input columns of the tokens valid_start - 3 .. valid_start - 1.
            self.wpred = sb((_P_MAX, 2, _CIN_COLS), "wpred", _U8)
            # The window source: the conv state at the columns of the window in its block (other columns unused).
            self.hsb = sb((_P_MAX, _CONV_TILES, _CIN_COLS), "hsb", _BF16)
            self.hist = nl.ndarray(
                (_CONV_TILES * _P_MAX, _HIST_COLS), dtype=_BF16, buffer=nl.private_hbm, name=self.nm("hist")
            )
        self.lead_on = bool(lead_outputs)
        if self.lead_on:
            # g on every partition (fp32, exact small integers), and as the int32 offset of the conv_lead read.
            self.lead_f = nl.ndarray((_P_MAX, 1), dtype=_F32, buffer=nl.sbuf, name=self.nm("lead_f"))
            self.lead_i = nl.ndarray((1, 1), dtype=_I32, buffer=nl.sbuf, name=self.nm("lead_i"))
            if self.grouped:
                self.cl_out = lead_out
            else:
                self.cl_out = nl.ndarray(
                    (3, _PROJ, _CONV_HIST), dtype=_BF16, buffer=nl.shared_hbm, name=self.nm("conv_lead")
                )

    # ------------------------------------------------------------------ streams

    def _build_streams(self):
        """The weight streams without the f_a tile; x^T comes from the image (``load_x``)."""
        korder, KC = self.korder, self.KC
        if self.mx:
            # FP8 keeps the conv weights resident (load_mx_weights); no bf16 conv-weight stream. mx=1 also keeps g|b
            # resident; mx=2 leaves g|b bf16, so it still streams w_gb.
            self.w_index = {}
            self.xs_stream = None
            self.w_stream = _Stream(self.wr, [])
            self.w_per_tile = -(-KC // _W_GROUP)
            self.gb_per_block = -(-KC // _GB_GROUP)
            gl = []
            if self.mx == 2:

                def gb_loader_mx2(chunks):
                    def load(ring, slot):
                        for pos, c0, cnt in _runs(chunks):
                            nisa.dma_copy(
                                dst=ring[:, slot, pos : pos + cnt, :],
                                src=self.w_gb.ap(
                                    pattern=[[KC * (_PROJ + _HEADS), _P_MAX], [1, cnt * (_PROJ + _HEADS)]],
                                    offset=c0 * (_PROJ + _HEADS),
                                ),
                                **_DMA_LOAD,
                            )

                    return load

                for _b in range(len(self.blocks)):
                    for j0 in range(0, KC, _GB_GROUP):
                        gl.append(gb_loader_mx2(korder[j0 : j0 + _GB_GROUP]))
            self.gb_stream = _Stream(self.gbr, gl)
            return

        def w_loader(tile, chunks):
            def load(ring, slot):
                for pos, c0, cnt in _runs(chunks):
                    nisa.dma_copy(
                        dst=ring[:, slot, pos : pos + cnt, :],
                        src=self.w_qkvf.ap(
                            pattern=[[KC * _P_MAX, _P_MAX], [1, cnt * _P_MAX]],
                            offset=(self.w_tile_base + tile) * _P_MAX * KC * _P_MAX + c0 * _P_MAX,
                        ),
                        **_DMA_LOAD,
                    )

            return load

        def gb_loader(chunks):
            def load(ring, slot):
                for pos, c0, cnt in _runs(chunks):
                    if self.grouped:
                        # Group-major w_gb [128, KC, ng * 387]: read this group's 387 columns of each hidden chunk.
                        nisa.dma_copy(
                            dst=ring[:, slot, pos : pos + cnt, :],
                            src=self.w_gb.ap(
                                pattern=[[KC * self.gb_full, _P_MAX], [self.gb_full, cnt], [1, _PROJ + _HEADS]],
                                offset=c0 * self.gb_full + self.gb_col,
                            ),
                            **_DMA_LOAD,
                        )
                    else:
                        nisa.dma_copy(
                            dst=ring[:, slot, pos : pos + cnt, :],
                            src=self.w_gb.ap(
                                pattern=[[KC * (_PROJ + _HEADS), _P_MAX], [1, cnt * (_PROJ + _HEADS)]],
                                offset=c0 * (_PROJ + _HEADS),
                            ),
                            **_DMA_LOAD,
                        )

            return load

        wl, gl = [], []
        # Block 0 with K-pass pieces: the loaders in the order of use (piece, tile, chunk group).
        self.w_index = {}
        # The row-half pre-pass runs first: the conv tiles of each block pair.
        for p in range(self.row_halves):
            for pos, tile in enumerate(self.w_order):
                self.w_index[("A", p, pos)] = len(wl)
                for j0 in range(0, KC, _W_GROUP):
                    wl.append(w_loader(tile, korder[j0 : j0 + _W_GROUP]))
        for b, _ in enumerate(self.blocks):
            if b == 0 and len(self.pieces) > 1:
                for pi, (c0, c1, _) in enumerate(self.pieces):
                    for pos, tile in enumerate(self.w_order):
                        self.w_index[(0, pi, pos)] = len(wl)
                        for j0 in range(c0, c1, _W_GROUP):
                            wl.append(w_loader(tile, korder[j0 : min(j0 + _W_GROUP, c1)]))
            else:
                for pos, tile in enumerate(self.w_order):
                    self.w_index[(b, 0, pos)] = len(wl)
                    for j0 in range(0, KC, _W_GROUP):
                        wl.append(w_loader(tile, korder[j0 : j0 + _W_GROUP]))
            for j0 in range(0, KC, _GB_GROUP):
                gl.append(gb_loader(korder[j0 : j0 + _GB_GROUP]))
        self.xs_stream = None
        self.w_stream = _Stream(self.wr, wl)
        self.gb_stream = _Stream(self.gbr, gl)
        self.w_per_tile = -(-KC // _W_GROUP)
        self.gb_per_block = -(-KC // _GB_GROUP)

    def _tile_src(self, b, s):
        """``(rank, local tile)`` of slot ``s`` of block ``b``."""
        t0, _ = self.blocks[b]
        g = t0 // _P_MAX + s
        return g // self.nT, g % self.nT

    def _load_half(self, b, c0, cnt, h, col):
        """DMA row half ``h`` of the chunks ``c0 .. c0 + cnt - 1`` of block ``b``'s slots into x^T columns
        ``col(s) .. col(s) + 63`` (``ag_row_halves``)."""
        _, ns = self.blocks[b]
        img = self.halves[h]
        width = img.shape[1]
        for s in range(ns):
            r, i = self._tile_src(b, s)
            nisa.dma_copy(
                dst=self.xT[:, c0 : c0 + cnt, col(s) : col(s) + _HALF],
                src=img.ap(
                    pattern=[[width, _P_MAX], [_HALF, cnt], [1, _HALF]],
                    offset=r * _P_MAX * width + (i * self.KC + c0) * _HALF,
                ),
                **self.x_dma,
            )

    def _in_pass_a(self, b):
        """Whether block ``b`` projects its first-half rows in the pre-pass (ag_row_halves)."""
        return b < 2 * self.row_halves

    def load_x(self, b, c0, cnt):
        """DMA the hidden chunks ``c0 .. c0 + cnt - 1`` of block ``b`` from the image pieces into x^T."""
        _, ns = self.blocks[b]
        if self.row_halves:
            for h in range(2):
                self._load_half(b, c0, cnt, h, lambda s, h=h: s * _P_MAX + h * _HALF)
            return
        for p0, p1, img in self.pieces:
            lo, hi = max(c0, p0), min(c0 + cnt, p1)
            if lo >= hi:
                continue
            width, n = img.shape[1], p1 - p0
            for s in range(ns):
                r, i = self._tile_src(b, s)
                nisa.dma_copy(
                    dst=self.xT[:, lo:hi, s * _P_MAX : (s + 1) * _P_MAX],
                    src=img.ap(
                        pattern=[[width, _P_MAX], [_P_MAX, hi - lo], [1, _P_MAX]],
                        offset=r * _P_MAX * width + (i * n + lo - p0) * _P_MAX,
                    ),
                    **self.x_dma,
                )

    def _mx_ready_chunks(self, built_upto):
        """How many x^T chunks are loaded: image DMAs land on ``_X_GROUP_BLOCK`` boundaries, the last one at KC."""
        if built_upto >= self.KC - 1:
            return self.KC
        return (built_upto + 1) // _X_GROUP_BLOCK * _X_GROUP_BLOCK

    def xT_chunk(self, bn, j):
        """Refill x^T for block ``bn`` behind the G-phase reads: one image DMA per group of chunks.

        The G phase of the previous block reads chunk ``j`` just before this call. A group DMA writes its
        chunks only after the last read of the group, so no read sees the next block's rows.
        """
        if bn == 0:
            return  # block 0 loads at the start (run)
        if (j + 1) % _X_GROUP_BLOCK == 0 or j == self.KC - 1:
            c0 = (j // _X_GROUP_BLOCK) * _X_GROUP_BLOCK
            self.load_x(bn, c0, j + 1 - c0)

    def _xt_mov(self, kc, ns):
        """The moving operand of a W tile: x^T of chunk ``kc`` for the block's ``ns`` token tiles."""
        return self.xT[:, kc, 0 : ns * _P_MAX]

    def _xt_mov_half_b(self, kc, ns):
        """The moving operand of a pre-pass block's W tile: the second-half rows of its slots, ``[128, ns, 64]``."""
        return self.xT.ap(pattern=[[self.KC * _BLOCK, _P_MAX], [_P_MAX, ns], [1, _HALF]], offset=kc * _BLOCK + _HALF)

    def _xt_stat(self, kc, s, ns):
        """The stationary operand of G slot ``s``: rows t0 + s + ns * m (m = 0 .. 127) of chunk ``kc``."""
        return self.xT[:, kc, s : ns * _P_MAX : ns]

    # ------------------------------------------------------------------ masks

    def load_constants(self):
        super().load_constants()
        if self.masked:
            nisa.memset(dst=self.zero_proj, value=0.0, engine=_VEC)
        if self.block0_in_order:
            # Block 0's conv history (the old state), and the old state rows of the pre-conv scratch (for a new
            # conv state with valid_len < 3), as block0_prep of the vendored front writes them.
            for ci in range(9):
                i3, h = divmod(ci, 3)
                row = i3 * _PROJ + h * _P_MAX  # pre scratch row (per group, 384 wide)
                row_in = i3 * self.pf + (self.h0 + h) * _P_MAX  # conv_state input row (full width)
                nisa.dma_copy(
                    dst=self.cin[:, ci, 0:_CONV_HIST],
                    src=self.conv_state.ap(pattern=[[_CONV_HIST, _P_MAX], [1, _CONV_HIST]], offset=row_in * _CONV_HIST),
                    **_DMA_LOAD,
                )
                nisa.dma_copy(
                    dst=self.pre.ap(
                        pattern=[[_CONV_HIST + self.T, _P_MAX], [1, _CONV_HIST]], offset=row * (_CONV_HIST + self.T)
                    ),
                    src=self.cin[:, ci, 0:_CONV_HIST],
                    **_DMA_STORE,
                )
        if self.vs_on:
            self.load_valid_start()
        if self.lead_on:
            self.lead_row()

    def lead_row(self):
        """g of ``conv_lead``: max(vs, ((max(valid_len, 1) - 1) // 128) * 128), on every partition and as an int32."""
        g = self.lead_f
        nisa.tensor_scalar(
            dst=g, data=self.vl_f, op0=nl.maximum, operand0=1.0, op1=nl.subtract, operand1=1.0, engine=_VEC
        )
        # g % 128 (g < T): subtract the steps of _mod_steps(T, 128) where they fit (exact small integers in fp32).
        r = nl.ndarray((_P_MAX, 1), dtype=_F32, buffer=nl.sbuf, name=self.nm("lead_r"))
        step = nl.ndarray((_P_MAX, 1), dtype=_F32, buffer=nl.sbuf, name=self.nm("lead_step"))
        nisa.tensor_copy(dst=r, src=g, engine=_VEC)
        for th in _mod_steps(self.T, _P_MAX):
            nisa.tensor_scalar(
                dst=step,
                data=r,
                op0=nl.greater_equal,
                operand0=float(th),
                op1=nl.multiply,
                operand1=float(th),
                engine=_VEC,
            )
            nisa.tensor_tensor(dst=r, data1=r, data2=step, op=nl.subtract, engine=_VEC)
        nisa.tensor_tensor(dst=g, data1=g, data2=r, op=nl.subtract, engine=_VEC)
        if self.vs_on:
            nisa.tensor_tensor(dst=g, data1=g, data2=self.vs_f, op=nl.maximum, engine=_VEC)
        nisa.tensor_copy(dst=self.lead_i, src=g[0:1, 0:1], engine=_VEC)

    def load_valid_start(self):
        """valid_start (clamped to [0, valid_len]), its column in its block, and the conv-state window source."""
        rows = _CONV_TILES * _P_MAX
        vs_raw = nl.ndarray((_P_MAX, 1), dtype=_I32, buffer=nl.sbuf, name=self.nm("vs_raw"))
        nisa.dma_copy(dst=vs_raw, src=self.valid_start.ap(pattern=[[0, _P_MAX], [1, 1]], offset=0), **_DMA_LOAD)
        nisa.tensor_copy(dst=self.vs_f, src=vs_raw, engine=_VEC)
        nisa.tensor_scalar(
            dst=self.vs_f, data=self.vs_f, op0=nl.maximum, operand0=0.0, op1=nl.minimum, operand1=self.vl_f, engine=_VEC
        )
        nisa.tensor_copy(dst=self.vs_i, src=self.vs_f[0:1, 0:1], engine=_VEC)
        # r = vs % 512 (vs < T): subtract the steps of _mod_steps(T, 512) where they fit (exact small integers in fp32).
        r_f = nl.ndarray((1, 1), dtype=_F32, buffer=nl.sbuf, name=self.nm("vs_rf"))
        step = nl.ndarray((1, 1), dtype=_F32, buffer=nl.sbuf, name=self.nm("vs_step"))
        nisa.tensor_copy(dst=r_f, src=self.vs_f[0:1, 0:1], engine=_VEC)
        for th in _mod_steps(self.T, _BLOCK):
            nisa.tensor_scalar(
                dst=step,
                data=r_f,
                op0=nl.greater_equal,
                operand0=float(th),
                op1=nl.multiply,
                operand1=float(th),
                engine=_VEC,
            )
            nisa.tensor_tensor(dst=r_f, data1=r_f, data2=step, op=nl.subtract, engine=_VEC)
        nisa.tensor_copy(dst=self.vs_r, src=r_f, engine=_VEC)
        # The conv state at the columns r .. r + 2 of the scratch (column c of a block's conv input is token
        # t0 + c - 3), and at r + 512 .. r + 514 for a window that starts in the last columns of the block before.
        # The other columns are never selected.
        for off in (0, _BLOCK) if self.T > _BLOCK else (0,):
            if not self.grouped:
                nisa.dma_copy(
                    dst=self.hist.ap(
                        pattern=[[_HIST_COLS, rows], [1, _CONV_HIST]],
                        offset=off,
                        scalar_offset=self.vs_r,
                        indirect_dim=1,
                    ),
                    src=self.conv_state.ap(pattern=[[_CONV_HIST, rows], [1, _CONV_HIST]], offset=0),
                    **_DMA_STORE,
                )
                continue
            # A head group: its q / k / v channels of the full-width conv state (rows i3 * p + 128 * (3 g + h)).
            for i3 in range(3):
                nisa.dma_copy(
                    dst=self.hist.ap(
                        pattern=[[_HIST_COLS, _PROJ], [1, _CONV_HIST]],
                        offset=i3 * _PROJ * _HIST_COLS + off,
                        scalar_offset=self.vs_r,
                        indirect_dim=1,
                    ),
                    src=self.conv_state.ap(
                        pattern=[[_CONV_HIST, _PROJ], [1, _CONV_HIST]],
                        offset=(i3 * self.pf + self.h0 * _P_MAX) * _CONV_HIST,
                    ),
                    **_DMA_STORE,
                )
        nisa.dma_copy(
            dst=self.hsb,
            src=self.hist.ap(
                pattern=[[_HIST_COLS, _P_MAX], [_P_MAX * _HIST_COLS, _CONV_TILES], [1, _CIN_COLS]], offset=0
            ),
            **_DMA_LOAD,
        )
        nisa.tensor_copy(dst=self.ramp_w[:, 0:_BLOCK], src=self.ramp, engine=_VEC)
        nisa.tensor_scalar(
            dst=self.ramp_w[:, _BLOCK:_CIN_COLS],
            data=self.ramp[:, 0:_CONV_HIST],
            op0=nl.add,
            operand0=float(_BLOCK),
            engine=_VEC,
        )

    def vs_masks(self, b):
        """With valid_start: the rows before it are pad rows (keep and kpred), and the block's window predicate."""
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        par = b % 2
        ge = nl.ndarray((_P_MAX, _SLOTS), dtype=_F32, buffer=nl.sbuf, name=self.nm("vs_ge"))
        nisa.tensor_scalar(
            dst=ge[:, 0:ns], data=self.mtok[:, 0:ns], op0=nl.greater_equal, operand0=self.vs_f, engine=_VEC
        )
        keep = self.keep[:, par, 0, 0:ns]
        nisa.tensor_tensor(dst=keep, data1=keep, data2=ge[:, 0:ns], op=nl.multiply, engine=_VEC)
        lt = nl.ndarray((_P_MAX, _BLOCK), dtype=_U8, buffer=nl.sbuf, name=self.nm("vs_lt"))
        nisa.tensor_scalar(
            dst=lt[:, 0:N],
            data=self.ramp[:, 0:N],
            op0=nl.add,
            operand0=float(t0),
            op1=nl.less,
            operand1=self.vs_f,
            engine=_VEC,
        )
        kp = self.kpred[:, par, 0:N]
        nisa.tensor_tensor(dst=kp, data1=kp, data2=lt[:, 0:N], op=nl.maximum, engine=_VEC)
        # Window columns: t0 + c - 3 in [vs - 3, vs). Block 0 also selects in its history columns (vs < 3); a later
        # block's history comes from the history copy of the block before.
        c0, c1 = (0 if b == 0 else _CONV_HIST), _CONV_HIST + N
        wa = nl.ndarray((_P_MAX, _CIN_COLS), dtype=_F32, buffer=nl.sbuf, name=self.nm("vs_wa"))
        wb = nl.ndarray((_P_MAX, _CIN_COLS), dtype=_F32, buffer=nl.sbuf, name=self.nm("vs_wb"))
        nisa.tensor_scalar(
            dst=wa[:, c0:c1],
            data=self.ramp_w[:, c0:c1],
            op0=nl.add,
            operand0=float(t0),
            op1=nl.greater_equal,
            operand1=self.vs_f,
            engine=_VEC,
        )
        nisa.tensor_scalar(
            dst=wb[:, c0:c1],
            data=self.ramp_w[:, c0:c1],
            op0=nl.add,
            operand0=float(t0 - _CONV_HIST),
            op1=nl.less,
            operand1=self.vs_f,
            engine=_VEC,
        )
        nisa.tensor_tensor(
            dst=self.wpred[:, par, c0:c1], data1=wa[:, c0:c1], data2=wb[:, c0:c1], op=nl.multiply, engine=_VEC
        )

    def block_masks(self, b):
        """The vendored masks, then the token-major pad predicate of the z and b selects."""
        super().block_masks(b)
        if self.vs_on:
            self.vs_masks(b)
        _, ns = self.blocks[b]
        par = b % 2
        pad = nl.ndarray((_P_MAX, _SLOTS), dtype=_F32, buffer=nl.sbuf, name=self.nm("padf"))
        nisa.tensor_scalar(
            dst=pad[:, 0:ns],
            data=self.keep[:, par, 0, 0:ns],
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.add,
            operand1=1.0,
            engine=_VEC,
        )
        for s in range(ns):
            nisa.tensor_scalar(
                dst=self.tokpred[:, par, s, :],
                data=self.zero_proj,
                op0=nl.add,
                operand0=pad[:, s : s + 1],
                engine=_VEC,
            )

    def silu_tile(self, b, ci, ca):
        """silu(conv) -> bf16; select 0 in the pad rows of q, k and v; schedule the prep of q and k."""
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        y = self.cin[:, ci, 3 : 3 + N]
        nisa.activation(dst=y, op=nl.silu, data=ca)
        if self.masked:
            nisa.tensor_copy_predicated(dst=y, src=0.0, predicate=self.kpred[:, b % 2, 0:N])
        if ci < 6:
            self.defer(0.2, ("ds1", b, ci), lambda: self.ds1_qk(b, ci))

    def f_b(self, b, slots):
        """z = bf16(f_a) @ W_fb^T, token-major; the bf16 minimum in the pad rows (a select)."""
        _, ns = self.blocks[b]
        N = ns * _P_MAX
        for s in slots:
            dst = self.psM[2 + s % 2][:, 0:_PROJ]
            nisa.nc_matmul(dst=dst, stationary=self.faT[:, s:N:ns], moving=self.wfb, accumulate=False)
            nisa.tensor_copy(dst=self.zb[:, s, :], src=dst, engine=_VEC)
            if self.masked:
                nisa.tensor_copy_predicated(dst=self.zb[:, s, :], src=_BF16_MIN, predicate=self.tokpred[:, b % 2, s, :])

    def conv_tile(self, b, ci, acc):
        """Round, store the pre-conv rows, run the causal conv (block 0 too, with block0_in_order)."""
        if self.vs_on:
            return self.conv_tile_vs(b, ci, acc)
        if not (b == 0 and self.block0_in_order):
            return super().conv_tile(b, ci, acc)
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        i3, h = divmod(ci, 3)
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
        self.conv(b, ci)

    def conv_tile_vs(self, b, ci, acc):
        """``conv_tile`` with valid_start (block0_in_order): round, select the conv-state window, store, conv.

        The pre-conv scratch then holds the window too, so the new conv state of a block that ends within 3 rows
        of valid_start reads the cached rows.
        """
        t0, ns = self.blocks[b]
        N = ns * _P_MAX
        i3, h = divmod(ci, 3)
        if b > 0:
            # As the vendored conv_tile: the previous block's readers and writer of this conv buffer go first.
            self.force_tile(b - 1, ci, "ds1" if ci < 6 else "silu")
            if ci >= 6:
                self.force(("v", b - 1, ci - 6))
        cin = self.cin[:, ci, :]
        _copy_rows(cin[:, _CONV_HIST : _CONV_HIST + N], acc)
        c0, c1 = (0 if b == 0 else _CONV_HIST), _CONV_HIST + N
        nisa.tensor_copy_predicated(
            dst=cin[:, c0:c1], src=self.hsb[:, ci, c0:c1], predicate=self.wpred[:, b % 2, c0:c1]
        )
        nisa.dma_copy(
            dst=self.pre.ap(
                pattern=[[_CONV_HIST + self.T, _P_MAX], [1, N]],
                offset=(i3 * _PROJ + h * _P_MAX) * (_CONV_HIST + self.T) + _CONV_HIST + t0,
            ),
            src=cin[:, _CONV_HIST : _CONV_HIST + N],
            **_DMA_STORE,
        )
        self.conv(b, ci)

    def conv(self, b, ci):
        """The vendored conv steps; with block0_in_order, block 0 also keeps its last rows as the next history."""
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
        if b > 0 or self.block0_in_order:
            # The next block's history: the last 3 pre-conv rows of this block.
            nisa.tensor_copy(dst=cin[:, 0:_CONV_HIST], src=cin[:, N : N + _CONV_HIST], engine=_VEC)
        self.defer(0.2, ("silu", b, ci), lambda: self.silu_tile(b, ci, ca))

    # ------------------------------------------------------------------ W and G phases

    def fa_tile(self, b):
        """Load the block's bf16 f_a^T from the image (in place of the f_a W tile), then schedule f_b."""
        _, ns = self.blocks[b]
        p0, p1, img = self.pieces[-1]
        width = img.shape[1]
        fa_col = self.nT * (p1 - p0) * _P_MAX
        for part in range(2):
            self.force(("fb", b - 1, part))
        if self.row_halves:
            # f_a^T of each row half follows the half image's x^T columns.
            for h in range(2):
                himg = self.halves[h]
                hw = himg.shape[1]
                for s in range(ns):
                    r, i = self._tile_src(b, s)
                    nisa.dma_copy(
                        dst=self.faT[:, s * _P_MAX + h * _HALF : s * _P_MAX + (h + 1) * _HALF],
                        src=himg.ap(
                            pattern=[[hw, _P_MAX], [1, _HALF]],
                            offset=r * _P_MAX * hw + self.nT * self.KC * _HALF + i * _HALF,
                        ),
                        **self.x_dma,
                    )
        else:
            for s in range(ns):
                r, i = self._tile_src(b, s)
                nisa.dma_copy(
                    dst=self.faT[:, s * _P_MAX : (s + 1) * _P_MAX],
                    src=img.ap(pattern=[[width, _P_MAX], [1, _P_MAX]], offset=r * _P_MAX * width + fa_col + i * _P_MAX),
                    **self.x_dma,
                )
        self.defer(0.12, ("fb", b, 0), lambda: self.f_b(b, range(0, min(2, ns))))
        if ns > 2:
            self.defer(0.33, ("fb", b, 1), lambda: self.f_b(b, range(2, ns)))

    def w_tile(self, b, pos):
        """The accumulating matmuls of conv W tile ``w_order[pos]`` of block ``b`` (channel-major output)."""
        if self.mx:
            return self._w_tile_fp8(b, pos) if self.mx == 2 else self._w_tile_mx(b, pos)
        _, ns = self.blocks[b]
        N = ns * _P_MAX
        tile = self.w_order[pos]
        if self.row_halves and self._in_pass_a(b):
            return self._w_tile_half_b(b, pos)
        acc = self.psM[pos % 2][:, 0:N]
        base = self.w_index[(b, 0, pos)]
        step_us = N * _US_PER_COL
        for j, kc in enumerate(self.korder):
            slot = self.w_stream.slot(base + j // _W_GROUP)
            nisa.nc_matmul(
                dst=acc,
                stationary=self.wr[:, slot, j % _W_GROUP, :],
                moving=self._xt_mov(kc, ns),
                accumulate=(j > 0),
            )
            self.tick(step_us)
        self.conv_tile(b, tile, acc)

    # ------------------------------------------------------------------ row halves (ag_row_halves)

    def _cin_half(self, ci, h, ns):
        """Columns of row half ``h`` of the ``ns`` slots of conv input tile ``ci``: ``[128, ns, 64]``."""
        row = _CONV_HIST + _BLOCK
        return self.cin.ap(
            pattern=[[_CONV_TILES * row, _P_MAX], [_P_MAX, ns], [1, _HALF]], offset=ci * row + _CONV_HIST + h * _HALF
        )

    def _pre_half(self, ci, t0, h, ns):
        """The pre-conv scratch rows of conv tile ``ci`` at row half ``h`` of the slots of the block at ``t0``."""
        width = _CONV_HIST + self.T
        return self.pre.ap(
            pattern=[[width, _P_MAX], [_P_MAX, ns], [1, _HALF]],
            offset=ci * _P_MAX * width + _CONV_HIST + t0 + h * _HALF,
        )

    def _h_a_tile(self, ci, ns):
        """Tile ``ci`` of the first-half staging, as ``[128, ns, 64]``."""
        width = _SLOTS * _HALF
        return self.h_a.ap(pattern=[[_CONV_TILES * width, _P_MAX], [_HALF, ns], [1, _HALF]], offset=ci * width)

    def _load_pair(self, p, c0, cnt):
        """The first-half x^T rows of blocks ``2 p`` and ``2 p + 1`` into x^T (block ``2 p + q`` at columns
        ``256 q + 64 s .. 256 q + 64 s + 63`` for slot ``s``)."""
        for q in range(2):
            self._load_half(2 * p + q, c0, cnt, 0, lambda s, q=q: q * _SLOTS * _HALF + s * _HALF)

    def pass_a(self):
        """The pre-pass: the conv W tiles of the first-half rows of block pairs ``(0, 1), (2, 3), ...`` while the
        second half's AG moves. One accumulation of 512 columns per tile, the chunks in order, as each block's own
        W tile would sum them; the bf16 results go to the pre-conv scratch. The last tile of a pair refills x^T
        behind its reads: the next pair's rows, then block 0's full rows."""
        KC = self.KC
        n_w = len(self.w_order)
        for c0 in range(0, KC, _X_GROUP_BLOCK):
            self._load_pair(0, c0, min(_X_GROUP_BLOCK, KC - c0))
        for p in range(self.row_halves):
            for pos, tile in enumerate(self.w_order):
                acc = self.psM[pos % 2][:, 0:_BLOCK]
                base = self.w_index[("A", p, pos)]
                for j, kc in enumerate(self.korder):
                    slot = self.w_stream.slot(base + j // _W_GROUP)
                    nisa.nc_matmul(
                        dst=acc,
                        stationary=self.wr[:, slot, j % _W_GROUP, :],
                        moving=self.xT[:, kc, 0:_BLOCK],
                        accumulate=(j > 0),
                    )
                    if pos == n_w - 1 and ((j + 1) % _X_GROUP_BLOCK == 0 or j == KC - 1):
                        c0 = (j // _X_GROUP_BLOCK) * _X_GROUP_BLOCK
                        if p + 1 < self.row_halves:
                            self._load_pair(p + 1, c0, j + 1 - c0)
                        else:
                            self.load_x(0, c0, j + 1 - c0)
                # Round to bf16 in the conv input tile (free until block 0's convs), then store both blocks' rows.
                stage = self.cin[:, tile, _CONV_HIST : _CONV_HIST + _BLOCK]
                nisa.tensor_copy(dst=stage, src=acc, engine=_VEC)
                row = _CONV_HIST + _BLOCK
                for q in range(2):
                    t0, ns = self.blocks[2 * p + q]
                    nisa.dma_copy(
                        dst=self._pre_half(tile, t0, 0, ns),
                        src=self.cin.ap(
                            pattern=[[_CONV_TILES * row, _P_MAX], [_HALF, ns], [1, _HALF]],
                            offset=tile * row + _CONV_HIST + q * _SLOTS * _HALF,
                        ),
                        **_DMA_STORE,
                    )

    def _load_h_a(self, b):
        """Reload pre-pass block ``b``'s first-half pre-conv rows from the scratch, before its W phase."""
        t0, ns = self.blocks[b]
        for ci in range(_CONV_TILES):
            nisa.dma_copy(dst=self._h_a_tile(ci, ns), src=self._pre_half(ci, t0, 0, ns), **_DMA_STORE)

    def _w_tile_half_b(self, b, pos):
        """A pre-pass block's conv W tile on its second-half rows only (256 columns), then its conv input."""
        _, ns = self.blocks[b]
        tile = self.w_order[pos]
        acc = self.psM[pos % 2].ap(pattern=[[_BLOCK, _P_MAX], [_HALF, ns], [1, _HALF]], offset=0)
        base = self.w_index[(b, 0, pos)]
        step_us = ns * _HALF * _US_PER_COL
        for j, kc in enumerate(self.korder):
            slot = self.w_stream.slot(base + j // _W_GROUP)
            nisa.nc_matmul(
                dst=acc,
                stationary=self.wr[:, slot, j % _W_GROUP, :],
                moving=self._xt_mov_half_b(kc, ns),
                accumulate=(j > 0),
            )
            self.tick(step_us)
        t0, _ = self.blocks[b]
        if b > 0:
            # As the vendored conv_tile: the previous block's readers and writer of this conv buffer go first.
            self.force_tile(b - 1, tile, "ds1" if tile < 6 else "silu")
            if tile >= 6:
                self.force(("v", b - 1, tile - 6))
        nisa.tensor_copy(dst=self._cin_half(tile, 1, ns), src=acc, engine=_VEC)
        nisa.tensor_copy(dst=self._cin_half(tile, 0, ns), src=self._h_a_tile(tile, ns), engine=_VEC)
        nisa.dma_copy(dst=self._pre_half(tile, t0, 1, ns), src=self._cin_half(tile, 1, ns), **_DMA_STORE)
        self.conv(b, tile)

    def w_block0_pieces(self):
        """Block 0's conv W tiles in K-passes, one pass per image piece; the partial sums wait in SBUF.

        Pass ``i`` needs only piece ``i`` of the image, so it runs while the later pieces move. Each pass
        restores the fp32 partial sum into PSUM and accumulates its chunks in order, so every output
        element sums the same products in the same order as one pass.
        """
        _, ns = self.blocks[0]
        N = ns * _P_MAX
        n_w = len(self.w_order)
        part = nl.ndarray((_P_MAX, n_w, N), dtype=_F32, buffer=nl.sbuf, name=self.nm("w0part"))
        step_us = N * _US_PER_COL
        last = len(self.pieces) - 1
        for pi, (c0, c1, _) in enumerate(self.pieces):
            if pi > 0:
                self.load_x(0, c0, c1 - c0)
            for pos, tile in enumerate(self.w_order):
                acc = self.psM[pos % 2][:, 0:N]
                if pi > 0:
                    nisa.tensor_copy(dst=acc, src=part[:, pos, :], engine=_VEC)
                base = self.w_index[(0, pi, pos)]
                for kc in range(c0, c1):
                    slot = self.w_stream.slot(base + (kc - c0) // _W_GROUP)
                    nisa.nc_matmul(
                        dst=acc,
                        stationary=self.wr[:, slot, (kc - c0) % _W_GROUP, :],
                        moving=self._xt_mov(kc, ns),
                        accumulate=(pi > 0 or kc > c0),
                    )
                    self.tick(step_us)
                if pi < last:
                    nisa.tensor_copy(dst=part[:, pos, :], src=acc, engine=nisa.engine.scalar)
                else:
                    self.conv_tile(0, tile, acc)

    def g_phase(self, b):
        """g|b token-major (ns token tiles), with the next block's x^T loaded group by group behind it."""
        if self.mx in (1, 3):
            return self._g_phase_mx(b)
        t0, ns = self.blocks[b]
        nb = len(self.blocks)
        nxt = b + 1 if b + 1 < nb else None
        for part in range(2):
            self.force(("fb", b, part))
        step_us = ns * (_PROJ + _HEADS) * _US_PER_COL
        cast_state = [0]  # mx=2: the next x^T chunk of block nxt to cast to e4m3, under these matmuls
        for j, kc in enumerate(self.korder):
            slot = self.gb_stream.slot(b * self.gb_per_block + j // _GB_GROUP)
            mov = self.gbr[:, slot, j % _GB_GROUP, :]
            for s in range(ns):
                self.last_matmul = nisa.nc_matmul(
                    dst=self.psM[_G_BANK[s]][:, 0 : _PROJ + _HEADS],
                    stationary=self._xt_stat(kc, s, ns),
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
        """g out; b -> bf16 with the bf16 minimum in the pad rows (a select); beta; the gate; the prep."""
        t0, ns = self.blocks[b]
        par = b % 2
        H, D = _HEADS, _HEAD_DIM
        self.force_blocks(b - 2)  # beta of this parity
        for s in [s for s in _G_EVAC_ORDER if s < ns]:
            gs = self.gst[:, self.next_ring(self.ring2, "gst") % 2, :]
            nisa.activation(dst=gs, op=nl.copy, data=self.psM[_G_BANK[s]][:, 0:_PROJ])
            if self.masked:
                # g of a pad row -> 0 (a select): the output of a pad row is then exactly 0 (its scan output is
                # 0), and the tail needs no pad mask.
                nisa.tensor_copy_predicated(dst=gs, src=0.0, predicate=self.tokpred[:, par, s, :])
            nisa.dma_copy(
                dst=self.g.ap(pattern=[[ns * self.pf, _P_MAX], [1, _PROJ]], offset=(t0 + s) * self.pf + self.fb_col),
                src=gs,
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.scalar,
            )
            nisa.tensor_copy(dst=self.bl[:, s, :], src=self.psM[_G_BANK[s]][:, _PROJ : _PROJ + H], engine=_VEC)
            if self.masked:
                nisa.tensor_copy_predicated(dst=self.bl[:, s, :], src=_BF16_MIN, predicate=self.tokpred[:, par, s, 0:H])
        nisa.activation(dst=self.beta[:, par, 0:ns, :], op=nl.sigmoid, data=self.bl[:, 0:ns, :])
        if b == 0:
            nisa.tensor_copy(dst=self.beta0[:, 0:ns, :], src=self.beta[:, par, 0:ns, :], engine=_VEC)
        for h in range(H):
            # With more than 2 slots, a z-gate tile is rewritten only after the gtok store of an older head ends.
            zf = self.zf[:, self.next_ring(self.ring2, "zf") % self.zf_slots, 0:ns, :]
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
                dst=self.gtok.ap(pattern=[[ns * D, _P_MAX], [1, ns * D]], offset=((self.h0 + h) * self.T + t0) * D),
                src=zf,
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.scalar,
            )
        if b > 0 or self.block0_in_order:
            self.ds_tail(b)

    # ------------------------------------------------------------------ top level

    def run(self):
        self.load_constants()
        # The first weight groups load before the image: the image loads wait for AG_x, and the DMA queue is
        # in order.
        self.w_stream.slot(0)
        self.gb_stream.slot(0)
        if self.row_halves:
            # The pre-pass loads its own rows, and block 0's x^T behind its last tile.
            self.pass_a()
        else:
            # Block 0's x^T (its first piece with K-passes); the later blocks load behind the G phase of the
            # block before.
            c_end = self.pieces[0][1] if len(self.pieces) > 1 else self.KC
            for c0 in range(0, c_end, _X_GROUP_BLOCK):
                self.load_x(0, c0, min(_X_GROUP_BLOCK, c_end - c0))
        for b in range(len(self.blocks)):
            if self.masked:
                self.block_masks(b)
            if self.row_halves and self._in_pass_a(b):
                self._load_h_a(b)
            if b == 0 and len(self.pieces) > 1:
                # f_a^T is in the last piece: load it after the passes, so its DMA does not hold the queue.
                self.w_block0_pieces()
                self.fa_tile(0)
            else:
                self.fa_tile(b)
                # Both forms quantize block 0 up front; blocks 1+ are quantized under the previous block's g-phase.
                if self.mx and b == 0:
                    self.quantize_block(b)
                for pos in range(len(self.w_order)):
                    self.w_tile(b, pos)
            self.g_phase(b)
        self.flush()
        if not self.block0_in_order:
            self.block0_prep()
            self.flush()
        rows = 3 * _PROJ
        pattern = [[3 + self.T, rows], [1, 3]]
        if not self.grouped:
            if self.vs_on:
                # The conv state as the pre-conv rows of tokens vs - 3 .. vs - 1, for a new state within 3 rows of vs.
                nisa.dma_copy(
                    dst=self.pre.ap(pattern=pattern, offset=0, scalar_offset=self.vs_i, indirect_dim=2),
                    src=self.conv_state.ap(pattern=[[3, rows], [1, 3]], offset=0),
                    **_DMA_LOAD,
                )
            if self.masked:
                src = self.pre.ap(pattern=pattern, offset=0, scalar_offset=self.vl_sb, indirect_dim=2)
            else:
                src = self.pre.ap(pattern=pattern, offset=self.T)
            nisa.dma_copy(dst=self.cs_out.ap(pattern=[[3, rows], [1, 3]], offset=0), src=src, **_DMA_LOAD)
            if self.lead_on:
                # conv_lead: the pre-conv rows of tokens g - 3 .. g - 1, after the window rows above.
                nisa.dma_copy(
                    dst=self.cl_out.ap(pattern=[[3, rows], [1, 3]], offset=0),
                    src=self.pre.ap(pattern=pattern, offset=0, scalar_offset=self.lead_i, indirect_dim=2),
                    **_DMA_LOAD,
                )
        else:
            # The new conv state into this group's channels of the shared [3, hc * D, 3] output: one DMA per q/k/v.
            pqkv = [[3 + self.T, _PROJ], [1, _CONV_HIST]]
            for i3 in range(3):
                src_off = i3 * _PROJ * (_CONV_HIST + self.T)
                # The group's channels of a full-width [3, hc * D, 3] conv tensor.
                chan = [[_CONV_HIST, _PROJ], [1, _CONV_HIST]]
                chan_off = (i3 * self.pf + self.h0 * _P_MAX) * _CONV_HIST
                if self.vs_on:
                    # The group's conv state as the pre-conv rows of tokens vs - 3 .. vs - 1 (as the 3-head form).
                    nisa.dma_copy(
                        dst=self.pre.ap(pattern=pqkv, offset=src_off, scalar_offset=self.vs_i, indirect_dim=2),
                        src=self.conv_state.ap(pattern=chan, offset=chan_off),
                        **_DMA_LOAD,
                    )
                if self.masked:
                    src = self.pre.ap(pattern=pqkv, offset=src_off, scalar_offset=self.vl_sb, indirect_dim=2)
                else:
                    src = self.pre.ap(pattern=pqkv, offset=src_off + self.T)
                nisa.dma_copy(dst=self.cs_out.ap(pattern=chan, offset=chan_off), src=src, **_DMA_LOAD)
                if self.lead_on:
                    nisa.dma_copy(
                        dst=self.cl_out.ap(pattern=chan, offset=chan_off),
                        src=self.pre.ap(pattern=pqkv, offset=src_off, scalar_offset=self.lead_i, indirect_dim=2),
                        **_DMA_LOAD,
                    )
        if self.lead_on:
            return self.qT, self.kT, self.kbT, self.vb, self.gtok, self.g, self.cs_out, self.cl_out
        return self.qT, self.kT, self.kbT, self.vb, self.gtok, self.g, self.cs_out
