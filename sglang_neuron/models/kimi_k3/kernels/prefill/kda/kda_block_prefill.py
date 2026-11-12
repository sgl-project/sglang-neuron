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

"""One NKI call for the whole Kimi-K3 prefill KDA attention block of one layer, with the collectives inside.

The kernel takes and gives the layout of the torch KDA prefill block in the
sequence-parallel form: the normed row block ``x [Tl, H]`` of this rank in, the row block of the
``o_proj`` sum out.

Order of the work (see DESIGN.md of the kernel lead for the reasons):

Region 1, straight-line code:
    1. Load the rows of this rank and transpose them on TensorE (transpose mode) to the hidden-major
       image ``x^T``. Run f_a on these rows (the front's matmul order) and append ``f_a^T`` to the image.
    2. All-gather the image (AG_x, one HBM collective).
Region 2, straight-line code: the KDA front (``_KdaBlockFront``). It reads x^T and f_a^T from the
    gathered image, and it masks the pad rows with selects.
Region 3: the KDA scan ``kda_prefill_v2`` as a nested call. It has dynamic loops, so no collective is
    in flight there.
Region 4, straight-line code:
    1. The output norm and gate, 0 in the pad rows, and TensorE
       transposes to ``y^T [128, H_local, T]``.
    2. ``o_proj`` in column pieces. After each piece, start its reduce-scatter (RS piece), so it moves
       while TensorE computes the next piece.
    3. Copy the RS results into the output row block.

All collectives run in ``private_hbm`` tensors: no peer accesses SBUF.
"""

from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ..utils.kernel_assert import kernel_assert
from .kda_prefill_v2_kernel import kda_prefill_v2
from ..utils.trn_target import is_trn2
from ..utils.valid_row_window import KDA_ITEMS, window_items
from .block_cc import gather_rows_feature_major, pe_transpose, reduce_scatter_rows, transpose_rows_bf16
from .in_proj_conv import _KdaBlockFront
from .kda_front import _GB_COLS, _HEAD_DIM, _HEADS, _check_mx, _new_trace_id

_P = 128
_BANK_FP32 = 512
_F32 = nl.float32
_BF16 = nl.bfloat16
_U8 = nl.uint8
_VEC = nisa.engine.vector
_ACT = nisa.engine.scalar
_EVAC = (_ACT, _VEC)
_DMA_SYNC = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}
_DMA_ACT = {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.scalar}
# Tokens per partition of a post block (2 KiB runs of the fp32 scan output per head).
_POST_J = 4
# Rows of one call: T = W * Tl.
_MAX_TOKENS = 8192
# Local heads x rows of one call: the tail keeps y^T [128, hc, T] bf16 (2 * hc * T bytes a partition) resident, so 12 x
# 4096 and 6 x 8192 (96 KiB) fit with the o_proj weight and the post ring.
_MAX_HEAD_TOKENS = 12 * 4096
# Above this many rows the post block takes 2 tokens a partition (see _KdaBlock.tail).
_POST_J_SMALL_ABOVE = 4096
# PSUM banks of o_proj accumulators in flight for one 128-row group.
_OPROJ_BANKS = 4
# Post blocks whose o and g loads run ahead of the post.
_POST_RING = 3
# Column pieces of the o_proj weight ring above 2048 rows (one reduce-scatter); 4 pieces of H / 4 fit SBUF at T 4096.
_SINGLE_RS_RING_PIECES = 4
# The scan head groups of a Trn3 rank with 12 local heads (the TP8 form), the default of scan_head_groups there: the
# 12 heads' pair-matrix and inverse PSUM tiles do not fit at once, and 6 groups of 2 heads avoid most of the scan's
# PSUM spills and reloads. Every other form keeps 1 (with 3 heads, Trn2, the groups only remove overlap).
TP8_SCAN_HEAD_GROUPS = 6
# With ag_shared on Trn3, the gathered input images (the input all-gather destinations) are the only shared_hbm
# scratch tensors; every per-core scratch tensor is private HBM, so ``--lnc1-with-shared-mem`` gives the two cores of
# an HBM stack one buffer only where both read the same gathered data. A caller can check this before it gives
# ag_shared with that option. The kernel does not read it.
AG_SHARED_TRN3_PRIVATE_SCRATCH = True


@dataclass(frozen=True)
class KdaBlockPlan:
    """Shapes and groups of one KDA block call on one rank.

    Dimensions:
        W: Ranks of the group (one rank list of ``world_ranks``).
        Tl: Rows of this rank (the sequence-parallel row block).
        T: Rows of the group, ``W * Tl``, at most 8192, with ``hc * T`` at most ``12 * 4096``.
        H: Hidden size.
        hc: Local KDA heads. D: head dimension.

    Example:
        >>> plan = KdaBlockPlan((tuple(range(32)),), 128, 7168)
        >>> plan.world, plan.tokens
        (32, 4096)
    """

    world_ranks: tuple
    rows: int
    hidden: int
    heads: int = _HEADS
    head_dim: int = _HEAD_DIM

    @property
    def world(self) -> int:
        """Return W, the size of one rank list."""
        return len(self.world_ranks[0])

    @property
    def tokens(self) -> int:
        """Return T = W * Tl."""
        return self.world * self.rows

    def check(self) -> None:
        """Raise an AssertionError if the plan is not consistent.

        Example:
            >>> KdaBlockPlan((tuple(range(2)),), 128, 512).check()
        """
        sizes = {len(group) for group in self.world_ranks}
        kernel_assert(len(sizes) == 1, f"The rank lists of world_ranks have different sizes: {sorted(sizes)}")
        kernel_assert(self.rows % _P == 0 and self.rows > 0, f"Tl {self.rows} is not a positive multiple of 128")
        kernel_assert(self.hidden % _P == 0, f"H {self.hidden} is not a multiple of 128")
        kernel_assert(self.tokens <= _MAX_TOKENS, f"T = {self.tokens} is above {_MAX_TOKENS}")
        kernel_assert(
            self.heads * self.tokens <= _MAX_HEAD_TOKENS,
            f"heads x T = {self.heads} x {self.tokens} is above 12 x 4096: y^T [128, heads, T] bf16 does not fit SBUF",
        )


class _Namer:
    """Give each buffer a name that is unique in one kernel trace: ``kb<call>_<tag>_<n>``."""

    def __init__(self, trace_id):
        self.prefix = f"kb{trace_id}"
        self.n = 0

    def __call__(self, tag):
        self.n += 1
        return f"{self.prefix}_{tag}_{self.n}"


def _post_blocks(T, J):
    """Token blocks ``(t0, J)`` of 128 * J tokens: token ``t0 + J * p + j`` on partition ``p``, slot ``j``."""
    out, t0 = [], 0
    while T - t0 >= _P * J:
        out.append((t0, J))
        t0 += _P * J
    if T - t0 > 0:
        out.append((t0, (T - t0) // _P))
    return out


def _pieces_of(H, cols):
    """Split ``H`` columns into pieces of ``cols`` columns (the last one shorter): ``[(n0, width)]``."""
    return [(c, min(cols, H - c)) for c in range(0, H, cols)]


def _pieces(H, n):
    """Split ``H`` columns into ``n`` pieces of whole 128-column tiles: ``[(n0, width)]``."""
    tiles = H // _P
    n = max(1, min(n, tiles))
    out, start = [], 0
    for g in range(n):
        cnt = tiles // n + (1 if g < tiles % n else 0)
        out.append((start * _P, cnt * _P))
        start += cnt
    return out


class _KdaBlock:
    """One call of the KDA block kernel (trace-time state)."""

    def __init__(self, plan: KdaBlockPlan, group: ReplicaGroup, opts: dict):
        self.plan, self.group, self.opts = plan, group, opts
        self.H, self.W, self.Tl, self.T = plan.hidden, plan.world, plan.rows, plan.tokens
        self.hc, self.D = plan.heads, plan.head_dim
        self.p = self.hc * self.D
        # f_a is one shared 128-wide feature after the 3 * hc q/k/v conv tiles of w_qkvf (tile 9 at 3 heads).
        self.fa_tile = 0 if opts["fa_only"] else 3 * self.hc
        self.nT = plan.rows // _P
        self.HK = plan.hidden // _P
        self.nm = _Namer(_new_trace_id())

    def _sb(self, shape, dtype, tag):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf, name=self.nm(tag))

    def _ps(self, shape, dtype, tag):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.psum, name=self.nm(tag))

    def _hbm(self, shape, dtype, tag, buffer=nl.private_hbm):
        return nl.ndarray(shape, dtype=dtype, buffer=buffer, name=self.nm(tag))

    # ------------------------------------------------------------------ region 1

    def gather_input(self, x, w_qkvf, ident):
        """x^T and f_a^T of the rows of this rank into the image pieces, then one AG per piece.

        Piece ``i`` holds the hidden chunks ``[c0, c1)``: ``src_i [128, nT * (c1 - c0) * 128]``, column
        ``(t * (c1 - c0) + kc - c0) * 128 + r`` for row ``r`` of local tile ``t``. The last piece also holds
        ``f_a^T`` after its x^T columns. Each piece starts its AG as soon as its columns are stored, so the
        front can start on the first chunks while the later pieces move.
        """
        H, HK, nT, Tl = self.H, self.HK, self.nT, self.Tl
        chunks = self.opts["ag_piece_chunks"]
        bounds = _pieces_of(HK * _P, chunks * _P) if chunks else _pieces(HK * _P, self.opts["ag_pieces"])
        self.img_pieces = []
        id32 = self._sb((_P, _P), _F32, "id32")
        nisa.dma_copy(dst=id32, src=ident, **_DMA_SYNC)
        self.id_bf = self._sb((_P, _P), _BF16, "id_bf")
        nisa.tensor_copy(dst=self.id_bf, src=id32, engine=_VEC)
        # The f_a weight: tile 9 of w_qkvf, [128 hidden-in-chunk, HK, 128 f_a columns].
        w_fa = self._sb((_P, HK, _P), _BF16, "w_fa")
        nisa.dma_copy(
            dst=w_fa.reshape((_P, HK * _P)),
            src=w_qkvf.ap(pattern=[[HK * _P, _P], [1, HK * _P]], offset=self.fa_tile * _P * HK * _P),
            **_DMA_SYNC,
        )
        xts = []
        for i in range(nT):
            rows = self._sb((_P, H), _BF16, f"x_rows{i}")
            nisa.dma_copy(dst=rows, src=x[i * _P : (i + 1) * _P, :], **_DMA_SYNC)
            xts.append((rows, self._sb((_P, HK, _P), _BF16, f"xT{i}")))
        for pi, (h0, hw) in enumerate(bounds):
            c0, c1 = h0 // _P, (h0 + hw) // _P
            cnt = c1 - c0
            last = pi == len(bounds) - 1
            cols = nT * cnt * _P + (Tl if last else 0)
            src = self._hbm((_P, cols), _BF16, f"img_src{pi}")
            # ag_shared: with NKI_LNC1_SHARED_HBM=1, the two cores of an HBM stack share one copy of the gathered
            # image (the AG writes each stack once). Without the flag, shared_hbm scratch is private.
            dst = self._hbm(
                (self.W * _P, cols),
                _BF16,
                f"img_dst{pi}",
                buffer=nl.shared_hbm if self.opts["ag_shared"] else nl.private_hbm,
            )
            for i, (rows, xt) in enumerate(xts):
                transpose_rows_bf16(rows[:, c0 * _P : c1 * _P], xt[:, c0:c1, :], self.id_bf, self.nm(f"xT{i}_{pi}"))
                nisa.dma_copy(
                    dst=src.ap(pattern=[[cols, _P], [_P, cnt], [1, _P]], offset=i * cnt * _P),
                    src=xt[:, c0:c1, :],
                    **_DMA_SYNC,
                )
            if last:
                for i, (_, xt) in enumerate(xts):
                    # f_a^T [128 f, 128 t]: the front's stationary weight tile, the chunks in order 0, 1, ...
                    acc = self._ps((_P, _P), _F32, f"fa_ps{i}")
                    for kc in range(HK):
                        nisa.nc_matmul(dst=acc, stationary=w_fa[:, kc, :], moving=xt[:, kc, :], accumulate=kc > 0)
                    fa = self._sb((_P, _P), _BF16, f"fa{i}")
                    nisa.tensor_copy(dst=fa, src=acc, engine=_VEC)
                    off = nT * cnt * _P + i * _P
                    nisa.dma_copy(dst=src[:, off : off + _P], src=fa, **_DMA_SYNC)
            gather_rows_feature_major(src, dst, self.group, self.nm(f"agx{pi}"))
            self.img_pieces.append((c0, c1, dst))
        return self.img_pieces

    def gather_input_halves(self, x, w_qkvf, ident):
        """The row-half form of ``gather_input``: two images and two AGs, so the front starts on the first half.

        Half ``h`` holds rows ``64 h .. 64 h + 63`` of every local tile: ``src_h [128, nT * HK * 64 + nT * 64]``,
        column ``(t * HK + kc) * 64 + r`` for row ``64 h + r`` of local tile ``t``, then ``f_a^T`` of those rows at
        column ``nT * HK * 64 + t * 64 + r``. Each half gets its own AG, half 0 first. The values are those of the
        one-image form: the same transposes, and f_a sums the same products in the same chunk order per column.
        """
        H, HK, nT = self.H, self.HK, self.nT
        R = _P // 2
        id32 = self._sb((_P, _P), _F32, "id32")
        nisa.dma_copy(dst=id32, src=ident, **_DMA_SYNC)
        self.id_bf = self._sb((_P, _P), _BF16, "id_bf")
        nisa.tensor_copy(dst=self.id_bf, src=id32, engine=_VEC)
        w_fa = self._sb((_P, HK, _P), _BF16, "w_fa")
        nisa.dma_copy(
            dst=w_fa.reshape((_P, HK * _P)),
            src=w_qkvf.ap(pattern=[[HK * _P, _P], [1, HK * _P]], offset=self.fa_tile * _P * HK * _P),
            **_DMA_SYNC,
        )
        xh = []
        for i in range(nT):
            rows = self._sb((_P, H), _BF16, f"x_rows{i}")
            nisa.dma_copy(dst=rows, src=x[i * _P : (i + 1) * _P, :], **_DMA_SYNC)
            halves = [self._sb((_P, HK, R), _BF16, f"xT{i}_h{h}") for h in range(2)]
            for first in range(0, HK, 8):
                count = min(8, HK - first)
                ps = nl.ndarray((_P, 8, _P), dtype=_BF16, buffer=nl.psum, name=self.nm(f"xT{i}_ps{first}"))
                for j in range(count):
                    k = first + j
                    pe_transpose(ps[:, j, :], rows[:, k * _P : (k + 1) * _P], self.id_bf)
                for h in range(2):
                    nisa.tensor_copy(
                        dst=halves[h][:, first : first + count, :], src=ps[:, 0:count, h * R : (h + 1) * R], engine=_VEC
                    )
            xh.append(halves)
        cols = nT * HK * R + nT * R
        self.img_halves = []
        for h in range(2):
            src = self._hbm((_P, cols), _BF16, f"img_src_h{h}")
            dst = self._hbm(
                (self.W * _P, cols),
                _BF16,
                f"img_dst_h{h}",
                buffer=nl.shared_hbm if self.opts["ag_shared"] else nl.private_hbm,
            )
            for i in range(nT):
                nisa.dma_copy(
                    dst=src[:, i * HK * R : (i + 1) * HK * R], src=xh[i][h].reshape((_P, HK * R)), **_DMA_SYNC
                )
            for i in range(nT):
                # f_a^T [128 f, 64 t] of the half: one accumulation over the chunks 0, 1, ... in its own PSUM tile.
                acc = self._ps((_P, R), _F32, f"fa_ps{i}_h{h}")
                for kc in range(HK):
                    nisa.nc_matmul(dst=acc, stationary=w_fa[:, kc, :], moving=xh[i][h][:, kc, :], accumulate=kc > 0)
                fa = self._sb((_P, R), _BF16, f"fa{i}_h{h}")
                nisa.tensor_copy(dst=fa, src=acc, engine=_VEC)
                off = nT * HK * R + i * R
                nisa.dma_copy(dst=src[:, off : off + R], src=fa, **_DMA_SYNC)
            gather_rows_feature_major(src, dst, self.group, self.nm(f"agx_h{h}"))
            self.img_halves.append(dst)
        self.img_pieces = [(0, HK, self.img_halves[0])]
        return self.img_pieces

    # ------------------------------------------------------------------ regions 2 and 3

    def front(
        self,
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
        valid_start,
        mx=0,
        w_qkv_mx=None,
        w_qkv_scale=None,
        w_gb_mx=None,
        w_gb_scale=None,
        lead_outputs=False,
    ):
        engine = nisa.engine.scalar if self.opts["x_dma_scalar"] else nisa.engine.sync
        # One 3-head front per group of 3 local heads; a single group keeps the vendored outputs, more groups share.
        ng = self.hc // _HEADS
        out = None if ng == 1 else self._front_out(lead=lead_outputs)
        ret = None
        for g in range(ng):
            fr = _KdaBlockFront(
                self.img_pieces,
                self.Tl,
                self.W,
                self.H,
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
                x_dma_engine=engine,
                block0_in_order=self.opts["block0_in_order"],
                zgate_slots=self.opts["zgate_slots"],
                row_halves=self.opts["ag_row_halves"],
                halves=getattr(self, "img_halves", None),
                valid_start=valid_start,
                mx=mx,
                w_qkv_mx=w_qkv_mx,
                w_qkv_scale=w_qkv_scale,
                w_gb_mx=w_gb_mx,
                w_gb_scale=w_gb_scale,
                out_hbm=out,
                head_group=g,
                p_full=self.p,
                lead_outputs=lead_outputs,
                lead_out=None if out is None else out.get("cl_out"),
            )
            r = fr.run()
            if ng == 1:
                ret = r
        if ng == 1:
            return ret
        ret = out["qT"], out["kT"], out["kbT"], out["vb"], out["gtok"], out["g"], out["cs_out"]
        return (*ret, out["cl_out"]) if lead_outputs else ret

    def _front_out(self, lead=False):
        """Shared full-width front outputs for the head-group loop: [hc, ...] / [T, hc * D], conv_new [3, hc * D, 3]
        (and conv_lead [3, hc * D, 3] with ``lead``)."""
        T, D, hc, p = self.T, self.D, self.hc, self.p
        out = {
            "qT": self._hbm((hc, D, T), _F32, "fqT"),
            "kT": self._hbm((hc, D, T), _F32, "fkT"),
            "kbT": self._hbm((hc, D, T), _F32, "fkbT"),
            "vb": self._hbm((hc, T, D), _F32, "fvb"),
            "gtok": self._hbm((hc, T, D), _F32, "fgtok"),
            "g": self._hbm((T, p), _F32, "fg"),
            "cs_out": self._hbm((3, p, 3), _BF16, "fcs", buffer=nl.shared_hbm),
        }
        if lead:
            out["cl_out"] = self._hbm((3, p, 3), _BF16, "fcl", buffer=nl.shared_hbm)
        return out

    # ------------------------------------------------------------------ region 4

    def tail(self, o, g, o_norm_w, o_proj_w_t, valid_len):
        """Norm and gate, y^T, o_proj in pieces with one RS each; returns y [Tl, H] bf16."""
        H, T, Tl, hc, D, p = self.H, self.T, self.Tl, self.hc, self.D, self.p
        Tp = o.shape[1]
        eps = self.opts["eps"]
        masked = valid_len is not None
        debug = self.opts["debug_outputs"]
        cols = self.opts["rs_piece_cols"]
        pieces = _pieces_of(H, cols) if cols else _pieces(H, self.opts["rs_pieces"])
        # Weights: o_proj K-major [128 d, hc, cols], one tile per piece, in piece order (piece 0 arrives first).
        # Above 6 heads the whole weight (hc * H * 2 bytes) plus y^T overflow SBUF at T 4096, so when the caller
        # splits o_proj into pieces the weight streams through a two-slot ring: piece gi is resident in slot gi % 2
        # while piece gi + 1 prefetches into the other slot. The 3-head and 6-head programs keep the all-resident
        # form, so their traces do not change.
        wo_ring = hc > _HEADS * 2 and len(pieces) > 1
        # Above 2048 rows with > 6 heads and one RS piece, stream the weight through the ring into one RS source.
        single_rs_ring = T > 2048 and hc > _HEADS * 2 and cols == 0 and self.opts["rs_pieces"] in (0, 1)
        if single_rs_ring:
            wpieces = _pieces(H, _SINGLE_RS_RING_PIECES)
            wo_pw = max(pw for _, pw in wpieces)
            wo_slots = [self._sb((_P, hc, wo_pw), _BF16, f"wo_ring{s}") for s in range(2)]

            def load_wo(gi):
                n0, pw = wpieces[gi]
                nisa.dma_copy(
                    dst=wo_slots[gi % 2][:, :, 0:pw],
                    src=o_proj_w_t.ap(pattern=[[H, _P], [_P * H, hc], [1, pw]], offset=n0),
                    **_DMA_ACT,
                )

            def wo_tile(gi):
                return wo_slots[gi % 2][:, :, 0 : wpieces[gi][1]]
        elif wo_ring:
            wo_pw = max(pw for _, pw in pieces)
            wo_slots = [self._sb((_P, hc, wo_pw), _BF16, f"wo_ring{s}") for s in range(2)]

            def load_wo(gi):
                n0, pw = pieces[gi]
                nisa.dma_copy(
                    dst=wo_slots[gi % 2][:, :, 0:pw],
                    src=o_proj_w_t.ap(pattern=[[H, _P], [_P * H, hc], [1, pw]], offset=n0),
                    **_DMA_ACT,
                )

            def wo_tile(gi):
                return wo_slots[gi % 2][:, :, 0 : pieces[gi][1]]
        else:
            wo = []
            for gi, (n0, pw) in enumerate(pieces):
                t = self._sb((_P, hc, pw), _BF16, f"wo{gi}")
                nisa.dma_copy(dst=t, src=o_proj_w_t.ap(pattern=[[H, _P], [_P * H, hc], [1, pw]], offset=n0), **_DMA_ACT)
                wo.append(t)

            def wo_tile(gi):
                return wo[gi]

        w_sb = self._sb((_P, D), _F32, "onw")
        nisa.dma_copy(dst=w_sb, src=o_norm_w.ap(pattern=[[0, _P], [1, D]], offset=0), **_DMA_SYNC)
        yT = self._sb((_P, hc, T), _BF16, "yT")
        if debug:
            self.y_pre = self._hbm((T, p), _BF16, "y_pre", buffer=nl.shared_hbm)

        # Above 6 heads at T 4096, and at 6 heads above 4096 rows (y^T is 96 KiB a partition at T 8192), the post
        # ring (o_ring + g_ring) and y^T together near-fill the usable SBUF, so halve the post block to J = 2:
        # o_ring, g_ring and the per-block scratch all scale with J, and the total o_proj work (T / 128 groups) does
        # not change. 3 and 6 heads keep J = 4 up to 4096 rows, so their traces do not change.
        post_j = self.opts["post_block_tokens"]
        if not post_j:
            post_j = 2 if hc > _HEADS * 2 or T > _POST_J_SMALL_ABOVE else _POST_J
        blocks = _post_blocks(T, post_j)
        n_ring = min(_POST_RING, len(blocks))
        o_ring = self._sb((_P, n_ring, hc, post_j * D), _F32, "o_ring")
        g_ring = self._sb((_P, n_ring, post_j * p), _F32, "g_ring")
        loaded = [0]

        def load_upto(n):
            """Issue the o and g loads of the post blocks before ``n`` (a ring of ``_POST_RING`` blocks)."""
            while loaded[0] < min(n, len(blocks)):
                bi = loaded[0]
                t0, J = blocks[bi]
                k = bi % n_ring
                nisa.dma_copy(
                    dst=o_ring[:, k, :, 0 : J * D],
                    src=o.ap(pattern=[[J * D, _P], [Tp * D, hc], [1, J * D]], offset=t0 * D),
                    **_DMA_SYNC,
                )
                nisa.dma_copy(
                    dst=g_ring[:, k, 0 : J * p], src=g.ap(pattern=[[J * p, _P], [1, J * p]], offset=t0 * p), **_DMA_ACT
                )
                loaded[0] += 1

        def post_block(bi, t0, J):
            """Norm and gate of one block; y^T columns. Returns its (col, t0, J, j) groups."""
            load_upto(bi + n_ring)
            k = bi % n_ring
            o_t = o_ring[:, k, :, 0 : J * D]
            g_t = g_ring[:, k, 0 : J * p]
            # The output norm and gate steps.
            sq = self._sb((_P, hc, J * D), _F32, f"sq{bi}")
            st = self._sb((_P, 3, hc, J), _F32, f"st{bi}")
            ss, ms, rs = st[:, 0], st[:, 1], st[:, 2]
            nisa.activation(dst=sq, op=nl.square, data=o_t)
            nisa.activation(dst=g_t, op=nl.sigmoid, data=g_t)
            nisa.tensor_reduce(dst=ss, op=nl.add, data=sq.reshape_dim(2, (J, D)), axis=3)
            nisa.activation(dst=ms, op=nl.copy, data=ss, scale=1.0 / D, bias=float(eps))
            nisa.tensor_scalar(dst=rs, data=ms, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
            o4 = o_t.reshape_dim(2, (J, D))
            a4 = sq.reshape_dim(2, (J, D))
            nisa.tensor_tensor(dst=a4, data1=o4, data2=rs.expand_dim(3).broadcast(3, D), op=nl.multiply, engine=_VEC)
            w4 = w_sb.expand_dim(1).broadcast(1, J).expand_dim(1).broadcast(1, hc)
            nisa.tensor_tensor(dst=a4, data1=a4, data2=w4, op=nl.multiply, engine=_VEC)
            y_t = self._sb((_P, J * p), _BF16, f"y{bi}")
            nisa.tensor_tensor(
                dst=y_t.reshape_dim(1, (J, hc, D)),
                data1=a4.permute((0, 2, 1, 3)),
                data2=g_t.reshape_dim(1, (J, hc, D)),
                op=nl.multiply,
                engine=_VEC,
            )
            # The pad rows need no mask here: their scan output is 0 and the front selects their g to 0, so y is
            # exactly 0 there.
            if debug:
                nisa.dma_copy(dst=self.y_pre.ap(pattern=[[J * p, _P], [1, J * p]], offset=t0 * p), src=y_t, **_DMA_ACT)
            for h in range(hc):
                ps = self._ps((_P, J, _P), _BF16, f"yT_ps{bi}_{h}")
                for j in range(J):
                    pe_transpose(ps[:, j, :], y_t[:, j * p + h * D : j * p + (h + 1) * D], self.id_bf)
                nisa.tensor_copy(
                    dst=yT[:, h, t0 : t0 + J * _P], src=ps.reshape((_P, J * _P)), engine=_EVAC[(bi * hc + h) % 2]
                )
            return [(t0 + j * _P, t0, J, j) for j in range(J)]

        if single_rs_ring:
            src_all = self._hbm((T, H), _BF16, "rs_src")
        else:
            srcs = [self._hbm((T, pw), _BF16, f"rs_src{gi}") for gi, (_, pw) in enumerate(pieces)]
        n_ev = [0]
        # With oproj_batch_stores: one stage tile for all groups, and one store per PSUM column batch, issued right
        # after the batch's evacuation. A batch's evacuation then waits only for the store of the same columns of
        # the group before, which left one group earlier. With a new tile per group and one store per group (the
        # default), the allocator puts every group's tile at one address, so the first evacuation of a group waits
        # for the whole store of the group before.
        op_stage = None
        if self.opts["oproj_batch_stores"] and not single_rs_ring:
            op_stage = self._sb((_P, max(pw for _, pw in pieces)), _BF16, "op_stage")

        def proj_group(gi, qi, grp):
            """o_proj of one 128-row group for piece ``gi``, stored to the rows of its tokens in the RS source."""
            if single_rs_ring:
                n0, pw = wpieces[gi]
                target, row_pitch, col0 = src_all, H, n0
            else:
                n0, pw = pieces[gi]
                target, row_pitch, col0 = srcs[gi], pw, 0
            wsrc = wo_tile(gi)
            col, t0, J, j = grp
            chunks = [(c, min(_BANK_FP32, pw - c)) for c in range(0, pw, _BANK_FP32)]
            if op_stage is not None:
                stage = op_stage[:, 0:pw]
            else:
                stage = self._sb((_P, pw), _BF16, f"op_sb{gi}_{qi}")
            # Row m of the tile is token t0 + J m + j. The stores alternate between the two DMA queues: the tail
            # writes the whole 58.7 MB RS source.
            dma_q = _DMA_SYNC if qi % 2 == 0 else _DMA_ACT
            # At most 4 PSUM banks of accumulators at a time (a wide piece runs in column batches).
            for b0 in range(0, len(chunks), _OPROJ_BANKS):
                batch = chunks[b0 : b0 + _OPROJ_BANKS]
                accs = [self._ps((_P, w), _F32, f"op{gi}_{qi}_{c}") for (c, w) in batch]
                # The stationary y^T tile serves all chunks of the batch; each element sums h = 0, 1, ...
                for h in range(hc):
                    for ci, (c, w) in enumerate(batch):
                        nisa.nc_matmul(
                            dst=accs[ci],
                            stationary=yT[:, h, col : col + _P],
                            moving=wsrc[:, h, c : c + w],
                            accumulate=h > 0,
                        )
                for ci, (c, w) in enumerate(batch):
                    nisa.tensor_copy(dst=stage[:, c : c + w], src=accs[ci], engine=_EVAC[n_ev[0] % 2])
                    n_ev[0] += 1
                if op_stage is not None:
                    c0, c1 = batch[0][0], batch[-1][0] + batch[-1][1]
                    nisa.dma_copy(
                        dst=target.ap(
                            pattern=[[J * row_pitch, _P], [1, c1 - c0]], offset=(t0 + j) * row_pitch + col0 + c0
                        ),
                        src=stage[:, c0:c1],
                        **dma_q,
                    )
            if op_stage is None:
                nisa.dma_copy(
                    dst=target.ap(pattern=[[J * row_pitch, _P], [1, pw]], offset=(t0 + j) * row_pitch + col0),
                    src=stage,
                    **dma_q,
                )

        self.y = nl.ndarray((Tl, H), dtype=_BF16, buffer=nl.shared_hbm, name=self.nm("y"))

        rs_dsts = []

        def finish_piece(gi):
            n0, pw = pieces[gi]
            dst = self._hbm((Tl, pw), _BF16, f"rs_dst{gi}")
            reduce_scatter_rows(srcs[gi], dst, self.group, self.nm(f"rs{gi}"))
            rs_dsts.append((n0, pw, dst))

        # The whole post first, then o_proj. With o_proj one post block behind, the PSUM evacuations of o_proj
        # sat behind the post's ScalarE and VectorE steps of the next block (in-order queues with cross-engine
        # waits), and TensorE waited in each block.
        groups = []
        for bi, (t0, J) in enumerate(blocks):
            groups.extend(post_block(bi, t0, J))
        if single_rs_ring:
            load_wo(0)
            for gi in range(len(wpieces)):
                if gi + 1 < len(wpieces):
                    load_wo(gi + 1)
                for qi, grp in enumerate(groups):
                    proj_group(gi, qi, grp)
            dst = self._hbm((Tl, H), _BF16, "rs_dst")
            reduce_scatter_rows(src_all, dst, self.group, self.nm("rs"))
            nisa.dma_copy(dst=self.y.ap(pattern=[[H, Tl], [1, H]], offset=0), src=dst, **_DMA_ACT)
            return self.y
        if wo_ring:
            # Prefetch piece gi + 1's weight into the free slot while piece gi's matmuls run.
            load_wo(0)
            for gi in range(len(pieces)):
                if gi + 1 < len(pieces):
                    load_wo(gi + 1)
                for qi, grp in enumerate(groups):
                    proj_group(gi, qi, grp)
                finish_piece(gi)
        else:
            for qi in range(len(groups)):
                proj_group(0, qi, groups[qi])
            finish_piece(0)
            for gi in range(1, len(pieces)):
                for qi, grp in enumerate(groups):
                    proj_group(gi, qi, grp)
                finish_piece(gi)
        # The copies of the RS results into y come after the last piece. A DMA waits in its engine's in-order
        # queue for its RS: a copy issued right after an RS would stop that engine (and the PSUM evacuations of
        # the next piece on it), so TensorE would wait for each RS.
        for n0, pw, dst in rs_dsts:
            nisa.dma_copy(dst=self.y.ap(pattern=[[H, Tl], [1, pw]], offset=n0), src=dst, **_DMA_ACT)
        return self.y


@nki.jit
def kimi_k3_kda_block_prefill(
    x,
    w_qkvf,
    w_gb,
    w_fb_t,
    conv_w,
    conv_state,
    A_log,
    dt_bias,
    S0,
    lcat,
    wcat,
    mask_a3,
    mask_b3,
    mask_lvl,
    ident,
    o_norm_w,
    o_proj_w_t,
    valid_len=None,
    valid_start=None,
    world_ranks: tuple = (),
    lower_bound: float = -5.0,
    eps: float = 1e-5,
    rs_pieces: int = 1,
    rs_piece_cols: int = 0,
    ag_pieces: int = 1,
    ag_piece_chunks: int = 0,
    ag_shared: bool = False,
    block0_in_order: bool = True,
    x_dma_scalar: bool = False,
    scan_decay_onezero: bool = False,
    oproj_batch_stores: bool = False,
    zgate_slots: int = 2,
    scan_count_tile: bool = False,
    ag_row_halves: int = 0,
    scan_chunks_per_iter: int = 2,
    valid_row_window: tuple | None = None,
    debug_outputs: bool = False,
    kernel_digest: str = "",
    mx: int = 0,
    w_qkv_mx=None,
    w_qkv_scale=None,
    w_gb_mx=None,
    w_gb_scale=None,
    lead_outputs: bool = False,
    scan_head_groups: int | None = None,
    scan_block_inverse=None,
    post_block_tokens: int = 0,
):
    """Run the Kimi-K3 prefill KDA attention block of one layer on the row block of this rank.

    Serving shape (TP32): W 32, Tl 128, T 4096, H 7168, 3 local heads of 128.
    TP16 form: W 16, 6 local heads (Tl 256 at T 4096, Tl 512 at T 8192). TP8 form: W 8, 12 local heads (up to T
    4096). hc = S0.shape[0] (3 or a multiple of 3). With more than 3 heads the kernel runs one 3-head front per group
    of 3 heads (ng = hc / 3 groups), sharing one input all-gather and one f_a, and the scan and the tail run once over
    all hc heads. The 3-head program is unchanged.

    Dimensions:
        W: Ranks of one rank list of ``world_ranks``.
        Tl: Rows of this rank (a multiple of 128). T = W * Tl, at most 8192, with hc * T at most 12 * 4096: the tail
            keeps y^T [128, hc, T] bf16 resident.
        H: Hidden (7168). hc: local heads (3 or a multiple of 3). D: head dimension (128). p = hc * D. ng = hc / 3.

    Args:
        x (nl.NkiTensor): [Tl, H] bf16, the normed row block of this rank (the ``input_layernorm`` output).
        w_qkvf (nl.NkiTensor): [3 * hc + 1, 128, H / 128, 128] bf16, group-major: group g's q, k, v conv tiles
            (9 tiles) at [9 g : 9 g + 9], then one shared f_a tile at [3 * hc]. At 3 heads this is the [10, ...]
            vendored layout. With ``mx`` 1 the kernel reads only the f_a tile, so ``w_qkvf`` can also be that tile
            alone, [1, 128, H / 128, 128] (the f_a-only form).
        w_gb (nl.NkiTensor): [128, H / 128, ng * (p3 + 3)] bf16 (p3 = 384), group-major: group g's g_proj (384)
            and b_proj (3) at columns [387 g : 387 g + 387]. With ``mx`` 1 the kernel does not read it, so it can be
            None. The f_a-only form and None give the bits of the full layouts.
        w_fb_t (nl.NkiTensor): [128, p] bf16, ``W_fb`` transposed, group-major: group g at columns [384 g : 384 g + 384].
        conv_w (nl.NkiTensor): [3, p, 4] fp32, the q, k and v conv weights (channel = head * 128 + c, heads in order).
        conv_state (nl.NkiTensor): [3, p, 3] bf16, the old conv state (same channel layout as conv_w).
        A_log (nl.NkiTensor): [1, hc] fp32.
        dt_bias (nl.NkiTensor): [hc, D] fp32.
        S0 (nl.NkiTensor): [hc, D, D] fp32, the old KDA state. ``hc`` is read from its first axis.
        lcat, wcat, mask_a3, mask_b3, mask_lvl, ident (nl.NkiTensor): the ``kda_scan_constants``
            tensors (``prefill/kda.py``). ``ident`` [128, 128] fp32 also serves the TensorE transposes.
        o_norm_w (nl.NkiTensor): [1, D] fp32, the output norm weight.
        o_proj_w_t (nl.NkiTensor): [p, H] bf16, the K-major ``o_proj`` row shard.
        valid_len (nl.NkiTensor): Optional [1, 1] int32. Rows from ``valid_len`` on are pad rows.
        valid_start (nl.NkiTensor): Optional [1, 1] int32. Rows before ``valid_start`` are pad rows (the cached rows of a
            prefix-cache hit). Needs ``valid_len``, and must be below T (see Notes).
        world_ranks (tuple): The replica partition of the group (tuple of rank tuples).
        lower_bound (float): The log-gate lower bound, in [-8, 0).
        eps (float): The output norm epsilon.
        rs_pieces (int): Column pieces of ``o_proj``; each piece has its own reduce-scatter. The default 1 is
            the fastest form on trn3 (a DMA issued during a collective waits for it, so pieces do not overlap
            the stores), and its RS has the size and layout of the compiler's RS of the torch block. Above 6
            local heads, with more than 1 piece, the tail streams the ``o_proj`` weight through a two-slot ring
            (one piece resident, the next prefetched) so the weight and y^T fit SBUF at T 4096; 3 and 6 heads
            keep the all-resident weight. For 12 heads at the serving hidden, 4 pieces (1792 columns) fit.
            Above 6 heads and 2048 rows with 1 piece, the tail keeps the single reduce-scatter but streams the
            weight through the same two-slot ring in 4 column pieces, so the weight and y^T fit SBUF at T 4096.
            The outputs and the T <= 2048 program do not change.
        rs_piece_cols (int): If > 0, pieces of this many columns (a multiple of 128) in place of ``rs_pieces``
            equal pieces. A power-of-two piece size (for example 2048 at Tl 128: 512 KiB per RS result) keeps
            the RS results aligned with no padding.
        ag_shared (bool): Put the gathered image in shared_hbm. With NKI_LNC1_SHARED_HBM=1 at compile time, the
            two cores of an HBM stack then share one copy, and the all-gather moves half the bytes into each stack.
            Without the flag the image stays private. The outputs do not change.
        block0_in_order (bool): Run the conv and prep of the first 512 rows in order (the conv state is ready at
            the start). False keeps the vendored order (block 0 last). The outputs do not change.
        ag_piece_chunks (int): If > 0, AG pieces of this many hidden chunks (128 each) in place of ``ag_pieces``
            equal pieces (for example 32 at H 7168: pieces of 32 and 24 chunks, 32 MiB per gathered piece).
        ag_pieces (int): Hidden pieces of the layer-input all-gather. With more than 1, the front starts the
            first block on the first pieces while the later pieces move. The outputs do not change.
        x_dma_scalar (bool): Load the gathered image on the ScalarE DMA queue (default: the Sync queue).
        scan_decay_onezero (bool): Give the nested scan ``decay_onezero``: its two decay-sum matmuls skip the
            low-half pass of their 0/1 moving operand. The outputs do not change.
        oproj_batch_stores (bool): Store the o_proj output to the RS source per PSUM column batch (up to 2048
            columns), each store right after its batch's evacuation, from one stage tile for all 128-row groups. The
            default stores each group once, after its last batch, and the next group's first evacuation into the
            stage waits for that whole store. The same SBUF, 4x the stores; the outputs do not change.
        zgate_slots (int): Slots of the front's z-gate tile ring (each slot is the gate of one head, stored to the
            scan's gate input). The default 2 makes the gate of head h + 2 wait for the store of head h; 3 adds
            2 KiB per partition. The outputs do not change.
        scan_count_tile (bool): Give the nested scan ``own_count_tile``: it computes its loop counts from
            ``valid_len`` in a tile of its own, so it does not write the tile its ``valid_len`` load shares with the
            front's. The outputs do not change.
        ag_row_halves (int): If > 0, gather the layer input in two row halves (rows 0-63 and 64-127 of every
            local tile, one AG each) and start the front on the first half: the conv projections of the first-half
            rows of the first ``2 * ag_row_halves`` 512-row blocks run, as pairs of blocks, while the second AG
            moves. Their bf16 results wait in the pre-conv scratch, and those blocks then project only their
            second-half rows. Each projection sums the same products in the same chunk order and rounds the same
            way, so the outputs do not change. Needs one input-gather piece and ``block0_in_order``.
        scan_chunks_per_iter (int): Chunks per iteration of the nested scan's main loop (its
            ``chunks_per_iter``: 2, the default, or 4). Each iteration ends with all-engine barriers; with 4 the
            scan runs about half as many. Each chunk runs the same steps in the same order, so the outputs do not
            change.
        scan_head_groups (int | None): The nested scan's ``head_groups``: its main loop runs the hc heads in this
            many groups, one after the other; the outputs do not change. None (default): ``TP8_SCAN_HEAD_GROUPS`` (6)
            on Trn3 with 12 local heads (the TP8 form), else 1. 1 runs all heads in one group (the program without
            head groups).
        scan_block_inverse: The nested scan's ``block_inverse`` hook for the low step-3 inverse levels. None, the
            default, gives the program without the hook. A hook changes the rounding of the scan (an accuracy-class
            option).
        post_block_tokens (int): Tokens per partition of a tail post block (J): 2 or 4. 0, the default, takes 4, or 2
            above 6 local heads or above 4096 rows, where the post ring and y^T need the smaller ring to fit SBUF. J
            only moves tokens between partitions and post blocks: each token gets the same norm and gate steps, and
            each o_proj element the same sum over h = 0, 1, ... At H 7168 the PSUM evacuation engine of an element
            also does not depend on J (an even number of 512-column chunks per row group). The outputs do not
            change. A forced 4 above 4096 rows at 6 heads and H 7168 compiles, but the compiler spills about
            120 KiB a partition of SBUF to HBM.
        valid_row_window (tuple | None): With ``valid_start``, the parts that skip the work of their whole row
            tiles before it (see ``valid_row_window.py``). The default None takes ``("scan",)`` on Trn2 and ()
            on Trn3. () gives the program without the window, and so does any window without
            ``valid_start``. The outputs do not change. Items:

            - ``"scan"``: give ``valid_start`` to the nested scan too. The scan's loop then starts at the first
              128-row chunk that holds a row at or after valid_start (at least chunk 2), so the whole chunks of
              cached rows before it do not run. Those rows are identity steps of the scan (zero inputs, ``S0`` the
              state after them): they leave the state unchanged and have zero output, which the skipped chunks
              keep.
            - ``"oproj"``: not built in this version (refused).
        debug_outputs (bool): Also return the intermediates (see Returns).
        kernel_digest (str): Unused in the trace; it keys the NKI caches.
        lead_outputs (bool): Also return S_lead and conv_lead, the state and conv history of row
            g = max(vs, ((max(valid_len, 1) - 1) // 128) * 128), vs the valid_start clamped to [0, valid_len] (0
            without one): the start of the last 128-row chunk with a valid row (see Notes). Needs ``valid_len``,
            refuses ``ag_row_halves``, and with ``valid_start`` needs ``"scan"`` in ``valid_row_window``. False, the
            default, gives the program without it.

    Returns:
        y [Tl, H] bf16 (the pad rows are 0), conv_new [3, p, 3] bf16, S_new [hc, D, D] fp32. With ``lead_outputs``,
        then S_lead [hc, D, D] fp32 and conv_lead [3, p, 3] bf16 (see Notes). With ``debug_outputs``: then qT, kT,
        kbT [hc, D, T] fp32, vb, gtok [hc, T, D] fp32, g [T, p] fp32, o [hc, T, D] fp32 (the raw scan output),
        y_pre [T, p] bf16 (the ``o_proj`` input).

    Notes:
        - The valid rows follow the unfused KDA path: the front, the scan ``kda_prefill_v2`` and the output
          norm and gate.
        - A NaN or Inf in a pad row does not reach a valid row or the state: every pad mask is a select.
        - With ``valid_start`` = vs (clamped to [0, valid_len]): the rows before vs are identity steps of the scan (q, k,
          v, kb = 0, decay 1) and their y is 0. ``conv_state`` is the conv history of row vs (the pre-conv rows of the
          tokens vs - 3 .. vs - 1) and ``S0`` the state after them. ``conv_new`` holds the pre-conv rows before
          ``valid_len``: rows of ``conv_state`` when ``valid_len - vs < 3``. vs = 0 gives the bits of no
          ``valid_start``. It needs ``block0_in_order`` and refuses ``ag_row_halves``. Each head group of a
          multi-group block takes its channels of ``conv_state``.
        - With ``lead_outputs`` (g as above): ``S_lead`` is the scan state at row g (after the rows before g) and
          ``conv_lead`` the conv history of row g (the pre-conv rows of the tokens g - 3 .. g - 1). A later call on
          the same segment with ``valid_start`` = g, ``S0`` = S_lead and ``conv_state`` = conv_lead gives the bits
          of this call on the rows from g on: the 128-row chunks of the scan start at the call's first row, so they
          are the same chunks. vs = valid_len < T gives S_lead = S0 = S_new and conv_lead = conv_new.
        - valid_start must be below T. At vs = valid_len = T, with T a power-of-two multiple of 512 and at least
          1024, the conv-state window has no column for the state: its second write ends one element past the
          window scratch (the simulator stops with an out-of-bound access; hardware does not check). A hit
          gives vs = P % T < T. The kernel cannot refuse it: valid_start is a runtime value.

    Pseudocode:
        img = all_gather(x^T | (x @ W_fa^T)^T)
        qT, kT, kbT, vb, gtok, g, conv_new = kda_front(img, conv_state, valid_len, valid_start)
        o, S_new = kda_prefill_v2(qT, kT, kbT, vb, gtok, S0, valid_len)
        y = reduce_scatter(post(o, g) @ W_o^T)
    """
    kernel_assert(len(world_ranks) > 0, "world_ranks must hold the replica partition of the group")
    Tl, H = x.shape
    hc = int(S0.shape[0])
    plan = KdaBlockPlan(tuple(tuple(int(r) for r in grp) for grp in world_ranks), int(Tl), int(H), heads=hc)
    plan.check()
    kernel_assert(x.dtype == _BF16, f"x must be bfloat16, got {x.dtype}")
    kernel_assert(hc > 0 and hc % _HEADS == 0, f"heads {hc} must be a positive multiple of {_HEADS}")
    kernel_assert(tuple(S0.shape) == (plan.heads, plan.head_dim, plan.head_dim), f"S0 shape {S0.shape}")
    kernel_assert(tuple(o_proj_w_t.shape) == (plan.heads * plan.head_dim, H), f"o_proj_w_t shape {o_proj_w_t.shape}")
    kernel_assert(tuple(o_norm_w.shape) == (1, plan.head_dim), f"o_norm_w shape {o_norm_w.shape}")
    if plan.heads > _HEADS:
        # These options are not generalized to the head-group loop yet; refuse them rather than run a wrong program.
        kernel_assert(int(mx) in (0, 1), "the multi-group KDA block supports mx 0 or 1 (MXFP8), not mx 2 or 3")
        kernel_assert(int(ag_row_halves) == 0, "the multi-group KDA block refuses ag_row_halves")
        kernel_assert(bool(block0_in_order), "the multi-group KDA block needs block0_in_order")
    group = ReplicaGroup(plan.world_ranks)
    opts = dict(
        eps=float(eps),
        rs_pieces=int(rs_pieces),
        rs_piece_cols=int(rs_piece_cols),
        ag_pieces=int(ag_pieces),
        ag_piece_chunks=int(ag_piece_chunks),
        ag_shared=bool(ag_shared),
        block0_in_order=bool(block0_in_order),
        x_dma_scalar=bool(x_dma_scalar),
        oproj_batch_stores=bool(oproj_batch_stores),
        zgate_slots=int(zgate_slots),
        ag_row_halves=int(ag_row_halves),
        debug_outputs=bool(debug_outputs),
        fa_only=int(w_qkvf.shape[0]) == 1,
        post_block_tokens=int(post_block_tokens),
    )
    kernel_assert(opts["zgate_slots"] >= 2, f"zgate_slots must be >= 2, got {zgate_slots}")
    kernel_assert(
        opts["post_block_tokens"] in (0, 2, 4), f"post_block_tokens must be 0, 2 or 4, got {post_block_tokens}"
    )
    if opts["fa_only"] or w_gb is None:
        # Only the MXFP8 front (mx 1) has no read of the bf16 conv tiles and of w_gb.
        kernel_assert(int(mx) == 1, f"the f_a-only w_qkvf and a None w_gb need mx 1, got mx {mx}")
    if opts["ag_row_halves"]:
        n_blocks = -(-plan.tokens // 512)
        kernel_assert(
            opts["ag_pieces"] == 1 and opts["ag_piece_chunks"] == 0,
            f"ag_row_halves takes one input-gather piece, got ag_pieces {ag_pieces}, ag_piece_chunks {ag_piece_chunks}",
        )
        kernel_assert(opts["block0_in_order"], "ag_row_halves needs block0_in_order")
        kernel_assert(
            0 < opts["ag_row_halves"] and 2 * opts["ag_row_halves"] <= n_blocks and plan.tokens % 512 == 0,
            f"ag_row_halves {ag_row_halves} must be in [1, {n_blocks // 2}] with T a multiple of 512, T = {plan.tokens}",
        )
    if valid_start is not None:
        kernel_assert(valid_len is not None, "valid_start needs valid_len")
        kernel_assert(
            tuple(valid_start.shape) == (1, 1) and valid_start.dtype == nl.int32,
            f"valid_start must be [1, 1] int32, got {valid_start.shape} {valid_start.dtype}",
        )
        kernel_assert(not opts["ag_row_halves"], "valid_start refuses ag_row_halves")
        kernel_assert(opts["block0_in_order"], "valid_start needs block0_in_order")
    # The items of valid_row_window that apply (none without valid_start). The default is the scan skip on Trn2,
    # where it is measured, and no window on Trn3.
    if valid_row_window is None:
        valid_row_window = ("scan",) if is_trn2() else ()
    window = window_items(valid_row_window, KDA_ITEMS, ("scan",), "kimi_k3_kda_block_prefill")
    opts["window"] = window if valid_start is not None else frozenset()
    if is_trn2():
        # A multi-piece input gather restores partial sums into PSUM with a copy and accumulates onto them
        # (block_front.w_block0_pieces). Trn2 does not support a matmul accumulation onto such PSUM.
        kernel_assert(
            opts["ag_pieces"] == 1 and opts["ag_piece_chunks"] == 0,
            f"Trn2 takes one input-gather piece, got ag_pieces {ag_pieces}, ag_piece_chunks {ag_piece_chunks}",
        )
    if int(mx) != 0:
        # The MX weights are group-major: 3 * hc conv tiles, ng * 387 g|b columns (the identity at 3 heads).
        ng = plan.heads // _HEADS
        _check_mx(
            int(mx), plan.hidden // 128, 3 * plan.heads, ng * _GB_COLS, w_qkv_mx, w_qkv_scale, w_gb_mx, w_gb_scale
        )
        kernel_assert(
            opts["ag_pieces"] == 1 and opts["ag_piece_chunks"] == 0 and not opts["ag_row_halves"],
            "the MXFP8 KDA block front needs one input-gather piece and no ag_row_halves",
        )
    lead_outputs = bool(lead_outputs)
    if lead_outputs:
        kernel_assert(valid_len is not None, "lead_outputs needs valid_len")
        kernel_assert(not opts["ag_row_halves"], "lead_outputs refuses ag_row_halves")
        # With valid_start, the scan's identity guard (whole chunks before it keep S0) needs the scan to see it.
        kernel_assert(
            valid_start is None or "scan" in opts["window"],
            'lead_outputs with valid_start needs "scan" in valid_row_window',
        )
    blk = _KdaBlock(plan, group, opts)
    if opts["ag_row_halves"]:
        blk.gather_input_halves(x, w_qkvf, ident)
    else:
        blk.gather_input(x, w_qkvf, ident)
    front_out = blk.front(
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
        valid_start,
        mx=int(mx),
        w_qkv_mx=w_qkv_mx,
        w_qkv_scale=w_qkv_scale,
        w_gb_mx=w_gb_mx,
        w_gb_scale=w_gb_scale,
        lead_outputs=lead_outputs,
    )
    qT, kT, kbT, vb, gtok, g, conv_new = front_out[:7]
    # The scan output is scratch of this kernel: private (see o_private). Its state is an output.
    # The front makes the pad rows of all five scan inputs exact zeros (selects), so the scan skips its own
    # pad-row zeroing (pad_rows_zero). The pair-matrix PSUM zeroing runs early (pair_psum_zero), which gives the
    # same bits in less time.
    scan_kwargs = {
        "lower_bound": float(lower_bound),
        "o_private": True,
        "pad_rows_zero": True,
        "pair_psum_zero": "early",
    }
    if scan_decay_onezero:
        scan_kwargs["decay_onezero"] = True
    if int(scan_chunks_per_iter) != 2:
        scan_kwargs["chunks_per_iter"] = int(scan_chunks_per_iter)
    if scan_head_groups is None:
        scan_head_groups = TP8_SCAN_HEAD_GROUPS if hc == 12 and not is_trn2() else 1
    if int(scan_head_groups) != 1:
        scan_kwargs["head_groups"] = int(scan_head_groups)
    if scan_block_inverse is not None:
        scan_kwargs["block_inverse"] = scan_block_inverse
    if valid_len is not None:
        scan_kwargs["valid_len"] = valid_len
        if scan_count_tile:
            scan_kwargs["own_count_tile"] = True
    if "scan" in opts["window"]:
        # The front makes the rows before valid_start zero scan inputs (selects), and S0 is the state after them.
        scan_kwargs["valid_start"] = valid_start
    if lead_outputs:
        # The scan also returns its state at row g, the start of the last chunk it runs (see lead_outputs).
        scan_kwargs["lead_state"] = True
        o, s_new, s_lead = kda_prefill_v2(
            qT, kT, kbT, vb, gtok, S0, lcat, wcat, mask_a3, mask_b3, mask_lvl, ident, **scan_kwargs
        )
        lead = (s_lead, front_out[7])
    else:
        o, s_new = kda_prefill_v2(
            qT, kT, kbT, vb, gtok, S0, lcat, wcat, mask_a3, mask_b3, mask_lvl, ident, **scan_kwargs
        )
        lead = ()
    y = blk.tail(o, g, o_norm_w, o_proj_w_t, valid_len)
    if debug_outputs:
        return (y, conv_new, s_new, *lead, qT, kT, kbT, vb, gtok, g, o, blk.y_pre)
    return (y, conv_new, s_new, *lead)
