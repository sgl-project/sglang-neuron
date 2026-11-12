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

"""Standalone V-up + output projection (CTE), MX or BF16.

KERNEL B of the split sparse-MLA forward. Reads the latent attention
output out_attn_hbm[B=1, S, H*L] (from kernel A, mla_sparse_attention_cte_kernel), V-ups each
head's latent with W_uv, then projects the H*d_v activation with W_o into
out_hbm[B=1, S, HID]. S-sharded (each core does its own queries over the full HID).

MX (:func:`_vupmx_oproj_stage`, the default): fp8x4 weights with block-128 scales. Budget-aware
o_proj weight residency — the full W_o is loaded once when it fits SBUF (TIER 1), else
K-slab-streamed double-buffered (TIER 3). Consumes the cross-kernel MX 4-pack column layout
kernel A writes into out_attn (each (k512, sub)'s latent groups contiguous in HBM), so the
latent transpose is a contiguous swdge dma_transpose off the Tensor Engine.

BF16 (:func:`_vup_oproj_stage_bf16`): plain bf16 weights in natural column order and back-to-back
nc_matmul. Required for GLM-MoE-DSA (GLM-5.2), whose v_head_dim is 256 — a value the MX
path cannot express at all, since its pre-swizzled transpose hard-requires d_v == 128.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from .common import _get_psum_bank_size
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from .mla_bf16 import (
    _BF16_K_TILE,
    _NUM_TRANSPOSE_BANKS,
    _bf16_compute_k_slab_tiles,
    _bf16_k_tile_sizes,
    _load_bf16_weights_k_slab,
    _transpose_bf16_to_k_major,
)
from .common import (
    _H_PACK,
    _MM1_TILE,
    _NUM_HW_PSUM_BANKS,
    _P_MAX,
    MlaPrecision,
    _dma_transpose_latent_for_mx,
    _load_mx_weights,
    _load_mx_weights_k_slab,
    _new_sbm,
    _transpose_preswizzled_for_mx,
)
from .validate import _validate_mla_vupmx_oproj_inputs

_OPROJ_S_TILE = 128
_OPROJ_HID_TILE = 1024
_OPROJ_K_SLAB_512 = 8
_OPROJ_STAGE_512 = 8


@nki.jit
def mla_vupmx_oproj_cte_kernel(
    out_attn_hbm: nl.NkiTensor,
    wuv_qtz_hbm: nl.NkiTensor,
    wuv_scale_hbm: nl.NkiTensor,
    wo_qtz_hbm: nl.NkiTensor,
    wo_scale_hbm: nl.NkiTensor,
    compact_scales: bool = True,
    precision: MlaPrecision = MlaPrecision.MX,
    kv_lora_rank: int = _P_MAX * _H_PACK,
) -> nl.NkiTensor:
    """Standalone V-up + output projection, MX or BF16 (S-sharded across cores).

    KERNEL B of the split sparse-MLA forward. Reads the latent attention
    output from kernel A (mla_sparse_attention_cte_kernel), V-ups each head's latent with W_uv
    (MX fp8x4), then projects the H*d_v activation with W_o (MX) into out_hbm[B=1, S, HID].
    Each core projects its own queries over the full HID. Intended for Context Encoding with
    GLM-5.2's dims (L == 512 kv_lora_rank, d_v == 128, H a multiple of 4 up to 128);
    requires B == 1 and S divisible by the number of cores. Budget-aware o_proj weight
    residency: full W_o loaded once when it fits SBUF, else K-slab-streamed double-buffered.

    Dimensions:
        B: Batch size (must be 1)
        S: Sequence length (this rank's S-shard)
        H: Number of attention heads (a multiple of 4)
        L: Latent (kv_lora_rank) dimension (fixed at 512 = P_MAX * 4)
        d_v: Per-head V dimension (must be 128 = P_MAX)
        HID: Output projection (hidden) dimension
        Hdv: H * d_v (the o_proj contraction)

    ``precision`` selects the arithmetic format; the args below are described for the default
    MX path. BF16 is what GLM-MoE-DSA (GLM-5.2) needs — note GLM's ``v_head_dim`` is
    **256**, which the MX path cannot express (it hard-requires ``d_v == 128``).

    Args:
        out_attn_hbm (nl.NkiTensor): [B, S, H*L] bf16 latent attention output from kernel A,
            carrying the cross-kernel MX 4-pack column layout.
            BF16: same shape, but NATURAL latent column order (no 4-pack permutation).
        wuv_qtz_hbm (nl.NkiTensor): [H*L // 4, d_v] fp8x4 packed MX V-up weight.
            BF16: [H*L, d_v] bfloat16.
        wuv_scale_hbm (nl.NkiTensor): [H*L // 128, ceil(d_v / 128)] uint8 compact block-128 scales.
            BF16: ignored, pass ``None``.
        wo_qtz_hbm (nl.NkiTensor): [H*d_v // 4, HID] fp8x4 packed MX o_proj weight.
            BF16: [H*d_v, HID] bfloat16.
        wo_scale_hbm (nl.NkiTensor): [H*d_v // 128, ceil(HID / 128)] uint8 compact block-128 scales.
            BF16: ignored, pass ``None``.
        compact_scales (bool): MX only — see ``wuv_scale_hbm``.
        precision (MlaPrecision): ``MX`` (default) or ``BF16``. BF16 drops the 4-packing, the
            Hdv output-column swizzle and both ``quantize_mx`` passes, and lifts the MX path's
            ``d_v == 128`` and ``H % 4 == 0`` restrictions (GLM needs d_v=256). Pass ``None``
            for both scale tensors.
        kv_lora_rank (int): L, the latent dimension. Only read on the BF16 path; the MX path
            pins L to 512 (one MX 512-tile per head) and ignores this. Defaults to 512, which
            is what GLM-5.2 uses.

    Returns:
        out (nl.NkiTensor): [B, S, HID] bf16 output projection result.

    Notes:
        - H is recovered from tensor shapes as ``H*L / L``: wuv is [H*L // 4, d_v] (MX) or
          [H*L, d_v] (BF16), and L comes from ``kv_lora_rank``. Do NOT pass H as a runtime
          scalar — the framework materializes it as an HBM tensor, not a trace-time int, so
          shape math derived from it is garbage.
        - MX consumes the cross-kernel 4-pack column layout kernel A writes into out_attn
          (each (k512, sub)'s latent groups contiguous in HBM), so the latent transpose is a
          contiguous swdge dma_transpose off the Tensor Engine. BF16 has no such layout: the
          latent is natural and is transposed on the Tensor Engine.

    Pseudocode:
        attn = out_attn[q]                       # de-permute 4-pack layout -> natural
        for h in range(H):
            attn_v[..., h, :] = attn[..., h, :] @ W_uv[h]   # MX V-up per head
        out[q] = attn_v.reshape(S, H*d_v) @ W_o             # MX o_proj
        return out
    """
    _validate_mla_vupmx_oproj_inputs(
        out_attn_hbm,
        wuv_qtz_hbm,
        wuv_scale_hbm,
        wo_qtz_hbm,
        wo_scale_hbm,
        compact_scales=compact_scales,
        precision=precision,
        kv_lora_rank=kv_lora_rank,
    )

    B, S, HL = out_attn_hbm.shape
    # MX pins L to one 512-wide MX tile per head; BF16 has no packing constraint and takes L
    # from kv_lora_rank.
    L = kv_lora_rank if precision.is_bf16() else _P_MAX * _H_PACK
    H = HL // L
    d_v = wuv_qtz_hbm.shape[1]
    HID = wo_qtz_hbm.shape[1]

    _, n_prgs, prg_id = get_verified_program_sharding_info("mla_vupmx_oproj_cte_kernel", (0, 1))
    kernel_assert(S % n_prgs == 0, f"S={S} must be divisible by n_prgs={n_prgs}")
    s_per_core = S // n_prgs
    s_start = prg_id * s_per_core

    out_hbm = nl.ndarray((B, S, HID), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    sbm = _new_sbm("sparse_mla_vupmx_oproj")
    # S-sharded, full HID.
    if precision.is_bf16():
        _vup_oproj_stage_bf16(out_attn_hbm, wuv_qtz_hbm, wo_qtz_hbm, out_hbm, H, L, sbm, s_start, s_per_core, 0, HID)
    else:
        _vupmx_oproj_stage(
            out_attn_hbm,
            wuv_qtz_hbm,
            wuv_scale_hbm,
            wo_qtz_hbm,
            wo_scale_hbm,
            out_hbm,
            H,
            L,
            sbm,
            s_start,
            s_per_core,
            0,
            HID,
            compact_scales=compact_scales,
        )
    return out_hbm


def _vup_oproj_stage_bf16(
    out_attn_hbm,
    wuv_hbm,
    wo_hbm,
    out_hbm,
    H,
    L,
    sbm,
    s_start,
    s_per_core,
    hid_start,
    hid_per_core,
    s_tile_cap=_OPROJ_S_TILE,
):
    """BF16 V-up + o_proj (``MlaPrecision.BF16``). Same two matmuls as :func:`_vupmx_oproj_stage`
    with every MX-format step deleted rather than reimplemented.

    Beyond dropping the 4-pack / swizzle / quantize machinery, the bf16 dataflow is
    structurally cheaper by one whole transpose stage. The MX path must produce ``attn_v`` in
    ``[s, Hdv]`` (M-major, because ``nc_matmul_mx`` puts the s-tile on the stationary free
    axis), then transpose + quantize all of Hdv to feed the o_proj. Here the V-up instead
    makes the WEIGHT the stationary operand, so it emits ``attn_v`` already TRANSPOSED as
    ``[d_v, s]`` per head:

        dst[M=d_v_tile, N=s] = stationary(W_uv[L_tile, d_v_tile]).T @ moving(attn_t[L_tile, s])

    Stacking those per-head ``[128, s]`` results over the head axis lands exactly the
    ``[P_MAX, Hdv/128, s]`` K-major buffer the o_proj wants as ITS stationary operand. So the
    only transposes in the kernel are on the latent, and Hdv is never transposed at all.

    GLM-MoE-DSA needs this path for a second reason beyond dtype: its ``v_head_dim`` is 256,
    and the MX path hard-requires ``d_v == 128`` (its pre-swizzled transpose assumes one head
    per 128-column group). Here ``d_v`` is tiled at 128 on the stationary free axis, so 256
    is just two tiles. ``H % 4 == 0`` also drops away with the 4-pack.

    Weight residency: W_o is K-slab-streamed, double-buffered when the slab depth survives the
    second buffer (see the slab-depth block below), and W_uv is loaded per head group. bf16
    costs 2 B/elem against MX's ~1.25, so the MX path's TIER-1 full-W_o residency almost never
    applies here. See the same note on ``_qkv_stage_bf16``.
    """
    d_v = wuv_hbm.shape[1]
    HID = wo_hbm.shape[1]
    HL = H * L
    Hdv = H * d_v

    kernel_assert(
        hid_per_core % _MM1_TILE == 0,
        f"_vup_oproj_stage_bf16: hid_per_core ({hid_per_core}) must be a multiple of {_MM1_TILE}",
    )
    kernel_assert(L % _BF16_K_TILE == 0, f"_vup_oproj_stage_bf16: L ({L}) must be a multiple of {_BF16_K_TILE}")
    """
    d_v must tile evenly at 128 so head h owns exactly ``d_v // 128`` whole partition tiles of
    the K-major attn_v buffer, i.e. its Hdv rows never straddle a tile boundary. A d_v of 128 and
    GLM-5.2's 256 both satisfy this; it is far weaker than the MX path's ``d_v == 128``.
    """
    kernel_assert(
        d_v % _BF16_K_TILE == 0,
        f"_vup_oproj_stage_bf16: d_v ({d_v}) must be a multiple of {_BF16_K_TILE}",
    )
    kernel_assert(0 < H <= _P_MAX, f"_vup_oproj_stage_bf16: H (heads/rank) must be in (0, {_P_MAX}], got {H}")

    oproj_s_tile = min(s_tile_cap, s_per_core)
    kernel_assert(
        s_per_core % oproj_s_tile == 0, f"_vup_oproj_stage_bf16: s_per_core must be a multiple of {oproj_s_tile}"
    )
    num_s_batches = s_per_core // oproj_s_tile
    # Same real/padded s split as the MX stage: a stationary free dim of 1 (s_per_core == 1,
    # e.g. S=2 over 2 cores) fails MLIR verification, so compute on a padded width and use the
    # real width for every HBM access.
    _S_TILE_PAD_MULT = 16
    oproj_s_tile_pad = div_ceil(oproj_s_tile, _S_TILE_PAD_MULT) * _S_TILE_PAD_MULT

    L_K_TILES = L // _BF16_K_TILE
    DV_TILES = d_v // _BF16_K_TILE
    HDV_K_TILES = Hdv // _BF16_K_TILE  # == H * DV_TILES

    # HID PSUM-tile, shrunk until the o_proj N-banks fit alongside the reserved transpose banks.
    oproj_hid_tile = _OPROJ_HID_TILE
    while (
        hid_per_core % oproj_hid_tile != 0 or _NUM_TRANSPOSE_BANKS + (oproj_hid_tile // _MM1_TILE) > _NUM_HW_PSUM_BANKS
    ):
        oproj_hid_tile -= _MM1_TILE
    kernel_assert(
        oproj_hid_tile >= _MM1_TILE, f"_vup_oproj_stage_bf16: cannot fit PSUM banks for hid_per_core={hid_per_core}"
    )
    num_hid_tiles = hid_per_core // oproj_hid_tile
    num_n_tiles = oproj_hid_tile // _MM1_TILE
    PSUM_BANK_SIZE = _get_psum_bank_size()

    _CALIBRATED_SBUF_BUDGET_BYTES = 240 * 1024
    _sbuf_budget = min(int(nl.tile_size.total_available_sbuf_size), _CALIBRATED_SBUF_BUDGET_BYTES)

    sbm.open_scope()

    """
    attn_v in K-major form, live across both phases: the V-up writes it, the o_proj reads it as
    its stationary operand. [P_MAX, Hdv/128, s_pad] bf16 = Hdv/128 * s_pad * 2 B/partition
    (32KB at GLM's unsharded Hdv=16384, s=128).
    """
    attn_vt = None  # allocated below, once S_GROUP is known (one buffer per s-tile in flight)

    """
    Head-group size for the V-up: the latent stage buffer, its transpose and the W_uv slice all
    scale with the group, so pick the largest power-of-two-ish divisor of H that fits a modest
    slice of the budget. Per head the three cost L*2 + L*2 + L*d_v/128*2 bytes/partition, so
    even large groups are cheap; the cap mostly keeps the latent DMA chunky.
    """
    _VUP_GROUP_BUDGET = 96 * 1024
    _per_head_bytes = L * 2 + L * 2 + L_K_TILES * d_v * 2
    _head_group_choices = [g for g in range(H, 0, -1) if H % g == 0 and g * _per_head_bytes <= _VUP_GROUP_BUDGET] or [1]
    VUP_HEAD_GROUP = _head_group_choices[0]
    num_vup_groups = div_ceil(H, VUP_HEAD_GROUP)

    """
    ---- W_o K-slab depth ----
    bf16 W_o is [Hdv, HID]: Hdv/128 K-tiles x hid_per_core x 2 B per partition (1.5MB/partition
    at GLM's unsharded Hdv=16384, HID=6144), so it always slabs. Reserve attn_vt, the
    accumulator + drain buffers and the V-up group working set.
    """
    _reserve_common = (
        HDV_K_TILES * oproj_s_tile_pad * 2  # attn_vt
        + 2 * hid_per_core * 2  # out_acc + out_sb
        + VUP_HEAD_GROUP * _per_head_bytes  # V-up group working set
    )
    _reserve = _reserve_common + 32 * 1024  # allocator slack
    slab_tiles = _bf16_compute_k_slab_tiles(Hdv, hid_per_core, _sbuf_budget, _reserve)
    """
    ---- W_o slab double-buffering ----
    Single-buffering serializes the slab DMA against the slab's matmuls: slab N+1 cannot start
    loading until every matmul reading slab N has retired (WAR on the one buffer).

    Two rules keep this from costing more than it buys:
      * Full residency (num_k_slabs == 1) has nothing to prefetch, so stay single-buffered.
      * Never buy the second buffer by shrinking the slab. Halving slab_tiles doubles
        num_k_slabs, and every extra slab is another full PSUM->SBUF accumulate pass over
        [s_tile, hid_per_core] on the Vector engine. Only double-buffer if the depth survives;
        the probe spends half the allocator slack to make that fit at the current depth.
    """
    n_wo_buf = 1
    if HDV_K_TILES // slab_tiles > 1:
        slab_tiles_db = _bf16_compute_k_slab_tiles(Hdv, hid_per_core * 2, _sbuf_budget, _reserve_common + 16 * 1024)
        if slab_tiles_db >= slab_tiles:
            n_wo_buf, slab_tiles = 2, slab_tiles_db
        elif num_s_batches * oproj_s_tile <= 2 * _MM1_TILE:
            # Short-s shapes invert the rule above: with few output rows per W_o pass the slab
            # DMA, not the matmul, is the stage's critical path (measured: the DMA outlasts the
            # PE per 12 MiB slab at s=64, alternating with zero overlap). The extra accumulate
            # pass per extra slab is a small amount of Vector work, so buy the second buffer by halving
            # the slab depth instead of staying single-buffered.
            cap = min(slab_tiles // 2, slab_tiles_db)
            new_slab = next((t for t in range(cap, 0, -1) if HDV_K_TILES % t == 0), 0)
            if new_slab:
                n_wo_buf, slab_tiles = 2, new_slab
    num_k_slabs = HDV_K_TILES // slab_tiles

    """
    ---- s-tile grouping (W_o re-stream amortization) ----
    W_o streams once per s-batch, so the same [Hdv, HID] weight is re-read num_s_batches times:
    at the GLM 128k row that is 201 MB x 8 = 1.6 GB, 88% of the kernel's entire DMA traffic.
    Double-buffering hides that DMA but cannot shrink it, so once overlap is in place the
    re-stream IS the memory roofline.

    Fix: process ``S_GROUP`` s-tiles per W_o pass. Phase B keeps the slab loop outer and adds an
    inner s-tile loop, so one resident slab feeds S_GROUP stationary operands and W_o traffic
    falls by S_GROUP. Phase V likewise hoists the head-group loop above the s-tile loop so W_uv
    is fetched once per group instead of once per s-batch.

    Cost: S_GROUP live ``attn_vt`` + ``out_acc`` buffers. Funded by shrinking VUP_HEAD_GROUP
    when needed but NEVER by shrinking the slab depth or dropping the second W_o buffer.

    No unroll budget is needed alongside this: the s-group driver below is an ``nl.fori_loop``,
    so the traced graph is O(S_GROUP) rather than O(S) and even the deepest shapes (S_local =
    16384) trace the same size as the shortest. S_GROUP is therefore chosen purely on SBUF.
    """

    def _group_fits(s_grp, head_grp, slack_bytes):
        return (
            s_grp * HDV_K_TILES * oproj_s_tile_pad * 2  # attn_vt, one per s-tile in flight
            + s_grp * hid_per_core * 2  # out_acc, one per s-tile in flight
            + hid_per_core * 2  # out_sb (shared drain staging)
            + head_grp * _per_head_bytes  # V-up group working set
            + n_wo_buf * slab_tiles * hid_per_core * 2  # W_o slab buffer(s)
            + slack_bytes
        ) <= _sbuf_budget

    S_GROUP = 1
    for s_grp in range(num_s_batches, 0, -1):
        if num_s_batches % s_grp != 0:
            continue
        head_grp = next((g for g in _head_group_choices if _group_fits(s_grp, g, 16 * 1024)), None)
        if head_grp is not None:
            S_GROUP, VUP_HEAD_GROUP = s_grp, head_grp
            break
    num_vup_groups = div_ceil(H, VUP_HEAD_GROUP)
    num_s_groups = num_s_batches // S_GROUP

    hdv_k_tile_sizes = _bf16_k_tile_sizes(Hdv)
    wuv_k_tile_sizes = _bf16_k_tile_sizes(HL)

    """
    PSUM map. Banks [0, _NUM_TRANSPOSE_BANKS) are owned by _transpose_bf16_to_k_major (the
    latent transposes); everything else lives above it. The V-up accumulators and the o_proj
    accumulators both sit there but never overlap in time — per s-batch the V-up fully precedes
    the o_proj, same phase ordering the MX stage relies on.
    """
    n_psum_above = _NUM_HW_PSUM_BANKS - _NUM_TRANSPOSE_BANKS
    kernel_assert(
        DV_TILES <= n_psum_above,
        f"_vup_oproj_stage_bf16: d_v={d_v} needs {DV_TILES} V-up PSUM banks, only {n_psum_above} free",
    )

    attn_vt = [
        sbm.alloc_stack(
            (_P_MAX, HDV_K_TILES, oproj_s_tile_pad), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"vup_attn_vt_{j}"
        )
        for j in range(S_GROUP)
    ]
    out_acc = [
        sbm.alloc_stack((oproj_s_tile_pad, hid_per_core), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"oproj_out_acc_{j}")
        for j in range(S_GROUP)
    ]
    out_sb = sbm.alloc_stack((oproj_s_tile_pad, hid_per_core), dtype=nl.bfloat16, buffer=nl.sbuf, name="oproj_out")
    wo_slab = [
        sbm.alloc_stack(
            (_P_MAX, slab_tiles, hid_per_core), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"oproj_wo_slab_{buf_idx}"
        )
        for buf_idx in range(n_wo_buf)
    ]

    """
    ---- s-group driver: rolled (device) vs unrolled (python) ----
    The s-group loop is the ONLY loop here whose trip count scales with S (everything else is
    set by H, Hdv or HID). Driven by python `range` it is fully unrolled, so the traced graph
    grows linearly in sequence length: ~2.7k instructions per s-batch means the 128k shape
    traces ~22k and the 1M shape ~173k, and the latter takes a long time to compile (and dies
    outright when several such compiles run concurrently).

    `nl.fori_loop` emits the body ONCE, making the trace O(S_GROUP) instead of O(S). The cost is
    a scheduling barrier per iteration and no cross-iteration prefetching -- cheap here
    because the kernel is PE-bound, so Phase V and Phase B were competing for the
    tensor engine rather than overlapping. The loop also carries no state (attn_vt and out_acc
    are fully rewritten every iteration), which is what makes Pallas-style `fori_loop` legal.

    Both drivers share one body; the only difference is whether the s-group index reaching the
    two S-dependent DMAs is a python int or a traced scalar, which `scalar_offset` accepts
    either way.
    """
    s_chunk_rows = S_GROUP * oproj_s_tile
    # Leading-dim index of this core's first s-group, so scalar_offset only has to add the
    # iteration. s_start is a multiple of s_per_core == num_s_groups * s_chunk_rows.
    s_group_base = s_start // s_chunk_rows
    # A single s-group needs no loop at all, and `scalar_offset` rejects a plain python int
    # (it must be a traced NkiScalar), so that case keeps static addressing.
    _rolled = num_s_groups > 1
    attn_flat = out_attn_hbm.reshape((out_attn_hbm.shape[1], HL))
    attn_grouped = out_attn_hbm.reshape((out_attn_hbm.shape[1] // s_chunk_rows, s_chunk_rows, HL))
    out_grouped = out_hbm.reshape((out_hbm.shape[1] // s_chunk_rows, s_chunk_rows, HID))

    def _latent_src(sel, j, group_start, grp_l):
        """Latent rows for s-tile ``j`` of the selected s-group, this head group's columns."""
        within = j * oproj_s_tile * HL + group_start * L
        pattern = [[HL, oproj_s_tile], [1, grp_l]]
        if _rolled:
            return attn_grouped.ap(pattern=pattern, offset=within, scalar_offset=sel, indirect_dim=0)
        return attn_flat.ap(pattern=pattern, offset=s_start * HL + within)

    def _out_dst(sel, j):
        """This core's HID slice of the output rows for s-tile ``j`` of the selected s-group."""
        within = j * oproj_s_tile * HID + hid_start
        pattern = [[HID, oproj_s_tile], [1, hid_per_core]]
        if _rolled:
            return out_grouped.ap(pattern=pattern, offset=within, scalar_offset=sel, indirect_dim=0)
        return out_hbm.ap(pattern=pattern, offset=s_start * HID + within)

    def _s_group_body(s_group_idx):
        """One s-group: V-up all S_GROUP s-tiles, then o_proj them against a shared W_o stream.

        ``s_group_idx`` is the LOCAL iteration (python int when unrolled, traced scalar under
        fori_loop); it reaches HBM only through ``scalar_offset``.
        """
        grp_sel = None if s_group_idx is None else s_group_base + s_group_idx

        """
        ---- Phase V: V-up, emitting attn_v TRANSPOSED straight into attn_vt. ----
        Head group OUTER, s-tile INNER: W_uv for a head group is fetched once and reused by all
        S_GROUP s-tiles, so W_uv traffic falls by S_GROUP alongside W_o's.
        """
        for group_idx in range(num_vup_groups):
            group_start = group_idx * VUP_HEAD_GROUP
            group_count = min(VUP_HEAD_GROUP, H - group_start)
            grp_l = group_count * L
            grp_k_tiles = group_count * L_K_TILES

            sbm.open_scope()
            lat_sb = sbm.alloc_stack((_P_MAX, grp_l), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"vup_lat_g{group_idx}")
            lat_t = sbm.alloc_stack(
                (_P_MAX, grp_k_tiles, oproj_s_tile_pad),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name=f"vup_lat_t_g{group_idx}",
            )
            wuv_grp = sbm.alloc_stack(
                (_P_MAX, grp_k_tiles, d_v), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"vup_wuv_g{group_idx}"
            )

            _load_bf16_weights_k_slab(
                wuv_hbm,
                wuv_grp,
                out_dim=d_v,
                k_tile_start=group_start * L_K_TILES,
                k_tile_count=grp_k_tiles,
                k_tile_sizes=wuv_k_tile_sizes,
                full_out_dim=d_v,
            )

            for j in range(S_GROUP):
                # This group's latent columns [group_start*L, +grp_l) for the real s rows.
                # scalar_offset walks the s-group chunk (runtime under fori_loop); `offset`
                # carries the static within-chunk part (this s-tile and this head group).
                nisa.dma_copy(
                    dst=lat_sb[0:oproj_s_tile, 0:grp_l],
                    src=_latent_src(grp_sel, j, group_start, grp_l),
                    dge_mode=dge_mode.hwdge,
                )
                # Latent -> K-major: the V-up contracts over L, which is on the latent's FREE axis.
                _transpose_bf16_to_k_major(
                    lat_sb,
                    lat_t,
                    m_sz=oproj_s_tile,
                    m_pad=oproj_s_tile_pad,
                    k_tile_sizes=[_BF16_K_TILE] * grp_k_tiles,
                    src_row_stride=grp_l,
                )

                for local_h in range(group_count):
                    head_idx = group_start + local_h
                    for dv in range(DV_TILES):
                        vup_psum = nl.ndarray(
                            (_BF16_K_TILE, oproj_s_tile_pad),
                            dtype=nl.bfloat16,
                            buffer=nl.psum,
                            address=(0, (_NUM_TRANSPOSE_BANKS + dv) * PSUM_BANK_SIZE),
                        )
                        for kt in range(L_K_TILES):
                            k_local = local_h * L_K_TILES + kt
                            nisa.nc_matmul(
                                dst=vup_psum[0:_BF16_K_TILE, 0:oproj_s_tile_pad],
                                stationary=wuv_grp[0:_BF16_K_TILE, k_local, nl.ds(dv * _BF16_K_TILE, _BF16_K_TILE)],
                                moving=lat_t[0:_BF16_K_TILE, k_local, nl.ds(0, oproj_s_tile_pad)],
                                accumulate=(kt > 0),
                            )
                        # Head head_idx's dv-th 128-row slice of Hdv == K-major tile head_idx*DV_TILES+dv.
                        nisa.tensor_copy(
                            dst=attn_vt[j][0:_BF16_K_TILE, head_idx * DV_TILES + dv, 0:oproj_s_tile_pad],
                            src=vup_psum[0:_BF16_K_TILE, 0:oproj_s_tile_pad],
                            engine=nisa.engine.scalar if dv % 2 == 0 else nisa.engine.vector,
                        )
            sbm.close_scope()

        """
        ---- Phase B: o_proj, K-slab outer, SBUF-accumulated across slabs. ----
        attn_vt is already the stationary operand in the right layout, so there is no
        transpose or quantize between the two phases.
        """
        """
        One PSUM bank per (hid_tile, N-tile), cycling the ``n_psum_above`` free banks so the
        next tile's matmul can start while the previous tile's SBUF accumulate still reads its
        bank — the same de-serialization the MX stage gets from its ping-pong buffers.
        """
        bank_rr = 0
        # Double-buffered prologue: prime buffer 0 with slab 0, then each iteration issues the
        # NEXT slab's load into the alternate buffer before consuming the current one.
        if n_wo_buf > 1:
            _load_bf16_weights_k_slab(
                wo_hbm,
                wo_slab[0],
                out_dim=hid_per_core,
                k_tile_start=0,
                k_tile_count=slab_tiles,
                k_tile_sizes=hdv_k_tile_sizes,
                full_out_dim=HID,
                out_col_offset=hid_start,
            )
        for slab_id in range(num_k_slabs):
            wb = slab_id % n_wo_buf
            if n_wo_buf > 1:
                if slab_id + 1 < num_k_slabs:
                    _load_bf16_weights_k_slab(
                        wo_hbm,
                        wo_slab[(slab_id + 1) % n_wo_buf],
                        out_dim=hid_per_core,
                        k_tile_start=(slab_id + 1) * slab_tiles,
                        k_tile_count=slab_tiles,
                        k_tile_sizes=hdv_k_tile_sizes,
                        full_out_dim=HID,
                        out_col_offset=hid_start,
                    )
            else:
                _load_bf16_weights_k_slab(
                    wo_hbm,
                    wo_slab[0],
                    out_dim=hid_per_core,
                    k_tile_start=slab_id * slab_tiles,
                    k_tile_count=slab_tiles,
                    k_tile_sizes=hdv_k_tile_sizes,
                    full_out_dim=HID,
                    out_col_offset=hid_start,
                )
            for hid_idx in range(num_hid_tiles):
                hid_off = hid_idx * oproj_hid_tile
                for i_n in range(num_n_tiles):
                    n_sz = min(_MM1_TILE, oproj_hid_tile - i_n * _MM1_TILE)
                    n_base = hid_off + i_n * _MM1_TILE
                    # S_GROUP stationary operands against the SAME resident slab: this is what
                    # turns one W_o pass into S_GROUP s-tiles of output.
                    for j in range(S_GROUP):
                        ps = nl.ndarray(
                            (_P_MAX, _MM1_TILE),
                            dtype=nl.bfloat16,
                            buffer=nl.psum,
                            address=(0, (_NUM_TRANSPOSE_BANKS + bank_rr) * PSUM_BANK_SIZE),
                        )
                        for k_local in range(slab_tiles):
                            k_global = slab_id * slab_tiles + k_local
                            nisa.nc_matmul(
                                dst=ps[0:oproj_s_tile_pad, 0:n_sz],
                                stationary=attn_vt[j][0:_BF16_K_TILE, k_global, nl.ds(0, oproj_s_tile_pad)],
                                moving=wo_slab[wb][0:_BF16_K_TILE, k_local, nl.ds(n_base, n_sz)],
                                accumulate=(k_local > 0),
                            )
                        # First slab initializes out_acc (copy, no read-dep); later slabs add.
                        if slab_id == 0:
                            nisa.tensor_copy(
                                dst=out_acc[j][0:oproj_s_tile_pad, nl.ds(n_base, n_sz)],
                                src=ps[0:oproj_s_tile_pad, 0:n_sz],
                                engine=nisa.engine.vector,
                            )
                        else:
                            nisa.tensor_tensor(
                                dst=out_acc[j][0:oproj_s_tile_pad, nl.ds(n_base, n_sz)],
                                data1=out_acc[j][0:oproj_s_tile_pad, nl.ds(n_base, n_sz)],
                                data2=ps[0:oproj_s_tile_pad, 0:n_sz],
                                op=nl.add,
                                engine=nisa.engine.vector,
                            )
                        bank_rr = (bank_rr + 1) % n_psum_above

        for j in range(S_GROUP):
            nisa.tensor_copy(
                dst=out_sb[0:oproj_s_tile, :], src=out_acc[j][0:oproj_s_tile, :], engine=nisa.engine.scalar
            )
            # hwdge is mandatory inside fori_loop (dge_mode.none is rejected there).
            nisa.dma_copy(
                dst=_out_dst(grp_sel, j),
                src=out_sb[0:oproj_s_tile, :],
                dge_mode=dge_mode.hwdge,
            )

    if _rolled:
        nl.fori_loop(0, num_s_groups, _s_group_body, step=1)
    else:
        _s_group_body(None)  # static addressing; the selector is unused

    sbm.close_scope()


@nki.jit
def mla_vup_cte_kernel(
    out_attn_hbm: nl.NkiTensor,
    wuv_hbm: nl.NkiTensor,
    kv_lora_rank: int = _P_MAX * _H_PACK,
    output_tensor: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """BF16 V-up ONLY (Phase V of :func:`_vup_oproj_stage_bf16`), emitting the K-major activation.

    First half of the row-parallel o_proj split: each rank V-ups its OWN tokens over ALL heads
    and hands the small activation (not the o_proj weight) to the collective. The output is the
    exact stationary layout the o_proj matmul wants, dumped to HBM K-tile-major so an
    ``all_gather(collective_dim=0)`` concatenates ranks without any relayout:

        vup_kt_hbm[kt, p, s] == attn_v_transposed[p of K-tile kt, token s]
        (K-tile kt covers Hdv rows [kt * 128, (kt + 1) * 128))

    Args:
        out_attn_hbm (nl.NkiTensor): [1, S, H*L] bf16 latent attention output, natural column order.
        wuv_hbm (nl.NkiTensor): [H*L, d_v] bf16 V-up weight.
        kv_lora_rank (int): L, the latent dimension (H is recovered as ``H*L / L``).

    Returns:
        vup_kt_hbm (nl.NkiTensor): [H*d_v / 128, 128, S] bf16 K-major V-up activation.

    Notes:
        - Small-S kernel (a CP rank's own tokens, 64-512): asserts ``S <= 512`` and runs plain
          static s-batches of <= 128 tokens, no fori_loop / s-group machinery.
        - S-sharded over programs like :func:`mla_vupmx_oproj_cte_kernel`; each program writes its
          own token columns of the shared output (LNC1 callers see n_prgs == 1).
    """
    B, S, HL = out_attn_hbm.shape
    L = kv_lora_rank
    H = HL // L
    d_v = wuv_hbm.shape[1]
    Hdv = H * d_v

    kernel_assert(B == 1, f"mla_vup_cte_kernel: B must be 1, got {B}")
    kernel_assert(H * L == HL, f"mla_vup_cte_kernel: H*L ({HL}) must be a multiple of kv_lora_rank ({L})")
    kernel_assert(wuv_hbm.shape[0] == HL, f"mla_vup_cte_kernel: wuv rows {wuv_hbm.shape[0]} != H*L {HL}")
    kernel_assert(L % _BF16_K_TILE == 0, f"mla_vup_cte_kernel: L ({L}) must be a multiple of {_BF16_K_TILE}")
    kernel_assert(d_v % _BF16_K_TILE == 0, f"mla_vup_cte_kernel: d_v ({d_v}) must be a multiple of {_BF16_K_TILE}")
    kernel_assert(0 < H <= _P_MAX, f"mla_vup_cte_kernel: H must be in (0, {_P_MAX}], got {H}")

    _, n_prgs, prg_id = get_verified_program_sharding_info("mla_vup_cte_kernel", (0, 1))
    kernel_assert(S % n_prgs == 0, f"mla_vup_cte_kernel: S={S} must be divisible by n_prgs={n_prgs}")
    s_per_core = S // n_prgs
    kernel_assert(s_per_core <= 512, f"mla_vup_cte_kernel: small-S kernel, s_per_core {s_per_core} > 512")
    s_start = prg_id * s_per_core

    # s-batches of <= 128 (the latent transpose caps the token tile at the partition limit).
    s_tile = min(_OPROJ_S_TILE, s_per_core)
    kernel_assert(
        s_per_core % s_tile == 0, f"mla_vup_cte_kernel: s_per_core ({s_per_core}) must be a multiple of {s_tile}"
    )
    num_s_batches = s_per_core // s_tile
    _S_TILE_PAD_MULT = 16  # same stationary-free-dim padding rule as the fused stage
    s_tile_pad = div_ceil(s_tile, _S_TILE_PAD_MULT) * _S_TILE_PAD_MULT

    L_K_TILES = L // _BF16_K_TILE
    DV_TILES = d_v // _BF16_K_TILE
    HDV_K_TILES = Hdv // _BF16_K_TILE
    kernel_assert(
        DV_TILES <= _NUM_HW_PSUM_BANKS - _NUM_TRANSPOSE_BANKS,
        f"mla_vup_cte_kernel: d_v={d_v} needs {DV_TILES} V-up PSUM banks, "
        f"only {_NUM_HW_PSUM_BANKS - _NUM_TRANSPOSE_BANKS} free",
    )
    PSUM_BANK_SIZE = _get_psum_bank_size()

    # Caller-provided output (e.g. a private_hbm collective send buffer, so the activation goes
    # straight to the all-gather without a staging copy), else a fresh shared_hbm tensor.
    out_shape = (HDV_K_TILES, _P_MAX, S)
    if output_tensor is not None:
        kernel_assert(
            tuple(output_tensor.shape) == out_shape and output_tensor.dtype == nl.bfloat16,
            f"mla_vup_cte_kernel: output_tensor must be {list(out_shape)} bf16, "
            f"got {output_tensor.shape} {output_tensor.dtype}",
        )
        vup_kt_hbm = output_tensor
    else:
        vup_kt_hbm = nl.ndarray(out_shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)

    def _drain(attn_vt_sb, s_batch, row0):
        """One s-batch's K-major tile to HBM (real s columns only)."""
        dst = vup_kt_hbm.ap(pattern=[[S, _P_MAX], [_P_MAX * S, HDV_K_TILES], [1, s_tile]], offset=row0)
        nisa.dma_copy(dst=dst, src=attn_vt_sb[0:_P_MAX, 0:HDV_K_TILES, 0:s_tile])

    sbm = _new_sbm("mla_vup_cte")
    sbm.open_scope()

    # Head-group sizing: same per-head cost model as _vup_oproj_stage_bf16 (lat_sb + lat_t + wuv).
    _VUP_GROUP_BUDGET = 96 * 1024
    _per_head_bytes = L * 2 + L_K_TILES * s_tile_pad * 2 + L_K_TILES * d_v * 2
    _head_group_choices = [g for g in range(H, 0, -1) if H % g == 0 and g * _per_head_bytes <= _VUP_GROUP_BUDGET] or [1]
    VUP_HEAD_GROUP = _head_group_choices[0]
    num_vup_groups = div_ceil(H, VUP_HEAD_GROUP)

    attn_flat = out_attn_hbm.reshape((S, HL))
    wuv_k_tile_sizes = _bf16_k_tile_sizes(HL)

    # One s-batch of the K-major activation, drained to HBM per batch.
    attn_vt = sbm.alloc_stack((_P_MAX, HDV_K_TILES, s_tile_pad), dtype=nl.bfloat16, buffer=nl.sbuf, name="vup_attn_vt")

    for s_batch in range(num_s_batches):
        row0 = s_start + s_batch * s_tile
        for group_idx in range(num_vup_groups):
            group_start = group_idx * VUP_HEAD_GROUP
            group_count = min(VUP_HEAD_GROUP, H - group_start)
            grp_l = group_count * L
            grp_k_tiles = group_count * L_K_TILES

            # Names must be unique per s-batch (the tracer rejects a repeated op name); batch 0 keeps the
            # unsuffixed names, so an S <= 128 trace has the names of a one-batch trace.
            sfx = f"g{group_idx}" + (f"_s{s_batch}" if s_batch else "")
            sbm.open_scope()
            lat_sb = sbm.alloc_stack((_P_MAX, grp_l), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"vup_lat_{sfx}")
            lat_t = sbm.alloc_stack(
                (_P_MAX, grp_k_tiles, s_tile_pad), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"vup_lat_t_{sfx}"
            )
            wuv_grp = sbm.alloc_stack(
                (_P_MAX, grp_k_tiles, d_v), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"vup_wuv_{sfx}"
            )
            _load_bf16_weights_k_slab(
                wuv_hbm,
                wuv_grp,
                out_dim=d_v,
                k_tile_start=group_start * L_K_TILES,
                k_tile_count=grp_k_tiles,
                k_tile_sizes=wuv_k_tile_sizes,
                full_out_dim=d_v,
            )
            if _VUP_DMA_TRANSPOSE and s_tile % 16 == 0:
                # The latent lands K-major straight from HBM in ONE 4-D HW DGE
                # transpose per head group, instead of a load plus one PE transpose and PSUM eviction per
                # 128-column K tile. src[a, k, c, d] = attn[row0 + 16c + a, col0 + 128k + d], and the
                # (3, 1, 2, 0) permutation writes dst[d, k, c, a], whose (c, a) free order is the token.
                n_c = s_tile // 16
                nisa.dma_transpose(
                    dst=lat_t.ap(
                        pattern=[[grp_k_tiles * s_tile_pad, _BF16_K_TILE], [s_tile_pad, grp_k_tiles], [16, n_c], [1, 16]],
                        offset=0,
                    ),
                    src=attn_flat.ap(
                        pattern=[[HL, 16], [_BF16_K_TILE, grp_k_tiles], [16 * HL, n_c], [1, _BF16_K_TILE]],
                        offset=row0 * HL + group_start * L,
                    ),
                    axes=(3, 1, 2, 0),
                    dge_mode=dge_mode.hwdge,
                )
            else:
                # This group's latent columns [group_start*L, +grp_l) for the batch's real s rows.
                nisa.dma_copy(
                    dst=lat_sb[0:s_tile, 0:grp_l],
                    src=attn_flat.ap(pattern=[[HL, s_tile], [1, grp_l]], offset=row0 * HL + group_start * L),
                    dge_mode=dge_mode.hwdge,
                )
                # Latent -> K-major: the V-up contracts over L, which is on the latent's FREE axis.
                _transpose_bf16_to_k_major(
                    lat_sb,
                    lat_t,
                    m_sz=s_tile,
                    m_pad=s_tile_pad,
                    k_tile_sizes=[_BF16_K_TILE] * grp_k_tiles,
                    src_row_stride=grp_l,
                )
            for local_h in range(group_count):
                head_idx = group_start + local_h
                for dv in range(DV_TILES):
                    vup_psum = nl.ndarray(
                        (_BF16_K_TILE, s_tile_pad),
                        dtype=nl.bfloat16,
                        buffer=nl.psum,
                        address=(0, (_NUM_TRANSPOSE_BANKS + dv) * PSUM_BANK_SIZE),
                    )
                    for kt in range(L_K_TILES):
                        k_local = local_h * L_K_TILES + kt
                        nisa.nc_matmul(
                            dst=vup_psum[0:_BF16_K_TILE, 0:s_tile_pad],
                            stationary=wuv_grp[0:_BF16_K_TILE, k_local, nl.ds(dv * _BF16_K_TILE, _BF16_K_TILE)],
                            moving=lat_t[0:_BF16_K_TILE, k_local, nl.ds(0, s_tile_pad)],
                            accumulate=(kt > 0),
                        )
                    # Head head_idx's dv-th 128-row slice of Hdv == K-major tile head_idx*DV_TILES+dv.
                    nisa.tensor_copy(
                        dst=attn_vt[0:_BF16_K_TILE, head_idx * DV_TILES + dv, 0:s_tile_pad],
                        src=vup_psum[0:_BF16_K_TILE, 0:s_tile_pad],
                        engine=nisa.engine.scalar if dv % 2 == 0 else nisa.engine.vector,
                    )
            sbm.close_scope()

        # Drain this s-batch straight to HBM in the K-major [kt, p, s] layout (real s columns only).
        _drain(attn_vt, s_batch, row0)

    sbm.close_scope()
    return vup_kt_hbm


# mla_vup_cte_kernel: load the latent K-major by DMA transpose (see its use).
_VUP_DMA_TRANSPOSE = True


@nki.jit
def mla_oproj_partial_cte_kernel(
    vup_all_kt_hbm: nl.NkiTensor,
    wo_shard_hbm: nl.NkiTensor,
    wo_shard_rank: nl.NkiTensor,
    n_ranks: int,
    hdv_total: int,
    output_tensor: nl.NkiTensor = None,
    input_pre_sliced: bool = False,
) -> nl.NkiTensor:
    """Row-parallel o_proj partial: all-gathered V-up activations x this rank's W_o row shard.

    Second half of the row-parallel o_proj split. The all-gather of :func:`mla_vup_cte_kernel`'s
    output concatenates ranks on dim 0, so ``vup_all_kt_hbm`` is ``[w, Hdv/128, 128, S]`` viewed
    flat as ``[w * Hdv/128, 128, S]``. Rank ``i`` of the wo group holds W_o rows
    ``[i * Hdv/w, (i + 1) * Hdv/w)`` as ``wo_shard_hbm`` and therefore needs, from EVERY source
    rank j, only the K-tiles covering that same Hdv row range:

        partial[j * S + t, :] = vup_j[t, i * Hdv/w : (i + 1) * Hdv/w] @ wo_shard_i

    A ``reduce_scatter(collective_dim=0, op=add)`` of the partials over the wo group then hands
    each rank its own tokens' full o_proj output.

    Args:
        vup_all_kt_hbm (nl.NkiTensor): [w * Hdv/128, 128, S] bf16 gathered K-major activations,
            rank-major on dim 0.
        wo_shard_hbm (nl.NkiTensor): [Hdv/w, HID] bf16 W_o row shard, natural column order.
        wo_shard_rank (nl.NkiTensor): [1, 1] int32 this rank's wo-shard index (runtime value; the
            K-tile selection is a runtime ``scalar_offset = wo_shard_rank * Hdv/w/128`` on the
            gathered tensor's K-tile axis, so the shard index never has to be a trace-time int).
        n_ranks (int): w, the wo group size.
        hdv_total (int): the FULL o_proj contraction depth H * d_v (the shard covers hdv_total / w).
        output_tensor (nl.NkiTensor): optional caller-provided [w*S, HID] bf16 HBM output (any HBM
            kind, e.g. a private_hbm reduce-scatter source); allocated shared_hbm when None.
        input_pre_sliced (bool): the input came from an ALL-TO-ALL instead of an all-gather:
            ``[w * Hdv/(128*w), 128, S]``, where block j is source rank j's K-tiles for THIS
            rank's Hdv row range already. The per-source K-tile base is then the static
            ``j * Hdv/(128*w)`` and ``wo_shard_rank`` is unused (pass any [1, 1] int32).

    Returns:
        partial_hbm (nl.NkiTensor): [w*S, HID] bf16 partial sums, source-rank-major rows.

    Notes:
        - ``S > 128`` must be a multiple of 128 and runs as 128-token tiles, each the ``S == 128``
          instruction sequence on its token columns (W_o is streamed once per tile).
        - Each tile's tokens of a source rank ride one stationary tile. The gathered
          activations are tiny (Hdv * S * 2 / w bytes per rank), so ALL ranks' needed K-tiles are
          loaded up front and stay resident; ``n_ranks`` is folded into the stationary free axis
          (``rpm`` ranks per matmul) so the PE output fills the full 128 partitions.
        - W_o slab streaming / double buffering mirrors Phase B of :func:`_vup_oproj_stage_bf16`;
          programs shard HID (LNC1 callers see n_prgs == 1).
    """
    return _oproj_partial_bf16(
        vup_all_kt_hbm, wo_shard_hbm, wo_shard_rank, n_ranks, hdv_total, output_tensor=output_tensor,
        input_pre_sliced=input_pre_sliced,
    )


def _oproj_partial_bf16(
    vup_all_kt_hbm: nl.NkiTensor,
    wo_shard_hbm: nl.NkiTensor,
    wo_shard_rank: nl.NkiTensor,
    n_ranks: int,
    hdv_total: int,
    output_tensor: nl.NkiTensor = None,
    input_pre_sliced: bool = False,
) -> nl.NkiTensor:
    """The body of :func:`mla_oproj_partial_cte_kernel`."""
    w = n_ranks
    HID = wo_shard_hbm.shape[1]
    S_all = vup_all_kt_hbm.shape[-1]
    S = min(S_all, _P_MAX)  # tokens per tile
    kernel_assert(
        hdv_total % (w * _BF16_K_TILE) == 0,
        f"mla_oproj_partial_cte_kernel: hdv_total ({hdv_total}) must be a multiple of w*128 ({w * _BF16_K_TILE})",
    )
    KT_TOTAL = hdv_total // _BF16_K_TILE
    hdv_shard = hdv_total // w
    KT_SHARD = hdv_shard // _BF16_K_TILE
    kernel_assert(
        wo_shard_hbm.shape[0] == hdv_shard,
        f"mla_oproj_partial_cte_kernel: wo_shard rows {wo_shard_hbm.shape[0]} != Hdv/w {hdv_shard}",
    )
    expected_in = (hdv_total if input_pre_sliced else w * hdv_total) * S_all
    kernel_assert(
        vup_all_kt_hbm.size == expected_in,
        f"mla_oproj_partial_cte_kernel: gathered activation size {vup_all_kt_hbm.size} != {expected_in}",
    )
    kernel_assert(
        0 < S_all and S_all % S == 0,
        f"mla_oproj_partial_cte_kernel: S must be in (0, {_P_MAX}] or a multiple of {_P_MAX}, got {S_all}",
    )
    n_s_tiles = S_all // S

    _, n_prgs, prg_id = get_verified_program_sharding_info("mla_oproj_partial_cte_kernel", (0, 1))
    kernel_assert(HID % n_prgs == 0, f"mla_oproj_partial_cte_kernel: HID={HID} must be divisible by n_prgs={n_prgs}")
    hid_per_core = HID // n_prgs
    hid_start = prg_id * hid_per_core

    _S_TILE_PAD_MULT = 16
    st_pad = div_ceil(S, _S_TILE_PAD_MULT) * _S_TILE_PAD_MULT
    # Ranks per matmul: pack as many source ranks as fit the 128-wide stationary free axis so the
    # PE output uses all 128 partitions instead of S of them.
    rpm = 1
    for cand in range(w, 0, -1):
        if w % cand == 0 and cand * st_pad <= _P_MAX:
            rpm = cand
            break
    n_grp = w // rpm
    grp_rows = rpm * st_pad

    if output_tensor is not None:
        kernel_assert(
            output_tensor.shape[0] == w * S_all and output_tensor.shape[-1] == HID,
            f"mla_oproj_partial_cte_kernel: output_tensor {output_tensor.shape} != ({w * S_all}, {HID})",
        )
        partial_hbm = output_tensor
    else:
        partial_hbm = nl.ndarray((w * S_all, HID), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    PSUM_BANK_SIZE = _get_psum_bank_size()
    sbm = _new_sbm("mla_oproj_partial")
    sbm.open_scope()

    # ---- W_o K-slab depth over the SHARD's contraction (same budget model as the fused stage) ----
    _CALIBRATED_SBUF_BUDGET_BYTES = 240 * 1024
    _sbuf_budget = min(int(nl.tile_size.total_available_sbuf_size), _CALIBRATED_SBUF_BUDGET_BYTES)
    _reserve = (
        n_grp * KT_SHARD * grp_rows * 2  # resident gathered activations
        + n_grp * hid_per_core * 2  # per-group accumulators
        + 32 * 1024  # allocator slack
    )
    # The resident set grows with the group count (one accumulator per rank group); very large wo
    # groups (where the partial itself would dwarf the saved weight traffic) are rejected early.
    kernel_assert(
        _reserve + hid_per_core * 2 <= _sbuf_budget,
        f"mla_oproj_partial_cte_kernel: resident activations + accumulators ({_reserve} B) + one W_o "
        f"K-tile exceed the SBUF budget ({_sbuf_budget} B); n_ranks={w} is too large for this kernel",
    )
    slab_single = _bf16_compute_k_slab_tiles(hdv_shard, hid_per_core, _sbuf_budget, _reserve, 1)
    if slab_single >= KT_SHARD:
        n_wo_buf, slab_tiles = 1, KT_SHARD  # full shard residency: nothing to prefetch
    else:
        n_wo_buf = 2
        slab_tiles = _bf16_compute_k_slab_tiles(hdv_shard, hid_per_core, _sbuf_budget, _reserve, 2)
    num_k_slabs = KT_SHARD // slab_tiles
    wo_k_tile_sizes = _bf16_k_tile_sizes(hdv_shard)

    vup_sb = sbm.alloc_stack((_P_MAX, n_grp, KT_SHARD, grp_rows), dtype=nl.bfloat16, buffer=nl.sbuf, name="oprp_vup")
    out_acc = [
        sbm.alloc_stack((grp_rows, hid_per_core), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"oprp_out_acc_{g}")
        for g in range(n_grp)
    ]
    wo_slab = [
        sbm.alloc_stack(
            (_P_MAX, slab_tiles, hid_per_core), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"oprp_wo_slab_{buf_idx}"
        )
        for buf_idx in range(n_wo_buf)
    ]

    tile_base = None
    if not input_pre_sliced:
        # Runtime K-tile base of this rank's Hdv row range: wo_shard_rank * KT_SHARD. scalar_offset
        # takes a [1, 1] SBUF scalar and indexes dim 0 of the (reshaped) source at runtime.
        rank_sb = sbm.alloc_stack((1, 1), dtype=nl.int32, buffer=nl.sbuf, name="oprp_rank")
        nisa.dma_copy(dst=rank_sb, src=wo_shard_rank.reshape((1, 1)))
        tile_base = sbm.alloc_stack((1, 1), dtype=nl.int32, buffer=nl.sbuf, name="oprp_tile_base")
        nisa.tensor_scalar(dst=tile_base, data=rank_sb, op0=nl.multiply, operand0=KT_SHARD)

    rows_per_src = KT_SHARD if input_pre_sliced else KT_TOTAL
    gathered = vup_all_kt_hbm.reshape((w * rows_per_src, _P_MAX * S_all))
    num_n_tiles = div_ceil(hid_per_core, _MM1_TILE)
    # One pass per S tile (S <= 128: one pass); the buffers are reused, each pass as the S <= 128 kernel.
    for s_tile in range(n_s_tiles):
        s0 = s_tile * S
        # ---- resident activation load: every source rank's needed K-tiles, hoisted above the slab
        # stream. Pad token columns are zeroed so no garbage reaches the (discarded) pad PE rows. ----
        if st_pad != S:
            nisa.memset(dst=vup_sb[0:_P_MAX, 0:n_grp, 0:KT_SHARD, 0:grp_rows], value=0.0)
        for j in range(w):
            g, i = j // rpm, j % rpm
            pattern = [[S_all, _P_MAX], [_P_MAX * S_all, KT_SHARD], [1, S]]
            if input_pre_sliced:
                src_ap = gathered.ap(pattern=pattern, offset=j * KT_SHARD * _P_MAX * S_all + s0)
            else:
                src_ap = gathered.ap(
                    pattern=pattern,
                    offset=j * KT_TOTAL * _P_MAX * S_all + s0,
                    scalar_offset=tile_base,
                    indirect_dim=0,
                )
            nisa.dma_copy(dst=vup_sb[0:_P_MAX, g, 0:KT_SHARD, nl.ds(i * st_pad, S)], src=src_ap)

        # ---- W_o slab stream OUTER, source-rank groups INNER, SBUF-accumulated across slabs (the
        # exact Phase-B pattern of _vup_oproj_stage_bf16, with the resident activation as stationary).
        bank_rr = 0
        if n_wo_buf > 1:
            _load_bf16_weights_k_slab(
                wo_shard_hbm,
                wo_slab[0],
                out_dim=hid_per_core,
                k_tile_start=0,
                k_tile_count=slab_tiles,
                k_tile_sizes=wo_k_tile_sizes,
                full_out_dim=HID,
                out_col_offset=hid_start,
            )
        for slab_id in range(num_k_slabs):
            wb = slab_id % n_wo_buf
            if n_wo_buf > 1:
                if slab_id + 1 < num_k_slabs:
                    _load_bf16_weights_k_slab(
                        wo_shard_hbm,
                        wo_slab[(slab_id + 1) % n_wo_buf],
                        out_dim=hid_per_core,
                        k_tile_start=(slab_id + 1) * slab_tiles,
                        k_tile_count=slab_tiles,
                        k_tile_sizes=wo_k_tile_sizes,
                        full_out_dim=HID,
                        out_col_offset=hid_start,
                    )
            else:
                _load_bf16_weights_k_slab(
                    wo_shard_hbm,
                    wo_slab[0],
                    out_dim=hid_per_core,
                    k_tile_start=slab_id * slab_tiles,
                    k_tile_count=slab_tiles,
                    k_tile_sizes=wo_k_tile_sizes,
                    full_out_dim=HID,
                    out_col_offset=hid_start,
                )
            for i_n in range(num_n_tiles):
                n_sz = min(_MM1_TILE, hid_per_core - i_n * _MM1_TILE)
                n_base = i_n * _MM1_TILE
                for g in range(n_grp):
                    ps = nl.ndarray(
                        (_P_MAX, _MM1_TILE),
                        dtype=nl.bfloat16,
                        buffer=nl.psum,
                        address=(0, bank_rr * PSUM_BANK_SIZE),
                    )
                    for k_local in range(slab_tiles):
                        k_global = slab_id * slab_tiles + k_local
                        nisa.nc_matmul(
                            dst=ps[0:grp_rows, 0:n_sz],
                            stationary=vup_sb[0:_P_MAX, g, k_global, nl.ds(0, grp_rows)],
                            moving=wo_slab[wb][0:_P_MAX, k_local, nl.ds(n_base, n_sz)],
                            accumulate=(k_local > 0),
                        )
                    # First slab initializes out_acc (copy, no read-dep); later slabs add.
                    if slab_id == 0:
                        nisa.tensor_copy(
                            dst=out_acc[g][0:grp_rows, nl.ds(n_base, n_sz)],
                            src=ps[0:grp_rows, 0:n_sz],
                            engine=nisa.engine.vector,
                        )
                    else:
                        nisa.tensor_tensor(
                            dst=out_acc[g][0:grp_rows, nl.ds(n_base, n_sz)],
                            data1=out_acc[g][0:grp_rows, nl.ds(n_base, n_sz)],
                            data2=ps[0:grp_rows, 0:n_sz],
                            op=nl.add,
                            engine=nisa.engine.vector,
                        )
                    bank_rr = (bank_rr + 1) % _NUM_HW_PSUM_BANKS

        # ---- drain: source rank j's tile rows land at [j*S_all + s0, +S) of the output ----
        for g in range(n_grp):
            for i in range(rpm):
                j = g * rpm + i
                nisa.dma_copy(
                    dst=partial_hbm.ap(pattern=[[HID, S], [1, hid_per_core]], offset=(j * S_all + s0) * HID + hid_start),
                    src=out_acc[g][nl.ds(i * st_pad, S), 0:hid_per_core],
                )

    sbm.close_scope()
    return partial_hbm


def _vupmx_oproj_stage(
    out_attn_hbm,
    wuv_qtz_hbm,
    wuv_scale_hbm,
    wo_qtz_hbm,
    wo_scale_hbm,
    out_hbm,
    H,
    L,
    sbm,
    s_start,
    s_per_core,
    hid_start,
    hid_per_core,
    s_tile_cap=_OPROJ_S_TILE,
    compact_scales: bool = True,
):
    """MX V-up + MX o_proj.

    Reads attn_latent from out_attn_hbm[B, S, H*L] for queries [s_start, s_start+
    s_per_core), V-ups to attn_v[s_tile, H*d_v], projects with W_o, and writes
    out_hbm[B, S, HID] for the HID slice [hid_start, hid_start+hid_per_core). Single
    source of truth for the standalone o_proj kernel (and reusable by a fused parent).

    Sharding: S is sharded via (s_start, s_per_core); the o_proj OUTPUT (HID) is
    sharded via (hid_start, hid_per_core) — kernel A is S-sharded and kernel B may
    instead shard HID (each core does all S, its HID slice), which only changes these
    two pairs. The full H*d_v contraction is always local.
    """
    HID = wo_qtz_hbm.shape[1]
    d_v = wuv_qtz_hbm.shape[1]
    HL = H * L
    Hdv = H * d_v

    kernel_assert(
        hid_per_core % _MM1_TILE == 0,
        f"_vupmx_oproj_stage: hid_per_core ({hid_per_core}) must be a multiple of {_MM1_TILE}",
    )
    # d_v == P_MAX and H % H_PACK == 0 so each swizzled 512-tile spans exactly H_PACK heads
    # (the 4-pack sub axis crosses head boundaries; see the Phase-V1 write AP comment).
    kernel_assert(d_v == _P_MAX, f"_vupmx_oproj_stage: pre-swizzled transpose requires d_v == {_P_MAX}, got {d_v}")
    # H = per-rank head count. MX o_proj tiles H*d_v in 512-wide blocks of H_PACK=4 heads, so
    # H must be a multiple of H_PACK (>= 4); fewer heads/rank (tp=64 -> 2) is unsupported.
    kernel_assert(0 < H <= _P_MAX, f"_vupmx_oproj_stage: H (heads/rank) must be in (0, {_P_MAX}], got {H}")
    kernel_assert(
        H % _H_PACK == 0,
        f"_vupmx_oproj_stage: H (heads/rank) must be a multiple of {_H_PACK} (>= {_H_PACK} heads); "
        f"MX o_proj tiles {_H_PACK} heads per 512-block. got H={H}",
    )
    oproj_s_tile = min(s_tile_cap, s_per_core)
    kernel_assert(
        s_per_core % oproj_s_tile == 0, f"_vupmx_oproj_stage: s_per_core must be a multiple of {oproj_s_tile}"
    )
    num_s_batches = s_per_core // oproj_s_tile
    """
    Two s-tile widths, mirroring the qkv stage's ``s_tile_sz`` / ``s_tile_pad`` split:

      * ``oproj_s_tile``     - the REAL rows this s-batch owns. Used for every HBM access
                               (latent read, output write) and for stepping s-batches, so a
                               shard never reads/writes outside its own slice.
      * ``oproj_s_tile_pad`` - the PADDED compute width. SBUF/PSUM buffers, transposes,
                               quantize and both matmuls run on this; the extra columns are
                               don't-care.

    Padding is needed because the V-up nc_matmul_mx puts the s-tile on its stationary FREE
    axis: a free dim of 1 (s_per_core == 1, e.g. S=2 over 2 cores) fails MLIR verification.
    Rounding UP to a multiple of _S_TILE_PAD_MULT also keeps the XBar dma_transpose output
    32B-aligned, so the tiny-tile latent load stays on the fast DMA-transpose path.
    """
    _S_TILE_PAD_MULT = 16  # 32B / bf16: XBar transpose output alignment
    oproj_s_tile_pad = div_ceil(oproj_s_tile, _S_TILE_PAD_MULT) * _S_TILE_PAD_MULT

    # HID PSUM-tile: only one oproj_hid_tile slice's PSUM is live at a time (Phase B is
    # K-slab-outer), so shrink it until num_n_tiles + the H_PACK transpose banks fit 8.
    oproj_hid_tile = _OPROJ_HID_TILE
    while hid_per_core % oproj_hid_tile != 0 or _H_PACK + (oproj_hid_tile // _MM1_TILE) > _NUM_HW_PSUM_BANKS:
        oproj_hid_tile -= _MM1_TILE
    kernel_assert(
        oproj_hid_tile >= _MM1_TILE, f"_vupmx_oproj_stage: cannot fit PSUM banks for hid_per_core={hid_per_core}"
    )

    num_mx_k_tiles = Hdv // (_P_MAX * _H_PACK)
    """
    KNOWN LIMIT: multi-s-batch (s_per_core > oproj_s_tile) is only correct when
    num_mx_k_tiles == 1 (H=4, Hdv=512, the validated head-sharded long-seq path). With
    num_mx_k_tiles > 1 the per-s_batch loop mis-handles the activation -> NaN. Fail early.
    """
    kernel_assert(
        num_s_batches == 1 or num_mx_k_tiles == 1,
        f"_vupmx_oproj_stage: multi-s-batch (s_per_core={s_per_core} > oproj_s_tile={oproj_s_tile}, "
        f"num_s_batches={num_s_batches}) is only supported when num_mx_k_tiles==1 (H={_H_PACK}, "
        f"Hdv=512); got H={H} -> num_mx_k_tiles={num_mx_k_tiles} (KNOWN-BROKEN regime).",
    )
    num_hid_tiles = hid_per_core // oproj_hid_tile
    num_n_tiles = oproj_hid_tile // _MM1_TILE  # PSUM banks per HID-slice.
    kernel_assert(
        _H_PACK + num_n_tiles <= _NUM_HW_PSUM_BANKS,
        f"stage-2 PSUM banks {_H_PACK + num_n_tiles} exceed {_NUM_HW_PSUM_BANKS}",
    )
    PSUM_BANK_SIZE = _get_psum_bank_size()

    sbm.open_scope()

    stage_512 = min(_OPROJ_STAGE_512, num_mx_k_tiles)
    kernel_assert(
        num_mx_k_tiles % stage_512 == 0,
        f"num_mx_k_tiles ({num_mx_k_tiles}) must be a multiple of stage tiles ({stage_512})",
    )
    num_stage_chunks = num_mx_k_tiles // stage_512

    """
    K-slab depth for the contiguous full-width weight load: pick the largest slab dividing
    num_mx_k_tiles whose double-buffered [P_MAX, slab_512, hid_per_core] fp8x4+scale buffer
    fits the budget (both buffers + out_acc + in_qtz must coexist).
    """
    _SLAB_BUDGET_BYTES = 70 * 1024  # per single buffer (fp8x4 4B + uint8 1B per elem)
    slab_512 = min(_OPROJ_K_SLAB_512, num_mx_k_tiles)
    while slab_512 > 1 and (num_mx_k_tiles % slab_512 != 0 or slab_512 * hid_per_core * (4 + 1) > _SLAB_BUDGET_BYTES):
        slab_512 -= 1
    kernel_assert(
        num_mx_k_tiles % slab_512 == 0,
        f"num_mx_k_tiles ({num_mx_k_tiles}) must be a multiple of K-slab tiles ({slab_512})",
    )
    num_k_slabs = num_mx_k_tiles // slab_512

    # V-up MX. L=512 => exactly H MX 512-tiles in the latent: head h == k-tile h.
    num_lat_k_tiles = HL // (_P_MAX * _H_PACK)  # == H
    lat_stage_512 = min(_OPROJ_STAGE_512, num_lat_k_tiles)
    kernel_assert(
        num_lat_k_tiles % lat_stage_512 == 0,
        f"num_lat_k_tiles ({num_lat_k_tiles}) must be a multiple of {lat_stage_512}",
    )
    num_lat_stage_chunks = num_lat_k_tiles // lat_stage_512

    # o_proj activation (quantized H*d_v contraction), resident for ALL s_per_core across
    # BOTH phases (small, ~10KB/part). Lives in the outer scope so it survives Phase B.
    # Covers the PADDED write of the last s-batch: it writes oproj_s_tile_pad columns at
    # offset (num_s_batches-1)*oproj_s_tile, which can exceed s_per_core when padding is on.
    s_qtz_width = (num_s_batches - 1) * oproj_s_tile + oproj_s_tile_pad
    in_qtz = sbm.alloc_stack(
        (_P_MAX, num_mx_k_tiles, s_qtz_width), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf, name="oproj_in_qtz"
    )
    in_scale = sbm.alloc_stack(
        (_P_MAX, num_mx_k_tiles, s_qtz_width), dtype=nl.uint8, buffer=nl.sbuf, name="oproj_in_scale"
    )

    # V-up scratch in its OWN scope so attn_v/lat_*/wuv_* free before Phase B allocates the
    # weight slab + accumulator.
    sbm.open_scope()
    attn_v = sbm.alloc_stack((oproj_s_tile_pad, Hdv), dtype=nl.bfloat16, buffer=nl.sbuf, name="attn_v")
    # Double-buffer the V0 latent-load (load N+1 overlaps N's V-up + Phase A). Only pays off
    # with >1 s-batch (head-sharded long-seq, where H is small so these buffers are tiny).
    n_lat_buf = 2 if num_s_batches > 1 else 1
    lat_qtz, lat_scale, lat_in_t, lat_planes = [], [], [], []
    for buf_idx in range(n_lat_buf):
        lat_qtz.append(
            sbm.alloc_stack(
                (_P_MAX, num_lat_k_tiles, oproj_s_tile_pad),
                dtype=nl.float8_e4m3fn_x4,
                buffer=nl.sbuf,
                name=f"vup_lat_qtz_{buf_idx}",
            )
        )
        lat_scale.append(
            sbm.alloc_stack(
                (_P_MAX, num_lat_k_tiles, oproj_s_tile_pad),
                dtype=nl.uint8,
                buffer=nl.sbuf,
                name=f"vup_lat_scale_{buf_idx}",
            )
        )
        lat_in_t.append(
            sbm.alloc_stack(
                (_P_MAX, lat_stage_512, oproj_s_tile_pad * _H_PACK),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name=f"vup_lat_t_{buf_idx}",
            )
        )
        # Contiguous sub-outer planes [P_MAX, lat_stage_512, H_PACK, s_tile] written by the
        # swdge dma_transpose, then vector-permuted into lat_in_t (sub innermost) for quantize.
        lat_planes.append(
            sbm.alloc_stack(
                (_P_MAX, lat_stage_512, _H_PACK, oproj_s_tile_pad),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name=f"vup_lat_planes_{buf_idx}",
            )
        )
    in_t = sbm.alloc_stack(
        (_P_MAX, stage_512, oproj_s_tile_pad * _H_PACK), dtype=nl.bfloat16, buffer=nl.sbuf, name="oproj_t"
    )
    wuv_qtz, wuv_scale = _load_mx_weights(
        wuv_qtz_hbm, wuv_scale_hbm, HL, d_v, sbm, name="vup_wuv", compact_scales=compact_scales
    )

    # Transpose scratch: Phase-A transpose uses all 8 PSUM banks. Phase-V1 reuses bank H_PACK
    # for vup_psum, but V1 fully precedes Phase A per s-batch so they don't overlap in time.
    tpsum = []
    for bank_idx in range(_NUM_HW_PSUM_BANKS):
        tpsum.append(
            nl.ndarray((_P_MAX, _P_MAX), dtype=nl.bfloat16, buffer=nl.psum, address=(0, bank_idx * PSUM_BANK_SIZE))
        )

    """
    Phase V (per s-batch): build the quantized o_proj activation for all S into in_qtz.
    V0 loads the latent via _dma_transpose_latent_for_mx — the latent->partition transpose
    runs on the DMA (swdge) engine straight from HBM (no nc_transpose), enabled by the
    cross-kernel 4-pack column order the attention stage writes (each (k512, sub)'s groups
    contiguous). Prologue loads s-batch 0; each iteration prefetches the next s-batch first.
    """
    attn_row_base = s_start * HL
    for chunk in range(num_lat_stage_chunks):
        k512_start = chunk * lat_stage_512
        _dma_transpose_latent_for_mx(
            out_attn_hbm,
            HL,
            oproj_s_tile_pad,
            lat_stage_512,
            lat_planes[0],
            lat_in_t[0],
            attn_row_base + k512_start * _MM1_TILE,
            sbm_scratch=sbm,
            tpsum_scratch=tpsum,
            s_valid=oproj_s_tile,
        )
        nisa.quantize_mx(
            src=lat_in_t[0][0:_P_MAX, 0:lat_stage_512, 0 : oproj_s_tile_pad * _H_PACK],
            dst=lat_qtz[0][0:_P_MAX, nl.ds(k512_start, lat_stage_512), 0:oproj_s_tile_pad],
            dst_scale=lat_scale[0][0:_P_MAX, nl.ds(k512_start, lat_stage_512), 0:oproj_s_tile_pad],
        )
    for s_batch in range(num_s_batches):
        buf = s_batch % n_lat_buf
        s_off = s_batch * oproj_s_tile
        # Prefetch next s-batch's latent into the alternate buffer (overlaps this V-up + A).
        if n_lat_buf > 1 and s_batch + 1 < num_s_batches:
            ld_buf = (s_batch + 1) % n_lat_buf
            ld_row_base = (s_start + (s_batch + 1) * oproj_s_tile) * HL
            for chunk in range(num_lat_stage_chunks):
                k512_start = chunk * lat_stage_512
                _dma_transpose_latent_for_mx(
                    out_attn_hbm,
                    HL,
                    oproj_s_tile_pad,
                    lat_stage_512,
                    lat_planes[ld_buf],
                    lat_in_t[ld_buf],
                    ld_row_base + k512_start * _MM1_TILE,
                    sbm_scratch=sbm,
                    tpsum_scratch=tpsum,
                    s_valid=oproj_s_tile,
                )
                nisa.quantize_mx(
                    src=lat_in_t[ld_buf][0:_P_MAX, 0:lat_stage_512, 0 : oproj_s_tile_pad * _H_PACK],
                    dst=lat_qtz[ld_buf][0:_P_MAX, nl.ds(k512_start, lat_stage_512), 0:oproj_s_tile_pad],
                    dst_scale=lat_scale[ld_buf][0:_P_MAX, nl.ds(k512_start, lat_stage_512), 0:oproj_s_tile_pad],
                )
        lat_qtz_b, lat_scale_b = lat_qtz[buf], lat_scale[buf]

        """
        Phase V1: per head, V-up MX-matmul -> vup_psum[s, d_v], evicted into attn_v at
        PRE-SWIZZLED Hdv column positions so Phase A can read contiguous [s,128] slices.
        The Hdv swizzle (the MLA column permutation) crosses head boundaries (a 512-tile spans 4
        heads), so it can't be applied per-head; instead the per-head evict uses a strided
        write AP that lands head h's output at the swizzled positions a 512-tile expects:
        base = (h//H_PACK)*P_MAX + (h%H_PACK)*(P_MAX//H_PACK); free pattern
        [[1, P_MAX//H_PACK], [num_mx_k_tiles*P_MAX, H_PACK]]. Result: assembled attn_v ==
        the MLA column permutation of the natural order, so in_qtz is bit-identical to the natural
        path (ref unchanged).
        """
        sub_stride = num_mx_k_tiles * _P_MAX
        for h in range(H):
            vup_psum = nl.ndarray(
                (oproj_s_tile_pad, d_v), dtype=nl.bfloat16, buffer=nl.psum, address=(0, _H_PACK * PSUM_BANK_SIZE)
            )
            nisa.nc_matmul_mx(
                dst=vup_psum[0:oproj_s_tile_pad, 0:d_v],
                stationary=lat_qtz_b[0:_P_MAX, h, nl.ds(0, oproj_s_tile_pad)],
                moving=wuv_qtz[0:_P_MAX, h, nl.ds(0, d_v)],
                stationary_scale=lat_scale_b[0:_P_MAX, h, nl.ds(0, oproj_s_tile_pad)],
                moving_scale=wuv_scale[0:_P_MAX, h, nl.ds(0, d_v)],
            )
            base = (h // _H_PACK) * _P_MAX + (h % _H_PACK) * (_P_MAX // _H_PACK)
            nisa.tensor_scalar(
                attn_v.ap(
                    pattern=[[Hdv, oproj_s_tile_pad], [1, _P_MAX // _H_PACK], [sub_stride, _H_PACK]],
                    offset=base,
                ),
                vup_psum,
                op0=nl.multiply,
                operand0=1.0,
                engine=nisa.engine.scalar,
            )

        """
        Phase A: pre-swizzled-transpose + quantize attn_v -> in_qtz[:, :, s_off:] (this
        s-batch's slice of the resident, all-S quantized activation). attn_v is now in
        swizzled Hdv column order, so the contiguous-read 8-bank transpose applies.
        """
        for chunk in range(num_stage_chunks):
            k512_start = chunk * stage_512
            _transpose_preswizzled_for_mx(
                attn_v, oproj_s_tile_pad, num_mx_k_tiles, stage_512, in_t, tpsum, base_col_512=k512_start
            )
            nisa.quantize_mx(
                src=in_t[0:_P_MAX, 0:stage_512, 0 : oproj_s_tile_pad * _H_PACK],
                dst=in_qtz[0:_P_MAX, nl.ds(k512_start, stage_512), nl.ds(s_off, oproj_s_tile_pad)],
                dst_scale=in_scale[0:_P_MAX, nl.ds(k512_start, stage_512), nl.ds(s_off, oproj_s_tile_pad)],
            )

    sbm.close_scope()  # free V-up scratch (attn_v, lat_*, wuv_*, in_t) before Phase B.

    """
    ---- Phase B (K-slab outer, SBUF-accumulated): per s-batch, accumulate the H*d_v
    contraction across K-slabs into out_acc. Each slab loads the full hid_per_core weight
    width contiguously (high MBU), one matmul per HID PSUM-slice, one PSUM slice live at a time.
    """
    sbm.open_scope()
    """
    Weight residency (budget-aware). TIER 1: if the full W_o fits free SBUF, load it ONCE
    and reuse across s-batches — kills the per-s_batch reload that made o_proj DMA-bound at
    large s_per_core (streamed W_o num_s_batches times). TIER 3 (W_o too big, e.g. 128-head
    Hdv=16384 ~1.1MB/part): per-s_batch double-buffered K-slab streaming (slab N+1 DMA
    overlaps slab N compute). fp8x4 = 4B/elem + uint8 scale = 1B/elem, per partition.
    """
    wo_full_bytes = num_mx_k_tiles * hid_per_core * (4 + 1)
    budget = int(0.95 * sbm.get_free_space())
    weights_resident = wo_full_bytes <= budget
    slab_w, slab_scale = [], []
    if weights_resident:
        # Single resident buffer holding ALL K-tiles (loaded once below, before s_batch loop).
        slab_w.append(
            sbm.alloc_stack(
                (_P_MAX, num_mx_k_tiles, hid_per_core), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf, name="oproj_w_full"
            )
        )
        slab_scale.append(
            sbm.alloc_stack(
                (_P_MAX, num_mx_k_tiles, hid_per_core), dtype=nl.uint8, buffer=nl.sbuf, name="oproj_scale_full"
            )
        )
    else:
        # Weight slab spans the FULL hid_per_core width (contiguous HBM read), DOUBLE-BUFFERED
        # so slab N+1's weight DMA overlaps slab N's matmul+accumulate.
        for buf_idx in range(2):
            slab_w.append(
                sbm.alloc_stack(
                    (_P_MAX, slab_512, hid_per_core),
                    dtype=nl.float8_e4m3fn_x4,
                    buffer=nl.sbuf,
                    name=f"oproj_slab_w_{buf_idx}",
                )
            )
            slab_scale.append(
                sbm.alloc_stack(
                    (_P_MAX, slab_512, hid_per_core), dtype=nl.uint8, buffer=nl.sbuf, name=f"oproj_slab_scale_{buf_idx}"
                )
            )
    """
    bf16 accumulator: the within-slab K-contraction accumulates in the fp32 PSUM
    register (independent of the bank's declared dtype), so out_acc only sums the few
    per-slab partials (num_k_slabs = Hdv/512/8, e.g. 4 at 128 heads of 128). A handful of
    bf16 adds of already-fp32-summed partials stays within tolerance, and keeping the
    accumulator bf16 removes the bf16<->fp32 casts on every slab eviction and the drain.
    """
    out_acc = sbm.alloc_stack((oproj_s_tile_pad, hid_per_core), dtype=nl.bfloat16, buffer=nl.sbuf, name="oproj_out_acc")
    out_sb = sbm.alloc_stack((oproj_s_tile_pad, hid_per_core), dtype=nl.bfloat16, buffer=nl.sbuf, name="oproj_out")
    """
    PSUM accumulators DOUBLE-BUFFERED (banks H_PACK..7) so the matmul of the next
    (slab,hid) tile can proceed while the SBUF add of the previous tile still reads its
    bank — breaks the matmul->add->matmul serialization. n_psum_buf pairs cycle.
    """
    n_psum_banks = _NUM_HW_PSUM_BANKS - _H_PACK
    n_psum_buf = max(1, n_psum_banks // num_n_tiles)
    psum_bufs = []
    for psum_buf_idx in range(n_psum_buf):
        row = []
        for i_n in range(num_n_tiles):
            row.append(
                nl.ndarray(
                    (_P_MAX, _MM1_TILE),
                    dtype=nl.bfloat16,
                    buffer=nl.psum,
                    address=(0, (_H_PACK + psum_buf_idx * num_n_tiles + i_n) * PSUM_BANK_SIZE),
                )
            )
        psum_bufs.append(row)
    # TIER 1: load the FULL W_o (all K-tiles) ONCE, before the s_batch loop, so every
    # s-batch reuses the resident weights (no per-s_batch reload).
    if weights_resident:
        _load_mx_weights_k_slab(
            wo_qtz_hbm,
            wo_scale_hbm,
            slab_w[0],
            slab_scale[0],
            in_dim_full=Hdv,
            out_dim=hid_per_core,
            k_tile_start=0,
            k_tile_count=num_mx_k_tiles,
            sbm=sbm,
            name="oproj_w_full_load",
            full_out_dim=HID,
            out_col_offset=hid_start,
            weights_dge_mode=dge_mode.hwdge,
            compact_scales=compact_scales,
        )
    for s_batch in range(num_s_batches):
        s_off = s_batch * oproj_s_tile
        pbuf = 0  # ping-pong index across (slab,hid,i_n) tiles
        # TIER 3 prologue: prefetch slab 0's weight into buffer 0 (resident path skips this).
        if not weights_resident:
            _load_mx_weights_k_slab(
                wo_qtz_hbm,
                wo_scale_hbm,
                slab_w[0],
                slab_scale[0],
                in_dim_full=Hdv,
                out_dim=hid_per_core,
                k_tile_start=0,
                k_tile_count=slab_512,
                sbm=sbm,
                name=f"oproj_slab_{s_batch}_0",
                full_out_dim=HID,
                out_col_offset=hid_start,
                weights_dge_mode=dge_mode.hwdge,
                compact_scales=compact_scales,
            )
        for slab_id in range(num_k_slabs):
            wb = slab_id % 2
            # TIER 3: prefetch the NEXT slab's weight into the alternate buffer so its DMA
            # overlaps this slab's matmul+accumulate below. (resident path: weights already in.)
            if not weights_resident and slab_id + 1 < num_k_slabs:
                _load_mx_weights_k_slab(
                    wo_qtz_hbm,
                    wo_scale_hbm,
                    slab_w[(slab_id + 1) % 2],
                    slab_scale[(slab_id + 1) % 2],
                    in_dim_full=Hdv,
                    out_dim=hid_per_core,
                    k_tile_start=(slab_id + 1) * slab_512,
                    k_tile_count=slab_512,
                    sbm=sbm,
                    name=f"oproj_slab_{s_batch}_{slab_id + 1}",
                    full_out_dim=HID,
                    out_col_offset=hid_start,
                    weights_dge_mode=dge_mode.hwdge,
                    compact_scales=compact_scales,
                )
            for hid_idx in range(num_hid_tiles):
                hid_off = hid_idx * oproj_hid_tile
                for i_n in range(num_n_tiles):
                    n_sz = min(_MM1_TILE, oproj_hid_tile - i_n * _MM1_TILE)
                    n_base = hid_off + i_n * _MM1_TILE
                    ps = psum_bufs[pbuf][i_n]
                    for k_local in range(slab_512):
                        k_global = slab_id * slab_512 + k_local
                        # Resident path: index the single full buffer at the GLOBAL k-tile;
                        # streamed path: index the active double-buffer at the LOCAL k-tile.
                        w_buf = slab_w[0] if weights_resident else slab_w[wb]
                        sc_buf = slab_scale[0] if weights_resident else slab_scale[wb]
                        w_k = k_global if weights_resident else k_local
                        nisa.nc_matmul_mx(
                            dst=ps[0:oproj_s_tile_pad, 0:n_sz],
                            stationary=in_qtz[0:_P_MAX, k_global, nl.ds(s_off, oproj_s_tile_pad)],
                            moving=w_buf[0:_P_MAX, w_k, nl.ds(n_base, n_sz)],
                            stationary_scale=in_scale[0:_P_MAX, k_global, nl.ds(s_off, oproj_s_tile_pad)],
                            moving_scale=sc_buf[0:_P_MAX, w_k, nl.ds(n_base, n_sz)],
                            accumulate=(k_local > 0),
                        )
                    # First slab initializes out_acc (copy, no read-dep); later slabs add.
                    if slab_id == 0:
                        nisa.tensor_copy(
                            dst=out_acc[0:oproj_s_tile_pad, nl.ds(n_base, n_sz)],
                            src=ps[0:oproj_s_tile_pad, 0:n_sz],
                            engine=nisa.engine.vector,
                        )
                    else:
                        nisa.tensor_tensor(
                            dst=out_acc[0:oproj_s_tile_pad, nl.ds(n_base, n_sz)],
                            data1=out_acc[0:oproj_s_tile_pad, nl.ds(n_base, n_sz)],
                            data2=ps[0:oproj_s_tile_pad, 0:n_sz],
                            op=nl.add,
                            engine=nisa.engine.vector,
                        )
                    pbuf = (pbuf + 1) % n_psum_buf
        # Drain fp32 accumulator -> bf16 -> HBM.
        out_row_off = (s_start + s_batch * oproj_s_tile) * HID + hid_start
        nisa.tensor_copy(dst=out_sb[0:oproj_s_tile, :], src=out_acc[0:oproj_s_tile, :], engine=nisa.engine.scalar)
        nisa.dma_copy(
            dst=out_hbm.ap(pattern=[[HID, oproj_s_tile], [1, hid_per_core]], offset=out_row_off),
            src=out_sb[0:oproj_s_tile, :],
        )

    sbm.close_scope()  # Phase-B scope
    sbm.close_scope()  # outer scope (in_qtz/in_scale)
