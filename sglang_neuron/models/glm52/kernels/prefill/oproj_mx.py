"""GLM_GK_PRJ_OMX=1: the head-TP V-up + o_proj of the prefill GK with the o_proj on the MX PE path.

One kernel for the wo group's ``S = g * T`` tokens of this rank's ``H`` heads (the o all-to-all's receive buffer):

  Phase V  V-up exactly as ``mla_vup_cte_kernel``'s pipelined path (bf16 weights, bf16 latent DMA-transposed K-major per 128-token
           chunk, the same matmuls into bf16 PSUM), but each ``[128 dv, 128 tokens]`` PSUM tile is evicted straight into the MX
           interleaved layout of the o_proj stationary and quantized on the DVE (``nisa.quantize_mx``, MXFP8 e4m3, one e8m0 scale
           per 32 K values). The quantized activation stays in SBUF (no vup_kt HBM round trip).
  Phase O  ``partial[S, HID] = vup[S, Hdv] @ wo[Hdv, HID]`` with ``nc_matmul_mx``: MXFP8 x MXFP8, K = 512 per instruction, the whole
           Hdv contraction (4 instructions at Hdv = 2,048) accumulated in one fp32 PSUM bank, one bf16 eviction per
           [128 tokens, 512 cols] tile (no K-slab bf16 adds). wo is streamed once in column slabs (all slabs resident when they
           fit, loads issued under Phase V).

Layouts (the nc_matmul_mx interleave of the NKI MXFP guide, which the prefill loaders' wo_mx_pack writes):
  K-tile t of 512 Hdv rows, partition p, x4 sub j  <->  Hdv row ``512 t + 128 j + p``; one scale group = 8 partitions x 4 subs.
  activation  aq[p, c, t, s] (fp8x4; token 128 c + s), a_s[32 q + i, c, t, s] = the e8m0 of group 4 q + i (quantize_mx's
              quadrant layout).
  wo_q_hbm    [Hdv / 4, HID] int32 (fp8x4 bytes, little-endian, byte j = sub j), row ``128 t + p``.
  wo_s_hbm    [Hdv / 32, HID] uint8 e8m0, row ``16 t + g`` = group g (partitions 8 g .. 8 g + 7) of K-tile t.
The loader builds both (``weight_loaders.wo_mx_pack``). Head h's V-up rows ``h * d_v + 128 dv + p`` land at t = (h d_v + 128 dv) // 512,
j = ((h d_v + 128 dv) % 512) // 128, so every PSUM tile is one (t, j) plane and the eviction is one strided write.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from .attention.mla_bf16 import _BF16_K_TILE, _bf16_k_tile_sizes, _load_bf16_weights_k_slab
from .attention.common import _NUM_HW_PSUM_BANKS, _P_MAX, _get_psum_bank_size, _new_sbm
from .utils.kernel_assert import kernel_assert

_MX_K = 512  # contraction rows per nc_matmul_mx (128 partitions x 4)
_N = 512  # moving columns per matmul / fp32 PSUM bank
_S_TILE = 128  # tokens per stationary tile
_VUP_BANKS = 2  # PSUM banks of Phase V (the last two); Phase O rotates over the rest
_SLAB_COLS = 1536  # wo column slab (3 x 512)


@nki.jit
def glm_vup_oproj_mx_kernel(
    lat_hbm: nl.NkiTensor,
    wuv_hbm: nl.NkiTensor,
    wo_q_hbm: nl.NkiTensor,
    wo_s_hbm: nl.NkiTensor,
    out_hbm: nl.NkiTensor,
    kv_lora_rank: int = 512,
):
    """``out_hbm[S, HID] (bf16) = V-up(lat_hbm[S, H * L]) @ wo`` with the o_proj on MXFP8 (see the module docstring).

    Args:
        lat_hbm: [S, H * L] bf16, this rank's H heads' latent attention output for the group's S tokens (row-major).
        wuv_hbm: [H * L, d_v] bf16 V-up weight of the H heads.
        wo_q_hbm: [H * d_v / 4, HID] int32, MXFP8 o_proj row shard (fp8x4 bytes).
        wo_s_hbm: [H * d_v / 32, HID] uint8 e8m0 scales.
        out_hbm: [S, HID] bf16 HBM partial (the reduce-scatter source).
    """
    S, HL = lat_hbm.shape
    L = kv_lora_rank
    H = HL // L
    d_v = wuv_hbm.shape[1]
    Hdv = H * d_v
    KT = Hdv // _MX_K
    HID = wo_q_hbm.shape[1]
    kernel_assert(H * L == HL and L % _BF16_K_TILE == 0, f"glm_vup_oproj_mx: H*L {HL} / L {L}")
    kernel_assert(d_v % _BF16_K_TILE == 0 and Hdv % _MX_K == 0, f"glm_vup_oproj_mx: Hdv {Hdv} must be a multiple of {_MX_K}")
    kernel_assert(tuple(wo_q_hbm.shape) == (Hdv // 4, HID), f"glm_vup_oproj_mx: wo_q {wo_q_hbm.shape} != {(Hdv // 4, HID)}")
    kernel_assert(tuple(wo_s_hbm.shape) == (Hdv // 32, HID), f"glm_vup_oproj_mx: wo_s {wo_s_hbm.shape} != {(Hdv // 32, HID)}")
    kernel_assert(tuple(out_hbm.shape) == (S, HID), f"glm_vup_oproj_mx: out {out_hbm.shape} != {(S, HID)}")
    kernel_assert(S % _S_TILE == 0, f"glm_vup_oproj_mx: S {S} must be a multiple of {_S_TILE}")
    kernel_assert(HID % _SLAB_COLS == 0, f"glm_vup_oproj_mx: HID {HID} must be a multiple of {_SLAB_COLS}")
    nc = S // _S_TILE
    LK = L // _BF16_K_TILE
    DVT = d_v // _BF16_K_TILE
    grp_k = H * LK
    n_slabs = HID // _SLAB_COLS
    n_per_slab = _SLAB_COLS // _N
    PSUM_BANK = _get_psum_bank_size()
    o_banks = _NUM_HW_PSUM_BANKS - _VUP_BANKS

    sbm = _new_sbm("glm_vup_oproj_mx")
    sbm.open_scope()
    # ---- resident: quantized activation (all S tokens), V-up weight, wo slabs ----
    # chunk-major [p, c, t, s]: quantize_mx's dst must be one contiguous run per partition (one chunk's KT x 128)
    aq = sbm.alloc_stack((_P_MAX, nc, KT, _S_TILE), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf, name="vomx_aq")
    a_s = sbm.alloc_stack((_P_MAX, nc, KT, _S_TILE), dtype=nl.uint8, buffer=nl.sbuf, name="vomx_as")
    wuv_all = sbm.alloc_stack((_P_MAX, grp_k, d_v), dtype=nl.bfloat16, buffer=nl.sbuf, name="vomx_wuv")
    lat_ts = [
        sbm.alloc_stack((_P_MAX, grp_k, _S_TILE), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"vomx_lat{i}") for i in range(2)
    ]
    in_ts = [
        sbm.alloc_stack((_P_MAX, KT, _S_TILE * 4), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"vomx_in{i}") for i in range(2)
    ]
    o_sbs = [sbm.alloc_stack((_P_MAX, _SLAB_COLS), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"vomx_o{i}") for i in range(2)]
    slab_bytes = KT * _SLAB_COLS * (4 + 1)
    n_wbuf = n_slabs if n_slabs * slab_bytes <= int(0.95 * sbm.get_free_space()) else 2
    w_q = [
        sbm.alloc_stack((_P_MAX, KT, _SLAB_COLS), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf, name=f"vomx_wq{i}")
        for i in range(n_wbuf)
    ]
    w_s = [sbm.alloc_stack((_P_MAX, KT, _SLAB_COLS), dtype=nl.uint8, buffer=nl.sbuf, name=f"vomx_ws{i}") for i in range(n_wbuf)]

    def _load_w(sl: int):
        """wo column slab ``sl`` -> buffer ``sl % n_wbuf``: one fp8x4 DMA + one scale DMA per partition quadrant."""
        b, c0 = sl % n_wbuf, sl * _SLAB_COLS
        nisa.dma_copy(
            dst=w_q[b][0:_P_MAX, 0:KT, 0:_SLAB_COLS],
            src=wo_q_hbm.ap(pattern=[[HID, _P_MAX], [_P_MAX * HID, KT], [1, _SLAB_COLS]], offset=c0, dtype=nl.float8_e4m3fn_x4),
            dge_mode=dge_mode.hwdge,
        )
        for q in range(4):
            # groups 4q .. 4q+3 of every K-tile (their data partitions are quadrant q) -> partitions 32q .. 32q+3
            nisa.dma_copy(
                dst=w_s[b][nl.ds(32 * q, 4), 0:KT, 0:_SLAB_COLS],
                src=wo_s_hbm.ap(pattern=[[HID, 4], [16 * HID, KT], [1, _SLAB_COLS]], offset=4 * q * HID + c0, dtype=nl.uint8),
                dge_mode=dge_mode.hwdge,
            )

    lat_flat = lat_hbm.reshape((S, HL))
    n_c16 = _S_TILE // 16

    def _issue_latent(c: int):
        # The VUP1 path's latent load: src[a, k, c, d] = lat[row0 + 16c + a, 128k + d], permuted (3, 1, 2, 0) to dst[d, k, c, a].
        nisa.dma_transpose(
            dst=lat_ts[c % 2].ap(pattern=[[grp_k * _S_TILE, _BF16_K_TILE], [_S_TILE, grp_k], [16, n_c16], [1, 16]], offset=0),
            src=lat_flat.ap(
                pattern=[[HL, 16], [_BF16_K_TILE, grp_k], [16 * HL, n_c16], [1, _BF16_K_TILE]], offset=c * _S_TILE * HL
            ),
            axes=(3, 1, 2, 0),
            dge_mode=dge_mode.hwdge,
        )

    # ---- Phase V ----
    _issue_latent(0)
    _load_bf16_weights_k_slab(
        wuv_hbm, wuv_all, out_dim=d_v, k_tile_start=0, k_tile_count=grp_k, k_tile_sizes=_bf16_k_tile_sizes(HL), full_out_dim=d_v
    )
    w_issued = 0
    if n_wbuf == n_slabs:
        _load_w(0)
        w_issued = 1
    for c in range(nc):
        if c + 1 < nc:
            _issue_latent(c + 1)
        if n_wbuf == n_slabs and w_issued < n_slabs:
            # the resident wo slabs ride the latent loads, one per chunk (the latent of chunk c + 1 is queued first)
            _load_w(w_issued)
            w_issued += 1
        lat_t, in_t = lat_ts[c % 2], in_ts[c % 2]
        for h in range(H):
            for dv in range(DVT):
                i = h * DVT + dv
                ps = nl.ndarray(
                    (_BF16_K_TILE, _S_TILE),
                    dtype=nl.bfloat16,
                    buffer=nl.psum,
                    address=(0, (o_banks + i % _VUP_BANKS) * PSUM_BANK),
                )
                for kt in range(LK):
                    nisa.nc_matmul(
                        dst=ps[0:_BF16_K_TILE, 0:_S_TILE],
                        stationary=wuv_all[0:_BF16_K_TILE, h * LK + kt, nl.ds(dv * _BF16_K_TILE, _BF16_K_TILE)],
                        moving=lat_t[0:_BF16_K_TILE, h * LK + kt, nl.ds(0, _S_TILE)],
                        accumulate=(kt > 0),
                    )
                row = h * d_v + dv * _BF16_K_TILE
                t, j = row // _MX_K, (row % _MX_K) // _BF16_K_TILE
                # MX interleave: in_t[p, t, s * 4 + j] = vup[s, 512 t + 128 j + p]
                nisa.tensor_copy(
                    dst=in_t.ap(pattern=[[KT * _S_TILE * 4, _P_MAX], [4, _S_TILE]], offset=t * _S_TILE * 4 + j),
                    src=ps[0:_BF16_K_TILE, 0:_S_TILE],
                    engine=nisa.engine.scalar if i % 2 == 0 else nisa.engine.vector,
                )
        nisa.quantize_mx(
            src=in_t[0:_P_MAX, 0:KT, 0 : _S_TILE * 4],
            dst=aq[0:_P_MAX, c, 0:KT, 0:_S_TILE],
            dst_scale=a_s[0:_P_MAX, c, 0:KT, 0:_S_TILE],
        )
    while n_wbuf == n_slabs and w_issued < n_slabs:
        _load_w(w_issued)
        w_issued += 1
    if n_wbuf < n_slabs:
        _load_w(0)

    # ---- Phase O ----
    bank = 0
    for sl in range(n_slabs):
        b = sl % n_wbuf
        if n_wbuf < n_slabs and sl + 1 < n_slabs:
            _load_w(sl + 1)
        for c in range(nc):
            o_sb = o_sbs[(sl * nc + c) % 2]
            for n in range(n_per_slab):
                ps = nl.ndarray((_P_MAX, _N), dtype=nl.float32, buffer=nl.psum, address=(0, bank * PSUM_BANK))
                bank = (bank + 1) % o_banks
                for t in range(KT):
                    nisa.nc_matmul_mx(
                        dst=ps[0:_S_TILE, 0:_N],
                        stationary=aq[0:_P_MAX, c, t, 0:_S_TILE],
                        moving=w_q[b][0:_P_MAX, t, nl.ds(n * _N, _N)],
                        stationary_scale=a_s[0:_P_MAX, c, t, 0:_S_TILE],
                        moving_scale=w_s[b][0:_P_MAX, t, nl.ds(n * _N, _N)],
                        accumulate=(t > 0),
                    )
                nisa.tensor_copy(
                    dst=o_sb[0:_S_TILE, nl.ds(n * _N, _N)],
                    src=ps[0:_S_TILE, 0:_N],
                    engine=nisa.engine.scalar if (c * n_per_slab + n) % 2 == 0 else nisa.engine.vector,
                )
            nisa.dma_copy(
                dst=out_hbm.ap(pattern=[[HID, _S_TILE], [1, _SLAB_COLS]], offset=c * _S_TILE * HID + sl * _SLAB_COLS),
                src=o_sb[0:_S_TILE, 0:_SLAB_COLS],
            )
    sbm.close_scope()
    return out_hbm
