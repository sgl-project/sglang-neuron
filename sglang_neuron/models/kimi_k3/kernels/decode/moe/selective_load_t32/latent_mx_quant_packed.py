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

"""MXFP8 quantization of a bf16 latent [T, H] (no norm) into the gate/up stationary layout of the kimi_t32 expert body,
in the latent hidden order of the latent-MoE semantics. A copy of the "transpose" form of the kimi subpackage."""

from typing import Callable, List, Optional, Tuple

import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_assert import kernel_assert

_P_MAX = 128
_X4 = 4


def latent_scratch_tiles(
    alloc: Callable[..., nl.NkiTensor], T: int, H: int, in_sbuf: bool, prefix: str = ""
) -> Tuple[nl.NkiTensor, Optional[nl.NkiTensor]]:
    """The SBUF scratch of quantize_latent_mx: (swizzled, latent_sb), latent_sb None for an SBUF latent."""
    swizzled = alloc((_P_MAX, (H // 512) * T * _X4), nl.bfloat16, name=f"{prefix}latent_swizzled")
    latent_sb = None if in_sbuf else alloc((T, H), nl.bfloat16, name=f"{prefix}latent_sb")
    return swizzled, latent_sb


def quantize_latent_mx(
    latent: nl.NkiTensor,
    alloc: Callable[..., nl.NkiTensor],
    latent_ps: nl.NkiTensor,
    hidden_qtz: nl.NkiTensor,
    hidden_scale: nl.NkiTensor,
    engine: nisa.engine,
    prefix: str = "",
    scratch: Optional[Tuple[nl.NkiTensor, Optional[nl.NkiTensor]]] = None,
) -> Tuple[List[nisa.NkiInstruction], nisa.NkiInstruction]:
    """Load a bf16 latent [T, H] (HBM) and quantize it to the MXFP8 gate/up stationary of the expert body. No norm.

    Layouts (q = x4 lane 0..3, p = partition, hb = H512 block):
        latent_sb   bf16 [T, H] SBUF: the latent as loaded (one HWDGE DMA).
        latent_ps   bf16 [128, (H/512) * 4 * T] PSUM: [p, (hb, q, t)] = latent[t, 512*hb + 4*p + q]. 4 PE transposes per
                    H512 block (one per lane q) of the stride-4 column view of latent_sb.
        swizzled    bf16 [128, (H/512) * T * 4] SBUF: [p, (hb, t, q)], q innermost (the lanes of one x4 element).
        hidden_qtz / hidden_scale [128, (H/512) * T]: quantize_mx of swizzled, element [p, hb*T + t] and the MX
                    scales in the quadrant layout. Each MX group (partitions 8r .. 8r + 7 x 4 lanes) holds the latent
                    columns 512*hb + 32*r .. + 31 of token t, so the gate/up weights must use the hidden order
                    h = 512*hb + 4*p + q.

    Args:
        latent (nl.NkiTensor): bfloat16 [T, H] @HBM, T <= 128, or @SBUF with token t on partition t (no load: the
            transposes read it in place).
        alloc (Callable): SBUF allocator for the latent_sb and swizzled tiles.
        latent_ps (nl.NkiTensor): bfloat16 [128, (H/512) * 4 * T] @PSUM scratch (one bank at H = 3584, T = 32).
        hidden_qtz (nl.NkiTensor): float8_e4m3fn_x4 [128, (H/512) * T] @SBUF output.
        hidden_scale (nl.NkiTensor): uint8 [128, (H/512) * T] @SBUF output.
        engine (nisa.engine): the HWDGE queue of the latent load.
        prefix (str): prefix of the instruction and tensor names.
        scratch (Optional[Tuple]): (swizzled, latent_sb) from latent_scratch_tiles, allocated before (latent_sb None
            for an SBUF latent). None (default): alloc allocates them here.

    Returns:
        (the latent load in a list, empty for an SBUF latent; the quantize_mx).
    """
    T, H = latent.shape
    n_h512 = H // 512
    kernel_assert(T <= _P_MAX and H % 512 == 0, f"quantize_latent_mx needs T <= 128 and H % 512 == 0, got {T}, {H}")
    in_sbuf = latent.buffer == nl.sbuf
    if scratch == None:
        scratch = latent_scratch_tiles(alloc, T, H, in_sbuf, prefix)
    swizzled, latent_sb = scratch
    loads = []
    if in_sbuf:
        latent_sb = latent  # the transposes read the caller's tile
    else:
        loads.append(
            nisa.dma_copy(
                dst=latent_sb[0:T, 0:H],
                src=latent,
                dge_mode=nisa.dge_mode.hwdge,
                engine=engine,
                name=f"{prefix}load_latent",
            )
        )
    for block in range(n_h512):
        for lane in range(_X4):
            psum_col = (block * _X4 + lane) * T
            nisa.nc_transpose(
                dst=latent_ps[:, psum_col : psum_col + T],  # [p, t]: lane `lane` of block `block`
                # [t, p]: latent column 512*block + 4*p + lane of each token
                data=latent_sb.ap(pattern=[[latent_sb.shape[1], T], [_X4, _P_MAX]], offset=block * 512 + lane),
                name=f"{prefix}transpose_latent_b{block}_q{lane}",
            )
    nisa.tensor_copy(
        # [p, hb, t, q]: the values to quantize, q innermost
        dst=swizzled.ap(pattern=[[n_h512 * T * _X4, _P_MAX], [T * _X4, n_h512], [_X4, T], [1, _X4]], offset=0),
        # [p, hb, t, q]: latent_ps[p, (hb, q, t)]
        src=latent_ps.ap(pattern=[[n_h512 * _X4 * T, _P_MAX], [_X4 * T, n_h512], [1, T], [T, _X4]], offset=0),
        engine=nisa.vector_engine,
        name=f"{prefix}swizzle_latent",
    )
    quantize = nisa.quantize_mx(dst=hidden_qtz, src=swizzled, dst_scale=hidden_scale, name=f"{prefix}quantize_latent")
    return loads, quantize
