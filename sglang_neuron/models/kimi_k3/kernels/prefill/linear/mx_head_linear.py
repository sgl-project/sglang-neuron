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

"""Kimi-K3 prefill MXFP8 per-head linear (Trn3): ``out[h] = W[h]^T x[h]`` for G heads with a short contraction."""

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from .mx_linear import MX_LANES, P_MAX, SCALE_GROUP

QUADRANT = 32
"""Partitions of one PE row tile; one head a row tile."""
HEADS_PER_ROUND = P_MAX // QUADRANT
MAX_CONTRACT = QUADRANT * MX_LANES
MAX_COLS = 4096
MAX_ROWS = 8192
_M_CHUNK = 512  # moving columns of one MX matmul (one 2 KiB PSUM bank of BF16 holds 1024)
_N_CHUNK = P_MAX  # output partitions of one MX matmul (the stationary free size)
_SCALE_ROWS = QUADRANT // 8  # scale rows of one row tile


@nki.jit
def prefill_mx_head_linear(
    x: nl.NkiTensor,
    w_mx: nl.NkiTensor,
    w_scale: nl.NkiTensor,
) -> nl.NkiTensor:
    """
    MXFP8 per-head linear of the Kimi-K3 prefill: ``out[h] = W[h]^T x[h]`` for G heads, K <= 128, on one core (LNC1).

    Built for the MLA absorb ``q_c = W_UK q_nope`` (K = nope 128, N = kv rank 512, M = query rows, G = local heads).
    A 128-deep contraction fills only 32 partitions of one MX matmul, so the kernel runs 4 heads at once on the 4 PE
    row groups (32-row MX tiles at rows 32 b). x and out are feature-major, the layouts of the flash feature path.

    Dimensions:
        G: heads, 1 .. 64
        K: contraction a head, K % 32 == 0, 32 .. 128
        N: outputs a head, N % 128 == 0, 128 .. 4096
        M: rows (tokens), 1 .. 8192

    Args:
        x (nl.NkiTensor): bfloat16 [G, K, M] @HBM; element (h, k, m) is input k of row m of head h.
        w_mx (nl.NkiTensor): uint8 [G, K / 4, 4 N] @HBM; byte (h, r, 4 n + q) is the e4m3fn value of W[h][4 r + q, n]
            (the consecutive layout of ``prefill_mx_linear`` for each head).
        w_scale (nl.NkiTensor): uint8 [G, K / 32, N] @HBM; row s of head h is the E8M0 scale of the inputs
            32 s .. 32 s + 31 of output n.

    Returns:
        out (nl.NkiTensor): bfloat16 [G, N, M] @HBM.

    Notes:
        - Accuracy class: MXFP8 x MXFP8 (the hardware quantize_mx rule on x, groups of 32 consecutive inputs).
        - One MX matmul a head covers the whole contraction, so the PSUM output is BF16 (one rounding).

    Pseudocode:
        for each round of 4 heads (head b in partitions 32 b .. 32 b + 31):
            w, ws = load the 4 heads' MXFP8 weights and scales           # [128, N]
            for each 512-row chunk:
                xq, xs = quantize_mx(interleave(x[round heads, :, chunk]))   # [128, 512]
                for each 128-output chunk:
                    for b: psum_b = nc_matmul_mx(w[32 b:, n], xq[32 b:], tile_position=(32 b, 0))
                    out[head b, n, chunk] = psum_b
    """
    groups, contract, rows = x.shape
    cols = w_scale.shape[2]
    out = nl.ndarray((groups, cols, rows), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="prefill_mx_head_linear_out")
    prefill_mx_head_linear_body(x, w_mx, w_scale, out)
    return out


def prefill_mx_head_linear_body(
    x: nl.NkiTensor,
    w_mx: nl.NkiTensor,
    w_scale: nl.NkiTensor,
    out: nl.NkiTensor,
) -> None:
    """The body of ``prefill_mx_head_linear`` for a caller kernel that inlines it (not ``@nki.jit``).

    Args:
        x, w_mx, w_scale: as in ``prefill_mx_head_linear``.
        out (nl.NkiTensor): bfloat16 [G, N, M] @HBM, written in full.
    """
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "prefill_mx_head_linear requires Trn3 or newer")
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "prefill_mx_head_linear supports LNC1 only",
    )
    kernel_assert(
        x.dtype == nl.bfloat16 and len(x.shape) == 3, f"x must be bfloat16 [G, K, M], got {x.dtype} {x.shape}"
    )
    groups, contract, rows = x.shape
    cols = w_scale.shape[2]
    kernel_assert(1 <= groups <= 64, f"G={groups} must be in 1 .. 64")
    kernel_assert(
        contract % SCALE_GROUP == 0 and SCALE_GROUP <= contract <= MAX_CONTRACT,
        f"K={contract} must be a multiple of {SCALE_GROUP} in {SCALE_GROUP} .. {MAX_CONTRACT}",
    )
    kernel_assert(
        cols % _N_CHUNK == 0 and cols <= MAX_COLS, f"N={cols} must be a multiple of {_N_CHUNK}, <= {MAX_COLS}"
    )
    kernel_assert(1 <= rows <= MAX_ROWS, f"M={rows} must be in 1 .. {MAX_ROWS}")
    data_rows = contract // MX_LANES
    scale_rows = contract // SCALE_GROUP
    kernel_assert(
        w_mx.dtype == nl.uint8 and tuple(w_mx.shape) == (groups, data_rows, MX_LANES * cols),
        f"w_mx must be uint8 [{groups}, {data_rows}, {MX_LANES * cols}], got {w_mx.dtype} {w_mx.shape}",
    )
    kernel_assert(
        w_scale.dtype == nl.uint8 and tuple(w_scale.shape) == (groups, scale_rows, cols),
        f"w_scale must be uint8 [{groups}, {scale_rows}, {cols}], got {w_scale.dtype} {w_scale.shape}",
    )
    kernel_assert(tuple(out.shape) == (groups, cols, rows), f"out must be [{groups}, {cols}, {rows}], got {out.shape}")
    # One x4 element (uint32) per weight row and column: same-width DMA, then an x4 view of the SBUF tile.
    w_words = w_mx.view(nl.uint32).reshape((groups * data_rows, cols))
    whole = data_rows == QUADRANT
    n_chunks = cols // _N_CHUNK
    evictions = 0

    for first_head in range(0, groups, HEADS_PER_ROUND):
        heads = min(HEADS_PER_ROUND, groups - first_head)
        used = QUADRANT * heads
        w_sb = nl.ndarray((P_MAX, cols), dtype=nl.uint32, buffer=nl.sbuf)
        ws_sb = nl.ndarray((P_MAX, cols), dtype=nl.uint8, buffer=nl.sbuf)
        if whole:
            nisa.dma_copy(dst=w_sb[0:used, :], src=w_words[first_head * data_rows : first_head * data_rows + used, :])
        else:
            # Zero partitions past K / 4 in each row tile: zero data adds nothing.
            nisa.memset(dst=w_sb[0:used, :], value=0, engine=nisa.engine.gpsimd)
            for head_pos in range(heads):
                row = (first_head + head_pos) * data_rows
                nisa.dma_copy(
                    dst=w_sb[QUADRANT * head_pos : QUADRANT * head_pos + data_rows, :],
                    src=w_words[row : row + data_rows, :],
                )
        for head_pos in range(heads):
            base = QUADRANT * head_pos
            if scale_rows < _SCALE_ROWS:
                nisa.memset(dst=ws_sb[base : base + _SCALE_ROWS, :], value=0, engine=nisa.engine.gpsimd)
            nisa.dma_copy(dst=ws_sb[base : base + scale_rows, :], src=w_scale[first_head + head_pos, :, :])
        w_x4 = w_sb.view(nl.float8_e4m3fn_x4)

        for row0 in range(0, rows, _M_CHUNK):
            width = min(_M_CHUNK, rows - row0)
            # nc_matmul_mx needs an even moving size; the pad column is zero.
            padded = width + width % 2
            # [p, lane, m]: partition 32 b + r, lane q holds input 4 r + q of head b.
            lanes_sb = nl.ndarray((P_MAX, MX_LANES, _M_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
            if not whole or padded > width:
                nisa.memset(dst=lanes_sb[0:used, :, :], value=0.0, engine=nisa.engine.gpsimd)
            if whole:
                nisa.dma_copy(
                    dst=lanes_sb[0:used, :, 0:width],
                    src=x.ap(
                        pattern=[[MX_LANES * rows, used], [rows, MX_LANES], [1, width]],
                        offset=first_head * contract * rows + row0,
                    ),
                )
            else:
                for head_pos in range(heads):
                    nisa.dma_copy(
                        dst=lanes_sb[QUADRANT * head_pos : QUADRANT * head_pos + data_rows, :, 0:width],
                        src=x.ap(
                            pattern=[[MX_LANES * rows, data_rows], [rows, MX_LANES], [1, width]],
                            offset=(first_head + head_pos) * contract * rows + row0,
                        ),
                    )
            interleaved = nl.ndarray((P_MAX, _M_CHUNK, MX_LANES), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=interleaved[0:used, 0:padded, :],
                src=lanes_sb[0:used, :, 0:padded].permute((0, 2, 1)),
                engine=nisa.engine.scalar,
            )
            x_data = nl.ndarray((P_MAX, _M_CHUNK), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
            x_scales = nl.ndarray((P_MAX, _M_CHUNK), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.quantize_mx(
                dst=x_data[0:used, 0:padded],
                src=interleaved[0:used, 0:padded, :],
                dst_scale=x_scales[0:used, 0:padded],
            )

            out_sb = [nl.ndarray((P_MAX, n_chunks, _M_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(heads)]
            for chunk in range(n_chunks):
                col0 = chunk * _N_CHUNK
                # One 2 KiB bank a head: the 4 row tiles run at once, each into its own bank at offset 0.
                psums = [nl.ndarray((P_MAX, 2 * _M_CHUNK), dtype=nl.bfloat16, buffer=nl.psum) for _ in range(heads)]
                for head_pos in range(heads):
                    base = QUADRANT * head_pos
                    nisa.nc_matmul_mx(
                        dst=psums[head_pos][:, 0:padded],
                        stationary=w_x4[base : base + QUADRANT, col0 : col0 + _N_CHUNK],
                        moving=x_data[base : base + QUADRANT, 0:padded],
                        stationary_scale=ws_sb[base : base + _SCALE_ROWS, col0 : col0 + _N_CHUNK],
                        moving_scale=x_scales[base : base + _SCALE_ROWS, 0:padded],
                        tile_position=(base, 0),
                        tile_size=(QUADRANT, P_MAX),
                    )
                for head_pos in range(heads):
                    engine = nisa.engine.scalar if evictions % 3 == 2 else nisa.engine.vector
                    evictions += 1
                    nisa.tensor_copy(
                        dst=out_sb[head_pos][:, chunk, 0:width], src=psums[head_pos][:, 0:width], engine=engine
                    )
            for head_pos in range(heads):
                nisa.dma_copy(
                    dst=out.ap(
                        pattern=[[rows, _N_CHUNK], [_N_CHUNK * rows, n_chunks], [1, width]],
                        offset=(first_head + head_pos) * cols * rows + row0,
                    ),
                    src=out_sb[head_pos][:, :, 0:width],
                )
