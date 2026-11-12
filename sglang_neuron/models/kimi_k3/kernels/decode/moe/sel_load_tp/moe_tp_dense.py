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

"""Kimi-K3 TP128 dense routed-expert decode on the selective-load packed layout: every expert slice, once per call."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.logging import get_logger
from ...utils.trace_names import next_call, restart_calls
from .moe_sel_load_tp import COMPILER_SBUF_RESERVE, _broadcast_routing
from .selective_load_packed import _Suffixed

_PMAX = 128
_X4 = 4
_H512 = 512
_TOP_K = 16
_GROUP = 16  # experts per group: their gate (or up) columns fill one moving tile, their 8 MX rows the 128 partitions
_MX_ROWS = 8
_BLOCKS = 4  # H512 blocks per gate/up scale chunk
_PSUM_COLS = 512  # gate at column 0 of a gate/up PSUM bank, up at column _PSUM_COLS
_PSUM_BANK_BYTES = 2048
_BANK_FP32 = _PSUM_BANK_BYTES // 4
_DOWN_CHUNK = 512
MAX_TOKENS = 128
# Ring depths: packed gate/up rows, moving gate/up tiles, down weights. A load for group g + depth waits for the reads
# of group g.
_ROW_SLOTS = 3
_MOVING_SLOTS = 2
_DOWN_SLOTS = 3
_ISA, _LANGUAGE = nisa, nl


@nki.jit
def moe_tp_dense(
    latent: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> nl.NkiTensor:
    """Compute one TP128 rank's routed-expert partial with every expert slice: the dense form of ``moe_sel_load_tp``.

    The kernel streams the 896 expert slices of the selective-load packed layout once, 16 experts per group, with all
    T tokens as the stationary operand. The affinity of token t for expert e is zero when e is not routed for t, so
    such pairs add zero. The cost does not depend on the routing.

    Dimensions:
        T: tokens, 1 <= T <= 128. K: 16. E: 896. H: latent size, 3584. I: intermediate slice, 24. G = E/16 groups.

    Args:
        latent (nl.NkiTensor): bfloat16 [T, H] @HBM, or @SBUF with token t on partition t.
        expert_index (nl.NkiTensor): int32 (or uint32) [T, K]: the routed experts of each token.
        expert_affinities (nl.NkiTensor): float32 [T, K]: their routing weights. Same buffer as expert_index.
        expert_gate_up_packed (nl.NkiTensor): uint8 [E, 128, 768] @HBM.
        expert_down_weights (nl.NkiTensor): float4_e2m1fn_x4 (or uint16) [E, 8, H] @HBM.
        expert_down_weights_scale (nl.NkiTensor): uint8 [E, H] @HBM.
        beta (float): SiTU gate soft-clipping factor.
        linear_beta (float): SiTU up soft-clipping factor.

    Returns:
        output (nl.NkiTensor): bfloat16 [T, H] @HBM, this rank's partial.

    Notes:
        - Per group: gate/up MX matmuls (moving: 16 experts x I columns), SiTU in FP32, the activation times the
          affinity of each (token, expert) in one MXFP8 term, and the down MX matmul that adds the 16 experts. PSUM
          accumulates the groups in BF16.
        - Every token column follows the same operations, so the result of a token does not depend on T or on the
          other tokens.
    """
    restart_calls()
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "moe_tp_dense supports LNC1 only",
    )
    kernel_assert(nisa.get_nc_version() >= nisa.nc_version.gen4, "moe_tp_dense requires Trn3 or newer")
    return moe_tp_dense_body(
        latent,
        expert_index,
        expert_affinities,
        expert_gate_up_packed,
        expert_down_weights,
        expert_down_weights_scale,
        beta,
        linear_beta,
    )


def moe_tp_dense_body(
    latent: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    sbuf=None,
    output: nl.NkiTensor = None,
    output_row: int = 0,
    private_out: bool = False,
) -> nl.NkiTensor:
    """The body of ``moe_tp_dense`` for a caller that inlines it (not ``@nki.jit``).

    Args:
        latent, expert_index, expert_affinities, expert_gate_up_packed, expert_down_weights,
        expert_down_weights_scale, beta, linear_beta: as in ``moe_tp_dense``.
        sbuf: the SBUF region (an ``SbufManager`` with explicit addresses, or an object with an ``sbm`` attribute).
            None: a manager of SBUF without the top COMPILER_SBUF_RESERVE bytes.
        output: optional bfloat16 HBM destination with H columns. output_row: its row for token 0.
        private_out: the new HBM output of output None is private HBM instead of shared HBM (opt-in; for a caller
            kernel that keeps every per-core scratch tensor private, as one compiled with ``--lnc1-with-shared-mem``,
            where a ``shared_hbm`` scratch tensor is one buffer for the two cores of an HBM stack). Only the buffer
            changes, so the result is the same. No effect with an output. False (default) keeps the program.

    Returns:
        output (nl.NkiTensor): bfloat16 [T, H] @HBM.
    """
    call = next_call("tp_dense")
    suffix = f"_c{call}" if call else ""
    nisa, nl = _Suffixed(_ISA, suffix), _Suffixed(_LANGUAGE, suffix, functions=("ndarray",))
    T, H = latent.shape
    E, _, row_bytes = expert_gate_up_packed.shape
    n_h512 = H // _H512
    n_groups = E // _GROUP
    n_scale_chunks = div_ceil(n_h512, _BLOCKS)
    n_segments = div_ceil(2 * n_scale_chunks, _X4)
    I = row_bytes // (_X4 * (n_h512 + n_segments))
    down_rows = I // _X4
    group_cols = _GROUP * I
    gate_up_cols = 2 * n_h512 * I  # x4 elements of one packed expert row
    n_down_chunks = H // _DOWN_CHUNK
    padded = div_ceil(T, _X4) * _X4
    kernel_assert(1 <= T <= MAX_TOKENS, f"Expected 1 <= T <= {MAX_TOKENS}, got T={T}")
    kernel_assert(latent.dtype == nl.bfloat16, f"Expected a bf16 latent, got {latent.dtype}")
    kernel_assert(E % (8 * _GROUP) == 0, f"Expected E % 128 == 0, got E={E}")
    kernel_assert(
        H % _H512 == 0 and n_h512 < 9 and I % _X4 == 0 and I <= 32, f"Unsupported H={H} or I={I} for moe_tp_dense"
    )
    kernel_assert(
        tuple(expert_index.shape) == (T, _TOP_K) and tuple(expert_affinities.shape) == (T, _TOP_K),
        f"Expected [T, {_TOP_K}] routing, got {expert_index.shape} and {expert_affinities.shape}",
    )
    if expert_index.dtype == nl.uint32:
        expert_index = expert_index.view(nl.int32)
    if expert_down_weights.dtype == nl.uint16:
        expert_down_weights = expert_down_weights.view(nl.float4_e2m1fn_x4)
    if sbuf is None:
        sbuf = SbufManager(
            0, nl.tile_size.total_available_sbuf_size - COMPILER_SBUF_RESERVE, get_logger("moe_tp_dense")
        )
    sbm = getattr(sbuf, "sbm", sbuf)
    if output is None:
        buffer = nl.private_hbm if private_out else nl.shared_hbm
        output = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=buffer, name="moe_tp_dense_out")
    dve, act = nisa.vector_engine, nisa.scalar_engine

    sbm.open_scope(name="tp_dense" + suffix)
    sbm.align_stack_curr_addr(512)

    def alloc(shape, dtype, name):
        return sbm.alloc_stack(shape, dtype, name=name + suffix)

    # The weight rings first, so their DMA writes stay aligned.
    gate_up_w = alloc((_PMAX, _ROW_SLOTS * _GROUP * row_bytes // 2), nl.uint16, "dense_gate_up_w")
    gate_up_moving = alloc((_PMAX, _MOVING_SLOTS * _GROUP * gate_up_cols), nl.uint16, "dense_gate_up_moving")
    moving_s_cols = n_segments * _BLOCKS * group_cols // 2  # uint16 per moving slot
    gate_up_moving_s = alloc((_PMAX, _MOVING_SLOTS * moving_s_cols), nl.uint16, "dense_gate_up_moving_s")
    down_w = alloc((_PMAX, _DOWN_SLOTS * H), expert_down_weights.dtype, "dense_down_w")
    # down_s: partition 32q + 4s + j holds the scale row of expert 4q + j of the group in slot s = g % 4.
    down_s = alloc((_PMAX, H), nl.uint8, "dense_down_s")
    hidden_qtz = alloc((_PMAX, n_h512 * padded), nl.float8_e4m3fn_x4, "dense_hidden_qtz")  # [p, (hb, t)]
    hidden_scale = alloc((_PMAX, n_h512 * padded), nl.uint8, "dense_hidden_scale")
    # dense_affinity[m, (g, t)] = the affinity of token t for expert 16g + m/8 (zero when it is not routed)
    dense_affinity = alloc((_PMAX, n_groups * padded), nl.bfloat16, "dense_affinity")
    gate_act = alloc((padded, 2 * group_cols), nl.float32, "dense_gate_act")
    situ_sigmoid = alloc((padded, 2 * group_cols), nl.float32, "dense_situ_sigmoid")
    situ_up = alloc((padded, 2 * group_cols), nl.float32, "dense_situ_up")
    gated = alloc((padded, 2 * _X4 * _PMAX), nl.bfloat16, "dense_gated")  # [t, (q, c, p)], rows p >= I/4 zero
    qmx_in = alloc((_PMAX, 2 * padded * _X4), nl.float16, "dense_qmx_in")
    inter = alloc((_PMAX, 2 * padded), nl.float8_e4m3fn_x4, "dense_inter")
    inter_s = alloc((_PMAX, 2 * padded), nl.uint8, "dense_inter_s")
    out_sb = alloc((padded, H), nl.bfloat16, "dense_out_sb")

    gate_up_ps_by_parity = [
        nl.ndarray(
            (padded, 2 * _PSUM_COLS),
            dtype=nl.bfloat16,
            buffer=nl.psum,
            address=(0, bank * _PSUM_BANK_BYTES),
            name=f"dense_gate_up_ps{bank}",
        )
        for bank in range(2)
    ]
    out_ps = nl.ndarray(
        (padded, H), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 2 * _PSUM_BANK_BYTES), name="dense_out_ps"
    )
    gated_ps_by_parity = [
        nl.ndarray(
            (_PMAX, _X4 * padded),
            dtype=nl.bfloat16,
            buffer=nl.psum,
            address=(0, (6 + parity) * _PSUM_BANK_BYTES),
            name=f"dense_gated_ps{parity}",
        )
        for parity in range(2)
    ]

    # Weight loads (HWDGE). Rows: 16 experts x 768 B on each partition; down: 128 contiguous rows; scales: 4 DMAs.
    rows_hbm = expert_gate_up_packed.view(nl.uint16)
    kernel_assert(
        expert_down_weights.shape[1] == _MX_ROWS,
        f"tp_dense loads 16 experts x 8 down rows in one DMA. It needs the padded down weights [E, 8, H], got "
        f"{expert_down_weights.shape}. Use the sel_load kernel for the compact down weights.",
    )
    down_rows_hbm = expert_down_weights.reshape((E * _MX_ROWS, H))
    queues = (nisa.engine.sync, nisa.engine.scalar)

    def load_rows(group):
        slot = group % _ROW_SLOTS
        words = row_bytes // 2  # uint16 words of a packed row
        nisa.dma_copy(
            dst=gate_up_w.ap(
                pattern=[[_ROW_SLOTS * _GROUP * words, _PMAX], [words, _GROUP], [1, words]],
                offset=slot * _GROUP * words,
            ),
            # [p, e, word]: row p of the 16 experts of the group
            src=rows_hbm.ap(
                pattern=[[words, _PMAX], [_PMAX * words, _GROUP], [1, words]],
                offset=group * _GROUP * _PMAX * words,
            ),
            dge_mode=nisa.dge_mode.hwdge,
            engine=queues[group % 2],
            name=f"dense_load_rows_g{group}",
        )

    def load_down(group):
        slot = group % _DOWN_SLOTS
        nisa.dma_copy(
            dst=down_w[:, slot * H : (slot + 1) * H],
            src=down_rows_hbm[group * _PMAX : (group + 1) * _PMAX, :],
            dge_mode=nisa.dge_mode.hwdge,
            engine=queues[(group + 1) % 2],
            name=f"dense_load_down_g{group}",
        )
        scale_slot = group % _X4
        for quadrant in range(_X4):
            first = 32 * quadrant + _X4 * scale_slot
            nisa.dma_copy(
                dst=down_s[first : first + _X4, :],
                src=expert_down_weights_scale[
                    group * _GROUP + _X4 * quadrant : group * _GROUP + _X4 * (quadrant + 1), :
                ],
                dge_mode=nisa.dge_mode.hwdge,
                engine=queues[quadrant % 2],
                name=f"dense_load_down_s_g{group}_q{quadrant}",
            )

    for group in range(min(_ROW_SLOTS, n_groups)):
        load_rows(group)
    for group in range(min(_DOWN_SLOTS, n_groups)):
        load_down(group)

    # Prologue: the latent in the MXFP8 stationary layout, and the dense affinities on the expert partitions.
    _quantize_latent_blocks(sbm, latent, hidden_qtz, hidden_scale, padded, suffix)
    _dense_affinities(sbm, expert_index, expert_affinities, dense_affinity, padded, n_groups, suffix)
    nisa.memset(dst=gated, value=0.0, engine=dve, name="dense_zero_gated")

    def relayout(group):
        """Copy the packed rows of group to its moving slot: DVE the weights, ACT the scale segments."""
        rows_base = (group % _ROW_SLOTS) * _GROUP * row_bytes // 2
        moving_base = (group % _MOVING_SLOTS) * _GROUP * gate_up_cols
        for gate_or_up in range(2):
            nisa.tensor_copy(
                # [p, hb, c, I]
                dst=gate_up_moving.ap(
                    pattern=[[_MOVING_SLOTS * _GROUP * gate_up_cols, _PMAX], [group_cols, n_h512], [I, _GROUP], [1, I]],
                    offset=moving_base + gate_or_up * n_h512 * group_cols,
                ),
                src=gate_up_w.ap(
                    pattern=[
                        [_ROW_SLOTS * _GROUP * row_bytes // 2, _PMAX],
                        [I, n_h512],
                        [row_bytes // 2, _GROUP],
                        [1, I],
                    ],
                    offset=rows_base + gate_or_up * n_h512 * I,
                ),
                engine=dve,
                name=f"dense_relayout_{('gate', 'up')[gate_or_up]}_g{group}",
            )
        for segment in range(n_segments):
            nisa.tensor_copy(
                dst=gate_up_moving_s.ap(
                    pattern=[
                        [_MOVING_SLOTS * moving_s_cols, _PMAX],
                        [group_cols // 2, _BLOCKS],
                        [I // 2, _GROUP],
                        [1, I // 2],
                    ],
                    offset=(group % _MOVING_SLOTS) * moving_s_cols + segment * _BLOCKS * group_cols // 2,
                ),
                src=gate_up_w.ap(
                    pattern=[
                        [_ROW_SLOTS * _GROUP * row_bytes // 2, _PMAX],
                        [I // 2, _BLOCKS],
                        [row_bytes // 2, _GROUP],
                        [1, I // 2],
                    ],
                    offset=rows_base + gate_up_cols + segment * _BLOCKS * I // 2,
                ),
                engine=act,
                name=f"dense_relayout_scales_g{group}_s{segment}",
            )

    def gate_up_projection(group, blocks, after):
        gate_up_ps = gate_up_ps_by_parity[group % 2]
        matmul = after
        for block in blocks:
            for gate_or_up in range(2):
                weight_col = (group % _MOVING_SLOTS) * _GROUP * gate_up_cols + (
                    gate_or_up * n_h512 + block
                ) * group_cols
                scale_chunk = gate_or_up * n_scale_chunks + block // _BLOCKS
                scale_col = ((group % _MOVING_SLOTS) * n_segments + scale_chunk // _X4) * _BLOCKS * group_cols + (
                    block % _BLOCKS
                ) * group_cols
                previous = matmul
                matmul = nisa.nc_matmul_mx(
                    dst=gate_up_ps[:, gate_or_up * _PSUM_COLS : gate_or_up * _PSUM_COLS + group_cols],
                    stationary=hidden_qtz[:, block * padded : (block + 1) * padded],
                    moving=gate_up_moving.view(nl.float4_e2m1fn_x4)[:, weight_col : weight_col + group_cols],
                    stationary_scale=hidden_scale[:, block * padded : (block + 1) * padded],
                    moving_scale=gate_up_moving_s.view(nl.uint8)[
                        _X4 * (scale_chunk % _X4) :, scale_col : scale_col + group_cols
                    ],
                    accumulate=block > 0,
                    name=f"dense_mm_{('gate', 'up')[gate_or_up]}_g{group}_b{block}",
                )
                if previous is not None:
                    matmul.depends_on(previous, kind=nisa.dependency.schedule)
        return matmul

    def situ(group):
        """gated[t, (q, c, p)] = lb*tanh(u/lb) * (beta*lb*tanh(g/beta)) * sigmoid(g) / lb, the order of the
        selective-load body."""
        gate_up_ps = gate_up_ps_by_parity[group % 2]
        col = (group % 2) * group_cols
        gate_ps, up_ps = gate_up_ps[:, :group_cols], gate_up_ps[:, _PSUM_COLS : _PSUM_COLS + group_cols]
        gate_term, up_term = gate_act[:, col : col + group_cols], situ_up[:, col : col + group_cols]
        sigmoid_term = situ_sigmoid[:, col : col + group_cols]
        nisa.activation(dst=gate_term, op=nl.tanh, data=gate_ps, scale=1.0 / beta, name=f"dense_tanh_gate_g{group}")
        nisa.activation(dst=up_term, op=nl.tanh, data=up_ps, scale=1.0 / linear_beta, name=f"dense_tanh_up_g{group}")
        nisa.activation(dst=sigmoid_term, op=nl.sigmoid, data=gate_ps, name=f"dense_sigmoid_g{group}")
        nisa.scalar_tensor_tensor(
            dst=gate_term,
            data=gate_term,
            op0=nl.multiply,
            operand0=beta * linear_beta,
            op1=nl.multiply,
            operand1=sigmoid_term,
            name=f"dense_situ_gate_g{group}",
        )
        pairs = [[2 * group_cols, padded], [down_rows, _X4], [I, _GROUP], [1, down_rows]]
        nisa.tensor_tensor(
            dst=gated.ap(
                pattern=[[2 * _X4 * _PMAX, padded], [_PMAX, _X4], [_MX_ROWS, _GROUP], [1, down_rows]],
                offset=(group % 2) * _X4 * _PMAX,
            ),
            data1=situ_up.ap(pattern=pairs, offset=col),
            data2=gate_act.ap(pattern=pairs, offset=col),
            op=nl.multiply,
            name=f"dense_mul_up_g{group}",
        )

    def finish(group, after):
        """The down stationary of group: transposes, times the dense affinities, one MXFP8 term."""
        gated_ps = gated_ps_by_parity[group % 2]
        transpose = after
        for q in range(_X4):
            previous = transpose
            transpose = nisa.nc_transpose(
                dst=gated_ps[:, q * padded : (q + 1) * padded],
                data=gated[:, ((group % 2) * _X4 + q) * _PMAX : ((group % 2) * _X4 + q + 1) * _PMAX],
                name=f"dense_transpose_g{group}_q{q}",
            )
            if previous is not None:
                transpose.depends_on(previous, kind=nisa.dependency.schedule)
        slot = (group % 2) * padded
        nisa.tensor_tensor(
            dst=qmx_in[:, slot * _X4 : (slot + padded) * _X4],
            data1=gated_ps.ap(pattern=[[_X4 * padded, _PMAX], [1, padded], [padded, _X4]], offset=0),  # [m, (t, q)]
            data2=dense_affinity.ap(pattern=[[n_groups * padded, _PMAX], [1, padded], [0, _X4]], offset=group * padded),
            op=nl.multiply,
            name=f"dense_qmx_in_g{group}",
        )
        nisa.quantize_mx(
            dst=inter[:, slot : slot + padded],
            src=qmx_in[:, slot * _X4 : (slot + padded) * _X4],
            dst_scale=inter_s[:, slot : slot + padded],
            name=f"dense_qmx_g{group}",
        )
        return transpose

    def down_projection(group, after):
        slot = (group % 2) * padded
        weights = (group % _DOWN_SLOTS) * H
        scale_row = _X4 * (group % _X4)
        matmul = after
        for chunk in range(n_down_chunks):
            col = chunk * _DOWN_CHUNK
            previous = matmul
            matmul = nisa.nc_matmul_mx(
                dst=out_ps[:, col : col + _DOWN_CHUNK],
                stationary=inter[:, slot : slot + padded],
                moving=down_w[:, weights + col : weights + col + _DOWN_CHUNK],
                stationary_scale=inter_s[:, slot : slot + padded],
                moving_scale=down_s[scale_row:, col : col + _DOWN_CHUNK],
                accumulate=group > 0,
                name=f"dense_mm_down_g{group}_c{chunk}",
            )
            if previous is not None:
                matmul.depends_on(previous, kind=nisa.dependency.schedule)
        return matmul

    # The PE pipeline: gate_up(0) | gate_up(g + 1)[:h] finish(g) gate_up(g + 1)[h:] down(g) | ... | finish, down.
    blocks = list(range(n_h512))
    half = n_h512 // 2
    relayout(0)
    last = gate_up_projection(0, blocks, after=None)
    situ(0)
    for group in range(n_groups):
        nxt = group + 1
        if nxt < n_groups:
            relayout(nxt)
            if nxt + _ROW_SLOTS - 1 < n_groups:
                load_rows(nxt + _ROW_SLOTS - 1)
            last = gate_up_projection(nxt, blocks[:half], after=last)
        last = finish(group, after=last)
        if nxt < n_groups:
            last = gate_up_projection(nxt, blocks[half:], after=last)
            situ(nxt)
        last = down_projection(group, after=last)
        if group + _DOWN_SLOTS < n_groups:
            load_down(group + _DOWN_SLOTS)

    # The output: evict each 512-column chunk (DVE and ACT) and store it.
    for chunk in range(n_down_chunks):
        cols = slice(chunk * _DOWN_CHUNK, (chunk + 1) * _DOWN_CHUNK)
        if chunk % 3 == 1:
            nisa.activation(dst=out_sb[:, cols], op=nl.copy, data=out_ps[:, cols], name=f"dense_evict_c{chunk}")
        else:
            nisa.tensor_copy(dst=out_sb[:, cols], src=out_ps[:, cols], engine=dve, name=f"dense_evict_c{chunk}")
        nisa.dma_copy(
            dst=output[output_row : output_row + T, cols],
            src=out_sb[0:T, cols],
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"dense_store_c{chunk}",
        )
    sbm.close_scope()
    return output


def _quantize_latent_blocks(
    sbm: SbufManager, latent: nl.NkiTensor, hidden_qtz: nl.NkiTensor, hidden_scale: nl.NkiTensor, padded: int, suffix
):
    """Quantize the natural [T, H] latent (T <= 128) to MXFP8 [p, (hb, t)]: H = 512*hb + 4*p + q of token t."""
    nisa, nl = _Suffixed(_ISA, suffix), _Suffixed(_LANGUAGE, suffix, functions=("ndarray",))
    T, H = latent.shape
    n_h512 = H // _H512
    depth = len(sbm.heap)
    swizzled = sbm.alloc_heap((_PMAX, n_h512 * padded * _X4), nl.bfloat16, name="dense_latent_swizzled" + suffix)
    if padded > T:
        nisa.memset(dst=swizzled, value=0.0, name="dense_latent_pad")
    source = latent
    if latent.buffer != nl.sbuf:
        source = sbm.alloc_heap((T, H), nl.bfloat16, name="dense_latent" + suffix)
        nisa.dma_copy(
            dst=source, src=latent, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync, name="dense_load_latent"
        )
    for block in range(n_h512):
        transposed = nl.ndarray(
            (_PMAX, _X4 * padded),
            dtype=nl.bfloat16,
            buffer=nl.psum,
            address=(0, (block % 2) * _PSUM_BANK_BYTES),
            name=f"dense_latent_ps{block}",
        )
        for q in range(_X4):
            nisa.nc_transpose(
                dst=transposed[:, q * padded : q * padded + T],
                data=source.ap(pattern=[[H, T], [_X4, _PMAX]], offset=block * _H512 + q),
                name=f"dense_latent_transpose_b{block}_q{q}",
            )
        nisa.tensor_copy(
            dst=swizzled.ap(pattern=[[n_h512 * padded * _X4, _PMAX], [_X4, T], [1, _X4]], offset=block * padded * _X4),
            src=transposed.ap(pattern=[[_X4 * padded, _PMAX], [1, T], [padded, _X4]], offset=0),
            engine=nisa.vector_engine,
            name=f"dense_latent_swizzle_b{block}",
        )
    nisa.quantize_mx(dst=hidden_qtz, src=swizzled, dst_scale=hidden_scale, name="dense_latent_quantize")
    while len(sbm.heap) > depth:
        sbm.pop_heap()


def _dense_affinities(
    sbm: SbufManager,
    expert_index: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    dense_affinity: nl.NkiTensor,
    padded: int,
    n_groups: int,
    suffix: str,
):
    """dense_affinity[m, (g, t)] = the affinity of token t for expert 16g + m/8 (zero when it is not routed).

    The routing goes to every partition (``_broadcast_routing``). Tile j of 128 experts: a compare with the expert
    id of each partition and a sum over k give A^T[e, t] on partition e % 128. A matmul with a 0/1 stationary per
    group then moves expert 16g + c to partitions 8c .. 8c + 7. Every sum has one nonzero term, so it is exact.
    """
    nisa, nl = _Suffixed(_ISA, suffix), _Suffixed(_LANGUAGE, suffix, functions=("ndarray",))
    dve = nisa.vector_engine
    depth = len(sbm.heap)

    def alloc_heap(shape, dtype, name):
        return sbm.alloc_heap(shape, dtype, name=name + suffix)

    n_tiles = n_groups * _GROUP // _PMAX
    groups_per_tile = _PMAX // _GROUP
    index_all = alloc_heap((_PMAX, padded * _TOP_K), nl.int32, "dense_index_all")
    affinity_all = alloc_heap((_PMAX, padded * _TOP_K), nl.bfloat16, "dense_affinity_all")
    _broadcast_routing(sbm, expert_index, expert_affinities, index_all, affinity_all, padded, suffix + "_d")
    expert_id = alloc_heap((_PMAX, n_tiles), nl.float32, "dense_expert_id")
    nisa.iota(dst=expert_id, pattern=[[_PMAX, n_tiles]], offset=0, channel_multiplier=1, name="dense_iota_expert_id")
    hot = alloc_heap((_PMAX, padded * _TOP_K), nl.bfloat16, "dense_hot")
    transposed = alloc_heap((_PMAX, n_tiles * padded), nl.bfloat16, "dense_affinity_t")
    for tile in range(n_tiles):
        nisa.scalar_tensor_tensor(
            dst=hot,
            data=index_all,
            op0=nl.equal,
            operand0=expert_id[:, tile : tile + 1],
            op1=nl.multiply,
            operand1=affinity_all,
            name=f"dense_hot{tile}",
        )
        nisa.tensor_reduce(
            dst=transposed[:, tile * padded : (tile + 1) * padded],
            op=nl.add,
            data=hot.ap(pattern=[[padded * _TOP_K, _PMAX], [_TOP_K, padded], [1, _TOP_K]], offset=0),
            axis=[2],
            name=f"dense_affinity_t{tile}",
        )
    # select[e, (r, m)] = 1 if e == 16r + m/8
    select_val = alloc_heap((_PMAX, groups_per_tile * _PMAX), nl.float32, "dense_select_val")
    select = alloc_heap((_PMAX, groups_per_tile * _PMAX), nl.bfloat16, "dense_select")
    nisa.iota(
        dst=select_val,
        pattern=[[-_GROUP, groups_per_tile], [-1, _GROUP], [0, _MX_ROWS]],
        offset=0,
        channel_multiplier=1,
        name="dense_iota_select",
    )
    nisa.tensor_scalar(dst=select, data=select_val, op0=nl.equal, operand0=0.0, engine=dve, name="dense_select_eq")
    batch = _BANK_FP32 // padded  # groups per PSUM bank
    for first in range(0, n_groups, batch):
        groups = range(first, min(n_groups, first + batch))
        moved = nl.ndarray(
            (_PMAX, _BANK_FP32),
            dtype=nl.float32,
            buffer=nl.psum,
            address=(0, (6 + (first // batch) % 2) * _PSUM_BANK_BYTES),
            name=f"dense_affinity_ps{first}",
        )
        for group in groups:
            tile, r = divmod(group, groups_per_tile)
            nisa.nc_matmul(
                dst=moved[:, (group - first) * padded : (group - first + 1) * padded],
                stationary=select[:, r * _PMAX : (r + 1) * _PMAX],
                moving=transposed[:, tile * padded : (tile + 1) * padded],
                name=f"dense_mm_affinity_g{group}",
            )
        nisa.tensor_copy(
            dst=dense_affinity[:, first * padded : (first + len(groups)) * padded],
            src=moved[:, 0 : len(groups) * padded],
            engine=dve,
            name=f"dense_copy_affinity{first}",
        )
    while len(sbm.heap) > depth:
        sbm.pop_heap()
