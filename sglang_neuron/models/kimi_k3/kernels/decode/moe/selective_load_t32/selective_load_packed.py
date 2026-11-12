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

"""Packed selective-load expert MLP of the kimi_t32 fused MoE block (T = 32 tokens, K = 16, H = 3584, I = 24,
E = 896): groups of 16 routed experts, with optional duplicate-expert skipping (dedupe).

The docstring of group_body explains the algorithm, the layouts and the design choices.
"""

import inspect
from typing import Callable, Generator, List, Optional, Sequence, Tuple

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil

_ISA, _LANGUAGE = nisa, nl


class _Suffixed:
    """A view of a module whose functions (or only the listed ones) append ``suffix`` to their ``name`` argument.

    Op and tensor names must be unique in a kernel, so a kernel that calls the body more than once gives each call its
    own name suffix (name_suffix of group_body). A copy of the kimi subpackage helper."""

    def __init__(self, module, suffix: str, functions: tuple = ()):
        self._module, self._suffix, self._functions = module, suffix, functions

    def __getattr__(self, attr):
        value = getattr(self._module, attr)
        if attr in self._functions or (not self._functions and inspect.isfunction(value)):

            def call(*args, **kwargs):
                if kwargs.get("name") != None:
                    kwargs["name"] = kwargs["name"] + self._suffix
                return value(*args, **kwargs)

            return call
        return value


def named_modules(name_suffix: Optional[str]):
    """(nisa, nl) for one call: the modules themselves without a suffix, else views that append name_suffix to every
    op and tensor name."""
    if not name_suffix:
        return _ISA, _LANGUAGE
    return _Suffixed(_ISA, name_suffix), _Suffixed(_LANGUAGE, name_suffix, functions=("ndarray",))


def cached_alloc(sbm: SbufManager, name_suffix: Optional[str], buffer_cache: Optional[dict], heap: bool = False):
    """sbm.alloc_stack (or alloc_heap) with name_suffix on the tensor name; with buffer_cache, the tensor of an earlier
    call with the same name, shape, dtype and address (the calls then share the DMA destinations)."""
    base = sbm.alloc_heap if heap else sbm.alloc_stack
    if not name_suffix and buffer_cache == None:
        return base

    def alloc(shape, dtype, name=None, **kwargs) -> nl.NkiTensor:
        tile = base(shape, dtype, name=None if name == None else name + (name_suffix or ""), **kwargs)
        if buffer_cache == None:
            return tile
        address = sbm.get_heap_curr_addr() if heap else sbm.get_stack_curr_addr()
        return buffer_cache.setdefault((name, tuple(shape), str(dtype), address, heap), tile)

    return alloc


_P_MAX = 128
_QUADRANT = 32
_X4 = 4
_MX_ROWS = 8
_MX_GROUP = 32  # values of one checkpoint MX block
_GROUP = 16  # slots (experts) per group
_BLOCKS = 4  # H512 blocks per gate/up scale chunk
_PSUM_BANK_BYTES = 2048
_PSUM_COLS = 512  # bf16 gate/up PSUM tile: gate at column 0, up at column _PSUM_COLS
_DOWN_CHUNK = 512  # H columns per down-projection matmul
_FOLD_CHUNK = 512  # H columns per fold matmul (one fp32 PSUM bank)
_FINISH_AFTER_UNITS = 2  # the transposes of group i - 1 run after this many gate/up units of group i
_ACT_RELAYOUT_BLOCKS = 4  # ACT copies the first H512 blocks of each re-layout copy, DVE the others
# int16 0xFFF0: an AND with the bf16 bits keeps the sign, the exponent and the top 3 mantissa bits, so the result
# (4 significant bits) is exactly representable in MXFP8 e4m3.
_E4M3_BF16_MASK = -16
# Expert id of an empty dedupe position before it is mapped to expert 0 (fp16 holds id - _PAD_ID exactly for
# 0 <= id < _PAD_ID).
_PAD_ID = 2048
# Offset that puts the position of an expert without a slot out of range in the one-hot compare (dedupe).
_NO_POSITION = 4096.0
# skip_empty_slots: the expert id of an empty dedupe position in the gate/up and down gathers, past every weight row
# (the gather skips it); 2^20 experts, so the row indices stay below 2^31.
_SKIP_ID = float(1 << 20)
_RW_PART_COLS = 168  # dedupe: columns of each position-matmul part (fp32 PSUM, bank 7 from 1024 B)


def run_phases(phases: Generator) -> nl.NkiTensor:
    """Run the phases of a body generator to its end and return its output."""
    while True:
        try:
            next(phases)
        except StopIteration as stop:
            return stop.value


def group_body(*args, **kwargs) -> nl.NkiTensor:
    """The routed-expert MLP in one phase: group_body_phases (same arguments) run to its end."""
    kernel_assert(not kwargs.get("two_phase", False), "group_body runs one phase: use group_body_phases")
    return run_phases(group_body_phases(*args, **kwargs))


def group_body_phases(
    sbm: SbufManager,
    hidden_qtz: nl.NkiTensor,
    hidden_scale: nl.NkiTensor,
    expert_gate_up_packed: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    index_tk: nl.NkiTensor,
    affinity_tk: nl.NkiTensor,
    router_logits: nl.NkiTensor,
    router_top: nl.NkiTensor,
    router_top_recip: nl.NkiTensor,
    after_norm: Optional[nisa.NkiInstruction] = None,
    dedupe_experts: Optional[str] = None,
    sp_group_period: int = 0,
    gate_up_slots: int = 4,
    moving_slots: int = 3,
    down_slots: int = 4,
    dedupe_static_groups: int = 1,
    dedupe_block_groups: Optional[Sequence[int]] = None,
    dedupe_prefetch: bool = False,
    situ: Optional[Tuple[float, float]] = None,
    sigmoid_from_tanh: bool = False,
    affinity_dtype: str = "fp32",
    router_scores: Optional[nl.NkiTensor] = None,
    dedupe_early_indices: bool = True,
    group0_queues: str = "sp_act",
    dedupe_block_mode: str = "binary",
    expert_offset: int = 0,
    n_experts: Optional[int] = None,
    output: Optional[nl.NkiTensor] = None,
    output_row: int = 0,
    gate_up_pair_gathers: bool = False,
    evict_split: str = "rows",
    name_suffix: Optional[str] = None,
    buffer_cache: Optional[dict] = None,
    quantize_first: bool = False,
    merged_down_gathers: bool = True,
    dedupe_late_trips: bool = False,
    dedupe_fold_first: bool = False,
    setup_memsets: str = "pool",
    dedupe_onehot_parts: bool = False,
    valid_tokens: Optional[int] = None,
    skip_down_padding: bool = False,
    dedupe_order: str = "id",
    down_lag: int = 1,
    skip_empty_slots: bool = False,
    weights_arena: bool = False,
    dedupe_early_first: bool = False,
    scales_after_gate_up: bool = False,
    pair_head_in_ring: bool = False,
    act_slots: int = 2,
    fold_dma: bool = False,
    skip_scale_duplicates: bool = False,
    early_ids_bank: int = 7,
    down_psum_dtype: str = "bf16",
    rank_info: Optional[nl.NkiTensor] = None,
    stationary_slots: int = 3,
    head_finish_first: bool = False,
    dedupe_token_rank: str = "tiles",
    dedupe_index_first: bool = False,
    late_setup: Optional[Callable] = None,
    dedupe_precomputed: Optional[Tuple[nl.NkiTensor, nl.NkiTensor]] = None,
    prefetch_groups: int = 0,
    prefetch_load: str = "gather",
    prefetch_indices: Optional[nl.NkiTensor] = None,
    fold_store_order: str = "columns",
    fold_store_cols: int = 512,
    dedupe_trips_after_stores: bool = False,
    two_phase: bool = False,
    two_phase_latent_first: bool = False,
) -> Generator[str, Optional[nisa.NkiInstruction], nl.NkiTensor]:
    """Routed-expert MLP (SiLU(gate) * up, no bias, POST_SCALE affinities) of the kimi_t32 fused block.

    A generator (group_body runs it to its end). With two_phase it yields "routing" once, after the work that needs
    only the routing: the setup, the compaction, the gather indices, the loads of the head pipeline's ring and the
    re-layout of group 0. Nothing before the yield reads hidden_qtz / hidden_scale, so the caller writes them (the
    latent quantization) and then resumes the generator (send: the last instruction of the quantization, or None).
    The generator returns the output.

    Optimized for decode with T = 32 tokens on Trn3 (LNC1), MXFP4 expert weights in the packed layouts of the kimi
    subpackage, a small intermediate size (I = 24) and E = 896.

    Dimensions:
        T: tokens. K: experts per token (16). N = T*K routed pairs, n = t*K + k. E: experts. H: hidden size.
        I: intermediate size of one expert (I <= 32). n_h512 = H/512, hb: H512 block. q: x4 lane. p: row of an
        expert's 8-row down MX block, i = q*(I/4) + p. a, m: SBUF partition.
        Slot: one expert with an affinity per token (0 for the tokens that do not route to it). Group: 16 slots,
        16G .. 16G + 15 for group G. P = N slot positions.

    Args:
        sbm (SbufManager): SBUF manager; the body allocates its buffers in its own stack scope.
        hidden_qtz, hidden_scale (nl.NkiTensor): float8_e4m3fn_x4 / uint8 [128, n_h512 * T] @SBUF, the MXFP8 RMSNorm
            output: element [p, hb*T + t] packs the 4 values of hidden index q*H/4 + 128*hb + p of token t.
        expert_gate_up_packed (nl.NkiTensor): uint8 [E, 128, 4*I*(H/512 + S)] @HBM, the packed gate/up rows of the kimi
            subpackage (weights [gate_or_up][hb][I], then S scale segments).
        expert_down_weights (nl.NkiTensor): float4_e2m1fn_x4 [E, 8, H] @HBM, the 8 MX rows of each expert. Or the
            compact [E, I/4, H]: only the rows that the down projection uses (needs skip_down_padding). The two
            forms give the same outputs.
        expert_down_weights_scale (nl.NkiTensor): uint8 [E, H] @HBM, the MX scale row of each expert. Or uint8
            [E, 2, H]: two down scale groups (each down row on the scale of its own checkpoint block; see
            rank_info). Needs down_psum_dtype="fp32".
        index_tk (nl.NkiTensor): uint32 [T, K] @SBUF, the router's expert index.
        affinity_tk (nl.NkiTensor): float32 [T, K] @SBUF, the normalized top-K affinities.
        router_logits (nl.NkiTensor): float32 [T, E] @PSUM, the router logits (+ bias). Dedupe only.
        router_top (nl.NkiTensor): float32 [T, K] @SBUF, the top-K logits in descending order. Dedupe only.
        router_top_recip (nl.NkiTensor): float32 [T, 1] @SBUF, 1 / sum over k of exp(top[t, k] - top[t, 0]).
            Dedupe only.
        after_norm (Optional[nisa.NkiInstruction]): the constant setup is scheduled after it (the end of the
            RMSNorm), so it does not delay the RMSNorm.
        dedupe_experts (Optional[str]): None (default, the baseline): slot n = pair n, so group G holds the 16
            experts of token G (N/16 groups). "dynamic": one slot per distinct expert. The head group holds the 16
            experts of token 0 (distinct by construction; their loads start right after the router). The other
            U - 16 distinct experts, in expert-id order, fill the groups of the 0/1-trip blocks of
            dedupe_block_groups (nl.fori_loop): the blocks whose group counts add up to ceil((U - 16) / 16) run once,
            the others not at all. The empty slots of the last executed group load expert 0 with zero affinities.
        sp_group_period (int): 0 (default): the Pool gathers load the gate/up rows of every group except group 0 of
            the baseline (16 HWDGE DMAs on the SP and ACT queues, no gather index needed). p > 0: in every pipeline,
            group i with i % p == p - 1 loads its gate/up rows with 16 SP HWDGE DMAs instead (the Pool gathers limit
            the pipeline; SP issues one expert at a time).
        gate_up_slots, moving_slots, down_slots (int): ring slots of the packed gate/up rows (4), of the re-laid
            moving weights (3) and of the down weights (4).
        dedupe_static_groups (int): dedupe: groups of the static head (no loop): token 0's group, then the first
            dedupe_static_groups - 1 groups of distinct experts. Default 1 (token 0 only). A larger head runs fewer
            0/1-trip blocks at the likely U (each executed block refills and drains the pipeline), but computes empty
            groups when U < 16 * dedupe_static_groups.
        dedupe_block_groups (Optional[Sequence[int]]): group counts of the dedupe blocks, decreasing, adding up to
            N/16 - dedupe_static_groups, each at most 1 + the sum of the smaller counts (for example powers of two).
            None (default): the powers of two below N/16 - dedupe_static_groups + 1: 16, 8, 4, 2, 1 for
            dedupe_static_groups = 1, 8, 4, 2, 1 for 17 (other values need an explicit list).
        dedupe_prefetch (bool): dedupe: issue the first loads of each block (scale tile 0, the gate/up rows of its
            first group, its first down unit) before its loop, so its pipeline does not start with a load. Default
            False: with a compiler that drains the DMA queues at every loop entry, a block that does not run then
            waits for these loads too (more time per skipped block, measured on the stock compiler).
        situ (Optional[Tuple[float, float]]): None (default): SiLU(gate) * up. (beta, linear_beta): the SiTU
            activation beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta) in fp32 (Step 3.4): ACT computes
            tanh(g/beta), tanh(u/linear_beta) and sigmoid(g) into fp32 tiles, DVE forms tanh(g/beta) * (beta*linear_beta)
            * sigmoid(g), multiplies it by the affinity and then by tanh(u/linear_beta) into gated (bf16).
        sigmoid_from_tanh (bool): with situ, sigmoid(g) = (1 + tanh(g/2)) / 2 (ACT: tanh(g/2); DVE: (1 + tanh(g/2)) *
            tanh(g/beta), the affinity, then times linear_beta*beta/2 * tanh(u/linear_beta)), so ACT uses only the
            tanh table. Default False.
        affinity_dtype (str): "fp32" (default) or "bf16": aff_cols is bf16, so each affinity is rounded to bf16 once
            (the rounding of the latent-MoE semantics).
        router_scores (Optional[nl.NkiTensor]): float32 [T, E] @SBUF, the sigmoid score of each selected expert and 0
            elsewhere (the router with router_score="sigmoid"). Dedupe: the compaction reads it instead of the
            softmax terms of router_logits; router_top_recip is then 1 / (the sum of the selected scores + 1e-20).
        dedupe_early_indices (bool): dedupe with dedupe_static_groups >= 5: the gather indices of head groups 1-4 (one
            down-scale tile) are formed right after the first part of the position matmuls, before the other parts, so
            the gate/up gather of group 1 does not wait for the whole compaction. Default True. The outputs are the
            same.
        group0_queues (str): the gate/up loads of the baseline's group 0. "sp_act" (default): 16 HWDGE loads on SP and
            ACT in turn. "sp": all on SP, so the ACT loads do not wait behind the re-layouts of groups 1 and 2 in the
            ACT instruction stream. "pool": one Pool gather like the other groups (no HWDGE load with a scalar expert
            id). The outputs are the same.
        dedupe_block_mode (str): dedupe: how the blocks place the experts after the static groups. "binary" (default):
            block b runs if the groups that are still needed are at least its size (largest block first), so the
            executed blocks add up to the needed groups exactly. "alternatives": dedupe_block_groups is an increasing
            list whose last entry is N/16 - dedupe_static_groups; exactly one block runs, the smallest that holds the
            needed groups (none if the static groups hold all experts). All blocks start at the same position group,
            so one executed block costs one pipeline fill and drain, and its last groups can be empty.
        expert_offset (int): the weight row of expert 0: expert e reads gate/up rows 128*(expert_offset + e) ..,
            down rows R*(expert_offset + e) .. (R = 8, or I/4 for the compact down weights) and down-scale row
            expert_offset + e. With stacked weights of L layers
            ([L*E, ...] or [L, E, ...]), layer l uses expert_offset = l*E. The offset goes into the dynamic index of
            every weight DMA (no DMA has a static offset). Default 0.
        n_experts (Optional[int]): E, the experts of one layer. None (default): the first dimension of the weights
            (one layer), or their second dimension for [L, E, ...] stacks.
        output (Optional[nl.NkiTensor]): bfloat16 HBM destination with H columns and at least output_row + T rows. None
            (default): a new [T, H] tensor.
        output_row (int): the row of output for token 0. Default 0.
        skip_down_padding (bool): the down gathers skip the padding rows of each expert's 8-row MX block (rows
            I/4 .. 7, 2 of 8 at I = 24): their gather indices are out of bounds, so the gather does not move them
            (8.8% fewer bytes per group at I = 24). Their SBUF rows keep old values; the down stationary rows they
            meet are zero and the FP4 values are finite, so the products are zero and the outputs are the same.
            The compact down weights [E, I/4, H] need it. Default False.
        dedupe_order (str): dedupe with dedupe_block_mode="alternatives": the order of the block positions. "id"
            (default): ascending expert id. "token": by the first token (1 .. T-1) that routes to the expert, then by
            id. The down projection adds each position group into bf16 PSUM, so a token's running sum is rounded once
            per group that holds one of its experts; in token order its experts fill fewer groups. Same experts, the
            outputs differ in the last bits.
        dedupe_token_rank (str): dedupe_order="token": how the compaction ranks the experts in the token order (Step
            4.3t of compact_experts). "tiles" (default): a scan per token over the expert tiles, then the experts first
            seen at earlier tokens (a third matmul and two more DVE ops). "flat": one scan over the (token, tile) axis
            and two matmuls (the second with the constant -(e' >= e)), so the head's critical path has 5 DVE ops less.
            The positions are the same, so the outputs are the same.
        dedupe_index_first (bool): dedupe with dedupe_static_groups > 1 and dedupe_block_mode="alternatives": the
            program order puts the path to the first gather indices first. The constants of the later steps (the
            zero memsets of gated / qmx_in, the transpose identity, the fold selector), the head affinities, the block
            plan, the affinity eviction of the first position part and late_setup follow the gather indices of head
            groups 1-4 (the early indices), and the position part 0 evicts its ids before its affinities. With the
            pair and merged gathers, the gather indices skip the single-group indices that no gather reads, and the
            pair indices come first. Default False. The outputs are the same.
        late_setup (Optional[Callable]): with dedupe_index_first: late_setup(after) emits the caller's work that only
            the affinities need (for example the normalization of the selected scores into router_top_recip),
            right after the early indices (after: their last instruction). Else not used.
        dedupe_precomputed (Optional[Tuple[nl.NkiTensor, nl.NkiTensor]]): dedupe with dedupe_static_groups > 1 (not
            with dedupe_index_first or sp_group_period): the results of the compaction and of the gather indices,
            computed by the caller: (indices int32 [128, C] @HBM, affinities [T, P] @HBM in the dtype of
            affinity_dtype). The body loads them (HWDGE on SP, or ACT with after_norm) and skips the compaction, so the gathers wait only for
            these loads: a measurement of the compaction cost. The columns of indices: gi_gu (P), gi_gu2 (P), gi_dn
            (P / 16), gi_dn2 (P / 8), gi_sc (one per down-scale tile), then the trip counts of the blocks (row 0).
            Optional third and fourth items: a [P, 1] SBUF tensor that the first Pool gather waits for (the input of
            a caller's op: two Pool ops add 0 * its first value to the first gather index), and pool_cols > 0: a
            stand-in Pool op of pool_cols fp32 columns on its P partitions between them (a cost a column).
            Default None.
        prefetch_groups (int): dedupe (alternatives, token order with dedupe_token_rank="flat"), a measurement of a
            hot-expert prefetch: the N = 16 * prefetch_groups experts expert_offset + 0 .. N - 1 (the hot experts of a
            swizzled layer) are positions 0 .. N - 1 (the first groups of the head, loaded at the start without
            waiting for the routing). The compaction places only the selected experts with id >= N, in token order
            over all tokens (no token-0 head), at positions N ..; the positions grow to N + T * K. The head pipeline
            starts with a single group 0 (prefetch_groups odd) or a pair (even), and dedupe_static_groups must have
            the parity of prefetch_groups. A hot expert that no token selects adds zero (affinity 0). Default 0.
        prefetch_load (str): the loads of the prefetch groups. "gather" (default): Pool gathers with static indices
            (built in the setup). "contiguous": the gate/up rows and the down rows of the hot experts are contiguous
            in HBM, so each unit is one HWDGE DMA (the down padding rows included) on the SP queue (ACT in the fused
            block, where SP carries the router weights); the down-scale tiles stay Pool gathers.
        prefetch_indices (Optional[nl.NkiTensor]): with prefetch_groups, the gather indices of the prefetch groups
            (a constant of the layer and expert_offset): int32 [128, C] @HBM in the column layout of
            dedupe_precomputed (only the columns of the prefetch groups and their scale tiles are read; HWDGE loads at
            the start). None (default): the setup builds them (DVE).
        fold_store_order (str): with fold_dma, the order of the store DMAs of the static output (one queue, in
            order). "columns" (default): each store follows its staging copy, in column order. "ready": all staging
            copies first, then the stores in the order that the copies end in a model of the copy engines (DVE copies
            the even chunks, ACT the odd ones, each engine at its own fixed copy time). The outputs are the same.
        fold_store_cols (int): with fold_dma, the columns of one store DMA of the static output: 512 (default) or
            a multiple of 512 (larger DMA rows; the store waits for the copies of all its chunks). The outputs are
            the same.
        two_phase (bool): dedupe (dynamic, dedupe_static_groups > 1): yield "routing" before the first read of
            hidden_qtz / hidden_scale (see above). Between the phases the body's stack scope of sbm stays open and
            its PSUM banks 6-7 hold the compaction; the caller's tiles must not overlap the stack scope, and the
            latent quantization may use PSUM bank 0 (the gate/up banks 0-1 are free until the group loop). Default
            False. The outputs are the same.
        two_phase_latent_first (bool): two_phase: the DVE copies of the re-layout of head group 1 are scheduled after
            the instruction sent at the resume (the latent quantization), so the scheduler does not put the latent
            swizzle and quantization behind them. Default False. The outputs are the same.
        dedupe_trips_after_stores (bool): dedupe with fold_dma: the trip-count tile of the blocks is written by an
            add of 0 (the compiler can remove the plain copy of dedupe_late_trips, and its schedule dependency with
            it), scheduled after the last store DMA of the static output, so the register loads on SP follow the
            store DMAs. Default False. The outputs are the same.
        down_lag (int): the PE order of the group pipeline. 1 (default): gate_up(i) (2 units), the transposes of
            group i - 1, the rest of gate_up(i), down(i - 1). 2: gate_up(i), the transposes of group i - 1, down(i - 2).
            With 1, the PE waits for the SwiGLU of group i - 1 before its transposes and for its MXFP8 quantization
            before its down projection (ACT and DVE need time after the end of gate_up(i - 1)). With 2, the
            gate/up of the next group runs in that time. The rings do not change. The outputs are the same.
        skip_empty_slots (bool): dedupe: the gate/up and down gathers skip the empty positions (after the U distinct
            experts; without the flag they load expert 0): their gather indices are out of bounds. Their SBUF rows
            keep the values of an earlier group (FP4 weights, so finite) and their affinities are zero, so they add
            zero. The scale segments of the gate/up row ring are set to zero once in the setup, so a ring slot that no
            gather wrote has finite scales. The down scales still load expert 0. Default False. The outputs are the
            same.
        weights_arena (bool): the gate/up row ring, the down weights and the down-scale tiles are views of one SBUF
            tensor, so every Pool gather writes one tensor. A program with many calls then gets one DGE table entry per
            call with the merged down gathers (a multi-column gather adds one entry per call for each destination
            tensor). Default False. The outputs are the same.
        dedupe_early_first (bool): dedupe with dedupe_onehot_parts and dedupe_early_indices: the one-hot tiles of the
            later position parts are scheduled after the gate/up gather index of head groups 1-4, so DVE forms that
            index first (the gate/up gather of group 1 waits for it). Default False. The outputs are the same.
        scales_after_gate_up (bool): the first down-scale gather of each pipeline goes after the gate/up gathers of its
            groups 0 and 1 in the Pool stream (default: before the gate/up gather of group 0), so the first gate/up
            rows load earlier; the down projection of group 0 needs the scales only later. Default False. The outputs
            are the same.
        pair_head_in_ring (bool): dedupe with gate_up_pair_gathers: the head's group 0 loads into ring slot R - 2
            instead of an extra slot R (the pair of groups R - 1 and R that uses this slot is gathered after the
            re-layout of group 0). With gate_up_slots=6 and moving_slots=2 the ring holds three pairs within the SBUF
            of four single slots and three moving slots plus one slot. Default False. The outputs are the same.
        act_slots (int): ring slots of the fp32 activation temporaries (gate_act and the SiTU tiles, 3,072 bytes per
            partition and slot). 2 (default) or 1: the activation of group i + 1 starts after gate_up(i + 1), so the
            activation of group i is done with one slot (4,608 bytes less SBUF with SiTU). The outputs are the same.
        fold_dma (bool): the output stores add the hi and lo rows of each token in the DMA (dma_compute: fp32, then
            one rounding to bf16), instead of a fold matmul, a copy to out_sb and a plain store. Removes the fold
            matmuls and the out_sb copies from the end of the call (not in the dedupe blocks: the compiler takes no DMA
            compute in a loop body). Default False. The outputs are the same when the
            DMA rounds like the PSUM copy (round to nearest even).
        skip_scale_duplicates (bool): the down-scale gathers skip partitions 32q + 16 .. 32q + 31 of each scale tile
            (the MX scale reads use partitions 32q + 4l + j, l and j < 4, only): their indices are out of bounds, so
            each scale gather moves half the bytes. Default False. The outputs are the same.
        early_ids_bank (int): dedupe_early_indices: the PSUM bank of the broadcast of the early position ids. 7
            (default): with the compaction tiles, whose position matmuls accumulate in the same bank. 6: from byte 512
            of bank 6 (after the transposes), where no matmul accumulates. The fused block uses 6 with the softmax
            router: there the scheduler can put the broadcast inside an accumulation of bank 7, which the compiler
            rejects (NCC_ISCH713, accumulation group interleaving). The outputs are the same.
        down_psum_dtype (str): the dtype of out_ps, the down-projection output that PSUM accumulates over the
            groups. "bf16" (default): out_ps [2T, H] in banks 2-5, rounded at every accumulation, then the fold adds
            the hi and lo rows. "fp32": T = 32 only. out_ps is [4T, H/2] fp32 in banks 2-5 (one bank per 512-column
            chunk: 3 x 512 + 256 columns at H = 3584). Rows 0 .. 2T - 1 hold the hi and lo rows of output half 0
            (H columns 0 .. H/2 - 1), rows 2T .. 4T - 1 those of half 1. Each down stationary slot is [zeros 2T |
            hi T | lo T]: half 0 reads columns 2T .. 4T - 1 (rows 0 .. 2T - 1), half 1 all 4T columns (its zero
            block adds +0 to the rows of half 0). With dedupe, the head output and the executed block add into an
            fp32 SBUF accumulator. The store DMAs add the hi and lo rows in fp32, so each output value is rounded to
            bf16 once. The stores follow the dedupe blocks (the compiler takes no DMA compute in a loop body), so
            dedupe_fold_first and fold_dma do not apply. The outputs differ from bf16 in the last bits.
        rank_info (Optional[nl.NkiTensor]): int32 [1, 4] @HBM, (global rank, DP rank, TP rank, 0). Needed with down
            scales [E, 2, H] (and only read then). The rank's slice holds values [I*r, I*r + I) of each expert (r =
            the global rank). Its first checkpoint block of 32 values holds n = min(I, 32 - (I*r) % 32) of them.
            Down row p holds values 4p .. 4p + 3, so it uses scale group 0 if 4p < n, else group 1 (I = 24: 6, 2, 4
            and 6 group-0 rows for r % 4 = 0, 1, 2, 3). Each scale group has its own MXFP8 down stationary (the
            rows of the other group are zero, so its MX scales come from its own rows) and its own down projection
            into the same PSUM rows: twice the down matmuls, the MXFP8 work and the down-scale bytes.
        stationary_slots (int): ring slots of the down stationary (qmx_in, inter, inter_s). 3 (default) or 2: the PE
            runs down(i - 2) before the transposes of group i, so quantize(i) never overwrites a slot that a down
            projection still reads (832 bytes per partition less; 1,152 with fp32 down PSUM, 2,304 with two scale groups).
            The outputs are the same.
        head_finish_first (bool): dedupe: the head pipeline finishes its group 0 (SwiGLU, transposes, MXFP8, down
            projection) before the re-layout and the gate/up of group 1. The rows of group 1 come from the first
            part of the compaction, so the PE would otherwise idle until they arrive, with group 0's transposes and
            down projection queued behind gate_up(1). Default False. The outputs are the same.
        valid_tokens (Optional[int]): only tokens 0 .. valid_tokens - 1 are stored (the others are padding: give them
            valid expert ids and, with dedupe, zero selected scores, so they add no experts). None (default): all T.
        gate_up_pair_gathers (bool): one Pool gather loads the gate/up rows of two groups (a [128, 32] index: 4096
            rows), so the gate/up stream has half the gathers (each gather has a fixed cost plus a gap).
            The pairs start at the first group of each pipeline after a dedupe head group (which keeps its own
            gather and an extra ring slot). Needs an even gate_up_slots, sp_group_period=0, and group0_queues="pool"
            without dedupe. Default False. The outputs are the same.
        evict_split (str): dedupe: how DVE and ACT share the copy of the static output from PSUM to the fp32
            accumulator before the blocks. "rows" (default): DVE the hi rows, ACT the lo rows. "columns": each
            engine copies half of the columns of both, so each copy is half as long. "banks": DVE copies the columns
            of PSUM banks 2-3 (0 .. 2047), ACT the others: the two copies do not share a PSUM bank, so they run at
            the same time (with "columns" the compiler orders them, because both read bank 3). The outputs are the
            same.
        quantize_first (bool): the ACT and DVE copies of relayout(i + 1) are scheduled after the ACT copy and the
            quantization of group i - 1 (Step 3.5), as in program order. Without it the scheduler can put a re-layout
            that waits for its gather ahead of a ready quantization on the in-order ACT queue, so the down projection
            of group i - 1 waits for the gather of group i + 1. Default False. The outputs are the same.
        merged_down_gathers (bool): True (default): one Pool gather loads the down rows of two groups (a [128, 2]
            index). False: one gather per group (a [128, 1] index), so the down ring holds down_slots single units. In a
            program with many calls, every merged gather adds DGE table entries per call. The outputs are the same.
        dedupe_late_trips (bool): dedupe: the trip-count tile of the blocks is written after the static output leaves
            PSUM (a copy of the compaction's result, scheduled after the eviction). The trip counts are register loads
            on every engine, and the scheduler puts them right after their source is written: without this flag they
            sit in the head, where every engine waits for the end of the compaction. Default False. The outputs are
            the same.
        dedupe_fold_first (bool): dedupe with dedupe_block_mode="alternatives": the static output is folded and
            stored before the blocks, from PSUM through a bf16 copy of all its columns (fold_full), as in the
            baseline. A block that runs adds its PSUM output to fold_full and folds and stores again. Without a block
            the tail is then the baseline's (no fp32 accumulator, no second copy); the loop boundaries come after the
            stores. With a block, the first fold and stores are extra work. The outputs are the same as without the
            flag: the static output is exact in bf16, and one bf16 rounding of the fp32 sum follows the one block.
            With dedupe_prefetch, the first loads of the blocks are issued before the first fold. Default False.
        setup_memsets (str): the engine of the constant memsets of the setup (the zero rows of gated and qmx_in; with
            dedupe, the ones tiles of the compaction). "pool" (default) or "dve": GpSimd and DVE share one SBUF port,
            so the Pool memsets at the start slow the DVE ops of the head (the gather indices of the first group and
            the compaction). The outputs are the same.
        dedupe_onehot_parts (bool): dedupe: the one-hot tiles of the position matmuls are built per part of the
            positions (the parts of the position matmuls), so the first part, which gives the gather indices of head
            groups 1-4, does not wait for the one-hot columns of the other parts. Default False. The outputs are the
            same.
        name_suffix (Optional[str]): appended to every op and tensor name of the call, so a kernel can run the body more
            than once (one suffix per call). None (default): no suffix.
        buffer_cache (Optional[dict]): the SBUF buffers of earlier calls on the same SbufManager. A buffer with the
            name, shape, dtype and address of an earlier one is that tensor, so the DMAs of all calls have the same
            destinations and can share their descriptor table entries. None (default): new buffers.

    Returns:
        output (nl.NkiTensor): bfloat16 [T, H] @HBM.

    Notes:
        - Same per-group work as the kimi subpackage (the gate/up columns of 16 experts in one moving tile, 16
          experts x 8 MX rows in the down stationary), with all T tokens as the gate/up stationary: the PE time of a
          group does not depend on T.
        - pre_quant affinities as hi + lo MXFP8 terms (the kimi fused block's numerics): v = a * SiLU(gate) * up is
          split into hi (4 significant bits, exact in e4m3) and lo = v - hi, and a final bf16 matmul (fold) adds the
          hi and lo output rows of each token.
        - Dedupe numerics: the affinities come from the router logits (exp on ACT, fp16 compaction matmuls), and a
          token's output adds terms of the head and of the blocks in fp32, so the output differs slightly from the
          baseline.

    Algorithm (the code comments use the same step numbers):
        Step 1: Allocate the buffers (rings, routing tiles, PSUM tiles at fixed banks) and build the constants.
        Step 2: Routing of the slot positions.
            Step 2.1: Baseline: aff_cols[t, n] = the affinity of pair n for its token (block diagonal), ids_b[a, n] =
                the expert of pair n on every partition (a ones matmul of the block-diagonal index).
            Step 2.2: Dedupe: the head positions 0..15 (token 0) from the router index, then the compaction (Step 4)
                for the block positions.
            Step 2.3: Gather indices of every position group from ids_b (Step 3 reads them): gate/up rows
                (128 * expert + row), down rows (8 * expert + row) and the down-scale tiles (one per 4 groups of a
                pipeline).
        Step 3: The group pipeline (run_groups) for a list of position groups (baseline: all; dedupe: the head, then
            each block in its nl.fori_loop). Per group i:
            Step 3.1: Loads: the 16 experts' packed gate/up rows (one Pool gather, or 16 HWDGE DMAs), the down
                rows (one Pool gather of 128 rows) and, per 4 groups, the down-scale tile (one Pool gather). Each load
                follows (in program order) the last reader of its ring slot.
            Step 3.2: relayout(i): ACT and DVE copy the packed rows to the moving layout, ACT the scale segments.
            Step 3.3: gate_up(i): per H512 block, the gate and the up matmul: stationary = block hb of all T tokens,
                moving = the 16*I gate (or up) columns of the group. PSUM bf16 [T, 2 * 512].
            Step 3.4: swiglu(i): ACT SiLU(gate); DVE multiplies it by the affinity a[t, slot] (aff_cols) and by up:
                gated[t, (q, c, p)] in bf16 (rows p >= I/4 stay zero).
            Step 3.5: finish(i): 4 PE transposes put the rows of the 16 experts on 128 partitions; ACT copies them to
                qmx_in, DVE splits hi / lo and quantizes both terms: the down stationary [128, 2T].
            Step 3.6: down(i): 7 MX matmuls of 512 columns against the down rows; PSUM out_ps [2T, H] accumulates the
                groups.
            PE order: gate_up(i)[first 2 units], finish(i - 1), gate_up(i)[rest], down(i - 1) (down_lag=2:
            gate_up(i), finish(i - 1), down(i - 2)).
        Step 4 (dedupe): compaction on ACT, DVE and PE during the head group (compact_experts).
        Step 5: Output: out_ps (baseline) or the fp32 accumulator (dedupe: the head and each block add their out_ps)
            is folded by a bf16 matmul that adds the hi and lo rows of each token, and stored per 512 columns.

    Layouts:
        HBM: as in the kimi subpackage (expert_gate_up_packed row [gate_or_up][hb][I] then the scale segments).
        SBUF:
        - gu_rows uint16 [128, R * 16 * row_bytes/2]: packed rows of the 16 slots of a group, as loaded (slot c at bytes
          c * row_bytes of the ring slot).
        - gu_moving uint16 [128, Rm * 16 * 2 * n_h512 * I] and gu_moving_s [128, Rm * S * 4 * 16 * I bytes]: the moving
          layout [gate_or_up][hb][c][I] (one moving tile = 16*I contiguous columns) and its scales [segment][hb%4][c][I].
        - dn_w float4_e2m1fn_x4 [128, Rd * H]: partition 8c + p holds row p of slot c.
        - dn_s uint8 [128, 2 * G * H] (G: 1 or 2 down scale groups): 2 down-scale tiles; partition 32q + 4l + j holds
          the scale row of slot 4q + j of the l-th group of the tile (scale group s at column s * H of the tile).
        - gate_act fp32 [T, 2 * 16 * I], gated bf16 [T, 2 * 4 * 128]: rings of 2 groups.
        - qmx_in bf16 [128, 3 * G * 2T * 4] ([m, (s, term, t, q)]), inter / inter_s [128, 3 * G * 2T] ([m, (s, term,
          t)]; fp32 down PSUM: [128, 3 * G * 4T], [m, (s, zeros 2T | term, t)]): rings of 3 groups.
        - aff_cols fp32 [T, P]: the affinity of token t for the slot at position n.
        - ids_b fp32 [128, P]: the expert of position n on every partition; gather index tiles int32.
        - out_acc fp32 [2T, H] (dedupe), out_sb bf16 [T, H]; fp32 down PSUM: out_acc fp32 [4T, H/2] (out_ps rows).
        PSUM: banks 0-1 gu_ps (group parity), banks 2-5 out_ps bf16 [2T, H] (row t: hi of token t, T + t: lo; fp32
        down PSUM: [4T, H/2], see down_psum_dtype), bank 6 the transposes [128, 2 * 4T] bf16, bank 7 the routing /
        compaction tiles; banks 0-1 the fold tiles at the end.
    """
    # name_suffix: every op and tensor name of this call ends with it (the module views of named_modules).
    nisa, nl = named_modules(name_suffix)
    # Step 1: Sizes and constraints.
    T, K = index_tk.shape
    N = T * K + _GROUP * prefetch_groups  # dedupe positions (prefetch: the hot positions first)
    if len(expert_gate_up_packed.shape) == 4:
        # [L, E, ...] stacks: one expert dimension of L*E rows
        n_experts = expert_gate_up_packed.shape[1] if n_experts == None else n_experts
        expert_gate_up_packed = expert_gate_up_packed.reshape(
            (expert_gate_up_packed.shape[0] * expert_gate_up_packed.shape[1],) + tuple(expert_gate_up_packed.shape[2:])
        )
        expert_down_weights = expert_down_weights.reshape(
            (expert_down_weights.shape[0] * expert_down_weights.shape[1],) + tuple(expert_down_weights.shape[2:])
        )
        expert_down_weights_scale = expert_down_weights_scale.reshape(
            (expert_down_weights_scale.shape[0] * expert_down_weights_scale.shape[1],)
            + tuple(expert_down_weights_scale.shape[2:])
        )
    E_rows, _, row_bytes = expert_gate_up_packed.shape  # E_rows: all stacked experts
    E = E_rows if n_experts == None else n_experts
    kernel_assert(
        0 <= expert_offset and expert_offset + E <= E_rows,
        f"expert_offset {expert_offset} + E {E} exceeds the {E_rows} weight rows",
    )
    H = expert_down_weights.shape[2]
    n_h512 = H // 512
    n_chunks = div_ceil(n_h512, _BLOCKS)
    n_segments = div_ceil(2 * n_chunks, _X4)  # S: 4 column chunks per 4*I-byte scale segment
    I = row_bytes // (_X4 * (n_h512 + n_segments))
    down_rows = I // _X4
    # The HBM rows of the down weights of one expert: 8 (the padded MX block) or I / 4 (the compact form). The SBUF
    # layout keeps 8 rows for each slot in the two forms. Only the HBM row of each down gather index changes.
    dn_stride = expert_down_weights.shape[1]
    group_cols = _GROUP * I  # gate (or up) columns of a group in the moving layout
    gate_up_cols = 2 * n_h512 * I  # float4_e2m1fn_x4 elements of one expert's gate/up weights
    n_pos_groups = N // _GROUP
    n_et = E // _P_MAX
    dynamic = dedupe_experts == "dynamic"
    two_t = 2 * T
    R, Rm, Rd = gate_up_slots, moving_slots, down_slots
    # down ring: Rd group slots as Rd / 2 merged units of 2 groups, or Rd single units
    ring_units, unit_cols = (Rd // 2, 2 * H) if merged_down_gathers else (Rd, H)
    kernel_assert(
        dedupe_experts in (None, "dynamic"), f"dedupe_experts must be None or 'dynamic', got {dedupe_experts}"
    )
    kernel_assert(K == _GROUP, f"Expected K == 16 (one token per baseline group), got K={K}")
    # The lo output rows start at partition T, and an instruction's first partition must be a multiple of 32.
    kernel_assert(
        2 * T <= _P_MAX and T % _QUADRANT == 0,
        f"Expected T = 32 or 64, got T={T}. For fewer tokens (for example a 1-2 token tail) pad the call to T = 32 "
        "with valid_tokens (padding tokens: zero selected scores, and any valid expert ids).",
    )
    # At T = 64 the compaction's PSUM tiles in bank 7 overlap (the head row weights over the token-expert counts).
    kernel_assert(
        not dynamic or T == _QUADRANT,
        f"dedupe_experts='dynamic' needs T = 32, got T={T}. Use dedupe_experts=None for T = 64.",
    )
    kernel_assert(
        H == 512 * n_h512 and _BLOCKS <= n_h512 and 2 * H <= 4 * _PSUM_BANK_BYTES,
        f"Expected H % 512 == 0 and 2048 <= H <= 4096, got H={H}",
    )
    kernel_assert(I % _X4 == 0 and I <= _QUADRANT, f"Expected I % 4 == 0 and I <= 32, got I={I}")
    kernel_assert(
        dn_stride in (_MX_ROWS, down_rows),
        f"Expected down weights [E, {_MX_ROWS}, H] or the compact [E, {down_rows}, H], got {expert_down_weights.shape}",
    )
    # In the compact form, an index of the rows I/4 .. 7 of a slot points to the next expert. skip_down_padding moves
    # these indices out of bounds. The contiguous prefetch load reads 8 HBM rows for each slot, and the gather indices
    # of a caller (dedupe_precomputed, prefetch_indices) have their own row stride.
    kernel_assert(
        dn_stride == _MX_ROWS
        or (
            skip_down_padding
            and prefetch_load != "contiguous"
            and dedupe_precomputed == None
            and prefetch_indices == None
        ),
        f"The compact down weights [E, {down_rows}, H] need skip_down_padding=True, prefetch_load='gather', and no "
        "caller gather indices. Set these options, or give the padded down weights [E, 8, H].",
    )
    kernel_assert(E % _P_MAX == 0 and E <= _PAD_ID, f"Expected E % 128 == 0 and E <= {_PAD_ID}, got E={E}")
    kernel_assert(
        R >= 2 and Rm >= 2 and Rd >= 2 and Rd % 2 == 0,
        "gate_up_slots and moving_slots must be >= 2, down_slots even and >= 2",
    )
    kernel_assert(sp_group_period >= 0, f"sp_group_period must be >= 0, got {sp_group_period}")
    kernel_assert(down_lag in (1, 2), f"down_lag must be 1 or 2, got {down_lag}")
    kernel_assert(act_slots in (1, 2), f"act_slots must be 1 or 2, got {act_slots}")
    kernel_assert(early_ids_bank in (6, 7), f"early_ids_bank must be 6 or 7, got {early_ids_bank}")
    kernel_assert(
        down_psum_dtype in ("bf16", "fp32"), f"down_psum_dtype must be 'bf16' or 'fp32', got {down_psum_dtype}"
    )
    down_fp32 = down_psum_dtype == "fp32"
    # Down scale groups: 1 ([E, H]) or 2 ([E, 2, H], each down row on the scale of its own checkpoint block).
    kernel_assert(
        tuple(expert_down_weights_scale.shape) in ((E_rows, H), (E_rows, 2, H)),
        f"Expected down scales [{E_rows}, {H}] or [{E_rows}, 2, {H}], got {expert_down_weights_scale.shape}",
    )
    n_scale_sets = len(expert_down_weights_scale.shape) - 1  # G
    kernel_assert(
        n_scale_sets == 1 or (down_fp32 and rank_info != None),
        "Down scales [E, 2, H] need down_psum_dtype='fp32' and rank_info",
    )
    # fp32 out_ps: the 4T rows of the two output halves in one quadrant-aligned stationary (4T <= 128 columns)
    kernel_assert(
        not down_fp32 or (T == _QUADRANT and (H // 2) % (_DOWN_CHUNK // 2) == 0),
        f"down_psum_dtype='fp32' needs T = 32 and H / 2 a multiple of 256, got T={T}, H={H}",
    )
    kernel_assert(
        not pair_head_in_ring or (gate_up_pair_gathers and dedupe_experts == "dynamic" and gate_up_slots >= 4),
        "pair_head_in_ring needs gate_up_pair_gathers, dedupe_experts='dynamic' and gate_up_slots >= 4",
    )
    kernel_assert(
        evict_split in ("rows", "columns", "banks"),
        f"evict_split must be 'rows', 'columns' or 'banks', got {evict_split}",
    )
    kernel_assert(
        not dedupe_fold_first or (dynamic and dedupe_block_mode == "alternatives"),
        "dedupe_fold_first needs dedupe_experts='dynamic' and dedupe_block_mode='alternatives'",
    )
    kernel_assert(not two_phase_latent_first or two_phase, "two_phase_latent_first needs two_phase")
    kernel_assert(
        not two_phase or (dynamic and dedupe_static_groups > 1),
        "two_phase needs dedupe_experts='dynamic' and dedupe_static_groups > 1",
    )
    kernel_assert(
        fold_store_order in ("columns", "ready"),
        f"fold_store_order must be 'columns' or 'ready', got {fold_store_order}",
    )
    kernel_assert(
        fold_store_cols % _FOLD_CHUNK == 0 and _FOLD_CHUNK <= fold_store_cols <= H,
        f"fold_store_cols must be a multiple of {_FOLD_CHUNK} in {_FOLD_CHUNK} .. {H}, got {fold_store_cols}",
    )
    kernel_assert(
        (fold_store_order == "columns" and fold_store_cols == _FOLD_CHUNK and not dedupe_trips_after_stores)
        or (fold_dma and dedupe_fold_first),
        "fold_store_order, fold_store_cols and dedupe_trips_after_stores need fold_dma and dedupe_fold_first",
    )
    pairs_gu = gate_up_pair_gathers
    kernel_assert(
        not pairs_gu or (R % 2 == 0 and sp_group_period == 0 and (dynamic or group0_queues == "pool")),
        "gate_up_pair_gathers needs an even gate_up_slots, sp_group_period=0 and (without dedupe) group0_queues='pool'",
    )
    kernel_assert(
        group0_queues in ("sp_act", "sp", "pool"),
        f"group0_queues must be 'sp_act', 'sp' or 'pool', got {group0_queues}",
    )
    kernel_assert(
        dedupe_block_mode in ("binary", "alternatives"),
        f"dedupe_block_mode must be 'binary' or 'alternatives', got {dedupe_block_mode}",
    )
    kernel_assert(dedupe_order in ("id", "token"), f"dedupe_order must be 'id' or 'token', got {dedupe_order}")
    kernel_assert(
        dedupe_order == "id" or (dynamic and dedupe_block_mode == "alternatives"),
        "dedupe_order='token' needs dedupe_experts='dynamic' and dedupe_block_mode='alternatives'",
    )
    dve, act, pool = nisa.vector_engine, nisa.scalar_engine, nisa.gpsimd_engine
    kernel_assert(setup_memsets in ("pool", "dve"), f"setup_memsets must be 'pool' or 'dve', got {setup_memsets}")
    memset_engine = dve if setup_memsets == "dve" else pool

    # Pipelines: (tag, position groups, HWDGE groups); dedupe blocks: (first position group, groups).
    blocks = []
    S_head = dedupe_static_groups
    # head_lead: the head pipeline starts with a single group (1) or a pair (0, prefetch_groups even)
    head_lead = 1 if prefetch_groups == 0 else prefetch_groups % 2
    kernel_assert(
        prefetch_groups == 0
        or (
            dynamic
            and dedupe_block_mode == "alternatives"
            and dedupe_order == "token"
            and dedupe_token_rank == "flat"
            and not dedupe_index_first
            and dedupe_precomputed == None
            and sp_group_period == 0
            and 0 < prefetch_groups < dedupe_static_groups
        ),
        "prefetch_groups needs dedupe_experts='dynamic', alternatives, dedupe_order='token', dedupe_token_rank='flat', "
        "no dedupe_index_first / dedupe_precomputed / sp_group_period, and 0 < prefetch_groups < dedupe_static_groups",
    )
    kernel_assert(
        prefetch_load in ("gather", "contiguous"),
        f"prefetch_load must be 'gather' or 'contiguous', got {prefetch_load}",
    )
    if dynamic:
        kernel_assert(
            1 <= S_head < n_pos_groups, f"dedupe_static_groups must be in 1 .. {n_pos_groups - 1}, got {S_head}"
        )
        # The merged down gathers pair the groups (G, G + 1) with G odd from group 1 on (prefetch_groups even: G even
        # from group 0 on), so the blocks must start at a group of that parity.
        kernel_assert(
            S_head % 2 == head_lead,
            f"dedupe_static_groups must be {'odd' if head_lead else 'even'}, got {S_head}",
        )
        n_block_groups = n_pos_groups - S_head
        alternatives = dedupe_block_mode == "alternatives"
        if alternatives:
            kernel_assert(
                dedupe_block_groups != None
                and all(b > a for a, b in zip(dedupe_block_groups, dedupe_block_groups[1:]))
                and dedupe_block_groups[0] >= 1
                and dedupe_block_groups[-1] == n_block_groups,
                f"dedupe_block_mode='alternatives' needs increasing dedupe_block_groups that end with {n_block_groups}, "
                f"got {dedupe_block_groups}",
            )
        elif dedupe_block_groups == None:
            dedupe_block_groups = [
                1 << k for k in reversed(range(n_block_groups.bit_length())) if (1 << k) <= n_block_groups
            ]
        # The plan runs a block when the groups still needed are at least its size (largest block first). It covers
        # every count only if each block is at most 1 + the sum of the smaller blocks (true for powers of two).
        kernel_assert(
            alternatives
            or sum(dedupe_block_groups) == n_block_groups
            and all(b > a for a, b in zip(dedupe_block_groups[1:], dedupe_block_groups))
            and all(size <= 1 + sum(dedupe_block_groups[j + 1 :]) for j, size in enumerate(dedupe_block_groups)),
            f"dedupe_block_groups must be decreasing, add up to {n_block_groups}, and each count must be at most 1 + "
            f"the sum of the smaller counts, got {dedupe_block_groups}",
        )
        first = S_head
        for size in dedupe_block_groups:
            blocks.append((first, size))
            first += 0 if alternatives else size
        pipelines = [("head", list(range(S_head)), head_lead)]
        pipelines += [(f"blk{b}", list(range(f, f + size)), 0) for b, (f, size) in enumerate(blocks)]
    else:
        pipelines = [("base", list(range(n_pos_groups)), 0)]

    def pipeline_units(n_groups: int, lead: int, cut: int = 0) -> Tuple[List[List[int]], List[List[int]]]:
        """The down units (1 or 2 groups: one gather each) and the down-scale tiles (up to 4 groups: one gather
        each) of a pipeline of n_groups groups, as lists of group indices. With lead = 1 (the dedupe head) group 0
        has its own unit and tile, so its loads do not wait for the compaction. cut (prefetch_groups of the head): no
        tile holds groups of both sides of cut (the static prefetch groups and the compacted ones)."""
        down_units = [[0]] if lead else []
        unit_groups = 2 if merged_down_gathers else 1
        down_units += [list(range(j, min(j + unit_groups, n_groups))) for j in range(lead, n_groups, unit_groups)]
        tiles = [[0]] if lead else []
        if cut > lead:
            tiles += [list(range(j, min(j + _BLOCKS, cut))) for j in range(lead, cut, _BLOCKS)]
        tiles += [list(range(j, min(j + _BLOCKS, n_groups))) for j in range(max(lead, cut), n_groups, _BLOCKS)]
        return down_units, tiles

    def cut_of(tag: str) -> int:
        """The static prefix of a pipeline: the prefetch groups of the head."""
        return prefetch_groups if tag == "head" else 0

    # Down-scale tiles of all pipelines (tile l-th group at partition offset 4l).
    scale_tiles = []  # (tag, position groups of the tile)
    for tag, groups, lead in pipelines:
        for tile in pipeline_units(len(groups), lead, cut_of(tag))[1]:
            scale_tiles.append((tag, [groups[i] for i in tile]))
    n_scale_tiles = len(scale_tiles)
    # The position groups that a single-group gather reads (gate/up: gi_gu; down: gi_dn), in any pipeline. With the
    # pair and merged gathers, the other groups need no single-group index (dedupe_index_first skips them).
    single_gu, single_dn = set(), set()
    for tag, groups, lead in pipelines:
        n = len(groups)
        for i in range(n):
            paired = gate_up_pair_gathers and i >= lead and lead + 2 * ((i - lead) // 2) + 1 < n
            if not paired:
                single_gu.add(groups[i])
        for unit in pipeline_units(n, lead, cut_of(tag))[0]:
            if len(unit) == 1:
                single_dn.add(groups[unit[0]])

    def group_runs(groups: Sequence[int], wanted) -> List[Tuple[int, int]]:
        """The maximal runs (first, count) of consecutive groups of `groups` that are in wanted."""
        runs = []
        for g in groups:
            if g in wanted:
                if runs and runs[-1][0] + runs[-1][1] == g:
                    runs[-1] = (runs[-1][0], runs[-1][1] + 1)
                else:
                    runs.append((g, 1))
        return runs

    # Step 1.2: SBUF buffers (weights first, so their DMA writes stay aligned).
    sbm.open_scope()
    alloc = cached_alloc(sbm, name_suffix, buffer_cache)
    # pair mode with a dedupe head: the head's group 0 has its own ring slot (R), so the pairs fill slots 0 .. R - 1
    # pair_head_in_ring: the head's group 0 uses ring slot R - 2 (the pair of groups R - 1 and R, gathered after its
    # re-layout), so the ring has no extra slot.
    gu_slots = R + (1 if pairs_gu and dynamic and head_lead == 1 and not pair_head_in_ring else 0)
    gu_stride = gu_slots * _GROUP * row_bytes // 2  # uint16 elements per partition of the row ring tensor
    if weights_arena:
        # One tensor: the row ring, the down weights (Rd groups), then the down-scale tiles (2 * G * H bytes).
        rows_elems, down_elems = gu_stride, Rd * H
        gu_stride = rows_elems + down_elems + n_scale_sets * H
    gu_rows = alloc((_P_MAX, gu_stride), nl.uint16, name="gu_rows")
    gu_moving = alloc((_P_MAX, Rm * _GROUP * gate_up_cols), nl.uint16, name="gu_moving")
    gu_moving_s = alloc((_P_MAX, Rm * n_segments * _BLOCKS * group_cols // 2), nl.uint16, name="gu_moving_s")
    if weights_arena:
        dn_w = gu_rows.view(expert_down_weights.dtype)[:, rows_elems : rows_elems + down_elems]
        dn_s = gu_rows.view(nl.uint8)[:, 2 * (rows_elems + down_elems) : 2 * gu_stride]
    else:
        dn_w = alloc((_P_MAX, Rd * H), expert_down_weights.dtype, name="dn_w")
        dn_s = alloc((_P_MAX, 2 * n_scale_sets * H), nl.uint8, name="dn_s")
    tile_cols = n_scale_sets * H  # columns of one down-scale tile of dn_s
    act_cols = act_slots * group_cols  # the activation temporaries: rings of act_slots groups
    gate_act = alloc((T, act_cols), nl.float32, name="gate_act")
    if situ != None:
        # SiTU (Step 3.4): rings of 2 groups like gate_act, [t, (c, I)]: sigmoid(g) (or tanh(g/2)) and tanh(u/lb).
        situ_sigmoid = alloc((T, act_cols), nl.float32, name="situ_sigmoid")
        situ_up = alloc((T, act_cols), nl.float32, name="situ_up")
    gated = alloc((T, 2 * _X4 * _P_MAX), nl.bfloat16, name="gated")
    qmx_cols = two_t * _X4  # qmx_in columns of one scale group of a slot: [term, t, q]
    qmx_slot = n_scale_sets * qmx_cols
    Rs = stationary_slots
    kernel_assert(Rs in (2, 3), f"stationary_slots must be 2 or 3, got {Rs}")
    qmx_in = alloc((_P_MAX, Rs * qmx_slot), nl.bfloat16, name="qmx_in")
    # Down stationary of one scale group of a slot: [term, t] (2T columns), or with fp32 down PSUM [zeros 2T | term,
    # t] (4T columns; the zero block stays zero).
    set_cols = 2 * two_t if down_fp32 else two_t
    term_first = two_t if down_fp32 else 0  # the first hi column in the stationary of a scale group
    stat_slot = n_scale_sets * set_cols
    inter = alloc((_P_MAX, Rs * stat_slot), nl.float8_e4m3fn_x4, name="inter")
    inter_s = alloc((_P_MAX, Rs * stat_slot), nl.uint8, name="inter_s")
    if n_scale_sets == 2:
        # group_mask[a, s] = 1.0 if down row a % 8 uses scale group s, else 0 (from rank_info, Step 1.4)
        group_mask = alloc((_P_MAX, 2), nl.float32, name="down_group_mask")
        group_work = alloc((_P_MAX, 2), nl.int32, name="down_group_work")
    ident_offsets = alloc((T, T), nl.float32, name="ident_offsets")
    ident_bf16 = alloc((T, T), nl.bfloat16, name="ident_bf16")
    kernel_assert(affinity_dtype in ("fp32", "bf16"), f"affinity_dtype must be 'fp32' or 'bf16', got {affinity_dtype}")
    kernel_assert(
        situ == None or (len(situ) == 2 and situ[0] > 0 and situ[1] > 0), f"situ must be (beta, linear_beta) > 0"
    )
    # affinity_dtype="bf16": every write rounds the affinity to bf16 once
    aff_cols = alloc((T, N), nl.bfloat16 if affinity_dtype == "bf16" else nl.float32, name="aff_cols")
    ids_b = alloc((_P_MAX, N), nl.float32, name="ids_b")
    # a % 16, a / 16, a / 8, (a / 4) % 4, a % 8, a % 2, (a / 2) % 8
    part = alloc((_P_MAX, 7), nl.int32, name="part")
    part_f = alloc((_P_MAX, 7), nl.float32, name="part_f")
    slot_offsets = alloc((_P_MAX, _GROUP), nl.float32, name="slot_offsets")
    sel16 = alloc((_P_MAX, _GROUP), nl.float32, name="sel16")  # [a, c] = (c == a % 16)
    sel8 = alloc((_P_MAX, _GROUP), nl.float32, name="sel8")  # [a, c] = (c == a / 8)
    sel_sc = alloc((_P_MAX, _GROUP), nl.float32, name="sel_sc")  # [a, c] = (c == 4 (a / 32) + a % 4)
    sel_l = alloc((_P_MAX, _BLOCKS), nl.float32, name="sel_l")  # [a, l] = (l == (a / 4) % 4)
    picked = alloc((_P_MAX, n_pos_groups, _GROUP), nl.float32, name="picked")
    e16 = alloc((_P_MAX, n_pos_groups), nl.float32, name="e16")
    e8 = alloc((_P_MAX, n_pos_groups), nl.float32, name="e8")
    e_sc = alloc((_P_MAX, n_pos_groups), nl.float32, name="e_sc")
    tile_pick = alloc((_P_MAX, n_scale_tiles, _BLOCKS), nl.float32, name="tile_pick")
    gi_rows = alloc((_P_MAX, _GROUP), nl.float32, name="gi_rows")  # [a, b] = 8b + a / 16
    skip_empty = skip_empty_slots and dynamic
    if skip_scale_duplicates:
        sc_skip = alloc((_P_MAX, 1), nl.int32, name="sc_skip")  # 2^20 on partitions 32q + 16 .. 32q + 31
    if skip_empty:
        skip_ids = alloc((_P_MAX, n_pos_groups), nl.float32, name="skip_ids")  # _SKIP_ID on the empty positions
    gi_gu = alloc((_P_MAX, n_pos_groups * _GROUP), nl.int32, name="gi_gu")
    if pairs_gu:
        # pair (G, G + 1): gi_gu2[a, (16 G + b)] for b < 32 (the 32 columns of both groups' regions)
        gi_gu2 = alloc((_P_MAX, n_pos_groups * _GROUP), nl.int32, name="gi_gu2")
        e32 = alloc((_P_MAX, n_pos_groups // 2), nl.float32, name="e32")
        picked32 = alloc((_P_MAX, n_pos_groups // 2, 2 * _GROUP), nl.float32, name="picked32")
        sel32 = alloc((_P_MAX, 2 * _GROUP), nl.float32, name="sel32")  # [a, j] = (j == a % 32)
        rows32 = alloc((_P_MAX, 2 * _GROUP), nl.float32, name="rows32")  # [a, b] = 4b + a / 32
        p32 = alloc((_P_MAX, 2), nl.int32, name="p32")  # a % 32, a / 32
        p32_f = alloc((_P_MAX, 2), nl.float32, name="p32_f")
        j32 = alloc((_P_MAX, 2 * _GROUP), nl.float32, name="j32")
    gi_dn = alloc((_P_MAX, n_pos_groups), nl.int32, name="gi_dn")
    gi_sc = alloc((_P_MAX, n_scale_tiles), nl.int32, name="gi_sc")
    # Merged down gathers of group pairs (G, G + 1): gi_dn2[a, (G, b)] for the first group G of each pair.
    pair_j = alloc((_P_MAX, 2 * _GROUP), nl.float32, name="pair_j")
    pair_off = alloc((_P_MAX, 2), nl.float32, name="pair_off")
    sel2 = alloc((_P_MAX, 2, 2 * _GROUP), nl.float32, name="sel2")
    pick2 = alloc((_P_MAX, n_pos_groups // 2, 2, 2 * _GROUP), nl.float32, name="pick2")
    e_pair = alloc((_P_MAX, n_pos_groups // 2, 2), nl.float32, name="e_pair")
    gi_dn2 = alloc((_P_MAX, 2 * n_pos_groups), nl.int32, name="gi_dn2")
    # fp32 down PSUM: no fold (the store DMAs add the hi and lo rows of out_acc)
    if not down_fp32:
        fold_sel = alloc((two_t, T), nl.bfloat16, name="fold_sel")
        fold_offsets = alloc((two_t, T), nl.int32, name="fold_offsets")
    half_cols = H // 2  # fp32 down PSUM: H columns of one output half
    if not dynamic and down_fp32:
        out_acc = alloc((2 * two_t, half_cols), nl.float32, name="out_acc")  # the rows of out_ps
    elif not dynamic:
        fold_in = alloc((two_t, 2 * _FOLD_CHUNK), nl.bfloat16, name="fold_in")
        out_sb = alloc((T, H), nl.bfloat16, name="out_sb")
    if not dynamic:
        eye_tk = alloc((T, N), nl.float32, name="eye_tk")
        if expert_offset != 0:
            index_off = alloc((T, K), nl.int32, name="index_off")  # index + expert_offset (the HWDGE loads)
        index_moving = alloc((T, N), nl.float16, name="index_moving")
        ones_tokens = alloc((T, _P_MAX), nl.float16, name="ones_tokens")
    else:
        # dedupe: out_acc, out_sb and fold_in come after the compaction, at the addresses of its temporaries.
        ids_rest = alloc((T + 1, N), nl.float16, name="ids_rest")  # row T: id - _PAD_ID of each block position
        if sp_group_period > 0:
            # row T: the expert of each block position (0 if empty)
            ids_i32 = alloc((T + 1, N), nl.int32, name="ids_i32")
            ids_f32 = alloc((T + 1, N), nl.float32, name="ids_f32")
        idx0 = alloc((1, K), nl.float16, name="idx0")
        ones_row = alloc((1, _P_MAX), nl.float16, name="ones_row")
        ones_row_t = alloc((T + 1, _P_MAX), nl.float16, name="ones_row_t")  # row T: the ones of the rest broadcast

    # Step 1.3: PSUM tiles. A dedupe block gets its own pipeline tensors (same addresses): a PSUM tensor cannot be
    # live across a loop boundary.
    def psum_tiles(suffix: str) -> Tuple[nl.NkiTensor, List[nl.NkiTensor], nl.NkiTensor]:
        # bf16 [2T, H] or fp32 [4T, H/2] (down_psum_dtype), from bank 2: 7 KB per partition
        out_tile = nl.ndarray(
            (2 * two_t, half_cols) if down_fp32 else (two_t, H),
            dtype=nl.float32 if down_fp32 else nl.bfloat16,
            buffer=nl.psum,
            address=(0, 2 * _PSUM_BANK_BYTES),
            name=f"out_ps{suffix}",
        )
        gu_tiles = [
            nl.ndarray(
                (T, 2 * _PSUM_COLS),
                dtype=nl.bfloat16,
                buffer=nl.psum,
                address=(0, parity * _PSUM_BANK_BYTES),
                name=f"gu_ps{parity}{suffix}",
            )
            for parity in range(2)
        ]
        tr_tile = nl.ndarray(
            (_P_MAX, 2 * _X4 * T),
            dtype=nl.bfloat16,
            buffer=nl.psum,
            address=(0, 6 * _PSUM_BANK_BYTES),
            name=f"tr_ps{suffix}",
        )
        return out_tile, gu_tiles, tr_tile

    fold_ps = [
        nl.ndarray(
            (T, _FOLD_CHUNK), dtype=nl.float32, buffer=nl.psum, address=(0, r * _PSUM_BANK_BYTES), name=f"fold_ps{r}"
        )
        for r in range(2)
    ]
    bank7 = 7 * _PSUM_BANK_BYTES
    if output == None:
        output = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="output")
    kernel_assert(
        output.shape[-1] == H and output_row + T <= output.shape[0],
        f"output needs {H} columns and at least {output_row + T} rows, got {output.shape}",
    )
    kernel_assert(
        valid_tokens == None or 1 <= valid_tokens <= T, f"valid_tokens must be in 1 .. {T}, got {valid_tokens}"
    )

    def after_norm_op(instruction: nisa.NkiInstruction) -> nisa.NkiInstruction:
        """Schedule a setup instruction after the RMSNorm, so the setup does not delay it."""
        if after_norm != None:
            instruction.depends_on(after_norm, kind=nisa.dependency.schedule)
        return instruction

    # Step 1.4: Constants. ident_bf16[t, t'] = (t' == t), the moving of the transposes; fold_sel[r, t] = (r % T == t),
    # the fold stationary; the partition fields and the selectors of the gather indices.
    index_first = dedupe_index_first and dynamic
    kernel_assert(
        not dedupe_index_first or (dynamic and dedupe_static_groups > 1 and dedupe_block_mode == "alternatives"),
        "dedupe_index_first needs dedupe_experts='dynamic', dedupe_static_groups > 1 and dedupe_block_mode="
        "'alternatives'",
    )
    precomputed = dedupe_precomputed != None
    kernel_assert(
        not precomputed or (dynamic and dedupe_static_groups > 1 and not index_first and sp_group_period == 0),
        "dedupe_precomputed needs dedupe_experts='dynamic', dedupe_static_groups > 1, no dedupe_index_first and "
        "sp_group_period=0",
    )

    def late_constants() -> None:
        """The constants of the group pipeline and the fold (not of the gather indices)."""
        after_norm_op(
            nisa.iota(dst=ident_offsets, pattern=[[1, T]], offset=0, channel_multiplier=-1, name="iota_ident")
        )
        after_norm_op(
            nisa.tensor_scalar(
                dst=ident_bf16, data=ident_offsets, op0=nl.equal, operand0=0.0, engine=dve, name="ident_op"
            )
        )
        if not down_fp32:
            after_norm_op(
                nisa.iota(dst=fold_offsets, pattern=[[-1, T]], offset=0, channel_multiplier=1, name="iota_fold")
            )
            for term in range(2):
                after_norm_op(
                    nisa.tensor_scalar(
                        dst=fold_sel[term * T : (term + 1) * T, :],
                        data=fold_offsets[term * T : (term + 1) * T, :],
                        op0=nl.equal,
                        operand0=term * T,
                        engine=dve,
                        name=f"fold_sel_op{term}",
                    )
                )
        after_norm_op(nisa.memset(dst=qmx_in, value=0.0, engine=memset_engine, name="memset_qmx_in"))
        if down_fp32:
            # the zero blocks of the down stationary slots (zero values, finite E8M0 scales)
            after_norm_op(nisa.memset(dst=inter.view(nl.uint32), value=0, engine=memset_engine, name="memset_inter"))
            after_norm_op(nisa.memset(dst=inter_s, value=0, engine=memset_engine, name="memset_inter_s"))
        # rows p >= I/4 stay zero
        after_norm_op(nisa.memset(dst=gated, value=0.0, engine=memset_engine, name="memset_gated"))

    if not index_first:
        late_constants()
    if skip_empty:
        # The scale segments of every gate/up ring slot (row bytes gate_up_cols * 2 ..) are zero (finite E8M0 scales)
        # until a gather writes them: an empty position keeps the values of the slot.
        words, first_word = row_bytes // 2, gate_up_cols  # uint16 words of a packed row, the first scale word
        after_norm_op(
            nisa.memset(
                # [p, slot c, scale words]: the scale segments of the ring
                dst=gu_rows.ap(
                    pattern=[[gu_stride, _P_MAX], [words, gu_slots * _GROUP], [1, words - first_word]],
                    offset=first_word,
                ),
                value=0,
                engine=memset_engine,
                name="memset_ring_scales",
            )
        )
    # part[a] = (a % 16, a / 16, a / 8, (a / 4) % 4, a % 8, a % 2, (a / 2) % 8) from an iota and bit operations.
    after_norm_op(nisa.iota(dst=part, pattern=[[0, 7]], offset=0, channel_multiplier=1, name="iota_part"))
    for col, (op0, operand0, op1, operand1) in enumerate(
        (
            (nl.bitwise_and, _GROUP - 1, None, None),  # a % 16
            (nl.right_shift, 4, None, None),  # a / 16
            (nl.right_shift, 3, None, None),  # a / 8
            (nl.right_shift, 2, nl.bitwise_and, 3),  # (a / 4) % 4
            (nl.bitwise_and, _MX_ROWS - 1, None, None),  # a % 8
            (nl.bitwise_and, 1, None, None),  # a % 2
            (nl.right_shift, 1, nl.bitwise_and, _MX_ROWS - 1),  # (a / 2) % 8
        )
    ):
        after_norm_op(
            nisa.tensor_scalar(
                dst=part[:, col : col + 1],
                data=part[:, col : col + 1],
                op0=op0,
                operand0=operand0,
                op1=op1,
                operand1=operand1,
                engine=dve,
                name=f"part_op{col}",
            )
        )
    after_norm_op(nisa.tensor_copy(dst=part_f, src=part, engine=dve, name="part_f_op"))
    if n_scale_sets == 2:
        # group_mask[a, 0] = (4 * (a % 8) < n), n = min(I, 32 - (I*r) % 32) for the global rank r; [a, 1] = 1 - [a, 0].
        # (An arithmetic op and a bitwise op do not share one instruction: the arithmetic runs in fp32.)
        after_norm_op(
            nisa.dma_copy(
                dst=group_work[:, 0:1],
                src=rank_info.ap(pattern=[[0, _P_MAX], [1, 1]], offset=0),  # the global rank on every partition
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name="load_rank",
            )
        )
        nisa.tensor_scalar(
            dst=group_work[:, 0:1],
            data=group_work[:, 0:1],
            op0=nl.multiply,
            operand0=I,
            engine=dve,
            name="down_group_start",
        )
        nisa.tensor_scalar(
            dst=group_work[:, 0:1],
            data=group_work[:, 0:1],
            op0=nl.bitwise_and,
            operand0=_MX_GROUP - 1,
            engine=dve,
            name="down_group_offset",
        )
        # n = min(32 - offset, I) as fp32 in group_mask[:, 1]
        nisa.tensor_scalar(
            dst=group_mask[:, 1:2],
            data=group_work[:, 0:1],
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.add,
            operand1=float(_MX_GROUP),
            engine=dve,
            name="down_group_first_block",
        )
        nisa.tensor_scalar(
            dst=group_mask[:, 1:2],
            data=group_mask[:, 1:2],
            op0=nl.minimum,
            operand0=float(I),
            engine=dve,
            name="down_group_count",
        )
        # 4 * (a % 8) < n: the first value of the partition's down row is in the first block
        nisa.tensor_scalar(
            dst=group_mask[:, 0:1],
            data=part_f[:, 4:5],
            op0=nl.multiply,
            operand0=float(_X4),
            op1=nl.less,
            operand1=group_mask[:, 1:2],
            engine=dve,
            name="down_group0",
        )
        nisa.tensor_scalar(
            dst=group_mask[:, 1:2],
            data=group_mask[:, 0:1],
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.add,
            operand1=1.0,
            engine=dve,
            name="down_group1",
        )
    if skip_scale_duplicates:
        # ((a / 16) & 1) << 20 = 2^20 (past every scale row) on the upper half of each quadrant of a scale tile
        kernel_assert(E_rows < (1 << 20), f"skip_scale_duplicates needs fewer than 2^20 weight rows, got {E_rows}")
        after_norm_op(
            nisa.tensor_scalar(
                dst=sc_skip,
                data=part[:, 1:2],
                op0=nl.bitwise_and,
                operand0=1,
                op1=nl.left_shift,
                operand1=20,
                engine=dve,
                name="sc_skip_op",
            )
        )
    # slot_offsets[a, c] = c; sel16 = (c == a % 16); sel8 = (c == a / 8);
    # sel_sc = (c == 4 (a / 32) + a % 4) = (c == 4 * (a / 32) + (a % 16) % 4); sel_l = (l == (a / 4) % 4).
    after_norm_op(nisa.iota(dst=slot_offsets, pattern=[[1, _GROUP]], offset=0, channel_multiplier=0, name="iota_slots"))
    after_norm_op(
        nisa.tensor_scalar(
            dst=sel16, data=slot_offsets, op0=nl.equal, operand0=part_f[:, 0:1], engine=dve, name="sel16_op"
        )
    )
    after_norm_op(
        nisa.tensor_scalar(
            dst=sel8, data=slot_offsets, op0=nl.equal, operand0=part_f[:, 2:3], engine=dve, name="sel8_op"
        )
    )
    # c(a) = 4 * (a / 32) + a % 4 = a % 4 + 4 * ((a / 16) / 2): from part (a / 16) and a % 16.
    sc_slot = alloc((_P_MAX, 2), nl.int32, name="sc_slot")
    sc_slot_f = alloc((_P_MAX, 1), nl.float32, name="sc_slot_f")
    after_norm_op(
        nisa.tensor_scalar(
            dst=sc_slot[:, 0:1],
            data=part[:, 1:2],
            op0=nl.right_shift,
            operand0=1,
            op1=nl.left_shift,
            operand1=2,
            engine=dve,
            name="sc_slot_hi",
        )
    )
    after_norm_op(
        nisa.tensor_scalar(
            dst=sc_slot[:, 1:2], data=part[:, 0:1], op0=nl.bitwise_and, operand0=3, engine=dve, name="sc_slot_lo"
        )
    )
    after_norm_op(
        nisa.tensor_tensor(
            dst=sc_slot_f, data1=sc_slot[:, 0:1], data2=sc_slot[:, 1:2], op=nl.add, engine=dve, name="sc_slot_op"
        )
    )
    after_norm_op(
        nisa.tensor_scalar(
            dst=sel_sc, data=slot_offsets, op0=nl.equal, operand0=sc_slot_f, engine=dve, name="sel_sc_op"
        )
    )
    after_norm_op(
        nisa.tensor_scalar(
            dst=sel_l,
            data=slot_offsets[:, 0:_BLOCKS],
            op0=nl.equal,
            operand0=part_f[:, 3:4],
            engine=dve,
            name="sel_l_op",
        )
    )
    # gi_rows[a, b] = 8b + a / 16: the source row of gate/up gather index [a, b] (row 128b + a lands on partition
    # 8b + a / 16, slot a % 16).
    after_norm_op(
        nisa.iota(dst=gi_rows, pattern=[[_MX_ROWS, _GROUP]], offset=0, channel_multiplier=0, name="iota_gi_rows")
    )
    after_norm_op(
        nisa.tensor_scalar(
            dst=gi_rows, data=gi_rows, op0=nl.add, operand0=part_f[:, 1:2], engine=dve, name="gi_rows_op"
        )
    )
    if pairs_gu:
        # p32[a] = (a % 32, a / 32); sel32[a, j] = (j == a % 32); rows32[a, b] = 4b + a / 32 (+ 128 * expert_offset)
        after_norm_op(nisa.iota(dst=p32, pattern=[[0, 2]], offset=0, channel_multiplier=1, name="iota_p32"))
        after_norm_op(
            nisa.tensor_scalar(
                dst=p32[:, 0:1], data=p32[:, 0:1], op0=nl.bitwise_and, operand0=31, engine=dve, name="p32_mod"
            )
        )
        after_norm_op(
            nisa.tensor_scalar(
                dst=p32[:, 1:2], data=p32[:, 1:2], op0=nl.right_shift, operand0=5, engine=dve, name="p32_div"
            )
        )
        after_norm_op(nisa.tensor_copy(dst=p32_f, src=p32, engine=dve, name="p32_f_op"))
        after_norm_op(nisa.iota(dst=j32, pattern=[[1, 2 * _GROUP]], offset=0, channel_multiplier=0, name="iota_j32"))
        after_norm_op(
            nisa.tensor_scalar(dst=sel32, data=j32, op0=nl.equal, operand0=p32_f[:, 0:1], engine=dve, name="sel32_op")
        )
        after_norm_op(
            nisa.tensor_scalar(
                dst=rows32,
                data=j32,
                op0=nl.multiply,
                operand0=4.0,
                op1=nl.add,
                operand1=p32_f[:, 1:2],
                engine=dve,
                name="rows32_op",
            )
        )
        if expert_offset != 0:
            after_norm_op(
                nisa.tensor_scalar(
                    dst=rows32,
                    data=rows32,
                    op0=nl.add,
                    operand0=float(_P_MAX * expert_offset),
                    engine=dve,
                    name="rows32_offset",
                )
            )
    if expert_offset != 0:
        # expert_offset: gate/up row 128 * (offset + e) + .., down row dn_stride * (offset + e) + .. (exact in fp32
        # below 2^24)
        kernel_assert(_P_MAX * (expert_offset + E) < (1 << 24), f"expert_offset {expert_offset} is too large")
        after_norm_op(
            nisa.tensor_scalar(
                dst=gi_rows,
                data=gi_rows,
                op0=nl.add,
                operand0=float(_P_MAX * expert_offset),
                engine=dve,
                name="gi_rows_offset",
            )
        )
        after_norm_op(
            nisa.tensor_scalar(
                dst=part_f[:, 4:5],
                data=part_f[:, 4:5],
                op0=nl.add,
                operand0=float(dn_stride * expert_offset),
                engine=dve,
                name="dn_rows_offset",
            )
        )
        after_norm_op(
            nisa.tensor_scalar(
                dst=part_f[:, 6:7],
                data=part_f[:, 6:7],
                op0=nl.add,
                operand0=float(dn_stride * expert_offset),
                engine=dve,
                name="dn2_rows_offset",
            )
        )

    if skip_down_padding and down_rows < _MX_ROWS:
        # The padding rows (row a % 8 of a single down gather, (a / 2) % 8 of a merged one, at least I/4) get an index
        # past the last down row: the gather skips them (oob_mode skip).
        past_end = float(dn_stride * E_rows)
        for col, label in ((4, "dn"), (6, "dn2")):
            skip = alloc((_P_MAX, 1), nl.float32, name=f"{label}_skip")
            after_norm_op(
                nisa.tensor_scalar(
                    dst=skip,
                    data=part[:, col : col + 1],
                    op0=nl.greater_equal,
                    operand0=float(down_rows),
                    op1=nl.multiply,
                    operand1=past_end,
                    engine=dve,
                    name=f"{label}_skip_op",
                )
            )
            after_norm_op(
                nisa.tensor_tensor(
                    dst=part_f[:, col : col + 1],
                    data1=part_f[:, col : col + 1],
                    data2=skip,
                    op=nl.add,
                    engine=dve,
                    name=f"{label}_skip_add",
                )
            )

    # Merged down gather of groups (G, G + 1): index [a, b] feeds row r = 128b + a, which lands on partition
    # m = 64b + a / 2, slot a % 2 (group G + a % 2): row (a / 2) % 8 of slot m / 8 = 8b + a / 16. So it reads the expert of
    # position 16G + j with j = 16 (a % 2) + 8b + a / 16: sel2[a, b, j] = (j == pair_off[a, b]).
    after_norm_op(nisa.iota(dst=pair_j, pattern=[[1, 2 * _GROUP]], offset=0, channel_multiplier=0, name="iota_pair_j"))
    # (pair_off follows part_f, which follows the RMSNorm)
    nisa.scalar_tensor_tensor(
        dst=pair_off[:, 0:1],
        data=part_f[:, 5:6],
        op0=nl.multiply,
        operand0=float(_GROUP),
        op1=nl.add,
        operand1=part_f[:, 1:2],
        name="pair_off_op0",
    )
    after_norm_op(
        nisa.tensor_scalar(
            dst=pair_off[:, 1:2],
            data=pair_off[:, 0:1],
            op0=nl.add,
            operand0=float(_MX_ROWS),
            engine=dve,
            name="pair_off_op1",
        )
    )
    for b in range(2):
        after_norm_op(
            nisa.tensor_scalar(
                dst=sel2[:, b, :],
                data=pair_j,
                op0=nl.equal,
                operand0=pair_off[:, b : b + 1],
                engine=dve,
                name=f"sel2_op{b}",
            )
        )

    in_loop = [False]  # ops inside a dedupe block (nl.fori_loop body) get no name= (the body is traced more than once)
    after_latent = [None]  # two_phase_latent_first: the latent quantization (sent at the resume)
    gather_after = [None, None]  # dedupe_precomputed: the [P, 1] value that the first Pool gather waits for, a scratch

    def nm(name: str) -> Optional[str]:
        """name, or None inside a dedupe block."""
        return None if in_loop[0] else name

    def chain(instruction: nisa.NkiInstruction, previous: List) -> nisa.NkiInstruction:
        """Schedule instruction after previous[0] (the engine's last chained instruction) and make it the last."""
        if previous[0] != None:
            instruction.depends_on(previous[0], kind=nisa.dependency.schedule)
        previous[0] = instruction
        return instruction

    # the first group of the gate/up pair gathers (after a dedupe head group; prefetch_groups even: from group 0)
    pair_parity = head_lead if dynamic else 0

    def pad_empty(ids: nl.NkiTensor, label: str, skip: bool) -> None:
        """The expert ids of empty dedupe positions (id >= E, fp32 [128, ...]): expert 0 (id * (id < E)), or with skip
        id + _SKIP_ID, past the last weight row (skip_empty_slots)."""
        if not skip:
            nisa.scalar_tensor_tensor(
                dst=ids, data=ids, op0=nl.less, operand0=float(E), op1=nl.multiply, operand1=ids, name=label
            )
            return
        cols = 1
        for size in ids.shape[1:]:
            cols *= size
        # skip_ids viewed with the shape of ids
        flags = skip_ids.ap(pattern=[[n_pos_groups, _P_MAX]] + [[1, cols]], offset=0)
        if len(ids.shape) == 3:
            flags = skip_ids.ap(
                pattern=[[n_pos_groups, _P_MAX], [ids.shape[2], ids.shape[1]], [1, ids.shape[2]]], offset=0
            )
        nisa.tensor_scalar(
            dst=flags,
            data=ids,
            op0=nl.greater_equal,
            operand0=float(E),
            op1=nl.multiply,
            operand1=_SKIP_ID,
            engine=dve,
            name=label + "_skip",
        )
        nisa.tensor_tensor(dst=ids, data1=ids, data2=flags, op=nl.add, engine=dve, name=label)

    def gather_indices(groups: Sequence[int], tiles: Sequence[int], suffix: str) -> nisa.NkiInstruction:
        """Step 2.3: the gather indices of the position groups `groups` and of the scale tiles `tiles`, from ids_b.

        e16[a, G] = the expert of position 16G + a % 16 (gate/up gather row 128b + a: slot a % 16, row 8b + a / 16);
        e8[a, G] = the expert of position 16G + a / 8 (down gather row a: slot a / 8, row a % 8); e_sc[a, G] = the
        expert of position 16G + 4 (a / 32) + a % 4 (the down scale of that slot, in partition a of a scale tile
        when the group is the tile's (a / 4) % 4-th). Empty dedupe positions (id >= E) load expert 0."""
        g0, n_g = groups[0], len(groups)
        kernel_assert(list(groups) == list(range(g0, g0 + n_g)), "gather_indices needs consecutive groups")

        def picks(g0: int, n_g: int, which, run: str = "") -> None:
            """e16 / e8 / e_sc (the selectors in which) of the groups g0 .. g0 + n_g - 1."""
            ids_view = ids_b.ap(pattern=[[N, _P_MAX], [_GROUP, n_g], [1, _GROUP]], offset=g0 * _GROUP)  # [a, G, c]
            for selector, dst, label in which:
                nisa.tensor_tensor(
                    dst=picked[:, 0:n_g, :],
                    data1=ids_view,
                    data2=selector.ap(pattern=[[_GROUP, _P_MAX], [0, n_g], [1, _GROUP]], offset=0),  # [a, G, c]
                    op=nl.multiply,
                    engine=dve,
                    name=f"pick_{label}{suffix}{run}",
                )
                nisa.tensor_reduce(
                    dst=dst[:, g0 : g0 + n_g],
                    op=nl.add,
                    data=picked[:, 0:n_g, :],
                    axis=2,
                    name=f"{label}_op{suffix}{run}",
                )
                # id >= E (an empty dedupe position) -> expert 0: id * (id < E) (skip_empty_slots: past the last row)
                if dynamic:
                    pad_empty(dst[:, g0 : g0 + n_g], f"{label}_pad{suffix}{run}", skip_empty and label != "esc")

        def gu_singles(g0: int, n_g: int, run: str = "") -> None:
            # gi_gu[a, (G, b)] = 128 * e16[a, G] + 8b + a / 16
            nisa.scalar_tensor_tensor(
                dst=gi_gu.ap(pattern=[[n_pos_groups * _GROUP, _P_MAX], [_GROUP, n_g], [1, _GROUP]], offset=g0 * _GROUP),
                data=e16.ap(pattern=[[n_pos_groups, _P_MAX], [1, n_g], [0, _GROUP]], offset=g0),  # [a, G, b]: e16[a, G]
                op0=nl.multiply,
                operand0=float(_P_MAX),
                op1=nl.add,
                operand1=gi_rows.ap(
                    pattern=[[_GROUP, _P_MAX], [0, n_g], [1, _GROUP]], offset=0
                ),  # [a, G, b]: 8b + a/16
                name=f"gi_gu_op{suffix}{run}",
            )

        def pair_indices() -> None:
            if pairs_gu:
                # Pair gathers (G, G + 1) for the pair starts G of the pipelines (odd with a dedupe head, else even):
                # gi_gu2[a, (16 G + b)] = 128 * e32[a, G] + 4b + a / 32, e32 = the expert of position 16 G + a % 32.
                starts = [g for g in groups if (g - pair_parity) % 2 == 0 and g + 1 < n_pos_groups]
                if starts:
                    s0, n_s = starts[0], len(starts)
                    nisa.tensor_tensor(
                        dst=picked32[:, 0:n_s, :],
                        # [a, P, j]: the expert of position 16 (s0 + 2P) + j
                        data1=ids_b.ap(pattern=[[N, _P_MAX], [2 * _GROUP, n_s], [1, 2 * _GROUP]], offset=s0 * _GROUP),
                        data2=sel32.ap(pattern=[[2 * _GROUP, _P_MAX], [0, n_s], [1, 2 * _GROUP]], offset=0),
                        op=nl.multiply,
                        engine=dve,
                        name=f"pick_e32{suffix}",
                    )
                    nisa.tensor_reduce(
                        dst=e32[:, 0:n_s], op=nl.add, data=picked32[:, 0:n_s, :], axis=2, name=f"e32_op{suffix}"
                    )
                    if dynamic:
                        pad_empty(e32[:, 0:n_s], f"e32_pad{suffix}", skip_empty)
                    nisa.scalar_tensor_tensor(
                        # [a, P, b]: columns 16 (s0 + 2P) + b, b < 32
                        dst=gi_gu2.ap(
                            pattern=[[n_pos_groups * _GROUP, _P_MAX], [2 * _GROUP, n_s], [1, 2 * _GROUP]],
                            offset=s0 * _GROUP,
                        ),
                        data=e32.ap(pattern=[[n_pos_groups // 2, _P_MAX], [1, n_s], [0, 2 * _GROUP]], offset=0),
                        op0=nl.multiply,
                        operand0=float(_P_MAX),
                        op1=nl.add,
                        operand1=rows32.ap(pattern=[[2 * _GROUP, _P_MAX], [0, n_s], [1, 2 * _GROUP]], offset=0),
                        name=f"gi_gu2_op{suffix}",
                    )

        def dn_singles(g0: int, n_g: int, run: str = "") -> nisa.NkiInstruction:
            # gi_dn[a, G] = dn_stride * e8[a, G] + a % 8 (rows a % 8 >= I/4 are padding rows, not used)
            down_index = nisa.tensor_scalar(
                dst=gi_dn[:, g0 : g0 + n_g],
                data=e8[:, g0 : g0 + n_g],
                op0=nl.multiply,
                operand0=float(dn_stride),
                op1=nl.add,
                operand1=part_f[:, 4:5],
                engine=dve,
                name=f"gi_dn_op{suffix}{run}",
            )
            return down_index

        def merged_indices() -> None:
            # Merged down gathers: pairs (G, G + 1) from the first group of `groups`; an odd last group keeps gi_dn.
            n_pairs_g = n_g // 2
            if n_pairs_g > 0:
                nisa.tensor_tensor(
                    dst=pick2[:, 0:n_pairs_g, :, :],
                    # [a, P, b, j]: the expert of position 16 (g0 + 2P) + j, for each b
                    data1=ids_b.ap(
                        pattern=[[N, _P_MAX], [2 * _GROUP, n_pairs_g], [0, 2], [1, 2 * _GROUP]], offset=g0 * _GROUP
                    ),
                    data2=sel2.ap(
                        pattern=[[4 * _GROUP, _P_MAX], [0, n_pairs_g], [2 * _GROUP, 2], [1, 2 * _GROUP]], offset=0
                    ),
                    op=nl.multiply,
                    engine=dve,
                    name=f"pick_pair{suffix}",
                )
                nisa.tensor_reduce(
                    dst=e_pair[:, 0:n_pairs_g, :],
                    op=nl.add,
                    data=pick2[:, 0:n_pairs_g, :, :],
                    axis=3,
                    name=f"e_pair_op{suffix}",
                )
                if dynamic:
                    pad_empty(e_pair[:, 0:n_pairs_g, :], f"e_pair_pad{suffix}", skip_empty)
                # gi_dn2[a, (g0 + 2P, b)] = dn_stride * e_pair[a, P, b] + (a / 2) % 8
                nisa.tensor_scalar(
                    dst=gi_dn2.ap(pattern=[[2 * n_pos_groups, _P_MAX], [4, n_pairs_g], [1, 2]], offset=2 * g0),
                    data=e_pair[:, 0:n_pairs_g, :],
                    op0=nl.multiply,
                    operand0=float(dn_stride),
                    op1=nl.add,
                    operand1=part_f[:, 6:7],
                    engine=dve,
                    name=f"gi_dn2_op{suffix}",
                )

        def tile_indices() -> None:
            for tile in tiles:
                _, tile_groups = scale_tiles[tile]
                first = tile_groups[0]
                # tile_pick[a, l] = e_sc[a, G_l] * (l == (a / 4) % 4); a missing l repeats the first group.
                cols = [g for g in tile_groups] + [first] * (_BLOCKS - len(tile_groups))
                contiguous = cols == list(range(first, first + _BLOCKS))
                if contiguous:
                    src = e_sc[:, first : first + _BLOCKS]
                else:
                    for l, g in enumerate(cols):
                        nisa.tensor_copy(
                            dst=tile_pick[:, tile, l : l + 1],
                            src=e_sc[:, g : g + 1],
                            engine=dve,
                            name=f"tile_cols{tile}_{l}",
                        )
                    src = tile_pick[:, tile, :]
                nisa.tensor_tensor(
                    dst=tile_pick[:, tile, :],
                    data1=src,
                    data2=sel_l,
                    op=nl.multiply,
                    engine=dve,
                    name=f"tile_pick{tile}",
                )
                nisa.tensor_reduce(
                    dst=gi_sc[:, tile : tile + 1], op=nl.add, data=tile_pick[:, tile, :], axis=1, name=f"gi_sc{tile}"
                )
                if expert_offset != 0:
                    nisa.tensor_scalar(
                        dst=gi_sc[:, tile : tile + 1],
                        data=gi_sc[:, tile : tile + 1],
                        op0=nl.add,
                        operand0=expert_offset,
                        engine=dve,
                        name=f"gi_sc_offset{tile}",
                    )
                if skip_scale_duplicates:
                    # partitions 32q + 16 .. 31 (never read by the MX scale tiles) get an index past the last scale row
                    nisa.tensor_tensor(
                        dst=gi_sc[:, tile : tile + 1],
                        data1=gi_sc[:, tile : tile + 1],
                        data2=sc_skip,
                        op=nl.add,
                        engine=dve,
                        name=f"gi_sc_skip{tile}",
                    )

        all_picks = ((sel16, e16, "e16"), (sel8, e8, "e8"), (sel_sc, e_sc, "esc"))
        if not index_first:
            picks(g0, n_g, all_picks)
            gu_singles(g0, n_g)
            pair_indices()
            down_index = dn_singles(g0, n_g)
            merged_indices()
            tile_indices()
            return down_index
        # dedupe_index_first: the indices of the first loads first (merged down pairs, gate/up pairs, the scale
        # tiles), then only the single-group indices that a gather reads.
        merged_indices()
        pair_indices()
        picks(g0, n_g, all_picks[2:])
        tile_indices()
        last = None
        for r, (first, count) in enumerate(group_runs(groups, single_gu)):
            picks(first, count, all_picks[0:1], f"_r{r}")
            gu_singles(first, count, f"_r{r}")
        for r, (first, count) in enumerate(group_runs(groups, single_dn)):
            picks(first, count, all_picks[1:2], f"_r{r}")
            last = dn_singles(first, count, f"_r{r}")
        return last

    # All stacked experts (E_rows): row 128e + p of expert e (gate/up), dn_stride e + p (down), e (down scale)
    gate_up_rows_hbm = expert_gate_up_packed.reshape((E_rows * _P_MAX, row_bytes))
    down_w_rows_hbm = expert_down_weights.reshape((E_rows * dn_stride, H))
    down_s_rows_hbm = expert_down_weights_scale.reshape((E_rows, n_scale_sets * H))  # [scale group 0 | 1]

    def run_groups(
        tag: str, groups: Sequence[int], tiles: Tuple, lead: int = 0, mode: str = "all"
    ) -> Optional[Callable[[], None]]:
        """Step 3: the group pipeline for the position groups `groups` (consecutive), from scratch into out_ps.

        mode "prefetch" emits only the first loads (scale tile 0, the gate/up rows of group 0, down pair 0) and
        returns: a dedupe block issues them before its loop, so its pipeline does not start with a load. mode
        "after_prefetch" is the rest of the pipeline (inside the loop). mode "start" (two_phase) emits the loads of
        the ring and the re-layout of group 0 (no read of hidden_qtz) and returns the group loop."""
        out_ps, gu_ps, tr_ps = tiles
        n = len(groups)
        q_last = [None, None]  # the ACT copy and the quantization of the last quantize() (quantize_first)
        pe_last, pool_last, sp_last, act_last = [None], [None], [None], [None]
        my_tiles = [s for s, (t, _) in enumerate(scale_tiles) if t == tag]
        down_units, tile_units = pipeline_units(n, lead, cut_of(tag))
        unit_of = {i: (u, i - unit[0]) for u, unit in enumerate(down_units) for i in unit}  # (unit, sub-slot)

        def slot_of(i: int) -> int:
            """The gate/up ring slot of group i (pair mode: the pairs fill slots 0 .. R - 1 from group `lead` on;
            a dedupe head group has slot R, or R - 2 with pair_head_in_ring)."""
            if pairs_gu:
                return (R - 2 if pair_head_in_ring else R) if i < lead else (i - lead) % R
            return i % R

        tile_of = {i: (t, i - tile[0]) for t, tile in enumerate(tile_units) for i in tile}  # (tile, place)

        def gather(dst: nl.NkiTensor, rows: nl.NkiTensor, index: nl.NkiTensor, label: str) -> None:
            """One batched Pool SWDGE gather: row 128b + a is HBM row index[a, b] (chained in Pool order)."""
            if gather_after[0] != None:
                # index[0, 0] += 0 * token on Pool (in the Pool order of the gathers): the gather reads the index after
                # the token is written
                token, zero = gather_after
                gather_after[0] = None
                nisa.tensor_scalar(
                    dst=zero, data=token[0:1, 0:1], op0=nl.multiply, operand0=0.0, engine=pool, name="pre_zero_op"
                )
                nisa.tensor_scalar(
                    dst=index[0:1, 0:1],
                    data=index[0:1, 0:1],
                    op0=nl.add,
                    operand0=zero,
                    engine=pool,
                    name="pre_after_op",
                )
            chain(
                nisa.dma_copy(
                    dst=dst,
                    src=rows.ap(
                        pattern=[[rows.shape[1], _P_MAX * index.shape[1]], [1, rows.shape[1]]],
                        offset=0,
                        vector_offset=index,
                        indirect_dim=0,
                    ),
                    dge_mode=nisa.dge_mode.swdge,
                    oob_mode=nisa.oob_mode.skip,
                    name=nm(label),
                ),
                pool_last,
            )

        # prefetch_load="contiguous": the HWDGE queue of the hot loads. SP in the standalone body (the ACT queue
        # issues a DMA in its instruction order, after ACT work of the head); ACT in the fused block (SP carries the
        # router weights).
        hot_queue = nisa.engine.sync if after_norm == None else nisa.engine.scalar
        hot_last = sp_last if after_norm == None else act_last

        def contiguous(G: int, n_groups: int) -> bool:
            """prefetch_load="contiguous": groups G .. G + n_groups - 1 are prefetch groups (HWDGE loads)."""
            return prefetch_load == "contiguous" and tag == "head" and G + n_groups <= prefetch_groups

        def load_rows_contiguous(base: int, G: int, n_groups: int) -> None:
            """The packed rows of the experts of positions 16G .. (contiguous in HBM: expert expert_offset + 16G + c)
            into ring slots from byte base: one HWDGE DMA on the ACT queue, [p, c, bytes] <- rows (e, p)."""
            n_exp, words = n_groups * _GROUP, row_bytes // 2  # uint16 words of a packed row
            chain(
                nisa.dma_copy(
                    dst=gu_rows.ap(pattern=[[gu_stride, _P_MAX], [words, n_exp], [1, words]], offset=base // 2),
                    src=expert_gate_up_packed.view(nl.uint16).ap(
                        pattern=[[words, _P_MAX], [_P_MAX * words, n_exp], [1, words]],
                        offset=(expert_offset + G * _GROUP) * _P_MAX * words,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=hot_queue,
                    name=nm(f"{tag}_load_gate_up_g{G}_hot"),
                ),
                hot_last,
            )

        def load_gate_up(i: int) -> None:
            """Step 3.1: the packed rows of the 16 slots of group i into ring slot slot_of(i): one Pool gather, or 16
            HWDGE DMAs (group 0 of the baseline: SP and ACT alternating; sp_group_period groups: SP). With
            gate_up_pair_gathers, the call for the second group of a pair loads both groups (one gather of 32 slots),
            and the call for the first group emits nothing. prefetch_load="contiguous": one HWDGE DMA for a prefetch
            group or pair."""
            G = groups[i]
            base = slot_of(i) * _GROUP * row_bytes
            if pairs_gu and i >= lead:
                first = lead + 2 * ((i - lead) // 2)
                if first + 1 < n:
                    if i == first:
                        return  # the pair is loaded with its second group
                    G = groups[first]
                    base = slot_of(first) * _GROUP * row_bytes
                    if contiguous(G, 2):
                        load_rows_contiguous(base, G, 2)
                        return
                    gather(
                        gu_rows.view(nl.uint8)[:, base : base + 2 * _GROUP * row_bytes],
                        gate_up_rows_hbm,
                        gi_gu2[:, G * _GROUP : (G + 2) * _GROUP],
                        f"{tag}_gather_gate_up_g{G}_pair",
                    )
                    return
            if contiguous(G, 1):
                load_rows_contiguous(base, G, 1)
                return
            baseline_first = tag == "base" and i == 0 and group0_queues != "pool"
            sp_group = sp_group_period > 0 and i % sp_group_period == sp_group_period - 1
            if not (baseline_first or sp_group):
                gather(
                    gu_rows.view(nl.uint8)[:, base : base + _GROUP * row_bytes],
                    gate_up_rows_hbm,
                    gi_gu[:, G * _GROUP : (G + 1) * _GROUP],
                    f"{tag}_gather_gate_up_g{G}",
                )
                return
            for c in range(_GROUP):
                n_pos = G * _GROUP + c
                on_sp = sp_group or group0_queues == "sp" or c % 2 == 0
                # The expert id of position n_pos: the router tile (baseline) or the compacted ids (dedupe).
                if dynamic:
                    id_src = ids_i32[T : T + 1, n_pos : n_pos + 1]
                elif expert_offset != 0:
                    id_src = index_off[n_pos // K : n_pos // K + 1, n_pos % K : n_pos % K + 1]
                else:
                    id_src = index_tk.view(nl.int32)[n_pos // K : n_pos // K + 1, n_pos % K : n_pos % K + 1]
                chain(
                    nisa.dma_copy(
                        dst=gu_rows.view(nl.uint8)[:, base + c * row_bytes : base + (c + 1) * row_bytes],
                        # [p, bytes]: the 128 packed rows of the expert of position n_pos
                        src=expert_gate_up_packed.ap(
                            pattern=[[row_bytes, _P_MAX], [1, row_bytes]],
                            offset=0,
                            scalar_offset=nisa.scalar_load(id_src),
                            indirect_dim=0,
                        ),
                        dge_mode=nisa.dge_mode.hwdge,
                        engine=nisa.engine.sync if on_sp else nisa.engine.scalar,
                        name=nm(f"{tag}_load_gate_up_g{G}_c{c}"),
                    ),
                    sp_last if on_sp else act_last,
                )

        def load_down(u: int) -> None:
            """Step 3.1: the 8 down rows of the 16 slots of down unit u (one Pool gather: 256 rows for 2 groups, 128
            for 1) into down pair slot u % (Rd / 2)."""
            unit = down_units[u]
            G = groups[unit[0]]
            col = (u % ring_units) * unit_cols
            if contiguous(G, len(unit)):
                # prefetch_load="contiguous": partition m of group G + s holds down row m % 8 of slot m / 8, that is
                # HBM row 8 * (expert_offset + 16 (G + s)) + m (the merged layout too): one DMA per group
                for s_g in range(len(unit)):
                    first_row = _MX_ROWS * (expert_offset + _GROUP * (G + s_g))
                    chain(
                        nisa.dma_copy(
                            dst=dn_w[:, col + s_g * H : col + (s_g + 1) * H],
                            src=down_w_rows_hbm[first_row : first_row + _P_MAX, :],
                            dge_mode=nisa.dge_mode.hwdge,
                            engine=hot_queue,
                            name=nm(f"{tag}_load_down_g{G + s_g}_hot"),
                        ),
                        hot_last,
                    )
                return
            if len(unit) == 2:
                gather(
                    dn_w[:, col : col + 2 * H], down_w_rows_hbm, gi_dn2[:, 2 * G : 2 * G + 2], f"{tag}_gather_down_g{G}"
                )
            else:
                gather(dn_w[:, col : col + H], down_w_rows_hbm, gi_dn[:, G : G + 1], f"{tag}_gather_down_g{G}")

        def load_scales(t: int) -> None:
            """Step 3.1: the down-scale tile t of the pipeline into ring slot t % 2 (one Pool gather; G * H bytes
            per row: the scale groups of the expert)."""
            gather(
                dn_s[:, (t % 2) * tile_cols : (t % 2 + 1) * tile_cols],
                down_s_rows_hbm,
                gi_sc[:, my_tiles[t] : my_tiles[t] + 1],
                f"{tag}_gather_down_s{t}",
            )

        def relayout(i: int) -> None:
            """Step 3.2: packed rows of group i (row slot i % R) -> moving slot i % Rm. ACT copies H512 blocks
            0 .. 3 of the gate and of the up weights and the scale segments, DVE the other blocks."""
            G = groups[i]
            rows_base = slot_of(i) * _GROUP * row_bytes // 2
            moving_base = (i % Rm) * _GROUP * gate_up_cols
            for gate_or_up in range(2):
                for first_block, n_blocks, engine in (
                    (0, _ACT_RELAYOUT_BLOCKS, act),
                    (_ACT_RELAYOUT_BLOCKS, n_h512 - _ACT_RELAYOUT_BLOCKS, dve),
                ):
                    copy = nisa.tensor_copy(
                        # [p, hb, c, I]: moving slot, gate_or_up columns of blocks first_block ..
                        dst=gu_moving.ap(
                            pattern=[
                                [Rm * _GROUP * gate_up_cols, _P_MAX],
                                [group_cols, n_blocks],
                                [I, _GROUP],
                                [1, I],
                            ],
                            offset=moving_base + (gate_or_up * n_h512 + first_block) * group_cols,
                        ),
                        # [p, hb, c, I]: packed row [gate_or_up][hb][I] of each slot
                        src=gu_rows.ap(
                            pattern=[
                                [gu_stride, _P_MAX],
                                [I, n_blocks],
                                [row_bytes // 2, _GROUP],
                                [1, I],
                            ],
                            offset=rows_base + (gate_or_up * n_h512 + first_block) * I,
                        ),
                        engine=engine,
                        name=nm(f"{tag}_relayout_{('gate', 'up')[gate_or_up]}_g{G}_b{first_block}"),
                    )
                    after = q_last[0] if engine is act else q_last[1]
                    if quantize_first and after != None:
                        copy.depends_on(after, kind=nisa.dependency.schedule)
                    if after_latent[0] != None and engine is dve and tag == "head" and i == 1:
                        copy.depends_on(after_latent[0], kind=nisa.dependency.schedule)
            for segment in range(n_segments):
                copy = nisa.tensor_copy(
                    # [p, hb % 4, c, I/2 words]: moving scale slot, segment `segment`
                    dst=gu_moving_s.ap(
                        pattern=[
                            [Rm * n_segments * _BLOCKS * group_cols // 2, _P_MAX],
                            [group_cols // 2, _BLOCKS],
                            [I // 2, _GROUP],
                            [1, I // 2],
                        ],
                        offset=((i % Rm) * n_segments + segment) * _BLOCKS * group_cols // 2,
                    ),
                    # [p, hb % 4, c, I/2 words]: scale segment `segment` of each slot, after its weights
                    src=gu_rows.ap(
                        pattern=[
                            [gu_stride, _P_MAX],
                            [I // 2, _BLOCKS],
                            [row_bytes // 2, _GROUP],
                            [1, I // 2],
                        ],
                        offset=rows_base + gate_up_cols + segment * _BLOCKS * I // 2,
                    ),
                    engine=act,
                    name=nm(f"{tag}_relayout_scales_g{G}_s{segment}"),
                )
                if quantize_first and q_last[0] != None:
                    copy.depends_on(q_last[0], kind=nisa.dependency.schedule)

        units = [(hb, gate_or_up) for hb in range(n_h512) for gate_or_up in range(2)]

        def gate_up(i: int, unit_range: Tuple[int, int]) -> None:
            """Step 3.3: gate and up matmuls of group i for units unit_range (H512 block hb, gate_or_up)."""
            G = groups[i]
            for hb, gate_or_up in units[unit_range[0] : unit_range[1]]:
                weight_col = (i % Rm) * _GROUP * gate_up_cols + (gate_or_up * n_h512 + hb) * group_cols
                # Scales of chunk k = gate_or_up * n_chunks + hb / 4: segment k / 4, partition offset 4 (k % 4).
                scale_chunk = gate_or_up * n_chunks + hb // _BLOCKS
                scale_col = ((i % Rm) * n_segments + scale_chunk // _X4) * _BLOCKS * group_cols + (
                    hb % _BLOCKS
                ) * group_cols
                chain(
                    nisa.nc_matmul_mx(
                        dst=gu_ps[i % 2][0:T, gate_or_up * _PSUM_COLS : gate_or_up * _PSUM_COLS + group_cols],
                        stationary=hidden_qtz[:, hb * T : (hb + 1) * T],  # [p, t]: block hb of all T tokens
                        moving=gu_moving.view(nl.float4_e2m1fn_x4)[:, weight_col : weight_col + group_cols],
                        stationary_scale=hidden_scale[:, hb * T : (hb + 1) * T],
                        moving_scale=gu_moving_s.view(nl.uint8)[
                            _X4 * (scale_chunk % _X4) :, scale_col : scale_col + group_cols
                        ],
                        accumulate=hb > 0,
                        name=nm(f"{tag}_mm_{('gate', 'up')[gate_or_up]}_g{G}_b{hb}"),
                    ),
                    pe_last,
                )

        def swiglu(i: int) -> None:
            """Step 3.4: gated[t, (q, c, p)] = SiLU(gate) * a[t, slot c] * up of i = q*(I/4) + p, in bf16."""
            G = groups[i]
            gate_col = (i % act_slots) * group_cols
            if situ != None:
                situ_activation(i)
                return
            nisa.activation(
                dst=gate_act[:, gate_col : gate_col + group_cols],
                op=nl.silu,
                data=gu_ps[i % 2][0:T, 0:group_cols],  # [t, (c, I)]: gate
                name=nm(f"{tag}_silu_g{G}"),
            )
            nisa.tensor_tensor(
                dst=gate_act.ap(pattern=[[act_cols, T], [I, _GROUP], [1, I]], offset=gate_col),  # [t, c, I]
                data1=gate_act.ap(pattern=[[act_cols, T], [I, _GROUP], [1, I]], offset=gate_col),
                # [t, c, I]: the affinity of token t for the slot at position 16G + c
                data2=aff_cols.ap(pattern=[[N, T], [1, _GROUP], [0, I]], offset=G * _GROUP),
                op=nl.multiply,
                engine=dve,
                name=nm(f"{tag}_mul_aff_g{G}"),
            )
            nisa.tensor_tensor(
                # [t, q, c, p]: gated slot of the group, rows p < I/4
                dst=gated.ap(
                    pattern=[[2 * _X4 * _P_MAX, T], [_P_MAX, _X4], [_MX_ROWS, _GROUP], [1, down_rows]],
                    offset=(i % 2) * _X4 * _P_MAX,
                ),
                # [t, q, c, p]: a * SiLU(gate) of slot c, i = q*(I/4) + p
                data1=gate_act.ap(
                    pattern=[[act_cols, T], [down_rows, _X4], [I, _GROUP], [1, down_rows]], offset=gate_col
                ),
                # [t, q, c, p]: up of slot c, i = q*(I/4) + p
                data2=gu_ps[i % 2].ap(
                    pattern=[[2 * _PSUM_COLS, T], [down_rows, _X4], [I, _GROUP], [1, down_rows]], offset=_PSUM_COLS
                ),
                op=nl.multiply,
                engine=dve,
                name=nm(f"{tag}_mul_up_g{G}"),
            )

        def situ_activation(i: int) -> None:
            """Step 3.4 with situ: gated[t, (q, c, p)] = lb*tanh(u/lb) * beta*tanh(g/beta) * sigmoid(g) * a[t, slot c]
            in fp32, written as bf16 (ACT: the tanh and sigmoid terms into fp32 tiles; DVE: the products)."""
            beta, linear_beta = situ
            G = groups[i]
            col = (i % act_slots) * group_cols  # slot of the group in gate_act, situ_sigmoid and situ_up
            gate_ps = gu_ps[i % 2][0:T, 0:group_cols]  # [t, (c, I)]: gate
            up_ps = gu_ps[i % 2][0:T, _PSUM_COLS : _PSUM_COLS + group_cols]  # [t, (c, I)]: up
            gate_term, up_term = gate_act[:, col : col + group_cols], situ_up[:, col : col + group_cols]
            sigmoid_term = situ_sigmoid[:, col : col + group_cols]
            nisa.activation(dst=gate_term, op=nl.tanh, data=gate_ps, scale=1.0 / beta, name=nm(f"{tag}_tanh_gate_g{G}"))
            nisa.activation(
                dst=up_term, op=nl.tanh, data=up_ps, scale=1.0 / linear_beta, name=nm(f"{tag}_tanh_up_g{G}")
            )
            # [t, c, I] of a gate_act slot, and the affinity of token t for the slot at position 16G + c
            slot_view = gate_act.ap(pattern=[[act_cols, T], [I, _GROUP], [1, I]], offset=col)
            aff_view = aff_cols.ap(pattern=[[N, T], [1, _GROUP], [0, I]], offset=G * _GROUP)
            # [t, q, c, p]: slot c, i = q*(I/4) + p, of a gate_act / situ_up slot, and of the gated slot of the group
            pairs = [[act_cols, T], [down_rows, _X4], [I, _GROUP], [1, down_rows]]
            gated_pairs = gated.ap(
                pattern=[[2 * _X4 * _P_MAX, T], [_P_MAX, _X4], [_MX_ROWS, _GROUP], [1, down_rows]],
                offset=(i % 2) * _X4 * _P_MAX,
            )
            if sigmoid_from_tanh:
                # 2 sigmoid(g) = 1 + tanh(g/2): gate_term = (1 + tanh(g/2)) * tanh(g/beta) * a; beta*lb/2 goes to up.
                nisa.activation(
                    dst=sigmoid_term, op=nl.tanh, data=gate_ps, scale=0.5, name=nm(f"{tag}_tanh_half_gate_g{G}")
                )
                nisa.scalar_tensor_tensor(
                    dst=gate_term,
                    data=sigmoid_term,
                    op0=nl.add,
                    operand0=1.0,
                    op1=nl.multiply,
                    operand1=gate_term,
                    name=nm(f"{tag}_situ_gate_g{G}"),
                )
                up_factor = linear_beta * beta / 2.0  # with the affinity: (gate_term * up_factor) * a
            else:
                nisa.activation(dst=sigmoid_term, op=nl.sigmoid, data=gate_ps, name=nm(f"{tag}_sigmoid_gate_g{G}"))
                # gate_term = tanh(g/beta) * (beta*lb) * sigmoid(g)
                nisa.scalar_tensor_tensor(
                    dst=gate_term,
                    data=gate_term,
                    op0=nl.multiply,
                    operand0=beta * linear_beta,
                    op1=nl.multiply,
                    operand1=sigmoid_term,
                    name=nm(f"{tag}_situ_gate_g{G}"),
                )
                up_factor = None
            if up_factor == None:
                nisa.tensor_tensor(
                    dst=slot_view,
                    data1=slot_view,
                    data2=aff_view,
                    op=nl.multiply,
                    engine=dve,
                    name=nm(f"{tag}_mul_aff_g{G}"),
                )
            else:
                nisa.scalar_tensor_tensor(
                    dst=slot_view,
                    data=slot_view,
                    op0=nl.multiply,
                    operand0=up_factor,
                    op1=nl.multiply,
                    operand1=aff_view,
                    name=nm(f"{tag}_mul_aff_g{G}"),
                )
            # gated = tanh(u/lb) * gate_term (tensor_tensor: the 3 free dimensions of the gated layout)
            nisa.tensor_tensor(
                dst=gated_pairs,
                data1=situ_up.ap(pattern=pairs, offset=col),
                data2=gate_act.ap(pattern=pairs, offset=col),
                op=nl.multiply,
                engine=dve,
                name=nm(f"{tag}_mul_up_g{G}"),
            )

        def transpose(i: int) -> None:
            """Step 3.5: tr_ps[(c, p), (q, t)] = gated[t, (q, c, p)] (4 PE transposes, identity moving)."""
            G = groups[i]
            for q in range(_X4):
                col = (i % 2) * _X4 * T + q * T
                chain(
                    nisa.nc_matmul(
                        dst=tr_ps[:, col : col + T],  # [(c, p), t] of lane q
                        stationary=gated[:, ((i % 2) * _X4 + q) * _P_MAX : ((i % 2) * _X4 + q + 1) * _P_MAX],
                        moving=ident_bf16,
                        is_moving_onezero=True,
                        is_transpose=True,
                        accumulate=False,
                        name=nm(f"{tag}_transpose_g{G}_q{q}"),
                    ),
                    pe_last,
                )

        def quantize(i: int) -> None:
            """Step 3.5: qmx_in[m, (s, term, t, q)] = hi / lo of the transposed values (two scale groups: times the
            group mask of partition m), then MXFP8 into inter[i % Rs]."""
            G = groups[i]
            slot3 = i % Rs
            hi_first = slot3 * qmx_slot
            lo_first = hi_first + T * _X4
            for s in range(n_scale_sets):
                # two scale groups: ACT keeps the rows of group s (the mask column as a per-partition scale)
                mask_scale = {} if n_scale_sets == 1 else {"scale": group_mask[:, s : s + 1]}
                q_last[0] = nisa.activation(
                    # [m, t, q]: the lo block of scale group s of the slot, q innermost
                    dst=qmx_in.ap(
                        pattern=[[Rs * qmx_slot, _P_MAX], [_X4, T], [1, _X4]], offset=lo_first + s * qmx_cols
                    ),
                    op=nl.copy,
                    # [m, t, q]: tr_ps[m, q*T + t]
                    data=tr_ps.ap(pattern=[[2 * _X4 * T, _P_MAX], [1, T], [T, _X4]], offset=(i % 2) * _X4 * T),
                    name=nm(f"{tag}_qmx_copy_g{G}" + (f"_s{s}" if n_scale_sets > 1 else "")),
                    **mask_scale,
                )

            def terms(view: nl.NkiTensor, first: int) -> nl.NkiTensor:
                """[m, (s, t, q)]: the hi (or lo) block of every scale group of the slot, from column first."""
                if n_scale_sets == 1:
                    return view[:, first : first + T * _X4]
                return view.ap(pattern=[[Rs * qmx_slot, _P_MAX], [qmx_cols, n_scale_sets], [1, T * _X4]], offset=first)

            nisa.tensor_scalar(
                dst=terms(qmx_in.view(nl.int16), hi_first),
                data=terms(qmx_in.view(nl.int16), lo_first),
                op0=nl.bitwise_and,
                operand0=_E4M3_BF16_MASK,
                engine=dve,
                name=nm(f"{tag}_qmx_hi_g{G}"),
            )
            nisa.tensor_tensor(
                dst=terms(qmx_in, lo_first),
                data1=terms(qmx_in, lo_first),
                data2=terms(qmx_in, hi_first),
                op=nl.subtract,
                engine=dve,
                name=nm(f"{tag}_qmx_lo_g{G}"),
            )
            # one quantization per scale group: its term columns follow its zero block (the dst must be contiguous)
            for s in range(n_scale_sets):
                stat_first = slot3 * stat_slot + s * set_cols + term_first  # the hi columns of scale group s
                q_first = slot3 * qmx_slot + s * qmx_cols
                q_last[1] = nisa.quantize_mx(
                    dst=inter[:, stat_first : stat_first + two_t],
                    src=qmx_in[:, q_first : q_first + qmx_cols],
                    dst_scale=inter_s[:, stat_first : stat_first + two_t],
                    name=nm(f"{tag}_qmx_g{G}" + (f"_s{s}" if n_scale_sets > 1 else "")),
                )

        def down(i: int) -> None:
            """Step 3.6: out_ps[(term, t), h] += stationary[(c, p), (term, t)] x down rows [(c, p), h] (fp32 down
            PSUM: per 512-column chunk, output half 1 then 0, then each scale group with its own stationary and
            scales)."""
            G = groups[i]
            slot3 = i % Rs
            tile, place = tile_of[i]
            scale_slot, scale_offset = tile % 2, _X4 * place
            unit, sub = unit_of[i]
            down_col = (unit % ring_units) * unit_cols + sub * H  # group i in its down unit slot
            if down_fp32:
                for chunk in range(div_ceil(half_cols, _DOWN_CHUNK)):
                    c0 = chunk * _DOWN_CHUNK
                    width = min(_DOWN_CHUNK, half_cols - c0)  # 512, or 256 for the last chunk at H = 3584
                    for half in (1, 0):
                        h0 = half * half_cols + c0  # the first H column of the chunk
                        rows = (half + 1) * two_t  # half 1: its zero block adds +0 to the rows of half 0
                        for s in range(n_scale_sets):
                            last = (slot3 * n_scale_sets + s + 1) * set_cols  # the end of scale group s's stationary
                            scale_col = scale_slot * tile_cols + s * H + h0
                            chain(
                                nisa.nc_matmul_mx(
                                    dst=out_ps[0:rows, c0 : c0 + width],  # [(half, term, t), h]: one PSUM bank
                                    # [(c, p), (zeros, term, t)]: from the zero block (half 1) or the hi columns
                                    stationary=inter[:, last - rows : last],
                                    moving=dn_w[:, down_col + h0 : down_col + h0 + width],  # [(c, p), h]
                                    stationary_scale=inter_s[:, last - rows : last],
                                    # scale group s of the tile slot, from partition 4 * (the group's place)
                                    moving_scale=dn_s[scale_offset:, scale_col : scale_col + width],
                                    accumulate=i > 0 or half == 0 or s > 0,
                                    name=nm(
                                        f"{tag}_mm_down_g{G}_h{half}_c{chunk}" + (f"_s{s}" if n_scale_sets > 1 else "")
                                    ),
                                ),
                                pe_last,
                            )
                return
            for chunk in range(H // _DOWN_CHUNK):
                c0 = chunk * _DOWN_CHUNK
                chain(
                    nisa.nc_matmul_mx(
                        dst=out_ps[0:two_t, c0 : c0 + _DOWN_CHUNK],
                        stationary=inter[:, slot3 * two_t : (slot3 + 1) * two_t],  # [(c, p), (term, t)]
                        moving=dn_w[:, down_col + c0 : down_col + c0 + _DOWN_CHUNK],  # [(c, p), 512]
                        stationary_scale=inter_s[:, slot3 * two_t : (slot3 + 1) * two_t],
                        # scale rows of the group: tile slot, from partition 4 * (its place in the tile)
                        moving_scale=dn_s[scale_offset:, scale_slot * H + c0 : scale_slot * H + c0 + _DOWN_CHUNK],
                        accumulate=i > 0,
                        name=nm(f"{tag}_mm_down_g{G}_c{chunk}"),
                    ),
                    pe_last,
                )

        # Loads ahead of the compute (program order): the Pool stream in PE consumption order.
        n_units = len(down_units)
        # The gate/up loads of a prefetch: group 0, or with pair gathers the pair (0, 1) (its load is the call for 1).
        prefetched = (0, 1) if pairs_gu and lead == 0 else (0,)
        if mode == "prefetch":
            load_scales(0)
            load_gate_up(prefetched[-1] if n > 1 else 0)
            load_down(0)
            return
        # scales_after_gate_up: the first scale tile follows the gate/up gathers of groups 0 and 1
        first_scales = min(1, n - 1) if scales_after_gate_up else -1
        for i in range(min(R, n)):
            if i == 0 and mode != "after_prefetch" and first_scales < 0:
                load_scales(0)
            if not (i in prefetched and mode == "after_prefetch"):
                load_gate_up(i)
            if i == first_scales and mode != "after_prefetch":
                load_scales(0)
            u, sub = unit_of[i]
            if sub == 0 and u < min(ring_units, n_units) and not (u == 0 and mode == "after_prefetch"):
                load_down(u)
        for u in range(unit_of[min(R, n) - 1][0] + 1, min(ring_units, n_units)):
            load_down(u)
        for t in range(1, min(2, len(my_tiles))):
            load_scales(t)
        relayout(0)
        if R < n:
            load_gate_up(R)  # relayout(0) frees row slot 0
        # down_lag 1: the transposes of group i - 1 after 2 gate/up units of group i, then down(i - 1). down_lag 2: the
        # transposes after all of gate_up(i), then down(i - 2), so the SwiGLU and MXFP8 chains of a group have a group
        # of PE work to hide behind.
        split = _FINISH_AFTER_UNITS if down_lag == 1 else len(units)

        def down_and_loads(j: int) -> None:
            """down(j), then the loads it frees: its down pair slot (after the pair's second group) and, after a
            tile's last group, the scale slot."""
            down(j)
            u, sub = unit_of[j]
            if sub == len(down_units[u]) - 1 and u + ring_units < n_units:
                load_down(u + ring_units)
            tile, place = tile_of[j]
            if place == len(tile_units[tile]) - 1 and tile + 2 < len(my_tiles):
                load_scales(tile + 2)

        # head_finish_first: the dedupe head finishes its group 0 (SwiGLU, transposes, MXFP8, down) before the
        # re-layout of group 1, so the PE runs them while it waits for the rows of group 1.
        early = head_finish_first and lead == 1 and n > 1

        def pipeline() -> None:
            """The group loop, from gate_up(0) (the first read of hidden_qtz / hidden_scale)."""
            for step in range(n + down_lag):
                i, prev, late = step, step - 1, step - down_lag
                if i < n:
                    gate_up(i, (0, split))
                if 0 <= prev < n and not (early and prev == 0):
                    transpose(prev)
                if i < n and split < len(units):
                    gate_up(i, (split, len(units)))
                if 0 <= prev < n and not (early and prev == 0):
                    quantize(prev)
                if early and i == 0:
                    swiglu(0)
                    transpose(0)
                    quantize(0)
                    down_and_loads(0)
                if i + 1 < n:
                    relayout(i + 1)
                    # The relayout frees row slot (i + 1) % R for group i + 1 + R.
                    if i + 1 + R < n:
                        load_gate_up(i + 1 + R)
                if i < n and not (early and i == 0):
                    swiglu(i)
                if late >= 0 and not (early and late == 0):
                    down_and_loads(late)

        if mode == "start":
            return pipeline  # two_phase: the caller runs the loop after the latent quantization
        pipeline()

    def store_groups(n_chunks: int) -> List[List[int]]:
        """The static store DMAs of fold_dma: runs of fold_store_cols / 512 chunks, in column order, or with
        fold_store_order="ready" in the order their staging copies end (DVE copies chunk c of the last down matmuls,
        which end at a fixed interval, for even c, ACT the odd c, each engine at its own fixed copy time)."""
        per = fold_store_cols // _FOLD_CHUNK
        groups = [list(range(c, min(c + per, n_chunks))) for c in range(0, n_chunks, per)]
        if fold_store_order == "columns":
            return groups
        free, ready = {0: 0.0, 1: 0.0}, []
        for chunk in range(n_chunks):
            engine = chunk % 2
            free[engine] = max(0.22 * (chunk + 1), free[engine]) + (0.38 if engine == 0 else 0.64)
            ready.append(free[engine])
        return sorted(groups, key=lambda group: (max(ready[c] for c in group), group[0]))

    def fold_and_store(
        source: nl.NkiTensor,
        staging: Optional[nl.NkiTensor],
        tiles: List[nl.NkiTensor],
        add: bool = False,
        stores: Optional[List] = None,
    ) -> List:
        """Step 5: per 512 columns, a bf16 matmul with fold_sel adds the hi and lo rows of each token into PSUM
        tiles[chunk % 2], then out_sb gets the bf16 result and the output rows are stored.

        source: the [2T, H] output terms (out_ps, out_acc or fold_full). staging: the SBUF tile that each chunk is
        copied into before its fold: fold_in (a ring of 2 chunks), fold_full (the same columns), or None (the fold
        reads source, a bf16 SBUF tile). add: DVE adds each chunk of source to the same columns of staging (fold_full)
        instead of the copy. Returns the last copy to out_sb (in a list). With fold_dma, the store DMA of each chunk
        adds the hi and lo rows of staging (in fp32, one rounding to bf16), so there is no fold matmul and no out_sb
        copy; returns the last staging copy. A fold inside a dedupe block (a loop body) keeps the fold matmul: the
        compiler does not take a DMA compute there. stores: a list that gets the store DMAs of fold_dma.
        With fold_store_order="ready" or fold_store_cols > 512, the stores follow all staging copies (store_groups)."""
        dma_fold = fold_dma and not in_loop[0]
        kernel_assert(not dma_fold or staging != None, "fold_dma needs a staging tile")
        deferred = dma_fold and (fold_store_order != "columns" or fold_store_cols != _FOLD_CHUNK)
        kernel_assert(not deferred or staging.shape[1] == H, "fold_store_order / fold_store_cols need fold_full")
        pe_fold = [None]
        last = None
        for chunk in range(H // _FOLD_CHUNK):
            c0, half = chunk * _FOLD_CHUNK, (chunk % 2) * _FOLD_CHUNK
            moving = source[:, c0 : c0 + _FOLD_CHUNK]
            if staging != None:
                col = half if staging.shape[1] == 2 * _FOLD_CHUNK else c0
                moving = staging[:, col : col + _FOLD_CHUNK]
                if add:
                    staged = nisa.tensor_tensor(
                        dst=moving, data1=source[:, c0 : c0 + _FOLD_CHUNK], data2=moving, op=nl.add, engine=dve
                    )
                elif chunk % 2 == 0:
                    staged = nisa.tensor_copy(
                        dst=moving, src=source[:, c0 : c0 + _FOLD_CHUNK], engine=dve, name=nm(f"fold_in_c{chunk}")
                    )
                else:
                    staged = nisa.activation(
                        dst=moving, op=nl.copy, data=source[:, c0 : c0 + _FOLD_CHUNK], name=nm(f"fold_in_c{chunk}")
                    )
            if dma_fold:
                last = staged
                if deferred:
                    continue
                # [t, h] = hi[t, h] + lo[t, h] in the store DMA: rows 0 .. T - 1 and T .. 2T - 1 of the staging chunk
                rows = T if valid_tokens == None else valid_tokens
                store = nisa.dma_compute(
                    dst=output[output_row : output_row + rows, c0 : c0 + _FOLD_CHUNK],
                    srcs=[moving[0:rows, :], moving[T : T + rows, :]],
                    reduce_op=nl.add,
                    name=nm(f"store_fold_c{chunk}"),
                )
                if stores != None:
                    stores.append(store)
                continue
            chain(
                nisa.nc_matmul(
                    dst=tiles[chunk % 2],  # [t, h] = hi[t, h] + lo[t, h]
                    stationary=fold_sel,
                    moving=moving,
                    is_stationary_onezero=True,
                    accumulate=False,  # the PSUM tile was written before (chunk - 2)
                    name=nm(f"mm_fold_c{chunk}"),
                ),
                pe_fold,
            )
            if chunk % 2 == 0:
                last = nisa.activation(
                    dst=out_sb[:, c0 : c0 + _FOLD_CHUNK],
                    op=nl.copy,
                    data=tiles[chunk % 2],
                    name=nm(f"fold_out_c{chunk}"),
                )
            else:
                last = nisa.tensor_copy(
                    dst=out_sb[:, c0 : c0 + _FOLD_CHUNK],
                    src=tiles[chunk % 2],
                    engine=dve,
                    name=nm(f"fold_out_c{chunk}"),
                )
            stored = out_sb[:, c0 : c0 + _FOLD_CHUNK]  # the stored tokens: all T, or 0 .. valid_tokens - 1
            if valid_tokens != None:
                stored = out_sb[0:valid_tokens, c0 : c0 + _FOLD_CHUNK]
            nisa.dma_copy(
                dst=output[output_row : output_row + stored.shape[0], c0 : c0 + _FOLD_CHUNK],
                src=stored,
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=nm(f"store_out_c{chunk}"),
            )
        if deferred:
            # The store DMAs after all staging copies: [t, h] = hi[t, h] + lo[t, h] over the columns of each group.
            # "ready": each store is scheduled after the previous one (the scheduler orders them by column otherwise).
            rows = T if valid_tokens == None else valid_tokens
            previous = None
            for group in store_groups(H // _FOLD_CHUNK):
                c0, c1 = group[0] * _FOLD_CHUNK, (group[-1] + 1) * _FOLD_CHUNK
                store = nisa.dma_compute(
                    dst=output[output_row : output_row + rows, c0:c1],
                    srcs=[staging[0:rows, c0:c1], staging[T : T + rows, c0:c1]],
                    reduce_op=nl.add,
                    name=nm(f"store_fold_c{group[0]}"),
                )
                if fold_store_order == "ready" and previous != None:
                    store.depends_on(previous, kind=nisa.dependency.schedule)
                previous = store
                if stores != None:
                    stores.append(store)
        return [last]

    def evict_fp32(source: nl.NkiTensor, label: str, chunks: Optional[Sequence[int]] = None) -> List:
        """fp32 down PSUM: out_acc[:, chunk] = source[:, chunk] for the 512-column chunks (one PSUM bank each; DVE
        the even chunks, ACT the odd ones). chunks None: two copies, DVE banks 2-3 and ACT banks 4-5 (no PSUM bank
        is read by both). Returns the copies."""
        if chunks == None:
            split = min(half_cols, 2 * _DOWN_CHUNK)
            spans = [(0, split, dve), (split, half_cols, act)] if split < half_cols else [(0, split, dve)]
            label_of = lambda j: f"{label}_evict_b{j}"  # noqa: E731
        else:
            spans = [(c * _DOWN_CHUNK, min(half_cols, (c + 1) * _DOWN_CHUNK), act if c % 2 else dve) for c in chunks]
            label_of = lambda j: f"{label}_evict_c{chunks[j]}"  # noqa: E731
        done = []
        for j, (first, stop, engine) in enumerate(spans):
            if engine == dve:
                done.append(
                    nisa.tensor_copy(
                        dst=out_acc[:, first:stop], src=source[:, first:stop], engine=dve, name=nm(label_of(j))
                    )
                )
            else:
                done.append(
                    nisa.activation(
                        dst=out_acc[:, first:stop], op=nl.copy, data=source[:, first:stop], name=nm(label_of(j))
                    )
                )
        return done

    def store_fp32(chunks: Sequence[int]) -> None:
        """Step 5 (fp32 down PSUM): per output half and 512-column chunk of out_acc, the store DMA adds the hi and lo
        rows of each token (fp32, then one rounding to bf16)."""
        rows = T if valid_tokens == None else valid_tokens
        for chunk in chunks:
            c0 = chunk * _DOWN_CHUNK
            width = min(_DOWN_CHUNK, half_cols - c0)
            for half in range(2):
                hi = half * two_t  # the hi rows of the output half; its lo rows follow at hi + T
                nisa.dma_compute(
                    dst=output[output_row : output_row + rows, half * half_cols + c0 : half * half_cols + c0 + width],
                    srcs=[out_acc[hi : hi + rows, c0 : c0 + width], out_acc[hi + T : hi + T + rows, c0 : c0 + width]],
                    reduce_op=nl.add,
                    name=nm(f"store_out_h{half}_c{chunk}"),
                )

    # Step 2: Routing of the slot positions.
    if not dynamic:
        # Step 2.1: aff_cols[t', (t, k)] = affinity[t', k] where t' == t; ids_b[a, n] = index of pair n.
        after_norm_op(
            nisa.iota(dst=eye_tk, pattern=[[1, T], [0, K]], offset=0, channel_multiplier=-1, name="iota_eye_tk")
        )
        after_norm_op(
            nisa.tensor_scalar(dst=eye_tk, data=eye_tk, op0=nl.equal, operand0=0.0, engine=dve, name="eye_tk_op")
        )
        after_norm_op(nisa.memset(dst=ones_tokens, value=1.0, engine=dve, name="memset_ones_tokens"))
        nisa.tensor_tensor(
            dst=index_moving,
            data1=index_tk.ap(pattern=[[K, T], [0, T], [1, K]], offset=0),  # [t', (t, k)]: index[t', k]
            data2=eye_tk,
            op=nl.multiply,
            engine=dve,
            name="index_block_diag",
        )
        ids_ps = nl.ndarray((_P_MAX, N), dtype=nl.float32, buffer=nl.psum, address=(0, bank7), name="ids_ps")
        nisa.nc_matmul(dst=ids_ps, stationary=ones_tokens, moving=index_moving, accumulate=False, name="mm_ids_bcast")
        nisa.tensor_copy(dst=ids_b, src=ids_ps, engine=dve, name="ids_b_copy")
        if expert_offset != 0:
            nisa.tensor_scalar(
                dst=index_off,
                data=index_tk.view(nl.int32),
                op0=nl.add,
                operand0=expert_offset,
                engine=dve,
                name="index_offset_op",
            )
        gather_indices(list(range(n_pos_groups)), list(range(n_scale_tiles)), "")
        nisa.tensor_tensor(
            dst=aff_cols,
            data1=affinity_tk.ap(pattern=[[K, T], [0, T], [1, K]], offset=0),  # [t', (t, k)]: affinity[t', k]
            data2=eye_tk,
            op=nl.multiply,
            engine=dve,
            name="aff_block_diag",
        )
        out_ps, _, _ = base_tiles = psum_tiles("")
        tag, groups, lead = pipelines[0]
        run_groups(tag, groups, base_tiles, lead=lead)
        fold_source = None if down_fp32 else out_ps
        if down_fp32:
            # Step 5: each chunk leaves PSUM, then its store DMAs add the hi and lo rows.
            for chunk in range(div_ceil(half_cols, _DOWN_CHUNK)):
                evict_fp32(out_ps, "out", [chunk])
                store_fp32([chunk])
    else:
        # Step 2.2: The head positions 0..15 (token 0's experts) on every partition, its gather indices, then the
        # compaction of the other distinct experts (dedupe_precomputed: the loads of the caller's results instead).
        n_hot = _GROUP * prefetch_groups  # prefetch: the hot positions 0 .. n_hot - 1
        first_compacted = n_hot if prefetch_groups else K  # the first position that the compaction writes
        if prefetch_groups:
            after_norm_op(nisa.memset(dst=ones_row_t, value=1.0, engine=memset_engine, name="memset_ones_row_t"))
            # the hot experts 0 .. n_hot - 1 (before expert_offset) at positions 0 .. n_hot - 1 on every partition
            after_norm_op(
                nisa.iota(dst=ids_b[:, 0:n_hot], pattern=[[1, n_hot]], offset=0, channel_multiplier=0, name="pf_ids")
            )
        elif not precomputed:
            after_norm_op(nisa.memset(dst=ones_row, value=1.0, engine=dve, name="memset_ones_row"))
            after_norm_op(nisa.memset(dst=ones_row_t, value=1.0, engine=memset_engine, name="memset_ones_row_t"))
            nisa.tensor_copy(dst=idx0, src=index_tk[0:1, :], engine=dve, name="idx0_copy")
            head_ps = nl.ndarray(
                (_P_MAX, K),
                dtype=nl.float32,
                buffer=nl.psum,
                address=(0, 6 * _PSUM_BANK_BYTES + 1024),
                name="head_ids_ps",
            )
            nisa.nc_matmul(dst=head_ps, stationary=ones_row, moving=idx0, accumulate=False, name="mm_head_ids")
            nisa.tensor_copy(dst=ids_b[:, 0:K], src=head_ps, engine=dve, name="head_ids_copy")
        # The head's group 0 has its own down-scale tile (the first "head" tile): its indices do not wait for the
        # compaction. prefetch_groups: all tiles of the prefetch groups.
        head_tile_list = [s for s, (t, _) in enumerate(scale_tiles) if t == "head"]
        if prefetch_groups:
            head_tiles_ids = [s for s in head_tile_list if scale_tiles[s][1][-1] < prefetch_groups]
        else:
            head_tiles_ids = head_tile_list[:1]
        # dedupe_early_indices: head scale tile 1 (head groups 1-4; prefetch: the first tile of compacted groups) gets
        # its gather indices after the first part of the position matmuls.
        early_tiles, early_groups = [], []
        compacted_tiles = [s for s in head_tile_list if s not in head_tiles_ids]
        if dedupe_early_indices and compacted_tiles and len(scale_tiles[compacted_tiles[0]][1]) == _BLOCKS:
            early_tiles = [compacted_tiles[0]]
            early_groups = list(scale_tiles[early_tiles[0]][1])

        def early_indices(part: int, first: int, cols: int) -> Optional[nisa.NkiInstruction]:
            """The ids of the first position part on every partition, then the gather indices of early_groups (returns
            the down index instruction, which DVE runs right after the gate/up index)."""
            if part != 0 or not early_groups:
                return None
            kernel_assert(
                first <= _GROUP * early_groups[0] and _GROUP * (early_groups[-1] + 1) <= first + cols,
                "dedupe_early_indices: the first position part must cover head groups 1-4",
            )
            # PSUM bank 7 (default) or bank 6 from 512 B (after the transposes of tr_ps), where no matmul accumulates
            early_address = bank7 if early_ids_bank == 7 else 6 * _PSUM_BANK_BYTES + 512
            kernel_assert(early_ids_bank == 7 or 4 * cols <= 512, f"early_ids_bank=6 needs <= 128 columns, got {cols}")
            early_ps = nl.ndarray(
                (_P_MAX, cols), dtype=nl.float32, buffer=nl.psum, address=(0, early_address), name="early_ids_ps"
            )
            nisa.nc_matmul(
                dst=early_ps,
                stationary=ones_row_t[T : T + 1, :],
                moving=ids_rest[T : T + 1, first : first + cols],
                accumulate=False,
                name="mm_early_ids",
            )
            nisa.tensor_scalar(
                dst=ids_b[:, first : first + cols],
                data=early_ps,
                op0=nl.add,
                operand0=float(_PAD_ID),
                engine=dve,
                name="early_ids_copy",
            )
            return gather_indices(early_groups, early_tiles, "_early")

        # dedupe_precomputed: the columns of the caller's index tensor (see the docstring)
        n16 = n_pos_groups * _GROUP
        pre_cols = {"gu": 0, "gu2": n16, "dn": 2 * n16, "dn2": 2 * n16 + n_pos_groups}
        pre_cols["sc"] = pre_cols["dn2"] + 2 * n_pos_groups
        pre_cols["trips"] = pre_cols["sc"] + n_scale_tiles

        # (HWDGE on SP, or on ACT in the fused block, where SP carries the router weights)
        pre_queue = nisa.engine.sync if after_norm == None else nisa.engine.scalar

        def load_precomputed(dst: nl.NkiTensor, src: nl.NkiTensor, label: str) -> None:
            nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=pre_queue, name=f"load_pre_{label}")

        if precomputed:
            pre_index, pre_aff = dedupe_precomputed[0:2]
            kernel_assert(
                tuple(pre_index.shape) == (_P_MAX, pre_cols["trips"] + len(dedupe_block_groups))
                and tuple(pre_aff.shape) == (T, N),
                f"dedupe_precomputed needs int32 [{_P_MAX}, {pre_cols['trips'] + len(dedupe_block_groups)}] and "
                f"[{T}, {N}], got {pre_index.shape} and {pre_aff.shape}",
            )
            # the indices of the head's group 0 first
            for label, dst, cols in (
                ("gu", gi_gu, n16),
                ("dn", gi_dn, n_pos_groups),
                ("sc", gi_sc, n_scale_tiles),
                ("gu2", gi_gu2 if pairs_gu else None, n16),
                ("dn2", gi_dn2, 2 * n_pos_groups),
            ):
                if dst != None:
                    load_precomputed(dst, pre_index[:, pre_cols[label] : pre_cols[label] + cols], label)
            load_precomputed(aff_cols, pre_aff, "aff")
        elif prefetch_groups and prefetch_indices != None:
            # the static gather indices of the prefetch groups from the caller: their columns only
            kernel_assert(
                tuple(prefetch_indices.shape) == (_P_MAX, pre_cols["trips"] + len(dedupe_block_groups)),
                f"prefetch_indices needs int32 [{_P_MAX}, {pre_cols['trips'] + len(dedupe_block_groups)}]",
            )
            tile_end = head_tiles_ids[-1] + 1  # the tiles of the prefetch groups come first
            for label, dst, cols in (
                ("gu", gi_gu, n_hot),
                ("dn", gi_dn, prefetch_groups),
                ("sc", gi_sc, tile_end),
                ("gu2", gi_gu2 if pairs_gu else None, n_hot),
                ("dn2", gi_dn2, 2 * prefetch_groups),
            ):
                if dst != None:
                    nisa.dma_copy(
                        dst=dst[:, 0:cols],
                        src=prefetch_indices[:, pre_cols[label] : pre_cols[label] + cols],
                        dge_mode=nisa.dge_mode.hwdge,
                        engine=pre_queue,
                        name=f"load_pf_{label}",
                    )
        elif prefetch_groups:
            # the static gather indices of the prefetch groups: calls that start at a pair start
            calls = [[0], list(range(1, prefetch_groups))] if head_lead else [list(range(prefetch_groups))]
            calls = [c for c in calls if c]
            for j, call in enumerate(calls):
                gather_indices(call, head_tiles_ids if j == len(calls) - 1 else [], f"_pf{j}")
        else:
            gather_indices([0], head_tiles_ids, "_head")
        out_ps, _, _ = head_tiles = psum_tiles("")

        def run_head(mode: str = "all") -> Optional[Callable[[], None]]:
            """The head pipeline (token 0's group), emitted right after the head affinities of the compaction, so
            DVE / ACT run it before the rest of the compaction."""
            tag, groups, lead = pipelines[0]
            return run_groups(tag, groups, head_tiles, lead=lead, mode=mode)

        trips = alloc((1, len(dedupe_block_groups)), nl.int32, name="dd_trips")
        # dedupe_late_trips: the compaction writes trips_early; a copy into trips follows the eviction.
        trips_early = (
            alloc((1, len(dedupe_block_groups)), nl.int32, name="dd_trips_early")
            if dedupe_late_trips or dedupe_trips_after_stores
            else trips
        )
        if precomputed:
            first = pre_cols["trips"]
            load_precomputed(trips_early, pre_index[0:1, first : first + len(dedupe_block_groups)], "trips")
        # The compaction options, passed only when they are on.
        compaction_options = {}
        if setup_memsets != "pool":
            compaction_options.update(memset_engine=setup_memsets)
        if dedupe_onehot_parts:
            compaction_options.update(onehot_parts=True)
        if dedupe_early_first:
            compaction_options.update(early_first=True)
        if dedupe_order != "id":
            compaction_options.update(order=dedupe_order)
        if dedupe_token_rank != "tiles":
            compaction_options.update(token_rank=dedupe_token_rank)
        if index_first:

            def after_early(after) -> None:
                """dedupe_index_first: after the early indices, the caller's affinity-only work (late_setup) and the
                constants of the group pipeline."""
                if late_setup != None:
                    late_setup(after)
                late_constants()

            compaction_options.update(index_first=True, after_early=after_early)
        # The compaction temporaries live in their own scope; out_acc, out_sb and fold_in take their addresses after
        # it (a later tile at these addresses waits for the last reads of the temporaries).
        sbm.open_scope()
        if precomputed:
            # The first gather waits for `after` (the point where the caller's results would be ready), and for a
            # stand-in Pool op of pool_cols fp32 columns (an iota) after it.
            token = dedupe_precomputed[2] if len(dedupe_precomputed) > 2 else None
            pool_cols = dedupe_precomputed[3] if len(dedupe_precomputed) > 3 else 0
            if token != None and pool_cols > 0:
                # the stand-in op: pool_op = iota + token (Pool), read by the token op of the first gather
                rows = token.shape[0]
                pool_src = alloc((rows, pool_cols), nl.float32, name="pre_pool_src")
                pool_op = alloc((rows, pool_cols), nl.float32, name="pre_pool_op")
                if token.dtype != nl.float32:
                    # the per-partition operand of the Pool op is fp32
                    token_f32 = alloc((rows, 1), nl.float32, name="pre_token_f32")
                    nisa.tensor_copy(dst=token_f32, src=token[:, 0:1], engine=pool, name="pre_token_f32_op")
                    token = token_f32
                nisa.iota(dst=pool_src, pattern=[[1, pool_cols]], offset=0, channel_multiplier=0, name="pre_pool_iota")
                nisa.tensor_scalar(
                    dst=pool_op,
                    data=pool_src,
                    op0=nl.add,
                    operand0=token[:, 0:1],
                    engine=pool,
                    name="pre_pool_op_add",
                )
                token = pool_op
            gather_after[0] = token
            gather_after[1] = alloc((1, 1), nl.float32, name="pre_zero")
        else:
            compact_experts(
                alloc,
                index_tk,
                router_logits,
                router_top,
                router_top_recip,
                aff_cols,
                ids_rest,
                N,
                [_GROUP * g for g in dedupe_block_groups],
                after_norm_op,
                after_head=run_head if S_head == 1 else None,
                static_slots=_GROUP * (S_head - (prefetch_groups if prefetch_groups else 1)),
                router_scores=router_scores,
                after_part=early_indices,
                trips=trips_early,
                alternatives=dedupe_block_mode == "alternatives",
                name_suffix=name_suffix,
                prefetch=n_hot,
                **compaction_options,
            )
        sbm.close_scope()
        # fp32 down PSUM: out_acc (the out_ps rows) takes the head and the executed block; the stores follow the
        # blocks (dedupe_fold_first does not apply).
        fold_first = dedupe_fold_first and not down_fp32
        if down_fp32:
            out_acc = alloc((2 * two_t, half_cols), nl.float32, name="out_acc")
        elif fold_first:
            # fold_full: the bf16 output terms [2T, H] (static, then + the executed block), the moving of the folds
            fold_full = alloc((two_t, H), nl.bfloat16, name="fold_full")
            out_sb = alloc((T, H), nl.bfloat16, name="out_sb")
        else:
            out_acc = alloc((two_t, H), nl.float32, name="out_acc")
            if not fold_dma:
                out_sb = alloc((T, H), nl.bfloat16, name="out_sb")  # fold_dma: the only fold (after the blocks) stores
            fold_in = alloc((two_t, 2 * _FOLD_CHUNK), nl.bfloat16, name="fold_in")
        # The block positions on every partition (fp16 id - _PAD_ID: exact), then their gather indices.
        if not precomputed:
            rest_ps = nl.ndarray(
                (_P_MAX, N - first_compacted), dtype=nl.float32, buffer=nl.psum, address=(0, bank7), name="rest_ids_ps"
            )
            nisa.nc_matmul(
                dst=rest_ps,
                stationary=ones_row_t[T : T + 1, :],
                moving=ids_rest[T : T + 1, first_compacted:N],
                accumulate=False,
                name="mm_rest_ids",
            )
            nisa.tensor_scalar(
                dst=ids_b[:, first_compacted:N],
                data=rest_ps,
                op0=nl.add,
                operand0=float(_PAD_ID),
                engine=dve,
                name="rest_ids_copy",
            )
            rest_first = (prefetch_groups if prefetch_groups else 1) + len(early_groups)
            gather_indices(
                list(range(rest_first, n_pos_groups)),
                [s for s in range(n_scale_tiles) if s not in head_tiles_ids + early_tiles],
                "_rest",
            )
        if sp_group_period > 0:
            # The expert ids of the block positions as int32 on partition T (SP HWDGE groups): empty -> expert 0.
            nisa.tensor_scalar(
                dst=ids_f32[T : T + 1, K:N],
                data=ids_rest[T : T + 1, K:N],
                op0=nl.add,
                operand0=float(_PAD_ID),
                engine=dve,
                name="ids_f32_op",
            )
            nisa.scalar_tensor_tensor(
                dst=ids_i32[T : T + 1, K:N],
                data=ids_f32[T : T + 1, K:N],
                op0=nl.less,
                operand0=float(E),
                op1=nl.multiply,
                operand1=ids_f32[T : T + 1, K:N],
                name="ids_i32_op",
            )
            if expert_offset != 0:
                nisa.tensor_scalar(
                    dst=ids_i32[T : T + 1, K:N],
                    data=ids_i32[T : T + 1, K:N],
                    op0=nl.add,
                    operand0=expert_offset,
                    engine=dve,
                    name="ids_i32_offset",
                )
        if S_head > 1 and two_phase:
            # The ring loads and the re-layout of group 0 before the latent; the group loop after it.
            head_loop = run_head("start")
            latent_done = yield "routing"
            if two_phase_latent_first:
                after_latent[0] = latent_done
            head_loop()
        elif S_head > 1:
            run_head()  # groups 1 .. S_head - 1 read the gather indices of the compaction
        alternatives_mode = dedupe_block_mode == "alternatives"
        static_stores = []  # the store DMAs of the static output (fold_dma with dedupe_fold_first)
        if down_fp32:
            # The head's output leaves PSUM before the blocks: DVE copies banks 2-3, ACT banks 4-5.
            static_done = evict_fp32(out_ps, "head")
        elif fold_first:
            if dedupe_prefetch:
                # All alternatives start at group S_head with the same positions: the first loads of the largest one go
                # on Pool while the first fold runs (none runs: the loads are not used).
                tag, groups, lead = pipelines[-1]
                run_groups(tag, groups, psum_tiles("_pre"), lead=lead, mode="prefetch")
            # The static output: folded and stored now (through fold_full), as in the baseline.
            static_done = fold_and_store(out_ps, fold_full, fold_ps, stores=static_stores)
        elif evict_split == "columns":
            # The head's output leaves PSUM before the blocks: DVE and ACT copy half of the columns each.
            half = H // 2
            static_done = [
                nisa.tensor_copy(dst=out_acc[:, 0:half], src=out_ps[:, 0:half], engine=dve, name="head_evict_c0"),
                nisa.activation(dst=out_acc[:, half:H], op=nl.copy, data=out_ps[:, half:H], name="head_evict_c1"),
            ]
        elif evict_split == "banks":
            # DVE copies the first half of the PSUM banks of out_ps (bf16 columns 0 .. 2047 at H = 3584: banks 2-3),
            # ACT the others: no PSUM bank is read by both.
            bank_cols = _PSUM_BANK_BYTES // 2
            split = bank_cols * div_ceil(div_ceil(H, bank_cols), 2)
            static_done = [
                nisa.tensor_copy(dst=out_acc[:, 0:split], src=out_ps[:, 0:split], engine=dve, name="head_evict_b0"),
                nisa.activation(dst=out_acc[:, split:H], op=nl.copy, data=out_ps[:, split:H], name="head_evict_b1"),
            ]
        else:
            # The head's output leaves PSUM before the blocks: DVE copies the hi rows, ACT the lo rows.
            static_done = [
                nisa.tensor_copy(dst=out_acc[0:T, :], src=out_ps[0:T, :], engine=dve, name="head_evict_hi"),
                nisa.activation(dst=out_acc[T:two_t, :], op=nl.copy, data=out_ps[T:two_t, :], name="head_evict_lo"),
            ]
        if dedupe_trips_after_stores:
            # The trip counts after the static stores (an add of 0, which the compiler keeps): the register loads
            # (every engine) are scheduled after the store DMAs.
            late = nisa.tensor_scalar(
                dst=trips, data=trips_early, op0=nl.add, operand0=0, engine=dve, name="dd_trips_late"
            )
            for instruction in static_done + static_stores:
                late.depends_on(instruction, kind=nisa.dependency.schedule)
        elif dedupe_late_trips:
            # The trip counts after the static output: their register loads (every engine) are scheduled here.
            late = nisa.tensor_copy(dst=trips, src=trips_early, engine=dve, name="dd_trips_late")
            for instruction in static_done:
                late.depends_on(instruction, kind=nisa.dependency.schedule)
        trip_counts = [nisa.scalar_load(trips[0:1, b : b + 1]) for b in range(len(blocks))]
        if dedupe_prefetch and alternatives_mode and not fold_first:
            # All alternatives start at group S_head with the same positions, so one prefetch of the first loads of the
            # largest alternative serves the one that runs (none runs: the loads are not used).
            tag, groups, lead = pipelines[-1]
            run_groups(tag, groups, psum_tiles("_pre"), lead=lead, mode="prefetch")
        for b in range(len(blocks)):
            block_tiles = psum_tiles(f"_blk{b}")
            tag, groups, lead = pipelines[b + 1]
            if dedupe_prefetch and not alternatives_mode:
                # The first loads of the block before its loop (they load expert 0 rows if the block does not run).
                run_groups(tag, groups, block_tiles, lead=lead, mode="prefetch")
            block_fold_ps = None
            if fold_first:
                # the fold tiles of the block (banks 0-1, like fold_ps): a PSUM tensor cannot be live across a loop
                block_fold_ps = [
                    nl.ndarray(
                        (T, _FOLD_CHUNK),
                        dtype=nl.float32,
                        buffer=nl.psum,
                        address=(0, r * _PSUM_BANK_BYTES),
                        name=f"fold_ps{r}_blk{b}",
                    )
                    for r in range(2)
                ]

            def block_body(_, tag=tag, groups=groups, lead=lead, block_tiles=block_tiles, fold_tiles=block_fold_ps):
                """One 0/1-trip block: its position groups, then out_acc += out_ps (with dedupe_fold_first:
                fold_full += out_ps, then the fold and the stores again)."""
                in_loop[0] = True
                run_groups(tag, groups, block_tiles, lead=lead, mode="after_prefetch" if dedupe_prefetch else "all")
                if fold_first:
                    # fold_full += out_ps per 512 columns, each followed by its fold and store
                    fold_and_store(block_tiles[0], fold_full, fold_tiles, add=True)
                else:
                    nisa.tensor_tensor(dst=out_acc, data1=block_tiles[0], data2=out_acc, op=nl.add, engine=dve)
                in_loop[0] = False

            nl.fori_loop(0, trip_counts[b], block_body)
        fold_source = None if fold_first or down_fp32 else out_acc
        if down_fp32:
            store_fp32(list(range(div_ceil(half_cols, _DOWN_CHUNK))))

    # Step 5: Fold the hi and lo rows of each token (bf16 matmul with fold_sel) and store each 512-column chunk.
    if fold_source != None:
        fold_and_store(fold_source, fold_in, fold_ps)
    sbm.close_scope()
    return output


def compact_experts(
    alloc,
    index_tk: nl.NkiTensor,
    router_logits: nl.NkiTensor,
    router_top: nl.NkiTensor,
    router_top_recip: nl.NkiTensor,
    aff_cols: nl.NkiTensor,
    ids: nl.NkiTensor,
    n_positions: int,
    dedupe_blocks: Sequence[int],
    after_norm_op,
    after_head: Optional[Callable[[], None]] = None,
    static_slots: int = 0,
    router_scores: Optional[nl.NkiTensor] = None,
    after_part: Optional[Callable[[int, int, int], None]] = None,
    trips: Optional[nl.NkiTensor] = None,
    alternatives: bool = False,
    name_suffix: Optional[str] = None,
    memset_engine: str = "pool",
    onehot_parts: bool = False,
    early_first: bool = False,
    order: str = "id",
    token_rank: str = "tiles",
    index_first: bool = False,
    after_early: Optional[Callable] = None,
    prefetch: int = 0,
) -> nl.NkiTensor:
    """Step 4 of group_body: the distinct experts, their positions in the dedupe blocks, the affinity of every token
    for the expert of every position, and the block trip counts.

    Writes aff_cols[:, 0:K] (head: the K experts of token 0), aff_cols[:, K:] and ids[T, K:] (block positions, fp16
    id - _PAD_ID; an empty position gets 0, that is id _PAD_ID, and zero affinities). dedupe_blocks gives the block
    sizes in slots. Returns trips int32 [1, n_blocks] on partition 0 (1 if block b runs). With router_scores (the
    sigmoid router), am is router_scores (the selected scores) and router_top_recip is 1 / (their sum + 1e-20).
    after_part(part, first, cols), if given, runs after each part of the position matmuls has written the ids of the
    positions first .. first + cols - 1. trips: the int32 [1, n_blocks] output tile (None: allocated with alloc).
    alternatives: dedupe_blocks is increasing and exactly one block runs (the smallest that holds the rest slots);
    the positions are not shifted, because all blocks start at the same position. name_suffix: appended to every op
    and PSUM tensor name (alloc appends it to the SBUF tensor names). memset_engine: "pool" or "dve", the engine of
    the ones memsets. onehot_parts: the one-hot tiles are built per part of the position matmuls (the part-0 matmuls
    then wait only for the part-0 columns). early_first: with onehot_parts, the one-hot tiles of the parts after the
    first are scheduled after the instruction that after_part returns for part 0 (the early gather indices).
    order: "id" (ascending expert id) or "token" (alternatives only: by the
    first token >= 1 that routes to the expert, then by id; Step 4.3t). token_rank: how Step 4.3t ranks the experts in
    the token order. "tiles": a scan per token over the expert tiles, then the experts first seen at earlier tokens.
    "flat": one scan over the (token, tile) axis, whose order is the token order, so two matmuls give the rank (the
    same positions, fewer ops on the head's critical path). index_first (alternatives, no after_head): the head
    affinities and the block plan (the trip counts) are emitted after the last position part, and part 0 evicts its
    ids before its affinities; after_early(after), if given, runs right after after_part(0, ...) (after: what that
    returned), before the affinity eviction of part 0. prefetch (token order, flat rank): the experts 0 .. prefetch - 1
    are positions 0 .. prefetch - 1 (their affinities: the selected scores, rounded to fp16, times router_top_recip);
    the compaction places the selected experts with id >= prefetch, in token order over all T tokens (token 0 has no
    head), from position prefetch on.

    Layouts (e = 128c + p, c = 0 .. E/128 - 1):
        am fp32 [T, E]: exp(logit - top_0) where the logit is >= the K-th largest of the token, else 0.
        AT_ps fp32 [128, (c, t)] (PSUM bank 7): am transposed per 128-expert tile.
        atx fp16 [128, c, T + 1]: AT_ps rounded to fp16, and in column T the id offset e - _PAD_ID: the stationary of
            the position matmuls.
        present / slot_r fp32 [128, c]: 1 if a token other than 0 routes to e and token 0 does not; the number of
            such experts with a smaller id.
        pos fp32 [128, c]: the block position of expert e, or >= _NO_POSITION if it has none.
        onehot fp16 [128, c, n_rest]: (pos == K + j) for rest position j.
        rw_ps fp32 [T + 1, n_rest / n_parts] (PSUM bank 7, one pass per part): rows 0..T-1 the affinity (before 1 / sum) of token t
            for the expert at the position, row T the id - _PAD_ID (0 for an empty position).
    """
    nisa, nl = named_modules(name_suffix)
    T, K = index_tk.shape
    E = router_logits.shape[1]
    n_et = E // _P_MAX
    head = prefetch if prefetch else K  # the positions before the compacted ones
    n_rest = n_positions - head
    n_blocks = len(dedupe_blocks)
    granule = dedupe_blocks[-1]
    dve, act, pool = nisa.vector_engine, nisa.scalar_engine, nisa.gpsimd_engine
    bank7 = 7 * _PSUM_BANK_BYTES

    if router_scores == None:
        neg_top0 = alloc((T, 1), nl.float32, name="dd_neg_top0")
        ex = alloc((T, E), nl.float32, name="dd_exp")
        am = alloc((T, E), nl.float32, name="dd_am")
    ident32_offsets = alloc((T, T), nl.float32, name="dd_ident32_offsets")
    ident32 = alloc((T, T), nl.float32, name="dd_ident32")
    atx = alloc((_P_MAX, n_et, T + 1), nl.float16, name="dd_atx")
    rest_sum = alloc((_P_MAX, n_et), nl.float32, name="dd_rest_sum")
    head_zero = alloc((_P_MAX, n_et), nl.float32, name="dd_head_zero")
    present = alloc((_P_MAX, n_et), nl.float32, name="dd_present")
    present_row = alloc((_P_MAX, 1), nl.float32, name="dd_present_row")
    incl = alloc((_P_MAX, n_et), nl.float32, name="dd_incl")
    excl = alloc((_P_MAX, n_et), nl.float32, name="dd_excl")
    lower_offsets = alloc((_P_MAX, _P_MAX), nl.float32, name="dd_lower_offsets")
    lower = alloc((_P_MAX, _P_MAX), nl.float32, name="dd_lower")
    ones = alloc((_P_MAX, _P_MAX), nl.float32, name="dd_ones")
    rest_left = alloc((_P_MAX, 1), nl.float32, name="dd_rest_left")
    run = alloc((_P_MAX, n_blocks), nl.float32, name="dd_run")
    base = alloc((_P_MAX, n_blocks), nl.float32, name="dd_base")
    skip_w = alloc((_P_MAX, n_blocks), nl.float32, name="dd_skip_w")
    cmp = alloc((_P_MAX, n_et, n_blocks), nl.float32, name="dd_cmp")
    shift_sum = alloc((_P_MAX, n_et), nl.float32, name="dd_shift_sum")
    pos = alloc((_P_MAX, n_et), nl.float32, name="dd_pos")
    pos_iota = alloc((_P_MAX, n_rest), nl.float32, name="dd_pos_iota")
    onehot = alloc((_P_MAX, n_et, n_rest), nl.float16, name="dd_onehot")
    e_iota = alloc((_P_MAX, n_et), nl.float32, name="dd_e_iota")
    idx0 = alloc((1, K), nl.float16, name="dd_idx0")
    ones_row = alloc((1, _P_MAX), nl.float16, name="dd_ones_row")
    head_onehot = alloc((_P_MAX, n_et, K), nl.float16, name="dd_head_onehot")
    if trips == None:
        trips = alloc((1, n_blocks), nl.int32, name="dd_trips")
    token_order = order == "token"
    kernel_assert(
        order in ("id", "token") and (not token_order or alternatives),
        f"order must be 'id' or 'token' (alternatives), got {order}",
    )
    kernel_assert(
        not prefetch or (token_order and token_rank == "flat" and alternatives), "prefetch needs token order, flat rank"
    )
    # tokens 1 .. T-1 (token 0 holds the head positions); prefetch: all tokens (column tok0 of at_ps is the first)
    n_tok, tok0 = (T, 0) if prefetch else (T - 1, 1)
    kernel_assert(token_rank in ("tiles", "flat"), f"token_rank must be 'tiles' or 'flat', got {token_rank}")
    flat_rank = token_order and token_rank == "flat"
    if token_order:
        tok_weight = alloc((_P_MAX, n_tok), nl.float32, name="dd_tok_weight")
        if not flat_rank:
            tok_reset = alloc((_P_MAX, n_tok * n_et), nl.float32, name="dd_tok_reset")
        tok_last = alloc((_P_MAX, n_et, n_tok), nl.float32, name="dd_tok_last")
        tok_first = alloc((_P_MAX, n_et), nl.float32, name="dd_tok_first")
        tok_onehot = alloc((_P_MAX, n_tok * n_et), nl.float32, name="dd_tok_onehot")
        tok_incl = alloc((_P_MAX, n_tok * n_et), nl.float32, name="dd_tok_incl")
        if flat_rank:
            # neg_upper[e', e] = -(e' >= e) = lower - 1: with ones^T incl it gives the rank in the flat order
            neg_upper = alloc((_P_MAX, _P_MAX), nl.float32, name="dd_tok_neg_upper")
        else:
            tok_excl = alloc((_P_MAX, n_tok * n_et), nl.float32, name="dd_tok_excl")
            tok_offs_incl = alloc((_P_MAX, n_tok), nl.float32, name="dd_tok_offs_incl")
            tok_offs = alloc((_P_MAX, n_tok), nl.float32, name="dd_tok_offs")
        tok_terms = alloc((_P_MAX, n_tok * n_et), nl.float32, name="dd_tok_terms")

    # PSUM bank 7 (64 B aligned tiles): at_ps [0, 896), slot_ps [896, 928), the head affinities, then the position
    # matmuls (in parts of _RW_PART_COLS columns) from 1024, hb_ps and tot_ps after them.
    n_parts = -(-n_rest // _RW_PART_COLS)
    while n_rest % n_parts != 0:
        n_parts += 1
    part_cols = n_rest // n_parts
    rw_end = 1024 + 4 * part_cols
    at_ps = nl.ndarray((_P_MAX, n_et * T), dtype=nl.float32, buffer=nl.psum, address=(0, bank7), name="dd_at_ps")
    slot_ps = nl.ndarray((_P_MAX, n_et), dtype=nl.float32, buffer=nl.psum, address=(0, bank7 + 896), name="dd_slot_ps")
    # The head affinities use the bytes of rw_ps before the position matmuls.
    hrw_ps = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.psum, address=(0, bank7 + 1024), name="dd_hrw_ps")
    rw_ps = nl.ndarray((T + 1, part_cols), dtype=nl.float32, buffer=nl.psum, address=(0, bank7 + 1024), name="dd_rw_ps")
    hb_start = -(-rw_end // 64) * 64
    hb_ps = nl.ndarray((_P_MAX, K), dtype=nl.float32, buffer=nl.psum, address=(0, bank7 + hb_start), name="dd_hb_ps")
    tot_start = -(-(hb_start + 4 * K) // 64) * 64
    kernel_assert(tot_start + 4 <= _PSUM_BANK_BYTES, f"Compaction PSUM tiles exceed bank 7 ({tot_start} B)")
    tot_ps = nl.ndarray((_P_MAX, 1), dtype=nl.float32, buffer=nl.psum, address=(0, bank7 + tot_start), name="dd_tot_ps")
    if token_order:
        # rank_ps [e % 128, (t, c)] takes the bytes of at_ps after its last read; cnt_ps after tot_ps.
        kernel_assert(4 * n_tok * n_et <= 4 * n_et * T, "dd_tok_rank_ps exceeds the bytes of dd_at_ps")
        rank_ps = nl.ndarray(
            (_P_MAX, n_tok * n_et), dtype=nl.float32, buffer=nl.psum, address=(0, bank7), name="dd_tok_rank_ps"
        )
        if not flat_rank:
            cnt_start = -(-(tot_start + 4) // 64) * 64
            kernel_assert(cnt_start + 4 * n_tok <= _PSUM_BANK_BYTES, f"dd_tok_cnt_ps exceeds bank 7 ({cnt_start} B)")
            cnt_ps = nl.ndarray(
                (_P_MAX, n_tok), dtype=nl.float32, buffer=nl.psum, address=(0, bank7 + cnt_start), name="dd_tok_cnt_ps"
            )

    # Constants (after the RMSNorm).
    after_norm_op(
        nisa.iota(dst=ident32_offsets, pattern=[[1, T]], offset=0, channel_multiplier=-1, name="dd_iota_ident")
    )
    after_norm_op(
        nisa.tensor_scalar(
            dst=ident32, data=ident32_offsets, op0=nl.equal, operand0=0.0, engine=dve, name="dd_ident_op"
        )
    )
    # lower[e', e] = (e' < e): iota e - e' > 0. ones: the stationary that sums over the partitions.
    after_norm_op(
        nisa.iota(dst=lower_offsets, pattern=[[1, _P_MAX]], offset=0, channel_multiplier=-1, name="dd_iota_lower")
    )
    after_norm_op(
        nisa.tensor_scalar(dst=lower, data=lower_offsets, op0=nl.greater, operand0=0.0, engine=dve, name="dd_lower_op")
    )
    ones_engine = dve if memset_engine == "dve" else pool
    after_norm_op(nisa.memset(dst=ones, value=1.0, engine=ones_engine, name="dd_memset_ones"))
    after_norm_op(nisa.memset(dst=ones_row, value=1.0, engine=ones_engine, name="dd_memset_ones_row"))
    # e_iota[p, c] = 128c + p; atx[:, c, T] = 128c + p - _PAD_ID.
    after_norm_op(nisa.iota(dst=e_iota, pattern=[[_P_MAX, n_et]], offset=0, channel_multiplier=1, name="dd_iota_e"))
    after_norm_op(
        nisa.iota(
            dst=atx.ap(pattern=[[n_et * (T + 1), _P_MAX], [T + 1, n_et], [1, 1]], offset=T),  # [p, c, T]
            pattern=[[_P_MAX, n_et], [0, 1]],
            offset=-_PAD_ID,
            channel_multiplier=1,
            name="dd_iota_id_col",
        )
    )
    if token_order:
        # tok_weight[p, j] = n_tok - j (token j + 1); tok_reset[p, (j, c)] = (c > 0): the scan restarts at each token.
        after_norm_op(
            nisa.iota(dst=tok_weight, pattern=[[1, n_tok]], offset=0, channel_multiplier=0, name="dd_iota_tok")
        )
        after_norm_op(
            nisa.tensor_scalar(
                dst=tok_weight,
                data=tok_weight,
                op0=nl.multiply,
                operand0=-1.0,
                op1=nl.add,
                operand1=float(n_tok),
                engine=dve,
                name="dd_tok_weight_op",
            )
        )
        if flat_rank:
            after_norm_op(
                nisa.tensor_scalar(
                    dst=neg_upper, data=lower, op0=nl.add, operand0=-1.0, engine=dve, name="dd_tok_neg_upper_op"
                )
            )
        else:
            after_norm_op(
                nisa.iota(
                    dst=tok_reset, pattern=[[0, n_tok], [1, n_et]], offset=0, channel_multiplier=0, name="dd_iota_reset"
                )
            )
            after_norm_op(
                nisa.tensor_scalar(
                    dst=tok_reset, data=tok_reset, op0=nl.greater, operand0=0.0, engine=dve, name="dd_tok_reset_op"
                )
            )
    # pos_iota[p, j] = K + j (prefetch: prefetch + j).
    after_norm_op(nisa.iota(dst=pos_iota, pattern=[[1, n_rest]], offset=head, channel_multiplier=0, name="dd_iota_pos"))

    def flat_positions(p_jc) -> None:
        """Step 4.3t with token_rank="flat": pos = the rank of (first token, id) among the present experts (K is added
        by dd_pos_off). An expert that is not present gets tok_first 0, which matches no token weight (>= 1).
        incl[e, (j, c)] = sum over (j', c') <= (j, c) of onehot[e, (j', c')] (one scan over the flattened axis, whose
        order is the token order); rank = ones^T incl + neg_upper^T onehot (the experts before e in the same column
        are those with a smaller partition)."""
        nisa.tensor_tensor(
            dst=tok_first, data1=tok_first, data2=present, op=nl.multiply, engine=dve, name="dd_tok_first_mask"
        )
        nisa.tensor_tensor(
            dst=tok_onehot.ap(pattern=p_jc, offset=0),
            data1=tok_first.ap(pattern=[[n_et, _P_MAX], [0, n_tok], [1, n_et]], offset=0),
            data2=tok_weight.ap(pattern=[[n_tok, _P_MAX], [1, n_tok], [0, n_et]], offset=0),
            op=nl.equal,
            engine=dve,
            name="dd_tok_onehot_op",
        )
        nisa.tensor_tensor_scan(
            dst=tok_incl,
            data0=ones.ap(pattern=[[_P_MAX, _P_MAX], [0, n_tok * n_et]], offset=0),  # 1: the scan never restarts
            data1=tok_onehot,
            initial=0.0,
            op0=nl.multiply,
            op1=nl.add,
            name="dd_tok_scan",
        )
        nisa.nc_matmul(dst=rank_ps, stationary=ones, moving=tok_incl, accumulate=False, name="dd_tok_mm_incl")
        nisa.nc_matmul(dst=rank_ps, stationary=neg_upper, moving=tok_onehot, accumulate=True, name="dd_tok_mm_upper")
        nisa.tensor_tensor(
            dst=tok_terms, data1=rank_ps, data2=tok_onehot, op=nl.multiply, engine=dve, name="dd_tok_pick_op"
        )
        nisa.tensor_reduce(
            dst=pos,
            op=nl.add,
            data=tok_terms.ap(pattern=[[n_tok * n_et, _P_MAX], [1, n_et], [n_et, n_tok]], offset=0),  # [p, c, j]
            axis=2,
            name="dd_tok_pos_sum",
        )

    def token_positions() -> None:
        """Step 4.3t: pos = K + the rank of (first token >= 1 that routes to e, e) among the present experts (exact fp32).

        tok_first = max over j of (n_tok - j) where token j + 1 routes to e; tok_onehot[e, (j, c)] = present and f(e) =
        token j + 1; rank = within-tile prefix (lower) + earlier tiles (scan per token, ones); offs = the experts
        first seen at earlier tokens. The rank matmuls write the bytes of at_ps after its last read (tok_last)."""
        p_jc = [[n_tok * n_et, _P_MAX], [n_et, n_tok], [1, n_et]]  # [p, j, c] over a (j, c) tile
        nisa.scalar_tensor_tensor(
            dst=tok_last,
            data=at_ps.ap(
                pattern=[[n_et * T, _P_MAX], [T, n_et], [1, n_tok]], offset=tok0
            ),  # [p, c, j]: token j + tok0
            op0=nl.greater,
            operand0=0.0,
            op1=nl.multiply,
            operand1=tok_weight.ap(pattern=[[n_tok, _P_MAX], [0, n_et], [1, n_tok]], offset=0),
            name="dd_tok_last_op",
        )
        nisa.tensor_reduce(dst=tok_first, op=nl.maximum, data=tok_last, axis=2, name="dd_tok_first_op")
        if flat_rank:
            flat_positions(p_jc)
            return
        nisa.tensor_tensor(
            dst=tok_onehot.ap(pattern=p_jc, offset=0),
            data1=tok_first.ap(pattern=[[n_et, _P_MAX], [0, n_tok], [1, n_et]], offset=0),
            data2=tok_weight.ap(pattern=[[n_tok, _P_MAX], [1, n_tok], [0, n_et]], offset=0),
            op=nl.equal,
            engine=dve,
            name="dd_tok_onehot_op",
        )
        nisa.tensor_tensor(
            dst=tok_onehot.ap(pattern=p_jc, offset=0),
            data1=tok_onehot.ap(pattern=p_jc, offset=0),
            data2=present.ap(pattern=[[n_et, _P_MAX], [0, n_tok], [1, n_et]], offset=0),
            op=nl.multiply,
            engine=dve,
            name="dd_tok_present_op",
        )
        # incl[e, (j, c)] = sum over c' <= c of onehot[e, (j, c')] (the scan restarts at c = 0), excl = incl - onehot.
        nisa.tensor_tensor_scan(
            dst=tok_incl,
            data0=tok_reset,
            data1=tok_onehot,
            initial=0.0,
            op0=nl.multiply,
            op1=nl.add,
            name="dd_tok_scan",
        )
        nisa.tensor_tensor(
            dst=tok_excl, data1=tok_incl, data2=tok_onehot, op=nl.subtract, engine=dve, name="dd_tok_excl_op"
        )
        nisa.nc_matmul(dst=rank_ps, stationary=lower, moving=tok_onehot, accumulate=False, name="dd_tok_mm_in")
        nisa.nc_matmul(dst=rank_ps, stationary=ones, moving=tok_excl, accumulate=True, name="dd_tok_mm_tiles")
        # cnt[j] = the experts first seen at token j + 1 (every partition): the last scan column of each token.
        nisa.nc_matmul(
            dst=cnt_ps,
            stationary=ones,
            moving=tok_incl.ap(pattern=[[n_tok * n_et, _P_MAX], [n_et, n_tok]], offset=n_et - 1),
            accumulate=False,
            name="dd_tok_mm_cnt",
        )
        nisa.tensor_tensor_scan(
            dst=tok_offs_incl,
            data0=ones[:, 0:n_tok],
            data1=cnt_ps,
            initial=0.0,
            op0=nl.multiply,
            op1=nl.add,
            name="dd_tok_offs_scan",
        )
        nisa.tensor_tensor(
            dst=tok_offs, data1=tok_offs_incl, data2=cnt_ps, op=nl.subtract, engine=dve, name="dd_tok_offs_op"
        )
        nisa.tensor_tensor(
            dst=tok_terms.ap(pattern=p_jc, offset=0),
            data1=rank_ps.ap(pattern=p_jc, offset=0),
            data2=tok_offs.ap(pattern=[[n_tok, _P_MAX], [1, n_tok], [0, n_et]], offset=0),
            op=nl.add,
            engine=dve,
            name="dd_tok_terms_op",
        )
        nisa.tensor_tensor(
            dst=tok_terms, data1=tok_terms, data2=tok_onehot, op=nl.multiply, engine=dve, name="dd_tok_pick_op"
        )
        nisa.tensor_reduce(
            dst=pos,
            op=nl.add,
            data=tok_terms.ap(pattern=[[n_tok * n_et, _P_MAX], [1, n_et], [n_et, n_tok]], offset=0),  # [p, c, j]
            axis=2,
            name="dd_tok_pos_sum",
        )
        nisa.tensor_scalar(dst=pos, data=pos, op0=nl.add, operand0=float(K), engine=dve, name="dd_tok_pos_op")

    # Step 4.1: am = (logit >= top[:, K - 1]) * exp(logit - top_0) (sigmoid router: the selected scores); A^T per
    # 128-expert tile.
    if router_scores != None:
        am = router_scores
    else:
        nisa.tensor_scalar(
            dst=neg_top0, data=router_top[:, 0:1], op0=nl.multiply, operand0=-1.0, engine=dve, name="dd_neg_top0_op"
        )
        nisa.activation(dst=ex, op=nl.exp, data=router_logits, bias=neg_top0, name="dd_exp_op")
        nisa.scalar_tensor_tensor(
            dst=am,
            data=router_logits,
            op0=nl.greater_equal,
            operand0=router_top[:, K - 1 : K],  # [t, 1]: the K-th largest logit of token t
            op1=nl.multiply,
            operand1=ex,
            name="dd_am_op",
        )
    for c in range(n_et):
        nisa.nc_matmul(
            dst=at_ps[:, c * T : (c + 1) * T],  # [e % 128, t]
            stationary=am[:, c * _P_MAX : (c + 1) * _P_MAX],  # [t, e % 128]
            moving=ident32,
            is_moving_onezero=True,
            is_transpose=True,
            accumulate=False,
            name=f"dd_transpose_e{c}",
        )
    nisa.activation(
        dst=atx.ap(pattern=[[n_et * (T + 1), _P_MAX], [T + 1, n_et], [1, T]], offset=0),  # [p, c, t]
        op=nl.copy,
        data=at_ps,
        name="dd_atx_copy",
    )

    kernel_assert(
        not index_first or (alternatives and after_head == None), "index_first needs alternatives and no after_head"
    )

    def head_affinities() -> None:
        """Step 4.2: Head affinities: head_onehot[e, (c, k)] = (128c + p == index[0, k]); aff_cols[:, k] = A^T
        one-hot."""
        nisa.tensor_copy(dst=idx0, src=index_tk[0:1, :], engine=dve, name="dd_idx0_copy")
        nisa.nc_matmul(
            dst=hb_ps, stationary=ones_row, moving=idx0, accumulate=False, name="dd_idx0_bcast"
        )  # [p, k] = index[0, k]
        nisa.tensor_tensor(
            dst=head_onehot,
            data1=e_iota.ap(pattern=[[n_et, _P_MAX], [1, n_et], [0, K]], offset=0),  # [p, c, k]: 128c + p
            data2=hb_ps.ap(pattern=[[K, _P_MAX], [0, n_et], [1, K]], offset=0),  # [p, c, k]: index[0, k]
            op=nl.equal,
            engine=dve,
            name="dd_head_onehot_op",
        )
        for c in range(n_et):
            nisa.nc_matmul(
                dst=hrw_ps,  # [t, k]
                stationary=atx[:, c, 0:T],
                moving=head_onehot[:, c, :],
                accumulate=c > 0,
                name=f"dd_mm_head_e{c}",
            )
        nisa.tensor_scalar(
            dst=aff_cols[:, 0:K],
            data=hrw_ps,
            op0=nl.multiply,
            operand0=router_top_recip,
            engine=dve,
            name="dd_head_aff",
        )

    def hot_affinities() -> None:
        """prefetch: aff_cols[:, e] = fp16(am[:, e]) * router_top_recip for the hot experts e < prefetch (position e)."""
        hot16 = alloc((T, prefetch), nl.float16, name="dd_hot16")
        nisa.tensor_copy(dst=hot16, src=am[:, 0:prefetch], engine=dve, name="dd_hot16_op")
        nisa.tensor_scalar(
            dst=aff_cols[:, 0:prefetch],
            data=hot16,
            op0=nl.multiply,
            operand0=router_top_recip,
            engine=dve,
            name="dd_hot_aff",
        )

    if prefetch:
        hot_affinities()
    elif not index_first:
        head_affinities()

    if after_head != None:
        after_head()  # the caller's head pipeline reads aff_cols[:, 0:K]

    # Step 4.3: present[e] = (sum over t >= 1 of A^T > 0) * (A^T[e, 0] == 0); exclusive prefix count over e.
    # (prefetch: the sum over all t, and e >= prefetch instead of token 0's test)
    nisa.tensor_reduce(
        dst=rest_sum,
        op=nl.add,
        data=at_ps.ap(pattern=[[n_et * T, _P_MAX], [T, n_et], [1, T - tok0]], offset=tok0),  # [p, c, t >= tok0]
        axis=2,
        name="dd_rest_sum_op",
    )
    if prefetch:
        nisa.tensor_scalar(
            dst=head_zero, data=e_iota, op0=nl.greater_equal, operand0=float(prefetch), engine=dve, name="dd_cold_op"
        )
    else:
        nisa.tensor_scalar(
            dst=head_zero,
            data=at_ps.ap(pattern=[[n_et * T, _P_MAX], [T, n_et]], offset=0),  # [p, c]: token 0
            op0=nl.equal,
            operand0=0.0,
            engine=dve,
            name="dd_head_zero_op",
        )
    nisa.scalar_tensor_tensor(
        dst=present,
        data=rest_sum,
        op0=nl.greater,
        operand0=0.0,
        op1=nl.multiply,
        operand1=head_zero,
        name="dd_present_op",
    )

    def present_total() -> None:
        """present_row[p] = the present experts of partition p; tot_ps = U - K on every partition."""
        nisa.tensor_reduce(dst=present_row, op=nl.add, data=present, axis=1, name="dd_present_row_op")
        if index_first:
            nisa.nc_matmul(
                dst=tot_ps, stationary=ones, moving=present_row, accumulate=False, name="dd_mm_total"
            )  # U - K on every partition

    if not index_first:
        present_total()
    if token_order:
        token_positions()
    # incl[p, c] = sum over c' <= c of present[p, c'] (scan), excl = incl - present.
    # scan: incl[:, c] = ones * incl[:, c - 1] + present[:, c]
    nisa.tensor_tensor_scan(
        dst=incl, data0=ones[:, 0:n_et], data1=present, initial=0.0, op0=nl.multiply, op1=nl.add, name="dd_scan"
    )
    nisa.tensor_tensor(dst=excl, data1=incl, data2=present, op=nl.subtract, engine=dve, name="dd_excl_op")
    # slot_r[e, c] = sum over e' < e of present[e', c] + sum over e' of excl[e', c] (fp32 matmuls, exact).
    nisa.nc_matmul(dst=slot_ps, stationary=lower, moving=present, accumulate=False, name="dd_mm_prefix_in")
    nisa.nc_matmul(dst=slot_ps, stationary=ones, moving=excl, accumulate=True, name="dd_mm_prefix_tiles")
    if not index_first:
        nisa.nc_matmul(
            dst=tot_ps, stationary=ones, moving=present_row, accumulate=False, name="dd_mm_total"
        )  # U - K on every partition

    # Step 4.4: Block plan (exact in fp32), greedy over the blocks from the largest: with rest_left = the rest slots not
    # yet placed, block b (size_b) runs if rest_left >= size_b - granule + 1 (so ceil(rest_left / granule) has bit k
    # set); base[b] = the slots of the executed larger blocks; skip_w[b] = (1 - run[b]) * size_b.
    # The static slots after the head come first (always run): the blocks place the rest.
    def alternatives_plan() -> None:
        """Alternatives: block b runs if size_{b-1} < rest_left <= size_b (size_{-1} = 0)."""
        nisa.tensor_scalar(
            dst=rest_left, data=tot_ps, op0=nl.add, operand0=-float(static_slots), engine=dve, name="dd_rest_left_op"
        )
        for b, size in enumerate(dedupe_blocks):
            previous = dedupe_blocks[b - 1] if b > 0 else 0
            nisa.tensor_scalar(
                dst=run[:, b : b + 1],
                data=rest_left,
                op0=nl.greater_equal,
                operand0=float(previous + 1),
                engine=dve,
                name=f"dd_run_ge{b}",
            )
            nisa.scalar_tensor_tensor(
                dst=run[:, b : b + 1],
                data=rest_left,
                op0=nl.less_equal,
                operand0=float(size),
                op1=nl.multiply,
                operand1=run[:, b : b + 1],
                name=f"dd_run{b}",
            )
        nisa.tensor_copy(dst=trips, src=run[0:1, :], engine=dve, name="dd_trips_op")

    if alternatives:
        # pos = K + slot_r (alternatives: the blocks start at the same position)
        if not index_first:
            alternatives_plan()
        if not token_order:
            nisa.tensor_scalar(dst=pos, data=slot_ps, op0=nl.add, operand0=float(K), engine=dve, name="dd_pos_op")
    else:
        nisa.tensor_scalar(
            dst=rest_left, data=tot_ps, op0=nl.add, operand0=-float(static_slots), engine=dve, name="dd_rest_left_op"
        )
        for b, size in enumerate(dedupe_blocks):
            nisa.tensor_scalar(
                dst=run[:, b : b + 1],
                data=rest_left,
                op0=nl.greater_equal,
                operand0=float(size - granule + 1),
                engine=dve,
                name=f"dd_run{b}",
            )
            nisa.tensor_tensor(
                dst=base[:, b : b + 1], data1=tot_ps, data2=rest_left, op=nl.subtract, engine=dve, name=f"dd_base{b}"
            )
            nisa.scalar_tensor_tensor(
                dst=rest_left,
                data=run[:, b : b + 1],
                op0=nl.multiply,
                operand0=-float(size),
                op1=nl.add,
                operand1=rest_left,
                name=f"dd_rest_left{b}",
            )
            nisa.tensor_scalar(
                dst=skip_w[:, b : b + 1],
                data=run[:, b : b + 1],
                op0=nl.multiply,
                operand0=-float(size),
                op1=nl.add,
                operand1=float(size),
                engine=dve,
                name=f"dd_skip_w{b}",
            )
        nisa.tensor_copy(dst=trips, src=run[0:1, :], engine=dve, name="dd_trips_op")
        # pos[e, c] = K + slot_r + sum over b of skip_w[b] * (slot_r >= base[b]); + _NO_POSITION if not present.
        nisa.tensor_tensor(
            dst=cmp,
            data1=slot_ps.ap(pattern=[[n_et, _P_MAX], [1, n_et], [0, n_blocks]], offset=0),  # [p, c, b]: slot_r
            data2=base.ap(pattern=[[n_blocks, _P_MAX], [0, n_et], [1, n_blocks]], offset=0),  # [p, c, b]: base[b]
            op=nl.greater_equal,
            engine=dve,
            name="dd_cmp_op",
        )
        nisa.tensor_tensor(
            dst=cmp,
            data1=cmp,
            data2=skip_w.ap(pattern=[[n_blocks, _P_MAX], [0, n_et], [1, n_blocks]], offset=0),  # [p, c, b]: skip_w[b]
            op=nl.multiply,
            engine=dve,
            name="dd_cmp_w",
        )
        nisa.tensor_reduce(dst=shift_sum, op=nl.add, data=cmp, axis=2, name="dd_shift_sum_op")
        nisa.scalar_tensor_tensor(
            dst=pos, data=slot_ps, op0=nl.add, operand0=float(K), op1=nl.add, operand1=shift_sum, name="dd_pos_op"
        )
    nisa.scalar_tensor_tensor(
        dst=pos, data=present, op0=nl.multiply, operand0=-_NO_POSITION, op1=nl.add, operand1=pos, name="dd_pos_mask"
    )
    # flat_rank: pos holds the rank, so K is added here
    pos_offset = _NO_POSITION + (float(head) if flat_rank else 0.0)
    nisa.tensor_scalar(dst=pos, data=pos, op0=nl.add, operand0=pos_offset, engine=dve, name="dd_pos_off")

    # Step 4.5: onehot[e, c, j] = (pos == K + j); rw = atx^T onehot in parts of the positions.
    early_done = [None]  # early_first: the instruction that the one-hot tiles of the later parts follow

    def onehot_columns(j0: int, cols: int, label: str) -> None:
        """onehot[:, c, j0 : j0 + cols] for every expert tile c."""
        for c in range(n_et):
            instruction = nisa.tensor_scalar(
                dst=onehot[:, c, j0 : j0 + cols],
                data=pos_iota[:, j0 : j0 + cols],
                op0=nl.equal,
                operand0=pos[:, c : c + 1],
                engine=dve,
                name=f"dd_onehot{label}_e{c}",
            )
            if early_done[0] is not None:
                instruction.depends_on(early_done[0], kind=nisa.dependency.schedule)

    if not onehot_parts:
        onehot_columns(0, n_rest, "")
    for part in range(n_parts):
        j0 = part * part_cols
        if onehot_parts:
            onehot_columns(j0, part_cols, f"_p{part}")
        for c in range(n_et):
            nisa.nc_matmul(
                dst=rw_ps,  # [t, j] affinity; row T: id - _PAD_ID
                stationary=atx[:, c, :],
                moving=onehot[:, c, j0 : j0 + part_cols],
                accumulate=c > 0,
                name=f"dd_mm_rw_p{part}_e{c}",
            )

        def evict_affinities() -> None:
            nisa.tensor_scalar(
                dst=aff_cols[:, head + j0 : head + j0 + part_cols],
                data=rw_ps[0:T, :],
                op0=nl.multiply,
                operand0=router_top_recip,
                engine=dve,
                name=f"dd_aff_p{part}",
            )

        def evict_ids() -> None:
            nisa.tensor_copy(
                dst=ids[T : T + 1, head + j0 : head + j0 + part_cols],
                src=rw_ps[T : T + 1, :],
                engine=dve,
                name=f"dd_ids_p{part}",
            )

        if index_first and part == 0:
            # the ids, the early indices and after_early, then the affinities
            evict_ids()
            done = after_part(part, head + j0, part_cols) if after_part != None else None
            if early_first and onehot_parts:
                early_done[0] = done
            if after_early != None:
                after_early(done)
            evict_affinities()
            continue
        evict_affinities()
        evict_ids()
        if after_part != None:
            done = after_part(part, head + j0, part_cols)
            if early_first and onehot_parts and part == 0:
                early_done[0] = done
    if index_first:
        # the work that only the trip counts and the head affinities need
        present_total()
        alternatives_plan()
        head_affinities()
    return trips
