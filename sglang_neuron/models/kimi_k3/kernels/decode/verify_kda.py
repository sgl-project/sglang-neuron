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

"""KDA verify attention body of the Kimi-K3 verify giga-kernel (TP32, LNC1, Trn3).

The verify body scores Q tokens of each of B sequences in one call, with the semantics of Q decode
steps in sequence (speculative decoding). Row ``j * B + b`` of the token-major hT input is token
``j`` of sequence ``b``.

The body keeps one COMMITTED state per sequence instead of a page per token. It replays the accepted
prefix of the previous call to rebuild the committed state, then scores the Q new tokens from it:

1. Load the committed state (recurrent state and conv window) from ``state_page``.
2. Load the replay records of the previous call from ``record_page``. A record of one token holds the
   pre-conv projection values and the fp32 decay gate and beta of that token.
3. Replay: for the first ``replay_len`` tokens of each sequence, rerun the per-token delta rule on a
   work copy of the committed state and commit the new state and conv window. The replay reproduces
   ``replay_len`` decode steps exactly, so the committed state round trips through bf16 as a page does.
4. Write the committed (replayed) state back to ``state_page``.
5. Score the Q new tokens from the committed state, chaining token to token in SBUF with no per-token
   page writes. Each new token is one decode step.
6. Write the records of the Q new tokens to ``record_page``, so the next call can replay them.

A select, not a multiply, zeros an off read and a record beyond ``replay_len``, so a NaN in a stale
page or a reused slot cannot reach the state or the output.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from .utils.kernel_assert import kernel_assert
from .utils.kernel_helpers import div_ceil
from .common import check_pair_group, dma_priority, is_trn3, token_reduce_scatter
from .tkg_weight_dma_gate import gate_weight_dma, order_after_gate_result, release_weight_gate
from .tkg_kda import (
    _PSUM_BANK_FP32,
    _QUADRANT,
    _RECURRENCE_GROUP,
    CONV_TAPS,
    GATE_LOWER_BOUND,
    HEAD_DIM,
    KDA_READOUTS,
    KDA_STATE_DTYPES,
    L2_EPS,
    MAX_LOCAL_HEADS,
    MAX_SEQUENCES,
    N_CONV_PROJ,
    NORM_EPS,
    P_MAX,
    PAGE_WIDTH,
    HeadMajorGemv,
    _bf16_diff_rows,
    _bf16_row_selector,
    _bf16_update_parts,
    _conv_scatter_rows,
    _conv_weights,
    _decay_state,
    _divide_by_one_plus,
    _even_pages_only,
    _gather_conv_rows,
    _gemv_channel_major,
    _load_conv_states,
    _off_to_null_page,
    _output_projection,
    _page_counts,
    _per_sequence,
    _recurrent_state_view,
    _runtime_after,
    _schedule_after,
    _stationary_parts,
    _store_conv_states,
    _w_in_buffers,
    after_w_in_loads,
    bf16_diff_cols,
    in_proj_copies_x_stationary,
    kda_a_exp,
    kda_default_readout,
    kda_in_columns,
    kda_in_columns_head_major,
    kda_page_bound,
    kda_page_offsets,
    kda_page_span,
    kda_params_a_exp,
    kda_small_columns,
    kda_w_out_form,
    kda_weight_dma_options,
    load_kda_w_out,
)
from .tkg_mx_quant import MX_LANES, mx_hT_shapes, mx_weight_format

MAX_QUERIES = 9
"""Most tokens of each sequence in one verify step: the bonus token and 8 drafts."""
EARLY_OUT_TOKENS = 6
"""Kill switch of the early output tail (0, not the default, turns the tail off): N > 0 runs the gated RMSNorm and the
output projection of the rows of tokens 0 .. N - 1 inside the sequential token chain, right after token N - 1, and those
of the later tokens after the chain, so only the later rows' tail stays between the last token and the output
all-reduce. Every step of the tail is per row (the norm, the gate, the MXFP8 quantization of 8 partitions x 4 lanes per
token, each matmul's moving column), so the rows keep their bits; the one all-reduce of all rows does not change. Only
with fewer than Q tokens and the sequential chain. 6: at Q = 8 (the k7 forms) the last 2 tokens' rows keep their tail
after the chain."""
HEAD_PIPELINE_FORMS = ("off", "g_last", "heads")
"""Forms of ``kda_verify_attention`` ``head_pipeline``: ``off`` keeps the program; ``g_last`` defers the input
projection's output-gate (g) blocks on the weight groups that stay in SBUF, so the chain starts before them; ``heads``
takes the head-major ``kda_w_in`` (``kda_decode_torch.kda_w_in_head_major``) and runs the projection and the conv /
readout pass in head stages, so the first heads' pass runs beside the last head's projection."""
LATE_KEEP_WARM_MAX = 8
"""The most warm-up matmuls of ``kda_verify_attention`` ``late_keep_warm``."""
LATE_KEEP_WARM_EVERY = 14
"""``late_keep_warm``: the hidden tiles of the late output projection between two hold matmuls. At the trn3 8/8 pace an
MXFP8 tile takes a fixed time, so a hold matmul every 14 tiles puts two N = 512 matmuls (full-row work) in each
di/dt window: the row duty stays above the 4/8 threshold while the projection runs."""
_KEEP_WARM_COLS = 512
"""The moving columns (and the fp32 PSUM bank width) of a ``late_keep_warm`` matmul."""
ROW_SELECTOR_ONCE = True
"""Kill switch (default True) of the ``bf16_update_rows`` update: each call builds the bf16 row selector of
``tkg_kda._bf16_update_parts`` once for its Q tokens, not once a token. The bits do not change; False (not the default)
builds the selector once a token."""
KDA_CHAINS = ("sequential", "chunked", "batched")
"""Forms of the scoring chain. ``sequential`` runs the delta rule token by token (a chain of decode steps).
``chunked`` solves the Q tokens of a sequence at once in the chunk-parallel (WY) form; see ``_chunked_chain``.
``batched`` solves the records replay and the Q scored tokens of all sequences and heads at once in the same
form, batched over the (sequence, head) pairs; see ``_batched_solve``."""
REPLAY_PREWORK_FORMS = ("one_pass", "per_token")
"""Forms of the records replay prework (``replay_prework``) and of the records scoring prework (``score_prework``): the
conv and readout vectors of the tokens in one pass before the chain, or one token at a time inside it. The bits are the
same."""
W_IN_PSUM_MAX_ROWS = 12
"""Most rows (Q * B) of a call whose input projection takes ``w_in_psum_bufs`` (more rows keep one PSUM tile per chunk).
MEASURED on Trn2 (2 ranks, 2 PSUM tiles vs 1): B = 3, Q = 4 (12 rows, the served form) a faster kernel; B = 2, Q = 8 (16
rows) a slower one."""
CHUNKED_MAX_QUERIES = 8
"""Most tokens of the ``chunked`` chain: its inverse of ``I + N`` has the factors up to ``N**4`` (``N**8 = 0``)."""
BATCHED_MAX_QUERIES = 8
"""Most tokens of the ``batched`` chain: its records form solves 2Q <= 16 tokens, whose inverse of ``I + N`` has the
factors up to ``N**8`` (``N**16 = 0``)."""
_BF16_FINITE_MAX = 3.3895313892515355e38
_FP32_FINITE_MAX = 3.4028234663852886e38
# Row isolation of the ``batched`` chain: the bounds of its solve inputs. The real inputs never reach them (normed k and q
# have |x| <= 1, the gate keeps the decay in [exp(-5), 1], beta is a sigmoid), so a finite input passes bit for bit.
_BATCHED_UNIT_BOUND = 2.0
_BATCHED_VALUE_BOUND = 2.0**24
_BATCHED_DECAY_FLOOR = 0.0024787521766663585
"""exp(-6): below the gate's floor exp(-5); a zero or negative decay (an unwritten record) gets it, so log stays finite."""
_BATCHED_RHS_BOUND = 2.0**60
_BATCHED_DELTA_BOUND = 2.0**100
_BATCHED_STORE_BOUND = 2.0**60
"""A stored state, conv window or record element at or above this magnitude is garbage (a clamped NaN or Inf reads as
the dtype's largest value): the batched chain stores the safe value instead. Real values stay far below it."""


def kda_record_offsets(heads: int, queries: int) -> dict:
    """Return the column layout of one sequence's replay record in a cache page.

    The record of each of ``queries`` tokens holds the pre-conv projection values (q, k, v blocks,
    bf16) and the decay gate and beta (fp32, stored as bf16 word pairs). The blocks are laid out token
    major and padded so each fp32 block starts on an even bf16 column.

    Args:
        heads (int): KDA heads of one rank.
        queries (int): Tokens of each sequence (Q).

    Returns:
        dict: The start columns ``cm``, ``decay`` and ``beta`` of the three blocks, the ``n_conv`` and
        ``gate_words`` block widths per token, and the total ``width`` in bf16 columns per sequence.

    Example:
        >>> kda_record_offsets(3, 8)["width"]
        168
    """
    n_conv = N_CONV_PROJ * heads
    gate_words = 2 * heads
    cm_block = queries * n_conv
    decay = cm_block + (cm_block % 2)
    beta = decay + queries * gate_words
    width = beta + queries * gate_words
    width += width % 2
    return {"cm": 0, "decay": decay, "beta": beta, "n_conv": n_conv, "gate_words": gate_words, "width": width}


def kda_verify_attention(
    x_hT: nl.NkiTensor,
    kda_w_in: nl.NkiTensor,
    kda_w_out: nl.NkiTensor,
    kda_w_small: nl.NkiTensor,
    kda_params: nl.NkiTensor,
    kv_pool: nl.NkiTensor,
    state_page: nl.NkiTensor,
    record_page: nl.NkiTensor,
    replay_len: nl.NkiTensor,
    state_position: nl.NkiTensor,
    tp_group: ReplicaGroup,
    tag: str,
    readout: str = None,
    ar_hbm: tuple = None,
    out_rs_tp: int = 0,
    replay: str = "records",
    state_slot_page: nl.NkiTensor = None,
    write_state_slots: nl.NkiTensor = None,
    prework: dict = None,
    conv_dtype: str = "bf16",
    chain: str = "sequential",
    token_tiles: bool = False,
    slot_store_queues: int = 1,
    slot_store_engine: str = "sync",
    w_in_buffers: int = 0,
    bf16_update_rows: bool = False,
    bf16_cols_once: bool = False,
    late_keep_warm: int = 0,
    kda_w_in_scale: nl.NkiTensor = None,
    state_dtype: str = "bf16",
    scatter_keep: list = None,
    w_in_ring=None,
    w_out_after_w_in: bool = False,
    w_out_after_w_in_fills: bool = False,
    w_in_marks: dict = None,
    in_proj_hook=None,
    token_hook=None,
    tail_hook=None,
    x_quant_hook=None,
    state_loads_first: bool = False,
    slot_pages_first: bool = False,
    conv_load_gather: bool = False,
    batched_conv_stores: bool = False,
    kda_w_out_scale: nl.NkiTensor = None,
    null_page: int = 0,
    slot_state_ring: tuple = None,
    slot_state_waits: dict = None,
    slot_state_depth: int = 2,
    slot_ring_per_call: bool = False,
    slot_ring_keep_sink: nl.NkiTensor = None,
    slot_ring_keep_defer: list = None,
    ar_sbuf: nl.NkiTensor = None,
    slot_pages_contiguous: bool = False,
    batched_slot_conv_stores: bool = False,
    slot_conv_rows: dict = None,
    slot_conv_waits: list = None,
    update_bcast_depth: int = 2,
    head_pipeline: str = "off",
    out_ar_rows: bool = False,
    in_proj_x_stationary: bool = False,
    state_loads_after_ring: bool = False,
    early_out_tokens: int = None,
    in_proj_lanes_view: bool = True,
    states_after_gate_result: bool = True,
    commit_after_w_in: bool = False,
    weight_dma: str = "pool",
    consts: dict = None,
    slot_conv_from_inputs: bool = False,
    conv_products_engine: str = "gpsimd",
    recurrence_group: int = None,
    paired_pages: bool = False,
    small_after_result: bool = False,
    shared_state_ring: bool = False,
    conv_zero_rows: nl.NkiTensor = None,
    page_group: dict = None,
    replay_prework: str = "one_pass",
    score_prework: str = "one_pass",
    w_in_psum_bufs: int = 1,
    w_in_refill_queues: str = "alternate",
    state_loads_after: tuple = (),
    merged_state_loads: bool = False,
    slot_store_regs: bool = False,
    slot_conv_after_ar: bool = False,
) -> nl.NkiTensor:
    """Run the KDA verify attention of one layer for Q tokens of each of B sequences on one TP rank.

    This is a body function, not a ``@nki.jit`` kernel. The body keeps one committed state per
    sequence. It replays the accepted prefix of the previous call to rebuild the committed state, then
    scores the Q new tokens from it. Each replayed or scored token equals one ``kda_decode_attention``
    step, so the scored output rows and the committed state are bitwise equal to a chain of decode
    steps. The attribute ``token_major`` is True: the body takes and returns token-major hT tiles.

    Dimensions:
        H: Hidden size, a multiple of 128.
        Hl: Local KDA heads.
        W: Local KDA width, Hl * 128.
        P: Pages in the cache pool.
        B: Sequences, 1 to 32.
        Q: Tokens of each sequence, 1 to 9.
        R: Rows, Q * B.

    Args:
        x_hT (nl.NkiTensor): [128, Q*B, H/128] bf16 SBUF, normalized hidden in token-major hT layout.
            Row ``j * B + b`` is token ``j`` of sequence ``b``.
        kda_w_in (nl.NkiTensor): [H, N] bf16 HBM, fused input projection, see ``tkg_kda``.
        kda_w_out (nl.NkiTensor): [W, H] bf16 HBM, output projection.
        kda_w_small (nl.NkiTensor): [128, W + 12*Hl] bf16 HBM, ``f_b`` and the conv weights.
        kda_params (nl.NkiTensor): [128, 2*Hl + 1] fp32 HBM, ``dt_bias``, ``A_log`` and ``o_norm`` (and the 12*Hl
            fp32 conv weights with ``conv_dtype`` fp32).
        kv_pool (nl.NkiTensor): [P, 73728] bf16 HBM, paged cache pool with one page per row.
        state_page (nl.NkiTensor): [B, 1] int32 HBM, the page that holds the committed state of each
            sequence. The body reads it and writes the new committed state back to it.
        record_page (nl.NkiTensor): [B, 1] int32 HBM, the page that holds the replay records of the
            previous call. The body reads it and overwrites it with the records of its Q new tokens.
        replay_len (nl.NkiTensor): [B, 1] int32 HBM, 0 to Q, tokens of the previous call to replay.
        state_position (nl.NkiTensor): [B, 1] int32 HBM, position of the committed state (0 is a zero
            state, as the decode body treats position 0).
        tp_group (ReplicaGroup): TP replica group of the output all-reduce.
        tag (str): Unique per-layer name prefix.
        readout (str): Form of the readout and the state matmuls, one of ``KDA_READOUTS``. None (the
            default) takes the form from ``kda_default_readout(B)``, the form the decode body uses.
        replay (str): ``records`` (default) replays the accepted prefix from the records. ``states`` writes
            the decode state page after each scored token to its slot page, and the next call loads the
            state after ``replay_len`` tokens from its slot, with no replay.
        state_slot_page (nl.NkiTensor): ``states``: [B, 1] int32 HBM, the slot page of the state to load
            (the slot of token ``replay_len - 1`` of the previous call). ``replay_len`` 0 reads
            ``state_page``.
        write_state_slots (nl.NkiTensor): ``states``: [B, Q] int32 HBM, the slot page of each scored token.
            Each sequence takes its own slot pages.
        prework (dict): None, or the result of ``kda_verify_prework`` for this layer, which a caller traces
            one phase ahead (beside the collective of the body before). None traces it here.
        conv_dtype (str): Form of the short convolution, one of ``CONV_DTYPES`` (see ``kda_decode_attention``).
            The replay and the scoring take the same form, so a replayed state equals that many decode steps of
            the same ``conv_dtype``.
        chain (str): Form of the scoring chain, one of ``KDA_CHAINS``. ``sequential`` (the default) chains
            the Q tokens as decode steps. ``chunked`` solves them at once (one head per rank, 1 or 2
            sequences, at most ``CHUNKED_MAX_QUERIES`` tokens); it is not bitwise equal to the decode steps.
            ``batched`` solves the records replay and the Q scored tokens of every sequence and head at once (any
            head count and batch, at most ``BATCHED_MAX_QUERIES`` tokens), in fp32 with one rounding of each stored
            state; it is not bitwise equal to the decode steps. ``batched`` and ``chunked`` write the Q slot states
            from per-call tiles that a later instruction can reuse before the slot stores finish (no ring), so they
            are not for serving until they take the slot-state ring and its edges (both are default-off).
        token_tiles (bool): ``states`` only: give each scored token its own state tiles at any local head count, as
            the one-head form does, so no token waits for the slot DMAs of the token before. False (the default)
            keeps the shared tiles above one head; the bits do not change.
        replay_prework (str): ``records`` replay with the ``sequential`` chain: ``one_pass`` (default) runs the conv and
            the readout vectors (q and k norms, moving vectors) of the Q replayed tokens in one pass before the replay
            chain, as the per-token scoring path does, since they do not read the recurrent state; only the delta rule
            stays in the chain. ``per_token`` runs them in the chain, one token at a time. The bits do not change.
        score_prework (str): ``records``, the shared-tile scoring path (more than one head, the ``sequential``
            chain): ``one_pass`` (default) runs the conv and the readout vectors of the Q scored tokens in one pass
            before the scoring chain, as the per-token path does; ``per_token`` runs them in the chain, one token at a
            time. The bits do not change.
        w_in_psum_bufs (int): Trn2, the BF16 ``kda_w_in``: the PSUM tiles that the input projection's chunk sums take
            in turn (``_gemv_channel_major`` ``psum_bufs``), for a call of at most ``W_IN_PSUM_MAX_ROWS`` rows. 1
            (default) keeps the program; 2 lets a chunk's matmuls run while VectorE adds the chunk before. The bits do
            not change.
        w_in_refill_queues (str): the queues of the BF16 ``kda_w_in`` ring refills (``_gemv_channel_major``
            ``refill_queues``): ``alternate`` (default) keeps the program; ``sync`` puts every refill on the SP queue.
            The bits do not change.
        slot_store_queues (int): ``states`` only: 1 (default) issues the per-token slot state stores on the Sync
            queue; 2 alternates them between the Sync and the ScalarE queues. The bits do not change.
        slot_store_engine (str): ``states`` only, with ``slot_store_queues`` 1: the queue of the per-token slot
            state stores. "sync" (default) keeps the Sync hardware DGE queue; "gpsimd" issues them from the GpSimd
            software DGE queue, off the in-order Sync queue that also carries the next MoE layer's first loads;
            "split" alternates them between the two. The same bytes go to the same pages, so the bits do not change.
        w_in_buffers (int): SBUF buffers of the ``kda_w_in`` ring of the input projection, as in
            ``kda_decode_attention``. 0 (the default) takes ``_w_in_buffers(Hl, B)`` and keeps the program. The bits
            do not change.
        bf16_update_rows (bool): with the ``decoupled_bf16`` readout, the state update of each token takes its bf16
            high and low rows from one bf16 TensorE transpose per 4 sequences and adds them in one broadcast matmul
            per head (``kda_decode_attention`` ``bf16_update_rows``). The bits do not change. No effect with the other
            readouts. False (default) keeps the program.
        bf16_cols_once (bool): with ``bf16_update_rows`` on the per-token states path (``token_tiles``, the sequential
            chain): the bf16 rows tiles of the update (``tkg_kda._bf16_diff_rows`` ``cols``) are zeroed once before
            the chain, two sets that the tokens take in turn, in place of a GpSimd memset in each token. The chain then
            has no GpSimd instruction that waits in order behind the slot conv-state scatters. The update writes only
            the high and low rows, so the zero columns stay zero and the same bytes reach the transposes; the bits do
            not change. False (default) keeps the program.
        late_keep_warm (int): 0 (default) keeps the program. N from 1 to ``LATE_KEEP_WARM_MAX``: N zero-data TensorE
            matmuls (bf16, 512 moving columns, one fp32 PSUM tile of their own, one accumulation group) before the
            late output projection, each after the log activation of the last rows' gated norm (a runtime edge, so
            they run late in the TensorE gap before the projection, not at its start), plus one between every
            ``LATE_KEEP_WARM_EVERY`` of the projection's hidden tiles (a schedule edge on the matmul before). They
            keep the TensorE row duty above the trn3 di/dt limiter's threshold in each window, so the
            projection runs at 8/8, not 4/8. Nothing reads them (the served compiler keeps unread NKI TensorE
            instructions; count them in the NEFF). They write only their own PSUM tile, so the bits do not change.
        kda_w_in_scale (nl.NkiTensor): None (default, a BF16 ``kda_w_in``), or the ``[128, S, N]`` uint8 scales of an
            MXFP8 ``kda_w_in`` in the hT_quad layout of ``tkg_mx_quant`` (``tkg_kda`` ``kda_w_in_scale``): the input
            projection runs MXFP8 x MXFP8 (``_gemv_channel_major_mx``). The values change.
        kda_w_out_scale (nl.NkiTensor): None (default, a BF16 ``kda_w_out``), or the scales of the MXFP8 padded-lane
            ``kda_w_out`` (``tkg_kda`` ``kda_w_out_scale``): the output projection runs MXFP8 x MXFP8. The values
            change.
        ar_sbuf (nl.NkiTensor): None (default: the all-reduce runs in HBM, through ``ar_hbm`` or a per-call pair), or a
            persistent bf16 SBUF ``[128, B * Q, H/128]`` source tile (allocated once for the kernel, for example from a
            region heap, and held to the kernel end): the output projection writes the partial into it, and the TP
            all-reduce runs SBUF -> SBUF into the result tile, with no HBM source store and no destination load. The
            Neuron runtime takes the same all-reduce form as for the HBM pair, with its reduce order for this tile
            shape, so the values do not change. Not with ``out_rs_tp``.
        state_dtype (str): Form of the recurrent state in the pages, one of ``KDA_STATE_DTYPES`` (see
            ``kda_decode_attention``). ``bf16`` (default) keeps the program. ``fp32`` and ``fp32_hilo`` read and write
            the fp32 state of the decode page (the committed state, and the slot pages in ``states`` mode), chain it
            in fp32 through the replay and the scoring, and give ``decoupled_bf16`` the bf16 stationary parts of each
            token's state, so every scored row and every written state stays bitwise equal to the decode steps of the
            same ``state_dtype``. The pool must be at least ``kda_page_offsets(Hl, state_dtype)["end"]`` wide and of
            even width. Not with ``chain="chunked"``. A ``prework`` must be traced with the same ``state_dtype``.
        paired_pages (bool): with an fp32 ``state_dtype``, every KDA page id (the committed state, record and slot
            pages) names the even page of an aligned pair of the pool (EVEN ids <= P - 2 only, so id + 1 < P: an
            odd id or an id >= P - 1 is off, as an out-of-range id), as ``kda_decode_attention`` ``paired_pages``:
            the pages, the records and the outputs have the bytes of the 512-row form of the same pool viewed as
            ``[P / 2, 2 * width]`` at id / 2. Not with a bf16 state or ``slot_pages_contiguous``. A ``prework`` must
            be traced with the same ``paired_pages``. False (default) keeps the program.
        scatter_keep (list): None (the default), or a list that the body extends by the SBUF views that its conv-state
            stores read and the stores themselves, ``(source rows [rows, 384], None, store DMAs)`` of each group of
            sequences of each store call (``kda_decode_attention`` ``scatter_keep``), for a caller that reads them
            after the stores (``kda_read_scatter_keep``). The output does not change.
        w_in_ring (WInRing): None (the default), or the ring of this layer whose first fills the caller issued
            earlier (``tkg_kda.prefetch_w_in_ring``), as in ``kda_decode_attention``. The first row chunk of the
            input projection reads it; the ring sets the buffer count (``w_in_buffers`` 0 or the same count). The
            bits do not change.
        w_out_after_w_in (bool): False (the default) keeps the program. True: the ``kda_w_out`` load is traced after the
            input projection (not before it), follows the last two ``kda_w_in`` loads of its last row chunk in the
            schedule and waits for their completion (``kda_decode_attention`` ``w_out_after_w_in``). The bits do not
            change.
        w_out_after_w_in_fills (bool): False (the default) keeps the program. True (the MXFP8 ``kda_w_in``): the
            ``kda_w_out`` load (and the loads of its MX scales) is traced after the input projection, follows the last
            two ``kda_w_in`` fills of the projection in the schedule and waits for their completion, so the
            ``kda_w_in`` stream does not share the DMA queues with it. The small weights, the state loads and their
            order do not change. The output projection reads ``kda_w_out`` at the end of the call. The same loads, so
            the bits do not change. Not with ``w_out_after_w_in`` (the BF16 form).
        w_in_marks (dict): None (the default), or a dict that the body fills with the ``kda_w_in`` loads and the last
            instruction of the last row chunk of the input projection (``kda_decode_attention`` ``w_in_marks``). The
            program does not change.
        in_proj_hook: None (the default) keeps the program, or a function that the body calls once, right after the
            input projection, with the marks of ``w_in_marks`` (``fills``: the ``kda_w_in`` loads, the refills last;
            ``end``). The caller traces work there that follows the projection's weight stream, for example the
            expert ring of the layer's MoE (``decode_block_body`` ``moe_prefetch_kda_after_in_proj``). The BF16 or the
            MXFP8 ``kda_w_in``. When the hook puts a list of DMAs at the marks' ``before_slot_stores`` key, every slot
            state store of the first scored token follows those DMAs in the schedule (a schedule edge, not a wait), so
            the in-order queue issues them all before the store chain (``decode_block_body``
            ``moe_prefetch_kda_before_slot_stores``). The values do not change.
        token_hook: None (the default) keeps the program, or a function that the body calls as ``token_hook(j, Q)``
            right after the chain's work on token ``j`` of the Q scored tokens (its state stores included). The caller
            traces work there that does not depend on this layer, so it fills the engine gaps of the token chain (for
            example the bank part of the layer's MLP mix, ``decode_block_body`` ``AHEAD_MLP_TOKEN_STEPS``).
        tail_hook: None (the default) keeps the program, or a function that the body calls once as
            ``tail_hook(marks)`` right after the output projection of its last rows is traced, before the all-reduce:
            ``marks`` holds that projection's last TensorE matmul (``"tensor"``) and its last VectorE PSUM copy
            (``"vector"``). The caller traces work there that follows the token chain and the projection on each
            engine, so it runs in the gap up to the all-reduce and in its flight (for example the bank part of the
            layer's MLP mix, ``decode_block_body`` ``attn_res_ahead_mlp_kda_tail``).
        x_quant_hook: None (the default) keeps the program, or a function that the body calls once with the
            quantize_mx instruction of its input, right after the MXFP8 input projection traces it and before its first
            matmul (the first row chunk). The caller traces VectorE work there that follows the quantize in the
            schedule, so it runs beside the projection's matmuls, not ahead of the quantize. The MXFP8 ``kda_w_in``
            only.
        state_loads_first (bool): ``states`` only: trace the state and slot state loads before the input projection,
            as ``kda_decode_attention`` ``state_loads_first``: the tiles then get their own bytes, not the bytes of a
            ``kda_w_in`` ring buffer, and the loads do not wait for the projection. Not with a ``prework``. The bits
            do not change; False (default) keeps the program. With ``records`` the moved record load gave wrong rows
            on hardware: nothing orders it after the record store of a call before it in the same kernel.
        slot_pages_first (bool): ``states`` only: trace the load of the slot pages (``write_state_slots``) first, and
            make the first fills of the ``kda_w_in`` ring follow it, so the small load is not queued behind the ring
            on the Sync queue. With a ``w_in_ring``, only the first fills that the body issues follow it. Not with a
            ``prework``. The bits do not change; False (default) keeps the program.
        conv_load_gather (bool): load the committed conv state of each group of ``128 // (3*Hl)`` sequences with one
            gather DMA on the Pool queue, not one DMA per sequence on the SP queue, as in ``kda_decode_attention``.
            The gather reads the same rows into the same partitions, and an off read still reads page 0, so the values
            do not change. Needs a pool width that is a multiple of the conv row. False (default) keeps the program.
        batched_conv_stores (bool): write the committed conv state of each group of ``128 // (3*Hl)`` sequences with
            one scatter DMA on the Pool queue, not one DMA per sequence, as in ``kda_decode_attention``. Each sequence
            writes its own ``state_page`` and an off write goes to page 0 (the null page, ``_page_counts``), so no two
            rows of one scatter share a target and the values do not change. Needs a pool width that is a multiple of
            the conv row. False (default) keeps the program.
        null_page (int): 0 (the default), or the KDA null page that the caller keeps for no request, as in
            ``kda_decode_attention``: a state, record or slot page at it reads and writes like an off one (zero reads,
            zero conv-state scatter rows), and every off access goes to it, not to page 0. The rows of the other pages
            do not change. ``prework`` must come from ``kda_verify_prework`` with the same ``null_page``.
        slot_state_ring (tuple): ``states`` with ``chain="sequential"`` only: the persistent SBUF ring tiles of the
            per-token slot-state store DMAs (``tkg_kda.kda_slot_state_tiles``), one set for the whole kernel. The
            compiler ends a DMA source's live range at the trigger, so a per-call state tile can lose its bytes to a
            later instruction while the store still reads them (nki-dma-source-liveness). A ring tile lives through the
            kernel, so its bytes stay, and the next write into it waits at runtime for the slot stores that read it
            last. None (the default) allocates the ring inside the call: the within-call edges hold, but a later block
            can still take the bytes (use the sources for a serving kernel). Each tile is ``[128, B, Hl, 128]`` of the
            state dtype. The ``batched`` chain takes a ring of Q tiles (token j writes ring[j]) and no in-call ring.
            The per-token ``sequential`` path takes 1 to Q tiles. With d tiles token j writes ring[j % d], so its
            update waits for the stores of token ``j - d`` (with Q tiles only for the previous call's stores of that
            tile). The bits do not change.
        slot_ring_per_call (bool): with ``token_tiles`` on the sequential ``states`` chain and no ``slot_state_ring``: a
            slot-state ring of Q tiles of this call (token j writes tile j), so no token's state update waits for the
            slot stores of an earlier token. The stores still read the tiles after the chain, so right after the
            all-reduce trigger one GpSimd add of one element of each tile into a call sink waits for the tile's stores
            (a runtime edge): the tiles stay live until their stores have read them, and a later writer of their bytes
            follows those reads. The same stores of the same bytes, so the bits do not change. False (default) keeps
            the program.
        slot_ring_keep_sink (nl.NkiTensor): ``slot_ring_per_call`` only: the SBUF sink of the keep-reads, a ``[1, 2]``
            bf16 or ``[1, 1]`` fp32 tile of the state dtype that the caller keeps live to the kernel end (the 2-server
            verify: ``hold_to_end``), so the compiler keeps the reads. None: a ``[1, 1]`` sink of the call that a GpSimd
            DMA stores to a private HBM word; the compiler deletes those reads as dead code (the simulator twin only).
        slot_ring_keep_defer (list): ``slot_ring_per_call`` with ``slot_ring_keep_sink`` only: a kernel-level record of
            the rings whose keep-reads wait for a later call. The call appends its ring in place of reading it after the
            all-reduce, and at its start (after its first GpSimd copy) reads the rings of the earlier calls, whose stores
            have long finished, so no keep-read waits on GpSimd ahead of the next layer's loads. The caller reads the
            last ring (``kda_flush_ring_keeps``) before the kernel end. The ring tiles stay live until those reads.
        slot_state_waits (dict): ``states`` with ``slot_state_ring`` only: a kernel-level mutable record, ring index to
            the slot-state DMAs that read that ring tile last. Each call reads it to make the first write into a ring
            tile wait for the previous call's stores, and writes back its own last stores, so the edges cross the KDA
            calls and the blocks (as ``decode_block`` ``kda_conv_waits``). None keeps only the within-call edges.
        slot_state_depth (int): tiles of an in-call ring (``slot_state_ring`` None); with a ring it is ``len(ring)``.
            2 (the default) alternates the tiles, so token ``j + 1`` reads the tile of token ``j`` while ``j + 2``
            reuses it; 1 keeps one tile, so token ``j + 1`` waits for token ``j`` (the FIT fallback). The bits do
            not change.
        shared_state_ring (bool): ``states`` with ``chain="sequential"`` and the shared state tile (more than one head
            and no ``token_tiles``): the tokens take the 2-tile slot-state ring in place of one state tile that each
            token updates in place. Token 0 reads the committed state tile, token ``j`` writes ``ring[j % 2]`` and token
            ``j + 1`` reads it, so the update of token ``j + 1`` waits only for the slot stores of token ``j - 1`` (or
            the previous call's), not for those of token ``j``. The update writes the whole new state from the old one,
            so the bits do not change. Needs a ring of 2 tiles (``slot_state_ring`` or ``slot_state_depth``). The
            per-token path (``token_tiles``) already cycles the ring, so there it changes nothing. False (default) keeps
            the program.
        slot_pages_contiguous (bool): ``states`` only: load the ``[B, Q]`` slot ids of ``write_state_slots`` in their
            HBM order, one contiguous run of ``B * Q`` int32 per partition, and read them as ``[Q, B]`` through a view
            in the VectorE cast (``_write_slot_pages`` ``contiguous``). The transposed broadcast moves ``128 * Q * B``
            4-byte elements, and its small packets hold its DMA queue for a long time. The same pages, so the bits do
            not change. False (default) keeps the program.
        batched_slot_conv_stores (bool): ``states`` only: write the slot conv windows of the scored tokens with one scatter
            DMA per group of ``128 // (3*Hl)`` rows on the Pool queue (``_store_conv_states`` ``scatter_rows``, as
            ``batched_conv_stores`` does for the committed window), not one Sync DMA per row. The per-row DMAs run one
            after another (each waits for the one before), so ``Q * B`` of them hold the Sync queue for a long
            time, and a later token's tile that takes the bytes of their sources waits for them. The rows of
            every slot page do not change; an off slot (page 0, or the null page) gets zeros, so only those pages
            differ. Needs a pool width that is a multiple of the conv row. False (default) keeps the program.
        update_bcast_depth (int): the PSUM tiles of each token's state-update broadcast ring (``_recurrence``
            ``bcast_depth``), 2 to 4, each in its own PSUM bank. At 2 (the default) the update of
            sequence ``s + 2`` waits for the VectorE reads of sequence ``s``; 4 lets the updates of 4 sequences issue
            back to back. The bits do not change.
        in_proj_x_stationary (bool): the MXFP8 input projection with the tokens as the stationary operand
            (``_gemv_channel_major_mx`` ``x_stationary``: one matmul a weight group and column chunk of at most 512,
            in place of one a group and 128-column block) and its bf16 copies through TensorE transposes
            (``in_proj_copies_x_stationary``). The same sums in the same group order and the same roundings on the same
            engines, so the bits do not change. Needs ``kda_w_in_scale``, not with ``head_pipeline`` g_last. False
            (default) keeps the program.
        state_loads_after_ring (bool): with ``state_loads_first`` and a ``w_in_ring``: every per-sequence state and
            conv-state load on the Sync queue (the committed and the slot pages; not the GpSimd gathers of
            ``conv_load_gather``) follows the ring's last prefetch fill in the schedule.
            Their dynamic DMAs read a VectorE page index and wait for the earlier DMAs of their queue, so on the
            in-order Sync queue a load between two fills holds the later fills until then; after the ring, the fills
            issue back to back. The VectorE selects of the loaded states and conv windows then follow the input
            projection's quantize (with the MXFP8 ``kda_w_in``, through its quant hook, before ``x_quant_hook``), so
            VectorE does not wait for the later loads ahead of the quantize. The same loads and selects, so the bits do
            not change. A layer without a ``w_in_ring`` keeps its program; False (default) keeps the program.
        head_pipeline (str): the order of the MXFP8 input projection's output blocks (``HEAD_PIPELINE_FORMS``). ``off``
            (default) keeps the program. ``g_last``: the matmuls of the output-gate (g) blocks on the weight groups
            that stay in the ``kda_w_in`` ring to the end of the call (``_gemv_channel_major_mx`` ``late_blocks``) run
            after the decay gate, and their copy and sigmoid after the conv and readout pass, so the conv and the chain
            do not wait for them. Only the step 12 tail and the early output tail read the gate. Each block adds its
            weight groups in the same order, so the bits do not change. Needs ``kda_w_in_scale`` and one row chunk of
            the input projection. ``heads``: ``kda_w_in`` / ``kda_w_in_scale`` in the head-major column order
            (``kda_decode_torch.kda_w_in_head_major``; a caller that gives the original order gets wrong values). The
            projection loads each weight group as two parts of >= 2 KB a partition (``tkg_kda.HeadMajorGemv``) in a
            ring of the same SBUF bytes as the ``w_in_buffers`` ring, and runs the f_a, b and q / k / v blocks of heads
            0 .. Hl - 2 first (pass A), then those of the last head (pass B) beside the conv / readout pass of the first
            heads, then the g blocks late (pass C, in quarters beside the last head's pass and the first chain tokens).
            The token chain is the same 3-head chain as with ``off``, on tiles that both stages fill. Each block and
            element keeps its operands and its order, so the bits do not change. Needs the sequential states chain with
            ``token_tiles``, the decoupled readouts, no ``w_in_ring`` (no feed-forward prefetch), at least 2 heads and
            ``w_in_buffers`` (or the default ring) of at least the weight groups' A parts.
        out_ar_rows (bool): ACCURACY CLASS (<= 1 BF16 ulp, not bitwise). The output all-reduce runs as two row chunks of
            the persistent HBM pair: the rows of the early output tail (``early_out_tokens``) right after its output
            projection, inside the token chain, and the later rows at the end. The collective re-blocks each chunk, so
            an element can take another owner rank and sum order. Needs the early output tail and the HBM all-reduce.
            False (default) keeps the program.
        early_out_tokens (int): the early output tail of the sequential chain for this call: None (default) takes
            ``EARLY_OUT_TOKENS``; 0 turns the tail off.
        in_proj_lanes_view (bool): the MXFP8 input projection's quantize reads the lanes of its input in place
            (``tkg_kda._gemv_channel_major`` ``lanes_view``). True (default); False makes the lane copies. The
            same bits.
        states_after_gate_result (bool): the committed-state stores and the slot-state loads follow the result loads
            of the gate collectives in the schedule (``tkg_weight_dma_gate.order_after_gate_result``). True (default); False gives
            the schedule without this order. The same bits.
        consts (dict): None (default: the call sets its own constants), or the kernel's all-ones tiles
            (``kda_verify_constants``): ``ones_sq`` fp32 ``[128, 128]`` and ``ones_rows`` bf16 ``[128, 128]``, set once
            for the kernel, in place of the two memsets of each call (the same values).
        weight_dma (str): the queue of the output-projection, small-weight and parameter loads (``KDA_WEIGHT_DMAS``):
            ``pool`` (default) the GpSimd software DGE; ``act`` the Activation hardware DGE queue,
            so the GpSimd queue holds no weight load ahead of the call's conv-window gathers. The same bytes.
        commit_after_w_in (bool): with ``w_in_ring``: each committed-state store follows the ring's last prefetch fill
            in the schedule, so on the in-order Sync queue no input-projection weight fill waits behind a store (a store
            waits for the state commit and for the store before it). The same stores, so the bits do not change. False
            (default) keeps the program.
        state_loads_after (tuple): () (default) keeps the program, or instructions of the block before (for example
            the MoE output all-reduce) that the state and conv-state loads of this call on the Sync queue follow in the
            schedule and wait for at runtime (``_verify_state_loads`` ``sp_loads_after``), so those loads do not hold
            the in-order Sync queue ahead of them. The gathered conv-state load (``conv_load_gather``, Pool queue)
            keeps its place. The same loads, so the bits do not change. Not with a ``prework`` (``kda_prework_ahead``):
            that path traces the state loads in the prework, so the call refuses the pair before any instruction.
        merged_state_loads (bool): ``states`` only: pick each sequence's page first (its slot page when its slot read is
            on, else its state page) and load the recurrent state and the conv window once from it, in place of loading
            the state page and the slot page and selecting (``_verify_state_loads``). Two dynamic-page loads a sequence
            fewer, and B x (state + conv window) bytes fewer, for the same selected bytes, so the bits do not change.
            False (default) keeps the program.
        slot_store_regs (bool): ``states`` only, with ``slot_store_queues`` 1 and ``slot_store_engine`` "sync": each
            scored token loads the slot page ids of its slot state stores into registers (``nisa.register_load``) before
            the stores, which take the registers as their page offsets. The Sync queue then loads them while the store
            before is in flight, in place of loading each one after it, when the store's wait for that store ends. The
            same pages, so the bits do not change. False (default) keeps the program.
        slot_conv_from_inputs (bool): ``states`` with the ``sequential`` chain: the transposes of the slot conv stores
            read the slot conv windows of the scored tokens from the conv inputs by position (``_store_conv_states``
            ``window_seq``), so no GpSimd copy builds a window tile first (3 a layer, behind the GpSimd scatter DMAs).
            The bits do not change; False (default) keeps the program.
        slot_conv_after_ar (bool): ``states`` on the per-token path: the slot conv-state stores of the scored tokens
            (their transposes, rows copies and scatter DMAs) trace after the call's output all-reduce, and the
            transposes follow it in the schedule. The pool writes of a call wait for the pool write before them, so
            the first slot state store then waits for its own state only, not for the conv-state scatters, which run
            in the all-reduce flight after the last slot state store. The pool rows of the scatters (the index and
            off-write tiles, the page group's tiles with ``page_group``) are still written before the all-reduce. The
            same rows and pages, so the bits do not change. False (default) keeps the program.
        conv_products_engine (str): the engine of the conv tap products of the scored tokens and of the replay
            (``_conv_tokens`` one-pass, ``_conv_token`` per token): ``gpsimd`` (default) or
            ``vector``. On the VectorE the tap products leave the GpSimd queue, which then issues the conv-state
            gathers and scatters sooner. The same IEEE products, so the bits of finite values do not change.
        recurrence_group (int): None (default: one group of up to ``_RECURRENCE_GROUP``
            sequences), or the sequences of each group of the delta rule of a token (``_recurrence`` ``group``), 1 to B.
            Each group has its own prediction PSUM tile and its own rows PSUM tile, so the difference chain of a group
            starts when its own predictions land and runs beside the update of the other group. The same ops on the
            same values, so the bits do not change.
        slot_conv_rows (dict): ``states``, per-token path only: persistent SBUF tiles, living through the kernel, of the
            slot conv-state stores (``kda_slot_conv_rows_tiles``): ``kda_conv_rows_<g>``, the rows of group ``g``, and
            for ``batched_slot_conv_stores`` ``kda_conv_index_<g>`` / ``kda_conv_off_<g>``, its scatter rows. The
            compiler ends a DMA source's live range at the trigger, so a later token's tile can take the bytes of a
            per-call source and then waits for the store to finish (nki-dma-source-liveness). A persistent tile keeps
            its bytes, so no tile of the token chain waits for the slot conv stores. The bits do not change. None
            (default) keeps per-call tiles.
        slot_conv_waits (list): with ``slot_conv_rows``: a kernel-level record of the slot conv stores that read the
            tiles last. The writes of this call into them wait at runtime for those stores, and the record then takes
            this call's stores, so the edges cross the KDA calls. None keeps only the within-call edges.
        conv_zero_rows (nl.NkiTensor): None (default), or the bf16 zero tile of the conv-state scatters that lives
            through the kernel (``tkg_kda.kda_conv_zero_rows_tile``). The scatters of ``batched_conv_stores`` and
            ``batched_slot_conv_stores`` take the zeros of their off writes from it (``_store_conv_states``
            ``zero_rows``), so this call zeroes no tile of its own. The values do not change.
        page_group (dict): None (default), or the page set (``kda_page_group_tiles``) of the KV cache group of this
            layer: the layers of one group have one block table, so their page metadata is the same. The first call of
            the group computes the state and slot DMA pages and masks, the slot write pages and the conv gather and
            scatter rows, and writes them into the set; a later call reads them and computes none of them. ``states``
            only, with the inline prework. The values do not change when every layer of the group has the pages of the
            first one; the caller checks that.
        small_after_result (bool): the ``kda_w_small``, ``kda_params`` and ``kda_w_out`` loads (the gated weight
            loads on the GpSimd queue) also wait for the load of the result of the previous body's last output
            collective (``tkg_weight_dma_gate.gate_weight_dma`` ``after_result``, with the block's consumer gates: a semaphore
            wait), so the in-order GpSimd queue issues them after that collective, not in its flight
            (``decode_block.decode_block_body`` ``kda_small_after_out_ar``). The bytes and the instructions do not
            change, so the values do not change. False (default) keeps the program.

    Returns:
        nl.NkiTensor: [128, Q*B, H/128] bf16 SBUF, the all-reduced attention output in token-major hT
        layout. Its attribute ``side_effects`` holds the state-store and record-store DMAs, which the
        output does not depend on. Its attribute ``side_effect_sources`` holds their SBUF source tiles.

    Notes:
        - Run with LNC1 on Trn3.
        - Page 0 is the null page. The committed state read is off (zero state) at ``state_page`` 0, at
          a page id of P or more, or at ``state_position`` 0. The record read is off (all masked) at
          ``record_page`` 0 or a page id of P or more. A select, not a multiply, puts zeros in place of
          an off read and of a record beyond ``replay_len``, so a NaN in a stale page or a reused slot
          cannot reach the state or the output. A write to a page of 0 or P or more goes to page 0.
        - The replay reruns the per-token delta rule on a work copy of the committed state. It commits
          the new state and conv window only for the sequences with ``j < replay_len``. The committed
          state chains in the page dtype (bf16, or fp32 with ``state_dtype``), which equals a page round trip,
          so the replayed state is bitwise equal to ``replay_len`` decode steps.
        - The scoring chains the Q new tokens token to token in SBUF with no per-token page writes.
          Each new token equals one decode step.
        - The record store follows the record read of the same page, so the next call reads this
          call's records. It clamps each value to the finite range of its dtype, so a record page never
          holds a NaN or an Inf. The all-reduce runs on private HBM tiles, as the decode body does.
    """
    n_rows = x_hT.shape[1]
    hidden_tiles = x_hT.shape[2]
    width, hidden = kda_w_out_form(kda_w_out, kda_w_out_scale, kda_w_small, hidden_tiles)
    heads = width // HEAD_DIM
    n_seq = state_page.shape[0]
    q_tokens = n_rows // n_seq
    cols = kda_in_columns(heads)
    small = kda_small_columns(heads, conv_dtype)
    page = kda_page_offsets(heads, state_dtype)
    rec = kda_record_offsets(heads, q_tokens)
    n_in = cols["width"]
    n_blocks = cols["blocks"]
    n_pages = kda_page_bound(kv_pool.shape[0], paired_pages)
    span = kda_page_span(paired_pages)

    kernel_assert(
        hidden == hidden_tiles * P_MAX, f"x_hT has {hidden_tiles} tiles of 128, kda_w_out has {hidden} columns"
    )
    kernel_assert(
        width == heads * HEAD_DIM and 1 <= heads <= MAX_LOCAL_HEADS,
        f"local KDA width {width} must be 128 * Hl with 1 <= Hl <= {MAX_LOCAL_HEADS}",
    )
    w_in_shape = (
        (hidden, n_in) if kda_w_in_scale is None else mx_hT_shapes(hidden, n_in, mx_weight_format(kda_w_in)[0])[0]
    )
    kernel_assert(tuple(kda_w_in.shape) == w_in_shape, f"kda_w_in shape {kda_w_in.shape} must be {w_in_shape}")
    kernel_assert(
        tuple(kda_w_small.shape) == (P_MAX, small["small_width"]),
        f"kda_w_small shape {kda_w_small.shape} must be ({P_MAX}, {small['small_width']})",
    )
    a_exp_ready = kda_params_a_exp(kda_params, small, heads)
    kernel_assert(not paired_pages or state_dtype != "bf16", "paired_pages is a form of the fp32 KDA state pages")
    kernel_assert(not paired_pages or null_page % 2 == 0, f"paired_pages needs an even null_page, got {null_page}")
    kernel_assert(
        not (paired_pages and slot_pages_contiguous), "paired_pages takes the slot page ids as they are, not contiguous"
    )
    kernel_assert(
        span * kv_pool.shape[1] >= page["end"],
        f"cache page width {span * kv_pool.shape[1]} must be >= {page['end']}",
    )
    kernel_assert(
        1 <= n_seq <= MAX_SEQUENCES,
        f"state_page carries {n_seq} sequences, the KDA verify body supports 1 to {MAX_SEQUENCES}",
    )
    kernel_assert(1 <= q_tokens <= MAX_QUERIES, f"x_hT carries {q_tokens} tokens, the body supports 1 to {MAX_QUERIES}")
    kernel_assert(n_rows == q_tokens * n_seq, f"x_hT carries {n_rows} rows, must be Q*B = {q_tokens} * {n_seq}")
    kernel_assert(
        n_rows * heads <= _PSUM_BANK_FP32,
        f"the per-head gate of {n_rows} rows does not fit one PSUM bank; use fewer Q*B rows",
    )
    kernel_assert(
        tuple(state_page.shape) == (n_seq, 1)
        and tuple(record_page.shape) == (n_seq, 1)
        and tuple(replay_len.shape) == (n_seq, 1)
        and tuple(state_position.shape) == (n_seq, 1),
        f"state_page {state_page.shape}, record_page {record_page.shape}, replay_len {replay_len.shape} and "
        f"state_position {state_position.shape} must be ({n_seq}, 1)",
    )
    kernel_assert(
        kv_pool.shape[1] % P_MAX == 0 and rec["width"] <= span * kv_pool.shape[1] // P_MAX,
        f"a replay record of {rec['width']} columns does not fit the {PAGE_WIDTH} bf16 page",
    )
    readout = kda_default_readout(n_seq) if readout is None else readout
    kernel_assert(readout in KDA_READOUTS, f"readout {readout!r} must be one of {KDA_READOUTS}")
    kernel_assert(replay in ("records", "states"), f"replay {replay!r} must be 'records' or 'states'")
    kernel_assert(chain in KDA_CHAINS, f"chain {chain!r} must be one of {KDA_CHAINS}")
    chunked = chain == "chunked"
    kernel_assert(state_dtype in KDA_STATE_DTYPES, f"state_dtype {state_dtype!r} must be one of {KDA_STATE_DTYPES}")
    kernel_assert(
        state_dtype == "bf16" or kv_pool.shape[1] % 2 == 0,
        f"an fp32 state needs an even page width (an fp32 view of the bf16 pool), got {kv_pool.shape[1]}",
    )
    kernel_assert(
        state_dtype == "bf16" or not chunked,
        "the chunked chain keeps a bf16 state; an fp32 state needs chain='sequential'",
    )
    # The bf16 stationary parts of each token's fp32 state for the decoupled_bf16 prediction (none for a bf16 state).
    n_parts = _stationary_parts(state_dtype) if readout == "decoupled_bf16" else 0
    if chunked:
        kernel_assert(
            heads == 1 and n_seq <= 2 and q_tokens <= CHUNKED_MAX_QUERIES,
            f"the chunked chain needs one local head, 1 or 2 sequences and at most {CHUNKED_MAX_QUERIES} tokens; "
            f"got {heads} heads, {n_seq} sequences, {q_tokens} tokens",
        )
    batched = chain == "batched"
    kernel_assert(
        not batched or q_tokens <= BATCHED_MAX_QUERIES,
        f"the batched chain takes at most {BATCHED_MAX_QUERIES} tokens, got {q_tokens}",
    )
    records_mode = replay == "records"
    kernel_assert(slot_store_queues in (1, 2), f"slot_store_queues {slot_store_queues!r} must be 1 or 2")
    kernel_assert(
        slot_store_engine in ("sync", "gpsimd", "split"),
        f"slot_store_engine {slot_store_engine!r} must be 'sync', 'gpsimd' or 'split'",
    )
    kernel_assert(
        slot_store_engine == "sync" or (slot_store_queues == 1 and replay == "states"),
        "slot_store_engine 'gpsimd' / 'split' need replay='states' and slot_store_queues 1",
    )
    kernel_assert(
        not slot_store_regs or (replay == "states" and slot_store_queues == 1 and slot_store_engine == "sync"),
        "slot_store_regs needs replay='states', slot_store_queues 1 and slot_store_engine 'sync'",
    )
    kernel_assert(w_in_buffers >= 0, f"w_in_buffers must be 0 (the default) or a buffer count, got {w_in_buffers}")
    kernel_assert(
        w_in_ring is None or w_in_buffers in (0, w_in_ring.buffers),
        f"w_in_ring has {getattr(w_in_ring, 'buffers', 0)} buffers; w_in_buffers must be 0 or that count",
    )
    gemv_marks = (
        {} if w_out_after_w_in or w_out_after_w_in_fills or w_in_marks is not None or in_proj_hook is not None else None
    )
    kernel_assert(
        not w_out_after_w_in_fills or (kda_w_in_scale is not None and not w_out_after_w_in),
        "w_out_after_w_in_fills orders the kda_w_out load after the MXFP8 kda_w_in fills (kda_w_in_scale): not with "
        "w_out_after_w_in",
    )
    kernel_assert(
        not (w_out_after_w_in or w_in_marks is not None) or kda_w_in_scale is None,
        "w_out_after_w_in and w_in_marks take the BF16 kda_w_in, not an MXFP8 one (kda_w_in_scale)",
    )
    kernel_assert(
        not (w_out_after_w_in and kda_w_out_scale is not None),
        "w_out_after_w_in orders the BF16 kda_w_out load: not with an MXFP8 kda_w_out (kda_w_out_scale)",
    )
    kernel_assert(
        prework is None or not (state_loads_first or slot_pages_first),
        "state_loads_first and slot_pages_first move loads of the inline prework: not with a prework",
    )
    kernel_assert(
        not state_loads_after or prework is None,
        "state_loads_after orders the inline state loads of the call: not with a prework (kda_prework_ahead), whose "
        "state loads the block traces before the call",
    )
    kernel_assert(
        not (state_loads_first or slot_pages_first) or not records_mode,
        "state_loads_first and slot_pages_first need replay='states'",
    )
    kernel_assert(
        not records_mode or (not token_tiles and slot_store_queues == 1),
        "token_tiles and slot_store_queues > 1 need replay='states'",
    )
    kernel_assert(not records_mode or not slot_pages_contiguous, "slot_pages_contiguous needs replay='states'")
    kernel_assert(not records_mode or not batched_slot_conv_stores, "batched_slot_conv_stores needs replay='states'")
    kernel_assert(update_bcast_depth in (2, 3, 4), f"update_bcast_depth must be 2, 3 or 4, got {update_bcast_depth!r}")
    kernel_assert(
        head_pipeline in HEAD_PIPELINE_FORMS,
        f"head_pipeline must be one of {HEAD_PIPELINE_FORMS}, got {head_pipeline!r}",
    )
    g_last = head_pipeline == "g_last"
    heads_mode = head_pipeline == "heads"
    kernel_assert(
        not heads_mode
        or (
            kda_w_in_scale is not None
            and w_in_ring is None
            and not records_mode
            and chain == "sequential"
            and token_tiles
            and readout != "reference"
            and heads >= 2
            and not late_keep_warm
        ),
        "head_pipeline heads needs the MXFP8 head-major kda_w_in (kda_w_in_scale), no w_in_ring, the sequential states "
        "chain with token_tiles, a decoupled readout, at least 2 heads and no late_keep_warm",
    )
    kernel_assert(
        not g_last or kda_w_in_scale is not None, "head_pipeline g_last takes the MXFP8 kda_w_in (kda_w_in_scale)"
    )
    kernel_assert(
        not in_proj_x_stationary or (kda_w_in_scale is not None and not g_last),
        "in_proj_x_stationary takes the MXFP8 kda_w_in (kda_w_in_scale), not with head_pipeline g_last",
    )
    kernel_assert(
        not state_loads_after_ring or state_loads_first,
        "state_loads_after_ring orders the loads of state_loads_first: give state_loads_first",
    )
    # state_loads_after_ring: the state loads follow the last prefetch fill of the ring (schedule only).
    ring_follow = (w_in_ring.fills[-1],) if state_loads_after_ring and w_in_ring is not None and w_in_ring.fills else ()
    # The VectorE selects of those later loads then run after the input projection's quantize (its quant hook), so
    # VectorE does not wait for the loads ahead of the quantize. The MXFP8 kda_w_in only (the hook's GEMV).
    deferred_selects = [] if ring_follow and kda_w_in_scale is not None else None
    # A prework with defer_selects (kda_verify_prework) gives the VectorE selects of its loads to the body, which runs
    # them where it runs those of state_loads_after_ring: after the input projection's quantize (the MXFP8 kda_w_in),
    # else after the projection. So VectorE does not wait for the prework loads where the caller traced them.
    prework_selects = None if prework is None else prework.get("deferred")
    if prework_selects is not None:
        deferred_selects = [*(deferred_selects or []), *prework_selects]

    def run_deferred_selects(after=()):
        while deferred_selects:
            traced = deferred_selects.pop(0)()
            for instr in traced if isinstance(traced, list) else [traced]:
                _schedule_after(instr, after)

    def proj_quant_hook(quant):
        run_deferred_selects((quant,))
        if x_quant_hook is not None:
            x_quant_hook(quant)

    proj_hook = proj_quant_hook if deferred_selects is not None and kda_w_in_scale is not None else x_quant_hook
    kernel_assert(
        not slot_conv_from_inputs or (not records_mode and chain == "sequential"),
        "slot_conv_from_inputs needs replay='states' and the sequential chain",
    )
    kernel_assert(
        conv_products_engine in ("gpsimd", "vector"),
        f"conv_products_engine must be 'gpsimd' or 'vector', got {conv_products_engine!r}",
    )
    kernel_assert(
        recurrence_group is None or (isinstance(recurrence_group, int) and recurrence_group >= 1),
        f"recurrence_group must be None or a positive int, got {recurrence_group!r}",
    )
    kernel_assert(slot_conv_rows is not None or slot_conv_waits is None, "slot_conv_waits needs slot_conv_rows")
    kernel_assert(
        page_group is None or (not records_mode and (prework is None or prework.get("page_group") is page_group)),
        "page_group keeps the page metadata of the states prework: replay='states' with no prework, or a prework "
        "traced with the same page set (kda_verify_prework page_group)",
    )
    if not records_mode:
        kernel_assert(
            state_slot_page is not None and write_state_slots is not None,
            "replay='states' needs state_slot_page and write_state_slots",
        )
        kernel_assert(
            tuple(state_slot_page.shape) == (n_seq, 1) and tuple(write_state_slots.shape) == (n_seq, q_tokens),
            f"state_slot_page {state_slot_page.shape} must be ({n_seq}, 1) and write_state_slots must be "
            f"({n_seq}, {q_tokens})",
        )

    bf16 = nl.bfloat16
    fp32 = nl.float32
    n_conv = N_CONV_PROJ * heads
    taps = CONV_TAPS - 1
    n_cm = n_blocks + heads

    # Step 1: shared constants.
    if consts is not None:
        # The kernel's all-ones tiles (kda_verify_constants), set once for the kernel.
        ones_rows, ones_sq = consts["ones_rows"], consts["ones_sq"]
        kernel_assert(
            tuple(ones_sq.shape) == (P_MAX, P_MAX)
            and ones_sq.dtype == fp32
            and ones_rows.dtype == bf16
            and ones_rows.shape[0] >= n_in - n_blocks * P_MAX
            and ones_rows.shape[1] == P_MAX,
            "consts takes the fp32 [128, 128] ones_sq and a bf16 [>= rows, 128] ones_rows (kda_verify_constants)",
        )
    else:
        ones_rows = nl.ndarray((n_in - n_blocks * P_MAX, P_MAX), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=ones_rows, value=1.0)
        ones_sq = nl.ndarray((P_MAX, P_MAX), dtype=fp32, buffer=nl.sbuf)
        nisa.memset(dst=ones_sq, value=1.0)
    ident = nl.shared_identity_matrix(n=P_MAX, dtype=bf16)
    # One row selector for the Q tokens of the call (a constant tile; ROW_SELECTOR_ONCE).
    row_selector = None
    if ROW_SELECTOR_ONCE and bf16_update_rows and readout == "decoupled_bf16":
        row_selector = _bf16_row_selector(ident, heads)
    ident32 = nl.shared_identity_matrix(n=P_MAX, dtype=fp32)

    # Step 2: the small weights, the params, and the page metadata of the state and the records.
    pages_loads = []
    early_pages = None
    if slot_pages_first:
        early_pages = _group_tile(
            page_group,
            "slot_write_pages",
            lambda: _write_slot_pages(
                write_state_slots,
                n_pages,
                q_tokens,
                f"{tag}_slot",
                loads=pages_loads,
                null_page=null_page,
                contiguous=slot_pages_contiguous,
                even_only=paired_pages,
            ),
        )
    # A prework with replay_ahead replayed the records and loaded the small weights (kda_verify_prework).
    replayed = prework is not None and prework.get("replayed") is not None
    if replayed:
        kernel_assert(
            records_mode and chain == "sequential",
            "a prework that replayed the records (replay_ahead) needs replay='records' and the sequential chain",
        )
        expected = _replay_options(
            conv_dtype,
            readout,
            recurrence_group,
            replay_prework,
            conv_products_engine,
            bf16_update_rows,
            update_bcast_depth,
        )
        kernel_assert(
            prework["replayed"] == expected,
            f"the prework replayed with {prework['replayed']}; the body has {expected}",
        )
        small_sb, params_sb = prework["small_sb"], prework["params_sb"]
        kernel_assert(
            tuple(params_sb.shape) == (P_MAX, kda_params.shape[1]),
            f"the prework loaded params {params_sb.shape}; the body has kda_params {kda_params.shape}",
        )
    else:
        small_sb = nl.ndarray((P_MAX, small["small_width"]), dtype=bf16, buffer=nl.sbuf)
        weight_queue = kda_weight_dma_options(weight_dma)
        gate_weight_dma(
            nisa.dma_copy(dst=small_sb, src=kda_w_small, **weight_queue, name=f"{tag}_kda_w_small_load"),
            after_result=small_after_result,
        )
        params_sb = nl.ndarray((P_MAX, kda_params.shape[1]), dtype=fp32, buffer=nl.sbuf)
        gate_weight_dma(
            nisa.dma_copy(dst=params_sb, src=kda_params, **weight_queue, name=f"{tag}_kda_params_load"),
            after_result=small_after_result,
        )
    inline = prework is None
    kernel_assert(
        inline or (prework.get("null_page", 0) == null_page and prework.get("paired_pages", False) == paired_pages),
        "prework must come from kda_verify_prework with the null_page and the paired_pages of the body",
    )
    if inline and page_group is not None and page_group["filled"]:
        prework = {key: page_group["tiles"][key] for key in _PAGE_META_KEYS}
    elif inline:
        prework = _verify_page_meta(
            state_page,
            record_page,
            replay_len,
            state_position,
            n_pages,
            q_tokens,
            tag,
            replay,
            null_page=null_page,
            even_only=paired_pages,
        )
        if page_group is not None:
            for key in _PAGE_META_KEYS:
                nisa.tensor_copy(dst=page_group["tiles"][key], src=prework[key], engine=nisa.engine.vector)
    if state_loads_first:
        _verify_state_loads(
            prework,
            kv_pool,
            heads,
            q_tokens,
            tag,
            replay,
            replay_len,
            state_slot_page,
            write_state_slots,
            state_dtype,
            early_pages,
            conv_load_gather=conv_load_gather,
            null_page=null_page,
            slot_pages_contiguous=slot_pages_contiguous,
            paired_pages=paired_pages,
            page_group=page_group,
            states_after_gate_result=states_after_gate_result,
            sp_loads_after=tuple(state_loads_after),
            merged_state_loads=merged_state_loads,
            loads_follow=ring_follow,
            deferred=deferred_selects,
        )

    # Step 3: the output projection weight load (issued before the input projection so it overlaps; with
    # w_out_after_w_in, after it).
    if kda_w_out_scale is not None and not w_out_after_w_in_fills:
        w_out_sb = load_kda_w_out(
            kda_w_out, kda_w_out_scale, heads, hidden, tag, 3, after_result=small_after_result, weight_dma=weight_dma
        )
    elif kda_w_out_scale is not None:
        w_out_sb = None  # w_out_after_w_in_fills: loaded after the input projection
    else:
        w_out_sb = nl.ndarray((P_MAX, heads, hidden), dtype=bf16, buffer=nl.sbuf)

        def load_w_out():
            return gate_weight_dma(
                nisa.dma_copy(
                    dst=w_out_sb,
                    src=kda_w_out.reshape((heads, P_MAX, hidden)).permute((1, 0, 2)),
                    priority=dma_priority(3),
                    **kda_weight_dma_options(weight_dma),
                    name=f"{tag}_kda_w_out_load",
                ),
                after_result=small_after_result,
            )

        if not (w_out_after_w_in or w_out_after_w_in_fills):
            load_w_out()

    # Step 4: the channel-major projection tile of the new tokens (q, k, v, g, f_a blocks, then b).
    # The input projection runs in row chunks that each keep its PSUM in one bank: a step of up to
    # 32 rows of each replica fits one chunk, and a 64-row verify step (B = 8, Q = 8) runs two.
    a_exp = kda_a_exp(params_sb, small, heads, a_exp_ready)
    conv_w = _conv_weights(small_sb, params_sb, small, n_conv, conv_dtype)
    conv_dt = fp32 if conv_dtype == "fp32" else bf16
    conv_engine = getattr(nisa.engine, conv_products_engine)
    cm = nl.ndarray((P_MAX, n_rows, n_cm), dtype=bf16, buffer=nl.sbuf)
    b_rows = n_in - n_blocks * P_MAX
    b_logits = nl.ndarray((b_rows, n_rows), dtype=bf16, buffer=nl.sbuf)
    proj_rows = _PSUM_BANK_FP32 // div_ceil(n_in, P_MAX)
    kernel_assert(
        not g_last or n_rows <= proj_rows,
        f"head_pipeline g_last takes one row chunk of the input projection ({proj_rows} rows), got {n_rows}",
    )
    # g_last: the output-gate (g) blocks are the late blocks of the projection (columns 3 Hl .. 4 Hl - 1 of cm).
    g_blocks = tuple(range(3 * heads, 4 * heads)) if g_last else ()
    late_proj: dict = {}
    head_gemv = None
    if heads_mode:
        # heads: the head-major projection in passes; pass A now (f_a, b, q / k / v of heads 0 .. Hl - 2).
        major = kda_in_columns_head_major(heads)
        ring_buffers = w_in_buffers or _w_in_buffers(heads, n_seq)
        n_groups = hidden_tiles // MX_LANES
        slot_cols = max(major["part_a"], n_in - major["part_a"])
        n_slots = n_groups if ring_buffers is None else ring_buffers * n_in // slot_cols
        head_gemv = HeadMajorGemv(
            x_hT[0:P_MAX, 0:n_rows, 0:hidden_tiles],
            kda_w_in,
            kda_w_in_scale,
            tag,
            heads,
            n_slots,
            marks=gemv_marks,
            quant_hook=x_quant_hook,
            fills_after=tuple(pages_loads),
        )
        psum_a = head_gemv.run("A")
        if gemv_marks is not None:
            gemv_marks["end"] = head_gemv.end
        n_a = N_CONV_PROJ * (heads - 1)
        cm_a = nl.ndarray((P_MAX, n_rows, n_a), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cm_a, src=psum_a[0:P_MAX, 0:n_a, 0:n_rows].permute((0, 2, 1)), engine=nisa.engine.vector)
        nisa.tensor_copy(
            dst=cm[0:P_MAX, 0:n_rows, 4 * heads : 4 * heads + 1],
            src=psum_a[0:P_MAX, n_a : n_a + 1, 0:n_rows].permute((0, 2, 1)),
            engine=nisa.engine.vector,
        )
        nisa.tensor_copy(
            dst=b_logits[0:b_rows, 0:n_rows], src=psum_a[0:b_rows, n_a + 1, 0:n_rows], engine=nisa.engine.scalar
        )
    for first in range(0, n_rows, proj_rows) if not heads_mode else ():
        count = min(proj_rows, n_rows - first)
        ring = w_in_ring if first == 0 else None
        proj_psum = _gemv_channel_major(
            x_hT[0:P_MAX, first : first + count, 0:hidden_tiles],
            kda_w_in,
            f"{tag}_proj{first}",
            buffers=ring.buffers if ring is not None else w_in_buffers or _w_in_buffers(heads, n_seq),
            **({"w_scale": kda_w_in_scale} if kda_w_in_scale is not None else {}),
            **({"ring": ring} if ring is not None else {}),
            **({"marks": gemv_marks} if gemv_marks is not None else {}),
            **({"quant_hook": proj_hook} if proj_hook is not None and first == 0 else {}),
            fills_after=tuple(pages_loads) if first == 0 else (),
            **({} if in_proj_lanes_view else {"lanes_view": False}),
            **({"psum_bufs": w_in_psum_bufs} if w_in_psum_bufs != 1 and n_rows <= W_IN_PSUM_MAX_ROWS else {}),
            **({"refill_queues": w_in_refill_queues} if w_in_refill_queues != "alternate" else {}),
            **({"late_blocks": g_blocks, "late": late_proj} if g_last else {}),
            **({"x_stationary": True} if in_proj_x_stationary else {}),
        )
        if in_proj_x_stationary:
            # The token-major sums: the served roundings, then TensorE transposes into cm and b_logits.
            in_proj_copies_x_stationary(proj_psum, cm, b_logits, first, n_blocks, b_rows)
            continue
        # g_last: the g blocks come from the late pass; the other blocks are copied here.
        for b0, b1 in ((0, 3 * heads), (4 * heads, n_blocks)) if g_last else ((0, n_blocks),):
            nisa.tensor_copy(
                dst=cm[0:P_MAX, first : first + count, b0:b1],
                src=proj_psum[0:P_MAX, b0:b1, 0:count].permute((0, 2, 1)),
                engine=nisa.engine.vector,
            )
        nisa.tensor_copy(
            dst=b_logits[0:b_rows, first : first + count],
            src=proj_psum[0:b_rows, n_blocks, 0:count],
            engine=nisa.engine.scalar,
        )
    # state_loads_after_ring: any select the quant hook did not run runs here, after the projection.
    run_deferred_selects()
    if w_out_after_w_in:
        after_w_in_loads(load_w_out(), gemv_marks)
    # w_out_after_w_in_fills: the output projection weight after the last two MXFP8 kda_w_in fills.
    if w_out_after_w_in_fills and kda_w_out_scale is not None:
        w_out_sb = load_kda_w_out(
            kda_w_out,
            kda_w_out_scale,
            heads,
            hidden,
            tag,
            3,
            after_result=small_after_result,
            weight_dma=weight_dma,
            # The last instruction of the input projection reads the last fills: a wait on it is a wait on their
            # completion with no runtime edge on a DMA (tkg_weight_dma_gate.wait_for_completion).
            after=(gemv_marks["end"],) if gemv_marks.get("end") is not None else tuple(gemv_marks["fills"][-2:]),
        )
    elif w_out_after_w_in_fills:
        after_w_in_loads(load_w_out(), gemv_marks)
    if w_in_marks is not None:
        w_in_marks.update(gemv_marks)
    before_slot_stores = ()
    if in_proj_hook is not None:
        in_proj_hook(gemv_marks)
        before_slot_stores = tuple(gemv_marks.get("before_slot_stores", ()))
    b_masked = nl.ndarray((b_rows, n_rows, heads), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=b_masked,
        data1=ident[0:b_rows, 0:heads].expand_dim(1).broadcast(1, n_rows),
        data2=b_logits.expand_dim(2).broadcast(2, heads),
        op=nl.multiply,
    )
    b_psum = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=b_psum, stationary=ones_rows[0:b_rows, 0:P_MAX], moving=b_masked)
    nisa.tensor_copy(dst=cm[0:P_MAX, 0:n_rows, n_blocks:n_cm], src=b_psum, engine=nisa.engine.scalar)

    # Step 5: decay gate, output gate and beta of the new tokens (do not need the states).
    decay = _decay_gate(cm, small_sb, params_sb, a_exp, ones_sq, heads, small)
    n_gb = n_cm - 3 * heads
    gb_exp = nl.ndarray((P_MAX, n_rows, n_gb), dtype=fp32, buffer=nl.sbuf)
    gb_sig = nl.ndarray((P_MAX, n_rows, n_gb), dtype=fp32, buffer=nl.sbuf)

    def gate_sigmoids(c0: int, c1: int) -> None:
        """Write the sigmoid of cm columns 3 Hl + c0 .. 3 Hl + c1 - 1 to gb_sig (elementwise, so the bits do not change)."""
        nisa.activation(
            dst=gb_exp[0:P_MAX, 0:n_rows, c0:c1],
            op=nl.exp,
            data=cm[0:P_MAX, 0:n_rows, 3 * heads + c0 : 3 * heads + c1],
            scale=-1.0,
        )
        _divide_by_one_plus(gb_sig[0:P_MAX, 0:n_rows, c0:c1], gb_exp[0:P_MAX, 0:n_rows, c0:c1], 1.0)

    if g_last:
        # The f_a and beta columns now; the g matmuls of the kept weight groups run here, behind the decay gate's.
        gate_sigmoids(heads, n_gb)
        g_psum = late_proj["run"]()
    elif heads_mode:
        # heads: the f_a and beta columns now; the last head's pass B here, so it runs beside the first heads' pass.
        gate_sigmoids(heads, n_gb)
        psum_b = head_gemv.run("B")
    else:
        nisa.activation(dst=gb_exp, op=nl.exp, data=cm[0:P_MAX, 0:n_rows, 3 * heads : n_cm], scale=-1.0)
        _divide_by_one_plus(gb_sig, gb_exp, 1.0)
    out_gate = gb_sig[0:P_MAX, 0:n_rows, 0:heads]
    beta = gb_sig[0:P_MAX, 0:n_rows, heads + 1 : 2 * heads + 1]

    if inline and not state_loads_first:
        _verify_state_loads(
            prework,
            kv_pool,
            heads,
            q_tokens,
            tag,
            replay,
            replay_len,
            state_slot_page,
            write_state_slots,
            state_dtype,
            early_pages,
            conv_load_gather=conv_load_gather,
            null_page=null_page,
            slot_pages_contiguous=slot_pages_contiguous,
            paired_pages=paired_pages,
            page_group=page_group,
            states_after_gate_result=states_after_gate_result,
            sp_loads_after=tuple(state_loads_after),
            merged_state_loads=merged_state_loads,
        )
    state_write, state_sb, conv_sb = prework["state_write"], prework["state_sb"], prework["conv_sb"]
    state_dt = bf16 if state_dtype == "bf16" else fp32
    kernel_assert(
        state_sb.dtype == state_dt,
        f"the prework state tile is {state_sb.dtype}; state_dtype {state_dtype!r} needs {state_dt}",
    )
    kernel_assert(
        slot_state_ring is None or not records_mode,
        "slot_state_ring is the per-token slot-state source: replay='states' only",
    )
    kernel_assert(
        slot_state_ring is not None or slot_state_depth in (1, 2),
        f"slot_state_depth {slot_state_depth!r} must be 1 or 2 for an in-call ring",
    )
    kernel_assert(
        not slot_ring_per_call
        or (slot_state_ring is None and token_tiles and not records_mode and not (batched or chunked)),
        "slot_ring_per_call needs token_tiles on the sequential states chain and no slot_state_ring",
    )
    ring_depth = (
        len(slot_state_ring) if slot_state_ring is not None else (q_tokens if slot_ring_per_call else slot_state_depth)
    )
    kernel_assert(
        not shared_state_ring or (not records_mode and not (batched or chunked) and ring_depth == 2),
        f"shared_state_ring needs replay='states', chain 'sequential' and a slot-state ring of 2 tiles, got replay "
        f"{replay!r}, chain {chain!r} and {ring_depth} tiles",
    )
    if slot_state_ring is not None:
        # The batched chain writes the Q slot states at once, so its ring holds one tile per token. The sequential chain
        # takes 1 to Q tiles: with d tiles token j writes ring[j % d] and waits only for the stores of token j - d.
        depths = (q_tokens,) if batched else tuple(range(1, q_tokens + 1))
        kernel_assert(ring_depth in depths, f"slot_state_ring takes {depths} tiles here, got {ring_depth}")
        for tile in slot_state_ring:
            kernel_assert(
                tuple(tile.shape) == (P_MAX, n_seq, heads, HEAD_DIM) and tile.dtype == state_dt,
                f"a slot_state_ring tile must be ({P_MAX}, {n_seq}, {heads}, {HEAD_DIM}) of {state_dt}, "
                f"got {tile.shape} of {tile.dtype}",
            )
    if records_mode:
        replay_pred, cm_rec, decay_rec, beta_rec = (
            prework[k] for k in ("replay_pred", "cm_rec", "decay_rec", "beta_rec")
        )
        rec_load, record_dma = prework["rec_load"], prework["record_dma"]
    if batched:
        # The records form solves the Q records and the Q new tokens of each pair as one chain of 2Q tokens.
        batched_masks = _batched_masks(2 * q_tokens if records_mode else q_tokens)
    if records_mode and batched:
        # Step 8 (batched): the records of the accepted prefix as identity-padded chain inputs (a record beyond
        # replay_len is an identity step), and the conv window after replay_len records. Step 10 solves them with the
        # new tokens and commits the state.
        batched_records = _batched_records(
            state_sb, conv_sb, replay_pred, cm_rec, decay_rec, beta_rec, conv_w, conv_dt, ones_sq
        )
    elif records_mode:
        # A prework with replay_ahead replayed the records already.
        if not replayed:
            _replay_records(
                state_sb,
                conv_sb,
                replay_pred,
                cm_rec,
                decay_rec,
                beta_rec,
                conv_w,
                conv_dt,
                conv_engine,
                ones_sq,
                ident,
                ident32,
                readout,
                tag,
                replay_prework,
                bf16_update_rows,
                n_parts,
                row_selector,
                recurrence_group,
                update_bcast_depth,
            )
    else:
        slot_write_pages = prework["slot_write_pages"]
    # slot_store_regs: the slot page ids of each scored token's slot state stores in registers, loaded one store ahead,
    # so the Sync queue loads each one while a store before it is in flight. Token 0's go ahead of the committed-state
    # stores below, so a store sits between each load and its own store too.
    slot_regs = {}
    if slot_store_regs:
        for seq in range(n_seq):
            _load_slot_reg(slot_regs, slot_write_pages, 0, seq, n_seq)

    # Step 9: write the committed (replayed) state and conv window to the state page.
    store_window = nl.ndarray((P_MAX, n_seq, n_conv, CONV_TAPS), dtype=bf16, buffer=nl.sbuf)
    window_copy = nisa.tensor_copy(
        dst=store_window[0:P_MAX, 0:n_seq, 0:n_conv, 1:CONV_TAPS], src=conv_sb, engine=nisa.engine.gpsimd
    )
    if slot_ring_keep_defer is not None:
        # slot_ring_keep_defer: the keep-reads of the earlier calls' rings, after this call's first GpSimd copy.
        kernel_assert(
            slot_ring_per_call and slot_ring_keep_sink is not None,
            "slot_ring_keep_defer needs slot_ring_per_call and slot_ring_keep_sink",
        )
        kda_flush_ring_keeps(slot_ring_keep_defer, slot_ring_keep_sink, after=window_copy)
    if batched:
        # Row isolation (batched): a non-finite conv window stores zeros, so it cannot poison its page for a later request.
        for tap in range(1, CONV_TAPS):
            _store_finite(store_window[0:P_MAX, 0:n_seq, 0:n_conv, tap], nisa.engine.gpsimd)
    stores = []

    def store_committed_states():
        for seq in range(n_seq):
            store = nisa.dma_copy(
                dst=_recurrent_state_view(
                    kv_pool, state_write[0:1, seq : seq + 1], heads, page["recurrent"], state_dtype
                ),
                src=state_sb[0:P_MAX, seq, 0:heads, 0:HEAD_DIM],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_kda_recurrent_state_store_{seq}",
            )
            # Only the next call reads the committed state. Ahead of the load of the previous body's output on the SP
            # queue, these stores (each after the slot-state select) held that load, which the residual add waits for.
            if commit_after_w_in and w_in_ring is not None and w_in_ring.fills:
                # commit_after_w_in: behind the input projection's prefetch fills on the Sync queue.
                _schedule_after(store, (w_in_ring.fills[-1],))
            stores.append(order_after_gate_result(store) if states_after_gate_result else store)

    if not (records_mode and batched):
        store_committed_states()
    committed_scatter = None
    if batched_conv_stores:
        committed_scatter = _group_index_sets(
            page_group,
            "committed",
            n_seq,
            n_conv,
            lambda kept: _conv_scatter_rows(
                kv_pool, state_write, n_conv, null_page=null_page, **({} if kept is None else {"persistent": kept})
            ),
        )
    committed_conv_dmas = _store_conv_states(
        kv_pool,
        state_write,
        store_window,
        f"{tag}_state",
        keep=scatter_keep,
        scatter_rows=committed_scatter,
        zero_rows=conv_zero_rows,
    )
    stores.extend(committed_conv_dmas)

    # A slot store reads sc_state and the conv rows tile, which a later token overwrites in place. The
    # compiler ends a DMA source's live range at the trigger, not at completion (nki-dma-source-liveness),
    # so the next write of either tile waits for the previous token's slot DMAs with a dependency edge.
    # Empty in records mode (no slot stores), so the records path adds no edge.
    prev_state_dmas: list = []
    prev_conv_dmas: list = committed_conv_dmas
    # slot_conv_after_ar: the slot conv-state stores of the per-token path, traced after the output all-reduce.
    late_slot_conv: list = []

    # Step 10: score the Q new tokens from the committed state, chaining in SBUF with no page writes.
    # At one head per rank (or with token_tiles in states mode) each token writes its own state tiles, so no token
    # waits for a read of the one before. The records replay of more than two sequences keeps the shared tiles. The
    # batched chain takes the one-pass conv and norms of the per-token path and writes no per-token tiles in records.
    per_token = (heads == 1 and (n_seq <= 2 or not records_mode)) or token_tiles or batched
    kernel_assert(score_prework in REPLAY_PREWORK_FORMS, f"score_prework must be one of {REPLAY_PREWORK_FORMS}")
    # The records scoring of the shared-tile path takes the one-pass conv and readout vectors too: none of them reads
    # the recurrent state, and records mode stores no per-token conv window (the window is only the next token's input).
    score_one_pass = records_mode and not per_token and not chunked and score_prework == "one_pass"
    kernel_assert(slot_conv_rows is None or per_token, "slot_conv_rows takes the per-token states path (token_tiles)")
    kernel_assert(not slot_conv_from_inputs or per_token, "slot_conv_from_inputs takes the per-token states path")
    kernel_assert(
        not slot_conv_after_ar or (per_token and not records_mode),
        "slot_conv_after_ar takes the per-token states path",
    )
    kernel_assert(
        slot_state_ring is None or len(slot_state_ring) <= 2 or batched or per_token,
        f"a slot_state_ring of {len(slot_state_ring or ())} tiles takes the per-token states path (token_tiles)",
    )
    # The slot-state store ring: a 2-tile ring (sequential states only) carries each token's output state, so a
    # persistent ring tile keeps its bytes past the store trigger and the next write waits for the store
    # (nki-dma-source-liveness). The sequential per-token path cycles the ring; batched takes the kernel's Q-tile ring
    # when given (token j writes ring[j]); chunked keeps per-call tiles. slot_state_ring (from the kernel) is
    # persistent, so the edges cross the KDA calls through slot_state_waits; None allocates the ring in the call,
    # which keeps only the within-call edges.
    # shared_state_ring: the shared-tile path (more than one head, no token_tiles) cycles the 2-tile ring as the
    # per-token path does, in place of one state tile updated in place.
    ring_tokens = shared_state_ring and not per_token
    use_ring = (per_token or ring_tokens) and not records_mode and not (batched or chunked)
    batched_ring = batched and not records_mode and slot_state_ring is not None
    shared_ring = not per_token and not ring_tokens and not records_mode and slot_state_ring is not None
    ring = None
    ring_last = None
    if use_ring or batched_ring:
        ring = (
            list(slot_state_ring)
            if slot_state_ring is not None
            else [
                nl.ndarray((P_MAX, n_seq, heads, HEAD_DIM), dtype=state_dt, buffer=nl.sbuf) for _ in range(ring_depth)
            ]
        )
        ring_last = [list((slot_state_waits or {}).get(r, ()) or ()) for r in range(len(ring))]
    kernel_assert(
        not bf16_cols_once or (bf16_update_rows and per_token and not records_mode and not (batched or chunked)),
        "bf16_cols_once needs bf16_update_rows on the per-token states path (token_tiles, the sequential chain)",
    )
    kernel_assert(
        type(late_keep_warm) is int and 0 <= late_keep_warm <= LATE_KEEP_WARM_MAX,
        f"late_keep_warm must be an int from 0 to {LATE_KEEP_WARM_MAX}, got {late_keep_warm!r}",
    )
    kw_zero = None
    if late_keep_warm:
        # late_keep_warm: the zero operands of the keep-warm matmuls, set at the call start (off the token chain).
        kw_zero = nl.ndarray((P_MAX, _KEEP_WARM_COLS), dtype=bf16, buffer=nl.sbuf, name=f"{tag}_kw_zero")
        nisa.memset(dst=kw_zero, value=0.0, engine=nisa.engine.gpsimd)
    cols_sets = None
    if per_token:
        if bf16_cols_once:
            # bf16_cols_once: the update's rows tiles, zeroed here before the conv pass (so ahead of the slot conv
            # scatters on GpSimd); token j takes set j % 2, one tile per sequence group of _recurrence.
            per_group = min(n_seq, _RECURRENCE_GROUP if recurrence_group is None else recurrence_group)
            cols_sets = [
                {
                    first: bf16_diff_cols(heads, min(n_seq, first + per_group) - first)
                    for first in range(0, n_seq, per_group)
                }
                for _ in range(2)
            ]
        # The batched chain writes the states of the scored tokens only in states mode (to their slots).
        sc_states = [state_sb]
        if use_ring or batched_ring:
            # Token j writes ring[j % depth]; token j + 1 reads it as its state_in (sc_states[j + 1] = ring[j % depth]).
            sc_states += [ring[t % len(ring)] for t in range(q_tokens)]
        elif not (batched and records_mode):
            sc_states += [
                nl.ndarray((P_MAX, n_seq, heads, HEAD_DIM), dtype=state_dt, buffer=nl.sbuf) for _ in range(q_tokens)
            ]
        fp32_sets = None
        if not batched:
            fp32_sets = [
                [nl.ndarray((P_MAX, heads, HEAD_DIM), dtype=fp32, buffer=nl.sbuf) for _ in range(n_seq)]
                for _ in range(2)
            ]
        # The conv inputs of all Q tokens are known after the input projection, so the conv, the norms and the
        # moving vectors of all tokens run in one pass before the chain, and the slot conv windows in one store.
        if heads_mode:
            # heads: the pass of heads 0 .. Hl - 2 (beside the projection's pass B), the first quarter of pass C, then
            # the pass of the last head. Both fill the tiles of the one chain.
            staged = _HeadStages(conv_sb, conv_w, decay, ones_sq, readout, heads, n_seq, q_tokens, conv_dt, conv_engine)
            staged.run(0, heads - 1, cm_a)
            head_gemv.run("C", _quarter(head_gemv.n_groups, 0))
            cm_b = nl.ndarray((P_MAX, n_rows, N_CONV_PROJ), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cm_b, src=psum_b.permute((0, 2, 1)), engine=nisa.engine.vector)
            staged.run(heads - 1, heads, cm_b)
            conv_all, conv_seq = staged.v_rows, staged.conv_seq
            k_all, q_all, mv_all, qk_all = staged.k_n, staged.q_n, staged.mv, staged.qk
        else:
            conv_all, conv_seq = _conv_tokens(
                conv_sb,
                cm[0:P_MAX, 0:n_rows, 0:n_conv],
                conv_w,
                q_tokens,
                conv_dt,
                products_engine=conv_engine,
            )
        if heads_mode:
            pass  # the staged pass filled k_all, q_all, mv_all and qk_all
        elif batched:
            # The batched chain takes only the normed k and q.
            k_all, q_all = _qk_norms(conv_all, ones_sq)
            mv_all = qk_all = None
        else:
            k_all, q_all, mv_all, qk_all = _readout_vectors(
                conv_all, decay[0:P_MAX, 0:n_rows, 0:heads], ones_sq, readout
            )
        if not records_mode:
            # slot_conv_from_inputs: the slot conv stores read the windows from conv_seq (no window tile).
            slot_all = None
            if not slot_conv_from_inputs:
                slot_all = nl.ndarray((P_MAX, n_rows, n_conv, CONV_TAPS), dtype=bf16, buffer=nl.sbuf)
            for tap in range(taps if slot_all is not None else 0):
                nisa.tensor_copy(
                    dst=slot_all[0:P_MAX, 0:n_rows, 0:n_conv, 1 + tap],
                    src=conv_seq[0:P_MAX, 1 + tap : 1 + tap + q_tokens].flatten_dims(1, 2),
                    engine=nisa.engine.gpsimd,
                )
                if batched:
                    _store_finite(slot_all[0:P_MAX, 0:n_rows, 0:n_conv, 1 + tap], nisa.engine.gpsimd)
            slot_pages_rows = slot_write_pages.flatten_dims(1, 2)
            # Persistent rows (and scatter rows) tiles: their writes wait for the previous call's slot conv stores.
            kept = (
                {} if slot_conv_rows is None else {"persistent": slot_conv_rows, "wait": tuple(slot_conv_waits or ())}
            )
            # The scatter rows are written here also when the stores trace after the all-reduce (slot_conv_after_ar):
            # the first call of a page group writes the group's rows tiles before it marks the group filled, and the
            # later calls of the group read them.
            slot_scatter = None
            if batched_slot_conv_stores and page_group is not None:
                # The group's rows tiles in place of the kept ones: written once by the group's first call.
                slot_scatter = _group_index_sets(
                    page_group,
                    "slot",
                    q_tokens * n_seq,
                    n_conv,
                    lambda tiles: _conv_scatter_rows(
                        kv_pool, slot_pages_rows, n_conv, null_page=null_page, persistent=tiles
                    ),
                )
            elif batched_slot_conv_stores:
                slot_scatter = _conv_scatter_rows(kv_pool, slot_pages_rows, n_conv, null_page=null_page, **kept)

            def slot_conv_stores(
                after: tuple = (),
                copy_wait: list = prev_conv_dmas,
                slot_all=slot_all,
                kept=kept,
                slot_scatter=slot_scatter,
            ) -> None:
                slot_conv_dmas = _store_conv_states(
                    kv_pool,
                    slot_pages_rows,
                    slot_all,
                    f"{tag}_sc_slot",
                    copy_wait=copy_wait,
                    keep=scatter_keep,
                    scatter_rows=slot_scatter,
                    zero_rows=conv_zero_rows,
                    transpose_after=after,
                    **({} if slot_conv_rows is None else {"sources": slot_conv_rows, "conv_wait": kept["wait"]}),
                    **({"window_seq": conv_seq} if slot_all is None else {}),
                )
                stores.extend(slot_conv_dmas)
                if slot_conv_waits is not None:
                    slot_conv_waits[:] = slot_conv_dmas

            if slot_conv_after_ar:
                late_slot_conv.append(slot_conv_stores)
            else:
                slot_conv_stores()
    elif ring_tokens:
        # shared_state_ring: token 0 reads the committed state tile and token j writes ring[j % 2], which token j + 1
        # reads (sc_states[j + 1] = ring[j % 2]), so no token writes the tile that the previous token's stores read.
        sc_states = [state_sb] + [ring[t % len(ring)] for t in range(q_tokens)]
    else:
        # The shared-tile path keeps one state tile and updates it in place each token. With a persistent ring tile
        # (states mode) it takes ring[0], so its bytes outlive the last token's store, and its first write waits for
        # the previous call's stores of ring[0] (slot_state_waits); None keeps a per-call tile.
        sc_state = (
            slot_state_ring[0]
            if shared_ring
            else nl.ndarray((P_MAX, n_seq, heads, HEAD_DIM), dtype=state_dt, buffer=nl.sbuf)
        )
        first_copy = nisa.tensor_copy(dst=sc_state, src=state_sb, engine=nisa.engine.gpsimd)
        if shared_ring and hasattr(first_copy, "depends_on"):
            prev = list((slot_state_waits or {}).get(0, ()) or ())
            if prev:
                first_copy.depends_on(*prev)
    if score_one_pass:
        conv_all, _ = _conv_tokens(conv_sb, cm[0:P_MAX, 0:n_rows, 0:n_conv], conv_w, q_tokens, conv_dt)
        k_all, q_all, mv_all, qk_all = _readout_vectors(conv_all, decay[0:P_MAX, 0:n_rows, 0:heads], ones_sq, readout)
    elif not per_token:
        sc_hist = nl.ndarray((P_MAX, n_seq, n_conv, taps), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sc_hist, src=conv_sb, engine=nisa.engine.gpsimd)
    if g_last:
        # The g blocks and their sigmoid after the conv and readout pass: the VectorE runs them behind that work.
        nisa.tensor_copy(
            dst=cm[0:P_MAX, 0:n_rows, 3 * heads : 4 * heads],
            src=g_psum[0:P_MAX, 0:heads, 0:n_rows].permute((0, 2, 1)),
            engine=nisa.engine.vector,
        )
        gate_sigmoids(0, heads)
    out_bf = nl.ndarray((P_MAX, n_rows, heads), dtype=bf16, buffer=nl.sbuf)
    if batched:
        # The batched chain writes the Q outputs of every sequence and head, and the committed state (records) or
        # the Q states (states mode).
        _batched_score(
            state_sb,
            conv_all[0:P_MAX, 0:n_rows, 2 * heads : 3 * heads],
            k_all,
            q_all,
            decay[0:P_MAX, 0:n_rows, 0:heads],
            beta,
            ident32,
            batched_masks,
            out_bf,
            None if records_mode else sc_states[1:],
            batched_records if records_mode else None,
            state_waits=ring_last if batched_ring else None,
        )
        if records_mode:
            # The committed state of a pair with a non-finite S0 is non-finite (its own terms): store zeros instead.
            _store_finite(state_sb.flatten_dims(1, 2), nisa.engine.vector)
            store_committed_states()
    if chunked:
        # The chunked chain writes the Q outputs and the Q states of each sequence at once.
        for seq in range(n_seq):
            _chunked_chain(
                state_sb,
                conv_all[0:P_MAX, 0:n_rows, 2 * heads],
                k_all[0:P_MAX, 0:n_rows, 0],
                q_all[0:P_MAX, 0:n_rows, 0],
                decay[0:P_MAX, 0:n_rows, 0],
                beta[0:P_MAX, 0:n_rows, 0],
                ident32,
                seq,
                out_bf[0:P_MAX, 0:n_rows, 0],
                sc_states[1:],
            )
    # EARLY_OUT_TOKENS: the gated RMSNorm and the output projection of the first tokens' rows run in the token chain.
    out_tokens = EARLY_OUT_TOKENS if early_out_tokens is None else early_out_tokens
    early_out = out_tokens if 0 < out_tokens < q_tokens and not (chunked or batched) else 0
    if early_out:
        part_rows = (
            ar_sbuf if ar_sbuf is not None else nl.ndarray((P_MAX, n_rows, hidden_tiles), dtype=bf16, buffer=nl.sbuf)
        )

        def gated_out(r0: int, r1: int, name: str, warm=None):
            """The Step 12 tail of rows ``r0 .. r1 - 1``: their gated RMSNorm and output projection into
            ``part_rows``. ``warm`` (``late_keep_warm``): the keep-warm matmuls of the projection."""
            rows = slice(r0, r1)
            out_sq = nl.ndarray((P_MAX, r1 - r0, heads), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=out_sq, data1=out_bf[0:P_MAX, rows, 0:heads], data2=out_bf[0:P_MAX, rows, 0:heads], op=nl.multiply
            )
            out_ss = nl.ndarray((P_MAX, r1 - r0, heads), dtype=fp32, buffer=nl.psum)
            nisa.nc_matmul(dst=out_ss, stationary=ones_sq, moving=out_sq, is_stationary_onezero=True)
            out_log = nl.ndarray((P_MAX, r1 - r0, heads), dtype=fp32, buffer=nl.sbuf)
            log_act = nisa.activation(dst=out_log, op=nl.log, data=out_ss, scale=1.0 / HEAD_DIM, bias=NORM_EPS)
            out_gated = nl.ndarray((P_MAX, r1 - r0, heads), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=out_gated,
                data1=out_bf[0:P_MAX, rows, 0:heads],
                data2=gb_sig[0:P_MAX, rows, 0:heads],
                op=nl.multiply,
            )
            out_rstd = nl.ndarray((P_MAX, r1 - r0, heads), dtype=fp32, buffer=nl.sbuf)
            nisa.activation(dst=out_rstd, op=nl.exp, data=out_log, scale=-0.5)
            o_rows = nl.ndarray((P_MAX, r1 - r0, heads), dtype=bf16, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=o_rows,
                data=out_rstd,
                op0=nl.multiply,
                operand0=params_sb[0:P_MAX, small["o_norm"] : small["o_norm"] + 1],
                op1=nl.multiply,
                operand1=out_gated,
            )
            if warm is not None:
                warm.start(log_act)
            _, last_matmul, last_copy = _output_projection(
                o_rows,
                w_out_sb,
                hidden_tiles,
                name,
                out=part_rows[0:P_MAX, rows, 0:hidden_tiles],
                **({"filler": warm.hold, "filler_every": LATE_KEEP_WARM_EVERY} if warm is not None else {}),
            )
            return {"tensor": last_matmul, "vector": last_copy}

    heads_g_done = False

    def g_ready():
        """heads: copy the g blocks of pass C into cm and take their sigmoid (the output gate)."""
        nonlocal heads_g_done
        nisa.tensor_copy(
            dst=cm[0:P_MAX, 0:n_rows, 3 * heads : 4 * heads],
            src=head_gemv.psum_bc[0:P_MAX, N_CONV_PROJ : N_CONV_PROJ + heads, 0:n_rows].permute((0, 2, 1)),
            engine=nisa.engine.vector,
        )
        gate_sigmoids(0, heads)
        heads_g_done = True

    ar_rows = bool(out_ar_rows)
    kernel_assert(
        not ar_rows or (early_out and ar_sbuf is None and not out_rs_tp),
        "out_ar_rows needs the early output tail (early_out_tokens) and the HBM all-reduce",
    )
    ar_chunks = []
    if ar_rows:
        # out_ar_rows: the HBM pair seen row-major ([rows, 128, H/128], the same bytes), so a row chunk is one
        # contiguous region of both tensors.
        if ar_hbm is not None:
            rows_src, rows_dst = ar_hbm
        else:
            rows_src = nl.ndarray(
                (P_MAX, n_rows, hidden_tiles), dtype=bf16, buffer=nl.private_hbm, name=f"{tag}_kda_ar_src"
            )
            rows_dst = nl.ndarray(
                (P_MAX, n_rows, hidden_tiles), dtype=bf16, buffer=nl.private_hbm, name=f"{tag}_kda_ar_dst"
            )
        check_pair_group(rows_dst, tp_group, f"{tag}_kda_all_reduce")
        rows_src = rows_src.reshape((n_rows, P_MAX, hidden_tiles))
        rows_dst = rows_dst.reshape((n_rows, P_MAX, hidden_tiles))

        def ar_chunk(r0: int, r1: int, src_tile, engine):
            """Store rows r0 .. r1 - 1 of ``src_tile`` to the HBM source and all-reduce that chunk."""
            nisa.dma_copy(
                dst=rows_src[r0:r1].permute((1, 0, 2)),
                src=src_tile[0:P_MAX, r0:r1, 0:hidden_tiles],
                dge_mode=nisa.dge_mode.hwdge,
                engine=engine,
                name=f"{tag}_kda_ar_src_store_{r0}",
            )
            return release_weight_gate(
                ncc.all_reduce(
                    srcs=[rows_src[r0:r1]],
                    dsts=[rows_dst[r0:r1]],
                    replica_group=tp_group,
                    op=nl.add,
                    name=f"{tag}_kda_all_reduce_{r0}",
                )
            )

    for j in range(q_tokens):
        token = slice(j * n_seq, (j + 1) * n_seq)
        state_out = (sc_states[j + 1] if len(sc_states) > j + 1 else None) if per_token or ring_tokens else sc_state
        if not (chunked or batched):
            if per_token or score_one_pass:
                conv_out = conv_all[0:P_MAX, token, 0:n_conv]
                k_n, q_n = k_all[0:P_MAX, token, 0:heads], q_all[0:P_MAX, token, 0:heads]
                mv_in = mv_all[0:P_MAX, token, 0:heads, 0 : mv_all.shape[3]]
                qk_psum = None if qk_all is None else qk_all[0:P_MAX, token, 0:heads]
            else:
                conv_out, window = _conv_token(
                    sc_hist, cm[0:P_MAX, token, 0:n_conv], conv_w, f"{tag}_sc{j}", conv_dt, conv_engine
                )
                k_n, q_n, mv_in, qk_psum = _readout_vectors(conv_out, decay[0:P_MAX, token, 0:heads], ones_sq, readout)
            state_in = sc_states[j] if per_token or ring_tokens else sc_state
            # With the ring, the write of ring[j % depth] waits for the stores that read it last: token j - depth of
            # this call, or the previous call's stores (ring_last, from slot_state_waits). Token j + 1 reads this
            # token's tile and does not wait. Without the ring the shared path keeps the token-to-token edge.
            token_wait = ring_last[j % len(ring)] if use_ring else prev_state_dmas
            # The last op of the readout writes the token's rows of out_bf (no bf16 copy).
            _recurrence(
                state_in,
                conv_out[0:P_MAX, 0:n_seq, 2 * heads : 3 * heads],
                k_n,
                q_n,
                decay[0:P_MAX, token, 0:heads],
                beta[0:P_MAX, token, 0:heads],
                mv_in,
                qk_psum,
                ident,
                ident32,
                readout,
                state_write_wait=token_wait,
                out_state=state_out,
                state_bufs=fp32_sets[j % 2] if per_token else None,
                ones_sq=ones_sq if per_token else None,
                bf16_update_rows=bf16_update_rows,
                stationary_parts=n_parts,
                row_selector=row_selector,
                bcast_depth=update_bcast_depth,
                group=recurrence_group,
                out=out_bf[0:P_MAX, token, 0:heads],
                **({"bf16_cols": cols_sets[j % 2]} if cols_sets is not None else {}),
            )
        if not records_mode:
            # states mode: write the decode state after scored token j (recurrent + conv) to slot j, so the
            # next call loads it directly instead of replaying. The slot is the decode page of that step.
            if batched:
                _store_finite(state_out.flatten_dims(1, 2), nisa.engine.vector)
            state_dmas = []
            page_regs = slot_regs.pop(j, None) if slot_store_regs else None
            for seq in range(n_seq):
                second = slot_store_queues == 2 and (j * n_seq + seq) % 2 == 1
                pool = slot_store_engine == "gpsimd" or (slot_store_engine == "split" and (j * n_seq + seq) % 2 == 1)
                if pool:
                    # The software DGE issues from the GpSimd queue; it takes no engine argument.
                    queue = {"dge_mode": nisa.dge_mode.swdge}
                else:
                    queue = {
                        "dge_mode": nisa.dge_mode.hwdge,
                        "engine": nisa.engine.scalar if second else nisa.engine.sync,
                    }
                state_dmas.append(
                    nisa.dma_copy(
                        dst=_recurrent_state_view(
                            kv_pool,
                            slot_write_pages[0:1, j, seq : seq + 1] if page_regs is None else page_regs[seq],
                            heads,
                            page["recurrent"],
                            state_dtype,
                        ),
                        src=state_out[0:P_MAX, seq, 0:heads, 0:HEAD_DIM],
                        **queue,
                        name=f"{tag}_sc{j}_slot_state_{seq}",
                    )
                )
                if j == 0 and before_slot_stores:
                    # The in_proj_hook's DMAs (the MoE expert ring) go out before the store chain on the in-order
                    # queue, not between its stores, so their bytes do not share the DMA rings with the stores.
                    _schedule_after(state_dmas[-1], before_slot_stores)
                if slot_store_regs and j + 1 < q_tokens:
                    # The next token's page id of this sequence, after this store: a load right before its own store
                    # would fold back into that store's address (its load after the wait for the store before).
                    _load_slot_reg(slot_regs, slot_write_pages, j + 1, seq, n_seq)
            stores.extend(state_dmas)
            if use_ring or batched_ring:
                # These stores now read ring[j % depth]; token j + depth (or the next call) waits for them.
                ring_last[j % len(ring)] = state_dmas
        if not records_mode and not per_token:
            slot_window = nl.ndarray((P_MAX, n_seq, n_conv, CONV_TAPS), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=slot_window[0:P_MAX, 0:n_seq, 0:n_conv, 1:CONV_TAPS],
                src=window[0:P_MAX, 0:n_seq, 0:n_conv, 1:CONV_TAPS],
                engine=nisa.engine.gpsimd,
            )
            conv_dmas = _store_conv_states(
                kv_pool,
                slot_write_pages[0:P_MAX, j, 0:n_seq],
                slot_window,
                f"{tag}_sc{j}_slot",
                copy_wait=prev_conv_dmas,
                keep=scatter_keep,
                scatter_rows=(
                    _group_index_sets(
                        page_group,
                        f"slot_{j}",
                        n_seq,
                        n_conv,
                        lambda kept, token=j: _conv_scatter_rows(
                            kv_pool,
                            slot_write_pages[0:P_MAX, token, 0:n_seq],
                            n_conv,
                            null_page=null_page,
                            **({} if kept is None else {"persistent": kept}),
                        ),
                    )
                    if batched_slot_conv_stores
                    else None
                ),
                zero_rows=conv_zero_rows,
            )
            stores.extend(conv_dmas)
            # The next token overwrites sc_state (in _recurrence) and the conv rows tile (in
            # _store_conv_states); make those writes wait for this token's slot DMAs.
            prev_state_dmas = state_dmas
            prev_conv_dmas = conv_dmas
        if j + 1 < q_tokens and not per_token and not score_one_pass:
            nisa.tensor_copy(
                dst=sc_hist, src=window[0:P_MAX, 0:n_seq, 0:n_conv, 1:CONV_TAPS], engine=nisa.engine.gpsimd
            )
        if token_hook is not None:
            token_hook(j, q_tokens)
        if heads_mode and j < 3:
            # heads: quarters 2-4 of pass C (the g blocks) between the first chain tokens.
            head_gemv.run("C", _quarter(head_gemv.n_groups, j + 1))
            if j == 2:
                g_ready()
        if early_out and j == early_out - 1:
            if heads_mode and not heads_g_done:
                g_ready()
            gated_out(0, early_out * n_seq, f"{tag}_early")
            if ar_rows:
                ar_chunks.append(ar_chunk(0, early_out * n_seq, part_rows, nisa.engine.sync))

    if heads_mode:
        # heads with fewer than 3 tokens: the quarters of pass C that the chain did not take, then the gate.
        for k in range(min(q_tokens, 3) + 1, 4):
            head_gemv.run("C", _quarter(head_gemv.n_groups, k))
        if not heads_g_done:
            g_ready()

    # The group's page set now holds this call's page metadata; the later calls of the group read it.
    if page_group is not None:
        page_group["filled"] = True

    # Record this call's last stores of each ring tile, so the next KDA call's first write into it waits for them.
    if slot_state_waits is not None:
        if use_ring or batched_ring:
            for r in range(len(ring)):
                slot_state_waits[r] = ring_last[r]
        elif shared_ring:
            slot_state_waits[0] = prev_state_dmas

    # The records are only written in records mode; states mode wrote the full state to the slots.
    if records_mode:
        # Step 11: write the records of the Q new tokens to the record page, after the record read above.
        # The copies clamp to the finite range of each dtype, so a NaN or an Inf is stored finite.
        rec_store = nl.ndarray((P_MAX, n_seq, rec["width"]), dtype=bf16, buffer=nl.sbuf)
        cm_src = cm[0:P_MAX, 0:n_rows, 0:n_conv].reshape_dim(1, (q_tokens, n_seq)).permute((0, 2, 1, 3))
        _finite_copy(
            rec_store[0:P_MAX, 0:n_seq, rec["cm"] : rec["cm"] + q_tokens * n_conv].reshape_dim(2, (q_tokens, n_conv)),
            cm_src,
            _BF16_FINITE_MAX,
        )
        decay_src = decay[0:P_MAX, 0:n_rows, 0:heads].reshape_dim(1, (q_tokens, n_seq)).permute((0, 2, 1, 3))
        _finite_copy(
            rec_store[0:P_MAX, 0:n_seq, rec["decay"] : rec["decay"] + q_tokens * rec["gate_words"]]
            .view(fp32)
            .reshape_dim(2, (q_tokens, heads)),
            decay_src,
            _FP32_FINITE_MAX,
        )
        beta_src = beta[0:P_MAX, 0:n_rows, 0:heads].reshape_dim(1, (q_tokens, n_seq)).permute((0, 2, 1, 3))
        _finite_copy(
            rec_store[0:P_MAX, 0:n_seq, rec["beta"] : rec["beta"] + q_tokens * rec["gate_words"]]
            .view(fp32)
            .reshape_dim(2, (q_tokens, heads)),
            beta_src,
            _FP32_FINITE_MAX,
        )
        if batched:
            # Row isolation (batched): a non-finite record element stores the identity-step value (projection 0, decay 1,
            # beta 0), so the record page of a failed row cannot poison a later replay. Each block is guarded in its own
            # dtype (the decay and beta words are fp32 pairs in the bf16 row).
            _store_finite(rec_store[0:P_MAX, 0:n_seq, rec["cm"] : rec["cm"] + q_tokens * n_conv], nisa.engine.vector)
            for name, safe in (("decay", 1.0), ("beta", 0.0)):
                _store_finite(
                    rec_store[0:P_MAX, 0:n_seq, rec[name] : rec[name] + q_tokens * rec["gate_words"]].view(fp32),
                    nisa.engine.vector,
                    safe,
                )
        for seq in range(n_seq):
            # The store overwrites the page that the record load reads, so its source waits for that load:
            # x * (1 + (bits(load) & 0)) keeps every bit of x. A schedule order alone lets the store land first.
            load_bits = rec_load[0:P_MAX, seq, 0:2].view(nl.int32)
            zero_i = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=zero_i, data=load_bits, op0=nl.bitwise_and, operand0=0, engine=nisa.engine.vector)
            one = nl.ndarray((P_MAX, 1), dtype=fp32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=one, data=zero_i, op0=nl.add, operand0=1.0, engine=nisa.engine.vector)
            first_col = rec_store[0:P_MAX, seq, 0:1]
            nisa.tensor_scalar(dst=first_col, data=first_col, op0=nl.multiply, operand0=one, engine=nisa.engine.vector)
            stores.append(
                nisa.dma_copy(
                    dst=_record_view(kv_pool, record_dma[0:1, seq : seq + 1], rec["width"], span),
                    src=rec_store[0:P_MAX, seq, 0 : rec["width"]],
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"{tag}_kda_record_store_{seq}",
                )
            )

    # Step 12: gated RMSNorm over v on all rows, then the output projection and the all-reduce.
    kernel_assert(ar_sbuf is None or not out_rs_tp, "ar_sbuf is the SBUF all-reduce: not with out_rs_tp")
    warm = _KeepWarm(kw_zero, late_keep_warm, f"{tag}_kw") if late_keep_warm else None
    if early_out:
        # The rows of the first tokens took their tail in the chain; the later rows take theirs here.
        tail_marks = gated_out(early_out * n_seq, n_rows, f"{tag}_late", warm=warm)
        part_bf = part_rows
    else:
        out_sq = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=out_sq, data1=out_bf, data2=out_bf, op=nl.multiply)
        out_ss = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.psum)
        nisa.nc_matmul(dst=out_ss, stationary=ones_sq, moving=out_sq, is_stationary_onezero=True)
        out_log = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.sbuf)
        log_act = nisa.activation(dst=out_log, op=nl.log, data=out_ss, scale=1.0 / HEAD_DIM, bias=NORM_EPS)
        out_gated = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=out_gated, data1=out_bf, data2=out_gate, op=nl.multiply)
        out_rstd = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.sbuf)
        nisa.activation(dst=out_rstd, op=nl.exp, data=out_log, scale=-0.5)
        o_bf = nl.ndarray((P_MAX, n_rows, heads), dtype=bf16, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=o_bf,
            data=out_rstd,
            op0=nl.multiply,
            operand0=params_sb[0:P_MAX, small["o_norm"] : small["o_norm"] + 1],
            op1=nl.multiply,
            operand1=out_gated,
        )

        if warm is not None:
            warm.start(log_act)
        part_bf, last_matmul, last_copy = _output_projection(
            o_bf,
            w_out_sb,
            hidden_tiles,
            tag,
            out=ar_sbuf,
            **({"filler": warm.hold, "filler_every": LATE_KEEP_WARM_EVERY} if warm is not None else {}),
        )
        tail_marks = {"tensor": last_matmul, "vector": last_copy}
    if tail_hook is not None:
        tail_hook(tail_marks)
    kernel_assert(
        not (slot_conv_after_ar and out_rs_tp), "slot_conv_after_ar takes an output all-reduce: not out_rs_tp"
    )
    if out_rs_tp:
        # attn_rs_moe_input: reduce-scatter the tokens over the TP group, this rank keeps its one row (bit-equal to the
        # all-reduce's row; a kernel test checks this). The residual, the mix and the norm then run on that one row.
        kept = token_reduce_scatter(part_bf, tp_group, tag, out_rs_tp)
        object.__setattr__(kept, "side_effects", tuple(stores))
        return kept
    reduced = nl.ndarray((P_MAX, n_rows, hidden_tiles), dtype=bf16, buffer=nl.sbuf)
    if ar_sbuf is not None:
        # SBUF -> SBUF all-reduce of the partial, in the (128, rows, H/128) tile shape of the HBM form (the runtime's
        # bitwise case; an SBUF collective takes the [128, rows * H/128] view of the same bytes). The collective and
        # the TP peers read the source until the all-reduce completes, but the SBUF allocator ends a tile's liveness
        # at its last reader's issue (the trigger here), so the source is the caller's persistent tile, held to the
        # kernel end: no other tile takes its bytes. Its next writer is the next KDA layer's output projection, which
        # follows this all-reduce's completion through the stream (the residual add reads the destination). The
        # destination is an ordinary tile: the runtime writes it only with the reduced blocks, which need this rank's
        # source, and its readers wait for the completion.
        flat = (P_MAX, n_rows * hidden_tiles)
        sb_ar = release_weight_gate(
            ncc.all_reduce(
                srcs=[part_bf.reshape(flat)],
                dsts=[reduced.reshape(flat)],
                replica_group=tp_group,
                op=nl.add,
                name=f"{tag}_kda_all_reduce",
            )
        )
        if slot_ring_per_call and use_ring and slot_ring_keep_defer is not None:
            slot_ring_keep_defer.append((ring, ring_last, f"{tag}_ring_keep"))
        elif slot_ring_per_call and use_ring:
            _keep_ring_until_stored(ring, ring_last, f"{tag}_ring_keep", sb_ar, slot_ring_keep_sink)
        for late in late_slot_conv:
            late((sb_ar,))
        object.__setattr__(reduced, "side_effects", tuple(stores))
        return reduced
    if ar_rows:
        # out_ar_rows: chunk 1 (the early rows) is in flight; chunk 2 takes the later rows, then both load back.
        last_ar = ar_chunk(early_out * n_seq, n_rows, part_bf, nisa.engine.scalar)
        if slot_ring_per_call and use_ring:
            _keep_ring_until_stored(ring, ring_last, f"{tag}_ring_keep", last_ar, slot_ring_keep_sink)
        for late in late_slot_conv:
            late((last_ar,))
        for r0, r1 in ((0, early_out * n_seq), (early_out * n_seq, n_rows)):
            nisa.dma_copy(
                dst=reduced[0:P_MAX, r0:r1, 0:hidden_tiles],
                src=rows_dst[r0:r1].permute((1, 0, 2)),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.scalar,
                name=f"{tag}_kda_ar_dst_load_{r0}",
            )
        object.__setattr__(reduced, "side_effects", tuple(stores))
        return reduced
    # The all-reduce runs in persistent HBM tensors: a per-call HBM source lets the compiler reuse its bytes
    # for a spill before a TP peer has read them. ``ar_hbm`` passes the kernel-level pair (allocated once,
    # live to the kernel end); without it (standalone or loop path) the pair is allocated per call.
    if ar_hbm is not None:
        ar_src, ar_dst = ar_hbm
    else:
        ar_src = nl.ndarray((P_MAX, n_rows, hidden_tiles), dtype=bf16, buffer=nl.private_hbm, name=f"{tag}_kda_ar_src")
        ar_dst = nl.ndarray((P_MAX, n_rows, hidden_tiles), dtype=bf16, buffer=nl.private_hbm, name=f"{tag}_kda_ar_dst")
    nisa.dma_copy(
        dst=ar_src,
        src=part_bf,
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.scalar,
        name=f"{tag}_kda_ar_src_store",
    )
    check_pair_group(ar_dst, tp_group, f"{tag}_kda_all_reduce")
    hbm_ar = release_weight_gate(
        ncc.all_reduce(srcs=[ar_src], dsts=[ar_dst], replica_group=tp_group, op=nl.add, name=f"{tag}_kda_all_reduce")
    )
    if slot_ring_per_call and use_ring and slot_ring_keep_defer is not None:
        # slot_ring_keep_defer: a later call (or the kernel end) reads this ring.
        slot_ring_keep_defer.append((ring, ring_last, f"{tag}_ring_keep"))
    elif slot_ring_per_call and use_ring:
        # slot_ring_per_call: the ring tiles stay live until their stores have read them (GpSimd, after the trigger).
        _keep_ring_until_stored(ring, ring_last, f"{tag}_ring_keep", hbm_ar, slot_ring_keep_sink)
    for late in late_slot_conv:
        late((hbm_ar,))
    nisa.dma_copy(
        dst=reduced,
        src=ar_dst,
        dge_mode=nisa.dge_mode.hwdge,
        engine=nisa.engine.scalar,
        name=f"{tag}_kda_ar_dst_load",
    )
    object.__setattr__(reduced, "side_effects", tuple(stores))
    return reduced


kda_verify_attention.token_major = True


def kda_slot_conv_rows_tiles(slot_rows: int, plan) -> dict:
    """Return the persistent SBUF tiles of the slot conv-state stores of the per-token states path (``slot_conv_rows``).

    For each group of the ``slot_rows`` (Q * B) slot rows: the bf16 rows tile ``[128, 128, CONV_TAPS-1]``
    (``kda_conv_rows_<g>``) and the int32 pool-row and uint16 off-write tiles ``[128, 1]`` of its scatter
    (``kda_conv_index_<g>`` / ``kda_conv_off_<g>``), as ``tkg_kda.kda_conv_rows_tiles`` / ``kda_conv_index_tiles`` give
    the decode conv-state stores. The groups are sized for MAX_LOCAL_HEADS, so they cover any head count. ``plan``
    allocates the tiles; the tile names carry a ``kda_slot_conv`` prefix.
    """
    group = P_MAX // (N_CONV_PROJ * MAX_LOCAL_HEADS)
    tiles = {}
    for g in range(-(-slot_rows // group)):
        tiles[f"kda_conv_rows_{g}"] = plan.tile(
            (P_MAX, HEAD_DIM, CONV_TAPS - 1), nl.bfloat16, f"kda_slot_conv_rows_{g}"
        )
        tiles[f"kda_conv_index_{g}"] = plan.tile((P_MAX, 1), nl.int32, f"kda_slot_conv_index_{g}")
        tiles[f"kda_conv_off_{g}"] = plan.tile((P_MAX, 1), nl.uint16, f"kda_slot_conv_off_{g}")
    return tiles


KDA_PAGE_GROUPS_MAX = 8
"""The most KV cache groups that ``kda_page_group_tiles`` sets can serve in one call (each group keeps its own tiles)."""


_PAGE_META_KEYS = ("state_read", "state_write", "read_mask")
"""The ``_verify_page_meta`` entries of ``states`` mode that a page set keeps."""


def _index_sets(n_seq: int, n_conv: int, plan, name: str) -> dict:
    """Return the int32 pool-row and uint16 off-write tiles ``[128, 1]`` of each group of ``128 // n_conv`` of the
    ``n_seq`` rows, in the ``_conv_scatter_rows`` ``persistent`` form."""
    group = P_MAX // n_conv
    tiles = {}
    for g in range(-(-n_seq // group)):
        tiles[f"kda_conv_index_{g}"] = plan.tile((P_MAX, 1), nl.int32, f"{name}_index_{g}")
        tiles[f"kda_conv_off_{g}"] = plan.tile((P_MAX, 1), nl.uint16, f"{name}_off_{g}")
    return tiles


def _kept_index_sets(tiles: dict, n_seq: int, n_conv: int) -> list:
    """Return the ``_conv_scatter_rows`` result that a call wrote into the ``_index_sets`` tiles ``tiles``."""
    group = P_MAX // n_conv
    sets = []
    for first in range(0, n_seq, group):
        n_rows = (min(n_seq, first + group) - first) * n_conv
        g = first // group
        sets.append((tiles[f"kda_conv_index_{g}"][0:n_rows, 0:1], tiles[f"kda_conv_off_{g}"][0:n_rows, 0:1]))
    return sets


def kda_page_group_tiles(
    n_seq: int,
    q_tokens: int,
    heads: int,
    plan,
    name: str,
    conv_load_gather: bool = False,
    batched_conv_stores: bool = False,
    batched_slot_conv_stores: bool = False,
    per_token: bool = False,
) -> dict:
    """Return the page set of one KV cache group: the persistent SBUF tiles of the page metadata of its KDA layers.

    In ``states`` mode each KDA verify call computes, from its pages and the positions and replay lengths of the step,
    the DMA pages and 0/1 masks of the state and the slot reads and writes (``_page_counts``), the slot write pages
    (``_write_slot_pages``), and the pool rows of its conv gather and scatters (``_conv_scatter_rows``). The layers of
    one KV cache group have one block table, so these values are the same in each of them. The first call of the group
    (``kda_verify_attention`` ``page_group``) writes them into these tiles, and the later calls read them in place of
    computing them again. ``plan`` allocates the tiles (a heap that lives through the kernel); ``name`` prefixes their
    names. The flags are those of the calls: they choose the index sets to keep (the conv gather, the committed
    scatter, and the slot scatter of all slots with ``per_token``, else one per token).

    Returns:
        dict: ``"filled"`` (False until the first call of the group) and ``"tiles"``.
    """
    n_conv = N_CONV_PROJ * heads
    tiles = {
        "state_read": plan.tile((P_MAX, n_seq), nl.int32, f"{name}_state_read"),
        "state_write": plan.tile((P_MAX, n_seq), nl.int32, f"{name}_state_write"),
        "read_mask": plan.tile((P_MAX, n_seq), nl.uint16, f"{name}_read_mask"),
        "slot_read": plan.tile((P_MAX, n_seq), nl.int32, f"{name}_slot_read"),
        "slot_mask": plan.tile((P_MAX, n_seq), nl.uint16, f"{name}_slot_mask"),
        "slot_write_pages": plan.tile((P_MAX, q_tokens, n_seq), nl.int32, f"{name}_slot_write_pages"),
    }
    if conv_load_gather:
        tiles["gather"] = _index_sets(n_seq, n_conv, plan, f"{name}_gather")
    if batched_conv_stores:
        tiles["committed"] = _index_sets(n_seq, n_conv, plan, f"{name}_committed")
    if batched_slot_conv_stores and per_token:
        tiles["slot"] = _index_sets(q_tokens * n_seq, n_conv, plan, f"{name}_slot")
    if batched_slot_conv_stores and not per_token:
        for j in range(q_tokens):
            tiles[f"slot_{j}"] = _index_sets(n_seq, n_conv, plan, f"{name}_slot{j}")
    return {"filled": False, "tiles": tiles}


def _group_tile(page_group, key: str, compute):
    """Return the page metadata ``key`` of the call: the tile of the group when an earlier call of the group wrote it,
    else ``compute()``, which this first call of the group then copies into the group's tile."""
    if page_group is None:
        return compute()
    kernel_assert(key in page_group["tiles"], f"the page set has no {key!r} tile (kda_page_group_tiles)")
    tile = page_group["tiles"][key]
    if page_group["filled"]:
        return tile
    value = compute()
    nisa.tensor_copy(dst=tile, src=value, engine=nisa.engine.vector)
    return value


def _group_page_meta(page_group, compute) -> dict:
    """Return the page metadata (``_PAGE_META_KEYS``) of a ``states`` prework: the tiles of the group when an earlier
    call of the group wrote them, else ``compute()``, which this first call of the group then copies into the group's
    tiles (``kda_verify_attention`` step 2)."""
    if page_group is not None and page_group["filled"]:
        return {key: page_group["tiles"][key] for key in _PAGE_META_KEYS}
    meta = compute()
    if page_group is not None:
        for key in _PAGE_META_KEYS:
            nisa.tensor_copy(dst=page_group["tiles"][key], src=meta[key], engine=nisa.engine.vector)
    return meta


def _group_index_sets(page_group, key: str, n_seq: int, n_conv: int, compute):
    """Return the conv gather or scatter rows ``key`` of the call: the rows of the group when an earlier call of the
    group wrote them, else ``compute(persistent)``, which writes them into the group's tiles (``persistent`` None
    without a group)."""
    if page_group is None:
        return compute(None)
    kernel_assert(key in page_group["tiles"], f"the page set has no {key!r} rows (kda_page_group_tiles flags)")
    tiles = page_group["tiles"][key]
    if page_group["filled"]:
        return _kept_index_sets(tiles, n_seq, n_conv)
    return compute(tiles)


def kda_verify_prework(
    kv_pool: nl.NkiTensor,
    state_page: nl.NkiTensor,
    record_page: nl.NkiTensor,
    replay_len: nl.NkiTensor,
    state_position: nl.NkiTensor,
    heads: int,
    q_tokens: int,
    tag: str,
    replay: str = "records",
    state_slot_page: nl.NkiTensor = None,
    write_state_slots: nl.NkiTensor = None,
    state_dtype: str = "bf16",
    conv_load_gather: bool = False,
    null_page: int = 0,
    slot_pages_contiguous: bool = False,
    paired_pages: bool = False,
    loads_after: tuple = (),
    replay_ahead: dict = None,
    merged_state_loads: bool = False,
    defer_selects: bool = False,
    page_group: dict = None,
) -> dict:
    """Return the tiles of the KDA verify steps that need only the page metadata and the cache pool.

    These are the page counts, the committed state and conv window (selected), and the replay records
    (``records``) or the slot state and the slot pages (``states``). They do not need the hidden or the
    weights of the layer, so a caller can trace them one phase ahead and pass the result as the
    ``prework`` of ``kda_verify_attention``. The arguments are those of ``kda_verify_attention``, and ``loads_after``:
    instructions that every cache-pool load follows in the schedule and waits for at runtime, for example the
    ``side_effects`` (page stores) of a call before it in the same kernel that wrote the pages it reads. Nothing else
    orders a load of a page after such a store. Empty (the default) keeps the program.

    ``replay_ahead`` (``records``, the ``sequential`` chain): None (the default) keeps the program. A dict also loads
    the layer's small weights (``kda_w_small``, ``kda_params``) and replays the accepted prefix here (step 8 of
    ``kda_verify_attention``, ``_replay_records``), so a caller that traces the prework one phase ahead runs the replay
    chain there, not in the body. The replay reads only the state and record pages and the small weights. Its other
    keys are the body options of the replay: ``conv_dtype``, ``readout``, ``recurrence_group``, ``replay_prework``,
    ``conv_products_engine`` and ``bf16_update_rows`` (the body's defaults when absent). The body then takes the
    replayed state and conv window and the small weights from the prework, skips step 8, and stops the trace when its
    options differ. The same instructions, so the bits do not change.

    ``merged_state_loads`` (``states``): one load of each sequence's
    selected page, as ``kda_verify_attention`` ``merged_state_loads``. ``defer_selects``: the VectorE selects of the
    loaded states and conv windows are not traced here; the result keeps them under ``deferred`` and the body runs
    them after its input projection's quantize (``kda_verify_attention``). The same loads and selects, so the bits do
    not change. The defaults keep the program. ``page_group`` (``states``): the page set of the layer's KV cache group
    (``kda_page_group_tiles``), as ``kda_verify_attention`` ``page_group``: the first prework of the group computes the
    page metadata, the slot pages and the conv gather rows and writes them into the set, and the later ones read them.
    The body of the layer then takes the same set; it marks the set filled.
    """
    kernel_assert(
        not (defer_selects or merged_state_loads or page_group is not None) or replay == "states",
        "defer_selects, merged_state_loads and page_group of the prework need replay='states'",
    )
    deferred = [] if defer_selects else None
    meta = _group_page_meta(
        page_group,
        lambda: _verify_page_meta(
            state_page,
            record_page,
            replay_len,
            state_position,
            kda_page_bound(kv_pool.shape[0], paired_pages),
            q_tokens,
            tag,
            replay,
            null_page=null_page,
            even_only=paired_pages,
        ),
    )
    _verify_state_loads(
        meta,
        kv_pool,
        heads,
        q_tokens,
        tag,
        replay,
        replay_len,
        state_slot_page,
        write_state_slots,
        state_dtype,
        conv_load_gather=conv_load_gather,
        null_page=null_page,
        slot_pages_contiguous=slot_pages_contiguous,
        paired_pages=paired_pages,
        loads_after=tuple(loads_after),
        **({"merged_state_loads": True} if merged_state_loads else {}),
        **({"deferred": deferred} if deferred is not None else {}),
        **({"page_group": page_group} if page_group is not None else {}),
    )
    meta["null_page"] = null_page
    meta["paired_pages"] = paired_pages
    if deferred is not None:
        meta["deferred"] = deferred
    if page_group is not None:
        meta["page_group"] = page_group
    if replay_ahead is not None:
        kernel_assert(replay == "records", f"replay_ahead replays the records: replay='records', got {replay!r}")
        _replay_ahead(meta, heads, tag, state_dtype, **replay_ahead)
    return meta


def kda_verify_prework_meta(
    kv_pool: nl.NkiTensor,
    state_page: nl.NkiTensor,
    record_page: nl.NkiTensor,
    replay_len: nl.NkiTensor,
    state_position: nl.NkiTensor,
    q_tokens: int,
    tag: str,
    state_slot_page: nl.NkiTensor,
    write_state_slots: nl.NkiTensor,
    null_page: int = 0,
    slot_pages_contiguous: bool = False,
    paired_pages: bool = False,
    merged_state_loads: bool = False,
    page_group: dict = None,
) -> dict:
    """Return the position-only part of the ``states`` prework of ``kda_verify_prework``: the page counts, the slot read
    and its mask, the slot write pages and, with ``merged_state_loads``, the selected page and read mask of each
    sequence. Only VectorE work and the load of the int32 slot page ids, so a caller can trace it in a window of its
    own. ``kda_verify_prework_loads`` then adds the loads. The arguments are those of ``kda_verify_prework``. With a
    ``page_group`` whose first layer filled it, the page counts, the slot read and the slot write pages are the tiles
    of the set: only the merged selects (``merged_state_loads``) trace here."""
    n_pages = kda_page_bound(kv_pool.shape[0], paired_pages)
    meta = _group_page_meta(
        page_group,
        lambda: _verify_page_meta(
            state_page,
            record_page,
            replay_len,
            state_position,
            n_pages,
            q_tokens,
            tag,
            "states",
            null_page=null_page,
            even_only=paired_pages,
        ),
    )
    slot_read, slot_mask = _slot_page_counts(
        page_group, state_slot_page, replay_len, n_pages, tag, null_page, paired_pages
    )
    meta["slot_counts"] = (slot_read, slot_mask)
    if merged_state_loads:
        meta["merged_pages"] = _merged_pages(meta["state_read"], meta["read_mask"], slot_read, slot_mask)
    meta["slot_write_pages"] = _group_tile(
        page_group,
        "slot_write_pages",
        lambda: _write_slot_pages(
            write_state_slots,
            n_pages,
            q_tokens,
            f"{tag}_slot",
            null_page=null_page,
            contiguous=slot_pages_contiguous,
            even_only=paired_pages,
        ),
    )
    meta.update(null_page=null_page, paired_pages=paired_pages, merged_state_loads=merged_state_loads)
    if page_group is not None:
        meta["page_group"] = page_group
    return meta


def kda_verify_prework_loads(
    meta: dict,
    kv_pool: nl.NkiTensor,
    heads: int,
    q_tokens: int,
    tag: str,
    replay_len: nl.NkiTensor,
    state_slot_page: nl.NkiTensor,
    write_state_slots: nl.NkiTensor,
    state_dtype: str = "bf16",
    conv_load_gather: bool = False,
    slot_pages_contiguous: bool = False,
    loads_after: tuple = (),
    defer_selects: bool = False,
) -> dict:
    """Add the loads of the ``states`` prework to ``meta`` (the ``kda_verify_prework_meta`` result) and return it: the
    committed state, the conv window (one gather per group with ``conv_load_gather``) and the slot state of each
    sequence, from the pages that ``meta`` gives. ``meta`` with these loads is a ``prework`` of
    ``kda_verify_attention``, as the result of ``kda_verify_prework``. The other arguments are those of
    ``kda_verify_prework``; the null page, the paired pages and ``merged_state_loads`` come from ``meta``. The same
    loads and selects as ``kda_verify_prework``, so the bits do not change."""
    deferred = [] if defer_selects else None
    _verify_state_loads(
        meta,
        kv_pool,
        heads,
        q_tokens,
        tag,
        "states",
        replay_len,
        state_slot_page,
        write_state_slots,
        state_dtype,
        slot_write_pages=meta["slot_write_pages"],
        conv_load_gather=conv_load_gather,
        null_page=meta["null_page"],
        slot_pages_contiguous=slot_pages_contiguous,
        paired_pages=meta["paired_pages"],
        loads_after=tuple(loads_after),
        merged_state_loads=meta["merged_state_loads"],
        deferred=deferred,
        slot_counts=meta["slot_counts"],
        merged_pages=meta.get("merged_pages"),
        **({"page_group": meta["page_group"]} if meta.get("page_group") is not None else {}),
    )
    if deferred is not None:
        meta["deferred"] = deferred
    return meta


def _replay_options(
    conv_dtype, readout, recurrence_group, replay_prework, conv_products_engine, bf16_update_rows, update_bcast_depth
):
    """Return the body options that the replay of the prework (``kda_verify_prework`` ``replay_ahead``) took."""
    return {
        "conv_dtype": conv_dtype,
        "readout": readout,
        "recurrence_group": recurrence_group,
        "replay_prework": replay_prework,
        "conv_products_engine": conv_products_engine,
        "bf16_update_rows": bf16_update_rows,
        "update_bcast_depth": update_bcast_depth,
    }


def _replay_ahead(
    meta,
    heads,
    tag,
    state_dtype,
    kda_w_small,
    kda_params,
    conv_dtype="bf16",
    readout=None,
    recurrence_group=None,
    replay_prework="one_pass",
    conv_products_engine="gpsimd",
    bf16_update_rows=False,
    update_bcast_depth=2,
    weight_dma="pool",
):
    """Load the small weights and replay the records of the prework ``meta`` in place (``kda_verify_prework``
    ``replay_ahead``): steps 1-2 (the replay's constants and small weights) and 8 of ``kda_verify_attention``."""
    bf16 = nl.bfloat16
    fp32 = nl.float32
    n_seq = meta["state_read"].shape[1]
    n_conv = N_CONV_PROJ * heads
    small = kda_small_columns(heads, conv_dtype)
    readout = kda_default_readout(n_seq) if readout is None else readout
    kernel_assert(readout in KDA_READOUTS, f"readout {readout!r} must be one of {KDA_READOUTS}")
    kernel_assert(replay_prework in REPLAY_PREWORK_FORMS, f"replay_prework must be one of {REPLAY_PREWORK_FORMS}")
    ones_sq = nl.ndarray((P_MAX, P_MAX), dtype=fp32, buffer=nl.sbuf)
    nisa.memset(dst=ones_sq, value=1.0)
    ident = nl.shared_identity_matrix(n=P_MAX, dtype=bf16)
    row_selector = None
    if ROW_SELECTOR_ONCE and bf16_update_rows and readout == "decoupled_bf16":
        row_selector = _bf16_row_selector(ident, heads)
    ident32 = nl.shared_identity_matrix(n=P_MAX, dtype=fp32)
    small_sb = nl.ndarray((P_MAX, small["small_width"]), dtype=bf16, buffer=nl.sbuf)
    # weight_dma: the queue of the small-weight and parameter loads (KDA_WEIGHT_DMAS).
    gate_weight_dma(
        nisa.dma_copy(
            dst=small_sb, src=kda_w_small, **kda_weight_dma_options(weight_dma), name=f"{tag}_kda_w_small_load"
        )
    )
    params_sb = nl.ndarray((P_MAX, kda_params.shape[1]), dtype=fp32, buffer=nl.sbuf)
    gate_weight_dma(
        nisa.dma_copy(
            dst=params_sb, src=kda_params, **kda_weight_dma_options(weight_dma), name=f"{tag}_kda_params_load"
        )
    )
    conv_w = _conv_weights(small_sb, params_sb, small, n_conv, conv_dtype)
    _replay_records(
        meta["state_sb"],
        meta["conv_sb"],
        meta["replay_pred"],
        meta["cm_rec"],
        meta["decay_rec"],
        meta["beta_rec"],
        conv_w,
        fp32 if conv_dtype == "fp32" else bf16,
        getattr(nisa.engine, conv_products_engine),
        ones_sq,
        ident,
        ident32,
        readout,
        tag,
        replay_prework,
        bf16_update_rows,
        _stationary_parts(state_dtype) if readout == "decoupled_bf16" else 0,
        row_selector,
        recurrence_group,
        update_bcast_depth,
    )
    meta.update(
        small_sb=small_sb,
        params_sb=params_sb,
        replayed=_replay_options(
            conv_dtype,
            readout,
            recurrence_group,
            replay_prework,
            conv_products_engine,
            bf16_update_rows,
            update_bcast_depth,
        ),
    )


def _verify_page_meta(
    state_page,
    record_page,
    replay_len,
    state_position,
    n_pages,
    q_tokens,
    tag,
    replay,
    null_page=0,
    even_only=False,
):
    """Return the page counts of the state and the records (``records``) of each sequence (step 2). ``null_page``: the
    KDA null page (``kda_verify_attention`` ``null_page``), or 0. ``even_only``: ``paired_pages`` (an odd id is off)."""
    state_read, state_write, read_mask, _ = _page_counts(
        state_page, state_position, n_pages, f"{tag}_state", null_page=null_page, even_only=even_only
    )
    meta = {"state_read": state_read, "state_write": state_write, "read_mask": read_mask}
    # states mode loads the per-token state directly, so it needs no replay records.
    if replay == "records":
        meta["record_dma"], meta["replay_pred"] = _record_meta(
            record_page, replay_len, n_pages, q_tokens, f"{tag}_record", null_page=null_page, even_only=even_only
        )
    return meta


def _slot_page_counts(page_group, state_slot_page, replay_len, n_pages, tag, null_page, paired_pages):
    """Return the slot read page and its 0/1 mask of each sequence (``_page_counts`` of ``state_slot_page`` at
    ``replay_len``), from the group's page set when the group's first layer filled it, else computed (and written into
    the set)."""
    if page_group is not None and page_group["filled"]:
        return page_group["tiles"]["slot_read"], page_group["tiles"]["slot_mask"]
    slot_read, _, slot_mask, _ = _page_counts(
        state_slot_page, replay_len, n_pages, f"{tag}_slot", null_page=null_page, even_only=paired_pages
    )
    if page_group is not None:
        nisa.tensor_copy(dst=page_group["tiles"]["slot_read"], src=slot_read, engine=nisa.engine.vector)
        nisa.tensor_copy(dst=page_group["tiles"]["slot_mask"], src=slot_mask, engine=nisa.engine.vector)
    return slot_read, slot_mask


def _merged_pages(state_read, read_mask, slot_read, slot_mask) -> tuple:
    """Return the page and the read mask of each sequence for ``merged_state_loads``: its slot page where the slot read
    is on, else its state page, and the read on where either is (VectorE)."""
    vector = nisa.engine.vector
    sel_page = nl.ndarray(state_read.shape, dtype=state_read.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sel_page, src=state_read, engine=vector)
    nisa.tensor_copy_predicated(dst=sel_page, src=slot_read, predicate=slot_mask)
    sel_mask = nl.ndarray(read_mask.shape, dtype=read_mask.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sel_mask, data1=read_mask, data2=slot_mask, op=nl.maximum, engine=vector)
    return sel_page, sel_mask


def _verify_state_loads(
    out,
    kv_pool,
    heads,
    q_tokens,
    tag,
    replay,
    replay_len,
    state_slot_page,
    write_state_slots,
    state_dtype="bf16",
    slot_write_pages=None,
    conv_load_gather=False,
    null_page=0,
    slot_pages_contiguous=False,
    paired_pages=False,
    page_group=None,
    states_after_gate_result=True,
    loads_after=(),
    sp_loads_after=(),
    merged_state_loads=False,
    loads_follow=(),
    deferred=None,
    slot_counts=None,
    merged_pages=None,
):
    """Add the selected state and conv window, and the replay records or the slot state, to ``out`` (steps 6-7).

    The state tiles take the page dtype of ``state_dtype`` (bf16, or fp32 for the fp32 forms).
    ``slot_write_pages``: the ``_write_slot_pages`` result when the caller loaded the slot pages before; None loads
    them here. ``page_group``: the page set of the call's KV cache group (``kda_verify_attention`` ``page_group``),
    or None. ``loads_after``: instructions that every cache-pool load follows (``kda_verify_prework``).
    ``sp_loads_after``: instructions that every Sync-queue cache-pool load follows (``kda_verify_attention``
    ``state_loads_after``); the gathered conv-state load on the Pool queue does not follow them.
    ``merged_state_loads`` (``states``): pick the page of each sequence first (its slot page when ``replay_len`` > 0, else
    its state page) and load the state and the conv window once from it, in place of loading both pages and selecting
    (``kda_verify_attention`` ``merged_state_loads``).
    ``loads_follow``: instructions that every per-sequence state and conv-state load follows in the schedule only
    (``kda_verify_attention`` ``state_loads_after_ring``); the gather DMAs of ``conv_load_gather`` run on the GpSimd
    queue and keep their order. ``deferred``: None traces the selects of the loaded states and conv windows here; a
    list gets them, in order, as functions that trace them and return their instructions, for the caller to run later
    (``state_loads_after_ring``: after the input projection's quantize). ``slot_counts``: None, or the
    ``_slot_page_counts`` result of the call, which a caller traced before (``kda_verify_prework_meta``);
    ``merged_pages``: None, or the ``_merged_pages`` result of ``merged_state_loads``, traced before in the same way."""

    sp_after = tuple(loads_after) + tuple(sp_loads_after)

    def pool_load(load):
        # Each pool load runs on the Sync hardware DGE queue.
        _schedule_after(load, list(sp_after) + list(loads_follow))
        _runtime_after(load, sp_after, engine=nisa.engine.sync)

    def vector_step(step):
        if deferred is None:
            step()
        else:
            deferred.append(step)

    n_seq = out["state_read"].shape[1]
    n_pages = kda_page_bound(kv_pool.shape[0], paired_pages)
    n_conv = N_CONV_PROJ * heads
    taps = CONV_TAPS - 1
    page = kda_page_offsets(heads, state_dtype)
    rec = kda_record_offsets(heads, q_tokens)
    records_mode = replay == "records"
    bf16 = nl.bfloat16
    fp32 = nl.float32
    state_read, read_mask = out["state_read"], out["read_mask"]
    state_dt = bf16 if state_dtype == "bf16" else fp32
    merged = merged_state_loads and not records_mode
    if merged and merged_pages is not None:
        state_read, read_mask = merged_pages
    elif merged:
        # merged_state_loads: the page of each sequence is its slot page where the slot read is on, else its state
        # page; the read is on where either is. One load of that page then gives the bytes that the two loads and the
        # selects below give: the slot state where the slot read is on, else the committed state, else zeros.
        slot_read, slot_mask = _slot_page_counts(
            page_group, state_slot_page, replay_len, n_pages, tag, null_page, paired_pages
        )
        state_read, read_mask = _merged_pages(state_read, read_mask, slot_read, slot_mask)

    # Step 6: load the committed state and conv window from the state page (zero where the read is off).
    state_sb = nl.ndarray((P_MAX, n_seq, heads, HEAD_DIM), dtype=state_dt, buffer=nl.sbuf)
    for seq in range(n_seq):
        pool_load(
            nisa.dma_copy(
                dst=state_sb[0:P_MAX, seq, 0:heads, 0:HEAD_DIM],
                src=_recurrent_state_view(
                    kv_pool, state_read[0:1, seq : seq + 1], heads, page["recurrent"], state_dtype
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=f"{tag}_kda_recurrent_state_load_{seq}",
            )
        )

        def select(loaded=state_sb[0:P_MAX, seq, 0:heads, 0:HEAD_DIM], seq=seq, read_mask=read_mask):
            return nisa.select_reduce(
                dst=loaded,
                predicate=_per_sequence(read_mask[0:P_MAX, seq : seq + 1], (heads, HEAD_DIM)).squeeze_dim(1),
                on_true=loaded,
                on_false=0.0,
            )

        vector_step(select)
    gathered_conv = None
    kernel_assert(not (conv_load_gather and loads_after), "loads_after does not order the gathered conv-state loads")
    if conv_load_gather:
        gather_rows = _group_index_sets(
            page_group,
            "gather",
            n_seq,
            n_conv,
            lambda kept: _conv_scatter_rows(
                kv_pool, state_read, n_conv, **({} if kept is None else {"persistent": kept})
            ),
        )
        gathered_conv = _gather_conv_rows(kv_pool, state_read, n_conv, f"{tag}_state", gather_rows=gather_rows)
    staged_conv = _load_conv_states(
        kv_pool,
        state_read,
        n_conv,
        f"{tag}_state",
        gathered=gathered_conv,
        load_after=sp_after,
        load_follow=loads_follow,
    )
    conv_sb = nl.ndarray((P_MAX, n_seq, n_conv, taps), dtype=bf16, buffer=nl.sbuf)

    def conv_select(conv_sb=conv_sb, staged_conv=staged_conv, read_mask=read_mask):
        return [
            nisa.memset(dst=conv_sb, value=0.0),
            nisa.tensor_copy_predicated(
                dst=conv_sb, src=staged_conv, predicate=_per_sequence(read_mask, (n_conv, taps))
            ),
        ]

    vector_step(conv_select)

    if records_mode:
        # Step 7: load the replay records of the previous call. Clean tiles hold the bf16 projection values
        # and the fp32 decay and beta (word pairs in the page).
        record_dma = out["record_dma"]
        rec_load = nl.ndarray((P_MAX, n_seq, rec["width"]), dtype=bf16, buffer=nl.sbuf)
        for seq in range(n_seq):
            pool_load(
                nisa.dma_copy(
                    dst=rec_load[0:P_MAX, seq, 0 : rec["width"]],
                    src=_record_view(
                        kv_pool, record_dma[0:1, seq : seq + 1], rec["width"], kda_page_span(paired_pages)
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=nisa.engine.sync,
                    name=f"{tag}_kda_record_load_{seq}",
                )
            )
        cm_rec = nl.ndarray((P_MAX, n_seq, q_tokens, n_conv), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=cm_rec,
            src=rec_load[0:P_MAX, 0:n_seq, rec["cm"] : rec["cm"] + q_tokens * n_conv].reshape_dim(
                2, (q_tokens, n_conv)
            ),
        )
        decay_rec = nl.ndarray((P_MAX, n_seq, q_tokens, heads), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=decay_rec,
            src=rec_load[0:P_MAX, 0:n_seq, rec["decay"] : rec["decay"] + q_tokens * rec["gate_words"]]
            .view(fp32)
            .reshape_dim(2, (q_tokens, heads)),
        )
        beta_rec = nl.ndarray((P_MAX, n_seq, q_tokens, heads), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=beta_rec,
            src=rec_load[0:P_MAX, 0:n_seq, rec["beta"] : rec["beta"] + q_tokens * rec["gate_words"]]
            .view(fp32)
            .reshape_dim(2, (q_tokens, heads)),
        )
        out.update(
            rec_load=rec_load,
            cm_rec=cm_rec,
            decay_rec=decay_rec,
            beta_rec=beta_rec,
        )
    else:
        # Step 7': load S_m from its per-token state slot and select it where replay_len > 0 (else keep
        # the committed state). The slot holds the full decode state (conv + recurrent) after m scored
        # tokens, so no replay recurrence runs. A select, not a multiply, keeps a null slot NaN-safe.
        if not merged:
            slot_read, slot_mask = (
                slot_counts
                if slot_counts is not None
                else _slot_page_counts(page_group, state_slot_page, replay_len, n_pages, tag, null_page, paired_pages)
            )
            slot_state = nl.ndarray((P_MAX, n_seq, heads, HEAD_DIM), dtype=state_dt, buffer=nl.sbuf)
            for seq in range(n_seq):
                # The slot state is read only by the select below and the recurrence, so these loads follow the load of
                # the previous body's output on the SP queue (each waits for its VectorE page index first).
                pool_load(
                    (order_after_gate_result if states_after_gate_result else (lambda dma: dma))(
                        nisa.dma_copy(
                            dst=slot_state[0:P_MAX, seq, 0:heads, 0:HEAD_DIM],
                            src=_recurrent_state_view(
                                kv_pool, slot_read[0:1, seq : seq + 1], heads, page["recurrent"], state_dtype
                            ),
                            dge_mode=nisa.dge_mode.hwdge,
                            engine=nisa.engine.sync,
                            name=f"{tag}_kda_slot_state_load_{seq}",
                        )
                    )
                )
            vector_step(
                lambda state_sb=state_sb, slot_state=slot_state, slot_mask=slot_mask: nisa.tensor_copy_predicated(
                    dst=state_sb, src=slot_state, predicate=_per_sequence(slot_mask, (heads, HEAD_DIM))
                )
            )
            slot_conv = _load_conv_states(
                kv_pool, slot_read, n_conv, f"{tag}_slot", load_after=sp_after, load_follow=loads_follow
            )
            vector_step(
                lambda conv_sb=conv_sb, slot_conv=slot_conv, slot_mask=slot_mask: nisa.tensor_copy_predicated(
                    dst=conv_sb, src=slot_conv, predicate=_per_sequence(slot_mask, (n_conv, taps))
                )
            )
        if slot_write_pages is None:
            slot_write_pages = _group_tile(
                page_group,
                "slot_write_pages",
                lambda: _write_slot_pages(
                    write_state_slots,
                    n_pages,
                    q_tokens,
                    f"{tag}_slot",
                    null_page=null_page,
                    contiguous=slot_pages_contiguous,
                    even_only=paired_pages,
                ),
            )
        out["slot_write_pages"] = slot_write_pages
    out.update(state_sb=state_sb, conv_sb=conv_sb)


def _replay_records(
    state_sb,
    conv_sb,
    replay_pred,
    cm_rec,
    decay_rec,
    beta_rec,
    conv_w,
    conv_dt,
    conv_engine,
    ones_sq,
    ident,
    ident32,
    readout,
    tag,
    replay_prework,
    bf16_update_rows,
    n_parts,
    row_selector,
    recurrence_group,
    bcast_depth,
):
    """Replay the accepted prefix of the records into ``state_sb`` and ``conv_sb`` in place (``kda_verify_attention``
    step 8, ``records`` with the ``sequential`` chain). The arguments are the body's tiles and options of that step.
    """
    heads = state_sb.shape[2]
    n_seq = state_sb.shape[1]
    q_tokens = replay_pred.shape[1]
    n_conv = conv_sb.shape[2]
    taps = CONV_TAPS - 1
    bf16 = nl.bfloat16
    fp32 = nl.float32
    state_dt = state_sb.dtype
    # Step 8: replay the accepted prefix. Each token reruns the delta rule on a work copy of the state,
    # and the new state and conv window are committed only for the sequences with j < replay_len. A
    # record beyond replay_len is zeroed first with a select, so a NaN cannot reach the committed state.
    # At 1 local head the gate record has a free dim of 1, so a select_reduce reduces it (NCC_IBIR211).
    # Zero it with a predicated copy of a zeros tile under the complement predicate (no reduce, NaN-safe);
    # above 1 head the select keeps the multi-head traced IR unchanged.
    one_head = heads == 1
    replay_pred_inv = None
    zero_gate = None
    if one_head:
        pred_f = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pred_f, src=replay_pred)
        inv_f = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=inv_f, data=pred_f, op0=nl.equal, operand0=0.0)
        replay_pred_inv = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=replay_pred_inv, src=inv_f)
        zero_gate = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.sbuf)
        nisa.memset(dst=zero_gate, value=0.0)
    work_state = nl.ndarray((P_MAX, n_seq, heads, HEAD_DIM), dtype=state_dt, buffer=nl.sbuf)

    def mask_record(j):
        token_pred = replay_pred[0:P_MAX, j, 0:n_seq]
        cm_j = cm_rec[0:P_MAX, 0:n_seq, j, 0:n_conv]
        decay_j = decay_rec[0:P_MAX, 0:n_seq, j, 0:heads]
        beta_j = beta_rec[0:P_MAX, 0:n_seq, j, 0:heads]
        nisa.select_reduce(dst=cm_j, predicate=_per_sequence(token_pred, (n_conv,)), on_true=cm_j, on_false=0.0)
        if one_head:
            token_inv = replay_pred_inv[0:P_MAX, j, 0:n_seq]
            nisa.tensor_copy_predicated(dst=decay_j, src=zero_gate, predicate=_per_sequence(token_inv, (heads,)))
            nisa.tensor_copy_predicated(dst=beta_j, src=zero_gate, predicate=_per_sequence(token_inv, (heads,)))
        else:
            nisa.select_reduce(
                dst=decay_j, predicate=_per_sequence(token_pred, (heads,)), on_true=decay_j, on_false=0.0
            )
            nisa.select_reduce(dst=beta_j, predicate=_per_sequence(token_pred, (heads,)), on_true=beta_j, on_false=0.0)
        return token_pred, cm_j, decay_j, beta_j

    kernel_assert(replay_prework in REPLAY_PREWORK_FORMS, f"replay_prework must be one of {REPLAY_PREWORK_FORMS}")
    one_pass = replay_prework == "one_pass"
    if one_pass:
        # The conv windows and readout vectors of the replayed tokens do not read the recurrent state, so they run
        # in one pass on the masked records (the per-token scoring path's form, the same bits for each token). A
        # token beyond replay_len gets other values than in the per_token form, but nothing of it is committed.
        masked = [mask_record(j) for j in range(q_tokens)]
        cm_rows = nl.ndarray((P_MAX, q_tokens, n_seq, n_conv), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cm_rows, src=cm_rec.permute((0, 2, 1, 3)), engine=nisa.engine.gpsimd)
        decay_rows = nl.ndarray((P_MAX, q_tokens, n_seq, heads), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=decay_rows, src=decay_rec.permute((0, 2, 1, 3)), engine=nisa.engine.gpsimd)
        conv_rep, conv_rep_seq = _conv_tokens(
            conv_sb, cm_rows.flatten_dims(1, 2), conv_w, q_tokens, conv_dt, products_engine=conv_engine
        )
        k_rep, q_rep, mv_rep, qk_rep = _readout_vectors(conv_rep, decay_rows.flatten_dims(1, 2), ones_sq, readout)
    for j in range(q_tokens):
        if one_pass:
            token_pred, cm_j, decay_j, beta_j = masked[j]
            token = slice(j * n_seq, (j + 1) * n_seq)
            conv_out = conv_rep[0:P_MAX, token, 0:n_conv]
            k_n, q_n = k_rep[0:P_MAX, token, 0:heads], q_rep[0:P_MAX, token, 0:heads]
            mv_in = mv_rep[0:P_MAX, token, 0:heads, 0 : mv_rep.shape[3]]
            qk_psum = None if qk_rep is None else qk_rep[0:P_MAX, token, 0:heads]
            # Positions j + 1 .. j + 3 of the inputs are the conv state after token j.
            next_conv = conv_rep_seq[0:P_MAX, j + 1 : j + 1 + taps].permute((0, 2, 3, 1))
            nisa.tensor_copy(dst=work_state, src=state_sb, engine=nisa.engine.gpsimd)
        else:
            token_pred, cm_j, decay_j, beta_j = mask_record(j)
            nisa.tensor_copy(dst=work_state, src=state_sb, engine=nisa.engine.gpsimd)
            conv_out, window = _conv_token(conv_sb, cm_j, conv_w, f"{tag}_rep{j}", conv_dt, conv_engine)
            k_n, q_n, mv_in, qk_psum = _readout_vectors(conv_out, decay_j, ones_sq, readout)
            next_conv = window[0:P_MAX, 0:n_seq, 0:n_conv, 1:CONV_TAPS]
        _recurrence(
            work_state,
            conv_out[0:P_MAX, 0:n_seq, 2 * heads : 3 * heads],
            k_n,
            q_n,
            decay_j,
            beta_j,
            mv_in,
            qk_psum,
            ident,
            ident32,
            readout,
            bf16_update_rows=bf16_update_rows,
            stationary_parts=n_parts,
            row_selector=row_selector,
            bcast_depth=bcast_depth,
            group=recurrence_group,
        )
        nisa.tensor_copy_predicated(
            dst=state_sb, src=work_state, predicate=_per_sequence(token_pred, (heads, HEAD_DIM))
        )
        nisa.tensor_copy_predicated(dst=conv_sb, src=next_conv, predicate=_per_sequence(token_pred, (n_conv, taps)))


def _finite_copy(dst, src, bound: float):
    """Copy ``src`` to ``dst`` clamped to ``[-bound, bound]``: NaN and Inf become finite, finite values stay."""
    nisa.tensor_scalar(dst=dst, data=src, op0=nl.minimum, operand0=bound, op1=nl.maximum, operand1=-bound)


def _record_view(kv_pool: nl.NkiTensor, page_index: nl.NkiTensor, width: int, span: int = 1) -> nl.NkiTensor:
    """Return the first ``width`` columns of one record page as an HBM view ``[128, width]``.

    The page is viewed as ``[128, page_width // 128]`` with a contiguous row per partition, so each
    partition holds the record of its own channel. ``span`` is ``kda_page_span``: a paired page is
    ``span * width`` elements wide, so its partition rows are those of the 512-row page form.
    """
    page_cols = span * kv_pool.shape[1] // P_MAX
    return kv_pool.ap(pattern=[[page_cols, P_MAX], [1, width]], offset=0, scalar_offset=page_index, indirect_dim=0)


def _record_meta(
    record_page: nl.NkiTensor,
    replay_len: nl.NkiTensor,
    n_pages: int,
    q_tokens: int,
    tag: str,
    null_page: int = 0,
    even_only: bool = False,
):
    """Return the record DMA page and the per-token replay predicate of each sequence.

    A broadcast DMA puts the ``B`` record pages and replay lengths on all 128 partitions, and the
    VectorE computes, for each sequence, whether the record page is in ``[1, n_pages)`` (the DMA page
    is the page id when valid, else page 0) and, for each token ``j``, whether ``j < replay_len`` on a
    valid page.

    Args:
        record_page (nl.NkiTensor): [B, 1] int32 HBM, record page of each sequence.
        replay_len (nl.NkiTensor): [B, 1] int32 HBM, tokens to replay of each sequence.
        n_pages (int): Pages in the cache pool.
        q_tokens (int): Tokens of each sequence (Q).
        tag (str): Unique per-layer name prefix.
        null_page (int): 0, or the KDA null page: a record page at it or below is not valid, and an off DMA page is it
            (``tkg_kda._page_counts``).

    Returns:
        tuple: ``record_dma`` SBUF int32 ``[128, B]`` (use partition 0 as a DMA offset) and
        ``replay_pred`` SBUF uint16 ``[128, Q, B]``.
    """
    n_seq = record_page.shape[0]
    vector = nisa.engine.vector
    meta_i = nl.ndarray((P_MAX, 2, n_seq), dtype=nl.int32, buffer=nl.sbuf)
    for col, src in ((0, record_page), (1, replay_len)):
        nisa.dma_copy(
            dst=meta_i[0:P_MAX, col, 0:n_seq],
            src=src.ap(pattern=[[0, P_MAX], [1, n_seq]], offset=0),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_kda_meta_load_{col}",
        )
    meta_f = nl.ndarray((P_MAX, 2, n_seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=meta_f, src=meta_i, engine=vector)
    page_f = meta_f[0:P_MAX, 0, 0:n_seq]
    len_f = meta_f[0:P_MAX, 1, 0:n_seq]
    high = nl.ndarray((P_MAX, n_seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=high, data=page_f, op0=nl.less, operand0=float(n_pages), engine=vector)
    if even_only:
        _even_pages_only(high, meta_i[0:P_MAX, 0, 0:n_seq], (P_MAX, n_seq))
    valid = nl.ndarray((P_MAX, n_seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=valid, data=page_f, op0=nl.greater_equal, operand0=float(1 + null_page), op1=nl.multiply, operand1=high
    )
    page_dma = nl.ndarray((P_MAX, n_seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=page_dma, data1=page_f, data2=valid, op=nl.multiply, engine=vector)
    _off_to_null_page(page_dma, valid, (P_MAX, n_seq), null_page)
    record_dma = nl.ndarray((P_MAX, n_seq), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=record_dma, src=page_dma, engine=vector)
    pred_f = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(q_tokens):
        nisa.scalar_tensor_tensor(
            dst=pred_f[0:P_MAX, j, 0:n_seq],
            data=len_f,
            op0=nl.greater,
            operand0=float(j),
            op1=nl.multiply,
            operand1=valid,
        )
    replay_pred = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=replay_pred, src=pred_f, engine=vector)
    return record_dma, replay_pred


def _load_slot_reg(slot_regs: dict, slot_write_pages: nl.NkiTensor, token: int, seq: int, n_seq: int) -> None:
    """Load the slot page id of ``seq`` at scored token ``token`` (``slot_write_pages[0, token, seq]``) into a new
    register, at ``slot_regs[token][seq]`` (``kda_verify_attention`` ``slot_store_regs``)."""
    regs = slot_regs.setdefault(token, [None] * n_seq)
    regs[seq] = nisa.register_alloc()
    nisa.register_load(regs[seq], slot_write_pages[0:1, token, seq : seq + 1])


def _write_slot_pages(
    write_state_slots: nl.NkiTensor,
    n_pages: int,
    q_tokens: int,
    tag: str,
    loads: list = None,
    null_page: int = 0,
    contiguous: bool = False,
    even_only: bool = False,
) -> nl.NkiTensor:
    """Return the ``[128, Q, B]`` int32 DMA pages of the Q per-token state slots of each sequence.

    A broadcast DMA puts the ``[B, Q]`` slot ids on all 128 partitions and the VectorE clamps each to
    ``[1, n_pages)``: the DMA page is the slot id when valid, else page 0, so a pad or null slot writes
    to page 0 (the null page). ``loads``: None, or a list that gets the DMA. With a KDA null page (``null_page`` > 0),
    a slot at it is not valid either, and a pad or null slot writes to the KDA null page. ``contiguous``: load the
    slot ids in their ``[B, Q]`` HBM order (one run of ``B * Q`` int32 per partition) and transpose them in the
    VectorE cast's source view, in place of the transposed broadcast of ``128 * Q * B`` 4-byte elements; the same
    pages.
    """
    n_seq = write_state_slots.shape[0]
    vector = nisa.engine.vector
    if contiguous:
        rows_i = nl.ndarray((P_MAX, n_seq * q_tokens), dtype=nl.int32, buffer=nl.sbuf)
        load = nisa.dma_copy(
            dst=rows_i,
            src=write_state_slots.ap(pattern=[[0, P_MAX], [1, n_seq * q_tokens]], offset=0),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_kda_slot_pages_load",
        )
        cast_src = rows_i.reshape_dim(1, (n_seq, q_tokens)).permute((0, 2, 1))
    else:
        pages_i = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=nl.int32, buffer=nl.sbuf)
        load = nisa.dma_copy(
            dst=pages_i,
            src=write_state_slots.ap(pattern=[[0, P_MAX], [1, q_tokens], [q_tokens, n_seq]], offset=0),
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
            name=f"{tag}_kda_slot_pages_load",
        )
        cast_src = pages_i
    if loads is not None:
        loads.append(load)
    pages_f = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pages_f, src=cast_src, engine=vector)
    high = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=high, data=pages_f, op0=nl.less, operand0=float(n_pages), engine=vector)
    if even_only:
        _even_pages_only(high, cast_src, (P_MAX, q_tokens, n_seq))
    valid = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=valid, data=pages_f, op0=nl.greater_equal, operand0=float(1 + null_page), op1=nl.multiply, operand1=high
    )
    page_dma_f = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=page_dma_f, data1=pages_f, data2=valid, op=nl.multiply, engine=vector)
    _off_to_null_page(page_dma_f, valid, (P_MAX, q_tokens, n_seq), null_page)
    page_dma = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=page_dma, src=page_dma_f, engine=vector)
    return page_dma


def _decay_gate(cm, small_sb, params_sb, a_exp, ones_sq, heads, small):
    """Return ``decay = exp(-5 * sigmoid(exp(A_log) * (bf16(f_a @ Wf_b) + dt_bias)))`` of all rows.

    ``cm`` is the channel-major projection ``[128, R, n_cm]``; column ``4 * heads`` holds f_a. The
    result is ``[128, R, heads]``.
    """
    n_rows = cm.shape[1]
    fp32 = nl.float32
    bf16 = nl.bfloat16
    gate_psum = nl.ndarray((P_MAX, heads, n_rows), dtype=fp32, buffer=nl.psum)
    for head_idx in range(heads):
        f_b_start = small["f_b"] + head_idx * HEAD_DIM
        nisa.nc_matmul(
            dst=gate_psum[0:P_MAX, head_idx, 0:n_rows],
            stationary=small_sb[0:P_MAX, f_b_start : f_b_start + HEAD_DIM],
            moving=cm[0:P_MAX, 0:n_rows, 4 * heads],
        )
    gate_bf = nl.ndarray((P_MAX, n_rows, heads), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gate_bf, src=gate_psum.permute((0, 2, 1)), engine=nisa.engine.vector)
    gate_shift = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gate_shift,
        data1=gate_bf,
        data2=params_sb[0:P_MAX, small["dt_bias"] : small["dt_bias"] + heads].expand_dim(1).broadcast(1, n_rows),
        op=nl.add,
    )
    gate_scaled = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gate_scaled, data1=gate_shift, data2=a_exp.expand_dim(1).broadcast(1, n_rows), op=nl.multiply
    )
    gate_exp = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=gate_exp, op=nl.exp, data=gate_scaled, scale=-1.0)
    gate_sig = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.sbuf)
    _divide_by_one_plus(gate_sig, gate_exp, 1.0)
    decay = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=decay, op=nl.exp, data=gate_sig, scale=GATE_LOWER_BOUND)
    return decay


def _conv_token(hist, cm_values, conv_w, tag, conv_dt=nl.bfloat16, products_engine=nisa.engine.gpsimd):
    """Run the short convolution of one token for its B sequences.

    ``hist`` ``[128, B, 3*Hl, 3]`` is the current conv window (oldest tap first), ``cm_values``
    ``[128, B, 3*Hl]`` the new projection value of this token, ``conv_w`` ``[128, 3*Hl, 4]`` the conv
    weight (BF16, or FP32 for the FP32 form). The products, the sum and the SiLU output take ``conv_dt``.
    Returns the SiLU output ``[128, B, 3*Hl]`` and the window ``[128, B, 3*Hl, 4]`` (its last three taps
    are the next conv state). ``products_engine`` (default the GpSimd) runs the products of the 3 window taps.
    """
    n_seq = cm_values.shape[1]
    n_conv = cm_values.shape[2]
    taps = CONV_TAPS - 1
    bf16 = nl.bfloat16
    fp32 = nl.float32
    window = nl.ndarray((P_MAX, n_seq, n_conv, CONV_TAPS), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=window[0:P_MAX, 0:n_seq, 0:n_conv, 0:taps], src=hist, engine=nisa.engine.gpsimd)
    nisa.tensor_copy(dst=window[0:P_MAX, 0:n_seq, 0:n_conv, taps], src=cm_values, engine=nisa.engine.gpsimd)
    conv_prod = nl.ndarray((P_MAX, n_seq, n_conv, CONV_TAPS), dtype=conv_dt, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=conv_prod[0:P_MAX, 0:n_seq, 0:n_conv, 0:taps],
        data1=window[0:P_MAX, 0:n_seq, 0:n_conv, 0:taps],
        data2=conv_w[0:P_MAX, 0:n_conv, 0:taps].expand_dim(1).broadcast(1, n_seq),
        op=nl.multiply,
        engine=products_engine,
    )
    nisa.tensor_tensor(
        dst=conv_prod[0:P_MAX, 0:n_seq, 0:n_conv, taps],
        data1=cm_values,
        data2=conv_w[0:P_MAX, 0:n_conv, taps].expand_dim(1).broadcast(1, n_seq),
        op=nl.multiply,
    )
    conv_sum = nl.ndarray((P_MAX, n_seq, n_conv), dtype=conv_dt, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=conv_sum, op=nl.add, data=conv_prod, axis=3)
    conv_exp = nl.ndarray((P_MAX, n_seq, n_conv), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=conv_exp, op=nl.exp, data=conv_sum, scale=-1.0)
    conv_out = nl.ndarray((P_MAX, n_seq, n_conv), dtype=conv_dt, buffer=nl.sbuf)
    _divide_by_one_plus(conv_out, conv_exp, conv_sum)
    return conv_out, window


def _conv_tokens(hist, cm_rows, conv_w, q_tokens, conv_dt=nl.bfloat16, products_engine=nisa.engine.gpsimd):
    """Run the short convolution of the Q tokens of B sequences in one pass (``conv_dt`` as ``_conv_token``).

    ``hist`` ``[128, B, 3*Hl, 3]`` is the conv window before token 0, ``cm_rows`` ``[128, Q*B, 3*Hl]`` the new
    projection values (row ``j * B + b``), ``conv_w`` ``[128, 3*Hl, 4]``. Each element gets the products, the engines
    and the 4-tap reduce of ``_conv_token``, so it has the same bits. Returns the SiLU output ``[128, Q*B, 3*Hl]`` and
    the inputs by position ``[128, 3 + Q, B, 3*Hl]``: positions ``j`` to ``j + 3`` are the window of token ``j``.
    ``products_engine`` (default the GpSimd, as ``_conv_token``) runs the products of the 3 window taps.

    On trn3 with VectorE products the products are tap-major (``[128, 4, Q*B, 3*Hl]``): each tap writes one contiguous
    run, because an SBUF write with a strided innermost dimension takes several times as long as a contiguous one on
    VectorE (a GpSimd product costs about the same either way). The reduce reads the 4 taps of each element with a
    stride, in the same tap order, so the sums keep their bits. Trn2 and the GpSimd products keep the tap-innermost
    products (``[128, Q*B, 3*Hl, 4]``) and their programs.
    """
    n_seq, n_conv = hist.shape[1], hist.shape[2]
    n_rows = cm_rows.shape[1]
    taps = CONV_TAPS - 1
    bf16 = nl.bfloat16
    fp32 = nl.float32
    conv_seq = nl.ndarray((P_MAX, taps + q_tokens, n_seq, n_conv), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=conv_seq[0:P_MAX, 0:taps], src=hist.permute((0, 3, 1, 2)), engine=nisa.engine.gpsimd)
    nisa.tensor_copy(
        dst=conv_seq[0:P_MAX, taps : taps + q_tokens],
        src=cm_rows.reshape_dim(1, (q_tokens, n_seq)),
        engine=nisa.engine.gpsimd,
    )
    tap_major = is_trn3() and products_engine == nisa.engine.vector
    if tap_major:
        conv_prod = nl.ndarray((P_MAX, CONV_TAPS, n_rows, n_conv), dtype=conv_dt, buffer=nl.sbuf)
    else:
        conv_prod = nl.ndarray((P_MAX, n_rows, n_conv, CONV_TAPS), dtype=conv_dt, buffer=nl.sbuf)

    def tap_dst(tap: int):
        if tap_major:
            return conv_prod[0:P_MAX, tap, 0:n_rows, 0:n_conv]
        return conv_prod[0:P_MAX, 0:n_rows, 0:n_conv, tap]

    for tap in range(taps):
        nisa.tensor_tensor(
            dst=tap_dst(tap),
            data1=conv_seq[0:P_MAX, tap : tap + q_tokens].flatten_dims(1, 2),
            data2=conv_w[0:P_MAX, 0:n_conv, tap].expand_dim(1).broadcast(1, n_rows),
            op=nl.multiply,
            engine=products_engine,
        )
    nisa.tensor_tensor(
        dst=tap_dst(taps),
        data1=cm_rows,
        data2=conv_w[0:P_MAX, 0:n_conv, taps].expand_dim(1).broadcast(1, n_rows),
        op=nl.multiply,
        **({"engine": nisa.engine.vector} if tap_major else {}),
    )
    conv_sum = nl.ndarray((P_MAX, n_rows, n_conv), dtype=conv_dt, buffer=nl.sbuf)
    by_tap = conv_prod.permute((0, 2, 3, 1)) if tap_major else conv_prod
    nisa.tensor_reduce(dst=conv_sum, op=nl.add, data=by_tap, axis=3)
    conv_exp = nl.ndarray((P_MAX, n_rows, n_conv), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=conv_exp, op=nl.exp, data=conv_sum, scale=-1.0)
    conv_out = nl.ndarray((P_MAX, n_rows, n_conv), dtype=conv_dt, buffer=nl.sbuf)
    _divide_by_one_plus(conv_out, conv_exp, conv_sum)
    return conv_out, conv_seq


def _quarter(n_groups: int, k: int) -> range:
    """Return the weight groups of quarter ``k`` (0 .. 3) of ``n_groups`` (``head_pipeline`` ``heads`` pass C)."""
    return range(n_groups * k // 4, n_groups * (k + 1) // 4)


class _HeadStages:
    """The conv / readout pass of ``kda_verify_attention`` ``head_pipeline`` ``heads`` in head stages.

    ``run(h0, h1, cm_stage)`` runs the pass of heads ``h0 .. h1 - 1`` from their projection columns ``cm_stage``
    ``[128, R, 3 (h1 - h0)]`` (q, k, v blocks of the stage): the conv window and the conv weights of the stage's columns,
    ``_conv_tokens``, then ``_readout_vectors`` into the stage's head columns of the call's tiles. The tiles hold what the
    one-pass path gives the chain: ``v_rows`` (the conv output at its v columns), ``conv_seq`` (the conv inputs by
    position, for the slot conv stores), ``k_n`` / ``q_n`` / ``mv`` and ``qk`` (each stage's PSUM q.k sums copied to one
    SBUF tile). Every op is per element or per column, and the copies are exact, so the values do not change.
    """

    def __init__(self, conv_sb, conv_w, decay, ones_sq, readout, heads, n_seq, q_tokens, conv_dt, conv_engine):
        fp32, bf16 = nl.float32, nl.bfloat16
        n_rows, n_conv, taps = q_tokens * n_seq, N_CONV_PROJ * heads, CONV_TAPS - 1
        split = readout == "decoupled_bf16"
        self.n_mv = 2 * (2 if split else 1)
        self.args = (conv_sb, conv_w, decay, ones_sq, readout, heads, n_seq, q_tokens, conv_dt, conv_engine)
        self.qk_n_all = nl.ndarray((P_MAX, n_rows, heads, 2), dtype=fp32, buffer=nl.sbuf)
        self.k_n = self.qk_n_all[0:P_MAX, 0:n_rows, 0:heads, 0]
        self.q_n = self.qk_n_all[0:P_MAX, 0:n_rows, 0:heads, 1]
        self.mv = nl.ndarray((P_MAX, n_rows, heads, self.n_mv), dtype=bf16 if split else fp32, buffer=nl.sbuf)
        # The q.k sums of all heads in SBUF (each stage's PSUM sums copied in, fp32 exact; the chain reads them).
        self.qk = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.sbuf)
        self.v_rows = nl.ndarray((P_MAX, n_rows, n_conv), dtype=conv_dt, buffer=nl.sbuf)
        self.conv_seq = nl.ndarray((P_MAX, taps + q_tokens, n_seq, n_conv), dtype=bf16, buffer=nl.sbuf)

    def run(self, h0: int, h1: int, cm_stage) -> None:
        conv_sb, conv_w, decay, ones_sq, readout, heads, n_seq, q_tokens, conv_dt, conv_engine = self.args
        n_rows, taps, nh = q_tokens * n_seq, CONV_TAPS - 1, h1 - h0
        gpsimd = nisa.engine.gpsimd
        hist = nl.ndarray((P_MAX, n_seq, N_CONV_PROJ * nh, taps), dtype=conv_sb.dtype, buffer=nl.sbuf)
        weights = nl.ndarray((P_MAX, N_CONV_PROJ * nh, CONV_TAPS), dtype=conv_w.dtype, buffer=nl.sbuf)
        for kind in range(N_CONV_PROJ):
            src, dst = slice(kind * heads + h0, kind * heads + h1), slice(kind * nh, (kind + 1) * nh)
            nisa.tensor_copy(
                dst=hist[0:P_MAX, 0:n_seq, dst, 0:taps], src=conv_sb[0:P_MAX, 0:n_seq, src, 0:taps], engine=gpsimd
            )
            nisa.tensor_copy(
                dst=weights[0:P_MAX, dst, 0:CONV_TAPS], src=conv_w[0:P_MAX, src, 0:CONV_TAPS], engine=gpsimd
            )
        conv, seq = _conv_tokens(hist, cm_stage, weights, q_tokens, conv_dt, products_engine=conv_engine)
        nisa.tensor_copy(
            dst=self.v_rows[0:P_MAX, 0:n_rows, 2 * heads + h0 : 2 * heads + h1],
            src=conv[0:P_MAX, 0:n_rows, 2 * nh : 3 * nh],
            engine=nisa.engine.vector,
        )
        for kind in range(N_CONV_PROJ):
            nisa.tensor_copy(
                dst=self.conv_seq[0:P_MAX, 0 : taps + q_tokens, 0:n_seq, kind * heads + h0 : kind * heads + h1],
                src=seq[0:P_MAX, 0 : taps + q_tokens, 0:n_seq, kind * nh : (kind + 1) * nh],
                engine=gpsimd,
            )
        _, _, _, qk_stage = _readout_vectors(
            conv,
            decay[0:P_MAX, 0:n_rows, h0:h1],
            ones_sq,
            readout,
            outs={
                "qk_n": self.qk_n_all[0:P_MAX, 0:n_rows, h0:h1, 0:2],
                "mv": self.mv[0:P_MAX, 0:n_rows, h0:h1, 0 : self.n_mv],
            },
        )
        nisa.tensor_copy(dst=self.qk[0:P_MAX, 0:n_rows, h0:h1], src=qk_stage, engine=nisa.engine.vector)


def _readout_vectors(conv_out, decay, ones_sq, readout, outs: dict = None):
    """Return the L2-normed k, q, the decayed moving vectors and the q.k matmul of one token.

    ``conv_out`` ``[128, B, 3*Hl]`` holds q, k, v blocks; ``decay`` is ``[128, B, Hl]``. Returns
    ``k_n``, ``q_n`` (``[128, B, Hl]`` views), ``mv_in`` ``[128, B, Hl, n_mv]`` and ``qk_psum``
    (``[128, B, Hl]`` or None), as the decode body forms them. ``outs`` (None: tiles of the call): views that take the
    results in place of new tiles: ``qk_n`` ``[128, B, Hl, 2]`` (k, q), ``mv`` ``[128, B, Hl, n_mv]`` and ``qk``
    (PSUM ``[128, B, Hl]``; ``kda_verify_attention`` ``head_pipeline`` ``heads`` fills the head columns of one tile
    per stage). The same ops, so the values do not change.
    """
    outs = outs or {}

    def given(name, shape, dtype, buffer):
        return outs[name] if name in outs else nl.ndarray(shape, dtype=dtype, buffer=buffer)

    n_seq = conv_out.shape[1]
    heads = conv_out.shape[2] // N_CONV_PROJ
    fp32 = nl.float32
    bf16 = nl.bfloat16
    qk_sq = nl.ndarray((P_MAX, n_seq, 2 * heads), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=qk_sq,
        data1=conv_out[0:P_MAX, 0:n_seq, 0 : 2 * heads],
        data2=conv_out[0:P_MAX, 0:n_seq, 0 : 2 * heads],
        op=nl.multiply,
    )
    qk_ss = nl.ndarray((P_MAX, n_seq, 2 * heads), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=qk_ss, stationary=ones_sq, moving=qk_sq, is_stationary_onezero=True)
    qk_log = nl.ndarray((P_MAX, n_seq, 2 * heads), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=qk_log, op=nl.log, data=qk_ss, bias=L2_EPS)
    qk_rnorm = nl.ndarray((P_MAX, n_seq, 2 * heads), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=qk_rnorm, op=nl.exp, data=qk_log, scale=-0.5)
    qk_n = given("qk_n", (P_MAX, n_seq, heads, 2), fp32, nl.sbuf)
    k_n = qk_n[0:P_MAX, 0:n_seq, 0:heads, 0]
    q_n = qk_n[0:P_MAX, 0:n_seq, 0:heads, 1]
    nisa.scalar_tensor_tensor(
        dst=q_n,
        data=conv_out[0:P_MAX, 0:n_seq, 0:heads],
        op0=nl.multiply,
        operand0=float(HEAD_DIM) ** -0.5,
        op1=nl.multiply,
        operand1=qk_rnorm[0:P_MAX, 0:n_seq, 0:heads],
    )
    nisa.tensor_tensor(
        dst=k_n,
        data1=conv_out[0:P_MAX, 0:n_seq, heads : 2 * heads],
        data2=qk_rnorm[0:P_MAX, 0:n_seq, heads : 2 * heads],
        op=nl.multiply,
    )
    decoupled = readout != "reference"
    split = readout == "decoupled_bf16"
    n_vec = 2 if decoupled else 1
    n_mv = n_vec * (2 if split else 1)
    mv_in = given("mv", (P_MAX, n_seq, heads, n_mv), bf16 if split else fp32, nl.sbuf)
    decay_cols = decay.expand_dim(3).broadcast(3, n_vec)
    if split:
        scaled = nl.ndarray((P_MAX, n_seq, heads, n_vec), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=scaled, data1=qk_n[0:P_MAX, 0:n_seq, 0:heads, 0:n_vec], data2=decay_cols, op=nl.multiply)
        high = mv_in[0:P_MAX, 0:n_seq, 0:heads, 0:n_mv:2]
        nisa.tensor_copy(dst=high, src=scaled, engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=mv_in[0:P_MAX, 0:n_seq, 0:heads, 1:n_mv:2], data1=scaled, data2=high, op=nl.subtract)
    else:
        nisa.tensor_tensor(dst=mv_in, data1=qk_n[0:P_MAX, 0:n_seq, 0:heads, 0:n_vec], data2=decay_cols, op=nl.multiply)
    qk_psum = None
    if decoupled:
        qk_prod = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=qk_prod, data1=q_n, data2=k_n, op=nl.multiply)
        qk_psum = given("qk", (P_MAX, n_seq, heads), fp32, nl.psum)
        nisa.nc_matmul(dst=qk_psum, stationary=ones_sq, moving=qk_prod, is_stationary_onezero=True)
    return k_n, q_n, mv_in, qk_psum


def _recurrence(
    staged_state,
    v_cols,
    k_n,
    q_n,
    decay,
    beta,
    mv_in,
    qk_psum,
    ident,
    ident32,
    readout,
    state_write_wait=None,
    out_state=None,
    state_bufs=None,
    ones_sq=None,
    bf16_update_rows=False,
    stationary_parts=0,
    row_selector=None,
    bcast_depth=2,
    group=None,
    out=None,
    bf16_cols=None,
):
    """Run the delta rule of one token for its B sequences and return the bf16 readout ``[128, B, Hl]``.

    This is the decode body ``_recurrence`` with the decoupled state update run in place (the verify
    body needs the new state for the next token, so it does not defer the update). ``staged_state``
    (bf16) holds the input state; the update writes the new bf16 state back into it.

    ``state_write_wait`` is an optional list of DMA instructions that read ``staged_state`` in the
    previous token (its slot stores). The in-place state write of this token waits for them, so a slot
    store finishes reading the state before this token overwrites it (nki-dma-source-liveness). When it
    is empty the call adds no edge, so the records and decode paths do not change.

    ``out_state`` (bf16, the shape of ``staged_state``) takes the new state in place of ``staged_state``, and
    ``state_bufs`` gives the fp32 state buffers (``min(B, 2 * _RECURRENCE_GROUP)`` tiles ``[128, Hl, 128]``). A
    caller that gives each token its own tiles removes the write-after-read waits between tokens.

    ``ones_sq`` (fp32 ones ``[128, 128]``) selects, at one head and the reference readout, the broadcast of the
    difference row as one VectorE diagonal and one ones matmul, in place of the row transpose, its PSUM copy and
    the one-hot matmul. Both forms add the same exact terms (``x * 1`` and signed zeros), so the bits are equal.

    ``bf16_update_rows`` (``decoupled_bf16`` only) takes the rows of the update from ``tkg_kda._bf16_diff_rows`` and
    adds the high and the low row in one broadcast matmul per head, as the decode body does.

    ``row_selector`` (with ``bf16_update_rows``): None (default) builds the bf16 row selector of the update here; else
    the selector that the caller built once for its tokens (``_bf16_row_selector``, the same values). ``bcast_depth``
    (default 2): the PSUM tiles of the update's broadcast ring, ``min(B, bcast_depth)``, each a PSUM tensor of its own
    bank; the broadcast matmuls of sequence ``s + depth`` wait for the reads of sequence ``s``. The bits do not change.

    ``group`` (None: ``_RECURRENCE_GROUP``): the sequences of each group. With a given ``group`` each group takes its own
    prediction PSUM tile and its own ``_bf16_diff_rows`` PSUM tile (the same values).
    ``bf16_cols`` (None: a memset in each call) maps the first sequence of each group to a zeroed rows tile of
    ``tkg_kda.bf16_diff_cols`` for ``_bf16_diff_rows`` (``kda_verify_attention`` ``bf16_cols_once``; the same bytes).

    An fp32 ``staged_state`` (the fp32 forms of ``tkg_kda.KDA_STATE_DTYPES``) chains in fp32, and ``out_state``
    is fp32 too. With ``decoupled_bf16`` the VectorE first writes ``stationary_parts`` bf16 stationary tiles of each
    sequence's input state, ``bf16(S)`` and for 2 parts ``bf16(S - bf16(S))``, and the prediction of each head adds
    their products in PSUM, as ``tkg_kda._recurrence`` does with the parts of ``_load_recurrent_states``. The
    ``reference`` and ``decoupled`` readouts use the fp32 state itself as their fp32 stationary operand.

    ``out`` (bf16 ``[128, B, Hl]``, None: a tile of the call) takes the readout: the last op of the readout writes it
    there, so a caller that keeps the readouts of its tokens in one tile needs no copy.
    """
    state_write_wait = tuple(state_write_wait or ())
    out_state = staged_state if out_state is None else out_state
    n_seq, heads = k_n.shape[1], k_n.shape[2]
    n_mv = mv_in.shape[3]
    fp32 = nl.float32
    bf16 = nl.bfloat16
    decoupled = readout != "reference"
    split = readout == "decoupled_bf16"
    fp32_state = staged_state.dtype == fp32
    kernel_assert(
        not split or (stationary_parts > 0) == fp32_state,
        "decoupled_bf16 needs the bf16 stationary parts of an fp32 state, and none for a bf16 state",
    )
    kernel_assert(out_state.dtype == staged_state.dtype, f"out_state {out_state.dtype} != state {staged_state.dtype}")
    stationaries = ()
    if split and fp32_state:
        # Each sequence has its own rows of each tile, so a NaN state reaches only its own prediction.
        stationaries = tuple(
            nl.ndarray((P_MAX, n_seq, heads, HEAD_DIM), dtype=bf16, buffer=nl.sbuf) for _ in range(stationary_parts)
        )
        for seq in range(n_seq):
            loaded = staged_state[0:P_MAX, seq, 0:heads, 0:HEAD_DIM]
            high = stationaries[0][0:P_MAX, seq, 0:heads, 0:HEAD_DIM]
            nisa.tensor_copy(dst=high, src=loaded, engine=nisa.engine.vector)
            if stationary_parts > 1:
                # S - bf16(S) is exact in fp32; its bf16 rounding is the low part.
                nisa.tensor_tensor(
                    dst=stationaries[1][0:P_MAX, seq, 0:heads, 0:HEAD_DIM],
                    data1=loaded,
                    data2=high,
                    op=nl.subtract,
                    engine=nisa.engine.vector,
                )
    split_groups = group is not None
    group = min(n_seq, _RECURRENCE_GROUP if group is None else group)
    n_bufs = min(n_seq, 2 * group)
    if state_bufs is None:
        state_bufs = [nl.ndarray((P_MAX, heads, HEAD_DIM), dtype=fp32, buffer=nl.sbuf) for _ in range(n_bufs)]
    groups = [(first, min(n_seq, first + group)) for first in range(0, n_seq, group)]
    if split_groups:
        # One prediction tile for each group (a PSUM tensor, its own bank): a group's VectorE reads do not share a bank
        # with the other group's TensorE writes.
        pred_tiles = {
            first: nl.ndarray((P_MAX, last - first, heads, n_mv), dtype=fp32, buffer=nl.psum) for first, last in groups
        }
    else:
        pred_psum = nl.ndarray((P_MAX, n_seq, heads, n_mv), dtype=fp32, buffer=nl.psum)
        pred_tiles = {first: pred_psum for first, _ in groups}
    pred_base = {first: first if split_groups else 0 for first, _ in groups}
    diff = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.sbuf)
    if out is None:
        out_bf = nl.ndarray((P_MAX, n_seq, heads), dtype=bf16, buffer=nl.sbuf)
    else:
        kernel_assert(
            tuple(out.shape) == (P_MAX, n_seq, heads) and out.dtype == bf16,
            f"out must be bf16 {(P_MAX, n_seq, heads)}, got {out.dtype} {tuple(out.shape)}",
        )
        out_bf = out
    qkb = None
    if decoupled:
        qkb = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=qkb, data1=qk_psum, data2=beta, op=nl.multiply)
    diag = ones_sq is not None and not split and heads == 1
    bf16_rows = bf16_update_rows and split

    def predict(first, last):
        width = last - first
        pred, base = pred_tiles[first], pred_base[first]
        for seq in range(first, last):
            state = state_bufs[seq % n_bufs]
            loaded = staged_state[0:P_MAX, seq, 0:heads, 0:HEAD_DIM]
            if split:
                parts = [part[0:P_MAX, seq, 0:heads, 0:HEAD_DIM] for part in stationaries] or [loaded]
            elif decoupled and fp32_state:
                # An fp32 state is the fp32 stationary itself; the update decays it into the buffer.
                parts = [loaded]
            else:
                nisa.tensor_copy(dst=state, src=loaded, engine=nisa.engine.gpsimd)
                parts = [state]
            stationary = parts[0]
            for head_idx in range(heads):
                for part_idx, part in enumerate(parts):
                    # The parts of a head accumulate back to back, so no other group opens in the PSUM zone between.
                    nisa.nc_matmul(
                        dst=pred[0:P_MAX, seq - base, head_idx, 0:n_mv],
                        stationary=part[0:P_MAX, head_idx, 0:HEAD_DIM],
                        moving=mv_in[0:P_MAX, seq, head_idx, 0:n_mv],
                        accumulate=part_idx > 0,
                    )
            if not decoupled:
                _decay_state(state, stationary, decay, seq)
        # The diagonal form keeps v - pred in the diff tile and multiplies by beta in the diagonal op.
        v_minus = (
            diff[0:P_MAX, first:last, 0:heads]
            if diag
            else nl.ndarray((P_MAX, width, heads), dtype=fp32, buffer=nl.sbuf)
        )
        nisa.tensor_tensor(
            dst=v_minus,
            data1=v_cols[0:P_MAX, first:last, 0:heads],
            data2=pred[0:P_MAX, first - base : last - base, 0:heads, 0],
            op=nl.subtract,
        )
        if split:
            nisa.tensor_tensor(
                dst=v_minus, data1=v_minus, data2=pred[0:P_MAX, first - base : last - base, 0:heads, 1], op=nl.subtract
            )
        if decoupled:
            q_part = nl.ndarray((P_MAX, width, heads), dtype=fp32, buffer=nl.sbuf)
            q_col = 2 if split else 1
            nisa.tensor_tensor(dst=q_part, data1=v_minus, data2=qkb[0:P_MAX, first:last, 0:heads], op=nl.multiply)
            for col in range(q_col, n_mv):
                last_col = col == n_mv - 1
                nisa.tensor_tensor(
                    dst=out_bf[0:P_MAX, first:last, 0:heads] if last_col else q_part,
                    data1=q_part,
                    data2=pred[0:P_MAX, first - base : last - base, 0:heads, col],
                    op=nl.add,
                    engine=nisa.engine.vector,
                )
        if not diag:
            nisa.tensor_tensor(
                dst=diff[0:P_MAX, first:last, 0:heads],
                data1=v_minus,
                data2=beta[0:P_MAX, first:last, 0:heads],
                op=nl.multiply,
            )

    def update(first, last, bcast_bufs, out_psum, bf16_parts=None):
        # Each sequence gets its own row tiles: the broadcast matmul of a sequence must read only its
        # own rows, because a matmul adds a 0 x row term for every other row of its moving operand and
        # 0 x Inf is NaN (the decoupled_bf16 split also makes Inf - Inf = NaN). A shared group tile would
        # let one sequence with a non-finite state break the new state of every sequence of the group. The
        # per-sequence PSUM tiles pack to the width of a shared group tile, so PSUM holds the same bytes.
        if bf16_parts is not None:
            slot = [g for g, _ in groups].index(first) if split_groups else None
            # bf16_cols: the caller's zeroed rows tile of this group (no memset here).
            cols = None if bf16_cols is None else bf16_cols.get(first)
            rows_of = _bf16_diff_rows(diff, first, last, bf16_parts[0], slot=slot, cols=cols)
        elif not diag:
            stride = -(-heads // _QUADRANT) * _QUADRANT
            per_tile = max(1, P_MAX // stride)
            width = last - first
            rows_psum = [
                nl.ndarray((min(per_tile, width - start) * stride, HEAD_DIM), dtype=fp32, buffer=nl.psum)
                for start in range(0, width, per_tile)
            ]
            rows_of = {}
            for seq in range(first, last):
                index = seq - first
                base = (index % per_tile) * stride
                source = rows_psum[index // per_tile][base : base + heads, 0:HEAD_DIM]
                nisa.nc_matmul(
                    dst=source,
                    stationary=diff[0:P_MAX, seq, 0:heads],
                    moving=ident32,
                    is_moving_onezero=True,
                    accumulate=False,
                )
                if split:
                    tiles = [nl.ndarray((heads, HEAD_DIM), dtype=bf16, buffer=nl.sbuf) for _ in range(2)]
                    nisa.tensor_copy(dst=tiles[0], src=source, engine=nisa.engine.scalar)
                    nisa.tensor_tensor(dst=tiles[1], data1=source, data2=tiles[0], op=nl.subtract)
                else:
                    tiles = [nl.ndarray((heads, HEAD_DIM), dtype=fp32, buffer=nl.sbuf)]
                    nisa.tensor_copy(dst=tiles[0], src=source, engine=nisa.engine.scalar)
                rows_of[seq] = tiles
        one_hot = ident if split else ident32
        for seq in range(first, last):
            state = state_bufs[seq % n_bufs]
            if decoupled:
                _decay_state(state, staged_state[0:P_MAX, seq, 0:heads, 0:HEAD_DIM], decay, seq)
            bcast = bcast_bufs[seq % len(bcast_bufs)]
            if diag:
                # bcast[p, v] = (v - pred)[v] * beta: the ones matmul sums column v of the diagonal, its one nonzero term.
                diag_t = nl.ndarray((P_MAX, HEAD_DIM), dtype=fp32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=diag_t,
                    data=ident32,
                    op0=nl.multiply,
                    operand0=diff[0:P_MAX, seq, 0:1],
                    op1=nl.multiply,
                    operand1=beta[0:P_MAX, seq, 0:1],
                )
                nisa.nc_matmul(
                    dst=bcast[0:P_MAX, 0, 0:HEAD_DIM],
                    stationary=ones_sq,
                    moving=diag_t,
                    is_stationary_onezero=True,
                    accumulate=False,
                )
            elif bf16_parts is not None:
                # One matmul adds the high and the low row of the head: high + low is exact in fp32.
                rows, base = rows_of[seq]
                for head_idx in range(heads):
                    nisa.nc_matmul(
                        dst=bcast[0:P_MAX, head_idx, 0:HEAD_DIM],
                        stationary=bf16_parts[1][base : base + 2 * heads, head_idx : head_idx + 1].broadcast(1, P_MAX),
                        moving=rows,
                        accumulate=False,
                    )
            else:
                for head_idx in range(heads):
                    for part_idx, part in enumerate(rows_of[seq]):
                        nisa.nc_matmul(
                            dst=bcast[0:P_MAX, head_idx, 0:HEAD_DIM],
                            stationary=one_hot[0:heads, head_idx : head_idx + 1].broadcast(1, P_MAX),
                            moving=part,
                            is_stationary_onezero=not split,
                            accumulate=part_idx > 0,
                        )
            for head_idx in range(heads):
                upd = nisa.scalar_tensor_tensor(
                    dst=out_state[0:P_MAX, seq, head_idx, 0:HEAD_DIM]
                    if decoupled
                    else state[0:P_MAX, head_idx, 0:HEAD_DIM],
                    data=bcast[0:P_MAX, head_idx, 0:HEAD_DIM],
                    op0=nl.multiply,
                    operand0=k_n[0:P_MAX, seq, head_idx : head_idx + 1],
                    op1=nl.add,
                    operand1=state[0:P_MAX, head_idx, 0:HEAD_DIM],
                )
                if decoupled and state_write_wait and hasattr(upd, "depends_on"):
                    upd.depends_on(*state_write_wait)
            if not decoupled:
                for head_idx in range(heads):
                    nisa.nc_matmul(
                        dst=out_psum[0:P_MAX, seq, head_idx : head_idx + 1],
                        stationary=state[0:P_MAX, head_idx, 0:HEAD_DIM],
                        moving=q_n[0:P_MAX, seq, head_idx : head_idx + 1],
                        accumulate=False,
                    )
                cp = nisa.tensor_copy(
                    dst=out_state[0:P_MAX, seq, 0:heads, 0:HEAD_DIM], src=state, engine=nisa.engine.gpsimd
                )
                if state_write_wait and hasattr(cp, "depends_on"):
                    cp.depends_on(*state_write_wait)

    def new_bcast_bufs():
        return [
            nl.ndarray((P_MAX, heads, HEAD_DIM), dtype=fp32, buffer=nl.psum) for _ in range(min(n_seq, bcast_depth))
        ]

    if decoupled:
        for first, last in groups:
            predict(first, last)
        bcast_bufs = new_bcast_bufs()
        bf16_parts = _bf16_update_parts(ident, heads, row_selector) if bf16_rows else None
        for first, last in groups:
            update(first, last, bcast_bufs, None, bf16_parts)
        return out_bf

    out_psum = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.psum)
    bcast_bufs = new_bcast_bufs()
    for first, last in groups:
        predict(first, last)
        update(first, last, bcast_bufs, out_psum)
    nisa.tensor_copy(dst=out_bf, src=out_psum, engine=nisa.engine.vector)
    return out_bf


def kda_verify_constants(ones_sq: nl.NkiTensor, plan) -> dict:
    """Return the ``consts`` of ``kda_verify_attention`` for a kernel: ``ones_sq``, a given fp32 ``[128, 128]`` all-ones
    tile of the kernel (for example ``decode_block.region_ones``), and a bf16 ``[128, 128]`` all-ones ``ones_rows`` tile
    from ``plan`` (``plan.tile(shape, dtype, name)``, a tile that lives to the kernel end), set here once. Every KDA verify
    call of the kernel then reads them in place of setting its own."""
    kernel_assert(
        tuple(ones_sq.shape) == (P_MAX, P_MAX) and ones_sq.dtype == nl.float32,
        "kda_verify_constants takes an fp32 [128, 128] all-ones tile",
    )
    ones_rows = plan.tile((P_MAX, P_MAX), nl.bfloat16, "kda_ones_rows")
    nisa.memset(dst=ones_rows, value=1.0, engine=nisa.engine.gpsimd)
    return {"ones_sq": ones_sq, "ones_rows": ones_rows}


def kda_flush_ring_keeps(record: list, sink: nl.NkiTensor, after=None) -> None:
    """Trace the keep-reads of every ring in ``record`` (``kda_verify_attention`` ``slot_ring_keep_defer``) into
    ``sink``, each read after ``after`` (None: no schedule edge) and after its tile's stores, and empty the record."""
    while record:
        ring, ring_last, name = record.pop(0)
        _keep_ring_until_stored(ring, ring_last, name, after, sink)


def _keep_ring_until_stored(ring, ring_last, name: str, after, sink=None):
    """``kda_verify_attention`` ``slot_ring_per_call``: one GpSimd add of one element of each ring tile into ``sink``,
    each after the stores that read the tile last (the read follows an in-place GpSimd write of the tile, which waits
    for the completion of those stores). The compiler ends a DMA source's live range at the trigger; after these reads a
    later writer of the tiles' bytes follows a read that waits for the stores. Every instruction follows ``after`` (the
    all-reduce) in the schedule, so none sits ahead of its trigger on GpSimd. ``sink``: a tile that the caller keeps
    live to the kernel end, so the compiler keeps the reads; None: a ``[1, 1]`` sink of the call, stored to a private
    HBM word (dead code for the compiler: the simulator only)."""
    own = sink is None
    if own:
        sink = nl.ndarray((1, 1), dtype=ring[0].dtype, buffer=nl.sbuf, name=f"{name}_sink")
        _schedule_after(nisa.memset(dst=sink, value=0.0, engine=nisa.engine.gpsimd), (after,))
    kernel_assert(sink.dtype == ring[0].dtype, f"the keep sink takes the ring dtype {ring[0].dtype}, got {sink.dtype}")
    acc = sink[0:1, 0:1]
    for r, tile in enumerate(ring):
        anchors = tuple(after) if isinstance(after, (list, tuple)) else ((after,) if after is not None else ())
        if ring_last[r]:
            # A runtime edge on a DMA breaks the released neuronx-cc 2.28 (tkg_weight_dma_gate.wait_for_completion). So one GpSimd
            # x * 1 writes one element of each partition and sequence of the tile in place. Each store reads one of
            # those bytes, so the compiler orders the write after each store (a WAR edge, a semaphore wait on the
            # DMA). The read below reads the written element (a RAW edge). The values do not change.
            view = tile[0:P_MAX, 0 : tile.shape[1], 0, 0:1]
            touch = nisa.tensor_scalar(
                dst=view, data=view, op0=nl.multiply, operand0=1.0, engine=nisa.engine.gpsimd, name=f"{name}{r}_touch"
            )
            _schedule_after(touch, anchors)
        read = nisa.tensor_tensor(
            dst=acc, data1=acc, data2=tile[0:1, 0, 0, 0:1], op=nl.add, engine=nisa.engine.gpsimd, name=f"{name}{r}"
        )
        _schedule_after(read, anchors)
    if own:
        word = nl.ndarray((1, 1), dtype=ring[0].dtype, buffer=nl.private_hbm, name=f"{name}_word")
        nisa.dma_copy(dst=word, src=sink, dge_mode=nisa.dge_mode.swdge, name=f"{name}_store")


class _KeepWarm:
    """The keep-warm matmuls of ``kda_verify_attention`` ``late_keep_warm``: zero-data bf16 matmuls (a ``[128, 128]``
    stationary and ``_KEEP_WARM_COLS`` moving columns of the call's zero tile) into one fp32 PSUM tile of their own, as
    one accumulation group. ``start(anchor)`` traces the warm-up matmuls, each after ``anchor`` (a runtime edge);
    ``hold(matmul)`` traces one more after ``matmul`` in the schedule and returns it (``_output_projection`` ``filler``
    puts the next tile's matmuls after it, so the hold stays between the two tile groups)."""

    def __init__(self, zero, warmups: int, name: str):
        self.zero, self.warmups, self.name, self.count = zero, warmups, name, 0
        self.bank = nl.ndarray((P_MAX, _KEEP_WARM_COLS), dtype=nl.float32, buffer=nl.psum, name=f"{name}_psum")

    def _matmul(self, after, kind):
        matmul = nisa.nc_matmul(
            dst=self.bank,
            stationary=self.zero[0:P_MAX, 0:P_MAX],
            moving=self.zero[0:P_MAX, 0:_KEEP_WARM_COLS],
            accumulate=self.count > 0,
            name=f"{self.name}_mm{self.count}",
        )
        self.count += 1
        if after is not None:
            handle = matmul if hasattr(matmul, "depends_on") else nisa.NkiInstruction(matmul)
            anchor = after if hasattr(after, "depends_on") else nisa.NkiInstruction(after)
            handle.depends_on(anchor, kind=kind)
        return matmul

    def start(self, anchor):
        for _ in range(self.warmups):
            self._matmul(anchor, nisa.dependency.runtime)

    def hold(self, matmul):
        return self._matmul(matmul, nisa.dependency.schedule)


def _token_masks(q_tokens: int) -> dict:
    """Return the fp32 0/1 masks ``[Q, Q]`` of the chunked chain, built from one iota of ``row - column``.

    ``lower``: row > column (strictly lower). ``upper``: row < column. ``upper_eq``: row <= column.
    """
    fp32 = nl.float32
    diff_i = nl.ndarray((q_tokens, q_tokens), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=diff_i, pattern=[[-1, q_tokens]], offset=0, channel_multiplier=1)
    diff = nl.ndarray((q_tokens, q_tokens), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=diff, src=diff_i)
    masks = {}
    for name, op in (("lower", nl.greater), ("upper", nl.less), ("upper_eq", nl.less_equal)):
        masks[name] = nl.ndarray((q_tokens, q_tokens), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=masks[name], data=diff, op0=op, operand0=0.0)
    return masks


def _chunked_chain(state_sb, v_rows, k_rows, q_rows, decay_rows, beta_rows, ident32, seq, out_rows, out_states):
    """Score the Q tokens of sequence ``seq`` (one head) at once with the chunk-parallel (WY) delta rule.

    The rows are token-major: row ``j * B + seq`` is token ``j``. ``v_rows`` (bf16), ``k_rows``, ``q_rows``,
    ``decay_rows`` and ``beta_rows`` (fp32) are ``[128, Q*B]``; ``q_rows`` holds the scaled query and
    ``beta_rows`` the same beta on every partition. ``state_sb`` ``[128, B, 1, 128]`` (bf16) is the state
    before token 0. The chain writes the bf16 output of token ``j`` to ``out_rows[:, j * B + seq]`` and the
    bf16 state after token ``j`` to ``out_states[j][:, seq, 0, :]``.

    With ``G[t]`` the running sum of the log decay and ``r = Q - 1``, every exponent stays in a bounded range:
    ``e4 = exp(G[t] - G[r])`` in ``[1, exp((Q-1)|lb|)]``, ``kf = exp(G[r] - G[s])`` and ``e1 = exp(G[t])`` in
    ``(0, 1]``. Then, for one head (rows = key or value channels, columns = tokens):

    * ``N[t, s] = beta_t (k_t e4_t) . (k_s kf_s)`` for ``s < t``, the strictly lower pair matrix.
    * ``rhs_t = beta_t (v_t - S0^T (k_t e1_t))``, and ``Delta = (I + N)^-1 rhs = (I - N)(I + N^2)(I + N^4) rhs``
      (exact for Q <= 8, because ``N**8 = 0``).
    * ``S_t = e4_t * (e1_r * S0 + sum_{s <= t} (k_s kf_s) Delta_s^T)`` (row scale ``e4_t``), so
      ``e4_t * e1_r = e1_t`` and ``e4_t * kf_s = exp(G[t] - G[s])``.
    * ``out_t = S_t^T q_t = (e1_r S0)^T (q_t e4_t) + sum_{s <= t} ((k_s kf_s) . (q_t e4_t)) Delta_s``.

    All the math is fp32 (fp32 matmul operands); each state rounds to bf16 once, at its store.
    """
    fp32 = nl.float32
    n_rows = k_rows.shape[1]
    n_seq = state_sb.shape[1]
    q_tokens = n_rows // n_seq
    last = q_tokens - 1
    rows = slice(seq, n_rows, n_seq)
    k_t = k_rows[0:P_MAX, rows]
    q_t = q_rows[0:P_MAX, rows]
    v_t = v_rows[0:P_MAX, rows]
    beta_t = beta_rows[0:P_MAX, rows]

    def tile(shape):
        return nl.ndarray(shape, dtype=fp32, buffer=nl.sbuf)

    def psum(shape):
        return nl.ndarray(shape, dtype=fp32, buffer=nl.psum)

    pq = (P_MAX, q_tokens)
    qq = (q_tokens, q_tokens)
    qd = (q_tokens, HEAD_DIM)
    masks = _token_masks(q_tokens)

    # Running log decay G and the three bounded exponentials.
    log_d = tile(pq)
    nisa.activation(dst=log_d, op=nl.log, data=decay_rows[0:P_MAX, rows])
    ones_pq = tile(pq)
    nisa.memset(dst=ones_pq, value=1.0)
    zero_col = tile((P_MAX, 1))
    nisa.memset(dst=zero_col, value=0.0)
    g_run = tile(pq)
    nisa.tensor_tensor_scan(dst=g_run, data0=ones_pq, data1=log_d, initial=zero_col, op0=nl.multiply, op1=nl.add)
    neg_g_last = tile((P_MAX, 1))
    nisa.tensor_scalar(dst=neg_g_last, data=g_run[0:P_MAX, last : last + 1], op0=nl.multiply, operand0=-1.0)
    e4 = tile(pq)
    nisa.activation(dst=e4, op=nl.exp, data=g_run, bias=neg_g_last)
    kf = tile(pq)
    nisa.activation(dst=kf, op=nl.exp, data=g_run, scale=-1.0, bias=g_run[0:P_MAX, last : last + 1])
    e1 = tile(pq)
    nisa.activation(dst=e1, op=nl.exp, data=g_run)

    # The scaled key, query and value columns.
    kh = tile(pq)
    nisa.tensor_tensor(dst=kh, data1=k_t, data2=e4, op=nl.multiply)
    nisa.tensor_tensor(dst=kh, data1=kh, data2=beta_t, op=nl.multiply)
    kc = tile(pq)
    nisa.tensor_tensor(dst=kc, data1=k_t, data2=kf, op=nl.multiply)
    kd_neg = tile(pq)
    nisa.tensor_tensor(dst=kd_neg, data1=k_t, data2=e1, op=nl.multiply)
    nisa.scalar_tensor_tensor(dst=kd_neg, data=kd_neg, op0=nl.multiply, operand0=-1.0, op1=nl.multiply, operand1=beta_t)
    qh = tile(pq)
    nisa.tensor_tensor(dst=qh, data1=q_t, data2=e4, op=nl.multiply)
    vb = tile(pq)
    nisa.tensor_tensor(dst=vb, data1=v_t, data2=beta_t, op=nl.multiply)
    s0 = tile((P_MAX, HEAD_DIM))
    nisa.tensor_copy(dst=s0, src=state_sb[0:P_MAX, seq, 0, 0:HEAD_DIM], engine=nisa.engine.gpsimd)
    s0r = tile((P_MAX, HEAD_DIM))
    nisa.tensor_scalar(dst=s0r, data=s0, op0=nl.multiply, operand0=e1[0:P_MAX, last : last + 1])

    # The pair matrix N, its transpose and the powers of the inverse.
    a_psum = psum(qq)
    nisa.nc_matmul(dst=a_psum, stationary=kh, moving=kc, accumulate=False)
    at_psum = psum(qq)
    nisa.nc_matmul(dst=at_psum, stationary=kc, moving=kh, accumulate=False)
    n_mat = tile(qq)
    nisa.tensor_tensor(dst=n_mat, data1=a_psum, data2=masks["lower"], op=nl.multiply)
    nt_mat = tile(qq)
    nisa.tensor_tensor(dst=nt_mat, data1=at_psum, data2=masks["upper"], op=nl.multiply)
    nt_neg = tile(qq)
    nisa.tensor_scalar(dst=nt_neg, data=nt_mat, op0=nl.multiply, operand0=-1.0)
    n2_psum = psum(qq)
    nisa.nc_matmul(dst=n2_psum, stationary=nt_mat, moving=n_mat, accumulate=False)
    n2t_psum = psum(qq)
    nisa.nc_matmul(dst=n2t_psum, stationary=n_mat, moving=nt_mat, accumulate=False)
    n2_mat = tile(qq)
    nisa.tensor_copy(dst=n2_mat, src=n2_psum, engine=nisa.engine.vector)
    n2t_mat = tile(qq)
    nisa.tensor_copy(dst=n2t_mat, src=n2t_psum, engine=nisa.engine.scalar)
    n4t_psum = psum(qq)
    nisa.nc_matmul(dst=n4t_psum, stationary=n2_mat, moving=n2t_mat, accumulate=False)
    n4t_mat = tile(qq)
    nisa.tensor_copy(dst=n4t_mat, src=n4t_psum, engine=nisa.engine.vector)

    # rhs = beta (v - S0^T (k e1)), token-major [Q, 128].
    rhs_psum = psum(qd)
    nisa.nc_matmul(dst=rhs_psum, stationary=vb, moving=ident32, accumulate=False)
    nisa.nc_matmul(dst=rhs_psum, stationary=kd_neg, moving=s0, accumulate=True)
    rhs = tile(qd)
    nisa.tensor_copy(dst=rhs, src=rhs_psum, engine=nisa.engine.vector)

    # Delta = (I - N)(I + N^2)(I + N^4) rhs.
    delta = rhs
    for factor in (n4t_mat, n2t_mat, nt_neg):
        step_psum = psum(qd)
        nisa.nc_matmul(dst=step_psum, stationary=factor, moving=delta, accumulate=False)
        nxt = tile(qd)
        nisa.tensor_tensor(dst=nxt, data1=step_psum, data2=delta, op=nl.add)
        delta = nxt

    # Outputs: out^T [128 v, Q] = (e1_r S0)^T qh + Delta^T (B^T masked to s <= t).
    bt_psum = psum(qq)
    nisa.nc_matmul(dst=bt_psum, stationary=kc, moving=qh, accumulate=False)
    bt_mat = tile(qq)
    nisa.tensor_tensor(dst=bt_mat, data1=bt_psum, data2=masks["upper_eq"], op=nl.multiply)
    out_psum = psum(pq)
    nisa.nc_matmul(dst=out_psum, stationary=s0r, moving=qh, accumulate=False)
    nisa.nc_matmul(dst=out_psum, stationary=delta, moving=bt_mat, accumulate=True)
    nisa.tensor_copy(dst=out_rows[0:P_MAX, rows], src=out_psum, engine=nisa.engine.vector)

    # States: S_t = e4_t * P_t + e1_t * S0 with P_t = kc[:, :t+1] Delta[:t+1], each rounded to bf16 once. The prefix sums
    # of all tokens run as matmuls of up to 512 columns on a token-masked copy of Delta, with one stationary.
    kct_psum = psum((q_tokens, P_MAX))
    nisa.nc_matmul(dst=kct_psum, stationary=kc, moving=ident32, accumulate=False)
    kct = tile((q_tokens, P_MAX))
    nisa.tensor_copy(dst=kct, src=kct_psum, engine=nisa.engine.scalar)
    s0_decayed = []
    for t in range(q_tokens):
        s0_t = tile((P_MAX, HEAD_DIM))
        nisa.tensor_scalar(
            dst=s0_t,
            data=s0,
            op0=nl.multiply,
            operand0=e1[0:P_MAX, t : t + 1],
            engine=nisa.engine.vector if t % 2 else nisa.engine.scalar,
        )
        s0_decayed.append(s0_t)
    delta_masked = tile((q_tokens, q_tokens, HEAD_DIM))
    nisa.tensor_tensor(
        dst=delta_masked,
        data1=delta.expand_dim(1).broadcast(1, q_tokens),
        data2=masks["upper_eq"].expand_dim(2).broadcast(2, HEAD_DIM),
        op=nl.multiply,
    )
    per_matmul = _PSUM_BANK_FP32 // HEAD_DIM
    for t0 in range(0, q_tokens, per_matmul):
        t1 = min(q_tokens, t0 + per_matmul)
        prefix_psum = psum((P_MAX, (t1 - t0) * HEAD_DIM))
        nisa.nc_matmul(
            dst=prefix_psum,
            stationary=kct,
            moving=delta_masked[0:q_tokens, t0:t1, 0:HEAD_DIM],
            accumulate=False,
        )
        for t in range(t0, t1):
            nisa.scalar_tensor_tensor(
                dst=out_states[t][0:P_MAX, seq, 0, 0:HEAD_DIM],
                data=prefix_psum[0:P_MAX, (t - t0) * HEAD_DIM : (t - t0 + 1) * HEAD_DIM],
                op0=nl.multiply,
                operand0=e4[0:P_MAX, t : t + 1],
                op1=nl.add,
                operand1=s0_decayed[t],
            )


def _batched_masks(tokens: int) -> dict:
    """Return the fp32 0/1 masks of the batched chain for a group of ``P_MAX // tokens`` pairs of ``tokens`` tokens.

    Rows and columns index (pair, token) pair-major, ``pair * tokens + token``, within the group.

    * ``lower`` ``[n, n]``: the same pair and column < row. ``upper``: the same pair and row < column.
      ``upper_eq``: the same pair and row <= column.
    * ``upto[t]`` ``[n, G]``: row (pair, s) and column pair' are the same pair and s <= t.

    The masks of a smaller last group are the top-left blocks of these.
    """
    group = P_MAX // tokens
    n = group * tokens
    fp32 = nl.float32
    vector = nisa.engine.vector

    def iota(pattern, cols):
        raw = nl.ndarray((n, cols), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=raw, pattern=pattern, offset=0, channel_multiplier=1)
        values = nl.ndarray((n, cols), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=values, src=raw, engine=vector)
        return values

    def same_pair(offset_rows, cols):
        # offset_rows = row - tokens * pair(column): the same pair when it is in [0, tokens).
        lower_bound = nl.ndarray((n, cols), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lower_bound, data=offset_rows, op0=nl.greater_equal, operand0=0.0, engine=vector)
        return lower_bound

    row_minus_col = iota([[-1, n]], n)
    row_offset = iota([[-tokens, group], [0, tokens]], n)
    row_offset_pair = iota([[-tokens, group]], group)
    same = nl.ndarray((n, n), dtype=fp32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=same,
        data=row_offset,
        op0=nl.less_equal,
        operand0=float(tokens - 1),
        op1=nl.multiply,
        operand1=same_pair(row_offset, n),
    )
    masks = {"tokens": tokens}
    for name, op in (("lower", nl.greater), ("upper", nl.less), ("upper_eq", nl.less_equal)):
        masks[name] = nl.ndarray((n, n), dtype=fp32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=masks[name], data=row_minus_col, op0=op, operand0=0.0, op1=nl.multiply, operand1=same
        )
        # The select predicate of the same mask: the solve selects, not multiplies, so a non-finite gram entry of
        # another pair stays out (0 * NaN is NaN).
        masks[f"{name}_pred"] = nl.ndarray((n, n), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=masks[f"{name}_pred"], src=masks[name], engine=vector)
    pair_start = same_pair(row_offset_pair, group)
    masks["upto"] = []
    for token in range(tokens):
        upto = nl.ndarray((n, group), dtype=fp32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=upto,
            data=row_offset_pair,
            op0=nl.less_equal,
            operand0=float(token),
            op1=nl.multiply,
            operand1=pair_start,
        )
        masks["upto"].append(upto)
    return masks


def _qk_norms(conv_out: nl.NkiTensor, ones_sq: nl.NkiTensor):
    """Return the L2-normed k and the scaled, normed q ``[128, R, Hl]`` (fp32) of the conv outputs ``[128, R, 3*Hl]``.

    The ops are those of ``_readout_vectors`` (the norms only).
    """
    n_rows = conv_out.shape[1]
    heads = conv_out.shape[2] // N_CONV_PROJ
    fp32 = nl.float32
    qk_sq = nl.ndarray((P_MAX, n_rows, 2 * heads), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=qk_sq,
        data1=conv_out[0:P_MAX, 0:n_rows, 0 : 2 * heads],
        data2=conv_out[0:P_MAX, 0:n_rows, 0 : 2 * heads],
        op=nl.multiply,
        engine=nisa.engine.vector,
    )
    qk_ss = nl.ndarray((P_MAX, n_rows, 2 * heads), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(dst=qk_ss, stationary=ones_sq, moving=qk_sq, is_stationary_onezero=True)
    qk_log = nl.ndarray((P_MAX, n_rows, 2 * heads), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=qk_log, op=nl.log, data=qk_ss, bias=L2_EPS, engine=nisa.engine.scalar)
    qk_rnorm = nl.ndarray((P_MAX, n_rows, 2 * heads), dtype=fp32, buffer=nl.sbuf)
    nisa.activation(dst=qk_rnorm, op=nl.exp, data=qk_log, scale=-0.5, engine=nisa.engine.scalar)
    q_n = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=q_n,
        data=conv_out[0:P_MAX, 0:n_rows, 0:heads],
        op0=nl.multiply,
        operand0=float(HEAD_DIM) ** -0.5,
        op1=nl.multiply,
        operand1=qk_rnorm[0:P_MAX, 0:n_rows, 0:heads],
    )
    k_n = nl.ndarray((P_MAX, n_rows, heads), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=k_n,
        data1=conv_out[0:P_MAX, 0:n_rows, heads : 2 * heads],
        data2=qk_rnorm[0:P_MAX, 0:n_rows, heads : 2 * heads],
        op=nl.multiply,
        engine=nisa.engine.vector,
    )
    return k_n, q_n


def _pair_major(rows: nl.NkiTensor, n_seq: int, engine) -> nl.NkiTensor:
    """Return the token-major rows ``[128, Q*B, Hl]`` (row ``j * B + b``) as an fp32 pair-major tile ``[128, B*Hl*Q]``.

    Column ``(b * Hl + h) * Q + j`` holds token ``j`` of sequence ``b`` and head ``h``.
    """
    n_rows, heads = rows.shape[1], rows.shape[2]
    tokens = n_rows // n_seq
    out = nl.ndarray((P_MAX, n_seq * heads * tokens), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=out.reshape_dim(1, (n_seq, heads, tokens)),
        src=rows.reshape_dim(1, (tokens, n_seq)).permute((0, 2, 3, 1)),
        engine=engine,
    )
    return out


def _check_engine(engine):
    """Return the engine of a compare or min / max ``tensor_scalar`` of the finite checks: ``engine``, but the VectorE in
    place of the GpSimd on Trn2, whose GpSimd has no such ``tensor_scalar`` (ISA ``opcode_engine_check``)."""
    return nisa.engine.vector if engine == nisa.engine.gpsimd and not is_trn3() else engine


def _finite_flags(x: nl.NkiTensor, engine, bound: float = None):
    """Return ``(flag, pred)``: ``flag`` fp32 1.0 where ``x`` is finite, else 0.0 (``x * 0 == 0`` is false for NaN and
    Inf), and ``pred``, the same as a uint16 select predicate. With ``bound``, ``flag`` is 1.0 only where also
    ``|x| < bound`` (``x * x < bound**2``, false for NaN, Inf and overflow). ``x`` has at most two free dimensions."""
    engine = _check_engine(engine)
    flag = nl.ndarray(tuple(x.shape), dtype=nl.float32, buffer=nl.sbuf)
    if bound is None:
        nisa.tensor_scalar(dst=flag, data=x, op0=nl.multiply, operand0=0.0, op1=nl.equal, operand1=0.0, engine=engine)
    else:
        nisa.tensor_tensor(dst=flag, data1=x, data2=x, op=nl.multiply, engine=engine)
        nisa.tensor_scalar(dst=flag, data=flag, op0=nl.less, operand0=bound * bound, engine=engine)
    pred = nl.ndarray(tuple(x.shape), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pred, src=flag, engine=engine)
    return flag, pred


def _sanitized(x: nl.NkiTensor, low: float, high: float, safe: float, engine):
    """Return ``(x', flag)``: ``x`` clamped to ``[low, high]`` with ``safe`` in place of each non-finite element, and the
    fp32 finite flag of ``x`` (``_finite_flags``). A finite ``x`` inside the bounds passes bit for bit."""
    flag, pred = _finite_flags(x, engine)
    clamped = nl.ndarray(tuple(x.shape), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=clamped, data=x, op0=nl.maximum, operand0=low, op1=nl.minimum, operand1=high, engine=_check_engine(engine)
    )
    out = nl.ndarray(tuple(x.shape), dtype=nl.float32, buffer=nl.sbuf)
    nisa.select_reduce(dst=out, predicate=pred, on_true=clamped, on_false=safe)
    return out, flag


def _store_finite(x: nl.NkiTensor, engine, safe: float = 0.0) -> None:
    """Put ``safe`` in place of each element of ``x`` (a store source) that is not finite or not below
    ``_BATCHED_STORE_BOUND`` in magnitude, in place. Other elements keep their bits, so a row that went non-finite stores
    safe values instead of poisoning its page for the next user."""
    _, pred = _finite_flags(x, engine, _BATCHED_STORE_BOUND)
    nisa.select_reduce(dst=x, predicate=pred, on_true=x, on_false=safe)


def _batched_solve(kk, qq, vv, dd, bb, s0, ident32, masks, out_pairs=None, state_dsts=(), ref=None, state_waits=None):
    """Solve the delta rule of ``tokens`` tokens for every (sequence, head) pair at once in the chunk-parallel (WY) form.

    All tiles are fp32 and pair-major: ``kk`` (the normed keys), ``qq`` (the scaled queries, or None for no
    outputs), ``vv`` (values), ``dd`` (the per-channel decay) and ``bb`` (beta, the same on every partition) are
    ``[128, P*T]`` with column ``pair * T + token``; the key or value channel is the partition. ``s0`` ``[128, P, 128]``
    is the state before token 0 of each pair (key rows, value columns), pair ``b * heads + h``.

    For each pair, with ``G`` the running sum of the log decay and the reference token ``r`` (``ref``, default
    ``T - 1``; ``_chunked_chain`` has the derivation): ``e4 = exp(G_t - G_r)``, ``kf = exp(G_r - G_s)``, ``e1 = exp(G_t)``;
    ``N[t, s] = beta_t (k_t e4_t) . (k_s kf_s)`` for ``s < t``; ``Delta = (I + N)^-1 beta (v - S0^T (k e1))``;
    ``out_t = S0^T (q_t e1_t) + sum_{s <= t} ((k_s kf_s) . (q_t e4_t)) Delta_s``; and the state after token ``t`` is
    ``S_t = e4_t * sum_{s <= t} (k_s kf_s) Delta_s^T + e1_t * S0``.

    The pairs run in groups of ``P_MAX // T``: one matmul forms the grams of a group (block-diagonal, masked), and
    the inverse ``(I + N)^-1 = (I - N)(I + N^2)(I + N^4)(I + N^8)`` (up to the power ``N^(2^L) = 0`` with
    ``2^L >= T``, ``T <= 16``) is a few matmuls of the group. A token with beta 0 and decay 1 is an identity step. All the math is fp32 (fp32 matmul operands).

    ``out_pairs`` (``[128, P*T]``, value rows) takes the outputs. ``state_dsts`` lists ``(token, tile)``: the state
    after ``token`` goes to ``tile[:, b, h, :]`` (``[128, B, heads, 128]``), rounded to its dtype once.
    ``state_waits`` (None or a list by token) holds the DMAs that read a tile last: each write into it waits for them.
    """
    fp32 = nl.float32
    vector, scalar, gpsimd = nisa.engine.vector, nisa.engine.scalar, nisa.engine.gpsimd
    tokens = masks["tokens"]
    width = kk.shape[1]
    n_pairs = width // tokens
    group = P_MAX // tokens
    want_out = qq is not None
    ref = tokens - 1 if ref is None else ref
    kernel_assert(tokens <= 2 * BATCHED_MAX_QUERIES, f"the batched solve takes at most 16 tokens, got {tokens}")

    def tile(shape):
        return nl.ndarray(shape, dtype=fp32, buffer=nl.sbuf)

    def psum(shape):
        return nl.ndarray(shape, dtype=fp32, buffer=nl.psum)

    def by_pair(x):
        return x.reshape_dim(1, (n_pairs, tokens))

    # Row isolation. The group matmuls below sum over all pairs of a group, and the decay scan runs across pairs, so one
    # non-finite input (a pad row, a stale record) would reach every pair (0 * NaN is NaN). Each input gets safe values in
    # place of its non-finite elements and bounds that the real inputs never reach (they pass bit for bit), so every value
    # of the solve stays finite. good[(pair, t)] is 1 where the pair's inputs of tokens 0..t are finite on all partitions.
    unit, value = _BATCHED_UNIT_BOUND, _BATCHED_VALUE_BOUND
    kk, good = _sanitized(kk, -unit, unit, 0.0, vector)
    vv, flag = _sanitized(vv, -value, value, 0.0, gpsimd)
    nisa.tensor_tensor(dst=good, data1=good, data2=flag, op=nl.multiply, engine=vector)
    dd, flag = _sanitized(dd, _BATCHED_DECAY_FLOOR, 1.0, 1.0, vector)
    nisa.tensor_tensor(dst=good, data1=good, data2=flag, op=nl.multiply, engine=vector)
    bb, flag = _sanitized(bb, 0.0, 1.0, 0.0, gpsimd)
    nisa.tensor_tensor(dst=good, data1=good, data2=flag, op=nl.multiply, engine=vector)
    if want_out:
        qq, flag = _sanitized(qq, -unit, unit, 0.0, gpsimd)
        nisa.tensor_tensor(dst=good, data1=good, data2=flag, op=nl.multiply, engine=vector)
    good_pairs = by_pair(good)
    for token in range(1, tokens):
        nisa.tensor_tensor(
            dst=good_pairs[0:P_MAX, 0:n_pairs, token],
            data1=good_pairs[0:P_MAX, 0:n_pairs, token],
            data2=good_pairs[0:P_MAX, 0:n_pairs, token - 1],
            op=nl.multiply,
            engine=vector,
        )
    ones = tile((P_MAX, P_MAX))
    nisa.memset(dst=ones, value=1.0, engine=gpsimd)
    good_ps = psum((P_MAX, width))
    nisa.nc_matmul(dst=good_ps, stationary=ones, moving=good, is_stationary_onezero=True, accumulate=False)
    good_all = tile((P_MAX, width))
    nisa.tensor_scalar(dst=good_all, data=good_ps, op0=nl.equal, operand0=float(P_MAX), engine=vector)
    good_pred = nl.ndarray((P_MAX, width), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=good_pred, src=good_all, engine=vector)

    # Running log decay G (the scan restarts at token 0 of each pair) and the bounded exponentials.
    log_d = tile((P_MAX, width))
    nisa.activation(dst=log_d, op=nl.log, data=dd, engine=scalar)
    carry = tile((P_MAX, width))
    nisa.memset(dst=carry, value=1.0, engine=gpsimd)
    nisa.memset(dst=by_pair(carry)[0:P_MAX, 0:n_pairs, 0], value=0.0, engine=gpsimd)
    g_run = tile((P_MAX, width))
    nisa.tensor_tensor_scan(dst=g_run, data0=carry, data1=log_d, initial=0.0, op0=nl.multiply, op1=nl.add)
    g_pairs = by_pair(g_run)
    rel = tile((P_MAX, width))
    nisa.tensor_tensor(
        dst=by_pair(rel),
        data1=g_pairs,
        data2=g_pairs[0:P_MAX, 0:n_pairs, ref].expand_dim(2).broadcast(2, tokens),
        op=nl.subtract,
        engine=vector,
    )
    e4 = tile((P_MAX, width))
    nisa.activation(dst=e4, op=nl.exp, data=rel, engine=scalar)
    kf = tile((P_MAX, width))
    nisa.activation(dst=kf, op=nl.exp, data=rel, scale=-1.0, engine=scalar)
    e1 = tile((P_MAX, width))
    nisa.activation(dst=e1, op=nl.exp, data=g_run, engine=scalar)

    # The scaled key, query and value columns, and the moving columns of the S0 reads (k e1, then q e1).
    kh = tile((P_MAX, width))
    nisa.tensor_tensor(dst=kh, data1=kk, data2=e4, op=nl.multiply, engine=vector)
    nisa.tensor_tensor(dst=kh, data1=kh, data2=bb, op=nl.multiply, engine=vector)
    kc = tile((P_MAX, width))
    nisa.tensor_tensor(dst=kc, data1=kk, data2=kf, op=nl.multiply, engine=gpsimd)
    vb = tile((P_MAX, width))
    nisa.tensor_tensor(dst=vb, data1=vv, data2=bb, op=nl.multiply, engine=gpsimd)
    n_mv = 2 * tokens if want_out else tokens
    mv = tile((P_MAX, n_pairs, n_mv))
    nisa.tensor_tensor(
        dst=mv[0:P_MAX, 0:n_pairs, 0:tokens], data1=by_pair(kk), data2=by_pair(e1), op=nl.multiply, engine=vector
    )
    if want_out:
        nisa.tensor_tensor(
            dst=mv[0:P_MAX, 0:n_pairs, tokens:n_mv],
            data1=by_pair(qq),
            data2=by_pair(e1),
            op=nl.multiply,
            engine=vector,
        )
        qh = tile((P_MAX, width))
        nisa.tensor_tensor(dst=qh, data1=qq, data2=e4, op=nl.multiply, engine=gpsimd)

    for p0 in range(0, n_pairs, group):
        p1 = min(n_pairs, p0 + group)
        pairs = p1 - p0
        n = pairs * tokens
        cols = slice(p0 * tokens, p1 * tokens)
        eye = ident32[0:n, 0:n]

        # The block-diagonal pair grams: N (strictly lower) and its transpose.
        n_ps = psum((n, n))
        nisa.nc_matmul(dst=n_ps, stationary=kh[0:P_MAX, cols], moving=kc[0:P_MAX, cols], accumulate=False)
        # The masks select (not multiply), so an entry of another pair stays out even if it is not finite.
        n_mat = tile((n, n))
        nisa.select_reduce(dst=n_mat, predicate=masks["lower_pred"][0:n, 0:n], on_true=n_ps, on_false=0.0)
        nt_ps = psum((n, n))
        nisa.nc_matmul(dst=nt_ps, stationary=kc[0:P_MAX, cols], moving=kh[0:P_MAX, cols], accumulate=False)
        nt_mat = tile((n, n))
        nisa.select_reduce(dst=nt_mat, predicate=masks["upper_pred"][0:n, 0:n], on_true=nt_ps, on_false=0.0)

        # M^T = ((I + N)^-1)^T = (I - N^T)(I + N^T^2)(I + N^T^4)...; matmul(stationary=X, moving=Y) = X^T Y. Each
        # level squares N and N^T and multiplies the product Y (and, before the last level, Y^T) by I + N^T^(2^l).
        levels = max(1, (tokens - 1).bit_length())
        y_mat = tile((n, n))
        nisa.tensor_tensor(dst=y_mat, data1=eye, data2=nt_mat, op=nl.subtract, engine=gpsimd)
        y_t = tile((n, n))
        nisa.tensor_tensor(dst=y_t, data1=eye, data2=n_mat, op=nl.subtract, engine=gpsimd)
        power, power_t = n_mat, nt_mat
        mt_ps = None
        for level in range(1, levels):
            last = level == levels - 1
            pt_ps = psum((n, n))
            nisa.nc_matmul(dst=pt_ps, stationary=power, moving=power_t, accumulate=False)
            factor = tile((n, n))
            nisa.tensor_tensor(dst=factor, data1=pt_ps, data2=eye, op=nl.add, engine=vector)
            if not last:
                p_ps = psum((n, n))
                nisa.nc_matmul(dst=p_ps, stationary=power_t, moving=power, accumulate=False)
                next_power = tile((n, n))
                nisa.tensor_copy(dst=next_power, src=p_ps, engine=scalar)
                next_power_t = tile((n, n))
                nisa.tensor_copy(dst=next_power_t, src=pt_ps, engine=scalar)
            y_ps = psum((n, n))
            nisa.nc_matmul(dst=y_ps, stationary=y_t, moving=factor, accumulate=False)
            if last:
                mt_ps = y_ps
                break
            yt_ps = psum((n, n))
            nisa.nc_matmul(dst=yt_ps, stationary=factor, moving=y_t, accumulate=False)
            y_mat = tile((n, n))
            nisa.tensor_copy(dst=y_mat, src=y_ps, engine=scalar)
            y_t = tile((n, n))
            nisa.tensor_copy(dst=y_t, src=yt_ps, engine=vector)
            power, power_t = next_power, next_power_t
        if mt_ps is None:
            mt = y_mat
        else:
            mt = tile((n, n))
            nisa.tensor_copy(dst=mt, src=mt_ps, engine=scalar)

        # rhs^T [v, (pair, t)] = beta (v - S0^T (k e1)): one S0 read per pair (the q e1 columns give the output part).
        sv_ps = psum((P_MAX, pairs, n_mv))
        for i in range(pairs):
            nisa.nc_matmul(
                dst=sv_ps[0:P_MAX, i, 0:n_mv],
                stationary=s0[0:P_MAX, p0 + i, 0:HEAD_DIM],
                moving=mv[0:P_MAX, p0 + i, 0:n_mv],
                accumulate=False,
            )
        rhs_t = tile((P_MAX, n))
        nisa.tensor_tensor(
            dst=rhs_t.reshape_dim(1, (pairs, tokens)),
            data1=sv_ps[0:P_MAX, 0:pairs, 0:tokens],
            data2=by_pair(bb)[0:P_MAX, p0:p1, 0:tokens],
            op=nl.multiply,
            engine=vector,
        )
        nisa.tensor_tensor(dst=rhs_t, data1=vb[0:P_MAX, cols], data2=rhs_t, op=nl.subtract, engine=vector)
        # A non-finite state S0 of a pair (a page read ON) makes its rhs non-finite; the Delta matmul below sums over the
        # pairs of the group, so the rhs gets the safe value there. That pair's outputs and state still go non-finite
        # through its own S0 terms, as the sequential chain gives them.
        rhs_t, _ = _sanitized(rhs_t, -_BATCHED_RHS_BOUND, _BATCHED_RHS_BOUND, 0.0, vector)
        if want_out:
            s0_out = tile((P_MAX, pairs, tokens))
            nisa.tensor_copy(dst=s0_out, src=sv_ps[0:P_MAX, 0:pairs, tokens:n_mv], engine=scalar)
        rt_ps = psum((n, HEAD_DIM))
        nisa.nc_matmul(dst=rt_ps, stationary=rhs_t, moving=ident32, is_moving_onezero=True, accumulate=False)
        rhs_tok = tile((n, HEAD_DIM))
        nisa.tensor_copy(dst=rhs_tok, src=rt_ps, engine=scalar)
        delta_ps = psum((n, HEAD_DIM))
        nisa.nc_matmul(dst=delta_ps, stationary=mt, moving=rhs_tok, accumulate=False)
        delta, _ = _sanitized(delta_ps, -_BATCHED_DELTA_BOUND, _BATCHED_DELTA_BOUND, 0.0, vector)

        if want_out:
            # out^T [v, (pair, t)] = Delta^T B^T (B[t, s] = (k_s kf_s) . (q_t e4_t), s <= t) + S0^T (q e1).
            bq_ps = psum((n, n))
            nisa.nc_matmul(dst=bq_ps, stationary=kc[0:P_MAX, cols], moving=qh[0:P_MAX, cols], accumulate=False)
            bq = tile((n, n))
            nisa.select_reduce(dst=bq, predicate=masks["upper_eq_pred"][0:n, 0:n], on_true=bq_ps, on_false=0.0)
            out_ps = psum((P_MAX, n))
            nisa.nc_matmul(dst=out_ps, stationary=delta, moving=bq, accumulate=False)
            nisa.tensor_tensor(
                dst=by_pair(out_pairs)[0:P_MAX, p0:p1, 0:tokens],
                data1=out_ps.reshape_dim(1, (pairs, tokens)),
                data2=s0_out,
                op=nl.add,
                engine=vector,
            )

        if state_dsts:
            # S_t [k, v] = e4_t * sum_{s <= t} (k_s kf_s) Delta_s^T + e1_t * S0: the token-masked Delta of each pair is
            # the moving operand of one matmul per 4 pairs, with the transposed keys of the group as the stationary.
            kct_ps = psum((n, P_MAX))
            nisa.nc_matmul(
                dst=kct_ps, stationary=kc[0:P_MAX, cols], moving=ident32, is_moving_onezero=True, accumulate=False
            )
            kct = tile((n, P_MAX))
            nisa.tensor_copy(dst=kct, src=kct_ps, engine=scalar)
            per_matmul = _PSUM_BANK_FP32 // HEAD_DIM
            for token, dst_tile in state_dsts:
                delta_masked = tile((n, pairs, HEAD_DIM))
                nisa.tensor_tensor(
                    dst=delta_masked,
                    data1=delta.expand_dim(1).broadcast(1, pairs),
                    data2=masks["upto"][token][0:n, 0:pairs].expand_dim(2).broadcast(2, HEAD_DIM),
                    op=nl.multiply,
                    engine=gpsimd,
                )
                # e1_t * S0 of the group's pairs in one op: the decay of each key row and pair, broadcast over v.
                s0_decayed = tile((P_MAX, pairs, HEAD_DIM))
                nisa.tensor_tensor(
                    dst=s0_decayed,
                    data1=s0[0:P_MAX, p0:p1, 0:HEAD_DIM],
                    data2=by_pair(e1)[0:P_MAX, p0:p1, token].expand_dim(2).broadcast(2, HEAD_DIM),
                    op=nl.multiply,
                    engine=gpsimd,
                )
                dst_pairs = dst_tile.flatten_dims(1, 2)
                for c0 in range(0, pairs, per_matmul):
                    c1 = min(pairs, c0 + per_matmul)
                    st_ps = psum((P_MAX, (c1 - c0) * HEAD_DIM))
                    nisa.nc_matmul(
                        dst=st_ps, stationary=kct, moving=delta_masked[0:n, c0:c1, 0:HEAD_DIM], accumulate=False
                    )
                    sums = st_ps.reshape_dim(1, (c1 - c0, HEAD_DIM))
                    dst = dst_pairs[0:P_MAX, p0 + c0 : p0 + c1, 0:HEAD_DIM]
                    if token == ref:
                        # e4 of the reference token is exp(0) = 1.
                        write = nisa.tensor_tensor(
                            dst=dst, data1=sums, data2=s0_decayed[0:P_MAX, c0:c1, 0:HEAD_DIM], op=nl.add, engine=vector
                        )
                    else:
                        scaled = tile((P_MAX, c1 - c0, HEAD_DIM))
                        nisa.tensor_tensor(
                            dst=scaled,
                            data1=sums,
                            data2=by_pair(e4)[0:P_MAX, p0 + c0 : p0 + c1, token].expand_dim(2).broadcast(2, HEAD_DIM),
                            op=nl.multiply,
                            engine=vector,
                        )
                        write = nisa.tensor_tensor(
                            dst=dst,
                            data1=scaled,
                            data2=s0_decayed[0:P_MAX, c0:c1, 0:HEAD_DIM],
                            op=nl.add,
                            engine=gpsimd,
                        )
                    waits = state_waits[token] if state_waits else None
                    if waits and hasattr(write, "depends_on"):
                        write.depends_on(*waits)

    # A pair whose inputs of tokens 0..t were not all finite gets a NaN output at token t, as the sequential chain gives
    # it (the caller then fails that request alone), and a zero state, so it writes no NaN to a page that a later request reuses.
    if want_out:
        nisa.tensor_copy_predicated(dst=out_pairs, src=float("nan"), predicate=good_pred, reverse_pred=True)
    for token, dst_tile in state_dsts:
        nisa.tensor_copy_predicated(
            dst=dst_tile.flatten_dims(1, 2),
            src=0.0,
            predicate=by_pair(good_pred)[0:P_MAX, 0:n_pairs, token].expand_dim(2).broadcast(2, HEAD_DIM),
            reverse_pred=True,
        )


def _state_fp32(state_sb: nl.NkiTensor) -> nl.NkiTensor:
    """Return the state ``[128, B, Hl, 128]`` as an fp32 tile ``[128, B*Hl, 128]`` (pair ``b * Hl + h``)."""
    n_seq, heads = state_sb.shape[1], state_sb.shape[2]
    s0 = nl.ndarray((P_MAX, n_seq * heads, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s0, src=state_sb.flatten_dims(1, 2), engine=nisa.engine.gpsimd)
    return s0


def _batched_records(state_sb, conv_sb, replay_pred, cm_rec, decay_rec, beta_rec, conv_w, conv_dt, ones_sq):
    """Return the chain inputs of the Q records of every sequence and head and commit the conv window (``batched``).

    ``state_sb`` ``[128, B, Hl, 128]`` and ``conv_sb`` ``[128, B, 3*Hl, 3]`` hold the committed state and window
    before the records; ``conv_sb`` takes the window after ``replay_len`` records in place. A record ``j >=
    replay_len`` (or of an off record page) gets zero projection values, decay 1 and beta 0 by a predicated copy, so
    it is an identity step and a NaN in its page cannot reach the state.

    Returns:
        dict: fp32 pair-major ``[128, B*Hl*Q]`` tiles ``k``, ``v``, ``decay`` and ``beta`` of the records, and ``s0``
        ``[128, B*Hl, 128]``, the fp32 state before the records.
    """
    n_seq, n_conv = conv_sb.shape[1], conv_sb.shape[2]
    heads = state_sb.shape[2]
    q_tokens = cm_rec.shape[2]
    n_rows = q_tokens * n_seq
    taps = CONV_TAPS - 1
    fp32, bf16 = nl.float32, nl.bfloat16
    vector, gpsimd = nisa.engine.vector, nisa.engine.gpsimd

    # The complement of the replay predicate: 1 where record j is beyond replay_len or the record page is off.
    pred_f = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pred_f, src=replay_pred, engine=vector)
    off_f = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=off_f, data=pred_f, op0=nl.equal, operand0=0.0, engine=vector)
    off = nl.ndarray((P_MAX, q_tokens, n_seq), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=off, src=off_f, engine=vector)

    cm_rows = nl.ndarray((P_MAX, n_rows, n_conv), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cm_rows.reshape_dim(1, (q_tokens, n_seq)), src=cm_rec.permute((0, 2, 1, 3)), engine=gpsimd)
    width = n_seq * heads * q_tokens
    dd = nl.ndarray((P_MAX, width), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=dd.reshape_dim(1, (n_seq, heads, q_tokens)), src=decay_rec.permute((0, 1, 3, 2)), engine=gpsimd
    )
    bb = nl.ndarray((P_MAX, width), dtype=fp32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bb.reshape_dim(1, (n_seq, heads, q_tokens)), src=beta_rec.permute((0, 1, 3, 2)), engine=gpsimd)
    zero_cm = nl.ndarray((P_MAX, n_seq, n_conv), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(dst=zero_cm, value=0.0, engine=gpsimd)
    one_gate = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.sbuf)
    nisa.memset(dst=one_gate, value=1.0, engine=gpsimd)
    zero_gate = nl.ndarray((P_MAX, n_seq, heads), dtype=fp32, buffer=nl.sbuf)
    nisa.memset(dst=zero_gate, value=0.0, engine=gpsimd)
    dd4 = dd.reshape_dim(1, (n_seq, heads, q_tokens))
    bb4 = bb.reshape_dim(1, (n_seq, heads, q_tokens))
    for j in range(q_tokens):
        token_off = off[0:P_MAX, j, 0:n_seq]
        nisa.tensor_copy_predicated(
            dst=cm_rows[0:P_MAX, j * n_seq : (j + 1) * n_seq, 0:n_conv],
            src=zero_cm,
            predicate=_per_sequence(token_off, (n_conv,)),
        )
        nisa.tensor_copy_predicated(
            dst=dd4[0:P_MAX, 0:n_seq, 0:heads, j], src=one_gate, predicate=_per_sequence(token_off, (heads,))
        )
        nisa.tensor_copy_predicated(
            dst=bb4[0:P_MAX, 0:n_seq, 0:heads, j], src=zero_gate, predicate=_per_sequence(token_off, (heads,))
        )

    # The conv, the norms and the values of all records in one pass, from the committed window.
    conv_rec, conv_seq = _conv_tokens(conv_sb, cm_rows, conv_w, q_tokens, conv_dt)
    k_rec, _ = _qk_norms(conv_rec, ones_sq)
    records = {
        "k": _pair_major(k_rec, n_seq, gpsimd),
        "v": _pair_major(conv_rec[0:P_MAX, 0:n_rows, 2 * heads : 3 * heads], n_seq, gpsimd),
        "decay": dd,
        "beta": bb,
        "s0": _state_fp32(state_sb),
    }

    # The window after replay_len records: the inputs at positions replay_len .. replay_len + 2 of conv_seq.
    for j in range(q_tokens):
        nisa.tensor_copy_predicated(
            dst=conv_sb,
            src=conv_seq[0:P_MAX, j + 1 : j + 1 + taps, 0:n_seq, 0:n_conv].permute((0, 2, 3, 1)),
            predicate=_per_sequence(replay_pred[0:P_MAX, j, 0:n_seq], (n_conv, taps)),
        )
    return records


def _batched_score(
    state_sb,
    v_rows,
    k_rows,
    q_rows,
    decay_rows,
    beta_rows,
    ident32,
    masks,
    out_bf,
    slot_states,
    records=None,
    state_waits=None,
):
    """Score the Q tokens of every sequence and head at once (``batched``) and write ``out_bf`` ``[128, Q*B, Hl]``.

    The rows are token-major (row ``j * B + b``) ``[128, Q*B, Hl]`` fp32 views (``v_rows`` may be bf16).
    ``state_sb`` ``[128, B, Hl, 128]`` is the state before token 0. ``slot_states`` (states mode) lists the Q
    tiles ``[128, B, Hl, 128]`` that take the state after each token; None writes no state.

    ``state_waits`` (states mode with a persistent ring) lists for each token the slot stores that read its tile last;
    every write into that tile waits for them (``_batched_solve``).

    ``records`` (the records form, from ``_batched_records``) puts the Q records of each pair before its Q new
    tokens: one solve of 2Q tokens from the state before the records gives the outputs of the new tokens and writes
    the state after the last record (the committed state) to ``state_sb``. The outputs read the fp32 committed
    state.
    """
    n_seq, heads = state_sb.shape[1], state_sb.shape[2]
    n_rows = k_rows.shape[1]
    q_tokens = n_rows // n_seq
    n_pairs = n_seq * heads
    fp32 = nl.float32
    vector, gpsimd = nisa.engine.vector, nisa.engine.gpsimd
    new = {
        "k": _pair_major(k_rows, n_seq, gpsimd),
        "q": _pair_major(q_rows, n_seq, gpsimd),
        "v": _pair_major(v_rows, n_seq, vector),
        "decay": _pair_major(decay_rows, n_seq, vector),
        "beta": _pair_major(beta_rows, n_seq, gpsimd),
    }
    if records is None:
        tokens, first = q_tokens, 0
        inputs = new
        s0 = _state_fp32(state_sb)
        state_dsts = () if slot_states is None else tuple((j, slot_states[j]) for j in range(q_tokens))
        ref = None
    else:
        tokens, first = 2 * q_tokens, q_tokens
        inputs = {}
        for name in ("k", "q", "v", "decay", "beta"):
            joined = nl.ndarray((P_MAX, n_pairs * tokens), dtype=fp32, buffer=nl.sbuf)
            pairs = joined.reshape_dim(1, (n_pairs, tokens))
            if name == "q":
                # The records give no outputs: their query columns are zero.
                nisa.memset(dst=pairs[0:P_MAX, 0:n_pairs, 0:q_tokens], value=0.0, engine=gpsimd)
            else:
                nisa.tensor_copy(
                    dst=pairs[0:P_MAX, 0:n_pairs, 0:q_tokens],
                    src=records[name].reshape_dim(1, (n_pairs, q_tokens)),
                    engine=gpsimd,
                )
            nisa.tensor_copy(
                dst=pairs[0:P_MAX, 0:n_pairs, q_tokens:tokens],
                src=new[name].reshape_dim(1, (n_pairs, q_tokens)),
                engine=vector,
            )
            inputs[name] = joined
        s0 = records["s0"]
        state_dsts = ((q_tokens - 1, state_sb),)
        ref = q_tokens - 1
    out_pairs = nl.ndarray((P_MAX, n_pairs * tokens), dtype=fp32, buffer=nl.sbuf)
    _batched_solve(
        inputs["k"],
        inputs["q"],
        inputs["v"],
        inputs["decay"],
        inputs["beta"],
        s0,
        ident32,
        masks,
        out_pairs=out_pairs,
        state_dsts=state_dsts,
        ref=ref,
        state_waits=None if records is not None else state_waits,
    )
    nisa.tensor_copy(
        dst=out_bf.reshape_dim(1, (q_tokens, n_seq)).permute((0, 2, 3, 1)),
        src=out_pairs.reshape_dim(1, (n_seq, heads, tokens))[0:P_MAX, 0:n_seq, 0:heads, first:tokens],
        engine=vector,
    )
