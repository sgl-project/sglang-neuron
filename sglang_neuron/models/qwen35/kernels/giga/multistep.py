"""Multi-step on-device greedy decode: N full decode steps per kernel launch (NKI, LNC=1).

Every launch on the 128 ranks pays a launch skew (ranks start independently; the first
collective waits for the slowest). One launch here runs ``num_steps`` complete steps:

    tok [T_shard] --embedding rows--> x [T_shard, H]                    (token-SP: own rows only)
      -> the qwen35_hybrid_stack layer loop (3 GDN + 1 gated attention, x15, MoE everywhere)
      -> final RMSNorm (own rows) -> AllGather[EP] -> this rank's vocab slice of lm_head + local argmax
      -> AllGather[EP] of the (max, id) candidates -> combine own rows -> next tokens -> next step

Design choices:
  * Embedding table REPLICATED per rank ([V, H] bf16 = 2.03 GB of ~18 GB HBM per NeuronCore): the
    next token's row is one indirect row gather (T_shard x 8 KB). A vocab-sharded table would
    need a masked gather of all C rows on every rank + a ReduceScatter[EP] of [C, H] in every step.
  * lm_head VOCAB-SHARDED over all ranks (V/128 = 1940 rows, 15.9 MB bf16 per rank). Every rank
    multiplies all C gathered rows by its slice (tokens stationary, ~62k streamed weight columns per
    128 rows); candidates are combined after one small AllGather (C x 128 x 8 bytes), each rank
    combining only its own T_shard rows (QWEN35_MS_OWNCOMB=1).
  * Per-step bookkeeping on device: positions pos + s (attention mask / KV write slot and the RoPE
    cos / sin rows gathered from [Lmax, 32] tables), the GDN conv / recurrent states carried in place
    from step to step, and the KV caches written at pos + s.
  * The step loop is unrolled at trace time (``QWEN35_MS_LOOP=0``, default: a collective inside an
    ``nl.fori_loop`` body fails at run time on trn3); ``QWEN35_MS_LOOP=1`` = device loop (one step traced).

Engine state (``state_mode``):
  2 POOL (default, no staging): the inputs ARE the engine's layer-major state pools - the P -> D hand-off
    layout of qwen35_prefill_stack:
        conv_state [L_g * Sp, Cc, K-1]      GDN layer g (the g-th non-attention layer), slot s at row g * Sp + s
        rec_state  [L_g * Sp, 128, vh, 128] fp32 (k-major [128, L_g * Sp, vh, 128] with QWEN35_GDN_KMAJOR=1)
        k_cache_t  [L_a * Spa, d, S], v_cache [L_a * Spa, S, d]   attention layer a, slot s at a * Spa + s
    and decode row r of this launch is slot r (persistent decode batch: a new request is placed into a
    freed row and its prefill hand-off writes that row's states). GDN states are updated IN PLACE in the pool
    (rec_row0 = g * Sp + r0), attention reads its history from and writes K/V at pos + s straight into its
    layer's pool block. The prologue / epilogue move no state bytes; only the tokens are written out.
  3 SLOT (pool + per-row slots ``gdn_slot`` [B_g, 1], ``kv_slot`` [B_attn, 1] int32, any slot order):
    GDN states are STAGED: each layer's active rows are gathered from the pool (indirect DMA through SBUF)
    into a private working pool [L_g * B_g, ...] just before step 0 reaches the layer (one layer of look-ahead,
    so the copies overlap step 0's earlier layers), updated in place there for N steps, and scattered back to
    their slots right after the last step's update of the layer (overlapping the rest of that step). Reason:
    gdn_decode's resident-state recurrence loads whole row blocks with one DMA; per-row indirect loads /
    stores inside the step body would cost ~2 x B_g small DMAs per layer per STEP instead of once per launch.
    Attention stages NOTHING: attn_decode_tkg reads the history from and writes the new K/V into the pool
    directly through its per-row slot input (slot = a * Spa + kv_slot[b], ``kv_write`` "flat").
  ``kv_slot`` in mode 2 likewise addresses the attention rows' slots in each layer block (GDN rows stay
  identity); without it mode 2 uses the static-address path on the layer's pool slice.
  1 SCAFFOLD (timing ids): one shared input state copied into per-layer private buffers before step 0, never
    written back (with ``kv_slot`` the copied caches are [NSLOT, ...] pools addressed by slot). 0 SHARED: one state / cache updated by every layer (false layer-to-layer dependency).
``return_state`` (modes 2 / 3): also return the pools (aliased inputs, after the write-back) - for tests.

Weights (``per_layer_weights``): False (timing ids) = every layer of a kind reads the SAME tensors. True = the
real model: every per-layer weight carries a leading layer axis and layer i reads its view ``w[k]`` (views of a
stacked HBM tensor address layer k exactly; no copy). k is
    the GDN index g (g-th non-attention layer)    w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out, w_out_ext, wx
    the attention index a (i % 4 == 3)            w_qkv, q_norm_w, k_norm_w, w_gate_p, w_o, w_o_ext, wx_a
    the layer index i                             in_ln_w, post_ln_w, w_router_p, w_shared_p, w_sd,
                                                  gate_up_w, gate_up_scale, down_w, down_scale
Each stacked tensor needs at least that many entries (a 60-layer pack also runs its first few layers). Shared
in both modes: ``sel`` (the rank's expert placement is the same in every layer), the TP / EP positions, the
embedding, final norm and lm_head. The expert-weight prefetch (``_early_experts`` -> issue_expert_weights) is
issued from the layer's own expert views, so it streams that layer's experts.

Persistent mode (``persistent=True``, state_mode 2): one launch runs until the host stops it. The step is traced
once inside an ``nl.while_loop``; every iteration ends with a vote, ``ctl != 0 and steps < max_steps`` on each
rank, summed by one AllReduce over ``ep_group`` (= all ranks), so all ranks leave on the same iteration (a rank
leaving alone would hang the others in their next collective). ``ctl`` is a [1, 1] int32 HBM input the host
rewrites mid-execution (nrt_tensor_write). Output: the ring [T_shard, 2, N] int32,
[:, 0, j] the token of the step whose 1-based number is [:, 1, j]; step k lands in slot (k - 1) % N and the
host trusts a slot once its stamp is the step it expects (token and stamp of a row go out in one DMA).

Padding rows (``row_valid``, optional [C_ep, 1] int32, EP-gathered order, identical on every rank): rows with
0 get all-zero routed affinities (layer_decode.moe_row_valid / moe_front ``valid``), so they take no
routed-row compaction slot in any layer of any step. Without it, padding rows sharing one state row and
position (and, from step 1 on, their fed-back argmax) route alike and overflow the compaction cap on a few
ranks in every MoE layer. Their tokens are still produced (don't-care).

Rank-dependent values (vocab slice base, EP position, TP positions) are DATA, so every rank runs the
same NEFF. Output: the generated ids [T_shard, num_steps] int32 of this rank's rows (rank r owns global
rows r*T_shard ..; concatenated over ranks: [C, num_steps]).
"""

import os
import re

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from nkilib.core.utils.kernel_assert import kernel_assert
from . import attn_decode as _attn_dec
from .gdn_decode import _KMAJOR
from .layer_decode import _SBUF_CC, _attn_mixer, _early_experts, _fold_inputs, _gdn_mixer, _moe_half, moe_row_valid
from .moe_experts_mx import _PREFETCH, LOOP_BODY, close_scale_pool, open_scale_pool
from .moe_front import rmsnorm_zc_tok
from .mtp import _bcast_scalar, _row_offsets, argmax_combine_body, lm_argmax_body, rope_rows_body

P = 128
STATE_SHARED, STATE_SCAFFOLD, STATE_POOL, STATE_SLOT = 0, 1, 2, 3
# Device loop over the steps (nl.fori_loop): fails at run time on trn3 with collectives inside the loop
# body (the same program passes unrolled), so the steps are unrolled at trace time by default.
_LOOP = os.environ.get("QWEN35_MS_LOOP", "0") == "1"
# Device loop: expert scale tiles (partly written by their DMAs, so loop-carried inside a loop body) come
# from a pool of "gu,dn" zeroed tiles allocated outside the loop; "0,0" memsets every scale chunk instead.
_SPOOL = tuple(int(v) for v in os.environ.get("QWEN35_MS_SPOOL", "16,4").split(","))
# Greedy combine of only this rank's own rows (one indirect gather of their candidates) instead of all C
# rows + an ids round trip through HBM + an indirect gather of the own ids (0: that path).
_OWN_COMBINE = os.environ.get("QWEN35_MS_OWNCOMB", "1") == "1"
_STAGE_COLS = int(os.environ.get("QWEN35_MS_STAGE_COLS", "8192"))  # elements per partition per staging DMA
# state_mode 3, one-step launch (first = last step): write every layer's state back after the step's last layer
# instead of right after the layer (where it interleaves with the next layer's stage-in). A diagnostic for a C=256
# N=1 state_mode 3 mismatch (wrong first tokens; state_mode 2 and N=8 state_mode 3 correct).
_WB_LATE = os.environ.get("QWEN35_MS_WB_LATE", "0") == "1"
# state_mode 3: build every GDN slot-index vector once in the prologue in ONE SBUF tile that stays live to the
# epilogue (read there), instead of a fresh tile per staging call (0). A fresh per-call index tile is not kept
# alive for the SW-DGE indirect DMA that reads it as vector_offset: its SBUF was reused while the copy was in
# flight, so staging moved wrong rows (wrong first tokens at C=256, N=1; with QWEN35_MS_WB_LATE=1 an
# out-of-bounds trap). With 1: correct. One-step launches at C=256 put the most staging DMAs in flight at once.
_IDX_ONCE = os.environ.get("QWEN35_MS_IDX_ONCE", "1") == "1"
# Trace-time DMA / collective census (count and bytes per phase and tensor) written to this file ("" = off).
_CENSUS = os.environ.get("QWEN35_MS_CENSUS", "")


# ----------------------------------------------------------------------------- trace-time census

_ISZ = {"float32": 4, "int32": 4, "uint32": 4, "bfloat16": 2, "float16": 2, "int16": 2, "uint16": 2,
        "float8": 1, "int8": 1, "uint8": 1}


def _nbytes(t):
    try:
        shape = getattr(t, "shape", None)
        if shape is None and hasattr(t, "pattern"):
            shape = [c for _, c in t.pattern]
        dt = getattr(t, "dtype", None) or getattr(getattr(t, "tensor", None), "dtype", None)
        k = next((v for nm, v in _ISZ.items() if nm in str(dt)), 0)
        m = 1
        for x in shape:
            m *= int(x)
        return m * k
    except Exception:
        return 0


def _tname(t):
    for o in (t, getattr(t, "tensor", None), getattr(t, "base", None)):
        nm = getattr(o, "name", None)
        if isinstance(nm, str) and nm:
            return re.sub(r"(_S\d+.*|_L\d+.*|_l\d+.*)$", "", nm)[:24]
    return "?"


def _is_hbm(t):
    for o in (t, getattr(t, "tensor", None)):
        b = getattr(o, "buffer", None)
        if b is not None:
            return "hbm" in str(b).lower()
    return False


class _Census:
    """Wraps nisa.dma_copy and the ncc collectives while the kernel is traced and tallies count / bytes per
    (phase, op, HBM tensor). Phases: prologue, S<s> (step s), stage_in / stage_out, epilogue."""

    def __init__(self, path):
        self.path, self.phase, self.rows, self.saved = path, "prologue", {}, []

    def _add(self, op, label, nb):
        r = self.rows.setdefault((self.phase, op, label), [0, 0])
        r[0] += 1
        r[1] += nb

    def __enter__(self):
        census = self
        real = nisa.dma_copy

        def dma(*a, **k):
            dst = k.get("dst", a[0] if a else None)
            src = k.get("src", a[1] if len(a) > 1 else None)
            hb = [t for t in (src, dst) if _is_hbm(t)]
            census._add("dma", _tname(hb[0] if hb else src), _nbytes(dst) or _nbytes(src))
            return real(*a, **k)

        self.saved.append((nisa, "dma_copy", real))
        nisa.dma_copy = dma
        for name in ("all_gather", "reduce_scatter", "all_reduce", "all_to_all"):
            f = getattr(ncc, name, None)
            if f is None:
                continue

            def cc(*a, _f=f, _n=name, **k):
                srcs = k.get("srcs") or []
                census._add("cc_" + _n, _tname(srcs[0]) if srcs else "?", sum(_nbytes(t) for t in srcs))
                return _f(*a, **k)

            self.saved.append((ncc, name, f))
            setattr(ncc, name, cc)
        return self

    def __exit__(self, *exc):
        for mod, name, f in self.saved:
            setattr(mod, name, f)
        try:
            tot = {}
            for (ph, op, _), (c, b) in self.rows.items():
                t = tot.setdefault((ph, "dma" if op == "dma" else "cc"), [0, 0])
                t[0] += c
                t[1] += b
            out = ["phase op count MB"] + [f"{ph} {op} {c} {b / 1e6:.3f}" for (ph, op), (c, b) in sorted(tot.items())]
            for ph in sorted({k[0] for k in self.rows}):
                top = sorted(((v[1], v[0], k[1], k[2]) for k, v in self.rows.items() if k[0] == ph), reverse=True)
                out += [f"  {ph} {op} {lab} n={c} {b / 1e6:.3f}MB" for b, c, op, lab in top[:16]]
            with open(self.path, "w") as f:
                f.write("\n".join(out) + "\n")
        except Exception as e:  # the census never breaks the trace
            print(f"census write failed: {e}")
        return False


class _NoCensus:
    phase = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ----------------------------------------------------------------------------- state staging (mode 3)


def _slot_idx(slot_hbm, r0, n, base):
    """[n, 1] int32 SBUF = slot_hbm[r0 : r0 + n] + base (pool row indices; < 2^24, exact in the fp32 ALU)."""
    t = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:n]
    nisa.dma_copy(dst=t, src=slot_hbm[r0 : r0 + n])
    if base:
        nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=base)
    return t


def gather_rows(dst2d, drow0, src2d, idx, n):
    """dst2d[drow0 + j, :] = src2d[idx[j], :] for j < n <= 128 (idx [n, 1] int32 SBUF): per column chunk one
    indirect HBM -> SBUF DMA (one contiguous run per row) and one plain SBUF -> HBM DMA."""
    W = src2d.shape[1]
    for c0 in range(0, W, _STAGE_COLS):
        cw = min(_STAGE_COLS, W - c0)
        t = nl.ndarray((P, cw), dtype=src2d.dtype, buffer=nl.sbuf)[:n]
        nisa.dma_copy(
            dst=t, src=src2d.ap(pattern=[[W, n], [1, cw]], offset=c0, vector_offset=idx, indirect_dim=0),
            dge_mode=nisa.dge_mode.swdge,
        )
        nisa.dma_copy(dst=dst2d.ap(pattern=[[W, n], [1, cw]], offset=drow0 * W + c0), src=t)


def scatter_rows(dst2d, idx, src2d, srow0, n):
    """dst2d[idx[j], :] = src2d[srow0 + j, :] for j < n <= 128 (the inverse of gather_rows)."""
    W = src2d.shape[1]
    for c0 in range(0, W, _STAGE_COLS):
        cw = min(_STAGE_COLS, W - c0)
        t = nl.ndarray((P, cw), dtype=src2d.dtype, buffer=nl.sbuf)[:n]
        nisa.dma_copy(dst=t, src=src2d.ap(pattern=[[W, n], [1, cw]], offset=srow0 * W + c0))
        nisa.dma_copy(
            dst=dst2d.ap(pattern=[[W, n], [1, cw]], offset=c0, vector_offset=idx, indirect_dim=0), src=t,
            dge_mode=nisa.dge_mode.swdge,
        )


def _flat2(t):
    """[R, ...] HBM tensor -> [R, prod(rest)] view."""
    w = 1
    for x in t.shape[1:]:
        w *= x
    return t.reshape((t.shape[0], w))


def stage_gdn_in(wconv, wrec, conv_pool, rec_pool, slot_hbm, g, Sp, B, idx_of=None):
    """Working rows g*B + r <- pool rows g*Sp + slot[r] (conv and recurrent state of GDN layer g, r < B)."""
    for r0 in range(0, B, P):
        rn = min(P, B - r0)
        idx = idx_of(g, r0, rn) if idx_of else _slot_idx(slot_hbm, r0, rn, g * Sp)
        gather_rows(_flat2(wconv), g * B + r0, _flat2(conv_pool), idx, rn)
        gather_rows(_flat2(wrec), g * B + r0, _flat2(rec_pool), idx, rn)


def stage_gdn_out(conv_pool, rec_pool, wconv, wrec, slot_hbm, g, Sp, B, idx_of=None):
    """Pool rows g*Sp + slot[r] <- working rows g*B + r (after the last step's update of GDN layer g)."""
    for r0 in range(0, B, P):
        rn = min(P, B - r0)
        idx = idx_of(g, r0, rn) if idx_of else _slot_idx(slot_hbm, r0, rn, g * Sp)
        scatter_rows(_flat2(conv_pool), idx, _flat2(wconv), g * B + r0, rn)
        scatter_rows(_flat2(rec_pool), idx, _flat2(wrec), g * B + r0, rn)


def _layer_block_table(block_table, a, Spa, B):
    """HBM [B, MP] int32 = a * Spa + block_table: attention layer a's pool rows of every row's pages (the paged KV's
    layer-major pools hold layer a's pages at rows a * Spa ..; attn_decode_tkg ``block_table``)."""
    MP = block_table.shape[1]
    t = nl.ndarray((P, MP), dtype=nl.int32, buffer=nl.sbuf)[:B]
    nisa.dma_copy(dst=t, src=block_table)
    if a:
        nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=a * Spa)
    out = nl.ndarray((B, MP), dtype=nl.int32, buffer=nl.private_hbm, name=f"ms_abt_l{a}")
    nisa.dma_copy(dst=out, src=t)
    return out


def _attn_slots(kv_slot, a, Spa, B):
    """HBM [B, 1] int32 = a * Spa + kv_slot: attention layer a's per-row pool slots (attn_decode_tkg ``slot``)."""
    idx = _slot_idx(kv_slot, 0, B, a * Spa)
    out = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.private_hbm, name=f"ms_aslot_l{a}")
    nisa.dma_copy(dst=out, src=idx)
    return out


# ----------------------------------------------------------------------------- per-layer weights


class _LayerWeights:
    """Weights of one layer: views ``w[k]`` of the stacked tensors with ``per_layer_weights``, else the shared
    tensors themselves (module docstring). Groups gdn / gdn_fold are indexed by the GDN index, attn / attn_fold
    by the attention index, layer (norms + MoE) by the layer index."""

    def __init__(self, per_layer, num_layers, L_g, L_a, **groups):
        self.pl, self.groups = per_layer, groups
        self.count = {"gdn": L_g, "gdn_fold": L_g, "attn": L_a, "attn_fold": L_a, "layer": num_layers}
        for nm, ws in groups.items():
            for w in ws:
                kernel_assert(
                    not per_layer or w is None or w.shape[0] >= self.count[nm],
                    f"per_layer_weights: every {nm} weight needs a leading layer axis of >= {self.count[nm]}",
                )

    def get(self, nm, k):
        return tuple(w[k] if self.pl and w is not None else w for w in self.groups[nm])

    def has(self, nm):
        return not self.pl or self.count[nm] > 0

    def first(self, nm):
        return self.get(nm, 0)

    def fold(self, fold, nm, k):
        """Router-fold tuple of layer k: its (w_ext, wx) views + the shared TP position tile."""
        if fold is None or not self.pl:
            return fold
        return self.get(nm, k) + (fold[2],)


# ----------------------------------------------------------------------------- step pieces


def _embed_rows(table, ids_sb, n, tag):
    """x [n, H] bf16 private HBM = table[ids] (ids an [n, 1] int32 SBUF tile; one indirect row gather)."""
    H = table.shape[1]
    e = nl.ndarray((P, H), dtype=table.dtype, buffer=nl.sbuf)[:n]
    nisa.dma_copy(
        dst=e, src=table.ap(pattern=[[H, n], [1, H]], offset=0, vector_offset=ids_sb, indirect_dim=0),
        dge_mode=nisa.dge_mode.swdge,
    )
    x = nl.ndarray((n, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name="x_emb" + tag)
    nisa.dma_copy(dst=x, src=e)
    return x


def _final_norm_weight(final_ln_w, n):
    """(1 + w) broadcast to n partitions, fp32 SBUF [n, H]: loaded ONCE per launch (step-invariant)."""
    H = final_ln_w.shape[0]
    w_row = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)[:n]
    nisa.dma_copy(dst=w_row, src=final_ln_w.ap(pattern=[[0, n], [1, H]], offset=0))
    nisa.tensor_scalar(dst=w_row, data=w_row, op0=nl.add, operand0=1.0)
    return w_row


def _final_norm(x_own, w_row, tag):
    """Final zero-centred RMSNorm of this rank's rows -> bf16 private HBM [n, H]."""
    n, H = x_own.shape
    x_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)[:n]
    nisa.dma_copy(dst=x_sb, src=x_own[:, 0:H])
    hn = rmsnorm_zc_tok(x_sb, w_row, n, H, 1e-6, nl.bfloat16)
    out = nl.ndarray((n, H), dtype=nl.bfloat16, buffer=nl.private_hbm, name="hn" + tag)
    nisa.dma_copy(dst=out, src=hn)
    return out


def greedy_next_tokens(x_own, w_row, w_lm_p, vbase, own_off, ep_group, ep_size, tag, vb_sb=None):
    """Greedy next token of this rank's rows: final norm -> AllGather[EP] -> vocab-slice logits + local
    argmax -> AllGather[EP] of the candidates -> combine -> own rows. Returns an [n, 1] int32 SBUF tile.

    w_row: [n, H] fp32 SBUF (1 + final_ln_w), from _final_norm_weight.
    own_off: [n, 1] int32 SBUF, ep_rank * n + i (this rank's rows among the gathered ones).
    vb_sb: optional [min(T, 128), 1] int32 SBUF broadcast of vbase (step-invariant, loaded once)."""
    n, H = x_own.shape[0], w_row.shape[1]
    T = n * ep_size
    hbm = nl.private_hbm
    hn = _final_norm(x_own, w_row, tag)
    lm_in = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=hbm, name="lm_in" + tag)
    ncc.all_gather(dsts=[lm_in], srcs=[hn], replica_group=ep_group, collective_dim=0)
    cand = nl.ndarray((T, 2), dtype=nl.float32, buffer=hbm, name="cand" + tag)
    lm_argmax_body(lm_in, w_lm_p, vbase, cand, vb_sb)
    cand_all = nl.ndarray((T * ep_size, 2), dtype=nl.float32, buffer=hbm, name="cand_all" + tag)
    ncc.all_gather(dsts=[cand_all], srcs=[cand], replica_group=ep_group, collective_dim=0)
    if _OWN_COMBINE:  # only the own rows' candidates: [n, R, 2] in one indirect gather, ids stay in SBUF
        return argmax_combine_body(cand_all, T, ep_size, 0, n, own_off)
    ids_hbm = nl.ndarray((T, 1), dtype=nl.int32, buffer=hbm, name="ids_all" + tag)
    for t0 in range(0, T, P):
        tn = min(P, T - t0)
        ids = argmax_combine_body(cand_all, T, ep_size, t0, tn)
        nisa.dma_copy(dst=ids_hbm[t0 : t0 + tn], src=ids)
    own = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:n]
    nisa.dma_copy(
        dst=own, src=ids_hbm.ap(pattern=[[1, n], [1, 1]], offset=0, vector_offset=own_off, indirect_dim=0),
        dge_mode=nisa.dge_mode.swdge,
    )
    return own


@nki.jit
def qwen35_hybrid_multistep(
    tok_shard,  # [T_shard, 1] int32 current tokens of this rank's rows (last sampled / prompt's last)
    embed_table,  # [V_e, H] bf16 embed_tokens (replicated)
    in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out,  # GDN shard (+ state)
    w_qkv, q_norm_w, k_norm_w, w_gate_p,  # attention shard
    cos_tab, sin_tab,  # [Lmax, 32] fp32 RoPE tables (rows gathered at pos + s on device)
    k_cache_t, v_cache,  # attention KV: state_mode 2 / 3 the layer-major pools (module docstring); 0 / 1 one
    # layer's caches ([B_attn, ...], or a [NSLOT, ...] pool with kv_slot)
    pos,  # [B_attn, 1] int32 position of step 0's token for each attention-group sequence
    w_o,
    post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale,  # MoE
    final_ln_w,  # [H] fp32 model.norm (zero-centred)
    w_lm_p,  # [128, H/128, Vs] bf16 this rank's lm_head^T vocab slice, rows pre-permuted
    vbase,  # [1, 1] int32 global id of the slice's first token
    ep_rank,  # [1, 1] int32 this rank's EP position (its rows among the EP-gathered ones)
    tp_group: ReplicaGroup,
    atp_group: ReplicaGroup,
    ep_group: ReplicaGroup,
    tp_size: int,
    atp_size: int,
    ep_size: int,
    num_layers: int,
    num_steps: int,
    state_mode: int = STATE_POOL,  # 2 pool in place | 3 pool + per-row slots (staged) | 1 scaffold | 0 shared
    gdn_slot=None,  # state_mode 3: [B_g, 1] int32 pool slot of each GDN state row (B_g = T_shard * tp_size)
    return_state: bool = False,  # modes 2 / 3: return (tokens, conv_state, rec_state, k_cache_t, v_cache)
    # router fold (as qwen35_hybrid_stack)
    w_out_ext=None, wx=None, tp_rank=None, w_o_ext=None, wx_a=None, atp_rank=None,
    kv_slot=None,  # [B_attn, 1] int32 KV-pool slot per attention row (caches [NSLOT, ...]); None: row b -> slot b
    kv_write="flat",  # attn_decode_tkg new-token K/V write mode with kv_slot
    per_layer_weights: bool = False,  # weights stacked on a leading layer axis, layer reads w[k] (module docstring)
    row_valid=None,  # [T_shard * ep_size, 1] int32, same on every rank: 0 = padding row, routed to no expert
    persistent: bool = False,  # run until the host says stop (module docstring "Persistent mode")
    ctl=None,  # persistent: [1, 1] int32 HBM run flag of this rank (host writes 0 to stop)
    max_steps: int = (1 << 24) - 1,  # persistent: bound on the steps of one launch (fp32 step stamps exact)
    block_table=None,  # state_mode 2 / 3: [B_attn, MP] int32 pages of each attention row (paged KV, patch 0004)
    kv_cp: int = 1,  # decode context parallelism over the atp group (patch 0009): the attention layers run
    # _attn_mixer_cp on the 4-D physical-block pool ([rows, 2, d, 2048/kv_cp] / [rows, 2, 2048/kv_cp, d])
    cp_sink=None,  # kv_cp > 1: [1, 1] int32 pool row the non-owners' new-token writes drop into
):
    """``num_steps`` greedy decode steps of ``num_layers`` hybrid layers in one launch (module docstring).

    Same per-layer structure as ``qwen35_hybrid_stack`` (every layer issues all of its DMAs and collectives);
    layers of a kind share weights, or read their own layer's views with ``per_layer_weights``. Returns tokens
    [T_shard, num_steps] int32 (+ the state pools with ``return_state``); persistent: the token ring
    [T_shard, num_steps, 2] int32 (token, step stamp) instead of tokens."""
    with _Census(_CENSUS) if _CENSUS else _NoCensus() as census:
        return _multistep_body(
            census, tok_shard, embed_table, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w,
            rec_state, w_out, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos_tab, sin_tab, k_cache_t, v_cache, pos, w_o,
            post_ln_w, w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, final_ln_w,
            w_lm_p, vbase, ep_rank, tp_group, atp_group, ep_group, tp_size, atp_size, ep_size, num_layers,
            num_steps, state_mode, gdn_slot, return_state, w_out_ext, wx, tp_rank, w_o_ext, wx_a,
            atp_rank, kv_slot, kv_write, per_layer_weights, row_valid, persistent, ctl, max_steps, block_table,
            kv_cp, cp_sink,
        )


def _multistep_body(
    census, tok_shard, embed_table, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state,
    w_out, w_qkv, q_norm_w, k_norm_w, w_gate_p, cos_tab, sin_tab, k_cache_t, v_cache, pos, w_o, post_ln_w,
    w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, final_ln_w, w_lm_p, vbase,
    ep_rank, tp_group, atp_group, ep_group, tp_size, atp_size, ep_size, num_layers, num_steps, state_mode,
    gdn_slot, return_state, w_out_ext, wx, tp_rank, w_o_ext, wx_a, atp_rank, kv_slot, kv_write, per_layer_weights,
    row_valid=None, persistent=False, ctl=None, max_steps=(1 << 24) - 1, block_table=None, kv_cp=1, cp_sink=None,
):
    _PREFETCH.clear()
    close_scale_pool()
    kernel_assert(not _SBUF_CC, "multi-step decode uses the HBM residual stream (QWEN35_SBUF_CC=0)")
    n = tok_shard.shape[0]
    B_attn = pos.shape[0]
    B_g = n * tp_size
    half = cos_tab.shape[1]
    N = num_steps
    hbm = nl.private_hbm
    gdn_layers = [i for i in range(num_layers) if i % 4 != 3]
    attn_layers = [i for i in range(num_layers) if i % 4 == 3]
    gi = {i: k for k, i in enumerate(gdn_layers)}
    ai = {i: k for k, i in enumerate(attn_layers)}
    L_g, L_a = len(gdn_layers), len(attn_layers)
    kernel_assert(state_mode in (0, 1, 2, 3), "state_mode: 0 shared, 1 scaffold, 2 pool, 3 pool + slots")
    if kv_cp > 1:  # DCP (patch 0009): the paged pool only; every atp rank's block table is the same
        kernel_assert(
            state_mode in (STATE_POOL, STATE_SLOT) and block_table is not None and cp_sink is not None
            and atp_rank is not None and kv_cp == atp_size,
            f"kv_cp > 1 needs state_mode 2/3, block_table, cp_sink, atp_rank and kv_cp == atp_size "
            f"(kv_cp={kv_cp}, atp_size={atp_size})",
        )
    lw = _LayerWeights(
        per_layer_weights, num_layers, L_g, L_a,
        gdn=(w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out), gdn_fold=(w_out_ext, wx),
        attn=(w_qkv, q_norm_w, k_norm_w, w_gate_p, w_o), attn_fold=(w_o_ext, wx_a),
        layer=(in_ln_w, post_ln_w, w_router_p, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale),
    )
    # router fold: shapes checked on one layer's tensors, TP positions loaded once; lw.fold picks each layer's pair
    fold_g = _fold_inputs(*lw.first("gdn_fold"), tp_rank, lw.first("gdn")[5], "GDN") if lw.has("gdn") else None
    fold_a = _fold_inputs(*lw.first("attn_fold"), atp_rank, lw.first("attn")[4], "attention") if lw.has("attn") else None
    kernel_assert(not return_state or state_mode >= STATE_POOL, "return_state needs a state pool (state_mode 2/3)")
    kernel_assert(
        not persistent or (state_mode == STATE_POOL and ctl is not None),
        "persistent: state_mode 2 (rows = slots, nothing staged per launch) and a ctl run flag",
    )

    # ---- per-layer state: gdn_st[i] = (conv, rec, row0), attn_kv[i] = (k, v, slot); prologue / epilogue hooks ----
    gdn_st, attn_kv, stage, flush = {}, {}, {}, {}
    attn_pool_off = {}  # attention layer i's pool-row offset when the block loop reads the raw block table
    idx_keep = [None]  # _IDX_ONCE: the slot-index tile, read in the epilogue so it stays live (keep_alive below)
    if state_mode == STATE_SHARED:
        for i in gdn_layers:
            gdn_st[i] = (conv_state, rec_state, 0)
        for i in attn_layers:
            attn_kv[i] = (k_cache_t, v_cache, kv_slot, None)
    elif state_mode == STATE_SCAFFOLD:
        for i in range(num_layers):
            if i % 4 == 3:
                kc = nl.ndarray(k_cache_t.shape, dtype=k_cache_t.dtype, buffer=hbm, name=f"kc_l{i}")
                vc = nl.ndarray(v_cache.shape, dtype=v_cache.dtype, buffer=hbm, name=f"vc_l{i}")
                nisa.dma_copy(dst=kc, src=k_cache_t)
                nisa.dma_copy(dst=vc, src=v_cache)
                attn_kv[i] = (kc, vc, kv_slot, None)
            else:
                cs = nl.ndarray(conv_state.shape, dtype=conv_state.dtype, buffer=hbm, name=f"conv_l{i}")
                rs = nl.ndarray(rec_state.shape, dtype=rec_state.dtype, buffer=hbm, name=f"rec_l{i}")
                nisa.dma_copy(dst=cs, src=conv_state)
                nisa.dma_copy(dst=rs, src=rec_state)
                gdn_st[i] = (cs, rs, 0)
    else:
        rec_rows = rec_state.shape[1] if _KMAJOR else rec_state.shape[0]
        Sp = conv_state.shape[0] // L_g if L_g else 0
        Spa = k_cache_t.shape[0] // L_a if L_a else 0
        kernel_assert(
            not L_g or (conv_state.shape[0] == L_g * Sp and rec_rows == L_g * Sp),
            "GDN state pool: conv_state / rec_state rows must be (number of GDN layers) x slots",
        )
        kernel_assert(
            not L_a
            or (k_cache_t.shape[0] == L_a * Spa and v_cache.shape[0] == L_a * Spa)
            or (kv_cp > 1 and k_cache_t.shape[0] == L_a * Spa + 1 and v_cache.shape[0] == L_a * Spa + 1),
            "KV pool: k_cache_t / v_cache slots must be (number of attention layers) x slots"
            " (plus one trailing CP write sink row under kv_cp)",
        )
        kernel_assert((not L_g or Sp >= B_g) and (not L_a or Spa >= B_attn or block_table is not None),
                      "state pool: fewer slots than rows")
        if state_mode == STATE_POOL:  # decode row r = slot r: GDN states in place in the pool
            for i in gdn_layers:
                gdn_st[i] = (conv_state, rec_state, gi[i] * Sp)
        else:
            kernel_assert(gdn_slot is not None, "state_mode 3 needs gdn_slot")
            kernel_assert(not _KMAJOR, "state_mode 3 stages token-major recurrent states (QWEN35_GDN_KMAJOR=0)")
            if L_g:
                wconv = nl.ndarray((L_g * B_g,) + tuple(conv_state.shape[1:]), dtype=conv_state.dtype, buffer=hbm,
                                   name="ms_wconv")
                wrec = nl.ndarray((L_g * B_g,) + tuple(rec_state.shape[1:]), dtype=rec_state.dtype, buffer=hbm,
                                  name="ms_wrec")
            idx_of = None
            if _IDX_ONCE and L_g:  # column g * n_ch + c = pool rows g*Sp + slot[c*128 : c*128 + 128]
                n_ch = (B_g + P - 1) // P
                idx_all = nl.ndarray((P, L_g * n_ch), dtype=nl.int32, buffer=nl.sbuf)
                nisa.memset(dst=idx_all, value=0)  # rows past B_g stay 0 (the epilogue reads the whole tile)
                for c in range(n_ch):
                    rn = min(P, B_g - c * P)
                    nisa.dma_copy(dst=idx_all[:rn, c : c + 1], src=gdn_slot[c * P : c * P + rn])
                    for g in range(1, L_g):
                        nisa.tensor_scalar(dst=idx_all[:rn, g * n_ch + c : g * n_ch + c + 1],
                                           data=idx_all[:rn, c : c + 1], op0=nl.add, operand0=g * Sp)
                idx_keep[0] = idx_all
                idx_of = lambda g, r0, rn: idx_all[:rn, g * n_ch + r0 // P : g * n_ch + r0 // P + 1]  # noqa: E731
            for i in gdn_layers:
                g = gi[i]
                gdn_st[i] = (wconv, wrec, g * B_g)
                stage[i] = lambda g=g: stage_gdn_in(wconv, wrec, conv_state, rec_state, gdn_slot, g, Sp, B_g, idx_of)
                flush[i] = lambda g=g: stage_gdn_out(conv_state, rec_state, wconv, wrec, gdn_slot, g, Sp, B_g, idx_of)
        for i in attn_layers:  # attention: in place in the pool (layer slice, or per-row slots), no staging
            a = ai[i]
            if block_table is not None:
                kernel_assert(kv_slot is None, "paged KV: block_table replaces kv_slot")
                kernel_assert(L_a * Spa < (1 << 24), "paged KV: pool rows must stay exact in the fp32 ALU")
                if kv_cp > 1:
                    # DCP: the 4-D pool rows are the physical blocks; the mixer maps local block lb to table
                    # column lb in-kernel. Short spans take the staged table (its rows already carry a * Spa).
                    if _attn_dec.cp_block_loop_engages(block_table.shape[1] * k_cache_t.shape[3]):
                        # the CP block loop reads the RAW table (a kernel input: dynamic-offset DMAs from
                        # an HBM intermediate trip the allocator inside a device loop — NCC_IBIR853); the
                        # layer's a * Spa pool offset is applied in-kernel after each pass's column DMA
                        attn_kv[i] = (k_cache_t, v_cache, None, block_table)
                        attn_pool_off[i] = a * Spa
                    else:
                        attn_kv[i] = (k_cache_t, v_cache, None, _layer_block_table(block_table, a, Spa, B_attn))
                elif _attn_dec.block_loop_engages(True, block_table.shape[1] * k_cache_t.shape[2],
                                                k_cache_t.shape[2], "dp"):
                    # the block loop reads the RAW table (a kernel input: dynamic-offset DMAs from an HBM
                    # intermediate trip the allocator inside a device loop — NCC_IBIR853); the layer's
                    # a * Spa pool offset is applied in-kernel after each pass's column DMA
                    attn_kv[i] = (k_cache_t, v_cache, None, block_table)
                    attn_pool_off[i] = a * Spa
                else:
                    attn_kv[i] = (k_cache_t, v_cache, None, _layer_block_table(block_table, a, Spa, B_attn))
            elif kv_slot is None:
                attn_kv[i] = (k_cache_t[a * Spa : a * Spa + B_attn], v_cache[a * Spa : a * Spa + B_attn], None, None)
            else:
                attn_kv[i] = (k_cache_t, v_cache, _attn_slots(kv_slot, a, Spa, B_attn), None)

    staged = set()

    def ensure_staged(i):  # issue layer i's prologue copies once (no-op without staging)
        if i in stage and i not in staged:
            staged.add(i)
            ph, census.phase = census.phase, "stage_in"
            stage[i]()
            census.phase = ph

    def write_back(i):
        if i in flush:
            ph, census.phase = census.phase, "stage_out"
            flush[i]()
            census.phase = ph

    # ---- small loop-carried SBUF tiles + step-invariant values (loaded once per launch) ----
    tok_sb = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:n]
    nisa.dma_copy(dst=tok_sb, src=tok_shard)
    pos_sb = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)[:B_attn]
    nisa.dma_copy(dst=pos_sb, src=pos)
    r_base = _bcast_scalar(ep_rank, n)
    nisa.tensor_scalar(dst=r_base, data=r_base, op0=nl.multiply, operand0=n)
    own_off = _row_offsets(n, 1, r_base)  # this rank's rows among the EP-gathered ones: ep_rank * n + i
    w_fin = _final_norm_weight(final_ln_w, n)
    vb_sb = _bcast_scalar(vbase, min(P, n * ep_size))
    cnt = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:n]  # step counter
    nisa.memset(cnt, 0.0)
    step_i = nl.ndarray((P, N), dtype=nl.int32, buffer=nl.sbuf)[:n]
    nisa.iota(step_i, pattern=[[1, N]], offset=0, channel_multiplier=0)
    step_f = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)[:n]
    nisa.tensor_copy(dst=step_f, src=step_i)
    out_f = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)[:n]  # generated ids (exact in fp32)
    nisa.memset(out_f, 0.0)
    if persistent:  # step stamps of the ring slots + steps done so far (exact in fp32 below 2^24)
        stamp_f = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.memset(stamp_f, 0.0)
        total = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.memset(total, 0.0)
    # padding rows (row_valid 0) are routed to no expert in every layer of every step: static for the launch
    moe_valid = moe_row_valid(row_valid, n, ep_size)

    def step(tg, first, last):
        """One decode step; ``first`` / ``last``: issue the staged layers' prologue copies (one layer of
        look-ahead) / epilogue write-backs (right after the layer's last update) inside this step."""
        if first and num_layers:
            ensure_staged(0)
        x = _embed_rows(embed_table, tok_sb, n, tg)
        pos_h = nl.ndarray((B_attn, 1), dtype=nl.int32, buffer=hbm, name="pos" + tg)
        nisa.dma_copy(dst=pos_h, src=pos_sb)
        cos_h = nl.ndarray((B_attn, half), dtype=nl.float32, buffer=hbm, name="cos" + tg)
        sin_h = nl.ndarray((B_attn, half), dtype=nl.float32, buffer=hbm, name="sin" + tg)
        rope_rows_body(cos_tab, sin_tab, pos_sb, B_attn, cos_h, sin_h)
        for i in range(num_layers):
            tag = f"{tg}_L{i}"
            if first:
                ensure_staged(i)
            # layer i's weights: views w[k] of the stacked tensors with per_layer_weights, else the shared ones
            l_in_ln, l_post_ln, l_router, l_shared, l_sd, l_gu, l_gus, l_dn, l_dns = lw.get("layer", i)
            pre = _early_experts(l_gu, l_gus, l_dn, l_dns, "layer")
            if i % 4 == 3:
                kc, vc, ksl, bt = attn_kv[i]
                l_qkv, l_qn, l_kn, l_gate, l_o = lw.get("attn", ai[i])
                resid1 = _attn_mixer(
                    x, l_in_ln, l_qkv, l_qn, l_kn, l_gate, cos_h, sin_h, kc, vc, pos_h, l_o, atp_group, atp_size,
                    tag, lw.fold(fold_a, "attn_fold", ai[i]), ksl, kv_write, block_table=bt,
                    pool_off=attn_pool_off.get(i, 0), kv_cp=kv_cp, cp_rank=atp_rank, cp_sink=cp_sink,
                )
            else:
                cs, rs, row0 = gdn_st[i]
                l_proj, l_conv, l_alog, l_dtb, l_norm, l_out = lw.get("gdn", gi[i])
                resid1, _, _ = _gdn_mixer(
                    x, l_in_ln, l_proj, l_conv, cs, l_alog, l_dtb, l_norm, rs, l_out, tp_group, tp_size, tag,
                    lw.fold(fold_g, "gdn_fold", gi[i]), True, row0,
                )
            if last and not (first and _WB_LATE):  # layer i's state is final: write it back while the step runs
                write_back(i)
            if first and i + 1 < num_layers:  # next layer's state copies overlap this layer's MoE half
                ensure_staged(i + 1)
            x = _moe_half(
                resid1, l_post_ln, l_router, sel, l_shared, l_sd, l_gu, l_gus, l_dn, l_dns, ep_group, ep_size, tag,
                hbm, pre, moe_valid,
            )
        if last and first and _WB_LATE:
            for i in range(num_layers):
                write_back(i)
        tok_new = greedy_next_tokens(x, w_fin, w_lm_p, vbase, own_off, ep_group, ep_size, tg, vb_sb)
        # record: out_f[:, s] = tok (one-hot of the step counter), then advance tokens / positions / counter
        tok_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.tensor_copy(dst=tok_f, src=tok_new)
        hot = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)[:n]
        nisa.tensor_scalar(dst=hot, data=step_f, op0=nl.equal, operand0=cnt)
        if persistent:  # ring slot cnt is reused: clear it, stamp it with the step number (1-based)
            keep = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)[:n]
            nisa.tensor_scalar(dst=keep, data=hot, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
            nisa.tensor_tensor(dst=out_f, data1=out_f, data2=keep, op=nl.multiply)
            nisa.tensor_scalar(dst=total, data=total, op0=nl.add, operand0=1.0)
            nisa.tensor_tensor(dst=stamp_f, data1=stamp_f, data2=keep, op=nl.multiply)
            nisa.scalar_tensor_tensor(
                dst=stamp_f, data=hot, op0=nl.multiply, operand0=total, op1=nl.add, operand1=stamp_f
            )
        nisa.scalar_tensor_tensor(dst=out_f, data=hot, op0=nl.multiply, operand0=tok_f, op1=nl.add, operand1=out_f)
        nisa.tensor_scalar(dst=cnt, data=cnt, op0=nl.add, operand0=1.0)
        if persistent:  # ring position wraps: cnt = cnt * (cnt < N)
            wrap = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)[:n]
            nisa.tensor_scalar(dst=wrap, data=cnt, op0=nl.less, operand0=float(N))
            nisa.tensor_tensor(dst=cnt, data1=cnt, data2=wrap, op=nl.multiply)
        nisa.tensor_copy(dst=tok_sb, src=tok_new)
        nisa.tensor_scalar(dst=pos_sb, data=pos_sb, op0=nl.add, operand0=1)

    if persistent:  # device while-loop until any rank's host clears its ctl flag (or max_steps)
        ring = nl.ndarray((n, 2, N), dtype=nl.int32, buffer=nl.shared_hbm, name="tok_ring")
        ring_sb = nl.ndarray((P, 2, N), dtype=nl.int32, buffer=nl.sbuf)[:n]
        vote_h = nl.ndarray((1, 1), dtype=nl.float32, buffer=hbm, name="ms_vote")
        vsum_h = nl.ndarray((1, 1), dtype=nl.float32, buffer=hbm, name="ms_vote_sum")
        ctl_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)

        def vote():
            """1 on every rank iff every rank's ctl != 0 and steps < max_steps: one AllReduce(add) over the
            EP group (= all ranks), so all ranks leave the loop on the same iteration."""
            nisa.dma_copy(dst=ctl_sb, src=ctl)
            v = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v, src=ctl_sb)
            nisa.tensor_scalar(dst=v, data=v, op0=nl.not_equal, operand0=0.0)
            lim = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=lim, data=total[0:1], op0=nl.less, operand0=float(max_steps))
            nisa.tensor_tensor(dst=v, data1=v, data2=lim, op=nl.multiply)
            nisa.dma_copy(dst=vote_h, src=v)
            ncc.all_reduce(dsts=[vsum_h], srcs=[vote_h], op=nl.add, replica_group=ep_group)
            s = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s, src=vsum_h)
            ok = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ok, data=s, op0=nl.equal, operand0=float(ep_size))
            r = nisa.register_alloc()
            nisa.register_load(r, ok)
            return r

        census.phase = "S*"
        if (_SPOOL[0] > 0 or _SPOOL[1] > 0) and lw.has("layer"):
            open_scale_pool(lw.first("layer")[5], lw.first("layer")[7], _SPOOL[0], _SPOOL[1])
        LOOP_BODY[0] = True
        try:
            r0 = vote()

            def body(_):
                step("", False, False)
                nisa.tensor_copy(dst=ring_sb[:, 0, :], src=out_f)
                nisa.tensor_copy(dst=ring_sb[:, 1, :], src=stamp_f)
                nisa.dma_copy(dst=ring, src=ring_sb)  # one DMA: a row's token and its stamp land together
                return vote()

            nl.while_loop(r0, body)
        finally:
            LOOP_BODY[0] = False
            close_scale_pool()
        census.phase = "epilogue"
        if return_state:
            return ring, conv_state, rec_state, k_cache_t, v_cache
        return ring

    if _LOOP:  # one traced step: all staging before the loop, all write-backs after it (no overlap)
        for i in range(num_layers):
            ensure_staged(i)
        census.phase = "S*"
        if (_SPOOL[0] > 0 or _SPOOL[1] > 0) and lw.has("layer"):
            open_scale_pool(lw.first("layer")[5], lw.first("layer")[7], _SPOOL[0], _SPOOL[1])  # one layer's shapes
        LOOP_BODY[0] = True
        try:
            nl.fori_loop(0, N, lambda _s: step("", False, False))
        finally:
            LOOP_BODY[0] = False
            close_scale_pool()
        for i in range(num_layers):
            write_back(i)
    else:
        for s in range(N):
            census.phase = f"S{s}"
            step(f"_S{s}", s == 0, s == N - 1)

    census.phase = "epilogue"
    if idx_keep[0] is not None:  # out_f += 0 * sum(slot indices): a tracked read after every staging DMA's issue
        ia = idx_keep[0]
        fa = nl.ndarray(ia.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=fa, src=ia)
        z = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=z, op=nl.add, data=fa, axis=1)
        nisa.tensor_scalar(dst=z, data=z, op0=nl.multiply, operand0=0.0)
        nisa.tensor_scalar(dst=out_f, data=out_f, op0=nl.add, operand0=z[:n])
    out_i = nl.ndarray((P, N), dtype=nl.int32, buffer=nl.sbuf)[:n]
    nisa.tensor_copy(dst=out_i, src=out_f)
    tokens = nl.ndarray((n, N), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=tokens, src=out_i)
    if return_state:
        return tokens, conv_state, rec_state, k_cache_t, v_cache
    return tokens
