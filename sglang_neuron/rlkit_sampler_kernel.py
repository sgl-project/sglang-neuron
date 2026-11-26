# SPDX-License-Identifier: Apache-2.0
"""RLKIT: the shard-local stage of the temperature sampler (sampling.sample_sharded_logits, v0.1.0) as one NKI kernel.

Selected by RLKIT_SAMPLER_KERNEL=1 (the guarded branch in sampling.sample_sharded_logits). See
sglang_rl/SAMPLER_KERNEL_NOTES.md for the decision and the equality promised.

Layout (the point of the kernel): the engine's torch ops run on logits [R, 37984] with the R rows on partitions, so
every elementwise op walks 37,984 elements per partition whatever R is (a constant cost per step at R = 1, 8, 32).
Here the shard is viewed as [R * S, W / S] (S segments per row, R * S <= 128 partitions, a free reshape of a contiguous
tensor), and the kernel walks W * R / 128 elements per partition (1,187 at R = 1 with S = 32; 2,374 at R = 8; 9,496 at
R = 32), in chunks of CHUNK columns. Per (partition, chunk) it writes 4 partials:
    best   max of the score (logits * inv + gate * gumbel)          (exact: max is exact)
    code   BIG - (smallest global id holding best)                  (exact float integers below 2**24)
    m      max(logits) * scale                                      (= max(logits * scale), scale > 0)
    s      sum(exp(logits * scale - m))                             (ScalarE exp, approximate)
Torch then merges the R * S * C partials per row with the engine's own tie rule (sampling._max_smallest_id), gathers
the raw logit at the winner from the logits as the engine does, forms the same payload [value, token, raw, lse] and
runs the engine's own all-gather and global merge (copied unchanged from sampling.py v0.1.0 lines 287-299).

Noise: computed in the kernel by the engine's hash (sampling.py:60-63, 90-119): xor, logical shifts and masks on the
Vector engine (bitvec ops, exact), the two multiplies as int32 tensor_tensor (GpSimd native integer arithmetic,
nki-docs api-nki-isa-tensor.md:1146-1149), u by exact float ops, -log(-log(u)) by two ScalarE nl.log activations.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

CHUNK = 2048          # free-axis columns per chunk; ~12 fp32/int32 tiles of 8 KiB per partition live at once
P_MAX = 128           # partitions (nki-docs api-nki-isa-tensor.md tensor_reduce "Tile size": must not exceed 128)
BIG = float(1 << 24)  # id code: BIG - id is exact for 0 <= id < 2**24 (sampling.py:276 refuses vocab >= 2**24)
NPART = 4             # partials per (partition, chunk)

# the engine's hash constants, sampling.py:60-63 (v0.1.0)
_LANE = 0x7FFFFFFF
_MIX_C1 = 0x05EBCA6B
_MIX_C2 = 0x42B2AE35
_COLUMN_SALT = 0x1E3779B9
_LOW23 = 0x7FFFFF
_INV_2_23 = 1.0 / 8388608.0


# ----------------------------------------------------------------------------------------------- layout (host ints)
def segments(rows: int, width: int) -> int:
    """S: the largest divisor of width with rows * S <= 128 (S = 1 when rows alone fill or exceed 128)."""
    if rows > P_MAX:
        raise RuntimeError(f"RLKIT sampler kernel: {rows} sampled rows exceed {P_MAX} partitions; this patch has no "
                           "row tiling (no silent fallback: unset RLKIT_SAMPLER_KERNEL)")
    best = 1
    for s in range(1, P_MAX // rows + 1):
        if width % s == 0:
            best = s
    return best


def n_chunks(seg_width: int) -> int:
    return -(-seg_width // CHUNK)


# ----------------------------------------------------------------------------------------------- the NKI kernel
def gumbel_partials(x, keys, inv, gate, scale, base):
    """x [P, F] fp32 logits (P = R * S partitions, F = W / S columns of a segment); per-partition columns [P, 1]:
    keys int32 (row noise key), inv fp32 (1/T, 1 for greedy rows), gate fp32 (-1 sampled, 0 greedy: the sign of the
    noise folded in), scale fp32 (lse scale), base fp32 (global id of the segment's column 0, exact below 2**24).
    Returns partials [P, C * 4] fp32 (C = ceil(F / CHUNK)): best, code, m, s per chunk.

    Plain function; sglang_neuron.backend.stack._as_nki_jit (stack.py:68-74) jits it, tests call nki.jit on it."""
    import nki.isa as nisa          # nki 0.7.0 package names as in lora_kernel/lora_merge.py:51-53
    import nki.language as nl

    P, F = x.shape
    C = -(-F // CHUNK)
    out = nl.ndarray((P, C * NPART), dtype=nl.float32, buffer=nl.shared_hbm)   # lora_merge.py:90 pattern

    pid = nl.program_id(0)          # lora_merge.py:92-93 (LNC: chunks dealt round-robin to programs)
    nprog = nl.num_programs(0)

    key_sb = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    inv_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    gate_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    scl_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    base_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=key_sb, src=keys[0:P, 0:1])
    nisa.dma_copy(dst=inv_sb, src=inv[0:P, 0:1])
    nisa.dma_copy(dst=gate_sb, src=gate[0:P, 0:1])
    nisa.dma_copy(dst=scl_sb, src=scale[0:P, 0:1])
    nisa.dma_copy(dst=base_sb, src=base[0:P, 0:1])

    # multiplier tiles for the int32 multiplies (iota with step 0: exact int32 constants), made once at the widest chunk
    # width; a chunk of n columns uses c1[0:P, 0:n]
    nmax = min(CHUNK, F)
    c1w = nl.ndarray((P, nmax), dtype=nl.int32, buffer=nl.sbuf)
    c2w = nl.ndarray((P, nmax), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=c1w, pattern=[[0, nmax]], offset=_MIX_C1, channel_multiplier=0)
    nisa.iota(dst=c2w, pattern=[[0, nmax]], offset=_MIX_C2, channel_multiplier=0)

    for c in range(C):
        if c % nprog != pid:        # the round-robin of lora_kernel/lora_merge.py:170-171 (simulator-tested at LNC 1, 2;
            continue                # this kernel only at LNC 1 in the simulator)
        c0 = c * CHUNK
        n = min(CHUNK, F - c0)

        c1 = c1w[0:P, 0:n]
        c2 = c2w[0:P, 0:n]

        xs = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=xs, src=x[0:P, c0:c0 + n])

        # global column ids: iota (api-nki-isa-utility.md:103-160) then + base (exact: values < 2**24)
        col = nl.ndarray((P, n), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=col, pattern=[[1, n]], offset=c0, channel_multiplier=0)
        idf = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=idf, data=col, op0=nl.add, operand0=base_sb)       # tensor.md:826 (fp32 math)
        nisa.tensor_copy(dst=col, src=idf)                                          # memory.md:525; exact ints

        h =nl.ndarray((P, n), dtype=nl.int32, buffer=nl.sbuf)
        t = nl.ndarray((P, n), dtype=nl.int32, buffer=nl.sbuf)

        def mix31(h):
            # sampling.py:90-98: h ^= h >> 16; h = (h * C1) & LANE; h ^= h >> 13; h = (h * C2) & LANE; h ^= h >> 16
            # right_shift = logical shift right (nki.api.shared.md:95); values stay in [0, 2**31) so it equals torch's >>
            nisa.tensor_scalar(dst=t, data=h, op0=nl.right_shift, operand0=16)
            nisa.tensor_tensor(dst=h, data1=h, data2=t, op=nl.bitwise_xor)
            # UNVERIFIED: int32 multiply wraps modulo 2**32 on GpSimd (doc says "exact results for all 32-bit integer
            # values", tensor.md:1146-1149, not what happens on overflow); the & LANE then keeps it modulo 2**31
            nisa.tensor_tensor(dst=h, data1=h, data2=c1, op=nl.multiply)
            nisa.tensor_scalar(dst=h, data=h, op0=nl.bitwise_and, operand0=_LANE)
            nisa.tensor_scalar(dst=t, data=h, op0=nl.right_shift, operand0=13)
            nisa.tensor_tensor(dst=h, data1=h, data2=t, op=nl.bitwise_xor)
            nisa.tensor_tensor(dst=h, data1=h, data2=c2, op=nl.multiply)
            nisa.tensor_scalar(dst=h, data=h, op0=nl.bitwise_and, operand0=_LANE)
            nisa.tensor_scalar(dst=t, data=h, op0=nl.right_shift, operand0=16)
            nisa.tensor_tensor(dst=h, data1=h, data2=t, op=nl.bitwise_xor)

        # sampling.py:117: h = mix31(row_key ^ mix31(columns ^ COLUMN_SALT))
        nisa.tensor_scalar(dst=h, data=col, op0=nl.bitwise_xor, operand0=_COLUMN_SALT)
        mix31(h)
        nisa.tensor_scalar(dst=h, data=h, op0=nl.bitwise_xor, operand0=key_sb)   # bitvec op with an int32 [P,1] vector
        mix31(h)
        nisa.tensor_scalar(dst=h, data=h, op0=nl.bitwise_and, operand0=_LOW23)

        # sampling.py:118-119: u = (float(h & 0x7FFFFF) + 0.5) * 2**-23 (exact), noise = -log(-log(u))
        u = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=u, src=h)
        nisa.tensor_scalar(dst=u, data=u, op0=nl.add, operand0=0.5, op1=nl.multiply, operand1=_INV_2_23)
        l1 = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=l1, op=nl.log, data=u)                       # scalar.md:206; log(u) in [-16.7, -6e-8]
        nisa.activation(dst=l1, op=nl.log, data=l1, scale=-1.0)          # log(-log(u)); input in the valid range
        # clamp l1 to a finite range first, so a greedy row (gate 0) cannot turn an inf from the device log into 0 * inf
        # = NaN (the engine's torch.where never touches the noise of a greedy row). Finite l1 lies in [-16.7, 2.8], so
        # the clamp changes no finite value. The clamp is one tensor_scalar with two ops (the form of the negm line
        # below); nl.minimum exists in nki 0.7.0 (WSL nki_sim_venv). UNVERIFIED on a device: a NaN input of the log
        # (max/min with NaN) is not covered by this clamp.
        nisa.tensor_scalar(dst=l1, data=l1, op0=nl.maximum, operand0=-1.0e4, op1=nl.minimum, operand1=1.0e4)
        # noise term = gate * log(-log u): -noise for sampled rows (gate -1), 0 for greedy rows (gate 0; l1 finite)
        nisa.tensor_scalar(dst=l1, data=l1, op0=nl.multiply, operand0=gate_sb)
        # score = logits * inv + noise (sampling.py:321: two roundings, mul then add, as torch does)
        sc = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sc, data=xs, op0=nl.multiply, operand0=inv_sb)
        nisa.tensor_tensor(dst=sc, data1=sc, data2=l1, op=nl.add)

        res = nl.ndarray((P, NPART), dtype=nl.float32, buffer=nl.sbuf)
        best = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=best, op=nl.maximum, data=sc, axis=1)     # tensor.md:728
        # smallest id holding best: max over (score == best) * (BIG - id)   (sampling.py:159-168 rule, _max_smallest_id)
        nisa.tensor_scalar(dst=sc, data=sc, op0=nl.equal, operand0=best)
        nisa.tensor_scalar(dst=idf, data=idf, op0=nl.subtract, operand0=BIG, reverse0=True)
        nisa.tensor_tensor(dst=sc, data1=sc, data2=idf, op=nl.multiply)
        code = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=code, op=nl.maximum, data=sc, axis=1)

        # log-sum-exp partial of logits * scale
        m = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=m, op=nl.maximum, data=xs, axis=1)
        nisa.tensor_scalar(dst=m, data=m, op0=nl.multiply, operand0=scl_sb)
        negm = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=negm, data=m, op0=nl.maximum, operand0=-3.0e38, op1=nl.multiply, operand1=-1.0)
        s = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=u, op=nl.exp, data=xs, bias=negm, scale=scl_sb, reduce_op=nl.add, reduce_res=s,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)          # scalar.md:206-260

        nisa.tensor_copy(dst=res[0:P, 0:1], src=best)
        nisa.tensor_copy(dst=res[0:P, 1:2], src=code)
        nisa.tensor_copy(dst=res[0:P, 2:3], src=m)
        nisa.tensor_copy(dst=res[0:P, 3:4], src=s)
        nisa.dma_copy(dst=out[0:P, c * NPART:(c + 1) * NPART], src=res)
    return out


# ----------------------------------------------------------------------------------------------- CPU reference
def gumbel_partials_reference(x, keys, inv, gate, scale, base):
    """The kernel's arithmetic in torch on the CPU, chunk by chunk, same argument and output layout.
    The noise uses the engine's own _mix31 formula (exact on CPU int32 as in sampling.py) and torch.log."""
    P, F = x.shape
    C = n_chunks(F)
    out = torch.empty((P, C * NPART), dtype=torch.float32)
    keys = keys.reshape(P, 1).to(torch.int32)

    def mix31(h):
        h = h ^ (h >> 16)
        h = (h * _MIX_C1) & _LANE
        h = h ^ (h >> 13)
        h = (h * _MIX_C2) & _LANE
        return h ^ (h >> 16)

    for c in range(C):
        c0 = c * CHUNK
        n = min(CHUNK, F - c0)
        xs = x[:, c0:c0 + n].to(torch.float32)
        idf = torch.arange(c0, c0 + n, dtype=torch.float32).reshape(1, n) + base.reshape(P, 1)
        col = idf.to(torch.int32)
        h = mix31(keys ^ mix31(col ^ _COLUMN_SALT))
        u = ((h & _LOW23).to(torch.float32) + 0.5) * _INV_2_23
        l1 = torch.log(-torch.log(u)) * gate.reshape(P, 1)
        sc = xs * inv.reshape(P, 1) + l1
        best = sc.amax(dim=1, keepdim=True)
        code = ((sc == best).to(torch.float32) * (BIG - idf)).amax(dim=1, keepdim=True)
        m = xs.amax(dim=1, keepdim=True) * scale.reshape(P, 1)
        negm = -torch.clamp(m, min=-3.0e38)
        s = torch.exp(xs * scale.reshape(P, 1) + negm).sum(dim=1, keepdim=True)
        out[:, c * NPART:(c + 1) * NPART] = torch.cat([best, code, m, s], dim=1)
    return out


# ----------------------------------------------------------------------------------------------- the device call
_logged = []


@torch._dynamo.assume_constant_result       # the engine's way to log at trace time: sampling_topk.py:106-108
def _log_first_use(rows: int, width: int, segs: int, part_shape: str, rank_info: str) -> None:
    if not _logged:
        _logged.append(1)
        logger.info("[rlkit] sampler by the kernel: rows=%d shard_width=%d segments=%d partitions=%d partials=%s "
                    "pid=%d %s", rows, width, segs, rows * segs, part_shape, os.getpid(), rank_info)


def _device_partials(x, keys, inv, gate, scale, base):
    from sglang_neuron.backend.stack import wrap_nki              # stack.py:37-48
    from sglang_neuron.sampling_topk import _programs              # sampling_topk.py:73-75 (LNC degree)

    return wrap_nki(gumbel_partials)[_programs()](x, keys, inv, gate, scale, base)


def sample_sharded_logits_kernel(logits, group, rank, vocabulary_size, temperatures, row_keys, top_logprobs,
                                 partials_fn=None):
    """Same contract and result as sampling.sample_sharded_logits (v0.1.0, lines 258-302) with top_logprobs == 0.
    partials_fn: the shard-local stage (default: the NKI kernel; the selftest passes the CPU reference)."""
    from sglang_neuron import sampling as S
    from sglang_neuron.parallel import collectives as neuron_collectives

    if top_logprobs:
        raise RuntimeError("RLKIT_SAMPLER_KERNEL=1 supports top logprobs 0 only (SGLANG_NEURON_MAX_TOP_LOGPROBS=0); "
                           "no silent fallback")
    if vocabulary_size >= 2**24:                                     # sampling.py:276-277
        raise ValueError("Packed float32 token IDs require a vocabulary smaller than 2**24")
    rows, width = logits.shape
    world = group.world_size
    logits = logits.float()
    rank64 = rank.to(torch.int64)
    if width * world > vocabulary_size:                              # padded columns exist: the engine's mask
        ids = torch.arange(width, dtype=torch.int64, device=logits.device) + rank64 * width
        logits = logits.masked_fill(ids.unsqueeze(0) >= vocabulary_size, float("-inf"))
    # per-row columns, as sampling._tempered_scores (sampling.py:316-319)
    temperature = temperatures.reshape(rows, 1).to(torch.float32)
    sampled = temperature > 0
    inverse = torch.where(sampled, 1.0 / torch.where(sampled, temperature, torch.ones_like(temperature)),
                          torch.ones_like(temperature))
    scale = torch.ones_like(inverse) if S.return_original_logprob() else inverse     # sampling.py:285
    gate = torch.where(sampled, torch.full_like(temperature, -1.0), torch.zeros_like(temperature))

    segs = segments(rows, width)
    seg_w = width // segs
    parts_n = rows * segs
    seg_index = torch.arange(segs, dtype=torch.int64, device=logits.device).repeat(rows)            # [R*S]
    base = (seg_index * seg_w + rank64 * width).to(torch.float32).reshape(parts_n, 1)

    def per_part(v):
        return v.reshape(rows, 1).repeat_interleave(segs, dim=0).reshape(parts_n, 1)

    keys = per_part(row_keys.to(torch.int32) & _LANE)
    x2 = logits.reshape(parts_n, seg_w)
    fn = partials_fn or _device_partials
    if partials_fn is None:
        _log_first_use(rows, width, segs, f"[{parts_n},{n_chunks(seg_w) * NPART}]",
                       f"rank_in_group={getattr(group, 'rank_in_group', '?')}")   # UNVERIFIED attribute name
    parts = fn(x2, keys, per_part(inverse), per_part(gate), per_part(scale), base)
    parts = parts.reshape(rows, segs * n_chunks(seg_w), NPART)

    # merge the partials of a row with the engine's tie rule (sampling._max_smallest_id, sampling.py:159-168)
    ids_all = BIG - parts[:, :, 1]
    value, token_f = S._max_smallest_id(parts[:, :, 0], ids_all, vocabulary_size - 1)
    token = token_f.to(torch.int64)
    local_id = (token - rank64 * width).clamp(0, width - 1)
    raw = logits.gather(1, local_id)                                    # sampling.py:284
    m_all, s_all = parts[:, :, 2], parts[:, :, 3]
    big_m = m_all.amax(dim=-1, keepdim=True)
    safe_m = torch.where(torch.isfinite(big_m), big_m, torch.zeros_like(big_m))
    lse = safe_m + torch.log((s_all * torch.exp(m_all - safe_m)).sum(dim=-1, keepdim=True))

    # from here on: sampling.py:287-299 (v0.1.0) unchanged, top_logprobs == 0
    payload = [value, token.float(), raw, lse]
    gathered = neuron_collectives.all_gather(group, torch.cat(payload, dim=-1), dim=-1)
    candidates = gathered.reshape(rows, group.world_size, 4 + 2 * top_logprobs)
    _, token, winner = S._shard_winner(candidates, vocabulary_size)
    raw = candidates[:, :, 2].masked_fill(~winner, float("-inf")).amax(dim=-1)
    global_lse = torch.logsumexp(candidates[:, :, 3], dim=-1)
    scale = scale.reshape(rows)
    return {
        "token_ids": token.squeeze(1).to(torch.int32),
        "logprobs": raw * scale - global_lse,
    }
