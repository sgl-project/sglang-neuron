#!/usr/bin/env python3
"""Selftest of the sampler kernel (sglang_neuron/rlkit_sampler_kernel.py and the guarded branch in sglang_neuron/sampling.py), CPU only.

    python tests/rlkit_sampler/selftest_sampler_kernel.py [--repo REPO] [--scratch DIR] [--seeds N] [--sim]

Builds a scratch package sglang_neuron/ holding the RELEASED sampling.py (git show v0.1.0:sglang_neuron/sampling.py,
or, when git cannot read the repository, released_sampling_v0.1.0.py in this folder), a stub of sglang_neuron.parallel.collectives (all_gather over a simulated TP4
group) and a copy of this repository's sglang_neuron/rlkit_sampler_kernel.py. Then, for random logits at the TP4 shard width 37,984:
  T1  same token id and logprob as the engine's own sample_sharded_logits (every rank), temperatures 0 / 0.6 / 1.0 and
      mixed rows, rows 1 / 8 / 32, many seeds (kernel stage = gumbel_partials_reference)
  T2  per-chunk best score of the reference equals the max of the engine's _tempered_scores over that chunk, bit for bit
      (this is the noise check: the score includes the Gumbel noise)
  T3  ties (equal maxima in several segments and ranks), constant rows, very negative and -inf logits
  T4  planted faults that MUST be caught: wrong noise column (base + 1), wrong row key, no tie rule (largest id)
  T5  this repository's sampling.py equals the released one except for the guarded branch (difflib)
  --sim  (WSL with nki 0.7.0) runs the NKI kernel in nki.simulate on reduced shapes against the reference
Exit 0 when every test passes. No device, no docker, no network.
"""
from __future__ import annotations

import argparse
import difflib
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEF_REPO = HERE.parents[1]                      # repository root: tests/rlkit_sampler/ -> tests/ -> root
PKG = DEF_REPO / "sglang_neuron"
WIDTH = 37984          # 151,936 / 4 (Qwen3 vocabulary at TP4)
WORLD = 4
VOCAB = WIDTH * WORLD
RESULTS = []

COLLECTIVES_STUB = '''"""Selftest stub: all_gather over a simulated group (group.gather(t) returns the world's payloads)."""
import torch


def all_gather(group, tensor, dim=-1):
    return group.gather(tensor, dim)
'''


def check(name, ok, info=""):
    RESULTS.append((name, ok, info))
    print(("PASS " if ok else "FAIL ") + name + (" | " + info if info else ""), flush=True)
    return ok


def build_scratch(repo: Path, scratch: Path):
    pkg = scratch / "sglang_neuron"
    (pkg / "parallel").mkdir(parents=True, exist_ok=True)
    for f in (pkg / "__init__.py", pkg / "parallel" / "__init__.py"):
        f.write_text("", encoding="utf-8", newline="\n")
    (pkg / "parallel" / "collectives.py").write_text(COLLECTIVES_STUB, encoding="utf-8", newline="\n")
    got = subprocess.run(["git", "-C", str(repo), "show", "v0.1.0:sglang_neuron/sampling.py"], capture_output=True)
    if got.returncode == 0:
        src = got.stdout
    else:                                   # e.g. git cannot read this worktree (WSL on a Windows checkout)
        src = (HERE / "released_sampling_v0.1.0.py").read_bytes()   # blob e1cf4227cb6d8a3b44283281e406026ae1f666d1
        print("git show failed; using released_sampling_v0.1.0.py (blob e1cf4227...) from this folder")
    (pkg / "sampling.py").write_bytes(src)
    shutil.copyfile(PKG / "rlkit_sampler_kernel.py", pkg / "rlkit_sampler_kernel.py")
    sys.path.insert(0, str(scratch))
    return src


class Group:
    """TP group of WORLD ranks. Pass 1 records each rank's payload; pass 2 returns them all (payloads are pure)."""

    def __init__(self, store, rank):
        self.world_size, self.rank_in_group, self.store, self.rank = WORLD, rank, store, rank

    def gather(self, t, dim):
        self.store[self.rank] = t.clone()
        if len(self.store) < WORLD:
            return torch.cat([t] * WORLD, dim=dim)      # pass 1 placeholder, its result is not used
        return torch.cat([self.store[r] for r in range(WORLD)], dim=dim)


def run_tp(fn, logits_full, temps, keys):
    """fn(shard, group, rank_tensor) per rank, twice (record, then real gather). Returns the per-rank results."""
    store = {}
    shards = [logits_full[:, r * WIDTH:(r + 1) * WIDTH] for r in range(WORLD)]
    for r in range(WORLD):
        fn(shards[r], Group(store, r), torch.tensor(r))
    return [fn(shards[r], Group(store, r), torch.tensor(r)) for r in range(WORLD)]


def engine_fn(S):
    def f(shard, g, rank, temps, keys):
        return S.sample_sharded_logits(shard, g, rank, VOCAB, temps, keys, top_logprobs=0)
    return f


def kernel_fn(K, partials):
    def f(shard, g, rank, temps, keys):
        return K.sample_sharded_logits_kernel(shard, g, rank, VOCAB, temps, keys, 0, partials_fn=partials)
    return f


def compare(S, K, logits, temps, keys, partials=None):
    partials = partials or K.gumbel_partials_reference
    e = run_tp(lambda sh, g, r: engine_fn(S)(sh, g, r, temps, keys), logits, temps, keys)
    k = run_tp(lambda sh, g, r: kernel_fn(K, partials)(sh, g, r, temps, keys), logits, temps, keys)
    same_ids = all(torch.equal(a["token_ids"], b["token_ids"]) for a, b in zip(e, k))
    dlp = max(float((a["logprobs"] - b["logprobs"]).abs().max()) for a, b in zip(e, k))
    return same_ids, dlp, e, k


def make_temps(kind, rows, gen):
    if kind == "mixed":
        choices = torch.tensor([0.0, 0.6, 1.0, 0.8])
        return choices[torch.randint(0, 4, (rows,), generator=gen)]
    return torch.full((rows,), float(kind))


def t1(S, K, seeds):
    gen = torch.Generator().manual_seed(1234)
    worst = 0.0
    n = 0
    all_ok = True
    for rows in (1, 8, 32):
        for kind in ("0", "0.6", "1.0", "mixed"):
            for sd in range(seeds):
                logits = (torch.randn(rows, VOCAB, generator=gen) * 4.0).to(torch.bfloat16).float()
                keys = S.noise_row_keys(torch.randint(0, 2**31 - 1, (rows,), generator=gen, dtype=torch.int64),
                                        torch.randint(0, 4096, (rows,), generator=gen))
                temps = make_temps(kind, rows, gen)
                ok, dlp, _, _ = compare(S, K, logits, temps, keys)
                worst = max(worst, dlp)
                n += 1
                if not ok or dlp > 1e-5:
                    all_ok = False
                    check(f"T1 rows={rows} T={kind} seed={sd}", False, f"same ids {ok}, max |dlogprob| {dlp:.3g}")
    return check("T1 ids equal the engine's, logprob within 1e-5 (CPU)", all_ok,
                 f"{n} cases x {WORLD} ranks, rows 1/8/32, T 0/0.6/1.0/mixed; max |dlogprob| {worst:.3g}")


def t2(S, K):
    gen = torch.Generator().manual_seed(7)
    ok_all = True
    for rows in (1, 8, 32):
        logits = (torch.randn(rows, WIDTH, generator=gen) * 4.0).float()
        temps = make_temps("mixed", rows, gen)
        keys = S.noise_row_keys(torch.randint(0, 2**31 - 1, (rows,), generator=gen, dtype=torch.int64),
                                torch.randint(0, 4096, (rows,), generator=gen))
        for r in range(WORLD):
            rank = torch.tensor(r)
            _, _, _, score = S._tempered_scores(logits, rank, VOCAB, temps, keys)
            segs = K.segments(rows, WIDTH)
            seg_w = WIDTH // segs
            temperature = temps.reshape(rows, 1)
            sampled = temperature > 0
            inv = torch.where(sampled, 1.0 / torch.where(sampled, temperature, torch.ones_like(temperature)),
                              torch.ones_like(temperature))
            gate = torch.where(sampled, -torch.ones_like(temperature), torch.zeros_like(temperature))
            rep = lambda v: v.reshape(rows, 1).repeat_interleave(segs, 0)          # noqa: E731
            base = (torch.arange(segs).repeat(rows) * seg_w + r * WIDTH).float().reshape(-1, 1)
            parts = K.gumbel_partials_reference(logits.reshape(rows * segs, seg_w), rep(keys & 0x7FFFFFFF), rep(inv),
                                                rep(gate), rep(inv), base)
            sc = score.reshape(rows * segs, seg_w)
            for c in range(K.n_chunks(seg_w)):
                c0 = c * K.CHUNK
                want = sc[:, c0:c0 + K.CHUNK].amax(dim=1)
                got = parts[:, c * K.NPART]
                if not torch.equal(want.view(torch.int32), got.view(torch.int32)):
                    ok_all = False
    return check("T2 per-chunk best score bit-equal to the engine's _tempered_scores (noise included)", ok_all,
                 "rows 1/8/32, 4 ranks, mixed temperatures")


def t3(S, K):
    gen = torch.Generator().manual_seed(99)
    ok_all, info = True, []
    for rows in (1, 8, 32):
        keys = torch.randint(0, 2**31 - 1, (rows,), generator=gen).to(torch.int32)
        # equal maxima in several segments and ranks; greedy rows must take the smallest id
        logits = torch.randn(rows, VOCAB, generator=gen)
        for col in (WIDTH * 3 + 5, WIDTH + 1200, 2400, 40000):
            logits[:, col] = 50.0
        ok, dlp, e, _ = compare(S, K, logits, torch.zeros(rows), keys)
        ok = ok and int(e[0]["token_ids"][0]) == 2400
        info.append(f"ties r{rows} {ok}")
        ok_all &= ok
        # constant rows (every column ties)
        ok, dlp, e, _ = compare(S, K, torch.zeros(rows, VOCAB), make_temps("mixed", rows, gen), keys)
        info.append(f"const r{rows} {ok}")
        ok_all &= ok and dlp <= 1e-5
        # very negative logits and -inf columns
        lg = torch.randn(rows, VOCAB, generator=gen) * 3 - 1e30
        lg[:, ::7] = float("-inf")
        lg[:, 12345] = -1e29
        ok, dlp, _, _ = compare(S, K, lg, make_temps("mixed", rows, gen), keys)
        info.append(f"neg r{rows} {ok} d{dlp:.2g}")
        ok_all &= ok and (dlp <= 1e-5 or dlp != dlp)
    return check("T3 ties, constant rows, very negative and -inf logits", ok_all, ", ".join(info))


def t4(S, K):
    ref = K.gumbel_partials_reference
    faults = {
        "wrong noise column (base + 1)": lambda x, k, i, g, s, b: ref(x, k, i, g, s, b + 1.0),
        "wrong row key (key ^ 1)": lambda x, k, i, g, s, b: ref(x, k ^ 1, i, g, s, b),
        "no tie rule (largest id)": lambda x, k, i, g, s, b: _largest_id(ref(x, k, i, g, s, b), b, x.shape[1], K),
    }
    gen = torch.Generator().manual_seed(5)
    ok_all = True
    for name, fn in faults.items():
        caught = False
        for rows in (1, 8, 32):
            logits = (torch.randn(rows, VOCAB, generator=gen) * 4.0)
            if "tie" in name:
                logits[:, 100] = 60.0
                logits[:, 90000] = 60.0
                temps = torch.zeros(rows)
            else:
                temps = torch.full((rows,), 1.0)
            keys = torch.randint(0, 2**31 - 1, (rows,), generator=gen).to(torch.int32)
            ok, _, _, _ = compare(S, K, logits, temps, keys, partials=fn)
            caught |= not ok
        ok_all &= check(f"T4 planted fault caught: {name}", caught)
    return ok_all


def _largest_id(parts, base, width, K):
    """Fault: per chunk, take the LARGEST id among the maxima (code of the largest = BIG - largest)."""
    out = parts.clone()
    # replace code with BIG - (last column of the chunk) where any max exists: a wrong tie rule
    for c in range(K.n_chunks(width)):
        n = min(K.CHUNK, width - c * K.CHUNK)
        out[:, c * K.NPART + 1] = K.BIG - (base.reshape(-1) + c * K.CHUNK + n - 1)
    return out


def t5(src_orig: bytes):
    new = (PKG / "sampling.py").read_bytes()
    a, b = src_orig.decode().splitlines(), new.decode().splitlines()
    diff = [d for d in difflib.unified_diff(a, b, lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")]
    removed = [d for d in diff if d.startswith("-")]
    added = [d[1:] for d in diff if d.startswith("+")]
    guarded = all(("RLKIT" in x or "rlkit" in x or x.strip() in ("",) or "row_keys, top_logprobs)" in x) for x in added)
    ok = not removed and guarded and len(added) <= 10 and b"\r" not in new
    for d in diff:
        print("   diff " + d)
    return check("T5 patched sampling.py = released + guarded branch only", ok,
                 f"removed {len(removed)}, added {len(added)}")


def sim_tests(K):
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")       # lora_kernel/test_merge_kernel_sim.py:15-16
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    import nki
    import numpy as np

    kern = nki.jit(K.gumbel_partials)
    gen = torch.Generator().manual_seed(3)
    ok_all = True
    for P, F in ((4, 300), (32, 1187), (128, 2500)):
        x = (torch.randn(P, F, generator=gen) * 4.0).float()
        x[0, 7] = x[0, 9] = 40.0
        keys = torch.randint(0, 2**31 - 1, (P, 1), generator=gen).to(torch.int32)
        inv = torch.full((P, 1), 1.0 / 0.6)
        inv[0] = 1.0
        gate = torch.full((P, 1), -1.0)
        gate[0] = 0.0
        scale = inv.clone()
        base = (torch.arange(P) * F + 37984).float().reshape(P, 1)
        ref = K.gumbel_partials_reference(x, keys, inv, gate, scale, base).numpy()
        t0 = time.time()
        got = np.asarray(nki.simulate(kern)(x.numpy(), keys.numpy(), inv.numpy(), gate.numpy(), scale.numpy(),
                                            base.numpy()))
        dt = time.time() - t0
        C = K.n_chunks(F)
        r4, g4 = ref.reshape(P, C, 4), got.reshape(P, C, 4)
        best_eq = int((r4[..., 0].view(np.int32) == g4[..., 0].view(np.int32)).sum())
        code_eq = int((r4[..., 1] == g4[..., 1]).sum())
        m_eq = int((r4[..., 2].view(np.int32) == g4[..., 2].view(np.int32)).sum())
        s_rel = float(np.max(np.abs(r4[..., 3] - g4[..., 3]) / np.maximum(np.abs(r4[..., 3]), 1e-30)))
        n = P * C
        d_best = float(np.max(np.abs(r4[..., 0] - g4[..., 0])))
        greedy_eq = bool((r4[0, :, 0].view(np.int32) == g4[0, :, 0].view(np.int32)).all())
        # the simulator's nl.log need not be torch.log bit for bit: the noise term is compared on its own below
        ok = code_eq == n and m_eq == n and s_rel < 1e-4 and greedy_eq and d_best < 1e-5
        ok_all &= check(f"SIM P={P} F={F} chunks={C}", ok,
                        f"best bit-equal {best_eq}/{n} (max |d| {d_best:.3g}; greedy row bit-equal {greedy_eq}), "
                        f"id code equal {code_eq}/{n}, m bit-equal {m_eq}/{n}, max rel diff of s {s_rel:.3g}, "
                        f"sim {dt:.1f}s")
    # the noise alone: one partition per hash value, gate -1, inv 1, logits 0 -> best = max noise of a 1-column chunk
    P, F = 128, 1
    x = torch.zeros(P, F)
    keys = torch.randint(0, 2**31 - 1, (P, 1), generator=gen).to(torch.int32)
    ones = torch.ones(P, 1)
    base = torch.arange(P).float().reshape(P, 1) * 977.0
    ref = K.gumbel_partials_reference(x, keys, ones, -ones, ones, base).numpy()[:, 0]
    got = np.asarray(nki.simulate(kern)(x.numpy(), keys.numpy(), ones.numpy(), (-ones).numpy(), ones.numpy(),
                                        base.numpy()))[:, 0]
    ulp = np.abs(ref.view(np.int32).astype(np.int64) - got.view(np.int32).astype(np.int64))
    check("SIM noise of single columns vs torch (informational: sim log vs torch.log)", True,
          f"bit-equal {int((ulp == 0).sum())}/{P}, max ulp {int(ulp.max())}")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(DEF_REPO))
    ap.add_argument("--scratch", default=os.environ.get("RLKIT_SCRATCH", str(Path(tempfile.gettempdir()) / "selftest_sampler_scratch")))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--sim-only", action="store_true")
    a = ap.parse_args()
    global torch
    import torch  # noqa: F811
    src = build_scratch(Path(a.repo), Path(a.scratch))
    S = importlib.import_module("sglang_neuron.sampling")
    K = importlib.import_module("sglang_neuron.rlkit_sampler_kernel")
    t0 = time.time()
    if not a.sim_only:
        t5(src)
        t2(S, K)
        t3(S, K)
        t4(S, K)
        t1(S, K, a.seeds)
    if a.sim or a.sim_only:
        sim_tests(K)
    bad = [r for r in RESULTS if not r[1]]
    print(f"SELFTEST_SAMPLER_KERNEL {'OK' if not bad else 'FAILED'}: {len(RESULTS) - len(bad)} pass, {len(bad)} fail, "
          f"{time.time() - t0:.0f}s")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
