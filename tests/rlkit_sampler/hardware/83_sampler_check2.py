#!/usr/bin/env python3
"""Second hardware check of the sampler kernel (2026-11-24), after run r10 of 82_sampler_kernel_hw.py:
the kernel server (B) was as fast as the greedy server and ran the kernel on every rank, but at temperature 0 it
differed from the released temperature server (A) on 3 of 20 prompts (largest logprob difference 0.10), and the
sampled requests of r10 were refused (HTTP 400: they did not carry top_p 1, top_k -1, min_p 0, see rollout_client.chat).

Questions, each with its control:
  Q1  temperature 0: is the A-B difference larger than the difference between two servers that should agree?
      Controls: A against A2 (same image, same flags, same sampler, another chip) and A against C (greedy sampler).
  Q2  sampled, same request seed: does B give A's tokens? Control: A against A2 with the same seed, A against itself
      with the same seed, A with another seed.
  Q3  sampled, first token of one prompt, N draws: is B's distribution A's? Control: A against A2. Also each
      server's observed frequencies against the probabilities the server itself reports (logprob of the drawn token).

    python 83_sampler_check2.py --results DIR --tag r11 [--draws 3000] [--dry-run]

Servers as in 82 (cores 20-23, 24-27, 28-31) plus A2 on cores 32-35 with A's compile cache. Result:
<results>/83_sampler_check2_<tag>.jsonl ending {"done": true}. Every server is stopped in finally."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import signal
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load("hw82", "82_sampler_kernel_hw.py")
O, rc, server_ctl = H.O, H.rc, H.server_ctl
SERVERS = dict(H.SERVERS)
SERVERS["A2"] = {"cores": "32-35", "port": 30094, "sampler": "temperature", "patch": False,
                 "cache": "sgl_cache_v0.1.0_samplerk_A"}
H.SERVERS.update(SERVERS)                      # server_env reads H.SERVERS
FULL = {"top_p": 1.0, "top_k": -1, "min_p": 0.0}   # full-vocabulary sampling, as rollout_client.chat sends it
DIST_PROMPT = "Pick a number between 1 and 10. Answer with the number only."


def chat(srv, prompt, temperature, seed=None, max_tokens=48):
    body = {"model": srv.served_name(), "messages": O.C.messages_for(prompt), "max_tokens": max_tokens,
            "temperature": temperature, "logprobs": True, "chat_template_kwargs": rc.THINK_OFF}
    if temperature > 0:
        body.update(FULL)
    if seed is not None:
        body["seed"] = seed
    code, out, _ = rc.post(srv.base_url + "/v1/chat/completions", body, 300)
    if code != 200 or not isinstance(out, dict) or not out.get("choices"):
        return {"ok": False, "status": code, "error": str(out)[:300]}
    ch = out["choices"][0]
    content = ((ch.get("logprobs") or {}).get("content")) or []
    return {"ok": True, "text": (ch.get("message") or {}).get("content"),
            "tokens": [c.get("token") for c in content], "logprobs": [c.get("logprob") for c in content]}


def pair_stats(xs, ys):
    """xs, ys: lists of chat answers to the same requests. Differences are taken before the first different token."""
    same, forks, maxes, sums, n, bit, errors = 0, [], [], 0.0, 0, 0, 0
    for a, b in zip(xs, ys):
        if not (a["ok"] and b["ok"]):
            errors += 1
            continue
        m = min(len(a["tokens"]), len(b["tokens"]))
        first = next((i for i in range(m) if a["tokens"][i] != b["tokens"][i]), None)
        if first is None and len(a["tokens"]) == len(b["tokens"]):
            same += 1
        else:
            forks.append(m if first is None else first)
        upto = m if first is None else first
        d = [abs(float(x) - float(y)) for x, y in zip(a["logprobs"][:upto], b["logprobs"][:upto])
             if x is not None and y is not None]
        if d:
            maxes.append(max(d))
            sums += sum(d)
            n += len(d)
            bit += sum(1 for x in d if x == 0.0)
    return {"requests": len(xs), "errors": errors, "same_tokens": same, "fork_positions": forks,
            "tokens_compared_before_fork": n, "bit_equal_logprobs": bit,
            "max_abs_dlogprob_before_fork": max(maxes) if maxes else None,
            "mean_abs_dlogprob_before_fork": (sums / n) if n else None}


def run_all(srv, prompts, temperature, seeds=None, workers=1):
    jobs = [(p, None) for p in prompts] if seeds is None else [(p, s) for p in prompts for s in seeds]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda j: chat(srv, j[0], temperature, j[1]), jobs))


def draws(srv, n, seed0, temperature=1.0):
    with ThreadPoolExecutor(max_workers=32) as ex:
        rs = list(ex.map(lambda i: chat(srv, DIST_PROMPT, temperature, seed0 + i, max_tokens=1), range(n)))
    toks, probs, bad = Counter(), {}, 0
    for r in rs:
        if not r["ok"] or not r["tokens"]:
            bad += 1
            continue
        t = r["tokens"][0]
        toks[t] += 1
        if r["logprobs"][0] is not None:
            probs.setdefault(t, []).append(math.exp(float(r["logprobs"][0])))
    return toks, {t: sum(v) / len(v) for t, v in probs.items()}, bad


def tv(c1, c2):
    n1, n2 = sum(c1.values()), sum(c2.values())
    if not n1 or not n2:
        return None
    return 0.5 * sum(abs(c1.get(t, 0) / n1 - c2.get(t, 0) / n2) for t in set(c1) | set(c2))


def chi2_against_reported(counts, probs):
    """Observed first-token counts against the probability the server reported for each drawn token."""
    n = sum(counts.values())
    stat, cells, mass = 0.0, 0, 0.0
    for t, c in counts.items():
        p = probs.get(t)
        if p is None or p * n < 5:
            continue
        stat += (c - p * n) ** 2 / (p * n)
        cells += 1
        mass += p
    return {"draws": n, "cells_with_expected_5_or_more": cells, "probability_mass_of_those_cells": round(mass, 4),
            "chi2": round(stat, 2), "degrees_of_freedom_about": max(0, cells - 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--ready-cap", type=float, default=2400)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    data_root = os.environ.get("DATA_ROOT", str(Path.home() / "w"))
    models_root = os.environ.get("MODELS_ROOT", str(Path(data_root) / "models"))
    model_dir = Path(models_root) / a.model
    keys = ["A", "A2", "B", "C"]
    names = {k: f"samplerk_{a.tag}_{k}" for k in keys}
    if a.dry_run:
        for k in keys:
            print(k, names[k], SERVERS[k], H.server_env(k, data_root))
        print("CHECK2_DONE dry-run")
        return 0
    a.results.mkdir(parents=True, exist_ok=True)
    ev = H.Events(a.results / f"83_sampler_check2_{a.tag}.jsonl")
    backend = dict(server_ctl.load_backend(HERE / "backends.json", "sglang_neuron"), ready_timeout_s=a.ready_cap)
    servers = {k: server_ctl.Server(backend, SERVERS[k]["port"], a.results / "server_logs", name=names[k],
                                    env=H.server_env(k, data_root)) for k in keys}

    def on_signal(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)
    ev.write(event="open", tag=a.tag, servers={k: dict(SERVERS[k], env=H.server_env(k, data_root)) for k in keys})
    status = "failed"
    try:
        with ThreadPoolExecutor(max_workers=len(keys)) as ex:
            started = dict(zip(keys, ex.map(lambda k: servers[k].start(model_dir), keys)))
        for k in keys:
            ev.write(event="server_ready", server=k, **started[k])
        up = [k for k in keys if started[k].get("ready_s") is not None]
        if not {"A", "B"} <= set(up):
            ev.write(event="stop_rule", reason="A or B not ready", up=up)
            return 1
        S = servers
        # Q1: temperature 0
        t0 = {k: run_all(S[k], H.PROMPTS, 0.0) for k in up}
        for x, y in (("A", "A2"), ("A", "B"), ("A", "C"), ("B", "C")):
            if x in up and y in up:
                ev.write(event="q1_temperature0", pair=f"{x} vs {y}", **pair_stats(t0[x], t0[y]))
        # Q2: sampled, same seeds
        seeds = [11, 222, 3333]
        s1 = {k: run_all(S[k], H.PROMPTS, 0.8, seeds) for k in up if k != "C"}
        a_again = run_all(S["A"], H.PROMPTS, 0.8, seeds)
        a_other = run_all(S["A"], H.PROMPTS, 0.8, [s + 100000 for s in seeds])
        ev.write(event="q2_sampled", pair="A vs A again, same seeds", **pair_stats(s1["A"], a_again))
        ev.write(event="q2_sampled", pair="A vs A, other seeds", **pair_stats(s1["A"], a_other))
        if "A2" in up:
            ev.write(event="q2_sampled", pair="A vs A2, same seeds", **pair_stats(s1["A"], s1["A2"]))
        ev.write(event="q2_sampled", pair="A vs B, same seeds", **pair_stats(s1["A"], s1["B"]))
        b_again = run_all(S["B"], H.PROMPTS, 0.8, seeds)
        ev.write(event="q2_sampled", pair="B vs B again, same seeds", **pair_stats(s1["B"], b_again))
        # Q3: distribution of the first token
        d = {k: draws(S[k], a.draws, 1000) for k in up if k != "C"}
        d["A_b"] = draws(S["A"], a.draws, 500000)          # A again with other seeds: the size of sampling noise
        for k, (cnt, pr, bad) in d.items():
            ev.write(event="q3_first_token", server=k, errors=bad, distinct_tokens=len(cnt),
                     top=[[t, c] for t, c in cnt.most_common(8)], against_reported=chi2_against_reported(cnt, pr))
        ev.write(event="q3_total_variation", pair="A vs A again, other seeds", tv=tv(d["A"][0], d["A_b"][0]))
        if "A2" in d:
            ev.write(event="q3_total_variation", pair="A vs A2", tv=tv(d["A"][0], d["A2"][0]))
        ev.write(event="q3_total_variation", pair="A vs B", tv=tv(d["A"][0], d["B"][0]))
        ev.write(event="q3_total_variation", pair="A again vs B", tv=tv(d["A_b"][0], d["B"][0]))
        status = "ok"
        return 0
    except BaseException as e:  # noqa: BLE001
        ev.write(event="error", error=f"{type(e).__name__}: {e}"[:400])
        return 1
    finally:
        for k in keys:
            try:
                ev.write(event="stop", server=k, seconds=servers[k].stop())
            except Exception as e:  # noqa: BLE001
                ev.write(event="stop_error", server=k, error=str(e)[:200])
        ev.write(event="end", status=status)
        ev.write(done=True)
        print(("CHECK2_DONE" if status == "ok" else "CHECK2_FAILED") + f" status={status}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
