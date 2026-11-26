#!/usr/bin/env python3
"""Hardware run of the sampler kernel (sglang_neuron/rlkit_sampler_kernel.py) against the released temperature and greedy
samplers, Qwen3-0.6B TP4, three servers up at once on three chips. Run on 2026-11-24 as r10 (0.6B) and r12 (8B, through
84_sampler_kernel_8b.py). It needs the launcher kit of the companion repository (see README.md in the parent folder).

    python 82_sampler_kernel_hw.py --results DIR --tag TAG [--model Qwen3-0.6B] [--only-correctness] [--skip C]
                                   [--ready-cap 3000] [--dry-run]

Servers (srv_sglang_neuron.sh through server_ctl; containers samplerk_<tag>_A/B/C):
  A  release image, SGLANG_NEURON_SAMPLER=temperature, cores 20-23, port 30091, cache $DATA_ROOT/sgl_cache_v0.1.0_samplerk_A
  B  + SGL_PATCH_DIR=<patch folder>, SGL_ENV=RLKIT_SAMPLER_KERNEL=1, temperature, cores 24-27, port 30092,
     cache $DATA_ROOT/sgl_cache_v0.1.0_sampler_kernel (its own: kernel sources are part of the cache digest,
     backend/compile_cache.py:40 of v0.1.0)
  C  release image, greedy, cores 28-31, port 30093, cache $DATA_ROOT/sgl_cache_v0.1.0_samplerk_C
  Own cache folders for A and C too, so no two servers compile into one folder at the same time.
  Flags of all three (SGL_EXTRA): FLAGS below.
Steps: start all three in parallel (ready cap each); a server that is not ready is recorded with its last log lines and
the script goes on with what is up. Correctness (A and B both up): (i) 20 prompts, temperature 0, 48 tokens, token
logprobs: B's text and tokens equal A's, every logprob within LOGPROB_TOL (largest |d| reported; bit-equal count too);
(ii) seed check on A (same seed twice must give the same text, two seeds must differ somewhere; else "seed not
effective" and (ii) is inconclusive), then 20 prompts x 3 seeds at temperature 0.8, 48 tokens: B's tokens equal A's;
(iii) "sampler by the kernel" in B's log (count of lines and of distinct pids) and absent from A's. Speed, one server at a
time, the others up and idle: 22b_llmbench_grid.sh LB_PROMPTS=1024 LB_CONC=1,8,32 LB_OUT=256 LB_REQ=64 LB_WARM=4 on
A, B, C, then A and B again. Every server stopped in finally, then docker rm -f of the three names.
(iv) 8 and 32 near-equal-length prompts at once to A and B, temperature 0: share of equal token lists, information
only. (i) and (ii) send one request at a time, so they cover R = 1 only; the C > 1 chunk path and the LNC 2 split are
seen on the device only through (iv).
Result: <results>/82_sampler_kernel_<tag>.jsonl (events, last line {"done": true}) and 82_sampler_kernel_<tag>.md.
Exit: 0 correctness passed and every bench returned cells (benches skipped with --only-correctness), 1 a correctness
check failed, B not up, a bench returned no cells, or an unexpected error, 2 setup error. Last stdout line
SAMPLERK_DONE ... or SAMPLERK_FAILED ... with seed_inconclusive=<bool> kernel_mismatch=<bool> concurrent_same_tokens=
<{n: equal}>. SIGTERM and SIGHUP stop the run through the same finally as Ctrl-C (servers stopped, containers removed).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


O = _load("opt81", "81_opt_sweep.py")       # read_rows, single_cell, tail_lines, force_remove, parallel, S76, SCRIPTS
rc, server_ctl = O.rc, O.server_ctl

PATCH = HERE / "patches" / "sglang_neuron_v0.1.0_sampler_kernel"
FLAGS = ("--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 131072 "
         "--cuda-graph-bs-decode 1 8 32")
SERVERS = {
    "A": {"cores": "20-23", "port": 30091, "sampler": "temperature", "patch": False, "cache": "sgl_cache_v0.1.0_samplerk_A"},
    "B": {"cores": "24-27", "port": 30092, "sampler": "temperature", "patch": True, "cache": "sgl_cache_v0.1.0_sampler_kernel"},
    "C": {"cores": "28-31", "port": 30093, "sampler": "greedy", "patch": False, "cache": "sgl_cache_v0.1.0_samplerk_C"},
}
KERNEL_LINE = "sampler by the kernel"          # rlkit_sampler_kernel.py _log_first_use
MAX_TOKENS = 48
LOGPROB_TOL = 1e-3                             # SAMPLER_KERNEL_NOTES.md "Equality promised"
SEEDS = [11, 222, 3333]
SAMPLED_T = 0.8
CONC_SIZES = (8, 32)                           # concurrent requests of check (iv)
BENCH_ORDER = ["A", "B", "C", "A", "B"]
LB = {"LB_PROMPTS": "1024", "LB_CONC": "1,8,32", "LB_OUT": "256", "LB_REQ": "64", "LB_WARM": "4"}
PROMPTS = list(O.GUARD_PROMPTS) + [
    "Explain in two sentences why the sky is blue.",
    "Write a haiku about a train crossing snow.",
    "List three prime numbers larger than 100.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Translate 'good morning, how are you' into French.",
    "A car travels 150 km in 2.5 hours. What is its average speed?",
    "Give one advantage and one drawback of solar power.",
    "Describe the water cycle in three steps.",
    "What does a compiler do? Answer briefly.",
    "Name four planets of the solar system in order from the sun.",
    "If a shirt costs 40 dollars after a 20 percent discount, what was the original price?",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "Why do leaves change colour in autumn?",
    "Write a Python one-liner that reverses a string.",
    "What is 17 squared minus 13 squared?",
]
assert len(PROMPTS) == 20


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Events:
    def __init__(self, path):
        self.path = path

    def write(self, **kv):
        kv.setdefault("t", utc())
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(kv, default=str) + "\n")
        print(json.dumps(kv, default=str)[:400], flush=True)


def server_env(key, data_root):
    s = SERVERS[key]
    env = {"SGL_CACHE": str(Path(data_root) / s["cache"]), "SGL_CORES": s["cores"], "SGL_SAMPLER": s["sampler"],
           "SGL_EXTRA": FLAGS, "SGL_SCRIPT": O.SCRIPTS["Qwen3-0.6B"], "SGL_TOP_LOGPROBS": "",
           "SGL_PATCH_DIR": str(PATCH) if s["patch"] else "", "SGL_ENV": "RLKIT_SAMPLER_KERNEL=1" if s["patch"] else ""}
    return env


def chat(srv, prompt, temperature, seed=None):
    body = {"model": srv.served_name(), "messages": O.C.messages_for(prompt), "max_tokens": MAX_TOKENS,
            "temperature": temperature, "logprobs": True, "chat_template_kwargs": rc.THINK_OFF}
    if seed is not None:
        body["seed"] = seed                       # rollout_client.py:56-57; reaches SamplingParams.sampling_seed? UNVERIFIED
    code, out, _ = rc.post(srv.base_url + "/v1/chat/completions", body, 300)
    if code != 200 or not isinstance(out, dict) or not out.get("choices"):
        return {"ok": False, "status": code, "error": str(out)[:300]}
    ch = out["choices"][0]
    content = ((ch.get("logprobs") or {}).get("content")) or []
    return {"ok": True, "text": (ch.get("message") or {}).get("content"),
            "tokens": [c.get("token") for c in content], "logprobs": [c.get("logprob") for c in content]}


def compare(a, b):
    if not (a["ok"] and b["ok"]):
        return {"same_text": False, "same_tokens": False, "max_abs_dlogprob": None, "bit_equal": 0, "n": 0,
                "first_diff": None}
    n = min(len(a["tokens"]), len(b["tokens"]))
    first = next((i for i in range(n) if a["tokens"][i] != b["tokens"][i]), None)
    if first is None and len(a["tokens"]) != len(b["tokens"]):
        first = n
    m = n if first is None else first
    d = [abs(float(x) - float(y)) for x, y in zip(a["logprobs"][:m], b["logprobs"][:m]) if x is not None and y is not None]
    return {"same_text": a["text"] == b["text"], "same_tokens": first is None, "first_diff": first,
            "max_abs_dlogprob": max(d) if d else None, "bit_equal": sum(1 for x in d if x == 0.0), "n": len(d)}


def docker_logs(srv):
    try:
        return subprocess.run(srv._fill(srv.b["logs"]), cwd=HERE, env=srv.env, capture_output=True, text=True,
                              timeout=120).stdout
    except Exception as e:  # noqa: BLE001
        return f"(logs failed: {e})"


def correctness(ev, A, B):
    ok = True
    # (i) temperature 0
    rows = []
    for i, q in enumerate(PROMPTS):
        a, b = chat(A, q, 0.0), chat(B, q, 0.0)
        c = compare(a, b)
        rows.append(c)
        ev.write(event="greedy_pair", i=i, **c)
    worst = max([r["max_abs_dlogprob"] or 0.0 for r in rows] + [0.0])
    p1 = all(r["same_text"] and r["same_tokens"] for r in rows) and worst <= LOGPROB_TOL and all(r["n"] for r in rows)
    ev.write(event="check_i", ok=p1, same=sum(r["same_tokens"] for r in rows), of=len(rows), max_abs_dlogprob=worst,
             bit_equal=sum(r["bit_equal"] for r in rows), compared=sum(r["n"] for r in rows), tol=LOGPROB_TOL)
    ok &= p1
    # (ii) seed effective on A?
    s1, s2, s3 = (chat(A, PROMPTS[5], SAMPLED_T, seed=s) for s in (SEEDS[0], SEEDS[0], SEEDS[1]))
    effective = s1["ok"] and s2["ok"] and s3["ok"] and s1["tokens"] == s2["tokens"] and s1["tokens"] != s3["tokens"]
    ev.write(event="seed_check", effective=effective, same_seed_same=s1.get("tokens") == s2.get("tokens"),
             other_seed_differs=s1.get("tokens") != s3.get("tokens"))
    rows = []
    for i, q in enumerate(PROMPTS):
        for sd in SEEDS:
            c = compare(chat(A, q, SAMPLED_T, seed=sd), chat(B, q, SAMPLED_T, seed=sd))
            rows.append(c)
            ev.write(event="sampled_pair", i=i, seed=sd, **c)
    same = sum(r["same_tokens"] for r in rows)
    p2 = effective and same == len(rows)
    ev.write(event="check_ii", ok=p2, inconclusive=not effective, same=same, of=len(rows),
             first_diffs=[r["first_diff"] for r in rows if not r["same_tokens"]][:20])
    ok &= p2
    # (iii) log lines
    la, lb = docker_logs(A), docker_logs(B)
    lines = [x for x in lb.splitlines() if KERNEL_LINE in x]
    pids = sorted(set(re.findall(r"pid=(\d+)", "\n".join(lines))))
    p3 = len(lines) >= 1 and KERNEL_LINE not in la
    ev.write(event="check_iii", ok=p3, b_lines=len(lines), b_distinct_pids=len(pids), a_has_line=KERNEL_LINE in la,
             b_examples=[x[-300:] for x in lines[:4]],
             note="TP4: expect 4 distinct pids if each rank is a process (not verified how ranks map to processes)")
    ok &= p3
    # (iv) concurrent requests, information only (never fails the run): (i) and (ii) above send one request at a time,
    # so every decode step had R = 1. 8 and 32 requests at once reach R = 8 / 32 (graph batch sizes 1 8 32 in FLAGS),
    # the C > 1 chunk path and the LNC 2 split of the kernel. A and B may batch the requests differently, so a
    # difference here is not a verdict.
    conc = {}
    for n in CONC_SIZES:
        prompts = [f"Write the numbers from {100 + i} to {110 + i} separated by commas." for i in range(n)]
        outs = {}
        for key, srv in (("A", A), ("B", B)):
            with ThreadPoolExecutor(max_workers=n) as ex:
                outs[key] = list(ex.map(lambda q, s=srv: chat(s, q, 0.0), prompts))
        cmp_ = [compare(a, b) for a, b in zip(outs["A"], outs["B"])]
        conc[n] = sum(1 for c in cmp_ if c["same_tokens"])
        d = [c["max_abs_dlogprob"] for c in cmp_ if c["max_abs_dlogprob"] is not None]
        ev.write(event="check_iv", n=n, same_tokens=conc[n], of=n, share=round(conc[n] / n, 3),
                 errors={k: sum(1 for o in v if not o["ok"]) for k, v in outs.items()},
                 first_diffs=[c["first_diff"] for c in cmp_ if not c["same_tokens"]][:32],
                 max_abs_dlogprob=max(d) if d else None,
                 note="information only; (i)/(ii) cover R=1 only")
    info = {"seed_inconclusive": not effective, "kernel_mismatch": not (p1 and p3 and (p2 or not effective)),
            "conc_same": conc}
    return ok, info


def bench(ev, a, key, srv, k):
    tag = f"{a.tag}_{key}{k}"
    cmd = ["env"] + [f"{n}={v}" for n, v in LB.items()] + ["bash", "22b_llmbench_grid.sh", tag, srv.base_url,
                                                          srv.served_name(), a.model_dir, str(a.results), "4"]
    rcode, sec, last = O.S76.run_bench(cmd, 3600, a.results / f"82_{tag}_bench.out")
    cells = []
    ok = rcode == 0 and last.startswith("LLMBENCH_DONE")
    if ok:
        try:
            cells = [O.single_cell(r) for r in O.read_rows(a.results / f"llmbench_{tag}.csv")]
        except (OSError, KeyError, ValueError) as e:
            ok = False
            last = f"csv: {e}"
    ev.write(event="bench", server=key, pass_no=k, tag=tag, cmd=cmd, exit=rcode, seconds=sec, ok=ok, last_line=last[:300],
             cells=[{"conc": c["conc"], "tps": c["tps"], "ttft_p50_s": c["ttft_p50_s"], "n_error": c["n_error"]}
                    for c in cells])
    return {c["conc"]: c["tps"] for c in cells}


def write_table(path, table):
    lines = ["| server | pass | c=1 tok/s | c=8 tok/s | c=32 tok/s |", "|---|---|---|---|---|"]
    for key, k, d in table:
        lines.append(f"| {key} | {k} | " + " | ".join(str(d.get(c, "")) for c in (1, 8, 32)) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="Qwen3-0.6B")
    # three servers compile from empty caches at once on the host CPUs (12 shapes took 555 s for one server, per the
    # review of this script; B also compiles one NKI kernel per shape); wait_ready returns as soon as a server is ready, so a high cap costs nothing
    ap.add_argument("--ready-cap", type=float, default=3000.0)
    ap.add_argument("--only-correctness", action="store_true")
    ap.add_argument("--skip", default="", help="comma list of servers not to start (C only makes sense)")
    ap.add_argument("--backends-json", default=str(HERE / "backends.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not re.match(r"^[A-Za-z0-9_.-]+$", a.tag):
        print("SAMPLERK_FAILED bad tag")
        return 2
    data_root = os.environ.get("DATA_ROOT") or str(Path.home() / "w")
    models_root = os.environ.get("MODELS_ROOT") or str(Path(data_root) / "models")
    os.environ["MODELS_ROOT"] = models_root
    a.model_dir = str(Path(models_root) / a.model)
    a.results = Path(a.results)
    keys = [k for k in ("A", "B", "C") if k not in a.skip.split(",")]
    names = {k: f"samplerk_{a.tag}_{k}" for k in keys}
    backend = dict(server_ctl.load_backend(a.backends_json, O.BACKEND))
    backend["ready_timeout_s"] = a.ready_cap
    if a.dry_run:
        print(f"DRY RUN: nothing is started. model {a.model_dir} results {a.results} patch {PATCH}")
        for k in keys:
            env = server_env(k, data_root)
            argv = [str(x).format(model=a.model_dir, port=SERVERS[k]["port"], name=names[k]) for x in backend["start"]]
            print(f"[{k}] " + " ".join(f"{n}='{v}'" for n, v in env.items()) + " " + " ".join(argv))
        for i, k in enumerate(x for x in BENCH_ORDER if x in keys):
            print(f"[bench {i + 1}] " + " ".join(f"{n}={v}" for n, v in LB.items()) +
                  f" bash 22b_llmbench_grid.sh {a.tag}_{k}<pass> http://127.0.0.1:{SERVERS[k]['port']} /models/{a.model} "
                  f"{a.model_dir} {a.results} 4")
        print("SAMPLERK_DONE dry-run")
        return 0
    if not (Path(a.model_dir) / "config.json").is_file():
        print(f"SAMPLERK_FAILED no {a.model_dir}/config.json")
        return 2
    if not (PATCH / "rlkit_sampler_kernel.py").is_file():
        print(f"SAMPLERK_FAILED no patch folder {PATCH}")
        return 2
    a.results.mkdir(parents=True, exist_ok=True)
    ev = Events(a.results / f"82_sampler_kernel_{a.tag}.jsonl")
    ev.write(event="open", tag=a.tag, servers={k: dict(SERVERS[k], env=server_env(k, data_root)) for k in keys},
             flags=FLAGS, patch=str(PATCH))
    servers = {k: server_ctl.Server(backend, SERVERS[k]["port"], a.results / "server_logs", name=names[k],
                                    env=server_env(k, data_root)) for k in keys}
    up = {}
    status, code = "error", 1
    info = {"seed_inconclusive": False, "kernel_mismatch": False, "conc_same": {}}

    def on_signal(signum, _frame):               # as 81_opt_sweep.py:1084-1087: SIGTERM/SIGHUP run the finally below
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            signal.signal(sig, on_signal)
    try:
        O.force_remove(list(names.values()))
        res = O.parallel([lambda s=servers[k]: s.start(a.model_dir) for k in keys])
        for k, r in zip(keys, res):
            if isinstance(r, dict) and r.get("ready_s") is not None:
                up[k] = servers[k]
                ev.write(event="server_ready", server=k, **r)
            else:
                log = r.get("log") if isinstance(r, dict) else None
                ev.write(event="not_ready", server=k, result=str(r)[:300], last_lines=O.tail_lines(log, 15) if log else [])
        ok = False
        if "A" in up and "B" in up:
            ok, info = correctness(ev, up["A"], up["B"])
        else:
            ev.write(event="correctness_skipped", reason="A or B not up", up=sorted(up))
        table = []
        if not a.only_correctness:
            count = {}
            for k in BENCH_ORDER:
                if k in up:
                    count[k] = count.get(k, 0) + 1
                    table.append((k, count[k], bench(ev, a, k, up[k], count[k])))
            write_table(a.results / f"82_sampler_kernel_{a.tag}.md", table)
            ev.write(event="table", rows=[{"server": k, "pass": n, "tps": d} for k, n, d in table])
        # exit 0 needs every bench that ran to have returned cells (a failed bench has an empty dict)
        code = 0 if ok and (a.only_correctness or (table and all(d for _, _, d in table))) else 1
        status = "ok" if code == 0 else "failed"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as e:  # noqa: BLE001      an unexpected error still ends with the SAMPLERK_ line below
        status = "error"
        ev.write(event="exception", error=f"{type(e).__name__}: {e}"[:300])
    finally:
        for k, s in servers.items():
            try:
                ev.write(event="stop", server=k, seconds=s.stop())
            except Exception as e:  # noqa: BLE001
                ev.write(event="stop_error", server=k, error=str(e)[:200])
        O.force_remove(list(names.values()))
        ev.write(event="end", status=status, up=sorted(up))
        ev.write(done=True)
    print(("SAMPLERK_DONE " if code == 0 else "SAMPLERK_FAILED ") + f"status={status} up={','.join(sorted(up))} "
          f"seed_inconclusive={info['seed_inconclusive']} kernel_mismatch={info['kernel_mismatch']} "
          f"concurrent_same_tokens={info['conc_same']}")
    return code


if __name__ == "__main__":
    sys.exit(main())
