#!/usr/bin/env python3
"""The sampler-kernel hardware run of 82_sampler_kernel_hw.py on Qwen3-8B (2026-11-24): same three servers
(A released temperature sampler, B kernel, C released greedy sampler), same checks and benches, the image's Qwen3-8B
example script, compile caches of their own.

    python 84_sampler_kernel_8b.py --results DIR --tag r12 [--skip C] [--dry-run]     (other options: see 82)"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hw82", HERE / "82_sampler_kernel_hw.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
MODEL = "Qwen3-8B"
for k in H.SERVERS:
    H.SERVERS[k] = dict(H.SERVERS[k], cache=H.SERVERS[k]["cache"] + "_8b")
_env = H.server_env


def server_env(key, data_root):
    e = _env(key, data_root)
    e["SGL_SCRIPT"] = H.O.SCRIPTS[MODEL]
    return e


H.server_env = server_env
if "--model" not in sys.argv:
    sys.argv += ["--model", MODEL]
sys.exit(H.main())
