# SPDX-License-Identifier: Apache-2.0
"""Whether MiMo-V2.6's MXFP4 routed experts convert to Trn2's FP8 e4m3 (max 240) without loss, over the whole
checkpoint: each weight dequantized (models/mimo_v2/weights.dequant_mxfp4), divided by a power-of-two scale per output
row (the smallest that brings the row's largest value to 240 or below), cast to float8_e4m3fn and back.

A power of two only shifts exponents, so a value converts exactly when its quotient is an e4m3 number: at most 240, and
a normal (at least 2^-6) or a subnormal multiple of 2^-9. An e2m1 value has at most two significant bits, so above
2^-6 it is always representable; the check counts the values that are not.

    python3 tools/mimo_v2/check_fp8_experts.py --model /models/MiMo-V2.6-Flash-RL --workers 64

Prints, per projection, the tensors checked, those not bit-exact, the worst relative error and the smallest quotient
seen; exits 1 if any tensor is not exact."""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import torch

FP8_MAX = 240.0


def _one_thread() -> None:
    # each worker one torch thread: a worker's default pool is every core, and the workers would oversubscribe the host
    torch.set_num_threads(1)


def _check_file(args) -> list:
    from safetensors import safe_open

    from sglang_neuron.models.mimo_v2.weights import dequant_mxfp4

    path, names = args
    out = []
    with safe_open(path, framework="pt") as f:
        for name in names:
            w = dequant_mxfp4(f.get_slice(name), f.get_slice(name.removesuffix(".weight") + ".weight_scale"))
            amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
            scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
            x = w / scale
            back = x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale
            exact = torch.equal(back, w)
            rel = 0.0 if exact else ((back - w).norm() / w.norm().clamp(min=1e-30)).item()
            nonzero = x.abs()[x != 0]
            out.append((name, exact, rel, nonzero.min().item() if nonzero.numel() else 0.0, x.abs().max().item()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with open(os.path.join(args.model, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    pattern = re.compile(r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$")
    by_file = {}
    for name, file in weight_map.items():
        if pattern.match(name):
            by_file.setdefault(os.path.join(args.model, file), []).append(name)
    jobs = [(path, names[i:i + 64]) for path, names in by_file.items() for i in range(0, len(names), 64)]
    stats = {}
    bad = []
    with ProcessPoolExecutor(args.workers, initializer=_one_thread) as pool:
        for results in pool.map(_check_file, jobs):
            for name, exact, rel, qmin, qmax in results:
                proj = name.rsplit(".", 2)[-2]
                s = stats.setdefault(proj, {"tensors": 0, "inexact": 0, "worst_rel": 0.0, "min_quotient": 1e30,
                                            "max_quotient": 0.0})
                s["tensors"] += 1
                s["inexact"] += not exact
                s["worst_rel"] = max(s["worst_rel"], rel)
                s["min_quotient"] = min(s["min_quotient"], qmin)
                s["max_quotient"] = max(s["max_quotient"], qmax)
                if not exact:
                    bad.append((name, rel))
    for proj, s in sorted(stats.items()):
        print(f"{proj}: {s['tensors']} tensors, {s['inexact']} not exact, worst relative L2 {s['worst_rel']:.2e}, "
              f"|quotient| in [{s['min_quotient']:.4g}, {s['max_quotient']:.4g}]")
    for name, rel in bad[:20]:
        print(f"  not exact: {name} (relative L2 {rel:.2e})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
