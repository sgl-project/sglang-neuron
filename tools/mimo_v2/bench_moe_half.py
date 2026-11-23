#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device time of MiMo-V2's decode MoE half kernel (models/mimo_v2/kernels/moe_half.py) on one NeuronCore: the kernel
chained R times inside one kernel (each call's output the next one's input, as consecutive layers), timed at R = 1 and
R = REPS, the difference over REPS - 1 calls (the launch overhead cancels). Seeded random weights at the served
per-rank shapes (check_moe_half.py's), the local experts' bias raised so every row routes to them.

    NEURON_RT_VISIBLE_CORES=0 python3 tools/mimo_v2/bench_moe_half.py --rows 4 32

Run on a free NeuronCore."""

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[4, 32])
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    import os

    os.environ["UNSAFE_FP8FNCAST"] = "1"
    from sglang_neuron.process_init import init_stack

    init_stack()
    import nki
    from nki._backends.mlir_tracer import context as tracer

    # the chained calls name their outputs alike: a repeat gets a suffix (kernels/layer._suffix_duplicate_op_names)
    setup_loc = tracer.MLIRTracerContext.setup_loc

    def unique_setup_loc(self, name=None):
        if name and name in self._op_names:
            n = 1
            while f"{name}__u{n}" in self._op_names:
                n += 1
            name = f"{name}__u{n}"
        return setup_loc(self, name)

    tracer.MLIRTracerContext.setup_loc = unique_setup_loc
    import torch

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.mimo_v2.experts_fp8 import FP8, requant_rows
    from sglang_neuron.models.mimo_v2.kernels.moe_half import moe_half
    from sglang_neuron.models.mimo_v2.launch import FP8_FN_AS_E4M3
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    prefix = "--internal-hlo2tensorizer-options="
    cc = [f"{a} {FP8_FN_AS_E4M3}" if a.startswith(prefix) else a for a in COMPILER_ARGS]
    if not any(a.startswith(prefix) for a in cc):
        cc.append(prefix + FP8_FN_AS_E4M3)
    H, E, EA, I, top_k, eps, ebase = 4096, 4, 256, 2048, 8, 1e-6, 8  # noqa: E741
    g = torch.Generator().manual_seed(0)
    ln_w = (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)
    router_w = (0.02 * torch.randn(EA, H, generator=g)).to(torch.bfloat16)
    bias = 0.01 * torch.randn(EA, generator=g)
    bias[ebase:ebase + E] += 1.0  # every row routes to all four local experts: the served worst case
    gw, gs, dw, ds = [], [], [], []
    for _ in range(E):
        qs = [requant_rows(0.02 * torch.randn(*s, generator=g)) for s in ((I, H), (I, H), (H, I))]
        gw.append(torch.stack([qs[0][0].float().t(), qs[1][0].float().t()], dim=1))
        gs.append(torch.stack([qs[0][1], qs[1][1]]))
        dw.append(qs[2][0].float().t())
        ds.append(qs[2][1])
    dev = torch.device("neuron", 0)
    ws = [ln_w.view(1, H), router_w.t().contiguous(), bias.view(1, EA), torch.tensor([[float(ebase)]]),
          torch.stack(gw).to(FP8), torch.stack(dw).to(FP8), torch.stack(gs), torch.stack(ds)]
    ws = [w.contiguous().to(dev) for w in ws]

    def chain(reps):
        @nki.jit
        def chained(x, ln, rw, eb, base, guw, dnw, gus, dns):
            h = x
            for _ in range(reps):
                h = moe_half(h, ln, rw, eb, base, guw, dnw, gus, dns, eps=eps, route_scale=1.0, top_k=top_k)
            return h
        return chained

    for rows in args.rows:
        x = (torch.randn(rows, H, generator=g) * 2).to(torch.bfloat16).to(dev)
        times = {}
        for reps in (1, args.reps):
            kern = chain(reps)
            with stack.compile_scope(cc):
                fn = torch.compile(lambda a, *b: wrap_nki(kern)[get_logical_nc_config()](a, *b),
                                   backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
                for _ in range(3):
                    fn(x, *ws).cpu()
                t = time.perf_counter()
                for _ in range(args.iters):
                    out = fn(x, *ws)
                out.cpu()
                times[reps] = (time.perf_counter() - t) / args.iters
            torch._dynamo.reset()
        per = (times[args.reps] - times[1]) / (args.reps - 1)
        print(f"rows {rows}: moe_half {per * 1e6:7.1f} us a call (R=1 {times[1] * 1e6:.0f} us, "
              f"R={args.reps} {times[args.reps] * 1e6:.0f} us)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
