#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check of Hy4-Flash's prefill iHC mix kernel (models/hy_v4_flash/kernels/hc_prefill.py) against ops.hc_mix on
seeded random streams and weights at the model's shapes (hc 4, H 3072), with the device time a call.

    python3 tools/hy4_flash/check_hc_prefill.py --simulate --grid 2 --rows 256   # nki.simulate on the CPU
    NEURON_RT_VISIBLE_CORES=0 python3 tools/hy4_flash/check_hc_prefill.py        # one NeuronCore"""

import argparse
import sys
import time

H, HC = 3072, 4
EPS, HC_EPS, MAG = 1e-5, 1e-6, 2.0
KWARGS = dict(eps=EPS, hc_eps=HC_EPS, hc_mag=MAG)


def make_inputs(rows: int, seed: int = 0):
    import torch

    g = torch.Generator().manual_seed(seed)
    streams = (torch.randn(rows, HC, H, generator=g) * 0.5).to(torch.bfloat16)
    fn = 0.01 * torch.randn(2 * HC, HC * H, generator=g)
    base = 0.1 * torch.randn(2 * HC, generator=g)
    scale = torch.tensor([0.5, 0.7])
    return streams, fn, base, scale


def kernel_args(streams, fn, base, scale):
    import torch

    fnT = fn.t().contiguous()
    hi = fnT.to(torch.bfloat16)
    lo = (fnT - hi.float()).to(torch.bfloat16)
    return [streams, hi, lo, base.view(1, -1), scale.view(1, -1)]


def run_simulate(args_t, grid):
    import ml_dtypes
    import nki
    import numpy as np
    import torch

    from sglang_neuron.models.hy_v4_flash.kernels import hc_prefill

    def np_of(t):
        return t.float().numpy().astype(ml_dtypes.bfloat16) if t.dtype == torch.bfloat16 else t.numpy()

    k = hc_prefill.hc_mix_rows
    outs = nki.simulate(k[grid] if grid > 1 else k)(*[np_of(t) for t in args_t], **KWARGS)
    return [torch.from_numpy(np.asarray(o).astype(np.float32)) for o in outs]


def run_device(args_t):
    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.hy_v4_flash.kernels import hc_prefill
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    device = torch.device("neuron", 0)
    dev = [t.contiguous().to(device) for t in args_t]

    def fn(*ts):
        return wrap_nki(hc_prefill.hc_mix_rows)[get_logical_nc_config()](*ts, **KWARGS)

    with stack.compile_scope(COMPILER_ARGS):
        compiled = torch.compile(fn, backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
        outs = [o.cpu().float() for o in compiled(*dev)]
        for _ in range(3):
            o = compiled(*dev)
        o[0].cpu()
        n = 20
        t0 = time.time()
        for _ in range(n):
            o = compiled(*dev)
        o[0].cpu()
        print(f"  device: {(time.time() - t0) / n * 1e3:.3f} ms a call", flush=True)
    torch._dynamo.reset()
    return outs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[2048])
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--grid", type=int, default=1, help="programs under --simulate (the device runs its LNC)")
    args = parser.parse_args()
    import torch

    from sglang_neuron.models.hy_v4_flash import ops

    failures = []
    for rows in args.rows:
        streams, fn, base, scale = make_inputs(rows)
        a = kernel_args(streams, fn, base, scale)
        want_post, want_x = ops.hc_mix(streams, fn, base, scale, HC, MAG, HC_EPS, EPS)
        got_post, got_x = run_simulate(a, args.grid) if args.simulate else run_device(a)
        want_x = want_x.float()
        rel_x = ((got_x - want_x).norm() / want_x.norm()).item()
        rel_p = ((got_post - want_post).norm() / want_post.norm()).item()
        # bf16 output rounding (the graph casts the same fp32 sum)
        ok = rel_x < 1e-2 and rel_p < 1e-4
        print(f"{'PASS' if ok else 'FAIL'} rows {rows}: block input rel L2 {rel_x:.2e}, post rel L2 {rel_p:.2e}",
              flush=True)
        if not ok:
            failures.append(rows)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
