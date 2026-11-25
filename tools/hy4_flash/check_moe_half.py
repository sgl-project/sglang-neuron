#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check of Hy4-Flash's decode MoE half kernel (models/hy_v4_flash/kernels/moe_half.py) against a torch reference of
the model's served form built from the graph's own pieces: ops.hc_mix, ops.rms_norm, ops.route, pre_expert_proj, the
local experts on their FP8 weights and per-row scales (fp8.requant_rows), post_expert_proj and the shared expert.

Seeded random weights at Hy4-Flash's per-rank shapes at TP32 / EP32 (``--experts`` narrows the local experts for a
quick run). The local experts' selection bias is raised so that most rows route to some of them; a control with the
bias dropped from the reference must fail, so a kernel that ignored it would show.

    python3 tools/hy4_flash/check_moe_half.py --simulate            # nki.simulate on the CPU, grid 1
    NEURON_RT_VISIBLE_CORES=0 python3 tools/hy4_flash/check_moe_half.py   # one NeuronCore"""

import argparse
import os
import sys

H, HC, L, I, EA, TOP_K, S = 3072, 4, 1536, 1536, 512, 16, 48  # noqa: E741
EPS, HC_EPS, MAG, SCALING, LIMIT = 1e-5, 1e-6, 2.0, 3.998, 10.0


def make_inputs(rows: int, experts: int, rank: int, seed: int = 0):
    import torch

    from sglang_neuron.models.hy_v4_flash.fp8 import requant_rows

    g = torch.Generator().manual_seed(seed)
    w = {}
    w["streams"] = (torch.randn(rows, HC, H, generator=g) * 0.5).to(torch.bfloat16)
    w["fn"] = 0.01 * torch.randn(2 * HC, HC * H, generator=g)
    w["base"] = 0.1 * torch.randn(2 * HC, generator=g)
    w["scale"] = torch.tensor([0.5, 0.7])
    w["ln_w"] = (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)
    w["router_w"] = (0.02 * torch.randn(EA, H, generator=g)).to(torch.bfloat16)
    bias = 0.01 * torch.randn(EA, generator=g)
    bias[rank * experts:(rank + 1) * experts] += 1.0
    w["bias"] = bias
    w["pre"] = (0.02 * torch.randn(H, L, generator=g)).to(torch.bfloat16)
    w["post"] = (0.02 * torch.randn(L, H, generator=g)).to(torch.bfloat16)
    gu, gus, dn, dns = [], [], [], []
    for _ in range(experts):
        gate = 0.03 * torch.randn(I, L, generator=g)
        up = 0.03 * torch.randn(I, L, generator=g)
        down = 0.03 * torch.randn(L, I, generator=g)
        qs = [requant_rows(m) for m in (gate, up, down)]
        gu.append(torch.stack([qs[0][0].float().t(), qs[1][0].float().t()], dim=1))  # [L, 2, I]
        gus.append(torch.stack([qs[0][1], qs[1][1]]))  # [2, I]
        dn.append(qs[2][0].float().t())  # [I, L]
        dns.append(qs[2][1])  # [L]
    w["gate_up"], w["gate_up_scale"] = torch.stack(gu), torch.stack(gus)
    w["down"], w["down_scale"] = torch.stack(dn), torch.stack(dns)
    w["sh_gate_up"] = (0.02 * torch.randn(H, 2 * S, generator=g)).to(torch.bfloat16)
    w["sh_down"] = (0.02 * torch.randn(S, H, generator=g)).to(torch.bfloat16)
    return w


def _w8a8(x):
    """x through the kernel's per-row FP8 activation (240 / the row's absolute max), back in fp32."""
    import torch

    x = x.float()
    sc = 240.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    return (x * sc).clamp(-240.0, 240.0).to(torch.float8_e4m3fn).float() / sc


def reference(w, rank: int, experts: int, use_bias: bool = True, w8a8: bool = False):
    """(partial [T, H] fp32, post [T, hc] fp32) by the graph's formulas; ``w8a8`` quantizes the experts' inputs as
    the kernel does (an exact match then checks the kernel's arithmetic apart from its FP8 activations)."""
    import torch

    from sglang_neuron.models.hy_v4_flash import ops

    post, x = ops.hc_mix(w["streams"], w["fn"], w["base"], w["scale"], HC, MAG, HC_EPS, EPS)
    xn = ops.rms_norm(x, w["ln_w"], EPS)
    bias = w["bias"] if use_bias else torch.zeros_like(w["bias"])
    aff, _ = ops.route(xn, w["router_w"], bias, TOP_K, True, SCALING)
    lat = xn @ w["pre"]
    routed = torch.zeros(lat.shape[0], L)
    for e in range(experts):
        a = aff[:, rank * experts + e:rank * experts + e + 1]
        li = _w8a8(lat) if w8a8 else lat.float()
        gu = li @ (w["gate_up"][e].float() * w["gate_up_scale"][e].reshape(1, 2, -1)).reshape(L, -1)
        gate, up = gu.chunk(2, dim=-1)
        inter = ops.swiglu(gate, up, LIMIT).to(torch.bfloat16).float()
        if w8a8:
            inter = _w8a8(inter)
        routed += a * (inter @ (w["down"][e].float() * w["down_scale"][e].reshape(1, -1)))
    out = (routed.to(torch.bfloat16) @ w["post"]).float()
    sg, su = (xn @ w["sh_gate_up"]).chunk(2, dim=-1)
    out = out + (ops.swiglu(sg, su, 0.0) @ w["sh_down"]).float()
    return out, post


def kernel_args(w, rank: int):
    import torch

    from sglang_neuron.models.hy_v4_flash.fp8 import FP8

    fnT = w["fn"].t().contiguous()
    hi = fnT.to(torch.bfloat16)
    lo = (fnT - hi.float()).to(torch.bfloat16)
    return [w["streams"], hi, lo, w["base"].view(1, -1), w["scale"].view(1, -1), w["ln_w"].view(1, -1),
            w["router_w"].t().contiguous(), w["bias"].view(1, -1), torch.tensor([[float(rank)]]), w["pre"],
            w["post"], w["gate_up"].to(FP8), w["down"].to(FP8), w["gate_up_scale"], w["down_scale"],
            w["sh_gate_up"], w["sh_down"]]


GRID = 1
KWARGS = dict(eps=EPS, hc_eps=HC_EPS, hc_mag=MAG, route_scale=SCALING, top_k=TOP_K, limit=LIMIT)


def run_simulate(args_t):
    import ml_dtypes
    import numpy as np
    import nki
    import torch

    from sglang_neuron.models.hy_v4_flash.kernels.moe_half import moe_half

    def np_of(t):
        if t.dtype == torch.bfloat16:
            return t.float().numpy().astype(ml_dtypes.bfloat16)
        if t.dtype == torch.float8_e4m3fn:
            return t.float().numpy().astype(ml_dtypes.float8_e4m3)
        return t.numpy()

    kernel = moe_half[GRID] if GRID > 1 else moe_half
    out, post = nki.simulate(kernel)(*[np_of(t) for t in args_t], **KWARGS)
    return torch.from_numpy(np.asarray(out).astype(np.float32)), torch.from_numpy(np.asarray(post).astype(np.float32))


def run_device(args_t):
    os.environ["UNSAFE_FP8FNCAST"] = "1"
    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.hy_v4_flash.kernels.moe_half import moe_half
    from sglang_neuron.models.hy_v4_flash.launch import FP8_FN_AS_E4M3
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    prefix = "--internal-hlo2tensorizer-options="
    args_cc = [f"{a} {FP8_FN_AS_E4M3}" if a.startswith(prefix) else a for a in COMPILER_ARGS]
    device = torch.device("neuron", 0)
    dev = [t.contiguous().to(device) for t in args_t]

    def fn(*ts):
        return wrap_nki(moe_half)[get_logical_nc_config()](*ts, **KWARGS)

    with stack.compile_scope(args_cc):
        compiled = torch.compile(fn, backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
        out, post = compiled(*dev)
        out, post = out.cpu().float(), post.cpu().float()
        import time

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
    return out, post


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--fp8-act", action="store_true", help="the kernel's W8A8 experts")
    parser.add_argument("--grid", type=int, default=1, help="programs under --simulate (the device runs its LNC)")
    parser.add_argument("--no-split", action="store_true",
                        help="the latent-split expert loop (every program runs every expert) for an A/B")
    args = parser.parse_args()
    global GRID
    GRID = args.grid
    failures = []
    for rows in args.rows:
        w = make_inputs(rows, args.experts, args.rank)
        want, want_post = reference(w, args.rank, args.experts)
        control, _ = reference(w, args.rank, args.experts, use_bias=False)
        exact, _ = reference(w, args.rank, args.experts, w8a8=True)
        KWARGS["fp8_act"] = args.fp8_act
        KWARGS["split_experts"] = not args.no_split
        got, got_post = (run_simulate if args.simulate else run_device)(kernel_args(w, args.rank))
        rel = ((got - want).norm() / want.norm().clamp(min=1e-30)).item()
        rel_c = ((got - control).norm() / control.norm().clamp(min=1e-30)).item()
        rel_p = ((got_post - want_post).norm() / want_post.norm()).item()
        rel_x = ((got - exact).norm() / exact.norm().clamp(min=1e-30)).item()
        # bf16 activations match the reference to rounding; W8A8 is about 3% a matmul off it
        ok = (rel < 5e-2 and rel_x < 3e-2 if args.fp8_act else rel < 1e-2) and rel_p < 1e-3 and rel_c > 0.2
        print(f"{'PASS' if ok else 'FAIL'} rows {rows}: partial rel L2 {rel:.2e} (no-bias control {rel_c:.2e}), "
              f"post rel L2 {rel_p:.2e}, vs the W8A8 reference {rel_x:.2e}", flush=True)
        if not ok:
            failures.append(rows)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
