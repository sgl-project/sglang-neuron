#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's decode MoE half kernel (models/mimo_v2/kernels/moe_half.py) on one NeuronCore, against
a torch reference of the model's served form: torch's RMSNorm (models/transformer/model.RMSNorm), the noaux router
(models/hy_v3/moe.hy3_route, the affinities rounded through bf16 as moe.MiMoV2MoE.forward) and the local experts on
their FP8 weights and power-of-two per-row scales (models/mimo_v2/experts_fp8.py), in fp32.

Seeded random weights at MiMo-V2.6-Flash's per-rank shapes at EP64 (H 4096, 4 local experts of 256, I 2048); the
local experts' selection bias is raised so that most tokens route to some of them. A control with the reference's
selection bias dropped must fail, so a kernel that ignored the bias would show.

    NEURON_RT_VISIBLE_CORES=0 python3 tools/mimo_v2/check_moe_half.py

Run on a free NeuronCore."""

import argparse
import sys


def reference(x, ln_w, eps, router_w, bias, ebase, gate_up, gate_up_scale, down, down_scale, top_k, scaling,
              use_bias=True):
    """[T, H] fp32: this rank's MoE partial by the torch path's formulas."""
    import torch

    x32 = x.float()
    normed = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
    normed = (ln_w.to(torch.bfloat16) * normed).to(torch.bfloat16)
    scores = torch.sigmoid(normed.float() @ router_w.float().t())
    _, experts = torch.topk(scores + (bias if use_bias else 0), top_k, dim=-1)
    chosen = scores.gather(1, experts)
    chosen = chosen / (chosen.sum(dim=-1, keepdim=True) + 1e-20)
    aff = torch.zeros_like(scores).scatter(1, experts, chosen * scaling).to(torch.bfloat16).float()
    out = torch.zeros(x.shape[0], x.shape[1])
    for j in range(gate_up.shape[0]):
        a = aff[:, ebase + j : ebase + j + 1]
        gu = normed.float() @ (gate_up[j].float() * gate_up_scale[j].reshape(1, 2, -1)).reshape(x.shape[1], -1)
        g, u = gu.chunk(2, dim=-1)
        inter = (torch.nn.functional.silu(g) * u).to(torch.bfloat16).float()
        out += a * (inter @ (down[j].float() * down_scale[j].reshape(1, -1)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8])
    args = parser.parse_args()
    import os

    # the FP8 experts compile float8_e4m3fn as Trn2's e4m3, as a serving launch sets (models/mimo_v2/launch.py)
    os.environ["UNSAFE_FP8FNCAST"] = "1"
    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.mimo_v2.experts_fp8 import FP8, requant_rows
    from sglang_neuron.models.mimo_v2.kernels.moe_half import moe_half
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    from sglang_neuron.models.mimo_v2.launch import FP8_FN_AS_E4M3

    prefix = "--internal-hlo2tensorizer-options="
    args_cc = [f"{a} {FP8_FN_AS_E4M3}" if a.startswith(prefix) else a for a in COMPILER_ARGS]
    if not any(a.startswith(prefix) for a in args_cc):
        args_cc.append(prefix + FP8_FN_AS_E4M3)
    H, E, EA, I, top_k, eps, scaling, ebase = 4096, 4, 256, 2048, 8, 1e-6, 1.0, 8  # noqa: E741
    g = torch.Generator().manual_seed(0)
    ln_w = (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)
    router_w = (0.02 * torch.randn(EA, H, generator=g)).to(torch.bfloat16)
    bias = 0.01 * torch.randn(EA, generator=g)
    bias[ebase:ebase + E] += 0.3
    gate_up_w, gate_up_s, down_w, down_s = [], [], [], []
    for _ in range(E):
        gate = 0.02 * torch.randn(I, H, generator=g)
        up = 0.02 * torch.randn(I, H, generator=g)
        dn = 0.02 * torch.randn(H, I, generator=g)
        qs = [requant_rows(m) for m in (gate, up, dn)]
        gate_up_w.append(torch.stack([qs[0][0].float().t(), qs[1][0].float().t()], dim=1))  # [H, 2, I]
        gate_up_s.append(torch.stack([qs[0][1], qs[1][1]]))  # [2, I]
        down_w.append(qs[2][0].float().t())  # [I, H]
        down_s.append(qs[2][1])  # [H]
    gate_up_w, gate_up_s = torch.stack(gate_up_w), torch.stack(gate_up_s)
    down_w, down_s = torch.stack(down_w), torch.stack(down_s)
    device = torch.device("neuron", 0)
    weights = [ln_w.view(1, H), router_w.t().contiguous(), bias.view(1, EA), torch.tensor([[float(ebase)]]),
               gate_up_w.to(FP8), down_w.to(FP8), gate_up_s, down_s]
    weights_d = [w.contiguous().to(device) for w in weights]

    def fn(x, *ws):
        return wrap_nki(moe_half)[get_logical_nc_config()](x, *ws, eps=eps, route_scale=scaling, top_k=top_k)

    failures = []
    for rows in args.rows:
        x = (torch.randn(rows, H, generator=g) * 2).to(torch.bfloat16)
        with stack.compile_scope(args_cc):
            compiled = torch.compile(fn, backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
            got = compiled(x.to(device), *weights_d).cpu().float()
        torch._dynamo.reset()
        want = reference(x, ln_w, eps, router_w, bias, ebase, gate_up_w, gate_up_s, down_w, down_s, top_k, scaling)
        control = reference(x, ln_w, eps, router_w, bias, ebase, gate_up_w, gate_up_s, down_w, down_s, top_k,
                            scaling, use_bias=False)
        rel = ((got - want).norm() / want.norm().clamp(min=1e-30)).item()
        rel_control = ((got - control).norm() / control.norm().clamp(min=1e-30)).item()
        routed = int((want.abs().sum(dim=1) > 0).sum())
        ok = rel < 2e-2 and rel_control > 0.1
        print(f"{'PASS' if ok else 'FAIL'} rows {rows}: relative L2 {rel:.2e} vs the reference, {rel_control:.2e} vs "
              f"the no-bias control; {routed}/{rows} rows routed to this rank")
        if not ok:
            failures.append(rows)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
