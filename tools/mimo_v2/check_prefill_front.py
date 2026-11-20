#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's prefill attention front kernel (models/mimo_v2/kernels/prefill.py) on one NeuronCore
against the torch it replaces (attention.MiMoAttention.prefill's norm, projection, RoPE, FP8 scaling and K/V write),
over a bf16 and an FP8 cache. The query heads and cache rows must match to bf16 / FP8 rounding.

    NEURON_RT_VISIBLE_CORES=0 python3 tools/mimo_v2/check_prefill_front.py"""

import sys


def main() -> int:
    import os

    # an FP8 cache's float8_e4m3fn compiles as Trn2's e4m3 (models/mimo_v2/launch.py sets both for a server)
    os.environ.setdefault("UNSAFE_FP8FNCAST", "1")
    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_front
    from sglang_neuron.models.mimo_v2.rope import rotate
    from sglang_neuron.models.transformer.model import RMSNorm
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    from sglang_neuron.models.mimo_v2.launch import FP8_FN_AS_E4M3

    prefix = "--internal-hlo2tensorizer-options="
    cc = [f"{a} {FP8_FN_AS_E4M3}" if a.startswith(prefix) else a for a in COMPILER_ARGS]
    if not any(a.startswith(prefix) for a in cc):
        cc.append(prefix + FP8_FN_AS_E4M3)
    g = torch.Generator().manual_seed(7)
    T = int(os.environ.get("CK_T", 512))
    H, nq, d, page, blocks = 4096, 1, 256, 128, T // 128 + 8
    x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16)
    norm = RMSNorm(H, 1e-6)
    norm.weight.data = (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)
    w = (torch.randn(H, (nq + 2) * d, generator=g) * H ** -0.5).to(torch.bfloat16)
    pos = torch.arange(T) + 300
    inv = 1.0 / (10000.0 ** (torch.arange(0, 86, 2).float() / 86))
    ang = pos.float()[:, None] * inv[None, :]
    cos = torch.ones(T, d // 2)
    sin = torch.zeros(T, d // 2)
    cos[:, :inv.numel()], sin[:, :inv.numel()] = ang.cos(), ang.sin()
    cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
    perm = torch.randperm(blocks * page - page, generator=g)[:T] + page
    slots = perm.to(torch.int32)
    slots[-3:] = torch.tensor([-1, blocks * page + 5, -7], dtype=torch.int32)  # padding rows -> slot 0

    qkv = norm(x) @ w
    q, k, v = qkv.split([nq * d, d, d], dim=-1)
    q = rotate(q.view(T, nq, d), cos, sin)
    k = rotate(k.view(T, 1, d), cos, sin)
    v = v.view(T, 1, d)
    ok = True
    for fp8, scale in ((False, 1.0), (True, 0.37)):
        dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        kr, vr = ((t * scale).clamp(-240, 240) if fp8 else t for t in (k, v))
        live = (slots >= 0) & (slots < blocks * page)
        tgt = torch.where(live, slots.long(), torch.zeros_like(slots.long()))
        kc_ref = torch.zeros(blocks * page, d, dtype=dtype)
        vc_ref = torch.zeros(blocks * page, d, dtype=dtype)
        for i in range(T):  # in order: the padding rows' writes to slot 0 land last, as the kernel's tiles do
            kc_ref[tgt[i]] = kr[i, 0].to(dtype)
            vc_ref[tgt[i]] = vr[i, 0].to(dtype)
        with stack.compile_scope(cc):
            fn = torch.compile(lambda *a: wrap_nki(prefill_front)[get_logical_nc_config()](
                *a, eps=1e-6, num_heads=nq, kv_scale=scale, q_scale=0.07), backend=stack.compile_backend_name(),
                fullgraph=True, dynamic=False)
            dev = torch.device("neuron", 0)
            kc = torch.zeros(blocks, 1, page, d, dtype=dtype).to(dev)
            vc = torch.zeros(blocks, 1, page, d, dtype=dtype).to(dev)
            outs = fn(x.to(dev), norm.weight.view(1, -1).to(dev), w.to(dev), cos.to(dev), sin.to(dev),
                      slots.view(-1, 1).to(dev), kc, vc)
        q_k, k_k, v_k, kc_k, vc_k = (o.cpu() for o in outs)
        dq = (q_k.float() - (q * 0.07).transpose(0, 1).float()).abs().max().item()
        real = tgt[live]
        dk = (kc_k.reshape(-1, d)[real].float() - kc_ref[real].float()).abs().max().item()
        dv = (vc_k.reshape(-1, d)[real].float() - vc_ref[real].float()).abs().max().item()
        untouched = torch.ones(blocks * page, dtype=torch.bool)
        untouched[tgt] = False
        drk = (k_k.float() - kr[:, 0].to(dtype).float()).abs().max().item()
        drv = (v_k.float() - vr[:, 0].to(dtype).float()).abs().max().item()
        rows_ok = max(drk, drv) <= max(dk, dv) + 1e-6 or max(drk, drv) < 0.07
        clean = kc_k.reshape(-1, d)[untouched].float().abs().max().item() == 0
        # one ulp of the cache's dtype at the largest magnitude (e4m3: 3 mantissa bits, bf16: 7)
        import math

        bits = 3 if fp8 else 7
        ulp = lambda m: 2.0 ** (math.floor(math.log2(max(m, 1e-30))) - bits)  # noqa: E731
        good = (dq < 0.05 and dk <= ulp(kc_ref.float().abs().max().item())
                and dv <= ulp(vc_ref.float().abs().max().item()) and clean and rows_ok)
        ok &= good
        print(f"  {'fp8' if fp8 else 'bf16'} cache: q max diff {dq:.3e}, k rows {dk:.3e}, v rows {dv:.3e}, "
              f"K/V rows out {max(drk, drv):.3e}, other rows untouched {clean}: {'ok' if good else 'DIFF'}", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
