#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's FP8 prior-packing kernel (models/mimo_v2/kernels/prefill.prefill_pack) on one
NeuronCore against the torch it replaces, attention.pack_prior: the gathered K (packed) and V pages and the remapped
table must match exactly, for a full layer's whole table and a sliding layer's window columns.

    NEURON_RT_VISIBLE_CORES=0 python3 tools/mimo_v2/check_prefill_pack.py"""

import sys


def main() -> int:
    import os

    os.environ.setdefault("UNSAFE_FP8FNCAST", "1")
    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.mimo_v2.attention import pack_prior
    from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_pack
    from sglang_neuron.models.mimo_v2.launch import FP8_FN_AS_E4M3
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    prefix = "--internal-hlo2tensorizer-options="
    cc = [f"{a} {FP8_FN_AS_E4M3}" if a.startswith(prefix) else a for a in COMPILER_ARGS]
    if not any(a.startswith(prefix) for a in cc):
        cc.append(prefix + FP8_FN_AS_E4M3)
    g = torch.Generator().manual_seed(13)
    blocks, page, d, width = 300, 128, 256, 160
    fp8 = torch.float8_e4m3fn
    k = (torch.randn(blocks, 1, page, d, generator=g) * 20).clamp(-240, 240).to(fp8)
    v = (torch.randn(blocks, 1, page, d, generator=g) * 20).clamp(-240, 240).to(fp8)
    used = 140
    table = torch.full((1, width), -1, dtype=torch.int32)
    table[0, :used] = torch.randperm(blocks - 1, generator=g)[:used].to(torch.int32) + 1
    ok = True
    for window, cached, tokens in ((0, 2048, 2048), (128, 9000, 2048)):
        if window:
            n_cols = (tokens + window - 1) // page + 2
            first = torch.clamp(torch.tensor([cached]) - window + 1, min=0) // page
            kr, vr, tr = pack_prior(k, v, table, first_col=first, n_cols=n_cols)
        else:
            n_cols = width
            kr, vr, tr = pack_prior(k, v, table)
        with stack.compile_scope(cc):
            fn = torch.compile(lambda *a: wrap_nki(prefill_pack)[get_logical_nc_config()](
                *a, n_cols=n_cols, window=window), backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
            dev = torch.device("neuron", 0)
            outs = fn(k.to(dev), v.to(dev), table.to(dev), torch.tensor([[cached]], dtype=torch.int32).to(dev))
        kk, vk, tk = (o.cpu() for o in outs)
        res = [torch.equal(kk.view(torch.uint8), kr.view(torch.uint8)), torch.equal(vk.view(torch.uint8),
                                                                                    vr.view(torch.uint8)),
               torch.equal(tk.long(), tr.long())]
        ok &= all(res)
        names = ("K", "V", "table")
        print(f"  window {window} ({n_cols} columns): "
              + ", ".join(f"{n} {'exact' if r else 'DIFF'}" for n, r in zip(names, res)), flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
