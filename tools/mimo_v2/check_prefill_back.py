#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's prefill attention back kernel (models/mimo_v2/kernels/prefill.prefill_back), its
all-reduce inside, over a world of NeuronCores: every rank's (h, hn) must equal

    h  = x + all_reduce((attn / kv_scale)[:, :, :v_head_dim] @ o_w)      the all-reduce on the host (gloo)
    hn = RMSNorm(h)

each rank with its own seeded heads and weight, the same residual stream, at a bf16 (scale 1) and an FP8 (scale 0.37)
cache's scale.

    torchrun --nproc-per-node 4 tools/mimo_v2/check_prefill_back.py

Run on free NeuronCores (rank r on core --first-core + r)."""

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-core", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=1024)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(args.first_core + rank)
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62184")
    cache = os.environ.get("CHECK_CACHE", "/cache/check_prefill_back")
    for name, sub in (("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"), ("TORCH_NEURONX_HLO_CACHE_DIR", "hlo"),
                      ("NKI_COMPILE_CACHE_URL", "nki")):
        os.environ[name] = os.path.join(cache, sub, f"rank{rank}")

    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch
    import torch.distributed as dist

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_back
    from sglang_neuron.models.transformer.model import RMSNorm
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks
    from sglang_neuron.parallel.state import _make_group
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    init_neuron_ranks(rank, world, rank, world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _make_group([list(range(world))], rank, "mimo_prefill_back_check")
    mesh = (tuple(range(world)),)

    T, H, nq, d, vd = args.tokens, 4096, 1, 256, 128
    shared = torch.Generator().manual_seed(3)
    own = torch.Generator().manual_seed(200 + rank)
    x = (torch.randn(T, H, generator=shared) * 0.5).to(torch.bfloat16)
    norm = RMSNorm(H, 1e-6)
    norm.weight.data = (1 + 0.1 * torch.randn(H, generator=shared)).to(torch.bfloat16)
    attn = torch.randn(nq, T, d, generator=own).to(torch.bfloat16)
    o_w = (torch.randn(nq * vd, H, generator=own) * (nq * vd * world) ** -0.5).to(torch.bfloat16)

    ok = True
    for scale in (1.0, 0.37):
        heads = attn / scale if scale != 1.0 else attn
        part = heads.transpose(0, 1)[..., :vd].reshape(T, nq * vd) @ o_w
        total = part.float()
        dist.all_reduce(total)
        h_ref = x + total.to(torch.bfloat16)
        hn_ref = norm(h_ref)
        with stack.compile_scope(list(COMPILER_ARGS)):
            fn = torch.compile(lambda *a: wrap_nki(prefill_back)[get_logical_nc_config()](
                *a, mesh, eps=1e-6, v_head_dim=vd, kv_scale=scale), backend=stack.compile_backend_name(),
                fullgraph=True, dynamic=False)
            dev = torch.device("neuron", rank)
            h, hn = fn(attn.to(dev), o_w.to(dev), x.to(dev), norm.weight.view(1, -1).to(dev))
        h, hn = h.cpu().float(), hn.cpu().float()
        dh = (h - h_ref.float()).abs().max().item()
        dn = (hn - hn_ref.float()).abs().max().item()
        rel = ((h - h_ref.float()).norm() / h_ref.float().norm()).item()
        # two bf16 ulps at the largest magnitude: the in-kernel all-reduce sums in bf16, the host's in fp32
        import math

        ulp = lambda m: 2.0 ** (math.floor(math.log2(max(m, 1e-30))) - 7)  # noqa: E731
        # h within two bf16 ulps; hn within four (its scale reads the all-reduced row, one more rounding chain)
        good = (dh <= 2 * ulp(h_ref.float().abs().max().item()) and dn <= 4 * ulp(hn_ref.float().abs().max().item())
                and rel < 5e-3)
        ok &= good
        print(f"rank {rank}: scale {scale}: h max diff {dh:.3e} (rel L2 {rel:.2e}), hn max diff {dn:.3e}: "
              f"{'ok' if good else 'DIFF'}", flush=True)
    print(f"{'PASS' if ok else 'FAIL'} rank {rank}/{world}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
