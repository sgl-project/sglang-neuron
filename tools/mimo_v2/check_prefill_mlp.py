#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's prefill dense MLP and residual kernels (models/mimo_v2/kernels/prefill.py prefill_mlp
and prefill_residual), their all-reduces inside, over a world of NeuronCores against the torch with the all-reduce on
the host (gloo):

    prefill_mlp       h = x + all_reduce((silu(hn @ gate) x (hn @ up)) @ down)      model.DenseMLP + the residual add
    prefill_residual  h = x + all_reduce(partial)                                    the MoE layer's residual add

each rank with its own seeded weights and partial, the same rows.

    torchrun --nproc-per-node 4 tools/mimo_v2/check_prefill_mlp.py"""

import argparse
import math
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
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62185")
    cache = os.environ.get("CHECK_CACHE", "/cache/check_prefill_mlp")
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
    from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_mlp, prefill_residual
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks
    from sglang_neuron.parallel.state import _make_group
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    init_neuron_ranks(rank, world, rank, world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _make_group([list(range(world))], rank, "mimo_prefill_mlp_check")
    mesh = (tuple(range(world)),)

    T, H, I = args.tokens, 4096, 256  # noqa: E741
    shared = torch.Generator().manual_seed(4)
    own = torch.Generator().manual_seed(300 + rank)
    x = (torch.randn(T, H, generator=shared) * 0.5).to(torch.bfloat16)
    hn = torch.randn(T, H, generator=shared).to(torch.bfloat16)
    gate = (torch.randn(H, I, generator=own) * H ** -0.5).to(torch.bfloat16)
    up = (torch.randn(H, I, generator=own) * H ** -0.5).to(torch.bfloat16)
    down = (torch.randn(I, H, generator=own) * (I * world) ** -0.5).to(torch.bfloat16)
    partial = (torch.randn(T, H, generator=own) * 0.3).to(torch.bfloat16)

    def reference(part):
        total = part.float()
        dist.all_reduce(total)
        return x + total.to(torch.bfloat16)

    mlp_ref = reference((torch.nn.functional.silu(hn @ gate) * (hn @ up)) @ down)
    res_ref = reference(partial)
    dev = torch.device("neuron", rank)
    with stack.compile_scope(list(COMPILER_ARGS)):
        grid = get_logical_nc_config()
        f_mlp = torch.compile(lambda *a: wrap_nki(prefill_mlp)[grid](*a, mesh), backend=stack.compile_backend_name(),
                              fullgraph=True, dynamic=False)
        h_mlp = f_mlp(hn.to(dev), x.to(dev), gate.to(dev), up.to(dev), down.to(dev)).cpu()
        f_res = torch.compile(lambda *a: wrap_nki(prefill_residual)[grid](*a, mesh),
                              backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
        h_res = f_res(x.to(dev), partial.to(dev)).cpu()

    def ulp(m):
        return 2.0 ** (math.floor(math.log2(max(m, 1e-30))) - 7)

    ok = True
    for name, got, ref in (("prefill_mlp", h_mlp, mlp_ref), ("prefill_residual", h_res, res_ref)):
        d = (got.float() - ref.float()).abs().max().item()
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        good = d <= 2 * ulp(ref.float().abs().max().item()) and rel < 5e-3
        ok &= good
        print(f"rank {rank}: {name}: max diff {d:.3e} (rel L2 {rel:.2e}): {'ok' if good else 'DIFF'}", flush=True)
    print(f"{'PASS' if ok else 'FAIL'} rank {rank}/{world}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
