#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's MTP prefill input kernel (models/mimo_v2/kernels/prefill.prefill_mtp_project), its
all-reduce and all-gather inside, over a world of NeuronCores against the torch it replaces: mtp.rotate and
mtp.MiMoMtp.project (the embedding, both RMSNorms and eh_proj, column-sharded over the ranks). The rotated ids must
match exactly, the layer input to bf16 rounding.

    torchrun --nproc-per-node 4 tools/mimo_v2/check_prefill_mtp.py"""

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
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62187")
    cache = os.environ.get("CHECK_CACHE", "/cache/check_prefill_mtp")
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
    from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_mtp_project
    from sglang_neuron.models.mimo_v2.mtp import rotate
    from sglang_neuron.models.transformer.model import RMSNorm
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks
    from sglang_neuron.parallel.state import _make_group
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    init_neuron_ranks(rank, world, rank, world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _make_group([list(range(world))], rank, "mimo_prefill_mtp_check")
    mesh = (tuple(range(world)),)

    # H 1024: over 4 ranks each holds 256 eh_proj columns (MiMo's 4096 over 64 ranks hold 64)
    T, H, Vr = args.tokens, 1024, 2048
    OR = H // world
    g = torch.Generator().manual_seed(21)
    table = torch.randn(world * Vr, H, generator=g).to(torch.bfloat16)
    ids = torch.randint(0, world * Vr, (T,), generator=g, dtype=torch.int32)
    at, token = T - 50, int(torch.randint(0, world * Vr, (1,), generator=g))
    prev = (torch.randn(T, H, generator=g) * 2).to(torch.bfloat16)
    enorm, hnorm = RMSNorm(H, 1e-6), RMSNorm(H, 1e-6)
    enorm.weight.data = (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)
    hnorm.weight.data = (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)
    eh = (torch.randn(H, 2 * H, generator=g) * (2 * H) ** -0.5).to(torch.bfloat16)  # eh_proj.weight [H, 2H]

    nxt = rotate(ids.view(1, -1), torch.tensor([at]), torch.tensor([token], dtype=torch.int32)).view(-1)
    ref = torch.cat([enorm(table[nxt.long()]), hnorm(prev)], dim=-1) @ eh.t()
    eh_wT = eh[rank * OR:(rank + 1) * OR].t().contiguous()
    dev = torch.device("neuron", rank)
    with stack.compile_scope(list(COMPILER_ARGS)):
        fn = torch.compile(lambda *a: wrap_nki(prefill_mtp_project)[get_logical_nc_config()](*a, mesh, eps=1e-6),
                           backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
        x, nx = fn(ids.view(-1, 1).to(dev), torch.tensor([[at]], dtype=torch.int32).to(dev),
                   torch.tensor([[token]], dtype=torch.int32).to(dev), prev.to(dev),
                   table[rank * Vr:(rank + 1) * Vr].to(dev), enorm.weight.view(1, -1).to(dev),
                   hnorm.weight.view(1, -1).to(dev), eh_wT.to(dev), torch.tensor([rank], dtype=torch.int32).to(dev))
    x, nx = x.cpu(), nx.cpu().view(-1)
    same = torch.equal(nx.long(), nxt.long())
    d = (x.float() - ref.float()).abs().max().item()
    rel = ((x.float() - ref.float()).norm() / ref.float().norm()).item()
    ulp = 2.0 ** (math.floor(math.log2(ref.float().abs().max().item())) - 7)
    ok = same and d <= 2 * ulp and rel < 5e-3
    print(f"rank {rank}: rotated ids {'exact' if same else 'DIFF'}; x max diff {d:.3e} (rel L2 {rel:.2e})", flush=True)
    print(f"{'PASS' if ok else 'FAIL'} rank {rank}/{world}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
