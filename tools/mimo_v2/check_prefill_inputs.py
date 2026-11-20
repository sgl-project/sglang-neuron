#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's prefill inputs kernel (models/mimo_v2/kernels/prefill.prefill_inputs), its all-reduce
inside, over a world of NeuronCores: the embeddings must equal the whole table's rows exactly (each id held by one
rank's vocabulary shard), and both kinds' RoPE tables rope.RotaryTables' to bf16 rounding.

    torchrun --nproc-per-node 4 tools/mimo_v2/check_prefill_inputs.py"""

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
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62186")
    cache = os.environ.get("CHECK_CACHE", "/cache/check_prefill_inputs")
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
    from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_inputs
    from sglang_neuron.models.mimo_v2.rope import RotaryTables
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks
    from sglang_neuron.parallel.state import _make_group
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    init_neuron_ranks(rank, world, rank, world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _make_group([list(range(world))], rank, "mimo_prefill_inputs_check")
    mesh = (tuple(range(world)),)

    T, H, Vr = args.tokens, 4096, 2048
    g = torch.Generator().manual_seed(9)
    table = torch.randn(world * Vr, H, generator=g).to(torch.bfloat16)
    ids = torch.randint(0, world * Vr, (T,), generator=g, dtype=torch.int32)
    positions = (torch.arange(T, dtype=torch.int32) + 123457)
    positions[-40:] = positions[-41]
    full, swa = RotaryTables(10000000.0, 64), RotaryTables(10000.0, 64)
    dev = torch.device("neuron", rank)
    with stack.compile_scope(list(COMPILER_ARGS)):
        fn = torch.compile(lambda i, p, w, r, f, s: wrap_nki(prefill_inputs)[get_logical_nc_config()](i, p, w, r, mesh,
                                                                                                    f, s),
                           backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
        outs = fn(ids.view(-1, 1).to(dev), positions.view(-1, 1).to(dev), table[rank * Vr:(rank + 1) * Vr].to(dev),
                  torch.tensor([rank], dtype=torch.int32).to(dev), full.inv_freq.view(1, -1).to(dev),
                  swa.inv_freq.view(1, -1).to(dev))
    emb, fc, fs, sc, ss = (o.cpu() for o in outs)
    ok = torch.equal(emb, table[ids.long()])
    print(f"rank {rank}: embeddings {'exact' if ok else 'DIFF'}", flush=True)
    for name, got, ref in (("full cos", fc, full(positions)[0]), ("full sin", fs, full(positions)[1]),
                           ("swa cos", sc, swa(positions)[0]), ("swa sin", ss, swa(positions)[1])):
        d = (got.float() - ref.float()).abs().max().item()
        ok &= d < 2e-2
        print(f"rank {rank}: {name} max diff {d:.2e}", flush=True)
    print(f"{'PASS' if ok else 'FAIL'} rank {rank}/{world}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
