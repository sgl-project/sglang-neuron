#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's DCP sliding-layer prefill KV kernel (models/mimo_v2/kernels/prefill.prefill_dcp_window)
over one DCP group of NeuronCores against the torch it replaces (attention._prefill_dcp's sliding branch): the
window's prior page (the group's last rank's), then the chunk's pages, in bf16; the local table and prior length.

    torchrun --nproc-per-node 8 tools/mimo_v2/check_prefill_dcp.py"""

import os
import sys


def main() -> int:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(rank)
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62188")
    os.environ["UNSAFE_FP8FNCAST"] = "1"
    cache = os.environ.get("CHECK_CACHE", "/cache/check_prefill_dcp")
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
    from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_dcp_window
    from sglang_neuron.models.mimo_v2.launch import FP8_FN_AS_E4M3
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks
    from sglang_neuron.parallel.state import _make_group
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    init_neuron_ranks(rank, world, rank, world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _make_group([list(range(world))], rank, "mimo_prefill_dcp_check")
    mesh = (tuple(range(world)),)
    prefix = "--internal-hlo2tensorizer-options="
    cc = [f"{a} {FP8_FN_AS_E4M3}" if a.startswith(prefix) else a for a in COMPILER_ARGS]
    if not any(a.startswith(prefix) for a in cc):
        cc.append(prefix + FP8_FN_AS_E4M3)

    blocks, page, d, width, T = 40, 128, 256, 24, 512
    shared = torch.Generator().manual_seed(17)
    own = torch.Generator().manual_seed(400 + rank)
    table = torch.randperm(blocks - 1, generator=shared)[:width].to(torch.int32).view(1, width) + 1
    ok = True
    for fp8, scale, cached in ((False, 1.0, world * page * 3), (True, 0.37, world * page * 5), (False, 1.0, 0)):
        dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        kc = (torch.randn(blocks, 1, page, d, generator=own) * 4).clamp(-240, 240).to(dtype)
        vc = (torch.randn(blocks, 1, page, d, generator=own) * 4).clamp(-240, 240).to(dtype)
        kn = (torch.randn(T, d, generator=own) * 4).clamp(-240, 240).to(dtype)
        vn = (torch.randn(T, d, generator=own) * 4).clamp(-240, 240).to(dtype)

        def deq(t):
            return (t.float() / scale).to(torch.bfloat16) if fp8 else t.to(torch.bfloat16)

        col = max(cached // (world * page) - 1, 0)
        page_id = max(int(table[0, col]), 0)
        mine = torch.stack([deq(kc[page_id, 0]), deq(vc[page_id, 0])])
        every = [torch.empty_like(mine) for _ in range(world)]
        dist.all_gather(every, mine)
        last = every[-1]
        k_ref = torch.cat([last[0].view(1, 1, page, d), deq(kn).view(T // page, page, 1, d).transpose(1, 2)])
        v_ref = torch.cat([last[1].view(1, 1, page, d), deq(vn).view(T // page, page, 1, d).transpose(1, 2)])
        n = 1 + T // page
        first = 1 if cached == 0 else 0
        t_ref = torch.clamp(torch.arange(n) + first, max=n - 1).view(1, n)
        p_ref = page if cached > 0 else 0
        dev = torch.device("neuron", rank)
        with stack.compile_scope(cc):
            fn = torch.compile(lambda *a: wrap_nki(prefill_dcp_window)[get_logical_nc_config()](
                *a, mesh, dcp=world, kv_scale=scale), backend=stack.compile_backend_name(), fullgraph=True,
                dynamic=False)
            outs = fn(kc.to(dev), vc.to(dev), table.to(dev), torch.tensor([[cached]], dtype=torch.int32).to(dev),
                      kn.to(dev), vn.to(dev))
        k_got, v_got, t_got, p_got = (o.cpu() for o in outs)
        res = (torch.equal(k_got, k_ref), torch.equal(v_got, v_ref), torch.equal(t_got.long(), t_ref.long()),
               int(p_got.view(-1)[0]) == p_ref)
        ok &= all(res)
        print(f"rank {rank}: fp8 {fp8} cached {cached}: K {res[0]} V {res[1]} table {res[2]} prior {res[3]}",
              flush=True)
    print(f"{'PASS' if ok else 'FAIL'} rank {rank}/{world}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
