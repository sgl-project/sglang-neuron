#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time a decode step's in-kernel all-reduces over a world of NeuronCores, three ways, for the one-kernel layer
(models/mimo_v2/kernels/layer.py):

- ``hbm``: the layer kernel's form today: the partial in shared HBM, a core barrier, the all-reduce HBM to HBM, a core
  barrier, and the residual add through SBUF back to HBM;
- ``sbuf``: nkilib transformer_tkg's ``sbuf_residual_and_cc`` form: each program's half of H in SBUF
  ([128, rows x H / 256]), the all-reduce SBUF to SBUF, the other program's half by sendrecv, the residual add in SBUF;
- ``xla``: the graph's all-reduce (parallel.collectives), the form the two half kernels run with.

Each runs ``--reps`` all-reduces of [rows, 4096] bf16 in one call (a step's 96 by default) and reports the mean time
of a call over ``--iters`` calls.

    torchrun --nproc-per-node 64 tools/mimo_v2/bench_cc.py --rows 32"""

import argparse
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-core", type=int, default=0)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--reps", type=int, default=96)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--modes", default="xla,hbm,sbuf")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(args.first_core + rank)
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62185")
    cache = os.environ.get("CHECK_CACHE", "/cache/bench_cc")
    for name, sub in (("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"), ("TORCH_NEURONX_HLO_CACHE_DIR", "hlo"),
                      ("NKI_COMPILE_CACHE_URL", "nki")):
        os.environ[name] = os.path.join(cache, sub, f"rank{rank}")

    from sglang_neuron.process_init import init_stack

    init_stack()
    import nki
    import nki.collectives as nccl
    import nki.isa as nisa
    import nki.language as nl
    import torch
    import torch.distributed as dist

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.parallel import collectives as neuron_collectives
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks
    from sglang_neuron.parallel.state import _make_group
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    init_neuron_ranks(rank, world, rank, world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    group = _make_group([list(range(world))], rank, "mimo_bench_cc")
    mesh = (tuple(range(world)),)
    H, R, REPS = 4096, args.rows, args.reps

    def replica_group(mesh_):
        cells = []
        for cell in mesh_:
            cells.append(list(cell))
        return nccl.ReplicaGroup(cells)

    @nki.jit
    def hbm_chain(x, mesh_, reps):
        B, Hd = x.shape
        n_prg = nl.num_programs(axes=0)
        rg = replica_group(mesh_)
        cur = x
        for i in range(reps):
            if n_prg > 1:
                nisa.core_barrier(data=cur, cores=(0, 1))
            red = nl.ndarray((B, Hd), dtype=x.dtype, buffer=nl.shared_hbm, name=f"red{i}")
            nccl.all_reduce(dsts=[red], srcs=[cur], op=nl.add, replica_group=rg)
            if n_prg > 1:
                nisa.core_barrier(data=red, cores=(0, 1))
            out = nl.ndarray((B, Hd), dtype=x.dtype, buffer=nl.shared_hbm, name=f"res{i}")
            a = nl.ndarray((B, Hd), dtype=x.dtype, buffer=nl.sbuf)
            b = nl.ndarray((B, Hd), dtype=x.dtype, buffer=nl.sbuf)
            s = nl.ndarray((B, Hd), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=a, src=cur)
            nisa.dma_copy(dst=b, src=red)
            nisa.tensor_tensor(dst=s, data1=a, data2=b, op=nl.add)
            nisa.dma_copy(dst=out, src=s)
            cur = out
        return cur

    @nki.jit
    def sbuf_chain(x, mesh_, reps):
        B, Hd = x.shape
        n_prg = nl.num_programs(axes=0)
        pid = nl.program_id(axis=0)
        rg = replica_group(mesh_)
        H0 = 128
        H1 = Hd // H0
        H1s = H1 // n_prg
        # this program's half of H as [128, B x H1s]: hidden index (pid x 128 + h0) x H1s + h1
        half = nl.ndarray((H0, B * H1s), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=half.reshape((H0, B, H1s)),
                      src=x.ap(pattern=[[H1s, H0], [Hd, B], [1, H1s]], offset=pid * H0 * H1s))
        # fixed buffers, reused every rep (a step's layers reuse theirs the same way)
        red = nl.ndarray((H0, B * H1s), dtype=x.dtype, buffer=nl.sbuf)
        other = nl.ndarray((H0, B * H1s), dtype=x.dtype, buffer=nl.sbuf)
        nxt = nl.ndarray((H0, B * H1s), dtype=x.dtype, buffer=nl.sbuf)
        for i in range(reps):
            src = half if i % 2 == 0 else nxt
            dst = nxt if i % 2 == 0 else half
            nccl.all_reduce(dsts=[red], srcs=[src], op=nl.add, replica_group=rg)
            if n_prg > 1:
                nisa.sendrecv(src=red, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
            nisa.tensor_tensor(dst=dst, data1=src, data2=red, op=nl.add)
        if reps % 2 == 1:
            nisa.tensor_copy(dst=half, src=nxt)
        out = nl.ndarray((B, Hd), dtype=x.dtype, buffer=nl.shared_hbm, name="sbuf_out")
        nisa.dma_copy(dst=out.ap(pattern=[[H1s, H0], [Hd, B], [1, H1s]], offset=pid * H0 * H1s),
                      src=half.reshape((H0, B, H1s)))
        return out

    def xla_chain(x):
        cur = x
        for _ in range(REPS):
            cur = cur + neuron_collectives.all_reduce(group, cur)
        return cur

    device = torch.device("neuron", rank)
    x = torch.zeros(R, H).to(torch.bfloat16).to(device)
    fns = {
        "xla": xla_chain,
        "hbm": lambda t: wrap_nki(hbm_chain)[get_logical_nc_config()](t, mesh, REPS),
        "sbuf": lambda t: wrap_nki(sbuf_chain)[get_logical_nc_config()](t, mesh, REPS),
    }
    for mode in args.modes.split(","):
        with stack.compile_scope(list(COMPILER_ARGS)):
            fn = torch.compile(fns[mode], backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
            try:
                out = fn(x)
                out.cpu()
                dist.barrier()
                t0 = time.time()
                for _ in range(args.iters):
                    out = fn(x)
                out.cpu()
                dist.barrier()
                ms = (time.time() - t0) / args.iters * 1000
                err = None
            except Exception as e:  # noqa: BLE001 - reported per mode
                ms, err = float("nan"), f"{type(e).__name__}: {str(e)[:300]}"
        torch._dynamo.reset()
        if rank == 0:
            print(f"{mode}: {REPS} all-reduces of [{R}, {H}] bf16 in {ms:.2f} ms a call ({ms / REPS * 1000:.1f} us "
                  f"each)" + (f"  FAILED {err}" if err else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
