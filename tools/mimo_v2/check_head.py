#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's step head and tail kernels (models/mimo_v2/kernels/head.py) over a world of
NeuronCores: ``embed`` must equal the vocabulary-sharded embedding summed over the ranks (exactly: one rank holds each
row), and ``greedy`` the argmax of the bf16 logits of the RMSNorm'd rows over the whole vocabulary.

    torchrun --nproc-per-node 8 tools/mimo_v2/check_head.py"""

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-core", type=int, default=0)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--vr", type=int, default=2384, help="vocabulary rows a rank (152576 / 64 served)")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(args.first_core + rank)
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62187")
    cache = os.environ.get("CHECK_CACHE", "/cache/check_head")
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
    from sglang_neuron.models.mimo_v2.kernels.head import embed, greedy
    from sglang_neuron.models.mimo_v2.kernels.mtp_head import mtp_project
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks
    from sglang_neuron.parallel.state import _make_group
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    init_neuron_ranks(rank, world, rank, world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _make_group([list(range(world))], rank, "mimo_head_check")
    mesh = (tuple(range(world)),)
    H, T, Vr, eps = 4096, args.rows, args.vr, 1e-6
    V = Vr * world
    shared = torch.Generator().manual_seed(7)
    full_emb = (0.02 * torch.randn(V, H, generator=shared)).to(torch.bfloat16)
    full_lm = (0.02 * torch.randn(V, H, generator=shared)).to(torch.bfloat16)
    norm_w = (1 + 0.1 * torch.randn(1, H, generator=shared)).to(torch.bfloat16)
    ids = torch.randint(0, V, (T, 1), generator=shared).to(torch.int32)
    x = torch.randn(T, H, generator=shared).to(torch.bfloat16)
    # a planted tie across two ranks' shards: the lower id must win
    my_emb = full_emb[rank * Vr:(rank + 1) * Vr].contiguous()
    my_lmT = full_lm[rank * Vr:(rank + 1) * Vr].t().contiguous()
    start = torch.tensor([rank], dtype=torch.int32)
    device = torch.device("neuron", rank)
    d = lambda t: t.contiguous().to(device)  # noqa: E731

    def compiled(fn, *inputs):
        with stack.compile_scope(list(COMPILER_ARGS)):
            out = torch.compile(fn, backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)(*inputs)
        torch._dynamo.reset()
        return out

    got_emb = compiled(lambda i, w, s: wrap_nki(embed)[get_logical_nc_config()](i, w, s, mesh),
                       d(ids), d(my_emb), d(start)).cpu()
    want_emb = full_emb[ids.reshape(-1).long()]
    emb_ok = torch.equal(got_emb, want_emb)

    got_tok = compiled(lambda t, g, w, s: wrap_nki(greedy)[get_logical_nc_config()](t, g, w, s, mesh, eps=eps),
                       d(x), d(norm_w), d(my_lmT), d(start)).cpu().reshape(-1)
    xf = x.float()
    normed = (norm_w.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).bfloat16().float()
              ).bfloat16().float()
    logits = (normed @ full_lm.float().t()).bfloat16().float()
    want_tok = torch.argmax(logits, dim=-1).to(torch.int32)
    # the kernel's fp32 accumulation order differs from torch's: a near-tie may flip; count exact agreement and check
    # that a disagreement is within one bf16 step of the max
    agree = (got_tok == want_tok)
    gap = (logits.gather(1, want_tok.long().view(-1, 1)) - logits.gather(1, got_tok.long().clamp(0, V - 1).view(-1, 1)))
    near = bool((gap.abs().view(-1)[~agree] <= logits.abs().max() * 2 ** -7).all()) if (~agree).any() else True
    tok_ok = bool(agree.float().mean() >= 0.9) and near and bool(((got_tok >= 0) & (got_tok < V)).all())
    # the MTP layer's input: eh_proj([enorm(embed(ids)) | hnorm(x)]), this rank's output columns all-gathered
    OR = H // world
    enorm_w = (1 + 0.1 * torch.randn(1, H, generator=shared)).to(torch.bfloat16)
    hnorm_w = (1 + 0.1 * torch.randn(1, H, generator=shared)).to(torch.bfloat16)
    eh_full = (0.02 * torch.randn(H, 2 * H, generator=shared)).to(torch.bfloat16)  # [out, in], as the checkpoint
    my_ehT = eh_full[rank * OR:(rank + 1) * OR].t().contiguous()
    got_x = compiled(lambda i, p_, w, en, hn, eh, s: wrap_nki(mtp_project)[get_logical_nc_config()](
        i, p_, w, en, hn, eh, s, mesh, eps=eps), d(ids), d(x), d(my_emb), d(enorm_w), d(hnorm_w), d(my_ehT),
        d(start)).cpu().float()

    def rms(t, w):
        t32 = t.float()
        return (w.float() * (t32 * torch.rsqrt(t32.pow(2).mean(-1, keepdim=True) + eps)).bfloat16().float()).bfloat16()

    want_x = (torch.cat([rms(want_emb, enorm_w), rms(x, hnorm_w)], dim=-1).float() @ eh_full.float().t()).bfloat16()
    rel_x = ((got_x - want_x.float()).norm() / want_x.float().norm()).item()
    proj_ok = rel_x < 1e-2
    print(f"  rank {rank}: mtp_project vs torch relative L2 {rel_x:.2e}", flush=True)
    ok = emb_ok and tok_ok and proj_ok
    print(f"{'PASS' if ok else 'FAIL'} rank {rank}/{world}, {T} rows, vocab {V}: embed exact {emb_ok}; greedy "
          f"agrees on {int(agree.sum())}/{T} rows (the rest near-ties: {near})", flush=True)
    flag = torch.tensor([0.0 if ok else 1.0])
    dist.all_reduce(flag)
    return 1 if int(flag) else 0


if __name__ == "__main__":
    sys.exit(main())
