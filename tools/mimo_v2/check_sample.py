#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's decode sampler kernel (models/mimo_v2/kernels/sample.py) over a world of NeuronCores,
against its rule in torch: every rank's top 16 of each half of its vocabulary shard, the pool's top 128, then
sampling.tail_rule_sample with the platform's draw (row_draws(noise_row_keys(seed, position))). Greedy rows must take
the argmax. The draws themselves are compared too (the kernel hashes on int32 lanes).

    torchrun --nproc-per-node 8 tools/mimo_v2/check_sample.py"""

import os
import sys


def main() -> int:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(rank)
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62191")
    cache = os.environ.get("CHECK_CACHE", "/cache/check_sample")
    for name, sub in (("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"), ("TORCH_NEURONX_HLO_CACHE_DIR", "hlo"),
                      ("NKI_COMPILE_CACHE_URL", "nki")):
        os.environ[name] = os.path.join(cache, sub, f"rank{rank}")
    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch
    import torch.distributed as dist

    from sglang_neuron import sampling
    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.mimo_v2.kernels.sample import K_POOL, K_RANK, sample
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks
    from sglang_neuron.parallel.state import _make_group
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    init_neuron_ranks(rank, world, rank, world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _make_group([list(range(world))], rank, "mimo_sample_check")
    mesh = (tuple(range(world)),)
    H, T, Vr, eps = 4096, 32, 2384, 1e-6
    V = Vr * world
    g = torch.Generator().manual_seed(11)
    full_lm = (0.05 * torch.randn(V, H, generator=g)).to(torch.bfloat16)
    norm_w = (1 + 0.1 * torch.randn(1, H, generator=g)).to(torch.bfloat16)
    x = torch.randn(T, H, generator=g).to(torch.bfloat16)
    temps = torch.tensor([0.0 if r % 4 == 0 else (0.6 if r % 4 == 1 else 1.0) for r in range(T)]).view(T, 1)
    filters = torch.tensor([[sampling.TOP_K_ALL if r % 3 else 20, 0.95 if r % 2 else 1.0, 0.0] for r in range(T)],
                           dtype=torch.float32)
    seeds = sampling.fold_seeds(torch.randint(0, 2 ** 62, (T,), generator=g)).view(T, 1)
    positions = torch.randint(0, 1 << 20, (T, 1), generator=g).to(torch.int32)
    my_lmT = full_lm[rank * Vr:(rank + 1) * Vr].t().contiguous()
    device = torch.device("neuron", rank)
    d = lambda t: t.contiguous().to(device)  # noqa: E731
    with stack.compile_scope(list(COMPILER_ARGS)):
        fn = torch.compile(lambda *a: wrap_nki(sample)[get_logical_nc_config()](
            a[0], a[1], a[2], a[3], mesh, a[4], a[5], a[6], a[7], eps=eps, span=1), backend=stack.compile_backend_name(),
            fullgraph=True, dynamic=False)
        got = fn(d(x), d(norm_w), d(my_lmT), d(torch.tensor([rank], dtype=torch.int32)), d(temps), d(seeds),
                 d(filters), d(positions)).cpu().reshape(-1)
    # torch: the rule over the same candidates
    xf = x.float()
    normed = (norm_w.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).bfloat16().float()
              ).bfloat16().float()
    logits = (normed @ full_lm.float().t()).bfloat16().float()
    half = Vr // 2
    pool_v, pool_i = [], []
    for prog in range(2):
        for r in range(world):
            lo = r * Vr + prog * half
            v, i = logits[:, lo:lo + half].topk(K_RANK, dim=-1)
            pool_v.append(v)
            pool_i.append(i + lo)
    pool_v, pool_i = torch.cat(pool_v, -1), torch.cat(pool_i, -1)
    vals, order = pool_v.topk(K_POOL, dim=-1)
    ids = pool_i.gather(1, order)
    draws = sampling.row_draws(sampling.noise_row_keys(seeds.view(-1), positions.view(-1))).view(T, 1)
    params = torch.cat([filters[:, 0:1].clamp(max=K_POOL), filters[:, 1:2], temps, draws], dim=1)
    want = sampling.tail_rule_sample(vals, ids, params).to(torch.int32)
    greedy = temps.view(-1) < 1e-5
    agree = got == want
    greedy_ok = bool((got[greedy] == logits[greedy].argmax(-1).to(torch.int32)).all())
    # where a draw disagrees: the kernel's and the rule's candidate places (nearby places are a bf16 logit
    # rounding moving a draw in a flat tail, far ones a wrong draw)
    places = []
    for r in torch.nonzero(~agree).view(-1).tolist():
        hit = (ids[r] == got[r]).nonzero().view(-1)
        places.append((r, int(hit[0]) if len(hit) else -1, int((ids[r] == want[r]).nonzero().view(-1)[0])))
    near = all(abs(a - b) <= 5 for _, a, b in places)
    if rank == 0:
        print(f"  (row, kernel place, rule place): {places}", flush=True)
    ok = agree.float().mean().item() >= 0.75 and greedy_ok and near
    print(f"{'PASS' if ok else 'FAIL'} rank {rank}/{world}: sampled tokens agree with the torch rule on "
          f"{int(agree.sum())}/{T} rows; greedy rows argmax {greedy_ok}; disagreeing rows "
          f"{torch.nonzero(~agree).view(-1).tolist()[:8]}", flush=True)
    flag = torch.tensor([0.0 if ok else 1.0])
    dist.all_reduce(flag)
    return 1 if int(flag) else 0


if __name__ == "__main__":
    sys.exit(main())
