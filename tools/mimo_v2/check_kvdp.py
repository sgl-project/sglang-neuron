#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of nkilib's KV data parallelism on MiMo-V2's 256-channel heads (kernels/attention_block.
decode_posid_kvdp) over one KVDP group of NeuronCores: every rank holds the same KV head and cache and its own query
head; each attends B / world rows, and every rank must get its own head's attention over all B rows, computed here
in torch.

    torchrun --nproc-per-node 8 tools/mimo_v2/check_kvdp.py"""

import argparse
import os
import sys

D, VD, H, PAGE, BLOCKS = 256, 128, 4096, 128, 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--first-core", type=int, default=0)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(args.first_core + rank)
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62189")
    cache = os.environ.get("CHECK_CACHE", "/tmp/check_kvdp")
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
    from sglang_neuron.models.mimo_v2.kernels.attention_block import decode_posid_kvdp
    from sglang_neuron.models.mimo_v2.rope import RotaryTables, rotate
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks
    from sglang_neuron.parallel.state import _make_group
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    init_neuron_ranks(rank, world, rank, world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _make_group([list(range(world))], rank, "mimo_kvdp_check")
    mesh = (tuple(range(world)),)
    B, eps, scale = world, 1e-6, 192 ** -0.5
    Ba = B // world
    shared = torch.Generator().manual_seed(3)
    lengths = torch.randint(1, BLOCKS * PAGE - 1, (B,), generator=shared)
    x = torch.randn(B, H, generator=shared).to(torch.bfloat16)
    w_k = (0.02 * torch.randn(H, D, generator=shared))
    w_v = (0.02 * torch.randn(H, D, generator=shared))
    w_k[:, 192:] = 0
    w_v[:, VD:] = 0
    k_cache = torch.randn(B * BLOCKS + 1, 1, PAGE, D, generator=shared).to(torch.bfloat16)
    v_cache = torch.randn(B * BLOCKS + 1, 1, PAGE, D, generator=shared).to(torch.bfloat16)
    k_cache[..., 192:] = 0
    v_cache[..., VD:] = 0
    table = torch.stack([1 + b * BLOCKS + torch.arange(BLOCKS) for b in range(B)]).to(torch.int32)
    own = torch.Generator().manual_seed(50 + rank)
    w_q = 0.02 * torch.randn(H, D, generator=own)
    w_q[:, 192:] = 0
    w_qkv = torch.cat([w_q, w_k, w_v], dim=1).to(torch.bfloat16)
    ln_w = torch.ones(1, H)
    cos, sin = RotaryTables(1e7, 64)(lengths)

    # torch: this rank's head over every row
    want = torch.zeros(B, D)
    xf = x.float()
    normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).bfloat16().float()
    qkv = normed @ w_qkv.float()
    q, k, v = (t.view(B, 1, D) for t in qkv.split(D, dim=-1))
    q, k = rotate(q, cos, sin)[:, 0], rotate(k, cos, sin)[:, 0]
    q, k, v = q.bfloat16().float(), k.bfloat16().float(), v[:, 0].bfloat16().float()
    for b in range(B):
        pos = int(lengths[b])
        pages = table[b][: pos // PAGE + 1]
        keys = torch.cat([k_cache[pages][:, 0].reshape(-1, D)[:pos].float(), k[b:b + 1]])
        vals = torch.cat([v_cache[pages][:, 0].reshape(-1, D)[:pos].float(), v[b:b + 1]])
        probs = torch.softmax((keys @ q[b]) * scale, dim=0)
        want[b] = (probs @ vals).bfloat16().float()

    my_rows = slice(rank * Ba, (rank + 1) * Ba)
    q_heads_attn = world
    active = torch.ones((1, Ba, q_heads_attn, 1), dtype=torch.uint8)
    pos = lengths.to(torch.float32).view(B, 1)[my_rows].contiguous()
    kvdp_rank = torch.tensor([rank], dtype=torch.uint32)
    device = torch.device("neuron", rank)
    d = lambda t: t.contiguous().to(device)  # noqa: E731
    inputs = [x.view(B, 1, H), ln_w, w_qkv, cos.t().unsqueeze(-1).contiguous(), sin.t().unsqueeze(-1).contiguous(),
              table[my_rows], k_cache, v_cache, active, pos, kvdp_rank]

    def fn(X, g, w, c, s, tbl, kc, vc, m, p, r):
        return wrap_nki(decode_posid_kvdp)[get_logical_nc_config()](
            X, H, eps, g, w, c, s, tbl, kc, vc, m, None, scale, p, r, mesh, world)

    ok = False
    try:
        with stack.compile_scope(list(COMPILER_ARGS)):
            heads, k_new, v_new = torch.compile(fn, backend=stack.compile_backend_name(), fullgraph=True,
                                                dynamic=False)(*[d(t) for t in inputs])
        got = heads.cpu().float().reshape(B, -1, D, 1)[:, 0, :, 0]
        rel = ((got - want).norm() / want.norm()).item()
        ok = rel < 2e-2
        print(f"{'PASS' if ok else 'FAIL'} rank {rank}/{world}: KVDP heads vs torch relative L2 {rel:.2e}; heads "
              f"{tuple(heads.shape)} K {tuple(k_new.shape)} V {tuple(v_new.shape)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL rank {rank}: {type(e).__name__}: " + str(e)[-1500:].replace(chr(10), " | "), flush=True)
    flag = torch.tensor([0.0 if ok else 1.0])
    dist.all_reduce(flag)
    return 1 if int(flag) else 0


if __name__ == "__main__":
    sys.exit(main())
