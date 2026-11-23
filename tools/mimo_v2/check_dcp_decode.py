#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's DCP decode attention (kernels/attention_block.decode_masked_dcp: the vendored
attention_block_tkg whose context-parallel path takes heads of 256 channels, kernels/cp) over a world of NeuronCores,
one DCP group: every rank holds the same KV head's sequence in 128-token blocks kept round-robin
(distributed.decode_context_parallel.DcpLayout), its own query head, and must return its head's attention over the
WHOLE sequence and the row's new token, computed here in torch over the unsharded sequence. With and without sinks (the
group's sinks real on rank 0 only, decode_context_parallel.group_sinks).

    torchrun --nproc-per-node 8 tools/mimo_v2/check_dcp_decode.py

Run on free NeuronCores (rank r on core --first-core + r)."""

import argparse
import os
import sys

D, VD, H, PAGE = 256, 128, 4096, 128
LENGTHS = [5, 300, 700, 1000]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-core", type=int, default=0)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(args.first_core + rank)
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62184")
    cache = os.environ.get("CHECK_CACHE", "/tmp/check_dcp")
    for name, sub in (("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"), ("TORCH_NEURONX_HLO_CACHE_DIR", "hlo"),
                      ("NKI_COMPILE_CACHE_URL", "nki")):
        os.environ[name] = os.path.join(cache, sub, f"rank{rank}")
    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch
    import torch.distributed as dist

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.distributed.decode_context_parallel import IDLE_SINK
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.mimo_v2.kernels.attention_block import decode_masked_dcp
    from sglang_neuron.models.mimo_v2.rope import RotaryTables, rotate
    from sglang_neuron.ops.attention.attention_decode import dcp_collective_mode
    from sglang_neuron.models.mimo_v2.attention import qk_swap_layout
    from sglang_neuron.ops.attention.attention_decode_mask import gen_attention_decode_mask
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks
    from sglang_neuron.parallel.state import _make_group
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    init_neuron_ranks(rank, world, rank, world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _make_group([list(range(world))], rank, "mimo_dcp_check")
    mesh = (tuple(range(world)),)
    rows, eps, scale = len(LENGTHS), 1e-6, 192 ** -0.5
    shared = torch.Generator().manual_seed(1)
    # the shared KV head's projection and sequence, and every rank's query head
    w_k = 0.02 * torch.randn(H, D, generator=shared)
    w_v = 0.02 * torch.randn(H, D, generator=shared)
    w_q_all = 0.02 * torch.randn(world, H, D, generator=shared)
    for w in (w_k, w_q_all):
        w[..., 192:] = 0
    w_v[:, VD:] = 0
    x = torch.randn(rows, H, generator=shared).to(torch.bfloat16)
    ln_w = (1 + 0.1 * torch.randn(H, generator=shared)).to(torch.bfloat16)
    max_len = max(LENGTHS)
    k_seq = torch.randn(rows, max_len, D, generator=shared).to(torch.bfloat16)
    v_seq = torch.randn(rows, max_len, D, generator=shared).to(torch.bfloat16)
    k_seq[..., 192:] = 0
    v_seq[..., VD:] = 0
    sinks_all = torch.tensor([1.5, -0.5, 0.7, 2.0, 0.1, -1.0, 0.3, 1.1])[:world]
    w_qkv = torch.cat([w_q_all[rank], w_k, w_v], dim=1).to(torch.bfloat16)

    # this rank's shard: global block j of a row lives on rank j % world, as its (j // world)-th local block
    blocks_per_row = -(-(max_len + 1) // (PAGE * world)) + 1
    pages = 1 + rows * blocks_per_row
    k_cache = torch.zeros(pages, 1, PAGE, D, dtype=torch.bfloat16)
    v_cache = torch.zeros(pages, 1, PAGE, D, dtype=torch.bfloat16)
    table = torch.full((rows, blocks_per_row), -1, dtype=torch.int32)
    end = torch.zeros(rows)
    owner = torch.zeros(rows, dtype=torch.int32)
    for b, length in enumerate(LENGTHS):
        for i in range(blocks_per_row):
            table[b, i] = 1 + b * blocks_per_row + i
        kept = 0
        for p in range(length):
            j = p // PAGE
            if j % world == rank:
                local = j // world
                k_cache[table[b, local], 0, p % PAGE] = k_seq[b, p]
                v_cache[table[b, local], 0, p % PAGE] = v_seq[b, p]
                kept += 1
        end[b] = kept
        owner[b] = int((length // PAGE) % world == rank)
    positions = torch.tensor(LENGTHS)
    cos, sin = RotaryTables(1e7, 64)(positions)
    mask = gen_attention_decode_mask(end.reshape(1, rows), bs=rows, q_head=world, s_active=1,
                                     s_prior=blocks_per_row * PAGE, block_len=PAGE, active_gate=owner)
    mask = qk_swap_layout(mask, d_head=D, fp8_kv=False, fp8_packed=False)

    def reference(with_sink: bool):
        """[B, D] this rank's head over the whole sequence and the new token."""
        xn = x.float()
        xn = (ln_w.float() * (xn * torch.rsqrt(xn.pow(2).mean(-1, keepdim=True) + eps)).bfloat16().float())
        xn = xn.bfloat16().float()
        q = (xn @ w_q_all[rank].bfloat16().float()).view(rows, 1, D)
        k = (xn @ w_k.bfloat16().float()).view(rows, 1, D)
        v = (xn @ w_v.bfloat16().float()).bfloat16().float()
        q, k = rotate(q, cos, sin)[:, 0].bfloat16().float(), rotate(k, cos, sin)[:, 0].bfloat16().float()
        out = torch.zeros(rows, D)
        for b, length in enumerate(LENGTHS):
            keys = torch.cat([k_seq[b, :length].float(), k[b:b + 1]])
            vals = torch.cat([v_seq[b, :length].float(), v[b:b + 1]])
            scores = (keys @ q[b]) * scale
            if with_sink:
                scores = torch.cat([scores, sinks_all[rank].reshape(1)])
            probs = torch.softmax(scores, dim=0)[: length + 1]
            out[b] = probs @ vals
        return out

    device = torch.device("neuron", rank)
    failures = 0
    for with_sink in (False, True):
        group_sinks = sinks_all if rank == 0 else torch.full_like(sinks_all, IDLE_SINK)
        inputs = [x.view(rows, 1, H), ln_w.view(1, H).float(), w_qkv, cos.t().unsqueeze(-1).contiguous(),
                  sin.t().unsqueeze(-1).contiguous(), table, k_cache, v_cache, mask]
        if with_sink:
            inputs.append(group_sinks.reshape(world, 1).float())
        dev = [t.contiguous().to(device) for t in inputs]

        def fn(X, gamma, wq, c, s, tbl, kc, vc, m, *rest):
            return wrap_nki(decode_masked_dcp)[get_logical_nc_config()](
                X=X, X_hidden_dim_actual=H, rmsnorm_X_eps=eps, rmsnorm_X_gamma=gamma, W_qkv=wq, cos=c, sin=s,
                active_blocks_table=tbl, K_cache=kc, V_cache=vc, attention_mask=m, sink=rest[0] if rest else None,
                softmax_scale=scale, k_scale=None, v_scale=None, dcp_size=world, dcp_mesh=mesh,
                dcp_mode=dcp_collective_mode(world))

        with stack.compile_scope(COMPILER_ARGS):
            out, _, _ = torch.compile(fn, backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)(*dev)
        torch._dynamo.reset()
        got = out.cpu().float().reshape(rows, 1, D, 1)[:, 0, :, 0]
        want = reference(with_sink)
        rel = ((got - want).norm() / want.norm()).item()
        ok = rel < 2e-2
        failures += not ok
        print(f"rank {rank} {'PASS' if ok else 'FAIL'} sink {with_sink}: relative L2 {rel:.2e}", flush=True)
    dist.barrier()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
