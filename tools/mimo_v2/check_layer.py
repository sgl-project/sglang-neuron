#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's one-kernel decode layer (models/mimo_v2/kernels/layer.py), its two all-reduces inside,
over a world of NeuronCores: every rank's layer output must equal

    h   = x + all_reduce(attn_half(x))        the attention half kernel (checked by check_attn_half.py)
    out = h + all_reduce(moe_half(h))         the MoE half kernel (checked by check_moe_half.py)

with the all-reduces on the host (gloo) and the halves run as their own kernels: what is new in the layer kernel is
its in-kernel collectives, the fences around them and the residual adds. Each rank has its own seeded weights (a rank's
heads and experts differ), the same input rows, and its new token's K/V must land in its slots.

    torchrun --nproc-per-node 4 tools/mimo_v2/check_layer.py

Run on free NeuronCores (rank r on core --first-core + r)."""

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-core", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--dense", action="store_true", help="the dense layer kernel (layer 0) instead of the MoE one")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(args.first_core + rank)
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62183")
    os.environ["UNSAFE_FP8FNCAST"] = "1"
    cache = os.environ.get("CHECK_CACHE", "/cache/check_layer")
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
    from sglang_neuron.models.mimo_v2.experts_fp8 import FP8, requant_rows
    from sglang_neuron.models.mimo_v2.kernels.attn_half import attn_half
    from sglang_neuron.models.mimo_v2.kernels.layer import dense_layer, moe_layer
    from sglang_neuron.models.mimo_v2.kernels.moe_half import moe_half
    from sglang_neuron.models.mimo_v2.launch import FP8_FN_AS_E4M3
    from sglang_neuron.models.mimo_v2.rope import RotaryTables
    from sglang_neuron.ops.attention.attention_decode_mask import gen_attention_decode_mask, qk_swap_layout
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks
    from sglang_neuron.parallel.state import _make_group
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    init_neuron_ranks(rank, world, rank, world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _make_group([list(range(world))], rank, "mimo_layer_check")
    mesh = (tuple(range(world)),)
    prefix = "--internal-hlo2tensorizer-options="
    cc = [f"{a} {FP8_FN_AS_E4M3}" if a.startswith(prefix) else a for a in COMPILER_ARGS]
    if not any(a.startswith(prefix) for a in cc):
        cc.append(prefix + FP8_FN_AS_E4M3)

    D, VD, H, PAGE, BLOCKS, E, EA, I = 256, 128, 4096, 128, 8, 4, 256, 2048  # noqa: E741
    rows, eps, scale = args.rows, 1e-6, 192 ** -0.5
    shared = torch.Generator().manual_seed(1)  # the rows, positions and table: the same on every rank
    own = torch.Generator().manual_seed(100 + rank)  # this rank's weights and cache
    x = torch.randn(rows, H, generator=shared).to(torch.bfloat16)
    positions = torch.randint(1, BLOCKS * PAGE - 1, (rows,), generator=shared)
    table = torch.stack([1 + b * BLOCKS + torch.arange(BLOCKS) for b in range(rows)]).to(torch.int32)
    slots = (table.gather(1, (positions // PAGE).view(-1, 1)).view(-1).long() * PAGE + positions % PAGE).to(torch.int32)
    k_cache = torch.randn(rows * BLOCKS + 1, 1, PAGE, D, generator=own).to(torch.bfloat16)
    v_cache = torch.randn(rows * BLOCKS + 1, 1, PAGE, D, generator=own).to(torch.bfloat16)
    attn_gamma = (1 + 0.1 * torch.randn(1, H, generator=own)).float()
    w_qkv = (0.02 * torch.randn(H, 3 * D, generator=own)).to(torch.bfloat16)
    w_o = (0.02 * torch.randn(VD, H, generator=own)).to(torch.bfloat16)
    sink = torch.tensor([[1.0]])
    cos, sin = RotaryTables(1e4, 64)(positions)
    start = (positions - 127).clamp(min=0).reshape(1, rows).float()
    mask = qk_swap_layout(gen_attention_decode_mask(positions.reshape(1, rows).float(), bs=rows, q_head=1, s_active=1,
                                                    s_prior=BLOCKS * PAGE, block_len=PAGE, start_pos=start),
                          d_head=D, fp8_kv=False, fp8_packed=False)
    moe_gamma = (1 + 0.1 * torch.randn(1, H, generator=own)).to(torch.bfloat16)
    router_wT = (0.02 * torch.randn(H, EA, generator=shared)).to(torch.bfloat16)  # replicated, as served
    bias = (0.01 * torch.randn(1, EA, generator=shared)).float()
    ebase = torch.tensor([[float(rank * E)]])
    gu_w, gu_s, dn_w, dn_s = [], [], [], []
    for _ in range(E):
        q = [requant_rows(0.02 * torch.randn(*shape, generator=own)) for shape in ((I, H), (I, H), (H, I))]
        gu_w.append(torch.stack([q[0][0].float().t(), q[1][0].float().t()], dim=1))
        gu_s.append(torch.stack([q[0][1], q[1][1]]))
        dn_w.append(q[2][0].float().t())
        dn_s.append(q[2][1])
    moe_args = [moe_gamma, router_wT, bias, ebase, torch.stack(gu_w).to(FP8), torch.stack(dn_w).to(FP8),
                torch.stack(gu_s), torch.stack(dn_s)]
    device = torch.device("neuron", rank)
    d = lambda t: t.contiguous().to(device)  # noqa: E731
    kernel_kw = dict(eps=eps, softmax_scale=scale, v_head_dim=VD, route_scale=1.0, top_k=8)

    def compiled(fn, *inputs):
        failed = torch.zeros(1)
        with stack.compile_scope(cc):
            try:
                out = torch.compile(fn, backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)(*inputs)
            except Exception:  # noqa: BLE001 - reported once every rank has run
                import traceback

                traceback.print_exc()
                failed += 1
                out = None
        dist.all_reduce(failed)
        if int(failed):
            raise RuntimeError("a rank failed")
        torch._dynamo.reset()
        return out

    attn_inputs = [x.view(rows, 1, H), attn_gamma, w_qkv, cos.t().unsqueeze(-1).contiguous(),
                   sin.t().unsqueeze(-1).contiguous(), table, k_cache.clone(), v_cache.clone(), mask, sink,
                   slots.view(rows, 1), w_o]

    # the reference: the two checked half kernels, the all-reduces on the host
    a_part, k_ref, v_ref = compiled(lambda *t: wrap_nki(attn_half)[get_logical_nc_config()](
        *t, rmsnorm_X_eps=eps, softmax_scale=scale, v_head_dim=VD), *[d(t) for t in attn_inputs])
    a = a_part.cpu().float()
    dist.all_reduce(a)
    h = (x.float() + a.bfloat16().float()).bfloat16()
    if args.dense:
        # the dense MLP's torch form (models/mimo_v2/model.DenseMLP) after torch's RMSNorm
        dense_args = [moe_gamma, (0.02 * torch.randn(H, 256, generator=own)).to(torch.bfloat16),
                      (0.02 * torch.randn(H, 256, generator=own)).to(torch.bfloat16),
                      (0.02 * torch.randn(256, H, generator=own)).to(torch.bfloat16)]
        h32 = h.float()
        normed = (moe_gamma.float() * (h32 * torch.rsqrt(h32.pow(2).mean(-1, keepdim=True) + eps)).bfloat16().float()
                  ).bfloat16().float()
        g_, u_, dn_ = (t.float() for t in dense_args[1:])
        m = ((torch.nn.functional.silu(normed @ g_) * (normed @ u_)).bfloat16().float() @ dn_)
    else:
        m_part = compiled(lambda *t: wrap_nki(moe_half)[get_logical_nc_config()](*t, eps=eps, route_scale=1.0,
                                                                                  top_k=8),
                          d(h), *[d(t) for t in moe_args])
        m = m_part.cpu().float()
    dist.all_reduce(m)
    want = (h.float() + m.bfloat16().float()).bfloat16().float()

    if args.dense:
        layer_inputs = [d(t) for t in attn_inputs] + [d(t) for t in dense_args]
        dense_kw = dict(eps=eps, softmax_scale=scale, v_head_dim=VD)
        out = compiled(lambda *t: wrap_nki(dense_layer)[get_logical_nc_config()](*t, mesh, **dense_kw), *layer_inputs)
    else:
        layer_inputs = [d(t) for t in attn_inputs] + [d(t) for t in moe_args]
        out = compiled(lambda *t: wrap_nki(moe_layer)[get_logical_nc_config()](*t, mesh, **kernel_kw), *layer_inputs)
    # the caches the kernel wrote in place: its device inputs
    kc, vc = layer_inputs[6], layer_inputs[7]
    out = out.cpu().float()
    rel = ((out - want).norm() / want.norm()).item()
    rows_of = lambda c: c.cpu().view(-1, D)[slots.long()]  # noqa: E731 - one KV head: a slot is a row
    written = torch.equal(rows_of(kc), rows_of(k_ref)) and torch.equal(rows_of(vc), rows_of(v_ref))
    # the dense reference's norm and MLP are torch's, not a checked kernel: bf16 rounding apart
    ok = rel < (2e-2 if args.dense else 1e-2) and written
    kind = "dense" if args.dense else "MoE"
    print(f"{'PASS' if ok else 'FAIL'} rank {rank}/{world}, {rows} rows, {kind}: layer kernel vs the reference "
          f"relative L2 {rel:.2e}; K/V rows as the half kernel's {written}", flush=True)
    flag = torch.tensor([0.0 if ok else 1.0])
    dist.all_reduce(flag)
    return 1 if int(flag) else 0


if __name__ == "__main__":
    sys.exit(main())
