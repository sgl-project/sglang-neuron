#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's prefill routing and blockwise mapping kernel (models/mimo_v2/kernels/prefill.py
prefill_route) on one NeuronCore against the torch it replaces: moe.route, the local experts' columns rounded through
bf16, and ops/moe/moe_blockwise.build_blockwise_mapping's torch mapping. The mapping must match exactly; the
affinities to bf16 rounding (a router score at a top-k tie can pick a different expert: those tokens are reported).

    NEURON_RT_VISIBLE_CORES=0 python3 tools/mimo_v2/check_prefill_route.py"""

import sys


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--padding", type=int, default=77)
    args = parser.parse_args()
    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.mimo_v2.experts_cte import BLOCK_SIZE
    from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_route
    from sglang_neuron.models.mimo_v2.moe import route
    from sglang_neuron.ops.moe.moe_blockwise import _mapping_in_torch, _num_blocks
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    g = torch.Generator().manual_seed(11)
    T, H, EA, E, K, ep_rank = args.tokens, 4096, 256, 4, 8, 5
    n_real = T - args.padding  # the chunk's padding: its last real position repeated
    hn = torch.randn(T, H, generator=g).to(torch.bfloat16)
    w = (torch.randn(EA, H, generator=g) * H ** -0.5).to(torch.bfloat16)
    bias = torch.randn(EA, generator=g) * 0.05
    # this rank's experts popular enough to fill several blocks
    bias[ep_rank * E:(ep_rank + 1) * E] += 0.3
    positions = torch.arange(T, dtype=torch.int32) + 1000
    positions[n_real:] = positions[n_real - 1]
    scaling = 2.5

    affinities, _ = route(hn, w, bias, K, True, scaling)
    local = affinities.to(torch.bfloat16).float()[:, ep_rank * E:(ep_rank + 1) * E]
    real = torch.arange(T) < n_real
    selected = ((local != 0) & real.unsqueeze(1)).float()
    blocks = _num_blocks(T, K, E, BLOCK_SIZE)
    tok_ref, b2e_ref = _mapping_in_torch(selected, E, E, False, blocks, BLOCK_SIZE, T, None)
    b2e_ref = torch.clamp(b2e_ref, max=E - 1)
    cond_ref = torch.cat([torch.any(tok_ref.view(blocks, BLOCK_SIZE) != -1, dim=1).int(), torch.zeros(2).int()])

    with stack.compile_scope(list(COMPILER_ARGS)):
        fn = torch.compile(lambda *a: wrap_nki(prefill_route)[get_logical_nc_config()](
            *a, top_k=K, route_scale=scaling, n_local=E, block_size=BLOCK_SIZE, num_blocks=blocks),
            backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
        dev = torch.device("neuron", 0)
        outs = fn(hn.to(dev), w.t().contiguous().to(dev), bias.view(1, -1).to(dev),
                  torch.full((1, 1), ep_rank, dtype=torch.int32).to(dev), positions.view(-1, 1).to(dev))
    aff, tok, b2e, cond = (o.cpu() for o in outs)
    aff = aff.float().view(T, E)
    picked_k, picked_r = aff != 0, local != 0
    ties = (picked_k != picked_r).any(dim=1)
    da = (aff[~ties] - local[~ties]).abs().max().item()
    print(f"  affinities: max diff {da:.3e} on {int((~ties).sum())} tokens; {int(ties.sum())} tokens pick another "
          f"local expert set (a top-k tie)", flush=True)
    ok = da < 2e-2 and int(ties.sum()) <= 2
    if int(ties.sum()) == 0:
        for name, got, ref in (("tokens", tok.long(), tok_ref.long()), ("block_to_expert", b2e.view(-1).long(),
                                                                       b2e_ref.view(-1).long()),
                               ("conditions", cond.view(-1).long(), cond_ref.long())):
            same = torch.equal(got, ref)
            ok &= same
            print(f"  {name}: {'exact' if same else 'DIFF'}", flush=True)
            if not same:
                bad = (got != ref).nonzero().view(-1)[:8].tolist()
                print(f"    first differences at {bad}: got {got[bad].tolist()} want {ref[bad].tolist()}")
    print(f"  {blocks} blocks, {int(cond_ref.sum())} used; tokens per expert {selected.sum(0).int().tolist()}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
