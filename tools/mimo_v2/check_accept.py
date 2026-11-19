#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's MTP verify / rotate / commit kernels (models/mimo_v2/kernels/accept.py) on one
NeuronCore against mtp_step.accept, mtp.rotate and the carry's index_copy_. Every output must match exactly.

    NEURON_RT_VISIBLE_CORES=0 python3 tools/mimo_v2/check_accept.py"""

import sys


def main() -> int:
    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.mimo_v2 import mtp, mtp_step
    from sglang_neuron.models.mimo_v2.kernels.accept import accept, commit, rotate
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    g = torch.Generator().manual_seed(9)
    W, S, N, OW = 8, mtp_step.SPEC, 9, 16
    tokens = torch.randint(0, 50, (W, S), generator=g)
    sampled = tokens.clone()
    sampled[:, :-1] = tokens[:, 1:]  # accept everything ...
    sampled[2, 1] = 999  # ... but row 2 stops after one draft, row 5 after none
    sampled[5, 0] = 998
    valid = torch.ones(W, S, dtype=torch.bool)
    valid[3, 1:] = False
    fresh = torch.zeros(W, dtype=torch.bool)
    fresh[4] = True
    live = torch.ones(W, dtype=torch.bool)
    live[7] = False
    span = {"valid": valid, "fresh": fresh, "live": live}
    want_out, want_kept = mtp_step.accept(tokens, sampled, span)
    want_bonus = sampled.gather(1, want_kept.view(-1, 1))
    draft = torch.randint(0, 50, (W, 1), generator=g)
    want_rot = mtp.rotate(tokens, want_kept, draft.view(-1))
    carry = torch.randint(0, 50, (N, S), generator=g).to(torch.int32)
    length = torch.randint(0, 99, (N,), generator=g).to(torch.int32)
    row_slots = torch.tensor([3, 0, 5, 1, 7, 2, 4, 8], dtype=torch.int32)
    base = torch.randint(0, 99, (W,), generator=g)
    drafts = torch.randint(0, 50, (W, S - 1), generator=g)
    want_carry, want_len = carry.clone(), length.clone()
    want_carry.index_copy_(0, row_slots.long(), torch.cat([want_bonus, drafts], 1).to(torch.int32))
    want_len.index_copy_(0, row_slots.long(), (base + want_kept + 1).to(torch.int32))

    col = lambda t: t.reshape(-1, 1).to(torch.int32)  # noqa: E731
    per_tok = lambda t: t.view(W, 1).expand(W, S)  # noqa: E731
    dev = torch.device("neuron", 0)
    d = lambda t: t.contiguous().to(dev)  # noqa: E731
    grid = get_logical_nc_config()
    with stack.compile_scope(list(COMPILER_ARGS)):
        def fn(t, s, v, f, l, dr, cr, ln, rs, b, ds):
            out, kept, bonus, krow = wrap_nki(accept)[grid](t, s, v, f, l, span=S, out_width=OW)
            rot = wrap_nki(rotate)[grid](t, kept, dr)
            res = wrap_nki(commit)[grid](cr, ln, rs, bonus, ds[:, 0:1], ds[:, 1:2], ds[:, 2:3], b, kept, out)
            return out, kept, bonus, krow, rot, res, cr, ln
        outs = torch.compile(fn, backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)(
            d(col(tokens)), d(col(sampled)), d(col(valid)), d(col(per_tok(fresh))), d(col(per_tok(live))),
            d(col(draft)), d(carry), d(length), d(row_slots), d(col(base)), d(drafts.to(torch.int32)))
    out, kept, bonus, krow, rot, res, cr, ln = (o.cpu() for o in outs)
    pad = torch.full((OW - W, S), -1, dtype=torch.int64)
    checks = {
        "out": torch.equal(out.long(), torch.cat([want_out, pad])),
        "kept": torch.equal(kept.view(-1).long(), want_kept),
        "bonus": torch.equal(bonus.view(-1).long(), want_bonus.view(-1)),
        "kept_row": torch.equal(krow.view(-1).long(), torch.arange(W) * S + want_kept),
        "rotate": torch.equal(rot.view(W, S).long(), want_rot.long()),
        "result": torch.equal(res.long(), out.long()),
        "carry": torch.equal(cr, want_carry),
        "length": torch.equal(ln, want_len),
    }
    for k, v in checks.items():
        print(f"  {k}: {'exact' if v else 'DIFF'}")
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
