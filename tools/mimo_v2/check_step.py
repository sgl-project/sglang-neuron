#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's MTP step-input kernel (models/mimo_v2/kernels/step.py) on one NeuronCore against
the torch it replaces: mtp_step.span_inputs, attention.span_slots, the layers' position masks' inputs and
rope.RotaryTables. Integers must match exactly; the RoPE tables to bf16 rounding.

    NEURON_RT_VISIBLE_CORES=0 python3 tools/mimo_v2/check_step.py"""

import sys


def main() -> int:
    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.mimo_v2 import mtp_step
    from sglang_neuron.models.mimo_v2.attention import span_slots
    from sglang_neuron.models.mimo_v2.kernels.step import step_inputs
    from sglang_neuron.models.mimo_v2.rope import RotaryTables
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    g = torch.Generator().manual_seed(5)
    W, S, N, page, Bf, Bs, window = 8, mtp_step.SPEC, 9, 128, 64, 6, 128
    carry = torch.randint(0, 150000, (N, S), generator=g).to(torch.int32)
    length = torch.randint(0, 8000, (N,), generator=g).to(torch.int32)
    ctrl = torch.zeros(W, mtp_step.CTRL_COLUMNS, dtype=torch.int32)
    ctrl[:6, mtp_step.CTRL_LIVE] = 1
    ctrl[1, mtp_step.CTRL_FRESH] = 1
    ctrl[1, mtp_step.CTRL_BASE] = 777
    ctrl[1, mtp_step.CTRL_TOKEN] = 4242
    row_slots = torch.tensor([3, 0, 5, 1, 7, 2, N - 1, N - 1], dtype=torch.int32)
    full_table = torch.randint(1, 20000, (W, Bf), generator=g).to(torch.int32)
    full_table[:, 50:] = -1
    swa_table = torch.randint(1, 3000, (W, Bs), generator=g).to(torch.int32)
    swa_off = (torch.clamp(length[row_slots.long()] - 256, min=0) // page * page).to(torch.int32)
    full_rope, swa_rope = RotaryTables(5e6, 64), RotaryTables(1e4, 64)

    span = mtp_step.span_inputs(carry.index_select(0, row_slots.long()), length.index_select(0, row_slots.long()),
                                ctrl)
    positions = span["positions"]
    md_full = {"block_table_tensor": full_table, "block_size": page}
    md_swa = {"block_table_tensor": swa_table, "block_size": page, "swa_kv_pos_offset": swa_off}
    want = {
        "tokens": span["tokens"].reshape(-1), "positions": positions.reshape(-1),
        "full_slots": span_slots(md_full, positions, span["valid"], window=None),
        "swa_slots": span_slots(md_swa, positions, span["valid"], window=window),
        "valid": span["valid"].reshape(-1).int(), "base": span["base"].reshape(-1, 1).expand(W, S).reshape(-1),
    }
    prior = span["base"].reshape(W, 1).expand(W, S).to(torch.int64)
    offset = swa_off.to(torch.int64).reshape(W, 1)
    end = (prior - offset).clamp(min=0)
    start = torch.minimum((positions.to(torch.int64) - offset - window + 1).clamp(min=0), end)
    fc, fs = full_rope(positions.reshape(-1))
    sc, ss = swa_rope(positions.reshape(-1))

    with stack.compile_scope(list(COMPILER_ARGS)):
        fn = torch.compile(lambda *a: wrap_nki(step_inputs)[get_logical_nc_config()](*a, window=window, mask_heads=(4, 8)), backend=stack.compile_backend_name(),
                           fullgraph=True, dynamic=False)
        dev = torch.device("neuron", 0)
        outs = fn(*(t.contiguous().to(dev) for t in (carry, length, ctrl, row_slots, full_table, swa_table, swa_off,
                                                      full_rope.inv_freq.view(1, -1), swa_rope.inv_freq.view(1, -1))))
    outs = [o.cpu() for o in outs]
    names = ["tokens", "positions", "full_slots", "swa_slots", "valid", "fresh", "live", "base"]
    ok = True
    for name, got in zip(names, outs[:8]):
        if name in want:
            same = torch.equal(got.reshape(-1).long(), want[name].reshape(-1).long())
            ok &= same
            print(f"  {name}: {'exact' if same else 'DIFF'}", flush=True)
    for name, got, ref in (("full pos", outs[8], prior), ("swa pos", outs[9], end), ("swa start", outs[10], start)):
        same = torch.equal(got.reshape(-1).long(), ref.reshape(-1).long())
        ok &= same
        print(f"  {name}: {'exact' if same else 'DIFF'}", flush=True)
    for name, got, ref in (("full cos", outs[11], fc), ("full sin", outs[12], fs), ("swa cos", outs[13], sc),
                           ("swa sin", outs[14], ss)):
        diff = (got.float().t() - ref.float()).abs().max().item()
        ok &= diff < 2e-2
        print(f"  {name}: max diff {diff:.2e}", flush=True)
    S = carry.shape[1]
    for h, got in zip((4, 8), outs[15:17]):
        k = torch.arange(S).reshape(S, 1, 1, 1)
        q = torch.arange(S).reshape(1, 1, 1, S)
        ref = (k <= q).expand(S, W, h, S).to(torch.uint8).reshape(S, W * h, S)
        same = tuple(got.shape) == tuple(ref.shape) and torch.equal(got, ref)
        ok &= same
        print(f"  active mask (heads {h}): {'exact' if same else 'DIFF'}", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
