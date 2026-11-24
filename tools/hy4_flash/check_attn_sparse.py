#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check of Hy4-Flash's decode attention kernel of a sparse layer (models/hy_v4_flash/kernels/attn_swa.attn_sparse)
against the graph's own pieces, on a random KV pool and a random IndexPool choice (511 complete groups before each
row's own and its own group, expanded to their tokens) at TP32's per-rank shapes (one head).

The reference writes each row's MLA row into the pool and reads the chosen tokens back (Attention.decode, sparse); the
kernel reads them before the write and takes the row's own key from its MLA row. A control with the sinks raised must
fail.

    python3 tools/hy4_flash/check_attn_sparse.py --simulate [--grid 2] [--fp8]
    NEURON_RT_VISIBLE_CORES=0 python3 tools/hy4_flash/check_attn_sparse.py"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import check_attn_swa as S  # noqa: E402

H, QL, NOPE, ROPE, V, LAT, KD, HALF, PAGE = S.H, S.QL, S.NOPE, S.ROPE, S.V, S.LAT, S.KD, S.HALF, S.PAGE  # noqa: E741
HEADS, GROUP, BUDGET, THETA, EPS = 1, 4, 511, 1.0e7, S.EPS
SCALE = (NOPE + ROPE) ** -0.5
K_PAD = 17 * 128
GRID = 1


def make_inputs(rows: int, fp8: bool, seed: int = 0):
    import torch

    g = torch.Generator().manual_seed(seed)
    bf = torch.bfloat16
    w = {"xn": (torch.randn(rows, H, generator=g)).to(bf), "q_resid": (torch.randn(rows, QL, generator=g)).to(bf),
         "q_b": (0.05 * torch.randn(QL, HEADS * (NOPE + ROPE), generator=g)).to(bf),
         "kv_a": (0.02 * torch.randn(H, KD, generator=g)).to(bf),
         "kv_ln": (1 + 0.1 * torch.randn(LAT, generator=g)).to(bf),
         "w_uk": (0.1 * torch.randn(HEADS, NOPE, LAT, generator=g)).to(bf),
         "w_uv": (0.05 * torch.randn(HEADS, V, LAT, generator=g)).to(bf),
         "gate": (0.02 * torch.randn(H, HEADS * V, generator=g)).to(bf),
         "o": (0.02 * torch.randn(HEADS * V, H, generator=g)).to(bf),
         "sinks": torch.randn(HEADS, generator=g)}
    blocks = 72  # a 9216-token table
    pages = rows * blocks + 8
    cache = (torch.randn(pages * PAGE, KD, generator=g) * 0.7).clamp(-6, 6)
    w["cache"] = cache.to(torch.float8_e4m3fn if fp8 else bf)
    perm = torch.randperm(pages - 1, generator=g) + 1
    pos, slots, keypos = [], [], []
    for r in range(rows):
        p = int(torch.randint(BUDGET * GROUP + 40, blocks * PAGE - 8, (1,), generator=g))
        own = p // GROUP
        chosen = torch.randperm(own, generator=g)[:BUDGET].sort().values
        groups = torch.cat([chosen, torch.tensor([own])])
        tokens = (groups.unsqueeze(1) * GROUP + torch.arange(GROUP)).flatten()
        table = perm[r * blocks:(r + 1) * blocks]
        sl = table[tokens // PAGE] * PAGE + tokens % PAGE
        kp = torch.where(tokens <= p, tokens, torch.full_like(tokens, -1)).float()
        pad = K_PAD - tokens.numel()
        slots.append(torch.cat([sl, torch.zeros(pad, dtype=sl.dtype)]))
        keypos.append(torch.cat([kp, torch.full((pad,), -1.0)]))
        pos.append(p)
        w.setdefault("own_slot", []).append(int(table[p // PAGE] * PAGE + p % PAGE))
    w["pos"] = torch.tensor(pos)
    w["slots"] = torch.stack(slots).to(torch.int32)
    w["keypos"] = torch.stack(keypos)
    return w


def tables(pos):
    from sglang_neuron.models.hy_v4_flash import ops

    cos, sin = ops.rope_tables(pos, ops.inv_freq(THETA, ROPE))
    return cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1), cos, sin


def reference(w, sinks=None):
    """(partial [T, H] fp32, rows [T, 576]) by Attention.decode's formulas (sparse) over the given choice."""
    import torch

    from sglang_neuron.models.hy_v4_flash import ops

    bf = torch.bfloat16
    rows_n = w["xn"].shape[0]
    x, q_resid = w["xn"], w["q_resid"]
    _, _, cos, sin = tables(w["pos"])
    q = (q_resid @ w["q_b"]).view(rows_n, HEADS, NOPE + ROPE)
    q_nope, q_rope = q[..., :NOPE], ops.rotate(q[..., NOPE:], cos.unsqueeze(1), sin.unsqueeze(1))
    ckv = x @ w["kv_a"]
    compact = torch.cat([ops.rms_norm(ckv[:, :LAT], w["kv_ln"], EPS), ops.rotate(ckv[:, LAT:], cos, sin)], dim=-1)
    q_latent = torch.einsum("rhd,hdl->rhl", q_nope, w["w_uk"])
    gate = torch.sigmoid((x @ w["gate"]).view(rows_n, HEADS, V).float())
    cache = w["cache"].clone()
    for r in range(rows_n):
        cache[w["own_slot"][r]] = compact[r].to(cache.dtype)
    keys = cache[w["slots"].long()].to(bf)
    allowed = w["keypos"] >= 0
    att = ops.mla_attend(q_latent, q_rope, keys, allowed, w["sinks"] if sinks is None else sinks, SCALE, LAT)
    values = torch.einsum("rhl,hdl->rhd", att.to(bf), w["w_uv"])
    values = (values.float() * gate).to(bf)
    return (values.reshape(rows_n, -1) @ w["o"]).float(), compact.float()


def kernel_args(w):
    cosf, sinf, _, _ = tables(w["pos"])
    cache = w["cache"]
    return [w["xn"], w["q_resid"], w["q_b"], w["kv_a"], w["kv_ln"].view(1, -1), w["w_uk"],
            w["w_uv"].transpose(1, 2).contiguous(), w["gate"], w["o"], w["sinks"].view(1, -1), cosf.contiguous(),
            sinf.contiguous(), S.pm_matrix(), cache[:, :HALF].contiguous(), cache[:, HALF:].contiguous(), w["slots"],
            w["keypos"].contiguous(), w["pos"].float().view(-1, 1)]


KWARGS = dict(eps=EPS, scale=SCALE)


def run(args_t, simulate: bool):
    from sglang_neuron.models.hy_v4_flash.kernels.attn_swa import attn_sparse

    if simulate:
        import ml_dtypes
        import nki
        import numpy as np
        import torch

        def np_of(t):
            if t.dtype == torch.bfloat16:
                return t.float().numpy().astype(ml_dtypes.bfloat16)
            if t.dtype == torch.float8_e4m3fn:
                return t.float().numpy().astype(ml_dtypes.float8_e4m3)
            return t.numpy()

        kernel = attn_sparse[GRID] if GRID > 1 else attn_sparse
        outs = nki.simulate(kernel)(*[np_of(t) for t in args_t], **KWARGS)
        return [torch.from_numpy(np.asarray(o).astype(np.float32)) for o in outs]
    return S.run_device(args_t, attn_sparse, KWARGS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--grid", type=int, default=1)
    parser.add_argument("--fp8", action="store_true")
    args = parser.parse_args()
    global GRID
    GRID = args.grid
    failures = []
    for rows in args.rows:
        w = make_inputs(rows, args.fp8)
        want, want_rows = reference(w)
        control, _ = reference(w, sinks=w["sinks"] + 20.0)
        got, got_rows = run(kernel_args(w), args.simulate)

        def rel(a, b):
            return ((a - b).norm() / b.norm().clamp(min=1e-30)).item()

        r, rc, rr = rel(got, want), rel(got, control), rel(got_rows, want_rows)
        ok = r < 2e-2 and rr < 1e-2 and rc > 3 * r
        print(f"{'PASS' if ok else 'FAIL'} rows {rows}: partial rel L2 {r:.2e} (sinks +20 control {rc:.2e}), "
              f"MLA rows {rr:.2e}", flush=True)
        if not ok:
            failures.append(rows)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
