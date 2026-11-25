#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check of Hy4-Flash's decode attention half kernel of a sliding layer (models/hy_v4_flash/kernels/attn_swa.py)
against the graph's own pieces (ops.hc_mix, ops.rms_norm, ops.rope_tables / ops.rotate, ops.mla_attend, the
Attention module's projections), on a random KV pool of the window's pages at TP32's per-rank shapes.

The reference writes each row's MLA row into the pool and reads its window back, as Attention.decode does; the kernel
reads the pool before the write and takes the row's own key from its MLA row. A control with the sinks dropped must
fail.

    python3 tools/hy4_flash/check_attn_swa.py --simulate [--grid 2] [--fp8]
    NEURON_RT_VISIBLE_CORES=0 python3 tools/hy4_flash/check_attn_swa.py"""

import argparse
import os
import sys

H, HC, QL, HEADS, NOPE, ROPE, V, LAT = 3072, 4, 1024, 2, 192, 64, 256, 512  # noqa: E741
KD, HALF, PAGE, NB, WINDOW = 576, 288, 128, 5, 512
EPS, HC_EPS, MAG, THETA = 1e-5, 1e-6, 2.0, 1.0e4
SCALE = (NOPE + ROPE) ** -0.5
GRID = 1


def make_inputs(rows: int, fp8: bool, seed: int = 0):
    import torch

    g = torch.Generator().manual_seed(seed)
    bf = torch.bfloat16
    w = {"streams": (torch.randn(rows, HC, H, generator=g) * 0.5).to(bf),
         "fn": 0.01 * torch.randn(2 * HC, HC * H, generator=g), "base": 0.1 * torch.randn(2 * HC, generator=g),
         "scale": torch.tensor([0.5, 0.7]), "ln_w": (1 + 0.1 * torch.randn(H, generator=g)).to(bf),
         "q_a": (0.02 * torch.randn(H, QL, generator=g)).to(bf),
         "qa_ln": (1 + 0.1 * torch.randn(QL, generator=g)).to(bf),
         "q_b": (0.05 * torch.randn(QL, HEADS * (NOPE + ROPE), generator=g)).to(bf),
         "kv_a": (0.02 * torch.randn(H, KD, generator=g)).to(bf),
         "kv_ln": (1 + 0.1 * torch.randn(LAT, generator=g)).to(bf),
         "w_uk": (0.1 * torch.randn(HEADS, NOPE, LAT, generator=g)).to(bf),
         "w_uv": (0.05 * torch.randn(HEADS, V, LAT, generator=g)).to(bf),
         "gate": (0.02 * torch.randn(H, HEADS * V, generator=g)).to(bf),
         "o": (0.02 * torch.randn(HEADS * V, H, generator=g)).to(bf),
         "sinks": torch.randn(HEADS, generator=g)}
    pages = rows * NB + 8
    dt = torch.float8_e4m3fn if fp8 else bf
    cache = (torch.randn(pages * PAGE, KD, generator=g) * 0.7).clamp(-6, 6)
    w["cache"] = cache.to(dt)
    pos, first, table = [], [], []
    perm = torch.randperm(pages - 1, generator=g) + 1
    for r in range(rows):
        p = int(torch.randint(WINDOW + 7, 6000, (1,), generator=g))
        f = (p - WINDOW + 1) // PAGE * PAGE
        pos.append(p)
        first.append(f)
        table.append(perm[r * NB:(r + 1) * NB])
    w["pos"] = torch.tensor(pos)
    w["first"] = torch.tensor(first)
    w["table"] = torch.stack(table).to(torch.int32)
    return w


def pm_matrix():
    import torch

    m = torch.zeros(ROPE, ROPE)
    for j in range(ROPE // 2):
        m[2 * j + 1, 2 * j] = -1.0
        m[2 * j, 2 * j + 1] = 1.0
    return m.to(torch.bfloat16)


def tables(pos):
    import torch

    from sglang_neuron.models.hy_v4_flash import ops

    cos, sin = ops.rope_tables(pos, ops.inv_freq(THETA, ROPE))
    return cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1), cos, sin


def reference(w, sinks=None):
    """(partial [T, H] fp32, post [T, hc], rows [T, 576]) by Attention.decode's formulas (sliding)."""
    import torch

    from sglang_neuron.models.hy_v4_flash import ops

    bf = torch.bfloat16
    rows_n = w["streams"].shape[0]
    post, x = ops.hc_mix(w["streams"], w["fn"], w["base"], w["scale"], HC, MAG, HC_EPS, EPS)
    x = ops.rms_norm(x, w["ln_w"], EPS)
    _, _, cos, sin = tables(w["pos"])
    q_resid = ops.rms_norm(x @ w["q_a"], w["qa_ln"], EPS)
    q = (q_resid @ w["q_b"]).view(rows_n, HEADS, NOPE + ROPE)
    q_nope, q_rope = q[..., :NOPE], ops.rotate(q[..., NOPE:], cos.unsqueeze(1), sin.unsqueeze(1))
    ckv = x @ w["kv_a"]
    compact = torch.cat([ops.rms_norm(ckv[:, :LAT], w["kv_ln"], EPS), ops.rotate(ckv[:, LAT:], cos, sin)], dim=-1)
    q_latent = torch.einsum("rhd,hdl->rhl", q_nope, w["w_uk"])
    gate = torch.sigmoid((x @ w["gate"]).view(rows_n, HEADS, V).float())
    cache = w["cache"].clone()
    keys, allowed = [], []
    for r in range(rows_n):
        p, f = int(w["pos"][r]), int(w["first"][r])
        slot = int(w["table"][r, (p - f) // PAGE]) * PAGE + (p - f) % PAGE
        cache[slot] = compact[r].to(cache.dtype)
        slots = (w["table"][r].long().unsqueeze(1) * PAGE + torch.arange(PAGE)).flatten()
        keys.append(cache[slots].to(bf))
        kp = f + torch.arange(NB * PAGE)
        allowed.append((kp <= p) & (kp > p - WINDOW))
    keys, allowed = torch.stack(keys), torch.stack(allowed)
    att = ops.mla_attend(q_latent, q_rope, keys, allowed, w["sinks"] if sinks is None else sinks, SCALE, LAT)
    values = torch.einsum("rhl,hdl->rhd", att.to(bf), w["w_uv"])
    values = (values.float() * gate).to(bf)
    return (values.reshape(rows_n, -1) @ w["o"]).float(), post, compact.float()


def window_keys(w):
    """[T, NB x 128, 576] each row's window pages of the pool before the rows' own writes (the graph's gather)."""
    import torch

    slots = (w["table"].long().unsqueeze(-1) * PAGE + torch.arange(PAGE)).flatten(1)
    return w["cache"][slots].contiguous()


def kernel_args(w):
    import torch

    fnT = w["fn"].t().contiguous()
    hi = fnT.to(torch.bfloat16)
    lo = (fnT - hi.float()).to(torch.bfloat16)
    cosf, sinf, _, _ = tables(w["pos"])
    cache = w["cache"]
    return [w["streams"], hi, lo, w["base"].view(1, -1), w["scale"].view(1, -1), w["ln_w"].view(1, -1), w["q_a"],
            w["qa_ln"].view(1, -1), w["q_b"], w["kv_a"], w["kv_ln"].view(1, -1), w["w_uk"],
            w["w_uv"].transpose(1, 2).contiguous(), w["gate"], w["o"], w["sinks"].view(1, -1), cosf.contiguous(),
            sinf.contiguous(), pm_matrix(), window_keys(w), w["first"].float().view(-1, 1),
            w["pos"].float().view(-1, 1)]


KWARGS = dict(eps=EPS, hc_eps=HC_EPS, hc_mag=MAG, scale=SCALE, window=WINDOW)


def run_simulate(args_t):
    import ml_dtypes
    import nki
    import numpy as np
    import torch

    from sglang_neuron.models.hy_v4_flash.kernels.attn_swa import attn_swa

    def np_of(t):
        if t.dtype == torch.bfloat16:
            return t.float().numpy().astype(ml_dtypes.bfloat16)
        if t.dtype == torch.float8_e4m3fn:
            return t.float().numpy().astype(ml_dtypes.float8_e4m3)
        return t.numpy()

    kernel = attn_swa[GRID] if GRID > 1 else attn_swa
    outs = nki.simulate(kernel)(*[np_of(t) for t in args_t], **KWARGS)
    return [torch.from_numpy(np.asarray(o).astype(np.float32)) for o in outs]


def run_device(args_t, kernel=None, kwargs=None):
    os.environ["UNSAFE_FP8FNCAST"] = "1"
    from sglang_neuron.process_init import init_stack

    init_stack()
    import time

    import torch

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.hy_v4_flash.kernels.attn_swa import attn_swa
    from sglang_neuron.models.hy_v4_flash.launch import FP8_FN_AS_E4M3
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    prefix = "--internal-hlo2tensorizer-options="
    args_cc = [f"{a} {FP8_FN_AS_E4M3}" if a.startswith(prefix) else a for a in COMPILER_ARGS]
    device = torch.device("neuron", 0)
    dev = [t.contiguous().to(device) for t in args_t]

    kernel = attn_swa if kernel is None else kernel
    kwargs = KWARGS if kwargs is None else kwargs

    def fn(*ts):
        return wrap_nki(kernel)[get_logical_nc_config()](*ts, **kwargs)

    with stack.compile_scope(args_cc):
        compiled = torch.compile(fn, backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
        outs = [o.cpu().float() for o in compiled(*dev)]
        for _ in range(3):
            o = compiled(*dev)
        o[0].cpu()
        n = 20
        t0 = time.time()
        for _ in range(n):
            o = compiled(*dev)
        o[0].cpu()
        print(f"  device: {(time.time() - t0) / n * 1e3:.3f} ms a call", flush=True)
    torch._dynamo.reset()
    return outs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--grid", type=int, default=1)
    parser.add_argument("--fp8", action="store_true", help="an FP8 KV pool")
    args = parser.parse_args()
    global GRID
    GRID = args.grid
    failures = []
    for rows in args.rows:
        w = make_inputs(rows, args.fp8)
        want, want_post, want_rows = reference(w)
        control, _, _ = reference(w, sinks=w["sinks"] + 8.0)
        got, got_post, got_rows = (run_simulate if args.simulate else run_device)(kernel_args(w))

        def rel(a, b):
            return ((a - b).norm() / b.norm().clamp(min=1e-30)).item()

        r, rc, rp, rr = rel(got, want), rel(got, control), rel(got_post, want_post), rel(got_rows, want_rows)
        ok = r < 2e-2 and rp < 1e-3 and rr < 1e-2 and rc > 3 * r
        print(f"{'PASS' if ok else 'FAIL'} rows {rows}: partial rel L2 {r:.2e} (sinks +8 control {rc:.2e}), "
              f"post {rp:.2e}, MLA rows {rr:.2e}", flush=True)
        if not ok:
            failures.append(rows)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
