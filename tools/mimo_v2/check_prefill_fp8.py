#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's FP8 prefill attention (models/mimo_v2/attention.pack_prior + nkilib's segmented
attention over packed FP8 K, heads of 256) on one NeuronCore: a 128-token chunk after a prior of 300 tokens, over
token-major FP8 caches holding each value times a scale of 0.5, against the bf16 kernel over the dequantized caches.
A full layer (every column packed) and a sliding one (window 128, sinks, only the window's columns packed).

    NEURON_RT_VISIBLE_CORES=0 python3 tools/mimo_v2/check_prefill_fp8.py

Run on a free NeuronCore."""

import os
import sys

D, PAGE, PRIOR, CHUNK, BLOCKS = 256, 128, 300, 128, 8
SCALE = 0.5


def main() -> int:
    # the FP8 cache compiles float8_e4m3fn as Trn2's e4m3, as a serving launch sets (models/mimo_v2/launch.py)
    os.environ["UNSAFE_FP8FNCAST"] = "1"
    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch

    from sglang_neuron.backend import stack
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.mimo_v2.attention import pack_prior
    from sglang_neuron.models.mimo_v2.launch import FP8_FN_AS_E4M3
    from sglang_neuron.ops.attention.attention_segmented_cte import segmented_attention

    prefix = "--internal-hlo2tensorizer-options="
    args_cc = [f"{a} {FP8_FN_AS_E4M3}" if a.startswith(prefix) else a for a in COMPILER_ARGS]
    if not any(a.startswith(prefix) for a in args_cc):
        args_cc.append(prefix + FP8_FN_AS_E4M3)
    g = torch.Generator().manual_seed(0)
    pages = BLOCKS + 3
    k = torch.randn(pages, 1, PAGE, D, generator=g)
    v = torch.randn(pages, 1, PAGE, D, generator=g)
    k8 = (k * SCALE).clamp(-240, 240).to(torch.float8_e4m3fn)
    v8 = (v * SCALE).clamp(-240, 240).to(torch.float8_e4m3fn)
    k_deq, v_deq = (k8.float() / SCALE).to(torch.bfloat16), (v8.float() / SCALE).to(torch.bfloat16)
    # the request's pages, out of order, then unallocated columns
    used = -(-(PRIOR + CHUNK) // PAGE)
    table = torch.full((1, BLOCKS), -1, dtype=torch.int32)
    table[0, :used] = torch.tensor([5, 2, 9, 7][:used], dtype=torch.int32)
    q = torch.randn(1, CHUNK, D, generator=g).to(torch.bfloat16)
    cached = torch.tensor([[PRIOR]], dtype=torch.int32)
    scale = 192 ** -0.5
    device = torch.device("neuron", 0)
    failures = 0
    for window, sink in ((None, None), (128, torch.tensor([[1.5]]))):
        def bf16(q_, kc, vc, tbl, pr, sk):
            return segmented_attention(q_, k_cache=kc, v_cache=vc, block_tables=tbl, prior_tokens=pr, block_size=PAGE,
                                       kv_segment_size=1024, scale=scale, sliding_window=window, sink=sk)

        def fp8(q_, kc, vc, tbl, pr, sk):
            extra = {}
            if window:
                extra = dict(first_col=torch.clamp(pr.to(torch.int64).reshape(-1) - window + 1, min=0) // PAGE,
                             n_cols=(CHUNK + window - 1) // PAGE + 2)
            kp, vp, t2 = pack_prior(kc, vc, tbl, **extra)
            out = segmented_attention(q_, k_cache=kp, v_cache=vp, block_tables=t2, prior_tokens=pr, block_size=PAGE,
                                      kv_segment_size=1024, scale=scale / SCALE, sliding_window=window, sink=sk)
            return out / SCALE

        outs = []
        for fn, kc, vc in ((bf16, k_deq, v_deq), (fp8, k8, v8)):
            dev = [t.to(device) if t is not None else None for t in (q, kc, vc, table, cached, sink)]
            with stack.compile_scope(args_cc):
                compiled = torch.compile(fn, backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
                outs.append(compiled(*dev).cpu().float())
            torch._dynamo.reset()
        rel = ((outs[1] - outs[0]).norm() / outs[0].norm()).item()
        ok = rel < 1e-2
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} window {window} sink {sink is not None}: FP8 packed vs bf16 dequantized "
              f"relative L2 {rel:.2e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
