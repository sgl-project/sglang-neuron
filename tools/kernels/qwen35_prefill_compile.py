# SPDX-License-Identifier: Apache-2.0
"""Compile the Qwen3.5 giga prefill graph family (models/qwen35/graph.py Qwen35PrefillGraph) the way the serving
runtime warms it (runtime.Qwen35PrefillRuntime._warm_shapes on the paged pool): one ``torch.compile`` graph per
(bucket, prior tier) program of ``prefix.prefill_buckets(--chunk, --slot)`` x ``prefix.prior_tiers(--context,
--slot)``, compiled under torch-neuronx's compile-only mode with ``sglang_model.prefill_compiler_args`` (neuronx-cc's
modular flow: the embed and head partitions plus one ``SEGMENT``-layer segment partition reused by every segment) and
``NKI_COMPILE_MODE`` = ``runtime.PREFILL_NKI_COMPILE_MODE``. No collective runs: one JSON line per program with its
trace + compile wall time and the NEFF cache entries it wrote.

``--nki BUCKET:TIER`` instead compiles one paged prefill segment kernel (``graph.SEGMENT`` layers, the call the
serving graph makes: ``n_chunks=1``, ``write_pages`` and, after a prior, ``prior_pages``/``prior_len``) with nki's
driver: no torch, no NeuronCore. It reports the trace and neuronx-cc times, the BIR instruction count and the NEFF
size.

Run inside the serving image on one free NeuronCore of a node (the torch path; --nki needs none):

    PYTHONPATH=$PWD:<sglang>/python python tools/kernels/qwen35_prefill_compile.py --core 0 \
        --layers 60 --chunk 16384 --slot 2048 --context 65536 --kv-slots 34 --state-slots 360 \
        --world 128 --dp 8 --config <config.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _neffs(cache_dir) -> dict:
    out = {}
    for root, _, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".neff"):
                p = os.path.join(root, f)
                out[p] = os.path.getsize(p)
    return out


def compile_torch(args) -> int:
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(args.core)
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"  # the giga kernels run at LNC 1
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62192")
    os.environ["SGLANG_NEURON_CPU_MODE"] = "0"
    for name, sub in (("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"), ("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "neff_local"),
                      ("NKI_COMPILE_CACHE_URL", "nki")):
        os.environ[name] = os.path.join(args.cache_root, sub)

    import torch

    from sglang_neuron.process_init import init_stack as _init_neuron_stack

    _init_neuron_stack()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from neuron_check_runtime import CheckGraph, start_rank_runtime
    from qwen35_decode_compile import _dims, bir_instruction_count  # noqa: F401 -- shared probe helpers

    start_rank_runtime(0, 1)
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.qwen35 import prefix
    from sglang_neuron.models.qwen35.config import Qwen35Layout
    from sglang_neuron.models.qwen35.graph import Qwen35DecodeGraph, Qwen35PrefillGraph, operand_shapes, \
        prefill_extra_shapes
    from sglang_neuron.models.qwen35.runtime import PREFILL_NKI_COMPILE_MODE, with_nki_compile_mode
    from sglang_neuron.models.qwen35.sglang_model import prefill_compiler_args

    dims = _dims(args.config)
    layout = Qwen35Layout(world=args.world, dp=args.dp, slot_tokens=args.slot, context=args.context)
    problems = dims.problems() + layout.problems(dims, attention_layers=args.layers // 4, slots=args.kv_slots)
    if problems:
        print(json.dumps({"ok": False, "refused": problems}))
        return 2
    buckets = prefix.prefill_buckets(args.chunk, args.slot)
    tiers = prefix.prior_tiers(layout.rope_len, args.slot)
    programs = [(b, t) for b in buckets for t in tiers]
    if args.programs:
        want = {(int(v.split(":")[0]), int(v.split(":")[1])) for v in args.programs.split(",")}
        programs = [p for p in programs if p in want]
    print(json.dumps({"ok": True, "event": "plan", "layers": args.layers, "chunk": args.chunk, "page": args.slot,
                      "context": layout.rope_len, "buckets": list(buckets), "tiers": list(tiers),
                      "programs": [list(p) for p in programs]}), flush=True)

    dev = torch.device("neuron", 0)
    t0 = time.time()
    decode = Qwen35DecodeGraph(dims, layout, layers=args.layers).to(dev)
    decode.load_operands({n: torch.zeros(s, dtype=d) for n, (s, d) in
                          operand_shapes(dims, layout, layers=args.layers).items()}, dev)
    L_g, L_a = args.layers - args.layers // 4, args.layers // 4
    r = layout.replica
    vh = dims.num_v_heads // r
    cc = operand_shapes(dims, layout, layers=args.layers)["conv_w"][0][1]
    sp = args.state_slots + 2
    decode.bind_pools(
        torch.zeros((L_g * sp, cc, dims.conv_kernel - 1), dtype=torch.bfloat16, device=dev),
        torch.zeros((L_g * sp, dims.head_k_dim, vh, dims.head_v_dim), dtype=torch.float32, device=dev),
        torch.zeros((L_a * args.kv_slots, dims.head_dim, args.slot), dtype=torch.bfloat16, device=dev),
        torch.zeros((L_a * args.kv_slots, args.slot, dims.head_dim), dtype=torch.bfloat16, device=dev))
    graph = Qwen35PrefillGraph(decode)
    graph.load_operands({n: torch.zeros(s, dtype=d) for n, (s, d) in
                         prefill_extra_shapes(dims, layout, layers=args.layers).items()}, dev)
    t_alloc = time.time() - t0

    cargs = prefill_compiler_args(list(COMPILER_ARGS))
    rc = 0
    import numpy as np

    for bucket, tier in programs:
        # the warm-up's padding chunk (runtime.Qwen35PrefillRuntime._pad_tables): every page is a layer's page 0,
        # one valid row at position 0, the state pool's sink
        base = (np.arange(L_a, dtype=np.int64) * args.kv_slots)[:, None]
        write = np.repeat(base, bucket // args.slot, axis=1).astype(np.int32)
        prior = np.repeat(base, tier // args.slot, axis=1).astype(np.int32) if tier else None
        positions = np.arange(bucket, dtype=np.int64)
        sink = args.state_slots + 1
        i32, i64 = torch.int32, torch.int64
        up = lambda a, dt: torch.as_tensor(a).to(dt).to(dev)  # noqa: E731
        inputs = (torch.zeros((bucket,), dtype=i32, device=dev), None, torch.tensor([sink], dtype=i64, device=dev),
                  up(positions, i64), torch.tensor([sink], dtype=i64, device=dev),
                  up(write, i32), torch.tensor([[1]], dtype=i32, device=dev),
                  up(prior, i32) if prior is not None else None,
                  torch.tensor([[0]], dtype=i32, device=dev) if tier else None)
        compiled = CheckGraph(graph, compiler_args=cargs)
        before = _neffs(os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"])
        t1 = time.time()
        try:
            with_nki_compile_mode(PREFILL_NKI_COMPILE_MODE, lambda: compiled.precompile(*inputs))
        except Exception as exc:  # noqa: BLE001 -- the report names the failure and the run goes on
            print(json.dumps({"ok": False, "prefill": [bucket, tier],
                              "error": f"{type(exc).__name__}: {exc}"[:4000],
                              "compile_s": round(time.time() - t1, 1)}), flush=True)
            rc = 1
            continue
        after = _neffs(os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"])
        new = sorted(((p, s) for p, s in after.items() if before.get(p) != s), key=lambda x: -x[1])
        print(json.dumps({"ok": True, "prefill": [bucket, tier], "layers": args.layers, "page": args.slot,
                          "context": layout.rope_len, "alloc_s": round(t_alloc, 1),
                          "compile_s": round(time.time() - t1, 1),
                          "neffs": new[:4], "neff_bytes": sum(s for _, s in new)}), flush=True)
    return rc


def compile_nki_segment(args) -> int:
    """One paged prefill segment (graph.SEGMENT layers) of a --nki BUCKET:TIER program, compiled by nki's driver."""
    import ml_dtypes
    import numpy as np
    import torch
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import TracerFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff

    sys.path.insert(0, os.path.dirname(__file__))
    from qwen35_decode_compile import _backend_options, _dims, bir_instruction_count

    from sglang_neuron.models.qwen35.config import Qwen35Layout
    from sglang_neuron.models.qwen35.graph import SEGMENT, operand_shapes, prefill_extra_shapes
    from sglang_neuron.ops import qwen35_giga

    dims = _dims(args.config)
    layout = Qwen35Layout(world=args.world, dp=args.dp, slot_tokens=args.slot, context=args.context)
    bucket, tier = (int(v) for v in args.nki.split(":"))
    r = layout.replica
    if bucket % r or bucket % args.slot or (tier and tier % args.slot):
        print(json.dumps({"ok": False, "refused": [f"the program {bucket}:{tier} is not whole {args.slot}-token "
                                                   f"pages split over the {r}-rank replica"]}))
        return 2
    bf16 = np.dtype(ml_dtypes.bfloat16)
    np_dtype = {torch.bfloat16: bf16, torch.float32: np.float32, torch.int32: np.int32, torch.uint8: np.uint8}
    full = {n: (s, np_dtype[d]) for n, (s, d) in operand_shapes(dims, layout, layers=SEGMENT).items()}
    full.update({n: (s, np_dtype[d]) for n, (s, d) in prefill_extra_shapes(dims, layout, layers=SEGMENT).items()})

    def z(name):
        shape, dt = full[name]
        return np.zeros(shape, dt)

    ng, na = SEGMENT - SEGMENT // 4, SEGMENT // 4
    vh = dims.num_v_heads // r
    la = dims.num_layers // 4  # the pools are the serving pool's full layer span; the segment indexes them
    i32 = np.int32
    inputs = dict(
        x_shard=np.zeros((bucket // r, dims.hidden), bf16), in_ln_g=z("in_ln_g"), w_proj_p=z("w_proj_p"),
        conv_w=z("conv_w"), a_log=z("a_log"), dt_bias=z("dt_bias"), norm_w=z("norm_w"), w_out=z("w_out"),
        in_ln_a=z("in_ln_a"), w_qkv=z("w_qkv"), q_norm_w=z("q_norm_w"), k_norm_w=z("k_norm_w"),
        w_gate_p=z("w_gate_p"), w_o=z("w_o"), cos=np.zeros((bucket, dims.rotary_dim // 2), np.float32),
        sin=np.zeros((bucket, dims.rotary_dim // 2), np.float32),
        conv_state=np.zeros((ng, full["conv_w"][0][1], dims.conv_kernel - 1), bf16),
        rec_state=np.zeros((ng, dims.head_k_dim, vh, dims.head_v_dim), np.float32),
        k_cache_t=np.zeros((la * args.kv_slots, dims.head_dim, args.slot), bf16),
        v_cache=np.zeros((la * args.kv_slots, args.slot, dims.head_dim), bf16),
        kv_slot=np.array([[bucket]], i32),  # unused by the paged kernel: the graph passes valid_len here
        attn_sub=np.zeros((1, 1), i32), post_ln_w=z("post_ln_w"),
        w_router_p=z("w_router_p"), w_shared_p=z("w_shared_full_p"), w_sd=z("w_sd_full"),
        expert_ids=z("expert_ids"), gate_up_w=z("gate_up_w"), gate_up_scale=z("gate_up_scale"),
        down_w=z("down_w"), down_scale=z("down_scale"),
        write_pages=np.zeros((na, bucket // args.slot), i32), valid_len=np.array([[bucket]], i32),
        tp_mesh=layout.tp_mesh(), ep_mesh=layout.ep_mesh(), seq_len=bucket, num_layers=SEGMENT)
    if tier:
        kernel = qwen35_giga._prefill_paged_prior_jit
        inputs.update(prior_pages=np.zeros((na, tier // args.slot), i32), prior_len=np.array([[tier]], i32))
    else:
        kernel = qwen35_giga._prefill_paged_jit
    kernel = getattr(kernel, "func", kernel)
    out = os.path.join(args.out, f"prefill_seg_b{bucket}_t{tier}_s{args.slot}")
    os.makedirs(out, exist_ok=True)
    opts = CompileOptions(target=os.environ.get("NKI_COMPILE_TARGET", "trn3"), lnc=1,
                          output_path=os.path.join(out, "file.neff"), artifacts_dir=os.path.join(out, "artifacts"))
    if args.harness_opts:  # nki-library's harness default, or one of its options (Q35_HARNESS_OPTS)
        opts = _backend_options(opts)
    t0 = time.monotonic()
    bir = compile_to_bir(kernel_func=kernel, frontend=TracerFrontend(), inputs=inputs, compile_opts=opts)
    t1 = time.monotonic()
    names = [s.name for s in bir.descriptor.input_specs]
    outs = [s.name for s in bir.descriptor.output_specs]
    compiled = compile_bir_to_neff(opts, bir, [], names, outs)
    neff = compiled.neff_path
    print(json.dumps({"ok": True, "mode": "nki-prefill-segment", "prefill": [bucket, tier], "segment": SEGMENT,
                      "slot": args.slot, "trace_s": round(t1 - t0, 1),
                      "neuronx_cc_s": round(time.monotonic() - t1, 1),
                      "bir_instructions": bir_instruction_count(kernel, inputs, opts),
                      "mac_count": getattr(bir, "mac_count", None), "neff": neff,
                      "neff_bytes": os.path.getsize(neff) if os.path.exists(neff) else None,
                      "inputs": len(names), "outputs": outs}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--core", type=int, default=0)
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=16384, help="SGLang --chunked-prefill-size")
    ap.add_argument("--slot", type=int, default=2048, help="KV page tokens (SGLang --page-size)")
    ap.add_argument("--context", type=int, default=65536, help="SGLang --context-length (paged pool)")
    ap.add_argument("--kv-slots", type=int, default=34, help="KV pages per layer, reserved page 0 included")
    ap.add_argument("--state-slots", type=int, default=360, help="GDN state slots (the pool adds slot 0 and the sink)")
    ap.add_argument("--world", type=int, default=128)
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument("--config", default=None, help="the model's config.json (default: the 397B dimensions)")
    ap.add_argument("--cache-root", default=os.path.expanduser("~/.cache/sglang_neuron_q35p"))
    ap.add_argument("--programs", default="", help="torch path: compile only these BUCKET:TIER programs (comma list)")
    ap.add_argument("--nki", default="", help="compile one paged segment kernel of BUCKET:TIER with nki's driver "
                    "(no torch, no core)")
    ap.add_argument("--out", default="/tmp/qwen35_prefill_nki")
    ap.add_argument("--harness-opts", action="store_true", help="with --nki: nki-library's harness compile options")
    args = ap.parse_args(argv)
    if args.nki:
        return compile_nki_segment(args)
    return compile_torch(args)


if __name__ == "__main__":
    sys.exit(main())
