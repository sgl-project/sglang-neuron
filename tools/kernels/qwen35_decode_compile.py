# SPDX-License-Identifier: Apache-2.0
"""Compile the Qwen3.5 giga decode graph (models/qwen35/graph.py) for one rank, the way the serving runtime compiles
it: a ``torch.compile`` graph on the ``neuron`` backend whose body is the one ``wrap_nki`` call of
``qwen35_hybrid_multistep`` with the stacked operands, the four in-place pools and the step's row tables, compiled
under torch-neuronx's compile-only mode with the serving compiler arguments. No collective runs: the graph is lowered
to its NEFF and the process reports the trace + compile time and the NEFF cache entries it wrote (one JSON line).

``--nki`` compiles the same kernel call without torch and without a NeuronCore: the graph's ``wrap_nki`` body
(``ops.qwen35_giga._multistep_jit``) traced to BIR and compiled to a NEFF by nki's driver (the path nki-library's
harness and tools/kernels/compile_smoke.py take), with the same operands, pools, row tables and static meshes.

The operands are zeros of the per-rank shapes of the layout (``graph.operand_shapes``): the compile depends on shapes
only. Run inside the serving image on one free NeuronCore of a node:

    PYTHONPATH=$PWD:<sglang>/python python tools/kernels/qwen35_decode_compile.py --core 0 \
        --layers 4 --rows 16 --slot 1280 --kv-slots 8 --state-slots 8 [--world 128 --dp 8] [--config <config.json>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def compiler_args() -> list:
    """The serving graph compiler arguments with the giga decode's changes: -O2 without the modular-flow threshold and
    without the FSDP annotation of collectives (the GPT-OSS giga decode's settings)."""
    import shlex

    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS

    prefix = "--internal-hlo2tensorizer-options="
    args = []
    for arg in COMPILER_ARGS:
        if arg in ("-O0", "-O1", "-O2", "-O3"):
            continue
        if arg.startswith(prefix):
            options = [o for o in shlex.split(arg[len(prefix):]) if not o.startswith("--modular-flow-mac-threshold=")]
            arg = prefix + shlex.join(options + ["--enable-annotate-fsdp-cc=false"])
        args.append(arg)
    return args + ["-O2"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--core", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rows", type=int, default=16, help="decode rows per replica")
    ap.add_argument("--slot", type=int, default=1280, help="KV slot tokens (SGLang --page-size)")
    ap.add_argument("--kv-slots", type=int, default=8, help="KV pages per layer, reserved page 0 included")
    ap.add_argument("--state-slots", type=int, default=8, help="GDN state slots (the pool adds slot 0 and the sink)")
    ap.add_argument("--world", type=int, default=128)
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument("--config", default=None, help="the model's config.json (default: the 397B dimensions)")
    ap.add_argument("--cache-root", default=os.path.expanduser("~/.cache/sglang_neuron_q35"))
    ap.add_argument("--nki", action="store_true", help="compile the kernel call with nki's driver (no torch, no core)")
    ap.add_argument("--out", default="/tmp/qwen35_decode_nki")
    ap.add_argument("--harness-opts", action="store_true", help="with --nki: nki-library's harness compile options "
                    "(backend optimizations off)")
    ap.add_argument("--context", type=int, default=0, help="a paged KV pool (models/qwen35 paged launch): --slot is "
                    "the KV page, the rows read through a block table of context / page pages")
    ap.add_argument("--kv-cp", type=int, default=1, help="with --nki and --context: the decode context-parallel "
                    "degree (Qwen35Layout.kv_cp = the replica size): the 4-D physical-block pool + cp_sink/atp_rank "
                    "inputs, the CP multistep (_multistep_jit_cp)")
    ap.add_argument("--prefill-graph", default="", help="torch path, paged pool: compile the serving prefill graph "
                    "(graph.Qwen35PrefillGraph, modular flow, prefill_compiler_args) of BUCKET:TIER instead of the decode")
    ap.add_argument("--run", action="store_true", help="torch path: also run the compiled graph and time it "
                    "(--reps runs; zeros operands)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--prefill", type=int, default=0, help="with --nki: compile one prefill segment of this many "
                    "prompt tokens instead of the decode")
    args = ap.parse_args(argv)
    if args.nki and args.prefill:
        return compile_nki_prefill(args)
    if args.nki:
        return compile_nki(args)

    os.environ["NEURON_RT_VISIBLE_CORES"] = str(args.core)
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"  # the giga kernels run at LNC 1
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62191")
    os.environ["SGLANG_NEURON_CPU_MODE"] = "0"
    for name, sub in (("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"), ("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "neff_local"),
                      ("NKI_COMPILE_CACHE_URL", "nki")):
        os.environ[name] = os.path.join(args.cache_root, sub)

    import torch

    from sglang_neuron.process_init import init_stack as _init_neuron_stack

    _init_neuron_stack()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from neuron_check_runtime import CheckGraph, start_rank_runtime

    start_rank_runtime(0, 1)
    from sglang_neuron.models.qwen35.config import Qwen35Layout
    from sglang_neuron.models.qwen35.graph import Qwen35DecodeGraph, operand_shapes

    dims = _dims(args.config)
    layout = Qwen35Layout(world=args.world, dp=args.dp, slot_tokens=args.slot, context=args.context)
    problems = dims.problems() + layout.problems(dims, attention_layers=args.layers // 4, slots=args.kv_slots)
    problems += layout.row_problems(dims, args.rows, kv_slots=None if layout.paged else args.kv_slots,
                                    state_slots=args.state_slots + 2)
    if problems:
        print(json.dumps({"ok": False, "refused": problems}))
        return 2
    dev = torch.device("neuron", 0)
    t0 = time.time()
    graph = Qwen35DecodeGraph(dims, layout, layers=args.layers).to(dev)
    shapes = operand_shapes(dims, layout, layers=args.layers)
    graph.load_operands({n: torch.zeros(s, dtype=d) for n, (s, d) in shapes.items()}, dev)
    L_g, L_a = args.layers - args.layers // 4, args.layers // 4
    r = layout.replica
    vh = dims.num_v_heads // r
    cc = shapes["conv_w"][0][1]
    sp = args.state_slots + 2
    graph.bind_pools(
        torch.zeros((L_g * sp, cc, dims.conv_kernel - 1), dtype=torch.bfloat16, device=dev),
        torch.zeros((L_g * sp, dims.head_k_dim, vh, dims.head_v_dim), dtype=torch.float32, device=dev),
        torch.zeros((L_a * args.kv_slots, dims.head_dim, args.slot), dtype=torch.bfloat16, device=dev),
        torch.zeros((L_a * args.kv_slots, args.slot, dims.head_dim), dtype=torch.bfloat16, device=dev))
    rows, shard = args.rows, args.rows // r
    i32 = torch.int32
    if args.prefill_graph:
        return _torch_prefill(args, dims, layout, graph, dev, CheckGraph, t0)
    kv = torch.zeros((rows, 1), dtype=i32, device=dev)
    bt = torch.zeros((rows, layout.pages_per_row), dtype=i32, device=dev) if layout.paged else None
    inputs = (torch.zeros((rows,), dtype=i32, device=dev), torch.zeros((rows, 1), dtype=i32, device=dev),
              None if layout.paged else kv,
              torch.full((rows, 1), sp - 1, dtype=i32, device=dev),
              torch.zeros((args.dp * rows, 1), dtype=i32, device=dev), bt)
    t_alloc = time.time() - t0
    compiled = CheckGraph(graph, compiler_args=compiler_args())
    t1 = time.time()
    try:
        compiled.precompile(*inputs)
    except Exception as exc:  # noqa: BLE001 -- the report names the failure
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:4000], "alloc_s": round(t_alloc, 1),
                          "compile_s": round(time.time() - t1, 1)}))
        raise
    neffs = []
    for root, _, files in os.walk(os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"]):
        neffs += [(os.path.join(root, f), os.path.getsize(os.path.join(root, f))) for f in files if f.endswith(".neff")]
    t_compile = time.time() - t1
    timing = _time_runs(args, compiled, inputs) if args.run else None
    print(json.dumps({"ok": True, "layers": args.layers, "rows": rows, "slot": args.slot, "context": args.context,
                      "alloc_s": round(t_alloc, 1), "compile_s": round(t_compile, 1), "timing": timing,
                      "neffs": sorted(neffs, key=lambda x: -x[1])[:3]}))
    return 0


def _time_runs(args, compiled, inputs) -> dict:
    """Run the compiled graph args.reps times after one warm call: per-call wall times (ms)."""
    import torch

    out = compiled(*inputs)
    _sync(out)
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        _sync(compiled(*inputs))
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"min_ms": round(ts[0], 3), "median_ms": round(ts[len(ts) // 2], 3), "reps": len(ts)}


def _sync(out):
    t = out["token_ids"] if isinstance(out, dict) else out
    t.cpu()


def _torch_prefill(args, dims, layout, decode, dev, CheckGraph, t0) -> int:
    """The serving prefill graph of one (bucket, prior tier) on the paged pool, compiled as the prefill family is:
    modular flow, prefill_compiler_args, NKI_COMPILE_MODE=enable-alloc=legacy (runtime.PREFILL_NKI_COMPILE_MODE)."""
    import torch

    from sglang_neuron.models.qwen35 import prefix
    from sglang_neuron.models.qwen35.graph import Qwen35PrefillGraph, prefill_extra_shapes
    from sglang_neuron.models.qwen35.runtime import PREFILL_NKI_COMPILE_MODE
    from sglang_neuron.models.qwen35.sglang_model import prefill_compiler_args

    bucket, tier = (int(v) for v in args.prefill_graph.split(":"))
    if not layout.paged:
        raise SystemExit("--prefill-graph compiles the paged prefill: pass --context past --slot")
    graph = Qwen35PrefillGraph(decode)
    graph.load_operands({n: torch.zeros(s, dtype=d) for n, (s, d) in
                         prefill_extra_shapes(dims, layout, layers=args.layers).items()}, dev)
    L_a = args.layers // 4
    tables = prefix.chunk_tables(pages=list(range(1, 2 + (tier + bucket) // args.slot)), start=tier, n=bucket,
                                 program=prefix.PrefillProgram(bucket, tier), page=args.slot,
                                 pages_per_layer=args.kv_slots, attention_layers=L_a, rope_len=layout.rope_len)
    i32, i64 = torch.int32, torch.int64
    up = lambda a, dt: torch.as_tensor(a).to(dt).to(dev)  # noqa: E731
    inputs = (torch.zeros((bucket,), dtype=i32, device=dev), None, torch.tensor([1], dtype=i64, device=dev),
              up(tables.positions, i64), torch.tensor([args.state_slots + 1], dtype=i64, device=dev),
              up(tables.write_pages, i32), torch.tensor([[bucket]], dtype=i32, device=dev),
              up(tables.prior_pages, i32) if tables.prior_pages is not None else None,
              torch.tensor([[tier]], dtype=i32, device=dev) if tier else None)
    t_alloc = time.time() - t0
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS

    os.environ["NKI_COMPILE_MODE"] = PREFILL_NKI_COMPILE_MODE
    compiled = CheckGraph(graph, compiler_args=prefill_compiler_args(list(COMPILER_ARGS)))
    t1 = time.time()
    try:
        compiled.precompile(*inputs)
    except Exception as exc:  # noqa: BLE001 -- the report names the failure
        print(json.dumps({"ok": False, "prefill": [bucket, tier], "error": f"{type(exc).__name__}: {exc}"[:4000],
                          "compile_s": round(time.time() - t1, 1)}))
        raise
    t_compile = time.time() - t1
    timing = _time_runs(args, compiled, inputs) if args.run else None
    print(json.dumps({"ok": True, "prefill": [bucket, tier], "layers": args.layers, "page": args.slot,
                      "context": args.context, "alloc_s": round(t_alloc, 1), "compile_s": round(t_compile, 1),
                      "timing": timing}))
    return 0


# Qwen3.5-397B-A17B's text config (the fields the host reads): the default when no --config is given
QWEN35_397B = {
    "model_type": "qwen3_5_moe_text", "hidden_size": 4096, "num_hidden_layers": 60, "full_attention_interval": 4,
    "linear_num_key_heads": 16, "linear_num_value_heads": 64, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "num_attention_heads": 32, "num_key_value_heads": 2,
    "head_dim": 256, "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024, "vocab_size": 248320, "rms_norm_eps": 1e-6, "mamba_ssm_dtype": "float32",
    "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25, "rope_type": "default"},
}


def _dims(config_path):
    from sglang_neuron.models.qwen35.config import Qwen35Dims

    if config_path:
        config = json.load(open(config_path))
        return Qwen35Dims.from_hf(config.get("text_config") or config)
    return Qwen35Dims.from_hf(QWEN35_397B)


def _backend_options(opts):
    """Q35_HARNESS_OPTS: "1" (default) nki-library's harness options (neuronx-cc instruction scheduling and linear-scan
    allocation off, register computes as affine expressions), or one alone: "sched", "alloc", "affine"."""
    from dataclasses import replace

    mode = os.environ.get("Q35_HARNESS_OPTS", "1")
    if mode == "sched":
        return opts.set_pipeline_options("enable-instruction-scheduling=false")
    if mode == "alloc":
        return opts.set_pipeline_options("enable-linear-scan-allocation=false")
    if mode == "affine":
        return replace(opts, emit_reg_compute_as_affine_expr=True)
    return opts.disable_backend_optimizations()


def bir_instruction_count(kernel_func, inputs, opts) -> "int | None":
    """The traced kernel's BIR instruction count (nested blocks included): a second trace with ``use_colz`` off —
    the timed compile keeps the served colz flow — counted from the kernel JSON the driver then writes
    (``NirResult.kernel_json_path``). None when the count trace fails."""
    import glob
    import tempfile
    from dataclasses import replace

    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import TracerFrontend

    try:
        with tempfile.TemporaryDirectory(prefix="q35count") as scratch:
            opts2 = replace(opts, use_colz=False, artifacts_dir=scratch)
            bir = compile_to_bir(kernel_func=kernel_func, frontend=TracerFrontend(), inputs=inputs,
                                 compile_opts=opts2)
            path = bir.kernel_json_path
            if not path or not os.path.exists(path):
                cands = [p for p in glob.glob(os.path.join(scratch, "**", "*.json"), recursive=True)
                         if os.path.getsize(p)]
                path = None
                for cand in sorted(cands, key=lambda p: -os.path.getsize(p)):
                    try:
                        if "functions" in json.load(open(cand)):
                            path = cand
                            break
                    except (OSError, ValueError):
                        continue
                if path is None:
                    return None
            doc = json.load(open(path))
    except Exception:  # noqa: BLE001 -- the count never fails the compile report
        return None
    n = 0

    def visit(block):
        nonlocal n
        for ins in block.get("instructions", []):
            n += 1
            for sub in ins.get("blocks", []) or []:
                visit(sub)

    for f in doc["functions"]:
        for b in f.get("blocks", []):
            visit(b)
    return n


def _probe_multistep_kernels():
    """@nki.jit wrappers with the multistep's FULL explicit operand list. qwen35_giga's own ``_multistep_jit`` /
    ``_multistep_jit_cp`` leave the MoE/LM-head operands as free variables (a serving-path concern); the driver's
    frontend binds raw ``.func`` signatures, so the probe traces these wrappers instead."""
    import nki

    from sglang_neuron.ops import qwen35_giga as _giga

    @nki.jit
    def _probe_ms(
        tok_shard, embed_table, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out,
        w_qkv, q_norm_w, k_norm_w, w_gate_p, cos_tab, sin_tab, k_cache_t, v_cache, pos, w_o, post_ln_w, w_router_p,
        sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, final_ln_w, w_lm_p, vbase, ep_rank,
        gdn_slot, kv_slot, row_valid,
        tp_mesh: tuple = (),
        ep_mesh: tuple = (),
        tp_size: int = 0,
        ep_size: int = 0,
        num_layers: int = 0,
        num_steps: int = 1,
        paged: bool = False,
    ):
        return _giga._trace_multistep(
            tok_shard, embed_table, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out,
            w_qkv, q_norm_w, k_norm_w, w_gate_p, cos_tab, sin_tab, k_cache_t, v_cache, pos, w_o, post_ln_w,
            w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, final_ln_w, w_lm_p,
            vbase, ep_rank, gdn_slot, kv_slot, row_valid,
            tp_mesh, ep_mesh, tp_size, ep_size, num_layers, num_steps, paged, 1, None, None)

    @nki.jit
    def _probe_ms_cp(
        tok_shard, embed_table, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out,
        w_qkv, q_norm_w, k_norm_w, w_gate_p, cos_tab, sin_tab, k_cache_t, v_cache, pos, w_o, post_ln_w, w_router_p,
        sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, final_ln_w, w_lm_p, vbase, ep_rank,
        gdn_slot, kv_slot, row_valid, cp_sink, atp_rank,
        tp_mesh: tuple = (),
        ep_mesh: tuple = (),
        tp_size: int = 0,
        ep_size: int = 0,
        num_layers: int = 0,
        num_steps: int = 1,
        kv_cp: int = 1,
    ):
        return _giga._trace_multistep(
            tok_shard, embed_table, in_ln_w, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out,
            w_qkv, q_norm_w, k_norm_w, w_gate_p, cos_tab, sin_tab, k_cache_t, v_cache, pos, w_o, post_ln_w,
            w_router_p, sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, final_ln_w, w_lm_p,
            vbase, ep_rank, gdn_slot, kv_slot, row_valid,
            tp_mesh, ep_mesh, tp_size, ep_size, num_layers, num_steps, True, kv_cp, cp_sink, atp_rank)

    return _probe_ms, _probe_ms_cp


def compile_nki(args) -> int:
    """The kernel call of the decode graph compiled by nki's driver (module docstring)."""
    import ml_dtypes
    import numpy as np
    import torch
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import TracerFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff

    from sglang_neuron.models.qwen35.config import Qwen35Layout
    from sglang_neuron.models.qwen35.graph import operand_shapes
    from sglang_neuron.ops import qwen35_giga

    qwen35_giga.require()
    dims = _dims(args.config)
    layout = Qwen35Layout(world=args.world, dp=args.dp, slot_tokens=args.slot, context=args.context,
                          kv_cp=args.kv_cp)
    problems = dims.problems() + layout.problems(dims, attention_layers=args.layers // 4, slots=args.kv_slots)
    problems += layout.row_problems(dims, args.rows, kv_slots=None if layout.paged else args.kv_slots,
                                    state_slots=args.state_slots + 2)
    if problems:
        print(json.dumps({"ok": False, "refused": problems}))
        return 2
    bf16 = np.dtype(ml_dtypes.bfloat16)
    np_dtype = {torch.bfloat16: bf16, torch.float32: np.float32, torch.int32: np.int32, torch.uint8: np.uint8}
    inputs = {n: np.zeros(s, np_dtype[d]) for n, (s, d) in operand_shapes(dims, layout, layers=args.layers).items()}
    L_g, L_a = args.layers - args.layers // 4, args.layers // 4
    vh = dims.num_v_heads // layout.replica
    sp = args.state_slots + 2
    cc = inputs["conv_w"].shape[1]
    rows, shard = args.rows, args.rows // layout.replica
    inputs.update(
        conv_state=np.zeros((L_g * sp, cc, dims.conv_kernel - 1), bf16),
        rec_state=np.zeros((L_g * sp, dims.head_k_dim, vh, dims.head_v_dim), np.float32),
        tok_shard=np.zeros((shard, 1), np.int32), pos=np.zeros((rows, 1), np.int32),
        kv_slot=(np.zeros((rows, layout.pages_per_row), np.int32) if layout.paged
                 else np.zeros((rows, 1), np.int32)),
        gdn_slot=np.full((rows, 1), sp - 1, np.int32),
        row_valid=np.zeros((args.dp * rows, 1), np.int32),
        tp_mesh=layout.tp_mesh(), ep_mesh=layout.ep_mesh(), tp_size=layout.replica, ep_size=layout.world,
        num_layers=args.layers, num_steps=1)
    probe_ms, probe_ms_cp = _probe_multistep_kernels()
    if args.kv_cp > 1:  # DCP: the 4-D physical-block pool (a rank's block of each page), the sink + the rank inputs
        cpb = args.slot // args.kv_cp
        inputs.update(
            k_cache_t=np.zeros((L_a * args.kv_slots + 1, dims.num_kv_heads, dims.head_dim, cpb), bf16),
            v_cache=np.zeros((L_a * args.kv_slots + 1, dims.num_kv_heads, cpb, dims.head_dim), bf16),
            cp_sink=np.full((1, 1), L_a * args.kv_slots, np.int32),
            atp_rank=np.zeros((1, 1), np.int32),
            kv_cp=args.kv_cp)
        kernel = getattr(probe_ms_cp, "func", probe_ms_cp)
    else:
        inputs.update(
            k_cache_t=np.zeros((L_a * args.kv_slots, dims.head_dim, args.slot), bf16),
            v_cache=np.zeros((L_a * args.kv_slots, args.slot, dims.head_dim), bf16),
            paged=layout.paged)
        kernel = getattr(probe_ms, "func", probe_ms)
    out = os.path.join(args.out, f"L{args.layers}_r{rows}_s{args.slot}" + (f"_ctx{args.context}" if layout.paged else "")
                       + (f"_cp{args.kv_cp}" if args.kv_cp > 1 else "")
                       + ("_harness" if args.harness_opts else ""))
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
    t2 = time.monotonic()
    neff = compiled.neff_path
    print(json.dumps({"ok": True, "mode": "nki", "layers": args.layers, "rows": rows, "slot": args.slot,
                      "context": args.context, "kv_cp": args.kv_cp, "harness_opts": bool(args.harness_opts),
                      "kv_slots": args.kv_slots, "state_slots": args.state_slots, "trace_s": round(t1 - t0, 1),
                      "neuronx_cc_s": round(t2 - t1, 1),
                      "bir_instructions": bir_instruction_count(kernel, inputs, opts),
                      "mac_count": getattr(bir, "mac_count", None), "neff": neff,
                      "neff_bytes": os.path.getsize(neff) if os.path.exists(neff) else None,
                      "inputs": len(names), "outputs": outs}))
    return 0


def compile_nki_prefill(args) -> int:
    """One prefill segment (graph.SEGMENT layers) of a ``--prefill``-token prompt per replica, compiled by nki's
    driver with the served operands' shapes."""
    import ml_dtypes
    import numpy as np
    import torch
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import TracerFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff

    from sglang_neuron.models.qwen35.config import Qwen35Layout
    from sglang_neuron.models.qwen35.graph import CHUNK, SEGMENT, operand_shapes, prefill_extra_shapes
    from sglang_neuron.ops import qwen35_giga

    dims = _dims(args.config)
    layout = Qwen35Layout(world=args.world, dp=args.dp, slot_tokens=args.slot)
    L = args.prefill
    nch = max(1, L // CHUNK)
    T_pc = L // nch // layout.replica
    bf16 = np.dtype(ml_dtypes.bfloat16)
    np_dtype = {torch.bfloat16: bf16, torch.float32: np.float32, torch.int32: np.int32, torch.uint8: np.uint8}
    full = {n: (s, np_dtype[d]) for n, (s, d) in operand_shapes(dims, layout, layers=SEGMENT).items()}
    full.update({n: (s, np_dtype[d]) for n, (s, d) in prefill_extra_shapes(dims, layout, layers=SEGMENT).items()})

    def z(name):
        shape, dt = full[name]
        return np.zeros(shape, dt)

    ng, na = SEGMENT - SEGMENT // 4, SEGMENT // 4
    vh = dims.num_v_heads // layout.replica
    inputs = dict(
        x_shard=np.zeros((nch * T_pc, dims.hidden), bf16), in_ln_g=z("in_ln_g"), w_proj_p=z("w_proj_p"),
        conv_w=z("conv_w"), a_log=z("a_log"), dt_bias=z("dt_bias"), norm_w=z("norm_w"), w_out=z("w_out"),
        in_ln_a=z("in_ln_a"), w_qkv=z("w_qkv"), q_norm_w=z("q_norm_w"), k_norm_w=z("k_norm_w"),
        w_gate_p=z("w_gate_p"), w_o=z("w_o"), cos=np.zeros((L, dims.rotary_dim // 2), np.float32),
        sin=np.zeros((L, dims.rotary_dim // 2), np.float32),
        conv_state=np.zeros((ng, full["conv_w"][0][1], dims.conv_kernel - 1), bf16),
        rec_state=np.zeros((ng, dims.head_k_dim, vh, dims.head_v_dim), np.float32),
        k_cache_t=np.zeros((dims.num_layers // 4 * args.kv_slots, dims.head_dim, args.slot), bf16),
        v_cache=np.zeros((dims.num_layers // 4 * args.kv_slots, args.slot, dims.head_dim), bf16),
        kv_slot=np.zeros((na, 1), np.int32), attn_sub=np.zeros((1, 1), np.int32), post_ln_w=z("post_ln_w"),
        w_router_p=z("w_router_p"), w_shared_p=z("w_shared_full_p"), w_sd=z("w_sd_full"),
        expert_ids=z("expert_ids"), gate_up_w=z("gate_up_w"), gate_up_scale=z("gate_up_scale"),
        down_w=z("down_w"), down_scale=z("down_scale"),
        tp_mesh=layout.tp_mesh(), ep_mesh=layout.ep_mesh(), seq_len=L, num_layers=SEGMENT, n_chunks=nch)
    kernel = getattr(qwen35_giga._prefill_jit, "func", qwen35_giga._prefill_jit)
    out = os.path.join(args.out, f"prefill_L{L}_s{args.slot}")
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
    print(json.dumps({"ok": True, "mode": "nki-prefill", "seq_len": L, "chunks": nch, "segment": SEGMENT,
                      "slot": args.slot, "trace_s": round(t1 - t0, 1), "neuronx_cc_s": round(time.monotonic() - t1, 1),
                      "bir_instructions": bir_instruction_count(kernel, inputs, opts),
                      "mac_count": getattr(bir, "mac_count", None),
                      "neff_bytes": os.path.getsize(neff) if os.path.exists(neff) else None, "outputs": outs}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
