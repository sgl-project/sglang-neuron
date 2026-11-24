# SPDX-License-Identifier: Apache-2.0
"""Compile Hy4-Flash's first ``--layers`` decoder layers as one graph on one NeuronCore at TP 1 with random weights,
in the prefill or decode form the server warms, to bisect compiler failures on a small graph.

    NEURON_RT_VISIBLE_CORES=0 python3 tools/hy4_flash/mini_compile.py --model /models/Hy4-flash-FP8 \
        --phase prefill --tokens 2048 --context 9216 [--layers 1]"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import cpu_check as C  # noqa: E402

PAGE = 128


KV_DTYPES = {"bf16": torch.bfloat16, "fp8_e4m3": torch.float8_e4m3fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", default="prefill")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--context", type=int, default=9216)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--kv-dtype", default="bf16", choices=sorted(KV_DTYPES), help="the KV pool's element type")
    ap.add_argument("--tp", type=int, default=1, help="shard the weights as this TP's rank 0")
    ap.add_argument("--no-kernels", action="store_true", help="the decode without the MoE half kernel")
    ap.add_argument("--profile", default="", help="record the device profile of 2 calls into this directory")
    ap.add_argument("--bench", type=int, default=0,
                    help="run the compiled graph this many more times and report ms a call (with --tp above 1 the "
                         "collectives are identities, so the time is one rank's compute alone)")
    args = ap.parse_args()
    import torch_neuronx  # noqa: F401

    group = C.install_fakes(args.tp)
    if args.tp > 1:
        # one process: every collective is the identity, so the graph runs on one core as rank 0 would
        from sglang_neuron.parallel import collectives

        collectives.all_reduce = lambda group, tensor, op="sum": tensor
    from transformers import AutoConfig

    from sglang_neuron.models.hy_v4_flash.config import hy4_flash_config
    from sglang_neuron.models.hy_v4_flash.model import Hy4FlashGraph
    from sglang_neuron.sglang_patches.hy4_flash_launch import register_config

    register_config()
    cfg = C.truncated(hy4_flash_config(AutoConfig.from_pretrained(args.model)), args.layers)
    dev = torch.device("neuron")
    with torch.device("cpu"):
        g = Hy4FlashGraph(cfg, args.phase)
    torch.manual_seed(0)
    for name, p in list(g.named_parameters()):
        owner, _, attr = name.rpartition(".")
        mod = g.get_submodule(owner)
        if p.dtype in (torch.uint8, torch.int32):
            t = torch.zeros(p.shape, dtype=p.dtype)
        elif p.dtype == torch.float8_e4m3fn:
            t = (torch.randn(p.shape) * 0.1).to(p.dtype)
        else:
            t = (torch.randn(p.shape) * 0.02).to(p.dtype)
        mod._parameters[attr] = torch.nn.Parameter(t.to(dev), requires_grad=False)
    for name, b in list(g.named_buffers()):
        owner, _, attr = name.rpartition(".")
        g.get_submodule(owner)._buffers[attr] = b.to(dev)
    for layer in g.layers:
        if layer.engram is not None:
            layer.engram.bind_hash()
    if not args.no_kernels:
        g.prepare_kernels()
    pages = args.context // PAGE
    slots = (pages + 2) * PAGE
    half = (cfg.kv_lora_rank + cfg.sparse.qk_rope_head_dim) // 2
    caches = {layer.self_attn.name: [torch.zeros(pages + 2, 1, PAGE, half, dtype=KV_DTYPES[args.kv_dtype], device=dev)
                                     for _ in range(2)] for layer in g.layers}
    g.bind_kv_cache(caches)
    g.bind_state(slots, dev)
    T = args.tokens if args.phase == "prefill" else args.tokens
    table = torch.arange(1, pages + 1, dtype=torch.int32).view(1, -1)
    if args.phase == "decode":
        table = table.expand(T, -1).contiguous()
    md_full = {"block_table_tensor": table.to(dev), "slot_mapping": (PAGE + torch.arange(T)).to(dev),
               "block_size": PAGE, "cached_seq_len": torch.tensor([[0]], dtype=torch.int32, device=dev)}
    md = {}
    for name, window in zip(g.layer_names, g.sliding_windows):
        if window and args.phase == "decode":
            nb = window // PAGE + 1
            md[name] = {**md_full, "block_table_tensor": table[:, :nb].contiguous().to(dev),
                        "swa_kv_pos_offset": torch.zeros(T, dtype=torch.int32, device=dev)}
        else:
            md[name] = md_full
    positions = torch.arange(T) if args.phase == "prefill" else torch.full((T,), args.context - 1)
    kwargs = dict(input_ids=torch.ones(T, dtype=torch.int32, device=dev), positions=positions.to(dev),
                  attn_metadata=md, sampling_positions=torch.tensor([T - 1] if args.phase == "prefill"
                                                                    else list(range(T))).to(dev),
                  rank=group.rank_in_group_tensor.to(dev))
    compiled = torch.compile(g, backend="neuron", fullgraph=True, dynamic=False)
    t0 = time.time()
    out = compiled(**kwargs)
    if args.bench:
        for _ in range(3):
            compiled(**kwargs)
        torch.neuron.synchronize() if hasattr(torch, "neuron") else None
        t1 = time.time()
        for _ in range(args.bench):
            out = compiled(**kwargs)
        out.float().cpu()
        print(f"BENCH tp={args.tp} {args.phase} T={T} ctx={args.context} layers={args.layers}: "
              f"{(time.time() - t1) / args.bench * 1e3:.2f} ms a call", flush=True)
    if args.profile:
        from sglang_neuron.model_executor.profiling import device_profile

        os.environ.setdefault("TORCH_NEURONX_NEFF_CACHE_DIR", "/tmp/neff_cache")
        with device_profile(args.profile):
            for _ in range(2):
                out = compiled(**kwargs)
            out.float().cpu()
        print(f"PROFILE written under {args.profile}", flush=True)
    print(f"OK tp={args.tp} {args.phase} T={T} ctx={args.context} layers={args.layers}: "
          f"{tuple(out.shape)} in {time.time() - t0:.0f} s, finite={bool(torch.isfinite(out.float().cpu()).all())}",
          flush=True)


if __name__ == "__main__":
    main()
