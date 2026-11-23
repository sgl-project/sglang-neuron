# SPDX-License-Identifier: Apache-2.0
"""Run Hy4-Flash's Neuron graphs (models/hy_v4_flash) on the CPU at TP 1 and compare them with golden outputs of the
reference model (transformers' modeling_hy_v4_flash).

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 tools/hy4_flash/cpu_check.py --model /models/Hy4-flash-FP8 \
        --golden /hy4f/ref/golden/prompt_a.pt [--layers 6] [--decode 8]

The graphs run as the runtime runs them: a prefill of the whole prompt through SGLang-style metadata (one request, a
block table of 128-token pages from page 1, page 0 reserved), then ``--decode`` single-token steps teacher-forced on
the golden continuation. With ``--layers`` below the model's the check compares the per-layer residual streams the
golden file holds; with the whole model it compares the prefill's last-token logits and each decode step's argmax."""

from __future__ import annotations

import argparse
import dataclasses
import os
import time

import torch
import torch.distributed as dist

PAGE = 128


class FakeGroup:
    """A one-rank GroupCoordinator stand-in; ``world_size`` above 1 shapes the shards as that TP's rank 0 (its
    collectives still run over the one process)."""

    rank_in_group = 0

    def __init__(self, world_size: int = 1):
        self.world_size = world_size
        self.rank_in_group_tensor = torch.zeros((), dtype=torch.int32)
        self.device_group = dist.group.WORLD
        self.cpu_group = dist.group.WORLD


def install_fakes(tp: int = 1):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    group = FakeGroup(tp)
    from sglang_neuron.models.hy_v4_flash import attention, model
    from sglang_neuron.parallel import groups
    from sglang_neuron.parallel.state import STATE

    for mod in (groups,):
        mod.get_neuron_attn_tp_group = lambda: group
        mod.get_neuron_ep_group = lambda: group
    attention.get_attn_tp_group = lambda: group
    model.get_attn_tp_group = lambda: group
    STATE.ep_degree = tp
    STATE.dp_size = 1
    return group


def truncated(cfg, layers: int):
    if layers >= cfg.num_hidden_layers:
        return cfg
    keep = [i for i, layer in enumerate(cfg.engram_layers) if layer < layers]
    return dataclasses.replace(
        cfg, num_hidden_layers=layers, swa_layers=cfg.swa_layers[:layers], moe_layers=cfg.moe_layers[:layers],
        engram_layers=tuple(cfg.engram_layers[i] for i in keep), engram_ids=tuple(cfg.engram_ids[i] for i in keep),
        engram_head_sizes=tuple(cfg.engram_head_sizes[i] for i in keep))


def metadata(graph, block_table, slot_mapping, cached: int, swa_offset=None) -> dict:
    full = {"block_table_tensor": block_table, "slot_mapping": slot_mapping, "block_size": PAGE,
            "cached_seq_len": torch.tensor([[cached]], dtype=torch.int32)}
    md = {}
    for name, window in zip(graph.layer_names, graph.sliding_windows):
        if window and swa_offset is not None:
            md[name] = {**full, **swa_offset(window)}
        else:
            md[name] = full
    return md


KV_DTYPES = {"bf16": torch.bfloat16, "fp8_e4m3": torch.float8_e4m3fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--decode", type=int, default=0)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--kv-dtype", default="bf16", choices=sorted(KV_DTYPES), help="the KV pool's element type")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    group = install_fakes()
    from transformers import AutoConfig

    from sglang_neuron.models.hy_v4_flash.config import hy4_flash_config
    from sglang_neuron.models.hy_v4_flash.model import Hy4FlashGraph
    from sglang_neuron.sglang_patches.hy4_flash_launch import register_config

    register_config()
    cfg = hy4_flash_config(AutoConfig.from_pretrained(args.model))
    cfg = truncated(cfg, args.layers or cfg.num_hidden_layers)
    golden = torch.load(args.golden, weights_only=False)
    ids = torch.as_tensor(golden["input_ids"]).reshape(-1)
    t0 = time.time()
    with torch.device("cpu"):
        graphs = {phase: Hy4FlashGraph(cfg, phase) for phase in ("prefill", "decode")}
    graphs["prefill"].load_weights(args.model, torch.device("cpu"))
    from sglang_neuron.models.host import share_weights, place_rest_on_device

    place_rest_on_device("Hy4-Flash", "prefill", graphs["prefill"], torch.device("cpu"))
    share_weights(graphs["prefill"], graphs["decode"], name="Hy4-Flash", phase="decode", source_phase="prefill")
    place_rest_on_device("Hy4-Flash", "decode", graphs["decode"], torch.device("cpu"))
    graphs["decode"].bind_hashes()
    print(f"loaded {cfg.num_hidden_layers} layers in {time.time() - t0:.0f} s", flush=True)

    pages = args.max_context // PAGE
    slots = (pages + 1) * PAGE
    caches = {}
    for g in graphs.values():
        for layer in g.layers:
            name = layer.self_attn.name
            if name not in caches:
                half = (cfg.kv_lora_rank + cfg.sparse.qk_rope_head_dim) // 2
                caches[name] = [torch.zeros(pages + 1, 1, PAGE, half, dtype=KV_DTYPES[args.kv_dtype]) for _ in range(2)]
        g.bind_kv_cache(caches)
    state = None
    for g in graphs.values():
        state = g.bind_state(slots, torch.device("cpu"), state)
    table = torch.arange(1, pages + 1, dtype=torch.int32).view(1, -1)

    streams = {}
    hooks = []
    for i, layer in enumerate(graphs["prefill"].layers):
        hooks.append(layer.register_forward_hook(lambda m, a, out, i=i: streams.__setitem__(i, out.detach())))
    n = ids.numel()
    t0 = time.time()
    logits = graphs["prefill"](input_ids=ids.to(torch.int32), positions=torch.arange(n),
                               attn_metadata=metadata(graphs["prefill"], table, PAGE + torch.arange(n), 0),
                               sampling_positions=torch.tensor([n - 1]), rank=group.rank_in_group_tensor)
    print(f"prefill {n} tokens in {time.time() - t0:.1f} s", flush=True)
    for h in hooks:
        h.remove()

    ref_streams = golden.get("hidden_streams", golden.get("layer_streams", {}))
    for i, ref in sorted(ref_streams.items()):
        i = int(i)
        if i not in streams:
            continue
        ours, ref = streams[i].float(), torch.as_tensor(ref).float().reshape(streams[i].shape)
        rel = (ours - ref).norm() / ref.norm()
        cos = torch.nn.functional.cosine_similarity(ours.flatten(), ref.flatten(), dim=0)
        print(f"layer {i}: rel err {rel:.4f} cosine {cos:.6f}", flush=True)
    if cfg.num_hidden_layers < 42:
        return
    ref_logits = torch.as_tensor(golden["prefill_logits_last8"]).float().reshape(-1, logits.shape[-1])[-1]
    ours = logits.float().reshape(-1)
    print(f"prefill logits cosine {torch.nn.functional.cosine_similarity(ours, ref_logits, dim=0):.6f} "
          f"argmax ours {int(ours.argmax())} ref {int(ref_logits.argmax())}", flush=True)
    generated = [int(t) for t in golden["generated_ids"]]
    tokens = [int(ours.argmax())]
    for step in range(min(args.decode, len(generated) - 1)):
        pos = n + step
        tok = generated[step]
        bt = table.clone()

        def swa(window, pos=pos, bt=bt):
            first = max(0, (pos - window + 1) // PAGE)
            nb = window // PAGE + 1
            cols = torch.arange(first, first + nb).clamp(max=bt.shape[1] - 1)
            return {"block_table_tensor": bt[:, cols], "swa_kv_pos_offset": torch.tensor([first * PAGE])}

        out = graphs["decode"](input_ids=torch.tensor([tok], dtype=torch.int32), positions=torch.tensor([pos]),
                               attn_metadata=metadata(graphs["decode"], bt, torch.tensor([PAGE + pos]), 0, swa),
                               sampling_positions=torch.tensor([0]), rank=group.rank_in_group_tensor)
        tokens.append(int(out.float().reshape(-1).argmax()))
    print("ours     ", tokens, flush=True)
    print("reference", generated[:len(tokens)], flush=True)


if __name__ == "__main__":
    main()
