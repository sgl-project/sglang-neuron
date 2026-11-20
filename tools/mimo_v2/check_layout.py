# SPDX-License-Identifier: Apache-2.0
"""CPU checks of MiMo-V2's checkpoint loaders and attention layout (models/mimo_v2/weights.py, rope.py) against
independent references, on real checkpoint tensors (tools/mimo_v2/fetch_tensors.py) and the HF model's formulas.

    python3 tools/mimo_v2/fetch_tensors.py --out /tmp/mimo_slices \\
        model.layers.0.self_attn.qkv_proj.weight model.layers.0.self_attn.qkv_proj.weight_scale_inv \\
        model.layers.1.self_attn.qkv_proj.weight model.layers.1.self_attn.qkv_proj.weight_scale_inv \\
        model.layers.1.self_attn.attention_sink_bias \\
        model.layers.1.mlp.experts.0.gate_proj.weight model.layers.1.mlp.experts.0.gate_proj.weight_scale \\
        model.layers.1.mlp.experts.0.down_proj.weight model.layers.1.mlp.experts.0.down_proj.weight_scale \\
        model.layers.0.mlp.down_proj.weight model.layers.0.mlp.down_proj.weight_scale_inv
    PYTHONPATH=. python3 tools/mimo_v2/check_layout.py --slices /tmp/mimo_slices

Needs torch and safetensors only (no SGLang, no Neuron). Checks:

1. the fused-QKV loader, every rank at TP 64, 16 and 8, equals an independent whole-tensor dequantization of the
   checkpoint's per-shard, per-section scale grids, de-interleaved and laid out per head;
2. the graphs' attention (the padded 256-channel heads, the moved RoPE pairs, the folded V scale) equals the HF
   model's eager attention (partial RoPE by rotate_half, V scale, sinks, sliding window) on a 300-token causal chunk,
   for a full and a sliding layer;
3. MXFP4 and flat-grid FP8 dequantization against bit-level references, and the bf16 views' windows against slices of
   the whole dequantized tensor."""

import argparse
import json
import math
import os
import sys

import torch
from safetensors import safe_open

from sglang_neuron.models.mimo_v2 import weights
from sglang_neuron.models.mimo_v2.config import KV_HEAD_DIM, mimo_v2_config
from sglang_neuron.models.mimo_v2.rope import RotaryTables, rotate

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# -- references, written apart from the loaders ------------------------------------------------------------------

def ref_fused_qkv(weight: torch.Tensor, scale: torch.Tensor, disk_tp: int, nh: int, nkv: int, hd: int, vhd: int):
    """fp32 (Q [nh, hd, H], K [nkv, hd, H], V [nkv, vhd, H]) of the stored fused QKV: each of ``disk_tp`` shards is
    [Q | K | V], each section scaled by its own 128-row grid."""
    sections = [(nh // disk_tp) * hd, (nkv // disk_tp) * hd, (nkv // disk_tp) * vhd]
    blocks = [math.ceil(s / 128) for s in sections]
    cols = weight.shape[1]
    out = {"q": [], "k": [], "v": []}
    row, blk = 0, 0
    for _ in range(disk_tp):
        for key, size, nb in zip("qkv", sections, blocks):
            w = weight[row:row + size].to(torch.float32)
            grid = scale[blk:blk + nb].to(torch.float32)
            full = grid.repeat_interleave(128, dim=0)[:size].repeat_interleave(128, dim=1)[:, :cols]
            out[key].append(w * full)
            row, blk = row + size, blk + nb
    q = torch.cat(out["q"]).view(nh, hd, cols)
    k = torch.cat(out["k"]).view(nkv, hd, cols)
    v = torch.cat(out["v"]).view(nkv, vhd, cols)
    return q, k, v


def ref_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """e2m1 by its bits (sign, two exponent bits, one mantissa bit), the low nibble first, times 2^(e8m0 - 127)."""
    def e2m1(n):
        sign = torch.where((n >> 3) & 1 == 1, -1.0, 1.0)
        exp, man = ((n >> 1) & 3).float(), (n & 1).float()
        mag = torch.where(exp == 0, man * 0.5, torch.exp2(exp - 1) * (1 + man / 2))
        return sign * mag
    p = packed.to(torch.int64)
    values = torch.stack([e2m1(p & 15), e2m1(p >> 4)], dim=-1).reshape(p.shape[0], -1)
    return values * torch.exp2(scale.to(torch.float32) - 127).repeat_interleave(32, dim=1)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def hf_attention(x, q_w, k_w, v_w, *, rope_theta, rotary, scaling, v_scale, window, sink, bf16_tables=False):
    """One query head's output [T, vhd] by the HF model (modeling_mimo_v2: MiMoV2Attention._forward_attention and
    eager_attention_forward) for x [T, H] at positions 0..T-1. ``bf16_tables`` rounds cos / sin to bf16, as the HF
    model does for bf16 activations and the graphs do."""
    tokens = x.shape[0]
    q, k, v = x @ q_w.t(), x @ k_w.t(), x @ v_w.t()
    v = v * v_scale
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary, 2, dtype=torch.int64).float() / rotary))
    freqs = torch.arange(tokens).float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos(), emb.sin()
    if bf16_tables:
        cos, sin = cos.to(torch.bfloat16).float(), sin.to(torch.bfloat16).float()

    def rope(t):
        r, n = t[:, :rotary], t[:, rotary:]
        return torch.cat([r * cos + rotate_half(r) * sin, n], dim=-1)

    q, k = rope(q), rope(k)
    scores = q @ k.t() * scaling
    i, j = torch.arange(tokens)[:, None], torch.arange(tokens)[None, :]
    allowed = j <= i
    if window:
        allowed &= j > i - window
    scores = scores.masked_fill(~allowed, float("-inf"))
    if sink is not None:
        scores = torch.cat([scores, sink.float().expand(tokens, 1)], dim=-1)
    probs = torch.softmax(scores - scores.max(dim=-1, keepdim=True).values, dim=-1)
    if sink is not None:
        probs = probs[:, :-1]
    return probs @ v


def graph_attention(x, qkv_w, *, rope_theta, rotary, scaling, window, sink, vhd):
    """The same head by the graphs' layout: x @ the loader's [H, 3 x 256] (one Q and one KV head), RoPE over the
    halves of the padded head (rope.rotate), attention over 256 channels, the first ``vhd`` channels out."""
    tokens = x.shape[0]
    qkv = x @ qkv_w
    q, k, v = (t.view(tokens, 1, KV_HEAD_DIM) for t in qkv.split(KV_HEAD_DIM, dim=-1))
    cos, sin = RotaryTables(rope_theta, rotary)(torch.arange(tokens))
    q = rotate(q.float(), cos.float(), sin.float())[:, 0]
    k = rotate(k.float(), cos.float(), sin.float())[:, 0]
    scores = q @ k.t() * scaling
    i, j = torch.arange(tokens)[:, None], torch.arange(tokens)[None, :]
    allowed = j <= i
    if window:
        allowed &= j > i - window
    scores = scores.masked_fill(~allowed, float("-inf"))
    if sink is not None:
        scores = torch.cat([scores, sink.float().expand(tokens, 1)], dim=-1)
    probs = torch.softmax(scores, dim=-1)
    if sink is not None:
        probs = probs[:, :-1]
    return (probs @ v[:, 0].float())[:, :vhd]


class Slices:
    """A safetensors file's tensors as the loaders see them (PySafeSlice: get_shape, get_dtype, basic indexing)."""

    def __init__(self, path):
        self._f = safe_open(os.path.join(path, "model.safetensors"), framework="pt")

    def __getitem__(self, name):
        return self._f.get_slice(name)

    def full(self, name):
        return self._f.get_tensor(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slices", required=True, help="the directory fetch_tensors.py wrote")
    args = parser.parse_args()
    with open(os.path.join(args.slices, "config.json")) as f:
        hf = json.load(f)
    cfg = mimo_v2_config(hf)
    s = Slices(args.slices)
    disk_tp = weights.qkv_disk_tp(args.slices)
    check("index metadata tp_size", disk_tp == 4, f"disk_tp={disk_tp}")

    # 0. the layout permutation
    order = weights.qk_order(192, cfg.rotary_dim)
    kept = [c for c in order if c >= 0]
    check("qk_order is a permutation of the 192 channels", sorted(kept) == list(range(192)) and len(order) == 256)
    check("rotary pairs (j, j + 32) land on (j, j + 128)",
          all(order[j] == j and order[j + 128] == j + 32 for j in range(cfg.rotary_dim // 2)))
    x = torch.randn(192, 3)
    laid_out = weights.qk_head_rows(x, 192, cfg.rotary_dim)
    check("unpermute_qk inverts the layout, the padding is zero",
          torch.equal(weights.unpermute_qk(laid_out.t(), 192, cfg.rotary_dim), x.t())
          and not laid_out[[c for c in range(256) if order[c] < 0]].any())

    # 1. the fused QKV loader against the whole-tensor reference
    for layer in (0, 1):
        kind = cfg.kind(layer)
        key = f"model.layers.{layer}.self_attn.qkv_proj"
        qr, kr, vr = ref_fused_qkv(s.full(f"{key}.weight"), s.full(f"{key}.weight_scale_inv"), disk_tp,
                                   kind.num_heads, kind.num_kv_heads, kind.head_dim, kind.v_head_dim)
        for tp in (64, 16, 8):
            loader = weights.fused_qkv_loader(num_heads=kind.num_heads, num_kv_heads=kind.num_kv_heads,
                                              head_dim=kind.head_dim, v_head_dim=kind.v_head_dim,
                                              rotary_dim=cfg.rotary_dim, v_scale=cfg.v_scale, tp=tp, disk_tp=disk_tp)
            qpr, replicas = kind.num_heads // tp, tp // kind.num_kv_heads
            worst = 0.0
            for rank in sorted({0, 1, tp // 2 - 1, tp // 2, tp - 1}):
                got = loader.load([s[f"{key}.weight"], s[f"{key}.weight_scale_inv"]], rank).float()
                kv = rank // replicas
                rows = [weights.qk_head_rows(qr[h], kind.head_dim, cfg.rotary_dim)
                        for h in range(rank * qpr, (rank + 1) * qpr)]
                rows += [weights.qk_head_rows(kr[kv], kind.head_dim, cfg.rotary_dim),
                         weights.v_head_rows(vr[kv], cfg.v_scale)]
                want = torch.cat(rows).t().to(torch.bfloat16).float()
                worst = max(worst, (got - want).abs().max().item())
            check(f"fused QKV loader, layer {layer} ({'sliding' if kind.sliding_window else 'full'}), TP{tp}",
                  worst == 0.0, f"max |diff| {worst:.3g}")
        # 2. the graphs' attention layout against the HF formulas, query head 5 and its KV head at TP64
        loader = weights.fused_qkv_loader(num_heads=kind.num_heads, num_kv_heads=kind.num_kv_heads,
                                          head_dim=kind.head_dim, v_head_dim=kind.v_head_dim,
                                          rotary_dim=cfg.rotary_dim, v_scale=cfg.v_scale, tp=64, disk_tp=disk_tp)
        head = 5
        kv = head // (64 // kind.num_kv_heads)
        qkv_w = loader.load([s[f"{key}.weight"], s[f"{key}.weight_scale_inv"]], head).float()
        sink = None
        if kind.sink:
            sink = weights.sink_loader(1).load([s[f"model.layers.{layer}.self_attn.attention_sink_bias"]], head)
        torch.manual_seed(0)
        x = torch.randn(300, cfg.hidden_size) * 0.05
        bf = lambda t: t.to(torch.bfloat16).float()  # noqa: E731 - the bf16 weights the graphs hold
        want = hf_attention(x, bf(qr[head]), bf(kr[kv]), bf(vr[kv]), rope_theta=kind.rope_theta,
                            rotary=cfg.rotary_dim, scaling=kind.scaling, v_scale=cfg.v_scale,
                            window=kind.sliding_window, sink=sink)
        got = graph_attention(x, qkv_w, rope_theta=kind.rope_theta, rotary=cfg.rotary_dim, scaling=kind.scaling,
                              window=kind.sliding_window, sink=sink, vhd=kind.v_head_dim)
        rel = ((got - want).norm() / want.norm()).item()
        # the bf16 rounding of the RoPE tables and of the V scale folded into the bf16 weight: ~3e-3
        check(f"attention vs HF eager (fp32 tables, V scaled after), layer {layer}", rel < 1e-2,
              f"relative L2 {rel:.2e}")
        # the same rounding on both sides: what remains is the layout, which must be exact up to fp32 sums
        strict = hf_attention(x, bf(qr[head]), bf(kr[kv]), bf(vr[kv] * cfg.v_scale), rope_theta=kind.rope_theta,
                              rotary=cfg.rotary_dim, scaling=kind.scaling, v_scale=1.0, window=kind.sliding_window,
                              sink=sink, bf16_tables=True)
        rel = ((got - strict).norm() / strict.norm()).item()
        check(f"attention layout vs HF eager at matched rounding, layer {layer}", rel < 1e-5, f"relative L2 {rel:.2e}")

    # 3. dequantization
    for proj in ("gate_proj", "down_proj"):
        key = f"model.layers.1.mlp.experts.0.{proj}"
        packed, scale = s.full(f"{key}.weight"), s.full(f"{key}.weight_scale")
        want = ref_mxfp4(packed, scale)
        got = weights.dequant_mxfp4(s[f"{key}.weight"], s[f"{key}.weight_scale"])
        check(f"MXFP4 dequant, expert {proj}", torch.equal(got, want), f"shape {tuple(got.shape)}")
        view = weights.DequantSlice(s[f"{key}.weight"], s[f"{key}.weight_scale"])
        rows, cols = view.get_shape()
        window = view[rows // 4:rows // 2, cols // 2:]
        check(f"MXFP4 bf16 view window, expert {proj}",
              torch.equal(window, want[rows // 4:rows // 2, cols // 2:].to(torch.bfloat16)))
    key = "model.layers.0.mlp.down_proj"
    w, sc = s.full(f"{key}.weight").to(torch.float32), s.full(f"{key}.weight_scale_inv")
    full = w * sc.repeat_interleave(128, 0)[:w.shape[0]].repeat_interleave(128, 1)[:, :w.shape[1]]
    view = weights.DequantSlice(s[f"{key}.weight"], s[f"{key}.weight_scale_inv"])
    rank, cols = 5, cfg.intermediate_size // 64
    got = view[:, rank * cols:(rank + 1) * cols]
    check("FP8 block bf16 view, a TP64 column shard", torch.equal(got, full[:, rank * cols:(rank + 1) * cols]
                                                                  .to(torch.bfloat16)))
    got = view[100:300]
    check("FP8 block bf16 view, an unaligned row window", torch.equal(got, full[100:300].to(torch.bfloat16)))

    print(f"\n{len(FAILURES)} failed" if FAILURES else "\nall passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
