# SPDX-License-Identifier: Apache-2.0
"""Pack Qwen3.5-397B-A17B(-FP8) into the giga kernels' per-rank operands.

    python -m tools.qwen35.build_pack.pack --ckpt <Qwen3.5-397B-A17B-FP8> \
        --out <qwen35-pack> [--layouts decode,prefill] [--workers 96]

Output: one raw (C-contiguous, little-endian) file per operand, memmapped and
written in place by parallel jobs, so the pack streams the checkpoint once and
never holds more than one layer / expert group per worker in RAM:

    meta.json                        config, layouts, {key: [shape, dtype]}, pack version
    g/<name>.bin                     identical on every rank (embedding, norms, router, ...)
    attn<tp>/t<j>/<name>.bin         attention TP shard j (q heads, its KV head, o_proj rows)
    gdn<tp>/t<j>/<name>.bin          GDN TP shard j (conv/recurrent heads, out_proj rows)
    ep<ep>/r<p>/<name>.bin           shared-expert column slice of EP position p (sliced layouts)
    exp/r<q>/<name>.bin              routed experts q*E_G .. q*E_G+E_G-1 (E_G = experts per rank
                                     of the finest EP layout; coarser EP layouts concatenate groups)
    lm/r<r>/w_lm_p.bin               lm_head^T vocab rows of rank r, rows pre-permuted
    .done/<job>                      resumability markers (a rerun skips finished jobs)

Weights stacked over layers carry a leading layer axis per kind: GDN operands
[45, ...], attention operands [15, ...], MoE operands [60, ...] (global layer).
bf16 operands are stored as their 2-byte patterns, MX expert data as uint8 with a
trailing x4 axis (mx.py). Dense weights stay bf16 (FP8 blocks dequantised and
rounded to bf16); routed experts are MXFP8.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from tools.qwen35.build_pack.config import Qwen35Config
from tools.qwen35.build_pack.geometry import RankLayout, layout_by_name, perm_rows
from tools.qwen35.build_pack.mx import expert_shapes

PACK_VERSION = 1
NP_DTYPES = {"bfloat16": np.uint16, "float32": np.float32, "uint8": np.uint8, "int32": np.int32}
ITEMSIZE = {"bfloat16": 2, "float32": 4, "uint8": 1, "int32": 4}


# ----------------------------------------------------------------------------- the operand catalog


def expert_group_size(cfg: Qwen35Config, layouts: Iterable[RankLayout]) -> int:
    """Experts per pack file: those of one rank of the finest EP layout."""
    return min(lay.experts_per_rank(cfg) for lay in layouts)


def catalog(cfg: Qwen35Config, layouts: List[RankLayout]) -> Dict[str, Tuple[tuple, str]]:
    """{key: (shape, dtype)} of every packed operand for these rank layouts."""
    H, d, E = cfg.hidden, cfg.head_dim, cfg.n_experts
    L, La, Lg = cfg.n_layers, len(cfg.attn_layers), len(cfg.gdn_layers)
    out: Dict[str, Tuple[tuple, str]] = {
        "g/embed": ((cfg.vocab, H), "bfloat16"),
        "g/final_ln": ((H,), "float32"),
        "g/in_ln_gdn": ((Lg, H), "float32"),
        "g/in_ln_attn": ((La, H), "float32"),
        "g/post_ln": ((L, H), "float32"),
        "g/q_norm": ((La, 1, d), "float32"),
        "g/k_norm": ((La, 1, d), "float32"),
        "g/gdn_norm": ((Lg, cfg.gdn_v_dim), "float32"),
        "g/w_router_p": ((L, 128, H // 128, E), "bfloat16"),
    }
    world = layouts[0].world
    for lay in layouts:
        if lay.world != world:
            raise ValueError("all layouts must span the same world")
        nq = cfg.n_q_heads // lay.attn_tp
        for j in range(lay.attn_tp):
            p = f"attn{lay.attn_tp}/t{j}/"
            out[p + "w_qkv"] = ((La, H, (nq + 2) * d), "bfloat16")
            out[p + "w_gate_p"] = ((La, 128, H // 128, nq * d), "bfloat16")
            out[p + "w_o"] = ((La, nq * d, H), "bfloat16")
            if lay.fold:
                out[p + "w_o_ext"] = ((La, nq * d, H + E + 1), "bfloat16")
                out[p + "wx_a"] = ((La, H // lay.attn_tp, E + 1), "bfloat16")
        vh, kh = lay.gdn_vh(cfg), lay.gdn_kh(cfg)
        Kd, Vd = kh * cfg.gdn_k_dim, vh * cfg.gdn_v_dim
        Cc = lay.gdn_conv_dim(cfg)
        for j in range(lay.gdn_tp):
            p = f"gdn{lay.gdn_tp}/t{j}/"
            out[p + "w_proj_p"] = ((Lg, 128, H // 128, 2 * Kd + 2 * Vd + 2 * vh), "bfloat16")
            out[p + "conv_w"] = ((Lg, Cc, cfg.conv_kernel), "bfloat16")
            out[p + "a_log"] = ((Lg, vh), "float32")
            out[p + "dt_bias"] = ((Lg, vh), "float32")
            out[p + "w_out"] = ((Lg, Vd, H), "bfloat16")
            if lay.fold:
                out[p + "w_out_ext"] = ((Lg, Vd, H + E + 1), "bfloat16")
                out[p + "wx"] = ((Lg, H // lay.gdn_tp, E + 1), "bfloat16")
        if lay.shared == "slice":
            Is = cfg.shared_inter // lay.ep
            for q in range(lay.ep):
                p = f"ep{lay.ep}/r{q}/"
                out[p + "w_shared_p"] = ((L, 128, H // 128, 2 * Is + 1), "bfloat16")
                out[p + "w_sd"] = ((L, Is, H), "bfloat16")
        else:
            Is = cfg.shared_inter
            out["g/w_shared_full_p"] = ((L, 128, H // 128, 2 * Is + 1), "bfloat16")
            out["g/w_sd_full"] = ((L, Is, H), "bfloat16")
    eg = expert_group_size(cfg, layouts)
    for q in range(E // eg):
        for name, shape in expert_shapes(H, cfg.moe_inter, eg, L).items():
            out[f"exp/r{q}/{name}"] = (shape, "uint8")
    Vr = cfg.vocab // world
    for r in range(world):
        out[f"lm/r{r}/w_lm_p"] = ((128, H // 128, Vr), "bfloat16")
    return out


def nbytes(shape: tuple, dtype: str) -> int:
    return int(np.prod(shape, dtype=np.int64)) * ITEMSIZE[dtype]


# ----------------------------------------------------------------------------- bf16 helpers


def bf16_bits(t) -> np.ndarray:
    """torch tensor / numpy array -> uint16 bf16 bit patterns (round to nearest even)."""
    import torch

    t = torch.as_tensor(np.asarray(t)) if not isinstance(t, torch.Tensor) else t
    return t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def bf16_from_bits(a) -> np.ndarray:
    import torch

    return torch.from_numpy(np.array(a, dtype=np.uint16).view(np.int16)).view(torch.bfloat16).float().numpy()


def _bfr(x):
    """Round fp32 (torch) to bf16 values, keep fp32."""
    import torch

    return x.to(torch.bfloat16).float()


# ----------------------------------------------------------------------------- per-shard math (pure, CPU-tested)


def attn_shard(cfg: Qwen35Config, lay: RankLayout, j: int, q_proj, k_proj, v_proj, o_proj) -> dict:
    """Attention operands of TP position j from HF weights (torch fp32, bf16 values):
    q_proj [n_q*2d, H] (per head: q rows then gate rows), k/v_proj [n_kv*d, H],
    o_proj [H, n_q*d]."""
    import torch

    d = cfg.head_dim
    heads = lay.q_heads(cfg, j)
    kvh = lay.kv_head(cfg, j)
    q = torch.cat([q_proj[h * 2 * d : h * 2 * d + d] for h in heads])  # [nq*d, H]
    gate = torch.cat([q_proj[h * 2 * d + d : (h + 1) * 2 * d] for h in heads])
    k = k_proj[kvh * d : (kvh + 1) * d]
    v = v_proj[kvh * d : (kvh + 1) * d]
    cols = torch.cat([torch.arange(h * d, (h + 1) * d) for h in heads])
    return {
        "w_qkv": torch.cat([q, k, v]).T.contiguous(),  # [H, (nq+2)d]: q heads | k | v
        "w_gate": gate.T.contiguous(),  # [H, nq*d]
        "w_o": o_proj[:, cols].T.contiguous(),  # [nq*d, H]
    }


def gdn_shard(cfg: Qwen35Config, lay: RankLayout, j: int, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
              conv1d, a_log, dt_bias, out_proj) -> dict:
    """GDN operands of TP position j (natural layouts; ``w_proj`` before row permutation):
    in_proj_qkv [2K+V, H], in_proj_z [V, H], in_proj_a/b [v_heads, H], conv1d [2K+V, 1, k],
    A_log / dt_bias [v_heads], out_proj [H, V]."""
    import torch

    ch = torch.as_tensor(lay.gdn_channels(cfg, j))
    vs = lay.gdn_v_heads(cfg, j)
    dv = cfg.gdn_v_dim
    vcols = torch.arange(vs.start * dv, vs.stop * dv)
    w_qkvz = torch.cat([in_proj_qkv[ch], in_proj_z[vcols]]).T  # [H, 2Kd + 2Vd]: q | k | v | z
    w_a = in_proj_a[vs.start : vs.stop].T
    w_b = in_proj_b[vs.start : vs.stop].T
    return {
        "w_proj": torch.cat([w_qkvz, w_a, w_b], dim=1).contiguous(),  # [H, 2Kd+2Vd+2vh]
        "conv_w": conv1d.reshape(conv1d.shape[0], -1)[ch].contiguous(),  # [Cc, k]
        "a_log": a_log[vs.start : vs.stop].float(),
        "dt_bias": dt_bias[vs.start : vs.stop].float(),
        "w_out": out_proj[:, vcols].T.contiguous(),  # [Vd, H]
    }


def router_fold(w_plain, router_t, w_sg, post_ln):
    """Router fold (lowc_layer_torch.pack_router_fold, scale 1): [w | (w diag(1+post)) [W_r | w_sg]].
    w_plain [rows, H], router_t [H, E], w_sg [H, 1], post_ln [H] -> fp32 [rows, H+E+1] (bf16 values
    after the caller's rounding)."""
    import torch

    W = torch.cat([router_t, w_sg], dim=1)
    M = (w_plain * (1.0 + post_ln)[None, :]) @ W
    return torch.cat([w_plain, M], dim=1)


def x_fold(router_t, w_sg, post_ln, pos: int, n: int):
    """lowc_layer_torch.pack_x_fold: (1 + post)[rows] * [W_r | w_sg][rows], rows = H/n slice of pos."""
    import torch

    H = router_t.shape[0]
    xr = H // n
    rows = slice(xr * pos, xr * (pos + 1))
    W = torch.cat([router_t, w_sg], dim=1)
    return (1.0 + post_ln)[rows, None] * W[rows]


def shared_slice(gate, up, down, w_sg, cols) -> dict:
    """Shared-expert operands for intermediate columns ``cols``: gate/up [Is, H], down [H, Is],
    w_sg [1, H] (shared_expert_gate) -> w_shared [H, 2n+1] (gate | up | gate column), w_sd [n, H]."""
    import torch

    c = torch.as_tensor(list(cols))
    return {
        "w_shared": torch.cat([gate[c].T, up[c].T, w_sg.reshape(1, -1).T], dim=1).contiguous(),
        "w_sd": down[:, c].T.contiguous(),
    }


# ----------------------------------------------------------------------------- jobs

_S: dict = {}


def _init(ckpt_dir: str, out: str, threads: int = 4):
    import torch

    from tools.qwen35.build_pack.ckpt import Checkpoint

    torch.set_num_threads(threads)
    meta = json.load(open(os.path.join(out, "meta.json")))
    _S.update(
        ck=Checkpoint(ckpt_dir), out=out, meta=meta,
        cfg=Qwen35Config.from_saved(meta["config"]),
        layouts=[RankLayout(**d) for d in meta["layouts"]],
    )


def _mm(key: str):
    shape, dt = _S["meta"]["operands"][key]
    return np.memmap(os.path.join(_S["out"], key + ".bin"), dtype=NP_DTYPES[dt], mode="r+", shape=tuple(shape))


def _put(key: str, index, value) -> None:
    m = _mm(key)
    dt = _S["meta"]["operands"][key][1]
    if dt == "bfloat16":
        value = bf16_bits(value)
    else:
        value = np.asarray(value.numpy() if hasattr(value, "numpy") else value, dtype=NP_DTYPES[dt])
    if index is None:
        m[...] = value.reshape(m.shape)
    else:
        m[index] = value.reshape(m[index].shape)
    m.flush()
    del m


def job_embed() -> str:
    ck = _S["ck"]
    emb = ck.get(ck.text("embed_tokens.weight"))
    _put("g/embed", None, emb)
    return "embed"


def job_head() -> str:
    ck, cfg, lay = _S["ck"], _S["cfg"], _S["layouts"][0]
    lm = ck.get(ck.lm_head_name())  # [V, H]
    for r in range(lay.world):
        rows = lay.vocab_rows(cfg, r)
        _put(f"lm/r{r}/w_lm_p", None, perm_rows(lm[rows.start : rows.stop].float().T.contiguous().numpy()))
    _put("g/final_ln", None, ck.getf(ck.text("norm.weight")))
    return "head"


def job_layer(li: int) -> str:
    """Everything of layer li except the routed experts."""
    ck, cfg = _S["ck"], _S["cfg"]
    is_attn, ki = cfg.kind_index(li)
    g = lambda tail: ck.getf(ck.layer(li, tail))  # noqa: E731
    in_ln, post_ln = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
    _put("g/post_ln", li, post_ln)
    router = g("mlp.gate.weight")  # [E, H]
    router_t = router.T.contiguous()
    _put("g/w_router_p", li, perm_rows(router_t.numpy()))
    sg = g("mlp.shared_expert_gate.weight")  # [1, H]
    w_sg = sg.reshape(1, -1).T.contiguous()  # [H, 1]
    sh_g, sh_u, sh_d = (g(f"mlp.shared_expert.{n}.weight") for n in ("gate_proj", "up_proj", "down_proj"))
    done_attn, done_gdn, done_ep = set(), set(), set()
    if is_attn:
        _put("g/in_ln_attn", ki, in_ln)
        _put("g/q_norm", ki, g("self_attn.q_norm.weight").reshape(1, -1))
        _put("g/k_norm", ki, g("self_attn.k_norm.weight").reshape(1, -1))
        qp, kp, vp, op = (g(f"self_attn.{n}.weight") for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
    else:
        _put("g/in_ln_gdn", ki, in_ln)
        la = lambda t: g("linear_attn." + t)  # noqa: E731
        qkv, z, a, b = la("in_proj_qkv.weight"), la("in_proj_z.weight"), la("in_proj_a.weight"), la("in_proj_b.weight")
        conv, A_log, dt_b, out_p = la("conv1d.weight"), la("A_log"), la("dt_bias"), la("out_proj.weight")
        _put("g/gdn_norm", ki, la("norm.weight"))
    for lay in _S["layouts"]:
        if is_attn and (lay.attn_tp, lay.fold) not in done_attn:
            done_attn.add((lay.attn_tp, lay.fold))
            for j in range(lay.attn_tp):
                s = attn_shard(cfg, lay, j, qp, kp, vp, op)
                p = f"attn{lay.attn_tp}/t{j}/"
                _put(p + "w_qkv", ki, s["w_qkv"])
                _put(p + "w_gate_p", ki, perm_rows(s["w_gate"].numpy()))
                _put(p + "w_o", ki, s["w_o"])
                if lay.fold:
                    w_o = _bfr(s["w_o"])
                    _put(p + "w_o_ext", ki, router_fold(w_o, router_t, w_sg, post_ln))
                    _put(p + "wx_a", ki, x_fold(router_t, w_sg, post_ln, j, lay.attn_tp))
        if not is_attn and lay.gdn_tp not in done_gdn:
            done_gdn.add(lay.gdn_tp)
            for j in range(lay.gdn_tp):
                s = gdn_shard(cfg, lay, j, qkv, z, a, b, conv, A_log, dt_b, out_p)
                p = f"gdn{lay.gdn_tp}/t{j}/"
                _put(p + "w_proj_p", ki, perm_rows(s["w_proj"].numpy()))
                _put(p + "conv_w", ki, s["conv_w"])
                _put(p + "a_log", ki, s["a_log"])
                _put(p + "dt_bias", ki, s["dt_bias"])
                _put(p + "w_out", ki, s["w_out"])
                if lay.fold:
                    w_out = _bfr(s["w_out"])
                    _put(p + "w_out_ext", ki, router_fold(w_out, router_t, w_sg, post_ln))
                    _put(p + "wx", ki, x_fold(router_t, w_sg, post_ln, j, lay.gdn_tp))
        if lay.shared == "slice" and lay.ep not in done_ep:
            done_ep.add(lay.ep)
            for q in range(lay.ep):
                s = shared_slice(sh_g, sh_u, sh_d, sg, lay.shared_cols(cfg, q))
                _put(f"ep{lay.ep}/r{q}/w_shared_p", li, perm_rows(s["w_shared"].numpy()))
                _put(f"ep{lay.ep}/r{q}/w_sd", li, s["w_sd"])
        elif lay.shared == "replicated" and "full" not in done_ep:
            done_ep.add("full")
            s = shared_slice(sh_g, sh_u, sh_d, sg, range(cfg.shared_inter))
            _put("g/w_shared_full_p", li, perm_rows(s["w_shared"].numpy()))
            _put("g/w_sd_full", li, s["w_sd"])
    return f"layer{li}"


def job_experts(li: int, q: int) -> str:
    """Routed experts q*E_G .. q*E_G+E_G-1 of layer li -> MXFP8 (mx.py)."""
    from tools.qwen35.build_pack.mx import pack_expert

    ck, cfg = _S["ck"], _S["cfg"]
    eg = expert_group_size(cfg, _S["layouts"])
    p = f"exp/r{q}/"
    mms = {n: _mm(p + n) for n in ("gate_up_w", "gate_up_scale", "down_w", "down_scale")}
    for j in range(eg):
        guw, gus, dw, ds = pack_expert(*ck.expert(li, q * eg + j))
        mms["gate_up_w"][li, j] = guw
        mms["gate_up_scale"][li, j] = gus
        mms["down_w"][li, j] = dw
        mms["down_scale"][li, j] = ds
    for m in mms.values():
        m.flush()
    return f"exp{li}_{q}"


def _run(job: tuple) -> Tuple[str, float]:
    t0 = time.time()
    kind, *args = job
    name = {"embed": job_embed, "head": job_head, "layer": job_layer, "experts": job_experts}[kind](*args)
    open(os.path.join(_S["out"], ".done", name), "w").close()
    return name, time.time() - t0


def job_name(job: tuple) -> str:
    kind, *args = job
    return {"embed": "embed", "head": "head", "layer": "layer{}", "experts": "exp{}_{}"}[kind].format(*args)


def all_jobs(cfg: Qwen35Config, layouts: List[RankLayout], layers: Optional[Iterable[int]] = None,
             only: Optional[set] = None) -> List[tuple]:
    layers = list(range(cfg.n_layers)) if layers is None else list(layers)
    eg = expert_group_size(cfg, layouts)
    jobs = []
    if not only or "embed" in only:
        jobs.append(("embed",))
    if not only or "head" in only:
        jobs.append(("head",))
    if not only or "layers" in only:
        jobs += [("layer", li) for li in layers]
    if not only or "experts" in only:
        jobs += [("experts", li, q) for li in layers for q in range(cfg.n_experts // eg)]
    return jobs


# ----------------------------------------------------------------------------- driver


def run_jobs(ckpt: str, out: str, jobs: List[tuple], workers: int = 0, threads: int = 2,
             log=lambda s: None) -> List[str]:
    """Run pack jobs (skipping finished ones): in a process pool, or in-process with workers=0."""
    jobs = [j for j in jobs if not os.path.exists(os.path.join(out, ".done", job_name(j)))]
    done, t0 = [], time.time()
    if workers <= 0:
        _init(ckpt, out, threads)
        for i, j in enumerate(jobs):
            name, dt = _run(j)
            done.append(name)
            log(f"[{time.time() - t0:7.0f}s] {i + 1}/{len(jobs)} {name} ({dt:.1f}s)")
        return done
    # layer jobs (big dense reads + fold matmuls) are listed first, then the expert jobs
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(ckpt, out, threads)) as ex:
        for i, (name, dt) in enumerate(ex.map(_run, jobs, chunksize=1)):
            done.append(name)
            if i % 64 == 0 or not name.startswith("exp"):
                log(f"[{time.time() - t0:7.0f}s] {i + 1}/{len(jobs)} {name} ({dt:.1f}s)")
    return done


def prepare(out: str, cfg: Qwen35Config, layouts: List[RankLayout], force: bool = False) -> dict:
    """Write meta.json and create (sparse) operand files; refuse a mismatching existing pack."""
    cat = catalog(cfg, layouts)
    meta = {
        "version": PACK_VERSION,
        "config": cfg.to_dict(),
        "layouts": [lay.to_dict() for lay in layouts],
        "expert_group": expert_group_size(cfg, layouts),
        "operands": {k: [list(s), dt] for k, (s, dt) in cat.items()},
    }
    path = os.path.join(out, "meta.json")
    if os.path.exists(path) and not force:
        old = json.load(open(path))
        if old != meta:
            raise RuntimeError(f"{path} describes a different pack (config / layouts); use --force or a new --out")
    os.makedirs(os.path.join(out, ".done"), exist_ok=True)
    for key, (shape, dt) in cat.items():
        fn = os.path.join(out, key + ".bin")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        size = nbytes(shape, dt)
        if not os.path.exists(fn) or os.path.getsize(fn) != size:
            with open(fn, "wb") as f:
                f.truncate(size)
    with open(path, "w") as f:
        json.dump(meta, f)
    return meta


def load_meta(pack: str) -> dict:
    return json.load(open(os.path.join(pack, "meta.json")))


def memmap(pack: str, key: str, meta: Optional[dict] = None, mode: str = "r") -> np.memmap:
    meta = meta or load_meta(pack)
    shape, dt = meta["operands"][key]
    return np.memmap(os.path.join(pack, key + ".bin"), dtype=NP_DTYPES[dt], mode=mode, shape=tuple(shape))


def verify(pack: str, ckpt_dir: str, samples: int = 4, seed: int = 0) -> List[str]:
    """Spot checks against the checkpoint: experts dequantised back (rel L2), router / attention /
    GDN shards bit-exact. Returns report lines; raises on a failure."""
    from tools.qwen35.build_pack.ckpt import Checkpoint
    from tools.qwen35.build_pack.mx import unpack_mx_fp8

    meta = load_meta(pack)
    cfg = Qwen35Config.from_saved(meta["config"])
    ck = Checkpoint(ckpt_dir)
    rng = np.random.default_rng(seed)
    eg = meta["expert_group"]
    lines = []
    for _ in range(samples):
        li, e = int(rng.integers(cfg.n_layers)), int(rng.integers(cfg.n_experts))
        q, j = divmod(e, eg)
        gate, up, down = ck.expert(li, e)
        gu = memmap(pack, f"exp/r{q}/gate_up_w", meta)[li, j]
        gs = memmap(pack, f"exp/r{q}/gate_up_scale", meta)[li, j]
        dq = unpack_mx_fp8(memmap(pack, f"exp/r{q}/down_w", meta)[li, j], memmap(pack, f"exp/r{q}/down_scale", meta)[li, j], "i")
        gq = unpack_mx_fp8(gu[:, 0], gs[:, 0], "h")
        for name, got, want in (("gate", gq, gate.T.numpy()), ("down", dq, down.T.numpy())):
            rel = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30))
            lines.append(f"expert L{li} e{e} {name}: MX rel L2 {rel:.4f}")
            if not rel <= 0.06:  # also fails on NaN
                raise AssertionError(lines[-1] + " (expected ~0.02-0.03 for MXFP8)")
        r = memmap(pack, "g/w_router_p", meta)[li]
        want = bf16_bits(perm_rows(ck.getf(ck.layer(li, "mlp.gate.weight")).T.contiguous().numpy()))
        if not np.array_equal(np.asarray(r), want):
            raise AssertionError(f"router L{li} differs from the checkpoint")
        lines.append(f"router L{li}: exact")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layouts", default="decode,prefill")
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 8))
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    ap.add_argument("--only", default="", help="subset of embed,head,layers,experts")
    ap.add_argument("--layers", default="", help="e.g. 0-3,59 (default all)")
    ap.add_argument("--force", action="store_true", help="overwrite a pack with different meta")
    ap.add_argument("--dry-run", action="store_true", help="print the catalog sizes and exit")
    ap.add_argument("--verify", type=int, default=0, help="spot-check N samples after packing")
    args = ap.parse_args(argv)

    cfg = Qwen35Config.from_checkpoint(args.ckpt)
    cfg.check_kernel_pattern()
    layouts = [layout_by_name(n) for n in args.layouts.split(",") if n]
    cat = catalog(cfg, layouts)
    total = sum(nbytes(s, d) for s, d in cat.values())
    if args.dry_run:
        groups: Dict[str, int] = {}
        for k, (s, d) in cat.items():
            groups[k.split("/")[0]] = groups.get(k.split("/")[0], 0) + nbytes(s, d)
        for gname, n in sorted(groups.items()):
            print(f"{gname:8s} {n / 2**30:9.1f} GiB")
        print(f"total    {total / 2**30:9.1f} GiB in {len(cat)} files")
        return 0
    os.makedirs(args.out, exist_ok=True)
    prepare(args.out, cfg, layouts, force=args.force)
    layers = None
    if args.layers:
        layers = []
        for part in args.layers.split(","):
            a, _, b = part.partition("-")
            layers += list(range(int(a), int(b or a) + 1))
    only = set(filter(None, args.only.split(","))) or None
    jobs = all_jobs(cfg, layouts, layers, only)
    print(f"pack {args.out}: {total / 2**30:.1f} GiB, {len(jobs)} jobs, {args.workers} workers", flush=True)
    t0 = time.time()
    done = run_jobs(args.ckpt, args.out, jobs, args.workers, args.threads, log=lambda s: print(s, flush=True))
    print(f"done: {len(done)} jobs run ({len(jobs) - len(done)} already finished) in {time.time() - t0:.0f}s",
          flush=True)
    if args.verify:
        for line in verify(args.out, args.ckpt, args.verify):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
