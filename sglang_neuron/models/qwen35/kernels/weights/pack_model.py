"""Pack the Qwen3.5-397B-A17B HF FP8 checkpoint into the per-rank stacked operands of
``qwen35_hybrid_multistep(..., per_layer_weights=True)``.
Pure numpy / torch / safetensors / ml_dtypes: no nki, runs on any CPU host.

    python -m sglang_neuron.models.qwen35.kernels.weights.pack_model <ckpt_dir> <out_dir> [--layers N] [--ranks 128]
        [--workers K] [--tp 64] [--atp 8] [--rope-len 32768] [--mx-scale ocp|nkilib]
    python -m sglang_neuron.models.qwen35.kernels.weights.pack_model <ckpt_dir> --print-layout [...]  (config.json only)

Decode layout (128 ranks, high C): GDN TP``tp`` (tp position = rank % tp, ranks/tp replicas), gated attention
TP``atp`` x DP (position = rank % atp, one KV head per rank), MoE EP = all ranks (rank r owns global experts
r*E_L .. r*E_L + E_L - 1 in every layer, so ``sel`` is the same per layer), lm_head vocab slice r*Vs .. per rank,
embedding replicated, router fold on (w_out_ext / wx and w_o_ext / wx_a packed).

Output (raw little-endian .bin files, np.memmap-able; ``load_rank`` assembles one rank's kernel inputs):

    g/<name>.bin        identical on every rank: embed_table, final_ln_w, cos_tab, sin_tab, in_ln_w, post_ln_w,
                        w_router_p, norm_w (GDN gated-norm weight, same for every head), q_norm_w, k_norm_w
    gdn<t>/<name>.bin   GDN TP position t: w_proj_p, conv_w, a_log, dt_bias, w_out, w_out_ext, wx, tp_rank
    attn<t>/<name>.bin  attention TP position t: w_qkv, w_gate_p, w_o, w_o_ext, wx_a, atp_rank
    r<r>/<name>.bin     rank r: sel, w_shared_p, w_sd, gate_up_w, gate_up_scale, down_w, down_scale, w_lm_p,
                        vbase, ep_rank
    meta.json           {group/name: [shape, dtype]}  (dtype "float8_e4m3fn_x4" = 4 packed e4m3 bytes, byte j =
                        element j, memmapped as uint32 unless nki's x4 dtype is available)
    layout.json         tp / atp / ranks / layers / L_g / L_a / E_L / mx_scale / config dims

Stacked axes (multistep.py docstring): GDN index g for w_proj_p, conv_w, a_log, dt_bias, norm_w, w_out, w_out_ext,
wx; attention index a for w_qkv, q_norm_w, k_norm_w, w_gate_p, w_o, w_o_ext, wx_a; layer index i for in_ln_w,
post_ln_w, w_router_p, w_shared_p, w_sd and the MX experts. Weights are the FP8 checkpoint dequantized by
``Ckpt.get`` (fp8 * 128x128 block scale_inv in fp32, rounded to bf16), the rule of the fork's accuracy reference;
dense operands are those bf16 values, MX experts are MXFP8-quantized from them.

MX scale rule (``--mx-scale``): "ocp" (default) = the fork's dense MX packer (scale exponent
floor(log2 amax) - 8, values clipped at 448; the rule of the accuracy measurements);
"nkilib" = nki-library mx_torch_common ``quantize_to_mx`` (exponent field of amax - 7, no clipping in practice).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import ml_dtypes
import numpy as np
import torch

BF16 = np.dtype(ml_dtypes.bfloat16)
F32 = np.dtype(np.float32)
I32 = np.dtype(np.int32)
U8 = np.dtype(np.uint8)
X4 = "float8_e4m3fn_x4"  # meta dtype name of the packed MX data (4 bytes per element of the kernel shape)
P = 128
Q = 4


# ----------------------------------------------------------------------------- config / layout


class Cfg:
    """Model dims from the checkpoint's config.json (text_config when present)."""

    def __init__(self, d):
        t = d.get("text_config", d)
        self.H = t["hidden_size"]
        self.n_layers = t["num_hidden_layers"]
        self.layer_types = t.get("layer_types") or [
            "full_attention" if i % t.get("full_attention_interval", 4) == t.get("full_attention_interval", 4) - 1
            else "linear_attention" for i in range(self.n_layers)
        ]
        self.nk, self.nv = t["linear_num_key_heads"], t["linear_num_value_heads"]
        self.dk, self.dv = t["linear_key_head_dim"], t["linear_value_head_dim"]
        self.kconv = t["linear_conv_kernel_dim"]
        self.nq, self.nkv, self.d = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        self.E, self.I, self.Is = t["num_experts"], t["moe_intermediate_size"], t["shared_expert_intermediate_size"]
        self.V = t["vocab_size"]
        rp = t.get("rope_parameters") or {}
        self.theta = float(rp.get("rope_theta", t.get("rope_theta", 1e7)))
        self.rot = int(self.d * float(rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25))))

    def as_dict(self):
        return {k: v for k, v in vars(self).items() if k != "layer_types"}


class Layout:
    """Parallel layout + every packed operand's (shape, dtype)."""

    def __init__(self, cfg, ranks=128, tp=None, atp=None, layers=None, rope_len=32768, mx_scale="ocp"):
        c = self.cfg = cfg
        self.ranks, self.tp, self.atp = ranks, tp or min(64, ranks), atp or min(8, ranks)
        self.L = layers or c.n_layers
        self.rope_len, self.mx_scale = rope_len, mx_scale
        tp, atp, H = self.tp, self.atp, c.H
        assert self.L >= 4, "need >= 4 layers (the kernel's 3 GDN + 1 attention period)"
        assert self.L <= c.n_layers
        for i in range(self.L):  # the kernel hard-codes the hybrid pattern
            want = "full_attention" if i % 4 == 3 else "linear_attention"
            assert c.layer_types[i] == want, f"layer {i} is {c.layer_types[i]}, kernel expects {want}"
        self.gdn_layers = [i for i in range(self.L) if i % 4 != 3]
        self.attn_layers = [i for i in range(self.L) if i % 4 == 3]
        self.L_g, self.L_a = len(self.gdn_layers), len(self.attn_layers)
        assert ranks % tp == 0 and ranks % atp == 0
        assert H % 512 == 0 and H % tp == 0 and H % atp == 0
        assert c.dk == c.dv == P, "GDN heads are 128 wide in the decode kernel"
        assert c.nv % tp == 0 and c.nv % c.nk == 0
        self.VH = c.nv // tp
        self.r = c.nv // c.nk  # v heads per k head (GQA)
        assert self.VH % self.r == 0 or self.r % self.VH == 0
        self.KH = max(1, self.VH // self.r)
        assert c.nq % atp == 0 and atp % c.nkv == 0, "attention: one KV head per rank"
        self.nq_l = c.nq // atp
        assert c.E % ranks == 0 and c.Is % ranks == 0 and c.V % ranks == 0 and c.I % 512 == 0
        self.E_L, self.Is_r, self.Vs = c.E // ranks, c.Is // ranks, c.V // ranks

    # head ownership (natural HF order) -------------------------------------------------------

    def gdn_heads(self, t):
        """(k heads, v heads) of GDN TP position t: v heads t*VH.., their GQA k heads."""
        v0 = t * self.VH
        return list(range(v0 // self.r, v0 // self.r + self.KH)), list(range(v0, v0 + self.VH))

    def attn_heads(self, t):
        """(q heads, kv head) of attention TP position t."""
        q = list(range(t * self.nq_l, (t + 1) * self.nq_l))
        return q, q[0] // (self.cfg.nq // self.cfg.nkv)

    def groups(self, rank):
        return ["g", f"gdn{rank % self.tp}", f"attn{rank % self.atp}", f"r{rank}"]

    def operands(self):
        """{group/name: (shape tuple, dtype)}; dtype X4 = the kernel's float8_e4m3fn_x4 (4 bytes each)."""
        c, H, L, L_g, L_a = self.cfg, self.cfg.H, self.L, self.L_g, self.L_a
        Kd, Vd = self.KH * c.dk, self.VH * c.dv
        Cc = 2 * Kd + Vd
        nqd = self.nq_l * c.d
        E, E_L, I, Is = c.E, self.E_L, c.I, self.Is_r
        g = {
            "embed_table": ((c.V, H), BF16), "final_ln_w": ((H,), F32),
            "cos_tab": ((self.rope_len, c.rot // 2), F32), "sin_tab": ((self.rope_len, c.rot // 2), F32),
            "in_ln_w": ((L, H), F32), "post_ln_w": ((L, H), F32), "w_router_p": ((L, P, H // P, E), BF16),
            "norm_w": ((L_g, c.dv), F32), "q_norm_w": ((L_a, 1, c.d), F32), "k_norm_w": ((L_a, 1, c.d), F32),
        }
        gd = {
            "w_proj_p": ((L_g, P, H // P, 2 * Kd + 2 * Vd + 2 * self.VH), BF16), "conv_w": ((L_g, Cc, c.kconv), BF16),
            "a_log": ((L_g, self.VH), F32), "dt_bias": ((L_g, self.VH), F32), "w_out": ((L_g, Vd, H), BF16),
            "w_out_ext": ((L_g, Vd, H + E + 1), BF16), "wx": ((L_g, H // self.tp, E + 1), BF16),
            "tp_rank": ((1, 1), I32),
        }
        at = {
            "w_qkv": ((L_a, H, nqd + 2 * c.d), BF16), "w_gate_p": ((L_a, P, H // P, nqd), BF16),
            "w_o": ((L_a, nqd, H), BF16), "w_o_ext": ((L_a, nqd, H + E + 1), BF16),
            "wx_a": ((L_a, H // self.atp, E + 1), BF16), "atp_rank": ((1, 1), I32),
        }
        r = {
            "sel": ((E, E_L), F32), "w_shared_p": ((L, P, H // P, 2 * Is + 1), BF16), "w_sd": ((L, Is, H), BF16),
            "gate_up_w": ((L, E_L, P, 2, H // 512, I), X4), "gate_up_scale": ((L, E_L, 16, 2, H // 512, I), U8),
            "down_w": ((L, E_L, P, I // 512, H), X4), "down_scale": ((L, E_L, 16, I // 512, H), U8),
            "w_lm_p": ((P, H // P, self.Vs), BF16), "vbase": ((1, 1), I32), "ep_rank": ((1, 1), I32),
        }
        out = {f"g/{k}": v for k, v in g.items()}
        out.update({f"gdn{t}/{k}": v for t in range(self.tp) for k, v in gd.items()})
        out.update({f"attn{t}/{k}": v for t in range(self.atp) for k, v in at.items()})
        out.update({f"r{i}/{k}": v for i in range(self.ranks) for k, v in r.items()})
        return out

    def info(self):
        return {
            "ranks": self.ranks, "tp": self.tp, "atp": self.atp, "ep": self.ranks, "layers": self.L,
            "L_g": self.L_g, "L_a": self.L_a, "gdn_layers": self.gdn_layers, "attn_layers": self.attn_layers,
            "E_L": self.E_L, "Is_rank": self.Is_r, "Vs": self.Vs, "rope_len": self.rope_len,
            "mx_scale": self.mx_scale, "config": self.cfg.as_dict(),
            "groups": "rank r reads g/, gdn<r % tp>/, attn<r % atp>/, r<r>/",
        }


def _nbytes(shape, dt):
    return int(np.prod(shape)) * (4 if dt == X4 else np.dtype(dt).itemsize)


def _dtname(dt):
    return X4 if dt == X4 else ("bfloat16" if dt == BF16 else np.dtype(dt).name)


def _np_dtype(name, x4_dtype=None):
    if name == X4:
        return x4_dtype if x4_dtype is not None else np.dtype(np.uint32)
    return BF16 if name == "bfloat16" else np.dtype(name)


# ----------------------------------------------------------------------------- checkpoint


class Ckpt:
    """HF safetensors checkpoint; ``get`` dequantizes FP8 -> bf16 with the 128x128 scale_inv (the fork's rule)."""

    def __init__(self, d):
        from safetensors import safe_open

        self._open = safe_open
        self.d = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self._files = {}
        self.pre = "model.language_model." if "model.language_model.embed_tokens.weight" in self.map else "model."

    def _f(self, name):
        fn = self.map[name]
        if fn not in self._files:
            self._files[fn] = self._open(os.path.join(self.d, fn), framework="pt")
        return self._files[fn]

    def get(self, name):
        t = self._f(name).get_tensor(name)
        if t.dtype == torch.float8_e4m3fn:
            s = self._f(name + "_scale_inv").get_tensor(name + "_scale_inv").float()
            o, i = t.shape
            s = s.repeat_interleave(128, 0)[:o].repeat_interleave(128, 1)[:, :i]
            t = (t.float() * s).to(torch.bfloat16)
        return t

    def f32(self, name):
        """Tensor as fp32 numpy (bf16 values are exact in fp32)."""
        return self.get(name).float().numpy()


def _bf(a):
    return np.asarray(a, dtype=np.float32).astype(BF16)


def perm_rows(w):
    """[H, C] -> [128, H/128, C] with w_p[p, h1, c] = w[h1*128 + p, c] (kernel row layout)."""
    H = w.shape[0]
    return np.ascontiguousarray(np.asarray(w).reshape(H // P, P, w.shape[1]).transpose(1, 0, 2))


def pack_router_fold(w_out, wr, w_sg, post_ln_w):
    """[W_out | W_out diag(1+w) [W_r | w_sg]] bf16 [rows, H + E + 1] (as the fork's low-concurrency torch reference)."""
    wo = np.asarray(w_out, dtype=np.float32)
    W = np.concatenate([np.asarray(wr, np.float32), np.asarray(w_sg, np.float32)], axis=1)
    M = (wo * (1.0 + np.asarray(post_ln_w, np.float32))[None, :]) @ W
    return np.concatenate([wo, M], axis=1).astype(BF16)


def pack_x_fold(wr, w_sg, post_ln_w, pos, n):
    """Rows pos*H/n .. of (1 + w) * [W_r | w_sg], bf16 [H/n, E + 1] (as the fork's low-concurrency torch reference)."""
    XR = wr.shape[0] // n
    rows = slice(XR * pos, XR * (pos + 1))
    W = np.concatenate([np.asarray(wr, np.float32), np.asarray(w_sg, np.float32)], axis=1)
    return ((1.0 + np.asarray(post_ln_w, np.float32))[rows, None] * W[rows]).astype(BF16)


def mx_pack(w, kind, mode="ocp"):
    """W [K, N] (fp32 values) -> (data uint32 [128, K/512, N] = 4 e4m3 bytes, byte j = slot j; scale uint8
    [16, K/512, N] E8M0). Slot j of (p, tile) holds k = j*(K/4) + tile*128 + p (kind "h") or
    tile*512 + j*128 + p (kind "i"); one scale per 8 partitions x 4 slots (nkilib MX layouts).

    mode "ocp": exponent floor(log2 amax) - 8, clip 448 (the fork's dense MX packer; an all-zero group gets scale
    byte 0 instead of that packer's wrapped 255). mode "nkilib": exponent field of amax - 7 (quantize_to_mx)."""
    w = np.asarray(w, dtype=np.float32)
    K, N = w.shape
    assert K % (Q * P) == 0, f"contraction dim {K} must be a multiple of 512"
    n_t = K // (Q * P)
    if kind == "h":
        g = w.reshape(Q, n_t, P, N).transpose(2, 1, 3, 0)  # [p, tile, N, j]
    else:
        g = w.reshape(n_t, Q, P, N).transpose(2, 0, 3, 1)
    g = np.ascontiguousarray(g).reshape(P // 8, 8, n_t, N, Q)
    amax = np.abs(g).max(axis=(1, 4))  # [16, n_t, N]
    if mode == "ocp":
        _, ex = np.frexp(np.maximum(amax, np.float32(2.0**-120)))
        e = ex.astype(np.int32) - 1 - 8  # floor(log2 amax) - 8, exact
        y = g / np.ldexp(np.float32(1.0), e)[:, None, :, :, None].astype(np.float32)
        y = np.clip(y, -448.0, 448.0).astype(np.float32)
        sc = np.clip(e + 127, 0, 254).astype(np.uint8)
    elif mode == "nkilib":
        field = ((amax.view(np.uint32) >> 23) & 0xFF).astype(np.int32)
        sc = np.clip(field - 7, 0, 255).astype(np.uint8)
        sf = np.power(2.0, field.astype(np.float64) - 7 - 127)[:, None, :, :, None]
        y = np.clip(g.astype(np.float64) / np.where(sf == 0, 1.0, sf), -448.0, 448.0).astype(np.float32)
    else:
        raise ValueError(mode)
    data = y.astype(ml_dtypes.float8_e4m3fn).view(np.uint8).reshape(P, n_t, N, Q)
    return np.ascontiguousarray(data).view(np.uint32).reshape(P, n_t, N), sc


# ----------------------------------------------------------------------------- pack jobs (one process each)

_S = {}


def _init(ckpt_dir, out_dir, lay_args):
    torch.set_num_threads(int(os.environ.get("PACK_TORCH_THREADS", "4")))
    cfg = Cfg(json.load(open(os.path.join(ckpt_dir, "config.json"))))
    lay = Layout(cfg, **lay_args)
    _S.update(ck=Ckpt(ckpt_dir), out=out_dir, lay=lay, ops=lay.operands())


def _mm(key):
    shape, dt = _S["ops"][key]
    dt = np.dtype(np.uint32) if dt == X4 else dt
    return np.memmap(os.path.join(_S["out"], key + ".bin"), dtype=dt, mode="r+", shape=shape)


def pack_globals():
    """Embedding, final norm, RoPE tables, lm_head slices, per-rank / per-position scalars and sel."""
    t0 = time.time()
    ck, lay, c = _S["ck"], _S["lay"], _S["lay"].cfg
    emb = ck.get(ck.pre + "embed_tokens.weight")
    dst = _mm("g/embed_table")
    for v0 in range(0, emb.shape[0], 16384):
        dst[v0 : v0 + 16384] = emb[v0 : v0 + 16384].to(torch.bfloat16).view(torch.int16).numpy().view(BF16)
    del emb, dst
    _mm("g/final_ln_w")[:] = ck.f32(ck.pre + "norm.weight")
    # HF default RoPE on the first `rot` dims (text mRoPE == RoPE)
    inv = 1.0 / np.float32(c.theta) ** (np.arange(0, c.rot, 2, dtype=np.float32) / c.rot)
    ang = np.arange(lay.rope_len, dtype=np.float32)[:, None] * inv.astype(np.float32)[None, :]
    _mm("g/cos_tab")[:] = np.cos(ang).astype(np.float32)
    _mm("g/sin_tab")[:] = np.sin(ang).astype(np.float32)
    lm = ck.get("lm_head.weight")  # [V, H]
    Vs = lay.Vs
    for r in range(lay.ranks):
        _mm(f"r{r}/w_lm_p")[:] = perm_rows(lm[r * Vs : (r + 1) * Vs].float().numpy().T.astype(BF16))
        _mm(f"r{r}/vbase")[:] = r * Vs
        _mm(f"r{r}/ep_rank")[:] = r
        sel = np.zeros((c.E, lay.E_L), np.float32)
        sel[r * lay.E_L + np.arange(lay.E_L), np.arange(lay.E_L)] = 1.0
        _mm(f"r{r}/sel")[:] = sel
    for t in range(lay.tp):
        _mm(f"gdn{t}/tp_rank")[:] = t
    for t in range(lay.atp):
        _mm(f"attn{t}/atp_rank")[:] = t
    return "globals", time.time() - t0


def pack_layer(i):
    """Layer i except its routed experts: norms, router, shared-expert slices, mixer shards + router fold."""
    t0 = time.time()
    ck, lay, c = _S["ck"], _S["lay"], _S["lay"].cfg
    lp = f"{ck.pre}layers.{i}."
    m = lp + "mlp."
    post = ck.f32(lp + "post_attention_layernorm.weight")
    _mm("g/in_ln_w")[i] = ck.f32(lp + "input_layernorm.weight")
    _mm("g/post_ln_w")[i] = post
    wr = ck.f32(m + "gate.weight").T  # [H, E]
    w_sg = ck.f32(m + "shared_expert_gate.weight").T  # [H, 1]
    _mm("g/w_router_p")[i] = perm_rows(_bf(wr))
    sg, su, sd = (ck.f32(m + f"shared_expert.{n}_proj.weight") for n in ("gate", "up", "down"))  # [Is,H] x2, [H,Is]
    Is = lay.Is_r
    for r in range(lay.ranks):
        cs = slice(r * Is, (r + 1) * Is)
        _mm(f"r{r}/w_shared_p")[i] = perm_rows(_bf(np.concatenate([sg[cs].T, su[cs].T, w_sg], axis=1)))
        _mm(f"r{r}/w_sd")[i] = _bf(sd[:, cs].T)
    if i % 4 != 3:
        _pack_gdn(i, lp + "linear_attn.", wr, w_sg, post)
    else:
        _pack_attn(i, lp + "self_attn.", wr, w_sg, post)
    return i, time.time() - t0


def _pack_gdn(i, a, wr, w_sg, post):
    ck, lay, c = _S["ck"], _S["lay"], _S["lay"].cfg
    g = lay.gdn_layers.index(i)
    Kt = c.nk * c.dk
    qkv = ck.f32(a + "in_proj_qkv.weight").T  # [H, 2Kt + Vt]: q heads | k heads | v heads
    z = ck.f32(a + "in_proj_z.weight").T  # [H, Vt]
    wa, wb = ck.f32(a + "in_proj_a.weight").T, ck.f32(a + "in_proj_b.weight").T  # [H, nv]
    conv = ck.f32(a + "conv1d.weight").reshape(qkv.shape[1], -1)  # [Cc_tot, K]
    a_log, dt_bias = ck.f32(a + "A_log").reshape(-1), ck.f32(a + "dt_bias").reshape(-1)
    w_out = ck.f32(a + "out_proj.weight").T  # [Vt, H]
    _mm("g/norm_w")[g] = ck.f32(a + "norm.weight").reshape(-1)

    def cols(heads, base, dim):
        return np.concatenate([np.arange(base + h * dim, base + (h + 1) * dim) for h in heads])

    for t in range(lay.tp):
        kh, vh = lay.gdn_heads(t)
        qc, kc, vc = cols(kh, 0, c.dk), cols(kh, Kt, c.dk), cols(vh, 2 * Kt, c.dv)
        zc = cols(vh, 0, c.dv)
        proj = np.concatenate([qkv[:, qc], qkv[:, kc], qkv[:, vc], z[:, zc], wa[:, vh], wb[:, vh]], axis=1)
        d = f"gdn{t}/"
        _mm(d + "w_proj_p")[g] = perm_rows(_bf(proj))
        _mm(d + "conv_w")[g] = _bf(conv[np.concatenate([qc, kc, vc])])
        _mm(d + "a_log")[g] = a_log[vh]
        _mm(d + "dt_bias")[g] = dt_bias[vh]
        wo = _bf(w_out[zc])
        _mm(d + "w_out")[g] = wo
        _mm(d + "w_out_ext")[g] = pack_router_fold(wo, wr, w_sg, post)
        _mm(d + "wx")[g] = pack_x_fold(wr, w_sg, post, t, lay.tp)


def _pack_attn(i, a, wr, w_sg, post):
    ck, lay, c = _S["ck"], _S["lay"], _S["lay"].cfg
    k = lay.attn_layers.index(i)
    H, d = c.H, c.d
    qg = ck.f32(a + "q_proj.weight").reshape(c.nq, 2, d, H)  # per head [query | gate] rows
    wk, wv = ck.f32(a + "k_proj.weight"), ck.f32(a + "v_proj.weight")  # [nkv*d, H]
    wo_all = ck.f32(a + "o_proj.weight").T  # [nq*d, H]
    _mm("g/q_norm_w")[k] = ck.f32(a + "q_norm.weight").reshape(1, d)
    _mm("g/k_norm_w")[k] = ck.f32(a + "k_norm.weight").reshape(1, d)
    for t in range(lay.atp):
        qh, kv = lay.attn_heads(t)
        w_q = qg[qh, 0].reshape(-1, H).T
        w_gate = qg[qh, 1].reshape(-1, H).T
        dd = f"attn{t}/"
        _mm(dd + "w_qkv")[k] = _bf(np.concatenate([w_q, wk[kv * d : (kv + 1) * d].T, wv[kv * d : (kv + 1) * d].T], 1))
        _mm(dd + "w_gate_p")[k] = perm_rows(_bf(w_gate))
        wo = _bf(wo_all[qh[0] * d : (qh[-1] + 1) * d])
        _mm(dd + "w_o")[k] = wo
        _mm(dd + "w_o_ext")[k] = pack_router_fold(wo, wr, w_sg, post)
        _mm(dd + "wx_a")[k] = pack_x_fold(wr, w_sg, post, t, lay.atp)


def pack_experts(job):
    """Rank r's E_L routed experts of layer i -> nkilib MX layouts."""
    i, r = job
    ck, lay = _S["ck"], _S["lay"]
    gu_w, gu_s, dn_w, dn_s = (_mm(f"r{r}/{n}") for n in ("gate_up_w", "gate_up_scale", "down_w", "down_scale"))
    for j in range(lay.E_L):
        e = r * lay.E_L + j
        lp = f"{ck.pre}layers.{i}.mlp.experts.{e}."
        gw, uw, dw = (ck.f32(lp + f"{n}_proj.weight") for n in ("gate", "up", "down"))  # [I,H], [I,H], [H,I]
        gq, gs = mx_pack(gw.T, "h", lay.mx_scale)
        uq, us = mx_pack(uw.T, "h", lay.mx_scale)
        dq, ds = mx_pack(dw.T, "i", lay.mx_scale)
        gu_w[i, j] = np.stack([gq, uq], axis=1)
        gu_s[i, j] = np.stack([gs, us], axis=1)
        dn_w[i, j] = dq
        dn_s[i, j] = ds
    return job


# ----------------------------------------------------------------------------- driver / loader


def _run(fn, jobs, workers, init_args, label):
    t0 = time.time()
    if workers <= 1:
        _init(*init_args)
        for k, j in enumerate(jobs):
            fn(j) if j is not None else fn()
            _progress(label, k + 1, len(jobs), t0)
        return
    with ProcessPoolExecutor(workers, initializer=_init, initargs=init_args) as ex:
        futs = [ex.submit(fn, j) if j is not None else ex.submit(fn) for j in jobs]
        for k, f in enumerate(futs):
            f.result()
            _progress(label, k + 1, len(jobs), t0)


def _progress(label, k, n, t0):
    if k == n or k % max(1, n // 20) == 0:
        print(f"{label}: {k}/{n} ({time.time() - t0:.0f}s)", flush=True)


def pack(ckpt_dir, out_dir, layers=None, ranks=128, workers=1, tp=None, atp=None, rope_len=32768, mx_scale="ocp"):
    cfg = Cfg(json.load(open(os.path.join(ckpt_dir, "config.json"))))
    lay_args = dict(ranks=ranks, tp=tp, atp=atp, layers=layers, rope_len=rope_len, mx_scale=mx_scale)
    lay = Layout(cfg, **lay_args)
    ops = lay.operands()
    for key, (shape, dt) in ops.items():
        os.makedirs(os.path.join(out_dir, os.path.dirname(key)), exist_ok=True)
        with open(os.path.join(out_dir, key + ".bin"), "wb") as f:
            f.truncate(_nbytes(shape, dt))
    json.dump({k: [list(s), _dtname(d)] for k, (s, d) in ops.items()}, open(os.path.join(out_dir, "meta.json"), "w"))
    json.dump(lay.info(), open(os.path.join(out_dir, "layout.json"), "w"), indent=1)
    init = (ckpt_dir, out_dir, lay_args)
    _run(pack_globals, [None], 1, init, "globals")
    _run(pack_layer, list(range(lay.L)), workers, init, "layers")
    _run(pack_experts, [(i, r) for i in range(lay.L) for r in range(lay.ranks)], workers, init, "experts")
    return lay


def load_layout(out_dir):
    return json.load(open(os.path.join(out_dir, "layout.json")))


def load_rank(out_dir, rank, x4_dtype=None):
    """{kernel kwarg: read-only np.memmap} of every packed operand of ``rank`` (weights, positions, sel, lm_head
    slice; not the runtime state / tokens / positions / replica groups). X4 operands are uint32 (4 packed e4m3
    bytes) unless ``x4_dtype`` is given; x4_dtype="auto" uses nki's float8_e4m3fn_x4 when importable."""
    if isinstance(x4_dtype, str) and x4_dtype == "auto":
        try:
            import nki.language as nl

            x4_dtype = nl.float8_e4m3fn_x4
        except Exception:
            x4_dtype = None
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    lay = load_layout(out_dir)
    groups = ["g", f"gdn{rank % lay['tp']}", f"attn{rank % lay['atp']}", f"r{rank}"]
    out = {}
    for g in groups:
        for key, (shape, dt) in meta.items():
            grp, name = key.split("/", 1)
            if grp == g:
                assert name not in out, name
                out[name] = np.memmap(os.path.join(out_dir, key + ".bin"), dtype=_np_dtype(dt, x4_dtype), mode="r",
                                      shape=tuple(shape))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt_dir")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--layers", type=int, default=None, help="pack the first N layers (default: all)")
    ap.add_argument("--ranks", type=int, default=128)
    ap.add_argument("--tp", type=int, default=None, help="GDN TP size (default min(64, ranks))")
    ap.add_argument("--atp", type=int, default=None, help="attention TP size (default min(8, ranks))")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    ap.add_argument("--rope-len", type=int, default=32768, help="rows of cos_tab / sin_tab (max position + 1)")
    ap.add_argument("--mx-scale", choices=("ocp", "nkilib"), default="ocp")
    ap.add_argument("--print-layout", action="store_true", help="print operand shapes / bytes and exit")
    a = ap.parse_args(argv)
    if a.print_layout:
        cfg = Cfg(json.load(open(os.path.join(a.ckpt_dir, "config.json"))))
        lay = Layout(cfg, a.ranks, a.tp, a.atp, a.layers, a.rope_len, a.mx_scale)
        ops = lay.operands()
        tot = sum(_nbytes(s, d) for s, d in ops.values())
        rank0 = {}
        for g in lay.groups(0):
            for k, (s, d) in ops.items():
                if k.split("/")[0] == g:
                    rank0[k.split("/", 1)[1]] = (g, s, d)
        per = 0
        for n, (g, s, d) in rank0.items():
            per += _nbytes(s, d)
            print(f"{n:14s} {g:7s} {str(list(s)):34s} {_dtname(d):17s} {_nbytes(s, d) / 1e6:10.1f} MB")
        print(f"per rank {per / 1e9:.3f} GB; whole pack {tot / 1e9:.2f} GB ({len(ops)} files)")
        return
    if a.out_dir is None:
        ap.error("out_dir is required")
    t0 = time.time()
    pack(a.ckpt_dir, a.out_dir, a.layers, a.ranks, a.workers, a.tp, a.atp, a.rope_len, a.mx_scale)
    print(f"done {time.time() - t0:.0f}s -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
