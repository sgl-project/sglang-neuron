# SPDX-License-Identifier: Apache-2.0
"""Load one rank's shard of one Kimi-K3 prefill stage from the checkpoint.

``load_stage_rank`` reads only the rows and columns of this rank and builds one weight container for each module of
the stage. Two backends:

* ``torch``: the forms the CPU references read. The MX classes below are ``MxLinear`` / ``MxHeadLinear`` with no
  kernel backend, so they run their MX-exact torch reference. The routed experts keep the checkpoint MXFP4 bytes and
  dequantize at the call. The attention containers hold the BF16 projections of the checkpoint as well as the kernel
  forms.
* ``kernel``: the forms of the device kernels and only those: the MX linears get the NKI MX linear backends, the routed
  experts are packed for the blockwise kernel (and the latent down rows are permuted into its column order), and the
  KDA projections exist only as the KDA block kernel copies.

MX classes (MXFP8, ``mx_linear``): KDA q/k/v/g/b (hT_quad, the KDA front layout), MLA q_b (hT_quad), MLA W_UK
(consecutive over the nope rows), MLA o_proj (consecutive), MoE latent down and up (hT_quad), MoE shared gate/up
(hT_quad) and shared down (consecutive). Everything else is the checkpoint value in its checkpoint dtype.

Symbols: ``H`` hidden, ``W`` stage width (= TP = EP), ``hc`` local heads (KDA heads / W and MLA heads / W),
``D`` head dim, ``p = hc D``, ``NH`` MLA heads, ``R`` kv lora rank, ``DR`` rope dim, ``VD`` v head dim, ``QD`` nope +
rope, ``QL`` q lora rank, ``E`` experts, ``E_L = E / W``, ``DL`` latent width, ``I`` expert intermediate,
``IS = shared experts x I / W``, ``ID`` dense intermediate / W, ``V`` vocab, ``G = H / 512``.

Containers and tensor names (``named_buffers`` of each container; ``T``/``K`` = in both backends / kernel only /
torch only):

================================  ============================================  ===============  =====
key, tensor                       shape                                         dtype            when
================================  ============================================  ===============  =====
``layers.<i>.attn`` (KDA)
  ``A_log``                       ``[hc]``                                      fp32             both
  ``dt_bias``                     ``[p]``                                       fp32             both
  ``conv_weight``                 ``[3, p, 4]`` (q, k, v short-conv taps)       fp32             both
  ``o_norm_weight``               ``[D]``                                       fp32             both
  ``front_w_fa``                  ``[1, 128, H / 128, 128]`` f_a tile           bf16             both
  ``front_w_fb_t``                ``[D, p]`` f_b transposed                     bf16             both
  ``front_conv_w``                ``[3, p, 4]``                                 fp32             both
  ``front_A_log``                 ``[1, hc]``                                   fp32             both
  ``front_dt_bias``               ``[hc, D]``                                   fp32             both
  ``front_w_qkv_mx``              ``[3 p / 128, G, 128, 512]``                  uint8            both
  ``front_w_qkv_scale``           ``[3 p / 128, 128, G, 128]``                  uint8            both
  ``front_w_gb_mx``               ``[G, 128, 4 (p + hc)]``                      uint8            both
  ``front_w_gb_scale``            ``[128, G, p + hc]``                          uint8            both
  ``block_o_proj_w_t``            ``[p, H]`` o_proj K-major                     bf16             both
  ``block_o_norm_w``              ``[1, D]``                                    fp32             both
  ``{q,k,v,g}_proj.weight``       ``[p, H]``                                    bf16             torch
  ``b_proj.weight``               ``[hc, H]``                                   bf16             torch
  ``f_a_proj.weight``             ``[D, H]``                                    bf16             torch
  ``f_b_proj.weight``             ``[p, D]``                                    bf16             torch
  ``o_proj.weight``               ``[H, p]``                                    bf16             torch
``layers.<i>.attn`` (MLA)
  ``q_a_proj.weight``             ``[QL, H]``                                   bf16             both
  ``q_a_layernorm.weight``        ``[QL]``                                      bf16             both
  ``q_b_proj.{w_mx,w_scale}``     MxLinear hT_quad of ``[hc QD, QL]``           uint8            both
  ``kv_a_proj_with_mqa.weight``   ``[R + DR, H]``                               bf16             both
  ``kv_a_layernorm.weight``       ``[R]``                                       bf16             both
  ``w_uk_mx.{w_mx,w_scale}``      MxHeadLinear of W_UK ``[hc, nope, R]``        uint8            both
  ``w_v_all``                     ``[NH, 128, R / 128, VD]`` W_V of all heads   bf16             both
  ``g_proj.weight``               ``[hc VD, H]``                                bf16             both
  ``o_proj.{w_mx,w_scale}``       MxLinear consecutive of ``[H, hc VD]``        uint8            both
  ``kv_b_proj.weight``            ``[hc (nope + VD), R]``                       bf16             torch
``layers.<i>.ffn`` (``LatentMoe``)
  ``gate_weight_t``               ``[H, E]``                                    bf16             both
  ``e_score_correction_bias``     ``[E]``                                       fp32             both
  ``routed_expert_down_proj.*``   MxLinear hT_quad of ``[DL, H]``               uint8            both
  ``routed_expert_up_proj.*``     MxLinear hT_quad of ``[H, DL]``               uint8            both
  ``routed_expert_norm.weight``   ``[DL]``                                      bf16             both
  ``shared_gate_up.*``            MxLinear hT_quad of ``[2 IS, H]`` (gate|up)   uint8            both
  ``shared_down.*``               MxLinear consecutive of ``[H, IS]``           uint8            both
  ``mx_gate_up_weights``          ``[E_L, 128, 2, DL / 512, I]``                uint16           kernel
  ``mx_gate_up_scales``           ``[E_L, 16, 2, DL / 512, I]``                 uint8            kernel
  ``mx_down_weights``             ``[E_L, 128, I / 512, DL]``                   uint16           kernel
  ``mx_down_scales``              ``[E_L, 16, I / 512, DL]``                    uint8            kernel
  ``local_expert_ids``            ``[1, E_L]``                                  int64            kernel
  ``dq_w{1,3}_codes``             ``[E_L, I, DL / 2]``                          uint8            torch
  ``dq_w{1,3}_scales``            ``[E_L, I, DL / 32]``                         uint8            torch
  ``dq_w2_codes``                 ``[E_L, DL, I / 2]``                          uint8            torch
  ``dq_w2_scales``                ``[E_L, DL, I / 32]``                         uint8            torch
``layers.<i>.ffn`` (``DenseMlp``)
  ``{gate,up}_proj.weight``       ``[ID, H]``                                   bf16             both
  ``down_proj.weight``            ``[H, ID]``                                   bf16             both
``layers.<i>.norms``
  ``input_layernorm.weight``, ``post_attention_layernorm.weight``, ``self_attention_res_norm.weight``,
  ``mlp_res_norm.weight``         ``[H]``                                       bf16             both
  ``self_attention_res_proj.weight``, ``mlp_res_proj.weight``  ``[1, H]``       bf16             both
``embed`` (``VocabEmbedding``, first stage)
  ``weight``                      ``[V / W, H]``                                bf16             both
``final`` (last stage)
  ``norm.weight``, ``output_attn_res_norm.weight``  ``[H]``                     bf16             both
  ``output_attn_res_proj.weight`` ``[1, H]``                                    bf16             both
  ``lm_head.weight``              ``[V / W, H]``                                bf16             both
``draft`` (stages with a drafter consumer layer, with ``draft_dir``)
  ``fc_<c>``                      ``[H, H / W]`` for each consumer layer ``c``  bf16             both
  ``context_norm``                ``[H]`` (the writer stage)                    bf16             both
  ``kv_a``                        ``[L_draft, R + DR, H]`` (the writer stage)   bf16             both
  ``kv_a_layernorm``              ``[L_draft, R]`` (the writer stage)           bf16             both
================================  ============================================  ===============  =====

Row and column slices of rank ``r``: KDA q/k/v/g/f_b/dt_bias/conv rows ``r p .. (r + 1) p``, b and A_log heads
``r hc ..``, o_proj columns ``r p ..``, f_a whole; MLA q_b rows ``r hc QD ..``, kv_b rows ``r hc (nope + VD) ..``
(W_V of all heads from the whole tensor), g_proj rows ``r hc VD ..``, o_proj columns ``r hc VD ..``; MoE experts
``r E_L .. (r + 1) E_L - 1`` whole, shared gate/up rows and down columns ``r IS ..``; dense gate/up rows and down
columns ``r ID ..``; embedding and lm_head rows ``r V / W ..``; drafter ``context_proj`` output rows ``r H / W ..``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from sglang_neuron.models.kimi_k3.prefill import fp4_pack
from sglang_neuron.models.kimi_k3.prefill.dense import DenseMlp, Weight
from sglang_neuron.models.kimi_k3.prefill.embed import VocabEmbedding
from sglang_neuron.models.kimi_k3.prefill.geometry import KimiK3Geometry
from sglang_neuron.models.kimi_k3.prefill.moe import KERNEL_EXPERT_BUFFERS, TORCH_EXPERT_BUFFERS, LatentMoe
from sglang_neuron.models.kimi_k3.prefill.mx_linear import (
    MxHeadLinear,
    MxLinear,
    install_mx_linear_backend,
    kda_front_mx_tensors,
)

BACKENDS = ("torch", "kernel")

MX_ORDERS = {
    "mla_q_b": "hT_quad",
    "mla_o_proj": "consecutive",
    "moe_latent_down": "hT_quad",
    "moe_latent_up": "hT_quad",
    "moe_shared_gate_up": "hT_quad",
    "moe_shared_down": "consecutive",
}
"""The group order of each MX linear class (KDA q/k/v/g/b take hT_quad in the KDA front layout; MLA W_UK takes
consecutive groups of the nope rows of each head)."""

NORM_NAMES = ("input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "mlp_res_norm",
              "self_attention_res_proj", "mlp_res_proj")
KDA_TORCH_LINEARS = ("q_proj", "k_proj", "v_proj", "g_proj", "b_proj", "f_a_proj", "f_b_proj", "o_proj")
KDA_FRONT_TILE = 128
KDA_FRONT_GROUP_HEADS = 3
"""Local heads of one KDA front program; more heads run one front per group of this many heads (group-major rows)."""
DRAFT_CHECKPOINT_FILE = "model.safetensors"

_PREFIX = "language_model."


class CheckpointReader:
    """Random access into a sharded safetensors checkpoint (``model.safetensors.index.json``).

    Keys are given without the ``language_model.`` prefix; the reader adds it when the checkpoint has it.
    """

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._prefix = _PREFIX if any(k.startswith(_PREFIX) for k in self.weight_map) else ""
        self._handles: dict[str, Any] = {}

    def key(self, key: str) -> str:
        return self._prefix + key

    def has(self, key: str) -> bool:
        return self.key(key) in self.weight_map

    def _slice(self, key: str):
        from safetensors import safe_open

        full = self.key(key)
        if full not in self.weight_map:
            raise KeyError(f"The checkpoint {self.path} has no tensor {full}.")
        name = self.weight_map[full]
        if name not in self._handles:
            self._handles[name] = safe_open(os.path.join(self.path, name), framework="pt", device="cpu")
        return self._handles[name].get_slice(full)

    def full(self, key: str, dtype: torch.dtype | None = None) -> Tensor:
        return _cast(self._slice(key)[:], dtype, key)

    def rows(self, key: str, start: int, stop: int, dtype: torch.dtype | None = None) -> Tensor:
        return _cast(self._slice(key)[start:stop], dtype, key)

    def cols(self, key: str, start: int, stop: int, dtype: torch.dtype | None = None) -> Tensor:
        return _cast(self._slice(key)[:, start:stop], dtype, key)

    def close(self) -> None:
        self._handles.clear()


def _lossless(source: torch.dtype, target: torch.dtype) -> bool:
    if source == target:
        return True
    if not (source.is_floating_point and target.is_floating_point):
        return False
    narrow, wide = torch.finfo(source), torch.finfo(target)
    return wide.eps <= narrow.eps and wide.max >= narrow.max and wide.tiny <= narrow.tiny


def _cast(t: Tensor, dtype: torch.dtype | None, key: str) -> Tensor:
    """Return a checkpoint tensor in ``dtype``; a cast that can change a value stops the load."""
    if dtype is None or t.dtype == dtype:
        return t.contiguous()
    if not _lossless(t.dtype, dtype):
        raise TypeError(f"The checkpoint tensor {key} is {t.dtype}; a cast to {dtype} would change its values.")
    return t.to(dtype).contiguous()


def stage_layers(plan: Any) -> list[int]:
    """The global layer ids of a stage plan: ``plan.layers``, ``plan.layer_range``, ``plan.stage_range``, or the plan
    itself as a range or sequence."""
    for name in ("layers", "layer_range", "stage_range", "layer_ids"):
        value = getattr(plan, name, None)
        if value is not None and not callable(value):
            return [int(i) for i in value]
    return [int(i) for i in plan]


def _split(total: int, rank: int, world: int, what: str) -> tuple[int, int]:
    if total % world:
        raise ValueError(f"{what} ({total}) does not divide by the stage width {world}.")
    step = total // world
    return rank * step, (rank + 1) * step


class WeightContainer(nn.Module):
    """A set of named tensors (buffers) and child weight modules of one stage module."""

    def __init__(self, kind: str, layer: int | None = None):
        super().__init__()
        self.kind = kind
        self.layer = layer

    def put(self, name: str, tensor: Tensor) -> None:
        self.register_buffer(name, tensor.contiguous())

    def extra_repr(self) -> str:
        return f"kind={self.kind}, layer={self.layer}"


def kda_front_head_groups(hc: int) -> int:
    """The head groups of the KDA front layouts for ``hc`` local heads."""
    if hc > KDA_FRONT_GROUP_HEADS and hc % KDA_FRONT_GROUP_HEADS == 0:
        return hc // KDA_FRONT_GROUP_HEADS
    return 1


def group_major(weights: list[Tensor], groups: int) -> Tensor:
    """Concatenate group 0 of each weight, then group 1 of each weight, and so on (rows)."""
    return torch.cat([w.chunk(groups)[i] for i in range(groups) for w in weights])


def kda_front_fa_tile(f_a: Tensor, tile: int = KDA_FRONT_TILE) -> Tensor:
    """``[1, tile, H / tile, tile]`` with ``out[0, p, c, j] = f_a[j, tile c + p]`` (``f_a [tile, H]``)."""
    rows, hidden = f_a.shape
    if rows != tile or hidden % tile:
        raise ValueError(f"The KDA f_a weight is [{rows}, {hidden}]; the front tile needs [{tile}, a multiple of "
                         f"{tile}].")
    return f_a.view(1, tile, hidden // tile, tile).permute(0, 3, 2, 1).contiguous()


def mla_w_v_layout(w_v: Tensor) -> Tensor:
    """W_V ``[NH, VD, R]`` of every head in the flash layout ``[NH, 128, R / 128, VD]``: ``out[h, c, k, d] =
    w_v[h, d, 128 k + c]``."""
    heads, v_dim, rank = w_v.shape
    return w_v.reshape(heads, v_dim, rank // 128, 128).permute(0, 3, 2, 1).contiguous()


def load_kda(reader: CheckpointReader, g: KimiK3Geometry, layer: int, rank: int, world: int,
             backend: str) -> WeightContainer:
    """The KDA attention weights of one layer and rank."""
    base = f"model.layers.{layer}.self_attn"
    bf16, fp32 = torch.bfloat16, torch.float32
    hc, d = g.kda_num_heads // world, g.kda_head_dim
    p0, p1 = rank * hc * d, (rank + 1) * hc * d
    attn = WeightContainer("kda", layer)
    lin = {name: reader.rows(f"{base}.{name}.weight", p0, p1, bf16) for name in ("q_proj", "k_proj", "v_proj",
                                                                                   "g_proj", "f_b_proj")}
    lin["b_proj"] = reader.rows(f"{base}.b_proj.weight", rank * hc, (rank + 1) * hc, bf16)
    lin["f_a_proj"] = reader.full(f"{base}.f_a_proj.weight", bf16)
    lin["o_proj"] = reader.cols(f"{base}.o_proj.weight", p0, p1, bf16)
    # A_log has zero entries after the heads; take the heads, then this rank's.
    a_log = reader.full(f"{base}.A_log", fp32)[:world * hc][rank * hc:(rank + 1) * hc]
    dt_bias = reader.rows(f"{base}.dt_bias", p0, p1, fp32)
    conv = torch.stack([reader.rows(f"{base}.{n}_conv1d.weight", p0, p1, fp32)[:, 0] for n in ("q", "k", "v")])
    o_norm = reader.full(f"{base}.o_norm.weight", fp32)
    attn.put("A_log", a_log)
    attn.put("dt_bias", dt_bias)
    attn.put("conv_weight", conv)
    attn.put("o_norm_weight", o_norm)
    attn.put("front_w_fa", kda_front_fa_tile(lin["f_a_proj"]))
    attn.put("front_w_fb_t", lin["f_b_proj"].t())
    attn.put("front_conv_w", conv.clone())
    attn.put("front_A_log", a_log.reshape(1, hc))
    attn.put("front_dt_bias", dt_bias.view(hc, d))
    groups = kda_front_head_groups(hc)
    mx = kda_front_mx_tensors(group_major([lin["q_proj"], lin["k_proj"], lin["v_proj"]], groups),
                              group_major([lin["g_proj"], lin["b_proj"]], groups))
    for name, value in mx.items():
        attn.put(f"front_{name}", value)
    attn.put("block_o_proj_w_t", lin["o_proj"].t())
    attn.put("block_o_norm_w", o_norm.reshape(1, -1))
    if backend == "torch":
        for name in KDA_TORCH_LINEARS:
            setattr(attn, name, Weight(lin[name]))
    return attn


def load_mla(reader: CheckpointReader, g: KimiK3Geometry, layer: int, rank: int, world: int,
             backend: str) -> WeightContainer:
    """The MLA attention weights of one layer and rank."""
    base = f"model.layers.{layer}.self_attn"
    bf16 = torch.bfloat16
    hc = g.num_heads // world
    qd, kvd = g.qk_nope_head_dim + g.qk_rope_head_dim, g.qk_nope_head_dim + g.v_head_dim
    attn = WeightContainer("mla", layer)
    attn.q_a_proj = Weight(reader.full(f"{base}.q_a_proj.weight", bf16))
    attn.q_a_layernorm = Weight(reader.full(f"{base}.q_a_layernorm.weight", bf16))
    q_b = reader.rows(f"{base}.q_b_proj.weight", rank * hc * qd, (rank + 1) * hc * qd, bf16)
    attn.q_b_proj = MxLinear.from_weight(q_b, MX_ORDERS["mla_q_b"], f"{base}.q_b_proj")
    attn.kv_a_proj_with_mqa = Weight(reader.full(f"{base}.kv_a_proj_with_mqa.weight", bf16))
    attn.kv_a_layernorm = Weight(reader.full(f"{base}.kv_a_layernorm.weight", bf16))
    kv_b_full = reader.full(f"{base}.kv_b_proj.weight", bf16).view(g.num_heads, kvd, g.kv_lora_rank)
    kv_b = kv_b_full[rank * hc:(rank + 1) * hc]
    attn.w_uk_mx = MxHeadLinear.from_weight(kv_b[:, :g.qk_nope_head_dim], f"{base}.w_uk_mx")
    attn.put("w_v_all", mla_w_v_layout(kv_b_full[:, g.qk_nope_head_dim:]))
    v0, v1 = rank * hc * g.v_head_dim, (rank + 1) * hc * g.v_head_dim
    attn.g_proj = Weight(reader.rows(f"{base}.g_proj.weight", v0, v1, bf16))
    o_proj = reader.cols(f"{base}.o_proj.weight", v0, v1, bf16)
    attn.o_proj = MxLinear.from_weight(o_proj, MX_ORDERS["mla_o_proj"], f"{base}.o_proj")
    if backend == "torch":
        attn.kv_b_proj = Weight(kv_b.reshape(hc * kvd, g.kv_lora_rank).contiguous())
    return attn


def read_experts(reader: CheckpointReader, base: str, experts: Iterable[int]) -> list[dict[str, tuple[Tensor, Tensor]]]:
    """The checkpoint MXFP4 ``(codes, scales)`` of w1, w2 and w3 of each expert."""
    return [{w: (reader.full(f"{base}.experts.{e}.{w}.weight_packed", torch.uint8),
                 reader.full(f"{base}.experts.{e}.{w}.weight_scale", torch.uint8))
             for w in ("w1", "w2", "w3")} for e in experts]


def load_moe(reader: CheckpointReader, g: KimiK3Geometry, layer: int, rank: int, world: int,
             backend: str) -> LatentMoe:
    """The latent MoE weights of one layer and rank."""
    base = f"model.layers.{layer}.block_sparse_moe"
    bf16, fp32 = torch.bfloat16, torch.float32
    e0, e1 = _split(g.num_experts, rank, world, "The expert count")
    s0, s1 = _split(g.num_shared_experts * g.moe_intermediate_size, rank, world, "The shared intermediate width")
    down = reader.full(f"{base}.routed_expert_down_proj.weight", bf16)
    if backend == "kernel":
        # The latent leaves the down projection in the kernel's column order.
        down = down.index_select(0, fp4_pack.bwmm_hidden_order(g.routed_expert_hidden_size))
    shared = f"{base}.shared_experts"
    gate_up = torch.cat([reader.rows(f"{shared}.gate_proj.weight", s0, s1, bf16),
                         reader.rows(f"{shared}.up_proj.weight", s0, s1, bf16)], dim=0)
    moe = LatentMoe(
        gate_weight_t=reader.full(f"{base}.gate.weight", bf16).t().contiguous(),
        e_score_correction_bias=reader.full(f"{base}.gate.e_score_correction_bias", fp32),
        routed_expert_down_proj=MxLinear.from_weight(down, MX_ORDERS["moe_latent_down"],
                                                     f"{base}.routed_expert_down_proj"),
        routed_expert_up_proj=MxLinear.from_weight(reader.full(f"{base}.routed_expert_up_proj.weight", bf16),
                                                   MX_ORDERS["moe_latent_up"], f"{base}.routed_expert_up_proj"),
        routed_expert_norm=reader.full(f"{base}.routed_expert_norm.weight", bf16),
        shared_gate_up=MxLinear.from_weight(gate_up, MX_ORDERS["moe_shared_gate_up"], f"{shared}.gate_up"),
        shared_down=MxLinear.from_weight(reader.cols(f"{shared}.down_proj.weight", s0, s1, bf16),
                                         MX_ORDERS["moe_shared_down"], f"{shared}.down"),
        e_start=e0, e_local=e1 - e0, top_k=g.top_k, renormalize=True, situ_beta=g.situ_beta,
        situ_linear_beta=g.situ_linear_beta, norm_eps=g.rms_norm_eps)
    if backend == "kernel":
        gu, gus, dn, dns = [], [], [], []
        for parts in read_experts(reader, base, range(e0, e1)):
            g, gs = fp4_pack.pack_expert_gate_up_bwmm(*parts["w1"], *parts["w3"])
            d, ds = fp4_pack.pack_expert_down_bwmm(*parts["w2"])
            gu.append(g)
            gus.append(gs)
            dn.append(d)
            dns.append(ds)
        buffers = dict(zip(KERNEL_EXPERT_BUFFERS, (torch.stack(gu), torch.stack(gus), torch.stack(dn),
                                                   torch.stack(dns))))
        moe.use_blockwise_experts(buffers, ep_degree=world)
    else:
        experts = read_experts(reader, base, range(e0, e1))
        buffers = {}
        for (codes, scales), w in zip(TORCH_EXPERT_BUFFERS, ("w1", "w3", "w2")):
            buffers[codes] = torch.stack([parts[w][0] for parts in experts])
            buffers[scales] = torch.stack([parts[w][1] for parts in experts])
        moe.use_torch_experts(buffers)
    return moe


def load_dense(reader: CheckpointReader, g: KimiK3Geometry, layer: int, rank: int, world: int) -> DenseMlp:
    """The dense MLP weights of one layer and rank."""
    base = f"model.layers.{layer}.mlp"
    bf16 = torch.bfloat16
    r0, r1 = _split(g.intermediate_size, rank, world, "The dense intermediate width")
    return DenseMlp(reader.rows(f"{base}.gate_proj.weight", r0, r1, bf16),
                    reader.rows(f"{base}.up_proj.weight", r0, r1, bf16),
                    reader.cols(f"{base}.down_proj.weight", r0, r1, bf16),
                    situ_beta=g.situ_beta, situ_linear_beta=g.situ_linear_beta)


def load_norms(reader: CheckpointReader, layer: int) -> WeightContainer:
    """The four norms and the two AttnRes projections of one layer (whole on every rank)."""
    norms = WeightContainer("norms", layer)
    for name in NORM_NAMES:
        setattr(norms, name, Weight(reader.full(f"model.layers.{layer}.{name}.weight", torch.bfloat16)))
    return norms


def load_embed(reader: CheckpointReader, g: KimiK3Geometry, rank: int, world: int) -> VocabEmbedding:
    v0, v1 = _split(g.vocab_size, rank, world, "The vocab size")
    return VocabEmbedding(reader.rows("model.embed_tokens.weight", v0, v1, torch.bfloat16))


def load_final(reader: CheckpointReader, g: KimiK3Geometry, rank: int, world: int) -> WeightContainer:
    """The output AttnRes, the final norm and this rank's LM head rows (the last stage)."""
    bf16 = torch.bfloat16
    final = WeightContainer("final")
    final.norm = Weight(reader.full("model.norm.weight", bf16))
    final.output_attn_res_norm = Weight(reader.full("model.output_attn_res_norm.weight", bf16))
    final.output_attn_res_proj = Weight(reader.full("model.output_attn_res_proj.weight", bf16))
    v0, v1 = _split(g.vocab_size, rank, world, "The vocab size")
    final.lm_head = Weight(reader.rows("lm_head.weight", v0, v1, bf16))
    return final


def _draft_config(draft_dir: str) -> dict:
    with open(os.path.join(draft_dir, "config.json")) as f:
        return json.load(f)


def draft_consumer_layers(draft_dir: str) -> tuple[int, ...]:
    """The target layers whose input stream is a drafter aux state: the drafter's target layer ids plus one."""
    return tuple(int(i) + 1 for i in _draft_config(draft_dir)["target_layer_ids"])


def load_draft(draft_dir: str, layers: list[int], rank: int, world: int) -> WeightContainer | None:
    """The drafter context weights of one stage rank, or None for a stage with no consumer layer.

    The rows of the drafter context land on the stage of the last consumer layer (the writer stage).
    """
    from safetensors import safe_open

    consumers = draft_consumer_layers(draft_dir)
    mine = [c for c in consumers if c in layers]
    if not mine:
        return None
    draft = WeightContainer("draft")
    with safe_open(os.path.join(draft_dir, DRAFT_CHECKPOINT_FILE), framework="pt", device="cpu") as f:
        proj = f.get_slice("context_proj.weight")
        hidden = proj.get_shape()[0]
        r0, r1 = _split(hidden, rank, world, "The drafter hidden size")
        for index, consumer in enumerate(consumers):
            if consumer in mine:
                block = _cast(proj[r0:r1, index * hidden:(index + 1) * hidden], torch.bfloat16, "context_proj.weight")
                draft.put(f"fc_{consumer}", block.t())
        if consumers[-1] in mine:
            count = int(_draft_config(draft_dir)["num_hidden_layers"])
            draft.put("context_norm", _cast(f.get_tensor("context_norm.weight"), torch.bfloat16, "context_norm"))
            draft.put("kv_a", torch.stack([_cast(f.get_tensor(f"layers.{i}.self_attn.kv_a_proj_with_mqa.weight"),
                                                 torch.bfloat16, "kv_a") for i in range(count)]))
            draft.put("kv_a_layernorm", torch.stack([_cast(f.get_tensor(f"layers.{i}.self_attn.kv_a_layernorm.weight"),
                                                           torch.bfloat16, "kv_a_layernorm") for i in range(count)]))
    return draft


def _install_backend(module: nn.Module, backend: str) -> None:
    if backend == "kernel":
        from sglang_neuron.models.kimi_k3.prefill import kernel_imports

        kernel_imports.prepare()
        install_mx_linear_backend(module, kernel_imports.mx_linear, kernel_imports.mx_head_linear)
    else:
        install_mx_linear_backend(module, None, None)


def load_stage_rank(checkpoint_dir: str, geometry: KimiK3Geometry, plan: Any, rank: int, world: int, *, backend: str,
                    draft_dir: str | None = None, layers: Iterable[int] | None = None) -> dict[str, nn.Module]:
    """Load the weight containers of one stage rank (module docstring).

    Args:
        checkpoint_dir: The checkpoint directory (safetensors shards and ``model.safetensors.index.json``).
        geometry: The model geometry (``geometry.read_geometry``).
        plan: The stage plan (``stage_layers``).
        rank: This rank in the stage group.
        world: The stage width.
        backend: ``torch`` or ``kernel``.
        draft_dir: The drafter checkpoint, or None for no drafter weights.
        layers: Only these global layers of the stage (a test loads one layer at a time); None loads all of them.
            ``embed`` and ``final`` come with the first and the last layer of the model.

    Returns:
        ``{"embed", "layers.<i>.attn", "layers.<i>.ffn", "layers.<i>.norms", "final", "draft"}`` (the keys the stage
        has), global layer ids.
    """
    if backend not in BACKENDS:
        raise ValueError(f"The weight backend {backend!r} is unknown. Use one of {BACKENDS}.")
    g = geometry
    stage = stage_layers(plan)
    wanted = stage if layers is None else [i for i in stage if i in set(layers)]
    reader = CheckpointReader(checkpoint_dir)
    out: dict[str, nn.Module] = {}
    try:
        with torch.device("cpu"):
            if 0 in wanted:
                out["embed"] = load_embed(reader, g, rank, world)
            for i in wanted:
                attn = load_mla if g.is_mla(i) else load_kda
                out[f"layers.{i}.attn"] = attn(reader, g, i, rank, world, backend)
                if g.is_moe(i):
                    out[f"layers.{i}.ffn"] = load_moe(reader, g, i, rank, world, backend)
                else:
                    out[f"layers.{i}.ffn"] = load_dense(reader, g, i, rank, world)
                out[f"layers.{i}.norms"] = load_norms(reader, i)
            if g.num_layers - 1 in wanted:
                out["final"] = load_final(reader, g, rank, world)
            # fc_<c> comes with consumer layer c; the writer weights with the last consumer layer.
            if draft_dir is not None:
                draft = load_draft(draft_dir, wanted, rank, world)
                if draft is not None:
                    out["draft"] = draft
    finally:
        reader.close()
    for module in out.values():
        _install_backend(module, backend)
    return out


def stage_rank_bytes(modules: dict[str, nn.Module]) -> int:
    """The tensor bytes of a loaded stage rank."""
    return sum(t.numel() * t.element_size() for m in modules.values() for t in m.buffers())


# ---------------------------------------------------------------------------------------------------------------------
# the stage modules of the loaded containers
# ---------------------------------------------------------------------------------------------------------------------

KDA_STATE_TENSORS = ("A_log", "dt_bias", "conv_weight", "o_norm_weight")
KDA_TORCH_TENSORS = tuple(f"{name}.weight" for name in KDA_TORCH_LINEARS)
KDA_KERNEL_TENSORS = {"w_qkv_mx": "front_w_qkv_mx", "w_qkv_scale": "front_w_qkv_scale", "w_gb_mx": "front_w_gb_mx",
                      "w_gb_scale": "front_w_gb_scale", "w_fa": "front_w_fa", "w_fb_t": "front_w_fb_t",
                      "conv_w": "front_conv_w", "A_log_2d": "front_A_log", "dt_bias_2d": "front_dt_bias",
                      "o_norm_w": "block_o_norm_w", "o_proj_w_t": "block_o_proj_w_t"}
"""The KDA block kernel inputs (``KdaAttention.kernel_buffers`` names) and the KDA container tensor of each."""


def _adopt(module: nn.Module, tensors: dict[str, Tensor]) -> nn.Module:
    """Give the parameters and buffers ``tensors`` (dotted names) of a module built on the meta device the loaded
    tensors themselves (same shape and dtype, no copy); refuse a tensor that stays on the meta device."""
    for name, tensor in tensors.items():
        owner_name, _, leaf = name.rpartition(".")
        owner = module.get_submodule(owner_name) if owner_name else module
        old = getattr(owner, leaf)
        if tuple(old.shape) != tuple(tensor.shape) or old.dtype != tensor.dtype:
            raise ValueError(f"{type(module).__name__}.{name} is {tuple(old.shape)} {old.dtype}; the loaded tensor is "
                             f"{tuple(tensor.shape)} {tensor.dtype}.")
        setattr(owner, leaf, nn.Parameter(tensor, requires_grad=False) if leaf in owner._parameters else tensor)
    missing = [n for n, t in [*module.named_parameters(), *module.named_buffers()] if t.is_meta]
    if missing:
        raise ValueError(f"{type(module).__name__} has no loaded tensor for {missing}.")
    return module


def kda_module(weights: WeightContainer, g: KimiK3Geometry, rank: int, world: int, backend: str) -> nn.Module:
    """The ``KdaAttention`` of a KDA container: the torch backend holds the BF16 projections, the kernel backend the
    kernel forms."""
    from sglang_neuron.models.kimi_k3.prefill.kda import KdaAttention, kda_scan_constants

    with torch.device("meta"):
        attn = KdaAttention.from_geometry(g, rank, world, backend=backend)
    tensors = dict(weights.named_buffers())
    names = KDA_STATE_TENSORS
    if backend == "torch":
        names = names + KDA_TORCH_TENSORS
    else:
        for name in KDA_TORCH_LINEARS:
            setattr(attn, name, None)
        attn.set_kernel_buffers({**{k: tensors[v] for k, v in KDA_KERNEL_TENSORS.items()},
                                 **{f"const{i}": c for i, c in enumerate(kda_scan_constants())}})
    return _adopt(attn, {name: tensors[name] for name in names})


def mla_module(weights: WeightContainer, g: KimiK3Geometry, rank: int, world: int, backend: str) -> nn.Module:
    """The ``MlaAttention`` of an MLA container, with its MX ``q_b_proj``, ``o_proj`` and W_UK (``w_uk_linear``)."""
    from sglang_neuron.models.kimi_k3.prefill.mla import MlaAttention

    with torch.device("meta"):
        attn = MlaAttention.from_geometry(g, rank, world, backend=backend)
    attn.q_b_proj = weights.q_b_proj
    attn.o_proj = weights.o_proj
    attn.w_uk_linear = weights.w_uk_mx
    attn.w_uk = None
    w_v = weights.w_v_all  # [NH, 128, R / 128, VD], mla_w_v_layout
    heads, _, _, v_dim = w_v.shape
    _adopt(attn, {"q_a_proj.weight": weights.q_a_proj.weight, "q_a_layernorm": weights.q_a_layernorm.weight,
                  "kv_a_proj_with_mqa.weight": weights.kv_a_proj_with_mqa.weight,
                  "kv_a_layernorm": weights.kv_a_layernorm.weight, "g_proj.weight": weights.g_proj.weight,
                  "w_v_all": w_v.permute(0, 3, 2, 1).reshape(heads, v_dim, -1)})
    if backend == "kernel":
        attn.set_kernel_buffers({"w_v": w_v, "ident": torch.eye(128, dtype=torch.bfloat16)})
    return attn


def stage_modules(loaded: dict[str, nn.Module], geometry: KimiK3Geometry, plan: Any, rank: int, world: int, *,
                  backend: str, drafter: Any = None) -> dict[str, nn.Module]:
    """The modules ``stage.KimiK3PrefillStage`` takes, from ``load_stage_rank``'s containers of the same stage rank.

    Each module holds the loaded tensors (no copy): ``layers.<i>.attn`` becomes ``kda.KdaAttention`` or
    ``mla.MlaAttention``, ``layers.<i>.norms`` ``layer.LayerNorms``, ``final`` ``stage.StageTail`` and ``draft``
    ``draft_context.DraftContext`` (with ``drafter``, its ``DrafterGeometry``, and the stage plan's consumer layers and
    writer role); ``embed`` and ``layers.<i>.ffn`` are the loaded modules. The kernel backend's attention modules
    still need ``prepare_kernel()`` for their trace shims.
    """
    from sglang_neuron.models.kimi_k3.prefill.draft_context import DraftContext
    from sglang_neuron.models.kimi_k3.prefill.layer import LayerNorms
    from sglang_neuron.models.kimi_k3.prefill.stage import StageTail

    g = geometry
    out: dict[str, nn.Module] = {}
    for key, module in loaded.items():
        if key.endswith(".attn"):
            build = mla_module if module.kind == "mla" else kda_module
            out[key] = build(module, g, rank, world, backend)
        elif key.endswith(".norms"):
            with torch.device("meta"):
                norms = LayerNorms(g.hidden_size, g.rms_norm_eps)
            out[key] = _adopt(norms, dict(module.named_buffers()))
        elif key == "final":
            with torch.device("meta"):
                tail = StageTail(g.hidden_size, g.vocab_size // world, g.rms_norm_eps)
            out[key] = _adopt(tail, dict(module.named_buffers()))
        elif key == "draft":
            if drafter is None or tuple(plan.all_draft_consumers) != tuple(t + 1 for t in drafter.target_layer_ids):
                raise ValueError("The loaded drafter weights need the drafter geometry and a stage plan with its "
                                 "target layers.")
            with torch.device("meta"):
                draft = DraftContext(drafter, world, plan.draft_consumers, plan.draft_writer)
            out[key] = _adopt(draft, dict(module.named_buffers()))
        else:
            out[key] = module
    if plan.drafter and "draft" not in out and not (plan.draft_consumers or plan.draft_writer):
        # a stage with no consumer layer loads no drafter weights; its empty context carries the accumulator through
        if drafter is None:
            raise ValueError("The stage plan has a drafter; give its drafter geometry.")
        out["draft"] = DraftContext(drafter, world, (), False)
    return out
