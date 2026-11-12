# SPDX-License-Identifier: Apache-2.0
"""One rank's giga decode operands from a packed Qwen3.5 checkpoint.

The serving packer (``tools/qwen35/build_pack``, run from the repository root) writes raw little-endian ``.bin``
files, memory-mappable, described by ``meta.json`` (``{"operands": {group/name: [shape,
dtype]}}``):

- ``g/``: identical on every rank: ``embed``, ``final_ln``, ``in_ln_gdn`` / ``in_ln_attn`` (each layer kind's input
  norms in kind order), ``post_ln``, ``q_norm``, ``k_norm``, ``gdn_norm``, ``w_router_p``;
- ``gdn<tp>/t<i>/``: GDN TP position ``i`` at GDN TP ``tp``: ``w_proj_p``, ``conv_w``, ``a_log``, ``dt_bias``,
  ``w_out``;
- ``attn<atp>/t<i>/``: attention TP position ``i``: ``w_qkv``, ``w_gate_p``, ``w_o``;
- ``ep<world>/r<r>/``: the shared expert's rank slice ``w_shared_p``, ``w_sd``;
- ``exp/r<r>/``: rank ``r``'s MXFP8 experts ``gate_up_w`` / ``down_w`` (four e4m3 bytes per element, a trailing axis
  of 4 uint8) and their ``gate_up_scale`` / ``down_scale``;
- ``lm/r<r>/``: rank ``r``'s LM-head vocabulary slice ``w_lm_p``.

``load_rank`` assembles the kernel's operands (``graph.operand_shapes``) for a rank of a ``config.Qwen35Layout``:
the GDN group at the replica's TP, the attention group at the replica's TP or, when the pack holds a finer attention
TP whose positions split the replica's evenly (the 397B pack's attention TP 8 for a 16-rank replica), the rank's
heads cut from it (``split_attention``: the query and gate columns of its heads, the same KV head, the output rows of
its heads), the per-layer input norms interleaved back into layer order, the expert one-hot ``sel``, the vocabulary
base and EP position, and the RoPE tables (HF default RoPE on the first ``rotary_dim`` dims). Every operand is checked
against the kernel's shape and dtype.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch

from .config import Qwen35Dims, Qwen35Layout
from .graph import operand_shapes

_NP = {"bfloat16": None, "float32": np.float32, "int32": np.int32, "uint8": np.uint8}


def _np_dtype(name: str):
    if name == "bfloat16":
        import ml_dtypes

        return np.dtype(ml_dtypes.bfloat16)
    if name not in _NP:
        raise ValueError(f"packed operand dtype {name!r} is not one the giga decode reads")
    return np.dtype(_NP[name])


class Pack:
    """A packed checkpoint directory (module docstring)."""

    def __init__(self, root: str):
        self.root = root
        meta = json.load(open(os.path.join(root, "meta.json")))
        self.operands = meta.get("operands", meta)

    def has(self, key: str) -> bool:
        return key in self.operands

    def get(self, key: str) -> np.ndarray:
        if key not in self.operands:
            raise KeyError(f"the pack at {self.root} holds no {key}")
        shape, dtype = self.operands[key]
        return np.memmap(os.path.join(self.root, key + ".bin"), dtype=_np_dtype(dtype), mode="r", shape=tuple(shape))

    def groups(self, prefix: str) -> list:
        """The TP sizes of the groups ``<prefix><tp>/`` the pack holds."""
        sizes = set()
        for key in self.operands:
            head = key.split("/", 1)[0]
            if head.startswith(prefix) and head[len(prefix):].isdigit():
                sizes.add(int(head[len(prefix):]))
        return sorted(sizes)


def _torch(a: np.ndarray) -> torch.Tensor:
    a = np.ascontiguousarray(a)
    if a.dtype.name == "bfloat16":
        return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(a)


def split_attention(parts: dict, *, coarse: int, fine: int, position: int, q_heads: int, head_dim: int) -> dict:
    """The attention operands of TP position ``position`` at TP ``fine`` from those of position
    ``position // (fine // coarse)`` at TP ``coarse``: the query and gate columns of its query heads, the same KV
    head, the output-projection rows of its heads. ``parts`` holds ``w_qkv [L, H, nq_c * d + 2d]``,
    ``w_gate_p [L, 128, H/128, nq_c * d]``, ``w_o [L, nq_c * d, H]`` of the coarse position."""
    if fine % coarse:
        raise ValueError(f"attention TP {fine} does not split the packed TP {coarse}")
    k = fine // coarse
    nq_c = q_heads // coarse
    if nq_c % k:
        raise ValueError(f"{nq_c} query heads per packed position do not split {k} ways")
    width = nq_c // k * head_dim
    j = position % k
    q = parts["w_qkv"][..., : nq_c * head_dim]
    kv = parts["w_qkv"][..., nq_c * head_dim:]
    return {
        "w_qkv": np.concatenate([q[..., j * width:(j + 1) * width], kv], axis=-1),
        "w_gate_p": parts["w_gate_p"][..., j * width:(j + 1) * width],
        "w_o": parts["w_o"][:, j * width:(j + 1) * width, :],
    }


def rope_tables(dims: Qwen35Dims, length: int) -> tuple:
    """(cos, sin) ``[length, rotary_dim / 2]`` float32: HF's default RoPE on the first ``rotary_dim`` dims (text
    mRoPE is plain RoPE)."""
    rot = dims.rotary_dim
    inv = 1.0 / np.float32(dims.rope_theta) ** (np.arange(0, rot, 2, dtype=np.float32) / rot)
    ang = np.arange(length, dtype=np.float32)[:, None] * inv.astype(np.float32)[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def kvcp_w_qkv(pack: Pack, dims: Qwen35Dims, r: int, c: int, L_a: int, split_qkv) -> "np.ndarray":
    """``w_qkv [L, H, nq_l * d + 2 * nkv * d]`` of a decode context-parallel rank (``layout.kv_cp``): its query heads'
    columns (``split_qkv``, the rank's split of its coarse position), then every KV head's K columns, then every KV
    head's V columns — the CP decode attends with all ``nkv`` KV heads of its block of each page. The pack keeps one
    KV head per attention position, head ``h`` at position ``h * c / nkv`` (pack_model.attn_heads)."""
    d, nkv = dims.head_dim, dims.num_kv_heads
    if c % nkv:
        raise ValueError(f"the pack holds attention TP {c}, not a multiple of the {nkv} KV heads: a kv_cp rank takes "
                         "every KV head, one whole packed position each (repack with a coarser attention TP)")
    nq_l = dims.num_q_heads // r
    cols = [split_qkv[..., : nq_l * d]]
    for kind in range(2):  # every K head, then every V head (natural channel order)
        for h in range(nkv):
            kv = pack.get(f"attn{c}/t{h * (c // nkv)}/w_qkv")[:L_a][..., -2 * d:]
            cols.append(kv[..., kind * d:(kind + 1) * d])
    return np.concatenate(cols, axis=-1)


def load_rank(pack: Pack, dims: Qwen35Dims, layout: Qwen35Layout, rank: int, *, layers: Optional[int] = None,
              rope_len: Optional[int] = None) -> dict:
    """{operand: host tensor} of ``rank`` (module docstring), checked against ``graph.operand_shapes``."""
    L = layers or dims.num_layers
    r = layout.replica
    t = layout.position_of(rank)
    kinds = ["attn" if i % 4 == 3 else "gdn" for i in range(L)]
    L_g, L_a = kinds.count("gdn"), kinds.count("attn")
    out = {}
    g = {name: pack.get(f"g/{name}") for name in ("embed", "final_ln", "in_ln_gdn", "in_ln_attn", "post_ln", "q_norm",
                                                   "k_norm", "gdn_norm", "w_router_p")}
    out["embed_table"] = g["embed"]
    out["final_ln_w"] = g["final_ln"]
    in_ln = np.empty((L, dims.hidden), np.float32)
    gi = ai = 0
    for i, kind in enumerate(kinds):
        if kind == "gdn":
            in_ln[i], gi = g["in_ln_gdn"][gi], gi + 1
        else:
            in_ln[i], ai = g["in_ln_attn"][ai], ai + 1
    out["in_ln_w"] = in_ln
    out["post_ln_w"] = g["post_ln"][:L]
    out["q_norm_w"] = g["q_norm"][:L_a]
    out["k_norm_w"] = g["k_norm"][:L_a]
    out["norm_w"] = g["gdn_norm"][:L_g]
    out["w_router_p"] = g["w_router_p"][:L]
    if r not in pack.groups("gdn"):
        raise ValueError(f"the pack holds GDN TP {pack.groups('gdn')}, not the replica's {r}")
    for name in ("w_proj_p", "conv_w", "a_log", "dt_bias", "w_out"):
        out[name] = pack.get(f"gdn{r}/t{t}/{name}")[:L_g]
    names = ("w_qkv", "w_gate_p", "w_o")
    if r in pack.groups("attn"):
        c = r
    else:
        coarse = [g for g in pack.groups("attn") if r % g == 0 and (dims.num_q_heads // g) % (r // g) == 0]
        if not coarse:
            raise ValueError(f"the pack holds attention TP {pack.groups('attn')}, none of which splits into the "
                             f"replica's {r}")
        c = max(coarse)
    parts = {n: pack.get(f"attn{c}/t{t // (r // c)}/{n}")[:L_a] for n in names}
    out.update(split_attention(parts, coarse=c, fine=r, position=t, q_heads=dims.num_q_heads,
                               head_dim=dims.head_dim))
    if layout.kv_cp > 1:
        out["w_qkv"] = kvcp_w_qkv(pack, dims, r, c, L_a, out["w_qkv"])
    ep = f"ep{layout.world}/r{rank}"
    out["w_shared_p"] = pack.get(f"{ep}/w_shared_p")[:L]
    out["w_sd"] = pack.get(f"{ep}/w_sd")[:L]
    for name in ("gate_up_w", "down_w"):
        x = pack.get(f"exp/r{rank}/{name}")[:L]
        if x.dtype != np.uint8 or x.shape[-1] != 4:
            raise ValueError(f"exp/r{rank}/{name}: MXFP8 x4 blocks are packed as a trailing axis of 4 bytes")
        out[name] = np.ascontiguousarray(x).view(np.int32).reshape(x.shape[:-1])
    for name in ("gate_up_scale", "down_scale"):
        out[name] = pack.get(f"exp/r{rank}/{name}")[:L]
    out["w_lm_p"] = pack.get(f"lm/r{rank}/w_lm_p")
    e_l = dims.num_experts // layout.world
    sel = np.zeros((dims.num_experts, e_l), np.float32)
    sel[rank * e_l + np.arange(e_l), np.arange(e_l)] = 1.0
    out["sel"] = sel
    out["vbase"] = np.array([[rank * (dims.vocab // layout.world)]], np.int32)
    out["ep_rank"] = np.array([[rank]], np.int32)
    out["cos_tab"], out["sin_tab"] = rope_tables(dims, rope_len or layout.rope_len)
    want = operand_shapes(dims, layout, layers=L, rope_len=rope_len or layout.rope_len)
    tensors = {}
    for name, (shape, dtype) in want.items():
        tensor = _torch(out[name])
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
            raise ValueError(f"packed operand {name}: {tuple(tensor.shape)} {tensor.dtype}; the kernel takes "
                             f"{tuple(shape)} {dtype}")
        tensors[name] = tensor
    return tensors


def load_prefill_extra(pack: Pack, dims: Qwen35Dims, layout: Qwen35Layout, rank: int, *,
                       layers: Optional[int] = None) -> dict:
    """{operand: host tensor} the prefill takes besides the decode's (``graph.prefill_extra_shapes``): the per-kind
    input norms, the whole shared expert (``g/w_shared_full_p``, ``g/w_sd_full``) and the rank's global expert ids."""
    from .graph import prefill_extra_shapes

    L = layers or dims.num_layers
    L_g = sum(1 for i in range(L) if i % 4 != 3)
    e_l = dims.num_experts // layout.world
    out = {
        "in_ln_g": pack.get("g/in_ln_gdn")[:L_g], "in_ln_a": pack.get("g/in_ln_attn")[:L - L_g],
        "w_shared_full_p": pack.get("g/w_shared_full_p")[:L], "w_sd_full": pack.get("g/w_sd_full")[:L],
        "expert_ids": (rank * e_l + np.arange(e_l, dtype=np.float32)).reshape(1, e_l),
    }
    tensors = {}
    for name, (shape, dtype) in prefill_extra_shapes(dims, layout, layers=L).items():
        tensor = _torch(out[name])
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
            raise ValueError(f"packed prefill operand {name}: {tuple(tensor.shape)} {tensor.dtype}; the kernel takes "
                             f"{tuple(shape)} {dtype}")
        tensors[name] = tensor
    return tensors
