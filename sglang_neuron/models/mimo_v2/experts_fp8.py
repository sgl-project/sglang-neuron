# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's routed experts in Trn2's FP8 (e4m3, max 240), converted from the checkpoint without loss.

Trn2 has no MX matmul, so V2.6's MXFP4 experts cannot run as stored. Each expert matrix is dequantized
(weights.dequant_mxfp4) and divided by a **power-of-two scale per output row**, the smallest that brings the row's
largest value to 240 or below. A power of two only shifts exponents, and an e2m1 value has two significant bits, so
every quotient is an e4m3 number: the conversion is exact (tools/mimo_v2/check_fp8_experts.py checks the whole
checkpoint; ``rowmax / 240``, a scale that is not a power of two, leaves about 7% of the values inexact). V2.5's
128 x 128 FP8-block experts take the same path; for them the per-row requantization is a rounding.

The experts then hold half the bf16 bytes (4.7 instead of 9.5 GB a rank at EP64) and run weight-only FP8 (bf16
activations, FP8 weights, the per-row scale applied after each matmul):

- decode: nkilib's moe_tkg in its ROW mode (gate/up scales [E, 2, I], down scales [E, H]), every local expert;
- prefill: the hybrid I-sharded moe_cte copy that feeds FP8 tiles to the matmul (experts_cte.py,
  kernels/moe_cte_fp8w.py), which takes per-channel scales.

The tensors are ``float8_e4m3fn`` holding values of at most 240, which the graphs compile as Trn2's e4m3 (the same
bytes; launch.py's UNSAFE_FP8FNCAST and the host's compiler option)."""

import os
import threading

import nki
import torch
from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from torch import nn

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.models.mimo_v2.weights import DequantSlice
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

FP8 = torch.float8_e4m3fn
FP8_MAX = 240.0
# moe_tkg's tokens a call
MAX_TOKENS = 128


def enabled(environ=os.environ) -> bool:
    """The routed experts run in Trn2's FP8 (converted without loss): always."""
    return True


def requant_rows(w: torch.Tensor) -> tuple:
    """fp32 [out, in] -> (FP8 [out, in], fp32 [out]) with a power-of-two scale per output row (module docstring)."""
    amax = w.abs().amax(dim=1).clamp(min=1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    q = (w / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q, scale


def _fp8_t(q: torch.Tensor) -> torch.Tensor:
    """An FP8 matrix transposed through its bytes (the CPU has no FP8 transpose kernels)."""
    return q.view(torch.uint8).t().contiguous().view(FP8)


class _Memo:
    """One requantization per (layer, rank, part) shared by a weight and its scale, whose loaders may run on the
    checkpoint loader's threads at once; each result is handed out once, so a reload recomputes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._slots = {}

    def take(self, key, part: str, compute):
        with self._lock:
            slot = self._slots.setdefault(key, {"lock": threading.Lock(), "parts": None})
        with slot["lock"]:
            if slot["parts"] is None:
                slot["parts"] = compute()
            value = slot["parts"].pop(part)
            if not slot["parts"]:
                with self._lock:
                    self._slots.pop(key, None)
        return value


def _views(slices: list) -> list:
    """bf16 views of [weights..., scales...] (weights.with_scales), or the slices of a bf16 checkpoint."""
    if not slices or isinstance(slices[0], DequantSlice):
        return slices
    n = len(slices) // 2
    return [DequantSlice(w, s) for w, s in zip(slices[:n], slices[n:])]


def _gate_up(slices: list, local: int, rows: slice) -> dict:
    """[E, H, 2, I] FP8 and [E, 2, I] scales of the local experts' gate and up (``rows`` this rank's I rows)."""
    views = _views(slices)
    weights, scales = [], []
    for g, u in zip(views[:local], views[local:]):
        pair_w, pair_s = [], []
        for view in (g, u):
            q, s = requant_rows(view[rows].float())
            pair_w.append(_fp8_t(q))
            pair_s.append(s)
        weights.append(torch.stack([w.view(torch.uint8) for w in pair_w], dim=1).view(FP8))
        scales.append(torch.stack(pair_s))
    return {"weight": torch.stack([w.view(torch.uint8) for w in weights]).view(FP8), "scale": torch.stack(scales)}


def _down(slices: list, cols: slice) -> dict:
    """[E, I, H] FP8 and [E, H] scales of the local experts' down (``cols`` this rank's I columns)."""
    weights, scales = [], []
    for view in _views(slices):
        q, s = requant_rows(view[:, cols].float())
        weights.append(_fp8_t(q).view(torch.uint8))
        scales.append(s)
    return {"weight": torch.stack(weights).view(FP8), "scale": torch.stack(scales)}


# keep each rank's converted experts on disk and read them back at the next launch (_ExpertCache)
CACHE_ENV = "SGLANG_NEURON_MIMO_EXPERT_CACHE"
_CHECKPOINT = {"path": None}


def set_checkpoint(path: str) -> None:
    """The checkpoint the next load reads (model.load_weights), which names the expert cache's directory."""
    _CHECKPOINT["path"] = path


def _sources_digest() -> str:
    """This module's and weights.py's code (their syntax trees, docstrings and comments left out, so an edit of the
    prose keeps the cache): the conversion the cached tensors came from."""
    import ast
    import hashlib

    from sglang_neuron.models.mimo_v2 import weights as weights_module

    def code_only(source: str) -> str:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
        return ast.dump(tree, include_attributes=False)

    digest = hashlib.sha256()
    for module_file in (__file__, weights_module.__file__):
        with open(module_file) as f:
            digest.update(code_only(f.read()).encode())
    return digest.hexdigest()[:16]


class _ExpertCache:
    """The converted experts of each (layer, gate_up | down, local expert range, intermediate shard), kept as one
    safetensors file (``SGLANG_NEURON_MIMO_EXPERT_CACHE=1``, off by default).

    The conversion is CPU work (dequantize MXFP4, requantize each row): 3 minutes of a 64-rank launch, every launch,
    for the same bytes. The first launch writes each result here, a later one reads it instead of reading and
    converting the checkpoint slices. A file is named by what decides its contents (the layer, which experts, which
    rows or columns, the shapes), not by the rank, so launches of other layouts with the same split share it; the
    directory by the checkpoint (its path and every file's size and mtime) and the conversion's sources. It sits
    beside torch-neuronx's NEFF cache root (``mimo_v2_experts/<fingerprint>``); about 1 byte a routed-expert
    parameter (~300 GB for V2.6-Flash), written once."""

    _directory = None
    _lock = threading.Lock()

    @classmethod
    def directory(cls):
        if os.environ.get(CACHE_ENV, "0").strip() != "1" or _CHECKPOINT["path"] is None:
            return None
        with cls._lock:
            if cls._directory is None:
                import hashlib
                from pathlib import Path

                root = Path(_CHECKPOINT["path"]).resolve()
                digest = hashlib.sha256(str(root).encode())
                for f in sorted(root.glob("*.safetensors")) + sorted(root.glob("*.json")):
                    st = f.stat()
                    digest.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
                digest.update(_sources_digest().encode())
                neff = os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR") or "/tmp/neff_cache"
                cls._directory = (Path(neff).resolve().parent / "mimo_v2_experts" / digest.hexdigest()[:24])
                cls._directory.mkdir(parents=True, exist_ok=True)
        return cls._directory

    @classmethod
    def get_or_compute(cls, name: str, compute) -> dict:
        directory = cls.directory()
        if directory is None:
            return compute()
        from safetensors.torch import load_file, save_file

        path = directory / f"{name}.safetensors"
        if path.exists():
            try:
                return load_file(str(path))
            except Exception:  # noqa: BLE001 - a torn or foreign file: convert again and replace it
                pass
        out = compute()
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
        save_file({k: v.contiguous() for k, v in out.items()}, str(tmp))
        os.replace(tmp, path)
        return out


def make_fp8(moe, layer_key: str) -> None:
    """Give ``moe`` (moe.MiMoV2MoE) FP8 expert weights and their per-row scales in place of its bf16 ones; each
    weight's mapping is its scale's too (moe.MiMoV2MoE.weight_mappings)."""
    e, (h, _, i) = moe.num_local_experts, moe.gate_up_weight.shape[1:]
    shard = slice(moe.expert_tp_rank * i, (moe.expert_tp_rank + 1) * i)
    memo = _Memo()
    experts = moe.local_experts
    # what decides a conversion's bytes (_ExpertCache); no cache without a layer key to tell the layers apart
    names = ({kind: f"{layer_key}.{kind}.x{experts.start}-{experts.stop}.s{shard.start}-{shard.stop}.h{h}"
              for kind in ("gate_up", "down")} if layer_key else None)
    moe.gate_up_weight = nn.Parameter(torch.empty(e, h, 2, i, dtype=FP8), requires_grad=False)
    moe.down_weight = nn.Parameter(torch.empty(e, i, h, dtype=FP8), requires_grad=False)
    moe.gate_up_scale = nn.Parameter(torch.empty(e, 2, i, dtype=torch.float32), requires_grad=False)
    moe.down_scale = nn.Parameter(torch.empty(e, h, dtype=torch.float32), requires_grad=False)

    def loader(kind: str, part: str):
        compute = (lambda slices: _gate_up(slices, e, shard)) if kind == "gate_up" else (
            lambda slices: _down(slices, shard))
        cached = compute if names is None else (lambda slices: _ExpertCache.get_or_compute(
            names[kind], lambda: compute(slices)))
        return SafetensorsWeightLoader(transform=lambda slices, rank: memo.take(
            (layer_key, kind, rank), part, lambda: cached(slices)))

    set_weight_loader(moe.gate_up_weight, loader("gate_up", "weight"))
    set_weight_loader(moe.gate_up_scale, loader("gate_up", "scale"))
    set_weight_loader(moe.down_weight, loader("down", "weight"))
    set_weight_loader(moe.down_scale, loader("down", "scale"))


@nki.jit
def _moe_tkg_row(hidden, gate_up, down, affinities, index, rank_id, gate_up_scale, down_scale):
    return moe_tkg(
        hidden, gate_up, down, affinities, index,
        is_all_expert=True,
        rank_id=rank_id,
        expert_gate_up_weights_scale=gate_up_scale,
        expert_down_weights_scale=down_scale,
        activation_fn=ActFnType.SiLU,
        mask_unselected_experts=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )


def decode_experts(moe, x: torch.Tensor, affinities: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """This rank's experts on the decode rows x [T, H] (ops/moe/moe_tkg_bf16.moe_experts over the FP8 weights)."""
    kernel = wrap_nki(_moe_tkg_row)[get_logical_nc_config()]
    rank_id = moe._ep_rank_id()
    tiles = [kernel(x[s:s + MAX_TOKENS], moe.gate_up_weight, moe.down_weight, affinities[s:s + MAX_TOKENS],
                    index[s:s + MAX_TOKENS], rank_id, moe.gate_up_scale, moe.down_scale)
             for s in range(0, x.shape[0], MAX_TOKENS)]
    return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)
