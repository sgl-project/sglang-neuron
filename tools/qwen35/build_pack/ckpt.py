# SPDX-License-Identifier: Apache-2.0
"""Streaming reader of the Qwen3.5 safetensors checkpoint (FP8 or BF16).

An FP8 weight has a 128x128 block ``<name>_scale_inv``. The reader dequantizes it to fp32 and rounds the result to
bf16. The kernel numerics checks use these bf16 values as the reference. The reader opens each file lazily, with one
handle for each process (safe_open memory-maps the file).
"""

from __future__ import annotations

import json
import os
from typing import Optional

TEXT_PREFIXES = ("model.language_model.", "model.")


class Checkpoint:
    def __init__(self, ckpt_dir: str):
        self.dir = ckpt_dir
        with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self._files: dict = {}
        self.prefix = next(
            (p for p in TEXT_PREFIXES if f"{p}embed_tokens.weight" in self.map), TEXT_PREFIXES[0]
        )

    def _f(self, name: str):
        from safetensors import safe_open

        fn = self.map[name]
        if fn not in self._files:
            self._files[fn] = safe_open(os.path.join(self.dir, fn), framework="pt")
        return self._files[fn]

    def has(self, name: str) -> bool:
        return name in self.map

    def raw(self, name: str):
        return self._f(name).get_tensor(name)

    def get(self, name: str, dtype=None):
        """The tensor as bf16 (FP8 dequantised with its block scale), or ``dtype``."""
        import torch

        t = self.raw(name)
        if t.dtype == torch.float8_e4m3fn:
            s = self.raw(name + "_scale_inv").float()
            o, i = t.shape
            s = s.repeat_interleave(128, 0)[:o].repeat_interleave(128, 1)[:, :i]
            t = (t.float() * s).to(torch.bfloat16)
        elif t.dtype not in (torch.float32,) and dtype is None:
            t = t.to(torch.bfloat16)
        return t.to(dtype) if dtype is not None else t

    def getf(self, name: str):
        """fp32 copy of the bf16-rounded weight (fp32 tensors stay exact)."""
        import torch

        return self.get(name).to(torch.float32)

    # -- names -----------------------------------------------------------------
    def text(self, tail: str) -> str:
        return self.prefix + tail

    def layer(self, li: int, tail: str) -> str:
        return f"{self.prefix}layers.{li}.{tail}"

    def lm_head_name(self) -> str:
        for n in ("lm_head.weight", self.text("lm_head.weight")):
            if n in self.map:
                return n
        return self.text("embed_tokens.weight")  # tied embeddings

    def expert(self, li: int, e: int):
        """(gate [I, H], up [I, H], down [H, I]) of routed expert e, bf16-rounded fp32."""
        import torch

        base = self.layer(li, f"mlp.experts.{e}.")
        if self.has(base + "gate_proj.weight"):
            return tuple(self.getf(base + n + ".weight") for n in ("gate_proj", "up_proj", "down_proj"))
        gu = self.layer(li, "mlp.experts.gate_up_proj")
        dn = self.layer(li, "mlp.experts.down_proj")
        if not (self.has(gu) and self.has(dn)):
            raise KeyError(f"layer {li}: no expert {e} weights in the checkpoint")
        w = self.raw(gu)[e].to(torch.bfloat16).float()  # [2I, H] (HF fused layout)
        I = w.shape[0] // 2
        return w[:I], w[I:], self.raw(dn)[e].to(torch.bfloat16).float()


def open_checkpoint(ckpt_dir: Optional[str]) -> Optional[Checkpoint]:
    return Checkpoint(ckpt_dir) if ckpt_dir else None
