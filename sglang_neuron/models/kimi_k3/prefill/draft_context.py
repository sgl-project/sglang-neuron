# SPDX-License-Identifier: Apache-2.0
"""The MLA DSpark drafter context rows of a prompt, computed on the prefill.

For drafter layer ``l`` and token ``q`` the decode drafter attends over the context row
``[kv_a_layernorm_l(x_q Wkv_a_l [:512]) | rope(x_q Wkv_a_l [512:])]`` with ``x_q = context_norm(context_proj(cat_i
aux_i,q))``, where ``aux_i`` is the stream after the drafter's target layer ``i`` (the input stream of the consumer
layer ``i + 1``).

1. ``context_proj`` is linear, so each stage adds ``bf16(aux_c) @ fc_c`` for its consumer layers ``c`` into an fp32
   accumulator ``[T, H / world]`` (this rank's output columns, all ``T`` rows). The accumulator rides the PP seams.
2. The last stage rounds the accumulator to bf16, gathers the columns over the group and computes the rows of all
   ``T`` tokens on every rank, then writes the FP8 rows of the tokens this rank owns at their write slots.

The bf16 rounding points: ``aux`` and ``context_proj`` operands bf16 with an fp32 sum; the accumulator rounded once
to bf16; each RMSNorm rounds the normed value, then the product with its weight; ``kv_a`` sums in fp32 and rounds;
the rope rotation runs in fp32 and rounds.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
from torch import nn

from sglang_neuron.models.kimi_k3.prefill.comm import StageComm
from sglang_neuron.models.kimi_k3.prefill.fp8_rows import STRIPE, owned_rows, owned_rows_from, quantize_rows, row_bytes

DRAFTER_MODEL_TYPE = "k3_dspark"

_EXPECTED = {"model_type": DRAFTER_MODEL_TYPE, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "num_target_layers": 5}
_MUST_BE_OFF = ("mla_use_nope", "mla_use_output_gate", "mla_use_qk_norm", "is_causal", "fc_norm")


@dataclass(frozen=True)
class DrafterGeometry:
    """The drafter facts the prefill context rows read."""

    hidden_size: int
    num_layers: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    rms_norm_eps: float
    target_layer_ids: tuple[int, ...]
    rope_theta: float
    yarn_factor: float
    yarn_original: int
    yarn_beta_fast: float
    yarn_beta_slow: float
    yarn_mscale: float
    yarn_mscale_all_dim: float

    @property
    def row_width(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim


def read_drafter_geometry(source, target_hidden: int | None = None) -> DrafterGeometry:
    """Read and check the drafter config (a checkpoint directory, its ``config.json`` path or a dict).

    Raises:
        ValueError: The drafter is not the MLA DSpark shape the context rows implement; the message names each field.
    """
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        with open(os.path.join(path, "config.json") if os.path.isdir(path) else path) as f:
            config = json.load(f)
    else:
        config = dict(source)
    problems = [f"{k} is {config.get(k)!r}, not {v!r}" for k, v in _EXPECTED.items() if config.get(k) != v]
    problems += [f"{k} is on" for k in _MUST_BE_OFF if config.get(k)]
    rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
    if rope.get("rope_type") != "yarn":
        problems.append(f"rope_type is {rope.get('rope_type')!r}, not 'yarn'")
    targets = tuple(int(i) for i in (config.get("target_layer_ids") or ()))
    if len(targets) != config.get("num_target_layers"):
        problems.append(f"target_layer_ids {list(targets)} does not hold num_target_layers ids")
    if target_hidden is not None and config.get("target_hidden_size", config.get("hidden_size")) != target_hidden:
        problems.append(f"target_hidden_size is {config.get('target_hidden_size')!r}, not {target_hidden}")
    if config.get("hidden_size") != config.get("target_hidden_size", config.get("hidden_size")):
        problems.append("hidden_size differs from target_hidden_size")
    if problems:
        raise ValueError("The drafter checkpoint is not the MLA DSpark drafter the prefill context rows implement:\n  "
                         + "\n  ".join(problems))
    return DrafterGeometry(
        hidden_size=int(config["hidden_size"]), num_layers=int(config["num_hidden_layers"]),
        kv_lora_rank=int(config["kv_lora_rank"]), qk_rope_head_dim=int(config["qk_rope_head_dim"]),
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-5)), target_layer_ids=targets,
        rope_theta=float(rope["rope_theta"]), yarn_factor=float(rope["factor"]),
        yarn_original=int(rope["original_max_position_embeddings"]), yarn_beta_fast=float(rope["beta_fast"]),
        yarn_beta_slow=float(rope["beta_slow"]), yarn_mscale=float(rope.get("mscale", 1.0)),
        yarn_mscale_all_dim=float(rope.get("mscale_all_dim", 1.0)))


def _yarn_mscale(factor: float, mscale: float) -> float:
    return 1.0 if factor <= 1 else 0.1 * mscale * math.log(factor) + 1.0


def yarn_inv_freq(drafter: DrafterGeometry) -> torch.Tensor:
    """The fp32 YaRN inverse frequencies ``[rope / 2]`` of the drafter (DeepSeek YaRN)."""
    dim, theta = drafter.qk_rope_head_dim, drafter.rope_theta
    pos_freqs = theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    extrapolation, interpolation = 1.0 / pos_freqs, 1.0 / (drafter.yarn_factor * pos_freqs)

    def correction_dim(rotations: float) -> float:
        return (dim * math.log(drafter.yarn_original / (rotations * 2 * math.pi))) / (2 * math.log(theta))

    low = max(math.floor(correction_dim(drafter.yarn_beta_fast)), 0)
    high = min(math.ceil(correction_dim(drafter.yarn_beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    return interpolation * (1 - mask) + extrapolation * mask


def draft_rope_rows(positions: torch.Tensor, drafter: DrafterGeometry) -> torch.Tensor:
    """The fp32 rope rows ``[T, 2 * rope] = [cos_x | sin_x]`` of integer positions, pre-expanded for the GPT-J
    pairs (elements ``2i`` and ``2i + 1`` hold ``cos_i``). A host input of the writer stage."""
    scale = _yarn_mscale(drafter.yarn_factor, drafter.yarn_mscale) / _yarn_mscale(drafter.yarn_factor,
                                                                                   drafter.yarn_mscale_all_dim)
    pos = positions.reshape(-1).to(torch.int64).clamp(min=0).to(torch.float32)
    freqs = pos.reshape(-1, 1) * yarn_inv_freq(drafter)
    cos, sin = freqs.cos() * scale, freqs.sin() * scale
    return torch.cat([cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)], dim=-1)


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    normed = _bf16(x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps))
    return _bf16(normed * weight.float())


def rotate_rows(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Rotate the GPT-J pairs of ``x [T, R]`` with the pre-expanded rows ``rope [T, 2R]`` in fp32, then round."""
    width = x.shape[-1]
    cos_x, sin_x = rope[:, :width].float(), rope[:, width:].float()
    x = x.float()
    rot = torch.stack((-x[:, 1::2], x[:, 0::2]), dim=-1).flatten(-2)
    return _bf16(x * cos_x + rot * sin_x)


def owned_row_index(rows: int, rank: torch.Tensor, world: int,
                    start: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """This rank's rows ``[whole // world]`` of a chunk of ``rows`` rows that starts at position ``start`` (``[1]``;
    None: 0) in the MLA owner layout (``fp8_rows.owned_rows_from``), and which of them are rows of the chunk. A chunk
    that is not whole stripe cycles is taken as its next whole cycles (``whole`` rows): its rows past the chunk read
    row 0 and are marked False. None: the chunk is whole cycles, every row is held."""
    cycle = STRIPE * world
    whole = -(-rows // cycle) * cycle
    if start is None:
        own = owned_rows(whole, rank, world, STRIPE)
    else:
        own = owned_rows_from(start, whole, rank, world, STRIPE)
    if whole == rows:
        return own, None
    held = own < rows
    return torch.where(held, own, torch.zeros_like(own)), held


def owned_fp8_rows(rows: torch.Tensor, own: torch.Tensor) -> torch.Tensor:
    """The FP8 e4m3fn rows ``[len(own), W]`` of the rows ``own`` of ``rows [T, W]`` (fp32 values of bf16): the rows
    clamped to +-448 and cast (scale 1, ``fp8_rows.quantize_rows``), then gathered. Row writers take them as FP8: the
    device compile lowers a torch uint8 view of a cast FP8 value with no other consumer as a value conversion, so the
    kernel writer views its pool as FP8 inside the kernel (``page_io.kernel_row_writer``) and only the host writer
    views the rows as bytes (``write_owned_fp8_rows``)."""
    return quantize_rows(rows).index_select(0, own)


def write_owned_rows(pool: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """Write row ``t`` of ``rows [T, W]`` (uint8) to local slot ``slots[t]`` of ``pool [blocks, 128, W]`` in place.

    A slot below 0 (a row this rank does not own, or a pad row) writes zeros to slot 0, which lies in the reserved
    block 0, so the write is one fixed-shape scatter.
    """
    flat = pool.view(-1, pool.shape[-1])
    valid = slots.reshape(-1) >= 0
    index = torch.where(valid, slots.reshape(-1), torch.zeros_like(slots.reshape(-1))).to(torch.int64)
    values = torch.where(valid.unsqueeze(-1), rows, torch.zeros_like(rows))
    flat.index_put_((index,), values)


class DraftContext(nn.Module):
    """The drafter context weights and math of one prefill stage rank.

    Buffers (bf16): ``fc_<c>`` ``[H, H / world]`` for each consumer layer ``c`` of the stage (this rank's
    ``context_proj`` output columns of the aux state of ``c``); on the writer stage also ``context_norm`` ``[H]``,
    ``kv_a`` ``[L, R + DR, H]`` and ``kv_a_layernorm`` ``[L, R]``.

    Args:
        drafter: The drafter geometry.
        world: The stage group size.
        consumers: The consumer layers of this stage.
        writer: This stage writes the rows.
    """

    def __init__(self, drafter: DrafterGeometry, world: int, consumers: Sequence[int], writer: bool):
        super().__init__()
        hidden = drafter.hidden_size
        if hidden % world:
            raise ValueError(f"The drafter context columns {hidden} do not divide over {world} ranks")
        self.drafter, self.world = drafter, world
        self.fc_width = hidden // world
        self.consumers, self.writer = tuple(consumers), bool(writer)
        for c in self.consumers:
            self.register_buffer(f"fc_{c}", torch.zeros(hidden, self.fc_width, dtype=torch.bfloat16),
                                 persistent=False)
        if self.writer:
            layers, width = drafter.num_layers, drafter.row_width
            self.register_buffer("context_norm", torch.ones(hidden, dtype=torch.bfloat16), persistent=False)
            self.register_buffer("kv_a", torch.zeros(layers, width, hidden, dtype=torch.bfloat16), persistent=False)
            self.register_buffer("kv_a_layernorm", torch.ones(layers, drafter.kv_lora_rank, dtype=torch.bfloat16),
                                 persistent=False)

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, torch.Tensor], drafter: DrafterGeometry, all_consumers: Sequence[int],
                     consumers: Sequence[int], writer: bool, rank: int, world: int) -> "DraftContext":
        """Build from the drafter checkpoint tensors (``context_proj.weight`` ``[H, n * H]``, ``context_norm.weight``,
        ``layers.<l>.self_attn.kv_a_proj_with_mqa.weight``, ``layers.<l>.self_attn.kv_a_layernorm.weight``).

        ``all_consumers`` orders the ``context_proj`` input blocks (block ``i`` is the aux state of consumer ``i``).
        """
        module = cls(drafter, world, consumers, writer)
        hidden = drafter.hidden_size
        cols = slice(rank * module.fc_width, (rank + 1) * module.fc_width)
        proj = tensors["context_proj.weight"]
        with torch.no_grad():
            for index, consumer in enumerate(all_consumers):
                if consumer in module.consumers:
                    block = proj[cols, index * hidden:(index + 1) * hidden]
                    getattr(module, f"fc_{consumer}").copy_(block.to(torch.bfloat16).T)
            if module.writer:
                module.context_norm.copy_(tensors["context_norm.weight"].to(torch.bfloat16))
                for layer in range(drafter.num_layers):
                    attn = f"layers.{layer}.self_attn"
                    module.kv_a[layer].copy_(tensors[f"{attn}.kv_a_proj_with_mqa.weight"].to(torch.bfloat16))
                    module.kv_a_layernorm[layer].copy_(tensors[f"{attn}.kv_a_layernorm.weight"].to(torch.bfloat16))
        return module

    def empty_acc(self, num_tokens: int, device=None) -> torch.Tensor:
        return torch.zeros(num_tokens, self.fc_width, dtype=torch.float32, device=device)

    def add(self, acc: torch.Tensor, layer: int, aux: torch.Tensor) -> torch.Tensor:
        """Add the fc partial of consumer ``layer``: ``acc + bf16(aux [T, H]) @ fc_layer`` (fp32 sum)."""
        if layer not in self.consumers:
            return acc
        return acc + aux.to(torch.bfloat16).float() @ getattr(self, f"fc_{layer}").float()

    def rows(self, acc: torch.Tensor, rope: torch.Tensor, comm: StageComm) -> torch.Tensor:
        """The context rows ``[L, T, R + DR]`` (fp32 values of bf16) of all ``T`` tokens, on every rank."""
        if not self.writer:
            raise ValueError("This stage does not write the drafter rows")
        eps, latent = self.drafter.rms_norm_eps, self.drafter.kv_lora_rank
        x = _rms(comm.all_gather(acc.to(torch.bfloat16), dim=-1).float(), self.context_norm, eps)
        rows = []
        for layer in range(self.drafter.num_layers):
            kv = _bf16(x @ self.kv_a[layer].float().T)
            normed = _rms(kv[:, :latent], self.kv_a_layernorm[layer], eps)
            rows.append(torch.cat((normed, rotate_rows(kv[:, latent:], rope)), dim=-1))
        return torch.stack(rows)

    def write(self, acc: torch.Tensor, rope: torch.Tensor, slots: torch.Tensor, pools: Sequence[torch.Tensor],
              comm: StageComm, writer: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None] | None = None,
              start: torch.Tensor | None = None) -> None:
        """Compute the rows and write the FP8 rows of the tokens this rank owns into the drafter row pools.

        Each layer's rows of this rank's stripes (``owned_row_index``, the MLA rows' owner layout from the chunk's
        first position ``start``) go to the writer at their slots as FP8 rows (``owned_fp8_rows``).

        Args:
            acc: ``[T, H / world]`` fp32 accumulator.
            rope: ``[T, 2 * DR]`` fp32 rope rows (``draft_rope_rows``).
            slots: int32 ``[T]``: the local slot of each row this rank owns, -1 for the other rows and pad rows.
            pools: One ``[blocks, 128, R + DR]`` uint8 pool per drafter layer, written in place.
            comm: The stage collectives.
            writer: The FP8 row writer ``(pool, slots, fp8 rows)`` (``write_owned_fp8_rows`` when None).
            start: ``[1]`` int64, the position of the chunk's first row (None: 0). After a prefix hit inside a stripe
                the rank's rows are not the rows of a chunk that starts at a stripe cycle.
        """
        rows = self.rows(acc, rope, comm)
        if len(pools) != rows.shape[0]:
            raise ValueError(f"{len(pools)} drafter row pools for {rows.shape[0]} drafter layers")
        own, held = owned_row_index(rows.shape[1], comm.rank_tensor.to(rows.device), self.world,
                                    None if start is None else start.to(rows.device))
        own_slots = slots.reshape(-1).index_select(0, own)
        if held is not None:
            own_slots = torch.where(held, own_slots, torch.full_like(own_slots, -1))
        write = write_owned_fp8_rows if writer is None else writer
        for layer_rows, pool in zip(rows, pools):
            write(pool, own_slots, owned_fp8_rows(layer_rows, own))


def write_owned_fp8_rows(pool: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """``write_owned_rows`` of FP8 rows ``[T, W]``: their bytes (a host dtype view) at the slots. The host writer of
    ``DraftContext.write``."""
    write_owned_rows(pool, slots, row_bytes(rows))
