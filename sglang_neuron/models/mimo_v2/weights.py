# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's checkpoint read as the graphs' bf16 parameters.

The FP8 checkpoints (MiMo-V2.5, V2.6-Flash-RL) store nearly every linear weight as ``float8_e4m3fn`` with one fp32
``weight_scale_inv`` per 128 x 128 tile (``quantization_config.weight_block_size``); V2.6's routed experts are MXFP4
instead (``store_dtype: mxfp4``): two e2m1 values a byte, the even column in the low nibble, and one e8m0
``weight_scale`` byte per 32 columns of a row. The output projections, the routers, the sinks, the embedding and the
LM head are bf16. Each rank dequantizes only the tiles of its own shard as it reads them, and the graphs compute in
bf16.

Two scale-grid layouts exist, and they are not interchangeable (measured on the published checkpoints, from the
vllm-neuron port, vllm-project/vllm-neuron#67):

1. A flat grid on every FP8 weight but the fused QKV: ``[R, C]`` pairs with ``[ceil(R / 128), ceil(C / 128)]``.
2. The fused ``self_attn.qkv_proj.weight`` is stored as ``disk_tp`` concatenated TP shards (the index's
   ``metadata.tp_size``, 4), each ``[Q | K | V]`` with its own ceil-padded 128-row grid per section: a full layer's
   shard is 16 x 192 + 1 x 192 + 1 x 128 = 3392 rows in 24 + 2 + 1 = 27 scale rows (108 for the 4 shards, not the
   flat 106). Reading it as layout 1 silently mis-scales most of Q.

The graphs' attention layout (``head_layout``): every head is ``config.KV_HEAD_DIM`` (256) channels. Q and K take
their 192 channels with the 64 RoPE channels split around the middle, rotary pair ``(j, j + 32)`` of the checkpoint
moved to ``(j, j + 128)``, which the kernels rotate as contiguous halves; the 128 un-rotated channels fill the rest of
each half, and the last 64 channels are zero. V takes its 128 channels times ``attention_value_scale`` (the HF model
scales V before caching it, so the cache holds the scaled V), then 128 zero channels. Q and K are permuted alike, so
every dot product is the checkpoint's."""

from __future__ import annotations

import math

import torch

from sglang_neuron.models.mimo_v2.config import KV_HEAD_DIM
from sglang_neuron.models.transformer.config import DTYPE
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader

FP8_BLOCK = 128
MXFP4_BLOCK = 32
# e2m1 by nibble; a list so the module imports under a meta default device
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ---------------------------------------------------------------------------------------------------------------------
# dequantization primitives (host side, load time)
# ---------------------------------------------------------------------------------------------------------------------

def _ranges(key, shape) -> tuple:
    """(r0, r1, c0, c1) of a 2-D basic index (slices with step 1, or Ellipsis) over ``shape``."""
    rows, cols = shape
    if key is Ellipsis:
        key = (slice(None), slice(None))
    if not isinstance(key, tuple):
        key = (key,)
    if len(key) == 1:
        key = (key[0], slice(None))
    if len(key) != 2 or not all(isinstance(k, slice) for k in key):
        raise IndexError(f"a dequantizing slice takes a 2-D slice index, not {key!r}")
    (r0, r1, rs), (c0, c1, cs) = key[0].indices(rows), key[1].indices(cols)
    if rs != 1 or cs != 1:
        raise IndexError(f"a dequantizing slice takes unit steps, not {key!r}")
    return r0, max(r0, r1), c0, max(c0, c1)


def _scale_row(scale_row: torch.Tensor, width: int, block: int) -> torch.Tensor:
    return scale_row.reshape(-1).to(torch.float32).repeat_interleave(block)[:width]


def dequant_fp8_blockwise(w_slice, s_slice, rows: tuple = None, cols: tuple = None) -> torch.Tensor:
    """fp32 ``[r1 - r0, c1 - c0]`` of a flat-grid 128 x 128 blockwise FP8 ``[R, C]`` weight; only the row blocks that
    meet the window are read."""
    R, C = (int(d) for d in w_slice.get_shape())
    r0, r1 = rows if rows is not None else (0, R)
    c0, c1 = cols if cols is not None else (0, C)
    if r1 <= r0 or c1 <= c0:
        return torch.zeros((r1 - r0, c1 - c0), dtype=torch.float32)
    first, last = r0 // FP8_BLOCK, (r1 - 1) // FP8_BLOCK
    parts = []
    for b in range(first, last + 1):
        lo, hi = b * FP8_BLOCK, min((b + 1) * FP8_BLOCK, R)
        w = w_slice[lo:hi, c0:c1].to(torch.float32)
        parts.append(w * _scale_row(s_slice[b:b + 1], C, FP8_BLOCK)[c0:c1].view(1, -1))
    out = torch.cat(parts, dim=0)
    offset = first * FP8_BLOCK
    return out[r0 - offset:r1 - offset]


def _section_rows(w_slice, s_slice, row_lo: int, row_hi: int, sec_start: int, sec_end: int, sec_blk: int):
    """fp32 absolute rows [row_lo, row_hi) inside one fused-QKV section, whose own scale grid starts at scale row
    ``sec_blk``: block indices are relative to the section."""
    C = int(w_slice.get_shape()[1])
    first, last = (row_lo - sec_start) // FP8_BLOCK, (row_hi - 1 - sec_start) // FP8_BLOCK
    parts = []
    for b in range(first, last + 1):
        lo = sec_start + b * FP8_BLOCK
        hi = min(lo + FP8_BLOCK, sec_end)
        w = w_slice[lo:hi].to(torch.float32)
        parts.append(w * _scale_row(s_slice[sec_blk + b:sec_blk + b + 1], C, FP8_BLOCK).view(1, -1))
    out = torch.cat(parts, dim=0)
    offset = sec_start + first * FP8_BLOCK
    return out[row_lo - offset:row_hi - offset]


def dequant_mxfp4(w_slice, s_slice, rows: tuple = None, cols: tuple = None) -> torch.Tensor:
    """fp32 ``[r1 - r0, c1 - c0]`` of an MXFP4 ``[R, C]`` weight (packed ``[R, C / 2]`` uint8, e8m0 scales
    ``[R, C / 32]``); a column window must be whole 32-column blocks."""
    R, half = (int(d) for d in w_slice.get_shape())
    r0, r1 = rows if rows is not None else (0, R)
    c0, c1 = cols if cols is not None else (0, half * 2)
    if c0 % MXFP4_BLOCK or c1 % MXFP4_BLOCK:
        raise ValueError(f"an MXFP4 column window [{c0}, {c1}) must be whole {MXFP4_BLOCK}-column blocks")
    with torch.device("cpu"):
        packed = w_slice[r0:r1, c0 // 2:c1 // 2].to("cpu", torch.uint8)
        table = torch.tensor(_E2M1, dtype=torch.float32)
        values = torch.stack((table[(packed & 0x0F).long()], table[(packed >> 4).long()]), dim=-1)
        values = values.reshape(r1 - r0, c1 - c0)
        exponent = s_slice[r0:r1, c0 // MXFP4_BLOCK:c1 // MXFP4_BLOCK].to("cpu", torch.int32)
        scale = torch.exp2((exponent - 127).to(torch.float32))
        return values * scale.repeat_interleave(MXFP4_BLOCK, dim=1)


def _is_packed_fp4(w_slice) -> bool:
    get = getattr(w_slice, "get_dtype", None)
    dtype = str(get()) if callable(get) else str(w_slice[0:1, 0:1].dtype)
    return dtype in ("U8", "uint8", "torch.uint8")


class DequantSlice:
    """A checkpoint slice of a quantized 2-D weight that reads as bf16 (``get_shape`` and basic 2-D slicing): a
    blockwise FP8 weight with its ``weight_scale_inv``, or an MXFP4 one (packed uint8) with its ``weight_scale``,
    which reports its unpacked shape."""

    def __init__(self, weight, scale):
        self._weight, self._scale = weight, scale
        self._fp4 = _is_packed_fp4(weight)
        rows, cols = (int(d) for d in weight.get_shape())
        self._shape = (rows, cols * 2) if self._fp4 else (rows, cols)

    def get_shape(self):
        return self._shape

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, key):
        r0, r1, c0, c1 = _ranges(key, self._shape)
        if self._fp4:
            out = dequant_mxfp4(self._weight, self._scale, (r0, r1), (c0, c1))
        else:
            out = dequant_fp8_blockwise(self._weight, self._scale, (r0, r1), (c0, c1))
        return out.to(DTYPE)


def with_scales(keys, scale_suffix: str) -> list:
    """The checkpoint keys of a quantized parameter: its weight keys, then each one's scale (``weight_scale_inv`` for
    FP8 blocks, ``weight_scale`` for MXFP4)."""
    keys = [keys] if isinstance(keys, str) else list(keys)
    if not all(k.endswith(".weight") for k in keys):
        raise ValueError(f"quantized weight keys end in .weight: {keys[:2]}")
    return keys + [k.removesuffix(".weight") + "." + scale_suffix for k in keys]


def dequantizing(loader: SafetensorsWeightLoader) -> SafetensorsWeightLoader:
    """``loader`` over the bf16 view of quantized weights, for a mapping built with ``with_scales``: the bf16 loaders
    (gate/up, row-parallel, the per-expert stacks) shard them unchanged, and only the shard they read is converted."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        n = len(slices) // 2
        return loader.load([DequantSlice(w, s) for w, s in zip(slices[:n], slices[n:])], rank)

    return SafetensorsWeightLoader(transform=transform)


# ---------------------------------------------------------------------------------------------------------------------
# the attention head layout (module docstring)
# ---------------------------------------------------------------------------------------------------------------------

def qk_order(head_dim: int, rotary_dim: int, padded: int = KV_HEAD_DIM) -> list:
    """For each of the ``padded`` channels of a graph Q/K head, the checkpoint channel it holds (-1: zero).

    The checkpoint rotates channels ``j`` and ``j + rotary_dim / 2`` for ``j < rotary_dim / 2``; the kernels rotate
    ``j`` and ``j + padded / 2``. Rotary channel ``j`` stays, ``j + rotary_dim / 2`` moves to ``j + padded / 2``, and
    the un-rotated channels fill each half after its rotary part, in order."""
    half, mid = rotary_dim // 2, padded // 2
    nope = list(range(rotary_dim, head_dim))
    first = mid - half
    if len(nope) > 2 * first or half > mid:
        raise ValueError(f"a head of {head_dim} channels ({rotary_dim} rotary) does not fit {padded}")
    order = list(range(half)) + nope[:first]
    order += [-1] * (mid - len(order))
    second = list(range(half, rotary_dim)) + nope[first:]
    order += second + [-1] * (mid - len(second))
    return order


def qk_head_rows(rows: torch.Tensor, head_dim: int, rotary_dim: int) -> torch.Tensor:
    """[head_dim, H] checkpoint rows of one Q or K head as the graph's [KV_HEAD_DIM, H]."""
    order = torch.tensor(qk_order(head_dim, rotary_dim))
    out = rows.new_zeros((KV_HEAD_DIM, rows.shape[1]))
    keep = order >= 0
    out[keep] = rows[order[keep]]
    return out


def v_head_rows(rows: torch.Tensor, v_scale: float) -> torch.Tensor:
    """[v_head_dim, H] checkpoint rows of one V head, scaled, as the graph's zero-padded [KV_HEAD_DIM, H]."""
    out = rows.new_zeros((KV_HEAD_DIM, rows.shape[1]))
    out[:rows.shape[0]] = rows * v_scale
    return out


def unpermute_qk(head: torch.Tensor, head_dim: int, rotary_dim: int) -> torch.Tensor:
    """The checkpoint's channel order of a graph Q/K head ``[..., KV_HEAD_DIM]`` (``[..., head_dim]``), for checks."""
    order = qk_order(head_dim, rotary_dim)
    index = torch.tensor([order.index(c) for c in range(head_dim)])
    return head.index_select(-1, index)


# ---------------------------------------------------------------------------------------------------------------------
# the fused QKV (pre-sharded on disk, a scale grid per section)
# ---------------------------------------------------------------------------------------------------------------------

def _qkv_geometry(disk_tp: int, nh: int, nkv: int, hd: int, vhd: int) -> tuple:
    q_rows, k_rows, v_rows = (nh // disk_tp) * hd, (nkv // disk_tp) * hd, (nkv // disk_tp) * vhd
    q_blks, k_blks, v_blks = (math.ceil(r / FP8_BLOCK) for r in (q_rows, k_rows, v_rows))
    return q_rows, k_rows, v_rows, q_rows + k_rows + v_rows, q_blks, k_blks, v_blks, q_blks + k_blks + v_blks


def qkv_disk_tp_candidates(scale_rows: int, nh: int, nkv: int, hd: int, vhd: int) -> list:
    """The shard counts whose per-section grids total ``scale_rows`` rows."""
    return [d for d in (1, 2, 4, 8, 16, 32, 64) if nh % d == 0 and nkv % d == 0
            and d * _qkv_geometry(d, nh, nkv, hd, vhd)[7] == scale_rows]


def fused_qkv_loader(*, num_heads: int, num_kv_heads: int, head_dim: int, v_head_dim: int, rotary_dim: int,
                     v_scale: float, tp: int, disk_tp: int) -> SafetensorsWeightLoader:
    """This rank's Q heads and KV head(s) of the fused ``qkv_proj`` as ``[H, (q + 2 kv) x KV_HEAD_DIM]`` in the graph's
    head layout: its Q heads, then its K heads, then its V heads. With more ranks than KV heads each rank holds one KV
    head, the one its query heads group over (``rank // (tp / num_kv_heads)``).

    ``disk_tp`` is the checkpoint's shard count (its index's ``metadata.tp_size``). The sliding layers' grid (8 KV
    heads) totals 116 rows for 1, 2 and 4 shards alike, so it cannot be told from the grid; a declared count the grid
    contradicts is refused. A bf16 re-export (one slice, no grid) is read as ``disk_tp`` shards too."""
    nh, nkv, hd, vhd = num_heads, num_kv_heads, head_dim, v_head_dim
    q_per_rank = nh // tp
    kv_per_rank = max(1, nkv // tp)
    replicas = max(1, tp // nkv)

    def transform(slices: list, rank: int) -> torch.Tensor:
        if len(slices) == 1:
            w_slice, s_slice = slices[0], None
        elif len(slices) == 2:
            w_slice, s_slice = slices
            allowed = qkv_disk_tp_candidates(int(s_slice.get_shape()[0]), nh, nkv, hd, vhd)
            if disk_tp not in allowed:
                raise ValueError(f"the fused QKV's scale grid ({int(s_slice.get_shape()[0])} rows for {nh} heads and "
                                 f"{nkv} KV heads) admits disk shard counts {allowed}, not the index's {disk_tp}")
        else:
            raise ValueError(f"the fused QKV loader takes the weight and its scale, not {len(slices)} slices")
        q_rows, k_rows, _, shard_rows, q_blks, k_blks, _, shard_blks = _qkv_geometry(disk_tp, nh, nkv, hd, vhd)
        q_disk, kv_disk = nh // disk_tp, nkv // disk_tp

        def read(section: str, head: int) -> torch.Tensor:
            if section == "q":
                ds, local, width = head // q_disk, head % q_disk, hd
                start, blk, length = ds * shard_rows, ds * shard_blks, q_disk * hd
            elif section == "k":
                ds, local, width = head // kv_disk, head % kv_disk, hd
                start, blk, length = ds * shard_rows + q_rows, ds * shard_blks + q_blks, kv_disk * hd
            else:
                ds, local, width = head // kv_disk, head % kv_disk, vhd
                start, blk = ds * shard_rows + q_rows + k_rows, ds * shard_blks + q_blks + k_blks
                length = kv_disk * vhd
            lo = start + local * width
            if s_slice is None:
                return w_slice[lo:lo + width].to(torch.float32)
            return _section_rows(w_slice, s_slice, lo, lo + width, start, start + length, blk)

        kv_lo = rank // replicas if replicas > 1 else rank * kv_per_rank
        rows = [qk_head_rows(read("q", h), hd, rotary_dim) for h in range(rank * q_per_rank, (rank + 1) * q_per_rank)]
        rows += [qk_head_rows(read("k", j), hd, rotary_dim) for j in range(kv_lo, kv_lo + kv_per_rank)]
        rows += [v_head_rows(read("v", j), v_scale) for j in range(kv_lo, kv_lo + kv_per_rank)]
        return torch.cat(rows, dim=0).t().contiguous().to(DTYPE)

    return SafetensorsWeightLoader(transform=transform)


def o_proj_loader(num_heads_per_rank: int, v_head_dim: int) -> SafetensorsWeightLoader:
    """``o_proj`` ([H, heads x v_head_dim], bf16) as this rank's ``[heads_per_rank x v_head_dim, H]``."""
    cols = num_heads_per_rank * v_head_dim

    def transform(slices: list, rank: int) -> torch.Tensor:
        if len(slices) != 1:
            raise ValueError("o_proj is stored bf16 (quantization_config.ignored_layers)")
        return slices[0][:, rank * cols:(rank + 1) * cols].t().to(DTYPE)

    return SafetensorsWeightLoader(transform=transform)


def sink_loader(num_heads_per_rank: int) -> SafetensorsWeightLoader:
    """This rank's heads of ``attention_sink_bias`` [heads], in float32."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][
        rank * num_heads_per_rank:(rank + 1) * num_heads_per_rank].to(torch.float32))


def qkv_disk_tp(checkpoint_path: str) -> int:
    """The checkpoint's fused-QKV shard count: its index's ``metadata.tp_size`` (4 on the published checkpoints), 1
    where the index does not say (the HF layout of one unsharded [Q | K | V])."""
    import json
    import os

    index = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return 1
    with open(index) as f:
        meta = json.load(f).get("metadata") or {}
    return int(meta.get("tp_size") or 1)
