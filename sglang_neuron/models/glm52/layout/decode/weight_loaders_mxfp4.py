# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 MXFP4 checkpoint to giga-kernel (GK) weight store.

The GK takes every per-layer weight with a leading ``[num_layers, ...]`` axis, in
one exact layout for each argument, already sharded for the rank that consumes
it. This module holds both halves of that translation:

- `build_rank_store` reads the checkpoint once and writes one ``.npy`` file for
  each rank and each kernel argument. It runs ONE time for a (checkpoint,
  geometry) pair. Point it at a persistent filesystem, because the store outlives
  the process that made it and rebuilding costs tens of minutes.
- `load_rank_weights` memory-maps one rank's files and returns the torch tensors
  that the kernel takes.

WHY A STORE AND NOT A PER-RANK LOADER. Each rank's weight set is about 8 GB at
78 layers, and each layer's conversion reads the same checkpoint shards for every
rank. A loader that ran for each rank would read the 438 GB checkpoint 128 times
and convert the experts 128 times. The store makes a rank's weight load a
memory-map open.

The MoE expert layouts come from the kernel package,
``sglang_neuron.models.glm52.kernels.decode.mxfp4_checkpoint``. The kernel
owns those rules, so this module asks the kernel package for them instead of
repeating them. Those conversions import nki when they run, so a host without it
still builds and tests the attention, dense and indexer files.

A shard with another rank's columns has the right shape and gives finite, wrong
tokens, and so does a store of another checkpoint's weights at the same shapes. So
the store records what it was built for in ``geometry.json``: the sharding fields,
the checkpoint format (`quark_compat`) and the checkpoint's digest
(`checkpoint_digest`). `load_rank_weights` refuses a store whose record differs
from the run's geometry or checkpoint.

Layout notes that the kernel asserts, and that a silent mismatch turns into a
wrong answer rather than an error:

- ``wkv_a`` columns are permuted to ``[k_pe | kv_latent]``. The checkpoint holds
  ``[kv_latent | k_pe]``.
- ``out_absorb_w`` is LATENT-MAJOR, ``[C, nh_out, v_head]``. The kernel loads it
  as contiguous latent tiles.
- ``q_absorb_w`` holds only the heads of this rank's ``wq_b`` column block.
- bf16 tensors are stored as their uint16 bit pattern, because ``np.save`` does
  not round-trip a custom bf16 dtype. `load_rank_weights` views them back.

Example:
    >>> geo = Glm52GkGeometry()
    >>> build_rank_store("/path/to/ckpt", "/path/to/store", geo)  # doctest: +SKIP
    >>> w = load_rank_weights("/path/to/store", geo, 0, "/path/to/ckpt")  # doctest: +SKIP
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import torch

from sglang_neuron.models.glm52.layout import quark_compat
from sglang_neuron.models.glm52.layout.decode.gk_geometry import Glm52GkGeometry
from sglang_neuron.models.glm52.layout.glm_kv_balance import balance_static_kv_stacks

logger = logging.getLogger(__name__)

# The largest finite float8_e4m3fn value.
E4M3FN_MAX = 448.0

# Store file name for each kernel argument. The kernel argument names carry the
# `_all` suffix, and the store keeps them, so a file name identifies the argument
# it feeds with no mapping table.
ATTN_FILES = (
    "input_norm_w_all",
    "wq_a_all",
    "wkv_a_all",
    "q_norm_w_all",
    "kv_norm_w_all",
    "wq_b_all",
    "q_absorb_w_all",
    "out_absorb_w_all",
    "o_proj_w_all",
)
MOE_FILES = (
    "gamma_all",
    "router_weights_all",
    "router_bias_all",
    "routed_gate_up_weights_all",
    "routed_gate_up_weights_scale_all",
    "routed_down_weights_all",
    "routed_down_weights_scale_all",
    "shared_gate_up_weights_all",
    "shared_gate_up_weights_scale_all",
    "shared_down_weights_all",
    "shared_down_weights_scale_all",
)
DENSE_FILES = (
    "dense_gamma_all",
    "dense_gate_weights_all",
    "dense_up_weights_all",
    "dense_down_weights_all",
)
# The dense MLP's column-sharded weights (`Glm52GkGeometry.dense_degree`).
DENSE_MLP_FILES = DENSE_FILES[1:]
# The shared expert's files (`Glm52GkGeometry.shared_degree`).
SHARED_FILES = tuple(name for name in MOE_FILES if name.startswith("shared_"))
IDX_FILES = (
    "idx_wq_b_all",
    "idx_wk_all",
    "idx_k_norm_w_all",
    "idx_k_norm_b_all",
    "idx_weights_proj_all",
)
ALL_FILES = ATTN_FILES + MOE_FILES + DENSE_FILES + IDX_FILES

# Files whose uint16 content is a bf16 bit pattern.
_BF16_FILES = frozenset(
    {
        "input_norm_w_all",
        "wq_a_all",
        "wkv_a_all",
        "wq_b_all",
        "q_absorb_w_all",
        "out_absorb_w_all",
        "o_proj_w_all",
        "gamma_all",
        "router_weights_all",
        "dense_gamma_all",
        "dense_gate_weights_all",
        "dense_up_weights_all",
        "dense_down_weights_all",
        "idx_wq_b_all",
        "idx_wk_all",
        "idx_weights_proj_all",
    }
)

# Files whose uint16 content is a packed MXFP4 tile. The decode op
# (`ops.gk_decode`) reinterprets them as `float4_e2m1fn_x4` inside the trace.
_MX_FILES = frozenset(
    {
        "routed_gate_up_weights_all",
        "routed_down_weights_all",
        "shared_gate_up_weights_all",
        "shared_down_weights_all",
    }
)


def store_spec(geo: Glm52GkGeometry) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    """Return ``{file name: (dtype, per-rank shape)}`` for one geometry.

    Args:
        geo: The sharding point.

    Returns:
        The dtype and the shape of every store file.

    Example:
        >>> store_spec(Glm52GkGeometry())["wq_a_all"][1]
        (78, 6144, 512)
    """
    ell, h, e = geo.num_layers, geo.hidden, geo.num_experts
    e_l, i = geo.experts_per_rank, geo.moe_intermediate
    i_sh, i_dense = geo.shared_width, geo.dense_intermediate
    lf, ld, lm = geo.num_full_layers, geo.num_dense_layers, geo.num_moe_layers
    return {
        "input_norm_w_all": (np.uint16, (ell, 1, h)),
        "wq_a_all": (np.uint16, (ell, h, geo.wq_a_shard)),
        "wkv_a_all": (np.uint16, (ell, h, geo.wkv_a_shard)),
        "q_norm_w_all": (np.float32, (ell, 1, geo.q_lora_rank)),
        "kv_norm_w_all": (np.float32, (ell, 1, geo.kv_lora_rank)),
        "wq_b_all": (np.uint16, (ell, geo.q_lora_rank, geo.wq_b_shard)),
        "q_absorb_w_all": (
            np.uint16,
            (ell, geo.q_absorb_heads, geo.qk_nope_head_dim, geo.kv_lora_rank),
        ),
        "out_absorb_w_all": (
            np.uint16,
            (ell, geo.kv_lora_rank, geo.nh_out, geo.v_head_dim),
        ),
        "o_proj_w_all": (np.uint16, (ell, geo.nh_out * geo.v_head_dim, h)),
        "gamma_all": (np.uint16, (lm, 1, h)),
        "router_weights_all": (np.uint16, (lm, h, geo.router_columns)),
        "router_bias_all": (np.float32, (lm, 1, e)),
        "routed_gate_up_weights_all": (np.uint16, (lm, e_l, 128, 2, h // 512, i)),
        "routed_gate_up_weights_scale_all": (
            np.uint8,
            (lm, e_l, 2, 64, h // 512, i // 4),
        ),
        "routed_down_weights_all": (np.uint16, (lm, e_l, 128, i // 512, h)),
        "routed_down_weights_scale_all": (np.uint8, (lm, e_l, 64, h)),
        "shared_gate_up_weights_all": (np.uint16, (lm, 1, 128, 2, h // 512, i_sh)),
        "shared_gate_up_weights_scale_all": (np.uint8, (lm, 1, 16, 2, h // 512, i_sh)),
        "shared_down_weights_all": (np.uint16, (lm, 1, i_sh // 4, 1, h)),
        "shared_down_weights_scale_all": (np.uint8, (lm, 1, i_sh // 32, 1, h)),
        "dense_gamma_all": (np.uint16, (ld, 1, h)),
        "dense_gate_weights_all": (np.uint16, (ld, h, i_dense)),
        "dense_up_weights_all": (np.uint16, (ld, h, i_dense)),
        "dense_down_weights_all": (np.uint16, (ld, i_dense, h)),
        "idx_wq_b_all": (np.uint16, (lf, geo.q_lora_rank, geo.idx_q_shard)),
        "idx_wk_all": (np.uint16, (lf, h, geo.idx_head_dim)),
        "idx_k_norm_w_all": (np.float32, (lf, 1, geo.idx_head_dim)),
        "idx_k_norm_b_all": (np.float32, (lf, 1, geo.idx_head_dim)),
        "idx_weights_proj_all": (np.uint16, (lf, h, geo.idx_n_heads)),
    }


class Glm52SafetensorsReader:
    """Read one checkpoint tensor at a time, by name, as numpy.

    The reader keeps one parsed header for each shard file and memory-maps the
    bytes, so a caller pays for the tensor it asks for and nothing else. bf16 and
    fp8 arrive as their uint16 or uint8 bit pattern, because numpy has no dtype
    for them; `read_f32` converts a bf16 tensor to float32, and it and
    `read_bf16_bits` refuse a quantized tensor (FP8, or MXFP4 words with a scale),
    which `read` returns as its stored bytes.

    Args:
        ckpt_dir: Directory that holds ``model.safetensors.index.json``.

    Example:
        >>> r = Glm52SafetensorsReader("/path/to/ckpt")  # doctest: +SKIP
        >>> r.read("model.embed_tokens.weight").shape  # doctest: +SKIP
        (154880, 6144)
    """

    # safetensors dtype tag to the numpy dtype a tensor is read as. numpy has no
    # bf16 or fp8, so those keep their bit patterns in uint16 and uint8, and the
    # MXFP4 blocks are uint8 words of two FP4 values.
    _DTYPES = {
        "BF16": np.uint16,
        "F16": np.float16,
        "F32": np.float32,
        "U8": np.uint8,
        "F8_E4M3": np.uint8,
        "I8": np.int8,
        "I32": np.int32,
        "I64": np.int64,
    }

    def __init__(self, ckpt_dir: str) -> None:
        self.dir = ckpt_dir if ckpt_dir.endswith("/") else ckpt_dir + "/"
        with open(self.dir + "model.safetensors.index.json") as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}
        self.format = quark_compat.checkpoint_format(self.dir, self.weight_map)

    def _header(self, shard: str) -> tuple[dict, int]:
        if shard not in self._headers:
            with open(shard, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                self._headers[shard] = (json.loads(f.read(size)), 8 + size)
        return self._headers[shard]

    def _record(self, name: str) -> tuple[str, dict, int]:
        """The shard file, the header record and the shard's data offset of one tensor.

        Every read goes through here, the bf16 and fp32 reads included (they ask
        for the dtype first), so a missing tensor always names the remedy.

        Raises:
            KeyError: If the checkpoint has no tensor of that name.
        """
        if name not in self.weight_map:
            raise KeyError(
                f"checkpoint has no tensor {name!r}. Check the checkpoint "
                f"revision, or the layer index."
            )
        shard = self.dir + self.weight_map[name]
        head, base = self._header(shard)
        return shard, head[name], base

    def has(self, name: str) -> bool:
        """Whether the checkpoint holds this tensor."""
        return name in self.weight_map

    def dtype(self, name: str) -> str:
        """The safetensors dtype tag of one tensor, for example ``"BF16"``.

        Raises:
            KeyError: If the checkpoint has no tensor of that name.
        """
        return self._record(name)[1]["dtype"]

    def _refuse_quantized(self, name: str) -> None:
        """Refuse a bf16 or float read of a quantized tensor: an FP8 one, or MXFP4 words with their scale."""
        dtype = self.dtype(name)
        if dtype == "F8_E4M3" or (dtype == "U8" and self.has(name + "_scale")):
            raise ValueError(
                f"{name} is a quantized {dtype} tensor in a {self.format} checkpoint, and this read takes a bf16 "
                "or float one. Check the config.json and the checkpoint revision."
            )

    def read(self, name: str) -> np.ndarray:
        """Read one tensor. bf16 comes back as uint16 bits.

        Raises:
            KeyError: If the checkpoint has no tensor of that name.
        """
        shard, rec, base = self._record(name)
        lo, hi = rec["data_offsets"]
        raw = np.array(
            np.memmap(
                shard, dtype=np.uint8, mode="r", offset=base + lo, shape=(hi - lo,)
            )
        )
        dtype = self._DTYPES[rec["dtype"]]
        return raw.view(dtype).reshape(rec["shape"])

    def read_rows(self, name: str, lo: int, hi: int) -> np.ndarray:
        """Read rows ``[lo, hi)`` of a 2-D tensor, and nothing else.

        The embedding and ``lm_head`` are 1.9 GB each, and a rank needs only its
        own vocabulary shard. This maps the row range alone, so 128 ranks read
        the tensor once between them instead of 128 times each.

        Args:
            name: Tensor name in the checkpoint.
            lo: First row.
            hi: One past the last row.

        Returns:
            ``[hi - lo, columns]``. bf16 comes back as uint16 bits.

        Raises:
            KeyError: If the checkpoint has no tensor of that name.
            ValueError: If the tensor is not 2-D, or the range is out of bounds.
        """
        shard, rec, base = self._record(name)
        shape = tuple(rec["shape"])
        if len(shape) != 2:
            raise ValueError(
                f"{name} has shape {shape} and read_rows takes a 2-D tensor. "
                "Use read for any other rank."
            )
        if not 0 <= lo <= hi <= shape[0]:
            raise ValueError(
                f"rows [{lo}, {hi}) are outside {name} with {shape[0]} rows."
            )
        dtype = np.dtype(self._DTYPES[rec["dtype"]])
        stride = shape[1] * dtype.itemsize
        start = base + rec["data_offsets"][0] + lo * stride
        raw = np.array(
            np.memmap(
                shard,
                dtype=np.uint8,
                mode="r",
                offset=start,
                shape=((hi - lo) * stride,),
            )
        )
        return raw.view(dtype).reshape(hi - lo, shape[1])

    def read_f32(self, name: str) -> np.ndarray:
        """Read one tensor as float32, bf16 bits included.

        Raises:
            KeyError: If the checkpoint has no tensor of that name.
        """
        self._refuse_quantized(name)
        raw = self.read(name)
        if raw.dtype == np.uint16:
            return bf16_bits_to_f32(raw)
        return np.asarray(raw, dtype=np.float32)

    def read_bf16_bits(self, name: str) -> np.ndarray:
        """Read one tensor as bf16 bit patterns, whatever the stored dtype.

        Raises:
            KeyError: If the checkpoint has no tensor of that name.
        """
        self._refuse_quantized(name)
        raw = self.read(name)
        if raw.dtype == np.uint16:
            return raw
        return f32_to_bf16_bits(np.asarray(raw, dtype=np.float32))


# The geometry.json key of the digest of the checkpoint a store's weights were converted from.
CHECKPOINT_DIGEST_KEY = "checkpoint_digest"
# The leading bytes of each shard's largest tensor that `checkpoint_digest` reads.
DIGEST_SAMPLE_BYTES = 1 << 16


def checkpoint_digest(ckpt_dir: str) -> str:
    """md5 that tells this checkpoint's weights from another checkpoint's, read from each shard's head.

    Each shard gives its safetensors header (every tensor's name, dtype, shape and
    place in the shard) and the first ``DIGEST_SAMPLE_BYTES`` of its largest
    tensor. The headers alone tell another architecture, export or format apart,
    but two trainings or two quantizations of one architecture have the same
    headers; their weights differ in every trained or quantized matrix, and a
    shard's largest tensor is one. Hashing every byte would read the whole
    checkpoint (438 GB for GLM-5.2) at each decode launch. The shards enter in the
    order of their own digests, so a copy whose shard files are named otherwise
    has the same digest.

    config.json does not enter: a launcher may serve a view of the checkpoint with
    its own config.json over the same weights, and the decode holds the config to
    the kernel's geometry itself (``layout.config.check_checkpoint``).

    Args:
        ckpt_dir: Directory that holds ``model.safetensors.index.json``.

    Example:
        >>> checkpoint_digest("/path/to/ckpt")  # doctest: +SKIP
        '0f343b0931126a20f133d67c2b018a3b'
    """
    root = ckpt_dir if ckpt_dir.endswith("/") else ckpt_dir + "/"
    with open(root + "model.safetensors.index.json") as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    digests = []
    for shard in shards:
        with open(root + shard, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = f.read(size)
            digest = hashlib.md5(header)
            records = [r for name, r in json.loads(header).items() if name != "__metadata__"]
            if records:
                lo, hi = max(records, key=lambda r: r["data_offsets"][1] - r["data_offsets"][0])["data_offsets"]
                f.seek(8 + size + lo)
                digest.update(f.read(min(hi - lo, DIGEST_SAMPLE_BYTES)))
        digests.append(digest.digest())
    return hashlib.md5(b"".join(sorted(digests))).hexdigest()


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Widen bf16 bit patterns to float32. Exact, because bf16 is a truncation.

    Args:
        bits: Any-shape uint16 array of bf16 bit patterns.

    Returns:
        The same shape in float32.

    Example:
        >>> float(bf16_bits_to_f32(np.array([0x3F80], np.uint16))[0])
        1.0
    """
    wide = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return wide.view(np.float32)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Narrow float32 to bf16 bit patterns with round-to-nearest-even.

    Torch does the rounding, because a hand-rolled shift truncates and a
    truncated weight is a systematic bias, not noise.

    Args:
        values: Any-shape float32 array.

    Returns:
        The same shape in uint16.

    Example:
        >>> int(f32_to_bf16_bits(np.array([1.0], np.float32))[0])
        16256
    """
    t = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    return t.to(torch.bfloat16).view(torch.uint16).numpy()


def _rank_dir(store_dir: str, rank_id: int) -> str:
    root = store_dir if store_dir.endswith("/") else store_dir + "/"
    return f"{root}rank{rank_id:02d}/"


def create_store_files(store_dir: str, geo: Glm52GkGeometry) -> int:
    """Pre-create every rank's store files, so workers only open them ``r+``.

    A worker writes disjoint layers of shared files, so the files must exist
    before the pool starts. `np.lib.format.open_memmap` writes a real header and
    leaves a sparse body, so this costs seconds and no disk.

    Args:
        store_dir: Root of the store.
        geo: The sharding point.

    Returns:
        Logical bytes of the whole store.
    """
    spec = store_spec(geo)
    total = 0
    for rank in range(geo.ranks):
        d = _rank_dir(store_dir, rank)
        os.makedirs(d, exist_ok=True)
        for name, (dtype, shape) in spec.items():
            path = d + name + ".npy"
            if not os.path.exists(path):
                np.lib.format.open_memmap(
                    path, mode="w+", dtype=dtype, shape=shape
                ).flush()
            total += int(np.prod(shape)) * np.dtype(dtype).itemsize
    return total


# Per-layer conversion.
def convert_attention_layer(
    reader: Glm52SafetensorsReader, layer: int, geo: Glm52GkGeometry
) -> dict[str, np.ndarray]:
    """Convert one layer's attention weights, unsharded, to the kernel's layouts.

    Every tensor comes back as bf16 bits, or float32 for the two norms that the
    kernel reads in float32.

    Args:
        reader: The checkpoint reader.
        layer: Model layer index.
        geo: The sharding point.

    Returns:
        The keys ``input_norm_w``, ``gamma``, ``q_norm_w``, ``kv_norm_w``,
        ``wq_a``, ``wkv_a``, ``wq_b``, ``q_absorb``, ``out_absorb`` and
        ``o_proj``.
    """
    p = f"model.layers.{layer}."
    c, n, v = geo.kv_lora_rank, geo.qk_nope_head_dim, geo.v_head_dim
    # kv_a_proj_with_mqa holds [kv_latent | k_pe] rows. The kernel reads the
    # columns as [k_pe | kv_latent], so swap the two blocks after the transpose.
    kv_a = reader.read_bf16_bits(p + "self_attn.kv_a_proj_with_mqa.weight")
    kv_a_t = np.ascontiguousarray(
        np.concatenate(
            [np.ascontiguousarray(kv_a[c:].T), np.ascontiguousarray(kv_a[:c].T)], axis=1
        )
    )
    # kv_b_proj is [num_heads * (nope + v), C]: the nope half absorbs into the
    # query, the v half into the output.
    kv_b = reader.read_bf16_bits(p + "self_attn.kv_b_proj.weight").reshape(
        geo.total_heads, n + v, c
    )
    return {
        "input_norm_w": reader.read_bf16_bits(p + "input_layernorm.weight").reshape(
            1, geo.hidden
        ),
        "gamma": reader.read_bf16_bits(p + "post_attention_layernorm.weight").reshape(
            1, geo.hidden
        ),
        "q_norm_w": reader.read_f32(p + "self_attn.q_a_layernorm.weight").reshape(
            1, geo.q_lora_rank
        ),
        "kv_norm_w": reader.read_f32(p + "self_attn.kv_a_layernorm.weight").reshape(
            1, c
        ),
        "wq_a": np.ascontiguousarray(
            reader.read_bf16_bits(p + "self_attn.q_a_proj.weight").T
        ),
        "wkv_a": kv_a_t,
        "wq_b": np.ascontiguousarray(
            reader.read_bf16_bits(p + "self_attn.q_b_proj.weight").T
        ),
        "q_absorb": np.ascontiguousarray(kv_b[:, :n, :]),
        "out_absorb": np.ascontiguousarray(kv_b[:, n:, :]),
        "o_proj": np.ascontiguousarray(
            reader.read_bf16_bits(p + "self_attn.o_proj.weight").T
        ),
    }


def convert_indexer_layer(
    reader: Glm52SafetensorsReader, layer: int, geo: Glm52GkGeometry
) -> dict[str, np.ndarray]:
    """Convert one full layer's indexer weights, unsharded.

    The indexer's ``k_norm`` is a true ``nn.LayerNorm``, so it has a bias as well
    as a gain, and the kernel reads both in float32.
    """
    p = f"model.layers.{layer}.self_attn.indexer."
    return {
        "idx_wq_b": np.ascontiguousarray(reader.read_bf16_bits(p + "wq_b.weight").T),
        "idx_wk": np.ascontiguousarray(reader.read_bf16_bits(p + "wk.weight").T),
        "idx_weights_proj": np.ascontiguousarray(
            reader.read_bf16_bits(p + "weights_proj.weight").T
        ),
        "idx_k_norm_w": reader.read_f32(p + "k_norm.weight").reshape(
            1, geo.idx_head_dim
        ),
        "idx_k_norm_b": reader.read_f32(p + "k_norm.bias").reshape(1, geo.idx_head_dim),
    }


def convert_shared_expert(
    reader: Glm52SafetensorsReader, layer: int, geo: Glm52GkGeometry
) -> list[tuple[np.ndarray, ...]]:
    """Convert one sparse layer's shared expert for each position of its group.

    The MX layout depends on the shard width (`intermediate_permutation`), so a
    wider group needs a new conversion, not a slice of a narrower one.

    Returns:
        ``geo.shared_degree`` tuples ``(gate_up, gate_up_scale, down, down_scale)``.
    """
    import nki.language as nl  # noqa: PLC0415  device-only import
    import neuron_dtypes  # noqa: F401, PLC0415  registers the x4 numpy dtype

    from sglang_neuron.models.glm52.kernels.decode.mxfp4_checkpoint import (  # noqa: E501, PLC0415
        convert_shared,
    )

    q = f"model.layers.{layer}.mlp.shared_experts."
    parts = [
        reader.read(q + name)
        for name in (
            "gate_proj.weight",
            "gate_proj.weight_scale",
            "up_proj.weight",
            "up_proj.weight_scale",
            "down_proj.weight",
            "down_proj.weight_scale",
        )
    ]
    x4 = np.dtype(nl.float4_e2m1fn_x4)
    return [
        convert_shared(*parts, x4, tp_rank=t, tp_size=geo.shared_degree)
        for t in range(geo.shared_degree)
    ]


def convert_moe_layer(
    reader: Glm52SafetensorsReader, layer: int, geo: Glm52GkGeometry
) -> dict[str, object]:
    """Convert one sparse layer's router and expert weights.

    The routed experts are converted for the whole model (256 of them) and
    scattered by expert-parallel rank. The shared expert is converted once for
    each position of its group (`convert_shared_expert`), because its
    intermediate axis is column-sharded.

    The MX layouts come from the kernel package. `convert_gate_up` reorders the
    intermediate axis so that the down projection loads losslessly, and it packs
    the gate/up scale in the PACK64 layout that the kernel reads.

    Returns:
        ``router_weights``, ``router_bias``, the four routed stacks for every
        expert, and ``shared`` as one entry for each shared-group position.
    """
    import nki.language as nl  # noqa: PLC0415  device-only import
    import neuron_dtypes  # noqa: F401, PLC0415  registers the x4 numpy dtype

    from sglang_neuron.models.glm52.kernels.decode.mxfp4_checkpoint import (  # noqa: E501, PLC0415
        convert_down,
        convert_gate_up,
    )

    x4 = np.dtype(nl.float4_e2m1fn_x4)
    p = f"model.layers.{layer}."
    h, e, i = geo.hidden, geo.num_experts, geo.moe_intermediate
    gu_w = np.empty((e, 128, 2, h // 512, i), np.uint16)
    gu_s = np.empty((e, 2, 64, h // 512, i // 4), np.uint8)
    dn_w = np.empty((e, 128, i // 512, h), np.uint16)
    dn_s = np.empty((e, 64, h), np.uint8)
    for expert in range(e):
        q = f"{p}mlp.experts.{expert}."
        packed, scale = convert_gate_up(
            reader.read(q + "gate_proj.weight"),
            reader.read(q + "gate_proj.weight_scale"),
            reader.read(q + "up_proj.weight"),
            reader.read(q + "up_proj.weight_scale"),
            x4,
        )
        gu_w[expert], gu_s[expert] = packed.view(np.uint16), scale
        packed, scale = convert_down(
            reader.read(q + "down_proj.weight"),
            reader.read(q + "down_proj.weight_scale"),
            x4,
        )
        dn_w[expert], dn_s[expert] = packed.view(np.uint16), scale

    shared = convert_shared_expert(reader, layer, geo)
    return {
        # The router contracts [H, E], and the checkpoint holds [E, H].
        "router_weights": np.ascontiguousarray(
            reader.read_bf16_bits(p + "mlp.gate.weight").T
        ),
        "router_bias": reader.read_f32(p + "mlp.gate.e_score_correction_bias").reshape(
            1, e
        ),
        "routed_gate_up_weights": gu_w,
        "routed_gate_up_weights_scale": gu_s,
        "routed_down_weights": dn_w,
        "routed_down_weights_scale": dn_s,
        "shared": shared,
    }


def convert_dense_layer(
    reader: Glm52SafetensorsReader, layer: int, geo: Glm52GkGeometry
) -> dict[str, np.ndarray]:
    """Convert one dense layer's MLP weights, unsharded and bf16.

    The first ``first_k_dense_replace`` layers of the model keep a dense MLP.
    Its weights are bf16 in the MXFP4 checkpoint, so the MX expert path does not
    apply to them.
    """
    p = f"model.layers.{layer}.mlp."
    return {
        "gate": np.ascontiguousarray(reader.read_bf16_bits(p + "gate_proj.weight").T),
        "up": np.ascontiguousarray(reader.read_bf16_bits(p + "up_proj.weight").T),
        "down": np.ascontiguousarray(reader.read_bf16_bits(p + "down_proj.weight").T),
    }


# The scatter.
def _put(path: str, index: int, value: np.ndarray) -> None:
    array = np.load(path, mmap_mode="r+")
    array[index] = value
    array.flush()


def _scatter_layer(
    store_dir: str,
    geo: Glm52GkGeometry,
    layer: int,
    attn: dict,
    moe: dict | None,
    dense: dict | None,
) -> None:
    """Write one model layer's converted weights into every rank's files."""
    v = geo.v_head_dim
    for rank in range(geo.ranks):
        d_lat = geo.latent_group_position(rank)
        d_q = geo.q_group_position(rank)
        d_out = geo.out_group_position(rank)
        tp = rank % geo.tp_size
        d = _rank_dir(store_dir, rank)

        _put(d + "input_norm_w_all.npy", layer, attn["input_norm_w"])
        if geo.moe_sparse_slots:
            # The MoE stacks hold the sparse layers only (`Glm52GkGeometry.num_moe_layers`).
            if not geo.is_dense(layer):
                _put(d + "gamma_all.npy", geo.moe_slot(layer), attn["gamma"])
        else:
            _put(d + "gamma_all.npy", layer, attn["gamma"])
        _put(d + "q_norm_w_all.npy", layer, attn["q_norm_w"])
        _put(d + "kv_norm_w_all.npy", layer, attn["kv_norm_w"])
        _put(
            d + "wq_a_all.npy",
            layer,
            attn["wq_a"][:, d_lat * geo.wq_a_shard : (d_lat + 1) * geo.wq_a_shard],
        )
        _put(
            d + "wkv_a_all.npy",
            layer,
            attn["wkv_a"][:, d_lat * geo.wkv_a_shard : (d_lat + 1) * geo.wkv_a_shard],
        )
        _put(
            d + "wq_b_all.npy",
            layer,
            attn["wq_b"][:, d_q * geo.wq_b_shard : (d_q + 1) * geo.wq_b_shard],
        )
        heads_q = slice(d_q * geo.q_absorb_heads, (d_q + 1) * geo.q_absorb_heads)
        _put(
            d + "q_absorb_w_all.npy",
            layer,
            np.ascontiguousarray(attn["q_absorb"][heads_q]),
        )
        heads_out = slice(d_out * geo.nh_out, (d_out + 1) * geo.nh_out)
        # [nh_out, v_head, C] -> [C, nh_out, v_head]: the kernel loads the latent
        # tiles as contiguous rows.
        _put(
            d + "out_absorb_w_all.npy",
            layer,
            np.ascontiguousarray(attn["out_absorb"][heads_out].transpose(2, 0, 1)),
        )
        _put(
            d + "o_proj_w_all.npy",
            layer,
            np.ascontiguousarray(
                attn["o_proj"][d_out * geo.nh_out * v : (d_out + 1) * geo.nh_out * v]
            ),
        )
        if moe is not None:
            m = geo.moe_slot(layer)
            # Under `router_tp` the rank holds its TP position's expert columns.
            cols = geo.router_columns
            c0 = tp * cols if geo.router_tp else 0
            _put(
                d + "router_weights_all.npy",
                m,
                np.ascontiguousarray(moe["router_weights"][:, c0 : c0 + cols]),
            )
            _put(d + "router_bias_all.npy", m, moe["router_bias"])
            experts = slice(
                geo.experts_per_rank * rank, geo.experts_per_rank * (rank + 1)
            )
            _put(
                d + "routed_gate_up_weights_all.npy",
                m,
                moe["routed_gate_up_weights"][experts],
            )
            _put(
                d + "routed_gate_up_weights_scale_all.npy",
                m,
                moe["routed_gate_up_weights_scale"][experts],
            )
            _put(
                d + "routed_down_weights_all.npy",
                m,
                moe["routed_down_weights"][experts],
            )
            _put(
                d + "routed_down_weights_scale_all.npy",
                m,
                moe["routed_down_weights_scale"][experts],
            )
            gu, gu_s, dn, dn_s = moe["shared"][rank % geo.shared_degree]
            _put(d + "shared_gate_up_weights_all.npy", m, gu.view(np.uint16))
            _put(d + "shared_gate_up_weights_scale_all.npy", m, gu_s)
            _put(d + "shared_down_weights_all.npy", m, dn.view(np.uint16))
            _put(d + "shared_down_weights_scale_all.npy", m, dn_s)
        if dense is not None:
            dp = rank % geo.dense_degree
            columns = slice(
                dp * geo.dense_intermediate, (dp + 1) * geo.dense_intermediate
            )
            _put(d + "dense_gamma_all.npy", layer, attn["gamma"])
            _put(d + "dense_gate_weights_all.npy", layer, dense["gate"][:, columns])
            _put(d + "dense_up_weights_all.npy", layer, dense["up"][:, columns])
            _put(d + "dense_down_weights_all.npy", layer, dense["down"][columns])


def _scatter_indexer_layer(
    store_dir: str, geo: Glm52GkGeometry, stack_index: int, idx: dict
) -> None:
    """Write one full layer's indexer weights into every rank's files."""
    for rank in range(geo.ranks):
        key_rank = geo.coords(rank)["key_rank"]
        d = _rank_dir(store_dir, rank)
        shard = geo.idx_q_shard
        _put(
            d + "idx_wq_b_all.npy",
            stack_index,
            np.ascontiguousarray(
                idx["idx_wq_b"][:, key_rank * shard : (key_rank + 1) * shard]
            ),
        )
        _put(d + "idx_wk_all.npy", stack_index, idx["idx_wk"])
        _put(d + "idx_weights_proj_all.npy", stack_index, idx["idx_weights_proj"])
        _put(d + "idx_k_norm_w_all.npy", stack_index, idx["idx_k_norm_w"])
        _put(d + "idx_k_norm_b_all.npy", stack_index, idx["idx_k_norm_b"])


def _build_one_layer(job: tuple) -> str:
    """Convert and scatter one model layer. One pool task.

    A sentinel file marks a finished layer, so a build resumes after an
    interruption instead of redoing hours of work.
    """
    ckpt_dir, store_dir, geo, layer = job
    root = store_dir if store_dir.endswith("/") else store_dir + "/"
    done = f"{root}.done_L{layer:03d}"
    if os.path.exists(done):
        return f"L{layer:03d} cached"
    start = time.time()
    reader = Glm52SafetensorsReader(ckpt_dir)
    stack_layer = layer - geo.layer_offset
    attn = convert_attention_layer(reader, layer, geo)
    moe = None if geo.is_dense(stack_layer) else convert_moe_layer(reader, layer, geo)
    dense = (
        convert_dense_layer(reader, layer, geo) if geo.is_dense(stack_layer) else None
    )
    _scatter_layer(store_dir, geo, stack_layer, attn, moe, dense)
    if geo.is_full(stack_layer):
        stack_index = geo.full_layers.index(stack_layer)
        _scatter_indexer_layer(
            store_dir, geo, stack_index, convert_indexer_layer(reader, layer, geo)
        )
    open(done, "w").close()
    kind = "dense" if geo.is_dense(stack_layer) else "moe"
    return f"L{layer:03d} {kind} {time.time() - start:.0f}s"


def build_rank_store(
    ckpt_dir: str,
    store_dir: str,
    geo: Glm52GkGeometry,
    layers: list[int] | None = None,
    workers: int = 12,
) -> dict[str, object]:
    """Build the per-rank GK weight store from an MXFP4 checkpoint.

    The build is resumable: each layer drops a sentinel, and a re-run skips the
    layers that have one. Delete ``<store>/.done_L*`` to force a rebuild.

    Args:
        ckpt_dir: Checkpoint directory, with ``model.safetensors.index.json``.
        store_dir: Root of the store. Use a persistent filesystem.
        geo: The sharding point. The store is valid for this geometry only.
        layers: Model layers to build. Defaults to the whole window.
        workers: Pool size. Each worker holds one layer's expert set in RAM.

    Returns:
        A summary with ``layers``, ``bytes`` and ``seconds``.

    Raises:
        ValueError: If the geometry does not divide (see `Glm52GkGeometry.validate`),
            the checkpoint is the Quark AttnFP8 export, or the store directory holds
            layers of a checkpoint of the other format or of another checkpoint.

    Example:
        >>> build_rank_store("/ckpt", "/store", Glm52GkGeometry())  # doctest: +SKIP
        {'layers': 78, 'bytes': 414..., 'seconds': 1...}
    """
    geo.validate()
    root = store_dir if store_dir.endswith("/") else store_dir + "/"
    os.makedirs(root, exist_ok=True)
    window = layers or list(range(geo.layer_offset, geo.layer_offset + geo.num_layers))
    fmt = _check_build_format(root, ckpt_dir)
    digest = checkpoint_digest(ckpt_dir)
    _check_build_checkpoint(root, ckpt_dir, digest)
    nbytes = create_store_files(store_dir, geo)
    logger.info(
        "[glm52] store %s: %d ranks x %d files, %.1f GB logical",
        root,
        geo.ranks,
        len(store_spec(geo)),
        nbytes / 1e9,
    )
    # The geometry and the checkpoint are written next to the store, so a loader can
    # refuse a store that a different sharding or checkpoint built. A silent
    # mismatch is a wrong answer.
    with open(root + "geometry.json", "w") as f:
        json.dump(
            {
                "ranks": geo.ranks,
                "num_layers": geo.num_layers,
                "dp_size": geo.dp_size,
                "cp_degree": geo.cp_degree,
                "head_shard": geo.head_shard,
                "oproj_tp": geo.oproj_tp,
                "qproj_tp": geo.qproj_tp,
                "latent_tp": geo.latent_tp,
                "tp_size": geo.tp_size,
                "router_tp": geo.router_tp,
                "t_shard": geo.t_shard,
                "context": geo.context,
                "index_share_period": geo.index_share_period,
                "index_skip_topk_offset": geo.index_skip_topk_offset,
                "layer_offset": geo.layer_offset,
                "first_dense_layers": geo.first_dense_layers,
                "checkpoint": ckpt_dir,
                CHECKPOINT_DIGEST_KEY: digest,
                quark_compat.FORMAT_KEY: fmt,
            },
            f,
            indent=1,
        )
    start = time.time()
    jobs = [(ckpt_dir, store_dir, geo, layer) for layer in window]
    if workers <= 1:
        for job in jobs:
            logger.info("[glm52] %s", _build_one_layer(job))
    else:
        with Pool(workers) as pool:
            for message in pool.imap_unordered(_build_one_layer, jobs):
                logger.info("[glm52] %s", message)
    seconds = time.time() - start
    logger.info("[glm52] store built in %.0f s", seconds)
    return {"layers": len(window), "bytes": nbytes, "seconds": seconds}


def _check_build_format(root: str, ckpt_dir: str) -> str:
    """Return the checkpoint format; refuse the Quark AttnFP8 export, and a store directory of the other format.

    A layer with a ``.done_L*`` sentinel is not built again, so a store of the other
    format in the directory would give a store that mixes the two.
    """
    quark_compat.refuse_attn_fp8(ckpt_dir, f"the GK store build into {root}")
    fmt = quark_compat.checkpoint_format(ckpt_dir)
    meta_path = root + "geometry.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            found = quark_compat.store_format(json.load(f))
    elif any(n.startswith(".done_L") for n in os.listdir(root)):
        found = quark_compat.MXFP4
    else:
        return fmt
    quark_compat.check_format(found, fmt, f"store {root}")
    return fmt


def _check_build_checkpoint(root: str, ckpt_dir: str, digest: str) -> None:
    """Refuse a store directory whose finished layers may be another checkpoint's.

    A layer with a ``.done_L*`` sentinel is not built again, so a build over another
    checkpoint's finished layers would give a store that mixes the two under this
    checkpoint's digest. A directory with no finished layer holds nothing to mix.
    """
    if not any(n.startswith(".done_L") for n in os.listdir(root)):
        return
    built = None
    if os.path.exists(root + "geometry.json"):
        with open(root + "geometry.json") as f:
            built = json.load(f).get(CHECKPOINT_DIGEST_KEY)
    if built != digest:
        what = (
            f"another checkpoint ({CHECKPOINT_DIGEST_KEY} {built})"
            if built
            else f"a checkpoint it does not record (no {CHECKPOINT_DIGEST_KEY})"
        )
        known = (
            ""
            if built
            else (
                f', or, for a store you know was converted from it, add "{CHECKPOINT_DIGEST_KEY}": '
                f'"{digest}" to its geometry.json'
            )
        )
        raise ValueError(
            f"store {root} holds layers converted from {what}, and this build "
            f"converts {ckpt_dir} ({CHECKPOINT_DIGEST_KEY} {digest}). A resumed "
            "build skips its finished layers, so the store would mix the two. Build "
            "into a new directory, or delete the store's .done_L* sentinels to "
            f"convert every layer again{known}."
        )


def shared_overlay_dir(store_dir: str, geo: Glm52GkGeometry) -> str:
    """The directory of a store's shared-expert files for a group wider than TP."""
    root = store_dir if store_dir.endswith("/") else store_dir + "/"
    return f"{root}shared_tp{geo.shared_degree}/"


def _build_shared_layer(job: tuple) -> str:
    ckpt_dir, overlay, geo, layer = job
    done = f"{overlay}.done_L{layer:03d}"
    if os.path.exists(done):
        return f"L{layer:03d} cached"
    stack_layer = layer - geo.layer_offset
    shared = convert_shared_expert(Glm52SafetensorsReader(ckpt_dir), layer, geo)
    m = geo.moe_slot(stack_layer)
    for rank in range(geo.ranks):
        d = _rank_dir(overlay, rank)
        gu, gu_s, dn, dn_s = shared[rank % geo.shared_degree]
        _put(d + "shared_gate_up_weights_all.npy", m, gu.view(np.uint16))
        _put(d + "shared_gate_up_weights_scale_all.npy", m, gu_s)
        _put(d + "shared_down_weights_all.npy", m, dn.view(np.uint16))
        _put(d + "shared_down_weights_scale_all.npy", m, dn_s)
    with open(done, "w"):
        pass
    return f"L{layer:03d} shared"


def build_shared_overlay(
    ckpt_dir: str, store_dir: str, geo: Glm52GkGeometry, workers: int = 12
) -> dict[str, object]:
    """Add the shared-expert files for `shared_tp` to a store built for the TP group.

    The rest of the store serves any `shared_tp`. Only the shared expert's MX
    layout depends on its shard width, so only its four files are rebuilt, in
    `shared_overlay_dir`, where `load_rank_weights` finds them.

    Args:
        ckpt_dir: Checkpoint directory.
        store_dir: Root of the existing store.
        geo: The sharding point, with ``shared_tp`` wider than ``tp_size``.
        workers: Pool size.

    Returns:
        A summary with ``layers`` and ``seconds``.

    Raises:
        ValueError: If the store holds the weights of another checkpoint, or records
            none (`check_store_checkpoint`).

    Example:
        >>> build_shared_overlay("/ckpt", "/store", Glm52GkGeometry(shared_tp=16))  # doctest: +SKIP
        {'layers': 75, 'seconds': ...}
    """
    geo.validate()
    root = store_dir if store_dir.endswith("/") else store_dir + "/"
    with open(root + "geometry.json") as f:
        check_store_checkpoint(json.load(f), ckpt_dir, root)
    overlay = shared_overlay_dir(store_dir, geo)
    spec = store_spec(geo)
    for rank in range(geo.ranks):
        d = _rank_dir(overlay, rank)
        os.makedirs(d, exist_ok=True)
        for name in SHARED_FILES:
            if not os.path.exists(d + name + ".npy"):
                dtype, shape = spec[name]
                np.lib.format.open_memmap(
                    d + name + ".npy", mode="w+", dtype=dtype, shape=shape
                ).flush()
    window = range(geo.layer_offset, geo.layer_offset + geo.num_layers)
    jobs = [
        (ckpt_dir, overlay, geo, layer)
        for layer in window
        if not geo.is_dense(layer - geo.layer_offset)
    ]
    start = time.time()
    if workers <= 1:
        for job in jobs:
            logger.info("[glm52] %s", _build_shared_layer(job))
    else:
        with Pool(workers) as pool:
            for message in pool.imap_unordered(_build_shared_layer, jobs):
                logger.info("[glm52] %s", message)
    return {"layers": len(jobs), "seconds": time.time() - start}


# The load.
@dataclass
class Glm52GkWeights:
    """One rank's stacked GK weights, as torch tensors.

    Attributes:
        tensors: The kernel argument name to tensor mapping.
        store_dir: Where the tensors came from, for a log line.
        rank_id: The rank whose shards these are.
    """

    tensors: dict[str, torch.Tensor]
    store_dir: str
    rank_id: int

    def as_kernel_kwargs(self) -> dict[str, torch.Tensor]:
        """Return the mapping that `glm52_gk_decode` takes as ``weights``."""
        return self.tensors

    def to(self, device: str | torch.device) -> "Glm52GkWeights":
        """Move every tensor to a device. Returns self for chaining."""
        self.tensors = {k: v.to(device) for k, v in self.tensors.items()}
        return self


def _as_torch(name: str, array: np.ndarray) -> torch.Tensor:
    """Map one store file to the dtype that the kernel reads."""
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if name in _BF16_FILES:
        return tensor.view(torch.bfloat16)
    # MX tiles stay uint16 here: torch has no float4_e2m1fn_x4, so the decode op
    # reinterprets them inside the trace, where NKI's dtype exists.
    return tensor


ATTN_W8_ENV = "SGLANG_NEURON_GLM52_ATTN_W8"
"""Whether the GK decode serves its two largest attention weights, ``wq_b`` and ``o_proj``, as float8_e4m3fn bytes,
half the bytes a rank streams for them each step. ``1``, the default, serves both in FP8; ``0`` serves them as stored, in
bf16. Each scale is a power of two folded exactly into a neighbouring bf16 / fp32 weight (`wq_b_fp8`, `o_proj_fp8`), so
the e4m3 rounding is the only change, and GSM8K holds within its noise."""


def attn_w8() -> bool:
    """Whether this process serves the FP8 attention weights (`ATTN_W8_ENV`, read at each weight load).

    Raises:
        ValueError: If the setting is neither 0 nor 1.
    """
    raw = os.environ.get(ATTN_W8_ENV, "").strip() or "1"
    if raw not in ("0", "1"):
        raise ValueError(f"{ATTN_W8_ENV}={raw!r}: give 1 (wq_b and o_proj in FP8, the default) or 0 (both in bf16)")
    return raw == "1"


def _pow2_factor(w: torch.Tensor) -> torch.Tensor:
    """Per stacked layer, the smallest power of two ``f`` with ``amax(w) / f <= E4M3FN_MAX``."""
    amax = w.float().reshape(w.shape[0], -1).abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
    return torch.exp2(torch.ceil(torch.log2(amax / E4M3FN_MAX)))


def wq_b_fp8(
    wq_b: torch.Tensor, q_norm_w: torch.Tensor, idx_wq_b: torch.Tensor, full_layers
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One rank's ``wq_b`` stack in FP8, with its per-layer scale folded where it is exact.

    Layer ``l``'s factor is the power of two ``f = 2**ceil(log2(amax / 448))``, so ``wq_b / f`` fits e4m3fn and every
    other use of ``f`` is exact: the q norm's gamma (fp32) takes ``* f``, which scales the normalized latent the matmul
    reads by exactly ``f``, and the indexer's query weight (bf16) of a full layer takes ``/ f``, so the indexer, which
    reads the same normalized latent, computes the same products bit for bit. The only change is the e4m3 rounding of
    ``wq_b / f``.

    Args:
        wq_b: ``[L, q_lora_rank, N]`` bfloat16.
        q_norm_w: ``[L, 1, q_lora_rank]`` float32.
        idx_wq_b: ``[L_full, q_lora_rank, N_idx]`` bfloat16, stack ``j`` belonging to layer ``full_layers[j]``.
        full_layers: The stack indices of the full layers, in order.

    Returns:
        ``wq_b`` as ``[L, q_lora_rank, N]`` uint8 (e4m3fn bits), the scaled ``q_norm_w``, the scaled ``idx_wq_b``.
    """
    factor = _pow2_factor(wq_b)
    q = (wq_b.float() / factor.view(-1, 1, 1)).to(torch.float8_e4m3fn)
    q_norm = q_norm_w * factor.view(-1, 1, 1)
    idx = idx_wq_b.clone()
    for j, layer in enumerate(full_layers):
        idx[j] = (idx_wq_b[j].float() / factor[layer]).to(torch.bfloat16)
    return q.view(torch.uint8).contiguous(), q_norm.contiguous(), idx.contiguous()


def o_proj_fp8(o_proj: torch.Tensor, out_absorb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One rank's ``o_proj`` stack in FP8, its per-layer power-of-two scale folded into the same rank's ``W_UV``.

    The exit computes ``(ctx @ W_UV) @ o_proj`` over this rank's out heads: ``out_absorb`` and ``o_proj`` hold the same
    heads, and nothing else reads ``out_absorb``. So ``W_UV * f`` (bf16 times a power of two: exact) makes the
    absorbed context exactly ``f`` times larger, before and after its bf16 rounding, and ``o_proj / f`` in e4m3 gives
    the same partial sum up to the e4m3 rounding of ``o_proj / f``. The factor is per rank and per layer: each rank's
    partial sum is exact on its own, so the TP reduce-scatter adds the same products.

    Args:
        o_proj: ``[L, nh_out * v_head, H]`` bfloat16, this rank's rows.
        out_absorb: ``[L, C, nh_out, v_head]`` bfloat16, the same heads.

    Returns:
        ``o_proj`` as ``[L, nh_out * v_head, H]`` uint8 (e4m3fn bits) and the scaled ``out_absorb`` (bfloat16).
    """
    factor = _pow2_factor(o_proj)
    q = (o_proj.float() / factor.view(-1, 1, 1)).to(torch.float8_e4m3fn)
    absorb = (out_absorb.float() * factor.view(-1, 1, 1, 1)).to(torch.bfloat16)
    return q.view(torch.uint8).contiguous(), absorb.contiguous()


# The geometry.json fields that a store's slices depend on. The layer window and the indexer cadence (``layer_offset``,
# ``first_dense_layers``, ``index_share_period``, ``index_skip_topk_offset``) can move a layer to another stack slot
# with every shape unchanged, which the load's shape check cannot see: at the default geometry, an
# ``index_skip_topk_offset`` of 2 instead of 3 keeps 21 full layers but moves which ones they are.
_STORE_SHARDING_KEYS = (
    "ranks",
    "cp_degree",
    "oproj_tp",
    "qproj_tp",
    "tp_size",
    "layer_offset",
    "first_dense_layers",
    "index_share_period",
    "index_skip_topk_offset",
)


def check_store_geometry(meta: dict, geo: Glm52GkGeometry, root: str) -> None:
    """Refuse a store whose ``geometry.json`` (``meta``) shards or stacks a weight differently from ``geo``.

    Not the context: no weight shard depends on it, so one store serves every compiled context of its sharding.
    t_shard moves idx_q_shard (through key_size), and the load's shape check refuses that. The latent split
    (``latent_tp``, the ``wq_a`` / ``wkv_a`` columns) may differ only from a token-group-shard store (``latent_tp``
    0), which serves any split by slicing at load; a side store linked by tools/glm52/build_latent_store.py records the
    split it serves. A field that ``meta`` does not record is not compared.

    Raises:
        ValueError: If a `_STORE_SHARDING_KEYS` field of ``meta`` differs from ``geo``'s, or ``meta`` records a latent
            split ``geo`` cannot slice.
    """
    for key in _STORE_SHARDING_KEYS:
        if key in meta and int(meta[key]) != int(getattr(geo, key)):
            raise ValueError(
                f"store {root} was built with {key}={meta[key]}, and this "
                f"geometry has {getattr(geo, key)}. Rebuild the store, or "
                f"correct the geometry."
            )
    if "latent_tp" in meta and int(meta["latent_tp"]) not in (0, int(geo.latent_tp)):
        raise ValueError(
            f"store {root} holds wq_a / wkv_a for latent_tp={meta['latent_tp']}, and this "
            f"geometry serves latent_tp={geo.latent_tp}. Link this split's side store from a "
            f"token-group-shard store with tools/glm52/build_latent_store.py, or rebuild the store."
        )


def check_store_checkpoint(meta: dict, checkpoint_dir: str, root: str) -> None:
    """Refuse a store whose ``geometry.json`` (``meta``) records another checkpoint's digest than ``checkpoint_dir``'s
    (`checkpoint_digest`), or none.

    The prefill reads the checkpoint itself, so a store of another checkpoint's weights at the same shapes (another
    training or quantization of the architecture) would decode finite, wrong tokens.

    Raises:
        ValueError: If ``meta`` has no ``checkpoint_digest``, or another one than ``checkpoint_dir``'s.
    """
    served = checkpoint_digest(checkpoint_dir)
    built = meta.get(CHECKPOINT_DIGEST_KEY)
    if built is None:
        raise ValueError(
            f"store {root} records no {CHECKPOINT_DIGEST_KEY}, so nothing ties its weights to the checkpoint "
            f"{checkpoint_dir} ({CHECKPOINT_DIGEST_KEY} {served}). Build the store from that checkpoint with "
            "build_rank_store(<checkpoint>, <store>, geometry) in a new directory, or, for a store you know was "
            f'converted from it, add "{CHECKPOINT_DIGEST_KEY}": "{served}" to its geometry.json.'
        )
    if built != served:
        raise ValueError(
            f"store {root} holds the weights of the checkpoint {meta.get('checkpoint')} ({CHECKPOINT_DIGEST_KEY} "
            f"{built}), not of {checkpoint_dir} ({served}): a decode that serves {checkpoint_dir} would run other "
            "weights than its prefill reads. Use the store built from that checkpoint, or build one with "
            "build_rank_store(<checkpoint>, <store>, geometry)."
        )


def load_rank_weights(
    store_dir: str,
    geo: Glm52GkGeometry,
    rank_id: int,
    checkpoint_dir: str,
    device: str | torch.device | None = None,
) -> Glm52GkWeights:
    """Load one rank's stacked GK weights from the store.

    The files are read with ``mmap_mode="r"``, so the pages arrive when the copy
    to the device asks for them, and a rank never holds the whole 8 GB in
    anonymous memory.

    Args:
        store_dir: Root of a store that `build_rank_store` wrote.
        geo: The sharding point. It must match the store's ``geometry.json``.
        rank_id: Global rank in ``[0, ranks)``.
        checkpoint_dir: The checkpoint the run serves. The store must hold its
            weights (`check_store_checkpoint`), so of its format too
            (`quark_compat`): the prefill reads the checkpoint itself, so a store
            of another checkpoint or export would decode with weights the
            prefill does not have.
        device: Optional device to copy the tensors to.

    Returns:
        The rank's weights.

    Raises:
        FileNotFoundError: If the store has no ``geometry.json`` or no file for
            an argument.
        ValueError: If the checkpoint is the Quark AttnFP8 export, or the store was
            built for a different geometry, from a checkpoint of another format, or
            from another checkpoint.

    Example:
        >>> load_rank_weights("/store", Glm52GkGeometry(), 0, "/ckpt")  # doctest: +SKIP
    """
    root = store_dir if store_dir.endswith("/") else store_dir + "/"
    meta_path = root + "geometry.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{root} has no geometry.json, which build_rank_store writes before it "
            "converts a layer, so it is not a GK weight store. Build the store for "
            "this geometry with build_rank_store(<checkpoint>, <store>, geometry), "
            "and point SGLANG_NEURON_GLM52_WEIGHT_STORE at it."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    check_store_geometry(meta, geo, root)
    quark_compat.refuse_attn_fp8(checkpoint_dir, f"the GK store load of {root}")
    fmt = quark_compat.store_format(meta)
    quark_compat.check_format(
        fmt, quark_compat.checkpoint_format(checkpoint_dir), f"store {root}"
    )
    check_store_checkpoint(meta, checkpoint_dir, root)
    spec = store_spec(geo)
    d = _rank_dir(store_dir, rank_id)
    tensors: dict[str, torch.Tensor] = {}
    overlay = shared_overlay_dir(store_dir, geo)
    for name, (_dtype, shape) in spec.items():
        path = d + name + ".npy"
        if name in SHARED_FILES and os.path.exists(
            _rank_dir(overlay, rank_id) + name + ".npy"
        ):
            # `shared_tp` on a store built for the TP group (build_shared_overlay).
            path = _rank_dir(overlay, rank_id) + name + ".npy"
        array = np.load(path, mmap_mode="r")
        if (
            name in ("wq_a_all", "wkv_a_all")
            and geo.latent_slices > 1
            and array.shape[-1] == shape[-1] * geo.latent_slices
        ):
            # A store with the token-group shard serves `latent_tp`. This rank's
            # columns lie in the token-group shard of position lat // slices,
            # which every rank of that position holds.
            lat = geo.latent_group_position(rank_id)
            owner = geo.coords(rank_id)["dp"] * geo.attn_size + lat // geo.latent_slices
            array = np.load(_rank_dir(store_dir, owner) + name + ".npy", mmap_mode="r")
            c0 = (lat % geo.latent_slices) * shape[-1]
            array = np.ascontiguousarray(array[..., c0 : c0 + shape[-1]])
        if name in DENSE_MLP_FILES and geo.dense_degree > geo.tp_size:
            # A store with the TP-group shard serves `dense_tp`. This rank's columns
            # lie in the TP shard of position p // ratio, which rank p // ratio holds.
            axis = 1 if name == "dense_down_weights_all" else 2
            ratio = geo.dense_degree // geo.tp_size
            if array.shape[axis] == shape[axis] * ratio:
                p = rank_id % geo.dense_degree
                array = np.load(
                    _rank_dir(store_dir, p // ratio) + name + ".npy", mmap_mode="r"
                )
                c0 = (p % ratio) * shape[axis]
                index = [slice(None)] * array.ndim
                index[axis] = slice(c0, c0 + shape[axis])
                array = np.ascontiguousarray(array[tuple(index)])
        if (
            name == "router_weights_all"
            and array.shape[-1] == geo.num_experts > shape[-1]
        ):
            # A store with the whole router serves `router_tp`: take this rank's
            # TP block of expert columns.
            cols = shape[-1]
            tp = rank_id % geo.tp_size
            array = np.ascontiguousarray(array[..., tp * cols : (tp + 1) * cols])
        if (
            name in MOE_FILES
            and array.shape[0] == geo.num_layers
            and tuple(array.shape[1:]) == tuple(shape[1:])
        ):
            # A store with a MoE slot for every layer (built without
            # `moe_sparse_slots`) serves the sparse-slot stacks: the dense layers
            # lead the window, so their slots are the leading ones. Dropped here,
            # after any column slice above, so one store serves both forms.
            array = array[geo.num_layers - shape[0] :]
        if tuple(array.shape) != tuple(shape):
            raise ValueError(
                f"{path} holds shape {array.shape}, and this geometry needs "
                f"{shape}. Rebuild the store for this geometry."
            )
        tensors[name] = _as_torch(name, np.asarray(array))
    if attn_w8():
        tensors["wq_b_all"], tensors["q_norm_w_all"], tensors["idx_wq_b_all"] = wq_b_fp8(
            tensors["wq_b_all"], tensors["q_norm_w_all"], tensors["idx_wq_b_all"], geo.full_layers
        )
        tensors["o_proj_w_all"], tensors["out_absorb_w_all"] = o_proj_fp8(tensors["o_proj_w_all"], tensors["out_absorb_w_all"])
    weights = Glm52GkWeights(
        tensors=balance_static_kv_stacks(tensors), store_dir=root, rank_id=rank_id
    )
    if device is not None:
        weights = weights.to(device)
    return weights


def zero_rank_weights(
    geo: Glm52GkGeometry, device: str | torch.device | None = None
) -> Glm52GkWeights:
    """Return one rank's stacked GK weights as zeros, with the shapes and dtypes of a store.

    For a run that needs the compiled graph and not the tokens, such as the
    server of a device profile, whose capture makes its own inputs. It reads
    nothing, so it skips the store read.

    Args:
        geo: The sharding point.
        device: Where to allocate the zeros.

    Returns:
        The rank's weights, every one zero.
    """
    tensors: dict[str, torch.Tensor] = {}
    for name, (dtype, shape) in store_spec(geo).items():
        torch_dtype = torch.from_numpy(np.zeros(0, dtype=dtype)).dtype
        tensor = torch.zeros(shape, dtype=torch_dtype, device=device)
        tensors[name] = tensor.view(torch.bfloat16) if name in _BF16_FILES else tensor
    # The FP8 attention weights (ATTN_W8_ENV) load as e4m3 bytes, so their zero twins are uint8 too, and a zeros server
    # compiles the served graph. The neighbours their scales fold into keep their dtypes.
    if attn_w8():
        for name in ("wq_b_all", "o_proj_w_all"):
            tensors[name] = torch.zeros(tensors[name].shape, dtype=torch.uint8, device=device)
    return Glm52GkWeights(tensors=tensors, store_dir="none (zeros)", rank_id=-1)


def load_head_shards(
    ckpt_dir: str, geo: Glm52GkGeometry, rank_id: int
) -> dict[str, np.ndarray]:
    """Load one rank's vocabulary shard of the embedding and of ``lm_head``.

    An engine runs the head ON DEVICE, in each of the 128 rank processes, so
    neither tensor can be replicated: they are 1.9 GB each. Both are sharded by
    ROW over the ranks, which makes the embedding one all-reduce and ``lm_head``
    one all-gather. The final norm is 12 KB, so it stays replicated.

    Args:
        ckpt_dir: Checkpoint directory.
        geo: The sharding point. ``vocab_size`` must divide by ``ranks``.
        rank_id: Global rank in ``[0, ranks)``.

    Returns:
        ``embed_tokens`` and ``lm_head`` as ``[vocab_size / ranks, hidden]``
        uint16 bf16 bits, and ``norm_weight`` as float32.

    Raises:
        ValueError: If the rank count does not divide the vocabulary.

    Example:
        >>> load_head_shards("/ckpt", Glm52GkGeometry(), 0)["lm_head"].shape
        ... # doctest: +SKIP
        (1210, 6144)
    """
    if geo.vocab_size % geo.ranks:
        raise ValueError(
            f"vocab_size {geo.vocab_size} must divide by ranks {geo.ranks} for a "
            "row shard. Pad the vocabulary, or change the rank count."
        )
    rows = geo.vocab_size // geo.ranks
    lo = rank_id * rows
    reader = Glm52SafetensorsReader(ckpt_dir)
    return {
        "embed_tokens": reader.read_rows("model.embed_tokens.weight", lo, lo + rows),
        "lm_head": reader.read_rows("lm_head.weight", lo, lo + rows),
        "norm_weight": reader.read_f32("model.norm.weight").ravel(),
    }

