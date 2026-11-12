# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 MTP draft layer (checkpoint layer 78) to a per-rank weight store.

The draft chain kernel takes each weight as one layer, already sharded for the rank that reads it.
``build_mtp_rank_store`` reads the checkpoint once and writes ``<store>/rank{r:02d}/mtp_<argument>.npy`` for each
argument of ``MTP_WEIGHT_ARGS``, then ``<store>/.done_mtp``; ``load_mtp_rank_weights`` memory-maps them back. The
files sit next to the GK store's and never share a name with them.

Layouts:

- Attention and indexer weights use the GK store's shards for a regular layer, except ``wq_b`` and ``q_absorb_w``,
  which are always token-group shards because the draft steps take the token-group query path
  (``mtp_query_geometry``). The sentinel records that layout, so a store built for another one is refused.
- Routed experts: rank ``r`` holds experts ``[E_L * r, E_L * (r + 1))``; the gate-up weight has ``(gate, up)`` on
  axis 2, and every expert weight is stored transposed, ``[in, out]``.
- Shared expert: TP position ``r % tp_size`` holds intermediate columns ``[t * I_sh, (t + 1) * I_sh)``.
- ``eh_proj_w_t`` is ``eh_proj.weight.T``; ``embed_shard_pad`` is this rank's vocabulary rows with a zero row on
  each side.

bf16 weights are stored as uint16 bits since ``np.save`` cannot round-trip bf16. Norm weights are widened to fp32
(exact), and ``router_bias`` stays fp32 because rounding it to bf16 changes the expert selection.

The experts are only stored in FP8: ``build_mtp_fp8_tensor_store`` quantizes the checkpoint's bf16 experts to
``float8_e4m3fn`` with one ``amax / 448`` factor per weight, written as ``mtp_fp8t_*`` files. No GLM-5.2 role serves
the Quark AttnFP8 export (layout/quark_compat), so every build and load here refuses it.

Example:
    >>> mtp_store_spec(Glm52GkGeometry())["eh_proj_w_t"][0]
    (12288, 6144)
    >>> build_mtp_rank_store("/path/to/ckpt", "/path/to/store", geo)  # doctest: +SKIP
"""

from __future__ import annotations

import json
import logging
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

from sglang_neuron.models.glm52.layout import quark_compat
from sglang_neuron.models.glm52.layout.decode.gk_geometry import Glm52GkGeometry, mtp_query_geometry
from sglang_neuron.models.glm52.layout.decode.weight_loaders_mxfp4 import (
    E4M3FN_MAX,
    Glm52SafetensorsReader,
    bf16_bits_to_f32,
    convert_attention_layer,
    convert_indexer_layer,
    f32_to_bf16_bits,
)

logger = logging.getLogger(__name__)

# Checkpoint layer index of the MTP draft layer: `num_hidden_layers` of GLM-5.2.
MTP_LAYER_INDEX = 78

# Sentinel that `build_mtp_rank_store` writes last. It holds the geometry that
# built the store, and `load_mtp_rank_weights` refuses a store without it.
MTP_DONE = ".done_mtp"

# Prefix of every MTP store file. The GK store files end in `_all`, so the two
# stores can share one directory.
MTP_FILE_PREFIX = "mtp_"

# Zero rows of `embed_shard_pad` around this rank's vocabulary rows: one before
# and one after. The chain kernel asserts the same count.
EMBED_PAD_ROWS = 2

# The chain kernel's weight arguments that `build_mtp_rank_store` writes, with the
# kernel's own names: every one but the experts (the FP8 files, `mtp_fp8t_store_spec`)
# and the lm_head shard, which the target model owns.
MTP_WEIGHT_ARGS: tuple[str, ...] = (
    "eh_proj_w_t",
    "enorm_w",
    "hnorm_w",
    "embed_shard_pad",
    "input_norm_w",
    "wq_a",
    "wkv_a",
    "q_norm_w",
    "kv_norm_w",
    "wq_b",
    "q_absorb_w",
    "out_absorb_w",
    "o_proj_w",
    "idx_wq_b",
    "idx_wk",
    "idx_k_norm_w",
    "idx_k_norm_b",
    "idx_weights_proj",
    "gamma",
    "router_weights",
    "router_bias",
    "shared_head_norm_w",
)

# Arguments that the kernel reads in fp32. The kernel loads each norm weight into
# an fp32 tile. `router_bias` must stay fp32: it is about 15.6 and decides the
# top-k selection, and the bf16 step at that value is 0.0625.
MTP_FP32_ARGS = frozenset(
    {
        "enorm_w",
        "hnorm_w",
        "input_norm_w",
        "q_norm_w",
        "kv_norm_w",
        "idx_k_norm_w",
        "idx_k_norm_b",
        "router_bias",
        "shared_head_norm_w",
    }
)

# The refusal of a Quark AttnFP8 checkpoint or store by the FP8 draft experts.
_QUARK_DRAFT = (
    "{what}: that export ships the MTP layer's experts in MXFP4, and FP8 draft expert files would quantize them a "
    "second time. The MTP draft serves the FP8 experts of the MXFP4 export, the only export the GLM-5.2 prefill "
    "serves too: build the weight store from that checkpoint."
)

# Arguments that the kernel reads in bf16. The store holds their uint16 bits.
MTP_BF16_ARGS = frozenset(MTP_WEIGHT_ARGS) - MTP_FP32_ARGS

# Sentinel and file prefix of the per-tensor FP8 expert files. They sit next to
# the bf16 files.
MTP_FP8T_DONE = ".done_mtp_fp8t"
MTP_FP8T_FILE_PREFIX = "mtp_fp8t_"

# Geometry fields that change a slice. The sentinel records them, and the loader
# compares them with the geometry that it gets.
_GEOMETRY_KEYS = (
    "ranks",
    "dp_size",
    "cp_degree",
    "head_shard",
    "tp_size",
    "t_shard",
    "spec",
    "oproj_tp",
    "qproj_tp",
    "hidden",
    "num_experts",
    "vocab_size",
)


def _sentinel_mismatch(meta: dict, geo: Glm52GkGeometry) -> str:
    """The first `_GEOMETRY_KEYS` field whose value in a sentinel's ``meta`` differs from ``geo``'s, or ``""``."""
    for key in _GEOMETRY_KEYS:
        if key in meta and int(meta[key]) != int(getattr(geo, key)):
            return key
    return ""


def mtp_store_mismatch(
    store_dir: str, geo: Glm52GkGeometry
) -> str:
    """Return how the built MTP store differs from the layout it must have, or ``""``.

    Args:
        store_dir: Root of the store.
        geo: The target sharding point.

    Returns:
        The first geometry key whose recorded value differs, as text. ``""`` when
        the store has no sentinel or the sentinel matches.

    Example:
        >>> mtp_store_mismatch("/store", Glm52GkGeometry())  # doctest: +SKIP
        ''
    """
    done = _root(store_dir) + MTP_DONE
    if not os.path.exists(done):
        return ""
    with open(done) as f:
        meta = json.load(f)
    layer_geo = mtp_query_geometry(geo)
    key = _sentinel_mismatch(meta, layer_geo)
    if not key:
        return ""
    why = (
        " (the MTP draft's query path is the token group's)"
        if key == "qproj_tp" and layer_geo is not geo
        else ""
    )
    return (
        f"it was built with {key}={meta[key]}, and this geometry has "
        f"{getattr(layer_geo, key)}{why}"
    )


def mtp_store_spec(
    geo: Glm52GkGeometry,
) -> dict[str, tuple[tuple[int, ...], type]]:
    """Return ``{argument: (per-rank shape, stored numpy dtype)}``.

    The stored dtype is ``np.uint16`` for a bf16 argument (its bit pattern) and
    ``np.float32`` for an fp32 argument; `load_mtp_rank_weights` gives the kernel
    the bf16 ones as ``torch.bfloat16``. The keys are in `MTP_WEIGHT_ARGS` order.

    Args:
        geo: The sharding point.

    Returns:
        The shape and the stored dtype of every MTP store file.

    Example:
        >>> spec = mtp_store_spec(Glm52GkGeometry())
        >>> spec["eh_proj_w_t"]
        ((12288, 6144), <class 'numpy.uint16'>)
    """
    geo = mtp_query_geometry(geo)
    h = geo.hidden
    v_s = geo.vocab_size // geo.ranks
    shapes: dict[str, tuple[int, ...]] = {
        "eh_proj_w_t": (2 * h, h),
        "enorm_w": (1, h),
        "hnorm_w": (1, h),
        "embed_shard_pad": (v_s + EMBED_PAD_ROWS, h),
        "input_norm_w": (1, h),
        "wq_a": (h, geo.q_lora_shard),
        "wkv_a": (h, geo.kv_row_shard),
        "q_norm_w": (1, geo.q_lora_rank),
        "kv_norm_w": (1, geo.kv_lora_rank),
        "wq_b": (geo.q_lora_rank, geo.wq_b_shard),
        "q_absorb_w": (geo.q_absorb_heads, geo.qk_nope_head_dim, geo.kv_lora_rank),
        "out_absorb_w": (geo.kv_lora_rank, geo.nh_out, geo.v_head_dim),
        "o_proj_w": (geo.nh_out * geo.v_head_dim, h),
        "idx_wq_b": (geo.q_lora_rank, geo.idx_q_shard),
        "idx_wk": (h, geo.idx_head_dim),
        "idx_k_norm_w": (1, geo.idx_head_dim),
        "idx_k_norm_b": (1, geo.idx_head_dim),
        "idx_weights_proj": (h, geo.idx_n_heads),
        "gamma": (1, h),
        "router_weights": (h, geo.num_experts),
        "router_bias": (1, geo.num_experts),
        "shared_head_norm_w": (1, h),
    }
    return {
        name: (shapes[name], np.float32 if name in MTP_FP32_ARGS else np.uint16)
        for name in MTP_WEIGHT_ARGS
    }


def mtp_rank_bytes(geo: Glm52GkGeometry) -> int:
    """Return the bytes of one rank's MTP store files, without the ``.npy`` headers.

    Example:
        >>> round(mtp_rank_bytes(Glm52GkGeometry()) / 1e6)
        217
    """
    return sum(
        int(np.prod(shape)) * np.dtype(dtype).itemsize
        for shape, dtype in mtp_store_spec(geo).values()
    )


def _root(store_dir: str) -> str:
    return store_dir if store_dir.endswith("/") else store_dir + "/"


def _rank_dir(store_dir: str, rank_id: int) -> str:
    return f"{_root(store_dir)}rank{rank_id:02d}/"


def _rank_chunks(geo: Glm52GkGeometry, workers: int) -> list[list[int]]:
    """Split the ranks into at most ``workers`` contiguous runs, one pool task each."""
    return [
        [int(r) for r in chunk]
        for chunk in np.array_split(np.arange(geo.ranks), max(min(workers, geo.ranks), 1))
        if len(chunk)
    ]


def _run(task, jobs: list[tuple], tag: str) -> None:
    """Run the build tasks, in a process pool when there is more than one.

    One task runs in this process: a pool would fork the caller for nothing.
    """
    if len(jobs) == 1:
        logger.info("[%s] %s", tag, task(jobs[0]))
        return
    with Pool(len(jobs)) as pool:
        for message in pool.imap_unordered(task, jobs):
            logger.info("[%s] %s", tag, message)


def _store_format(store_dir: str) -> str | None:
    """The checkpoint format that the GK store or the MTP files of a directory record.

    Returns:
        None for a directory that holds neither.

    Raises:
        ValueError: If the GK store and the MTP files record different formats.
    """
    found = {}
    for name in ("geometry.json", MTP_DONE):
        path = _root(store_dir) + name
        if os.path.exists(path):
            with open(path) as f:
                found[name] = quark_compat.store_format(json.load(f))
    if len(set(found.values())) > 1:
        raise ValueError(
            f"store {_root(store_dir)} mixes checkpoint formats {found}. Rebuild it in a new directory."
        )
    return next(iter(found.values()), None)


def _check_geometry(geo: Glm52GkGeometry) -> None:
    """Refuse a geometry that the MTP slices cannot use.

    Raises:
        ValueError: If the vocabulary or the experts do not divide by the ranks,
            or if the shared expert does not divide by the TP degree.
    """
    geo.validate()
    if geo.vocab_size % geo.ranks:
        raise ValueError(
            f"vocab_size {geo.vocab_size} does not divide by ranks {geo.ranks}. "
            "The embedding shard needs an equal row count on each rank."
        )
    if geo.num_experts < geo.ranks or geo.num_experts % geo.ranks:
        raise ValueError(
            f"num_experts {geo.num_experts} is not a multiple of ranks "
            f"{geo.ranks}. Each rank must hold the same number of routed experts."
        )
    if geo.shared_intermediate_full % geo.tp_size:
        raise ValueError(
            f"shared_intermediate_full {geo.shared_intermediate_full} does not "
            f"divide by tp_size {geo.tp_size}. Change tp_size."
        )


def _store_tensors(layer: int, geo: Glm52GkGeometry) -> list[str]:
    """Return every checkpoint tensor that `build_mtp_rank_store` reads (the experts are the FP8 files')."""
    p = f"model.layers.{layer}."
    names = [
        p + "eh_proj.weight",
        p + "enorm.weight",
        p + "hnorm.weight",
        p + "shared_head.norm.weight",
        p + "input_layernorm.weight",
        p + "post_attention_layernorm.weight",
        p + "self_attn.q_a_proj.weight",
        p + "self_attn.q_a_layernorm.weight",
        p + "self_attn.kv_a_proj_with_mqa.weight",
        p + "self_attn.kv_a_layernorm.weight",
        p + "self_attn.q_b_proj.weight",
        p + "self_attn.kv_b_proj.weight",
        p + "self_attn.o_proj.weight",
        p + "self_attn.indexer.wq_b.weight",
        p + "self_attn.indexer.wk.weight",
        p + "self_attn.indexer.weights_proj.weight",
        p + "self_attn.indexer.k_norm.weight",
        p + "self_attn.indexer.k_norm.bias",
        p + "mlp.gate.weight",
        p + "mlp.gate.e_score_correction_bias",
        "model.embed_tokens.weight",
    ]
    return names


def _required_tensors(layer: int, geo: Glm52GkGeometry) -> list[str]:
    """Return every checkpoint tensor that the MTP store reads: `_store_tensors` and the experts that
    `build_mtp_fp8_tensor_store` quantizes, so a checkpoint without them is refused before either build."""
    p = f"model.layers.{layer}.mlp."
    names = _store_tensors(layer, geo) + [p + f"shared_experts.{proj}.weight" for proj in ("gate_proj", "up_proj",
                                                                                         "down_proj")]
    for expert in range(geo.num_experts):
        names += [f"{p}experts.{expert}.{proj}.weight" for proj in ("gate_proj", "up_proj", "down_proj")]
    return names


def _bf16_matrix(
    reader: Glm52SafetensorsReader, name: str, shape: tuple[int, int]
) -> np.ndarray:
    """Read one 2-D tensor as bf16 bits, and check its shape.

    Raises:
        ValueError: If the checkpoint tensor has a different shape.
    """
    array = reader.read_bf16_bits(name)
    if tuple(array.shape) != tuple(shape):
        raise ValueError(
            f"checkpoint tensor {name} has shape {tuple(array.shape)}, and the "
            f"geometry needs {tuple(shape)}. Check the geometry, or the checkpoint."
        )
    return array


def convert_mtp_layer(
    reader: Glm52SafetensorsReader, layer: int, geo: Glm52GkGeometry
) -> dict[str, np.ndarray]:
    """Convert the MTP-only tensors of one layer, unsharded.

    The attention and indexer tensors come from `convert_attention_layer` and
    `convert_indexer_layer`, and the embedding rows are read for each rank,
    because each rank needs only its own part. The experts are the FP8 files'
    (`build_mtp_fp8_tensor_store`).

    Args:
        reader: The checkpoint reader.
        layer: Checkpoint layer index of the MTP layer.
        geo: The sharding point.

    Returns:
        ``eh_proj_w_t``, ``enorm_w``, ``hnorm_w`` and ``shared_head_norm_w``,
        ``router_weights`` and ``router_bias``. bf16 tensors are uint16 bits.

    Example:
        >>> convert_mtp_layer(reader, 78, Glm52GkGeometry())  # doctest: +SKIP
    """
    p = f"model.layers.{layer}."
    h, e = geo.hidden, geo.num_experts
    eh_proj = _bf16_matrix(reader, p + "eh_proj.weight", (h, 2 * h))
    bias = reader.read_f32(p + "mlp.gate.e_score_correction_bias").reshape(1, e)
    return {
        # [H, 2H] (out, in) to [2H, H]. Input columns [0, H) are the enorm half
        # of concat(enorm(e), hnorm(h)), so they become rows [0, H).
        "eh_proj_w_t": np.ascontiguousarray(eh_proj.T),
        "enorm_w": reader.read_f32(p + "enorm.weight").reshape(1, h),
        "hnorm_w": reader.read_f32(p + "hnorm.weight").reshape(1, h),
        "shared_head_norm_w": reader.read_f32(p + "shared_head.norm.weight").reshape(
            1, h
        ),
        # The router contracts [H, E], and the checkpoint holds [E, H].
        "router_weights": np.ascontiguousarray(
            _bf16_matrix(reader, p + "mlp.gate.weight", (e, h)).T
        ),
        "router_bias": np.ascontiguousarray(bias, dtype=np.float32),
    }


def read_routed_experts(
    reader: Glm52SafetensorsReader, layer: int, geo: Glm52GkGeometry, rank_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read one rank's routed experts in the kernel's bf16 layouts.

    Args:
        reader: The checkpoint reader.
        layer: Checkpoint layer index of the MTP layer.
        geo: The sharding point.
        rank_id: Global rank. It holds experts
            ``[experts_per_rank * rank_id, experts_per_rank * (rank_id + 1))``.

    Returns:
        ``gate_up [E_L, H, 2, I]`` (axis 2: 0 is gate, 1 is up) and
        ``down [E_L, I, H]``, both as uint16 bf16 bits.

    Example:
        >>> gu, dn = read_routed_experts(reader, 78, Glm52GkGeometry(), 77)  # doctest: +SKIP
        >>> gu.shape  # doctest: +SKIP
        (2, 6144, 2, 2048)
    """
    h, i, e_l = geo.hidden, geo.moe_intermediate, geo.experts_per_rank
    p = f"model.layers.{layer}.mlp.experts."
    gate_up = np.empty((e_l, h, 2, i), np.uint16)
    down = np.empty((e_l, i, h), np.uint16)
    for local in range(e_l):
        q = f"{p}{e_l * rank_id + local}."
        gate_up[local, :, 0, :] = _bf16_matrix(reader, q + "gate_proj.weight", (i, h)).T
        gate_up[local, :, 1, :] = _bf16_matrix(reader, q + "up_proj.weight", (i, h)).T
        down[local] = _bf16_matrix(reader, q + "down_proj.weight", (h, i)).T
    return gate_up, down


def read_embed_shard_pad(
    reader: Glm52SafetensorsReader, geo: Glm52GkGeometry, rank_id: int
) -> np.ndarray:
    """Read one rank's padded vocabulary shard of the embedding.

    The MTP layer has no embedding of its own, so the shard comes from
    ``model.embed_tokens.weight``. It holds the same rows as the target's
    embedding shard (`load_head_shards`).

    Args:
        reader: The checkpoint reader.
        geo: The sharding point.
        rank_id: Global rank.

    Returns:
        ``[V_s + 2, H]`` uint16 bf16 bits. Row 0 and row ``V_s + 1`` are zero.
        Rows ``1 .. V_s`` are vocabulary rows ``[V_s * rank_id, V_s * (rank_id + 1))``.

    Example:
        >>> read_embed_shard_pad(reader, Glm52GkGeometry(), 0).shape  # doctest: +SKIP
        (1212, 6144)
    """
    v_s = geo.vocab_size // geo.ranks
    rows = reader.read_rows(
        "model.embed_tokens.weight", v_s * rank_id, v_s * (rank_id + 1)
    )
    if rows.dtype != np.uint16:
        rows = f32_to_bf16_bits(np.asarray(rows, dtype=np.float32))
    if rows.shape[1] != geo.hidden:
        raise ValueError(
            f"model.embed_tokens.weight has {rows.shape[1]} columns, and the "
            f"geometry has hidden {geo.hidden}. Check the geometry."
        )
    padded = np.zeros((v_s + EMBED_PAD_ROWS, geo.hidden), np.uint16)
    padded[1 : v_s + 1] = rows
    return padded


def mtp_rank_arrays(
    reader: Glm52SafetensorsReader,
    geo: Glm52GkGeometry,
    rank_id: int,
    attn: dict[str, np.ndarray],
    idx: dict[str, np.ndarray],
    mtp: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return one rank's MTP weights, in the stored layouts and dtypes.

    The attention and indexer slices repeat the slices of
    `weight_loaders_mxfp4._scatter_layer` and
    `weight_loaders_mxfp4._scatter_indexer_layer`, so the MTP layer and a GK
    layer get the same shard on the same rank, except ``wq_b`` and ``q_absorb_w``,
    which are the token-group shards (`mtp_query_geometry`).

    Args:
        reader: The checkpoint reader, for the embedding rows.
        geo: The sharding point.
        rank_id: Global rank.
        attn: `convert_attention_layer` output for the MTP layer.
        idx: `convert_indexer_layer` output for the MTP layer.
        mtp: `convert_mtp_layer` output for the MTP layer.

    Returns:
        ``{argument: array}`` for every name in `MTP_WEIGHT_ARGS`.

    Example:
        >>> arrays = mtp_rank_arrays(reader, geo, 0, attn, idx, mtp)  # doctest: +SKIP
    """
    v = geo.v_head_dim
    coords = geo.coords(rank_id)
    pig = coords["pos_in_group"]
    key_rank = coords["key_rank"]
    # the draft's query path: the token group's
    q_geo = mtp_query_geometry(geo)
    d_q = q_geo.q_group_position(rank_id)
    d_out = geo.out_group_position(rank_id)

    heads_q = slice(d_q * q_geo.q_absorb_heads, (d_q + 1) * q_geo.q_absorb_heads)
    heads_out = slice(d_out * geo.nh_out, (d_out + 1) * geo.nh_out)
    q_shard = slice(pig * geo.q_lora_shard, (pig + 1) * geo.q_lora_shard)
    kv_shard = slice(pig * geo.kv_row_shard, (pig + 1) * geo.kv_row_shard)
    wq_b_cols = slice(d_q * q_geo.wq_b_shard, (d_q + 1) * q_geo.wq_b_shard)
    idx_cols = slice(key_rank * geo.idx_q_shard, (key_rank + 1) * geo.idx_q_shard)

    arrays = {
        "eh_proj_w_t": mtp["eh_proj_w_t"],
        "enorm_w": mtp["enorm_w"],
        "hnorm_w": mtp["hnorm_w"],
        "embed_shard_pad": read_embed_shard_pad(reader, geo, rank_id),
        # The GK store keeps input_layernorm in bf16. The MTP kernel reads it in
        # fp32, and the widening is exact.
        "input_norm_w": bf16_bits_to_f32(attn["input_norm_w"]),
        "wq_a": attn["wq_a"][:, q_shard],
        "wkv_a": attn["wkv_a"][:, kv_shard],
        "q_norm_w": attn["q_norm_w"],
        "kv_norm_w": attn["kv_norm_w"],
        "wq_b": attn["wq_b"][:, wq_b_cols],
        "q_absorb_w": attn["q_absorb"][heads_q],
        # [nh_out, v_head, C] to [C, nh_out, v_head]: the kernel loads the latent
        # tiles as contiguous rows.
        "out_absorb_w": attn["out_absorb"][heads_out].transpose(2, 0, 1),
        "o_proj_w": attn["o_proj"][
            d_out * geo.nh_out * v : (d_out + 1) * geo.nh_out * v
        ],
        "idx_wq_b": idx["idx_wq_b"][:, idx_cols],
        "idx_wk": idx["idx_wk"],
        "idx_k_norm_w": idx["idx_k_norm_w"],
        "idx_k_norm_b": idx["idx_k_norm_b"],
        "idx_weights_proj": idx["idx_weights_proj"],
        "gamma": attn["gamma"],
        "router_weights": mtp["router_weights"],
        "router_bias": mtp["router_bias"],
        "shared_head_norm_w": mtp["shared_head_norm_w"],
    }
    return {name: np.ascontiguousarray(arrays[name]) for name in MTP_WEIGHT_ARGS}


def mtp_fp8t_store_spec(
    geo: Glm52GkGeometry,
) -> dict[str, tuple[tuple[int, ...], type]]:
    """Return ``{argument: (per-rank shape, stored numpy dtype)}`` of the per-tensor FP8 expert files.

    Example:
        >>> mtp_fp8t_store_spec(Glm52GkGeometry())["routed_fp8_scales"]
        ((2, 3), <class 'numpy.float32'>)
    """
    h = geo.hidden
    e_l, i = geo.experts_per_rank, geo.moe_intermediate
    i_sh = geo.shared_intermediate
    return {
        "routed_gate_up_weights": ((e_l, h, 2, i), np.uint8),
        "routed_down_weights": ((e_l, i, h), np.uint8),
        "shared_gate_up_weights": ((1, h, 2, i_sh), np.uint8),
        "shared_down_weights": ((1, i_sh, h), np.uint8),
        "routed_fp8_scales": ((e_l, 3), np.float32),
        "shared_fp8_scales": ((1, 3), np.float32),
    }


def f32_to_e4m3fn_bits(x: np.ndarray) -> np.ndarray:
    """fp32 values in ``[-448, 448]`` to OCP ``float8_e4m3fn`` bits, rounding to nearest even.

    Pure numpy: the store builds in forked pool workers, where a torch cast can deadlock on the parent's
    thread pool. Normal values keep 3 mantissa bits (exponent bias 7, smallest normal ``2**-6``);
    smaller ones round to the ``2**-9`` subnormal grid.

    Raises:
        ValueError: If a value is outside the e4m3fn range (the caller scales by ``amax / 448`` first).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.all(np.abs(x) <= E4M3FN_MAX):
        raise ValueError(
            "f32_to_e4m3fn_bits takes values in [-448, 448]; scale the weight first."
        )
    a = np.abs(x)
    # Normals: round the fp32 mantissa to 3 bits, to nearest even, in the bit pattern.
    b = a.view(np.uint32)
    b = (b + np.uint32(0x7FFFF) + ((b >> np.uint32(20)) & np.uint32(1))) & np.uint32(
        0xFFF00000
    )
    rounded = b.view(np.float32)
    # Subnormals (and the values that round up into the smallest normal): the 2**-9 grid.
    sub = a < 2.0**-6
    rounded = np.where(sub, np.rint(a * 2.0**9) * 2.0**-9, rounded).astype(np.float32)
    mant, exp = np.frexp(rounded)  # rounded = mant * 2**exp, mant in [0.5, 1)
    unbiased = exp - 1
    normal = (rounded >= 2.0**-6) & (rounded > 0)
    e_field = np.where(normal, unbiased + 7, 0).astype(np.int64)
    m_field = np.where(
        normal, np.rint((mant * 2.0 - 1.0) * 8.0), np.rint(rounded * 2.0**9)
    ).astype(np.int64)
    sign = (x.view(np.uint32) >> np.uint32(31)).astype(np.int64)
    return ((sign << 7) | (e_field << 3) | m_field).astype(np.uint8)


def quantize_fp8_tensor(bits16: np.ndarray) -> tuple[np.ndarray, float]:
    """bf16 bits to ``float8_e4m3fn`` bits with one factor, ``amax / 448`` (round to nearest even).

    Returns:
        ``(bits, factor)``: uint8 FP8 bits of the same shape, and the fp32 factor; the dequantized weight is
        ``fp8(bits) * factor``.
    """
    w = bf16_bits_to_f32(bits16)
    amax = float(np.abs(w).max())
    factor = np.float32(amax / E4M3FN_MAX if amax > 0 else 1.0)
    scaled = np.clip(w / factor, -E4M3FN_MAX, E4M3FN_MAX).astype(np.float32)
    return f32_to_e4m3fn_bits(scaled), float(factor)


def read_fp8_tensor_experts(
    reader: Glm52SafetensorsReader,
    layer: int,
    geo: Glm52GkGeometry,
    rank_id: int,
    shared: dict[str, tuple[np.ndarray, float]],
) -> dict[str, np.ndarray]:
    """One rank's experts re-quantized with one factor per weight, in the kernel layouts.

    Args:
        reader: A reader of the bf16 checkpoint (layer 78 is bf16 in GLM-5.2 MXFP4).
        layer: Checkpoint layer index of the MTP layer.
        geo: The sharding point.
        rank_id: Global rank.
        shared: The whole shared expert, quantized once: ``{proj: ([out, in] bits, factor)}`` (see
            `_fp8_tensor_shared`).

    Returns:
        ``{argument: array}`` for every name in `mtp_fp8t_store_spec`.
    """
    spec = mtp_fp8t_store_spec(geo)
    out = {name: np.empty(shape, dtype) for name, (shape, dtype) in spec.items()}
    gate_up, down = read_routed_experts(reader, layer, geo, rank_id)
    for local in range(geo.experts_per_rank):
        for g in range(2):
            bits, factor = quantize_fp8_tensor(gate_up[local, :, g, :])
            out["routed_gate_up_weights"][local, :, g, :] = bits
            out["routed_fp8_scales"][local, g] = factor
        bits, factor = quantize_fp8_tensor(down[local])
        out["routed_down_weights"][local] = bits
        out["routed_fp8_scales"][local, 2] = factor
    i_sh = geo.shared_intermediate
    tp = rank_id % geo.tp_size
    cols = slice(tp * i_sh, (tp + 1) * i_sh)
    for g, proj in enumerate(("gate_proj", "up_proj")):
        bits, factor = shared[proj]
        out["shared_gate_up_weights"][0, :, g, :] = bits[cols].T
        out["shared_fp8_scales"][0, g] = factor
    bits, factor = shared["down_proj"]
    out["shared_down_weights"][0] = bits[:, cols].T
    out["shared_fp8_scales"][0, 2] = factor
    return out


def _fp8_tensor_shared(
    reader: Glm52SafetensorsReader, layer: int, geo: Glm52GkGeometry
) -> dict[str, tuple[np.ndarray, float]]:
    """The whole shared expert, ``[out, in]``, quantized with one factor per weight (shared by every TP shard)."""
    p = f"model.layers.{layer}.mlp.shared_experts."
    h, i_full = geo.hidden, geo.shared_intermediate_full
    shapes = {
        "gate_proj": (i_full, h),
        "up_proj": (i_full, h),
        "down_proj": (h, i_full),
    }
    return {
        proj: quantize_fp8_tensor(_bf16_matrix(reader, f"{p}{proj}.weight", shape))
        for proj, shape in shapes.items()
    }


def _build_fp8t_ranks(job: tuple) -> str:
    """Write the per-tensor FP8 expert files of some ranks. One pool task."""
    ckpt_dir, store_dir, geo, layer, ranks = job
    start = time.time()
    reader = Glm52SafetensorsReader(ckpt_dir)
    shared = _fp8_tensor_shared(reader, layer, geo)
    nbytes = 0
    for rank in ranks:
        d = _rank_dir(store_dir, rank)
        os.makedirs(d, exist_ok=True)
        for name, array in read_fp8_tensor_experts(
            reader, layer, geo, rank, shared
        ).items():
            _save(f"{d}{MTP_FP8T_FILE_PREFIX}{name}.npy", array)
            nbytes += array.nbytes
    return (
        f"ranks {ranks[0]}..{ranks[-1]}: {nbytes / 1e9:.2f} GB in "
        f"{time.time() - start:.0f} s"
    )


def build_mtp_fp8_tensor_store(
    ckpt_dir: str,
    store_dir: str,
    geo: Glm52GkGeometry,
    workers: int = 16,
    *,
    layer: int = MTP_LAYER_INDEX,
    force: bool = False,
) -> None:
    """Quantize the MTP experts of the bf16 checkpoint layer to per-tensor FP8 files.

    Writes ``<store>/rank{r:02d}/mtp_fp8t_<argument>.npy`` for every rank, then ``<store>/.done_mtp_fp8t``. Does
    nothing if that sentinel exists, unless ``force``.

    Raises:
        ValueError: The checkpoint is a Quark AttnFP8 export, the store holds another format's weights, or the
            geometry cannot shard the layer.
    """
    _check_geometry(geo)
    root = _root(store_dir)
    done = root + MTP_FP8T_DONE
    if quark_compat.checkpoint_format(ckpt_dir) != quark_compat.MXFP4:
        raise ValueError(
            _QUARK_DRAFT.format(what=f"checkpoint {ckpt_dir} is a Quark AttnFP8 export")
        )
    found = _store_format(store_dir)
    if found is not None:
        quark_compat.check_format(found, quark_compat.MXFP4, f"store {root}")
    if os.path.exists(done) and not force:
        logger.info(
            "[glm52 mtp] store %s: %s exists, nothing to build", root, MTP_FP8T_DONE
        )
        return
    os.makedirs(root, exist_ok=True)
    start = time.time()
    jobs = [(ckpt_dir, store_dir, geo, layer, chunk) for chunk in _rank_chunks(geo, workers)]
    _run(_build_fp8t_ranks, jobs, "glm52 mtp fp8t")
    meta = {key: int(getattr(geo, key)) for key in _GEOMETRY_KEYS}
    meta.update(
        {
            "layer": layer,
            "checkpoint": ckpt_dir,
            quark_compat.FORMAT_KEY: quark_compat.MXFP4,
            "args": list(mtp_fp8t_store_spec(geo)),
            "seconds": round(time.time() - start, 1),
            "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    with open(done + ".tmp", "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(done + ".tmp", done)
    logger.info("[glm52 mtp fp8t] store built in %.0f s", time.time() - start)


def load_mtp_fp8_tensor_expert_weights(
    store_dir: str,
    geo: Glm52GkGeometry,
    rank_id: int,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load one rank's FP8 draft experts: the per-tensor FP8 expert files of `build_mtp_fp8_tensor_store`.

    The four expert weights arrive as uint8 e4m3fn bits (the draft's only expert files) with the two fp32 factor
    tables. The files are memory-mapped copy-on-write, as `load_mtp_rank_weights`' files are.

    Raises:
        FileNotFoundError: If the store has no ``.done_mtp_fp8t`` or no file for an argument.
        ValueError: If the store is from a Quark AttnFP8 checkpoint, was built for a different geometry, or a file
            has the wrong shape or dtype.

    Example:
        >>> load_mtp_fp8_tensor_expert_weights("/store", Glm52GkGeometry(), 0)  # doctest: +SKIP
    """
    root = _root(store_dir)
    if _store_format(store_dir) == quark_compat.QUARK_ATTN_FP8:
        raise ValueError(
            _QUARK_DRAFT.format(what=f"store {root} is from a Quark AttnFP8 checkpoint")
        )
    done = root + MTP_FP8T_DONE
    if not os.path.exists(done):
        raise FileNotFoundError(
            f"weight store {root} has no {MTP_FP8T_DONE}, so it holds no FP8 draft experts, which the MTP draft "
            "serves. Build them into the store from its checkpoint with "
            "build_mtp_fp8_tensor_store(<checkpoint>, <store>, geometry)."
        )
    with open(done) as f:
        meta = json.load(f)
    key = _sentinel_mismatch(meta, geo)
    if key:
        raise ValueError(
            f"the FP8 draft experts of {root} were built with {key}={meta[key]}, and this geometry has "
            f"{getattr(geo, key)}. Build them again for this geometry with "
            "build_mtp_fp8_tensor_store(<checkpoint>, <store>, geometry, force=True)."
        )
    d = _rank_dir(store_dir, rank_id)
    tensors: dict[str, torch.Tensor] = {}
    for name, (shape, dtype) in mtp_fp8t_store_spec(geo).items():
        path = f"{d}{MTP_FP8T_FILE_PREFIX}{name}.npy"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FP8 draft expert file {path} does not exist. Build the files again with "
                "build_mtp_fp8_tensor_store(<checkpoint>, <store>, geometry, force=True)."
            )
        array = np.load(path, mmap_mode="c")
        if tuple(array.shape) != tuple(shape) or array.dtype != np.dtype(dtype):
            raise ValueError(
                f"{path} holds {array.dtype}{tuple(array.shape)}, and this geometry needs "
                f"{np.dtype(dtype)}{tuple(shape)}. Build the files again with "
                "build_mtp_fp8_tensor_store(<checkpoint>, <store>, geometry, force=True)."
            )
        tensor = torch.from_numpy(array)
        if torch.device(device).type != "cpu":
            tensor = tensor.to(device)
        tensors[name] = tensor
    return tensors


def _save(path: str, array: np.ndarray) -> None:
    """Write one ``.npy`` file through a temporary name, so a crash leaves no torn file."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def _build_ranks(job: tuple) -> str:
    """Convert the MTP layer and write the files of some ranks. One pool task.

    Each task converts the replicated tensors itself, so the job holds only
    paths, the geometry and rank indices. The pool start method does not matter.
    """
    ckpt_dir, store_dir, geo, layer, ranks = job
    start = time.time()
    reader = Glm52SafetensorsReader(ckpt_dir)
    attn = convert_attention_layer(reader, layer, geo)
    idx = convert_indexer_layer(reader, layer, geo)
    mtp = convert_mtp_layer(reader, layer, geo)
    spec = mtp_store_spec(geo)
    nbytes = 0
    for rank in ranks:
        arrays = mtp_rank_arrays(reader, geo, rank, attn, idx, mtp)
        d = _rank_dir(store_dir, rank)
        os.makedirs(d, exist_ok=True)
        for name, (shape, dtype) in spec.items():
            array = arrays[name]
            if tuple(array.shape) != tuple(shape) or array.dtype != np.dtype(dtype):
                raise RuntimeError(
                    f"rank {rank} {name} has {array.dtype}{tuple(array.shape)}, and "
                    f"the spec is {np.dtype(dtype)}{tuple(shape)}. Correct "
                    f"mtp_rank_arrays."
                )
            _save(f"{d}{MTP_FILE_PREFIX}{name}.npy", array)
            nbytes += array.nbytes
    return (
        f"ranks {ranks[0]}..{ranks[-1]}: {nbytes / 1e9:.2f} GB in "
        f"{time.time() - start:.0f} s"
    )


def build_mtp_rank_store(
    ckpt_dir: str,
    store_dir: str,
    geo: Glm52GkGeometry,
    workers: int = 16,
    *,
    layer: int = MTP_LAYER_INDEX,
    force: bool = False,
) -> None:
    """Build the per-rank MTP weight store from the checkpoint.

    Writes ``<store>/rank{r:02d}/mtp_<argument>.npy`` for every rank, then ``<store>/.done_mtp``. Does nothing if the
    sentinel exists, unless ``force``. Each pool task converts the replicated tensors once and writes a contiguous
    range of ranks.

    Raises:
        ValueError: The checkpoint is a Quark AttnFP8 export, or the geometry cannot shard the layer.
        KeyError: The checkpoint is missing an MTP tensor.
    """
    _check_geometry(geo)
    root = _root(store_dir)
    done = root + MTP_DONE
    if os.path.exists(done) and not force:
        logger.info("[glm52 mtp] store %s: %s exists, nothing to build", root, MTP_DONE)
        return
    quark_compat.refuse_attn_fp8(ckpt_dir, f"the MTP store build into {root}")
    reader = Glm52SafetensorsReader(ckpt_dir)
    found = _store_format(store_dir)
    if found is not None:
        quark_compat.check_format(found, reader.format, f"store {root}")
    missing = [name for name in _required_tensors(layer, geo) if not reader.has(name)]
    if missing:
        raise KeyError(
            f"checkpoint {ckpt_dir} has no tensor {missing[0]!r} "
            f"({len(missing)} MTP tensors are missing). Check the layer index "
            f"({layer}) and the checkpoint revision."
        )
    os.makedirs(root, exist_ok=True)
    if os.path.exists(done):
        os.remove(done)

    start = time.time()
    jobs = [(ckpt_dir, store_dir, geo, layer, chunk) for chunk in _rank_chunks(geo, workers)]
    logger.info(
        "[glm52 mtp] store %s: layer %d, %d ranks x %d files, %.1f GB",
        root,
        layer,
        geo.ranks,
        len(MTP_WEIGHT_ARGS),
        geo.ranks * mtp_rank_bytes(geo) / 1e9,
    )
    _run(_build_ranks, jobs, "glm52 mtp")
    seconds = time.time() - start

    # The layout's own geometry: the rows chain's store records its token-group qproj_tp.
    layer_geo = mtp_query_geometry(geo)
    meta = {key: int(getattr(layer_geo, key)) for key in _GEOMETRY_KEYS}
    meta.update(
        {
            "layer": layer,
            "checkpoint": ckpt_dir,
            "args": list(MTP_WEIGHT_ARGS),
            "rank_bytes": mtp_rank_bytes(geo),
            "seconds": round(seconds, 1),
            "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
            quark_compat.FORMAT_KEY: reader.format,
        }
    )
    with open(done + ".tmp", "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(done + ".tmp", done)
    logger.info("[glm52 mtp] store built in %.0f s", seconds)


def load_mtp_rank_weights(
    store_dir: str,
    geo: Glm52GkGeometry,
    rank_id: int,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load one rank's MTP weights from the store.

    The files are memory-mapped copy-on-write, so the pages arrive when a copy
    asks for them, and a write to a tensor does not change the file.

    Args:
        store_dir: Root of a store that `build_mtp_rank_store` wrote.
        geo: The sharding point. Its MTP layout (`mtp_query_geometry`) must match
            the geometry in ``.done_mtp``.
        rank_id: Global rank in ``[0, ranks)``.
        device: Device for the tensors. ``"cpu"`` keeps the memory maps.

    Returns:
        ``{argument: tensor}`` in `MTP_WEIGHT_ARGS` order. bf16 arguments are
        ``torch.bfloat16`` and fp32 arguments are ``torch.float32``.

    Raises:
        FileNotFoundError: If the store has no ``.done_mtp`` or no file for an
            argument.
        ValueError: If the store is from a Quark AttnFP8 checkpoint, was built for a
            different geometry, or a file has the wrong shape or dtype.

    Example:
        >>> w = load_mtp_rank_weights("/store", Glm52GkGeometry(), 0)  # doctest: +SKIP
        >>> w["router_bias"].dtype  # doctest: +SKIP
        torch.float32
    """
    root = _root(store_dir)
    done = root + MTP_DONE
    if not os.path.exists(done):
        raise FileNotFoundError(
            f"MTP weight store {root} has no {MTP_DONE} sentinel, so the draft's files were not built, or their "
            "build did not finish. Build them into the store from its checkpoint with "
            "build_mtp_rank_store(<checkpoint>, <store>, geometry)."
        )
    if not 0 <= rank_id < geo.ranks:
        raise ValueError(f"rank_id {rank_id} is outside [0, {geo.ranks}).")
    # The store records its token-group qproj_tp, so a store of another query layout is
    # refused here, with the field named.
    mismatch = mtp_store_mismatch(store_dir, geo)
    if mismatch:
        raise ValueError(
            f"MTP store {root}: {mismatch}. Build the draft's files again for this geometry with "
            "build_mtp_rank_store(<checkpoint>, <store>, geometry, force=True), "
            "or correct the geometry."
        )
    if _store_format(store_dir) == quark_compat.QUARK_ATTN_FP8:
        raise ValueError(f"MTP store {root} is from the Quark AttnFP8 export, which no GLM-5.2 role reads; build the "
                         "store from the MXFP4 export.")
    d = _rank_dir(store_dir, rank_id)
    tensors: dict[str, torch.Tensor] = {}
    for name, (shape, dtype) in mtp_store_spec(geo).items():
        path = f"{d}{MTP_FILE_PREFIX}{name}.npy"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"MTP store file {path} does not exist. Build the draft's files again with "
                "build_mtp_rank_store(<checkpoint>, <store>, geometry, force=True)."
            )
        array = np.load(path, mmap_mode="c")
        if tuple(array.shape) != tuple(shape) or array.dtype != np.dtype(dtype):
            raise ValueError(
                f"{path} holds {array.dtype}{tuple(array.shape)}, and this geometry needs {np.dtype(dtype)}"
                f"{tuple(shape)}. Build the draft's files again for this geometry with "
                "build_mtp_rank_store(<checkpoint>, <store>, geometry, force=True)."
            )
        tensor = torch.from_numpy(array)
        if name in MTP_BF16_ARGS:
            tensor = tensor.view(torch.bfloat16)
        if torch.device(device).type != "cpu":
            tensor = tensor.to(device)
        tensors[name] = tensor
    return tensors
