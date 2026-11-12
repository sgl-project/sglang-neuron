# SPDX-License-Identifier: Apache-2.0
"""Hold a GLM-5.2-family checkpoint to the decode giga-kernel (GK) geometry that serves it, and bound its context.

The decode does not derive its geometry from the checkpoint. It serves its launch's point
(``SGLANG_NEURON_GLM52_POINT``, models/glm52/points.py), whose geometry is built for GLM-5.2's shapes, and the weight
store, the KV formats and the PD hand-off's byte layout all follow that point. So a checkpoint is served only where
every value the kernel is traced with is the checkpoint's own (`check_checkpoint`). A shape that differs fails far
from its cause, as a trace-time assert in the kernel. A constant that differs fails silently: a checkpoint with
another routed scaling factor, indexer schedule or RoPE base decodes finite, wrong tokens with the kernel's values.

Example:
    >>> check_checkpoint(Glm52GkGeometry(), hf_config)  # doctest: +SKIP
    >>> max_served_model_len(Glm52GkGeometry())
    130944
"""

from __future__ import annotations

from typing import Any

from sglang_neuron.models.glm52.layout.decode.gk_geometry import (
    IDX_KEY_TILE,
    IDX_SNAKE_PARTITIONS,
    IDX_TOPK_CHUNK,
    Glm52GkGeometry,
    idx_topk_chunking,
)

#: The checkpoint's scalar fields, each with the geometry attribute the kernel is traced with in its place.
CHECKPOINT_FIELDS = {
    "num_hidden_layers": "num_layers",
    "hidden_size": "hidden",
    "num_attention_heads": "total_heads",
    "q_lora_rank": "q_lora_rank",
    "kv_lora_rank": "kv_lora_rank",
    "qk_nope_head_dim": "qk_nope_head_dim",
    "qk_rope_head_dim": "qk_rope_head_dim",
    "v_head_dim": "v_head_dim",
    "n_routed_experts": "num_experts",
    "num_experts_per_tok": "top_k",
    "moe_intermediate_size": "moe_intermediate",
    "vocab_size": "vocab_size",
    "index_topk": "index_topk",
    "index_n_heads": "idx_n_heads",
    "index_head_dim": "idx_head_dim",
    "first_k_dense_replace": "first_dense_layers",
    "intermediate_size": "dense_intermediate_full",
    "routed_scaling_factor": "routed_scaling_factor",
}


def checkpoint_problems(geo: Glm52GkGeometry, hf_config: Any) -> list[str]:
    """Each way the checkpoint's config differs from what the decode kernel is traced with, [] when it matches.

    The GLM-5.2 checkpoint's config, as SGLang hands it to the host, is transformers' ``GlmMoeDsaConfig``, which
    declares every field read here but ``scoring_func``. The per-layer schedules are compared as the lists the
    model runs (``indexer_types``, ``mlp_layer_types``): transformers derives them (from ``index_topk_pattern``, or
    ``index_topk_freq`` and ``index_skip_topk_offset``, and from ``first_k_dense_replace``) only when the config.json
    does not list them, so the lists are what the model runs either way.

    Args:
        geo: The decode's geometry, whose ``layer_offset`` is 0: the decode runs every layer.
        hf_config: The checkpoint's config.

    Returns:
        One ``"<field> <value> (the kernel's <value>)"`` entry for each field that differs, in field order.
    """
    problems = [f"{name} {getattr(hf_config, name)} (the kernel's {getattr(geo, attr)})"
                for name, attr in CHECKPOINT_FIELDS.items() if getattr(hf_config, name) != getattr(geo, attr)]
    # the kernel takes the shared expert as one intermediate of `shared_intermediate_full` columns
    shared = hf_config.n_shared_experts * hf_config.moe_intermediate_size
    if shared != geo.shared_intermediate_full:
        problems.append(f"n_shared_experts {hf_config.n_shared_experts} of {hf_config.moe_intermediate_size} columns "
                        f"(the kernel's shared expert of {geo.shared_intermediate_full})")
    full = [layer for layer, kind in enumerate(hf_config.indexer_types) if kind == "full"]
    if full != list(geo.full_layers):
        problems.append(f"indexer_types with full layers {full} (the kernel's {list(geo.full_layers)})")
    dense = [layer for layer, kind in enumerate(hf_config.mlp_layer_types) if kind == "dense"]
    if dense != list(range(geo.first_dense_layers)):
        problems.append(f"mlp_layer_types with dense layers {dense} (the kernel's first {geo.first_dense_layers})")
    # transformers standardizes the RoPE fields into this dict, with both keys, whatever the config.json spells
    rope = hf_config.rope_parameters
    if rope["rope_type"] != "default" or rope["rope_theta"] != geo.rope_theta:
        problems.append(f"rope_parameters {rope} (the kernel's default RoPE at theta {geo.rope_theta})")
    # The router: sigmoid scores, the top-k renormalized to sum to one, and no group limit. transformers'
    # GlmMoeDsaConfig does not declare scoring_func, because its router is sigmoid, so a config without it is sigmoid.
    scoring = getattr(hf_config, "scoring_func", "sigmoid")
    if scoring != "sigmoid":
        problems.append(f"scoring_func {scoring!r} (the kernel's 'sigmoid')")
    if not hf_config.norm_topk_prob:
        problems.append("norm_topk_prob False (the kernel renormalizes the routed weights)")
    if hf_config.topk_group != hf_config.n_group:
        problems.append(f"topk_group {hf_config.topk_group} of n_group {hf_config.n_group} (the kernel routes over "
                        "every expert, with no group limit)")
    return problems


def check_checkpoint(geo: Glm52GkGeometry, hf_config: Any) -> None:
    """Refuse a checkpoint the decode kernel does not serve at this geometry, or a geometry it does not trace.

    First the checkpoint against the geometry (`checkpoint_problems`). Then, with the two equal, the rules of the
    kernel the decode runs (the decode kernel, ``sglang_neuron.models.glm52.kernels.decode``) on the geometry's shapes and on
    the point's context and sharding. Each rule names its value, because the same mismatch inside the kernel is a
    trace-time assert with no config context.

    Raises:
        ValueError: The checkpoint differs from the geometry (every difference, in one message), or the geometry
            breaks one of the kernel's rules.
    """
    problems = checkpoint_problems(geo, hf_config)
    if problems:
        raise ValueError("the GLM GK decode is built for GLM-5.2-family shapes and constants; this checkpoint has "
                         + ", ".join(problems))
    if geo.index_topk % 128:
        raise ValueError(f"index_topk {geo.index_topk} must be a multiple of 128: the kernel gathers the selected keys "
                         "with an index tile over all 128 partitions (2048 takes its fast selection path)")
    if geo.hidden % 512:
        raise ValueError(f"hidden_size {geo.hidden} must be a multiple of 512, the kernel's MX contraction tile")
    pool_block = geo.block_size * geo.cp_degree
    if geo.context % pool_block:
        raise ValueError(f"the point's context {geo.context} must be a multiple of one pool block, block_size x "
                         f"cp_degree = {pool_block}: each CP rank holds whole blocks of a slot")
    # The decode kernel ranks the shard in one select, or in a power of two of equal chunks and then the chunk winners in
    # one more select (its `_pick_local_chunks`, `idx_topk_chunking`). No select is wider than IDX_TOPK_CHUNK keys, and
    # its score and residency kernels tile a chunked shard by IDX_KEY_TILE keys.
    chunking = idx_topk_chunking(geo.idx_keys, geo.index_topk)
    if chunking is not None and chunking[0] > 1 and chunking[1] & (chunking[1] - 1):
        # a chunked shard is ranked by topk_select's wide select, which carries a winner's in-chunk key as its shard
        # key masked by the chunk width (wide_publish)
        raise ValueError(
            f"one indexer shard of the point holds {geo.idx_keys} keys, ranked in {chunking[0]} chunks of "
            f"{chunking[1]}: a chunked shard needs a power-of-two chunk width (the wide select masks each winner's "
            "shard key by it). Give the point (models/glm52/points.py) a context whose shard is a power-of-two "
            "multiple of its chunk width")
    if chunking is None or (chunking[0] > 1 and geo.idx_keys % IDX_KEY_TILE):
        raise ValueError(
            f"one indexer shard of the point holds {geo.idx_keys} keys. The kernel ranks up to {IDX_TOPK_CHUNK} keys "
            f"at once, or a multiple of {IDX_KEY_TILE} keys in a power of two C <= "
            f"{max(IDX_TOPK_CHUNK // geo.index_topk, 1)} of equal chunks of at most {IDX_TOPK_CHUNK} keys, each a "
            f"multiple of {IDX_SNAKE_PARTITIONS}, whose C x {geo.index_topk} winners fit one more select. Give the "
            "point (models/glm52/points.py) a larger key_size (cp_degree, or t_shard 1) or a context whose shard has "
            "such chunks")


def max_served_model_len(geo: Glm52GkGeometry) -> int:
    """The longest sequence the GK decodes exactly at this geometry: its slot's context less one pool block.

    The kernel's causal keep test reads static positions, from ``prior_len = context - spec`` (its position edge), so
    a step that starts at ``prior_len + 2`` or later drops its newest prior keys and decodes wrong tokens, with no
    error. Serving one pool block (``block_size * cp_degree`` positions, 128 at CP4) short of the slot's end keeps
    every step below the edge, with room past the served context for a request's last step and the steps an
    overlapped scheduler may launch before it sees the request finish: 5 steps of ``spec`` 8 rows, 40 of the 128.

    Example:
        >>> max_served_model_len(Glm52GkGeometry(context=1048576))
        1048448
    """
    return geo.context - geo.block_size * geo.cp_degree
