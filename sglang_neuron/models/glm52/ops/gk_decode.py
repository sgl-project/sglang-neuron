# SPDX-License-Identifier: Apache-2.0
"""Functional wrapper for the GLM-5.2 decode giga-kernel (GK).

``glm52_decode_giga_kernel`` traces every sparse decoder layer of the decode step, collectives included, into one
``@nki.jit`` program. It takes this rank's ``[t_shard, hidden]`` slice of the residual stream; the per-layer weights
stacked on a leading layer axis (NKI takes no Python list, and ``W[i]`` drops that axis); the indexer weights on a
shorter axis, since shared layers have no indexer; the KV caches, which it updates in place (the paged MLA latent
stacks, four at the 1M point, and the indexer key cache); the layer-invariant control tensors; and nine replica
meshes. It returns the post-stack residual slice plus the updated caches for input-output aliasing.

The kernel is decode-only, trn3-only and MXFP4, with no torch reference. It comes from the vendored copy in
``sglang_neuron.models.glm52.kernels.decode`` (``KERNEL_PACKAGE``), which needs only the NKI DSL, so an import failure here
means a broken copy. The wrapper traces through the plugin's ``wrap_nki``, passes the FP8 ``wq_b`` / ``o_proj`` as
float8 views, and selects the indexer's carry-index merge in NKI, since the released toolchain has no GpSimd loader.

Example:
    >>> out = glm52_gk_decode(x_shard, weights=weights, control=control, meshes=meshes,
    ...                       scalars=scalars)  # doctest: +SKIP
"""

import importlib

import nki

import nki.language as nl
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from nki.collectives import ReplicaGroup as _ReplicaGroup

from sglang_neuron.utils.neuron_utils import can_run_kernel

#: The vendored kernel copy this process runs (``models/glm52/decode/kernel_limits.py`` checks a geometry against it).
KERNEL_PACKAGE = "sglang_neuron.models.glm52.kernels.decode"

# Not guarded: with the NKI DSL already imported, a failure here is a broken copy or a bad knob, and the server should
# stop now rather than at its first decode step.
_glm52_decode_giga_kernel = importlib.import_module(
    f"{KERNEL_PACKAGE}.giga_kernel").glm52_decode_giga_kernel


def _vendor_digest() -> str:
    """Return a digest of the vendored kernel files in use and the kernel knobs."""
    return importlib.import_module(KERNEL_PACKAGE).kernel_digest()


#: The NKI compile cache keys on the `wrap_nki` function's source and arguments, not on the kernel it calls or the
#: knobs that kernel read at import, so the wrapper takes this digest of the copy's files and knobs: editing either
#: compiles again.
_VENDOR_DIGEST = _vendor_digest()

# The four MoE weight tensors that the kernel reads as MXFP4. The host holds them
# as uint16 tiles, because torch has no fp4 dtype; the wrapper reinterprets them
# inside the trace so the MoE matmul sees the MX dtype.
_MX_WEIGHT_ARGS = (
    "routed_gate_up_weights_all",
    "routed_down_weights_all",
    "shared_gate_up_weights_all",
    "shared_down_weights_all",
)


def _as_mx(tensor):
    """Reinterpret a uint16 tile tensor as ``float4_e2m1fn_x4``.

    The bytes do not move. Only the dtype that the kernel's SBUF loads inherit
    changes, and the MX matmul needs the MX dtype to read the scale groups.
    """
    if tensor is not None and tensor.dtype == nl.uint16:
        return tensor.view(nl.float4_e2m1fn_x4)
    return tensor


def _as_fp8(tensor):
    """Reinterpret a uint8 weight (``SGLANG_NEURON_GLM52_ATTN_W8``'s e4m3fn ``wq_b`` / ``o_proj``) as ``float8_e4m3fn``.

    A bf16 weight passes unchanged. Both matmuls read the FP8 weight as the moving tile beside a bf16 stationary.
    """
    if tensor is not None and tensor.dtype == nl.uint8:
        return tensor.view(nl.float8_e4m3fn)
    return tensor


@nki.jit
def _glm52_gk_decode_wrapper(
    x_shard: nl.ndarray,
    # attention weights, stacked [L, ...]
    input_norm_w_all: nl.ndarray,
    wq_a_all: nl.ndarray,
    wkv_a_all: nl.ndarray,
    q_norm_w_all: nl.ndarray,
    kv_norm_w_all: nl.ndarray,
    wq_b_all: nl.ndarray,
    q_absorb_w_all: nl.ndarray,
    out_absorb_w_all: nl.ndarray,
    o_proj_w_all: nl.ndarray,
    cos: nl.ndarray,
    sin: nl.ndarray,
    # indexer weights, stacked [L_full, ...]
    idx_wq_b_all: nl.ndarray,
    idx_wk_all: nl.ndarray,
    idx_k_norm_w_all: nl.ndarray,
    idx_k_norm_b_all: nl.ndarray,
    idx_weights_proj_all: nl.ndarray,
    idx_cos: nl.ndarray,
    idx_sin: nl.ndarray,
    # caches, mutated in place
    kv_cache_all: nl.ndarray,
    idx_k_cache_all: nl.ndarray,
    # layer-invariant control
    kv_new_slot_mapping: nl.ndarray,
    block_table: nl.ndarray,
    positions: nl.ndarray,
    idx_new_slot_mapping: nl.ndarray,
    idx_positions: nl.ndarray,
    key_offset: nl.ndarray,
    cp_pos_offset: nl.ndarray,
    # MoE weights, stacked [L, ...]
    gamma_all: nl.ndarray,
    router_weights_all: nl.ndarray,
    router_bias_all: nl.ndarray,
    routed_gate_up_weights_all: nl.ndarray,
    routed_gate_up_weights_scale_all: nl.ndarray,
    routed_down_weights_all: nl.ndarray,
    routed_down_weights_scale_all: nl.ndarray,
    shared_gate_up_weights_all: nl.ndarray,
    shared_gate_up_weights_scale_all: nl.ndarray,
    shared_down_weights_all: nl.ndarray,
    shared_down_weights_scale_all: nl.ndarray,
    shared_affinities: nl.ndarray,
    # dense MLP weights, stacked [L_dense, ...]
    dense_gamma_all: nl.ndarray,
    dense_gate_weights_all: nl.ndarray,
    dense_up_weights_all: nl.ndarray,
    dense_down_weights_all: nl.ndarray,
    rank_id: nl.ndarray,
    shared_rank_id: nl.ndarray,
    # replica meshes as tuples of tuples -> ReplicaGroup, in-trace
    token_replica_ranks: tuple = (),
    cp_replica_ranks: tuple = (),
    head_replica_ranks: tuple = (),
    seq_replica_ranks: tuple = (),
    key_replica_ranks: tuple = (),
    ep_replica_ranks: tuple = (),
    tp_replica_ranks: tuple = (),
    block_replica_ranks: tuple = (),
    qblock_replica_ranks: tuple = (),
    latent_replica_ranks: tuple = (),
    # scalars
    num_layers: int = 0,
    index_share_period: int = 0,
    cp_degree: int = 0,
    head_shard: int = 1,
    token_shard: int = 0,
    key_size: int = 0,
    num_heads: int = 0,
    qk_nope_head_dim: int = 0,
    qk_rope_head_dim: int = 0,
    kv_lora_rank: int = 0,
    idx_n_heads: int = 0,
    idx_rope_dim: int = 0,
    index_topk: int = 0,
    spec: int = 0,
    softmax_scale: float = 1.0,
    kv_scale: float = 1.0,
    rms_norm_eps: float = 1e-5,
    latent_norm_eps: float = 1e-6,
    idx_norm_eps: float = 1e-6,
    ep_size: int = 0,
    tp_size: int = 0,
    top_k: int = 0,
    routed_scaling_factor: float = 1.0,
    index_skip_topk_offset: int = 0,
    layer_offset: int = 0,
    first_dense_layers: int = 0,
    # The pooled indexer (the chunk-dealt pool, `layout.idx_pool.IdxChunkPool`): the pool mesh, its degree and
    # tokens, and its per-step tables (None = no pool).
    pool_replica_ranks: tuple = (),
    idx_pool_degree: int = 0,
    idx_pool_tokens: int = 0,
    idx_pool_items: nl.ndarray = None,
    idx_pool_n_items: nl.ndarray = None,
    idx_pool_region_base: nl.ndarray = None,
    idx_pool_new_key_rows: nl.ndarray = None,
    idx_pool_unrot: nl.ndarray = None,
    idx_block_table: nl.ndarray = None,
    lm_head_weight_t: nl.ndarray = None,
    final_norm_w: nl.ndarray = None,
    sampling_params: nl.ndarray = None,
    lm_head_k: int = 64,
    kv_cache_hi: nl.ndarray = None,
    kv_cache_hi2: nl.ndarray = None,
    dense_replica_ranks: tuple = (),
    shared_replica_ranks: tuple = (),
    idx_shard_degree: int = 1,
    idx_block_size: int = 0,
    layer_start: int = 0,
    layer_stop: int = 0,
    vendor_digest: str = "",
    kv_cache_hi3: nl.ndarray = None,
):
    """Trace-time adapter: build the replica groups and cast the MX weights."""
    # Only the compile cache reads `vendor_digest`. See `_VENDOR_DIGEST`.
    del vendor_digest
    return _glm52_decode_giga_kernel(
        x_shard,
        input_norm_w_all,
        wq_a_all,
        wkv_a_all,
        q_norm_w_all,
        kv_norm_w_all,
        _as_fp8(wq_b_all),
        q_absorb_w_all,
        out_absorb_w_all,
        _as_fp8(o_proj_w_all),
        cos,
        sin,
        idx_wq_b_all,
        idx_wk_all,
        idx_k_norm_w_all,
        idx_k_norm_b_all,
        idx_weights_proj_all,
        idx_cos,
        idx_sin,
        kv_cache_all,
        idx_k_cache_all,
        kv_new_slot_mapping,
        block_table,
        positions,
        idx_new_slot_mapping,
        idx_positions,
        key_offset,
        cp_pos_offset,
        gamma_all,
        router_weights_all,
        router_bias_all,
        _as_mx(routed_gate_up_weights_all),
        routed_gate_up_weights_scale_all,
        _as_mx(routed_down_weights_all),
        routed_down_weights_scale_all,
        _as_mx(shared_gate_up_weights_all),
        shared_gate_up_weights_scale_all,
        _as_mx(shared_down_weights_all),
        shared_down_weights_scale_all,
        shared_affinities,
        dense_gamma_all,
        dense_gate_weights_all,
        dense_up_weights_all,
        dense_down_weights_all,
        rank_id,
        shared_rank_id,
        _ReplicaGroup([list(g) for g in token_replica_ranks]),
        _ReplicaGroup([list(g) for g in cp_replica_ranks]),
        _ReplicaGroup([list(g) for g in head_replica_ranks]),
        _ReplicaGroup([list(g) for g in seq_replica_ranks]),
        _ReplicaGroup([list(g) for g in key_replica_ranks]),
        _ReplicaGroup([list(g) for g in ep_replica_ranks]),
        _ReplicaGroup([list(g) for g in tp_replica_ranks]),
        num_layers,
        index_share_period,
        cp_degree,
        head_shard,
        token_shard,
        key_size,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        idx_n_heads,
        idx_rope_dim,
        index_topk,
        spec,
        softmax_scale,
        kv_scale,
        rms_norm_eps,
        latent_norm_eps,
        idx_norm_eps,
        ep_size,
        tp_size,
        top_k,
        routed_scaling_factor,
        index_skip_topk_offset,
        layer_offset,
        first_dense_layers,
        # The block-sharded exit and the block-sharded query path. Pass the
        # meshes by keyword, because the kernel takes the indexer block table
        # after them and a positional argument would land on it.
        block_replica_group=(
            _ReplicaGroup([list(g) for g in block_replica_ranks])
            if block_replica_ranks
            else None
        ),
        qblock_replica_group=(
            _ReplicaGroup([list(g) for g in qblock_replica_ranks])
            if qblock_replica_ranks
            else None
        ),
        # The indexer shares MLA's block table and deal. The kernel reads the
        # cache as PAGED through this table, which is what lets a connector
        # place a transferred indexer block where it places the latent block.
        idx_block_table=idx_block_table,
        lm_head_weight_t=lm_head_weight_t,
        final_norm_w=final_norm_w,
        sampling_params=sampling_params,
        lm_head_k=lm_head_k,
        # The second to fourth stacks of a split latent cache (four at the 1M point). Passed only when present, so a
        # single-stack call and its traced graph are unchanged.
        **({} if kv_cache_hi is None else {"kv_cache_hi": kv_cache_hi}),
        **({} if kv_cache_hi2 is None else {"kv_cache_hi2": kv_cache_hi2}),
        **({} if kv_cache_hi3 is None else {"kv_cache_hi3": kv_cache_hi3}),
        # The latent-sharded down-projection (`latent_tp`). Passed only when set,
        # so the token-group graph is unchanged.
        **(
            {}
            if not latent_replica_ranks
            else {
                "latent_replica_group": _ReplicaGroup(
                    [list(g) for g in latent_replica_ranks]
                )
            }
        ),
        # The dense MLP over a group wider than TP (`dense_tp`). Passed only when
        # set, so the TP-group graph is unchanged.
        **(
            {}
            if not dense_replica_ranks
            else {
                "dense_replica_group": _ReplicaGroup(
                    [list(g) for g in dense_replica_ranks]
                ),
                "dense_tp_size": len(dense_replica_ranks[0]),
            }
        ),
        # The shared expert over a group wider than TP (`shared_tp`), likewise.
        **(
            {}
            if not shared_replica_ranks
            else {
                "shared_replica_group": _ReplicaGroup(
                    [list(g) for g in shared_replica_ranks]
                ),
                "shared_tp_size": len(shared_replica_ranks[0]),
            }
        ),
        # The indexer shares MLA's block table: its keys are DEALT over the CP group in
        # idx_block_size units (MLA's deal).
        idx_shard_degree=idx_shard_degree,
        idx_block_size=idx_block_size,
        # The GK split (the decode point's gk_split): a layer window, passed only when set.
        **(
            {}
            if (layer_start, layer_stop) == (0, 0)
            else {"layer_start": layer_start, "layer_stop": layer_stop}
        ),
        **(
            {}
            if not idx_pool_degree
            else {
                "pool_replica_group": _ReplicaGroup(
                    [list(g) for g in pool_replica_ranks]
                ),
                "idx_pool_degree": idx_pool_degree,
                "idx_pool_tokens": idx_pool_tokens,
                # the chunk-dealt pool's tables (the GK tells its deal by `idx_pool_items`)
                "idx_pool_items": idx_pool_items,
                "idx_pool_n_items": idx_pool_n_items,
                "idx_pool_region_base": idx_pool_region_base,
                "idx_pool_new_key_rows": idx_pool_new_key_rows,
                "idx_pool_unrot": idx_pool_unrot,
            }
        ),
    )

# lnc=1: the kernel is traced at grid[1]. A 128-rank collective cannot load
# at lnc=2, where the runtime exposes 64 cores.
_wrapped_glm52_gk_decode = wrap_nki(_glm52_gk_decode_wrapper)


def _wide_mesh(meshes: dict[str, tuple], name: str) -> tuple:
    """Mesh `name` when it is wider than the TP mesh, else empty (the TP-group graph)."""
    mesh = meshes.get(name, ())
    if mesh and len(mesh[0]) > len(meshes["tp"][0]):
        return mesh
    return ()


def _latent_mesh(meshes: dict[str, tuple]) -> tuple:
    """The latent mesh when it is wider than the token mesh, else empty (the token-group graph)."""
    latent = meshes.get("latent", ())
    if latent and len(latent[0]) > len(meshes["token"][0]):
        return latent
    return ()


def can_use_glm52_gk_decode(x_shard: Tensor) -> bool:
    """Whether the GLM-5.2 GK decode kernel can run for this input.

    The tensors must be on a Neuron device: there is no torch fallback.

    Args:
        x_shard: The entry residual tensor. Its device selects the path.

    Returns:
        True when the kernel can run.

    Example:
        >>> can_use_glm52_gk_decode(torch.zeros(2, 8))  # doctest: +SKIP
        False
    """
    return can_run_kernel(x_shard)


def glm52_gk_decode(
    x_shard: Tensor,
    *,
    weights: dict[str, Tensor],
    control: dict[str, Tensor],
    meshes: dict[str, tuple],
    scalars: dict[str, object],
    head: dict[str, Tensor] | None = None,
    idx_shard_degree: int = 1,
    idx_block_size: int = 0,
    layer_window: tuple[int, int] = (0, 0),
):
    """Run the fused whole-stack GLM-5.2 decode giga-kernel.

    Args:
        x_shard: ``[t_shard, hidden]`` entry residual of this rank.
        weights: The stacked weight tensors, keyed by the kernel's parameter
            names. `Glm52GkWeights.as_kernel_kwargs` builds this mapping.
        control: The layer-invariant and step-dependent control tensors, keyed
            the same way. It must hold the caches. An optional
            ``idx_block_table`` makes the kernel read the indexer key cache as
            PAGED; its absence keeps the position-ordered read. A pooled indexer
            adds the chunk-dealt pool's per-step tables: ``idx_pool_items``,
            ``idx_pool_n_items``, ``idx_pool_region_base``, ``idx_pool_new_key_rows``
            and ``idx_pool_unrot`` (`layout.idx_pool.IdxChunkPool.tables`).
        meshes: The replica meshes from `Glm52GkGeometry.replica_meshes`.
        scalars: The scalar arguments from `Glm52GkGeometry.scalar_kwargs`.
        head: The LM-head tail's inputs ``lm_head_weight_t``, ``final_norm_w`` and ``sampling_params``, and the
            optional int ``k``, the candidates that the tail keeps (64 if absent). The served kernel runs the tail
            (``GLM52_GIGA_LM_HEAD``, which the launch pins on), so a served step passes them.
        idx_shard_degree: The degree the indexer's keys are dealt over (`Glm52GkGeometry.idx_shard_degree`).
        idx_block_size: The indexer cache's block, its deal unit (the latent cache's block).
        layer_window: The stack layers ``[start, stop)`` this call runs (the GK split), ``(0, 0)`` for all of them.

    Returns:
        A tuple of this rank's ``[t_shard, hidden]`` post-stack residual and the mutated caches: the latent stack and
        the indexer cache. With the LM-head tail, the tuple then has ``topk_values``, ``topk_indices`` and
        ``[t_shard, 1]`` uint32 ``tokens``. For a split latent stack (the ``kv_cache_hi``, ``kv_cache_hi2`` and
        ``kv_cache_hi3`` control keys), the later stacks come LAST, in that order.

    Raises:
        ValueError: If the scalars pool the indexer and the control lacks the pool's tables, or the other way round.
        RuntimeError: If the tensors are not on a Neuron device: the kernel has no CPU path.

    Example:
        >>> out = glm52_gk_decode(x, weights=w, control=c, meshes=m, scalars=s)  # doctest: +SKIP
    """
    pooled = int(scalars.get("idx_pool_degree", 0)) > 0
    if pooled != (control.get("idx_pool_items") is not None):
        raise ValueError(
            "the pooled indexer and its tables (idx_pool_items, ...) go together: the scalars pool "
            f"{scalars.get('idx_pool_degree', 0)} CP groups and the control "
            f"{'has' if control.get('idx_pool_items') is not None else 'lacks'} idx_pool_items"
        )
    if not can_use_glm52_gk_decode(x_shard):
        raise RuntimeError(
            "glm52_gk_decode needs a Neuron device: there is no torch fallback for the decode giga-kernel"
        )
    return _wrapped_glm52_gk_decode[1](
        x_shard,
        weights["input_norm_w_all"],
        weights["wq_a_all"],
        weights["wkv_a_all"],
        weights["q_norm_w_all"],
        weights["kv_norm_w_all"],
        weights["wq_b_all"],
        weights["q_absorb_w_all"],
        weights["out_absorb_w_all"],
        weights["o_proj_w_all"],
        control["cos"],
        control["sin"],
        weights["idx_wq_b_all"],
        weights["idx_wk_all"],
        weights["idx_k_norm_w_all"],
        weights["idx_k_norm_b_all"],
        weights["idx_weights_proj_all"],
        control["idx_cos"],
        control["idx_sin"],
        control["kv_cache_all"],
        control["idx_k_cache_all"],
        control["kv_new_slot_mapping"],
        control["block_table"],
        control["positions"],
        control["idx_new_slot_mapping"],
        control["idx_positions"],
        control["key_offset"],
        control["cp_pos_offset"],
        weights["gamma_all"],
        weights["router_weights_all"],
        weights["router_bias_all"],
        weights["routed_gate_up_weights_all"],
        weights["routed_gate_up_weights_scale_all"],
        weights["routed_down_weights_all"],
        weights["routed_down_weights_scale_all"],
        weights["shared_gate_up_weights_all"],
        weights["shared_gate_up_weights_scale_all"],
        weights["shared_down_weights_all"],
        weights["shared_down_weights_scale_all"],
        control["shared_affinities"],
        weights["dense_gamma_all"],
        weights["dense_gate_weights_all"],
        weights["dense_up_weights_all"],
        weights["dense_down_weights_all"],
        control["rank_id"],
        control["shared_rank_id"],
        tuple(tuple(int(r) for r in g) for g in meshes["token"]),
        tuple(tuple(int(r) for r in g) for g in meshes["cp"]),
        tuple(tuple(int(r) for r in g) for g in meshes["head"]),
        tuple(tuple(int(r) for r in g) for g in meshes["seq"]),
        tuple(tuple(int(r) for r in g) for g in meshes["key"]),
        tuple(tuple(int(r) for r in g) for g in meshes["ep"]),
        tuple(tuple(int(r) for r in g) for g in meshes["tp"]),
        tuple(tuple(int(r) for r in g) for g in meshes["block"]),
        tuple(tuple(int(r) for r in g) for g in meshes["qblock"]),
        # The latent group only when it is wider than the token group (`latent_tp`).
        tuple(tuple(int(r) for r in g) for g in _latent_mesh(meshes)),
        int(scalars["num_layers"]),
        int(scalars["index_share_period"]),
        int(scalars["cp_degree"]),
        int(scalars["head_shard"]),
        int(scalars["token_shard"]),
        int(scalars["key_size"]),
        int(scalars["num_heads"]),
        int(scalars["qk_nope_head_dim"]),
        int(scalars["qk_rope_head_dim"]),
        int(scalars["kv_lora_rank"]),
        int(scalars["idx_n_heads"]),
        int(scalars["idx_rope_dim"]),
        int(scalars["index_topk"]),
        int(scalars["spec"]),
        float(scalars["softmax_scale"]),
        float(scalars["kv_scale"]),
        float(scalars["rms_norm_eps"]),
        float(scalars["latent_norm_eps"]),
        float(scalars["idx_norm_eps"]),
        int(scalars["ep_size"]),
        int(scalars["tp_size"]),
        int(scalars["top_k"]),
        float(scalars["routed_scaling_factor"]),
        int(scalars["index_skip_topk_offset"]),
        int(scalars["layer_offset"]),
        int(scalars["first_dense_layers"]),
        # The pooled indexer (`Glm52GkGeometry.idx_pool_degree`): off when the scalars say 0.
        tuple(tuple(int(r) for r in g) for g in meshes.get("pool", ())),
        int(scalars.get("idx_pool_degree", 0)),
        int(scalars.get("idx_pool_tokens", 0)),
        # the chunk-dealt pool's tables (`IdxChunkPool.tables` / `IdxChunkPoolTables`), absent without a pool
        control.get("idx_pool_items"),
        control.get("idx_pool_n_items"),
        control.get("idx_pool_region_base"),
        control.get("idx_pool_new_key_rows"),
        control.get("idx_pool_unrot"),
        control.get("idx_block_table"),
        *(
            (head["lm_head_weight_t"], head["final_norm_w"], head["sampling_params"])
            if head
            else (None, None, None)
        ),
        int(head.get("k", 64)) if head else 64,
        control.get("kv_cache_hi"),
        control.get("kv_cache_hi2"),
        tuple(tuple(int(r) for r in g) for g in _wide_mesh(meshes, "dense")),
        tuple(tuple(int(r) for r in g) for g in _wide_mesh(meshes, "shared")),
        int(idx_shard_degree),
        int(idx_block_size),
        int(layer_window[0]),
        int(layer_window[1]),
        _VENDOR_DIGEST,
        control.get("kv_cache_hi3"),
    )
