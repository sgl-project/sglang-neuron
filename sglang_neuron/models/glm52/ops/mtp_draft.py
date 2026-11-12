# SPDX-License-Identifier: Apache-2.0
"""Functional wrapper for the GLM-5.2 MTP draft chain kernel.

The kernel (the vendored ``mtp/`` fork in ``sglang_neuron.models.glm52.kernels.decode``) runs the MTP layer ``num_steps``
times in one ``@nki.jit`` trace, collectives included, and returns one greedy draft token per sequence and step, the
same on every rank. It takes the step-0 embedding and the target's post-norm hidden rows, the layer's weights
(experts in FP8 with one factor per weight), the two MTP caches, which it updates in place on step 0 only, the
per-step control tensors, and the replica meshes.

Steps after the first reuse step 0's indexer selection, attend no keys of their own, and write nothing. The import is
guarded so this module still imports on a host without the Neuron SDK.

Example:
    >>> out = glm52_mtp_draft(embeds, hidden, weights=w, control=c, meshes=m, scalars=s)  # doctest: +SKIP
"""

import logging

import nki
import nki.language as nl
import torch
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from nki.collectives import ReplicaGroup as _ReplicaGroup

from sglang_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

try:
    import importlib

    # import the module, not the decorated function, so __file__ points at the kernel source
    _mtp_module = importlib.import_module(
        "sglang_neuron.models.glm52.kernels.decode.mtp.draft_chain"
    )
    _glm52_mtp_rows_chain_kernel = _mtp_module.glm52_mtp_draft_chain_kernel

    #: Absolute path of the kernel file in use, for a log line and a test.
    KERNEL_SOURCE = _mtp_module.__file__
    #: Zero rows before and after the vocabulary shard of the embedding table.
    EMBED_PAD_ROWS = int(_mtp_module.EMBED_PAD_ROWS)
    _MTP_AVAILABLE = True
    logger.debug("[glm52 MTP] kernel source: %s", KERNEL_SOURCE)
except Exception as e:  # pragma: no cover - import availability is env-dependent
    _glm52_mtp_rows_chain_kernel = None
    KERNEL_SOURCE = None
    EMBED_PAD_ROWS = 2
    _MTP_AVAILABLE = False
    logger.info(
        "glm52_mtp_draft_chain_kernel is not available (%s). The GLM-5.2 MTP "
        "draft path raises an error if you select it. This is expected on a "
        "host without the kernel.",
        e,
    )


def _vendor_digest() -> str:
    """Return a digest of the kernel copy's files and of the kernel knobs."""
    import sglang_neuron.models.glm52.kernels.decode as kernel_copy

    return kernel_copy.kernel_digest()


#: The NKI compile cache keys on the wrapper's source and arguments only, so the wrapper also takes a digest of the
#: kernel copy and its knobs; editing either recompiles instead of loading a stale kernel.
_VENDOR_DIGEST = _vendor_digest()

# Positions of the kept outputs in the kernel's return tuple (the greedy head's).
_OUT_KV_CACHE = 0
_OUT_IDX_K_CACHE = 1
_OUT_TOKENS_ALL = 3


if _MTP_AVAILABLE:

    @nki.jit
    def _glm52_mtp_draft_wrapper(
        # step-0 prologue inputs
        inputs_embeds: nl.ndarray,
        previous_hidden: nl.ndarray,
        mtp_positions_all: nl.ndarray,
        enorm_w: nl.ndarray,
        hnorm_w: nl.ndarray,
        eh_proj_w_t: nl.ndarray,
        # embedding feedback, vocabulary-sharded
        embed_shard_pad: nl.ndarray,
        vocab_base: nl.ndarray,
        # attention weights
        input_norm_w: nl.ndarray,
        wq_a: nl.ndarray,
        wkv_a: nl.ndarray,
        q_norm_w: nl.ndarray,
        kv_norm_w: nl.ndarray,
        wq_b: nl.ndarray,
        q_absorb_w: nl.ndarray,
        out_absorb_w: nl.ndarray,
        o_proj_w: nl.ndarray,
        cos_all: nl.ndarray,
        sin_all: nl.ndarray,
        # indexer weights
        idx_wq_b: nl.ndarray,
        idx_wk: nl.ndarray,
        idx_k_norm_w: nl.ndarray,
        idx_k_norm_b: nl.ndarray,
        idx_weights_proj: nl.ndarray,
        idx_cos: nl.ndarray,
        idx_sin: nl.ndarray,
        # caches, mutated in place
        kv_cache: nl.ndarray,
        idx_k_cache: nl.ndarray,
        # per-step control
        kv_new_slot_mapping_all: nl.ndarray,
        block_table: nl.ndarray,
        positions_all: nl.ndarray,
        idx_new_slot_mapping: nl.ndarray,
        idx_positions: nl.ndarray,
        key_offset: nl.ndarray,
        cp_pos_offset: nl.ndarray,
        shared_extra_keys: nl.ndarray,
        # MoE weights, the experts as uint8 FP8 bits
        gamma: nl.ndarray,
        router_weights: nl.ndarray,
        router_bias: nl.ndarray,
        routed_gate_up_weights: nl.ndarray,
        routed_down_weights: nl.ndarray,
        shared_gate_up_weights: nl.ndarray,
        shared_down_weights: nl.ndarray,
        rank_id: nl.ndarray,
        # the draft head
        shared_head_norm_w: nl.ndarray,
        lm_head_weight_t: nl.ndarray,
        sampling_params_all: nl.ndarray,
        live_rows: nl.ndarray,
        # the FP8 experts' factors and each sequence's draft row in this rank's token slice
        routed_fp8_scales: nl.ndarray,
        shared_fp8_scales: nl.ndarray,
        draft_rows: nl.ndarray,
        # replica meshes as tuples of tuples -> ReplicaGroup, in-trace
        token_replica_ranks: tuple = (),
        cp_replica_ranks: tuple = (),
        head_replica_ranks: tuple = (),
        seq_replica_ranks: tuple = (),
        key_replica_ranks: tuple = (),
        ep_replica_ranks: tuple = (),
        block_replica_ranks: tuple = (),
        qblock_replica_ranks: tuple = (),
        # scalars
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
        num_steps: int = 1,
        idx_block_table: nl.ndarray = None,
        vendor_digest: str = "",
        idx_shard_degree: int = 1,
        idx_block_size: int = 0,
    ):
        """Trace-time adapter: build the replica groups and call the chain."""
        # Only the compile cache reads `vendor_digest`. See `_VENDOR_DIGEST`.
        del vendor_digest
        # The FP8 experts travel as uint8 bits; the kernel reads them as float8_e4m3fn.
        routed_gate_up_weights = routed_gate_up_weights.view(nl.float8_e4m3fn)
        routed_down_weights = routed_down_weights.view(nl.float8_e4m3fn)
        shared_gate_up_weights = shared_gate_up_weights.view(nl.float8_e4m3fn)
        shared_down_weights = shared_down_weights.view(nl.float8_e4m3fn)
        block_group = (
            _ReplicaGroup([list(g) for g in block_replica_ranks])
            if block_replica_ranks
            else None
        )
        # Every argument goes by keyword. The kernel signature is long, and a
        # keyword cannot land on the wrong parameter.
        return _glm52_mtp_rows_chain_kernel(
            inputs_embeds=inputs_embeds,
            previous_hidden=previous_hidden,
            mtp_positions_all=mtp_positions_all,
            enorm_w=enorm_w,
            hnorm_w=hnorm_w,
            eh_proj_w_t=eh_proj_w_t,
            embed_shard_pad=embed_shard_pad,
            vocab_base=vocab_base,
            input_norm_w=input_norm_w,
            wq_a=wq_a,
            wkv_a=wkv_a,
            q_norm_w=q_norm_w,
            kv_norm_w=kv_norm_w,
            wq_b=wq_b,
            q_absorb_w=q_absorb_w,
            out_absorb_w=out_absorb_w,
            o_proj_w=o_proj_w,
            cos_all=cos_all,
            sin_all=sin_all,
            idx_wq_b=idx_wq_b,
            idx_wk=idx_wk,
            idx_k_norm_w=idx_k_norm_w,
            idx_k_norm_b=idx_k_norm_b,
            idx_weights_proj=idx_weights_proj,
            idx_cos=idx_cos,
            idx_sin=idx_sin,
            kv_cache=kv_cache,
            idx_k_cache=idx_k_cache,
            kv_new_slot_mapping_all=kv_new_slot_mapping_all,
            block_table=block_table,
            positions_all=positions_all,
            idx_new_slot_mapping=idx_new_slot_mapping,
            idx_positions=idx_positions,
            key_offset=key_offset,
            cp_pos_offset=cp_pos_offset,
            shared_extra_keys=shared_extra_keys,
            gamma=gamma,
            router_weights=router_weights,
            router_bias=router_bias,
            routed_gate_up_weights=routed_gate_up_weights,
            routed_down_weights=routed_down_weights,
            shared_gate_up_weights=shared_gate_up_weights,
            shared_down_weights=shared_down_weights,
            rank_id=rank_id,
            shared_head_norm_w=shared_head_norm_w,
            lm_head_weight_t=lm_head_weight_t,
            sampling_params_all=sampling_params_all,
            token_replica_group=_ReplicaGroup([list(g) for g in token_replica_ranks]),
            cp_replica_group=_ReplicaGroup([list(g) for g in cp_replica_ranks]),
            head_replica_group=_ReplicaGroup([list(g) for g in head_replica_ranks]),
            seq_replica_group=_ReplicaGroup([list(g) for g in seq_replica_ranks]),
            key_replica_group=_ReplicaGroup([list(g) for g in key_replica_ranks]),
            ep_replica_group=_ReplicaGroup([list(g) for g in ep_replica_ranks]),
            cp_degree=cp_degree,
            head_shard=head_shard,
            token_shard=token_shard,
            key_size=key_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            idx_n_heads=idx_n_heads,
            idx_rope_dim=idx_rope_dim,
            index_topk=index_topk,
            spec=spec,
            softmax_scale=softmax_scale,
            kv_scale=kv_scale,
            rms_norm_eps=rms_norm_eps,
            latent_norm_eps=latent_norm_eps,
            idx_norm_eps=idx_norm_eps,
            ep_size=ep_size,
            tp_size=tp_size,
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
            num_steps=num_steps,
            block_replica_group=block_group,
            qblock_replica_group=(
                _ReplicaGroup([list(g) for g in qblock_replica_ranks])
                if qblock_replica_ranks
                else None
            ),
            # The indexer shares MLA's block table: its keys are DEALT over the CP group in
            # idx_block_size units and read through idx_block_table. The MTP cache is private
            # to this layer.
            idx_block_table=idx_block_table,
            idx_shard_degree=idx_shard_degree,
            idx_block_size=idx_block_size,
            # The compacted routed path: one draft row for each sequence.
            live_rows=live_rows,
            draft_rows=draft_rows,
            # The rows chain shards eh_proj over the o_proj block group: one
            # all_to_all group for the step, not two.
            eh_proj_replica_group=block_group,
            routed_fp8_scales=routed_fp8_scales,
            shared_fp8_scales=shared_fp8_scales,
        )

    # lnc=1: the kernel is traced at grid[1]. A 128-rank collective cannot load
    # at lnc=2, where the runtime exposes 64 cores.
    _wrapped_glm52_mtp_draft = wrap_nki(_glm52_mtp_draft_wrapper)
else:
    _wrapped_glm52_mtp_draft = None


def can_use_glm52_mtp_draft(inputs_embeds: Tensor) -> bool:
    """Whether the draft chain kernel can run here: it imports and ``inputs_embeds`` is on a Neuron device or the NKI
    simulator. There is no torch fallback.
    """
    if not _MTP_AVAILABLE:
        return False
    return can_run_kernel(inputs_embeds)


def glm52_mtp_draft(
    inputs_embeds: Tensor,
    previous_hidden: Tensor,
    *,
    weights: dict[str, Tensor],
    control: dict[str, Tensor],
    meshes: dict[str, tuple],
    scalars: dict[str, object],
    num_steps: int = 7,
    idx_shard_degree: int = 1,
    idx_block_size: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the MTP draft chain for ``num_steps`` steps.

    Args:
        inputs_embeds: ``[t_shard, hidden]`` bf16 step-0 embedding of this rank's rows.
        previous_hidden: ``[t_shard, hidden]`` bf16 post-norm hidden state of the target for the same rows.
        weights: The MTP layer's weights by kernel parameter name, including this rank's ``eh_proj_w_t`` column
            shard, the FP8 experts and their ``routed_fp8_scales`` / ``shared_fp8_scales``.
        control: The control tensors by name, including the two caches (updated in place), ``live_rows`` and
            ``draft_rows``; an optional ``idx_block_table`` reads the indexer cache as paged.
        meshes: ``Glm52GkGeometry.replica_meshes``.
        scalars: ``mtp_draft.mtp_scalar_kwargs``.
        num_steps: Draft steps per launch.
        idx_shard_degree: Ranks the indexer keys are dealt over.
        idx_block_size: The indexer cache's block size.

    Returns:
        ``(tokens_all [num_steps, num_slots, 1] int32, kv_cache, idx_k_cache)``.

    Raises:
        RuntimeError: The kernel is not available, or the tensors are on CPU without the NKI simulator.
    """
    if not can_use_glm52_mtp_draft(inputs_embeds):
        raise RuntimeError(
            "glm52_mtp_draft cannot run. It needs the vendored GLM-5.2 MTP draft "
            "chain kernel and a Neuron device or the NKI simulator. "
            f"Kernel available={_MTP_AVAILABLE}. There is no torch fallback."
        )
    # The rows chain's draft steps run the TOKEN-GROUP query path, so its MTP layer holds wq_b /
    # q_absorb_w over the token group even under a target qproj_tp (`gk_geometry.mtp_query_geometry`),
    # and the chain takes no query block group. A block-sharded wq_b is refused here, not at trace.
    qblock_replica_ranks = tuple(tuple(int(r) for r in g) for g in meshes["qblock"])
    token_group = int(scalars["cp_degree"]) * int(scalars["head_shard"])
    q_degree = (
        int(scalars["num_heads"])
        * (int(scalars["qk_nope_head_dim"]) + int(scalars["qk_rope_head_dim"]))
        // int(weights["wq_b"].shape[1])
    )
    if q_degree > token_group:
        raise ValueError(
            f"glm52_mtp_draft: wq_b holds 1/{q_degree} of the query columns, and the rows "
            f"chain runs the query path of the token group ({token_group} ranks). Build "
            "the MTP store with rows_chain (wq_b / q_absorb_w over the token group)."
        )
    if qblock_replica_ranks and len(qblock_replica_ranks[0]) > token_group:
        qblock_replica_ranks = ()
    vendor_digest = _VENDOR_DIGEST
    out = _wrapped_glm52_mtp_draft[1](
        inputs_embeds,
        previous_hidden,
        control["mtp_positions_all"],
        weights["enorm_w"],
        weights["hnorm_w"],
        weights["eh_proj_w_t"],
        weights["embed_shard_pad"],
        control["vocab_base"],
        weights["input_norm_w"],
        weights["wq_a"],
        weights["wkv_a"],
        weights["q_norm_w"],
        weights["kv_norm_w"],
        weights["wq_b"],
        weights["q_absorb_w"],
        weights["out_absorb_w"],
        weights["o_proj_w"],
        control["cos_all"],
        control["sin_all"],
        weights["idx_wq_b"],
        weights["idx_wk"],
        weights["idx_k_norm_w"],
        weights["idx_k_norm_b"],
        weights["idx_weights_proj"],
        control["idx_cos"],
        control["idx_sin"],
        control["kv_cache"],
        control["idx_k_cache"],
        control["kv_new_slot_mapping_all"],
        control["block_table"],
        control["positions_all"],
        control["idx_new_slot_mapping"],
        control["idx_positions"],
        control["key_offset"],
        control["cp_pos_offset"],
        control["shared_extra_keys"],
        weights["gamma"],
        weights["router_weights"],
        weights["router_bias"],
        weights["routed_gate_up_weights"],
        weights["routed_down_weights"],
        weights["shared_gate_up_weights"],
        weights["shared_down_weights"],
        control["rank_id"],
        weights["shared_head_norm_w"],
        weights["lm_head_weight_t"],
        control["sampling_params_all"],
        control["live_rows"],
        weights["routed_fp8_scales"],
        weights["shared_fp8_scales"],
        control["draft_rows"],
        tuple(tuple(int(r) for r in g) for g in meshes["token"]),
        tuple(tuple(int(r) for r in g) for g in meshes["cp"]),
        tuple(tuple(int(r) for r in g) for g in meshes["head"]),
        tuple(tuple(int(r) for r in g) for g in meshes["seq"]),
        tuple(tuple(int(r) for r in g) for g in meshes["key"]),
        tuple(tuple(int(r) for r in g) for g in meshes["ep"]),
        tuple(tuple(int(r) for r in g) for g in meshes["block"]),
        qblock_replica_ranks,
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
        int(num_steps),
        control.get("idx_block_table"),
        vendor_digest,
        int(idx_shard_degree),
        int(idx_block_size),
    )
    # A value cast, as the decode giga-kernel tail does: torch has no usable
    # uint32. The ids are below 2**31, so the cast keeps every value.
    return out[_OUT_TOKENS_ALL].to(torch.int32), out[_OUT_KV_CACHE], out[_OUT_IDX_K_CACHE]
