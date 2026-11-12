# SPDX-License-Identifier: Apache-2.0
"""Launch settings of the GLM-5.2 PD prefill at its two points: PP4 x 32 ranks (the default) and PP16 x 8.

Most are compile knobs of the prefill kernel and its weight loaders. Each value is the one the accuracy and timing gates
ran with; another value builds a program nothing has validated, so the platform refuses a launch environment that sets
one differently (serving_policy.set_launch_setting).

Pure data: the catalog imports it at launch."""

from __future__ import annotations

import hashlib

from sglang_neuron.models.glm52.prefill.compile_args import SCRATCHPAD_PAGE_MB
from sglang_neuron.models.glm52.prefill.expert_placements import PLACEMENT_ENV, PLACEMENT_FILE, PLACEMENT_MD5

# (name, value, reason) of the PP4 stages' kernel and loader knobs
_KERNEL_KNOBS = (
    ("GLM_GK_ATTN_RESIDENT", "1",
     "keep each rank's wo-group slice of the attention weights resident, so no layer gathers them"),
    ("GLM_GK_KV_PLANS", "64",
     "sizes the KV buffer's spare row, which every served program allocates"),
    ("GLM_GK_MOE_AFF8", "1", "carry each MoE row's top-8 ids and affinities through the token all-gather"),
    ("GLM_GK_MOE_DYN_INNER", "1", "run the routed experts' dynamic block loop one block per iteration"),
    ("GLM_GK_MOE_PF_TILES", "3", "prefetch the next expert block's first three gate/up tiles"),
    ("GLM_GK_MOE_PF_TILES_T128", "1", "prefetch only one tile in the 4096-token graph, where three measured slower"),
    ("GLM_GK_MOE_WRELAY", "1", "store the routed gate/up weights tile-major so each tile streams contiguously"),
    ("GLM_GK_MOE_ZINIT_AT", "moe", "zero the routed output once per layer under the MoE token all-gather"),
    ("GLM_GK_MOE_ZINIT_MIN_T", "128", "move that zero-init only in the 4096-token graph"),
    ("GLM_GK_PRJ_NORMST", "1", "store the input norm's output tile by tile on a hardware DMA queue"),
    ("GLM_GK_PRJ_OMX", "1", "fuse the V-up and o_proj into one MXFP8 matmul call"),
    ("GLM_GK_PRJ_QKV", "1239", "the head-TP QKV's schedule options (issue order only)"),
    ("GLM_GK_PRJ_QKV_T", "64", "apply those QKV options only in the 2048-token graph, where they were validated"),
    ("GLM_GK_PRJ_XAG", "2", "split the head-TP x all-gather in two so the QKV starts on the first half"),
    ("GLM_GK_ROPE_ONCE", "1", "build the QKV's RoPE tables once per launch"),
    ("GLM_GK_SHARED_TP", "1", "replicate the shared expert on every rank"),
    ("GLM_GK_SHARED_UNDER_RS", "late", "run the replicated shared expert under the routed experts' reduce-scatter"),
    ("GLM_GK_SHMAP_ONCE", "1", "build the shared expert's token map once per launch"),
    ("GLM_GK_SYNC_HOL", "2", "reorder the full layers' Sync-queue loads so the indexer is not stuck behind the QKV"),
    ("GLM_GK_TC_MOE_AG", "split", "split the MoE token all-gather into tiles"),
    ("GLM_GK_UP_NORMRES", "1", "compute the hybrid input norm without the unused MX row"),
    ("GLM_GK_UP_S1DMAT", "1", "load the QKV's token tiles K-major with one transposing DMA each"),
    ("GLM52K_GK_ATTN_UNROLL_MAX_S", "128", "unroll the sparse attention up to 128 queries a rank"),
)

# (name, value, reason): the PP4 point's launch settings, in the order the platform applies them
SETTINGS: tuple = _KERNEL_KNOBS + (
    (PLACEMENT_ENV, PLACEMENT_FILE,
     f"the MoE expert placement the PP4 stages were measured with (md5 {PLACEMENT_MD5})"),
    ("NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION", "0",
     "complete executions without the runtime's polling thread, which competes with the scheduler for the host"),
    ("NEURON_SCRATCHPAD_PAGE_SIZE", str(SCRATCHPAD_PAGE_MB),
     "the scratchpad page in MiB, equal to the graphs' --hbm-scratchpad-page-size"),
    ("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", "10800",
     "wait out a stage NEFF's compile instead of compiling the same NEFF twice"),
)
_PP4_KNOBS = frozenset(name for name, _, _ in _KERNEL_KNOBS)

# The PP16 point's kernel knobs, in place of the PP4 ones, which are tuned for 32-rank stages
_PP16_KERNEL_KNOBS = (
    ("GLM_GK_MOE_PAD_MASK", "1", "give the bucket's padding rows zero router affinity so they take no expert block"),
    # without these four the PP16 prefill is nondeterministic run to run: each keeps a buffer peers still use
    ("GLM_GK_STAGE_SPLIT", "1", "split a collective's odd-sized source copy so the collective waits for its tail"),
    ("GLM_GK_SEND_LIFE", "1", "keep a weight prefetch's send buffers live until every peer has received them"),
    ("GLM_GK_KVL_LIFE", "1", "keep a layer's gathered local KV live until every peer's attention has read it"),
    ("GLM_GK_MOE_ZINIT_GATE", "1", "zero the expert outputs only after the MoE sync, when the buffer is free to reuse"),
)


def point_settings(pp_size: int) -> tuple:
    """The settings of the ``pp_size``-stage point. PP16 takes PP4's runtime settings, no placement file (its 8 EP
    ranks hold 32 experts each, in order) and its own kernel knobs."""
    if int(pp_size) == 4:
        return SETTINGS
    if int(pp_size) != 16:
        raise ValueError(f"pp_size {pp_size}: the prefill recipe has a PP4 and a PP16 point")
    kept = tuple(s for s in SETTINGS if s[0] not in _PP4_KNOBS and s[0] != PLACEMENT_ENV)
    return kept + _PP16_KERNEL_KNOBS


def names(pp_size: int = 4) -> frozenset:
    """The names the recipe of the ``pp_size``-stage point sets (point_settings)."""
    return frozenset(name for name, _, _ in point_settings(pp_size))


def digest(pp_size: int = 4) -> str:
    """sha256[:16] of the ``pp_size``-stage point's sorted (name, value) pairs: equal on every rank that runs it."""
    text = "\n".join(f"{name}={value}" for name, value in sorted((n, v) for n, v, _ in point_settings(pp_size)))
    return hashlib.sha256(text.encode()).hexdigest()[:16]
