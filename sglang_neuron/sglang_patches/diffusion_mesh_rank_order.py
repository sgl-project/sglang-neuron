# SPDX-License-Identifier: Apache-2.0
"""Build SGLang Diffusion's own model-parallel groups on the trn2 physical mesh, so the Neuron models use them.

SGLang Diffusion's ``initialize_model_parallel`` builds every group from one rank order:
``RankGenerator(tp, sp, pp, cfg, dp, "tp-sp-pp-cfg-dp").get_ranks(token)``. TP groups are consecutive ranks and SP
groups are ranks strided by ``tp``. On trn2 a strided SP group spans chips two torus hops apart (``[0, 8, ..., 56]``),
which the fabric cannot route, and the Neuron models need each TP and CP group to be a ring on the mesh
(``diffusion.distributed.parallel_state.mesh_tp_cp_cfg_groups``).

The hook wraps ``RankGenerator.get_ranks`` and renames every rank of the groups it returns through
``parallel_state.mesh_rank_permutation``: SGLang's TP group i becomes mesh row i and its SP group j mesh column j,
members in the mesh order. A rank keeps its process and its NeuronCore, and rank 0 stays rank 0, the rank SGLang's
scheduler takes requests on. For a generator that is not ``initialize_model_parallel``'s shape (pp = dp = 1, no rank
offset, that order), and off the trn2 mesh, the renaming is the identity. ``diffusion.distributed.groups.bind_groups``
checks the groups SGLang built before the models use them; if this hook did not apply, it raises.

Every partition SGLang builds from the generator is renamed, so its other groups (VAE decode, Ulysses / ring) list
their ranks in mesh order too. The Neuron models do not use them.

The diffusion plugin (``sglang_neuron.diffusion.plugin``) registers this module with SGLang Diffusion's HookRegistry;
SGLang applies it in every worker (``worker_bootstrap``) before the worker builds its groups.

Targets md5-pinned against upstream SGLang 43881a7: ``RankGenerator.get_ranks`` (wrapped); relied on:
``RankGenerator.__init__`` (the fields the hook reads) and ``parallel_state.initialize_model_parallel`` (builds the TP
and SP groups through ``get_ranks`` with that order).

Upstream: a platform hook for the rank layout of ``initialize_model_parallel``'s groups (a rank permutation for
``tp``, ``sp`` and ``cfg``), so a platform with a physical topology can lay out SGLang's groups; this module would
become the Neuron platform's implementation of that hook."""

from __future__ import annotations

TARGET = ("sglang.multimodal_gen.runtime.distributed.utils", "RankGenerator.get_ranks")
_TARGETS_MD5 = {
    TARGET: "d0e55432c72310198540b1b531b52298",
    # the fields the hook reads: tp, sp, pp, cfg, dp, rank_offset, order
    (
        "sglang.multimodal_gen.runtime.distributed.utils",
        "RankGenerator.__init__",
    ): "55d2cf9f13da984959ce593a6e226466",
    # builds the TP and SP groups through get_ranks with SGLANG_RANK_ORDER
    (
        "sglang.multimodal_gen.runtime.distributed.parallel_state",
        "initialize_model_parallel",
    ): "606391376d2e998e43d6237c9e64dc65",
}
SGLANG_RANK_ORDER = "tp-sp-pp-cfg-dp"


def permutation_for(generator) -> list[int] | None:
    """The renaming for one RankGenerator; ``None`` when no renaming applies."""
    from sglang_neuron.diffusion.distributed.parallel_state import mesh_rank_permutation

    if (
        generator.order != SGLANG_RANK_ORDER
        or generator.pp != 1
        or generator.dp != 1
        or generator.rank_offset
    ):
        return None
    return mesh_rank_permutation(generator.tp, generator.sp, generator.cfg)


def get_ranks(original, self, token):
    """RankGenerator.get_ranks, with every rank renamed onto the Neuron layout."""
    groups = original(self, token)
    perm = permutation_for(self)
    return groups if perm is None else [[perm[r] for r in group] for group in groups]


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(".".join(TARGET), get_ranks, around)
