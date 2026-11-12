# SPDX-License-Identifier: Apache-2.0
"""Classifier-free guidance (CFG) across CFG replicas: the compiled all-gather + combine.

Each CFG branch on Neuron is a full multi-core shard (TP x CP), not a single device: the CFG group
of size ``cfg_size`` pairs the corresponding ranks of the per-branch replicas
(``parallel_state.mesh_tp_cp_cfg_groups``). Under CFG parallel each replica runs its branch (the
compiled transformer NEFF), then :func:`_cfg_gather_combine`, compiled with ``fullgraph=True``,
exchanges the two branches on the Neuron CC fabric and combines them
(``NeuronWanModel.predict_noise_maybe_with_cfg`` and ``compile_helpers``). That needs the CFG
group's replica partition registered with TP / CP (``parallel_state.register_replica_groups``):
torch-neuronx's collective legalization resolves a process group's ``group_name`` to a full-world
``replica_groups`` partition through its mesh registry. There is no uncompiled fallback on device:
a CFG collective that fails to legalize raises.
"""

from __future__ import annotations

import torch

from sglang_neuron.diffusion.distributed.groups import get_cfg_group


def _cfg_gather_combine(
    local_pred: torch.Tensor,
    true_cfg_scale: torch.Tensor,
) -> torch.Tensor:
    """All-gather the two CFG branches and apply the CFG combine. Compiled.

    ``local_pred`` is this rank's branch output — the model's predicted-noise
    tensor, same shape on both CFG ranks (e.g. the latent ``[B, C, T, H, W]``).
    The all_gather stacks the two branches into ``[2, *local_pred.shape]`` (index 0
    = positive from rank 0, index 1 = negative from rank 1); the combine reduces
    that back to a single tensor of ``local_pred``'s shape.

    Runs on every CFG rank, then computes the deterministic,
    identical-on-all-ranks combine ``neg + scale*(pos - neg)``.

    ``true_cfg_scale`` is a ``[1]`` fp32 device tensor (``NeuronWanModel.cfg_scale_tensor``): a
    new scale reuses the compiled graph, where a Python float would be specialized into it. Its
    0-d element keeps the result in ``local_pred``'s dtype, as the sequential combine
    (``NeuronWanModel._cfg_combine``) does with the same ``scale[0]``.

    The collective is issued on the CFG group's ``device_group``; its replica
    partition is registered in the mesh registry by
    ``distributed.parallel_state.register_replica_groups`` so this lowers to a
    Neuron CC op under fullgraph compile.
    """
    cfg_group = get_cfg_group()
    gathered = cfg_group.all_gather(local_pred, separate_tensors=True)
    positive, negative = gathered[0], gathered[1]
    return negative + true_cfg_scale[0] * (positive - negative)
