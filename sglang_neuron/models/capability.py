# SPDX-License-Identifier: Apache-2.0
"""What a model's decode is built for beyond the default launch: the decode launch layouts it serves and the
speculative algorithms its own worker runs. A catalog entry (catalog.ModelSpec) names them, and the platform and the
serving policy lift their refusals for that model alone. Imports nothing, since the catalog imports it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeLaunchLayout:
    """One PD decode server of ``tp_size`` ranks over ``nnodes`` nodes, in ``dp_size`` DP-attention groups (``--dp-size
    dp_size --enable-dp-attention``, attention TP ``tp_size // dp_size``), experts over ``ep_size`` ranks, the KV in DCP
    groups of ``dcp_size`` ranks inside each attention-TP group, every rank on one logical core of
    ``NEURON_LOGICAL_NC_CONFIG=logical_nc``.

    SGLang places ranks node-major (node n holds ranks ``[n * ranks_per_node, (n + 1) * ranks_per_node)``), so every
    attention-TP group, and with it every DCP group, lies inside one node.

    ``runtime_env`` holds the Neuron runtime settings ``(name, value)`` every rank of the layout starts with in place of
    its role's defaults (model_executor.runtime_init.runtime_defaults); a launch environment that sets one keeps its
    value."""

    name: str
    nnodes: int
    tp_size: int
    dp_size: int
    ep_size: int
    dcp_size: int
    logical_nc: int
    runtime_env: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        problems = []
        if min(self.nnodes, self.tp_size, self.dp_size, self.ep_size, self.dcp_size) < 1:
            problems.append("every width is at least 1")
        elif self.tp_size % self.nnodes or self.tp_size % self.dp_size:
            problems.append("--tp-size splits evenly over the nodes and the DP-attention groups")
        elif self.attention_tp % self.dcp_size:
            problems.append("the DCP size divides the attention TP")
        elif self.ranks_per_node % self.attention_tp:
            problems.append("every attention-TP group lies inside one node")
        if self.logical_nc not in (1, 2):
            problems.append("NEURON_LOGICAL_NC_CONFIG is 1 or 2")
        if problems:
            raise ValueError(f"decode launch layout {self.name}: {'; '.join(problems)}")

    @property
    def attention_tp(self) -> int:
        return self.tp_size // self.dp_size

    @property
    def ranks_per_node(self) -> int:
        return self.tp_size // self.nnodes

    def flags(self) -> str:
        """The layout as the launch flags that select it."""
        return (f"--nnodes {self.nnodes} --tp-size {self.tp_size} --dp-size {self.dp_size} --enable-dp-attention "
                f"--ep-size {self.ep_size} --dcp-size {self.dcp_size}")
