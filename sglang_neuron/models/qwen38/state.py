# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8's GDN recurrent state, on the platform's recurrent-state framework (docs/design/recurrent-state.md).

The state is the same family, the same components and the same physical layout as Qwen3.5's, because the layers run
the same kernels: per rank, the temporal state ``(v_heads/tp, head_v, head_k)`` float32 (bfloat16 with
``--mamba-ssm-dtype bfloat16``) stored as the kernels read it (``rec_state [rows, 128, vh, 128]``, token-major) and the
conv window ``(conv_dim/tp, kernel - 1)`` bfloat16 in upstream's q | k | v order. Only the layer ids and the head
counts differ, so the pool and its slot operations are Qwen3.5's, reused unchanged (``Qwen35StatePool``,
``GdnStateOps``).
"""

from __future__ import annotations

import dataclasses

from sglang_neuron.mem_cache.state_spec import StateSpec, gdn_spec
# the pool and its slot operations are layout-driven, not model-specific: reused as they are
from sglang_neuron.models.qwen35.state import GdnStateOps, Qwen35StatePool, kernel_pool  # noqa: F401
from sglang_neuron.models.qwen38.config import Qwen38Dims, Qwen38Layout


class Qwen38StatePool(Qwen35StatePool):
    """The GDN state pool. Same storage and kernel views as Qwen3.5's; the name keeps the launch's refusals readable."""


#: the temporal state dtypes the kernels store (``--mamba-ssm-dtype``): the recurrence runs in fp32 in SBUF either way,
#: the state DMAs casting a bfloat16 state on the way in and out, so bfloat16 halves the state pool per request
TEMPORAL_DTYPES = ("float32", "bfloat16")


def temporal_dtype(server_args=None) -> str:
    """The launch's ``--mamba-ssm-dtype``, float32 when unset; refuses a dtype the kernels do not store."""
    if server_args is None:
        from sglang.srt.runtime_context import get_server_args

        server_args = get_server_args()
    value = str(getattr(server_args, "mamba_ssm_dtype", None) or "float32")
    value = {"bf16": "bfloat16", "fp32": "float32"}.get(value, value)
    if value not in TEMPORAL_DTYPES:
        raise ValueError(f"Qwen3.8 stores the GDN state in one of {TEMPORAL_DTYPES}; --mamba-ssm-dtype is {value!r}")
    return value


def state_spec(dims: Qwen38Dims, layout: Qwen38Layout, temporal: str = "float32") -> StateSpec:
    """The GDN state one rank holds, with the temporal component in the kernels' layout."""
    spec = gdn_spec(layer_ids=dims.gdn_layers, tp_size=layout.tp_size, num_k_heads=dims.num_k_heads,
                    num_v_heads=dims.num_v_heads, head_k_dim=dims.head_k_dim, head_v_dim=dims.head_v_dim,
                    conv_kernel=dims.conv_kernel, temporal_dtype=temporal, conv_dtype="bfloat16",
                    numerics="qwen35-giga")
    temporal = dataclasses.replace(
        spec.recurrent, physical_shape=(dims.head_k_dim, layout.gdn_v_heads, dims.head_v_dim))
    return dataclasses.replace(spec, components=(temporal, *spec.convs))
