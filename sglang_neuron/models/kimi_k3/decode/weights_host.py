# SPDX-License-Identifier: Apache-2.0
"""Host-side pieces of the Kimi-K3 decode weight conversion: the vendored host builders, global tensors, dtypes.

The layouts of the decode kernels come from the host builders of the vendored decode kernel tree (``KERNEL_ROOT``):
each builder takes the global ``weight[out, in]`` tensors of a layer and reads the rows and columns of one rank. The
loader gives it global tensors that hold the rank's shard at its position and zeros elsewhere (``embed``); the zeros
are fresh pages of ``np.zeros``, so the parts the builder does not read cost no memory.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor

KERNEL_ROOT = "sglang_neuron.models.kimi_k3.kernels.decode"

HOST_MODULES = {
    "kda": "kda_decode_torch",
    "mla": "mla_decode_torch",
    "moe_dp": "moe_decode_dp_torch",
    "dense": "multinode.ffn_world_torch",
    "routed": "multinode.moe_routed_layout",
    "draft_host": "draft_mla_host",
    "draft_torch": "draft_mla_torch",
}
"""key -> module under ``KERNEL_ROOT`` of each host builder module."""

MX_ORDER = "hT_quad"
MX_SCALE_SUFFIX = "_mx_scale"
"""A builder input ``name`` given as MXFP8 bytes comes with its E8M0 scales as ``name + MX_SCALE_SUFFIX``."""
KDA_MX = {"kda_w_in": MX_ORDER, "kda_w_out": MX_ORDER}
MLA_MX = {"mla_in_proj": MX_ORDER, "mla_q_b_proj": MX_ORDER, "mla_kv_b_proj": MX_ORDER, "mla_o_proj": MX_ORDER}
MOE_MX = {"moe_in_proj": MX_ORDER}
"""The MXFP8 classes of each builder: every projection of the attention and the MoE input projection take MXFP8
hT_quad groups; the builders quantize the BF16 weights at the ``nisa.quantize_mx`` rule. The column out projection of
the MoE (``kernel_calls.MOE_OUT_PROJ_FORM``) is BF16 (``weights_ffn.column_out_proj``)."""


class HostBuilderUnavailable(ImportError):
    """A host builder module of the vendored decode kernel tree does not import on this host."""


_MODULES: dict[str, Any] = {}


def host_modules() -> SimpleNamespace:
    """Import the host builder modules (``HOST_MODULES``) once; return them by key.

    Raises:
        HostBuilderUnavailable: A module does not import (no vendored decode tree, or no ``nki`` on this host).
    """
    if len(_MODULES) != len(HOST_MODULES):
        # A first import under a meta default device would make module-level tensors meta.
        with torch.device("cpu"):
            for key, module in HOST_MODULES.items():
                path = f"{KERNEL_ROOT}.{module}"
                try:
                    _MODULES[key] = importlib.import_module(path)
                except ImportError as err:
                    raise HostBuilderUnavailable(
                        f"The Kimi-K3 decode host builder module {path} does not import ({err!r}). The decode "
                        f"kernels are vendored under {KERNEL_ROOT}; run from a checkout that has them, on a host "
                        "with nki.") from err
    return SimpleNamespace(**_MODULES)


def lazy_zeros(shape: tuple[int, ...] | list[int], dtype: torch.dtype) -> Tensor:
    """A CPU zero tensor whose pages the OS maps on first touch (``np.zeros`` of a large array)."""
    count = 1
    for size in shape:
        count *= int(size)
    data = np.zeros(count * torch.empty((), dtype=dtype).element_size(), dtype=np.uint8)
    return torch.from_numpy(data).view(dtype).reshape(tuple(shape))


def embed(shard: Tensor, dim: int | None, parts: int, index: int) -> Tensor:
    """A zero global tensor with ``shard`` as part ``index`` of ``parts`` equal parts along ``dim``.

    ``dim`` None returns the shard: a whole tensor.

    Example:
        >>> full = embed(torch.ones(2, 3), 0, 4, 1)
        >>> full.shape, full[2:4].sum().item(), full.sum().item()
        (torch.Size([8, 3]), 6.0, 6.0)
    """
    if dim is None:
        return shard
    if not 0 <= index < parts:
        raise ValueError(f"The shard index {index} is outside {parts} parts.")
    size = shard.shape[dim]
    shape = list(shard.shape)
    shape[dim] = size * parts
    full = lazy_zeros(shape, shard.dtype)
    full.narrow(dim, index * size, size).copy_(shard)
    return full


def embed_row_block(block: Tensor, rows: int, start: int) -> Tensor:
    """A zero ``[rows, ...]`` tensor with ``block`` at rows ``[start, start + len(block))``; ``start`` at or past
    ``rows`` gives all zeros (a rank with no block)."""
    shape = [rows, *block.shape[1:]]
    full = lazy_zeros(shape, block.dtype)
    if start < rows:
        full.narrow(0, start, block.shape[0]).copy_(block)
    return full


def to_numpy(tensor: Tensor) -> np.ndarray:
    """A CPU tensor as numpy, with BF16 as ``ml_dtypes.bfloat16`` (no copy for a contiguous CPU tensor)."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        import ml_dtypes

        return tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return tensor.numpy()


def to_torch(array: Any) -> Tensor:
    """A host builder array as a torch tensor of the same bits (BF16 kept exact)."""
    if isinstance(array, Tensor):
        return array.contiguous()
    array = np.ascontiguousarray(array)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def router_rows(router_weight: Tensor, tp_rank: int, tp: int) -> Tensor:
    """The FP16 router rows of TP rank ``tp_rank``, hT-tiled: ``[128, H / 128, E / tp]``.

    Element ``(p, j, e)`` is ``router_weight[tp_rank * E / tp + e, 128 j + p]`` in FP16 (nearest; the router weights
    hold no value that FP16 overflows).

    Raises:
        ValueError: A router weight overflows FP16.
    """
    experts, hidden = router_weight.shape
    if experts % tp or hidden % 128:
        raise ValueError(f"The router weight {tuple(router_weight.shape)} does not split over TP {tp}.")
    count = experts // tp
    rows = router_weight[tp_rank * count:(tp_rank + 1) * count].to(torch.float16)
    if bool(torch.isinf(rows).any()):
        raise ValueError("The router weight overflows FP16.")
    return rows.t().reshape(hidden // 128, 128, count).permute(1, 0, 2).contiguous()


ROUTER_BIAS_COLUMN = 140
"""The first ``moe_vectors`` column of the router correction biases of all experts: after the 140 columns of the DP
form (the 28 local biases and the 112 latent norm weights of the TP rank)."""


def vectors_with_router_bias(vectors: np.ndarray, router_bias: Tensor) -> Tensor:
    """``moe_vectors`` ``[1, 140 + E]`` FP32: the DP-form columns, then the FP32 router bias of every expert (the
    router rows of the verify take no expert selection, so every rank adds the bias of every expert)."""
    bias = router_bias.detach().float().reshape(1, -1).numpy()
    if vectors.shape[0] != 1 or vectors.shape[1] < ROUTER_BIAS_COLUMN:
        raise ValueError(f"The DP moe_vectors {tuple(vectors.shape)} has no {ROUTER_BIAS_COLUMN} leading columns.")
    start = ROUTER_BIAS_COLUMN
    return torch.from_numpy(np.ascontiguousarray(
        np.concatenate([vectors[:, :start], bias, vectors[:, start:]], axis=1), dtype=np.float32))


def to_hT_order(vector: Tensor) -> Tensor:
    """``[..., N]`` in the hT order of the kernels: element ``p * (N / 128) + j`` is element ``128 j + p``."""
    width = vector.shape[-1]
    lead = vector.shape[:-1]
    return vector.reshape(*lead, width // 128, 128).transpose(-1, -2).contiguous().reshape(*lead, width)


def put_layer(stacks: dict[str, Tensor], arrays: Mapping[str, Any], layer: int, layers: int,
              device: torch.device | str | None = None) -> None:
    """Copy layer ``layer`` of each array into its stack ``[layers, ...]`` on ``device`` (None: the host); the first
    layer allocates the stacks."""
    for name, array in arrays.items():
        value = to_torch(array)
        stack = stacks.get(name)
        if stack is None:
            stack = stacks[name] = torch.empty((layers, *value.shape), dtype=value.dtype, device=device)
        elif tuple(stack.shape[1:]) != tuple(value.shape) or stack.dtype != value.dtype:
            raise ValueError(f"Layer {layer} of {name} is {value.dtype} {tuple(value.shape)}; the stack is "
                             f"{stack.dtype} {tuple(stack.shape[1:])}.")
        stack[layer].copy_(value)
