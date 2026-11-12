# SPDX-License-Identifier: Apache-2.0
"""Top-k of one device's tensor for the device sampler's candidate pool, by an NKI kernel: GpSIMD's nisa.topk where
it applies (Trainium3, bf16, within its bounds), else nkilib's rotational kernel. torch.topk runs on CPU tensors only;
on a device tensor it lowers to a custom call that returns wrong candidates after an all-gather, so a device tensor no
kernel takes is refused and the sampler takes its exact reduction instead (sampling._exact_topk)."""

import logging
from enum import Enum
from typing import Optional

import nki.language as nl
import torch
from nkilib.core.topk.rotational_topk import rotational_topk
from nkilib.core.topk.rotational_topk_utils import create_rotational_topk_config, create_topk_config
from nkilib.experimental.topk.gpsimd_topk import create_gpsimd_topk_config, gpsimd_topk
from torch import Tensor
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from sglang_neuron.backend import envs
from sglang_neuron.backend.stack import get_platform_target, torch_to_nki_dtype, wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

# nisa.topk's bounds: 8 <= vocab with the vocab padded to 16 under 65536, and 0 < k < 32768
_GPSIMD_MIN_VOCAB, _GPSIMD_MAX_VOCAB, _GPSIMD_MAX_K = 8, 65536, 32768


class TopkMethod(Enum):
    ROTATIONAL = 0
    GPSIMD = 1


def topk(tensor: Tensor, k: int, dim: int = -1, gather_dim: Optional[int] = None) -> tuple[Tensor, Tensor]:
    """``(values, int64 indices)`` of the k largest along the last dimension, sorted descending. ``gather_dim``, the
    dimension a caller's pool was gathered along, must be the reduced one: the input is one device's tensor."""
    dim = dim % tensor.dim()
    if gather_dim is not None and gather_dim % tensor.dim() != dim:
        raise ValueError(f"topk reduces one device's tensor: gather_dim={gather_dim} must be None or dim={dim}")
    method = select_topk_method(tensor, k, dim)
    if method is not None:
        return _topk_nki(tensor, k, method)
    if tensor.device.type != "cpu":
        raise RuntimeError(f"no NKI top-k kernel takes {tuple(tensor.shape)} {tensor.dtype} with k={k} on "
                           f"{tensor.device}; use an exact reduction (sampling._exact_topk)")
    return torch.topk(tensor, k, dim=dim)


def select_topk_method(tensor: Tensor, k: int, dim: int) -> Optional[TopkMethod]:
    """The kernel for this top-k, or None. Both kernels reduce the last dimension, and neither takes k == vocab (a sort
    of the whole row)."""
    if not can_run_kernel(tensor) or tensor.ndim < 2 or dim % tensor.ndim != tensor.ndim - 1:
        return None
    vocab = tensor.shape[-1]
    if not 0 < k < vocab:
        return None
    if (tensor.dtype == torch.bfloat16 and _is_trn3() and vocab >= _GPSIMD_MIN_VOCAB
            and -(-vocab // 16) * 16 < _GPSIMD_MAX_VOCAB and k < _GPSIMD_MAX_K):
        return TopkMethod.GPSIMD
    if tensor.dtype in (torch.bfloat16, torch.float32) and _rotational_config_builds(
            tensor.numel() // vocab, vocab, k, getattr(nl, torch_to_nki_dtype(tensor.dtype))):
        return TopkMethod.ROTATIONAL
    return None


# The helpers below run host code (a runtime query, nkilib's config factories and their logging) that dynamo cannot
# trace under fullgraph; assume_constant_result runs them once per traced shape and folds the result in.

@torch._dynamo.assume_constant_result
def _is_trn3() -> bool:
    # nisa.topk exists from the third NeuronCore generation, which only Trainium3 has
    return get_platform_target().startswith("trn3")


def _programs() -> int:
    """The SPMD grid of the kernels: the LNC degree the graph runs at, 1 on the CPU."""
    return 1 if envs.SGLANG_NEURON_CPU_MODE else get_logical_nc_config()


@torch._dynamo.assume_constant_result
def _rotational_config(rows: int, vocab: int, k: int, nki_dtype):
    topk_config = create_topk_config(inp_shape=(rows, vocab), inp_dtype=nki_dtype, k=k, num_programs=_programs())
    return create_rotational_topk_config(inp_shape=(rows, vocab), topk_config=topk_config)


@torch._dynamo.assume_constant_result
def _rotational_config_builds(rows: int, vocab: int, k: int, nki_dtype) -> bool:
    """Whether the rotational kernel compiles for this shape. Its SBUF bound is not monotonic in k, so no static cap
    captures it; building the kernel's own config answers it. The factories refuse a shape by assertion (stripped
    under python -O, the one mode where this check is unsafe); any other exception is a real error and propagates."""
    try:
        _rotational_config(rows, vocab, k, nki_dtype)
        return True
    except AssertionError:
        return False


@torch._dynamo.assume_constant_result
def _gpsimd_config(rows: int, vocab: int, k: int, nki_dtype):
    return create_gpsimd_topk_config(inp_shape=(rows, vocab), inp_dtype=nki_dtype, k=k, num_programs=_programs())


@torch._dynamo.assume_constant_result
def _log_kernel(kernel: str, rows: int, vocab: int, k: int) -> None:
    logger.info("[topk] kernel=%s rows=%d vocab=%d k=%d", kernel, rows, vocab, k)


_KERNELS = {
    TopkMethod.ROTATIONAL: (rotational_topk, _rotational_config, "rotational-nki"),
    TopkMethod.GPSIMD: (gpsimd_topk, _gpsimd_config, "gpsimd-nki"),
}


def _topk_nki(tensor: Tensor, k: int, method: TopkMethod) -> tuple[Tensor, Tensor]:
    # An all-gathered pool can arrive as a lazy AsyncCollectiveTensor, which the NKI call does not dispatch through.
    if isinstance(tensor, AsyncCollectiveTensor):
        tensor = tensor.wait()
    vocab = tensor.shape[-1]
    rows = tensor.reshape(-1, vocab)
    kernel, build_config, label = _KERNELS[method]
    _log_kernel(label, rows.shape[0], vocab, k)
    config = build_config(rows.shape[0], vocab, k, getattr(nl, torch_to_nki_dtype(rows.dtype)))
    values, indices = wrap_nki(kernel)[_programs()](rows, config)
    shape = (*tensor.shape[:-1], k)
    return values.reshape(shape), indices.reshape(shape).to(torch.int64)
