# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from safetensors import PySafeSlice

_WEIGHT_LOADER_ATTR = "weight_loader"


@dataclass
class ShardSpec:
    """A shard along ``dim``: ``num_shards`` shards (usually the TP size) of ``size`` each."""

    dim: int
    size: int
    num_shards: int


@dataclass
class SafetensorsWeightLoader:
    """How a parameter is built from its checkpoint slices: ``transform(slices, rank)`` gets one ``PySafeSlice`` per
    checkpoint key of the parameter's mapping and this rank, and returns the parameter's tensor (a TP shard, a fused
    QKV, a padded or relaid tensor). Without a transform the single slice loads as is.

    For example, ``transform=lambda slices, rank: slices[0][rank * 64:(rank + 1) * 64, :]`` is a column-parallel shard
    and ``lambda slices, rank: torch.cat([s[:] for s in slices], dim=0)`` fuses several checkpoint tensors."""

    transform: Callable[[list["PySafeSlice"], int], torch.Tensor] = None

    def load(self, slices: list["PySafeSlice"], rank: int) -> torch.Tensor:
        """The parameter's tensor from its checkpoint slices, contiguous."""
        if self.transform:
            result = self.transform(slices, rank)
        else:
            # By default, load first slice as-is
            assert len(slices) == 1, (
                f"SafetensorsWeightLoader without transform should only take in a single slice but got {len(slices)}"
            )
            result = slices[0][:]

        return result.contiguous()


def set_weight_loader(param: nn.Parameter, loader: SafetensorsWeightLoader) -> None:
    """Attach ``loader`` to ``param``; the checkpoint load calls it for that parameter."""
    setattr(param, _WEIGHT_LOADER_ATTR, loader)


def get_weight_loader(param: nn.Parameter) -> SafetensorsWeightLoader:
    """``param``'s loader, or the identity loader where none is attached."""
    return getattr(param, _WEIGHT_LOADER_ATTR, SafetensorsWeightLoader())


def with_rank_override(
    loader: SafetensorsWeightLoader, rank: int
) -> SafetensorsWeightLoader:
    """``loader`` called with ``rank`` instead of the rank the load passes: a weight sharded over a subgroup (an
    attention-TP group under attention DP) is sharded by the rank within that group, not the global TP rank."""
    return SafetensorsWeightLoader(
        transform=lambda slices, _: loader.load(slices, rank)
    )


# Common transform factories


def sharding_weight_loader(
    shard_dim: int,
    shard_size: int,
    num_shards: int,
    is_storage_transposed: bool = False,
    pad_shard: bool = False,
) -> SafetensorsWeightLoader:
    """This rank's shard along ``shard_dim`` of the parameter, read from the checkpoint alone (``get_shard``).

    ``is_storage_transposed``: the checkpoint stores the 2-D weight transposed against the parameter, so the shard is
    cut on the other dim and transposed. ``pad_shard``: zero-pad a short last shard to ``shard_size`` (a vocabulary
    the TP size does not divide). More than one tensor, or a transposed one of more than 2 dims, is refused."""

    def transform(slices: list["PySafeSlice"], rank: int) -> torch.Tensor:
        if len(slices) != 1:
            raise ValueError("sharding_weight_loader() only supports a single tensor")
        slice_obj = slices[0]
        if is_storage_transposed and len(slice_obj.get_shape()) > 2:
            raise ValueError("sharding_weight_loader() does not support transposed storage for tensors with more than "
                             "2 dims")
        storage_dim = 1 - shard_dim if is_storage_transposed and shard_dim in (0, 1) else shard_dim
        result = get_shard(slice_obj, storage_dim, shard_size, num_shards, rank)
        if is_storage_transposed:
            result = result.T
        if pad_shard:
            target_shape = list(result.shape)
            target_shape[shard_dim] = shard_size
            result = pad_to_shape(result, tuple(target_shape))
        return result

    return SafetensorsWeightLoader(transform=transform)


def _contiguous_expert_bounds(local_expert_indices: list) -> tuple[int, int]:
    """Validate a contiguous expert range and return ``(lo, hi)`` inclusive.

    Only contiguous expert ranges (e.g. ``[ep_rank*L, (ep_rank+1)*L)``) are
    supported; non-contiguous indices are rejected.
    """
    if len(local_expert_indices) == 0:
        raise ValueError("local_expert_indices cannot be empty")
    lo = local_expert_indices[0]
    hi = local_expert_indices[-1]
    if list(local_expert_indices) != list(range(lo, hi + 1)):
        raise ValueError(
            "expert-parallel weight loaders require contiguous "
            f"local_expert_indices, got {local_expert_indices}"
        )
    return lo, hi


class SliceView:
    """Lazy, composable view over a ``PySafeSlice`` (or tensor).

    Presents the checkpoint-slice interface (``get_shape()`` + ``__getitem__``)
    over a sub-region of ``base`` defined by a per-base-dim ``index`` (slices
    keep a dim, ints collapse it). Indexing the view composes the new key into
    that base selection and performs a SINGLE read on ``base``, so a caller
    that restricts one dim (e.g. an expert-parallel range on dim 0) and an
    inner loader that shards a different dim (e.g. a TP shard on dim 1)
    together materialize only the final shape -- no intermediate over-read.

    This exists because ``PySafeSlice`` (a native pyo3 type) supports only a
    single lazy ``__getitem__`` (which materializes a tensor) and cannot be
    subclassed or sub-sliced lazily. ``SliceView`` defers that one read until
    the composed selection is known.

    Supported index forms: ``slice``, ``int`` and ``Ellipsis`` -- the forms
    weight loaders use. ``None``/newaxis and integer lists are rejected
    (it inserts a dim with no backing data -- a reshape, not a sub-selection;
    ``unsqueeze`` the result instead), as is boolean/tensor advanced indexing.
    Unsupported forms raise so a gap fails loudly rather than silently slicing
    the wrong region.
    """

    def __init__(self, base, index=None):
        self._base = base
        shape = self._base_shape()
        if index is None:
            index = tuple(slice(None) for _ in shape)
        if len(index) != len(shape):
            raise ValueError(
                f"SliceView index length {len(index)} != base ndim {len(shape)}"
            )
        self._index = tuple(self._normalize(sel, n) for sel, n in zip(index, shape))

    def _base_shape(self) -> tuple[int, ...]:
        get_shape = getattr(self._base, "get_shape", None)
        return tuple(get_shape()) if get_shape is not None else tuple(self._base.shape)

    @staticmethod
    def _normalize(sel, n):
        """Resolve a selector against dim length n to concrete form."""
        if isinstance(sel, slice):
            return slice(*sel.indices(n))  # concrete (start, stop, step)
        if isinstance(sel, bool):  # a bool is an int to Python; as an index it is a mask, which is not supported
            raise TypeError(f"SliceView does not support boolean index {sel!r}; only slice / int")
        if isinstance(sel, int):
            return sel + n if sel < 0 else sel
        raise TypeError(f"SliceView selector must be slice or int, got {sel!r}")

    @staticmethod
    def _slice_len(s: slice) -> int:
        return len(range(s.start, s.stop, s.step))

    def get_shape(self) -> tuple[int, ...]:
        """Shape of the view: lengths of the kept (slice) dims, in order."""
        return tuple(
            self._slice_len(sel) for sel in self._index if isinstance(sel, slice)
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.get_shape()

    def _expand_ellipsis(self, key: tuple, n_view_dims: int) -> tuple:
        """Replace a single Ellipsis with full slices covering the unspecified
        view dims (numpy/torch semantics)."""
        n_ellipsis = sum(1 for k in key if k is Ellipsis)
        if n_ellipsis == 0:
            return key
        if n_ellipsis > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        non_ellipsis = [k for k in key if k is not Ellipsis]
        fill = n_view_dims - len(non_ellipsis)
        if fill < 0:
            raise IndexError("too many indices for SliceView")
        out = []
        for k in key:
            if k is Ellipsis:
                out.extend([slice(None)] * fill)
            else:
                out.append(k)
        return tuple(out)

    def _compose(self, base_sel: slice, key):
        """Compose an incoming key element (view space) with a base slice
        (underlying space), returning a selector in underlying space."""
        n = self._slice_len(base_sel)
        if isinstance(key, bool):
            raise TypeError(f"SliceView does not support boolean index {key!r}; only slice / int")
        if isinstance(key, slice):
            ks, ke, kstep = key.indices(n)
            new_start = base_sel.start + base_sel.step * ks
            new_step = base_sel.step * kstep
            new_stop = base_sel.start + base_sel.step * ke
            return slice(new_start, new_stop, new_step)
        if isinstance(key, int):
            k = key + n if key < 0 else key
            if not (0 <= k < n):
                raise IndexError(f"index {key} out of range for dim of size {n}")
            return base_sel.start + base_sel.step * k  # int -> collapses dim
        raise TypeError(
            f"SliceView does not support index element {key!r}; "
            f"only slice / int."
        )

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is None for k in key):
            raise TypeError(
                "SliceView does not support newaxis (None); unsqueeze the "
                "materialized result instead"
            )
        view_dims = [d for d, sel in enumerate(self._index) if isinstance(sel, slice)]
        key = self._expand_ellipsis(key, len(view_dims))
        if len(key) > len(view_dims):
            raise IndexError(
                f"too many indices: got {len(key)} for {len(view_dims)} view dims"
            )
        # Pad with full slices for unspecified trailing view dims.
        key = list(key) + [slice(None)] * (len(view_dims) - len(key))

        composed = list(self._index)
        for vd, k in zip(view_dims, key):
            composed[vd] = self._compose(self._index[vd], k)
        return self._base[tuple(composed)]


# Expert-parallel (EP) weight loader.
#
# Each rank owns a contiguous range of experts. expert_parallel_tensor_dim_loader
# restricts the inner loader's INPUT to the local experts *before* it runs: each
# checkpoint tensor (experts on ``expert_dim``) is wrapped in a SliceView over the
# local expert range, so per-expert work (e.g. MXFP4 dequant) only touches local
# experts and non-local experts are never read.
#
# ``original_loader`` must be constructed for ``num_local_experts`` and must
# define a transform (the wrapper calls ``original_loader.transform`` directly;
# a bare ``SafetensorsWeightLoader()`` is not supported).


def expert_parallel_tensor_dim_loader(
    local_expert_indices: list,
    original_loader: SafetensorsWeightLoader,
    expert_dim: int = 0,
) -> SafetensorsWeightLoader:
    """An expert-parallel loader for checkpoint tensors whose ``expert_dim`` indexes the experts (GPT-OSS ``[E, ...]``
    weights, scales and biases): each slice is narrowed to this rank's contiguous expert range with a ``SliceView``
    before ``original_loader`` runs, so its own indexing (a TP shard on another dim) composes into one read of just the
    final shape."""
    lo, hi = _contiguous_expert_bounds(local_expert_indices)

    def transform(slices: list, rank: int) -> torch.Tensor:
        local_slices = []
        for s in slices:
            index = [slice(None)] * len(s.get_shape())
            index[expert_dim] = slice(lo, hi + 1)
            local_slices.append(SliceView(s, index=index))
        return original_loader.transform(local_slices, rank)

    return SafetensorsWeightLoader(transform=transform)


# Common functions for composability


def get_shard(
    slice_obj: "PySafeSlice", dim: int, shard_size: int, num_shards: int, rank: int
) -> torch.Tensor:
    """``rank``'s shard along ``dim`` of a checkpoint slice, reading only that shard: ``[rank * shard_size, (rank + 1) *
    shard_size)`` (rank modulo ``num_shards``). An unpadded checkpoint gives a shorter last shard, which the caller
    pads: 119 rows in 3 shards of 40 give the last rank 39, and in 4 shards none."""
    shape = slice_obj.get_shape()

    # clamped by hand: older safetensors raise on a start or stop past the dimension instead of clamping it
    start = min((rank % num_shards) * shard_size, shape[dim])
    end = min(start + shard_size, shape[dim])

    slices = [slice(None)] * len(shape)
    slices[dim] = slice(start, end)

    return slice_obj[tuple(slices)]


def get_shard_deinterleaved(
    slice_obj: "PySafeSlice",
    dim: int,
    shard_size: int,
    num_shards: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rank``'s ``(gate, up)`` shards of a checkpoint that interleaves them along ``dim`` (``g0, u0, g1, u1, ...``):
    ``shard_size`` (gate plus up) interleaved entries per rank, split into the even and the odd ones. With
    ``shard_size`` 4 over 2 ranks, rank 0 gets gate ``[g0, g1]`` and up ``[u0, u1]``. An unpadded checkpoint gives
    shorter last shards, which the caller pads."""
    shape = slice_obj.get_shape()
    ndim = len(shape)

    # clamped by hand, as in get_shard
    start = min((rank % num_shards) * shard_size, shape[dim])
    end = min(start + shard_size, shape[dim])

    # Gate: even interleaved positions [start : end : 2]
    sl_gate = [slice(None)] * ndim
    sl_gate[dim] = slice(start, end, 2)
    gate_shard = slice_obj[tuple(sl_gate)]

    # Up: odd interleaved positions [start + 1 : end + 1 : 2]
    sl_up = [slice(None)] * ndim
    sl_up[dim] = slice(min(start + 1, shape[dim]), min(start + shard_size + 1, shape[dim]), 2)
    up_shard = slice_obj[tuple(sl_up)]

    return gate_shard, up_shard


def pad_to_shape(tensor: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """``tensor`` zero-padded at the end of each dim to ``target_shape``. A target with another number of dims, or a dim
    smaller than the tensor's, raises ``ValueError``: a checkpoint tensor larger than the config expects is refused,
    never cropped."""
    target = tuple(int(d) for d in target_shape)
    if tensor.ndim != len(target) or any(a > t for a, t in zip(tensor.shape, target)):
        raise ValueError(
            f"pad_to_shape: cannot pad a tensor of shape {tuple(tensor.shape)} to {target}; the target must have the "
            "same number of dims and no dim smaller than the tensor's (a checkpoint tensor larger than the model "
            "config expects is refused, not cropped)")
    if tuple(tensor.shape) == target:
        return tensor

    pad = []
    for actual, target in zip(reversed(tensor.shape), reversed(target_shape)):
        pad.extend([0, target - actual])

    return torch.nn.functional.pad(tensor, pad)
