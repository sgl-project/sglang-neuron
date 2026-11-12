# SPDX-License-Identifier: Apache-2.0
"""The recurrent-state declaration a model makes: which global layers carry state, which components each holds per
request slot (shape, dtype, physical layout), and how they compare with upstream SGLang's cache parameters.

One declaration serves every recurrent layer family the platform hosts: KDA (``kda``), the gated delta net (``gdn``,
HF ``layer_types`` value ``linear_attention``) and a Mamba-2 SSM (``mamba2``). A model declares one ``StateSpec`` for
the whole model; each pipeline stage's pool keeps the layers SGLang gives it (``StateSpec.stage``), so the pool, its
radix checkpoints and its PD entries agree on the components, their order, their bytes and their global layer ids by
construction.

Upstream's ``MambaPool`` holds one temporal tensor and a list of conv tensors per state layer, each
``[layers, slots + 1, *per-slot shape]``; a component here is one of those tensors. ``logical_shape`` is upstream's
per-slot shape (``cache_params.shape.temporal`` or ``cache_params.shape.conv[i]``); ``physical_shape`` is a layout a
model's kernel reads instead, with the same element count. Upstream sizes and transfers a slot by bytes, so only the
element count and dtype must agree.

Layer identity under PP: ``layer_ids`` are global layer indices, in the pool's buffer order, and every key derived from
them (checkpoints, PD transfer pairing) is global.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from math import prod
from typing import Iterable, Optional, Tuple

# state layer families with a defined meaning here
KDA = "kda"
GDN = "gdn"
MAMBA2 = "mamba2"
FAMILIES = (KDA, GDN, MAMBA2)

# HF layer_types values that name a recurrent family
LAYER_TYPE_FAMILY = {"linear_attention": GDN, "kda": KDA}

# component roles
CONV = "conv"
RECURRENT = "recurrent"

_DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "float8_e4m3fn": 1, "float8_e5m2": 1, "int8": 1}


def dtype_name(dtype) -> str:
    """'float32' for torch.float32 or 'float32'; refuses a dtype the platform has no state kernels for."""
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPE_BYTES:
        raise ValueError(f"state dtype {dtype!r} is not one the platform stores recurrent state in "
                         f"({', '.join(sorted(_DTYPE_BYTES))})")
    return name


@dataclass(frozen=True)
class StateComponent:
    """One per-slot state tensor of every state layer: upstream's ``temporal`` (role recurrent) or one ``conv[i]``."""

    name: str
    role: str
    logical_shape: Tuple[int, ...]
    dtype: str
    physical_shape: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.role not in (CONV, RECURRENT):
            raise ValueError(f"state component {self.name!r}: role {self.role!r} is neither {CONV} nor {RECURRENT}")
        object.__setattr__(self, "logical_shape", tuple(int(d) for d in self.logical_shape))
        object.__setattr__(self, "dtype", dtype_name(self.dtype))
        if not self.logical_shape or min(self.logical_shape) <= 0:
            raise ValueError(f"state component {self.name!r}: shape {self.logical_shape} has an empty axis")
        if self.physical_shape is not None:
            physical = tuple(int(d) for d in self.physical_shape)
            object.__setattr__(self, "physical_shape", physical)
            if prod(physical) != prod(self.logical_shape):
                raise ValueError(f"state component {self.name!r}: physical layout {physical} holds "
                                 f"{prod(physical)} elements, upstream's {self.logical_shape} {prod(self.logical_shape)}")

    @property
    def shape(self) -> Tuple[int, ...]:
        """The per-slot shape the pool allocates: the kernel's physical layout when declared."""
        return self.physical_shape or self.logical_shape

    @property
    def item_bytes(self) -> int:
        return _DTYPE_BYTES[self.dtype]

    @property
    def slot_bytes(self) -> int:
        """Bytes of one slot of one layer: the unit PD transfers per (layer, slot)."""
        return prod(self.logical_shape) * self.item_bytes


@dataclass(frozen=True)
class StateSpec:
    """The recurrent state one pipeline stage holds: a family, its components (temporal first, then conv in upstream's
    order) and the global ids of its state layers, in pool buffer order.

    ``working_rows``: a model whose kernel advances state in a separate working bank (canonical checkpoint slots plus
    working rows the kernel reads and writes, loaded before and published after a forward) declares the bank's row
    count, scratch row 0 included; None when the kernel updates the canonical slots in place.

    ``track_chunk``: the token grid at which the prefill kernel can capture state inside a forward (its core tile,
    upstream's mamba_chunk_size); 0 when it can capture only the state at a forward's end.

    ``conv_slice_axis``: the per-slot axis of the conv components that holds the attention-TP-sharded channels, when
    the declared conv shape orders its axes unlike upstream's cache parameters; None keeps upstream's axis."""

    family: str
    components: Tuple[StateComponent, ...]
    layer_ids: Tuple[int, ...]
    working_rows: Optional[int] = None
    track_chunk: int = 0
    tp_size: int = 1
    numerics: str = "default"
    extra: Tuple[Tuple[str, object], ...] = field(default_factory=tuple)
    conv_slice_axis: Optional[int] = None

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"state family {self.family!r} is not one the platform hosts ({', '.join(FAMILIES)})")
        object.__setattr__(self, "layer_ids", tuple(int(i) for i in self.layer_ids))
        object.__setattr__(self, "components", tuple(self.components))
        if not self.layer_ids:
            raise ValueError(f"{self.family} state spec declares no state layers")
        if len(set(self.layer_ids)) != len(self.layer_ids) or min(self.layer_ids) < 0:
            raise ValueError(f"{self.family} state layer ids {list(self.layer_ids)} repeat or are negative")
        roles = [c.role for c in self.components]
        if roles.count(RECURRENT) != 1 or roles[0] != RECURRENT or CONV not in roles:
            raise ValueError(f"{self.family} state spec needs one recurrent component first and at least one conv "
                             f"component (upstream MambaPool's temporal + conv list), got {roles}")
        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.family} state component names repeat: {names}")
        if self.working_rows is not None and int(self.working_rows) < 2:
            raise ValueError("a working bank needs a scratch row and at least one request row")
        if self.track_chunk < 0 or self.tp_size < 1:
            raise ValueError("track_chunk must be >= 0 and tp_size >= 1")

    # -- views --------------------------------------------------------------------------------------------------
    @property
    def recurrent(self) -> StateComponent:
        return self.components[0]

    @property
    def convs(self) -> Tuple[StateComponent, ...]:
        return self.components[1:]

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)

    def slot_bytes(self) -> int:
        """Bytes one request slot holds over every state layer of the stage (upstream mamba_cache_per_req)."""
        return sum(c.slot_bytes for c in self.components) * self.num_layers

    def layer_index(self, layer_id: int) -> int:
        """Buffer index of a global layer id; raises for a layer this stage does not hold."""
        try:
            return self.layer_ids.index(int(layer_id))
        except ValueError:
            raise KeyError(f"global layer {layer_id} carries no {self.family} state on this stage "
                           f"(state layers {list(self.layer_ids)})") from None

    def stage(self, start_layer: int, end_layer: int) -> "StateSpec":
        """The spec of the pipeline stage holding global layers [start_layer, end_layer): its own state layers, with
        their global ids. A stage without state layers has no state spec (None is the caller's)."""
        ids = tuple(i for i in self.layer_ids if start_layer <= i < end_layer)
        if not ids:
            raise ValueError(f"stage [{start_layer}, {end_layer}) holds none of the {self.family} state layers")
        from dataclasses import replace

        return replace(self, layer_ids=ids)

    # -- identity -----------------------------------------------------------------------------------------------
    def describe(self) -> dict:
        """Everything a peer or a cached checkpoint must agree on, rank-normalized (no slot counts, no ranks)."""
        return {
            "family": self.family,
            "components": [(c.name, c.role, list(c.logical_shape), c.dtype, list(c.shape)) for c in self.components],
            "layers": list(self.layer_ids),
            "working": self.working_rows is not None,
            "track_chunk": self.track_chunk,
            "tp": self.tp_size,
            "numerics": self.numerics,
        }

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.describe(), sort_keys=True).encode()).hexdigest()

    # -- upstream agreement -------------------------------------------------------------------------------------
    def check_cache_params(self, cache_params, mamba_layer_ids: Iterable[int]) -> None:
        """Raise unless upstream's resolved cache parameters size the same state this spec declares: the same global
        layers in the same order, temporal and conv per-slot element counts and dtypes, and the family (upstream's
        ``is_kda``). Upstream resolves the temporal dtype from the checkpoint and --mamba-ssm-dtype
        (SGLANG_MAMBA_SSM_DTYPE), the conv dtype from SGLANG_MAMBA_CONV_DTYPE; a disagreement is a launch refusal,
        never a silent cast."""
        problems = []
        ids = tuple(int(i) for i in mamba_layer_ids)
        if ids != self.layer_ids:
            problems.append(f"layers {list(ids)} != declared {list(self.layer_ids)}")
        shape = cache_params.shape
        if prod(tuple(shape.temporal)) != prod(self.recurrent.logical_shape):
            problems.append(f"temporal {tuple(shape.temporal)} != declared {self.recurrent.logical_shape}")
        convs = [tuple(c) for c in shape.conv]
        if [prod(c) for c in convs] != [prod(c.logical_shape) for c in self.convs]:
            problems.append(f"conv {convs} != declared {[c.logical_shape for c in self.convs]}")
        dtype = cache_params.dtype
        if dtype_name(dtype.temporal) != self.recurrent.dtype:
            problems.append(f"temporal dtype {dtype_name(dtype.temporal)} != declared {self.recurrent.dtype} (launch "
                            f"with --mamba-ssm-dtype {self.recurrent.dtype})")
        if any(dtype_name(dtype.conv) != c.dtype for c in self.convs):
            problems.append(f"conv dtype {dtype_name(dtype.conv)} != declared {[c.dtype for c in self.convs]} "
                            "(SGLANG_MAMBA_CONV_DTYPE)")
        is_kda = bool(getattr(cache_params, "is_kda", False))
        if is_kda != (self.family == KDA):
            problems.append(f"upstream is_kda={is_kda} for a {self.family} declaration")
        if problems:
            raise ValueError(f"{self.family} state: upstream's resolved cache parameters disagree with the model's "
                             "declaration (" + "; ".join(problems) + "); launch with the dtypes and layout the "
                             "model's kernels were built for")


def gdn_spec(*, layer_ids, tp_size: int, num_k_heads: int, num_v_heads: int, head_k_dim: int, head_v_dim: int,
             conv_kernel: int, temporal_dtype="float32", conv_dtype="bfloat16", **kw) -> StateSpec:
    """Gated delta net state per rank (upstream Mamba2StateShape.create for GDN): temporal ``(v_heads/tp, head_v_dim,
    head_k_dim)``, conv ``(conv_dim/tp, kernel - 1)`` with conv_dim = 2*K + V (q, k, v)."""
    for name, value in (("num_k_heads", num_k_heads), ("num_v_heads", num_v_heads)):
        if value % tp_size:
            raise ValueError(f"GDN {name}={value} does not shard over attention TP {tp_size}")
    conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    return StateSpec(
        family=GDN, layer_ids=tuple(layer_ids), tp_size=tp_size,
        components=(StateComponent("temporal", RECURRENT, (num_v_heads // tp_size, head_v_dim, head_k_dim),
                                   temporal_dtype),
                    StateComponent("conv0", CONV, (conv_dim // tp_size, conv_kernel - 1), conv_dtype)),
        extra=(("conv_shard_groups", (num_k_heads * head_k_dim, num_k_heads * head_k_dim, num_v_heads * head_v_dim)),),
        **kw)


WINDOW_FIRST = "window_first"
CHANNEL_MAJOR = "channel_major"


def kda_spec(*, layer_ids, tp_size: int, num_heads: int, head_dim: int, conv_kernel: int,
             temporal_dtype="float32", conv_dtype="bfloat16", conv_physical=None, conv_layout: str = WINDOW_FIRST,
             **kw) -> StateSpec:
    """KDA state per rank (upstream KimiLinearStateShape.create, q/k/v heads equal): temporal ``(heads/tp, dim,
    dim)``, conv ``(kernel - 1, 3 * heads/tp * dim)``, window first (upstream's), or with ``conv_layout=CHANNEL_MAJOR``
    ``(3 * heads/tp * dim, kernel - 1)``, taps last, the layout the KDA kernels read and write; either way channel
    ``j * heads/tp * dim + h * dim + d`` for ``j`` in (q, k, v). The dtypes default to upstream's (fp32 temporal unless
    --mamba-ssm-dtype, bf16 conv); ``conv_physical`` names a kernel layout of the conv window instead."""
    if num_heads % tp_size:
        raise ValueError(f"KDA num_heads={num_heads} does not shard over TP {tp_size}")
    if conv_layout not in (WINDOW_FIRST, CHANNEL_MAJOR):
        raise ValueError(f"KDA conv layout {conv_layout!r} is neither {WINDOW_FIRST} nor {CHANNEL_MAJOR}")
    local = num_heads // tp_size
    channels, taps = 3 * local * head_dim, conv_kernel - 1
    conv_shape, axis = ((taps, channels), None) if conv_layout == WINDOW_FIRST else ((channels, taps), 0)
    return StateSpec(
        family=KDA, layer_ids=tuple(layer_ids), tp_size=tp_size,
        components=(StateComponent("temporal", RECURRENT, (local, head_dim, head_dim), temporal_dtype),
                    StateComponent("conv0", CONV, conv_shape, conv_dtype, conv_physical)),
        extra=(("conv_shard_groups", (num_heads * head_dim,) * 3),), conv_slice_axis=axis,
        **kw)
