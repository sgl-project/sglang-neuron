# SPDX-License-Identifier: Apache-2.0
"""A megakernel's pool layout: which of SGLang's pool tensors its programs take, in which order, and which they write
in place and hand back as aliases.

SGLang owns every KV and state pool (allocation, sizing, prefix sharing, PD transfer). A whole-model kernel reads and
writes them in place: the phase graph takes each pool tensor as an input and returns the kernel's in-place-written
buffer as an output, which torch-neuronx aliases back to the input, so the write lands in SGLang's tensor and nothing
is copied. A ``PoolLayout`` declares that contract per model:

- ``PoolEntry``: one named group of pool tensors (``buffers`` of them, holding ``layers``, ``per_buffer`` layers per
  tensor) of one ``kind``, read through ``resolve(kv_pool, req_to_token_pool)``; ``aliased`` when the kernel writes it
  and returns it, ``zero_on_free`` when the kernel reads whole pages and so needs every freed page zeroed (a gather past
  a sequence's end must read zeros: a masked score does not protect ``0 * inf`` in the PV product).
- Layer ids are global (the model's layer index, not the stage-local module index), so a pipeline stage's layout
  names the layers that stage holds.
- ``pool_classes`` / ``state_pool_class``: the pool classes SGLang's KV cache configurator builds for the model
  (``pool_class(kind)``, the interface the platform seams read).
- ``bind`` resolves every entry against the live pools once they exist, checks the declared counts, one device and no
  tensor bound twice, and returns ``BoundPools``: the tensors in declaration order and their identities. A pool SGLang
  reallocates after the bind (a resize, a flush that rebuilds the pool) is refused at the next step's
  ``BoundPools.check`` rather than read through stale tensors.

The kinds are the closed set the platform knows how to size, transfer and checkpoint; a kind outside it is refused at
declaration. A spec that needs another kind (an MLA latent, a DSA index-K pool, recurrent or conv state, a token
relay) adds it here with its resolver; request state (recurrent or conv state pools) is served by the pool-layout
seam's ``state_pool_class`` (docs/design/megakernel.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

# kv: one K or V tensor per layer (SGLang's MHA pool); kv_pair: the GPT-OSS giga-kernel's paired-buffer form (one
# tensor per layer pair, kv_layout.KVLayout.paired_layers); mla_latent: the per-layer latent (ckv | kpe) of an MLA
# pool; index_k: a DSA indexer's per-layer key pool, paged with the latent; recurrent_state / conv_state: a
# linear-attention or SSM layer's per-request state rows (KDA, GDN; SGLang's mamba pool)
POOL_KINDS = frozenset({"kv", "kv_pair", "mla_latent", "index_k", "recurrent_state", "conv_state"})


@dataclass(frozen=True)
class PoolEntry:
    name: str
    kind: str
    layers: tuple
    resolve: Callable[[Any, Any], Sequence[Any]] = field(compare=False)
    buffers: int = 1
    per_buffer: int = 1
    aliased: bool = True
    zero_on_free: bool = False

    def __post_init__(self):
        if self.kind not in POOL_KINDS:
            raise ValueError(f"pool entry {self.name!r}: kind {self.kind!r} is not one of {sorted(POOL_KINDS)}")
        if self.buffers < 1 or self.per_buffer < 1:
            raise ValueError(f"pool entry {self.name!r}: {self.buffers} buffers of {self.per_buffer} layers")
        layers = tuple(int(i) for i in self.layers)
        object.__setattr__(self, "layers", layers)
        if len(layers) != self.buffers * self.per_buffer:
            raise ValueError(f"pool entry {self.name!r}: {len(layers)} layers in {self.buffers} buffers of "
                             f"{self.per_buffer}")
        if len(set(layers)) != len(layers) or any(i < 0 for i in layers):
            raise ValueError(f"pool entry {self.name!r}: layer ids {layers} repeat or are negative")


def _identity(t) -> tuple:
    # the tensor object and its storage: a pool that rebuilt its tensors (or rebound them to new storage) differs
    return (id(t), t.data_ptr(), tuple(t.shape), str(t.dtype))


@dataclass
class BoundPools:
    """The live tensors of a layout: ``tensors[name]`` in each entry's buffer order."""

    layout: "PoolLayout"
    tensors: dict
    identities: tuple
    sources: tuple  # (kv pool, req-to-token pool) objects the tensors were read from

    def check(self, kv_pool, req_to_token_pool) -> None:
        """Refuse a step whose pools are not the ones bound (SGLang replaced the pool object): per step, so it compares
        the pool objects only; ``recheck`` re-reads every tensor."""
        if (id(kv_pool), id(req_to_token_pool)) != self.sources:
            raise RuntimeError(f"megakernel pool layout {self.layout.family}: SGLang's pools were replaced after the "
                               "megakernel bound them; the compiled programs would read the old tensors")

    def recheck(self, kv_pool, req_to_token_pool) -> None:
        """Re-resolve every entry and refuse any tensor that is no longer the one bound."""
        again = self.layout.bind(kv_pool, req_to_token_pool)
        if again.identities != self.identities:
            changed = [name for name in self.tensors
                       if [_identity(t) for t in again.tensors[name]] != [_identity(t) for t in self.tensors[name]]]
            raise RuntimeError(f"megakernel pool layout {self.layout.family}: {changed} were reallocated after the bind")

    def flat(self, aliased_only: bool = False) -> list:
        out = []
        for entry in self.layout.entries:
            if entry.aliased or not aliased_only:
                out.extend(self.tensors[entry.name])
        return out


@dataclass(frozen=True)
class PoolLayout:
    family: str
    entries: tuple
    # the page the kernel's block tables count in (SGLang's --page-size it was built for), or None (no paged entry)
    page_size: Optional[int] = None
    # The pool classes SGLang's KV cache configurator builds for this model (the platform seams'
    # get_megakernel_kv_pool_cls): an attention pool kind ("mha", "mla", "dsa") -> its class, a subclass of the kind's
    # in-tree pool; and the request-state pool class (a MambaPool subclass), None for the request table's own
    pool_classes: tuple = ()
    state_pool_class: Optional[type] = field(default=None, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "entries", tuple(self.entries))
        classes = dict(self.pool_classes)
        object.__setattr__(self, "pool_classes", tuple(sorted(classes.items())))
        if not self.family or not self.entries:
            raise ValueError("a pool layout needs a family name and at least one entry")
        names = [e.name for e in self.entries]
        if len(set(names)) != len(names):
            raise ValueError(f"pool layout {self.family}: entry names repeat: {names}")
        if set(classes) - {"mha", "mla", "dsa"} or not all(isinstance(c, type) for c in classes.values()):
            raise ValueError(f"pool layout {self.family}: pool classes are classes of the kinds mha, mla and dsa; got "
                             f"{sorted(classes)}")
        if self.state_pool_class is not None and not isinstance(self.state_pool_class, type):
            raise TypeError(f"pool layout {self.family}: the state pool class is a class")

    def pool_class(self, kind: str) -> type:
        """The KV pool class of ``kind`` this layout's model builds; ValueError for a kind it does not declare."""
        classes = dict(self.pool_classes)
        if kind not in classes:
            raise ValueError(f"pool layout {self.family} declares no {kind} pool (it declares {sorted(classes)})")
        return classes[kind]

    @property
    def alias_count(self) -> int:
        """How many buffers the programs return as aliases, after their own outputs."""
        return sum(e.buffers for e in self.entries if e.aliased)

    @property
    def zero_on_free(self) -> bool:
        return any(e.zero_on_free for e in self.entries)

    def digest(self) -> tuple:
        """What every rank must declare the same (rank agreement): entries' names, kinds, layers and alias flags."""
        return (self.family, self.page_size,
                tuple((kind, f"{cls.__module__}.{cls.__qualname__}") for kind, cls in self.pool_classes),
                tuple((e.name, e.kind, e.layers, e.buffers, e.per_buffer, e.aliased, e.zero_on_free)
                      for e in self.entries))

    def bind(self, kv_pool, req_to_token_pool) -> BoundPools:
        tensors, identities, device = {}, [], None
        seen = set()
        for entry in self.entries:
            got = list(entry.resolve(kv_pool, req_to_token_pool))
            if len(got) != entry.buffers:
                raise RuntimeError(f"megakernel pool layout {self.family}: {entry.name} resolved {len(got)} tensors, "
                                   f"the layout declares {entry.buffers}")
            for t in got:
                if device is None:
                    device = t.device
                elif t.device != device:
                    raise RuntimeError(f"megakernel pool layout {self.family}: {entry.name} is on {t.device}, the "
                                       f"other pools on {device}")
                ident = _identity(t)
                if ident[:2] in seen:
                    raise RuntimeError(f"megakernel pool layout {self.family}: {entry.name} binds a tensor another "
                                       "entry already binds")
                seen.add(ident[:2])
                identities.append(ident)
            tensors[entry.name] = got
        return BoundPools(self, tensors, tuple(identities), (id(kv_pool), id(req_to_token_pool)))


# resolvers of SGLang's and the plugin's pools


def paired_kv(which: str) -> Callable:
    """The K (``which="k"``) or V per-pair tensors of the plugin's paired KV pool (``NeuronMHATokenToKVPool.layer_pairs``,
    allocated when the model declared ``kv_layout.KVLayout(paired_layers=True)``)."""
    if which not in ("k", "v"):
        raise ValueError(which)

    def resolve(kv_pool, req_to_token_pool):
        pairs = getattr(kv_pool, "layer_pairs", None)
        if pairs is None:
            raise RuntimeError(f"{type(kv_pool).__name__} holds no layer pairs: the model must declare "
                               "kv_layout.KVLayout(paired_layers=True) before SGLang builds its KV pool")
        return pairs[0] if which == "k" else pairs[1]

    return resolve


def per_layer(getter: str, layer_ids, index: Optional[int] = None) -> Callable:
    """``kv_pool.<getter>(layer_id)`` for each global layer id (SGLang's pools index by global layer id), the
    ``index``-th item of a tuple result (``get_kv_buffer``: 0 the K, 1 the V). The MLA / DSA pools' per-layer latent
    and index-K buffers resolve through it (mem_cache/mla_pool.py)."""
    layer_ids = tuple(int(i) for i in layer_ids)

    def resolve(kv_pool, req_to_token_pool):
        out = []
        for layer_id in layer_ids:
            value = getattr(kv_pool, getter)(layer_id)
            out.append(value if index is None else value[index])
        return out

    return resolve


def state_pool(field_name: str) -> Callable:
    """A request-state pool's stacked tensor(s) (``req_to_token_pool.mamba_pool.mamba_cache.<field_name>``: the
    temporal state ``[L, rows, ...]`` or the list of conv states), as SGLang's hybrid request pool holds them."""

    def resolve(kv_pool, req_to_token_pool):
        pool = getattr(req_to_token_pool, "mamba_pool", None)
        if pool is None:
            raise RuntimeError("the request pool holds no state pool (SGLang's HybridReqToTokenPool.mamba_pool)")
        value = getattr(pool.mamba_cache, field_name)
        return list(value) if isinstance(value, (list, tuple)) else [value]

    return resolve

