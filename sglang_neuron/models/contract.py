# SPDX-License-Identifier: Apache-2.0
"""What the runtime takes from a model: its attention layers, each full or sliding, which layers carry recurrent state
(``state_layer_ids``), and a digest every rank of the engine must agree on.

The layer kinds come from the config's ``layer_types``; any kind but full and sliding attention and the recurrent
kinds (``RECURRENT``) is refused, since the runtime's metadata and kernels serve those only, and a
``sliding_window`` without ``layer_types`` is refused rather than defaulted to full attention everywhere, since which
layers slide would be a guess (a Mistral-class config slides on every layer). A graph that names its own layers
declares each name and window instead (``declared_layer_kinds``). ``assert_fleet_consistent`` compares a digest of the
layer kinds and windows, head counts, hidden and vocab sizes, phase and parallel layout (no rank or device ids) across
the engine with one gloo all-gather as the runtime is built; other shape differences surface as a NEFF mismatch.

A model whose config declares recurrent layers must declare a state spec covering exactly them
(mem_cache/state_spec), checked against the contract at graph-runner build (``check_state_spec``, via
mem_cache/state_pool.check_model_state): a state spec that survives undetected on a dense config would size and wire
a pool that never runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

FULL = "full_attention"
SLIDING = "sliding_attention"
# a config's layer_types name a recurrent-state kind as upstream names them; the contract's state family names the
# state-spec family the model itself declares (mem_cache/state_spec; gdn: gated delta net, kda: Kimi delta attention)
RECURRENT = {"linear_attention": "gdn", "kda": "kda"}
# the stateful kinds: RECURRENT, whose layers are a state family's, and upstream's other names for Mamba-style state,
# which no state spec family yet serves
_STATEFUL = {"mamba", "mamba2", "ssm", "recurrent"} | frozenset(RECURRENT)


@dataclass(frozen=True)
class LayerKind:
    name: str
    sliding_window: int | None = None
    # a recurrent layer's state family (RECURRENT's values); None for an attention layer
    state: str | None = None

    def key(self) -> tuple:
        """The layer's identity for the fleet digest: a recurrent layer at a full-attention layer's place differs."""
        return (self.name, self.sliding_window, self.state)


@dataclass(frozen=True)
class ModelContract:
    architecture: str
    layers: tuple[LayerKind, ...]
    num_attention_heads: int
    num_kv_heads: int
    hidden_size: int
    vocab_size: int
    phase: str = "all"
    parallel_digest: tuple = field(default_factory=tuple)

    def sliding_windows(self) -> list:
        return [layer.sliding_window for layer in self.layers]

    def layer_names(self) -> list:
        return [layer.name for layer in self.layers]

    def state_layer_ids(self) -> tuple:
        return tuple(i for i, layer in enumerate(self.layers) if layer.state is not None)

    def check_state_spec(self, spec) -> None:
        """Raise unless the model's StateSpec covers exactly the contract's recurrent layers, in one family: a
        recurrent layer without state, or state for an attention layer, would read or write another layer's slots."""
        ids = self.state_layer_ids()
        families = {self.layers[i].state for i in ids}
        if spec is None:
            if ids:
                raise ValueError(f"{self.architecture}: layers {ids} are recurrent ({sorted(families)}) and the model "
                                 "declares no state spec (mem_cache/state_spec.StateSpec)")
            return
        if list(spec.layer_ids) != list(ids) or families != {spec.family}:
            raise ValueError(f"{self.architecture}: the model's {spec.family} state spec covers layers "
                             f"{list(spec.layer_ids)}; the contract's recurrent layers are {ids} "
                             f"({sorted(families)})")

    def fleet_hash(self) -> str:
        payload = {
            "architecture": self.architecture,
            "layers": [layer.key() for layer in self.layers],
            "heads": [self.num_attention_heads, self.num_kv_heads],
            "hidden": self.hidden_size, "vocab": self.vocab_size,
            "phase": self.phase,
            "parallel": list(self.parallel_digest),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def derive_layer_kinds(hf_config, architecture: str) -> tuple:
    n = int(hf_config.num_hidden_layers)
    window = getattr(hf_config, "sliding_window", None)
    layer_types = getattr(hf_config, "layer_types", None)
    if not layer_types:
        if window:
            # a Mistral-class config slides on EVERY layer: defaulting each to full attention would serve wrong
            # masks and KV budget, so refuse rather than guess
            raise ValueError(f"{architecture}: the config sets sliding_window={window} but has no layer_types; "
                             "deriving which layers slide would be a guess (a Mistral-class config slides on every "
                             "layer), so declare the model's layer kinds explicitly (layer_names / sliding_windows "
                             "on the model graph) or fix the config's layer_types")
        layer_types = [FULL] * n
    layer_types = list(layer_types)
    # a num_hidden_layers override builds the first n layers of the checkpoint's list
    if len(layer_types) < n:
        raise ValueError(f"{architecture}: layer_types has {len(layer_types)} entries for {n} hidden layers")
    kinds = []
    for i, kind in enumerate(layer_types[:n]):
        if kind in RECURRENT:
            kinds.append(LayerKind(f"layers.{i}.linear_attn", state=RECURRENT[kind]))
            continue
        if kind in _STATEFUL:
            raise ValueError(f"{architecture}: layer {i} is {kind!r}; no model on this platform declares state "
                             "components for this SSM family, so it has no state pool (mem_cache/state_spec "
                             "families: gdn, kda)")
        if kind not in (FULL, SLIDING):
            raise ValueError(f"{architecture}: layer {i} is {kind!r}; the runtime serves full and sliding "
                             "attention layers only")
        if kind == SLIDING and not (window and int(window) > 0):
            raise ValueError(f"{architecture}: layer {i} slides, but the config has no positive sliding_window")
        kinds.append(LayerKind(f"layers.{i}.self_attn", int(window) if kind == SLIDING else None))
    return tuple(kinds)


def declared_layer_kinds(hf_config, architecture: str, layer_names, sliding_windows, state_families=None) -> tuple:
    """The layers a graph declares: a name (the key of its attention metadata) and a window (None or 0 for full
    attention) per hidden layer, in global layer order, which the pool and pipeline stages index by. A hybrid model
    also declares each layer's recurrent-state family (None for an attention layer)."""
    names, windows = list(layer_names), list(sliding_windows)
    n = int(hf_config.num_hidden_layers)
    if len(names) != n or len(windows) != n:
        raise ValueError(f"{architecture}: the graph declares {len(names)} layer names and {len(windows)} windows for "
                         f"{n} hidden layers")
    if len(set(names)) != n:
        raise ValueError(f"{architecture}: the graph declares a layer name twice, and each name keys one layer's "
                         "attention metadata")
    states = list(state_families) if state_families is not None else [None] * n
    if len(states) != n:
        raise ValueError(f"{architecture}: the graph declares {len(states)} state families for {n} hidden layers")
    unknown = sorted({f for f in states if f is not None and f not in RECURRENT.values()})
    if unknown:
        raise ValueError(f"{architecture}: state families {unknown} are not ones the platform hosts")
    kinds = []
    for i, (name, window, state) in enumerate(zip(names, windows, states)):
        if window and int(window) < 0:
            raise ValueError(f"{architecture}: layer {i} declares the window {window}")
        kinds.append(LayerKind(str(name), int(window) if window else None, state=state))
    return tuple(kinds)


def derive(hf_config, *, architecture: str, phase: str = "all", parallel_digest: tuple = (),
           layers: tuple | None = None) -> ModelContract:
    """The contract of ``hf_config``, over its ``layer_types`` or the graph's declared ``layers``."""
    heads = int(hf_config.num_attention_heads)
    return ModelContract(
        architecture=architecture,
        layers=layers if layers is not None else derive_layer_kinds(hf_config, architecture),
        num_attention_heads=heads,
        num_kv_heads=int(getattr(hf_config, "num_key_value_heads", None) or heads),
        hidden_size=int(hf_config.hidden_size),
        vocab_size=int(hf_config.vocab_size),
        phase=phase,
        parallel_digest=tuple(parallel_digest),
    )


def assert_fleet_consistent(contract: ModelContract, group) -> str:
    """The contract's digest, raising unless every rank of ``group`` holds the same."""
    import torch
    import torch.distributed as dist

    digest = contract.fleet_hash()
    if group.world_size <= 1:
        return digest
    local = torch.tensor(bytearray.fromhex(digest), dtype=torch.uint8)
    gathered = [torch.zeros_like(local) for _ in range(group.world_size)]
    dist.all_gather(gathered, local, group=group.cpu_group)
    hashes = {bytes(t.tolist()).hex() for t in gathered}
    if len(hashes) != 1:
        raise RuntimeError(f"the model contract differs across the engine's ranks ({sorted(hashes)}): some rank "
                           "resolved another model shape or launch setting; check the ranks' environments")
    logger.info("[neuron] model contract %s (%s, %d layers) agreed on %d ranks", digest[:12], contract.architecture,
                len(contract.layers), group.world_size)
    return digest
