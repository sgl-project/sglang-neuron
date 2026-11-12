# SPDX-License-Identifier: Apache-2.0
"""PD KV hand-off for a decode whose KV lives in its kernel's own caches, not in SGLang's pool.

SGLang sends a prefill rank's KV to the allocator pages one decode rank publishes. A decode kernel with its own caches
(one slot per request, dealt over its context-parallel ranks) needs its own destination, rank mapping, registration
frame, peer check and send. The model provides them (``catalog.ModelSpec.kv_handoff``: a module with ``DECODE_HOOKS``
and ``PREFILL_HOOKS``, AROUND hooks by target); this module registers them, and refuses a hand-off that leaves a
target without a hook. Both sides also carry the pool's published layout to the KV manager before it registers with
its peers. A model without a hand-off keeps upstream's path."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional

_COMMON = "sglang.srt.disaggregation.common.conn"
_NIXL = "sglang.srt.disaggregation.nixl.conn"
_DECODE = "sglang.srt.disaggregation.decode"
_PREFILL = "sglang.srt.disaggregation.prefill"
_UTILS = "sglang.srt.disaggregation.utils"
# (module, qualified name) of each upstream source, pinned in upstream_pins.json: the wrapped targets of each side,
# then the sources they rely on
DECODE_TARGETS = (
    (_UTILS, "get_pd_kv_layout"),
    (_NIXL, "NixlKVManager.build_receiver_registration_message"),
    (_COMMON, "CommonKVManager._resolve_rank_mapping"),
    (_DECODE, "DecodePreallocQueue._transfer_page_indices"),
)
PREFILL_TARGETS = (
    (_UTILS, "get_pd_kv_layout"),
    (_NIXL, "KVArgsRegisterInfo.from_zmq"),
    (_NIXL, "NixlKVManager._add_remote_peer"),
    (_NIXL, "NixlKVManager.send_kvcache"),
)
_RELIED_ON = (
    # the managers take the pool's layout before they register with peers, and map ranks after upstream's checks
    (_DECODE, "DecodePreallocQueue._init_kv_manager"),
    (_PREFILL, "PrefillBootstrapQueue._init_kv_manager"),
    (_COMMON, "CommonKVManager.__init__"),
    (_NIXL, "NixlKVManager.__init__"),
    (_COMMON, "CommonKVManager.try_ensure_parallel_info"),
    # a receiver registers with every target and publishes each request's destination to all of them
    (_COMMON, "CommonKVReceiver._setup_bootstrap_infos"),
    (_COMMON, "CommonKVReceiver._register_kv_args"),
    (_COMMON, "CommonKVReceiver.send_metadata"),
    (_NIXL, "NixlKVManager.build_receiver_metadata_message"),
    (_DECODE, "DecodePreallocQueue._publish_destination"),
    (_DECODE, "DecodePreallocQueue._send_kv_metadata"),
    (_UTILS, "pre_alloc_fill_len"),
    # the send: per-chunk sends, entry pairing by layer id, the prepared descriptor lists
    (_NIXL, "NixlKVManager.transfer_worker"),
    (_NIXL, "NixlKVManager._send_kvcache_generic"),
    (_NIXL, "NixlKVManager._prepare_payload_xfer"),
    (_NIXL, "NixlKVManager._build_transfer_dst_indices"),
    (_UTILS, "build_transfer_entry_pairs"),
    (_NIXL, "NixlKVManager._prep_equal_tp_dlist"),
    (_NIXL, "repeat_indices_over_layers"),
    (_NIXL, "NixlKVManager._post_xfer"),
    (_NIXL, "NixlKVManager._reload_invalidated_peers"),
    # completion: a decode rank concludes once every prefill rank has reported
    (_COMMON, "CommonKVManager.apply_prefill_status"),
    (_COMMON, "CommonKVManager._prefill_unique_rank"),
)


def model_handoff() -> Optional[ModuleType]:
    """The serving model's hand-off module (``catalog.ModelSpec.kv_handoff``), or None for a model without one."""
    from sglang_neuron.models import catalog

    spec = catalog.served_spec()
    name = getattr(spec, "kv_handoff", "") if spec is not None else ""
    return importlib.import_module(name) if name else None


def _register(registry, around, targets, hooks, owner: str) -> None:
    from sglang_neuron.sglang_patches import source_guard

    missing = sorted(set(q for _, q in targets) - set(hooks))
    if missing:
        raise RuntimeError(f"the KV hand-off {owner} defines no hook for {missing}")
    source_guard.check_pinned((*targets, *_RELIED_ON), __name__)
    for module, qualname in targets:
        registry.register(f"{module}.{qualname}", hooks[qualname], around)


def register_decode(registry, around, handoff: ModuleType) -> None:
    """The decode side of ``handoff``."""
    _register(registry, around, DECODE_TARGETS, handoff.DECODE_HOOKS, handoff.__name__)


def register_prefill(registry, around, handoff: ModuleType) -> None:
    """The prefill side of ``handoff``, after every other hook on its peer registration and send."""
    _register(registry, around, PREFILL_TARGETS, handoff.PREFILL_HOOKS, handoff.__name__)


__all__ = ["DECODE_TARGETS", "PREFILL_TARGETS", "model_handoff", "register_decode", "register_prefill"]
