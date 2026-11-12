# SPDX-License-Identifier: Apache-2.0
"""A NIXL PD prefill refuses, when a decode rank registers, a decode that registered fewer state components than the
prefill sends, naming both sides' components.

A prefill sends each request's state components at its last chunk (``NixlKVManager.maybe_send_extra``), component i
into the entries the decode registered for it (``KVArgsRegisterInfo.dst_state_data_ptrs``, which ``from_zmq`` reads
from the decode's registration, ``build_receiver_registration_message``; [] for a decode that registered none) at the
indices the decode allocated for it. Upstream SGLang's registration (``NixlKVManager._add_remote_peer``) compares the
two sides' KV entries only (``_prepare_payload_xfer`` refuses a decode with fewer KV regions), and ``maybe_send_extra``
takes a component the decode did not register as empty entry and index lists. A hybrid SWA prefill
(``SGLANG_NEURON_HYBRID_SWA=1``, a ``NeuronSWAKVPool``) sends its sliding-window layers as a state component
(``setup_state_kv_args`` registers the KV storage's wire entries, of which a pool with a sliding-window sub-pool has
``StateType.SWA``); a decode launched without hybrid SWA memory
(a ``NeuronMHATokenToKVPool`` of every layer) registers none, and more KV entries than the prefill holds. Upstream
accepts that decode, and every request to it then fails at its last chunk, after its whole prefill ("State index length
mismatch at component 0"), with no word when the roles meet.

Here the prefill refuses such a decode rank in ``_add_remote_peer`` (wrapped), before upstream registers it, naming the
components it sends and the number the decode registered (``state_components_problem``, the one text of this refusal,
which a pipelined decode stage's registration takes too, sglang_patches/pp_decode_stage_transfer). A decode that registered
more state components than the prefill sends is upstream's to serve. The prefill keeps a decode it refuses as a
refused peer whose rooms fail at once with the reason, and its bootstrap thread goes on serving the other decodes
(sglang_patches/nixl_peer_refusal). The decode does not see the prefill's state
components (the prefill's bootstrap registration and SGLang's ``PrefillServerInfo`` carry none), so the refusal is the
prefill's only. Every rank of a prefill builds the same pool kind and every rank of a decode the same, so every prefill
rank refuses every rank of such a decode.

The same registration refuses a decode whose recurrent-state (``StateType.MAMBA``) entries cannot take the prefill's:
a paired entry of a different slot size, a prefill state layer the decode has no entry for, or different attention TPs
with an entry not head-sliceable (disaggregation/state_transfer.state_entries_problem). Upstream WRITEs the first with
the prefill's slot stride into the decode's slot grid, silently, and fails every request at its last chunk for the
other two.

Upstream: a generic refusal at registration, where upstream fails the same pairing at each request's last chunk.

Targets, each checked against upstream SGLang 43881a7: ``NixlKVManager._add_remote_peer`` (wrapped), and the sources the
refusal relies on (their reasons at ``_TARGETS_MD5``)."""

from __future__ import annotations

from typing import Optional, Sequence

_NIXL = "sglang.srt.disaggregation.nixl.conn"
_UTILS = "sglang.srt.disaggregation.utils"
_COMMON = "sglang.srt.disaggregation.common.conn"
TARGET = (_NIXL, "NixlKVManager._add_remote_peer")
# (module, qualified name): md5 of the source
_TARGETS_MD5 = {
    TARGET: "e2ee9f36cd7849d884c49275e4c10bde",
    # sends component i into dst_state_data_ptrs[i] at dst_state_indices[i], each [] past the decode's components, and
    # fails a request whose component has no destination indices at its last chunk
    (_NIXL, "NixlKVManager.maybe_send_extra"): "a0419a7cd7d96bc4a2b3f9e61c948c92",
    # the decode registers each of its state components' entries (from_zmq's msg[8], after the guard frame and the aux
    # item sizes), [] when it has none
    (_NIXL, "NixlKVManager.build_receiver_registration_message"): "1d28c9d9c8345a71db02d6840a7a3acc",
    (_NIXL, "KVArgsRegisterInfo.from_zmq"): "888b10e09310b5af69364bbf6ffbdfab",
    # both roles' state components, in the KV storage's wire order: a pool with a sliding-window sub-pool (a hybrid SWA
    # pool) registers its window component, an MHA pool none
    (_UTILS, "setup_state_kv_args"): "c09eb7e917545ca1d45e53b58ecdee67",
    # hands each registration to _add_remote_peer; a refusal raised there ends it
    (_NIXL, "NixlKVManager._start_bootstrap_thread"): "0ad34e79d2d77da1b2cb45c799e94748",
    # the prefill info the decode reads at first contact names no state component
    (_COMMON, "PrefillServerInfo"): "48c66f9592eff5eb430d870febbcfc55",
}


def _name(state_type) -> str:
    return str(getattr(state_type, "value", state_type))


def state_components_problem(state_types: Sequence, dst_state_data_ptrs: Optional[Sequence]) -> Optional[str]:
    """Why a decode that registered these state components' entries cannot take this prefill's state components, or
    None: it registered fewer than the prefill sends."""
    sent = [_name(st) for st in state_types or []]
    registered = len(dst_state_data_ptrs or [])
    if registered >= len(sent):
        return None
    missing = ", ".join(f"{i} ({name})" for i, name in enumerate(sent) if i >= registered)
    return (f"this prefill sends {len(sent)} state component(s) ({', '.join(sent)}) and the decode registered "
            f"{registered}: component(s) {missing} would have no decode entries")


def add_remote_peer(original_fn, self, decode_kv_args, *args, **kwargs):
    """AROUND NixlKVManager._add_remote_peer: a decode that registered fewer state components than this prefill sends
    is refused before it is registered."""
    problem = state_components_problem(getattr(self.kv_args, "state_types", None),
                                       decode_kv_args.dst_state_data_ptrs)
    if problem is not None:
        # upstream registers it, and every request to it fails at its last chunk: refused here, naming both sides
        raise RuntimeError(f"NIXL PD transfer to decode peer {decode_kv_args.agent_name}: {problem}, so every request "
                           "would fail at its last chunk; launch both roles with the same memory layout (hybrid SWA "
                           "memory, SGLANG_NEURON_HYBRID_SWA=1, on both roles at one attention TP, or on neither)")
    from sglang_neuron.disaggregation import state_transfer

    problem = state_transfer.state_entries_problem(
        self.kv_args, decode_kv_args, prefill_tp=int(getattr(self, "state_tp_size", None) or self.attn_tp_size),
        pp_size=int(getattr(self, "pp_size", 1) or 1))
    if problem is not None:
        # upstream registers it and then WRITEs each request's state with the prefill's slot stride into the decode's
        # slots, or fails the request at its last chunk: refused here, naming the entries
        raise RuntimeError(f"NIXL PD transfer to decode peer {decode_kv_args.agent_name}: {problem}; launch both roles "
                           "with the same recurrent-state dtypes and layout (and one attention TP for a kernel "
                           "layout upstream cannot slice)")
    return original_fn(self, decode_kv_args, *args, **kwargs)


def register(registry, around) -> None:
    import functools
    import importlib

    from sglang_neuron.sglang_patches import source_guard

    for (module, qualname), md5 in _TARGETS_MD5.items():
        obj = functools.reduce(getattr, qualname.split("."), importlib.import_module(module))
        source_guard.check(obj, md5, f"{module}.{qualname}", __name__)
    registry.register(".".join(TARGET), add_remote_peer, around)
