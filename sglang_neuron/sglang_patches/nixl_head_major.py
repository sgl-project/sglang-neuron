# SPDX-License-Identifier: Apache-2.0
"""NIXL KV transfer between a prefill and a decode of different attention-TP sizes, on head-major pages.

A Neuron KV pool stores each page head-major, ``[kv_heads, page_size, head_dim]``, the layout the attention kernels
read. When the two sides' attention-TP sizes differ, upstream's NIXL transfer slices each page per token, as if it
were ``[page_size, kv_heads, head_dim]``, and so addresses the wrong bytes. On a head-major page a head's share of the
page is one contiguous run, so here the slice transfer takes one slice per page and head group instead of one per
token. The prep (``_init_hetero_tp_prep_handle``) and the send (``send_kvcache_slice``) are upstream's source with the
slice unit set to one per page. The prep also refuses a decode whose per-head page bytes differ from the prefill's,
since a head's page is then not one run on both sides.

Upstream builds one source descriptor list (dlist) from the first decode peer of another TP size, and sends to every
later peer through it. That list's runs follow the first decode's head grouping, so a prefill serving decodes of two
attention-TP sizes would address the second decode's heads through the first's runs. Here the prep keeps one source
dlist per head grouping (``neuron_slice_srcs``), and each peer's entry names its grouping for the send to look up.

Each side also checks the other's layout, since a whole-page copy between the two layouts corrupts as surely as a
per-token slice:

- The decode announces its layout with one trailing frame on its registration, which upstream's parser ignores. The
  prefill refuses a decode peer without it.
- Both roles publish the Neuron layout through upstream's own kv_layout field (``get_pd_kv_layout``, which names a
  layout only for DeepSeek-V4's pool). The decode fails a request whose prefill publishes another layout, or none,
  before it registers with that prefill. A prefill without sglang-neuron would write token-major pages into the
  decode's head-major ones. It would also never ack the abort of a room it could not prove settled, and the decode's
  quarantine would then hold that room's pages until the prefill restarts.

Upstream: needs a KV layout field in the NIXL registration so the slice transfer can index either layout."""

from __future__ import annotations

from typing import Dict, Tuple

HEAD_MAJOR_FRAME = b"sglang-neuron:kv-head-major"
NEURON_KV_LAYOUT = "sglang-neuron:kv-head-major"
_CONN = "sglang.srt.disaggregation.nixl.conn"
_COMMON = "sglang.srt.disaggregation.common.conn"
_UTILS = "sglang.srt.disaggregation.utils"
_STAGING = "sglang.srt.disaggregation.common.staging_buffer"
_TARGETS_MD5: Dict[Tuple[str, str], str] = {
    (_CONN, "NixlKVManager._init_hetero_tp_prep_handle"): "85727ce66c9512bc9be63136dc52db2f",
    (_CONN, "NixlKVManager.send_kvcache_slice"): "caa8942af9f2fc181c7daf2618ad6cca",
    (_CONN, "NixlKVManager.build_receiver_registration_message"): "1d28c9d9c8345a71db02d6840a7a3acc",
    (_CONN, "NixlKVManager._add_remote_peer"): "e2ee9f36cd7849d884c49275e4c10bde",
    (_CONN, "KVArgsRegisterInfo.from_zmq"): "888b10e09310b5af69364bbf6ffbdfab",
    (_CONN, "expand_page_indices_for_slice"): "c7468a6078c8bb0085872a44a0e74ee8",
    # the head slice arithmetic the transfer keeps
    (_STAGING, "compute_head_slice_params"): "9963562e07776bf1ebd244343534ec62",
    # the layout both roles publish; a request to a prefill of another layout fails before the receiver's init
    (_UTILS, "get_pd_kv_layout"): "a4a82e44d0d94377a9be937568851228",
    (_COMMON, "CommonKVReceiver.init"): "fe1b27f354afbc1dfb46cf7dfa0be764",
}
UPSTREAM_PAGE_SIZE = "    page_size = self.kv_args.page_size\n"
NEURON_PAGE_SIZE = "    page_size = 1  # a head-major page is one slice unit per head\n"
# the per-head bytes check goes right after the prep reads both sides' page bytes
UPSTREAM_SRC_ITEM_LEN = "    src_kv_item_len = self.kv_args.kv_item_lens[0]\n"
PER_HEAD_BYTES_CHECK = (
    "    src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)\n"
    "    if src_kv_item_len // src_heads_per_rank != bytes_per_head_slice:\n"
    "        raise ValueError(\n"
    '            "Head-major KV slice transfer needs equal per-head page bytes on both sides (prefill "\n'
    '            f"{src_kv_item_len // src_heads_per_rank}, decode {bytes_per_head_slice})")\n')
# one source dlist per head grouping, and each peer's destination entry names its grouping
UPSTREAM_SHARED_SRC = "    if self.prep_handle_slice_src is None:\n"
NEURON_SRC_PER_GROUPING = (
    "    slice_srcs = self.__dict__.setdefault(\"neuron_slice_srcs\", {})  # head grouping -> its source dlist\n"
    "    if num_groups not in slice_srcs:\n")
UPSTREAM_SRC_STORE = ("        self.prep_handle_slice_src = (\n            src_handle,\n            num_groups,\n"
                      "            num_ptr_pairs,\n            num_slots,\n        )\n")
NEURON_SRC_STORE = "        slice_srcs[num_groups] = (src_handle, num_groups, num_ptr_pairs, num_slots)\n"
UPSTREAM_DST_STORE = ("    self.prep_handles_slice_dst[peer_name] = (\n        dst_handle,\n        num_slots_dst,\n"
                      "        head_group_idx,\n    )\n")
NEURON_DST_STORE = ("    self.prep_handles_slice_dst[peer_name] = (\n        dst_handle,\n        num_slots_dst,\n"
                    "        head_group_idx,\n        num_groups,\n    )\n")
UPSTREAM_SEND_LOOKUP = (
    "    assert self.prep_handle_slice_src is not None\n    assert peer_name in self.prep_handles_slice_dst\n"
    "    src_handle, num_groups, num_ptr_pairs, num_slots_src = (\n        self.prep_handle_slice_src\n    )\n"
    "    dst_handle, num_slots_dst, head_group_idx = self.prep_handles_slice_dst[\n        peer_name\n    ]\n")
NEURON_SEND_LOOKUP = (
    "    assert peer_name in self.prep_handles_slice_dst\n"
    "    dst_handle, num_slots_dst, head_group_idx, grouping = self.prep_handles_slice_dst[peer_name]\n"
    "    src_handle, num_groups, num_ptr_pairs, num_slots_src = self.neuron_slice_srcs[grouping]\n")
PREP_HANDLE_SUBSTITUTIONS = (
    (UPSTREAM_PAGE_SIZE, NEURON_PAGE_SIZE),
    (UPSTREAM_SRC_ITEM_LEN, UPSTREAM_SRC_ITEM_LEN + PER_HEAD_BYTES_CHECK),
    (UPSTREAM_SHARED_SRC, NEURON_SRC_PER_GROUPING),
    (UPSTREAM_SRC_STORE, NEURON_SRC_STORE),
    (UPSTREAM_DST_STORE, NEURON_DST_STORE),
)
SEND_SUBSTITUTIONS = ((UPSTREAM_PAGE_SIZE, NEURON_PAGE_SIZE), (UPSTREAM_SEND_LOOKUP, NEURON_SEND_LOOKUP))


def build_receiver_registration_message(original_fn, self, *args, **kwargs):
    return list(original_fn(self, *args, **kwargs)) + [HEAD_MAJOR_FRAME]


def from_zmq(original_fn, cls, msg, *args, **kwargs):
    info = original_fn(cls, msg, *args, **kwargs)
    info.dst_kv_head_major = any(frame == HEAD_MAJOR_FRAME for frame in msg)
    return info


def get_pd_kv_layout(original_fn, token_to_kv_pool, *args, **kwargs):
    """Name the Neuron layout for a Neuron pool, for which upstream names none."""
    from sglang_neuron.mem_cache.memory_pool import NeuronMHATokenToKVPool, NeuronMLATokenToKVPool

    layout = original_fn(token_to_kv_pool, *args, **kwargs)
    if layout is None and isinstance(token_to_kv_pool, (NeuronMHATokenToKVPool, NeuronMLATokenToKVPool)):
        return NEURON_KV_LAYOUT
    return layout


def receiver_init(original_fn, self, *args, **kwargs):
    """Fail a request whose prefill publishes another KV layout before it registers with that prefill, the way
    upstream's init already fails one of another allocation policy."""
    from sglang.srt.disaggregation.base import KVPoll

    info = self.kv_mgr.prefill_info_table.get(self.bootstrap_addr)
    if self.conclude_state != KVPoll.Failed and info is not None and info.kv_layout != self.kv_mgr.kv_layout:
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            f"prefill {self.bootstrap_addr} publishes KV layout {info.kv_layout!r}, not this decode's "
            f"{self.kv_mgr.kv_layout!r}: a prefill without sglang-neuron writes token-major pages into this decode's "
            "head-major pool (pair a Neuron prefill)")
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        self.conclude_state = KVPoll.Failed
        return None
    return original_fn(self, *args, **kwargs)


def add_remote_peer(original_fn, self, decode_kv_args, *args, **kwargs):
    if not getattr(decode_kv_args, "dst_kv_head_major", False):
        raise ValueError(
            f"PD KV transfer to decode peer {decode_kv_args.agent_name}: its registration lacks sglang-neuron's KV "
            "layout frame, and this Neuron prefill pairs only with a decode running sglang-neuron, whose pages are laid "
            "out as its own (pair a Neuron decode)")
    return original_fn(self, decode_kv_args, *args, **kwargs)


def init_hetero_tp_prep_handle():
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(source_guard.resolve(_CONN, "NixlKVManager._init_hetero_tp_prep_handle"),
                                           PREP_HANDLE_SUBSTITUTIONS)


def send_kvcache_slice():
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(source_guard.resolve(_CONN, "NixlKVManager.send_kvcache_slice"),
                                           SEND_SUBSTITUTIONS)


def register(registry, around, replace) -> None:
    from sglang_neuron.mem_cache.memory_pool import NeuronMHATokenToKVPool
    from sglang_neuron.sglang_patches import source_guard

    if not NeuronMHATokenToKVPool.kv_head_major:
        raise RuntimeError("sglang_patches/nixl_head_major serves head-major Neuron KV pools only")
    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_CONN}.NixlKVManager._init_hetero_tp_prep_handle", init_hetero_tp_prep_handle(), replace)
    registry.register(f"{_CONN}.NixlKVManager.send_kvcache_slice", send_kvcache_slice(), replace)
    registry.register(f"{_CONN}.NixlKVManager.build_receiver_registration_message",
                      build_receiver_registration_message, around)
    registry.register(f"{_CONN}.NixlKVManager._add_remote_peer", add_remote_peer, around)
    registry.register(f"{_CONN}.KVArgsRegisterInfo.from_zmq", from_zmq, around)
    registry.register(f"{_UTILS}.get_pd_kv_layout", get_pd_kv_layout, around)
    registry.register(f"{_COMMON}.CommonKVReceiver.init", receiver_init, around)
