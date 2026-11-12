# SPDX-License-Identifier: Apache-2.0
"""The PD hand-off of the Qwen3.5 kv_cp decode's KV layout: a whole-page prefill scatters each page block to the
decode rank that attends over it.

A kv_cp decode (sglang_neuron/models/qwen35, kv_cp = its attention TP size) holds, of every SGLang page, one block
of ``page / cp`` tokens (its CP rank's block) of EVERY KV head, while a prefill at kv_cp 1 holds whole pages with the
heads sharded across the attention TP group. pd_kv_cp_layout's equal-layout rule refuses the pairing; ``scatter_pair``
admits exactly this one, and the transfer is here:

* the decode announces its layout with the head count (pd_kv_cp_layout's fourth frame field, ``heads``);
* the prefill, at peer registration, checks the geometry (``geometry_problem``: equal attention TP = the CP degree,
  the heads split it, and every paired KV entry holds ``cp`` blocks on the prefill and ``heads`` of them on the
  decode), then preps, per peer, two strided dlists (``NixlKVManager._prepare_payload_xfer``, wrapped): the source
  dlist runs over this rank's KV stacks offset by the DECODE peer's block of each page (``block`` bytes of
  ``item_len`` per slot, ``item_len`` apart), the destination dlist over the decode's stacks offset by THIS rank's
  head columns (its ``heads`` KV heads, ``heads * block`` bytes of a page each, ``item_len`` apart). Each rank thus
  sends one block of each page — 1/cp of its KV — and the decode's ``heads`` sources (one per KV head shard) fill its
  page;
* the prefill sends over the peer's own source dlist (``NixlKVManager.send_kvcache``, wrapped), not the shared
  equal-TP dlist of whole pages;
* the decode maps each rank to its ``heads`` sources (``CommonKVManager._resolve_rank_mapping``, wrapped): rank e
  pulls from prefill attention-TP ranks ``e % group + h * group`` (group = tp / heads, h the KV head shard), one
  response each — upstream's multi-source machinery already marks every target of a non-MLA pool real.

The source rank's block offset is the PEER's CP rank and the destination offset is the SOURCE rank's head shard, so
every page id means the same page on both roles and the allocator, radix and block tables stay identical everywhere.

Upstream: generic: a decode that holds all KV heads of a block of each page, served by a prefill that holds whole
pages — one asymmetric CP pairing of layouts the equal-layout rule refuses.

Targets are md5-pinned against upstream SGLang 43881a7."""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from sglang_neuron import env_registry

# the scatter knobs (env_registry), read once at import: the descriptor bound of one posted batch, the drain of the
# previous batch before the next posts, and the prep/send dump
_SCATTER_MAX_DESCS = (env_registry.non_negative_int("SGLANG_NEURON_QWEN35_SCATTER_MAX_DESCS")
                      if "SGLANG_NEURON_QWEN35_SCATTER_MAX_DESCS" in os.environ else 32768)
_SCATTER_DRAIN = (env_registry.zero_or_one("SGLANG_NEURON_QWEN35_SCATTER_DRAIN")
                  if "SGLANG_NEURON_QWEN35_SCATTER_DRAIN" in os.environ else True)
_SCATTER_DUMP = env_registry.zero_or_one("SGLANG_NEURON_QWEN35_SCATTER_DUMP")

_COMMON = "sglang.srt.disaggregation.common.conn"
_NIXL = "sglang.srt.disaggregation.nixl.conn"

# (module, qualified name): md5 of upstream's source; the wrapped functions and the sources the transfer calls
_TARGETS_MD5 = {
    (_NIXL, "NixlKVManager._prepare_payload_xfer"): "0d52b004d8811f5b3306c8f184f306a7",
    (_NIXL, "NixlKVManager.send_kvcache"): "dc8b5f9e6114739fce8c623ba6bd6428",
    (_COMMON, "CommonKVManager._resolve_rank_mapping"): "70d83eef6dcb74b7cce329808eecba2f",
    # one strided run per KV entry: (ptr, xfer_len, dev, item_len stride, num_slots), the runs below
    (_NIXL, "NixlKVManager._init_equal_tp_prep_handle"): "55dbf1feaa081681ebd6afffdb068916",
    # its dlist builder, called directly for the per-peer source dlist (local agent, scatter key)
    (_NIXL, "NixlKVManager._prep_equal_tp_dlist"): "3f25702681e87ec5e77da6f40a7c7956",
    # descriptor index = entry * num_slots + slot, entry-major, the send below
    (_NIXL, "repeat_indices_over_layers"): "4c38a3c0e2c2a0320217f43c56806f88",
    # the last-chunk state send every KV source reaches; skips a component whose prefill indices are empty
    (_NIXL, "NixlKVManager.maybe_send_extra"): "a0419a7cd7d96bc4a2b3f9e61c948c92",
}

# the per-peer source dlist's key in prep_handles (the shared equal-TP source dlist is ""; a peer's name its own)
_SRC_KEY_SUFFIX = ":kv-cp-scatter-src"


def _src_key(peer_name: str) -> str:
    return peer_name + _SRC_KEY_SUFFIX


def _scatter_layout(info):
    """The peer's kv_cp layout when this pairing is a scatter one (set by prepare_payload_xfer), else None."""
    return getattr(info, "kv_cp_scatter_layout", None) if info is not None else None


_LOCK_INIT = threading.Lock()


def _drain_lock(manager):
    """The manager's lock on the drain's read-settle-post-update step (its transfer workers send concurrently), made
    on first use."""
    lock = getattr(manager, "_scatter_drain_lock", None)
    if lock is None:
        with _LOCK_INIT:
            lock = getattr(manager, "_scatter_drain_lock", None)
            if lock is None:
                lock = manager._scatter_drain_lock = threading.Lock()
    return lock


def geometry_problem(manager, peer, layout) -> Optional[str]:
    """Why this prefill rank cannot scatter its pages to a kv_cp decode of ``layout``, or None. Runs at peer
    registration (pd_kv_cp_layout.add_remote_peer), before any transfer is prepped."""
    from sglang_neuron.sglang_patches import pd_kv_cp_layout

    cp, heads = layout.cp_size, layout.heads
    if manager.is_mla_backend or getattr(manager, "is_hybrid_mla_backend", False):
        return "the kv_cp scatter serves an MHA KV pool on both roles; an MLA pool's pages are not head-sharded"
    if int(manager.attn_tp_size) != cp:
        return (f"the decode's CP group is its {cp} attention TP ranks holding one block of each page each; this "
                f"prefill's attention TP is {manager.attn_tp_size}")
    if cp % heads:
        return f"the decode's {heads} KV heads do not shard its {cp} attention TP ranks evenly"
    src_lens = [int(l) for l in manager.kv_args.kv_item_lens or []]
    dst_lens = [int(l) for l in peer.dst_kv_item_lens or []]
    if len(dst_lens) != len(src_lens):
        return ("the scatter serves one KV entry per layer on both roles "
                f"({len(src_lens)} prefill, {len(dst_lens)} decode; no decode-only speculative decoding)")
    pairs = pd_kv_cp_layout._pairs(getattr(manager.kv_args, "kv_layer_ids", None), peer.dst_kv_layer_ids,
                                   len(src_lens), len(dst_lens), int(getattr(manager, "pp_size", 1) or 1))
    if pairs is None:
        return "the roles' KV layer ids do not pair; upstream refuses this pairing itself"
    problems = []
    for i, j in pairs:
        src, dst = src_lens[i], dst_lens[j]
        if src % cp:
            problems.append(f"KV entry {i}: the prefill page {src} B is not {cp} whole blocks")
        elif dst != heads * (src // cp):
            problems.append(f"KV entry {i}: {heads} heads x the {src // cp} B block is {heads * (src // cp)} B, "
                            f"not the decode page {dst} B")
    if problems:
        return "the scatter geometry does not hold: " + "; ".join(problems)
    return None


# The prefill
def prepare_payload_xfer(original_fn, self, peer_info, *args, **kwargs):
    """AROUND NixlKVManager._prepare_payload_xfer: a scatter peer gets two per-peer strided dlists (module docstring);
    any other peer upstream's prep."""
    from sglang_neuron.sglang_patches import pd_kv_cp_layout

    layout = getattr(peer_info, "dst_kv_cp_layout", None)
    own = pd_kv_cp_layout.own_layout(int(self.kv_args.page_size))
    if not pd_kv_cp_layout.scatter_pair(own, layout):
        return original_fn(self, peer_info, *args, **kwargs)
    from sglang.srt.disaggregation.nixl.conn import _homogeneous_kv_mem_kind

    cp, heads = layout.cp_size, layout.heads
    group = cp // heads
    head = int(self.attn_tp_rank) // group  # this rank's KV head shard, the head columns it fills on the decode
    page_tokens = int(self.kv_args.page_size)
    items = [int(l) for l in self.kv_args.kv_item_lens]
    if len(items) % 2:
        raise RuntimeError("the kv_cp scatter needs the pool's K entries then its V entries, one pair per layer")
    n_layers = len(items) // 2
    block_tokens, rem = divmod(page_tokens, cp)
    if rem:
        raise RuntimeError(f"the page ({page_tokens} tokens) does not shard into {cp} whole token blocks")
    # A V page is [page_tokens, head_dim] (token-major), so a cp_rank's TOKEN block of it is one contiguous run.
    # A K page is either the dim-major transpose [head_dim, page_tokens] — the token block a 2-D strided slice:
    # one contiguous 2*block_tokens-byte run per dim row, dim rows page_tokens elements apart, and flat
    # item_len//cp blocks would send a DIM slice of every token instead of a token slice of every dim — or
    # token-major like V (the layout frame's k_tm; models/qwen35/config.KV_TM_ENV), the token block then one
    # contiguous run per layer on BOTH roles and the K entries collapse to one run per page like V's.
    if bool(own.k_tm) != bool(layout.k_tm):
        raise RuntimeError("the kv_cp scatter's K layout differs between the roles; cp_layout_problem should have "
                           "refused this peer at registration")
    esz = 2  # the pool refuses non-bf16 KV
    dim_row_bytes = page_tokens * esz
    block_bytes = block_tokens * esz
    blocks = [int(l) // cp for l in items]  # per V entry: one block's bytes (kept for the dump line)
    src_base = [int(p) for p in self.kv_args.kv_data_ptrs]
    dst_base = [int(p) for p in peer_info.dst_kv_ptrs]
    dst_item = [int(l) for l in peer_info.dst_kv_item_lens]
    dst_num_slots = peer_info.dst_num_slots if peer_info.dst_num_slots is not None else self._num_slots_src
    src_ptrs: list = []
    src_strides: list = []
    src_data: list = []
    src_xfer: list = []
    dst_ptrs: list = []
    dst_strides: list = []
    dst_data: list = []
    dst_xfer: list = []
    # the same layer-id pairing geometry_problem validated, so the transfer and the check cannot disagree
    pairs = pd_kv_cp_layout._pairs(getattr(self.kv_args, "kv_layer_ids", None), peer_info.dst_kv_layer_ids,
                                   len(items), len(dst_item), int(getattr(self, "pp_size", 1) or 1))
    if pairs is None:
        raise RuntimeError("the kv_cp scatter's KV layer ids do not pair; geometry_problem should have refused this "
                           "peer at registration")
    for i, j in pairs:
        if i < n_layers and not layout.k_tm:
            dims, rem = divmod(items[i], dim_row_bytes)
            if rem or dst_item[j] * cp != items[i] * heads:
                raise RuntimeError("the kv_cp scatter's K geometry does not tile into dim rows")
            plane = dst_item[j] // heads  # the decode's per-head K plane: dims x block_tokens, rows back to back
            for r in range(dims):
                src_ptrs.append(src_base[i] + r * dim_row_bytes + layout.cp_rank * block_bytes)
                src_strides.append(items[i])
                src_data.append(int(self.kv_args.kv_data_lens[i]))
                src_xfer.append(block_bytes)
                dst_ptrs.append(dst_base[j] + head * plane + r * block_bytes)
                dst_strides.append(dst_item[j])
                dst_data.append(dst_item[j] * dst_num_slots)
                dst_xfer.append(block_bytes)
        else:  # a V entry, or a token-major K entry: the rank's block is one contiguous run on both roles
            b = items[i] // cp
            src_ptrs.append(src_base[i] + layout.cp_rank * b)
            src_strides.append(items[i])
            src_data.append(int(self.kv_args.kv_data_lens[i]))
            src_xfer.append(b)
            dst_ptrs.append(dst_base[j] + head * b)
            dst_strides.append(dst_item[j])
            dst_data.append(dst_item[j] * dst_num_slots)
            dst_xfer.append(b)
    # NIXL preps a dlist against the NAMED agent's metadata, and "" names the local agent (upstream's src-side
    # convention, _init_equal_tp_prep_handle's docstring). The source dlist is this rank's memory: prep it against
    # the local agent and store it under the peer's scatter key — passing the key as the agent name makes NIXL look
    # it up among the remote agents, and the peer is refused at registration (NIXL_ERR_NOT_FOUND).
    self.prep_handles[_src_key(peer_info.agent_name)] = self._prep_equal_tp_dlist(
        "", src_ptrs, src_strides, src_data,
        int(self.kv_args.gpu_id), num_slots=self._num_slots_src, mem_kind=self.src_mem_kind, kv_xfer_lens=src_xfer)
    dst_lens = dst_item
    dst_mem_kind = _homogeneous_kv_mem_kind(peer_info.dst_kv_mem_kinds, "destination")
    self._init_equal_tp_prep_handle(
        peer_info.agent_name, dst_ptrs, int(peer_info.gpu_id), num_slots=peer_info.dst_num_slots,
        mem_kind=dst_mem_kind, kv_item_lens=dst_strides,
        kv_data_lens=dst_data, kv_xfer_lens=dst_xfer)
    peer_info.dst_homogeneous_mem_kind = dst_mem_kind
    peer_info.kv_cp_scatter_layout = layout
    if _SCATTER_DUMP:
        logging.getLogger(__name__).info(
            "Q35SCATTER prep peer=%s cp=%d heads=%d group=%d rank=%d head=%d entries=%d num_slots_src=%s "
            "dst_num_slots=%s blocks=%s src_item_lens=%s dst_item_lens=%s src_ptr0=%#x dst_ptr0=%#x "
            "src_data_lens=%s kv_layer_ids=%s dst_kv_layer_ids=%s",
            peer_info.agent_name, cp, heads, group, int(self.attn_tp_rank), head, len(blocks),
            self._num_slots_src, peer_info.dst_num_slots, blocks,
            [int(l) for l in self.kv_args.kv_item_lens], dst_lens, int(self.kv_args.kv_data_ptrs[0]),
            int(peer_info.dst_kv_ptrs[0]), [int(l) for l in self.kv_args.kv_data_lens],
            getattr(self.kv_args, "kv_layer_ids", None), getattr(peer_info, "dst_kv_layer_ids", None))


def send_kvcache(original_fn, self, peer_name, prefill_kv_indices, dst_kv_ptrs, dst_kv_indices, dst_gpu_id,
                 dst_mem_kind="VRAM", *args, **kwargs):
    """AROUND NixlKVManager.send_kvcache: a scatter peer's transfer runs over ITS source dlist (this rank's pages
    offset by the peer's block); upstream's prepped path would run the shared whole-page dlist."""
    info = self.decode_kv_args_table.get(peer_name)
    if _scatter_layout(info) is None:
        return original_fn(self, peer_name, prefill_kv_indices, dst_kv_ptrs, dst_kv_indices, dst_gpu_id,
                           dst_mem_kind=dst_mem_kind, *args, **kwargs)
    from sglang.srt.disaggregation.nixl.conn import repeat_indices_over_layers

    num_slots_dst = info.dst_num_slots if info.dst_num_slots is not None else self._num_slots_src
    # prepare_payload_xfer's dlists: a dim-major K entry expands into one run per dim row, a token-major K entry
    # (the layout frame's k_tm) is one run like V
    n_layers = len(self.kv_args.kv_item_lens) // 2
    dims = int(self.kv_args.kv_item_lens[0]) // (int(self.kv_args.page_size) * 2)
    entries = n_layers * (1 if _scatter_layout(info).k_tm else dims) + n_layers
    # A dim-major K 256K request's chunks total ~400K 256B descriptors (K expands one run per dim row; token-major
    # K posts 64 KiB runs, 2 per layer per page); every posted batch
    # stays outstanding on the EFA queue pair until the request's end-of-send settle, and an inundated QP aborts
    # the agent (libfabric "Internal queue pair error (local)"). Bound the outstanding work per source rank to
    # ONE batch (a rank scatters to ~8 peers; a per-peer bound still floods the QP): split at
    # SGLANG_NEURON_QWEN35_SCATTER_MAX_DESCS (default 32768, page-aligned so a page's runs stay in one xfer) and, under
    # SGLANG_NEURON_QWEN35_SCATTER_DRAIN (default on), settle the previous batch — including the previous send_kvcache
    # call's last handle of THIS room — before posting the next. Every earlier handle is DONE before the last
    # posts, so the single returned handle accounts for the whole call. The manager's drain lock makes the
    # read-settle-post-update one step across its transfer workers.
    if _SCATTER_DUMP:
        logging.getLogger(__name__).info(
            "Q35SCATTER send peer=%s room=%s src_pages=%s dst_pages=%s entries=%d num_slots_src=%s num_slots_dst=%s "
            "max_descs=%d drain=%d",
            peer_name, getattr(info, "room", None), [int(x) for x in prefill_kv_indices][:48],
            [int(x) for x in dst_kv_indices][:48], entries, self._num_slots_src, num_slots_dst, _SCATTER_MAX_DESCS,
            int(_SCATTER_DRAIN))
    pages_per_xfer = max(1, _SCATTER_MAX_DESCS // entries)
    n_pages = len(prefill_kv_indices)
    last_handle = None
    room = getattr(info, "room", None)
    with _drain_lock(self):
        if _SCATTER_DRAIN:
            prev = getattr(self, "_scatter_drain_prev", None)
            if prev is not None and prev[0] == room:
                settled, failed = self._await_handles([prev[1]], failure_seen=False)
                if not settled or failed:
                    self._scatter_drain_prev = None
                    raise RuntimeError(f"kv_cp scatter drain: the previous transfer to {peer_name} failed")
        for start in range(0, n_pages, pages_per_xfer):
            src_indices = repeat_indices_over_layers(prefill_kv_indices[start:start + pages_per_xfer], entries,
                                                     self._num_slots_src)
            dst_indices = repeat_indices_over_layers(dst_kv_indices[start:start + pages_per_xfer], entries,
                                                     num_slots_dst)
            xfer_handle = self.agent.make_prepped_xfer(
                "WRITE", self.prep_handles[_src_key(peer_name)], src_indices, self.prep_handles[peer_name],
                dst_indices)
            if not xfer_handle:
                raise Exception("KVSender failed to create prepped transfer")
            self._post_xfer(xfer_handle, "KVSender failed to post prepped transfer")
            if _SCATTER_DRAIN and start + pages_per_xfer < n_pages:
                settled, failed = self._await_handles([xfer_handle], failure_seen=False)
                if not settled or failed:
                    raise RuntimeError(f"kv_cp scatter drain: a transfer to {peer_name} failed")
            last_handle = xfer_handle
        if _SCATTER_DRAIN:
            self._scatter_drain_prev = (room, last_handle)
    return last_handle


def maybe_send_extra(original_fn, self, peer_name, prefill_state_indices, *args, **kwargs):
    """AROUND NixlKVManager.maybe_send_extra: in a scatter pairing only ONE of the peer's ``heads`` KV sources
    may send the recurrent state — the prefill rank of the peer's own attention-TP index, whose GDN heads the
    peer holds (the state is attention-TP sharded, not kv_cp sharded; the equal-TP pairing is rank-positional).
    Every source reaches this call at its last KV chunk and upstream pairs state slots positionally, so without
    the gate every source WRITEs the peer's one state slot and the request keeps whichever lands last (observed
    live: per-transfer r vs r^8 shard swaps on a random rank subset, and mixed-component slots where the two
    WRITEs interleaved). The non-owner source still sends its KV and aux; an emptied component list is skipped
    by both upstream's loop and the prepared-plan path."""
    info = self.decode_kv_args_table.get(peer_name)
    if _scatter_layout(info) is not None:
        decode_tp_size = kwargs.get("decode_tp_size", args[3] if len(args) > 3 else None)
        decode_tp_rank = kwargs.get("decode_tp_rank", args[4] if len(args) > 4 else 0)
        if not decode_tp_size:
            raise RuntimeError("the kv_cp scatter's state-ownership gate needs maybe_send_extra's decode_tp_size "
                               "(SGLang's fourth positional argument after prefill_state_indices, or the kwarg); "
                               "without it no rank owns the peer's GDN state — re-validate this call against the "
                               "pinned SGLang")
        own_rank = int(self.attn_tp_rank) % int(self.attn_tp_size)
        if int(decode_tp_rank) % int(decode_tp_size) != own_rank:
            prefill_state_indices = [[] for _ in prefill_state_indices]
    yield from original_fn(self, peer_name, prefill_state_indices, *args, **kwargs)


# The decode
def resolve_rank_mapping(original_fn, self, info, *args, **kwargs):
    """AROUND CommonKVManager._resolve_rank_mapping: a kv_cp decode rank pulls from the ``heads`` prefill ranks that
    hold its block's KV heads (module docstring); a decode without a kv_cp layout takes upstream's mapping."""
    from sglang_neuron.model_executor import kv_layout

    layout = kv_layout.current()
    cp = int(getattr(layout, "cp_size", 1))
    heads = int(getattr(layout, "kv_heads_per_rank", None) or 0)
    if cp <= 1 or heads <= 0:
        return original_fn(self, info, *args, **kwargs)
    if self.is_mla_backend or getattr(self, "is_hybrid_mla_backend", False):
        raise RuntimeError("the kv_cp decode's scatter serves an MHA KV pool; this decode is MLA")
    if int(self.attn_tp_size) != cp or cp % heads:
        raise RuntimeError(f"the kv_cp decode's CP group {cp} is its attention TP {self.attn_tp_size} ranks, "
                           f"sharded by the {heads} KV heads")
    if int(info.attn_tp_size) != cp:
        raise RuntimeError(f"the kv_cp scatter serves an equal-TP prefill of {cp} attention TP ranks, one KV head "
                           f"shard each; this prefill's attention TP is {info.attn_tp_size}")
    if int(info.attn_cp_size) != 1 or int(getattr(self, "attn_cp_size", 1)) != 1:
        raise RuntimeError("the kv_cp scatter serves a prefill without attention CP")
    group = cp // heads
    rank = self.kv_args.engine_rank % self.attn_tp_size  # this rank's position in the replica: its CP rank
    sources = sorted(rank % group + h * group for h in range(heads))
    info.target_tp_rank = sources[0]
    info.target_tp_ranks = sources
    info.target_cp_ranks = [int(getattr(self, "attn_cp_rank", 0))]
    responses = heads
    if int(info.pp_size) == int(self.pp_size):
        info.target_pp_ranks = [self.pp_rank]
    elif int(self.pp_size) == 1:
        info.target_pp_ranks = list(range(int(info.pp_size)))
        responses *= int(info.pp_size)
    else:
        raise RuntimeError(f"the kv_cp scatter serves a prefill pipelined like the decode; prefill PP "
                           f"{info.pp_size}, decode PP {self.pp_size}")
    info.required_dst_info_num = heads  # each prefill rank serves ``heads`` decode ranks, as upstream's sharded case
    info.required_prefill_response_num = responses


def register_prefill(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_NIXL}.NixlKVManager._prepare_payload_xfer", prepare_payload_xfer, around)
    registry.register(f"{_NIXL}.NixlKVManager.send_kvcache", send_kvcache, around)
    registry.register(f"{_NIXL}.NixlKVManager.maybe_send_extra", maybe_send_extra, around)


def register_decode(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_COMMON}.CommonKVManager._resolve_rank_mapping", resolve_rank_mapping, around)
