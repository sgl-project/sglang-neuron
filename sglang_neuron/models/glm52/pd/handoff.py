# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2's PD KV handoff over NIXL: the PP prefill (PP4 x TP32 or PP16 x TP8, whose page the decode serves,
models/glm52/launch.paired_prefill_point) pushes each request's prompt KV into the GK decode's slot-private caches,
every 32-row sub-block whole into the one decode block that holds it (models/glm52/pd/placement).

Both sides keep SGLang's push on its plain MLA path, and the prefill's per-peer prepared dlists. SGLang addresses a
decode by the page indices it publishes for a request and pairs each prefill stage's KV entries with the decode's by
global layer id. Three hooks on each side bend that to the GK, whose KV is not in SGLang's pool but in the kernel's own
caches: one slot per request (``slot = req_pool_idx - 1``) on the slot's ``cp_degree`` ranks, an identity block table,
the slot's 32-row blocks dealt over its ranks. One more on each side carries the layout the GLM pools publish.

Decode side (``register_decode``):

* ``DecodePreallocQueue._transfer_page_indices``: the destination is the request's slot and pages, ``slot x
  pages_per_slot + page`` (``PdPlacement.destination_pages``), the same list on every decode rank. An allocator page is
  never an address: the GK reads none of SGLang's pages.
* ``CommonKVManager._resolve_rank_mapping``: decode rank R takes its KV from the prefill ranks ``r`` with ``r %
  cp_degree == cp(R)`` on every stage, since those ranks hold the sub-blocks the deal puts at its CP position. Each of
  them therefore serves one decode rank of every slot (``required_dst_info_num`` = the slots), and R waits for all of
  them (``required_prefill_response_num``). Upstream maps R to one prefill rank alone, R // 4 behind PP4's 32-rank
  stages and R // 16 behind PP16's 8-rank ones, which holds an eighth of R's blocks at most (a half at PP16).
* ``NixlKVManager.build_receiver_registration_message``: the decode rank's placement, in a trailing frame
  (``Glm52Frame``) that upstream's parser does not read, as it does not read the other plugin frames.

Prefill side (``register_prefill``):

* ``KVArgsRegisterInfo.from_zmq`` reads the frame onto the peer.
* ``NixlKVManager._add_remote_peer``, registered after every other hook on it so that it runs first: a peer that is not
  a GK decode this prefill can serve is refused before anything registers it (``check_peer``), and
  sglang_patches/nixl_peer_refusal turns the raise into a refusal that leaves the bootstrap thread serving.
* ``NixlKVManager.send_kvcache``: each chunk goes whole to its holder, the slot's decode rank at this rank's CP
  position, at that rank's local blocks (``PdPlacement.send_plan``), over the holder's prepared dlist; every other peer
  of the room gets nothing (None, which the transfer worker skips). Behind a decode that pools its indexer keys in top-k
  chunks (the 1m point, models/glm52/points.py), a page's key sub-blocks of the target layers go to the holder
  of their chunk instead, at the same local block: one prepared WRITE to each peer of the entries and blocks it takes
  (``_send_entries``, upstream's prepared send over a subset of the entries). The MTP layer's keys (a drafting decode's)
  are not pooled: they go with the latent, as the draft's own indexer reads them from the slot's ranks. Every such
  holder is a decode rank at this rank's CP position, so a peer this rank already serves. It sits outside
  sglang_patches/pp_decode_stage_transfer's send, which passes an unpipelined decode through.

Both sides: ``disaggregation.utils.get_pd_kv_layout`` names the layout of a pool that published one
(``pd.placement.publish_decode_layout``, ``publish_prefill_layout``) as a ``Glm52KvLayout``, the string ``GLM52_KV_LAYOUT``
carrying the pool's placement. Each role's ``_init_kv_manager`` puts it on the KV args before it builds the manager,
whose constructor takes it as its ``kv_layout`` before its bootstrap registration and threads start, so no peer meets
a GLM manager without its placement. SGLang's own layout check then stops a GK decode at its first contact with a
prefill of another layout, and nixl_head_major's receiver init fails its request to a prefill of none, before it
registers with either: a GK decode reaches only a GLM-5.2 prefill, so no other prefill reads its frame.

On a GLM-5.2 server (its catalog entry names this module as its ``kv_handoff``) every NIXL KV manager is GLM's: one whose pool published no
layout would have upstream address the GK decode's slot caches by allocator page, so the decode's first contact and the
prefill's every peer refuse it. Only the destination hook passes such a manager through, for the fake backend's decode
alone, which moves no KV. Completion, aux, the status messages, quarantine and settle-then-ack are upstream's and the
plugin's, unchanged: every target is a real destination, so each receives aux and the room's status.

The hooks are registered by sglang_patches/kernel_kv_handoff, which pins their targets."""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional

import numpy as np

# the layout both GLM-5.2 pools publish lives in pd.placement, which the pools import without this module; its tag is
# also the tag of the decode's registration frame
from sglang_neuron.models.glm52.pd.placement import GLM52_KV_LAYOUT, LAYOUT_ATTR, Glm52KvLayout, Glm52PdLayout
from sglang_neuron.sglang_patches.pp_decode_stage_transfer import UPSTREAM_REGISTRATION_FRAMES

FRAME_TAG = GLM52_KV_LAYOUT.encode("ascii") + b"="
FRAME_VERSION = 2

@dataclasses.dataclass(frozen=True)
class Glm52Frame:
    """A GK decode rank's placement as its registration frame carries it: its rank, the slot and CP position the GK
    gives that rank, the geometry the prefill's sends are placed for (the CP degree, rows per block, blocks a slot
    holds, bytes of one indexer key block, the indexer key pool: the CP groups of a pool and the pools, 0 and 0
    unpooled) and whether it registered the MTP layer."""

    version: int
    rank: int
    slot: int
    cp: int
    cp_degree: int
    block_rows: int
    blocks_per_slot: int
    key_item_bytes: int
    key_pool_degree: int
    key_pool_split: int
    mtp: int

    def to_bytes(self) -> bytes:
        return FRAME_TAG + ",".join(str(int(v)) for v in dataclasses.astuple(self)).encode("ascii")

    @classmethod
    def parse(cls, payload: bytes) -> "Glm52Frame":
        """The frame from the bytes after its tag; the version comes first, so another version is named, not
        misread."""
        values = payload.decode("ascii").split(",")
        if int(values[0]) != FRAME_VERSION:
            raise ValueError(f"its GLM-5.2 frame is version {values[0]}; this prefill reads version {FRAME_VERSION}")
        names = [f.name for f in dataclasses.fields(cls)]
        if len(values) != len(names):
            raise ValueError(f"its GLM-5.2 frame has {len(values)} fields, not version {FRAME_VERSION}'s {len(names)}")
        return cls(*(int(v) for v in values))


@functools.lru_cache(maxsize=None)
def placement(geo, prefill_ranks: int):
    """The placement of GK point ``geo`` behind prefill stages of ``prefill_ranks`` ranks (``PdPlacement``, which
    refuses a pairing it cannot serve)."""
    from sglang_neuron.models.glm52.pd.placement import PdPlacement

    return PdPlacement(geo, int(prefill_ranks))


def published(mgr) -> Optional[Glm52PdLayout]:
    """The GLM layout a KV manager took with its ``kv_layout`` (a ``Glm52KvLayout``), None for a manager whose pool
    published none."""
    layout = getattr(mgr, "kv_layout", None)
    return layout.pd_layout if isinstance(layout, Glm52KvLayout) else None


def required(mgr, context: str) -> Glm52PdLayout:
    """The GLM layout of a NIXL KV manager of a GLM-5.2 server, or a raise: its pool published none, and upstream would
    address the GK decode's slot caches by allocator page."""
    layout = published(mgr)
    if layout is None:
        raise RuntimeError(f"{context}: this GLM-5.2 server's KV pool published no GLM-5.2 layout (pd_placement"
                           ".publish_decode_layout, publish_prefill_layout), so its KV has no place in the GK decode's "
                           "slots")
    return layout


# Both sides
def get_pd_kv_layout(original_fn, token_to_kv_pool, *args, **kwargs):
    """AROUND disaggregation.utils.get_pd_kv_layout: a pool with a GLM layout names ``GLM52_KV_LAYOUT``, carrying the
    layout to its KV manager, in either order with sglang_patches/nixl_head_major's, which names the Neuron layout only
    where upstream names none."""
    layout = getattr(token_to_kv_pool, LAYOUT_ATTR, None)
    if layout is not None:
        return Glm52KvLayout(layout)
    return original_fn(token_to_kv_pool, *args, **kwargs)


# The decode
def decode_frame(kv_args, layout: Glm52PdLayout) -> Glm52Frame:
    """This decode rank's frame: where the GK puts its rank, and the geometry its caches hold."""
    from sglang_neuron.models.glm52.pd import placement as pd_placement

    geo = layout.geo
    # the placement the rank mapping validated for this page at first contact (decode_placement), which comes first
    place = placement(geo, int(kv_args.page_size) // geo.block_size)
    rank = layout.kernel_rank
    return Glm52Frame(
        version=FRAME_VERSION, rank=rank, slot=place.rank_slot(rank), cp=place.rank_cp(rank),
        cp_degree=geo.cp_degree, block_rows=geo.block_size,
        # the blocks the caches hold a slot: the context's over the CP ranks
        blocks_per_slot=geo.num_blocks // geo.num_seqs_local, key_item_bytes=pd_placement.entry_bytes(geo)[1],
        **key_pool(geo), mtp=int(geo.num_layers in list(kv_args.kv_layer_ids)))


def key_pool(geo) -> dict:
    """The frame's indexer key pool of GK point ``geo``: its CP groups a pool and its pools, 0 and 0 unpooled."""
    degree = int(geo.idx_pool_degree)
    return dict(key_pool_degree=degree, key_pool_split=int(geo.idx_pool_split) if degree else 0)


def build_receiver_registration_message(original_fn, self, *args, **kwargs):
    """AROUND NixlKVManager.build_receiver_registration_message: a GK decode rank's frame trails upstream's frames."""
    layout = required(self, "the GLM-5.2 GK decode's registration with a prefill")
    return list(original_fn(self, *args, **kwargs)) + [decode_frame(self.kv_args, layout).to_bytes()]


def decode_placement(mgr, layout: Glm52PdLayout, info):
    """The placement of this GK decode behind prefill ``info`` (a PrefillServerInfo), refusing a prefill whose pages
    it cannot place: the KV moves only from unpipelined-decode, CP-free attention TP ranks of whole 32-row sub-blocks,
    into an MHA-kind accounting pool, which makes every target a real destination."""
    geo = layout.geo
    problems = []
    if mgr.pp_size != 1 or mgr.attn_cp_size != 1 or info.attn_cp_size != 1:
        problems.append(f"decode PP {mgr.pp_size} and attention CP {mgr.attn_cp_size}, prefill attention CP "
                        f"{info.attn_cp_size} (the GK decode is one stage, and the prefill's CP is its attention TP)")
    if mgr.is_mla_backend:
        # upstream marks every target but one dummy behind an MLA-kind decode pool, and a dummy target sends no KV
        problems.append("an MLA-kind decode pool (the GK decode's SGLang pool is its MHA-kind accounting pool)")
    page = int(mgr.kv_args.page_size)
    if page != info.attn_tp_size * geo.block_size:
        problems.append(f"a {page}-token page, not the prefill's {info.attn_tp_size} ranks x {geo.block_size} rows")
    try:
        place = placement(geo, info.attn_tp_size)
    except ValueError as error:
        problems.append(str(error))
    if problems:
        raise RuntimeError("the GLM-5.2 GK decode cannot take its KV from this prefill: " + "; ".join(problems))
    return place


def resolve_rank_mapping(original_fn, self, info, *args, **kwargs):
    """AROUND CommonKVManager._resolve_rank_mapping: a GK decode rank targets the prefill ranks of its CP position on
    every stage, each of which sends to one decode rank of every slot."""
    layout = required(self, "the GLM-5.2 GK decode's first contact with a prefill")
    place = decode_placement(self, layout, info)
    cp = place.rank_cp(layout.kernel_rank)
    info.target_tp_rank = cp  # it keys the receiver's bootstrap cache
    info.target_tp_ranks = [r for r in range(info.attn_tp_size) if r % layout.geo.cp_degree == cp]
    info.target_cp_ranks = [self.attn_cp_rank]
    info.target_pp_ranks = list(range(info.pp_size))
    info.required_dst_info_num = place.slots
    info.required_prefill_response_num = len(info.target_tp_ranks) * info.pp_size
    return None


def transfer_page_indices(original_fn, self, decode_req, kv_indices, page_size, *args, **kwargs):
    """AROUND DecodePreallocQueue._transfer_page_indices: a GK request's destination is its slot's pages, on every
    decode rank alike; the allocator's pages, which the GK never reads, are not."""
    layout = published(self.kv_manager)
    if layout is None:
        # the fake backend's manager, which moves no KV and takes no layout; a NIXL manager without one was refused at
        # its first contact (resolve_rank_mapping), before any request reaches its destination
        return original_fn(self, decode_req, kv_indices, page_size, *args, **kwargs)
    from sglang.srt.disaggregation.utils import pre_alloc_fill_len

    req = decode_req.req
    tokens = len(kv_indices)
    if tokens != pre_alloc_fill_len(req):
        # a decode prefix: the indices start past it, and the slot's pages would be numbered from the wrong page
        raise RuntimeError(f"the GLM-5.2 GK decode publishes a request's whole slot, but request {req.rid} holds a "
                           f"decode prefix ({tokens} of its {pre_alloc_fill_len(req)} positions to transfer); launch "
                           "the GK decode without --disaggregation-decode-enable-radix-cache")
    place = placement(layout.geo, int(page_size) // layout.geo.block_size)
    return np.asarray(place.destination_pages(int(req.kv.req_pool_idx) - 1, tokens), dtype=np.int32)


# The prefill
def from_zmq(original_fn, cls, msg, *args, **kwargs):
    """AROUND KVArgsRegisterInfo.from_zmq: a GK decode rank's frame, None from any other decode."""
    info = original_fn(cls, msg, *args, **kwargs)
    frames = [bytes(f) for f in msg[UPSTREAM_REGISTRATION_FRAMES:] if bytes(f).startswith(FRAME_TAG)]
    info.dst_glm52_frame = frames[-1][len(FRAME_TAG):] if frames else None
    info.glm52_decode = None
    return info


def check_peer(mgr, layout: Glm52PdLayout, peer) -> Glm52Frame:
    """The frame of a GK decode rank this prefill rank can send to, or a raise naming every reason it cannot: a decode
    of another layout or frame version, a decode that drafts with MTP against a prefill that does not send the MTP rows
    (or the reverse), another placement geometry (CP degree, rows per block, blocks a
    slot, the GK's rank count, or the slot and CP position of its rank), a paired entry of other item bytes (fp8
    against bf16 indexer keys), state on either side, or ranks whose NIXL rank is not their kernel rank.

    Which positions a request may hold is the launch's (the decode's context length admits it, and
    ``PdPlacement.destination_pages`` refuses more pages than a slot holds), not the pairing's."""
    from sglang_neuron.models.glm52.pd import placement as pd_placement

    def refuse(problems):
        raise RuntimeError(f"PD KV handoff to GLM-5.2 decode peer {peer.agent_name}: " + "; ".join(problems))

    if peer.dst_glm52_frame is None:
        refuse([f"its registration has no {GLM52_KV_LAYOUT} frame, so it is not a GK decode (pair the GLM-5.2 GK "
                "decode)"])
    try:
        frame = Glm52Frame.parse(peer.dst_glm52_frame)
    except ValueError as error:
        refuse([str(error)])
    geo, kv_args, problems = layout.geo, mgr.kv_args, []
    # the last stage sends the MTP layer's rows when it registers them (pd.placement.PD_MTP_ENV); a stage before it holds
    # no MTP rows, and a drafting decode's extra entries pair with nothing there
    sends_mtp = geo.num_layers in [int(i) for i in kv_args.kv_layer_ids]
    last_stage = int(getattr(mgr, "pp_rank", 0)) == int(getattr(mgr, "pp_size", 1)) - 1
    if frame.mtp and last_stage and not sends_mtp:
        problems.append(f"it drafts with the MTP layer (registers layer {geo.num_layers}'s caches), and this prefill "
                        f"does not send the MTP rows: launch the prefill with {pd_placement.PD_MTP_ENV}=1")
    if not frame.mtp and sends_mtp:
        problems.append(f"this prefill sends the MTP layer's rows ({pd_placement.PD_MTP_ENV}=1), which a decode that "
                        f"does not draft has no caches for: launch the prefill without {pd_placement.PD_MTP_ENV}")
    rows = int(kv_args.page_size) // int(mgr.attn_tp_size)
    if (frame.cp_degree, frame.block_rows) != (geo.cp_degree, geo.block_size) or rows != frame.block_rows:
        problems.append(f"its slots are dealt in {frame.block_rows}-row blocks over {frame.cp_degree} CP ranks, and "
                        f"this prefill places {rows}-row runs for the GK's {geo.block_size}-row blocks over "
                        f"{geo.cp_degree}")
    slot_blocks = geo.local_blocks_per_seq  # the GK context over its CP ranks, in blocks
    if frame.blocks_per_slot != slot_blocks or peer.dst_num_slots != slot_blocks:
        # the frame's count is the caches', and dst_num_slots the stride of the peer's prepared dlist (its shortest
        # registered entry): a send places blocks by both
        problems.append(f"it holds {frame.blocks_per_slot} blocks a slot and registers {peer.dst_num_slots}, not the "
                        f"{slot_blocks} of the GK's {geo.context}-position slot")
    if int(peer.decode_tp_size) != geo.ranks:
        # each prefill rank's room waits for one destination per slot (required_dst_info_num), which the GK's ranks send
        problems.append(f"its server runs {peer.decode_tp_size} attention TP ranks, not the GK's {geo.ranks}")
    try:
        place = placement(geo, mgr.attn_tp_size)
        where = (place.rank_slot(frame.rank), place.rank_cp(frame.rank))
    except ValueError as error:
        problems.append(str(error))
    else:
        if (frame.slot, frame.cp) != where or frame.rank != int(peer.decode_tp_rank):
            problems.append(f"its GK rank {frame.rank} (NIXL rank {peer.decode_tp_rank}) says slot {frame.slot} CP "
                            f"{frame.cp}, where the GK puts slot {where[0]} CP {where[1]}")
    key_bytes = pd_placement.entry_bytes(geo)[1]
    if frame.key_item_bytes != key_bytes:
        problems.append(f"its indexer key blocks are {frame.key_item_bytes} B, not the GK point's {key_bytes} B")
    pool = key_pool(geo)
    if (frame.key_pool_degree, frame.key_pool_split) != (pool["key_pool_degree"], pool["key_pool_split"]):
        problems.append(f"its indexer keys are pooled over {frame.key_pool_degree} CP groups in {frame.key_pool_split} "
                        f"pools, and this prefill places them for {pool['key_pool_degree']} in "
                        f"{pool['key_pool_split']} (0: unpooled): launch both with the same SGLANG_NEURON_GLM52_POINT")
    if kv_args.state_types or peer.dst_state_data_ptrs:
        # upstream sends a request's state to every real destination by the decode's own state indices
        # (_prepare_state_xfer, maybe_send_extra), which no GK slot holds
        problems.append(f"state components (this prefill's {list(kv_args.state_types)}, "
                        f"{len(peer.dst_state_data_ptrs)} of the decode's): the GK decode holds its KV in its slots "
                        "alone")
    if int(kv_args.engine_rank) != layout.kernel_rank:
        problems.append(f"this prefill rank's NIXL rank is {kv_args.engine_rank} and its kernel CP rank "
                        f"{layout.kernel_rank}: its pages hold the kernel rank's rows")
    problems += _paired_item_problems(kv_args, peer)
    if problems:
        refuse(problems)
    return frame


def _paired_item_problems(kv_args, peer) -> list:
    """Upstream's pairing of this stage's entries with the peer's (``_build_transfer_dst_indices`` at PP > 1, with layer
    ids on both sides), and every pair whose item bytes differ, which a send would write in part."""
    from sglang.srt.disaggregation.utils import build_transfer_entry_pairs

    try:
        pairs = build_transfer_entry_pairs(list(kv_args.kv_layer_ids), list(peer.dst_kv_layer_ids),
                                           len(kv_args.kv_item_lens), len(peer.dst_kv_item_lens))
    except RuntimeError as error:
        return [str(error)]
    differ = [(kv_args.kv_layer_ids[i], kv_args.kv_item_lens[i], peer.dst_kv_item_lens[j]) for i, j in pairs
              if int(kv_args.kv_item_lens[i]) != int(peer.dst_kv_item_lens[j])]
    if not differ:
        return []
    shown = ", ".join(f"layer {lid}: {src} B into {dst} B" for lid, src, dst in differ[:4])
    return [f"{len(differ)} paired entries differ in item bytes ({shown}): a block would be written in part (fp8 "
            "against bf16 indexer keys)"]


def add_remote_peer(original_fn, self, decode_kv_args, *args, **kwargs):
    """AROUND NixlKVManager._add_remote_peer, outermost: a GLM prefill rank registers only a GK decode rank it can send
    to (check_peer), and none while its pool has published no layout."""
    layout = required(self, f"PD KV handoff to GLM-5.2 decode peer {decode_kv_args.agent_name}")
    decode_kv_args.glm52_decode = check_peer(self, layout, decode_kv_args)
    return original_fn(self, decode_kv_args, *args, **kwargs)


def send_kvcache(original_fn, self, peer_name, prefill_kv_indices, dst_kv_ptrs, dst_kv_indices, *args, **kwargs):
    """AROUND NixlKVManager.send_kvcache: this rank's pages of a chunk go whole to the slot's decode rank at this rank's
    CP position, at its local blocks, over its prepared dlist; a pooled decode's key sub-blocks to their chunk's
    holder; to every other GK peer of the room, nothing."""
    frame = getattr(self.decode_kv_args_table.get(peer_name), "glm52_decode", None)
    if frame is None:
        # add_remote_peer put the frame on every peer it let upstream register
        raise RuntimeError(f"PD KV handoff to decode peer {peer_name}: this GLM-5.2 prefill registered no GK decode "
                           "rank of that name")
    layout = required(self, f"PD KV handoff to GLM-5.2 decode peer {peer_name}")
    plan = placement(layout.geo, self.attn_tp_size).send_plan(dst_kv_indices, layout.kernel_rank)
    latent, keys = plan.holder == frame.rank, plan.key_pages(frame.rank)
    if not latent and not keys:
        return None
    if "" not in self.prep_handles or peer_name not in self.prep_handles:
        # upstream's send without a prepared dlist (a peer whose reload failed, _reload_invalidated_peers) pairs this
        # stage's entries with the decode's by position, which puts them in other layers' caches
        raise RuntimeError(f"PD KV handoff to GLM-5.2 decode peer {peer_name}: its prepared dlist is gone")
    if latent and len(keys) == len(plan.blocks):
        # every entry of every page: upstream's own prepared send
        return original_fn(self, peer_name, prefill_kv_indices, dst_kv_ptrs, np.asarray(plan.blocks, dtype=np.int32),
                           *args, **kwargs)
    return _send_entries(self, peer_name, prefill_kv_indices, plan.blocks, latent, keys, layout.geo.num_layers)


@functools.lru_cache(maxsize=None)
def pooled_key_entries(layer_ids: tuple, mtp_layer: int) -> tuple:
    """Which of a stage's registered entries are a target layer's indexer key cache, whose sub-blocks go to their
    chunk's holder under a pooled decode: a layer id's second entry (pd.placement.stage_entry_ids: every layer's latent
    cache, then each full layer's keys; the MTP layer's latent before its keys), except the MTP layer's, which the
    draft reads unpooled from the slot's ranks."""
    seen, out = set(), []
    for layer in layer_ids:
        out.append(layer in seen and layer != mtp_layer)
        seen.add(layer)
    return tuple(out)


def _send_entries(mgr, peer_name, prefill_kv_indices, blocks, latent: bool, keys, mtp_layer: int) -> object:
    """Upstream's prepared send (``_send_kvcache_generic``'s prepped path) of part of a chunk to one peer: every
    entry but the target layers' keys of every page when ``latent`` (the latent's holder), every target layer's key
    entry of the pages ``keys`` (indices into the chunk), into the peer's local ``blocks`` (one per page), as one WRITE
    over the source's and the peer's prepared dlists."""
    is_key = pooled_key_entries(tuple(int(i) for i in mgr.kv_args.kv_layer_ids), int(mtp_layer))
    if len(is_key) != len(mgr.kv_args.kv_item_lens):
        raise RuntimeError(f"PD KV handoff to GLM-5.2 decode peer {peer_name}: {len(is_key)} layer ids for "
                           f"{len(mgr.kv_args.kv_item_lens)} registered entries")
    info = mgr.decode_kv_args_table[peer_name]
    dst_slots = info.dst_num_slots if info.dst_num_slots is not None else mgr._num_slots_src
    src_pages, dst_blocks = np.asarray(prefill_kv_indices, np.int64), np.asarray(blocks, np.int64)
    pages = {False: np.arange(len(dst_blocks)) if latent else np.empty(0, np.int64), True: np.asarray(keys, np.int64)}
    src, dst = [], []
    for entry, key in enumerate(is_key):
        pick = pages[key]
        src.append(entry * mgr._num_slots_src + src_pages[pick])
        dst.append(entry * dst_slots + dst_blocks[pick])
    handle = mgr.agent.make_prepped_xfer("WRITE", mgr.prep_handles[""], np.concatenate(src).astype(np.int32),
                                         mgr.prep_handles[peer_name], np.concatenate(dst).astype(np.int32))
    if not handle:
        raise Exception("KVSender failed to create prepped transfer")
    mgr._post_xfer(handle, "KVSender failed to post prepped transfer")
    return handle


#: The decode side's AROUND hooks by target (sglang_patches.kernel_kv_handoff registers them).
DECODE_HOOKS = {
    "get_pd_kv_layout": get_pd_kv_layout,
    "NixlKVManager.build_receiver_registration_message": build_receiver_registration_message,
    "CommonKVManager._resolve_rank_mapping": resolve_rank_mapping,
    "DecodePreallocQueue._transfer_page_indices": transfer_page_indices,
}
#: The prefill side's, after every other hook on its peer registration and send.
PREFILL_HOOKS = {
    "get_pd_kv_layout": get_pd_kv_layout,
    "KVArgsRegisterInfo.from_zmq": from_zmq,
    "NixlKVManager._add_remote_peer": add_remote_peer,
    "NixlKVManager.send_kvcache": send_kvcache,
}

