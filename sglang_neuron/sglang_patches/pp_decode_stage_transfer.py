# SPDX-License-Identifier: Apache-2.0
"""Let a pipelined PD decode stage take its own layers' KV from an unpipelined prefill over NIXL, and make a pipelined
prefill refuse a decode stage with another layer partition, naming both.

Upstream pairs stages only at equal PP sizes, or an unpipelined decode with every prefill stage, and asserts otherwise
(``CommonKVManager._resolve_rank_mapping``), so a PP2 decode behind a PP1 prefill stops at first contact. Past that
assert it still refuses a decode with fewer KV entries than the prefill (``NixlKVManager._prepare_payload_xfer``), and
its prepared send takes every source entry. At equal PP sizes it never compares layer partitions: with different
``SGLANG_PP_LAYER_PARTITION`` values the decode stage with more layers is served but never gets KV for the extra ones,
and the other is refused with a reason that does not name the partition.

- Decode side (``register_decode``): behind an unpipelined prefill, ``resolve_rank_mapping`` maps attention TP and CP
  ranks as at equal PP sizes, targets the prefill's one stage, and has the prefill expect one destination per decode
  stage. A prefill this decode cannot pair with is refused at first contact with ``PrefillIncompatibleError``, which
  fails only that prefill's requests and keeps the decode serving the others
  (sglang_patches/decode_incompatible_prefill): a prefill at another PP size, a transfer backend that does not pair
  decode stages (``register_decode`` marks the NIXL one ``NixlKVManager.SERVES_PP_DECODE_STAGES``), and the pairings
  the prefill would refuse at registration that the decode can see itself (an MLA pool, a prefill at attention CP, a
  decode at DCP, another attention TP), in the prefill's own words. Each stage sends its layer range [start, end) in a
  trailing registration frame (``STAGE_FRAME_TAG``) that upstream's parser ignores, like the nixl_head_major and
  dcp_page_transfer frames.
- Prefill side (every PD prefill, ``register_prefill``): ``from_zmq`` reads the frame. A pipelined prefill refuses a
  decode stage whose range is not its own, naming both, and otherwise uses upstream's per-stage pairing
  (``add_remote_peer``). An unpipelined prefill pairs its entries of the stage's layers with the stage's entries by
  global layer (the ids sglang_patches/pd_kv_layer_ids publishes), prepares a dlist of them (``prepare_payload_xfer``)
  and sends from those entries only (``send_kvcache``). It refuses at registration what it does not serve: an MLA pool,
  a decode at DCP, a DCP relayout or another attention TP, a pool with state, mixed destination memory kinds, a stage
  that registered no KV entry of its layers, and stage layers it does not hold or that fall outside the stage's range.

A refused registration fails that peer's rooms with the reason on both servers (sglang_patches/nixl_peer_refusal), and
the decode keeps serving the other prefills. Only the prefill knows its layer partition, so only it refuses on that.

Upstream: generic for NIXL; the stage-range registration frame is the protocol change.

Targets, checked against upstream SGLang 43881a7: ``CommonKVManager._resolve_rank_mapping``,
``DecodePreallocQueue._init_kv_manager`` and ``NixlKVManager.build_receiver_registration_message`` on the decode;
``KVArgsRegisterInfo.from_zmq``, ``NixlKVManager._add_remote_peer``, ``NixlKVManager._prepare_payload_xfer`` and
``NixlKVManager.send_kvcache`` on the prefill (all wrapped); and the sources the hooks rely on (``_RELIED_ON_MD5``)."""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import List, Optional, Tuple

_COMMON = "sglang.srt.disaggregation.common.conn"
_NIXL = "sglang.srt.disaggregation.nixl.conn"
_DECODE = "sglang.srt.disaggregation.decode"
_PREFILL = "sglang.srt.disaggregation.prefill"
# a decode stage's layer range, b"<tag><start>,<end>", after upstream's registration frames
STAGE_FRAME_TAG = b"sglang-neuron:pp-stage-layers="
# KVArgsRegisterInfo.from_zmq reads upstream's frames msg[0] to msg[24] (the registration after its guard frame); the
# plugin's frames trail them, so the tag is only looked for there
UPSTREAM_REGISTRATION_FRAMES = 25
# the rank mapping fields upstream's _resolve_rank_mapping sets besides target_pp_ranks
_MAPPING_FIELDS = ("target_tp_rank", "target_tp_ranks", "target_cp_ranks", "required_dst_info_num",
                   "required_prefill_response_num")

# (module, qualified name): md5 of upstream's source; the decode side's wrapped targets, then the prefill side's
_DECODE_WRAPPED = {
    (_COMMON, "CommonKVManager._resolve_rank_mapping"): "70d83eef6dcb74b7cce329808eecba2f",
    (_DECODE, "DecodePreallocQueue._init_kv_manager"): "a49fb126873eaddb433c4243112060ca",
    (_NIXL, "NixlKVManager.build_receiver_registration_message"): "1d28c9d9c8345a71db02d6840a7a3acc",
}
_PREFILL_WRAPPED = {
    (_NIXL, "KVArgsRegisterInfo.from_zmq"): "888b10e09310b5af69364bbf6ffbdfab",
    (_NIXL, "NixlKVManager._add_remote_peer"): "e2ee9f36cd7849d884c49275e4c10bde",
    (_NIXL, "NixlKVManager._prepare_payload_xfer"): "0d52b004d8811f5b3306c8f184f306a7",
    (_NIXL, "NixlKVManager.send_kvcache"): "dc8b5f9e6114739fce8c623ba6bd6428",
}
_RELIED_ON_MD5 = {
    # calls _resolve_rank_mapping after its own checks, and caches the mapped info
    (_COMMON, "CommonKVManager.try_ensure_parallel_info"): "15b2d05442275911ca74f44980700aac",
    # a decode stage's receiver bootstraps and registers with every target_pp_ranks entry (here the prefill's stage 0)
    (_COMMON, "CommonKVReceiver._setup_bootstrap_infos"): "f1bbcd0048d2d4b25f6ede85b28ed76b",
    (_COMMON, "CommonKVReceiver._register_kv_args"): "7bcce3ad2916529b71be6f656f3ba0f7",
    # the receiver sends the mapping's required_dst_info_num (one destination per decode stage) with each room's
    # metadata, and the prefill's bootstrap thread waits for that many destinations before it sends
    (_COMMON, "CommonKVReceiver.init"): "fe1b27f354afbc1dfb46cf7dfa0be764",
    (_NIXL, "NixlKVManager.build_receiver_metadata_message"): "7d27560f44a749e7f441c6e87fe15f08",
    (_NIXL, "TransferInfo.from_zmq"): "7ba9ad4939368443fdd8574389bd4f7e",
    # hands each registration to from_zmq and _add_remote_peer; a refusal raised there refuses the peer
    # (sglang_patches/nixl_peer_refusal rebuilds it)
    (_NIXL, "NixlKVManager._start_bootstrap_thread"): "0ad34e79d2d77da1b2cb45c799e94748",
    # sends an equal-TP MHA peer's chunk through send_kvcache, over its prepared dlists when it has them
    (_NIXL, "NixlKVManager.transfer_worker"): "aaccdcf4cb029cd5aa2f01d265e85c25",
    # the prepared send that _send_stage_prepped follows with the stage's source entries, and the post it calls
    (_NIXL, "NixlKVManager._send_kvcache_generic"): "23f6707348dbe01adfe24a8652a16eb6",
    (_NIXL, "NixlKVManager._post_xfer"): "98788e15a47fbf7631846e2a7b7fcd6a",
    (_NIXL, "repeat_indices_over_layers"): "4c38a3c0e2c2a0320217f43c56806f88",
    # the prepared dlists: one run of every slot per entry, entry after entry
    (_NIXL, "NixlKVManager._init_equal_tp_prep_handle"): "55dbf1feaa081681ebd6afffdb068916",
    (_NIXL, "NixlKVManager._prep_equal_tp_dlist"): "3f25702681e87ec5e77da6f40a7c7956",
    (_NIXL, "_homogeneous_kv_mem_kind"): "f35a8b8eb6307cc4b23816d5aec0ea45",
    # re-prepares a reloaded peer through _prepare_payload_xfer, rebuilding a stage's dlist of its own layers
    (_NIXL, "NixlKVManager._reload_invalidated_peers"): "7a3cfca8476a78c4e9868dd8e53fffad",
    # set the prefill stage's pool bounds (prefill_start_layer, prefill_end_layer) before the manager is built
    (_PREFILL, "PrefillBootstrapQueue._init_kv_manager"): "a64cc26a13904432901c1b256ae1e2d0",
    (_PREFILL, "_transfer_start_layer"): "0d87215ab1104b4905bff40540e72933",
}


# The stage pairings this module does not serve, seen by both roles and refused in one text
def stage_pairing_problems(mla: bool, prefill_attn_cp: int, decode_dcp: int, prefill_attn_tp: int,
                           decode_attn_tp: int) -> List[str]:
    problems = []
    if mla:
        problems.append("an MLA pool")
    if prefill_attn_cp > 1:
        problems.append(f"a prefill at attention CP {prefill_attn_cp}")
    if decode_dcp > 1:
        problems.append(f"a decode at DCP {decode_dcp}")
    if decode_attn_tp != prefill_attn_tp:
        problems.append(f"attention TP {prefill_attn_tp} into {decode_attn_tp}")
    return problems


def stage_refusal(stage: Optional[Tuple[int, int]], problems: List[str]) -> str:
    layers = f" of layers [{stage[0]}, {stage[1]})" if stage is not None else ""
    return (f"NIXL PD transfer into a pipelined decode stage{layers} serves an MHA pool without state at the decode's "
            "attention TP; not: " + "; ".join(problems))


# The decode
def resolve_rank_mapping(original_fn, self, info, *args, **kwargs):
    """AROUND CommonKVManager._resolve_rank_mapping: a decode stage behind an unpipelined prefill maps its attention TP
    and CP ranks as at equal PP sizes, targets the prefill's one stage, and has the prefill expect one destination per
    decode stage. A prefill this decode cannot pair with raises PrefillIncompatibleError at this first contact, which
    fails only that prefill's requests (sglang_patches/decode_incompatible_prefill); upstream's PP and CP asserts would
    leave the scheduler loop. Other pairings are upstream's."""
    from sglang_neuron.sglang_patches.decode_incompatible_prefill import PrefillIncompatibleError

    if self.pp_size == 1 or info.pp_size == self.pp_size:
        return original_fn(self, info, *args, **kwargs)
    if info.pp_size != 1:
        raise PrefillIncompatibleError(
            f"a decode at PP {self.pp_size} serves a prefill at PP {self.pp_size} or 1, not {info.pp_size}")
    if not getattr(self, "SERVES_PP_DECODE_STAGES", False):
        raise PrefillIncompatibleError(
            f"a decode at PP {self.pp_size} behind an unpipelined prefill needs the NIXL transfer backend (this one, "
            f"{type(self).__name__}, does not pair decode stages)")
    problems = stage_pairing_problems(self.is_mla_backend or self.is_hybrid_mla_backend, info.attn_cp_size,
                                      self.dcp_size, info.attn_tp_size, self.attn_tp_size)
    if problems:
        # the prefill would refuse these at registration and fail every room of the stage: refused here, before the
        # stage registers, in the prefill's own words
        raise PrefillIncompatibleError(stage_refusal(getattr(self.kv_args, "stage_layer_range", None), problems))
    staged = dataclasses.replace(info, pp_size=self.pp_size)
    original_fn(self, staged, *args, **kwargs)
    for field in _MAPPING_FIELDS:
        setattr(info, field, getattr(staged, field))
    info.target_pp_ranks = [0]
    # every decode stage sends this prefill rank its own destination for each room
    info.required_dst_info_num = staged.required_dst_info_num * self.pp_size
    return None


def init_kv_manager(original_fn, self, *args, **kwargs):
    """AROUND DecodePreallocQueue._init_kv_manager: record the stage's layer range (the model runner's, which its KV
    pool was built for) in its KV args."""
    kv_manager = original_fn(self, *args, **kwargs)
    layer_info = self.scheduler.tp_worker.model_runner.layer_info
    kv_manager.kv_args.stage_layer_range = (int(layer_info.start_layer), int(layer_info.end_layer))
    return kv_manager


def build_receiver_registration_message(original_fn, self, *args, **kwargs):
    """AROUND NixlKVManager.build_receiver_registration_message: the decode stage's layer range, in a trailing frame."""
    start, end = self.kv_args.stage_layer_range
    return list(original_fn(self, *args, **kwargs)) + [STAGE_FRAME_TAG + f"{start},{end}".encode("ascii")]


# The prefill
def from_zmq(original_fn, cls, msg, *args, **kwargs):
    """AROUND KVArgsRegisterInfo.from_zmq: the decode stage's layer range from its frame; None for an unpipelined
    decode."""
    info = original_fn(cls, msg, *args, **kwargs)
    frames = [bytes(f) for f in msg[UPSTREAM_REGISTRATION_FRAMES:] if bytes(f).startswith(STAGE_FRAME_TAG)]
    info.dst_stage_layer_range = None
    if frames:
        start, end = frames[-1][len(STAGE_FRAME_TAG):].decode("ascii").split(",")
        info.dst_stage_layer_range = (int(start), int(end))
    return info


def prefill_stage_layer_range(kv_args) -> Optional[Tuple[int, int]]:
    """This prefill stage's layers [start, end), as ``PrefillBootstrapQueue._init_kv_manager`` sets them: the pool's
    ``start_layer`` via ``_transfer_start_layer`` (no Neuron pool wraps a token KV view) and its ``end_layer``. None if
    either is unset."""
    start, end = getattr(kv_args, "prefill_start_layer", None), getattr(kv_args, "prefill_end_layer", None)
    return None if start is None or end is None else (int(start), int(end))


def partition_refusal(stage: Tuple[int, int], own: Tuple[int, int]) -> str:
    return (f"the decode stage holds layers [{stage[0]}, {stage[1]}) and this prefill stage [{own[0]}, {own[1]}): a "
            "pipelined prefill and a pipelined decode need the same layer partition (SGLANG_PP_LAYER_PARTITION)")


def add_remote_peer(original_fn, self, decode_kv_args, *args, **kwargs):
    """AROUND NixlKVManager._add_remote_peer: a pipelined prefill stage serves its rank's decode stage on upstream's
    per-stage pairing only if both hold the same layers. Upstream would serve a stage with more layers (never writing
    the extra ones) and refuse one with fewer; here both are refused, naming the two ranges."""
    stage = decode_kv_args.dst_stage_layer_range
    if self.pp_size > 1 and stage is not None:
        own = prefill_stage_layer_range(self.kv_args)
        if own is None:
            raise RuntimeError(f"the decode stage holds layers [{stage[0]}, {stage[1]}) and this prefill stage names no "
                               "layer range (prefill_start_layer, prefill_end_layer), so it cannot tell whether it "
                               "holds the same layers")
        if stage != own:
            raise RuntimeError(partition_refusal(stage, own))
        # the same layers: upstream's per-stage pairing and send
        decode_kv_args.dst_stage_layer_range = None
    return original_fn(self, decode_kv_args, *args, **kwargs)


def pair_stage_entries(src_layer_ids: List[int], dst_layer_ids: List[int], n_src: int, n_dst: int,
                       stage: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Pair this prefill's entries of the stage's layers [start, end) with the stage's entries by global layer id, in
    source order (K before V). Entries of other layers stay unpaired. Every entry in the range must pair on both sides,
    so no stage layer misses its KV."""
    start, end = stage
    if n_src == 0 and n_dst == 0:
        return []
    if (n_src and not src_layer_ids) or (n_dst and not dst_layer_ids):
        raise RuntimeError(f"a pipelined decode stage (layers [{start}, {end})) pairs its KV by layer id, which this PD "
                           "pair does not name on both sides")
    if len(src_layer_ids) != n_src or len(dst_layer_ids) != n_dst:
        raise RuntimeError("Layer metadata length must match transfer entries: "
                           f"src metadata={len(src_layer_ids)} entries={n_src}, "
                           f"dst metadata={len(dst_layer_ids)} entries={n_dst}")
    outside = sorted({lid for lid in dst_layer_ids if not start <= lid < end})
    if outside:
        raise RuntimeError(f"the decode stage of layers [{start}, {end}) registered entries of layers {outside} outside "
                           "its range")
    dst_pos = {}
    for j, lid in enumerate(dst_layer_ids):
        dst_pos.setdefault(lid, deque()).append(j)
    pairs = []
    for i, lid in enumerate(src_layer_ids):
        if not start <= lid < end:
            continue
        if not dst_pos.get(lid):
            raise RuntimeError(f"the decode stage of layers [{start}, {end}) is missing a transfer entry for model "
                               f"layer {lid}")
        pairs.append((i, dst_pos[lid].popleft()))
    unpaired = sorted({lid for lid, rest in dst_pos.items() if rest})
    if unpaired:
        raise RuntimeError(f"this prefill holds no transfer entry for layers {unpaired} of the decode stage of layers "
                           f"[{start}, {end}): their KV would never arrive")
    return pairs


def prepare_payload_xfer(original_fn, self, peer_info, *args, **kwargs):
    """AROUND NixlKVManager._prepare_payload_xfer: a decode stage gets a prepared dlist of its own layers' entries,
    where upstream refuses it for having fewer. At equal PP sizes add_remote_peer clears the range, so only an
    unpipelined prefill gets here with one."""
    if getattr(peer_info, "dst_stage_layer_range", None) is None:
        return original_fn(self, peer_info, *args, **kwargs)
    _prepare_stage_peer(self, peer_info)
    return None


def _prepare_stage_peer(self, peer_info) -> None:
    """Pair the stage's KV entries with this prefill's of the same layers and prepare its dlist. Any pairing this
    prefill does not serve is refused before a request reaches the stage."""
    from sglang.srt.disaggregation.nixl.conn import _homogeneous_kv_mem_kind

    stage = peer_info.dst_stage_layer_range
    problems = []
    if self.is_mla_backend or self.is_hybrid_mla_backend:
        problems.append("an MLA pool")
    if peer_info.dst_dcp_size > 1:
        problems.append(f"a decode at DCP {peer_info.dst_dcp_size}")
    if peer_info.requires_dcp_relayout and peer_info.dst_dcp_size <= 1:
        # the relayout repacks the stage's pages into a buffer of upstream's, which pairs entries by position
        problems.append("a DCP relayout")
    if peer_info.decode_tp_size != self.attn_tp_size:
        # the head-slice dlists pair a peer's entries by position (CommonKVManager.get_mha_kv_ptrs_with_pp)
        problems.append(f"attention TP {self.attn_tp_size} into {peer_info.decode_tp_size}")
    if self.kv_args.state_types:
        # a state component's entries are sent without the stage's range (maybe_send_extra)
        problems.append(f"state components {list(self.kv_args.state_types)}")
    if problems:
        raise RuntimeError(stage_refusal(stage, problems))
    pairs = pair_stage_entries(self.kv_args.kv_layer_ids, peer_info.dst_kv_layer_ids, len(self.kv_args.kv_data_ptrs),
                               len(peer_info.dst_kv_ptrs), stage)
    if not pairs:
        # every request's transfer into it would fail
        raise RuntimeError(f"NIXL PD transfer into a pipelined decode stage of layers [{stage[0]}, {stage[1]}): the "
                           "stage registered no KV entry of its layers")
    try:
        dst_mem_kind = _homogeneous_kv_mem_kind(peer_info.dst_kv_mem_kinds, "destination")
    except NotImplementedError as error:
        raise RuntimeError(f"NIXL PD transfer into a pipelined decode stage of layers [{stage[0]}, {stage[1]}): "
                           f"{error}") from error
    peer_info.stage_src_entries = [i for i, _ in pairs]
    peer_info.dst_homogeneous_mem_kind = dst_mem_kind
    # the source dlist over every entry, which upstream shares between its equal-TP peers; a stage's send takes its
    # paired entries of it
    if "" not in self.prep_handles:
        self._init_equal_tp_prep_handle("", self.kv_args.kv_data_ptrs, self.kv_args.gpu_id,
                                        num_slots=self._num_slots_src, mem_kind=self.src_mem_kind)
    dst_num_slots = peer_info.dst_num_slots if peer_info.dst_num_slots is not None else self._num_slots_src
    dst_kv_item_lens = [peer_info.dst_kv_item_lens[j] for _, j in pairs]
    self._init_equal_tp_prep_handle(
        peer_info.agent_name,
        [peer_info.dst_kv_ptrs[j] for _, j in pairs],
        peer_info.gpu_id,
        num_slots=peer_info.dst_num_slots,
        mem_kind=dst_mem_kind,
        kv_item_lens=dst_kv_item_lens,
        kv_data_lens=[item_len * dst_num_slots for item_len in dst_kv_item_lens],
        kv_xfer_lens=[self.kv_args.kv_item_lens[i] for i, _ in pairs],
    )


def send_kvcache(original_fn, self, peer_name, prefill_kv_indices, dst_kv_ptrs, dst_kv_indices, *args, **kwargs):
    """AROUND NixlKVManager.send_kvcache: a decode stage gets only this prefill's entries of its layers, each into its
    entry of the same layer, over the stage's prepared dlist (upstream's prepared send takes every source entry)."""
    peer = self.decode_kv_args_table.get(peer_name)
    if getattr(peer, "dst_stage_layer_range", None) is None:
        return original_fn(self, peer_name, prefill_kv_indices, dst_kv_ptrs, dst_kv_indices, *args, **kwargs)
    return _send_stage_prepped(self, peer_name, peer, prefill_kv_indices, dst_kv_indices)


def _send_stage_prepped(self, peer_name, peer, prefill_data_indices, dst_data_indices):
    """Upstream's prepared send (``_send_kvcache_generic``) with the stage's paired entries of the source dlist, in the
    order of the stage's dlist entries."""
    import numpy as np

    from sglang.srt.disaggregation.nixl.conn import repeat_indices_over_layers

    if "" not in self.prep_handles or peer_name not in self.prep_handles:
        # upstream's generic send would pair every source entry with the stage's by position; a peer whose reload
        # failed (_reload_invalidated_peers) has no dlist until a later reload succeeds
        raise RuntimeError(f"NIXL PD transfer into the pipelined decode stage {peer_name}: its prepared dlist is gone")
    entries = np.asarray(peer.stage_src_entries, dtype=np.int64)
    src_indices = (entries[:, None] * self._num_slots_src
                   + np.asarray(prefill_data_indices, dtype=np.int64)[None, :]).ravel().astype(np.int32)
    num_slots_dst = peer.dst_num_slots if peer.dst_num_slots is not None else self._num_slots_src
    dst_indices = repeat_indices_over_layers(dst_data_indices, len(entries), num_slots_dst)
    xfer_handle = self.agent.make_prepped_xfer("WRITE", self.prep_handles[""], src_indices,
                                               self.prep_handles[peer_name], dst_indices)
    if not xfer_handle:
        raise Exception("KVSender failed to create prepped transfer")
    self._post_xfer(xfer_handle, "KVSender failed to post prepped transfer")
    return xfer_handle


_DECODE_HOOKS = {
    "CommonKVManager._resolve_rank_mapping": resolve_rank_mapping,
    "DecodePreallocQueue._init_kv_manager": init_kv_manager,
    "NixlKVManager.build_receiver_registration_message": build_receiver_registration_message,
}
_PREFILL_HOOKS = {
    "KVArgsRegisterInfo.from_zmq": from_zmq,
    "NixlKVManager._add_remote_peer": add_remote_peer,
    "NixlKVManager._prepare_payload_xfer": prepare_payload_xfer,
    "NixlKVManager.send_kvcache": send_kvcache,
}


def register_decode(registry, around) -> None:
    """The decode side, on a pipelined decode."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets({**_DECODE_WRAPPED, **_RELIED_ON_MD5}, __name__)
    for module, qualname in _DECODE_WRAPPED:
        registry.register(f"{module}.{qualname}", _DECODE_HOOKS[qualname], around)
    from sglang.srt.disaggregation.nixl.conn import NixlKVManager

    # resolve_rank_mapping serves a decode stage behind an unpipelined prefill on this backend only; any other
    # backend is refused at first contact instead of silently proceeding
    NixlKVManager.SERVES_PP_DECODE_STAGES = True


def register_prefill(registry, around) -> None:
    """The prefill side, on every PD prefill: a pipelined decode may pair with any of them."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets({**_PREFILL_WRAPPED, **_RELIED_ON_MD5}, __name__)
    for module, qualname in _PREFILL_WRAPPED:
        registry.register(f"{module}.{qualname}", _PREFILL_HOOKS[qualname], around)
