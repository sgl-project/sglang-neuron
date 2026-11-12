# SPDX-License-Identifier: Apache-2.0
"""Send KV to a DCP decode by whole pages or whole stripes; upstream plans DCP transfers only from a prefill without
DCP, and by tokens.

Under ``--dcp-size D`` a Neuron pool puts a request's 128-token stripe (one pool page) k on DCP rank k mod D, at local
page k // D of the rank's block table (memory_pool.NeuronMHATokenToKVPool and NeuronMLATokenToKVPool,
``dcp_interleave``). Two pairings use that placement:

* A prefill without DCP sends an MHA DCP decode page k whole to rank k mod D, at decode page-list entry k // D.
  Upstream only sends MLA pools to a DCP decode (token round-robin) and refuses MHA on both sides.
* A DCP prefill of Cp ranks (DCP over its whole attention TP) sends a DCP decode of Cd = a multiple of Cp ranks its
  stripes: stripe j goes from prefill rank j % Cp to decode rank j % Cd (disaggregation/dcp_stripes.py). Recurrent
  state keeps upstream's attention-TP slicing, so decode rank g pairs with prefill ranks g % Cp (stripes) and
  g // (Cd / Cp) (state). Upstream refuses any DCP prefill whose DCP size differs from the decode's.

Decode side (``register_decode``, on a DCP decode):

* ``DecodePreallocQueue._init_kv_manager``, ``NixlKVManager.build_receiver_registration_message``: announce the
  pool's DCP run, DCP size and rank, page size, KV dtype, whether the decode pairs by stripes (its model declares
  ``prefill_context_parallel``) and its drafter-row form, in a trailing registration frame (``DCP_INTERLEAVE_FRAME``)
  that upstream's parser ignores.
* ``CommonKVManager.try_ensure_parallel_info`` (rebuilt from source): accept an MHA DCP decode whose pool gives each DCP
  rank whole pages.
* ``CommonKVManager._resolve_rank_mapping``: pair by stripes with a prefill of a smaller attention TP;
  ``CommonKVReceiver._get_bootstrap_info_from_server`` and ``CommonKVReceiver.send_metadata`` (rebuilt from source)
  tell each prefill rank how many decode ranks it serves, which differs between prefill ranks.

Prefill side (every PD prefill, ``register_prefill``; peers without DCP take upstream's paths; the stripe paths only on
a prefill whose model stores its KV in DCP placement, ``stripes``):

* ``KVArgsRegisterInfo.from_zmq``: read the decode's frame (a run of 1, token round-robin, without one), and the page
  rule a decode announces when its per-request pages are a function of the request's state slot (``PAGE_RULE_FRAME``,
  ``dcp_stripes.PageRule.frame``, read by ``read_page_rule``).
* ``NixlKVManager._add_remote_peer``: check the pairing first (``check_page_peer``, ``check_stripe_peer``; an MLA or DSA
  DCP decode of a prefill without DCP is refused, ``refuse_mla_page_peer``: upstream's token plan does not match the
  Neuron MLA pool's whole blocks per DCP rank and carries no DSA index keys); then
  ``CommonKVManager.requires_dcp_relayout`` says yes for an MHA DCP decode and for a stripe peer. A check that fails
  refuses that decode alone: sglang_patches/nixl_peer_refusal keeps the bootstrap thread serving every other decode
  and fails the refused decode's rooms with the reason on both servers.
* ``CommonKVManager._init_dcp_pack_buffers_once``: no pack buffers for pages or stripes.
* ``NixlKVManager._prepare_payload_xfer``: a page peer gets the dlists of a peer without DCP (equal-TP, or
  nixl_head_major head slices); a stripe peer needs none.
* ``common.utils.build_dcp_token_transfer_plan``: return the chunk (``DcpPagePlan``); an MHA page plan is built from it
  on first use. A decode prefix that is not whole decode pages fails the room.
* ``NixlKVManager.send_kvcache_dcp``: send a page plan over the page dlists, or a stripe peer the chunk's stripes it
  owns (ring drafter rows: only those in the drafter window, at the ring pages of the request's state slot); a rank
  with nothing for the peer posts nothing.
* ``SchedulerDisaggregationPrefillMixin.send_kv_chunk``: the request's prompt length goes on its TransferInfos, where
  a ring peer's chunks read the drafter window.
* ``PrefillBootstrapQueue._init_kv_manager``: a stage whose model keeps drafter rows of its own
  (``pd_draft_kv_pool``) registers them as SGLang's draft entries, which the stripe send sends as drafter rows.
* ``NixlKVManager.maybe_send_extra``: a stripe peer gets state only from its state rank, and only by the page rule it
  announced (send_rule_state: per recurrent layer its 4 parts at the request's state slot).

A stripe peer with a page rule is served only when the rule matches this prefill and the decode's registration
(dcp_stripes.page_rule_problem: page bytes, part names, order, offsets and sizes, groups, ring and window, and every
state entry at the page the rule names); a stripe peer with state and no rule, ring drafter rows without a rule, a rule
without ring rows, a frame this build cannot read, and a rule on any other pairing are refused at registration.

Upstream: a DCP plan next to the MLA token plan, picked by pool layout: whole pages (stripes) per DCP rank, as
upstream's NPU DCP already places them (mem_cache/kv/alloc/runner_ids at interleave_size = page_size), and a pairing
for a DCP prefill of another DCP size. Only the Neuron pools use that layout over NIXL today.

Targets are md5-pinned against upstream SGLang 43881a7."""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional, Tuple

import numpy as np

# the registration frame of a decode whose per-request pages are a function of the request's state slot
# (dcp_stripes.PageRule.frame: the codec the decode and this sender share)
from sglang_neuron.disaggregation.dcp_stripes import PAGE_RULE_FRAME

DCP_INTERLEAVE_FRAME = b"sglang-neuron:kv-dcp-interleave:"

_COMMON = "sglang.srt.disaggregation.common.conn"
_NIXL = "sglang.srt.disaggregation.nixl.conn"
_UTILS = "sglang.srt.disaggregation.common.utils"
_DISAGG_UTILS = "sglang.srt.disaggregation.utils"
_DECODE = "sglang.srt.disaggregation.decode"
_PREFILL = "sglang.srt.disaggregation.prefill"
# (module, qualified name): md5 of upstream's source; the decode side's, then the prefill side's
_DECODE_TARGETS_MD5 = {
    (_COMMON, "CommonKVManager.try_ensure_parallel_info"): "15b2d05442275911ca74f44980700aac",
    (_NIXL, "NixlKVManager.build_receiver_registration_message"): "1d28c9d9c8345a71db02d6840a7a3acc",
    (_DECODE, "DecodePreallocQueue._init_kv_manager"): "a49fb126873eaddb433c4243112060ca",
    (_COMMON, "CommonKVManager._resolve_rank_mapping"): "70d83eef6dcb74b7cce329808eecba2f",
    (_COMMON, "CommonKVReceiver._get_bootstrap_info_from_server"): "ddba17351cc8bffe54aabd1da888e799",
    (_COMMON, "CommonKVReceiver.send_metadata"): "6af740c31cb4a93febac0e80f68af77b",
    # fetches one bootstrap entry per target rank and stage, and marks every one a real sender when target_tp_rank is
    # None
    (_COMMON, "CommonKVReceiver._setup_bootstrap_infos"): "f1bbcd0048d2d4b25f6ede85b28ed76b",
}
_PREFILL_TARGETS_MD5 = {
    (_COMMON, "CommonKVManager.requires_dcp_relayout"): "f84990db91dfd31901080126008830f0",
    (_COMMON, "CommonKVManager._init_dcp_pack_buffers_once"): "2810bb36ba7bc2dccc42096f91a7011d",
    (_NIXL, "NixlKVManager._add_remote_peer"): "e2ee9f36cd7849d884c49275e4c10bde",
    (_NIXL, "NixlKVManager._prepare_payload_xfer"): "0d52b004d8811f5b3306c8f184f306a7",
    (_NIXL, "NixlKVManager.send_kvcache_dcp"): "d3c82d1c580cc38dc43a2f9dd9b5345c",
    (_NIXL, "KVArgsRegisterInfo.from_zmq"): "888b10e09310b5af69364bbf6ffbdfab",
    (_UTILS, "build_dcp_token_transfer_plan"): "87788be2090c0c03766d4d762e8f3646",
    (_NIXL, "NixlKVManager.maybe_send_extra"): "a0419a7cd7d96bc4a2b3f9e61c948c92",
    # every chunk of a request is queued through send_kv_chunk, after the decode's TransferInfos arrived
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.send_kv_chunk"): "0e5a574d819e4c08526f8751615bc139",
    # registers self.draft_token_to_kv_pool's entries after the target's (num_draft_entries) and their band ids
    (_PREFILL, "PrefillBootstrapQueue._init_kv_manager"): "a64cc26a13904432901c1b256ae1e2d0",
    # a relayout peer's chunk goes plan -> pack -> send_kvcache_dcp, its state through maybe_send_extra on the last
    # chunk, and the plan gets the request's own TransferInfo.dst_kv_indices (request_of); the stripe send posts
    # through the generic sender, by page; the state rank pairing is upstream's scatter slice
    (_NIXL, "NixlKVManager.transfer_worker"): "aaccdcf4cb029cd5aa2f01d265e85c25",
    (_NIXL, "NixlKVManager._send_kvcache_generic"): "23f6707348dbe01adfe24a8652a16eb6",
    (_DISAGG_UTILS, "compute_mamba_state_slice_blocks"): "1b42454cba603cb70fc3416ebdf8d156",
}
# try_ensure_parallel_info is rebuilt from source with this check replaced: an MHA pool whose DCP run is its page
# (whole pages per DCP rank) pairs too
UPSTREAM_DCP_CHECK = '''    if self.dcp_size > 1:
        if not (self.is_mla_backend or self.is_hybrid_mla_backend):
            raise RuntimeError(
                "PD decode DCP requires an MLA or hybrid-MLA KV pool."
            )
'''
NEURON_DCP_CHECK = '''    if self.dcp_size > 1:
        if not (
            self.is_mla_backend
            or self.is_hybrid_mla_backend
            or getattr(self.kv_args, "kv_dcp_interleave", 1) == self.kv_args.page_size
        ):
            raise RuntimeError(
                "PD decode DCP requires an MLA or hybrid-MLA KV pool, or an MHA "
                "pool that gives each DCP rank whole pages."
            )
'''
# send_metadata is rebuilt from source with each prefill rank's count of the decode ranks it serves
UPSTREAM_DST_INFO_NUM = "required_dst_info_num=self.required_dst_info_num,"
NEURON_DST_INFO_NUM = 'required_dst_info_num=bootstrap_info.get("neuron_dst_info_num", self.required_dst_info_num),'


# The page plan
def _prefix_pages(decode_prefix_len: int, physical_page_size: int, dcp_size: int) -> int:
    """Prefill pages in the decode prefix (what the decode's radix cache holds). The prefix must be whole virtual pages
    (page x DCP size), or the first page sent would land on the wrong rank."""
    if not decode_prefix_len:
        return 0
    virtual_page_size = physical_page_size * dcp_size
    if physical_page_size <= 0 or decode_prefix_len % virtual_page_size != 0:
        raise ValueError(
            "PD DCP page transfer requires decode_prefix_len to align to the virtual DCP page size "
            f"({virtual_page_size}), got {decode_prefix_len}")
    return decode_prefix_len // physical_page_size


def build_dcp_page_transfer_plan(src_page_indices, dst_page_indices, *, dcp_size: int, dcp_rank: int,
                                 src_page_offset: int = 0, decode_prefix_len: int = 0,
                                 physical_page_size: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """(source pages, destination pages) one DCP rank gets from a prefill without DCP. Request page k (prefix pages +
    src_page_offset + i) goes whole to rank k % dcp_size, at the decode page-list entry of virtual page k // dcp_size
    (the list starts after the prefix). Raises on an unaligned prefix, or when the decode's pages run out."""
    prefix_pages = _prefix_pages(decode_prefix_len, physical_page_size, dcp_size)
    src = np.asarray(src_page_indices, dtype=np.int64)
    dst = np.asarray(dst_page_indices, dtype=np.int64)
    pages = prefix_pages + src_page_offset + np.arange(src.size, dtype=np.int64)
    mine = pages % dcp_size == dcp_rank
    ordinals = pages[mine] // dcp_size - prefix_pages // dcp_size
    if ordinals.size and int(ordinals.max()) >= dst.size:
        raise ValueError(
            f"Insufficient destination DCP pages: required={int(ordinals.max()) + 1}, provided={dst.size}, "
            f"src_page_offset={src_page_offset}, dcp_rank={dcp_rank}")
    return src[mine], dst[ordinals]


@dataclasses.dataclass(frozen=True)
class DcpChunk:
    """One chunk of a DCP transfer as upstream's worker plans it: its prefill pages, the decode's pages after its
    prefix, the decode rank's DCP size and rank, and where the chunk starts."""

    src_page_indices: np.ndarray
    dst_page_indices: np.ndarray
    physical_page_size: int
    dcp_size: int
    dcp_rank: int
    src_page_offset: int = 0
    decode_prefix_len: int = 0
    num_kv_tokens: Optional[int] = None
    # the decode page list the worker passed: its request's TransferInfo.dst_kv_indices, by identity
    dst_source: object = dataclasses.field(default=None, compare=False, repr=False)


class DcpPagePlan:
    """The chunk upstream's worker hands send_kvcache_dcp. A page peer sends the pages of one chunk one DCP rank
    receives whole (build_dcp_page_transfer_plan, built on first use); a stripe peer plans the chunk's stripes."""

    def __init__(self, chunk: DcpChunk):
        self.chunk = chunk

    @property
    def target_src_token_indices(self) -> np.ndarray:
        """No tokens to pack (NixlKVManager._pack_dcp_rank_once packs none for an empty set)."""
        return np.empty((0,), dtype=np.int64)

    @functools.cached_property
    def _pages(self) -> Tuple[np.ndarray, np.ndarray]:
        c = self.chunk
        src = np.asarray(c.src_page_indices, dtype=np.int64)
        if c.num_kv_tokens is not None:
            src = src[: -(-int(c.num_kv_tokens) // int(c.physical_page_size))]
        return build_dcp_page_transfer_plan(src, c.dst_page_indices, dcp_size=c.dcp_size, dcp_rank=c.dcp_rank,
                                            src_page_offset=c.src_page_offset,
                                            decode_prefix_len=c.decode_prefix_len,
                                            physical_page_size=c.physical_page_size)

    @property
    def src_pages(self) -> np.ndarray:
        return self._pages[0]

    @property
    def dst_pages(self) -> np.ndarray:
        return self._pages[1]


def _mla(mgr) -> bool:
    return bool(mgr.is_mla_backend or mgr.is_hybrid_mla_backend)


def _page_peer(mgr, peer_info) -> bool:
    """Whether this prefill sends the DCP decode ``peer_info`` whole pages (MHA) instead of upstream's token plan
    (MLA)."""
    return bool(getattr(peer_info, "requires_dcp_relayout", False)) and not _mla(mgr)


def _stripe_target(mgr, dst_dcp_size) -> bool:
    """Whether this prefill sends a decode of DCP size ``dst_dcp_size`` by stripes: a DCP prefill and a DCP decode of
    another size (upstream's equal-DCP and DCP-to-none pairings stay upstream's)."""
    return int(mgr.dcp_size) > 1 and int(dst_dcp_size) not in (1, int(mgr.dcp_size))


def _frame(peer_info):
    from sglang_neuron.disaggregation import dcp_stripes

    frame = getattr(peer_info, "dst_dcp_frame", None)
    return frame if frame is not None else dcp_stripes.DecodeFrame(int(getattr(peer_info, "dst_dcp_interleave", 1)))


# The decode
def init_kv_manager(original_fn, self, *args, **kwargs):
    """AROUND DecodePreallocQueue._init_kv_manager: the manager's args carry the pool's DCP run, whether the decode
    pairs by stripes and its drafter-row form."""
    from sglang_neuron.disaggregation import dcp_stripes
    from sglang_neuron.mem_cache.memory_pool import full_attention_pool

    kv_manager = original_fn(self, *args, **kwargs)
    pool = full_attention_pool(self.token_to_kv_pool)
    kv_manager.kv_args.kv_dcp_interleave = int(getattr(pool, "dcp_interleave", 1) or 1)
    kv_manager.kv_args.neuron_stripe_pairing = pairs_by_stripes(kv_manager)
    kv_manager.kv_args.neuron_draft_rows = dcp_stripes.DECODE_DRAFT_ROWS
    return kv_manager


def pairs_by_stripes(kv_manager) -> bool:
    """A DCP decode pairs by stripes when its model's PD prefill stores DCP KV (ModelSpec.prefill_context_parallel)."""
    from sglang_neuron.models import catalog

    if int(getattr(kv_manager, "dcp_size", 1)) <= 1:
        return False
    return bool(catalog.model_spec(kv_manager.server_args)[1].prefill_context_parallel)


def decode_frame(mgr):
    """The frame a decode manager announces."""
    from sglang_neuron.disaggregation import dcp_stripes

    kv = mgr.kv_args
    return dcp_stripes.DecodeFrame(
        run=int(getattr(kv, "kv_dcp_interleave", 1) or 1), dcp_size=int(getattr(mgr, "dcp_size", 1)),
        dcp_rank=int(getattr(mgr, "dcp_rank", 0)), page_size=int(getattr(kv, "page_size", 0) or 0),
        kv_dtype=str(getattr(kv, "kv_cache_dtype_str", "") or ""),
        stripes=bool(getattr(kv, "neuron_stripe_pairing", False)),
        draft_rows=str(getattr(kv, "neuron_draft_rows", dcp_stripes.DRAFT_SHARDED)))


def build_receiver_registration_message(original_fn, self, *args, **kwargs):
    """AROUND NixlKVManager.build_receiver_registration_message: the decode's frame, trailing upstream's."""
    return list(original_fn(self, *args, **kwargs)) + [DCP_INTERLEAVE_FRAME + decode_frame(self).encode()]


def decode_stripe_pairing(mgr, info):
    """The stripe pairing of this decode manager with the prefill ``info`` (a PrefillServerInfo) describes: a decode
    that pairs by stripes and a prefill of a smaller attention TP, whose DCP is its attention TP. None for upstream's
    pairing; raises on one the stripe map does not serve."""
    from sglang_neuron.disaggregation import dcp_stripes

    if not getattr(mgr.kv_args, "neuron_stripe_pairing", False) or int(info.attn_tp_size) >= int(mgr.attn_tp_size):
        return None
    pairing = dcp_stripes.StripePairing(
        prefill_dcp=int(info.attn_tp_size), decode_dcp=int(mgr.dcp_size), prefill_tp=int(info.attn_tp_size),
        decode_tp=int(mgr.attn_tp_size), block=int(mgr.kv_args.page_size),
        draft_rows=str(getattr(mgr.kv_args, "neuron_draft_rows", dcp_stripes.DRAFT_SHARDED)),
        state=bool(getattr(mgr.kv_args, "state_types", None)))
    problem = pairing.problem()
    if problem:
        raise RuntimeError(f"PD DCP stripe pairing with a prefill of attention TP {info.attn_tp_size}: {problem}")
    return pairing


def resolve_rank_mapping(original_fn, self, info, *args, **kwargs):
    """AROUND CommonKVManager._resolve_rank_mapping: a stripe pairing takes stripes from prefill rank g % Cp and state
    from g // (Cd / Cp), waits for every one of them on every prefill stage, and tells each how many decode ranks it
    serves. Every target is a real sender (target_tp_rank None)."""
    out = original_fn(self, info, *args, **kwargs)
    pairing = decode_stripe_pairing(self, info)
    if pairing is None:
        return out
    rank = int(self.kv_args.engine_rank) % int(self.attn_tp_size)
    ranks = pairing.prefill_ranks(rank)
    # upstream's count for a larger decode attention TP: one prefill rank per CP rank and stage
    per_rank = int(info.required_prefill_response_num)
    info.target_tp_rank = None
    info.target_tp_ranks = list(ranks)
    info.required_prefill_response_num = per_rank * len(ranks)
    info.neuron_dst_info_num = {r: len(pairing.decode_peers(r)) for r in ranks}
    return out


def get_bootstrap_info_from_server(original_fn, self, prefill_dp_rank, prefill_cp_rank, target_tp_rank,
                                   target_pp_rank, *args, **kwargs):
    """AROUND CommonKVReceiver._get_bootstrap_info_from_server: a stripe pairing's entry carries how many decode ranks
    its prefill rank serves."""
    entry = original_fn(self, prefill_dp_rank, prefill_cp_rank, target_tp_rank, target_pp_rank, *args, **kwargs)
    served = getattr(getattr(self, "prefill_info", None), "neuron_dst_info_num", None)
    if entry is not None and served:
        entry["neuron_dst_info_num"] = served[int(target_tp_rank)]
    return entry


# The prefill
def from_zmq(original_fn, cls, msg, *args, **kwargs):
    """AROUND KVArgsRegisterInfo.from_zmq: the decode's frame (a run of 1 without one)."""
    from sglang_neuron.disaggregation import dcp_stripes

    info = original_fn(cls, msg, *args, **kwargs)
    frames = [f for f in msg if isinstance(f, (bytes, bytearray)) and bytes(f).startswith(DCP_INTERLEAVE_FRAME)]
    info.dst_dcp_frame = dcp_stripes.DecodeFrame.decode(bytes(frames[-1])[len(DCP_INTERLEAVE_FRAME):]) if frames \
        else None
    info.dst_dcp_interleave = info.dst_dcp_frame.run if frames else 1
    rules = [bytes(f) for f in msg if isinstance(f, (bytes, bytearray)) and bytes(f).startswith(PAGE_RULE_FRAME)]
    info.dst_page_frame = rules[-1] if rules else None
    info.dst_page_rule, info.dst_page_rule_problem = read_page_rule(info.dst_page_frame)
    return info


def page_rule_parser():
    """The page-frame reader (``dcp_stripes.parse_page_frame``, the codec the decode announces its rule with)."""
    from sglang_neuron.disaggregation import dcp_stripes

    return dcp_stripes.parse_page_frame


def read_page_rule(frame):
    """(the parsed page rule, None), (None, why it cannot be read) or (None, None) without a frame."""
    if frame is None:
        return None, None
    parse = page_rule_parser()
    try:
        return parse(frame), None
    except (ValueError, KeyError, UnicodeDecodeError) as exc:
        return None, f"its page frame does not parse ({exc})"


def requires_dcp_relayout(original_fn, self, dst_dcp_size, dst_dcp_rank, *args, **kwargs):
    """AROUND CommonKVManager.requires_dcp_relayout: a prefill without DCP relayouts for an MHA DCP decode, and a DCP
    prefill for a decode of another DCP size; upstream refuses both (_add_remote_peer has checked the pairing)."""
    if self.dcp_size == 1 and dst_dcp_size > 1 and not _mla(self):
        return True
    if _stripe_target(self, dst_dcp_size):
        return True
    return original_fn(self, dst_dcp_size, dst_dcp_rank, *args, **kwargs)


def check_page_peer(mgr, peer_info) -> None:
    """Refuse an MHA DCP decode this prefill cannot send whole pages to. Its DCP run must be this prefill's page, and
    each KV entry must hold the paired entry's page or, at a larger decode attention TP, a head slice of it the whole
    DCP group shares (nixl_head_major maps decode rank i to KV head i // (decode TP / KV heads), so the head replication
    must be a multiple of the DCP size). Pools with state are refused: hybrid SWA state is not sent to a DCP decode."""
    page = int(mgr.kv_args.page_size)
    run = int(getattr(peer_info, "dst_dcp_interleave", 1) or 1)
    if run != page:
        raise RuntimeError(
            f"PD DCP to an MHA decode needs it to give each DCP rank whole pages of this prefill's {page} tokens; the "
            f"decode's DCP run is {run} (a decode without the Neuron DCP pool)")
    if getattr(mgr.kv_args, "state_types", None):
        raise NotImplementedError(
            f"PD DCP page transfer sends no state ({mgr.kv_args.state_types}): the sliding-window state of a hybrid "
            "SWA pool is not sent to a DCP decode")
    n_src, n_dst = len(mgr.kv_args.kv_item_lens), len(peer_info.dst_kv_item_lens)
    dst_indices = mgr._build_transfer_dst_indices(peer_info=peer_info, n_src=n_src, n_dst=n_dst)
    ratios = {(int(mgr.kv_args.kv_item_lens[i]), int(peer_info.dst_kv_item_lens[j])) for i, j in enumerate(dst_indices)}
    decode_tp, prefill_tp, dcp = int(peer_info.decode_tp_size), int(mgr.attn_tp_size), int(peer_info.dst_dcp_size)
    if decode_tp == prefill_tp and all(src == dst for src, dst in ratios):
        return
    total_heads = int(getattr(mgr.kv_args, "total_kv_head_num", 0) or 0) or int(
        getattr(mgr.kv_args, "kv_head_num", 0) or 0) * prefill_tp
    replication = decode_tp // total_heads if total_heads else 0
    n = decode_tp // prefill_tp if decode_tp % prefill_tp == 0 else 0
    heads_ratio = n // max(1, replication) if n else 0
    if not (n and replication >= 1 and replication % dcp == 0 and heads_ratio >= 1
            and all(src == dst * heads_ratio for src, dst in ratios)):
        raise RuntimeError(
            "PD DCP page transfer needs each decode rank's KV page to be the prefill rank's page or a head slice of it "
            f"that every rank of its DCP group shares: pages {sorted(ratios)} (prefill, decode bytes), prefill "
            f"attention TP {prefill_tp}, decode attention TP {decode_tp}, DCP {dcp}, {total_heads} KV heads")


def refuse_mla_page_peer(mgr, peer_info) -> None:
    """Refuse an MLA (or DSA) DCP decode paired with this prefill without DCP. Upstream's MLA plan would send it by
    tokens round-robin over the DCP group and carries no DSA index keys, while the Neuron MLA pool keeps whole blocks
    per DCP rank (mem_cache/cp_pages.CpPages, its ``dcp_interleave``); cp_pages.latent_page_plan maps those blocks and
    nothing executes it. No model serves this pairing: refused until one needs it."""
    raise RuntimeError(
        f"PD DCP to an MLA decode (--dcp-size {int(peer_info.dst_dcp_size)}) from a prefill without DCP: upstream's "
        "MLA plan places tokens round-robin and sends no DSA index keys, and the Neuron MLA pool keeps whole blocks "
        "per DCP rank (mem_cache/cp_pages.latent_page_plan maps them; no transfer executes it). Serve the decode "
        "without --dcp-size, or pair it with a DCP prefill")


def check_stripe_peer(mgr, peer_info):
    """The decode rank ``peer_info`` as this DCP prefill rank serves it by stripes (dcp_stripes.StripePeer); raises
    on a pairing the stripe map does not serve, naming what to change."""
    from sglang.srt.disaggregation.base.conn import StateType
    from sglang.srt.disaggregation.utils import resolve_dcp_dst_entry_indices

    from sglang_neuron.disaggregation import dcp_stripes

    name = peer_info.agent_name
    kv = mgr.kv_args
    frame = _frame(peer_info)
    rule = None
    if getattr(peer_info, "dst_page_frame", None) is not None:
        problem = peer_info.dst_page_rule_problem
        if problem is None:
            try:
                rule = dcp_stripes.PageRule.of(peer_info.dst_page_rule)
            except (ValueError, TypeError) as exc:
                problem = f"its page rule is not one this prefill places by ({exc})"
        if problem is not None:
            raise RuntimeError(f"PD DCP stripe transfer to decode peer {name}: it announces a page rule (its KDA state "
                               f"and drafter rows at pages of its state slot), and {problem}")
    if int(mgr.dcp_rank) != int(mgr.attn_tp_rank):
        raise RuntimeError(f"PD DCP stripe transfer to decode peer {name}: this prefill rank's DCP rank "
                           f"{mgr.dcp_rank} is not its attention-TP rank {mgr.attn_tp_rank}")
    states = list(getattr(kv, "state_types", None) or [])
    if any(state != StateType.MAMBA for state in states):
        raise NotImplementedError(f"PD DCP stripe transfer to decode peer {name}: it sends recurrent (mamba) state "
                                  f"only, and this prefill's pools hold {states}")
    if states and rule is None:
        raise RuntimeError(f"PD DCP stripe transfer to decode peer {name}: this prefill's pools hold recurrent state, "
                           "which the stripe sender sends only to a decode that announces where each part of it lands "
                           "(a page rule, PAGE_RULE_FRAME); upstream's state slicing would write this prefill's "
                           "channel-major conv state into a pool whose axis order it cannot see")
    pairing = dcp_stripes.StripePairing(
        prefill_dcp=int(mgr.dcp_size), decode_dcp=int(peer_info.dst_dcp_size), prefill_tp=int(mgr.attn_tp_size),
        decode_tp=int(peer_info.decode_tp_size), block=int(kv.page_size), draft_rows=frame.draft_rows,
        state=bool(states))
    n_src, n_dst = len(kv.kv_item_lens), len(peer_info.dst_kv_item_lens)
    dst_indices = resolve_dcp_dst_entry_indices(list(kv.kv_layer_ids or []), list(peer_info.dst_kv_layer_ids or []),
                                                n_src, n_dst)
    peer = dcp_stripes.stripe_peer(
        name=name, pairing=pairing, prefill_rank=int(mgr.attn_tp_rank),
        decode_rank=int(peer_info.decode_tp_rank) % int(peer_info.decode_tp_size), frame=frame,
        upstream_dcp=(int(peer_info.dst_dcp_size), int(peer_info.dst_dcp_rank)), prefill_page=int(kv.page_size),
        prefill_kv_dtype=str(mgr.kv_cache_dtype_str), src_item_lens=list(kv.kv_item_lens),
        dst_item_lens=list(peer_info.dst_kv_item_lens), dst_indices=dst_indices,
        num_draft=int(getattr(kv, "num_draft_entries", 0) or 0), page_rule=rule)
    if rule is not None:
        i = _mamba_index(mgr) if states else 0
        dst_state = []
        for field in ("dst_state_data_ptrs", "dst_state_item_lens", "dst_state_layer_ids"):
            lists = getattr(peer_info, field, None) or []
            dst_state.append(list(lists[i]) if i < len(lists) else [])
        problem = dcp_stripes.page_rule_problem(
            rule, decode_dcp=pairing.decode_dcp, ratio=pairing.state_ratio, block=pairing.block,
            stripe_bytes=int(kv.kv_item_lens[0]), sources=rule_state_sources(mgr) if states else [],
            dst_kv_ptrs=list(peer_info.dst_kv_ptrs), dst_state_ptrs=dst_state[0], dst_state_strides=dst_state[1],
            dst_state_layer_ids=dst_state[2])
        if problem is not None:
            raise RuntimeError(f"PD DCP stripe transfer to decode peer {name}: it announces a page rule (its KDA "
                               f"state and drafter rows at pages of its state slot), and {problem}")
    return peer


def add_remote_peer(original_fn, self, decode_kv_args, *args, **kwargs):
    """AROUND NixlKVManager._add_remote_peer: a DCP decode is checked before it is registered: of a prefill
    without DCP, an MHA one for whole pages (check_page_peer) and an MLA one refused (refuse_mla_page_peer);
    a decode a DCP prefill sends by stripes for the stripe map (check_stripe_peer). A raise refuses that decode alone
    (nixl_peer_refusal.register_or_refuse, around the registration in the bootstrap thread)."""
    stripes = _stripe_target(self, decode_kv_args.dst_dcp_size)
    if getattr(decode_kv_args, "dst_page_frame", None) is not None and not stripes:
        raise RuntimeError(f"PD transfer to decode peer {decode_kv_args.agent_name}: it announces a page rule (its KDA "
                           "state and drafter rows at pages of its state slot), which only a DCP prefill sends, by "
                           "stripes, to a decode of a larger DCP size; this prefill runs --dcp-size "
                           f"{self.dcp_size} and the decode {decode_kv_args.dst_dcp_size}")
    if self.dcp_size == 1 and int(decode_kv_args.dst_dcp_size) > 1:
        if _mla(self):
            refuse_mla_page_peer(self, decode_kv_args)
        check_page_peer(self, decode_kv_args)
    if stripes:
        decode_kv_args.neuron_stripes = check_stripe_peer(self, decode_kv_args)
    return original_fn(self, decode_kv_args, *args, **kwargs)


def init_dcp_pack_buffers_once(original_fn, self, *args, **kwargs):
    """AROUND CommonKVManager._init_dcp_pack_buffers_once: an MHA prefill sends whole pages and a DCP prefill whole
    stripes; neither packs."""
    if not _mla(self) or int(self.dcp_size) > 1:
        return None
    return original_fn(self, *args, **kwargs)


def prepare_payload_xfer(original_fn, self, peer_info, *args, **kwargs):
    """AROUND NixlKVManager._prepare_payload_xfer: a stripe peer writes device memory by page and needs no dlist; a
    page peer gets the dlists upstream builds for a peer without DCP (equal-TP, else head slices), built from a copy
    without the relayout flag; send_kvcache_dcp uses them. Upstream's DCP preparation assumes equal per-token bytes on
    both sides, so it would refuse a head slice and a stripe-sharded drafter pool."""
    stripes = getattr(peer_info, "neuron_stripes", None)
    if stripes is not None:
        kinds = {peer_info.dst_kv_mem_kinds[j] for j in stripes.target_dst + stripes.draft_dst}
        if kinds - {"VRAM"}:
            raise NotImplementedError(f"PD DCP stripe transfer to decode peer {peer_info.agent_name}: its KV entries "
                                      f"are in {sorted(kinds)} memory; the stripe send writes device memory only")
        peer_info.dst_homogeneous_mem_kind = "VRAM"
        return None
    if not _page_peer(self, peer_info):
        return original_fn(self, peer_info, *args, **kwargs)
    plain = dataclasses.replace(peer_info, requires_dcp_relayout=False)
    out = original_fn(self, plain, *args, **kwargs)
    peer_info.dst_homogeneous_mem_kind = plain.dst_homogeneous_mem_kind
    peer_info.dcp_page_slices = int(peer_info.decode_tp_size) != int(self.attn_tp_size)
    return out


def build_dcp_token_transfer_plan(original_fn, src_page_indices, dst_page_indices, *, physical_page_size: int,
                                  dcp_size: int, dcp_rank: int, src_page_offset: int = 0, decode_prefix_len: int = 0,
                                  num_kv_tokens: Optional[int] = None):
    """AROUND common.utils.build_dcp_token_transfer_plan: a DCP decode only pairs with this prefill through an MHA page
    pool (check_page_peer) or by stripes (check_stripe_peer), so the plan is the chunk (DcpPagePlan), which
    send_kvcache_dcp plans by pages or by stripes. send_kvcache_dcp refuses a page plan on an MLA pool without
    DCP."""
    return DcpPagePlan(DcpChunk(np.asarray(src_page_indices, dtype=np.int64), np.asarray(dst_page_indices,
                                                                                         dtype=np.int64),
                                int(physical_page_size), int(dcp_size), int(dcp_rank), int(src_page_offset or 0),
                                int(decode_prefix_len or 0), None if num_kv_tokens is None else int(num_kv_tokens),
                                dst_source=dst_page_indices))


def note_prompt_len(mgr, room, prompt_len: int) -> None:
    """Keep a request's prompt length on each of its TransferInfos, where a ring peer's chunks read the drafter
    window (ring_window)."""
    for info in list((getattr(mgr, "transfer_infos", None) or {}).get(room, {}).values()):
        info.neuron_prompt_len = int(prompt_len)


def prefill_init_kv_manager(original_fn, self, *args, **kwargs):
    """AROUND PrefillBootstrapQueue._init_kv_manager: a stage whose model keeps drafter rows of its own
    (``pd_draft_kv_pool``, the Kimi-K3 drafter writer stage) registers them as SGLang's draft entries, after the
    target's. Upstream takes the draft pool from a speculative draft worker, which a Neuron prefill does not run, so
    without this the stage registers no draft entries and sends a ring decode no drafter rows."""
    runner = getattr(getattr(self.scheduler, "tp_worker", None), "model_runner", None)
    pool = getattr(getattr(runner, "model", None), "pd_draft_kv_pool", None)
    if pool is not None:
        if self.draft_token_to_kv_pool is not None:
            raise RuntimeError("PD prefill: the model keeps drafter rows of its own and SGLang also passed a draft "
                               "pool; a prefill registers one of them")
        self.draft_token_to_kv_pool = pool
    return original_fn(self, *args, **kwargs)


def send_kv_chunk(original_fn, self, req, *args, **kwargs):
    """AROUND SchedulerDisaggregationPrefillMixin.send_kv_chunk: the request's prompt length goes with its transfer
    (note_prompt_len) before any of its chunks is queued; upstream's chunk carries only its own pages and tokens."""
    sender = getattr(req, "disagg_kv_sender", None)
    if getattr(sender, "kv_mgr", None) is not None and getattr(sender, "bootstrap_room", None) is not None:
        note_prompt_len(sender.kv_mgr, sender.bootstrap_room, len(req.origin_input_ids))
    return original_fn(self, req, *args, **kwargs)


def request_of(mgr, peer_name, chunk: DcpChunk):
    """The TransferInfo of ``chunk``'s request and peer: the one whose decode page list the worker passed."""
    for infos in list(mgr.transfer_infos.values()):
        info = infos.get(peer_name)
        if info is not None and info.dst_kv_indices is chunk.dst_source:
            return info
    return None


def ring_window(mgr, peer_name, chunk: DcpChunk, rule, block: int) -> Tuple[int, int]:
    """(the drafter window's first stripe, the request's decode state slot) of ``chunk``'s request, for a ring peer.
    Raises when the request's prompt length is not known, a chunk ends past it, or the decode's prefix holds part of
    the window (the prefill sends nothing behind the prefix)."""
    info = request_of(mgr, peer_name, chunk)
    prompt = getattr(info, "neuron_prompt_len", None)
    if info is None or prompt is None:
        raise RuntimeError(f"PD DCP stripe transfer to decode peer {peer_name}: the request's prompt length is not "
                           "known, and its ring drafter rows are the last positions of the prompt "
                           "(SchedulerDisaggregationPrefillMixin.send_kv_chunk keeps it)")
    end = chunk.decode_prefix_len + chunk.src_page_offset * int(mgr.dcp_size) * block + int(chunk.num_kv_tokens or 0)
    if end > prompt:
        raise RuntimeError(f"PD DCP stripe transfer to decode peer {peer_name}: a chunk ends at token {end}, past the "
                           f"{prompt}-token prompt")
    lo = rule.window_start(prompt, block)
    if chunk.decode_prefix_len > lo:
        raise RuntimeError(f"PD DCP stripe transfer to decode peer {peer_name}: its prefix ({chunk.decode_prefix_len} "
                           f"tokens) holds drafter window positions from {lo} on, which the prefill does not send")
    return lo // block, int(info.dst_state_indices[0][0])


def send_stripes(mgr, peer_name, dst_info, chunk: DcpChunk) -> list:
    """Post the stripes of ``chunk`` this prefill rank owes the stripe peer ``dst_info``: its target stripes to their
    owner, and its drafter stripes to their owner (sharded) or to every decode rank (replicated), each a whole page
    WRITE per KV entry through upstream's generic sender."""
    from sglang_neuron.disaggregation import dcp_stripes

    peer = dst_info.neuron_stripes
    kv = mgr.kv_args
    n_draft = int(getattr(kv, "num_draft_entries", 0) or 0)
    n_target = len(kv.kv_data_ptrs) - n_draft
    stripes = dcp_stripes.chunk_stripes(peer.pairing, int(mgr.attn_tp_rank), chunk.src_page_indices,
                                        chunk.dst_page_indices, src_page_offset=chunk.src_page_offset,
                                        decode_prefix_len=chunk.decode_prefix_len, num_kv_tokens=chunk.num_kv_tokens)
    parts = []
    if peer.roles.stripes:
        owned = stripes.to(peer.decode_rank)
        parts.append((0, n_target, peer.target_dst, owned))
        if n_draft and peer.pairing.draft_rows == dcp_stripes.DRAFT_SHARDED:
            parts.append((n_target, n_target + n_draft, peer.draft_dst, owned))
        if n_draft and peer.pairing.draft_rows == dcp_stripes.DRAFT_RING:
            block = int(peer.pairing.block)
            first, slot = ring_window(mgr, peer_name, chunk, peer.page_rule, block)
            parts.append((n_target, n_target + n_draft, peer.draft_dst,
                          stripes.ring(peer.decode_rank, peer.page_rule, slot, first)))
    if n_draft and peer.pairing.draft_rows == dcp_stripes.DRAFT_REPLICATED:
        parts.append((n_target, n_target + n_draft, peer.draft_dst, stripes.replicated()))
    handles = []
    for lo, hi, dst_entries, (src, dst) in parts:
        if not src.size or lo == hi:
            continue
        handle = mgr._send_kvcache_generic(
            peer_name=peer_name, src_data_ptrs=list(kv.kv_data_ptrs[lo:hi]),
            dst_data_ptrs=[dst_info.dst_kv_ptrs[j] for j in dst_entries], item_lens=list(kv.kv_item_lens[lo:hi]),
            prefill_data_indices=src, dst_data_indices=dst, dst_gpu_id=dst_info.gpu_id,
            src_mem_kind=mgr.src_mem_kind, dst_mem_kind=dst_info.dst_homogeneous_mem_kind, force_flat=True,
            bypass_prepped=True)
        if handle is not None:
            handles.append(handle)
    return handles


def send_kvcache_dcp(original_fn, self, peer_name, dst_info, plan, packed_src, *args, **kwargs):
    """AROUND NixlKVManager.send_kvcache_dcp: a stripe peer gets the chunk's stripes this rank owes it (send_stripes);
    a page plan goes over the page dlists of its peer (prepare_payload_xfer); a rank that owes the peer nothing of the
    chunk posts nothing."""
    if not isinstance(plan, DcpPagePlan):
        return original_fn(self, peer_name, dst_info, plan, packed_src, *args, **kwargs)
    if getattr(dst_info, "neuron_stripes", None) is not None:
        return send_stripes(self, peer_name, dst_info, plan.chunk)
    if _mla(self):
        raise RuntimeError("PD DCP: an MLA pool's chunk got a page plan; MLA pools interleave tokens")
    if plan.src_pages.size == 0:
        return []
    if dst_info.dcp_page_slices:
        return [self.send_kvcache_slice(peer_name, plan.src_pages, plan.dst_pages)]
    if dst_info.dst_homogeneous_mem_kind is None:
        raise RuntimeError("Missing NIXL destination KV memory kind")
    return [self.send_kvcache(peer_name, plan.src_pages, dst_info.dst_kv_ptrs, plan.dst_pages, dst_info.gpu_id,
                              dst_mem_kind=dst_info.dst_homogeneous_mem_kind)]


def _mamba_index(mgr) -> int:
    from sglang.srt.disaggregation.base.conn import StateType

    return list(getattr(mgr.kv_args, "state_types", None) or []).index(StateType.MAMBA)


def rule_state_sources(mgr):
    """This prefill rank's recurrent layers' state entries (dcp_stripes.StateSource): the state pool's MAMBA component
    registers each layer's conv entry, then each layer's recurrent entry (state_pool.NeuronStatePool)."""
    from sglang_neuron.disaggregation import dcp_stripes

    kv, i = mgr.kv_args, _mamba_index(mgr)
    ptrs, items, ids = kv.state_data_ptrs[i], kv.state_item_lens[i], [int(x) for x in kv.state_layer_ids[i]]
    n = len(ids) // 2
    if len(ids) != 2 * n or ids[:n] != ids[n:] or len(ptrs) != len(ids) or len(items) != len(ids):
        raise RuntimeError(f"PD DCP stripe transfer: this prefill's state entries (layers {ids}) are not one conv and "
                           "one recurrent entry per layer")
    return [dcp_stripes.StateSource(ids[k], int(ptrs[k]), int(items[k]), int(ptrs[n + k]), int(items[n + k]))
            for k in range(n)]


def rule_state_parts(peer_info, index: int) -> dict:
    """The page-rule decode's state entries by layer id: ((address of slot 0, slot stride) of each of
    dcp_stripes.STATE_PARTS)."""
    from sglang_neuron.disaggregation import dcp_stripes

    n = len(dcp_stripes.STATE_PARTS)
    ptrs, strides = peer_info.dst_state_data_ptrs[index], peer_info.dst_state_item_lens[index]
    ids = [int(x) for x in peer_info.dst_state_layer_ids[index]]
    return {ids[k]: tuple((int(ptrs[k + p]), int(strides[k + p])) for p in range(n)) for k in range(0, len(ids), n)}


def send_rule_state(mgr, peer_name, dst_info, prefill_state_indices, dst_state_indices) -> list:
    """Post the state parts of the request this prefill rank owes the page-rule decode ``dst_info``: per recurrent
    layer of this stage, each of dcp_stripes.STATE_PARTS at its entry's ``ptr + s * slot stride`` for the request's
    decode state slot ``s``, in one WRITE."""
    from sglang_neuron.disaggregation import dcp_stripes

    peer, i = dst_info.neuron_stripes, _mamba_index(mgr)
    src_slot, dst_slot = int(prefill_state_indices[i][0]), int(dst_state_indices[i][0])
    lens = list(getattr(dst_info, "dst_state_data_lens", None) or [])
    if lens and lens[i]:
        slots = min(int(n) // int(s) for n, s in zip(lens[i], dst_info.dst_state_item_lens[i])) - 1
        if not 1 <= dst_slot <= slots:
            raise RuntimeError(f"PD DCP stripe transfer to decode peer {peer_name}: state slot {dst_slot} is outside "
                               f"its slots 1 .. {slots}")
    writes = dcp_stripes.state_writes(rule_state_sources(mgr), rule_state_parts(dst_info, i),
                                      decode_rank=peer.decode_rank, ratio=peer.pairing.state_ratio, src_slot=src_slot,
                                      dst_slot=dst_slot)
    if not writes:
        return []
    src = [(addr, size, mgr.kv_args.gpu_id) for addr, _, size in writes]
    dst = [(addr, size, dst_info.gpu_id) for _, addr, size in writes]
    handle = mgr.agent.initialize_xfer("WRITE", mgr.agent.get_xfer_descs(src, "VRAM"),
                                       mgr.agent.get_xfer_descs(dst, "VRAM"), peer_name)
    if not handle:
        raise RuntimeError(f"PD DCP stripe transfer to decode peer {peer_name}: no state transfer handle")
    mgr._post_xfer(handle, f"PD DCP stripe transfer to decode peer {peer_name}: posting the state transfer failed")
    return [handle]


def maybe_send_extra(original_fn, self, peer_name, *args, **kwargs):
    """AROUND NixlKVManager.maybe_send_extra: a stripe peer takes its state heads from one prefill rank (upstream's
    slice of a prefill rank's heads for decode rank g assumes prefill rank g // (decode TP / prefill TP)); another
    prefill rank that serves it stripes sends it no state. A peer with a page rule gets its state parts by that rule
    (send_rule_state) in place of upstream's state send."""
    peer = getattr(self.decode_kv_args_table.get(peer_name), "neuron_stripes", None)
    if peer is not None and not peer.roles.state:
        return iter(())
    if peer is not None and peer.page_rule is not None:
        import inspect

        bound = inspect.signature(original_fn).bind(self, peer_name, *args, **kwargs).arguments
        return send_rule_state(self, peer_name, self.decode_kv_args_table[peer_name], bound["prefill_state_indices"],
                               bound["dst_state_indices"])
    return original_fn(self, peer_name, *args, **kwargs)


def first_contact_check():
    """Upstream's CommonKVManager.try_ensure_parallel_info, with NEURON_DCP_CHECK in place of its DCP check."""
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(source_guard.resolve(_COMMON, "CommonKVManager.try_ensure_parallel_info"),
                                           [(UPSTREAM_DCP_CHECK, NEURON_DCP_CHECK)])


def receiver_send_metadata():
    """Upstream's CommonKVReceiver.send_metadata, telling each prefill rank the count of decode ranks its bootstrap
    entry carries."""
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(source_guard.resolve(_COMMON, "CommonKVReceiver.send_metadata"),
                                           [(UPSTREAM_DST_INFO_NUM, NEURON_DST_INFO_NUM)])


def register_decode(registry, around, replace) -> None:
    """The decode side, on a DCP decode."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_DECODE_TARGETS_MD5, __name__)
    registry.register(f"{_DECODE}.DecodePreallocQueue._init_kv_manager", init_kv_manager, around)
    registry.register(f"{_NIXL}.NixlKVManager.build_receiver_registration_message", build_receiver_registration_message,
                      around)
    registry.register(f"{_COMMON}.CommonKVManager.try_ensure_parallel_info", first_contact_check(), replace)
    registry.register(f"{_COMMON}.CommonKVManager._resolve_rank_mapping", resolve_rank_mapping, around)
    registry.register(f"{_COMMON}.CommonKVReceiver._get_bootstrap_info_from_server", get_bootstrap_info_from_server,
                      around)
    registry.register(f"{_COMMON}.CommonKVReceiver.send_metadata", receiver_send_metadata(), replace)


def register_prefill(registry, around, stripes: bool = False) -> None:
    """The prefill side, on every PD prefill: a DCP decode may pair with any of them. ``stripes``: the prefill stores
    its KV in DCP placement (its model's catalog ``prefill_context_parallel``), so it also sends a DCP decode its
    stripes, drafter rows and state by the decode's page rule (send_kv_chunk, maybe_send_extra, and the drafter row
    pool's registration); every other prefill keeps upstream's paths there."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_PREFILL_TARGETS_MD5, __name__)
    registry.register(f"{_NIXL}.KVArgsRegisterInfo.from_zmq", from_zmq, around)
    registry.register(f"{_COMMON}.CommonKVManager.requires_dcp_relayout", requires_dcp_relayout, around)
    registry.register(f"{_NIXL}.NixlKVManager._add_remote_peer", add_remote_peer, around)
    registry.register(f"{_COMMON}.CommonKVManager._init_dcp_pack_buffers_once", init_dcp_pack_buffers_once, around)
    registry.register(f"{_NIXL}.NixlKVManager._prepare_payload_xfer", prepare_payload_xfer, around)
    registry.register(f"{_UTILS}.build_dcp_token_transfer_plan", build_dcp_token_transfer_plan, around)
    registry.register(f"{_NIXL}.NixlKVManager.send_kvcache_dcp", send_kvcache_dcp, around)
    if not stripes:
        return
    registry.register(f"{_NIXL}.NixlKVManager.maybe_send_extra", maybe_send_extra, around)
    registry.register(f"{_PREFILL}.SchedulerDisaggregationPrefillMixin.send_kv_chunk", send_kv_chunk, around)
    registry.register(f"{_PREFILL}.PrefillBootstrapQueue._init_kv_manager", prefill_init_kv_manager, around)
