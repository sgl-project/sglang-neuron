# SPDX-License-Identifier: Apache-2.0
"""The PD hand-off of a token context-parallel KV layout: a prefill whose model declares token CP
(``kv_layout.KVLayout(cp_size=C)``) sends each page block for block to the decode rank of its own CP rank, and a NIXL
PD prefill refuses at registration a decode whose KV layout is not its own.

Under token CP a native page of ``C x B`` tokens is one ``B``-token block on each rank of the attention-TP group, rank
``r`` holding the page's tokens ``[r x B, (r + 1) x B)`` in its block of the page's own id (mem_cache/cp_pages.CpPages).
At equal attention TP upstream SGLang pairs decode rank ``e`` with prefill attention-TP rank ``e % attn_tp``
(``CommonKVManager._resolve_rank_mapping``), and the pair's transfer WRITEs page ``p`` of every KV entry
(``send_kvcache``, ``item_len`` bytes at ``ptr + p * item_len``) and every DSA index-K state entry
(``maybe_send_extra``) into the decode's page ``q``. When both roles split pages the same way (the same CP degree and
block) and the paired ranks hold the same CP rank, that per-rank page copy is exactly ``cp_pages.latent_page_plan`` at
``Cp = Cd``: every block reaches the rank that owns it, in the decode's page of the same ordinal. That is the
hand-off of an MLA model under token CP: the latent and the owner-sharded index keys of every DSA layer, by the pages
upstream already sends; the recurrent state travels as its equal-TP slots (disaggregation/state_transfer).

Upstream's registration carries none of this (no CP degree, block or rank of the KV pool), so it would WRITE a CP
prefill's blocks into a decode pool that holds whole pages, or a block of other tokens into a rank that reads its own,
silently. Here:

* the decode appends its layout, ``(cp_size, cp_rank, block)`` of the model's declared KV layout, to its registration as
  a trailing frame (``NixlKVManager.build_receiver_registration_message``, wrapped; ``CP_LAYOUT_FRAME``);
* the prefill reads it (``KVArgsRegisterInfo.from_zmq``, wrapped; ``dst_kv_cp_layout``, None when the frame is absent)
  and, in ``NixlKVManager._add_remote_peer`` (wrapped), refuses the decode rank unless both layouts are the same and
  the ranks hold the same CP rank, whenever either side runs token CP (``cp_layout_problem``). A pairing of two CP
  layouts also needs every paired KV entry and DSA index-K entry to hold pages of one size on both roles
  (``page_size_problem``): an owner-sharded index pool holds one block of a page, a replicated one the whole page.

The layout is the process's declaration (``model_executor/kv_layout.current()``: the model declares it during
construction, before SGLang builds the pool, which reads it), so neither the frame nor the check depends on state set
after a manager is built. A refused decode becomes a refused peer whose rooms fail with the reason
(sglang_patches/nixl_peer_refusal). Pairings without token CP on either side are upstream's, unchanged; a DCP decode
publishes no token-CP layout (its pages are the DCP page plan's, sglang_patches/dcp_page_transfer), so a CP prefill
refuses it.

One asymmetric pairing is served beyond equal layouts: a whole-page prefill into a decode whose layout holds every
KV head of its block of each page (the Qwen3.5 kv_cp decode; the frame then carries the head count as a fourth
field, ``heads``). ``scatter_pair`` admits it and sglang_patches/kv_cp_scatter moves the decode rank's block of every
page into its head columns of the decode page; the geometry the transfer relies on is checked at peer registration
(``kv_cp_scatter.geometry_problem``). The frame's fifth field carries the K page layout (token-major vs the
dim-major transpose): a hand-off needs one K layout on both roles — the whole-page transfer is layout-blind, and
the scatter's K block is one contiguous run only between two token-major pools — so a mismatch refuses the peer.

Not served: a CP prefill into a decode of another CP degree or block (a CP16 prefill into a decode that holds whole
pages, the reverse, or another degree). ``cp_pages.latent_page_plan`` maps those blocks, but no transfer sends that plan.

Upstream: generic: a KV pool's context-parallel layout in the PD registration and the refusal of a decode at another.

Targets, each checked against upstream SGLang 43881a7: ``NixlKVManager.build_receiver_registration_message``,
``KVArgsRegisterInfo.from_zmq`` and ``NixlKVManager._add_remote_peer`` (wrapped), and the sources the hand-off relies on
(their reasons at ``_TARGETS_MD5``)."""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

_COMMON = "sglang.srt.disaggregation.common.conn"
_NIXL = "sglang.srt.disaggregation.nixl.conn"
_UTILS = "sglang.srt.disaggregation.utils"
# a decode's KV layout, b"<tag><cp_size>,<cp_rank>,<block>", after upstream's registration frames
CP_LAYOUT_FRAME = b"sglang-neuron:kv-cp-layout:"
# KVArgsRegisterInfo.from_zmq reads upstream's frames msg[0] to msg[24]; the plugin's frames trail them
_UPSTREAM_REGISTRATION_FRAMES = 25

_WRAPPED = {
    (_NIXL, "NixlKVManager.build_receiver_registration_message"): "1d28c9d9c8345a71db02d6840a7a3acc",
    (_NIXL, "KVArgsRegisterInfo.from_zmq"): "888b10e09310b5af69364bbf6ffbdfab",
    (_NIXL, "NixlKVManager._add_remote_peer"): "e2ee9f36cd7849d884c49275e4c10bde",
}
# (module, qualified name): md5 of the source
_TARGETS_MD5 = {
    **_WRAPPED,
    # at equal attention TP decode rank e pulls from prefill attention-TP rank e % attn_tp, one real source per rank
    (_COMMON, "CommonKVManager._resolve_rank_mapping"): "70d83eef6dcb74b7cce329808eecba2f",
    # page p of every KV entry, item_len bytes at ptr + p * item_len, into the decode's page q (the flat MLA path)
    (_NIXL, "NixlKVManager.send_kvcache"): "dc8b5f9e6114739fce8c623ba6bd6428",
    # a DSA index-K component is sent page for page like the KV, at the last chunk
    (_NIXL, "NixlKVManager.maybe_send_extra"): "a0419a7cd7d96bc4a2b3f9e61c948c92",
    # hands each registration to from_zmq and _add_remote_peer; a refusal raised there refuses the peer
    (_NIXL, "NixlKVManager._start_bootstrap_thread"): "0ad34e79d2d77da1b2cb45c799e94748",
    # the KV args a hybrid DSA pool registers: the latent as KV, the index keys as a DSA state component
    (_UTILS, "setup_state_kv_args"): "c09eb7e917545ca1d45e53b58ecdee67",
}


class CpLayout(NamedTuple):
    """A rank's KV pages: ``cp_size`` blocks of ``block`` tokens per page, this rank holding block ``cp_rank``."""

    cp_size: int
    cp_rank: int
    block: int
    # the KV heads the rank holds of its block (a decode whose attention heads slice across the CP group, the
    # Qwen3.5 kv_cp decode, holds every head of its block); 0: the heads split as upstream's (the equal-layout hand-off)
    heads: int = 0
    # the pool's K pages are token-major ([block, head_dim], like V), not the dim-major transpose (the Qwen3.5
    # kv_cp decode and the prefill serving its scatter, models/qwen35/config.KV_TM_ENV); 0: dim-major. A PD
    # hand-off needs one K layout on both roles: the whole-page transfer is layout-blind, and the scatter's
    # contiguous K block exists only between two token-major pools.
    k_tm: int = 0

    @property
    def is_cp(self) -> bool:
        return self.cp_size > 1

    def describe(self) -> str:
        if not self.is_cp:
            return f"whole {self.block}-token pages (no token CP)"
        heads = f", all {self.heads} KV heads of the block" if self.heads else ""
        return (f"token CP {self.cp_size} in {self.block}-token blocks ({self.cp_size * self.block}-token pages), CP "
                f"rank {self.cp_rank}{heads}")


def own_layout(page_size: int) -> CpLayout:
    """This process's KV layout at SGLang page ``page_size``: the model's declared token CP, or whole pages."""
    from sglang_neuron.model_executor import kv_layout

    layout = kv_layout.current()
    k_tm = 1 if getattr(layout, "k_token_major", False) else 0
    if int(getattr(layout, "cp_size", 1)) <= 1:
        # a KVLayout without token-CP fields (no CP model declared one) holds whole pages
        return CpLayout(1, 0, int(page_size), 0, k_tm)
    return CpLayout(int(layout.cp_size), int(layout.cp_rank), int(layout.physical_block_size(int(page_size))),
                    int(getattr(layout, "kv_heads_per_rank", None) or 0), k_tm)


def encode(layout: CpLayout) -> bytes:
    # three fields when the heads split as upstream's and K is dim-major, so a peer of the older plugin reads the
    # same layout
    fields = f"{layout.cp_size},{layout.cp_rank},{layout.block}"
    if layout.heads or layout.k_tm:
        fields += f",{layout.heads}"
    if layout.k_tm:
        fields += ",1"
    return CP_LAYOUT_FRAME + fields.encode("ascii")


def decode(frame: bytes) -> CpLayout:
    fields = frame[len(CP_LAYOUT_FRAME):].decode("ascii").split(",")
    return CpLayout(int(fields[0]), int(fields[1]), int(fields[2]),
                    int(fields[3]) if len(fields) > 3 else 0,
                    int(fields[4]) if len(fields) > 4 else 0)


def scatter_pair(own: CpLayout, decode_layout: Optional[CpLayout]) -> bool:
    """A whole-page prefill rank and a kv_cp decode rank that holds every KV head of its block of each page (the
    Qwen3.5 DCP decode): the scatter transfer (sglang_patches/kv_cp_scatter) sends the decode rank's block of every
    page into its head columns of the decode page. Only this asymmetric pairing is served; its geometry the scatter
    checks at peer registration."""
    return (not own.is_cp and decode_layout is not None and decode_layout.is_cp and decode_layout.heads > 0
            and own.block == decode_layout.cp_size * decode_layout.block)


def cp_layout_problem(own: CpLayout, decode_layout: Optional[CpLayout]) -> Optional[str]:
    """Why this prefill rank's pages cannot go page for page into a decode rank of ``decode_layout``, or None."""
    if decode_layout is None:
        if not own.is_cp:
            if own.k_tm:
                return ("this prefill holds token-major K pages and the decode published no KV layout: the "
                        "whole-page transfer is layout-blind and would land them in a dim-major pool — a "
                        "token-major prefill serves the kv_cp scatter decode only")
            return None  # a decode without the frame: upstream's pairing of whole pages
        return (f"this prefill holds {own.describe()} and the decode published no KV layout; a token-CP hand-off sends "
                "each block to the decode rank of the same CP rank, which needs the decode's layout")
    if not own.is_cp and not decode_layout.is_cp:
        if own.k_tm != decode_layout.k_tm:
            return (f"this prefill's K pages are {'token-major' if own.k_tm else 'dim-major'} and the decode's are "
                    f"{'token-major' if decode_layout.k_tm else 'dim-major'}: the whole-page transfer is "
                    "layout-blind and needs one K layout on both roles")
        return None
    if scatter_pair(own, decode_layout):
        if own.k_tm != decode_layout.k_tm:
            return (f"the kv_cp scatter's K block of a page is one contiguous run only between two token-major K "
                    f"pools; this prefill's K is {'token-major' if own.k_tm else 'dim-major'} and the decode's is "
                    f"{'token-major' if decode_layout.k_tm else 'dim-major'}")
        return None  # the scatter transfer sends the decode rank its block of each page, not a page for a page
    if (own.cp_size, own.block) != (decode_layout.cp_size, decode_layout.block):
        return (f"this prefill holds {own.describe()} and the decode {decode_layout.describe()}; a token-CP hand-off "
                "moves each page block for block to the decode rank of the same CP rank, which needs one layout on "
                "both roles (mem_cache/cp_pages.latent_page_plan maps other degrees, but no transfer sends it)")
    if own.cp_rank != decode_layout.cp_rank:
        return (f"this prefill rank holds CP rank {own.cp_rank} of each page and its decode peer CP rank "
                f"{decode_layout.cp_rank}: the blocks it would write hold another rank's tokens")
    return None


def _pairs(src_ids, dst_ids, n_src: int, n_dst: int, pp_size: int):
    from sglang.srt.disaggregation.utils import build_transfer_entry_pairs

    try:
        return build_transfer_entry_pairs(list(src_ids or []), list(dst_ids or []), n_src, n_dst,
                                          allow_positional_fallback=pp_size == 1)
    except RuntimeError:
        return None  # upstream refuses the pairing itself


def page_size_problem(kv_args, peer, *, pp_size: int) -> Optional[str]:
    """Why a token-CP pair's paired KV and DSA index-K entries do not hold pages of one size on both roles, or None:
    upstream WRITEs with the prefill's page stride into the decode's page grid."""
    problems = []
    src_lens = list(kv_args.kv_item_lens or [])
    dst_lens = list(peer.dst_kv_item_lens or [])
    pairs = _pairs(getattr(kv_args, "kv_layer_ids", None), peer.dst_kv_layer_ids, len(src_lens), len(dst_lens),
                   pp_size)
    bad = [(i, src_lens[i], dst_lens[j]) for i, j in pairs or [] if int(src_lens[i]) != int(dst_lens[j])]
    if bad:
        problems.append("KV entries " + ", ".join(f"{i}: {a} != {b} bytes" for i, a, b in bad[:4]))
    for c, state_type in enumerate(getattr(kv_args, "state_types", None) or []):
        if str(getattr(state_type, "value", state_type)) != "dsa":
            continue
        src = list(_at(kv_args.state_item_lens, c))
        dst = list(_at(peer.dst_state_item_lens, c))
        if not src or not dst:
            continue
        pairs = _pairs(_at(kv_args.state_layer_ids, c), _at(peer.dst_state_layer_ids, c), len(src), len(dst), pp_size)
        bad = [(i, src[i], dst[j]) for i, j in pairs or [] if int(src[i]) != int(dst[j])]
        if bad:
            problems.append(f"DSA index-K component {c} entries " +
                            ", ".join(f"{i}: {a} != {b} bytes" for i, a, b in bad[:4]) +
                            " (an owner-sharded index pool holds one block of a page, a replicated one the page)")
    if not problems:
        return None
    return "the roles' pages differ in size: " + "; ".join(problems)


def _at(lists: Optional[Sequence], i: int) -> Sequence:
    return lists[i] if lists is not None and i < len(lists) and lists[i] is not None else ()


# The decode
def build_receiver_registration_message(original_fn, self, *args, **kwargs):
    """AROUND NixlKVManager.build_receiver_registration_message: the decode's KV layout, in a trailing frame."""
    message = list(original_fn(self, *args, **kwargs))
    message.append(encode(own_layout(int(self.kv_args.page_size))))
    return message


# The prefill
def from_zmq(original_fn, cls, msg, *args, **kwargs):
    """AROUND KVArgsRegisterInfo.from_zmq: the decode's KV layout (None without the frame), read from its frame, which
    upstream's parser does not see."""
    frames = list(msg)
    index = next((i for i in range(_UPSTREAM_REGISTRATION_FRAMES, len(frames))
                  if bytes(frames[i]).startswith(CP_LAYOUT_FRAME)), None)
    layout = decode(bytes(frames.pop(index))) if index is not None else None
    info = original_fn(cls, frames, *args, **kwargs)
    info.dst_kv_cp_layout = layout
    return info


def add_remote_peer(original_fn, self, decode_kv_args, *args, **kwargs):
    """AROUND NixlKVManager._add_remote_peer: a decode rank whose KV layout is not this prefill rank's (when either runs
    token CP) is refused before it is registered."""
    own = own_layout(int(self.kv_args.page_size))
    peer_layout = getattr(decode_kv_args, "dst_kv_cp_layout", None)
    problem = cp_layout_problem(own, peer_layout)
    if problem is None and scatter_pair(own, peer_layout):
        from sglang_neuron.sglang_patches import kv_cp_scatter

        problem = kv_cp_scatter.geometry_problem(self, decode_kv_args, peer_layout)
    elif problem is None and (own.is_cp or (peer_layout is not None and peer_layout.is_cp)):
        problem = page_size_problem(self.kv_args, decode_kv_args, pp_size=int(getattr(self, "pp_size", 1) or 1))
    if problem is not None:
        # upstream registers it and WRITEs this rank's blocks into pages that hold other tokens: refused here
        raise RuntimeError(f"NIXL PD transfer to decode peer {decode_kv_args.agent_name}: {problem}; launch both roles "
                           "of a token-CP model with its declared KV layout at one attention TP")
    return original_fn(self, decode_kv_args, *args, **kwargs)


_HOOKS = {
    "NixlKVManager.build_receiver_registration_message": build_receiver_registration_message,
    "KVArgsRegisterInfo.from_zmq": from_zmq,
    "NixlKVManager._add_remote_peer": add_remote_peer,
}


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    for module, qualname in _WRAPPED:
        registry.register(f"{module}.{qualname}", _HOOKS[qualname], around)
