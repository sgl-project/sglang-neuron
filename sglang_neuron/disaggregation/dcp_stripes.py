# SPDX-License-Identifier: Apache-2.0
"""The KV hand-off from a DCP prefill to a decode of a larger DCP size, as host metadata.

Both roles keep their latent (MLA) rows block-cyclic over their DCP group in stripes of one pool page
(mem_cache/cp_pages.CpPages, distributed/decode_context_parallel.DcpLayout): global stripe ``j`` of a request lives on
prefill rank ``j % Cp`` at its local page ``src_pages[j // Cp]`` and goes whole to decode rank ``j % Cd`` at
``dst_pages[j // Cd]``. Here each side's DCP group is its whole attention-TP group, and ``Cd`` is a multiple of
``Cp``. The recurrent state stays sharded by attention heads, which SGLang slices between attention TP sizes as it
pairs ranks: decode rank ``g`` takes its heads from prefill rank ``g // (Cd / Cp)``. So decode rank ``g`` pairs with
prefill ranks ``{g % Cp, g // (Cd / Cp)}``, and prefill rank ``r`` serves decode ranks ``{r, r + Cp, ...}`` for
stripes and ``{r k, ..., r k + k - 1}`` (``k = Cd / Cp``) for state.

Drafter context rows follow the target rows' stripes on the decode (``sharded``), or the decode keeps every stripe on
every rank, at block ``j % Cd`` of its ``Cd``-block virtual page (``replicated``), or the decode keeps only the drafter
window, sharded like the target rows, in the request's ring of rider pages its page rule names (``ring``, PageRule);
the prefill always holds them sharded. With replicated rows every decode rank pairs with every prefill rank. The decode
announces the form."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import numpy as np

DRAFT_SHARDED = "sharded"
DRAFT_REPLICATED = "replicated"
DRAFT_RING = "ring"
DRAFT_ROWS = (DRAFT_SHARDED, DRAFT_REPLICATED, DRAFT_RING)
# The drafter-row form a decode announces. It must name the form of the decode's drafter pool.
DECODE_DRAFT_ROWS = DRAFT_SHARDED


class Roles(NamedTuple):
    """What one prefill rank sends one decode rank: its target stripes, its state heads, its drafter stripes."""

    stripes: bool
    state: bool
    draft: bool

    @property
    def any(self) -> bool:
        return self.stripes or self.state or self.draft


@dataclass(frozen=True)
class StripePairing:
    """The rank pairing of a ``prefill_dcp``-rank DCP prefill and a ``decode_dcp``-rank DCP decode over
    ``block``-token stripes; ``state``: the pools carry attention-TP-sharded state."""

    prefill_dcp: int
    decode_dcp: int
    prefill_tp: int
    decode_tp: int
    block: int
    draft_rows: str = DRAFT_SHARDED
    state: bool = True

    def problem(self) -> Optional[str]:
        """Why the stripe map does not serve this pairing, or None."""
        cp, cd = self.prefill_dcp, self.decode_dcp
        if cp < 2:
            return f"the prefill has no DCP (--dcp-size {cp})"
        if cp != self.prefill_tp:
            return (f"the stripe map needs the prefill's DCP over its whole attention TP (--dcp-size {self.prefill_tp} "
                    f"with attention TP {self.prefill_tp}); the prefill runs --dcp-size {cp}")
        if cd != self.decode_tp:
            return (f"the stripe map needs the decode's DCP over its whole attention TP (--dcp-size {self.decode_tp} "
                    f"with attention TP {self.decode_tp}); the decode runs --dcp-size {cd}")
        if cd <= cp or cd % cp:
            return f"the decode's DCP size must be a larger multiple of the prefill's {cp}, not {cd}"
        if self.block < 1:
            return f"the stripe is {self.block} tokens"
        if self.draft_rows not in DRAFT_ROWS:
            return f"drafter rows {self.draft_rows!r} are not one of {DRAFT_ROWS}"
        return None

    @property
    def state_ratio(self) -> int:
        """Decode ranks per prefill rank's state heads."""
        return self.decode_tp // self.prefill_tp

    def stripe_source(self, decode_rank: int) -> int:
        return decode_rank % self.prefill_dcp

    def state_source(self, decode_rank: int) -> int:
        return decode_rank // self.state_ratio

    def prefill_ranks(self, decode_rank: int) -> Tuple[int, ...]:
        """The prefill ranks decode rank ``decode_rank`` pairs with, in order."""
        if self.draft_rows == DRAFT_REPLICATED:
            return tuple(range(self.prefill_tp))
        ranks = {self.stripe_source(decode_rank)}
        if self.state:
            ranks.add(self.state_source(decode_rank))
        return tuple(sorted(ranks))

    def decode_peers(self, prefill_rank: int) -> Tuple[int, ...]:
        """The decode ranks prefill rank ``prefill_rank`` serves, in order."""
        return tuple(g for g in range(self.decode_tp) if prefill_rank in self.prefill_ranks(g))

    def roles(self, prefill_rank: int, decode_rank: int) -> Roles:
        stripes = self.stripe_source(decode_rank) == prefill_rank
        state = self.state and self.state_source(decode_rank) == prefill_rank
        return Roles(stripes, state, True if self.draft_rows == DRAFT_REPLICATED else stripes)


@dataclass(frozen=True)
class ChunkStripes:
    """The stripes of one transfer chunk one prefill rank holds: global stripe ids, the local page of each on the
    prefill rank, and the decode page each lands in (on its owner ``stripe % Cd``, or on every decode rank for
    replicated rows)."""

    stripes: np.ndarray
    src_pages: np.ndarray
    dst_pages: np.ndarray
    decode_dcp: int

    def to(self, decode_rank: int) -> Tuple[np.ndarray, np.ndarray]:
        """(source pages, destination pages) of the stripes decode rank ``decode_rank`` owns."""
        mine = self.stripes % self.decode_dcp == decode_rank
        return self.src_pages[mine], self.dst_pages[mine]

    def ring(self, decode_rank: int, rule: "PageRule", slot: int, first: int) -> Tuple[np.ndarray, np.ndarray]:
        """(source pages, rider kernel pages) of the stripes decode rank ``decode_rank`` owns from stripe ``first`` on
        (the drafter window's first stripe), in state slot ``slot``'s ring."""
        mine = (self.stripes % self.decode_dcp == decode_rank) & (self.stripes >= first)
        return self.src_pages[mine], rule.ring_pages(slot, self.stripes[mine], self.decode_dcp)

    def replicated(self) -> Tuple[np.ndarray, np.ndarray]:
        """(source pages, destination blocks) of every stripe into a replicated pool: block ``j % Cd`` of decode page
        ``p`` is block ``p * Cd + j % Cd`` of the pool, in units of one stripe."""
        return self.src_pages, self.dst_pages * self.decode_dcp + self.stripes % self.decode_dcp


def chunk_stripes(pairing: StripePairing, prefill_rank: int, src_pages, dst_pages, *, src_page_offset: int = 0,
                  decode_prefix_len: int = 0, num_kv_tokens: Optional[int] = None) -> ChunkStripes:
    """The stripes prefill rank ``prefill_rank`` holds of one chunk. ``src_pages``: the chunk's allocator pages
    (``Cp`` stripes each), which start ``src_page_offset`` pages after the decode prefix; ``dst_pages``: the decode's
    pages after its prefix; ``num_kv_tokens``: the chunk's tokens (a partial last stripe moves whole).

    The decode prefix (its radix-cache hit) must be whole decode allocator pages (``Cd`` stripes), or its first stripe
    would land on the wrong rank."""
    cp, cd, block = pairing.prefill_dcp, pairing.decode_dcp, pairing.block
    if not 0 <= prefill_rank < cp:
        raise ValueError(f"prefill rank {prefill_rank} is outside the {cp}-rank DCP group")
    if decode_prefix_len < 0 or decode_prefix_len % (cd * block):
        raise ValueError(f"PD DCP stripe transfer: the decode prefix ({decode_prefix_len} tokens) must be whole decode "
                         f"pages of {cd} x {block} tokens")
    src = np.asarray(src_pages, dtype=np.int64).reshape(-1)
    dst = np.asarray(dst_pages, dtype=np.int64).reshape(-1)
    tokens = src.size * cp * block if num_kv_tokens is None else int(num_kv_tokens)
    if tokens < 0 or -(-tokens // (cp * block)) > src.size:
        raise ValueError(f"PD DCP stripe transfer: a chunk of {tokens} tokens does not fit its {src.size} prefill "
                         f"pages of {cp} x {block} tokens")
    first = decode_prefix_len // block + src_page_offset * cp  # a multiple of Cp: the chunk starts on a page
    offsets = np.arange(prefill_rank, -(-tokens // block), cp, dtype=np.int64)
    stripes = first + offsets
    dst_index = stripes // cd - decode_prefix_len // (cd * block)
    if dst_index.size and int(dst_index.max()) >= dst.size:
        raise ValueError(f"PD DCP stripe transfer: insufficient decode pages: stripe {int(stripes.max())} needs page "
                         f"{int(dst_index.max())} of {dst.size} after the decode prefix")
    return ChunkStripes(stripes, src[offsets // cp], dst[dst_index], cd)


@dataclass(frozen=True)
class DecodeFrame:
    """What a decode announces of its KV placement in its registration: the run of consecutive tokens a DCP rank
    keeps, its DCP size and rank, page size and KV dtype, whether it pairs by stripes with a DCP prefill, and its
    drafter-row form."""

    run: int
    dcp_size: int = 1
    dcp_rank: int = 0
    page_size: int = 0
    kv_dtype: str = ""
    stripes: bool = False
    draft_rows: str = DRAFT_SHARDED

    def encode(self) -> bytes:
        fields = [str(self.run), f"dcp={self.dcp_size}/{self.dcp_rank}", f"page={self.page_size}",
                  f"kv={self.kv_dtype}", f"pairing={'stripes' if self.stripes else 'tp'}", f"draft={self.draft_rows}"]
        return ";".join(fields).encode("ascii")

    @classmethod
    def decode(cls, payload: bytes) -> "DecodeFrame":
        """A frame's payload; a bare run is a decode that announces nothing else."""
        run, *rest = bytes(payload).decode("ascii").split(";")
        fields = dict(field.split("=", 1) for field in rest)
        if not fields:
            return cls(int(run))
        size, rank = fields["dcp"].split("/")
        return cls(int(run), int(size), int(rank), int(fields["page"]), fields["kv"], fields["pairing"] == "stripes",
                   fields["draft"])


STATE_PARTS = ("conv_q", "conv_k", "conv_v", "recurrent")
"""The parts of one recurrent layer's state a page-rule decode keeps, in its entry order: the q, k and v conv
channels of its heads, then the recurrent state of its heads."""
RULE_FIELDS = ("kda_start", "slot_pages", "group_pages", "page", "ring", "window", "state")
STATE_GROUPS = 3
"""Recurrent-state groups of a page-rule decode: recurrent layer ``j`` (in its layer order) is in group ``j % 3``."""


@dataclass(frozen=True)
class PageRule:
    """A decode's per-request pages as a function of the request's state slot ``s`` (1-based), from its registration's
    page frame: recurrent layer ``j`` (group ``j % 3``) at host page ``kda_start + (s - 1) slot_pages + (j % 3)
    group_pages`` of the buffer of its decode MLA entry ``j // 3``, its parts (``state``: (name, byte offset, bytes))
    in that page; drafter position ``p`` on decode rank ``(p // block) % Cd``, rider host page ``1 + (s - 1) ring +
    (p // (2 Cd block)) % ring``, kernel half ``(p // (Cd block)) % 2``, for the last ``window`` positions rounded down
    to a whole stripe. ``page``: bytes of a host page (two stripes' pages)."""

    kda_start: int
    slot_pages: int
    group_pages: int
    page: int
    ring: int
    window: int
    state: Tuple[Tuple[str, int, int], ...]

    @classmethod
    def of(cls, fields: dict) -> "PageRule":
        """The rule of a parsed page frame; raises a ValueError on missing or unknown fields."""
        names = set(fields)
        if names != set(RULE_FIELDS):
            raise ValueError(f"page rule fields {sorted(names)} are not {sorted(RULE_FIELDS)}")
        state = tuple((str(n), int(o), int(b)) for n, o, b in fields["state"])
        return cls(*(int(fields[k]) for k in RULE_FIELDS[:-1]), state)

    def window_start(self, prompt_len: int, block: int) -> int:
        """The first drafter position sent of a ``prompt_len``-token prompt: the window's start, rounded down to a
        whole stripe."""
        return max(0, int(prompt_len) - self.window) // block * block

    def ring_pages(self, slot: int, stripes, decode_dcp: int) -> np.ndarray:
        """The rider kernel page (one stripe each) of each of ``stripes`` in state slot ``slot``'s ring, on the stripe's
        owner rank."""
        j = np.asarray(stripes, dtype=np.int64)
        return 2 * (1 + (int(slot) - 1) * self.ring + (j // (2 * decode_dcp)) % self.ring) + (j // decode_dcp) % 2

    def frame(self) -> bytes:
        """The registration frame a decode announces this rule with (``PAGE_RULE_FRAME`` and ``key=value`` fields in
        ``RULE_FIELDS`` order, ``state`` as ``name@offset+bytes`` parts); ``parse_page_frame`` reads it back.

        Example:
            >>> rule = PageRule(1500, 15, 5, 147456, 7, 49152, (("conv_q", 0, 2304),))
            >>> PageRule.of(parse_page_frame(rule.frame())) == rule
            True
        """
        fields = [f"{name}={int(getattr(self, name))}" for name in RULE_FIELDS[:-1]]
        fields.append("state=" + ",".join(f"{n}@{int(o)}+{int(b)}" for n, o, b in self.state))
        return PAGE_RULE_FRAME + ";".join(fields).encode("ascii")


PAGE_RULE_FRAME = b"sglang-neuron:k3-decode-pages:"
"""The registration frame of a decode whose per-request pages are a function of the request's state slot
(``PageRule.frame``); the prefill's stripe sender reads it with ``parse_page_frame``."""


def parse_page_frame(payload: bytes) -> dict:
    """The fields of a page frame (``PageRule.frame``): ints, and ``state`` as (name, byte offset, bytes) parts. Raises
    a ValueError (or UnicodeDecodeError) on a payload that is not one."""
    text = bytes(payload).decode("ascii")
    prefix = PAGE_RULE_FRAME.decode("ascii")
    if not text.startswith(prefix):
        raise ValueError(f"not a page frame: {text[:40]!r}")
    fields = dict(item.split("=", 1) for item in text[len(prefix):].split(";"))
    if "state" not in fields:
        raise ValueError("a page frame without its state parts")
    out = {k: int(v) for k, v in fields.items() if k != "state"}
    out["state"] = tuple((n, int(o), int(b)) for n, rest in (p.split("@") for p in fields["state"].split(","))
                         for o, b in [rest.split("+")])
    return out


@dataclass(frozen=True)
class StateSource:
    """One recurrent layer's state entries on the prefill: the conv entry ``[3 x channels, taps]`` channel-major
    (q, k, v channels) and the recurrent entry ``[heads, ...]``, each at ``ptr + slot * item`` for a slot."""

    layer_id: int
    conv_ptr: int
    conv_item: int
    recurrent_ptr: int
    recurrent_item: int


def state_part_bytes(conv_item: int, recurrent_item: int, ratio: int) -> Tuple[int, ...]:
    """Bytes of each of STATE_PARTS a decode rank takes from one prefill rank's slot of one layer, when ``ratio``
    decode ranks share that prefill rank's heads."""
    if conv_item % (3 * ratio) or recurrent_item % ratio:
        raise ValueError(f"a conv slot of {conv_item} B and a recurrent slot of {recurrent_item} B do not split "
                         f"into 3 parts x {ratio} decode ranks")
    return (conv_item // (3 * ratio),) * 3 + (recurrent_item // ratio,)


def state_writes(sources, dst_parts, *, decode_rank: int, ratio: int, src_slot: int, dst_slot: int):
    """(source address, destination address, bytes) of every state part decode rank ``decode_rank`` takes from this
    prefill rank's ``sources`` (StateSource per layer), into ``dst_parts[layer] = ((ptr, slot stride), ...)`` (one per
    STATE_PARTS) at slot ``dst_slot``. Of conv part ``j`` the rank takes channels ``j C + h C / ratio .. + C /
    ratio`` (``C`` channels a part, ``h = decode_rank % ratio``), of the recurrent state heads ``h H / ratio .. + H /
    ratio``: one contiguous range of the channel-major slot each."""
    h = int(decode_rank) % ratio
    out = []
    for src in sources:
        parts = dst_parts[src.layer_id]
        sizes = state_part_bytes(src.conv_item, src.recurrent_item, ratio)
        conv0 = src.conv_ptr + int(src_slot) * src.conv_item
        offsets = [conv0 + j * (src.conv_item // 3) + h * sizes[j] for j in range(3)]
        offsets.append(src.recurrent_ptr + int(src_slot) * src.recurrent_item + h * sizes[3])
        for (ptr, stride), addr, size in zip(parts, offsets, sizes):
            out.append((addr, int(ptr) + int(dst_slot) * int(stride), size))
    return out


def page_rule_problem(rule: PageRule, *, decode_dcp: int, ratio: int, block: int, stripe_bytes: int, sources,
                      dst_kv_ptrs, dst_state_ptrs, dst_state_strides, dst_state_layer_ids) -> Optional[str]:
    """Why this prefill rank cannot place by ``rule`` what the decode registered, or None. The rule must give this
    prefill's state parts (``sources``: StateSource per layer of the stage) in STATE_PARTS order, contiguous from byte
    0 of a host page of two stripes' pages, 3 groups of ``group_pages``, and a ring that holds the window and one more
    stripe; and every state entry the decode registered (four a layer, in its layer order ``j``) must sit at slot 0 of
    the page the rule names in the buffer of its MLA entry ``j // 3``, with the rule's slot stride."""
    names = tuple(name for name, _, _ in rule.state)
    if names != STATE_PARTS:
        return f"its state parts are {list(names)}, not {list(STATE_PARTS)}"
    if rule.page != 2 * stripe_bytes:
        return f"its host page holds {rule.page} bytes, and two stripes' pages {2 * stripe_bytes}"
    sizes = {state_part_bytes(s.conv_item, s.recurrent_item, ratio) for s in sources}
    if len(sizes) > 1:
        return f"this prefill's layers hold state slots of different bytes ({sorted(sizes)})"
    offset, laid = 0, []
    for size in (sizes.pop() if sizes else tuple(b for _, _, b in rule.state)):
        laid.append((offset, size))
        offset += size
    if [(o, b) for _, o, b in rule.state] != laid or offset > rule.page:
        return (f"its state parts {[(n, o, b) for n, o, b in rule.state]} are not this prefill's, contiguous from byte "
                f"0 of a page: {[(n, o, b) for n, (o, b) in zip(STATE_PARTS, laid)]}")
    if rule.group_pages < 1 or rule.slot_pages != STATE_GROUPS * rule.group_pages or rule.kda_start < 1:
        return (f"its state slot of {rule.slot_pages} pages is not {STATE_GROUPS} groups of {rule.group_pages} "
                f"pages from page {rule.kda_start} >= 1")
    if rule.ring < 1 or rule.window < 1 or rule.window % block:
        return f"its ring of {rule.ring} pages and window of {rule.window} tokens are not whole stripes"
    if rule.ring * 2 * decode_dcp * block < rule.window + block:
        return (f"its ring of {rule.ring} pages holds {rule.ring * 2 * decode_dcp * block} tokens, less than the "
                f"window and one stripe ({rule.window + block})")
    n = len(STATE_PARTS)
    ids = [int(i) for i in dst_state_layer_ids]
    if len(ids) % n or any(ids[k:k + n] != [ids[k]] * n for k in range(0, len(ids), n)) or \
            len(dst_state_ptrs) != len(ids) or len(dst_state_strides) != len(ids):
        return f"its state entries (layers {ids}) are not {n} parts a layer"
    layers = ids[::n]
    missing = [s.layer_id for s in sources if s.layer_id not in layers]
    if missing:
        return f"it registers no state of this prefill's layers {missing}"
    stride = rule.slot_pages * rule.page
    for j, layer in enumerate(layers):
        ptrs, strides = dst_state_ptrs[j * n:(j + 1) * n], dst_state_strides[j * n:(j + 1) * n]
        if j // STATE_GROUPS >= len(dst_kv_ptrs):
            return f"its state layer {layer} (state layer {j}) has no MLA entry {j // STATE_GROUPS} buffer"
        base = int(dst_kv_ptrs[j // STATE_GROUPS]) + (rule.kda_start - rule.slot_pages + (j % STATE_GROUPS)
                                                       * rule.group_pages) * rule.page
        if [int(p) - base for p in ptrs] != [o for _, o, _ in rule.state] or {int(s) for s in strides} != {stride}:
            return (f"its state entries of layer {layer} sit at offsets {[int(p) - base for p in ptrs]} with strides "
                    f"{sorted({int(s) for s in strides})} from where its page rule puts state layer {j}: offsets "
                    f"{[o for _, o, _ in rule.state]}, stride {stride}")
    return None


@dataclass(frozen=True)
class StripePeer:
    """A decode rank as one prefill rank serves it: the pairing, the decode rank, what this prefill rank sends it,
    the decode's entry index of each target and drafter entry, and the decode's page rule (PageRule) when it announced
    one."""

    pairing: StripePairing
    decode_rank: int
    roles: Roles
    target_dst: Tuple[int, ...]
    draft_dst: Tuple[int, ...]
    page_rule: Optional[PageRule] = None


def stripe_peer(*, name: str, pairing: StripePairing, prefill_rank: int, decode_rank: int, frame: DecodeFrame,
                upstream_dcp: Tuple[int, int], prefill_page: int, prefill_kv_dtype: str, src_item_lens,
                dst_item_lens, dst_indices, num_draft: int, page_rule: Optional[PageRule] = None) -> StripePeer:
    """Check that prefill rank ``prefill_rank`` can serve decode rank ``decode_rank`` (registered as ``name``) by
    stripes, and return how. ``upstream_dcp``: the (size, rank) of the registration's own DCP frames;
    ``src_item_lens``: this rank's KV entries' page bytes, drafter entries last (``num_draft``); ``dst_indices``: the
    decode entry paired with each, by layer id; ``dst_item_lens``: the decode's entries' page bytes. Raises a
    RuntimeError that names what to change."""

    def refuse(why: str):
        raise RuntimeError(f"PD DCP stripe transfer to decode peer {name}: {why}")

    if not frame.stripes:
        refuse(f"this prefill stores its KV in DCP {pairing.prefill_dcp} stripes and sends a DCP decode by stripes, "
               "but the decode paired its ranks by attention TP; the decode must pair by stripes "
               "(sglang_patches/dcp_page_transfer.register_decode on a model with prefill_context_parallel)")
    if (frame.dcp_size, frame.dcp_rank) != tuple(upstream_dcp):
        refuse(f"its DCP frames disagree: {frame.dcp_size}/{frame.dcp_rank} announced, {upstream_dcp[0]}/"
               f"{upstream_dcp[1]} registered")
    problem = pairing.problem()
    if problem:
        refuse(problem)
    if frame.dcp_rank != decode_rank:
        refuse(f"its DCP rank {frame.dcp_rank} is not its attention-TP rank {decode_rank}")
    if frame.page_size != prefill_page:
        refuse(f"its page is {frame.page_size} tokens and this prefill's {prefill_page} (launch both with "
               f"--page-size {prefill_page})")
    if frame.run != pairing.block:
        refuse(f"its DCP rank keeps runs of {frame.run} tokens and the stripe map moves whole {pairing.block}-token "
               "stripes (a decode with the Neuron DCP pool keeps whole pages)")
    if frame.kv_dtype != prefill_kv_dtype:
        refuse(f"its KV dtype is {frame.kv_dtype} and this prefill's {prefill_kv_dtype} (launch both with "
               f"--kv-cache-dtype {prefill_kv_dtype})")
    if frame.draft_rows != pairing.draft_rows:
        refuse(f"it announces {frame.draft_rows} drafter rows and the pairing {pairing.draft_rows}")
    if (frame.draft_rows == DRAFT_RING) != (page_rule is not None):
        refuse(f"it announces {frame.draft_rows} drafter rows " + ("and no page rule; ring rows land at the pages of "
                                                                    "the request's state slot its page rule names"
                                                                    if page_rule is None else
                                                                    "and a page rule, whose drafter rows are a ring"))
    roles = pairing.roles(prefill_rank, decode_rank)
    if not roles.any:
        refuse(f"prefill rank {prefill_rank} sends decode rank {decode_rank} nothing: it pairs with prefill ranks "
               f"{list(pairing.prefill_ranks(decode_rank))}")
    n_target = len(src_item_lens) - num_draft
    dst_indices = tuple(int(j) for j in dst_indices)
    for i, (src_len, j) in enumerate(zip(src_item_lens, dst_indices)):
        draft = i >= n_target
        scale = pairing.decode_dcp if draft and pairing.draft_rows == DRAFT_REPLICATED else 1
        if int(dst_item_lens[j]) != int(src_len) * scale:
            what = f"{pairing.draft_rows} drafter entry {i - n_target}" if draft else f"target entry {i}"
            refuse(f"{what} holds {dst_item_lens[j]} bytes a page on the decode (entry {j}) and {src_len} on this "
                   f"prefill; a stripe moves whole, so the decode page must be {int(src_len) * scale} bytes")
    return StripePeer(pairing, decode_rank, roles, dst_indices[:n_target], dst_indices[n_target:], page_rule)
