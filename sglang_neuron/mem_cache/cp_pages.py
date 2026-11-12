# SPDX-License-Identifier: Apache-2.0
"""Context-parallel page ownership of a paged KV pool, as host metadata.

Every context-parallel KV layout the platform serves is the same block-cyclic map. An allocator page (the *native* page)
of ``cp_size * block`` tokens is one ``block``-token kernel block on each rank of the CP group, and rank ``r`` stores
the page's tokens ``[r * block, (r + 1) * block)`` in its local block of the same index as the native page. So a native
page id is the local block id on every rank, block tables are the same on every rank, and only the slots a rank writes
and the number of tokens it holds differ:

* decode context parallelism (``--dcp-size D``, distributed/decode_context_parallel.DcpLayout): ``block`` = the pool's
  page, native page = SGLang's widened ``D * page`` allocator page;
* token context parallelism (model_executor/kv_layout.KVLayout.cp_size ``C``): native page = SGLang's page, ``block`` =
  ``page / C``.

SGLang's own MLA DCP owns tokens round-robin (``position % D``); this layout keeps whole blocks on one rank instead, so a
page is a contiguous region a P/D transfer moves whole and a kernel gathers by block table.

All functions here are host (numpy) metadata. Write slots of tokens a rank does not own are ``DROP_SLOT`` (-1, which
the kernels skip), never a slot of the reserved page 0: the MLA / DSA pools keep page 0 all zeros (the zero-page
contract, mem_cache/mla_pool.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import NamedTuple, Sequence

import numpy as np

# A write slot the kernels skip: a token this rank does not own, or a padding row.
DROP_SLOT = -1
# int32 is the kernels' slot type
_INT32_MAX = int(np.iinfo(np.int32).max)


def _host_ints(values, what: str) -> tuple:
    device = getattr(values, "device", "cpu")
    if getattr(device, "type", device) != "cpu":
        raise ValueError(f"{what} must be host metadata, not a {device} tensor")
    try:
        return tuple(index(v) for v in np.asarray(values).reshape(-1).tolist())
    except TypeError as error:
        raise ValueError(f"{what} must be integers") from error


@dataclass(frozen=True)
class CpPages:
    """Block-cyclic page ownership over a ``cp_size``-rank group of ``block``-token kernel blocks."""

    cp_size: int
    block: int

    def __post_init__(self):
        if type(self.cp_size) is not int or type(self.block) is not int or self.cp_size < 1 or self.block < 1:
            raise ValueError(f"invalid CP page geometry {self}")

    @property
    def page(self) -> int:
        """The native (allocator) page: one block on every rank."""
        return self.cp_size * self.block

    def _rank(self, rank: int) -> int:
        if not 0 <= rank < self.cp_size:
            raise ValueError(f"CP rank {rank} is outside the {self.cp_size}-rank group")
        return rank

    def owner(self, positions) -> np.ndarray:
        """The CP rank that stores the token at each position (or native slot: both share the in-page offset)."""
        return (np.asarray(positions, dtype=np.int64) % self.page) // self.block

    def local_count(self, lengths, rank: int) -> np.ndarray:
        """How many of a sequence's first ``lengths`` tokens ``rank`` stores; a token at position ``p`` is local token
        ``local_count(p)`` of its owner."""
        rank = self._rank(rank)
        lengths = np.asarray(lengths, dtype=np.int64)
        full, rem = lengths // self.page, lengths % self.page
        return full * self.block + np.clip(rem - rank * self.block, 0, self.block)

    def pages_for(self, tokens: int) -> int:
        return -(-int(tokens) // self.page)

    def blocks_for(self, tokens: int) -> int:
        return -(-int(tokens) // self.block)

    def local_pages(self, native_pages: Sequence[int], tokens: int, rank: int) -> tuple:
        """The local blocks ``rank`` holds of a sequence of ``tokens`` tokens over ``native_pages`` (its allocator
        pages, in order): the first native pages up to the last one in which the rank owns a token. Raises on a page
        list that does not cover exactly the sequence, a repeated page or the reserved page 0."""
        rank = self._rank(rank)
        pages = _host_ints(native_pages, "native pages")
        tokens = index(tokens)
        if tokens < 0 or len(pages) != self.pages_for(tokens):
            raise ValueError(f"{len(pages)} native pages do not cover exactly {tokens} tokens of {self.page}-token "
                             "pages")
        if len(set(pages)) != len(pages) or any(p <= 0 for p in pages):
            raise ValueError("native pages must be unique and exclude the reserved page 0")
        owned = max(0, (self.blocks_for(tokens) + self.cp_size - 1 - rank) // self.cp_size)
        return pages[:owned]

    def write_slots(self, native_slots, rank: int) -> np.ndarray:
        """SGLang's native token slots -> ``rank``'s local slots (int32), ``DROP_SLOT`` where another rank owns the
        token or the slot is negative (a padding row). A slot of the reserved page 0 is refused: nothing writes it."""
        rank = self._rank(rank)
        slots = np.asarray(_host_ints(native_slots, "native slots"), dtype=np.int64)
        live = slots >= 0
        if np.any(slots[live] < self.page):
            raise ValueError("a native slot addresses the reserved page 0, which the zero-page contract keeps zero")
        local = (slots // self.page) * self.block + slots % self.block
        if local.size and int(local.max()) > _INT32_MAX:
            raise ValueError("local slots exceed the kernels' int32 slot type")
        mine = live & (self.owner(slots) == rank)
        return np.where(mine, local, DROP_SLOT).astype(np.int32)


class BlockTransfer(NamedTuple):
    """One kernel block of a P/D transfer: block ``src_offset`` of source rank ``src_rank``'s page ``src_page`` to block
    ``dst_offset`` of destination rank ``dst_rank``'s page ``dst_page``. The offsets are 0 for a sharded pool (a page
    is one block per rank); a replicated index-K pool holds whole native pages on every rank, ``cp_size`` blocks each."""

    src_rank: int
    src_page: int
    src_offset: int
    dst_rank: int
    dst_page: int
    dst_offset: int


def latent_page_plan(tokens: int, src: CpPages, src_pages: Sequence[int], dst: CpPages, dst_pages: Sequence[int], *,
                     decode_prefix_len: int = 0, replicated: bool = False) -> tuple:
    """The block moves that hand a request's latent (MLA) or index-K pages from a prefill to a decode, both block-cyclic
    over the same ``block`` (prefill CP ``src.cp_size``, decode DCP/CP ``dst.cp_size``; 1 for either side without CP).

    ``src_pages`` / ``dst_pages`` are the request's native pages on each side, from token 0. Global block ``j`` (tokens
    ``[j * block, (j + 1) * block)``) lives on prefill rank ``j % Cp`` in local page ``src_pages[j // Cp]`` and goes to
    decode rank ``j % Cd`` in local page ``dst_pages[j // Cd]``; a partial last block moves whole. Blocks behind the
    decode's cached prefix (``decode_prefix_len``, whole decode native pages) are not sent; an unaligned prefix is
    refused, since its first block would land on the wrong owner.

    ``replicated``: the pool keeps every token on every rank (a replicated DSA index-K pool, ``index_sharding``
    "replicated"): global block ``j`` is read from prefill rank ``j % Cp`` at block ``j % Cp`` of its native page and
    written to every decode rank at block ``j % Cd`` of its native page."""
    if src.block != dst.block:
        raise ValueError(f"prefill block {src.block} and decode block {dst.block} differ: a kernel block does not move "
                         "whole")
    tokens = index(tokens)
    src_pages = _host_ints(src_pages, "prefill pages")
    dst_pages = _host_ints(dst_pages, "decode pages")
    if len(src_pages) != src.pages_for(tokens) or len(dst_pages) != dst.pages_for(tokens):
        raise ValueError(f"the page lists ({len(src_pages)} prefill, {len(dst_pages)} decode) do not cover exactly "
                         f"{tokens} tokens")
    if any(p <= 0 for p in src_pages + dst_pages):
        raise ValueError("a request page is the reserved page 0")
    if decode_prefix_len < 0 or decode_prefix_len % dst.page:
        raise ValueError(f"the decode prefix ({decode_prefix_len} tokens) is not whole decode native pages "
                         f"({dst.page} tokens)")
    first = min(decode_prefix_len, tokens) // src.block
    moves = []
    for j in range(first, src.blocks_for(tokens)):
        s_rank, d_rank = j % src.cp_size, j % dst.cp_size
        s_page, d_page = src_pages[j // src.cp_size], dst_pages[j // dst.cp_size]
        if not replicated:
            moves.append(BlockTransfer(s_rank, s_page, 0, d_rank, d_page, 0))
            continue
        for rank in range(dst.cp_size):
            moves.append(BlockTransfer(s_rank, s_page, s_rank, rank, d_page, d_rank))
    return tuple(moves)


def rank_moves(moves, *, src_rank: int) -> tuple:
    """The moves prefill rank ``src_rank`` posts; a rank that owns no block of the request posts nothing."""
    return tuple(m for m in moves if m.src_rank == src_rank)
