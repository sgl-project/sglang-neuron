# SPDX-License-Identifier: Apache-2.0
"""The full-attention layers' KV pool, in the page layout the Qwen3.5 attention kernels read.

``attn_decode_tkg`` (``block_table``) and ``attn_prefill_tkg`` (``write_pages`` / ``prior_pages``) read and write a
request's history page by page: ``k_cache_t [L_a * NPAGE, d, PG]`` (K transposed per page, so the scores need no
transpose) and ``v_cache [L_a * NPAGE, PG, d]``, attention layer ``a``'s page ``p`` at row ``a * NPAGE + p``. SGLang's
page is ``PG`` and its block table names each request's pages, so the pool is SGLang's page pool in that layout.

This is Qwen3.5's slot pool (``models/qwen35/kv_pool.Qwen35SlotKVPool``) without its context-parallel paths, which
this model's graphs do not run: one KV head per rank, bfloat16, K dim-major. The kernels write the pool inside the
compiled graphs, so a token-addressed write is refused.
"""

from __future__ import annotations

from sglang_neuron.models.qwen35.kv_pool import Qwen35SlotKVPool


class Qwen38SlotKVPool(Qwen35SlotKVPool):
    """The paged KV pool (module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.kv_cp_size > 1:
            raise ValueError("Qwen3.8 runs no KV context parallelism; launch without --dcp-size")

    @property
    def k_tm(self) -> bool:
        """The attention kernels read K dim-major ``[d, PG]`` per page on every role here."""
        return False


def slot_kv(which: str):
    """A ``PoolEntry`` resolver of the slot pool's layer-major K (``which="k"``) or V tensor."""
    if which not in ("k", "v"):
        raise ValueError(which)

    def resolve(kv_pool, req_to_token_pool):
        pool = getattr(kv_pool, "full_kv_pool", kv_pool)
        if not isinstance(pool, Qwen38SlotKVPool):
            raise RuntimeError(f"the full-attention KV pool is {type(pool).__name__}, not the Qwen3.8 paged pool")
        return [pool.k_stack if which == "k" else pool.v_stack]

    return resolve
