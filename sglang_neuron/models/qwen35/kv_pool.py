# SPDX-License-Identifier: Apache-2.0
"""The full-attention layers' KV pool in the giga kernels' slot layout.

``qwen35_hybrid_multistep`` reads one request's whole history from a KV *slot* (``kv_slot``), not through a block
table: ``k_cache_t [L_a * Spa, d, S]`` (K transposed per slot) and ``v_cache [L_a * Spa, S, d]``, attention layer
``a``'s slot ``s`` at row ``a * Spa + s``, the new token's K / V written in place at its position. With SGLang's page
size equal to the slot (``--page-size S``), one SGLang page is one slot, so the page id SGLang allocated for a request
(``out_cache_loc // S``) is its ``kv_slot`` and the token's offset in the page is its position. The pool:

- allocates every attention layer's K and V as one layer-major tensor each (``k_stack`` / ``v_stack``, the kernel's
  operands), whose per-layer views are SGLang's ``k_buffer`` / ``v_buffer`` (``[pages, 1, d, S]`` and
  ``[pages, 1, S, d]``, one KV head per rank);
- keeps the platform pool's page-granular transfer entries: a page is ``d * S`` contiguous elements in both tensors,
  so SGLang's PD transfer moves whole slots by page id;
- refuses every token-addressed write (``set_kv_buffer``): the kernels write the pool inside the compiled graphs.

On the kv_cp=1 PD prefill role, ``config.KV_TM_ENV`` flips the K stacks to V's token-major layout
(``[rows, 1, S, d]`` — a pure permutation of the same page bytes) so the prefill->kv_cp-decode scatter moves a
rank's K block of a page as one contiguous run per layer (sglang_patches/kv_cp_scatter); the launch refuses the
flag on every other role (the kv_cp=1 decode's attention kernel reads the dim-major layout, and a whole-page PD
transfer is layout-blind).

Under decode context parallelism (``config.Qwen35Layout.kv_cp`` C, declared as ``model_executor.kv_layout.KVLayout``)
the same page id holds this rank's ``S / C``-token block of every page with *both* KV heads: the stacks are
``[L_a * Spa + 1, 2, d, S / C]`` / ``[L_a * Spa + 1, 2, S / C, d]`` (head axis after the page, as upstream's 4-D
buffers; the block of page ``p`` of layer ``a`` is one contiguous run at row ``a * Spa + p``), and the stacks' last
row is the CP write *sink* the kernel redirects a non-owner rank's new-token K / V writes to (never read, never
allocated: the allocator hands out SGLang's pages ``1 .. size / S`` only). A PD transfer then moves one block of each
page per peer (sglang_patches/kv_cp_scatter), ``2 * d * S / C`` contiguous bytes, instead of the whole page.

SGLang builds it as the full-attention pool of its hybrid linear pool (``HybridLinearKVPool``, which maps the global
attention layer ids onto layers ``0 .. L_a - 1`` here); the megakernel pool-layout seam hands SGLang this class.
"""
from __future__ import annotations

import torch

from sglang_neuron.mem_cache.memory_pool import NeuronMHATokenToKVPoolKind
from sglang_neuron.models.qwen35 import config as qwen35_config


class Qwen35SlotKVPool(NeuronMHATokenToKVPoolKind):
    """The slot-layout KV pool (module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # without KV CP one KV head per rank; with it every KV head (2) of this rank's block of each page
        want = 2 if self.kv_cp_size > 1 else 1
        if self.head_num != want:
            raise ValueError(f"the Qwen3.5 giga attention keeps {want} KV head{'s' if want > 1 else ''} per rank; the "
                             f"pool holds {self.head_num}")
        if self.dcp is not None or self.paired_layers:
            raise ValueError("the Qwen3.5 slot KV pool runs no (decode) context parallelism and no layer pairs")

    def _create_buffers(self):
        if self.store_dtype != torch.bfloat16:
            raise ValueError(f"the Qwen3.5 giga attention reads a bfloat16 KV cache; the pool stores {self.store_dtype}"
                             " (launch with --kv-cache-dtype auto / bfloat16)")
        nb, d, s = self.num_blocks, self.head_dim, self.block_size
        # kv_cp: the stacks get the head axis of upstream's 4-D buffers, plus one trailing row as the CP write sink
        # (never read, never allocated: it is not a page the allocator manages)
        h, rows = self.head_num, self.layer_num * nb + (1 if self.kv_cp_size > 1 else 0)
        # token-major K: the kv_cp decode's per-head pages are [.., tokens, head_dim] like V (attn_cp_core
        # reads and writes them that way) and the kv_cp=1 PD prefill's pages take the same layout under
        # config.KV_TM_ENV — a pure permutation of the same page bytes, so the scatter's K block of a page is
        # one contiguous run per layer on both roles (sglang_patches/kv_cp_scatter). The kv_cp=1 decode role
        # refuses the flag (its attention kernel reads the dim-major layout), so a token-major kv_cp=1 pool
        # only ever feeds the prefill graph.
        self.k_stack = torch.zeros((rows, h, s, d) if self.k_tm else (rows, h, d, s),
                                   dtype=self.store_dtype, device=self.kv_device)
        self.v_stack = torch.zeros((rows, h, s, d), dtype=self.store_dtype, device=self.kv_device)
        if self.kv_cp_size > 1:
            self.k_buffer = [self.k_stack[i * nb:(i + 1) * nb] for i in range(self.layer_num)]
            self.v_buffer = [self.v_stack[i * nb:(i + 1) * nb] for i in range(self.layer_num)]
        else:
            # the 3-D stacks of the one-head slot pool (the decode kernel's operands), per-layer views unsqueezed
            self.k_stack, self.v_stack = self.k_stack[:, 0], self.v_stack[:, 0]
            self.k_buffer = [self.k_stack[i * nb:(i + 1) * nb].unsqueeze(1) for i in range(self.layer_num)]
            self.v_buffer = [self.v_stack[i * nb:(i + 1) * nb].unsqueeze(1) for i in range(self.layer_num)]

    @property
    def k_tm(self) -> bool:
        """The pool's K pages are token-major [.., tokens, head_dim] like its V pages, not the dim-major
        transpose — unconditional on the kv_cp decode (attn_cp_core reads and writes them that way, and the
        PD scatter's K block of a page is then one contiguous run per layer on both roles), flag-gated
        (config.KV_TM_ENV) on the kv_cp=1 PD prefill that serves the scatter."""
        return self.kv_cp_size > 1 or (self.kv_cp_size == 1 and qwen35_config.kv_tm_launch())

    @property
    def cp_sink_row(self) -> int:
        """The flat K / V stack row a CP decode redirects a non-owner rank's new-token writes to (never read)."""
        if not self.kv_cp_size > 1:
            raise AttributeError("the Qwen3.5 slot KV pool keeps a CP write sink row only under KV CP")
        return self.layer_num * self.num_blocks

    def _clear_buffers(self):
        super()._clear_buffers()
        self.k_stack = self.v_stack = None

    def get_kv_size_bytes(self):
        """Upstream's MHA contract, (K bytes, V bytes) over every layer: SGLang's hybrid pools
        (``HybridLinearKVPool``) unpack it."""
        return self.k_stack.nbytes, self.v_stack.nbytes

    def get_kv_buffer_shape(self):
        shape = (self.num_blocks, self.head_num, self.block_size, self.head_dim)
        k_shape = shape if self.k_tm else (self.num_blocks, self.head_num, self.head_dim, self.block_size)
        return torch.Size(k_shape), torch.Size(shape)

    @property
    def slots(self) -> int:
        """Rows per layer block of the kernel's KV pools (``Spa``): SGLang's reserved page 0 and every page."""
        return self.num_blocks

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, *args, **kwargs):
        raise RuntimeError("the Qwen3.5 slot KV pool is written by the giga kernels inside the compiled graphs; a "
                           "token-addressed write would land in the transposed K layout")

    def get_device_pool_infos(self):
        """HiCache's device buffers (mem_cache/memory_pool.py). At kv_cp=1 the base class's, unchanged. Under kv_cp
        the pool serves the host tier with SGLANG_NEURON_QWEN35_HICACHE=1 (config.HICACHE_ENV): page p's row of the
        per-layer views is this rank's block of page p (both KV heads), the page id is the token page on every
        rank, and mem_cache/host_pool.py's NeuronCpHostKVPool moves the rows by that id; without the flag the base
        class's CP refusal stands (docs/design/context-parallelism.md#the-host-tier)."""
        if self.kv_cp_size == 1:
            return super().get_device_pool_infos()
        if not qwen35_config.hicache_launch():
            self._refuse_cp_hicache()
        return self._mha_device_pool_infos()


def slot_kv(which: str):
    """A ``PoolEntry`` resolver of the slot pool's layer-major K (``which="k"``) or V tensor."""
    if which not in ("k", "v"):
        raise ValueError(which)

    def resolve(kv_pool, req_to_token_pool):
        pool = getattr(kv_pool, "full_kv_pool", kv_pool)
        if not isinstance(pool, Qwen35SlotKVPool):
            raise RuntimeError(f"the full-attention KV pool is {type(pool).__name__}, not the Qwen3.5 slot pool")
        return [pool.k_stack if which == "k" else pool.v_stack]

    return resolve
