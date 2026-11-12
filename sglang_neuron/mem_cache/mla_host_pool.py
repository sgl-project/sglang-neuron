# SPDX-License-Identifier: Apache-2.0
"""HiCache L2 for the Neuron MLA latent and DSA index-K pools (mem_cache/mla_pool.py): their host copies, and the hooks
that make SGLang's HiCache assembler build them.

SGLang's assembler (mem_cache/hybrid_cache/hybrid_pool_assembler.py) builds an MLA pool's host copy as its own
``MLATokenToKVPoolHost`` (``build_kv_host_pool``), with no per-platform choice the way the MHA host pool has
(``get_mha_host_pool_cls``, mem_cache/host_pool.py), and a DSA pool's index keys as a ``DSAIndexerPoolHost`` side pool
(``_build_sidecar_host_pool``) beside it. Both copy with CUDA kernels over token-major rows. For a Neuron latent pool the
hooks here build the Neuron host pools instead, on the same TransferEngine as the MHA pool's (runtime batch DMA of
whole pages, XferEvents, the hold of hicache_hold.py):

* ``NeuronLatentHostPool``, the KV pool: the latent and every other paged buffer the model declares
  (``NeuronMLATokenToKVPool.buffers()``), except those a side pool carries;
* ``NeuronIndexKeyHostPool``, the INDEXER side pool of a DSA pool (``get_device_pool_infos``): the index keys. SGLang
  gives it the KV pool's host indices (``indices_from_pool=KV``), so its host page ``h`` mirrors the KV pool's.

Each region is one layer of one buffer, page-first, and a host page holds one device page of every region: this rank's
block of the SGLang page. Under token CP (a declared kv_layout) the native page id is the block id on every rank, so
each rank moves its own block of each page with the controller's indices, untranslated. SGLang's
DCP translation (``maybe_dcp_kernel_indices``) assumes its round-robin token owners, which the Neuron DCP layout does not
have (it keeps whole blocks per rank, mem_cache/cp_pages.py), and the platform serves HiCache on no decode server, so a
DCP pool's host pool is refused. The KPool request tails are request rows, not prefix pages: SGLang's host group carries
none (a hit ends on a whole page, which ends a whole pool), and they stay on the device.

Refused, each naming its reason: MTP draft pools packed into the host pool, a row width other than the pool's latent,
a transfer page other than the pool's, a host layout other than layer_first and an L3 storage backend (as for MHA).
"""

from __future__ import annotations

import logging

from sglang_neuron.mem_cache.host_pool import NeuronPagedHostPool

logger = logging.getLogger(__name__)

_ASSEMBLER = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler"
# the buffer each side pool of a Neuron latent pool carries, by upstream PoolName value
_SIDE_POOL_BUFFERS = {"indexer": "index_keys"}


def is_neuron_latent_pool(pool) -> bool:
    return bool(getattr(pool, "is_neuron_latent_pool", False))


def side_pool_buffers(kv_pool) -> set:
    """The buffers of ``kv_pool`` SGLang's assembler gives a side pool of their own: those its device pool infos name
    for every pool but KV, when upstream reads the infos at all (``collect_host_pool_infos``: a DSATokenToKVPool's)."""
    from sglang.srt.mem_cache.hybrid_cache.host_pool_config import collect_host_pool_infos

    out = set()
    for info in collect_host_pool_infos(kv_pool):
        if info.pool_name.value == "kv":
            continue
        name = _SIDE_POOL_BUFFERS.get(info.pool_name.value)
        if name is None:
            raise ValueError(f"{type(kv_pool).__name__} describes a {info.pool_name.value} pool no Neuron host pool "
                             "carries")
        out.add(name)
    return out


def _refuse_dcp(kv_pool) -> None:
    if getattr(kv_pool, "cp_source", "none") == "dcp":
        raise ValueError("--enable-hierarchical-cache over a decode-context-parallel MLA pool: SGLang's DCP index "
                         "translation assumes round-robin token owners, the Neuron DCP pool keeps whole blocks per "
                         "rank, and HiCache is served on no decode server")


class NeuronLatentHostPool(NeuronPagedHostPool):
    """The host copy of a Neuron MLA / DSA pool's paged buffers but those a side pool carries (``exclude``): the
    latent, and any paged buffer the model adds."""

    def __init__(self, kv_pool, *, host_to_device_ratio, host_size, page_size: int, layout: str, allocator_type,
                 pool_label: str = "kv", exclude=()):
        _refuse_dcp(kv_pool)
        if page_size != kv_pool.page_size:
            raise ValueError(f"HiCache's {page_size}-token transfer page is not the Neuron MLA pool's "
                             f"{kv_pool.page_size}-token page")
        if kv_pool.hicache_write_back_staging is not None or kv_pool.layer_shard_enabled:
            raise ValueError("the Neuron MLA pool stages no write-back on the device and shards no layers")
        views = [(name, view) for name, _, view in kv_pool.page_views() if name not in set(exclude)]
        self.buffer_names = tuple(dict.fromkeys(name for name, _ in views))
        self.kv_lora_rank = kv_pool.kv_lora_rank
        self.qk_rope_head_dim = kv_pool.qk_rope_head_dim
        self.kv_cache_dim = kv_pool.kv_cache_dim
        self.dsa_kv_cache_store_fp8 = False
        capacity = kv_pool.size if kv_pool.host_capacity_tokens is None else kv_pool.host_capacity_tokens
        super().__init__([view for _, view in views], dtype=kv_pool.store_dtype, page_size=page_size,
                         layer_num=kv_pool.layer_num, device_capacity_tokens=capacity,
                         start_layer=kv_pool.start_layer, end_layer=kv_pool.end_layer,
                         host_to_device_ratio=host_to_device_ratio, host_size=host_size, layout=layout,
                         allocator_type=allocator_type, pool_label=pool_label,
                         device_page_limit=kv_pool.paged_blocks, load_layer=0)


class NeuronIndexKeyHostPool(NeuronPagedHostPool):
    """The host copy of a Neuron DSA pool's index keys: SGLang's INDEXER side pool, with the KV host pool's pages
    (``root_host_pool``) and indices. ``layer_mapping`` is upstream's device layer -> host layer of the side pool's
    buffers (HostPoolConfig.device_to_host_layer)."""

    def __init__(self, kv_pool, *, root_host_pool, layer_mapping: dict, allocator_type=None):
        _refuse_dcp(kv_pool)
        # the device layer of each leading index of the index keys: every layer, or the layers that hold keys (a
        # model whose shared-indexer layers hold none, e.g. GLM-5.2, declares index_key_layers())
        layer_of = getattr(kv_pool, "index_key_layers", None)
        layer_of = tuple(layer_of()) if layer_of is not None else None
        views = dict((idx[0] if layer_of is None else layer_of[idx[0]], view)
                     for name, idx, view in kv_pool.page_views() if name == "index_keys")
        layers = sorted(layer_mapping, key=layer_mapping.get)
        if sorted(layer_mapping.values()) != list(range(len(layer_mapping))) or any(l not in views for l in layers):
            raise ValueError(f"index-key host layers {layer_mapping} are not layers of the pool's "
                             f"{len(views)} index-key layers")
        self.index_head_dim = kv_pool.index_head_dim
        super().__init__([views[layer] for layer in layers], dtype=kv_pool.store_dtype,
                         page_size=root_host_pool.logical_page_size, layer_num=len(layers),
                         device_capacity_tokens=root_host_pool.device_capacity_tokens,
                         start_layer=kv_pool.start_layer, end_layer=kv_pool.end_layer,
                         host_to_device_ratio=1.0, host_size=0, layout=root_host_pool.layout,
                         allocator_type=allocator_type, pool_label="indexer", host_token_capacity=root_host_pool.size,
                         device_page_limit=kv_pool.paged_blocks, load_layer=min(layer_mapping))
        if self.page_num != root_host_pool.page_num:
            raise ValueError(f"the index-key host pool's {self.page_num} pages are not the KV host pool's "
                             f"{root_host_pool.page_num}")


# ---- the hooks --------------------------------------------------------------------------------------------------


def build_kv_host_pool(original, *, kv_pool, page_size, use_mla=None, override_kv_cache_dim=None, host_size=None,
                       mtp_draft_device_pools=(), pool_label="kv"):
    """hybrid_pool_assembler.build_kv_host_pool: a Neuron latent pool's KV host pool is NeuronLatentHostPool, sized as
    upstream sizes the MLA host pool (--hicache-ratio of the pool's tokens, or --hicache-size)."""
    if not is_neuron_latent_pool(kv_pool):
        return original(kv_pool=kv_pool, page_size=page_size, use_mla=use_mla,
                        override_kv_cache_dim=override_kv_cache_dim, host_size=host_size,
                        mtp_draft_device_pools=mtp_draft_device_pools, pool_label=pool_label)
    if mtp_draft_device_pools:
        raise ValueError("Neuron HiCache keeps no MTP draft KV: the Neuron MLA pool serves no speculative decoding")
    if override_kv_cache_dim not in (None, kv_pool.kv_cache_dim):
        raise ValueError(f"a {override_kv_cache_dim}-wide host row is not the Neuron MLA pool's "
                         f"{kv_pool.kv_cache_dim}-wide latent")
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
    from sglang.srt.runtime_context import get_memory

    memory = get_memory()
    return NeuronLatentHostPool(kv_pool, host_to_device_ratio=memory.hicache_ratio,
                                host_size=memory.hicache_size if host_size is None else host_size, page_size=page_size,
                                layout=memory.hicache_mem_layout,
                                allocator_type=hybrid_pool_assembler._get_allocator_type(), pool_label=pool_label,
                                exclude=side_pool_buffers(kv_pool))


def build_sidecar_host_pool(original, *, config, root, root_host_pool):
    """hybrid_pool_assembler._build_sidecar_host_pool: a Neuron DSA pool's INDEXER side pool is
    NeuronIndexKeyHostPool."""
    if not is_neuron_latent_pool(root):
        return original(config=config, root=root, root_host_pool=root_host_pool)
    if _SIDE_POOL_BUFFERS.get(config.pool_name.value) != "index_keys":
        raise ValueError(f"no Neuron host pool carries a Neuron MLA pool's {config.pool_name.value} side pool")
    if not isinstance(root_host_pool, NeuronLatentHostPool):
        raise TypeError(f"the Neuron index-key host pool mirrors a NeuronLatentHostPool, not "
                        f"{type(root_host_pool).__name__}")
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler

    return NeuronIndexKeyHostPool(root, root_host_pool=root_host_pool, layer_mapping=config.device_to_host_layer,
                                  allocator_type=hybrid_pool_assembler._get_allocator_type())


# Upstream functions these hooks replace or rely on, pinned by md5 against SGLang 43881a7
# (sglang_patches.source_guard); a changed source stops the launch.
_SOURCES_MD5 = {
    # replaced for a Neuron latent pool: upstream builds MLATokenToKVPoolHost unconditionally
    (_ASSEMBLER, "build_kv_host_pool"): "3733320973aea2035d48663febbb4e65",
    # replaced for a Neuron latent pool: upstream builds DSAIndexerPoolHost
    (_ASSEMBLER, "_build_sidecar_host_pool"): "b6bff01c8583ef0f3a7af6e1e71af8bf",
    # relied on: the side pools share the KV pool's device pool and, through the sidecar specs, its host indices
    (_ASSEMBLER, "_build_pool_entries"): "7759b23e97b4f4ba9d33af3132091485",
    ("sglang.srt.mem_cache.hybrid_cache.host_pool_config", "HostPoolConfig.sidecar_spec"):
        "196c9bfd2696e060851bf83701e8906d",
    # relied on: only a DSATokenToKVPool's infos name side pools (side_pool_buffers)
    ("sglang.srt.mem_cache.hybrid_cache.host_pool_config", "collect_host_pool_infos"):
        "08c2316718d57469ac6fd7be1d409e3d",
    # relied on: a load calls each pool's per-layer load in transfer-layer order, through its layer mapper, so a pool's
    # first mapped layer comes first (NeuronPagedHostPool.load_layer)
    ("sglang.srt.mem_cache.l2_transfer", "L2TransferEngine.submit_host_to_device"): "463a30683200f30680308cd112db1596",
}


def register_hooks(registry, hook_type) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SOURCES_MD5, __name__)
    registry.register(f"{_ASSEMBLER}.build_kv_host_pool", build_kv_host_pool, hook_type.AROUND)
    registry.register(f"{_ASSEMBLER}._build_sidecar_host_pool", build_sidecar_host_pool, hook_type.AROUND)
