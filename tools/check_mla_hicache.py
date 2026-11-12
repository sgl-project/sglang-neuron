#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of the MLA / DSA pools' host copies (mem_cache/mla_host_pool.py, mla_pool.get_cpu_copy) on one
NeuronCore, at serving geometry.

Per geometry and request size (--ctx tokens) it builds the real NeuronMLATokenToKVPool on the device and the HiCache
host pools the plugin's hooks build for it (the latent, and the index keys as the INDEXER side pool), then:

* fills N host pages with random bytes and loads them into N device pages (runtime batch DMA, the HiCache load path);
* stores those device pages to other host pages and checks them bit-exact against the random bytes;
* loads them into N *other* device pages, stores those back and checks bit-exact again; reads one layer of the
  destination pages back with a plain ``.to("cpu")`` (independent of the DMA path) and checks it; checks block 0 is
  still zero (the zero-page contract) and the pages no copy targeted are untouched;
* does the same for get_cpu_copy / load_cpu_copy (SGLang's retraction backup) over the request's slots;
* times store and load of the request's pages, both pools submitted together as a HiCache transfer submits them, over
  contiguous and shuffled device pages, and reports GB/s.

Geometries: ``k31`` is a token-CP DSA rank: 11 DSA layers, CP16 over 64-token blocks (1024-token SGLang page),
576-wide latent, owner-sharded 128-wide index keys, so a rank holds 1/16 of each page; ``glm52`` is GLM-5.2: 78 layers,
64-token page, no CP, replicated index keys.

usage: NEURON_RT_VISIBLE_CORES=<one free core> python tools/check_mla_hicache.py --geom k31 glm52 --ctx 8192 32768
Run on a free NeuronCore only. Prints one JSON line per check and exits non-zero on a mismatch.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GEOMS = {
    "k31": dict(layers=11, page=1024, cp=16, index="owner"),
    "glm52": dict(layers=78, page=64, cp=1, index="replicated"),
}


def emit(**rec):
    print(json.dumps(rec, default=str), flush=True)


@contextlib.contextmanager
def published_topology():
    """The SGLang runtime context a served process has (a published topology of ServerArgs(model_path="dummy"), WORLD
    rank 0) for the block, published and reset the way SGLang's own tests do it: upstream SGLang refuses to read an
    unpublished configuration."""
    from sglang.srt import runtime_context
    from sglang.srt.server_args import ServerArgs

    runtime_context.reset_context()
    runtime_context.publish(ServerArgs(model_path="dummy"), role="test",
                            ranks=runtime_context.SpawnRanks(world_rank=0, dp_rank=None))
    try:
        yield
    finally:
        runtime_context.reset_context()


def build(g, pages, device):
    """The pool (the zero fill's compile skipped: nothing here frees a page) and its two host pools."""
    from sglang_neuron.mem_cache import mla_host_pool, mla_pool
    from sglang_neuron.model_executor import kv_layout

    with ExitStack() as st:
        st.enter_context(mock.patch.object(kv_layout, "_LAYOUT", kv_layout.KVLayout(cp_size=g["cp"], cp_rank=0)))
        st.enter_context(published_topology())
        st.enter_context(mock.patch("sglang_neuron.distributed.decode_context_parallel.runtime_layout",
                                    lambda page: None))
        st.enter_context(mock.patch("sglang_neuron.model_executor.runtime_init.neuron_device", lambda: device))
        st.enter_context(mock.patch.object(mla_pool, "warm_fill", lambda *a, **k: None))
        pool = mla_pool.NeuronMLATokenToKVPool(pages * g["page"], g["page"], torch.bfloat16, 512, 64, g["layers"],
                                               str(device), False, index_head_dim=128, index_sharding=g["index"])
        kv = mla_host_pool.NeuronLatentHostPool(pool, host_to_device_ratio=2.0, host_size=0, page_size=g["page"],
                                                layout="layer_first", allocator_type="default",
                                                exclude=("index_keys",))
        idx = mla_host_pool.NeuronIndexKeyHostPool(pool, root_host_pool=kv,
                                                   layer_mapping={i: i for i in range(g["layers"])})
    return pool, kv, idx


def slots(pages, page):
    return torch.cat([torch.arange(p * page, (p + 1) * page) for p in pages])


def move(hosts, to_host, host_pages, dev_pages, page):
    """One HiCache transfer: both pools submitted, then waited; seconds."""
    t0 = time.perf_counter()
    for h in hosts:
        h._submit(to_host, slots(host_pages, page), slots(dev_pages, page))
    for h in hosts:
        h.engine.wait_all()
    return time.perf_counter() - t0


def equal_bytes(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def run(geom, ctx, reps, device):
    from sglang_neuron.mem_cache.host_pool import page_runs

    g = GEOMS[geom]
    n = math.ceil(ctx / g["page"])
    pages = 2 * n + 2
    t0 = time.perf_counter()
    pool, kv, idx = build(g, pages, device)
    hosts = (kv, idx)
    emit(geom=geom, ctx=ctx, check="build", pages=n, pool_pages=pages, host_pages=kv.page_num,
         pool_gib=pool.get_kv_size_bytes() / 2**30, seconds=time.perf_counter() - t0)
    page = g["page"]
    src, dst = list(range(1, n + 1)), list(range(n + 1, 2 * n + 1))
    # host pages: the random reference, the first store, the store after the load into other device pages
    ref_h, mid_h, out_h = list(range(n)), list(range(n, 2 * n)), list(range(2 * n, 3 * n))
    assert 3 * n <= kv.page_num, (n, kv.page_num)
    gen = torch.Generator().manual_seed(1)
    ref = []
    for h in hosts:
        for region in h.host_regions:
            r = torch.randint(-128, 127, (n, region[0].numel() * region.element_size()), dtype=torch.int8,
                              generator=gen)
            region.view(torch.int8).view(region.shape[0], -1)[ref_h] = r
            ref.append(r)
    ok = True
    move(hosts, False, ref_h, src, page)          # host -> device src
    move(hosts, True, mid_h, src, page)           # device src -> host
    regions = [region for h in hosts for region in h.host_regions]
    rt1 = all(equal_bytes(region[mid_h].reshape(n, -1), r) for region, r in zip(regions, ref))
    move(hosts, False, mid_h, dst, page)          # host -> device dst (other pages)
    move(hosts, True, out_h, dst, page)           # device dst -> host
    rt2 = all(equal_bytes(region[out_h].reshape(n, -1), r) for region, r in zip(regions, ref))
    lat0 = pool.latent[0].to("cpu")[dst].reshape(n, -1)
    direct = equal_bytes(lat0, ref[0])
    zero0 = pool.zero_page_violations() == []
    untouched = bool((pool.latent[0].to("cpu")[2 * n + 1:] == 0).all())
    ok &= rt1 and rt2 and direct and zero0 and untouched
    emit(geom=geom, ctx=ctx, check="hicache_bit_exact", store=rt1, load_other_pages=rt2, direct_readback=direct,
         block0_zero=zero0, untouched_zero=untouched)

    # retraction backup: the request's slots (a partial last page) to host and back into other pages
    tokens = ctx - 1
    before = [v.to("cpu")[src] for _, _, v in pool.page_views()]
    t = time.perf_counter()
    copy = pool.get_cpu_copy(slots(src, page)[:tokens])
    t_get = time.perf_counter() - t
    t = time.perf_counter()
    pool.load_cpu_copy(copy, slots(dst, page)[:tokens])
    t_load = time.perf_counter() - t
    after = [v.to("cpu")[dst] for _, _, v in pool.page_views()]
    cpu_ok = all(equal_bytes(a, b) for a, b in zip(after, before))
    ok &= cpu_ok
    nbytes = n * (sum(kv.page_bytes) + sum(idx.page_bytes))
    emit(geom=geom, ctx=ctx, check="cpu_copy_bit_exact", ok=cpu_ok, get_s=t_get, load_s=t_load,
         get_gbps=nbytes / t_get / 1e9, load_gbps=nbytes / t_load / 1e9)

    # bandwidth: the request's pages, contiguous and shuffled on the device
    shuffled = [src[i] for i in torch.randperm(n, generator=gen).tolist()]
    for name, dev in (("contiguous", src), ("shuffled", shuffled)):
        stores = [move(hosts, True, mid_h, dev, page) for _ in range(reps)]
        loads = [move(hosts, False, mid_h, dev, page) for _ in range(reps)]
        emit(geom=geom, ctx=ctx, check="bandwidth", device_pages=name, pages=n, mib=nbytes / 2**20,
             runs_per_region=len(page_runs(dev, mid_h)),
             regions=len(regions), store_ms=1e3 * min(stores), load_ms=1e3 * min(loads),
             store_gbps=nbytes / min(stores) / 1e9, load_gbps=nbytes / min(loads) / 1e9,
             store_ms_median=1e3 * sorted(stores)[reps // 2], load_ms_median=1e3 * sorted(loads)[reps // 2])
    del pool, kv, idx
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geom", nargs="+", default=list(GEOMS), choices=list(GEOMS))
    ap.add_argument("--ctx", nargs="+", type=int, default=[8192, 32768])
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    import torch_neuronx  # noqa: F401

    device = torch.device("neuron", 0)
    ok = True
    for geom in a.geom:
        for ctx in a.ctx:
            ok &= run(geom, ctx, a.reps, device)
    emit(check="summary", ok=ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
