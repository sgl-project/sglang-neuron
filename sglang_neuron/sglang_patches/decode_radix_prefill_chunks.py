# SPDX-License-Identifier: Apache-2.0
"""Make a chunk-keyed PD decode reuse only KV its prefills computed for whole chunks, and keep apart KV from
prefills that compute different bits.

On the chunk-keyed tree (sglang_patches.chunk_keyed_tree) a prefill computes every prompt in whole chunks from position
0. A decode with ``--disaggregation-decode-enable-radix-cache`` on the same backend keys its tree on whole chunks of its
per-rank ``--chunked-prefill-size`` (serving_policy.decode_radix_prefill_chunks), so a hit ends on a chunk boundary of
any prefill whose chunk divides the tree's, and the prefill computes the rest with a cold prefill's own forwards. That
is only bit-exact if the cached KV came from a prefill that computes the same bits, one with the same KV fingerprint.
Upstream's protocol carries none of this, so each prefill rank advertises it through its bootstrap server:

* ``CommonKVManager._register_topology_row``: an on-grid prefill rank (chunk-keyed tree or ``--disable-radix-cache``)
  adds ``kv_chunk_grid``, its chunk, and ``kv_numerics``: its tree (which changes the prefill graph's block table,
  graph_runner's ``prefill_table_pad``) and its TP, EP, attention TP and DP, and PP layout (which changes how sums
  reduce). A prefill on the page-keyed tree resumes off the grid and advertises nothing.
* ``CommonKVBootstrapServer._handle_route_put`` and ``_handle_route_get``: the server keeps the first accepted
  fingerprint, refuses a later rank with another (HTTP 400, like upstream's DSpark layout check), and adds it to the
  info answer. Without a chunk both stay byte-identical to upstream.
* ``PrefillServerInfo``: replaced by a subclass with the two fields on every PD decode, since upstream's dataclass
  raises TypeError on an unknown key.
* ``CommonKVManager._resolve_rank_mapping``: the chunk-keyed decode refuses a prefill with no chunk, or one whose chunk
  does not divide the tree's, with PrefillIncompatibleError (sglang_patches.decode_incompatible_prefill).
* ``DecodePreallocQueue._init_receiver``: once paired, the request's radix ``extra_key`` becomes its own joined with
  the prefill's fingerprint (like SGLang's elastic-EP namespace), so no tree node mixes KV of two fingerprints.
  ``Req.build_rebootstrap_payload`` sends the request's own key, so the namespace stays on the decode.

All of it reads only advertised info and the launch config, never which prefill a rank reached first, so every decode
rank refuses the same prefills and the attention-TP ranks of a DP group keep the same tree.

Left out of the fingerprint: what first contact already checks (KV dtype, page size), what the checkpoint fixes
(quantization, platform.check_quantization), and what the operator keeps equal: the prefills behind one decode serve
the same checkpoint at the same ``--context-length`` (which sets the block table's width) from the same image.

Upstream: SGLang would carry the fingerprint in its bootstrap protocol next to a chunk-keyed radix tree. SGLang's Rust
bootstrap registry (``SGLANG_RUST_SERVER``) drops the keys, so a chunk-keyed decode refuses every prefill behind it.

Targets are md5-pinned against upstream SGLang 43881a7, one table per registration."""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_COMMON = "sglang.srt.disaggregation.common.conn"
_NIXL = "sglang.srt.disaggregation.nixl.conn"
_DECODE = "sglang.srt.disaggregation.decode"
_REQ = "sglang.srt.managers.schedule_batch.Req"
# the prefill's chunk and the rest of its KV fingerprint (advertised_numerics), as the registration, the info answer and
# PrefillServerInfo carry them
GRID_KEY = "kv_chunk_grid"
NUMERICS_KEY = "kv_numerics"
# a bootstrap server's (chunk, numerics), set by the first registration it accepts
_SERVER_ADVERT = "_neuron_kv_fingerprint"
# a request's extra_key as it reached the decode, before its prefill's fingerprint joined it
_OWN_EXTRA_KEY = "_neuron_own_extra_key"
# the decode's info query names every rank -1 (try_ensure_parallel_info)
_INFO_QUERY = ("prefill_dp_rank", "prefill_cp_rank", "target_tp_rank", "target_pp_rank")

# (module, qualified name): md5 of upstream's source, for each process role's hooks
_SERVER_TARGETS_MD5 = {
    (_COMMON, "CommonKVBootstrapServer._handle_route_put"): "e6d3daff4260d7db9207db63c2b3c3ef",
    (_COMMON, "CommonKVBootstrapServer._handle_route_get"): "a672abb72ddbed3b3e52cb37e6831b7c",
    # awaits self._handle_route_put and _handle_route_get by attribute, so the registry's synchronous wrapper of each
    # hook returns the coroutine it awaits
    (_COMMON, "CommonKVBootstrapServer._handle_route"): "f01eae2a78dcb3af86d7f1921241e783",
    # the NIXL backend's server inherits both handlers
    (_NIXL, "NixlKVBootstrapServer"): "71d4dff3dcca56af4058b9b2eef0cda2",
    ("sglang.srt.disaggregation.utils", "get_kv_class"): "611e883df7a0e779693b07c892ea71c9",
}
_PREFILL_TARGETS_MD5 = {
    (_COMMON, "CommonKVManager._register_topology_row"): "623206d3d4e08ee4e11f7b2771e954e8",
    # sends every row (the Rust registry's DP gather too) through _register_topology_row, from the KV manager's
    # constructor, which runs after the model set the prefill chunk grid
    (_COMMON, "CommonKVManager.register_to_bootstrap"): "0abfe2e7802db1970be4ee6ccfcb9e45",
}
_DECODE_TARGETS_MD5 = {
    (_COMMON, "PrefillServerInfo"): "48c66f9592eff5eb430d870febbcfc55",
    # builds PrefillServerInfo(**answer) from its module's globals, which the replacement rebinds
    (_COMMON, "CommonKVManager.try_ensure_parallel_info"): "15b2d05442275911ca74f44980700aac",
}
_CHUNK_KEYED_DECODE_TARGETS_MD5 = {
    (_COMMON, "CommonKVManager._resolve_rank_mapping"): "70d83eef6dcb74b7cce329808eecba2f",
    # runs upstream's own checks, then _resolve_rank_mapping, and caches the info only once it returns
    (_COMMON, "CommonKVManager.try_ensure_parallel_info"): "15b2d05442275911ca74f44980700aac",
    # inits the request's receiver
    (_DECODE, "DecodePreallocQueue._init_receiver"): "a87b638f3e357857174fe28cd2593dd1",
    # binds the receiver to its prefill's cached info, prefill_info
    (_COMMON, "CommonKVReceiver.init"): "fe1b27f354afbc1dfb46cf7dfa0be764",
    # init every receiver but a fake transfer's (no prefill) through _init_receiver, once its prefill's info is cached
    (_DECODE, "DecodePreallocQueue.add"): "a3fe64b4f09123d07f1d7bcf70db32b5",
    (_DECODE, "DecodePreallocQueue._resolve_pending_reqs"): "f77910873e66c71e120a97aa20b3a452",
    # matches a request only once its initialized receiver waits for input
    (_DECODE, "DecodePreallocQueue.pop_preallocated"): "b42db6c8d0b02438782c25253dc8a7da",
    # the decode's one match: under the request's extra_key (match_kv_cache), and a rebootstrap's re-bound root under it
    (_DECODE, "DecodePreallocQueue._match_prefix_and_lock"): "f3a7dc22286b4542f3e2d3a042bdc3f1",
    ("sglang.srt.mem_cache.common", "match_kv_cache"): "14bbc226d6cac76804d71f9478e1f00f",
    (_DECODE, "_bind_root_prefix"): "7a81ca1766d17b1501b23a87ce614e32",
    # the match keys the request's tokens on its extra_key, and so does every insert
    ("sglang.srt.mem_cache.common", "_req_radix_key"): "06537cd4b740f3cb416258650acd671f",
    ("sglang.srt.mem_cache.unified_radix_cache", "UnifiedRadixCache.checkpoint_into_tree"):
        "4082530bb41b73ce76ed16c2e1531de1",
    # sends the request's extra_key to the prefill
    ("sglang.srt.managers.schedule_batch", "Req.build_rebootstrap_payload"): "5806078eb40812545679873977011f82",
}


# The prefill
def advertised_chunk() -> int:
    """The chunk this prefill rank advertises: its grid unit on the chunk-keyed tree or without a tree, where every
    prompt is on the grid; 0 on SGLang's page-keyed tree or without a grid."""
    from sglang.srt.runtime_context import get_memory

    from sglang_neuron import serving_policy
    from sglang_neuron.sglang_patches import prefill_chunk_grid

    chunk = prefill_chunk_grid.grid_unit()
    if chunk is None:
        return 0
    return chunk if serving_policy.current().chunk_keyed_tree or get_memory().disable_radix_cache else 0


def advertised_numerics() -> dict:
    """The rest of the KV fingerprint: the tree, and the parallel layout the model's graphs are built over (models.host
    reads TP, EP and attention DP; distributed.pipeline the stages)."""
    from sglang_neuron import serving_policy, sglang_tree

    pc = sglang_tree.parallel_config()
    return {"tree": "chunk" if serving_policy.current().chunk_keyed_tree else "none", "tp_size": int(pc.tp_size),
            "ep_size": int(pc.ep_size or 1), "attn_tp_size": int(pc.attn_tp_size),
            "attn_dp_size": sglang_tree.attention_dp_size(pc), "pp_size": int(pc.pp_size)}


def fingerprint(chunk: int, numerics: Optional[dict]) -> str:
    """A prefill's KV fingerprint as text, chunk first and the rest by name; "none" without a chunk. Never holds "|"
    (namespaced_extra_key)."""
    if not chunk:
        return "none"
    return ",".join([f"{GRID_KEY}={chunk}", *(f"{name}={value}" for name, value in sorted((numerics or {}).items()))])


def register_topology_row(original_fn, self, url, payload, *args, **kwargs):
    """AROUND CommonKVManager._register_topology_row: add the rank's fingerprint to its registration; upstream's payload
    without a chunk."""
    chunk = advertised_chunk()
    if chunk:
        payload = {**payload, GRID_KEY: chunk, NUMERICS_KEY: advertised_numerics()}
    return original_fn(self, url, payload, *args, **kwargs)


# The prefill's bootstrap server
async def handle_route_put(original_fn, self, request, *args, **kwargs):
    """AROUND CommonKVBootstrapServer._handle_route_put: the first accepted registration sets the server's fingerprint.
    A later rank with another is refused before it registers, so the server never counts every rank and answers no
    decode. A registration upstream refuses or fails on (a malformed row) sets none."""
    from aiohttp import web

    # aiohttp keeps the body it read, and upstream's handler parses the same bytes again
    data = await request.json()
    advert = (int(data.get(GRID_KEY, 0)), data.get(NUMERICS_KEY))
    # read and set with no await between them, so of two first registrations in flight at once only one sets it
    first = getattr(self, _SERVER_ADVERT, None)
    if first is None:
        setattr(self, _SERVER_ADVERT, advert)
    elif advert != first:
        reason = (f"KV fingerprint differs across prefill ranks: {fingerprint(*first)} registered first, "
                  f"{fingerprint(*advert)} now")
        # upstream's registering rank logs only the status it got
        logger.error("[neuron] bootstrap server refused a prefill rank: %s", reason)
        return web.Response(text=reason, status=400)
    try:
        response = await original_fn(self, request, *args, **kwargs)
    except BaseException:
        _forget_unaccepted(self, first)
        raise
    if response.status != 200:
        _forget_unaccepted(self, first)
    elif getattr(self, _SERVER_ADVERT, None) is None:
        # a first registration that failed while this one was in flight cleared the fingerprint this one shares
        setattr(self, _SERVER_ADVERT, advert)
    return response


def _forget_unaccepted(server, first) -> None:
    """A registration upstream refused or failed on sets no fingerprint: the server's is the first one it accepts."""
    if first is None and not server._registered_count:
        setattr(server, _SERVER_ADVERT, None)


async def handle_route_get(original_fn, self, request, *args, **kwargs):
    """AROUND CommonKVBootstrapServer._handle_route_get: add the server's fingerprint to the info answer (every rank
    -1). Other answers, and all answers without a chunk, stay upstream's."""
    from aiohttp import web

    response = await original_fn(self, request, *args, **kwargs)
    chunk, numerics = getattr(self, _SERVER_ADVERT, (0, None))
    # upstream answers 200 only to a query whose four ranks it parsed as integers
    if not chunk or response.status != 200 or any(int(request.query[name]) != -1 for name in _INFO_QUERY):
        return response
    payload = json.loads(response.body)
    payload[GRID_KEY], payload[NUMERICS_KEY] = chunk, numerics
    return web.json_response(payload, status=200)


# The decode
def server_info_class():
    """Upstream's PrefillServerInfo plus the advertised fingerprint (``kv_chunk_grid``, 0 for none, and
    ``kv_numerics``), under upstream's name so the decode's logs read the same."""
    from sglang.srt.disaggregation.common.conn import PrefillServerInfo

    @dataclasses.dataclass
    class ChunkedPrefillServerInfo(PrefillServerInfo):
        kv_chunk_grid: int = 0
        kv_numerics: Optional[dict] = None

        def __post_init__(self):
            super().__post_init__()
            self.kv_chunk_grid = int(self.kv_chunk_grid)

    ChunkedPrefillServerInfo.__name__ = ChunkedPrefillServerInfo.__qualname__ = PrefillServerInfo.__qualname__
    return ChunkedPrefillServerInfo


def tree_chunk() -> int:
    """The chunk the chunk-keyed decode's tree keys: its per-rank --chunked-prefill-size (SGLang splits the flag over
    the attention-DP groups)."""
    from sglang.srt.runtime_context import get_schedule

    chunk = int(get_schedule().chunked_prefill_size or 0)
    if chunk <= 0:
        raise ValueError(f"--radix-cache-backend neuron-chunk on a decode needs --chunked-prefill-size > 0, got "
                         f"{chunk}")
    return chunk


def pairing_problem(prefill_chunk: int, tree_chunk: int, attention_dp: int) -> Optional[str]:
    """Why a decode whose tree keys ``tree_chunk``-token chunks over ``attention_dp`` groups can reuse no KV from a
    prefill advertising ``prefill_chunk`` (0: none); None when they pair."""
    if prefill_chunk <= 0:
        return ("this decode's radix tree holds only KV computed for whole prefill chunks, and this prefill advertises "
                "no chunk grid: launch it with --radix-cache-backend neuron-chunk or --disable-radix-cache. A prefill "
                "behind SGLang's Rust bootstrap registry (SGLANG_RUST_SERVER) advertises none, since the registry "
                "drops it")
    if tree_chunk % prefill_chunk == 0:
        return None
    over = f" (its --chunked-prefill-size over {attention_dp} attention-DP groups)" if attention_dp > 1 else ""
    return (f"this decode's radix tree keys whole chunks of {tree_chunk} tokens{over}, not a whole number of this "
            f"prefill's {prefill_chunk}-token chunks, so a hit would end inside a chunk whose KV the prefill computed "
            f"in one forward with the tokens after it: launch the decode with a --chunked-prefill-size that is a "
            f"multiple of {prefill_chunk * attention_dp}")


def check_pairing(original_fn, self, info, *args, **kwargs):
    """AROUND CommonKVManager._resolve_rank_mapping on the chunk-keyed decode: refuse a prefill whose KV the tree cannot
    reuse (PrefillIncompatibleError) before upstream maps and caches its info. Reads only advertised info and the launch
    config, so every rank refuses the same prefills."""
    from sglang_neuron import sglang_tree
    from sglang_neuron.sglang_patches.decode_incompatible_prefill import PrefillIncompatibleError

    problem = pairing_problem(info.kv_chunk_grid, tree_chunk(),
                              sglang_tree.attention_dp_size(sglang_tree.parallel_config()))
    if problem is not None:
        raise PrefillIncompatibleError(problem)
    return original_fn(self, info, *args, **kwargs)


def namespaced_extra_key(own: Optional[str], prefill_fingerprint: str) -> str:
    """A request's radix extra_key on the chunk-keyed decode: ``own`` joined with the prefill's fingerprint, as SGLang
    joins the elastic-EP namespace (Scheduler._maybe_namespace_elastic_radix_cache). The fingerprint follows the last
    "|", so different fingerprints never share a key."""
    return f"{own}|{prefill_fingerprint}" if own else prefill_fingerprint


def namespace_request(original_fn, self, decode_req, *args, **kwargs):
    """AROUND DecodePreallocQueue._init_receiver on the chunk-keyed decode: once the receiver has paired, join the
    request's own extra_key with its prefill's fingerprint. A receiver that failed before pairing leaves the key (its
    request fails). A rebootstrap pairs a new receiver and joins the own key again."""
    result = original_fn(self, decode_req, *args, **kwargs)
    # the info upstream's init bound the receiver to; the table's entry can go meanwhile (_handle_node_failure)
    info = getattr(decode_req.kv_receiver, "prefill_info", None)
    if info is not None:
        req = decode_req.req
        own = vars(req).setdefault(_OWN_EXTRA_KEY, req.extra_key)
        req.extra_key = namespaced_extra_key(own, fingerprint(info.kv_chunk_grid, info.kv_numerics))
    return result


def rebootstrap_payload(original_fn, self, *args, **kwargs):
    """AROUND Req.build_rebootstrap_payload on the chunk-keyed decode: a retracted request's recompute carries its own
    extra_key, not the decode's namespace."""
    payload = original_fn(self, *args, **kwargs)
    if _OWN_EXTRA_KEY in vars(self):
        payload = {**payload, "extra_key": vars(self)[_OWN_EXTRA_KEY]}
    return payload


def register_bootstrap_server(registry, around) -> None:
    """The bootstrap server's hooks, registered in every process: the server runs in the prefill's launching process
    and relays the fingerprint whatever the tree."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_SERVER_TARGETS_MD5, __name__)
    registry.register(f"{_COMMON}.CommonKVBootstrapServer._handle_route_put", handle_route_put, around)
    registry.register(f"{_COMMON}.CommonKVBootstrapServer._handle_route_get", handle_route_get, around)


def register_prefill(registry, around) -> None:
    """Every PD prefill advertises its KV fingerprint."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_PREFILL_TARGETS_MD5, __name__)
    registry.register(f"{_COMMON}.CommonKVManager._register_topology_row", register_topology_row, around)


def register_decode(registry, replace) -> None:
    """Every PD decode parses a prefill's fingerprint (an on-grid prefill without a tree advertises one too)."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_DECODE_TARGETS_MD5, __name__)
    registry.register(f"{_COMMON}.PrefillServerInfo", server_info_class(), replace)


def register_chunk_keyed_decode(registry, around) -> None:
    """The chunk-keyed decode pairs with each prefill on its chunk and keys each request's KV under its prefill's
    fingerprint (serving_policy.decode_radix_prefill_chunks)."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_CHUNK_KEYED_DECODE_TARGETS_MD5, __name__)
    registry.register(f"{_COMMON}.CommonKVManager._resolve_rank_mapping", check_pairing, around)
    registry.register(f"{_DECODE}.DecodePreallocQueue._init_receiver", namespace_request, around)
    registry.register(f"{_REQ}.build_rebootstrap_payload", rebootstrap_payload, around)
