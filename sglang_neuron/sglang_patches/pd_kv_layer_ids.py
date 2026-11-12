# SPDX-License-Identifier: Apache-2.0
"""Name the global model layer of every KV entry a Neuron pool registers for PD, so a pipelined prefill stage's entries
pair with the decode's entries of the same layers.

SGLang pairs KV entries by layer id when both peers publish ids, and otherwise refuses peers whose entry counts
differ, as a pipelined prefill stage's and a whole decode's do. Upstream's ``build_kv_layer_ids`` returns no ids for
the Neuron pools, so a pool whose class sets ``publishes_kv_layer_ids`` publishes its own (``get_kv_layer_ids``: the
MHA pool's K entries' layers, then the V entries'; a standalone MLA pool's one entry per layer). Between two whole
stacks the ids give the positional pairing, so a launch without PP is unchanged. A hybrid linear-attention pool (MLA
and recurrent-state layers) gets its ids from upstream, which reads them from the pool. A draft pool is refused: a
Neuron server serves no draft model.

The head-slice transfer (a decode at another attention TP) pairs a stage's entries in
``CommonKVManager.get_mha_kv_ptrs_with_pp``, which offsets the V entries wrongly for an uneven layer split; the wrapper
pairs them by layer instead, and refuses a pipelined decode stage of other layers."""

from __future__ import annotations

_UTILS = "sglang.srt.disaggregation.utils"
TARGET = f"{_UTILS}.build_kv_layer_ids"
CALL_SITES = ("sglang.srt.disaggregation.prefill", "sglang.srt.disaggregation.decode")
# (module, qualified name): md5 of the source
_TARGETS_MD5 = {
    (_UTILS, "build_kv_layer_ids"): "cf00a4da1b7083e667ca3be3179991ef",
    ("sglang.srt.disaggregation.prefill", "PrefillBootstrapQueue._init_kv_manager"): "a64cc26a13904432901c1b256ae1e2d0",
    ("sglang.srt.disaggregation.decode", "DecodePreallocQueue._init_kv_manager"): "a49fb126873eaddb433c4243112060ca",
    # where upstream's owner region takes its ids from (none for the Neuron pool)
    ("sglang.srt.mem_cache.static_kv_storage", "_static_owner_region"): "2763b5bde4ac9425e9c9662a90acb18b",
    ("sglang.srt.disaggregation.common.conn", "CommonKVManager.get_mha_kv_ptrs_with_pp"):
        "3410b1ab21dde5f51475310b500552b9",
}
STAGE_POINTERS = "sglang.srt.disaggregation.common.conn.CommonKVManager.get_mha_kv_ptrs_with_pp"


def build_kv_layer_ids(original_fn, *, token_to_kv_pool, draft_token_to_kv_pool, num_draft_entries: int,
                       num_hidden_layers: int, storage):
    """AROUND build_kv_layer_ids (the prefill's and the decode's KV registration): the global layer ids of a pool that
    publishes them (``publishes_kv_layer_ids``) where upstream reports none."""
    ids = original_fn(token_to_kv_pool=token_to_kv_pool, draft_token_to_kv_pool=draft_token_to_kv_pool,
                      num_draft_entries=num_draft_entries, num_hidden_layers=num_hidden_layers, storage=storage)
    if ids or not getattr(type(token_to_kv_pool), "publishes_kv_layer_ids", False):
        return ids
    if draft_token_to_kv_pool is not None or num_draft_entries:
        raise RuntimeError("a Neuron PD server serves no draft model, so its KV registration names no draft entries")
    return token_to_kv_pool.get_kv_layer_ids()


def mha_kv_ptrs_with_pp(original_fn, self, src_kv_ptrs, dst_kv_ptrs):
    """AROUND CommonKVManager.get_mha_kv_ptrs_with_pp: when the stage's layer count does not divide the decode's, pair
    its K and V entries with the decode's of the same global layers, where upstream would assume draft entries (a
    Neuron decode has none). The decode must hold the whole stack; every other case is upstream's."""
    layers, total = len(src_kv_ptrs) // 2, len(dst_kv_ptrs) // 2
    if not layers or layers >= total or total % layers == 0:
        return original_fn(self, src_kv_ptrs, dst_kv_ptrs)
    start = int(self.kv_args.prefill_start_layer)
    whole = _model_layers(self)
    if total != whole or start + layers > total:
        raise RuntimeError(f"a prefill stage of layers [{start}, {start + layers}) pairs by head slice with a decode of "
                           f"the whole stack ({whole} layers), not one of {total}: launch a pipelined decode with the "
                           "prefill's --pp-size and SGLANG_PP_LAYER_PARTITION")
    return (list(src_kv_ptrs[:layers]), list(src_kv_ptrs[layers:]), list(dst_kv_ptrs[start:start + layers]),
            list(dst_kv_ptrs[total + start:total + start + layers]), layers)


def _model_layers(manager) -> int:
    """The checkpoint's layer count, read once per KV manager (its config, which every role serves)."""
    layers = getattr(manager, "_neuron_model_layers", None)
    if layers is None:
        from sglang_neuron.models import catalog

        layers = manager._neuron_model_layers = int(catalog.checkpoint_config(manager.server_args)["num_hidden_layers"])
    return layers


def register(registry, around) -> None:
    import importlib

    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    upstream = importlib.import_module(_UTILS).build_kv_layer_ids
    for module in CALL_SITES:
        if getattr(importlib.import_module(module), "build_kv_layer_ids", None) is not upstream:
            raise RuntimeError(f"{module}.build_kv_layer_ids is not {TARGET}; re-validate {__name__} against this "
                               "SGLang before serving it")
    registry.register(TARGET, build_kv_layer_ids, around)
    registry.register(STAGE_POINTERS, mha_kv_ptrs_with_pp, around)
