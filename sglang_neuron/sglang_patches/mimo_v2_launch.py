# SPDX-License-Identifier: Apache-2.0
"""Two adjustments SGLang makes for a MiMo-V2 checkpoint that the Neuron graphs need otherwise.

1. **The fused-QKV TP check.** ``model_hook.handle_model_specific_adjustments`` refuses a ``MiMoV2ForCausalLM``
   launch whose attention TP does not divide the checkpoint's fused-QKV shard count, which
   ``get_mimo_v2_fused_qkv_expected_tp_size`` gives (its 4 KV heads), because SGLang's own MiMoV2 loader reads each
   rank's rows as stored. The Neuron graphs re-shard the fused QKV from the stored layout for any attention TP
   (models/mimo_v2/weights.fused_qkv_loader), so on this platform that function answers no expected size (the
   adjustment imports it when it runs, and SGLang's own MiMoV2 model, its other reader, is not loaded here).
2. **The KV pool's head dims.** ``ModelConfig._derive_model_shapes`` gives MiMoV2 K heads of 192 and V heads of 128
   channels (``head_dim``, ``v_head_dim``, ``swa_head_dim``, ``swa_v_head_dim``), which SGLang sizes its pools and
   the hybrid SWA pool's kwargs by. The Neuron pool holds one head dim for K and V on every layer
   (mem_cache/memory_pool._refuse_other_head_dims) and the graphs run every head at
   models/mimo_v2/config.KV_HEAD_DIM (256: nkilib's attention kernels take whole multiples of 128 above 128), so after
   the derivation all four read KV_HEAD_DIM, and the pool, its sizing and the host pools agree with the graphs. The
   graphs read their own dims from the checkpoint's config, which is left as it is.

3. **The MTP layers' KV.** Under ``--speculative-algorithm NEURON_MTP`` the graphs hold the checkpoint's MTP
   layers, sliding-window layers whose KV is the pool's layers after the target's (models/mimo_v2/mtp.py; the host
   sizes the pool deeper). ``ModelConfig._derive_hybrid_model`` splits the target's layers between the
   window and full pools from ``hybrid_layer_pattern``, which has no entry for them, so after the derivation the MTP
   layers the draft runs (from ``num_hidden_layers`` on) are listed among the sliding ones.

The targets are md5-pinned against upstream SGLang 43881a7."""

from __future__ import annotations

EXPECTED_TP = "sglang.srt.configs.model_config.get_mimo_v2_fused_qkv_expected_tp_size"
_EXPECTED_TP_MD5 = "ffe9726373f082e7a56f83be9f21b518"
SHAPES = "sglang.srt.configs.model_config.ModelConfig._derive_model_shapes"
_SHAPES_MD5 = "64913dc453d37806c81fb665fccd9802"
_HEAD_DIMS = ("head_dim", "v_head_dim", "swa_head_dim", "swa_v_head_dim")
HYBRID = "sglang.srt.configs.model_config.ModelConfig._derive_hybrid_model"
_HYBRID_MD5 = "4fc5ddc6ee6b948a2bffc07ede1501dc"


def _is_mimo_v2(architectures) -> bool:
    from sglang_neuron.models.mimo_v2.config import ARCHITECTURES

    return any(a in ARCHITECTURES for a in (architectures or ()))


def no_expected_tp_size(hf_config):
    """REPLACE get_mimo_v2_fused_qkv_expected_tp_size: no fused-QKV TP is required (module docstring)."""
    return None


def derive_model_shapes(original_fn, self, *args, **kwargs):
    result = original_fn(self, *args, **kwargs)
    if _is_mimo_v2(getattr(self.hf_config, "architectures", None)):
        from sglang_neuron.models.mimo_v2.config import KV_HEAD_DIM

        for name in _HEAD_DIMS:
            if getattr(self, name, None) is not None:
                setattr(self, name, KV_HEAD_DIM)
    return result


def derive_hybrid_model(original_fn, self, *args, **kwargs):
    result = original_fn(self, *args, **kwargs)
    from sglang_neuron.speculative.neuron_mtp import is_neuron_mtp

    ids = getattr(self, "swa_attention_layer_ids", None)
    if (_is_mimo_v2(getattr(self.hf_config, "architectures", None)) and ids is not None
            and is_neuron_mtp(getattr(self, "speculative_algorithm", None)) and not self.is_draft_model):
        from sglang_neuron.models.mimo_v2.mtp import num_layers

        first = int(self.hf_text_config.num_hidden_layers)
        self.swa_attention_layer_ids = [*ids, *(i for i in range(first, first + num_layers()) if i not in ids)]
    return result


def register(registry, around, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    for target, md5 in ((EXPECTED_TP, _EXPECTED_TP_MD5), (SHAPES, _SHAPES_MD5), (HYBRID, _HYBRID_MD5)):
        module_name = "sglang.srt.configs.model_config"
        source_guard.check(source_guard.resolve(module_name, target.removeprefix(module_name + ".")), md5, target,
                           __name__)
    registry.register(EXPECTED_TP, no_expected_tp_size, replace)
    registry.register(SHAPES, derive_model_shapes, around)
    registry.register(HYBRID, derive_hybrid_model, around)
