# SPDX-License-Identifier: Apache-2.0
"""What SGLang 43881a7 needs for a Hy4-Flash checkpoint (``HYV4FlashForConditionalGeneration``), which it predates
(upstream added the model in sgl-project/sglang#51668):

1. **The config type.** The checkpoint's ``model_type`` is ``hy_v4_flash`` and ships no configuration code. The
   config classes (a text-only copy of upstream's, models/hy_v4_flash/hf_config.py) are registered with SGLang's
   config registry and Transformers' AutoConfig in every process of a Neuron launch, before any reads the config.
2. **The KV pool's geometry.** Every layer caches one MLA row a token, its 512-wide latent and 64 RoPE'd key
   channels. The Neuron graphs keep that row in the platform's MHA pool as one KV head of 288 channels, the row's
   first half in K and its second in V (models/hy_v4_flash/attention.py), so that the sliding layers can use SGLang's
   hybrid SWA window pool, which is built over the MHA pool only. After ``ModelConfig._derive_model_shapes`` a Hy4-Flash
   config is MHA with K and V heads of ``ROW_HALF``; at attention TP 32 the config's 32 KV heads give each rank one.
3. **The window pool.** ``ModelConfig._derive_hybrid_model`` recognizes no Hy4-Flash layer as sliding (its
   ``layer_types`` name the others ``deepseek_sparse_attention``, not ``full_attention``), so before the derivation the
   text config opts in the generic way (``is_hybrid_swa`` and a ``hybrid_layer_pattern`` of 1 for each sliding layer);
   ``--disable-hybrid-swa-memory`` still keeps one pool.

Upstream: drop the module once the pinned SGLang carries Hy4-Flash. The targets are md5-pinned against SGLang
43881a7."""

from __future__ import annotations

SHAPES = "sglang.srt.configs.model_config.ModelConfig._derive_model_shapes"
_SHAPES_MD5 = "64913dc453d37806c81fb665fccd9802"
HYBRID = "sglang.srt.configs.model_config.ModelConfig._derive_hybrid_model"
_HYBRID_MD5 = "4fc5ddc6ee6b948a2bffc07ede1501dc"
# half of an MLA row (kv_lora_rank + qk_rope_head_dim = 576): one KV head of the MHA pool's K, and one of its V
ROW_HALF = 288


def _is_hy4_flash(architectures) -> bool:
    from sglang_neuron.models.hy_v4_flash.config import ARCHITECTURES

    return any(a in ARCHITECTURES for a in (architectures or ()))


def register_config() -> None:
    """Register the Hy4-Flash config types with SGLang's registry and AutoConfig; again is a no-op."""
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from sglang_neuron.models.hy_v4_flash.hf_config import HYV4FlashConfig, HYV4FlashTextConfig

    try:
        from sglang.srt.utils.hf_transformers import common

        common._CONFIG_REGISTRY.setdefault(HYV4FlashConfig.model_type, HYV4FlashConfig)
    except ImportError:  # an SGLang without the registry module: AutoConfig alone
        pass
    for cls in (HYV4FlashConfig, HYV4FlashTextConfig):
        if cls.model_type not in CONFIG_MAPPING:
            AutoConfig.register(cls.model_type, cls)


def derive_model_shapes(original_fn, self, *args, **kwargs):
    result = original_fn(self, *args, **kwargs)
    if _is_hy4_flash(getattr(self.hf_config, "architectures", None)):
        from sglang.srt.configs.model_config import AttentionArch

        text = self.hf_text_config
        if int(text.kv_lora_rank) + int(text.qk_rope_head_dim) != 2 * ROW_HALF:
            raise ValueError(f"Hy4-Flash: an MLA row of {text.kv_lora_rank} + {text.qk_rope_head_dim} channels is not "
                             f"two KV heads of {ROW_HALF}")
        self.attention_arch = AttentionArch.MHA
        for name in ("head_dim", "v_head_dim", "swa_head_dim", "swa_v_head_dim"):
            setattr(self, name, ROW_HALF)
    return result


def derive_hybrid_model(original_fn, self, *args, **kwargs):
    if _is_hy4_flash(getattr(self.hf_config, "architectures", None)):
        from sglang_neuron.models.hy_v4_flash.config import SLIDING

        text = self.hf_text_config
        text.is_hybrid_swa = True
        text.hybrid_layer_pattern = [1 if kind == SLIDING else 0 for kind in text.layer_types]
    return original_fn(self, *args, **kwargs)


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    register_config()
    module_name = "sglang.srt.configs.model_config"
    for target, md5 in ((SHAPES, _SHAPES_MD5), (HYBRID, _HYBRID_MD5)):
        source_guard.check(source_guard.resolve(module_name, target.removeprefix(module_name + ".")), md5, target,
                           __name__)
    registry.register(SHAPES, derive_model_shapes, around)
    registry.register(HYBRID, derive_hybrid_model, around)
