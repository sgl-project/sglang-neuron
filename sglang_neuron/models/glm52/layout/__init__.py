# SPDX-License-Identifier: Apache-2.0
"""The host side of the GLM-5.2 giga kernels' layout contract: the plugin's code that puts the checkpoint, the caches
and each step's tables in the layouts the kernels under sglang_neuron/models/glm52/kernels take (PROVENANCE.md there).

The kernels' own host code has functions these modules leave out on purpose, each needing a module SGLang-Neuron has no
equivalent of or serving something the plugin does not: quark_compat has no adapt_prefill_load, whose model-id
fallback lists the checkpoint's tensors by name, where the plugin's SafetensorsCheckpoint takes only a local directory
and lists no names; prefill/pipeline has no stage_layer_split, whose default split SGLang's get_pp_indices does not
reproduce (SGLang gives an uneven split's extra layers to the last stages, 19, 19, 20, 20 for 78 layers over 4 stages,
where that split gives 19, 20, 20, 19); weight_loaders_mtp has no external decode NEFF hooks (the plugin serves no
external decode NEFF); and mx_tiles keeps only the MX tile helpers the prefill loaders share with GPT-OSS, none of
the GPT-OSS loaders around them.

Shared by both kernels, at this package's top:
* config: the geometry of a Hugging Face config and of a deployment's knobs;
* quark_compat: which of the two Quark exports a checkpoint is, and the refusal of the AttnFP8 one, which no role serves;
* glm_kv_row and glm_kv_balance: the static latent row's format (the prefill's pool rows and the decode's caches) and
  its latent gain;
* rope_split: the two-level RoPE tables.

decode/, the decode giga kernel (and its MTP draft layer):
* gk_geometry: the sharding geometry, the rank decomposition and the structural tensors of each step;
* weight_loaders_mxfp4: the per-rank weight store, built from the MXFP4 or the Quark checkpoint;
* weight_loaders_mtp: the MTP draft layer's per-rank weight store;
* gk_kv_cache: the latent and indexer caches;
* idx_pool: the chunk-dealt indexer pool's membership, its deal and its per-step tables;
* idx_pool_torch: the same per-step tables as torch ops of fixed shape, which a traced graph builds on the device from
  its device positions (the MTP verify graph's).

prefill/, the prefill giga kernel, which both prefill points (PP4 and PP16) run:
* config, weight_loaders and pipeline;
* mx_tiles: the MX tile helpers its weight loaders share with GPT-OSS;
* mtp_kv: the prompt rows of the MTP layer, and its checkpoint keys.
"""
