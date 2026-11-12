# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2 (GLM-MoE-DSA) model config the prefill stage graphs are traced from.

GLM-5.2 is Multi-head Latent Attention (MLA), a DSA sparse-attention indexer, and a grouped-sigmoid MoE with
one shared expert, with cross-layer top-k sharing: a "shared" layer reuses the top-k token selection of the previous
"full" indexer layer. The defaults are the served checkpoint's geometry; a checkpoint's config.json overrides the
fields it declares (``from_configs``), and ``pipeline.stage_config`` cuts a PP stage's config from the full one.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from sglang_neuron.models.glm52.options import Glm52PrefillOptions


def _check_layer_types(name: str, types: list, allowed: tuple, num_layers: int) -> None:
    """Refuse a per-layer schedule a checkpoint supplies that is not one entry of ``allowed`` per layer: the stage
    builders branch on these strings (a "shared" layer holds no indexer weights, a "sparse" one the MoE), so an entry
    of another spelling would silently build the wrong layer kind, and a list of another length the wrong stage."""
    if len(types) != num_layers:
        raise ValueError(f"{name} lists {len(types)} layers of {num_layers}")
    bad = {i: t for i, t in enumerate(types) if t not in allowed}
    if bad:
        raise ValueError(f"{name}: the layers {bad} are not one of {allowed}")


@dataclass
class GlmMoeDsaConfig:
    """Configuration for the GLM-MoE-DSA (GLM-5.2) model.

    Architecture overview:
    - Transformer decoder with MLA attention and a DSA indexer per layer.
    - The first `first_k_dense_replace` layers use a dense MLP. The rest use MoE.
    - MoE routes each token to `num_experts_per_tok` of `n_routed_experts`
      experts with a grouped-sigmoid router, plus `n_shared_experts` shared
      experts that run for every token.
    - Interleaved RoPE in both the main attention and the indexer.
    """

    # Core transformer
    vocab_size: int = 154880
    hidden_size: int = 6144
    intermediate_size: int = 12288  # Dense-layer MLP hidden size
    num_hidden_layers: int = 78
    # The checkpoint index of the MTP layer, whose prompt rows the last stage writes. None means
    # `num_hidden_layers`: the MTP layer follows the last decoder layer. A test that keeps
    # the first layers of a full checkpoint sets the full layer count here.
    mtp_layer_index: int | None = None
    # The checkpoint index of decoder layer 0. A pipeline stage (or a test) that keeps the
    # layers [k, k + num_hidden_layers) of a full checkpoint sets k; the per-layer lists
    # then hold the types of those layers.
    first_layer_index: int = 0
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    rms_norm_eps: float = 1e-5
    # The MLA q_a and kv_a layernorms use a different eps from the decoder norms.
    mla_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    torch_dtype: torch.dtype = torch.bfloat16

    # MLA attention
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    attention_bias: bool = False

    # DSA indexer
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    # Per-layer indexer mode. "full" runs the indexer. "shared" reuses the
    # top-k of the previous full layer. `None` derives the schedule from
    # `index_topk_freq` / `index_skip_topk_offset` in `__post_init__`.
    indexer_types: list[str] | None = None
    index_topk_freq: int = 4
    index_skip_topk_offset: int = 3

    # MoE
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    first_k_dense_replace: int = 3
    # Per-layer MLP type: "dense" or "sparse". `None` derives it from
    # `first_k_dense_replace` in `__post_init__`.
    mlp_layer_types: list[str] | None = None

    # RoPE
    # GLM-5.2 uses default (non-YaRN) RoPE, so there is no mscale correction.
    rope_parameters: dict = field(
        default_factory=lambda: {"rope_type": "default", "rope_theta": 8000000.0}
    )
    max_position_embeddings: int = 1048576

    # GLM-5.2 keeps a separate lm_head. The models do not support a tied head.
    tie_word_embeddings: bool = False

    # Special tokens
    pad_token_id: int | None = 154820
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    # The prefill options (not model geometry)
    options: Glm52PrefillOptions | None = None

    def __post_init__(self):
        # MLA keeps one compressed KV representation, not per-head K/V.
        self.num_kv_heads = 1
        # The paged KV cache stores one combined [k_pe | compressed_kv] vector
        # per token. Its width is the MLA cache head size.
        self.head_dim = self.qk_rope_head_dim + self.kv_lora_rank  # 64 + 512 = 576
        # Per-head query/key width before the value split.
        self.qk_head_dim = (
            self.qk_nope_head_dim + self.qk_rope_head_dim
        )  # 192 + 64 = 256
        # RoPE applies only to the rope slice.
        self.rope_head_dim = self.qk_rope_head_dim  # 64
        # HuggingFace exposes `num_local_experts` as an alias of `n_routed_experts`.
        self.num_local_experts = self.n_routed_experts
        if self.mtp_layer_index is None:
            self.mtp_layer_index = self.num_hidden_layers

        # Per-layer indexer schedule. "full" runs the indexer. "shared" reuses
        # the previous full layer's top-k selection.
        if self.indexer_types is None:
            freq = max(self.index_topk_freq, 1)
            offset = self.index_skip_topk_offset
            self.indexer_types = [
                "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                for i in range(self.num_hidden_layers)
            ]

        # Per-layer MLP schedule. The first `first_k_dense_replace` layers are
        # dense. The rest are MoE.
        if self.mlp_layer_types is None:
            n_dense = min(self.first_k_dense_replace, self.num_hidden_layers)
            self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (
                self.num_hidden_layers - n_dense
            )
        _check_layer_types("indexer_types", self.indexer_types, ("full", "shared"), self.num_hidden_layers)
        _check_layer_types("mlp_layer_types", self.mlp_layer_types, ("dense", "sparse"), self.num_hidden_layers)

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, options: Glm52PrefillOptions):
        """Create the config from a HuggingFace config and the prefill options.

        Accept a path to a JSON file, a `PretrainedConfig`, or a dict. Filter the
        source down to the declared dataclass fields. Convert a string
        `torch_dtype` to the torch dtype. Attach the prefill options.

        The filter keeps the dataclass defaults for every field a checkpoint does not
        declare. In particular a top-level `rope_theta` is not a field, so the RoPE base
        stays `rope_parameters`' 8e6 unless the checkpoint declares `rope_parameters`:
        the GLM-5.2 config.json omits rope_theta, and 8e6 is the base the decode geometry
        rotates k_pe with too.
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            # transformers' to_dict serializes quantization_config whenever the attribute exists, which fails on
            # None, so a None one is hidden for the call and put back.
            if (
                hasattr(hf_config, "quantization_config")
                and hf_config.quantization_config is None
            ):
                delattr(hf_config, "quantization_config")
                config_dict = hf_config.to_dict()
                hf_config.quantization_config = None
            else:
                config_dict = hf_config.to_dict()
        else:
            config_dict = hf_config

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        filtered_dict["options"] = options

        return cls(**filtered_dict)
