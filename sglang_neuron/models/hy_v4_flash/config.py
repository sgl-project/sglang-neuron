# SPDX-License-Identifier: Apache-2.0
"""The shape of a Hy4-Flash checkpoint (``HYV4FlashForConditionalGeneration``) as the Neuron graphs serve it.

Hy4-Flash's text decoder (42 layers) differs from a DeepSeek-style MLA model in five ways the graphs follow:

| | sparse layers (12) | sliding layers (30) |
| --- | --- | --- |
| attention | gated MLA, 32 heads, over the IndexPool's selection | gated MLA, 64 heads, a 512-token causal window |
| RoPE base | 1e7 | 1e4 |

* **Gated MLA with sinks.** Every layer is MLA (``q_lora_rank`` 1024, ``kv_lora_rank`` 512, a Q/K head of 192 + 64
  RoPE channels, a V head of 256), its output gated by ``sigmoid(gate_proj(x))`` per head channel, a learned sink
  logit per head in the softmax, and the output projection closed by ``post_proj_norm``. RoPE rotates interleaved
  pairs (the checkpoint's interleaved MRoPE reduces to plain interleaved RoPE for text positions).
* **IndexPool.** A sparse layer pools every 4 tokens' indexer keys into one gated, RMS-normed group key; a query keeps
  the 511 best-scoring complete groups before its own group, plus its own group up to itself (``index_topk`` 2048
  tokens). Below 2048 tokens of context that is every token.
* **iHC.** The residual is ``hc_mult`` (4) streams; each block reads a sigmoid-weighted mix of them and adds its
  output to each stream times a per-stream sigmoid gate (``attn_hc`` / ``ffn_hc``), and ``hc_head`` mixes them once
  before the final norm.
* **LatentMoE.** 512 routed experts (16 a token) run in a 1536-wide latent space (``pre_expert_proj`` /
  ``post_expert_proj`` around them), their SwiGLU clamped at ``swiglu_limit``; a shared expert sees the full width,
  and ``combine_output_norm`` closes the block. Layer 0 is a dense MLP closed by ``post_mlp_layernorm``.
* **Engram.** Before 4 layers (``engram_layer_ids``, 1-based), a hashed bigram and trigram lookup (16 hash heads of
  96 channels, ~165M rows a layer) is gated per stream and added to the residual streams.
"""

from __future__ import annotations

from dataclasses import dataclass

NAME = "Hy4-Flash"
ARCHITECTURES = ("HYV4FlashForConditionalGeneration",)
SPARSE = "deepseek_sparse_attention"
SLIDING = "sliding_attention"
# the IndexPool kernels' fixed geometry (upstream SGLang asserts the same)
INDEX_DIM = 128
INDEX_RATIO = 4


def _get(cfg, name: str, default=None):
    value = cfg.get(name) if isinstance(cfg, dict) else getattr(cfg, name, None)
    return default if value is None else value


def text_config(hf_config):
    """The text decoder's config of the checkpoint's composite config (or dict), or the config itself."""
    text = _get(hf_config, "text_config")
    return text if text is not None else hf_config


@dataclass(frozen=True)
class AttentionKind:
    num_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float
    sliding_window: int | None

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def scaling(self) -> float:
        return self.qk_head_dim ** -0.5


@dataclass(frozen=True)
class Hy4FlashConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    q_lora_rank: int
    kv_lora_rank: int
    sparse: AttentionKind
    swa: AttentionKind
    # per layer: True for a sliding-window layer, True for an MoE layer
    swa_layers: tuple
    moe_layers: tuple
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    routed_hidden_size: int
    shared_intermediate_size: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    swiglu_limit: float
    hc_mult: int
    hc_magnitude: float
    hc_eps: float
    index_topk: int
    index_n_heads: int
    # Engram: the 0-based decoder layers that read one and each one's 1-based id (the hash seed, the vocab sizes)
    engram_layers: tuple
    engram_ids: tuple
    engram_max_ngram: int
    engram_heads_per_ngram: int
    engram_head_dim: int
    engram_pad_id: int
    engram_seed: int
    engram_fp8_embedding: bool
    # per engram layer: the hash heads' table sizes, n-gram-major (2-grams' heads, then 3-grams')
    engram_head_sizes: tuple
    # the lookup table's entries (the tokenizer's ids it normalizes)
    engram_lookup_size: int
    tie_word_embeddings: bool
    quant_method: str

    def kind(self, layer_idx: int) -> AttentionKind:
        return self.swa if self.swa_layers[layer_idx] else self.sparse

    @property
    def sparse_layer_ids(self) -> tuple:
        return tuple(i for i, s in enumerate(self.swa_layers) if not s)

    @property
    def engram_hash_heads(self) -> int:
        return (self.engram_max_ngram - 1) * self.engram_heads_per_ngram

    def layer_name(self, layer_idx: int) -> str:
        return f"layers.{layer_idx}.self_attn"

    @property
    def sliding_windows(self) -> list:
        return [self.swa.sliding_window if s else None for s in self.swa_layers]


def _rope_theta(text, kind: str, default: float) -> float:
    rope = _get(text, "layer_rope_parameters") or _get(text, "rope_parameters") or {}
    entry = rope.get(kind) if isinstance(rope, dict) else None
    if isinstance(entry, dict) and "rope_theta" in entry:
        return float(entry["rope_theta"])
    return default


def hy4_flash_config(hf_config) -> Hy4FlashConfig:
    """The graphs' config of a Hy4-Flash checkpoint, refusing what they do not serve."""
    text = text_config(hf_config)
    problems = config_problems(text)
    if problems:
        raise ValueError(f"{NAME}: " + "; ".join(problems))
    n = int(_get(text, "num_hidden_layers"))
    layer_types = [SPARSE if t in ("indexed_attention", "sparse", "sparse_attention") else t
                   for t in list(_get(text, "layer_types"))[:n]]
    mlp_types = list(_get(text, "mlp_layer_types", ["dense"] + ["sparse"] * (n - 1)))[:n]
    nope, rope, v = (int(_get(text, k)) for k in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"))
    sparse = AttentionKind(num_heads=int(_get(text, "num_attention_heads")), qk_nope_head_dim=nope,
                           qk_rope_head_dim=rope, v_head_dim=v,
                           rope_theta=_rope_theta(text, SPARSE, 1e7), sliding_window=None)
    swa = AttentionKind(num_heads=int(_get(text, "swa_num_attention_heads", sparse.num_heads)),
                        qk_nope_head_dim=int(_get(text, "swa_qk_nope_head_dim", nope)),
                        qk_rope_head_dim=int(_get(text, "swa_qk_rope_head_dim", rope)),
                        v_head_dim=int(_get(text, "swa_v_head_dim", v)),
                        rope_theta=_rope_theta(text, SLIDING, 1e4),
                        sliding_window=int(_get(text, "sliding_window", 512)))
    engram_ids = tuple(int(i) for i in _get(text, "engram_layer_ids", []) or [])
    sizes = _get(text, "engram_layer_head_vocab_sizes", {}) or {}
    head_sizes = tuple(tuple(int(s) for order in sizes[str(i)] for s in order) for i in engram_ids)
    quant = _get(text, "quantization_config") or _get(hf_config, "quantization_config") or {}
    quant = quant if isinstance(quant, dict) else quant.to_dict()
    moe_inter = int(_get(text, "moe_intermediate_size"))
    return Hy4FlashConfig(
        vocab_size=int(_get(text, "vocab_size")),
        hidden_size=int(_get(text, "hidden_size")),
        intermediate_size=int(_get(text, "intermediate_size")),
        num_hidden_layers=n,
        rms_norm_eps=float(_get(text, "rms_norm_eps", 1e-5)),
        q_lora_rank=int(_get(text, "q_lora_rank")),
        kv_lora_rank=int(_get(text, "kv_lora_rank")),
        sparse=sparse, swa=swa,
        swa_layers=tuple(t == SLIDING for t in layer_types),
        moe_layers=tuple(t == "sparse" for t in mlp_types),
        num_experts=int(_get(text, "n_routed_experts")),
        top_k=int(_get(text, "num_experts_per_tok")),
        moe_intermediate_size=moe_inter,
        routed_hidden_size=int(_get(text, "routed_expert_hidden_size", moe_inter)),
        shared_intermediate_size=moe_inter * int(_get(text, "n_shared_experts", 1)),
        norm_topk_prob=bool(_get(text, "norm_topk_prob", True)),
        routed_scaling_factor=float(_get(text, "routed_scaling_factor", 1.0)),
        swiglu_limit=float(_get(text, "swiglu_limit", 0.0)),
        hc_mult=int(_get(text, "hc_mult", 4)),
        hc_magnitude=float(_get(text, "hc_magnitude", 2.0)),
        hc_eps=float(_get(text, "hc_eps", 1e-6)),
        index_topk=int(_get(text, "index_topk", 2048)),
        index_n_heads=int(_get(text, "compressed_indexer_num_attention_heads", _get(text, "index_n_heads", 16))),
        engram_layers=tuple(i - 1 for i in engram_ids),
        engram_ids=engram_ids,
        engram_max_ngram=int(_get(text, "engram_max_ngram_size", 3)),
        engram_heads_per_ngram=int(_get(text, "engram_n_head_per_ngram", 8)),
        engram_head_dim=int(_get(text, "engram_n_embed_per_ngram", 768)) // int(_get(text, "engram_n_head_per_ngram",
                                                                                        8)),
        engram_pad_id=int(_get(text, "engram_pad_id", 120002)),
        engram_seed=int(_get(text, "engram_seed", 0)),
        engram_fp8_embedding=bool(_get(text, "engram_fp8_embedding", False)),
        engram_head_sizes=head_sizes,
        engram_lookup_size=int(_get(text, "engram_lookup_table_size", 120818)),
        tie_word_embeddings=bool(_get(text, "tie_word_embeddings", False)),
        quant_method=str(quant.get("quant_method") or ""),
    )


def config_problems(text) -> list:
    """Everything in the text config the graphs do not serve, each naming the field."""
    problems = []
    n = int(_get(text, "num_hidden_layers", 0))
    layer_types = list(_get(text, "layer_types", []) or [])
    allowed = {SPARSE, SLIDING, "indexed_attention", "sparse", "sparse_attention"}
    if len(layer_types) < n or set(layer_types[:n]) - allowed:
        problems.append(f"layer_types must name each of the {n} layers {SPARSE} or {SLIDING}")
    if int(_get(text, "compressed_indexer_compress_ratio", INDEX_RATIO)) != INDEX_RATIO:
        problems.append(f"compressed_indexer_compress_ratio must be {INDEX_RATIO}")
    for key in ("compressed_indexer_head_dim", "index_head_dim"):
        if int(_get(text, key, INDEX_DIM)) != INDEX_DIM:
            problems.append(f"{key} must be {INDEX_DIM}")
    ratio = int(_get(text, "compressed_indexer_compress_ratio", INDEX_RATIO))
    if int(_get(text, "index_topk", 2048)) != int(_get(text, "compressed_indexer_group_topk", 512)) * ratio:
        problems.append("index_topk must be compressed_indexer_group_topk x the compress ratio")
    if _get(text, "compressed_indexer_key_pooling", "gated") != "gated":
        problems.append("the IndexPool is served with gated key pooling (compressed_indexer_key_pooling)")
    if int(_get(text, "n_group", 1)) != 1 or int(_get(text, "topk_group", 1)) != 1:
        problems.append("grouped routing (n_group, topk_group above 1) is not served")
    if _get(text, "attention_bias", False):
        problems.append("attention biases (attention_bias) are not served")
    if _get(text, "hidden_act", "silu") != "silu":
        problems.append(f"hidden_act {_get(text, 'hidden_act')!r}: the MLPs are served SiLU-gated")
    if int(_get(text, "qk_rope_head_dim", 64)) != int(_get(text, "swa_qk_rope_head_dim", 64)):
        problems.append("the sparse and sliding layers must share the RoPE width")
    quant = _get(text, "quantization_config") or {}
    quant = quant if isinstance(quant, dict) else quant.to_dict()
    if quant:
        if quant.get("quant_method") != "fp8":
            problems.append(f"quantization {quant.get('quant_method')!r}: the graphs read bf16 or 128 x 128 FP8 blocks")
        elif list(quant.get("weight_block_size") or [128, 128]) != [128, 128]:
            problems.append(f"weight_block_size {quant.get('weight_block_size')}: the loaders read 128 x 128 FP8 "
                            "blocks")
    return problems
