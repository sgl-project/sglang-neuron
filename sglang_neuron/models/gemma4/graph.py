# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 dense as the runtime's role graphs (prefill: one request's chunk per forward; decode: one token per row),
on attention TP.

Attention is torch math with float32 scores (head_dim 256 and 512 are above what the platform's fused attention
kernels serve), and the cache is written and read only through the indirect-DMA kernels in
kv_access.py. Each layer reads and writes its own K and V pool tensors; a global layer views its tensors with wider
rows inside the kernels, so no graph input is aliased by two differently shaped views.

Per layer: input norm, attention (QK RMSNorm, V RMSNorm without scale, RoPE, scaling 1.0), post-attention norm and
residual, pre-feedforward norm, GeGLU MLP (tanh GELU), post-feedforward norm and residual, then the layer scalar.
The embedding is scaled by sqrt(hidden) rounded to bf16, and the logits are soft-capped."""

import math

import torch
from sglang.srt.distributed.parallel_state import get_attn_tp_group
from torch import nn

from sglang_neuron.layers import ColumnParallelLinear
from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.mem_cache.kv_cache import KVSpec, LayerSpec
from sglang_neuron.models.gemma4 import kv_access
from sglang_neuron.models.gemma4.config import DTYPE, Gemma4Config
from sglang_neuron.models.transformer import weights
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import set_weight_loader


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float, scale: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=DTYPE)) if scale else None
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        normed = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)
        return normed if self.weight is None else self.weight * normed


def gelu_tanh(x):
    # written out: torch's gelu(approximate="tanh") is not shown to lower on this stack
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


class Rotary(nn.Module):
    """Gemma 4 RoPE over the whole head (rotate-half). A proportional RoPE rotates the first
    ``int(factor * head_dim / 2)`` frequencies (denominator head_dim) and gives the rest frequency 0 (cos 1, sin 0)."""

    def __init__(self, head_dim: int, theta: float, factor: float):
        super().__init__()
        angles = int(factor * head_dim // 2)
        inv = 1.0 / (theta ** (torch.arange(0, 2 * angles, 2, dtype=torch.float32) / head_dim))
        inv = torch.cat([inv, torch.zeros(head_dim // 2 - angles, dtype=torch.float32)])
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, positions):
        """``[T, head_dim]`` cos and sin in DTYPE."""
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(DTYPE), emb.sin().to(DTYPE)


def _rope(x, cos, sin):
    """x ``[T, heads, d]``, cos and sin ``[T, d]``."""
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.name = f"layers.{layer_idx}.self_attn"
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = get_attn_tp_group().world_size
        self.is_global = config.is_global(layer_idx)
        self.head_dim = d = config.layer_head_dim(layer_idx)
        total_kv = config.layer_kv_heads(layer_idx)
        if config.num_attention_heads % tp:
            raise ValueError(f"{config.num_attention_heads} attention heads do not shard over an attention TP of {tp}")
        if total_kv % tp and tp % total_kv:
            raise ValueError(f"{total_kv} KV heads neither shard over nor replicate across an attention TP of {tp}")
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = max(1, total_kv // tp)
        self.groups = self.num_heads // self.num_kv_heads
        self.window = None if self.is_global else config.sliding_window
        self.k_eq_v = self.is_global and config.attention_k_eq_v
        width = (self.num_heads + 2 * self.num_kv_heads) * d
        self.qkv_weight = nn.Parameter(torch.empty(config.hidden_size, width, dtype=DTYPE))
        self.o_weight = nn.Parameter(torch.empty(self.num_heads * d, config.hidden_size, dtype=DTYPE))
        self.q_norm = RMSNorm(d, config.rms_norm_eps)
        self.k_norm = RMSNorm(d, config.rms_norm_eps)
        self.v_norm = RMSNorm(d, config.rms_norm_eps, scale=False)
        # bound by bind_kv_cache: this layer's own K and V pool tensors, and the [pages, heads, page, d] view of them
        # its rows use (more heads than it writes when the pool's rows are narrower)
        self.k_cache = None
        self.v_cache = None
        self.dims = None
        set_weight_loader(self.qkv_weight, weights.fused_qkv_weight_loader(tp, self.num_heads * d,
                                                                           self.num_kv_heads * d, total_kv))
        set_weight_loader(self.o_weight, weights.row_parallel_loader(self.num_heads * d))

    def bind(self, k_cache, v_cache):
        pages, heads, page, dp = k_cache.shape[:4]
        if (heads * dp) % self.head_dim or (heads * dp) // self.head_dim < self.num_kv_heads:
            raise ValueError(f"{self.name}: a pool page row of {heads} x {dp} cannot hold {self.num_kv_heads} KV heads "
                             f"of {self.head_dim}")
        self.k_cache, self.v_cache = k_cache, v_cache
        self.dims = (pages, heads * dp // self.head_dim, page, self.head_dim)

    def _project(self, x, cos, sin):
        """q ``[T, heads, d]``, k and v ``[T, kv_heads, d]``."""
        t, d = x.shape[0], self.head_dim
        qkv = x @ self.qkv_weight
        q, k, v = torch.split(qkv, [self.num_heads * d, self.num_kv_heads * d, self.num_kv_heads * d], dim=-1)
        q = self.q_norm(q.reshape(t, self.num_heads, d))
        k = self.k_norm(k.reshape(t, self.num_kv_heads, d))
        v = self.v_norm(v.reshape(t, self.num_kv_heads, d))
        return _rope(q, cos, sin), _rope(k, cos, sin), v

    def _write(self, k, v, slots, page):
        """Write this step's K and V rows at their slots; return the two ordering tokens."""
        slots = slots.to(torch.long)
        rows = kv_access.token_rows(slots // page, slots % page, self.num_kv_heads, self.dims[1], page)
        d = self.head_dim
        k_rows = k.transpose(0, 1).reshape(-1, d).to(self.k_cache.dtype)
        v_rows = v.transpose(0, 1).reshape(-1, d).to(self.v_cache.dtype)
        return kv_access.write_rows(self.k_cache, k_rows, rows), kv_access.write_rows(self.v_cache, v_rows, rows)

    def _read(self, table, tokens):
        """K and V ``[rows, kv_heads, pages * page, d]`` of each table row, read after the writes."""
        rows, nb = table.shape
        ids = torch.clamp(table.reshape(-1).to(torch.long), min=0)
        out = []
        for cache, token in zip((self.k_cache, self.v_cache), tokens):
            pages = kv_access.gather_pages(cache, ids, self.dims, after=token)
            pages = pages.view(rows, nb, self.dims[1], self.dims[2], self.head_dim)[:, :, :self.num_kv_heads]
            out.append(pages.permute(0, 2, 1, 3, 4).reshape(rows, self.num_kv_heads, nb * self.dims[2], self.head_dim))
        return out

    def _attend(self, q, k, v, q_pos, k_pos):
        """q ``[R, Tq, heads, d]``, k and v ``[R, kv_heads, S, d]``, q_pos ``[R, Tq]``, k_pos ``[R, S]``; returns
        ``[R, Tq, heads * d]``. Float32 scores with scaling 1.0, no GQA expansion."""
        r, tq = q.shape[:2]
        qg = q.view(r, tq, self.num_kv_heads, self.groups, self.head_dim).permute(0, 2, 1, 3, 4)
        qg = qg.reshape(r, self.num_kv_heads, tq * self.groups, self.head_dim)
        scores = torch.matmul(qg.float(), k.float().transpose(-2, -1))
        keep = k_pos[:, None, :] <= q_pos[:, :, None]
        if self.window is not None:
            keep = keep & (k_pos[:, None, :] > q_pos[:, :, None] - self.window)
        # [R, Tq, S] -> [R, 1, Tq * G, S]
        keep = keep.unsqueeze(2).expand(r, tq, self.groups, keep.shape[-1]).reshape(r, 1, tq * self.groups, -1)
        scores = scores.masked_fill(~keep, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v.float()).to(DTYPE)
        out = out.view(r, self.num_kv_heads, tq, self.groups, self.head_dim).permute(0, 2, 1, 3, 4)
        return out.reshape(r, tq, self.num_heads * self.head_dim)

    def prefill(self, x, cos, sin, positions, md):
        t = x.shape[0]
        q, k, v = self._project(x, cos, sin)
        tokens = self._write(k, v, md["slot_mapping"], md["block_size"])
        table = md["block_table_tensor"]
        k_all, v_all = self._read(table, tokens)
        k_pos = torch.arange(k_all.shape[2], device=x.device).view(1, -1)
        out = self._attend(q.unsqueeze(0), k_all, v_all, positions.view(1, t), k_pos).view(t, -1)
        return neuron_collectives.all_reduce(self.attn_tp_group, out @ self.o_weight)

    def decode(self, x, cos, sin, positions, md):
        b = x.shape[0]
        q, k, v = self._project(x, cos, sin)
        tokens = self._write(k, v, md["slot_mapping"], md["block_size"])
        k_all, v_all = self._read(md["block_table_tensor"], tokens)
        k_pos = torch.arange(k_all.shape[2], device=x.device).view(1, -1)
        offset = md.get("swa_kv_pos_offset")
        if offset is not None:
            k_pos = k_pos + offset.view(-1, 1).to(k_pos.dtype)
        k_pos = k_pos.expand(b, -1)
        out = self._attend(q.unsqueeze(1), k_all, v_all, positions.view(b, 1), k_pos).view(b, -1)
        return neuron_collectives.all_reduce(self.attn_tp_group, out @ self.o_weight)


class MLP(nn.Module):
    """GeGLU: down(gelu_tanh(gate(x)) * up(x)), the intermediate dimension sharded over the attention TP."""

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.attn_tp_group.world_size
        if config.intermediate_size % tp:
            raise ValueError(f"an intermediate size of {config.intermediate_size} does not shard over {tp} ranks")
        rows = config.intermediate_size // tp
        self.gate_up_weight = nn.Parameter(torch.empty(config.hidden_size, 2 * rows, dtype=DTYPE))
        self.down_weight = nn.Parameter(torch.empty(rows, config.hidden_size, dtype=DTYPE))
        set_weight_loader(self.gate_up_weight, weights.gate_up_loader(rows, False))
        set_weight_loader(self.down_weight, weights.row_parallel_loader(rows))

    def forward(self, x):
        gate, up = (x @ self.gate_up_weight).chunk(2, dim=-1)
        return neuron_collectives.all_reduce(self.attn_tp_group, (gelu_tanh(gate) * up) @ self.down_weight)


class DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps)
        self.mlp = MLP(config)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps)
        self.layer_scalar = nn.Parameter(torch.ones(1, dtype=DTYPE))

    def forward(self, x, attention):
        x = x + self.post_attention_layernorm(attention(self.input_layernorm(x)))
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return x * self.layer_scalar


class Gemma4Graph(nn.Module):
    """One role graph (``phase`` "prefill" or "decode")."""

    def __init__(self, config: Gemma4Config, phase: str, checkpoint_prefix: str = "model.language_model."):
        super().__init__()
        if phase not in ("prefill", "decode"):
            raise ValueError(f"a role graph is a prefill or a decode, not {phase!r}")
        self.config = config
        self.phase = phase
        self.prefix = checkpoint_prefix
        self.tp_group = get_attn_tp_group()
        self.lm_head_tp_group = group = groups.get_neuron_attn_tp_group()
        self.embed_tokens = VocabDimShardedEmbedding(config.vocab_size, config.hidden_size, dtype=DTYPE,
                                                     tp_group=group.device_group)
        # sqrt(hidden) rounded to bf16, as the reference stores it (sqrt(5376) = 73.32 -> 73.5)
        self.embed_scale = torch.tensor(math.sqrt(config.hidden_size), dtype=DTYPE).item()
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.sliding_rope = Rotary(config.head_dim, config.sliding_rope_theta, 1.0)
        self.global_rope = Rotary(config.global_head_dim, config.global_rope_theta,
                                  config.global_partial_rotary_factor)
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, dtype=DTYPE, gather_output=True,
                                            tp_group=group.device_group)
        set_weight_loader(self.embed_tokens.weight,
                          weights.vocab_rows_loader(self.embed_tokens.vocab_size_per_rank, config.hidden_size))
        if not config.tie_word_embeddings:
            raise ValueError("Gemma 4: an untied LM head is not served")
        self.lm_head.weight = self.embed_tokens.weight
        # the runtime's per-layer metadata keys and windows (models/contract.declared_layer_kinds)
        self.layer_names = [layer.self_attn.name for layer in self.layers]
        self.sliding_windows = [layer.self_attn.window for layer in self.layers]

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank) * self.embed_scale
        ropes = {False: self.sliding_rope(positions), True: self.global_rope(positions)}
        step = "prefill" if self.phase == "prefill" else "decode"
        for layer in self.layers:
            attn = layer.self_attn
            cos, sin = ropes[attn.is_global]
            md = attn_metadata[attn.name]
            run = getattr(attn, step)
            hidden = layer(hidden, lambda x, run=run, cos=cos, sin=sin, md=md: run(x, cos, sin, positions, md))
        hidden = self.norm(hidden.index_select(0, sampling_positions))
        logits = self.lm_head(hidden)
        cap = self.config.final_logit_softcapping
        if cap:
            logits = cap * torch.tanh(logits.float() / cap)
        return logits

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(layers=[LayerSpec(name=name) for name in self.layer_names])

    def bind_kv_cache(self, kv_caches: dict) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            attn.bind(*kv_caches[attn.name])

    def hf_weight_mappings(self) -> dict:
        p = self.prefix
        mappings = {"embed_tokens.weight": f"{p}embed_tokens.weight", "norm.weight": f"{p}norm.weight"}
        for i, layer in enumerate(self.layers):
            src, dst = f"{p}layers.{i}", f"layers.{i}"
            attn = f"{src}.self_attn"
            v = "k" if layer.self_attn.k_eq_v else "v"
            mappings.update({
                f"{dst}.self_attn.qkv_weight": [f"{attn}.q_proj.weight", f"{attn}.k_proj.weight",
                                                f"{attn}.{v}_proj.weight"],
                f"{dst}.self_attn.o_weight": f"{attn}.o_proj.weight",
                f"{dst}.self_attn.q_norm.weight": f"{attn}.q_norm.weight",
                f"{dst}.self_attn.k_norm.weight": f"{attn}.k_norm.weight",
                f"{dst}.mlp.gate_up_weight": [f"{src}.mlp.gate_proj.weight", f"{src}.mlp.up_proj.weight"],
                f"{dst}.mlp.down_weight": f"{src}.mlp.down_proj.weight",
                f"{dst}.layer_scalar": f"{src}.layer_scalar",
            })
            for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                         "post_feedforward_layernorm"):
                mappings[f"{dst}.{norm}.weight"] = f"{src}.{norm}.weight"
        return mappings

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        state = checkpoint.load_sharded_pipelined(self.tp_group.rank_in_group, self.tp_group.world_size, self,
                                                  self.hf_weight_mappings(), device).state_dict
        state = {k: (t.to(DTYPE) if t.dtype != DTYPE else t) for k, t in state.items()}
        self.load_state_dict(state, strict=False, assign=True)
        self.lm_head.weight = self.embed_tokens.weight
