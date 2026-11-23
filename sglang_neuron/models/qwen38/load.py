# SPDX-License-Identifier: Apache-2.0
"""The checkpoint mapping and the per-parameter loaders, on the platform's pipelined reader.

Every parameter names the checkpoint tensors it is built from (``hf_weight_mappings``) and carries a
``SafetensorsWeightLoader`` whose transform builds this rank's shard in the kernels' layout (``weights.py``). The
reader then reads each rank's share of the files into the page cache, applies the transforms and uploads the result,
as it does for every other model here.

The checkpoint prefixes the language model with ``model.language_model.`` and carries a ``model.visual.`` tower that
text-only serving never names, so it is never read.
"""

from __future__ import annotations

import torch

from sglang_neuron.models.qwen38 import weights as W
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, pad_to_shape, set_weight_loader


def _loader(fn) -> SafetensorsWeightLoader:
    return SafetensorsWeightLoader(transform=fn)


def set_loaders(graph) -> None:
    """Give every parameter of ``graph`` the transform that builds this rank's shard."""
    dims, layout = graph.dims, graph.layout
    tp = layout.tp_size
    head, kh, vh = dims.head_k_dim, dims.num_k_heads, dims.num_v_heads
    hidden = dims.hidden

    rows = graph.embed_tokens.vocab_size_per_rank
    set_weight_loader(graph.embed_tokens.weight, _loader(
        lambda s, rank: pad_to_shape(s[0][rank * rows:(rank + 1) * rows], (rows, hidden))))
    lm_rows = graph.lm_head.out_features_per_rank
    set_weight_loader(graph.lm_head.weight, _loader(
        lambda s, rank: pad_to_shape(s[0][rank * lm_rows:(rank + 1) * lm_rows], (lm_rows, hidden))))

    for layer in graph.layers:
        set_weight_loader(layer.mlp.gate_weight, _loader(
            lambda s, rank: W.column_shard(s[0][:], rank=rank, tp_size=tp)))
        set_weight_loader(layer.mlp.up_weight, _loader(
            lambda s, rank: W.column_shard(s[0][:], rank=rank, tp_size=tp)))
        set_weight_loader(layer.mlp.down_weight, _loader(
            lambda s, rank: W.row_shard(s[0][:], rank=rank, tp_size=tp)))
        if layer.is_gdn:
            set_weight_loader(layer.linear_attn.w_proj_p, _loader(
                lambda s, rank: W.gdn_projection(s[0][:], s[1][:], s[2][:], s[3][:], rank=rank, tp_size=tp,
                                                 num_k_heads=kh, num_v_heads=vh, head_dim=head)))
            set_weight_loader(layer.linear_attn.conv_w, _loader(
                lambda s, rank: W.gdn_conv(s[0][:], rank=rank, tp_size=tp, num_k_heads=kh, num_v_heads=vh,
                                           head_dim=head)))
            set_weight_loader(layer.linear_attn.w_out, _loader(
                lambda s, rank: W.gdn_out(s[0][:], rank=rank, tp_size=tp, num_v_heads=vh, head_dim=head)))
            per = vh // tp
            set_weight_loader(layer.linear_attn.a_log, _loader(
                lambda s, rank: s[0][rank * per:(rank + 1) * per].float().contiguous()))
            set_weight_loader(layer.linear_attn.dt_bias, _loader(
                lambda s, rank: s[0][rank * per:(rank + 1) * per].float().contiguous()))
            set_weight_loader(layer.linear_attn.norm_w, _loader(lambda s, rank: s[0][:].float().contiguous()))
        else:
            nq, nkv, d = dims.num_q_heads, dims.num_kv_heads, dims.head_dim
            pad = layout.kernel_q_heads
            set_weight_loader(layer.self_attn.w_qkv, _loader(
                lambda s, rank: W.attention_qkv(s[0][:], s[1][:], s[2][:], rank=rank, tp_size=tp, num_q_heads=nq,
                                                num_kv_heads=nkv, head_dim=d, padded_q_heads=pad)[0]))
            set_weight_loader(layer.self_attn.w_gate_p, _loader(
                lambda s, rank: W.row_permute(W.attention_qkv(
                    s[0][:], s[1][:], s[2][:], rank=rank, tp_size=tp, num_q_heads=nq, num_kv_heads=nkv,
                    head_dim=d, padded_q_heads=pad)[1])))
            set_weight_loader(layer.self_attn.w_o, _loader(
                lambda s, rank: W.attention_out(s[0][:], rank=rank, tp_size=tp, num_q_heads=nq, head_dim=d,
                                                padded_q_heads=pad)))
            set_weight_loader(layer.self_attn.q_norm_w, _loader(
                lambda s, rank: s[0][:].float().view(1, -1).contiguous()))
            set_weight_loader(layer.self_attn.k_norm_w, _loader(
                lambda s, rank: s[0][:].float().view(1, -1).contiguous()))


def hf_weight_mappings(graph) -> dict:
    """Parameter name to its checkpoint key, or the keys a built parameter is made from."""
    p = graph.prefix
    dims = graph.dims
    m = {
        "embed_tokens.weight": f"{p}embed_tokens.weight",
        "norm.weight": f"{p}norm.weight",
        "lm_head.weight": f"{p}embed_tokens.weight" if dims.tie_word_embeddings else "lm_head.weight",
    }
    for i, layer in enumerate(graph.layers):
        src, dst = f"{p}layers.{i}", f"layers.{i}"
        m[f"{dst}.input_layernorm.weight"] = f"{src}.input_layernorm.weight"
        m[f"{dst}.post_attention_layernorm.weight"] = f"{src}.post_attention_layernorm.weight"
        m[f"{dst}.mlp.gate_weight"] = f"{src}.mlp.gate_proj.weight"
        m[f"{dst}.mlp.up_weight"] = f"{src}.mlp.up_proj.weight"
        m[f"{dst}.mlp.down_weight"] = f"{src}.mlp.down_proj.weight"
        if layer.is_gdn:
            a = f"{src}.linear_attn"
            m[f"{dst}.linear_attn.w_proj_p"] = [f"{a}.in_proj_qkv.weight", f"{a}.in_proj_z.weight",
                                                f"{a}.in_proj_a.weight", f"{a}.in_proj_b.weight"]
            m[f"{dst}.linear_attn.conv_w"] = f"{a}.conv1d.weight"
            m[f"{dst}.linear_attn.w_out"] = f"{a}.out_proj.weight"
            m[f"{dst}.linear_attn.a_log"] = f"{a}.A_log"
            m[f"{dst}.linear_attn.dt_bias"] = f"{a}.dt_bias"
            m[f"{dst}.linear_attn.norm_w"] = f"{a}.norm.weight"
        else:
            s = f"{src}.self_attn"
            qkv = [f"{s}.q_proj.weight", f"{s}.k_proj.weight", f"{s}.v_proj.weight"]
            m[f"{dst}.self_attn.w_qkv"] = list(qkv)
            m[f"{dst}.self_attn.w_gate_p"] = list(qkv)
            m[f"{dst}.self_attn.w_o"] = f"{s}.o_proj.weight"
            m[f"{dst}.self_attn.q_norm_w"] = f"{s}.q_norm.weight"
            m[f"{dst}.self_attn.k_norm_w"] = f"{s}.k_norm.weight"
    return m


def load_into(graph, path: str, device) -> None:
    """Read this rank's shard of every mapped parameter onto ``device``."""
    from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint

    set_loaders(graph)
    checkpoint = SafetensorsCheckpoint(path)
    state = checkpoint.load_sharded_pipelined(graph.tp_group.rank_in_group, graph.tp_group.world_size, graph,
                                              hf_weight_mappings(graph), device).state_dict
    state = {k: (v.to(torch.bfloat16) if v.dtype.is_floating_point and v.dtype not in
                 (torch.bfloat16, torch.float32) else v) for k, v in state.items()}
    graph.load_state_dict(state, strict=False, assign=True)
