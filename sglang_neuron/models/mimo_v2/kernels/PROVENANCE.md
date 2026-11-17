# MiMo-V2 kernels

| file | from | change |
| --- | --- | --- |
| `moe_half.py` | Hy3's decode MoE block kernel, `models/hy_v3/kernels/moe_tkg_fused.py` (from the vllm-neuron Hy3 contrib, qingzwang/vllm-neuron#8) | no shared expert; bf16 activations against the FP8 weights (one 128-row matmul pass, in place of the static-FP8 W8A8 activations and double-row passes, since MiMo's checkpoint has no activation scales); the per-output-row weight scales of `experts_fp8.py` applied after the gate / up and down matmuls; the affinities rounded through bf16 as the torch path's; the RMSNorm rounded as `models/transformer/model.RMSNorm` |
| `../kv_write.py` | vllm-neuron's MiMo-V2.5 contrib, `contrib/models/mimo_v2_5/src/kernels/kv_cache_write.py` (vllm-project/vllm-neuron#67) | the plugin's imports; a slot outside the pool goes to SGLang's reserved slot 0 instead of the skip sentinel |
| `layer.py` | the structure of nkilib's `experimental/transformer/transformer_tkg.py` (its HBM path) | MiMo's halves (`attn_half.py`, `moe_half.py`, nkilib's `mlp`); the collective's destination fenced before it is read; a name for every shared HBM tensor |
| `attn_half.py`, `attention_block.py` | new, around nkilib's `attention_block_tkg` | |
