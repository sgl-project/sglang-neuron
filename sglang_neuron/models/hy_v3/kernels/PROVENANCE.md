# Hy3 kernels

| file | from | change |
| --- | --- | --- |
| `moe_tkg_fused.py` | vllm-neuron Hy3 contrib, `contrib/models/hy_v3/src/kernels/moe_tkg_fused.py` (qingzwang/vllm-neuron#8) | header; the static-DMA-descriptor switch (an environment variable this plugin's registry would refuse) removed |
| `bwmm_shard_on_I_fp8w.py` | nkilib 2.33 `nkilib/core/moe/moe_cte/bwmm_shard_on_I.py` (md5 3788ce5254164330a3f6b29497cab4d1), via the vllm-neuron Hy3 contrib | `UPCAST_FP8_WEIGHTS = False` gates the two per-channel FP8 weight upcasts (gate / up and down): the FP8 tiles go to the matmul; absolute nkilib imports |
| `moe_cte_fp8w.py` | vllm-neuron Hy3 contrib, `contrib/models/hy_v3/src/kernels/moe_cte_fp8w.py` | none |
| `attention_kv_parallel_fp8.py` | nkilib 2.33 `nkilib/core/attention/attention_kv_parallel_segmented_cte.py` (md5 8191f27b8e5951cd2bef47fb62490d39) | a `prior_seg_size` argument (default `seg_size`) for the inner segmented kernel's prior KV segment, so a query segment can be longer than it; absolute nkilib imports |
