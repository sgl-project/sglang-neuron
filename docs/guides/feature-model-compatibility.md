# Feature and model compatibility

Which features each model serves. Each cell comes from the launch and admission checks ([What a launch
refuses](../design/runtime.md#what-a-launch-refuses)) and the hardware runs. A refused launch fails at startup with a
message that names the setting at fault.

- **tested**: served, and run on hardware in the layout given; the [examples](../../examples/README.md) launch these
  layouts and check their accuracy;
- **served**: not refused, but not run on hardware;
- **refused**: the launch fails at startup;
- **—**: does not apply to the model.

The models are GPT-OSS-120B (`GptOssForCausalLM`), Qwen3-0.6B and Qwen3-8B (`Qwen3ForCausalLM`), Qwen3-30B-A3B
(`Qwen3MoeForCausalLM`), GLM-4-9B-0414 (`Glm4ForCausalLM`), Gemma 4 31B (`Gemma4ForConditionalGeneration`), DeepSeek-V2-Lite (`DeepseekV2ForCausalLM`), GLM-5.2
(`GlmMoeDsaForCausalLM`), Qwen3.5-397B-A17B (`Qwen3_5MoeForCausalLM`) and Kimi-K3 (`KimiK3ForConditionalGeneration`).

## Serving and parallelism

| Feature | GPT-OSS | Qwen3 | Qwen3-MoE | GLM-4 | Gemma 4 | DeepSeek-V2 | GLM-5.2 | Qwen3.5 | Kimi-K3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aggregated server | tested (TP8, EP8) | tested (TP4; Qwen3-8B also TP8) | tested (TP4, EP4) | tested (TP4) | tested (TP4 on a trn2.3xlarge) | tested (TP4) | refused | served (TP128 of eight DP-attention replicas, EP128; [Qwen3.5 radix](../model-recipes/qwen35-radix.md)) | refused |
| Disaggregated prefill and decode | tested ¹ | tested (TP4; one prefill to two decode servers for Qwen3-8B, to five for Qwen3-0.6B) | tested (TP4 EP4; one prefill to one decode server) | tested (TP4; one prefill to two decode servers) | served; not run ⁹ | tested (TP4; one prefill to five decode servers) | tested ⁵ | tested ⁷ | tested ⁸ |
| DP attention on a decode server | tested | served | refused | served | served | refused | refused | tested (eight replicas of attention TP16) | tested (eight groups of attention TP32) |
| DP attention on a prefill server | served | served | refused | served | served | refused | refused | tested (eight replicas of attention TP16) | refused |
| DP attention on an aggregated server | refused | refused | refused | refused | refused | refused | refused | served (every rank plans each step from SGLang's DP-attention exchange) | refused |
| Expert parallelism (`--ep-size`) | tested | refused (no experts) | tested (EP4); without `--ep-size` served | refused (no experts) | refused (no experts) | refused (its routed experts always split over attention TP) | tested (each role's EP its TP: 32 or 8 on the prefill, 128 on the decode) | tested (EP128 over every rank, required) | tested on the decode (EP256); the prefill splits its experts over each stage's TP group (`--ep-size` 1 or the TP size) |
| Async scheduling and execution | tested | tested | tested | tested | tested | tested | tested | tested | tested |
| Giga-kernel decode (a decode server that runs every layer of a step in one kernel call) | tested (`--tp-size 128 --dp-size 16 --enable-dp-attention --ep-size 128`; six TP4 prefill replicas to one giga-kernel decode) | refused | refused | refused | refused | refused | tested, its only decode (`--tp-size 128 --ep-size 128` without DP attention) ⁵ | tested, its only decode: the whole-model megakernel (`--enable-nvidia-megakernel`), on every role ⁷ | tested, its only decode: the verify program runs all 93 layers in one kernel call ⁸ |
| Pipeline parallelism (`--pp-size`) on a disaggregated prefill server | tested (PP2 x TP8) ² | refused | refused | refused | refused | refused | tested (PP4 x TP32, PP16 x TP8) ⁵ | refused | tested (PP8 x TP16) ⁸ |
| Pipeline parallelism on a disaggregated decode server, behind a prefill of the same stages | tested (PP2 x TP8) ² | refused | refused | refused | refused | refused | refused | refused | refused |
| Pipeline parallelism on a disaggregated decode server, behind an unpipelined prefill at the decode's attention TP | tested (PP2 x TP8 behind TP8) ² | refused | refused | refused | refused | refused | refused | refused | refused |
| Pipeline parallelism on an aggregated server | tested (PP2 x TP8) ⁴ | refused | refused | refused | refused | refused | refused | refused | refused |
| Decode context parallelism (`--dcp-size`) on a disaggregated decode server | tested (five TP4 prefill replicas to a TP16 `--dcp-size 2` decode) ³ | refused | refused | refused | refused | refused | refused | refused; the decode's own split of each replica's KV over its 16 ranks (`SGLANG_NEURON_QWEN35_KV_CP=16`) tested ⁷ | tested (DCP32 inside each DP-attention group) ⁸ |
| Attention context parallelism, decode context parallelism on another role or with DP attention or `--dp-size`, a server over several nodes | refused | refused | refused | refused | refused | refused | refused | refused | tested: DCP16 on the PD prefill, and DCP with DP attention over two nodes in its declared decode layout ⁸; the rest refused |
| Speculative decoding | refused | refused | refused | refused | refused | refused | tested on the decode (`--speculative-algorithm NEURON_MTP`) ⁶ | refused | tested on the decode (`--speculative-algorithm DSPARK`, k = 3 or 7) ⁸ |
| LoRA, mixed chunks | refused | refused | refused | refused | refused | refused | refused | refused | refused |
| Embedding server (`--is-embedding`) | refused | refused | refused | refused | refused | refused | refused | refused | refused |
| Memory saver (`--enable-memory-saver`) | refused | refused | refused | refused | refused | refused | refused | refused | refused |

¹ GPT-OSS: one prefill server of five data-parallel replicas of TP4 EP2 (`--dp-size 5`) to one decode
server of TP64 with eight DP-attention groups and EP64.

² GPT-OSS: two stages of 18 layers, each TP8 EP8, on one node
([Pipeline parallelism](../design/pipeline-parallelism.md)). A pipelined prefill and decode of different layer
partitions (`SGLANG_PP_LAYER_PARTITION`) are refused at pairing, naming both stage ranges. A pipelined server answers
`/pause_generation` in `in_place` or `retract` mode and `/flush_cache` with a timeout with HTTP 400.

³ GPT-OSS's per-layer decode: each pair of attention-TP16 ranks splits every sequence's KV, so the decode holds twice
the context per rank's pool
([Decode context parallelism](../design/decode-context-parallelism.md)).

⁴ GPT-OSS: two stages of 19 and 17 layers (`SGLANG_PP_LAYER_PARTITION=19,17`), each TP8 EP8, on one node, on SGLang's
synchronous PP loop, slower than the pipelined PD roles
([The pipelined aggregated server](../design/pipeline-parallelism.md#the-pipelined-aggregated-server)).

⁵ GLM-5.2: one prefill server, of four TP32 EP32 pipeline stages over 18, 20, 20 and 20 layers or of sixteen TP8 EP8
stages over 3 layers and then 5 each, to one decode server of TP128 on the decode giga kernel, its 32 slots at 128k or
1M positions; either prefill serves either decode point ([GLM-5.2](../model-recipes/glm52.md#the-served-pairings)).

⁶ GLM-5.2's decode: the async MTP K7 step, 7 drafts a step from the checkpoint's MTP layer, verified and carried on the
device ([Async MTP K7](../model-recipes/glm52.md#async-mtp-k7)). Every other algorithm is refused at launch, and so is
`NEURON_MTP` on a prefill or on a server that is not a PD decode.

⁷ Qwen3.5: one prefill server and one decode server, each on one node: TP128 as eight DP-attention replicas of
attention TP16 and EP128, on the whole-model megakernel, with the KV and the GDN state over NIXL. The decode splits each
replica's KV over its 16 ranks (`SGLANG_NEURON_QWEN35_KV_CP=16`) at a 256K context, and keeps its own radix tree
([Qwen3.5 PD](../model-recipes/qwen35-pd.md)). The aggregated server of the same layout is not run on hardware
([Qwen3.5 radix](../model-recipes/qwen35-radix.md)).

⁸ Kimi-K3: one prefill server of eight TP16 pipeline stages on one node, with DCP16 over each stage, to one decode
server over two nodes: TP256 as eight DP-attention groups of attention TP32, EP256, and DCP32 inside each group, with
DSPARK at k = 3 or 7 ([Kimi-K3](../model-recipes/kimi-k3.md)). Kimi-K3 serves the PD prefill and the PD decode only,
and a launch over several nodes in any other layout is refused.

⁹ Gemma 4: no disaggregated layout was run. The bf16 weights (about 62 GB) do not fit the two NeuronCores of half a
trn2.3xlarge, and no multi-node layout was tried, so the prefill/decode rows say what the launch checks do rather than
what has run.

GPT-OSS's launch check and graphs also refuse DP attention without `--ep-size`, heads, KV heads or experts that do not
split evenly over their ranks, a prefill attention TP under four and a decode attention TP under eight (so an aggregated
server, which builds both graphs, needs `--tp-size 8` or more), a checkpoint whose dtype is not bfloat16, and a
`--context-length` that, rounded up to 256 tokens, is not a multiple of 512 (10240, the tested context, is served).

DeepSeek-V2's launch check refuses any layout but four query heads per rank (`--tp-size 4` for DeepSeek-V2-Lite), a
`--context-length` that is not a multiple of 512 (the decode kernel's context tile) or is above 4096 (the longest
prefill window compiled on hardware), a `--chunked-prefill-size` other than the context length, and a launch without
`--max-running-requests` or with more than 128 (the most nkilib's `moe_tkg` takes in a decode step).

The checks of Qwen3.5's host and of the Kimi-K3 prefill are in
[When a scheduler starts](../design/runtime.md#when-a-scheduler-starts) and
[What a prefill launch refuses](../model-recipes/kimi-k3.md#what-a-prefill-launch-refuses).

## Weights, KV cache and attention

| Feature | GPT-OSS | Qwen3 | Qwen3-MoE | GLM-4 | Gemma 4 | DeepSeek-V2 | GLM-5.2 | Qwen3.5 | Kimi-K3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Checkpoint weights | MXFP4, required; experts run as MXFP4 (tested), or as BF16 dequantized on the host with `SGLANG_NEURON_GPT_OSS_EXPERTS=bf16` (tested ⁴ at `--tp-size 16 --ep-size 4`; served on the PD roles) | bf16; a quantized checkpoint is refused | bf16; a quantized checkpoint is refused | bf16; a quantized checkpoint is refused | bf16; a quantized checkpoint is refused | bf16; a quantized checkpoint is refused | the MXFP4 Quark export (`--quantization quark`), required; the Quark AttnFP8 export refused at both roles | a per-rank weight pack of the FP8 checkpoint (`SGLANG_NEURON_QWEN35_PACK`), required: dense weights in bf16, experts in MXFP8 | the MXFP4 experts (`compressed-tensors`), required |
| KV cache dtype (`--kv-cache-dtype`) | `fp8_e4m3` only | `auto`, `bf16`, `bfloat16` | `auto`, `bf16`, `bfloat16` | `auto`, `bf16`, `bfloat16` | `auto`, `bf16`, `bfloat16` | `auto`, `bf16`, `bfloat16` | `auto`, `bf16`, `bfloat16` (the kernels' own FP8 rows) | `auto`, `bf16`, `bfloat16` | `fp8_e4m3` only |
| KV pool | paged MHA | paged MHA | paged MHA | paged MHA | paged MHA, one uniform pool: the host declares two 512-wide pool heads a rank, which hold a sliding layer's 4 heads of 256 and a global layer's 1 head of 512 (`KVLayout`) | paged MLA | the prefill stages' paged pools of the kernels' 640-byte rows; the decode giga kernel's own slot caches | the paged pool of the 15 full-attention layers (`sglang_neuron/models/qwen35/kv_pool.py`); the GDN state in SGLang's request-table state pool | SGLang's hybrid linear-attention pool over the Neuron MLA pool (FP8 rows in DCP placement); the KDA state in SGLang's request-table state pool |
| Page size | 128 | 128 | 128 | 128 | 128 | 128 | 1,024 behind the PP4 prefill, 256 behind PP16 | 512 by default; 256, 1,024 or 2,048 (the recipes' 2,048) | 128 |
| Sliding-window layers | tested (the even layers, as the released checkpoints' `layer_types`) | refused | refused | refused (GLM-4 has none; a config.json that declares them is refused) | tested (50 of the 60 layers, window 1024, as the checkpoint's `layer_types`) | — | — | — | — |
| Hybrid SWA memory (SGLang's default on the chunk cache; `SGLANG_NEURON_HYBRID_SWA_RADIX=1` on the radix cache) | served on an aggregated server, tested pipelined on the chunk cache (PP2 x TP8); one pool on PD, with the hierarchical cache, and on the radix cache without the radix mode; the radix mode refused pipelined | one pool | one pool | one pool | refused (one pool; the launch rules require `--disable-hybrid-swa-memory`) | one pool | — | — | — |
| Prefix (radix) cache on a server that prefills | tested ³ | tested ³ | tested ² | tested ² | tested (hits logged on repeated prompts, tokens unchanged) | tested ³ | tested ² (whole pages on every stage) | refused on SGLang's default tree; tested on `neuron-chunk` (below) | refused on SGLang's default tree; tested on `--radix-cache-backend kimi-prompt` ⁵ |
| Radix cache on a disaggregated decode server (`--disaggregation-decode-enable-radix-cache`) | tested | served | served | served | served ⁹ | served | refused (the decode's caches are slot-private) | tested, on `neuron-chunk` (below) | refused (a DCP decode) |
| Prefix cache keyed on whole chunks (`--radix-cache-backend neuron-chunk`) on a server that prefills | tested, and served pipelined | served | served | served | served | refused | refused | tested on the PD prefill; served on the aggregated server | served, pipelined |
| Decode radix cache keyed on whole prefill chunks (`--radix-cache-backend neuron-chunk` with `--disaggregation-decode-enable-radix-cache`) | tested | served | served | served | served ⁹ | refused | refused | tested (each replica's tree on 2,048-token chunks, on the KV-split decode) | refused |
| Hierarchical cache (L2) on a server that prefills | tested (aggregated) | served | served | served | served | refused (the MLA pool keeps no host copy) | refused (no pipelined server's host tier has run on hardware) | served, on `neuron-chunk` | served, on `neuron-chunk` and pipelined; the kimi-prompt cache has a host tier of its own ⁵ |
| Radix cache on a pipelined or DCP decode server; hierarchical cache on a decode server, pipelined or on `neuron-chunk` | refused | refused | refused | refused | refused | refused | refused | hierarchical cache on the decode served with `SGLANG_NEURON_QWEN35_HICACHE=1`, on `neuron-chunk` too; no pipelined or `--dcp-size` decode | refused |

## Compiled shapes

| Feature | GPT-OSS | Qwen3 | Qwen3-MoE | GLM-4 | Gemma 4 | DeepSeek-V2 | GLM-5.2 | Qwen3.5 | Kimi-K3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chunk (`--chunked-prefill-size`) | 512, 1024, 2048, 4096 or 8192 | 512, 1024, 2048, 4096 or 8192 | 512, 1024, 2048, 4096 or 8192 | 512, 1024, 2048, 4096 or 8192 | the launch's value (the recipe's 512): the prefill graph is the model's own, not the segmented kernel's tiles | equal to `--context-length` | 4096 at PP4, 2048 at PP16 | whole KV pages and whole 512-token blocks of each DP replica's share (`--chunked-prefill-size` over the eight replicas: the recipes' 16,384 gives 2,048) | whole stripe cycles of 128 x `--dcp-size` tokens, at most 8,192 (the served 4,096) |
| Prefill tiles below the chunk | refused (the chunk's graph only) | tested ² | tested ² | tested ² | not built (the chunk's graph only) | tested ² (the smallest window that holds the cached prefix and the new tokens) | tested ² (1024 and 2048 at PP4, 1024 at PP16) | served: one tile of one KV page and at least 512 tokens, when the chunk is larger (none at the recipes' 2,048-token chunk and page) | tested ² (2,048 below the 4,096 chunk) |
| Decode batch ladder | tested on the aggregated server; one size on the DP-attention decode | tested | tested | tested | tested | tested | one size, the decode kernel's 32 slots | served (16, 32 and 64 rows a replica: `--cuda-graph-bs-decode 16 32 64`) | one size for each k: 4 requests a DP group at k = 3, 2 at k = 7 |
| Decode context ladder | not built (every step reads the whole context) | tested ² | tested ² | tested ² | tested (a quarter, a half and the whole context) | tested ² | not built (each slot holds the decode point's whole context) | served (a quarter, a half and the whole context, in whole pages; the DP cohort takes one rung a step) | not built (one table of the 1M context) |

² Run under the hardware runs' load, which reaches it; no check targets it alone.

³ Hits at every page and chunk edge against cold runs: aggregated on Qwen3-0.6B,
DeepSeek-V2-Lite and GPT-OSS, disaggregated on Qwen3-8B and on GPT-OSS's pipelined prefill.

⁴ One aggregated server of BF16 experts at TP16 EP4: determinism and GSM8K
0.8946, at the MXFP4 servers' floor of 0.868.

⁵ Kimi-K3's prefill: `--radix-cache-backend kimi-prompt` keeps the end of each finished prompt, so a next turn hits at
the exact end of the turn before. Its host tier (`SGLANG_NEURON_KIMI_PROMPT_HOST_SLOTS`,
`SGLANG_NEURON_KIMI_PROMPT_HOST_STATE_SLOTS`) takes the place of HiCache
([The kimi-prompt cache](../model-recipes/kimi-k3.md#the-kimi-prompt-cache)).

## Sampling and requests

| Feature | GPT-OSS | Qwen3 | Qwen3-MoE | GLM-4 | Gemma 4 | DeepSeek-V2 | GLM-5.2 | Qwen3.5 | Kimi-K3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `greedy` sampler | tested | tested | tested | tested | tested | tested | tested | tested, its only sampler | refused as the launch's sampler (the launch rules set `topk_topp`); a greedy request is tested on `topk_topp` |
| `temperature` and `topk_topp` samplers | served ¹ | served | served | served | served | served | `topk_topp` served on both roles; `temperature` refused | refused | `topk_topp` tested on both roles; `temperature` refused |
| `host` sampler (SGLang's sampler on the logits), with `--disable-overlap-schedule` | served ² | served | served | served | tested | served | refused | refused | refused |
| Structured outputs (`json_schema`, `regex`, `ebnf`, `structural_tag`), on the `host` sampler | served ² | served | served | served | tested | served | refused | refused | refused, at the tokenizer of both servers |
| Top logprobs (`SGLANG_NEURON_MAX_TOP_LOGPROBS`, 0 by default) | served ¹ | served | served | served | served | served | refused, and any logprobs | refused, and any logprobs | refused, and any logprobs |
| Streaming sessions on a server that prefills | refused | refused | refused | refused | refused | refused | refused | refused | refused |
| Deterministic inference (`--enable-deterministic-inference`), which SGLang serves only on its own attention backends | refused | refused | refused | refused | refused | refused | refused | refused | refused |

¹ Refused on the GPT-OSS giga-kernel decode, which serves greedy tokens only, with no logprobs (the
[GPT-OSS recipe](../model-recipes/gpt-oss.md#the-giga-kernel-decode)).
² Refused on a decode with DP attention, which GPT-OSS's disaggregated decode runs, and under pipeline parallelism.

The sampler is a launch setting, `SGLANG_NEURON_SAMPLER`: `greedy` by default, `temperature`, `topk_topp` or `host`
([Device sampling](features-guide.md#device-sampling)). On every model, a request the launch cannot serve exactly is
refused at admission with an error, never served with the option dropped. On a device sampler, among them:

- a sampled request on a `greedy` server (SGLang's default temperature is 1), top-k, top-p or min-p on a `temperature`
  server, and a top-k above 256 on a `topk_topp` server;
- more top logprobs than `SGLANG_NEURON_MAX_TOP_LOGPROBS`, token-ID logprobs, prompt logprobs (the prefill graph
  returns the last token's logits only) and the sampling mask;
- on the GPT-OSS and GLM-5.2 giga-kernel decodes, any logprobs (`return_logprob`): their fused tails return tokens
  only;
- on Qwen3.5, a sampled request and any logprobs: its kernels sample greedily and return no logprobs;
- on Kimi-K3, at the tokenizer of both servers: structured outputs, penalties, min-p, `min_new_tokens`, logit bias, a
  custom logit processor, logprobs and hidden states;
- penalties, `min_new_tokens`, logit bias, custom logit processors and grammar-constrained decoding (structured
  outputs), which the `host` sampler serves;
- `n` above 1 on a prefill or decode server, and a seeded sampled request with `n` above 1, whose copies would draw the
  same tokens;
- on every sampler, beam search, and on the `host` sampler a sampled request's seed, which SGLang's sampler reads only
  under deterministic inference;
- prompts given as `input_embeds` or with `positional_embed_overrides`, and token ids outside the vocabulary.

The full list is in [At admission](../design/runtime.md#at-admission).

## Target

GPT-OSS's graphs and the Kimi-K3 prefill run on Trainium3 only, and a launch on another target is refused. The Qwen3.5
megakernel declares Trainium3 as its one target. The other models are not refused on another target. They are run on
Trainium3 nodes, except Gemma 4, which is run on a Trainium2 trn2.3xlarge and has not been run on Trainium3.
Every model runs two physical NeuronCores per logical core (LNC2), and the GPT-OSS
giga-kernel decode and every role of GLM-5.2, Qwen3.5 and Kimi-K3 one (LNC1); the platform sets the degree
from the layout, and refuses a launch value other than the host's default that contradicts it.
