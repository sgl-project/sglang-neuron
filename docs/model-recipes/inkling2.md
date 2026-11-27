# Inkling-2-Mini on Neuron

`Inkling2LoopForCausalLM`, the language model of
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), runs on the
[runtime](../design/runtime.md) as an aggregated server, TP8 on the eight physical NeuronCores of a trn2.3xlarge (one rank
a core, LNC1), with the radix cache. The host is `sglang_neuron/models/inkling2/sglang_model.py`; the graphs are
`sglang_neuron/models/inkling2/graph.py` over the arithmetic in `ops.py`.

## The model

Inkling-2-Mini is a looped mixture-of-experts transformer: layers 6 to 17 of its 24 run three times with shared
weights, so a token takes 48 layer visits, with gated messages between the passes. 20 layers attend a sliding window of
1024 tokens and 4 the whole context, with a learned relative-position bias instead of RoPE, short causal convolutions
on the keys, values and both residual branches, per-token attention sinks, an output gate and attention residuals
across depth; the MoE has 128 experts, top 8, with a softcapped SwiGLU. The checkpoint stores the experts in NVFP4
(layers 0 to 19) and MXFP8 (20 to 23); the loader dequantizes them to bf16 (`weights.py`, bit-identical to Hugging
Face's load ops).

Per rank: one KV head of 128 and three query heads a layer, the experts' intermediate dimension split eight ways (each
rank keeps every expert), and the vocabulary split into the LM head's shards; the device sampler reads each rank's
shard. The decode MoE runs as one NKI kernel for buckets of up to two rows (`kernels/moe_decode_hw.py`: each routed
expert read with hardware DMA descriptors, the expert id in a register), the torch gather above.

## The caches and the radix cache

Each layer visit has its own KV layer, 48 in all, read through SGLang's block table: a full-attention visit reads the
request's pages, a sliding visit the pages of its window, and both mask past the window or the causal edge themselves,
so the platform sees full-attention layers. The short convolutions' input windows (three tokens a convolution, about
700,000 bf16 elements a request a rank) are a record kept in 11 more KV layers: read in the page of the request's last
processed token and written in the page of the step's last token, and a prefill chunk also writes it at the end of
every page it fills. Every full page so holds the record of the state after its last token, and SGLang's radix cache
needs nothing else: a request whose prompt hits cached pages reads its last page's record and continues as the cold
run does (checked below). The pool is the platform's MHA pool at those 59 layers, which the host builds itself
(`configure_kv_cache`), since SGLang would size it from the config's 24.

The pages are 256 tokens and a prompt prefills in whole pages (`--page-size 256`, `--chunked-prefill-size` a multiple
of it).

## Serving

The server loads a text view of the checkpoint: its `text_config` as `config.json`, the rest symlinked
(`examples/inkling2_mini/prepare_checkpoint.sh`, `tools/inkling2/make_text_view.sh`). SGLang would otherwise serve the
checkpoint's `Inkling2LoopForConditionalGeneration` as a multimodal model and look for its image processor. The
checkpoint's config type is newer than the image's transformers; the catalog entry's `hf_config_types` registers it in
every process when the plugin loads, before SGLang reads the config.

```bash
python3 -m sglang.launch_server --model-path /models/Inkling-2-Mini-text --served-model-name inkling \
  --tp-size 8 --context-length 16384 --page-size 256 --chunked-prefill-size 256 \
  --max-running-requests 2 --cuda-graph-bs-decode 2 --max-total-tokens 16384 --mem-fraction-static 0.85 \
  --dtype bfloat16 --reasoning-parser inkling --host 127.0.0.1 --port 30000
```

The graphs compile with the transformer model type and the hierarchy splitter, the decode at `-O3`: its 96
all-reduces a step run inside `-O1`'s modular flow otherwise (`neuron_compiler_args`). The prefill keeps the platform's
`-O1`. The scratchpad page is 1024 MiB (the catalog entry): beside the 6.5 GB of weights a core, the platform's
3584 MiB page leaves the prefill program no room.

## Validation

One trn2.3xlarge, the command above unless noted, greedy.

- Against Hugging Face's `Inkling2LoopForCausalLM` on CPU: greedy continuations of reference prompts of 759, 1047,
  1263, 1551 and 4863 tokens (past the 1024 window) match for 24 of 24 tokens; a 2021-token prompt diverges at step 3,
  where Hugging Face's top two logits are 0.000 apart. Teacher-forced over 24 steps of four prompts (5 to 4863
  tokens): 91 of 96 steps with Hugging Face's argmax; at the 5 others its top-two gap is 0.000, 0.000, 0.038, 0.039
  and 0.141.
- GSM8K, the first 100 questions, five-shot at temperature 0: 0.870, none unparsed.
- A code hidden at 10, 50 and 90 percent of filler, at 2147 and 12122 prompt tokens: 6 of 6.
- The radix cache: for prompts of 759 to 4863 tokens, a repeat (a full hit, 512 to 4608 tokens cached) and a second
  prompt on the first one's leading pages (a partial hit) give the cold run's tokens exactly.
- 30 minutes at 2 clients, prompts of 100 to 4000 tokens cut from one document (so requests hit each other's pages),
  a share cancelled midway: 332 completed, 61 cancelled, no error, median TPOT 38.8 ms in the first 10 minutes and
  35.7 in the last.

Speed, `sglang.bench_serving`, random 1024-token prompts and 1024 output tokens (no shared pages):

| Clients | TTFT p50 | TPOT p50 | Output tokens/s |
| --- | --- | --- | --- |
| 1 | 1.42 s | 30.1 ms | 32.0 |
| 2 | 1.08 s | 30.2 ms | 63.1 |

A conversation turn of 128 tokens on a cached history starts in 0.37 s at 1024, 4096 and 8192 history tokens (cold:
1.77, 5.94 and 11.52 s).

At a 4k context with more rows (`--context-length 4096 --max-running-requests 8 --cuda-graph-bs-decode 4`, decode
graphs of 4 and 8 rows, otherwise the same), same benchmark; greedy outputs and radix hits as above:

| Clients | TTFT p50 | TPOT p50 | Output tokens/s |
| --- | --- | --- | --- |
| 1 | 1.29 s | 36.9 ms | 26.4 |
| 2 | 0.98 s | 36.7 ms | 52.5 |
| 4 | 1.29 s | 37.2 ms | 101.4 |
| 8 | 5.37 s | 63.0 ms | 101.8 |

One and two clients run in the 4-row graph, whose MoE is the torch gather (the NKI kernel serves up to 2 rows).

Tool calls: with the chat template's tool declarations, the model calls tools in its format (`<|message_model|>` name
`<|content_invoke_tool_json|>` arguments): three tasks (two parallel weather calls and a difference; an order lookup and
a sum; a weather call answered in Portuguese) are answered right in 6 of 6 runs (two reasoning efforts), 2 to 3 turns
each, every later turn starting in 0.37 s on the radix cache. SGLang's `--tool-call-parser inkling` constrains a tool
call with a `structural_tag` grammar, which the device sampler does not apply: on the device samplers a request with
`tools` is refused, so the OpenAI tools API needs the `host` sampler (`SGLANG_NEURON_SAMPLER=host` with
`--disable-overlap-schedule`), not measured here.

## What the launch refuses

A `--page-size` other than 256 or a chunk that is not whole pages, a `--tp-size` other than 8 on the device, PD
(`--disaggregation-mode`), DP attention, pipeline stages, speculative decoding (the checkpoint's DSpark draft is not
served), expert parallelism, and a KV cache dtype other than bf16.

## Limitations

- The KV pool is 16,384 tokens at a 16k context: each core's HBM holds the weights, the pool and the programs'
  scratchpad (the decode program's 3 GB, then a page more for the prefill), so a larger pool leaves the prefill no
  room. The sliding visits keep full-length KV (SGLang's hybrid SWA pool and the radix cache's chunk tree are
  exclusive on this platform).
- Images and the DSpark draft are not served.
- The OpenAI tools API (`--tool-call-parser inkling`) needs the host sampler (above).
