# Llama 3 on Neuron

`LlamaForCausalLM` (Llama-3.1-8B-Instruct) runs on the [runtime](../design/runtime.md) in bf16 as an aggregated server,
TP4 on the four NeuronCores of a trn2.3xlarge. The host is `sglang_neuron/models/llama/sglang_model.py`; the graphs are
the shared dense transformer in `sglang_neuron/models/transformer/`, with the model's rotary tables over Llama 3.1's
rescaled frequencies (`sglang_neuron/models/llama/model.py`).

## The model

Llama 3 has GQA attention (32 query heads over 8 KV heads of 128) with no biases and no QK norm, RoPE over the whole
head, a SwiGLU MLP with separate gate and up projections, and an untied LM head over a 128,256-token vocabulary. Each
phase runs its attention in the paged attention kernels. Prefill runs `qkv_paged` (QKV, RoPE and the K/V write), then
the segmented attention over the prefix and the chunk, then the output projection. Decode runs `attention_block_tkg`,
which does the QKV, RoPE, attention over the row's pages, the K/V write and the output projection in one call. In both,
an all-reduce over the attention TP ranks follows, then the MLP shard and a second all-reduce.

The attention heads, the MLP and the vocabulary split over the attention TP ranks: at TP4 each rank holds eight query
heads and two KV heads, at TP8 four and one. The LM head stays sharded by vocabulary, and the device sampler reads
each rank's shard of the logits.

### The RoPE

Llama 3.1 and 3.2 checkpoints carry a `llama3` RoPE scaling (`rope_scaling`, or `rope_parameters` in current
transformers): `factor`, `low_freq_factor`, `high_freq_factor` and `original_max_position_embeddings` (8, 1, 4 and
8192 for Llama 3.1). The scaling is a one-time rescale of the inverse frequencies before the cos/sin tables are built:
by wavelength, frequencies above `original / low_freq_factor` are divided by `factor`, those below `original /
high_freq_factor` are kept, and the band between is blended linearly. For Llama 3.1's 64 frequencies that is 29 kept,
6 blended and 29 divided by 8. The tables themselves are the shared ones (`cos.to(bf16)` of position times frequency),
which the kernels consume unchanged, so only `inv_freq` is Llama 3.1's own. The
rescaling changes the tables at every position, not only past 8192, so it is needed for the model's output at any
context. A checkpoint without a scaling (Llama 3, or a `rope_type` of `default`) uses the plain frequencies.

## Serving

The commands below run inside the container, with the checkpoint mounted at `/models`; the [container
guide](../guides/container.md) has the `docker run` for one node. The example has the aggregated server as scripts
([Llama-3.1-8B-Instruct](../../examples/llama31_8b/)), and [examples/README.md](../../examples/README.md#run-a-model)
gives the order to run them.

Aggregated, on the four NeuronCores of a trn2.3xlarge, with the validated flags:

```bash
python3 -m sglang.launch_server --model-path /models/Llama-3.1-8B-Instruct --tp-size 4 \
  --context-length 4096 --chunked-prefill-size 2048 --max-running-requests 64 --max-total-tokens 343040 \
  --mem-fraction-static 0.6 \
  --host 0.0.0.0 --port 30000
```

`--host 0.0.0.0` listens inside the container; publish the port on the host's loopback only
(`-p 127.0.0.1:30000:30000`), as the [container guide](../guides/container.md#3-start-a-container) explains.

`--context-length` is required: without it SGLang takes the checkpoint's 131,072, which the graphs do not serve (the
segmented prefill kernel's tiles stop at 8192, and the decode attention's flash path is validated to 8K). The example
serves 4096; 8192 is also tested ([Measured](#measured)), on the same KV pool, which then holds 41 requests of a full
8192 tokens at once.

`--max-total-tokens 343040` is the KV pool `--mem-fraction-static 0.6` leaves room for on a trn2.3xlarge (42 GiB of KV:
an oversized pool is refused at 345,344 tokens, and the reading moves by about a page between launches).

Disaggregated, the prefill server and each decode server take the same model flags plus their role's:
`--disaggregation-mode prefill` or `--disaggregation-mode decode`, `--disaggregation-transfer-backend nixl`, and the
prefill's `--disaggregation-bootstrap-port`. The decode server is measured alone ([Measured](#measured)); the pair is
not run ([Limitations](#limitations)).

## Compiled shapes

- **Prefill.** A graph of the chunk (`--chunked-prefill-size`, one of the segmented kernel's KV segment tiles: 512,
  1024, 2048, 4096 or 8192) and one of each smaller token count `--cuda-graph-bs-prefill` lists (each also such a
  tile) or, without that flag, of each smaller tile that splits evenly over the attention TP ranks.
- **Decode.** A graph per batch size up to `--max-running-requests`: the powers of two below it and the count itself,
  unless `--cuda-graph-bs-decode` or the decode entry of `--cuda-graph-config` lists the sizes. Each compiles over the
  whole context and over a quarter and a half of it, and a step runs the smallest batch and context that hold it.
- `--max-total-tokens` sizes the KV pool, whose page count is a dimension of every graph; state it, so a relaunch finds
  its graphs in torch-neuronx's compile cache. A token of KV is 32 layers x 8 KV heads x 128 x 2 (K and V) x 2 bytes =
  128 KiB.

## Measured

All numbers are from one trn2.3xlarge, TP4, bf16, the flags above and the greedy device sampler unless a line says
otherwise, on the sglang-neuron image v0.1.0 (Neuron SDK 2.33). The checkpoint is `unsloth/Llama-3.1-8B-Instruct` at
revision `4699cc7`, an ungated copy of `meta-llama/Llama-3.1-8B-Instruct` with the same four weight files and
`model.safetensors.index.json`; its `config.json` adds a pad token id (128004) and an explicit `head_dim` of 128, which
the host reads the same either way.

Accuracy:

- Greedy tokens of three chat prompts (54, 2,500 and 3,900 tokens), 64 new tokens each, against Hugging Face
  transformers in bf16 on CPU: 151 of 151 identical, alone and sent at once.
- At `--context-length 8192`, prompts of 6,000 and 7,900 tokens: 64 of 64 identical each, alone. Sent four at once,
  the 7,900-token answer changes at step 41, where the reference's top two logits are 0.125 apart (one bf16 step).
- `examples/llama31_8b/run_gsm8k.sh` (first 100 questions, five-shot, temperature 0): 0.78, none unparsed.
- A code hidden at 10, 50 and 90 percent of filler text is answered exactly at 2,000, 6,000 and 7,900 tokens (9 of 9).
- Prefix cache: hits on 128, 2,048, 2,176 and 3,200 shared tokens give the cold run's tokens in 8 of 8 cases; with
  3,200 tokens shared a request takes 401 ms against 546 ms cold.
- The `host` sampler (`SGLANG_NEURON_SAMPLER=host`, `--disable-overlap-schedule`): one token flips at a 0.125 gap and
  one at an exact tie; a `json_schema` request returns a document of the schema.

Speed, one client, random-token prompts:

| Input / output tokens | TTFT p50 | TPOT p50 |
| --- | --- | --- |
| 1024 / 128 | 69 ms | 10.42 ms |
| 3900 / 128 | 203 ms | 10.60 ms |
| 7900 / 128 (`--context-length 8192`) | 473 ms | 10.76 ms |

- Throughput: 1024-token prompts with 128 new tokens give 92 output tokens/s at 1 client and 780, 1,131 and 1,381 at
  16, 32 and 64, where all 64 rows fill. 128-token prompts with 512 new tokens give 98 tokens/s at a TPOT of 10.1 ms
  at 1 client and 2,778 at 21.0 ms at 64.
- Steadiness, TPOT p90 and p99 against the median (the bounds of [Performance
  characteristics](../design/disaggregated-inference.md#performance-characteristics) are 1.01 and 1.02): a decode
  server alone (`--disaggregation-mode decode` on SGLang's fake transfer, driven by `sglang.bench_serving
  --fake-prefill`) is at 1.002x to 1.004x at 8, 16 and 64 clients. The aggregated server holds the bounds with long
  answers (1, 4 and 8 clients; 1.0099x and 1.0119x at 8 clients of 1024 / 1024), and is at 1.088x and 1.119x with
  128-token answers at 8 clients, where prompts' prefill chunks run between decode steps.
- Under load: 80 clients on the 64 rows for 30 minutes, prompts of 256 to 3,000 tokens, answers of 64 to 512 with
  `ignore_eos`, 15 percent of the requests cancelled midway: 8,284 requests, no error, every request not cancelled
  returned its full answer, a median TPOT of 54.4 ms in minutes 1 to 6 and 53.5 ms in the last five, and SGLang's idle
  check of the KV pool and request slots stayed silent.
- Traces ([Profiling](../guides/profiling.md)): on the aggregated server under a prefill-only load, every prefill
  launch is asynchronous, with a launch share of 0.05 to 0.07 (the bound is 0.25). On the decode server alone at 64
  clients, the p99 idle gap before a step is 0.004 ms (bound 0.05) and the p99 `cc_exec_barrier` 0.043 ms (bound 0.1).
- Disaggregated layout ([Onboarding a model](../model-dev/onboarding-models.md#5-the-disaggregated-layout)), from these
  rates: a 1024-token prefill runs 52.68 ms on the device (R_p 18.98 requests/s), and the decode alone at 64 rows
  finishes 5.51 requests/s, so one prefill server keeps three decode servers fed, busy 0.87 of the time.
- Startup: 12.5 to 13 minutes on a cold compile cache (84 decode graphs and 12 prefill graphs), 8 minutes from a warm
  one; the `host` sampler's first launch takes 26 minutes.

## What is refused

A launch the graphs can't serve fails at startup, naming the setting at fault:

- a model dtype other than bfloat16 (`--dtype`, or the checkpoint's with `auto`);
- a quantized checkpoint (the FP8 exports of Llama 3.1 are not served);
- sliding-window attention layers (`use_sliding_window`, or `sliding_attention` in `layer_types`);
- attention biases (`attention_bias`) and MLP biases (`mlp_bias`);
- a RoPE scaling other than `llama3` or the default (`yarn`, `linear`, `dynamic`, `longrope`);
- attention heads that do not split over the attention TP ranks, KV heads that neither split over them nor replicate
  evenly across them, and an intermediate size that does not split over them;
- an `--ep-size` other than 1, since the model has no experts to split;
- a KV cache dtype other than `auto`, `bf16` or `bfloat16`, and a page size other than 128;
- the features every model refuses ([What a launch refuses](../design/runtime.md#what-a-launch-refuses)).

## Limitations

- Disaggregated prefill and decode is not run as a pair: the KV moves over NIXL and EFA between nodes, and a
  trn2.3xlarge has no EFA. The layout above is computed from the decode server measured alone.
- Not run: a context above 8192, TP8, Trn3, and the `temperature` and `topk_topp` samplers. Quantized (FP8) checkpoints
  and speculative decoding are refused at launch ([What is refused](#what-is-refused)).
- Other Llama checkpoints the entry accepts but that have not been run: Llama-3.1-8B (base), Llama-3-8B (default
  RoPE), Llama-3.2-3B (factor 32, tied embeddings) and other `LlamaForCausalLM` models with a head dim of 128.
