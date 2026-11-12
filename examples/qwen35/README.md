# Qwen3.5-397B-A17B

Prefill/decode on two Trn3 nodes at a context of up to 262,144 tokens: a prefill server of 128 ranks (eight
attention-DP replicas of TP16 on the whole-model megakernel), and a decode server of the same layout that splits each
replica's KV cache over its 16 ranks and keeps a prefix cache of its own. Start the image on each node and run the
scripts in the order [the examples page](../README.md) gives.

| role | node | script |
|---|---|---|
| decode | decode node | `run_decode.sh` |
| prefill | prefill node | `run_prefill.sh` |
| balancer | prefill node | `run_balancer.sh` |

## Launch

```bash
HOST=<decode node address> PACK=/models/<pack> bash /opt/sglang-neuron/examples/qwen35/run_decode.sh      # decode node
HOST=<prefill node address> PACK=/models/<pack> bash /opt/sglang-neuron/examples/qwen35/run_prefill.sh    # prefill node
PREFILL_URL=http://<prefill node address>:30000 DECODE_URL=http://<decode node address>:30000 \
  bash /opt/sglang-neuron/examples/qwen35/run_balancer.sh                          # prefill node, second shell
```

## Check

```bash
bash /opt/sglang-neuron/examples/qwen35/run_client.sh    # the answer's text contains 391
bash /opt/sglang-neuron/examples/qwen35/run_gsm8k.sh     # prints the accuracy
```

Run both in a third shell on the prefill node, the balancer's host: `run_client.sh` sends one greedy request to
`/generate`, and `run_gsm8k.sh` asks the first `NUM_QUESTIONS` (100) of GSM8K's test split, five-shot and greedy.

## Notes

- **Checkpoint and weight pack:** the servers serve a text-only view of the Qwen3.5-397B-A17B FP8 checkpoint
  (`MODEL_PATH`, by default `/models/Qwen3.5-397B-A17B-FP8-text`): a directory whose `config.json` is the FP8
  checkpoint's `text_config` with the architecture `Qwen3_5MoeForCausalLM`, and every other file a link to the
  checkpoint's. Both load a weight pack built from it (`PACK`): the checkpoint's weights in each rank's own layout,
  which `sglang_neuron/models/qwen35/weights.py` describes. Build both from a checkout of this repository as the
  [Qwen3.5 recipe](../../docs/model-recipes/qwen35-radix.md#build-the-inputs) gives (`tools/qwen35/make_text_view.py`,
  then `python -m tools.qwen35.build_pack.pack`).
- **Restart both servers together:** if only one restarts, the other keeps a stale transfer peer and its KV transfers
  fail.
- **Prefix-cache hits across turns:** the balancer picks each request's prefill replica by load, so the turns of one
  session reach different replicas and reuse little of the prefix cache. The recipe explains the replica affinity a
  front end needs for that.

[The Qwen3.5 PD recipe](../../docs/model-recipes/qwen35-pd.md) explains the layout, its settings and the limits
of this layout.
