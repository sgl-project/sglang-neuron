# Llama-3.1-8B-Instruct

One aggregated server on one trn2.3xlarge, TP4 on its four NeuronCores. Start the image on the instance and run the
scripts below inside it.

| role | node | script |
|---|---|---|
| server (prefill and decode) | trn2.3xlarge, NeuronCores 0-3, port 30000 | `run_server.sh` |
| client and accuracy check | the same container, second shell | `run_client.sh`, `run_gsm8k.sh` |

## Launch

```bash
bash /opt/sglang-neuron/examples/llama31_8b/run_server.sh
```

The first launch compiles the decode graphs and the prefill graphs (about 13 minutes on a trn2.3xlarge); a relaunch
with the same flags and compile cache takes about 8 minutes.

## Check

```bash
# second shell (docker exec -it sglang bash), once the server is ready
bash /opt/sglang-neuron/examples/llama31_8b/run_client.sh    # the answer "content": "391"
bash /opt/sglang-neuron/examples/llama31_8b/run_gsm8k.sh     # Accuracy: about 0.78 on the first 100 questions
```

## Notes

- The checkpoint: `meta-llama/Llama-3.1-8B-Instruct` from Hugging Face (gated; accept Meta's licence on the model
  page), as `<models dir>/Llama-3.1-8B-Instruct`. The measured results were taken on `unsloth/Llama-3.1-8B-Instruct`
  (revision `4699cc7`), an ungated copy with the same weight files and index ([the
  recipe](../../docs/model-recipes/llama-3.md#measured)).
- Defaults: `MODEL_PATH=/models/Llama-3.1-8B-Instruct`, `PORT=30000`.
- The server takes 64 requests at a time, a request's prompt and answer together take up to 4,096 tokens, and the KV
  pool holds 343,040 tokens across all requests. `--context-length 8192` is also tested (pass it after the script's
  flags: the last value wins).
- Disaggregated prefill and decode is not run as a pair; the recipe gives the decode server measured alone and the
  layout computed from it.
- [The Llama 3 recipe](../../docs/model-recipes/llama-3.md) explains the flags and the model's RoPE.
