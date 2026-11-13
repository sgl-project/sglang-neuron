# Gemma 4 31B

One aggregated server on one trn2.3xlarge, TP4 on its four NeuronCores, text only. Start the image on the instance and
run the scripts below inside it.

| role | node | script |
|---|---|---|
| server (prefill and decode) | trn2.3xlarge, NeuronCores 0-3, port 30000 | `run_server.sh` |
| client and accuracy check | the same container, second shell | `run_client.sh`, `run_gsm8k.sh` |

## Launch

```bash
bash /opt/sglang-neuron/examples/gemma4_31b/run_server.sh
```

The first launch compiles 12 decode graphs and the prefill graph (about 74 minutes on a trn2.3xlarge); a relaunch with
the same flags and compile cache takes about 20 minutes.

## Check

```bash
# second shell (docker exec -it sglang bash), once the server is ready
bash /opt/sglang-neuron/examples/gemma4_31b/run_client.sh    # the answer "content": "391"
bash /opt/sglang-neuron/examples/gemma4_31b/run_gsm8k.sh     # Accuracy: about 0.82 on the first 100 questions
```

## Notes

- The checkpoint: `google/gemma-4-31B-it` from Hugging Face, as `<models dir>/gemma-4-31B-it`.
- Defaults: `MODEL_PATH=/models/gemma-4-31B-it`, `PORT=30000`.
- The server takes 8 requests at a time, a request's prompt and answer together take up to 16,384 tokens, and the KV
  pool holds 20,480 tokens across all requests.
- Disaggregated prefill and decode is not tested (the bf16 weights do not fit half a trn2.3xlarge).
- [The Gemma 4 recipe](../../docs/model-recipes/gemma4.md) explains the flags and the model.
