# Qwen3.8 27B

One aggregated server on one trn2.3xlarge, TP8 on its eight NeuronCores (one rank per physical core), text only.
Start the image on the instance and run the scripts below inside it.

| role | node | script |
|---|---|---|
| server (prefill and decode) | trn2.3xlarge, NeuronCores 0-7, port 30000 | `run_server.sh` |
| client and accuracy check | the same container, second shell | `run_client.sh`, `run_gsm8k.sh` |

## Launch

```bash
bash /opt/sglang-neuron/examples/qwen38_27b/run_server.sh
```

The first launch compiles 9 decode graphs and the prefill graphs of the 2048-token chunk and the 512-token tile (22
minutes on a trn2.3xlarge before the tile, which adds one prefill graph); a relaunch with the same flags and compile
cache takes about 6 minutes.

## Check

```bash
# second shell (docker exec -it sglang bash), once the server is ready
bash /opt/sglang-neuron/examples/qwen38_27b/run_client.sh    # the answer ends with "391"
bash /opt/sglang-neuron/examples/qwen38_27b/run_gsm8k.sh     # Accuracy: about 0.99 on the first 100 questions
```

## Notes

- The checkpoint: `Qwen/Qwen3.8-27B` from Hugging Face, as `<models dir>/Qwen3.8-27B`.
- Defaults: `MODEL_PATH=/models/Qwen3.8-27B`, `PORT=30000`, the prefix cache on (`RADIX=0` serves without it), the
  512-token prefill tile below the chunk (`PREFILL_TILES=2048` compiles the chunk alone).
- Extra arguments go to the server. `--mamba-ssm-dtype bfloat16 --max-mamba-cache-size 85` with `PREFILL_TILES="512
  1024"` stores the GDN state in bf16 and runs both tiles ([Prefill tiles](../../docs/model-recipes/qwen38.md)).
- The server takes 4 requests at a time, a request's prompt and answer together take up to 16,384 tokens, and the KV
  pool holds 65,536 tokens across all requests.
- The model reasons before it answers (its chat template turns thinking on): give a chat request enough `max_tokens`.
- [The Qwen3.8 recipe](../../docs/model-recipes/qwen38.md) explains the flags, the model and the measured results.
