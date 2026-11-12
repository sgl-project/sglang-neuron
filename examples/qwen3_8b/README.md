# Qwen3-8B

Prefill/decode on two Trn3 nodes: one prefill server and one decode server, each TP4 on four NeuronCores. Start the
image on each node and run the scripts in the order [the examples page](../README.md) gives.

| role | node | script |
|---|---|---|
| decode | decode node | `run_decode.sh` |
| prefill | prefill node | `run_prefill.sh` |
| balancer | prefill node | `run_balancer.sh` |
| client and accuracy check | prefill node | `run_client.sh`, `run_gsm8k.sh` |

## Launch

```bash
HOST=<decode node address> bash /opt/sglang-neuron/examples/qwen3_8b/run_decode.sh      # decode node
HOST=<prefill node address> bash /opt/sglang-neuron/examples/qwen3_8b/run_prefill.sh    # prefill node
# prefill node, second shell, once both servers are ready
PREFILL_URL=http://<prefill node address>:30000 DECODE_URL=http://<decode node address>:30001 \
  bash /opt/sglang-neuron/examples/qwen3_8b/run_balancer.sh
```

## Check

```bash
# prefill node, third shell (docker exec -it sglang bash)
bash /opt/sglang-neuron/examples/qwen3_8b/run_client.sh    # "content": "17 * 23 = 391"
bash /opt/sglang-neuron/examples/qwen3_8b/run_gsm8k.sh     # Accuracy: about 0.92 on the first 100 questions
```

## Aggregated (one node)

One server, `run_server.sh`, runs both prefill and decode, TP4 on four NeuronCores, on `127.0.0.1:30000`:

```bash
bash /opt/sglang-neuron/examples/qwen3_8b/run_server.sh
# second shell (docker exec -it sglang bash), once the server is ready
URL=http://127.0.0.1:30000 bash /opt/sglang-neuron/examples/qwen3_8b/run_client.sh     # "content": "17 * 23 = 391"
PORT=30000 bash /opt/sglang-neuron/examples/qwen3_8b/run_gsm8k.sh                      # Accuracy: about 0.92
```

## Notes

- Defaults: `MODEL_PATH=/models/Qwen3-8B`, `PORT=30000` (prefill) and `PORT=30001` (decode), and the decode server
  on NeuronCores `CORES=0-3`.
- Each server takes 64 requests at a time, and a request's prompt and answer together take up to 4,096 tokens.
- The validated layout runs a second decode server next to the first. Start it, and the balancer with both:

  ```bash
  # decode node, second shell
  CORES=4-7 PORT=30002 HOST=<decode node address> bash /opt/sglang-neuron/examples/qwen3_8b/run_decode.sh
  # prefill node, second shell, in place of the balancer above, once all three servers are ready
  PREFILL_URL=http://<prefill node address>:30000 DECODE_URL=http://<decode node address>:30001 \
    bash /opt/sglang-neuron/examples/qwen3_8b/run_balancer.sh --decode http://<decode node address>:30002
  ```
- `run_client.sh` turns Qwen3's thinking off (`"chat_template_kwargs": {"enable_thinking": false}`).
- [The Qwen3 recipe](../../docs/model-recipes/qwen3.md) explains the flags, and the TP8 variant.
