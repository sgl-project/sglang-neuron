# DeepSeek-V2-Lite

Prefill/decode on two Trn3 nodes: one prefill server and five decode servers, each TP4 on its own four
NeuronCores. Start the image on each node and run the scripts in the order [the examples page](../README.md) gives.

| role | node | script |
|---|---|---|
| decode, five servers | decode node, NeuronCores 0-3, 4-7, ... 16-19, ports 30001 to 30005 | `run_decode.sh` |
| prefill | prefill node | `run_prefill.sh` |
| balancer | prefill node | `run_balancer.sh` |
| client and accuracy check | prefill node | `run_client.sh`, `run_gsm8k.sh` |

## Launch

```bash
# decode node: five decode servers, each on its own four NeuronCores and port
for i in $(seq 0 4); do
  CORES=$((4 * i))-$((4 * i + 3)) PORT=$((30001 + i)) HOST=<decode node address> \
    bash /opt/sglang-neuron/examples/deepseek_v2_lite/run_decode.sh > decode$i.log 2>&1 &
done
cat decode*.log | grep -c 'fired up'    # decode node: 5 once every decode server is ready
HOST=<prefill node address> bash /opt/sglang-neuron/examples/deepseek_v2_lite/run_prefill.sh    # prefill node
D=<decode node address>                                                         # prefill node, second shell
PREFILL_URL=http://<prefill node address>:30000 DECODE_URL=http://$D:30001 \
  bash /opt/sglang-neuron/examples/deepseek_v2_lite/run_balancer.sh \
  --decode http://$D:30002 --decode http://$D:30003 \
  --decode http://$D:30004 --decode http://$D:30005
```

## Check

```bash
# prefill node, third shell
bash /opt/sglang-neuron/examples/deepseek_v2_lite/run_client.sh    # the answer starts "17 * 23 = 391."
bash /opt/sglang-neuron/examples/deepseek_v2_lite/run_gsm8k.sh     # Accuracy: about 0.38 on the first 100 questions
```

## Aggregated (one node)

One server, `run_server.sh`, runs both prefill and decode, TP4 on four NeuronCores, on `127.0.0.1:30000`:

```bash
bash /opt/sglang-neuron/examples/deepseek_v2_lite/run_server.sh
# second shell (docker exec -it sglang bash), once the server is ready
URL=http://127.0.0.1:30000 bash /opt/sglang-neuron/examples/deepseek_v2_lite/run_client.sh    # the answer starts "17 * 23 = 391."
PORT=30000 bash /opt/sglang-neuron/examples/deepseek_v2_lite/run_gsm8k.sh                     # Accuracy: about 0.37
```

## Notes

- The checkpoint: `deepseek-ai/DeepSeek-V2-Lite` from Hugging Face, as `<models dir>/DeepSeek-V2-Lite`.
- Defaults: `MODEL_PATH=/models/DeepSeek-V2-Lite`; `PORT=30000` for the prefill and the one-node server, `PORT=30001`
  and `CORES=0-3` for a decode server.
- The loop runs the decode servers in the background of one shell: stop them with `kill -INT $(jobs -p)` in that shell.
- Each server takes 64 requests at a time, and a request's prompt and answer together take up to 4,096 tokens.
- Fewer decode servers work the same way: start as many as you need and give the balancer each one's URL.
- DeepSeek-V2-Lite is a base model, not a chat model: its answer goes on after the number.
- [The DeepSeek-V2 recipe](../../docs/model-recipes/deepseek-v2.md) explains the flags.
