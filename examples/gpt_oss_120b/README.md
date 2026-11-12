# GPT-OSS-120B

Prefill/decode on two Trn3 nodes: a prefill server of five replicas (each attention TP4, MoE EP2) and a decode server
of eight DP-attention replicas (attention TP8, MoE EP64). Start the image on each node and run the scripts in the order
[the examples page](../README.md) gives.

| role | node | script |
|---|---|---|
| decode | decode node | `run_decode.sh`, or `run_decode_giga.sh` for the giga-kernel decode |
| prefill | prefill node | `run_prefill.sh` |
| balancer | prefill node | `run_balancer.sh` |
| client and accuracy check | prefill node | `run_client.sh`, `run_gsm8k.sh` |

## Launch

```bash
HOST=<decode node address> bash /opt/sglang-neuron/examples/gpt_oss_120b/run_decode.sh      # decode node
HOST=<prefill node address> bash /opt/sglang-neuron/examples/gpt_oss_120b/run_prefill.sh    # prefill node
# prefill node, second shell, once both servers are ready
PREFILL_URL=http://<prefill node address>:30000 DECODE_URL=http://<decode node address>:30000 \
  bash /opt/sglang-neuron/examples/gpt_oss_120b/run_balancer.sh
```

## Check

```bash
# prefill node, third shell (docker exec -it sglang bash)
bash /opt/sglang-neuron/examples/gpt_oss_120b/run_client.sh    # "content": "391", then a get_weather tool call
bash /opt/sglang-neuron/examples/gpt_oss_120b/run_gsm8k.sh     # Accuracy: about 0.89 on the first 100 questions
```

## The giga-kernel decode

The giga-kernel decode runs a whole decode step as one program, on 128 ranks, one on each of the decode node's 128
physical NeuronCores. For it, start `run_decode_giga.sh` instead of `run_decode.sh`, and the prefill with six
replicas. Each decode server takes all of the decode node's NeuronCores, so stop the other one first:

```bash
HOST=<decode node address> bash /opt/sglang-neuron/examples/gpt_oss_120b/run_decode_giga.sh             # decode node
REPLICAS=6 HOST=<prefill node address> bash /opt/sglang-neuron/examples/gpt_oss_120b/run_prefill.sh     # prefill node
```

The balancer and the checks are the same; the accuracy is about 0.89 too.

## Notes

- Defaults: `MODEL_PATH=/models/gpt-oss-120b`, `PORT=30000`, `REPLICAS=5`.
- A request's prompt and answer together take up to 10,240 tokens.
- The servers run SGLang's GPT-OSS chat parsers (`--reasoning-parser gpt-oss --tool-call-parser gpt-oss`), so a chat
  answer has its reasoning in `reasoning_content` and its tool calls in `tool_calls`.
- On a node without internet access, download the tokenizer's vocabulary elsewhere into the models directory, and
  start both servers with `TIKTOKEN_ENCODINGS_BASE=/models/tiktoken HF_HUB_OFFLINE=1` before the command:

  ```bash
  curl -L -o <models dir>/tiktoken/o200k_base.tiktoken --create-dirs \
    https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
  ```
- [The GPT-OSS recipe](../../docs/model-recipes/gpt-oss.md) explains the layouts and their flags.
