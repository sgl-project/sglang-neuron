# Inkling-2-Mini

One trn2.3xlarge, aggregated (prefill and decode in one server), TP8 on its eight physical NeuronCores, with the radix
cache. The recipe is [docs/model-recipes/inkling2.md](../../docs/model-recipes/inkling2.md).

## Launch

```bash
# once: the checkpoint's text view, which the server loads
SRC=/models/Inkling-2-Mini DST=/models/Inkling-2-Mini-text bash /opt/sglang-neuron/examples/inkling2_mini/prepare_checkpoint.sh
bash /opt/sglang-neuron/examples/inkling2_mini/run_server.sh
```

## Check

```bash
# second shell (docker exec -it sglang bash), once the server is ready
bash /opt/sglang-neuron/examples/inkling2_mini/run_client.sh    # "content": "391"
bash /opt/sglang-neuron/examples/inkling2_mini/run_gsm8k.sh     # Accuracy on the first 100 questions
```
