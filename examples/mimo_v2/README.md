# MiMo-V2.6-Flash-RL

One trn2.48xlarge, aggregated, TP64 / EP64, the text decoder. [The recipe](../../docs/model-recipes/mimo-v2.md) explains
the layout, the flags and what is not validated yet.

| script | does |
|---|---|
| `run_server.sh` | the server; `CONTEXT` 8192 for bring-up, 262144 / 1048576 for the AgentX points (`MTP=1` for NEURON_MTP) |
| `run_client.sh` | one greedy chat request |
| `run_needle.py` | needles in long haystacks (the full-attention layers' long-context check) |
| `run_agentx.sh` | one InferenceX AgentX point, from a separate CPU host |

```bash
CONTEXT=8192 CHUNK=2048 MAX_RUNNING=4 bash /opt/sglang-neuron/examples/mimo_v2/run_server.sh
# second shell, once the server is ready
bash /opt/sglang-neuron/examples/mimo_v2/run_client.sh
python3 /opt/sglang-neuron/examples/mimo_v2/run_needle.py --tokens 4000 7000
```

Before hardware, `tools/mimo_v2/check_layout.py` checks the loaders and the attention layout on CPU (torch and
safetensors only) against tensors `tools/mimo_v2/fetch_tensors.py` reads from the published checkpoint.
