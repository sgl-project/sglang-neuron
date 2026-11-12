# Kimi-K3

Prefill/decode on three Trn3 nodes at a context of up to 1,048,576 tokens: a prefill server of eight pipeline stages of
TP16 on one node, and one decode server over two nodes (TP256 as eight DP-attention groups of TP32, with DSPARK
speculative decoding). The prefill takes token ids only, so clients reach the pair through a chat front, which applies
the checkpoint's chat template. Start the image on each node and run the scripts in the order
[the examples page](../README.md) gives.

| role | node | script |
|---|---|---|
| decode | decode node 0 and decode node 1 | `run_decode.sh` |
| prefill | prefill node | `run_prefill.sh` |
| balancer | prefill node | `run_balancer.sh` |
| chat front | prefill node | `run_chat_front.sh` (runs `chat_front.py`) |

## Launch

```bash
# decode node 0
HOST=<decode node 0 address> NODE_RANK=0 DIST_INIT_ADDR=<decode node 0 address>:29500 \
  DRAFT_PATH=/models/Kimi-K3-DSpark bash /opt/sglang-neuron/examples/kimi_k3/run_decode.sh
# decode node 1
HOST=<decode node 1 address> NODE_RANK=1 DIST_INIT_ADDR=<decode node 0 address>:29500 \
  DRAFT_PATH=/models/Kimi-K3-DSpark bash /opt/sglang-neuron/examples/kimi_k3/run_decode.sh
# prefill node
HOST=<prefill node address> DRAFT_PATH=/models/Kimi-K3-DSpark bash /opt/sglang-neuron/examples/kimi_k3/run_prefill.sh
# prefill node, second shell, once both servers are up
PREFILL_URL=http://<prefill node address>:30000 DECODE_URL=http://<decode node 0 address>:30000 \
  bash /opt/sglang-neuron/examples/kimi_k3/run_balancer.sh
# prefill node, third shell, once the balancer is up
bash /opt/sglang-neuron/examples/kimi_k3/run_chat_front.sh
```

## Check

```bash
bash /opt/sglang-neuron/examples/kimi_k3/run_client.sh    # the answer: 391, the reasoning in reasoning_content
bash /opt/sglang-neuron/examples/kimi_k3/run_gsm8k.sh     # prints the score
```

Run both in a fourth shell on the prefill node, the chat front's host. `run_client.sh` sends one chat request;
`run_gsm8k.sh` runs SGLang's chat evaluation client (`python3 -m sglang.test.run_eval --eval-name gsm8k --api chat`)
through the chat front, at the model's recommended sampling (temperature 1.0, top-p 0.95). It takes the first five
questions of the test split as its examples and asks the next `NUM_QUESTIONS` (100). It prints `Score` (the accuracy);
the lines about a deprecation and about `spec_accept_length` that the client also prints do not matter here.

## Notes

- **Checkpoints:** `moonshotai/Kimi-K3` and its DSPARK drafter, `Inferact/Kimi-K3-DSpark`, both on Hugging Face, under
  `/models` on every node: `hf download moonshotai/Kimi-K3 --local-dir /models/Kimi-K3` and `hf download
  Inferact/Kimi-K3-DSpark --local-dir /models/Kimi-K3-DSpark`. Both servers take the drafter (`DRAFT_PATH`): the prefill
  writes its context rows, and the decode drafts with it.
- **The two decode nodes:** the same command on both, each with its own `HOST`, `NODE_RANK` 0 and 1, and the same
  `DIST_INIT_ADDR`; node 0 serves HTTP. Put them in one UltraServer, on two nodes whose UltraServer indices are 1 and 2,
  or 0 and 3: the decode's cross-node collectives run slower between the other pairs. `neuron-ls` in the container
  prints each node's `ultraserver-id` and `ultraserver-instance-index`. To check a pair before the decode starts, run a
  2-node all-reduce over the decode's 256 ranks from decode node 0, which starts node 1's half with `mpirun` over SSH
  (so node 0 needs SSH access to node 1), and compare its `Avg bus bandwidth` between pairs:

  ```bash
  NEURON_LOGICAL_NC_CONFIG=1 NEURON_RT_ROOT_COMM_ID=<decode node 0 address>:44444 \
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    nccom-test -r 256 -N 2 -s <decode node 0 address> <decode node 1 address> \
    -b 64M -e 64M -n 20 -w 5 -d bf16 all_reduce
  ```
- **Network:** all three nodes in one security group that allows all traffic between them. Besides the common page's
  ports, the decode's two nodes connect on ports derived from `DIST_INIT_ADDR`, and the prefill sends the KV to ranks on
  both decode nodes.
- **Draft length:** `K=3` (the default) or `K=7` on `run_decode.sh`; the decode programs are built for those two.
- **Host memory:** the prefill's prefix cache keeps a host tier of about 1,080 GB at the validated values
  (`SGLANG_NEURON_KIMI_PROMPT_HOST_SLOTS=18000`, `SGLANG_NEURON_KIMI_PROMPT_HOST_STATE_SLOTS=2000`). On a node with less
  host memory, set both lower before `run_prefill.sh`.
- **Requests:** send chat requests to the chat front, not to the balancer: a text request there fails, since the
  prefill takes token ids only.
- **Sampling:** the chat front passes a request's `temperature`, `top_p` and `top_k` through as given, and a request
  without them samples at temperature 1.0 and top-p 1.0. Set them in every request: the model's sampling is
  temperature 1.0 (0.6 with thinking off) and top-p 0.95, as `run_client.sh` and `run_gsm8k.sh` send.
- **Long cold prompts:** a request's KV can reach the decode late behind long cold prompts at the prefill, so the
  scripts raise SGLang's two disaggregation timeouts from their default of 300 to the same value on both servers:
  `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600` on the prefill and `SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600` on
  the decode. Set both before the scripts to change them.
- Defaults: `MODEL_PATH=/models/Kimi-K3`, `PORT=30000` for the servers, `K=3`. `run_chat_front.sh` listens on
  127.0.0.1, `PORT=8021`, and forwards to the balancer at `UPSTREAM=http://127.0.0.1:8000`; `LOG=<file>` writes each
  request and answer to that file as JSON lines. `run_client.sh` (`URL`) and `run_gsm8k.sh` (`HOST`, `PORT`) default
  to the chat front.

[The Kimi-K3 recipe](../../docs/model-recipes/kimi-k3.md) explains the layout and every flag.
