# Examples

Scripts that serve each model in the SGLang-Neuron container image with its validated flags, one folder per model.
A model's README gives its layout, the launch commands and the accuracy check; this page has what they all share.

| Model | Folder | Prefill/decode, two nodes | One node |
|---|---|---|---|
| GLM-5.2 (MXFP4) | [glm52](glm52/) | one prefill server (4- or 16-stage pipeline) and one decode server, up to 1M tokens | - |
| GPT-OSS-120B | [gpt_oss_120b](gpt_oss_120b/) | five prefill replicas and one decode server, or six and the giga-kernel decode | - |
| Qwen3-8B | [qwen3_8b](qwen3_8b/) | one prefill and one decode server | TP4 |
| Qwen3-0.6B | [qwen3_0p6b](qwen3_0p6b/) | one prefill and five decode servers | TP4 |
| GLM-4-9B | [glm4_9b](glm4_9b/) | one prefill and two decode servers | TP4 |
| DeepSeek-V2-Lite | [deepseek_v2_lite](deepseek_v2_lite/) | one prefill and five decode servers | TP4 |
| Qwen3.5-397B-A17B | [qwen35](qwen35/) | one prefill and one decode server, up to 256K tokens | - |
| Kimi-K3 | [kimi_k3](kimi_k3/) | one prefill server and a decode server over two nodes, up to 1M tokens (three nodes) | - |
| Wan2.2-T2V-A14B (text to video) | [wan22](wan22/) | - | one trn2.48xlarge, SGLang Diffusion |

The other diffusion pipelines (FLUX.1-dev, FLUX.3-Action, MiniMax-H3, LingBot-World-V2) run in the image as their
recipes' "In the container image" sections give:
[FLUX.1-dev](../docs/model-recipes/flux1-dev.md#in-the-container-image),
[FLUX.3-Action](../docs/model-recipes/flux3-action.md#in-the-container-image),
[MiniMax-H3](../docs/model-recipes/minimax-h3.md#in-the-container-image),
[LingBot-World-V2](../docs/model-recipes/lingbot-world-v2.md#in-the-container-image).


## The scripts

| Script | What it runs |
|---|---|
| `run_decode.sh` | The decode server: it generates each answer, from the KV cache the prefill server sends it over EFA. |
| `run_prefill.sh` | The prefill server: it computes each prompt's KV cache and sends it to the decode server. |
| `run_balancer.sh` | The balancer, the address clients use for a prefill/decode pair (`127.0.0.1:8000`). It sends each request to a prefill server and a decode server and returns the decode server's answer. It is the plugin's `python3 -m sglang_neuron.disaggregation.balancer` and needs no Neuron device. |
| `run_client.sh` | One request whose answer is known, to check that the model answers correctly. |
| `run_gsm8k.sh` | The accuracy check: the first 100 GSM8K questions through the balancer, with SGLang's few-shot client (`python3 -m sglang.test.few_shot_gsm8k`). Kimi-K3's goes through its chat front with SGLang's chat evaluation client and prints `Score`. |
| `run_server.sh` | A one-node server that runs both prefill and decode, for the models that have one. |
| `run_generate.sh` | Wan2.2's one-shot video generation (`sglang generate`). |

Every script is a short bash file; each server script sets the model's flags and runs SGLang's own launcher,
`python3 -m sglang.launch_server`. Read a script to see the exact command.

## Start the image

The commands are bash. On each node of a prefill/decode pair, start a container with the Neuron devices, the EFA
devices and the node's network, and open a shell in it:

```bash
IMAGE=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:<tag>
DEVICES=$(for d in /dev/neuron* /dev/infiniband/uverbs*; do printf -- '--device %s ' "$d"; done)
docker run -it --rm --name sglang --init --ipc=host --network host \
  --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 $DEVICES \
  -v <models dir>:/models:ro -v <cache dir>:/cache $IMAGE bash
```

For a one-node server, leave out the EFA devices, `--network host`, `IPC_LOCK` and `memlock`:

```bash
DEVICES=$(for d in /dev/neuron*; do printf -- '--device %s ' "$d"; done)
docker run -it --rm --name sglang --init --ipc=host --cap-add SYS_NICE $DEVICES \
  -v <models dir>:/models:ro -v <cache dir>:/cache $IMAGE bash
```

Open more shells in the same container with `docker exec -it sglang bash`. The scripts are in the image at
`/opt/sglang-neuron/examples/<folder>/`. The [container guide](../docs/guides/container.md) explains each option.

- `<models dir>` holds each model's checkpoint in a directory of its own, such as `<models dir>/Qwen3-8B`. The image
  has the Hugging Face CLI, for example
  `docker run --rm -v <models dir>:/models $IMAGE hf download Qwen/Qwen3-8B --local-dir /models/Qwen3-8B`.
- `<cache dir>` keeps the compiled programs: the first start of a model compiles them, and later starts load them.
  Use the same path every time, and one directory per image version and node role.
- All nodes of a deployment (Kimi-K3 has three) need one security group that allows all traffic between them, as
  EFA requires: besides the servers' port (30000) and the prefill server's KV bootstrap port (8998), each rank's KV
  transfer listens on a port the system picks. These ports accept any peer that reaches them, so give clients only
  the balancer.

## Run a model

1. Start the decode and the prefill servers, in either order, each on its own node. A server is ready when its log
   says `The server is fired up and ready to roll!`.
2. Start the balancer in a second shell on the prefill node (`docker exec -it sglang bash`), once every server is
   ready.
3. Run `run_client.sh`, then `run_gsm8k.sh`, in a third shell on the prefill node.

Each model's README gives these commands in order. For a one-node server, start `run_server.sh`, then run the client
and the accuracy check against it in a second shell.

## Settings

The scripts read these variables, with the defaults the model's README lists:

| Variable | Script | Meaning |
|---|---|---|
| `HOST` | `run_prefill.sh`, `run_decode.sh` | Required: an address of this node that the other node and the balancer reach. |
| `HOST` | `run_server.sh` | The address the server listens on, `127.0.0.1` by default. |
| `PORT` | servers | The server's port. |
| `MODEL_PATH` | servers, `run_client.sh` | The checkpoint, `/models/<checkpoint>`. |
| `CACHE_DIR` | servers | The compile cache, `/cache`. |
| `PREFILL_URL`, `DECODE_URL` | `run_balancer.sh` | Required: the two servers, `http://<node address>:<port>`. |
| `URL` | `run_client.sh` | The balancer, `http://127.0.0.1:8000`, or a one-node server. |
| `HOST`, `PORT` | `run_gsm8k.sh` | The balancer, `127.0.0.1` and `8000`, or a one-node server. |
| `NUM_QUESTIONS`, `GSM8K_DATA` | `run_gsm8k.sh` | How many questions (100), and a local copy of GSM8K's `test.jsonl`. |

Arguments after a script's name go to its command: SGLang's server flags for a server script, and the balancer's own
options (`--host`, `--port`, another `--decode`) for `run_balancer.sh`.

## Check accuracy

`run_gsm8k.sh` prints `Accuracy`, `Invalid` (the share of answers without a number) and `Latency`. Each model's
README gives the accuracy to expect, where one is measured. These are scores of SGLang's 5-shot client on 100
questions, not the model card's. The client downloads the questions from
[openai/grade-school-math](https://github.com/openai/grade-school-math).
On a host without internet access, download the file elsewhere and point `GSM8K_DATA` at it:

```bash
curl -L -o <models dir>/gsm8k/test.jsonl --create-dirs \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
GSM8K_DATA=/models/gsm8k/test.jsonl bash /opt/sglang-neuron/examples/<folder>/run_gsm8k.sh
```

## Stop

Stop the balancer first, then the servers: Ctrl+C in the shell of each one that runs in the foreground, and
`kill -INT $(jobs -p)` in a shell that started servers in the background (`&`). A server drains and stops every rank;
wait until its shell prompt is back before you exit the container. `docker stop` does not drain them: the servers run
under the container's shell, which passes the signal on to none of them.

## When it does not work

| What you see | Why | What to do |
|---|---|---|
| a server exits at start with `the compile cache ... was built from other NKI kernel sources` | the cache directory belongs to another image version | give this image its own cache directory |
| `Failed to load the backend extension: torch_neuronx` | a volume is mounted at `/workspace`, or Python was started there | mount data elsewhere and keep the working directory off `/workspace` |
| Neuron runtime errors on a host that just started | the host's Neuron driver is not ready yet | wait until `ls /dev/neuron*` lists the devices on the host |
| EFA or libfabric errors at start, or requests that hang in the KV transfer | the EFA devices, `IPC_LOCK` or `memlock` are missing | start the container as above |
| the decode server cannot reach the prefill server, so requests fail or time out | the wrong `HOST`, or traffic blocked between the nodes | give each server an address the other node reaches, and allow all traffic between the two nodes |
| a server refuses its peer at the first request | the two nodes run different image builds | run the same image on both |
