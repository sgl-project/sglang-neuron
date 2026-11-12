# Container runbook

From a checkout of this repository to a serving model and an accuracy check, with the SGLang-Neuron images. The image
reference is [docker/README.md](../../docker/README.md); the model recipes hold every flag's reasoning.

```
1. build the images            docker/build.sh                  (any host with Docker)
2. put them on the Neuron hosts docker save | docker load        (or your own registry)
3. one node, a small model     Qwen3-8B on 4 NeuronCores         (does the image serve here?)
4. prefill/decode              GPT-OSS-120B or GLM-5.2, two nodes, the balancer in front
5. check accuracy              the example's run_gsm8k.sh against the served model
```

## 1. Build the images

On a Linux x86_64 host with Docker Engine 23 or later, Python 3.11 or later, about 60 GB free under Docker's root and
network access ([docker/README.md](../../docker/README.md) lists every prerequisite):

```bash
git clone <this repository> sglang-neuron && cd sglang-neuron
git checkout <commit>                 # main, or a release commit
docker/build.sh
```

It builds `sglang-neuron:<tag>` (the image that serves) and `sglang-neuron:<tag>-qa` (the same plus the repository's
checkout), checks both and prints both names. The tag is `v<version>-<commit>-dev`.

## 2. Put the images on the Neuron hosts

Every node of a deployment runs the same build: a prefill and a decode server exchange the KV cache with the image's
patched NIXL. Build on each host, or copy the image:

```bash
docker save sglang-neuron:<tag> | ssh <neuron host> docker load
```

or push it to a registry of your own (`docker tag`, `docker push`). Each host also needs the Neuron driver Neuron SDK
2.33.0 lists (`aws-neuronx-dkms` 2.31.7.0 or later); on a host that just booted, wait until `ls /dev/neuron*` lists
the devices.

## The pieces every server's `docker run` takes

Set these in bash on each node (the commands below rely on bash splitting `$DEVICES` and `$ENVS`; zsh does not):

```bash
IMAGE=sglang-neuron:<tag>
MODELS=/path/to/models                    # the checkpoints, mounted at /models
CACHE=/path/to/cache/sglang-neuron-<tag>  # persistent and local: one directory per image version and server role
DEVICES=$(for d in /dev/neuron* /dev/infiniband/uverbs*; do [ -e "$d" ] && printf -- '--device %s ' "$d"; done)
ENVS="-e TORCH_NEURONX_NEFF_CACHE_DIR=/cache/neff -e TORCH_NEURONX_HLO_CACHE_DIR=/cache/hlo -e NKI_COMPILE_CACHE_URL=/cache/nki"
```

- `--init --ipc=host` and `--cap-add SYS_NICE` on every server (the worker processes, shared memory, NUMA);
  `--cap-add IPC_LOCK --ulimit memlock=-1 --network host` and the EFA devices for prefill/decode.
- With `--network host` the container uses the node's network, so `--host` is the address a server listens on: give
  each server the node's own address that the other node and the balancer reach. `0.0.0.0` would listen on every
  interface of the node.
- The cache mount: the first start of a model compiles its programs, and a later start at the same path loads them;
  a cache moved to another path compiles again. A cache built by an image with other kernel sources is refused: give
  each image version its own directory, and give it tens of GB.
- Mount nothing at `/workspace` and start nothing there: the base image's torch-neuronx is installed from it.
- On a node without internet access: `-e HF_HUB_OFFLINE=1`, and the checkpoints and tokenizer files local.
- A server answers `GET /health` with 200 once it has warmed up; its log then says `The server is fired up and ready
  to roll!`. The waits below poll until then (a cold start compiles every program first): if a container exits instead,
  `docker ps -a` and `docker logs <name>` say why. Stop a multi-rank server with `docker stop -t 120 <name>`: it
  drains and stops every rank on SIGTERM, which can take a prefill server longer than Docker's default 10 s, so that
  default would kill it first. `docker rm <name>` removes a stopped container; `--rm` on `docker run` would remove it
  on exit, and its logs with it.

## 3. One node: Qwen3-8B

The quickest check that the image serves on a host:

```bash
docker run -d --name qwen3 --init --ipc=host --cap-add SYS_NICE -p 127.0.0.1:30000:30000 $DEVICES \
  -v $MODELS:/models:ro -v $CACHE:/cache $ENVS \
  $IMAGE python3 -m sglang.launch_server --model-path /models/Qwen3-8B --tp-size 4 \
  --context-length 4096 --chunked-prefill-size 2048 --max-running-requests 64 --max-total-tokens 266240 \
  --mem-fraction-static 0.6 --host 0.0.0.0 --port 30000
until curl -sf http://127.0.0.1:30000/health; do
  [ "$(docker inspect -f '{{.State.Running}}' qwen3)" = true ] || { echo "exited: docker logs qwen3"; break; }
  sleep 10; done
curl -s http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "/models/Qwen3-8B", "temperature": 0, "max_tokens": 64,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}'
docker stop -t 120 qwen3 && docker rm qwen3
```

`--host 0.0.0.0` listens inside the container; `-p 127.0.0.1:30000:30000` publishes it to this host only. `docker logs
-f qwen3` shows the start (on a new cache, mostly compiling). The answer is `17 * 23 = 391`; the request turns Qwen3's
thinking off (`enable_thinking`), which would otherwise spend the 64 tokens before the answer.

## 4a. Prefill/decode: GPT-OSS-120B

One prefill server with six replicas runs on one node. One giga-kernel decode server, 128 ranks, one per physical
NeuronCore, runs on the other. A balancer in front sends every request to both. The KV cache
moves from prefill to decode over EFA with the image's patched NIXL.

```
client -> balancer (127.0.0.1:8000) -> prefill server (prefill node :30000, bootstrap :8998)
                                    -> decode server  (decode node  :30000)
          prefill --KV over EFA (NIXL)--> decode
```

### Before you start

On both nodes:
- **Image:** the same image, built with `docker/build.sh` (or `docker load`ed from the same build). The two servers must
  run the same build: the KV transfer uses the image's patched NIXL.
- **Hardware:** 16 Neuron devices (`/dev/neuron0` ... `/dev/neuron15`) and the EFA devices (`/dev/infiniband/uverbs*`).
- **Network:** both nodes in one security group that allows all traffic between its members, as EFA requires. Besides
  the servers' port (30000) and the prefill's KV bootstrap port (8998), every rank's KV transfer listens on a TCP port
  the system picks, so a firewall that opens only the listed ports breaks the transfer. The balancer's host reaches both
  servers on 30000.
- **Model:** the `gpt-oss-120b` checkpoint in a directory mounted at `/models`.
- **Tokenizer file, offline only:** on a node without internet access, also put `o200k_base.tiktoken` (GPT-OSS's
  tokenizer vocabulary) in a directory under it, here `/models/tiktoken`.
- **Compile cache:** a persistent directory for the compile cache, mounted at `/cache`. Use the same path on every
  start, and one directory per image version and node role.

On each node, set [the common pieces](#the-pieces-every-servers-docker-run-takes) (`$MODELS` holds
`gpt-oss-120b/`, and `tiktoken/` if offline) and the two addresses:

```bash
PREFILL=<prefill node address>            # an address the other node and the balancer reach
DECODE=<decode node address>
# on a node without internet access, also:
ENVS="$ENVS -e TIKTOKEN_ENCODINGS_BASE=/models/tiktoken -e HF_HUB_OFFLINE=1"
```

### 1. Start the decode server (decode node)

```bash
docker run -d --name gptoss-decode --init --ipc=host --network host \
  --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 $DEVICES \
  -v $MODELS:/models:ro -v $CACHE:/cache $ENVS \
  $IMAGE python3 -m sglang.launch_server --model-path /models/gpt-oss-120b \
  --kv-cache-dtype fp8_e4m3 --context-length 10240 --chunked-prefill-size 8192 \
  --disaggregation-mode decode --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 \
  --tp-size 128 --dp-size 16 --enable-dp-attention --ep-size 128 --enable-dp-attention-local-control-broadcast \
  --max-running-requests 128 --max-total-tokens 163840 --mem-fraction-static 0.5 --stream-interval 4 \
  --tokenizer-worker-num 4 --detokenizer-worker-num 4 \
  --reasoning-parser gpt-oss --tool-call-parser gpt-oss \
  --host $DECODE --port 30000
```

### 2. Start the prefill server (prefill node)

```bash
docker run -d --name gptoss-prefill --init --ipc=host --network host \
  --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 $DEVICES \
  -v $MODELS:/models:ro -v $CACHE:/cache $ENVS \
  $IMAGE python3 -m sglang.launch_server --model-path /models/gpt-oss-120b \
  --kv-cache-dtype fp8_e4m3 --context-length 10240 --chunked-prefill-size 8192 \
  --disaggregation-mode prefill --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 \
  --tp-size 4 --ep-size 2 --dp-size 6 --max-running-requests 8 --max-total-tokens 65536 --mem-fraction-static 0.51 \
  --reasoning-parser gpt-oss --tool-call-parser gpt-oss \
  --host $PREFILL --port 30000
```

`--reasoning-parser gpt-oss --tool-call-parser gpt-oss` turn the model's harmony output into OpenAI chat fields.
Without them, `content` carries the raw channel markup. They cost no decode time. `--host` is each node's own address
([the common pieces](#the-pieces-every-servers-docker-run-takes)).

Wait until both servers answer:

```bash
until curl -sf http://$DECODE:30000/health; do   # on the decode node
  [ "$(docker inspect -f '{{.State.Running}}' gptoss-decode)" = true ] || { echo "exited: docker logs gptoss-decode"; break; }
  sleep 10; done
until curl -sf http://$PREFILL:30000/health; do  # on the prefill node
  [ "$(docker inspect -f '{{.State.Running}}' gptoss-prefill)" = true ] || { echo "exited: docker logs gptoss-prefill"; break; }
  sleep 10; done
```

While they start:
- On a new cache a server compiles its programs first; on a warm one it loads them.
- `docker logs -f gptoss-prefill` shows progress. The server is up at the line `The server is fired up and ready to
  roll!`.

### 3. Start the balancer (the prefill node, or any host that reaches both servers)

The balancer needs no Neuron device:

```bash
docker run -d --name gptoss-balancer --init --network host $IMAGE \
  python3 -m sglang_neuron.disaggregation.balancer --prefill http://$PREFILL:30000 --decode http://$DECODE:30000
until curl -sf http://127.0.0.1:8000/health; do sleep 2; done; echo balancer ready
```

It listens on `127.0.0.1:8000`. For clients on other hosts, add `--host <address>` (and `--port`), and expose it only to
them.

### 4. Send a request

A chat request:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "/models/gpt-oss-120b", "temperature": 0, "max_tokens": 256,
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}'
```

Success looks like this:
- `choices[0].message.content` is the answer alone (`"391"`).
- `choices[0].message.reasoning_content` holds the model's reasoning.
- `finish_reason` is `"stop"`.
- `"stream": true` gives the same split, chunk by chunk.
- `curl -s http://$DECODE:30000/v1/models` shows the model id to use in `"model"`.

A tool call:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "/models/gpt-oss-120b", "temperature": 0, "max_tokens": 256,
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}'
```

Success: `finish_reason` is `"tool_calls"`, and `message.tool_calls[0].function` is
`{"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}`.

### 5. Stop

```bash
docker stop gptoss-balancer          # on its host
docker stop -t 120 gptoss-prefill    # prefill node
docker stop -t 120 gptoss-decode     # decode node
```

`-t 120` gives each server the time to stop every rank
([the common pieces](#the-pieces-every-servers-docker-run-takes)).

While a server stops, the decode's log shows a few `asyncio.exceptions.CancelledError` tracebacks (one per tokenizer
worker) and `Subprocess scheduler_0 ... crashed with exit code -9`: that is SGLang ending its own worker processes, and
the server is gone a few seconds later. On a first start, the decode's log also carries thousands of `WARNING` lines
from the kernel library while it compiles; they are informational.

## 4b. Prefill/decode: GLM-5.2

One pipelined prefill server (PP4 x TP32) runs on one node. One giga-kernel decode server, 128 ranks with MTP, runs on
the other. A balancer in front sends every request to both, and the KV cache moves from prefill to decode over EFA
with the image's patched NIXL. This section brings up the default pairing, **128k context on a PP4 prefill**, and checks
it with one chat request; [GLM-5.2 on Neuron](../model-recipes/glm52.md) explains every flag and value.

```
client -> balancer (127.0.0.1:8000) -> prefill server (prefill node :30000, bootstrap :8998), PP4 x TP32
                                    -> decode server  (decode node  :30000), TP128, MTP
          prefill --KV over EFA (NIXL)--> decode
```

### Before you start

The nodes, the image, the network and the compile cache are as for GPT-OSS ([4a](#before-you-start)), on Trn3
nodes. Also:
- **Model:** the `GLM-5.2-MXFP4` checkpoint in `$MODELS`, on both nodes.
- **Weight store:** the decode's weight store for the 128k point, on the decode node ([below](#the-weight-store)).

On each node, set [the common pieces](#the-pieces-every-servers-docker-run-takes) (`$MODELS` holds
`GLM-5.2-MXFP4/`), the two addresses and the store:

```bash
PREFILL=<prefill node address>            # an address the other node and the balancer reach
DECODE=<decode node address>
STORE=/path/to/glm52-store-128k           # the decode's weight store, mounted at /store
```

### The weight store

The decode loads its weights from a GK (giga-kernel) weight store: the checkpoint converted to the kernel's per-rank
layout. The store is built once per decode point, and with MTP it also holds the draft layer's files. The build runs on
CPU in the image, needs no Neuron device, can be resumed, and writes a `geometry.json` that the decode checks at load
(the point's sharding and the checkpoint it came from):

```bash
docker run --rm -v $MODELS:/models:ro -v $STORE:/store \
  -e SGLANG_NEURON_CPU_MODE=1 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 $IMAGE python3 -c '
from sglang_neuron.models.glm52.points import POINTS
from sglang_neuron.models.glm52.layout.decode.weight_loaders_mxfp4 import build_rank_store
from sglang_neuron.models.glm52.layout.decode.weight_loaders_mtp import build_mtp_rank_store, build_mtp_fp8_tensor_store
geo = POINTS["128k"].geometry()
ckpt = "/models/GLM-5.2-MXFP4"
build_rank_store(ckpt, "/store", geo, workers=48)
build_mtp_rank_store(ckpt, "/store", geo)
build_mtp_fp8_tensor_store(ckpt, "/store", geo)'
```

If you already have a store for this checkpoint and point, set `$STORE` to it. The decode refuses a store built for
another point, another latent split or another checkpoint, and says which field differs.

### 1. Start the decode server (decode node)

```bash
docker run -d --name glm52-decode --init --ipc=host --network host \
  --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 $DEVICES \
  -v $MODELS:/models:ro -v $STORE:/store:ro -v $CACHE:/cache \
  -e SGLANG_NEURON_GLM52_POINT=128k -e SGLANG_NEURON_GLM52_WEIGHT_STORE=/store \
  -e SGLANG_NEURON_SAMPLER=topk_topp $ENVS \
  $IMAGE python3 -m sglang.launch_server --model-path /models/GLM-5.2-MXFP4 --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl --tp-size 128 --ep-size 128 --page-size 1024 \
  --max-running-requests 32 --disaggregation-decode-extra-slots 0 --cuda-graph-bs-decode 32 \
  --context-length 130944 --max-total-tokens 4194304 --speculative-algorithm NEURON_MTP \
  --stream-interval 1 --trust-remote-code --reasoning-parser glm45 --host $DECODE --port 30000
```

### 2. Start the prefill server (prefill node)

```bash
docker run -d --name glm52-prefill --init --ipc=host --network host \
  --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 $DEVICES \
  -v $MODELS:/models:ro -v $CACHE:/cache \
  -e SGLANG_NEURON_GLM52_POINT=128k -e SGLANG_PP_LAYER_PARTITION=18,20,20,20 \
  -e SGLANG_NEURON_SAMPLER=topk_topp -e SGLANG_NEURON_GLM52_PD_MTP=1 $ENVS \
  $IMAGE python3 -m sglang.launch_server --model-path /models/GLM-5.2-MXFP4 --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl --pp-size 4 --tp-size 32 --ep-size 32 \
  --chunked-prefill-size 4096 --cuda-graph-bs-prefill 1024 2048 --context-length 130944 \
  --max-running-requests 16 --max-total-tokens 14679040 --reasoning-parser glm45 \
  --host $PREFILL --port 30000 --disaggregation-bootstrap-port 8998
```

Both servers must agree, or a server refuses its peer at first contact:
- the decode point (`SGLANG_NEURON_GLM52_POINT`);
- MTP: `--speculative-algorithm NEURON_MTP` on the decode goes with `SGLANG_NEURON_GLM52_PD_MTP=1` on the prefill;
- the sampler: `SGLANG_NEURON_SAMPLER=topk_topp` lets requests sample; without it, both serve greedy requests only.

`--reasoning-parser glm45` puts the model's thinking in `reasoning_content` and the answer alone in `content`.

A warm start of the prefill does not skip its warm-up: every rank traces its kernels again. A server opens its port
only after its warm-up: until then `curl` prints `000` (connection refused). Follow the progress with `docker logs -f
glm52-prefill`; a server logs `The server is fired up and ready to roll!` when it is ready.

### 3. Start the balancer (the prefill node, or any host that reaches both servers)

The balancer needs no Neuron device:

```bash
docker run -d --name glm52-balancer --init --network host $IMAGE \
  python3 -m sglang_neuron.disaggregation.balancer --prefill http://$PREFILL:30000 --decode http://$DECODE:30000
```

It listens on `127.0.0.1:8000`. For clients on other hosts, give `--host <address>` (and `--port`), and expose it
only to them.

### 4. Send a request

```bash
for url in http://$DECODE:30000 http://$PREFILL:30000 http://127.0.0.1:8000; do
  echo "$url $(curl -s -o /dev/null -w '%{http_code}' $url/health)"
done
MODEL=$(curl -s http://127.0.0.1:8000/v1/models | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])')
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\", \"temperature\": 0, \"max_tokens\": 1024,
  \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 times 23? Reply with the number only.\"}]}" \
  | python3 -m json.tool
```

Success looks like this:
- all three `/health` give `200`;
- `choices[0].message.content` is the answer alone (`391`);
- `reasoning_content` holds the model's thinking (a few hundred characters), and `usage` counts it as
  `reasoning_tokens`.

The thinking counts against `max_tokens`, so leave a request room for it: a request whose thinking uses up the budget
returns an empty `content`. The API names the model by its `--model-path` (`/models/GLM-5.2-MXFP4`); add
`--served-model-name <name>` to both servers to give clients a shorter one.

### 5. Stop

```bash
docker stop glm52-balancer           # on its host
docker stop -t 120 glm52-prefill     # prefill node
docker stop -t 120 glm52-decode      # decode node
```

`-t 120` gives each server the time to stop every rank
([the common pieces](#the-pieces-every-servers-docker-run-takes)).
The compile caches stay in `$CACHE`, so the next start is warm.

### 1M context on a PP16 prefill

The same commands, with these values:

| | 128k on PP4 | 1M on PP16 |
|---|---|---|
| `SGLANG_NEURON_GLM52_POINT` (both) | `128k` | `1m` |
| `--context-length` (both) | 130944 | 1048448 |
| decode `--max-total-tokens` / `--page-size` | 4194304 / 1024 | 33554432 / 256 |
| prefill `--pp-size` / `--tp-size` / `--ep-size` | 4 / 32 / 32 | 16 / 8 / 8 |
| prefill `SGLANG_PP_LAYER_PARTITION` | `18,20,20,20` | `3,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5` |
| prefill `--chunked-prefill-size` / `--cuda-graph-bs-prefill` | 4096 / `1024 2048` | 2048 / `1024` |
| prefill `--max-running-requests` / `--max-total-tokens` | 16 / 14679040 | 48 / 12582912 |
| the decode's weight store | the 128k store | the 1M latent-split store (below) |

The 1M decode serves a store built for the 1M point and a latent split of 8:
1. Build the 1M store as above with `POINTS["1m"].geometry()`.
2. Make the side store from it and the 128k store with `tools/glm52/build_latent_store.py`, which the image holds at
   `/opt/sglang-neuron/tools/glm52/`:
   `docker run --rm -v <stores>:/stores $IMAGE python3 /opt/sglang-neuron/tools/glm52/build_latent_store.py
   /stores/<1M store> /stores/<128k store> /stores/<1M latent-8 store> 8`.
   It links the 1M store's files as they are when it runs, so build the 1M store's MTP files first (step 1), and its
   links are absolute: mount the same stores directory at `/stores` on the decode.

The 1M decode's Neuron runtime settings are set by the launch itself; do not set `NEURON_RT_IO_RING_CACHE_SIZE`.

### When it does not come up

The table at [the end](#when-it-does-not-work) holds what every section shares. For GLM-5.2 also:
- **The decode refuses its weight store.** The error names the field (point, latent split or checkpoint digest):
  build or point `$STORE` at the store for this point and checkpoint.
- **The servers refuse each other.** The decode point, MTP or sampler settings differ between them (above).
- **An empty `content` with `--reasoning-parser glm45`:** the thinking used the whole `max_tokens`; raise it.

## 5. Check accuracy

Before the servers stop, each example's `run_gsm8k.sh` scores the first 100 GSM8K questions against them; the model's
README gives the accuracy to expect ([checking accuracy](../../examples/README.md#check-accuracy)). The one-node server
of section 3 listens on port 30000, a prefill/decode pair's balancer on 8000 (the script's default):

```bash
docker run --rm --network host -e PORT=30000 $IMAGE /opt/sglang-neuron/examples/qwen3_8b/run_gsm8k.sh
docker run --rm --network host $IMAGE /opt/sglang-neuron/examples/gpt_oss_120b/run_gsm8k.sh
```

## When it does not work

| what you see | why | what to do |
|---|---|---|
| a server exits at start with `the compile cache ... was built from other NKI kernel sources` | the cache directory belongs to another image version | give this image its own `$CACHE` directory |
| `ModuleNotFoundError: No module named 'torch_neuronx'` | a volume is mounted at `/workspace`, or Python was started there | mount data elsewhere and keep the working directory off `/workspace` |
| the decode logs NUMA warnings and the KV transfer falls back or fails | the container lacks `CAP_SYS_NICE` | add `--cap-add SYS_NICE` |
| EFA / libfabric errors at start, or requests hang in transfer | the EFA devices, `IPC_LOCK` or `memlock` are missing | pass every `/dev/infiniband/uverbs*`, `--cap-add IPC_LOCK --ulimit memlock=-1` |
| the decode cannot reach the prefill's bootstrap, so requests fail or time out | the wrong `--host`, or traffic blocked between the nodes | give each server its own reachable address; allow all traffic between the two nodes (the security group's rule for itself) |
| a server refuses its peer at the first request | the two nodes run different image builds, or (GLM-5.2) a different decode point, MTP or sampler setting | run the same image on both, with the same settings |
| chat `content` holds `<|channel|>analysis<|message|>...` | the decode server runs without the parsers | add `--reasoning-parser gpt-oss --tool-call-parser gpt-oss` |
| the server fails to load the tokenizer's vocabulary offline | no `o200k_base.tiktoken` | set `TIKTOKEN_ENCODINGS_BASE` to its directory |
| the balancer answers 400 "routes single-prompt requests only" | the request holds a list of prompts | send one prompt per request |
| a server logs `OpenAI Responses API (/v1/responses) disabled: ... ImportError` at start, and `/v1/responses` returns an error | SGLang's Responses API is off at this SGLang commit (its `openai` pin predates it); the rest of the API is unaffected | use `/v1/chat/completions`, `/v1/completions` or `/generate` |
| a server started on a host that just booted fails with `TDRV ... unable to retrieve device architecture` and `Failed to load the backend extension: torch_neuronx` | the host's Neuron driver was still loading | wait until `ls /dev/neuron*` lists every device, then start the server again |
| NIXL errors after Python packages were changed in a container | NIXL was reinstalled from PyPI, or `NIXL_PLUGIN_DIR` is set | use the image's NIXL as built: no `pip install nixl`, `NIXL_PLUGIN_DIR` unset |
