# The container image

The SGLang-Neuron image has everything a Trainium instance needs to serve a model with SGLang: SGLang, the
SGLang-Neuron plugin, the Neuron SDK and NIXL for prefill/decode disaggregation. This guide checks a host, starts a
container and serves a model in it.

```
public.ecr.aws/neuron/pytorch-inference-sglang-neuron:<tag>
```

## What is in the image

- SGLang, with its runtime dependencies and SGLang Diffusion's packages;
- the SGLang-Neuron plugin, installed in the image's Python;
- the Neuron SDK release the plugin is pinned to: the compiler, the runtime, torch and torch-neuronx;
- NIXL, built with the patches prefill/decode serving needs; its notice is in `/usr/share/doc/sglang-neuron/nixl/`;
- in `/opt/sglang-neuron/`, this repository at the image's commit, less its CI and build tooling:
  `README.md`, `LICENSE`, `pyproject.toml`, `docs/`, `examples/`, `sglang_neuron/` (the installed plugin's sources,
  plus the kernel records the package leaves out), the NIXL patches in `patches/nixl/` and
  `tools/glm52/build_latent_store.py` (the GLM-5.2 weight store tool). Python started in `/opt/sglang-neuron` imports
  this copy, which is the same code as the installed plugin. The docs' links to `docker/` resolve in the repository
  only.

`docker inspect -f '{{json .Config.Labels}}' $IMAGE` shows what an image was built from: the plugin revision, the
Neuron SDK release and the SGLang commit. Section 3 prints the versions of the installed packages.

## 1. Check the host

The host is a Trainium instance with Docker Engine and the Neuron driver of Neuron SDK 2.33.0 (`aws-neuronx-dkms`
2.31.7.0 or later). The language models in the [model recipes](../model-recipes/index.md) are validated on Trn3 (and
GPT-OSS runs on Trn3 only), the diffusion pipelines on trn2.48xlarge. The driver's devices are on the host:

```bash
ls /dev/neuron*
```

It lists one device per Neuron chip (16 on a trn2.48xlarge or a Trn3 instance). If it lists none, install or load the
driver first; a host that just booted can take a few minutes before the driver is ready. The image has its own
`neuron-ls`, which the next steps run in a container.

## 2. Get the image

```bash
IMAGE=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:<tag>
docker pull $IMAGE
```

## 3. Start a container

Set these once per shell. The commands below rely on bash splitting `$DEVICES` into separate arguments; zsh does
not.

```bash
MODELS=/path/to/models                 # the checkpoints, mounted at /models
CACHE=/path/to/cache                   # the compile cache: persistent and local, one directory per image version
DEVICES=$(for d in /dev/neuron*; do printf -- '--device %s ' "$d"; done)
```

Check that the container sees the devices and the software:

```bash
docker run --rm $DEVICES $IMAGE neuron-ls
docker run --rm $DEVICES $IMAGE python3 -c '
import importlib.metadata as m
for p in ("sglang", "sglang-neuron", "torch", "torch-neuronx", "neuronx-cc", "nixl"): print(p, m.version(p))'
```

`neuron-ls` lists the same devices as on the host, and the second command prints the installed versions.

A server's container takes these options:

```bash
docker run -d --name <name> --init --ipc=host --cap-add SYS_NICE $DEVICES \
  -v $MODELS:/models:ro -v $CACHE:/cache \
  -e TORCH_NEURONX_NEFF_CACHE_DIR=/cache/neff -e TORCH_NEURONX_HLO_CACHE_DIR=/cache/hlo \
  -e NKI_COMPILE_CACHE_URL=/cache/nki \
  -p 127.0.0.1:30000:30000 \
  $IMAGE <command>
```

- `--init` reaps the server's worker processes; `--ipc=host` gives SGLang's processes the shared memory they
  exchange tensors through (Docker's default `/dev/shm` is 64 MB); `--cap-add SYS_NICE` lets the NUMA library and
  NIXL place memory (without it NIXL cannot map its network rails to NUMA nodes).
- The cache mount keeps compiled programs across containers: the first start of a model compiles them, later starts
  at the same path load them. Mount it at the same path every time, and give each image version its own directory: a
  server refuses a cache built from other kernel sources and says so.
- `-p 127.0.0.1:30000:30000` publishes the server's port on the host's loopback only. The server listens with
  `--host 0.0.0.0` inside the container, and the publish is what limits who reaches it. For a client on another
  machine, forward the port over SSH (`ssh -L 30000:127.0.0.1:30000 <host>`) rather than publishing it wider.
- Mount nothing at `/workspace`, and start nothing from there: the Neuron SDK container installs torch-neuronx from
  `/workspace/torch_neuronx`.
- On a host without internet access, add `-e HF_HUB_OFFLINE=1` and keep the checkpoints and tokenizer files under
  `$MODELS`.

### Two nodes: prefill and decode

A prefill/decode pair moves the KV cache between the nodes with NIXL over EFA. Each of its servers also takes the EFA
devices, the memory lock and the host's network:

```bash
DEVICES=$(for d in /dev/neuron* /dev/infiniband/uverbs*; do printf -- '--device %s ' "$d"; done)
docker run -d --name <name> --init --ipc=host --network host \
  --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 $DEVICES \
  -v $MODELS:/models:ro -v $CACHE:/cache \
  -e TORCH_NEURONX_NEFF_CACHE_DIR=/cache/neff -e TORCH_NEURONX_HLO_CACHE_DIR=/cache/hlo \
  -e NKI_COMPILE_CACHE_URL=/cache/nki \
  $IMAGE <command>
```

- With `--network host` the container uses the node's network, so `-p` does not apply. Give each server
  `--host <the node's address>`, the address the other node and the balancer reach it on; `0.0.0.0` would listen on
  every interface of the node.
- Put both nodes in one security group that allows all traffic between its members, as EFA requires. Besides the
  servers' port (30000) and the prefill server's KV bootstrap port (8998), every rank's KV transfer listens on a TCP
  port the system picks, so a firewall that opens only the listed ports breaks the transfer. These ports accept any
  peer that reaches them: keep them inside the cluster's security group, and give clients only the balancer.
- Both nodes run the same image: the KV transfer uses the image's NIXL, whose handshake another NIXL build does not
  accept.
- The balancer sends each request to both halves. It needs no Neuron device and runs in a container of its own on
  any host that reaches both servers; it listens on `127.0.0.1:8000`:

  ```bash
  docker run -d --name balancer --init --network host $IMAGE \
    python3 -m sglang_neuron.disaggregation.balancer --prefill http://<prefill>:30000 --decode http://<decode>:30000
  ```

- Servers started with SGLang's `--api-key` refuse requests without the key; the balancer checks no key itself and
  passes each client's `Authorization` header on to both servers.

## 4. Serve a model

[`examples/`](../../examples/README.md) has a folder for each model with the scripts that serve it and the order to
start them in; in the image they are at `/opt/sglang-neuron/examples/`. Qwen3-8B on one node, with its checkpoint from
Hugging Face (the image has the `hf` command):

```bash
docker run --rm -v $MODELS:/models $IMAGE hf download Qwen/Qwen3-8B --local-dir /models/Qwen3-8B

docker run -d --name qwen3-8b --init --ipc=host --cap-add SYS_NICE $DEVICES \
  -v $MODELS:/models:ro -v $CACHE:/cache -p 127.0.0.1:30000:30000 -e HOST=0.0.0.0 \
  $IMAGE /opt/sglang-neuron/examples/qwen3_8b/run_server.sh

until curl -sf http://127.0.0.1:30000/health; do
  [ "$(docker inspect -f '{{.State.Running}}' qwen3-8b)" = true ] || { echo "exited: docker logs qwen3-8b"; break; }
  sleep 10; done

curl -s http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "/models/Qwen3-8B", "temperature": 0, "max_tokens": 64,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}'
```

The answer is `17 * 23 = 391`. The first start on a new cache compiles the model's programs; `docker logs -f
qwen3-8b` shows the progress, and the server is ready at `The server is fired up and ready to roll!`. `HOST=0.0.0.0`
lets the published port reach the server inside the container; the publish keeps it on the host's loopback.

Any other model starts the same way: its example's scripts, or SGLang's own launcher,
`python3 -m sglang.launch_server ...`, with the flags its [recipe](../model-recipes/index.md) gives. The image sets
`SGLANG_PLATFORM=neuron`, so SGLang loads the Neuron platform without further settings.

## 5. Check accuracy

Each example folder has `run_gsm8k.sh`, which scores the first 100 GSM8K questions against the running server; the
model's README gives the accuracy to expect ([checking accuracy](../../examples/README.md#check-accuracy)). The client
needs no Neuron device. For the server above:

```bash
docker run --rm --network host -e PORT=30000 $IMAGE /opt/sglang-neuron/examples/qwen3_8b/run_gsm8k.sh
```

## 6. Stop

```bash
docker stop -t 120 qwen3-8b && docker rm qwen3-8b
```

A server drains and stops every rank on `docker stop`; Docker's default of 10 s can be too short for that.

## Known limitations

- **Do not `pip install` or upgrade `nixl` or `nixl-cu13`.** A PyPI wheel replaces the image's patched NIXL, and the
  prefill/decode servers then fail to exchange the KV cache.
- **`pip check` reports requirements the image leaves out on purpose:** SGLang's own GPU stack (the `sglang ...`
  lines) and `nixl-cu12` (the image has the NIXL build that matches its torch).
- **SGLang's `/v1/responses` endpoint is off.** The `openai` package SGLang pins for this platform predates its
  Responses API code, so the server disables the endpoint at startup. `/v1/chat/completions`, `/v1/completions` and
  `/generate` serve as usual.
- **The GSM8K client prints a deprecation notice** that points to SGLang's newer `sgl-eval` tool, which the image
  does not install; the client works as before.

Building the image yourself is described in `docker/README.md` in the repository.
