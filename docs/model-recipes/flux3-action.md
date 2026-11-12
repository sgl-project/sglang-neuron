# FLUX.3-Action on Neuron

This plugin (`sglang_neuron`) runs Black Forest Labs' FLUX.3-Action DROID robot policy
(`black-forest-labs/flux-3-action-droid`) through SGLang Diffusion's own `Flux3ActionPipeline` on a trn2.48xlarge, with
BF16 weights, on torch-neuronx. From three camera frames, the robot state and a language instruction, the policy
denoises a chunk of future video latents together with a chunk of 32 actions, and returns the actions. SGLang's
[FLUX 3 Action cookbook page](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/vla/FLUX/FLUX-3-Action.mdx)
describes the model, its request schema and its GPU deployment. This page covers what is particular to Neuron.

Three layouts are validated, each on 4 or 8 of the instance's 64 logical NeuronCores (LNC=2):

| `--model-variant` | Recipe | Layout | Flags |
|---|---|---|---|
| `sd` | step-distilled, 1 step | TP4 x CP2 | `--num-gpus 8 --tp-size 4 --ulysses-degree 2` |
| `sd` | step-distilled, 1 step | TP4 | `--num-gpus 4 --tp-size 4` |
| `gd` | guidance-distilled, 4 steps, no CFG | TP4 x CP2 | `--num-gpus 8 --tp-size 4 --ulysses-degree 2` |
| `gd` | guidance-distilled, 4 steps, no CFG | TP4 | `--num-gpus 4 --tp-size 4` |
| `base` | 4 steps, guidance 4.0 (video) and 1.0 (action) | TP4 x CFG2 | `--num-gpus 8 --tp-size 4 --enable-cfg-parallel` |

Other layouts, the FP8r packages and trn3 are not validated in this repository.

## The model

SGLang's preprocess, text-encoding, observation and denoising stages run as Neuron subclasses
(`NeuronFlux3ActionPipeline.create_pipeline_stages`), and SGLang's postprocess stage is used as it is. No tensor op
runs eagerly on the device. Each piece below is a compiled graph:

- **DiT:** SGLang's `Flux3Transformer` with a Neuron block on NKI kernels. SGLang loads and TP-shards the checkpoint on
  the host. Each stream's 5 mode blocks run as one compiled graph, and the 28 joint blocks as four compiled chains of 7.
  Each rank holds 1/tp of its sequence-parallel shard: the rows are all-gathered over TP before the projections and
  reduce-scattered after them. With CP, the video stream's attention
  uses the const-max ring kernel from `kernels/flux`. The output heads run inside the last joint graph.
- **Text encoder:** Qwen3-VL-4B's language model, one compiled graph per pad bucket of 80 tokens.
- **Video VAE:** SGLang's `Flux3VideoVAE` encoder with a tiled neighborhood attention in place of NATTEN, which has no
  Neuron build. Each Swin block is split over the TP group, which is one chip's four cores. At CP2, the first two
  stages' canvas rows are split over the CP pair with a window halo. The DROID canvas (the wrist view over the two
  halved exterior views) is built from the uint8 pixels in the same graph.
- **Sampler:** SGLang's Cosmos UniPC, with each step's CFG combine and update as one device graph. The coefficients
  come from the sigmas and config of SGLang's scheduler. At CFG2, each CFG rank runs its own branch and the step graph
  gathers both.

The models use SGLang's own TP, SP and CFG process groups, bound and checked by `runtime.initialize_model_groups` as
FLUX.1-dev's are. Each worker runs 4 torch threads.

The code is in `sglang_neuron/diffusion/flux3_action/` (the pipeline, DiT, VAE, sampler and text encoder), and the
block's kernels are in `sglang_neuron/diffusion/kernels/flux3.py`.

## Setup

The container and the SGLang checkout are the ones in the [setup guide](../getting-started/setup-guide.md). A container
must see the Neuron devices of the cores it serves on. Build the venv as for FLUX.1-dev. The script also installs
`websockets` and `msgpack`, which the OpenPI endpoint needs:

```bash
# from this repository's checkout under /work, as in the setup guide
NEURON_RT_VISIBLE_CORES=0 scripts/setup_sglang_diffusion_env.sh /work/venv /work/sglang
source /work/venv/bin/activate
export PYTHONPATH=/work/sglang/python
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

The commands below run in this venv: its `python` and `sglang` have the plugin, `websockets` and `msgpack`.

`TORCH_DEVICE_BACKEND_AUTOLOAD=0` is required for every launch, for the reason in the FLUX.1-dev recipe's
[runtime path](flux1-dev.md#the-runtime-path).

The policy package holds the `sd` and `gd` variants under `variants/` and the `base` policy at its root. The policy
config names its frozen components in `black-forest-labs/flux-3-action-base` at a pinned revision: the text encoder as
the `text_encoder` folder and the video VAE as `video_vae.safetensors`. SGLang downloads them when no component path
is given. To serve from local copies of that revision, pass them as components:

```bash
--component-paths.text_encoder /work/models/flux-3-action-base/text_encoder \
--component-paths.vae /work/models/flux-3-action-base/video_vae.safetensors
```

At the pinned revision, the `text_encoder` folder has the same 713 tensors and the same tokenizer and config files
(by SHA-256) as `Qwen/Qwen3-VL-4B-Instruct`. The accuracy results on this page served a local copy of
`Qwen/Qwen3-VL-4B-Instruct`.

Cores: worker `i` takes the `i`-th core of `NEURON_VISIBLE_DEVICES`, else of the launcher's `NEURON_RT_VISIBLE_CORES`,
else core `i`. Each TP group is one chip's four cores when the list starts on a multiple of 4, as in every validated
run.

## Serving

```bash
sglang serve --model-path black-forest-labs/flux-3-action-droid --model-type diffusion --model-variant gd \
  --num-gpus 8 --tp-size 4 --ulysses-degree 2 --host 127.0.0.1 --port 30000
```

Before the server reports ready, the pipeline runs one DROID request through its stages on every rank
(`NeuronFlux3ActionPipeline.warm_up`). The request has three black uint8 360x640 views, a zero state, and the package's
default steps and guidance. The first served request therefore compiles no graph. SGLang's default warmup builds no
request for an action policy, and `--warmup-mode request` warms only a one-step copy on a float canvas.

Clients use SGLang's OpenPI WebSocket, `/openpi/policy`, with msgpack numpy payloads: three uint8 360x640 views as
`observation.images.<camera>` (`wrist`, `left`, `right`, or the LeRobot names), the 8-dim state as
`observation.state`, and the instruction as `prompt`. The response carries the `[32, 8]` chunk as `actions`. The
cookbook page has the full schema.

SGLang's HTTP action API, `POST /v1/actions/generations` with `GET /v1/actions/metadata`, serves the same pipeline. On
`gd` TP4 x CP2, three uint8 views sent as base64 PNG returned the same actions as the WebSocket, with a longer round
trip. Pixel lists also returned the same actions, with a much longer round trip: the three views become a large JSON
body.

```python
import asyncio

import numpy as np
import websockets
from sglang.multimodal_gen.runtime.entrypoints.action.protocol import pack_msgpack, unpack_msgpack


async def main():
    observation = {f"observation.images.{name}": np.zeros((360, 640, 3), dtype=np.uint8)
                   for name in ("wrist", "left", "right")}
    observation["observation.state"] = np.zeros(8, dtype=np.float32)
    observation["prompt"] = "put the marker in the cup"
    async with websockets.connect("ws://127.0.0.1:30000/openpi/policy", max_size=None) as ws:
        await ws.recv()
        await ws.send(pack_msgpack(observation))
        actions = np.asarray(unpack_msgpack(await ws.recv())["actions"])
        print(actions.shape)


asyncio.run(main())
```

SGLang's benchmark measures a running server:

```bash
python -m sglang.multimodal_gen.benchmarks.bench_flux3_action --url ws://127.0.0.1:30000 --warmup 10 --requests 100
```

### In the container image

The SGLang-Neuron container image ([docker/README.md](../../docker/README.md)) has SGLang, its diffusion packages
(`websockets` and `msgpack` included) and this plugin installed. Start it as
[docker/README.md](../../docker/README.md#run) says: the Neuron devices of the cores it serves on passed, the compile
caches mounted, and `-e TORCH_DEVICE_BACKEND_AUTOLOAD=0` added. Inside it, the commands of [Serving](#serving) run with
these differences:

- No setup script, venv, SGLang checkout or `PYTHONPATH`: skip [Setup](#setup) except its component paragraph, and run
  `sglang` and `python3` from the image.
- The checkpoint: the examples assume the policy package `black-forest-labs/flux-3-action-droid` is in a host directory
  mounted as `/models/flux-3-action-droid` and the base components of [Setup](#setup)
  (`black-forest-labs/flux-3-action-base` at the revision the policy config names: its `text_encoder` folder and
  `video_vae.safetensors`) as `/models/flux-3-action-base`. Pass the package as `--model-path` and the components with
  `--component-paths.text_encoder` and `--component-paths.vae`.
- Give `sglang serve` `--host 0.0.0.0` in place of `--host 127.0.0.1` for clients outside the container
  ([docker/README.md](../../docker/README.md#run)).
- The Python client and the benchmark import SGLang: run them with the image's `python3`, for example with
  `docker exec <container> python3 <script>` and the script in a mounted directory.

The server, from the host:

```bash
docker run --rm --init --ipc=host --cap-add SYS_NICE \
    $(for d in /dev/neuron*; do printf -- '--device %s ' "$d"; done) \
    -v <models>:/models -v <cache dir>:/cache -p 127.0.0.1:30000:30000 \
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    -e TORCH_NEURONX_NEFF_CACHE_DIR=/cache/neff -e TORCH_NEURONX_HLO_CACHE_DIR=/cache/hlo \
    -e NKI_COMPILE_CACHE_URL=/cache/nki \
    sglang-neuron:<tag> sglang serve --model-path /models/flux-3-action-droid \
      --component-paths.text_encoder /models/flux-3-action-base/text_encoder \
      --component-paths.vae /models/flux-3-action-base/video_vae.safetensors \
      --model-type diffusion --model-variant gd --num-gpus 8 --tp-size 4 --ulysses-degree 2 \
      --host 0.0.0.0 --port 30000
```

## Compiled graphs

The first start of a layout compiles its graphs before the server reports ready. torch-neuronx caches the compiled
graphs under `TORCH_NEURONX_NEFF_CACHE_DIR` (default `/tmp/neff_cache`). Keep that directory across containers to skip
the compile. A worker refuses a cache that was built from other NKI kernel sources, as in the FLUX.1-dev recipe's
[Compiled shapes](flux1-dev.md#compiled-shapes).

## Accuracy

Measured on trn2.48xlarge with the validated container and the SGLang checkout of the setup guide, in one run: 16 DROID
observations against SGLang's own pipeline on the CPU in FP32 (the baseline) and BF16. A layout passes when the
Bhattacharyya coefficient (BC) of the error distributions is at least 0.99 or its RMS error is at most BF16's, with
every per-fixture L-inf ratio below 5 and L2 ratio below 3.

| Variant | Layout | BC | RMS ratio | worst L-inf / L2 |
|---|---|---|---|---|
| `sd` | TP4 x CP2 | 0.9944 | 1.109 | 1.86 / 1.55 |
| `sd` | TP4 | 0.9937 | 1.118 | 1.98 / 1.42 |
| `gd` | TP4 x CP2 | 0.9894 | 0.860 | 1.72 / 2.06 |
| `gd` | TP4 | 0.9845 | 0.816 | 1.80 / 2.11 |
| `base` | TP4 x CFG2 | 0.9899 | 0.821 | 2.56 / 2.08 |

All five layouts pass. `gd` TP4 passes on its RMS error, with a BC of 0.9845.

## Environment variables

| Variable | Effect |
|---|---|
| `TORCH_DEVICE_BACKEND_AUTOLOAD=0` | Required for a diffusion launch (Setup) |
| `NEURON_VISIBLE_DEVICES`, `NEURON_RT_VISIBLE_CORES` | Core indices the workers take, in rank order. The first one set wins (Setup) |
| `NEURON_CC_FLAGS`, `NEURON_COMPILER_OPT_LEVEL` | Refused at launch when set. The DiT and the VAE pass their own compiler arguments |
| `TORCH_NEURONX_NEFF_CACHE_DIR` | torch-neuronx's compiled-graph cache (default `/tmp/neff_cache`) |

## What is refused

A launch that the pipeline does not serve fails before the server reports ready, and names the setting at fault:

- An FP8r package (`--model-variant fp8r`, `gd-fp8r`, `sd-fp8r`): `NotImplementedError`.
- A TP size that does not divide every VAE block's head count and MLP width, such as `--tp-size 3` or `8`:
  `ValueError`.
- An exported `NEURON_CC_FLAGS` or `NEURON_COMPILER_OPT_LEVEL`: `ValueError`. FLUX.3-Action has no compiler profile,
  so any value is refused.
- The platform's refusals for every diffusion launch: CPU or layerwise offload, SGLang's per-component placement, FSDP
  inference, torch.compile, the breakable CUDA graph, and a `--disagg-role` other than `monolithic` (see the FLUX.1-dev
  recipe's [What is refused](flux1-dev.md#what-is-refused)).

A request carries one observation and returns one action chunk. An `action_horizon` above the policy's 32 actions is
rejected.
