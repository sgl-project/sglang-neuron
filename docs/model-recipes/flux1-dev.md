# FLUX.1-dev on Neuron

`FluxPipeline` (`black-forest-labs/FLUX.1-dev`) runs text-to-image on SGLang Diffusion
(`sglang.multimodal_gen`), not on the plugin's LLM [runtime](../design/runtime.md), in bf16 on
torch-neuronx: 8-way tensor parallel (TP8) x 8-way context parallel (CP8) over the 64 logical
NeuronCores of a trn2.48xlarge (LNC=2, `NEURON_LOGICAL_NC_CONFIG=2`: one logical core is two
physical cores). The pipeline is `sglang_neuron/diffusion/flux_pipeline.py`; the DiT (the FLUX
diffusion transformer) and the text encoders are in `sglang_neuron/diffusion/models/flux/`, the
VAE in `sglang_neuron/diffusion/distributed/autoencoders/` and the NKI kernels in
`sglang_neuron/diffusion/kernels/flux/`.

The tested launch is `--num-gpus 64 --tp-size 8` on a trn2.48xlarge. Other worker counts, other TP
sizes and trn3 are not tested.

## The model

One SGLang stage (`NeuronFluxStage`) runs the whole request on every rank in lockstep:

- text encoding: CLIP-L (on every rank) and T5-XXL (sharded over all 64 ranks);
- per-request DiT conditioning: the text tokens and every step's modulation rows (the per-step
  shift, scale and gate values of the DiT's AdaLN layers, computed from the timestep, the
  guidance scale and the pooled CLIP text embedding);
- the denoise loop; the Euler update is inside the DiT's last graph (at 4096 x 4096, where the
  DiT runs one graph per block, the Euler step is a graph of its own);
- the VAE decode, tiled over all 64 ranks; rank 0 then converts the image to SGLang's output
  format (float32 in [0, 1]) on one CPU thread.

Context-parallel attention is chosen by the joint token count: `(H/16) x (W/16)` image tokens,
for an image of `H` x `W` pixels, plus 512 text tokens. Up to 4,608 tokens (1024 x 1024), K/V
are all-gathered over the 8 CP ranks inside NKI kernels: `flux_gather_attention_fwd` in the 19
double-stream blocks; in the 38 single-stream blocks, `flux_gather_kv_fwd` then
`flux_gathered_kv_attention_fwd`, with the block's MLP running while the gather is in flight.
Above that (2048 x 2048 is 16,896 tokens), every block uses the ring kernel
(`flux_ring_attention_fwd`), which passes K/V from rank to rank.

## The runtime path

The SGLang launcher is the process you start (`sglang generate`, `sglang serve` or a
`DiffGenerator` script); it starts one worker process per NeuronCore.

1. **Activation.** SGLang selects the platform through the `sglang.multimodal_gen.platforms`
   entry point when torch-neuronx is installed (`sglang_neuron.diffusion:activate`). The
   `sglang.multimodal_gen.plugins` entry point then registers `NeuronFluxPipeline` for
   `FluxPipeline` checkpoints, replacing SGLang's own.
   The launcher holds no NeuronCore, and must not start the Neuron runtime:
   1. `import torch` loads torch-neuronx through its `torch.backends` autoload entry point
      (`TORCH_DEVICE_BACKEND_AUTOLOAD=0` turns that off);
   2. once torch-neuronx is loaded, the Neuron device is torch's accelerator, and SGLang
      Diffusion queries it at import, which starts the runtime on every visible core. The
      workers then cannot claim their cores.

   The platform fails at start, naming the variable, when the launcher has started the
   runtime.
2. **Launcher.** Before the workers start, the platform:
   - refuses the launches listed in [What is refused](#what-is-refused);
   - sets the model's environment profile (`env_profiles.FLUX1_DEV`: Neuron runtime collective
     settings, the HBM scratchpad page size, `NEURON_LOGICAL_NC_CONFIG=2`, the compiler flags,
     `TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1` (a graph that fails on the device raises
     instead of running on the CPU), the compile-artifact directory with its artifacts kept, and
     the compile-cache lock timeout) as defaults, which the workers inherit; an exported value
     wins, except for the compiler variables ([Environment variables](#environment-variables));
   - turns on Gloo lazy init (`TORCH_GLOO_LAZY_INIT=1`): each worker creates hundreds of
     process groups, and an eager full-mesh connect can fail intermittently;
   - caps the launcher's CPU thread pool at `cpus // tp-size`, where `cpus` is the number of
     CPUs the launcher may run on: an uncapped pool busy-waits on the CPUs the workers use.

   The platform reads `_class_name` in the checkpoint's `model_index.json` to pick the profile.
   Without the file (a local directory that has none, or the Hub unreachable with no cached
   copy; only the second logs a warning), it launches without the FLUX profile: none of the
   profile's defaults are set.
3. **Workers.** Each of the 64 workers claims one logical core, checks the compile caches (a
   cache built from other NKI kernel sources or another kernel library is refused; see
   [Compiled shapes](#compiled-shapes)) and only then imports torch-neuronx
   (`runtime.configure_worker`). SGLang then initializes `torch.distributed` with
   `cpu:gloo,neuron:neuron`: torch-neuronx's `neuron` process group sets up the device runtime,
   Gloo carries the CPU collectives. SGLang builds its own TP and sequence-parallel (SP) groups,
   TP8 x SP8, and the pipeline uses the SP groups as its CP groups. Before SGLang builds them,
   each worker renames SGLang's rank order onto the trn2 physical mesh (a hook,
   `sglang_neuron/sglang_patches/diffusion_mesh_rank_order.py`), so every TP and CP group is a
   ring on the chip torus (a trn2.48xlarge's 16 chips form a 4 x 4 torus; e.g. TP
   `[0,1,2,3,12,13,14,15]`). SGLang's scheduler on rank 0 hands each request to
   the other 63 workers.
4. **Metrics.** The stage records `TextEncodingStage`, `ConditioningStage`, `DenoisingStage`
   (with per-step times) and `DecodingStage` in SGLang's request metrics. torch-neuronx queues
   device work and returns, so rank 0 times the stages with device events
   (`torch.neuron.Event`), read after the image is copied to the CPU. They appear in
   `GenerationResult.metrics` and in SGLang's perf logs, and rank 0 logs one
   `[NeuronFlux] ... e2e_forward=...` line per request.

## Setup

The Neuron container and the SGLang commit are the ones the plugin's LLM path uses: start the
container and check SGLang out at `/work/sglang` as the
[setup guide](../getting-started/setup-guide.md) says (its `docker run`, `git clone` and
`git checkout` lines). Do not run its `setup_sglang_main_env.sh` line: the script below runs it,
and it refuses a venv directory that already exists. The container must see all 16 Neuron
devices (`/dev/neuron*`).

```bash
# from this repository's checkout under /work, as in the setup guide
NEURON_RT_VISIBLE_CORES=0 scripts/setup_sglang_diffusion_env.sh /work/venv /work/sglang
export PYTHONPATH=/work/sglang/python
# run with /work/venv/bin/python and /work/venv/bin/sglang
```

The script:

- runs `scripts/setup_sglang_main_env.sh`: a venv over the container's site-packages, SGLang's
  runtime packages at the pins in the checkout's `python/pyproject_other.toml` with torch held at
  the container's build, and this plugin installed editable;
- adds the packages SGLang Diffusion and the diffusion pipelines import, from
  `docker/diffusion-requirements.txt`: diffusers 0.40.0, cache-dit 1.5.0, imageio 2.36.0,
  imageio-ffmpeg 0.5.1, addict 2.4.0, PyAV (`av`) 16.1.0, msgpack 1.2.4 and websockets 17.4, all
  with torch pinned to the container's;
- installs the SGLang checkout, editable and without its dependencies, which adds the `sglang`
  command (`sglang serve`, `sglang generate`). `pip check` therefore lists SGLang's CUDA packages
  and exact pins that this environment does not install
  ([Requirements left unmet](../getting-started/setup-guide.md#requirements-left-unmet)).

Checkpoint: FLUX.1-dev is gated on Hugging Face. Accept its license and log in (`hf auth login`,
the Hugging Face CLI, or set `HF_TOKEN`), then either pass the Hub ID as `--model-path` (SGLang
downloads it) or download it first, for example
`hf download black-forest-labs/FLUX.1-dev --local-dir /work/models/FLUX.1-dev`, and pass the
directory.

Cores: each worker takes one logical core, and the platform sets `NEURON_RT_VISIBLE_CORES` for
it. The cores come from `NEURON_VISIBLE_DEVICES`, else the launcher's `NEURON_RT_VISIBLE_CORES`,
else `0..63`, as core indices (a range like `0-63` or a comma list); worker `i` takes the `i`-th.
This is the order the plugin's LLM path reads the server's cores in (`cpu_placement.server_cores`).

## Serving

Every diffusion launch sets one variable, so that the SGLang launcher does not start the Neuron
runtime ([The runtime path](#the-runtime-path), step 1):

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

`--num-gpus` is the number of workers, one logical NeuronCore each. SGLang sets its SP degree to
`--num-gpus / --tp-size` (8), and the pipeline uses SGLang's SP groups as its CP groups; leave
`--sp-degree` and `--dp-size` unset. Pass the step count: SGLang's FLUX default is 50 steps,
which compiles its own graphs and is not validated; 25 is.

Offline, one image:

```bash
/work/venv/bin/sglang generate --model-path black-forest-labs/FLUX.1-dev \
  --num-gpus 64 --tp-size 8 \
  --prompt "A red fox sitting on a mossy log in a misty forest at sunrise" \
  --height 1024 --width 1024 --num-inference-steps 25 --guidance-scale 3.5 --seed 42 \
  --output-path out --output-file-name fox.png
```

Python (`DiffGenerator`, run with `/work/venv/bin/python`):

```python
import numpy as np
from sglang.multimodal_gen import DiffGenerator


def main():
    generator = DiffGenerator.from_pretrained(
        model_path="black-forest-labs/FLUX.1-dev", num_gpus=64, tp_size=8
    )
    result = generator.generate(sampling_params_kwargs=dict(
        prompt="...", height=1024, width=1024, num_inference_steps=25, guidance_scale=3.5,
        seed=42, save_output=False,
    ))
    image = np.asarray(result.frames[0])  # H x W x 3, uint8
    generator.shutdown()


if __name__ == "__main__":  # SGLang starts the workers as subprocesses
    main()
```

Server (OpenAI images API):

```bash
/work/venv/bin/sglang serve --model-type diffusion --model-path black-forest-labs/FLUX.1-dev \
  --num-gpus 64 --tp-size 8 --warmup-mode request --port 30000
```

`--warmup-mode request` warms up with a 1-step copy of the first request, so the first request
compiles the 1-step graphs of that copy and then its own graphs
([Compiled shapes](#compiled-shapes)). A request (fields beyond the
OpenAI ones go at the top level):

```bash
curl -s http://127.0.0.1:30000/v1/images/generations -H "Content-Type: application/json" -d '{
  "model": "black-forest-labs/FLUX.1-dev", "prompt": "A red fox sitting on a mossy log",
  "size": "1024x1024", "n": 1, "response_format": "b64_json", "output_format": "png",
  "num_inference_steps": 25, "guidance_scale": 3.5, "seed": 42}'
```

The response carries the image as base64 in `data[0].b64_json`, as PNG here; without
`output_format` it is JPEG.

Seeds follow SGLang: output `i` of a prompt uses seed `seed + i`. The initial noise is drawn on
the CPU, so it is the same on every host.

SGLang turns each prompt of a list, and each output of a prompt (`n` in the images API,
`num_outputs_per_prompt` in `DiffGenerator`), into a request of its own; the requests run one
after another.

Image sizes: 1024 and 2048 square at 25 steps are validated for quality. 4096 square runs and
matches the reference, but the image is near-uniform, so it is not useful for real images.

### In the container image

The SGLang-Neuron container image ([docker/README.md](../../docker/README.md)) has SGLang, its
diffusion packages and this plugin installed. Start it as
[docker/README.md](../../docker/README.md#run) says: the Neuron devices passed, the checkpoint and
the compile caches mounted, and `-e TORCH_DEVICE_BACKEND_AUTOLOAD=0` added. Inside it, the commands
of [Serving](#serving) run with these differences:

- No setup script, venv, SGLang checkout or `PYTHONPATH`: skip [Setup](#setup), and run `sglang` and
  `python3` in place of `/work/venv/bin/sglang` and `/work/venv/bin/python`.
- The checkpoint: the examples assume the `black-forest-labs/FLUX.1-dev` checkpoint is in a host
  directory mounted as `/models/FLUX.1-dev`, its Hub repo's name, and pass that directory as
  `--model-path`. Keep that name: SGLang picks the Neuron pipeline by it.
- A Python script, its inputs and its outputs must be in a mounted directory, since `--rm` removes
  the container's own files when it stops.
- `sglang serve` takes `--host 0.0.0.0` for clients outside the container
  ([docker/README.md](../../docker/README.md#run)).

One image, from the host:

```bash
docker run --rm --init --ipc=host --cap-add SYS_NICE \
    $(for d in /dev/neuron*; do printf -- '--device %s ' "$d"; done) \
    -v <models>:/models -v <cache dir>:/cache -v "$PWD/out":/out \
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e TORCH_NEURONX_NEFF_CACHE_DIR=/cache/neff \
    -e TORCH_NEURONX_HLO_CACHE_DIR=/cache/hlo -e NKI_COMPILE_CACHE_URL=/cache/nki \
    sglang-neuron:<tag> sglang generate --model-path /models/FLUX.1-dev \
      --num-gpus 64 --tp-size 8 \
      --prompt "A red fox sitting on a mossy log in a misty forest at sunrise" \
      --height 1024 --width 1024 --num-inference-steps 25 --guidance-scale 3.5 --seed 42 \
      --output-path /out --output-file-name fox.png
```

## Environment variables

| Variable | Effect |
|---|---|
| `TORCH_DEVICE_BACKEND_AUTOLOAD=0` | Required for a diffusion launch ([Serving](#serving)) |
| `NEURON_VISIBLE_DEVICES`, `NEURON_RT_VISIBLE_CORES` | Core indices the workers take, in rank order; the first set wins ([Setup](#setup)) |
| `TORCH_NEURONX_NEFF_CACHE_DIR`, `TORCH_NEURONX_HLO_CACHE_DIR` | torch-neuronx's compiled-graph caches (defaults `/tmp/neff_cache` and `/tmp/hlo_cache`; [Compiled shapes](#compiled-shapes)) |
| `TORCH_NEURONX_DEBUG_DIR` | Compile artifacts; the profile's default is `./compile_dir`, so launch from a writable directory or set it |
| `NEURON_LOGICAL_NC_CONFIG` | The profile sets 2; the NKI kernels refuse another value |
| `NEURON_CC_FLAGS`, `NEURON_COMPILER_OPT_LEVEL` | Set by the plugin: the profile's flags for most graphs, the DiT's and the VAE's own for theirs. A launch that exports `NEURON_CC_FLAGS` other than the profile's `-O1 --hbm-scratchpad-page-size=2048`, or any `NEURON_COMPILER_OPT_LEVEL`, is refused |
| `SGLANG_NEURON_REQUIRE_NKI_KERNELS=1` | An NKI kernel gate that takes its torch fallback raises `NkiKernelRequiredError` (fallbacks marked by design only log) |
| `SGLANG_NEURON_DISABLE_NKI_KERNELS=1` | Take the torch fallback of every NKI kernel |

The environment profile (`sglang_neuron/diffusion/env_profiles.py`) sets its other variables
only where they are unset. Each worker sets its own `NEURON_RT_VISIBLE_CORES` (its core) and
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` (1), whatever was exported.

## Compiled shapes

The first request at a new image size or step count compiles its graphs. The per-request
conditioning holds one modulation row per step, so the DiT graphs' input shapes depend on both.
Give the first request at each new size or step count a client timeout above its compile time.

torch-neuronx caches compiled graphs under `TORCH_NEURONX_NEFF_CACHE_DIR` (default
`/tmp/neff_cache`) and `TORCH_NEURONX_HLO_CACHE_DIR` (default `/tmp/hlo_cache`). Keep both
across containers of the same image to skip the compile. A cached graph's key covers neither the
NKI kernel sources nor the kernel library, so each cache records a digest of both (a
`KERNEL_SOURCES` file, which the first launch on an empty cache writes). A worker refuses a cache
with another digest (other kernel sources, or another neuronx-cc, which ships the kernel library),
and a filled cache with no record, and names the cache. A cache filled before the plugin recorded
digests has no record and is refused: remove it, or, if this checkout and this image built it,
record the digest with `/work/venv/bin/python -m sglang_neuron.backend.compile_cache stamp <dir>`.

## What is refused

The launcher refuses at start, naming the setting at fault:

- a launcher that has started the Neuron runtime (launch with `TORCH_DEVICE_BACKEND_AUTOLOAD=0`;
  [The runtime path](#the-runtime-path), step 1);
- CPU offload, layerwise offload, FSDP inference, torch.compile, the breakable CUDA graph, and
  SGLang's per-component placement (`--component-residency`, `--cpu-offload-components`,
  `--layerwise-offload-components`), when turned on by a flag or a `DiffGenerator.from_pretrained`
  argument (for example `--dit-cpu-offload`, `--enable-torch-compile`). SGLang's own defaults for
  them are turned off: the Neuron pipeline places and compiles its own modules;
- `--disagg-role` other than `monolithic` (disaggregated or pool serving);
- `NEURON_CC_FLAGS` or `NEURON_COMPILER_OPT_LEVEL` exported with another value than the
  profile's, and `TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE` exported with another value than `1`
  (torch-neuronx's vocab-sharding pass, which the plugin keeps off);
- an SGLang checkout whose functions the rank-order hook replaces or relies on differ from the
  ones at the [setup guide](../getting-started/setup-guide.md)'s commit.

The workers refuse at start, before they open their core, a compile cache whose digest differs
([Compiled shapes](#compiled-shapes)). They refuse at model load:

- CFG parallel (`--enable-cfg-parallel`, `--cfg-parallel-size` above 1): FLUX.1-dev uses
  guidance embedding;
- `--dp-size` above 1, or an `--sp-degree` other than `--num-gpus / --tp-size`;
- a `FluxPipeline` checkpoint without guidance embedding (for example FLUX.1-schnell): the
  plugin replaces SGLang's pipeline for every `FluxPipeline` checkpoint, but runs only
  guidance-distilled ones (`guidance_embeds` in `transformer/config.json`), such as FLUX.1-dev;
- `NEURON_LOGICAL_NC_CONFIG` other than 2.

A request fails when:

- its height or width is not a multiple of 64, or is below 512, or its image tokens
  (`(H/16) x (W/16)`) do not split over the 64 ranks: they must be a multiple of 64, so for
  example 576 x 576 fails;
- it has a negative prompt with `true_cfg_scale > 1`.

LoRA and image-to-image are not supported.
