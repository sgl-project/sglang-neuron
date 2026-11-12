# Wan2.2 A14B on Neuron

With this plugin, SGLang Diffusion runs Wan2.2 A14B text-to-video
(`Wan-AI/Wan2.2-T2V-A14B-Diffusers`) and image-to-video (`Wan-AI/Wan2.2-I2V-A14B-Diffusers`) on a
trn2.48xlarge: BF16 weights, torch-neuronx, one 81-frame video per request at 480 x 832,
720 x 1280 or 1072 x 1920 (height x width), 40 steps. The model runs 4-way tensor parallel x
8-way context parallel x 2-way CFG parallel (TP4 x CP8 x CFG2) on the 64 logical NeuronCores
(LNC=2). Its defaults are those of the reference pipeline it is validated against.

## The model

- **DiT (both experts):** TP4 splits the 40 heads (10 per rank) and the FFN, and the residual
  stream is sequence-parallel over the TP group. CP8 splits the video tokens: 32,760 / 75,600 /
  168,840 at 480p / 720p / 1080p, or 4,095 / 9,450 / 21,105 per rank. Every rank keeps all 512
  text tokens for cross-attention.
- **CFG2:** the 64 ranks form two 32-rank replicas, each a full TP4 x CP8 DiT. One replica runs
  the positive prompt, the other the negative one. Each of the 32 CFG groups pairs a rank with its
  counterpart in the other replica, and one compiled graph per step gathers the two predictions
  over that pair and combines them.
- **UMT5 text encoder:** TP over all 64 ranks.
- **VAE:** ranks `0..size-1` for the patch-parallel degree `size`, in 32 x 32-latent tiles at
  stride 24.

The TP, CP and CFG groups are SGLang's own. On trn2.48xlarge they are placed on the 8 x 8 physical
mesh so that each group is a torus ring (`sglang_neuron/diffusion/distributed/parallel_state.py`).

## Setup

The container has Python 3.14, torch 2.14.0 and torch-neuronx 2.14.3, and must see all 16 Neuron
devices. Build the venv on SGLang `43881a7`, the commit in the
[setup guide](../getting-started/setup-guide.md); the plugin's SGLang patches refuse another:

```bash
git clone https://github.com/sgl-project/sglang /work/sglang
git -C /work/sglang checkout 43881a7
# from this repository's checkout under /work, as in the setup guide
NEURON_RT_VISIBLE_CORES=0 scripts/setup_sglang_diffusion_env.sh /work/venv /work/sglang
source /work/venv/bin/activate  # the venv's sglang and python
export PYTHONPATH=/work/sglang/python
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

`TORCH_DEVICE_BACKEND_AUTOLOAD=0` is required for every launch. Without it, `import torch` in the
SGLang launcher autoloads torch-neuronx, which starts the Neuron runtime on every visible core and
leaves none for the 64 workers. The platform refuses to start when that has happened and names the
variable.

`--model-path` takes the Hub ID or a downloaded checkpoint directory, under any directory name.
Only Wan2.2 A14B checkpoints with BF16 weights run. The plugin refuses any other `WanPipeline` or
`WanImageToVideoPipeline` checkpoint (Wan2.1, Wan2.2 TI2V-5B, a quantized export) at model load.

## Serving

The layout is `--num-gpus 64 --tp-size 4 --cfg-parallel-size 2`. SGLang Diffusion sets its
sequence-parallel degree `sp_degree` to `num-gpus / (tp-size * cfg-parallel-size)` = 8, and the
pipeline uses SGLang's SP groups as its CP groups. CP attention is the plugin's own ring kernel over
each group; SGLang's Ulysses / ring split of `sp_degree` is unused. Leave `--sp-degree`,
`--ulysses-degree`, `--ring-degree` and `--dp-size` unset.

The VAE patch-parallel degree is fixed when the model loads. Set
`SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE` to the degree for the resolution you will serve:

| | 480p: `--height 480 --width 832` | 720p: `--height 720 --width 1280` | 1080p: `--height 1072 --width 1920` |
|---|---|---|---|
| T2V | 16 | 32 | 64 |
| I2V | 16 | 32 | 32 |

Unset, the degree comes from the first `--warmup-resolutions` entry, else 480p. A request at a size
whose degree differs still runs, with a WARNING.

For 1080p, request a height of 1072: I2V raises unless height and width are multiples of 16, and
T2V rounds them down. A request without a size gets 480 x 832.

T2V at 480p:

```bash
SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE=16 \
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --num-gpus 64 --tp-size 4 --cfg-parallel-size 2 \
  --prompt "A fluffy orange cat walking gracefully across a sunny garden path, high quality, detailed" \
  --negative-prompt "orange, cat, animal" \
  --height 480 --width 832 --num-frames 81 --num-inference-steps 40 \
  --guidance-scale 5.0 --guidance-scale-2 5.0 --seed 42 \
  --output-path out --output-file-name t2v_480p.mp4
```

I2V adds the first frame with `--image-path`. The image is resized to the requested size
(LANCZOS). This example uses Wan2.2's own sample image,
[`examples/i2v_input.JPG`](https://github.com/Wan-Video/Wan2.2/blob/main/examples/i2v_input.JPG) in
Wan-Video/Wan2.2:

```bash
curl -L -o i2v_input.JPG https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/i2v_input.JPG
```

```bash
SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE=16 \
sglang generate --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --num-gpus 64 --tp-size 4 --cfg-parallel-size 2 \
  --image-path i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
  --height 480 --width 832 --num-frames 81 --num-inference-steps 40 --seed 42 \
  --output-path out --output-file-name i2v_480p.mp4
```

From Python, export the VAE variable before `from_pretrained`, since the workers read it at load:

```python
import numpy as np
from sglang.multimodal_gen import DiffGenerator


def main():
    generator = DiffGenerator.from_pretrained(
        model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers", num_gpus=64, tp_size=4,
        cfg_parallel_degree=2,
    )
    result = generator.generate(sampling_params_kwargs=dict(
        prompt="...", negative_prompt="...", height=480, width=832, num_frames=81,
        num_inference_steps=40, seed=42, save_output=False,
        # I2V: image_path="i2v_input.JPG"
    ))
    frames = np.stack([np.asarray(f) for f in result.frames])  # T x H x W x 3, uint8
    generator.shutdown()


if __name__ == "__main__":  # SGLang starts the workers as subprocesses
    main()
```

Server (OpenAI videos API):

```bash
SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE=16 \
sglang serve --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --num-gpus 64 --tp-size 4 --cfg-parallel-size 2 --warmup-mode request --port 30000
```

A video request is a job: create it, poll it until `"status": "completed"`, then download the
mp4.

```bash
curl -s http://127.0.0.1:30000/v1/videos -H "Content-Type: application/json" -d '{
  "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "prompt": "A fluffy orange cat walking ...",
  "negative_prompt": "orange, cat, animal", "size": "832x480", "num_frames": 81, "fps": 16,
  "num_inference_steps": 40, "guidance_scale": 5.0, "guidance_scale_2": 5.0, "seed": 42}'
curl -s http://127.0.0.1:30000/v1/videos/<id>
curl -s -o t2v_480p.mp4 http://127.0.0.1:30000/v1/videos/<id>/content
```

Pass `num_frames` and `fps`: the API's default is 4 s at 24 fps. I2V sends the image as the
multipart file `input_reference`; a T2V server rejects it with HTTP 400.

Rank 0 logs one line per request:

```
[NeuronWan] t2v 832x480x81 steps=40 flow_shift=5.0 boundary=0.875 guidance=5.0/5.0 cfg_parallel=2 e2e_forward=..s text=..s vae_encode=..s denoise=..s step_p50=..s vae=..s
```

`e2e_forward` is the wall time of the request on rank 0, from text encoding to the video's copy to
the CPU.

### Defaults

The configs in `sglang_neuron/diffusion/models/wan2_2/wan_pipeline.py` carry the reference pipeline's values,
several of which differ from SGLang's Wan defaults:

| Parameter | This plugin | SGLang `43881a7` |
|---|---|---|
| Flow shift, T2V / I2V | 5.0 / 5.0 | 12.0 / 5.0 |
| Guidance, high- / low-noise expert, T2V | 5.0 / 5.0 | 4.0 / 3.0 |
| Guidance, high- / low-noise expert, I2V | 5.0 / 5.0 | 3.5 / 3.5 |
| Negative prompt | `""` (the negative branch still runs) | a long Chinese prompt |

A launch or request flag still wins. `--guidance-scale` sets the high-noise expert and
`--guidance-scale-2` the low-noise one; setting only the first leaves the second at 5.0, so pass
both. The scheduler applies the flow shift twice, at build and again in `set_timesteps`, as the
reference pipeline does. The checkpoint's `scheduler_config.json` (shift 3.0) is ignored.

### In the container image

The [Wan2.2 example](../../examples/wan22/) runs the T2V command below as a script.

The SGLang-Neuron container image ([docker/README.md](../../docker/README.md)) has SGLang, its
diffusion packages and this plugin installed. Start it as
[docker/README.md](../../docker/README.md#run) says: the Neuron devices passed, the checkpoint and
the compile caches mounted, and `-e TORCH_DEVICE_BACKEND_AUTOLOAD=0` added. Inside it, the commands
of [Serving](#serving) run with these differences:

- No setup script, venv, SGLang checkout or `PYTHONPATH`: skip [Setup](#setup), and run `sglang`
  and `python3` from the image.
- Pass `SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE` into the container (`-e`).
- The checkpoint: the examples assume `Wan-AI/Wan2.2-T2V-A14B-Diffusers` is in a host directory
  mounted as `/models/Wan2.2-T2V-A14B-Diffusers` (I2V: `Wan-AI/Wan2.2-I2V-A14B-Diffusers` as
  `/models/Wan2.2-I2V-A14B-Diffusers`), its Hub repo's name, and pass that directory as
  `--model-path`.
- I2V: mount the input image's directory and pass its path in the container as `--image-path`.
- A Python script, its inputs and its outputs must be in a mounted directory, since `--rm` removes
  the container's own files when it stops.
- `sglang serve` takes `--host 0.0.0.0` for clients outside the container
  ([docker/README.md](../../docker/README.md#run)).

T2V at 480p, from the host:

```bash
docker run --rm --init --ipc=host --cap-add SYS_NICE \
    $(for d in /dev/neuron*; do printf -- '--device %s ' "$d"; done) \
    -v <models>:/models -v <cache dir>:/cache -v "$PWD/out":/out \
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE=16 \
    -e TORCH_NEURONX_NEFF_CACHE_DIR=/cache/neff -e TORCH_NEURONX_HLO_CACHE_DIR=/cache/hlo \
    -e NKI_COMPILE_CACHE_URL=/cache/nki \
    sglang-neuron:<tag> sglang generate --model-path /models/Wan2.2-T2V-A14B-Diffusers \
      --num-gpus 64 --tp-size 4 --cfg-parallel-size 2 \
      --prompt "A fluffy orange cat walking gracefully across a sunny garden path, high quality, detailed" \
      --negative-prompt "orange, cat, animal" \
      --height 480 --width 832 --num-frames 81 --num-inference-steps 40 \
      --guidance-scale 5.0 --guidance-scale-2 5.0 --seed 42 \
      --output-path /out --output-file-name t2v_480p.mp4
```

## Compiled shapes

The first request at a new size compiles every graph. torch-neuronx caches the compiled graphs in
`TORCH_NEURONX_NEFF_CACHE_DIR` (default `/tmp/neff_cache`): use one persistent directory per
configuration, and give the first request a client timeout that covers the compile.

## Ring attention

At CP > 1 the DiT self-attention runs the const-max ring kernel
(`sglang_neuron/diffusion/kernels/wan/ring_attention_const_max_fwd.py`). Each rank keeps its own
K/V, and the kernel passes K/V around the CP group with `collective_permute`, overlapping the
exchange with the attention compute. The kernel has no online-max pass. It bounds each query row's
softmax maximum up front with a Cauchy–Schwarz bound, and because query rows do not rotate through
the ring, merging the ring steps is a plain sum.

No flag selects the attention path. Where the kernel cannot run (CPU tensors, fake-tensor tracing,
`SGLANG_NEURON_DISABLE_NKI_KERNELS=1`), the code all-gathers K/V and runs flash attention. Each
worker logs its choice once, at trace time:

```
WAN_RING_KERNEL=const_max ring_attention_const_max_fwd reason=eligible
WAN_RING_FALLBACK=allgather reason=...        (WARNING)
```

A run used the ring only if the workers logged the first line and never the second. With
`SGLANG_NEURON_REQUIRE_NKI_KERNELS=1` the fallback raises instead.

## Environment variables

| Variable | Effect |
|---|---|
| `TORCH_DEVICE_BACKEND_AUTOLOAD=0` | Required (Setup) |
| `SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE` | VAE patch-parallel degree (Serving) |
| `TORCH_NEURONX_NEFF_CACHE_DIR` | Compiled-graph cache (default `/tmp/neff_cache`) |
| `SGLANG_NEURON_WAN_SAVE_VIDEO_PT` | A `.pt` path. Rank 0 saves each request's decoded frames there as `[81, 3, H, W]` bf16 in [0, 1]. `{index}` in the path keeps one file per request |
| `SGLANG_NEURON_WAN_SAVE_VIDEO_PT_INDICES` | Comma list of 0-based request indices to save (unset: all). The save is synchronous, so keep it out of timed runs |
| `SGLANG_NEURON_VAE_MAX_DEVICE_GATHER_BYTES` | Size of one VAE output gather chunk (default 128 MiB). 512 MiB ran out of device memory at 1080p I2V |
| `SGLANG_NEURON_REQUIRE_NKI_KERNELS=1` | Raise when any NKI kernel, the ring kernel included, would fall back to torch |
| `SGLANG_NEURON_DISABLE_NKI_KERNELS=1` | Use the torch fallback of every NKI kernel |

Before the workers start, the platform applies `env_profiles.WAN22`
(`sglang_neuron/diffusion/env_profiles.py`). An exported value overrides any entry, with two
exceptions. The two CC DMA packet sizes are forced to 2048 with a WARNING. A launch that exports
`NEURON_CC_FLAGS` other than the profile's, or any `NEURON_COMPILER_OPT_LEVEL`, is refused:

```
TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1
NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=167772160
NEURON_SCRATCHPAD_PAGE_SIZE=2048
NEURON_RT_DBG_CC_DMA_PACKET_SIZE=2048
NEURON_RT_DBG_CC_DMA_COPY_DMA_PRIO_PKT_SIZE=2048
NEURON_LOGICAL_NC_CONFIG=2
NEURON_CC_FLAGS="-O1 --hbm-scratchpad-page-size=2048"
TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=True
TORCH_NEURONX_DEBUG_DIR=./compile_dir
TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=3600
```

Both packet-size names are set on purpose, so NRT's start-up WARN on every worker,
`NEURON_RT_DBG_CC_DMA_COPY_DMA_PRIO_PKT_SIZE is specified, ignoring NEURON_RT_DBG_CC_DMA_PACKET_SIZE`,
is expected. The 3600 s lock timeout lets the other ranks wait while one rank compiles a NEFF.

## What is refused

- A launcher that has started the Neuron runtime: the platform refuses to start and names
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0` (Setup).
- SGLang's CPU and layerwise offload, FSDP inference, `--enable-torch-compile`, the breakable CUDA
  graph, per-component placement (`--component-residency`, `--cpu-offload-components`,
  `--layerwise-offload-components`) and `--disagg-role` other than `monolithic`: the launcher
  refuses them at start, naming the flag.
- `NEURON_CC_FLAGS` other than the profile's, or any `NEURON_COMPILER_OPT_LEVEL`, in the launch
  environment.
- An I2V size whose height or width is not a multiple of 16 raises; T2V rounds it down (Serving).
- A `WanPipeline` or `WanImageToVideoPipeline` checkpoint other than Wan2.2 A14B with BF16 weights
  (Wan2.1, Wan2.2 TI2V-5B, a quantized export), at model load (Setup).
- Cache-DiT is not implemented.
- One prompt per request; several videos per prompt run one after another.
