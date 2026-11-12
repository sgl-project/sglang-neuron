# MiniMax-H3 on Neuron

With this plugin, SGLang Diffusion serves `MiniMaxAI/MiniMax-H3` on a trn2.48xlarge, with BF16
weights, on torch-neuronx: one video with a stereo soundtrack per request, for two tasks:

- **t2va**: the prompt alone;
- **fl2va**: the prompt and a first keyframe, a last keyframe, or both.

The model runs 8-way tensor parallel (TP8) x 8-way context parallel (CP8) on the 64 logical
NeuronCores (LNC=2); its CP group is SGLang's sequence-parallel (SP) group. The pipeline is
`sglang_neuron/diffusion/minimax_h3_pipeline.py`; the model is in
`sglang_neuron/diffusion/models/minimax_h3/`.

## The model

One SGLang stage (`NeuronMiniMaxH3Stage`) runs the whole request on every rank in lockstep
(`NeuronMiniMaxH3Model.generate`):

- the Qwen3-VL conditioner, TP-sharded: its first 50 decoder blocks (the DiT reads the output of
  block 49) and, for each keyframe, its vision tower in FP32. Rank 0 loads the keyframes and
  broadcasts them to every rank;
- each keyframe's latent, encoded by the video VAE, tiled over the ranks;
- the denoise loop over one packed sequence of text, video and audio rows, video and audio on
  their own schedules. The checkpoint is guidance-distilled: each step is one DiT graph and one
  scheduler-update graph, and the latents stay on the device. The blocks' AdaLN projections
  depend on the timestep alone, so they are computed once per schedule and reused by the
  requests that share it;
- the video VAE decode, tiled over the ranks, and the audio VAE decode of the soundtrack, in
  FP32, in windows on ranks other than rank 0.

Context-parallel self-attention uses the NKI const-max ring kernel
(`ring_attention_const_max_fwd`). The blocks whose QK-norm gains exceed its bound on the softmax
maximum (45, 48 and 49) all-gather K/V instead.

The noise is drawn on the CPU from the request's seed, so it is the same on every host. Rank 0
records `TextEncodingStage`, `ImageVAEEncodingStage`, `DenoisingStage` (with per-step times),
`DecodingStage` and `AudioDecodingStage` in SGLang's request metrics.

## Setup

Start the Neuron container and check SGLang out at `/work/sglang` as the [setup
guide](../getting-started/setup-guide.md) says, without its `setup_sglang_main_env.sh` line: the
script below runs it, and it refuses an existing venv directory. A container must see all 16 Neuron
devices. Build the SGLang Diffusion venv. The script also installs PyAV, which SGLang uses to check
each MiniMax-H3 MP4 it writes:

```bash
# from this repository's checkout under /work, as in the setup guide
NEURON_RT_VISIBLE_CORES=0 scripts/setup_sglang_diffusion_env.sh /work/venv /work/sglang
export PYTHONPATH=/work/sglang/python
export TORCH_DEVICE_BACKEND_AUTOLOAD=0  # every launch: keeps the launcher from starting the Neuron runtime
# run with /work/venv/bin/python and /work/venv/bin/sglang
```

t2va and fl2va read about 145 GB of the 498 GB checkpoint. Download those components and pass
the directory as `--model-path`:

```bash
/work/venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    "MiniMaxAI/MiniMax-H3", local_dir="/work/models/MiniMax-H3",
    allow_patterns=["model_index.json", "text_encoder/*", "tokenizer/*", "processor/*",
                    "transformer/*", "vae/*", "audio_vae/*", "scheduler/*", "audio_scheduler/*"],
)
EOF
```

A checkpoint whose `model_index.json` class is `MiniMaxH3ModularPipeline` gets the Neuron
pipeline in place of SGLang's own.

## Serving

Launch with `--num-gpus 64 --tp-size 8`: `--num-gpus` is the number of workers, one logical
NeuronCore each. SGLang sets `--sp-degree` to `num-gpus / tp-size` = 8, and the model runs CP on
that SP group.

A request is SGLang's MiniMax-H3 request, offline and over `/v1/videos` alike:

- `task`: `t2va` or `fl2va`;
- `conditions`: empty for t2va; for fl2va one or two
  `{"type": "image", "uri": ..., "role": "keyframe", "frame_index": ...}`, `frame_index` 0 for
  the first frame and -1 for the last;
- `target`: `short_edge` (768), `aspect_ratio` (`auto` takes the first keyframe's) and
  `duration_seconds` (4 to 15; below 5, the shortest length the checkpoint was released for, logs
  a warning). 768, `16:9` and 5 s give 768 x 1344 and 124 frames.

Unset, `num_inference_steps` is 50 (at least 2). The output is 24 fps.

Python (`DiffGenerator`, run with `/work/venv/bin/python`):

```python
import numpy as np
from sglang.multimodal_gen import DiffGenerator


def main():
    generator = DiffGenerator.from_pretrained(
        model_path="/work/models/MiniMax-H3", num_gpus=64, tp_size=8
    )
    result = generator.generate(sampling_params_kwargs=dict(
        prompt="A golden retriever runs along a beach at sunset", seed=42, task="t2va", conditions=[],
        target={"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
        save_output=True, output_path="out", return_file_paths_only=False,
    ))
    video = np.stack([np.asarray(frame) for frame in result.frames])  # T x H x W x 3, uint8
    audio = np.asarray(result.audio)  # 2 x samples, float32
    generator.shutdown()


if __name__ == "__main__":  # SGLang starts the workers as subprocesses
    main()
```

Server:

```bash
/work/venv/bin/sglang serve --model-type diffusion --model-path /work/models/MiniMax-H3 \
  --num-gpus 64 --tp-size 8 --warmup-mode request --port 30000
```

`--warmup-mode request` warms up on the first request: SGLang's default server warmup queries CUDA
in the launcher.

fl2va with a first and a last keyframe (`file://` URIs name files the server can read):

```bash
video_id=$(curl -s http://127.0.0.1:30000/v1/videos -H "Content-Type: application/json" -d '{
  "prompt": "A golden retriever runs along a beach at sunset", "seed": 0, "seconds": 5, "task": "fl2va",
  "conditions": [
    {"type": "image", "uri": "file:///work/first.png", "role": "keyframe", "frame_index": 0},
    {"type": "image", "uri": "file:///work/last.png", "role": "keyframe", "frame_index": -1}],
  "target": {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5.0}}' | jq -r .id)
curl -s http://127.0.0.1:30000/v1/videos/$video_id  # poll until "status" is "completed"
curl -s http://127.0.0.1:30000/v1/videos/$video_id/content -o fl2va.mp4
```

Any two images work as the keyframes (`aspect_ratio` `auto` takes the first one's). The MP4 holds
H.264 video at 24 fps and stereo AAC at 32 kHz.

### In the container image

The SGLang-Neuron container image ([docker/README.md](../../docker/README.md)) has SGLang, its
diffusion packages (PyAV included) and this plugin installed. Start it as
[docker/README.md](../../docker/README.md#run) says: the Neuron devices passed, the checkpoint and
the compile caches mounted, and `-e TORCH_DEVICE_BACKEND_AUTOLOAD=0` added. Inside it, the commands
of [Serving](#serving) run with these differences:

- No setup script, venv, SGLang checkout or `PYTHONPATH`: skip [Setup](#setup), and run `sglang`
  and `python3` in place of `/work/venv/bin/sglang` and `/work/venv/bin/python`.
- The checkpoint: the examples assume the `MiniMaxAI/MiniMax-H3` components that [Setup](#setup)
  lists are in a host directory mounted as `/models/MiniMax-H3`, its Hub repo's name, and pass that
  directory as `--model-path`.
- The `file://` URIs of fl2va's keyframes name files inside the container: mount their directory
  and change the URIs to match, for example `file:///data/first.png` and `file:///data/last.png`.
- A Python script, its inputs and its outputs must be in a mounted directory, since `--rm` removes
  the container's own files when it stops.
- `sglang serve` takes `--host 0.0.0.0` for clients outside the container
  ([docker/README.md](../../docker/README.md#run)).

The server, from the host (then send the requests above to `127.0.0.1:30000`, with the keyframes'
URIs under `/data`):

```bash
docker run --rm --init --ipc=host --cap-add SYS_NICE \
    $(for d in /dev/neuron*; do printf -- '--device %s ' "$d"; done) \
    -v <models>:/models -v <cache dir>:/cache -v <keyframes dir>:/data -p 127.0.0.1:30000:30000 \
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e TORCH_NEURONX_NEFF_CACHE_DIR=/cache/neff \
    -e TORCH_NEURONX_HLO_CACHE_DIR=/cache/hlo -e NKI_COMPILE_CACHE_URL=/cache/nki \
    sglang-neuron:<tag> sglang serve --model-type diffusion --model-path /models/MiniMax-H3 \
      --num-gpus 64 --tp-size 8 --warmup-mode request --host 0.0.0.0 --port 30000
```

## Compiled shapes

The DiT's packed sequence holds the text, keyframe, video and audio rows, so the first request at
a new canvas, frame count, set of keyframes or prompt length (in tokens) compiles its graphs.
torch-neuronx keeps the compiled graphs in `TORCH_NEURONX_NEFF_CACHE_DIR` (default
`/tmp/neff_cache`) and `TORCH_NEURONX_HLO_CACHE_DIR` (default `/tmp/hlo_cache`); keep both across
containers of the same image to skip the compile. Each cache records a digest of the NKI kernel
sources and the kernel library in its `KERNEL_SOURCES` file, and a worker refuses a cache with
another digest, or a filled cache without the file. To keep a cache that this checkout and image
filled before the file existed, record it with
`/work/venv/bin/python -m sglang_neuron.backend.compile_cache stamp <dir>`.

## What is refused

- At launch: CPU or layerwise offload, FSDP inference, torch.compile, the breakable CUDA graph and
  SGLang's per-component placement options; `--disagg-role` other than `monolithic`;
  `NEURON_CC_FLAGS` or `NEURON_COMPILER_OPT_LEVEL` exported with another value than the
  environment profile's.
- CFG parallel: the checkpoint is guidance-distilled.
- More than one prompt or output per request, fewer than 2 steps, the condition noise
  augmentation, Cache-DiT, Spectrum and rollout.

## Limitations

- The DiT graph is static in the prompt's token count: each new count compiles one more graph,
  and past Dynamo's recompile limit (8) a request at a new shape fails.
