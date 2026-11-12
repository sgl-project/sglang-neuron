#!/bin/bash
# Wan2.2-T2V-A14B, text to video on one trn2.48xlarge: one 480p video from a prompt, written to OUTPUT_DIR. Run it
# inside the image. SGLang Diffusion starts one worker per logical NeuronCore (64), TP4 x CFG-parallel 2.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/Wan2.2-T2V-A14B-Diffusers}
OUTPUT_DIR=${OUTPUT_DIR:-/out}
PROMPT=${PROMPT:-A fluffy orange cat walking gracefully across a sunny garden path, high quality, detailed}
NEGATIVE_PROMPT=${NEGATIVE_PROMPT-orange, cat, animal}
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki
# the launcher holds no NeuronCore, so its import of torch must not load torch-neuronx; each worker loads it
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# the VAE decode's patch parallelism, the recipe's value
export SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE=${SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE:-16}

exec sglang generate --model-path "$MODEL_PATH" \
    --num-gpus 64 --tp-size 4 --cfg-parallel-size 2 \
    --prompt "$PROMPT" --negative-prompt "$NEGATIVE_PROMPT" \
    --height 480 --width 832 --num-frames 81 --num-inference-steps 40 \
    --guidance-scale 5.0 --guidance-scale-2 5.0 --seed 42 \
    --output-path "$OUTPUT_DIR" --output-file-name t2v_480p.mp4 "$@"
