#!/bin/bash
# MiMo-V2.6-Flash-RL on one trn2.48xlarge, aggregated (prefill and decode in one server): TP64 / EP64 on the 64
# logical NeuronCores, the text decoder only.
#
#   CONTEXT=8192   bash run_server.sh    # bring-up: the smallest graphs, the quickest compile
#   CONTEXT=262144 bash run_server.sh    # the AgentX 256K point
#   CONTEXT=1048576 bash run_server.sh   # the AgentX 1M point (see the recipe's capacity table first)
#
# The radix cache keeps each session's prefix across turns; SGLANG_NEURON_HYBRID_SWA_RADIX=1 keeps the 39 sliding
# layers' KV in SGLang's window pool under it, so only the 9 full-attention layers hold the whole context (without it
# every layer would, 48 / 9 times the KV).
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/MiMo-V2.6-Flash-RL}
SERVED_NAME=${SERVED_NAME:-MiMo-V2.6-Flash-RL}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
CONTEXT=${CONTEXT:-262144}
# the segmented prefill tiles go up to 8192 tokens (utils/bucket_utils.SUPPORTED_KV_SEGMENT_SIZES)
CHUNK=${CHUNK:-8192}
MAX_RUNNING=${MAX_RUNNING:-16}
MEM_FRACTION=${MEM_FRACTION:-0.85}
# the window pool's tokens per full-pool token. SGLang's default, 0.8, spends most of the KV on the 39 sliding layers'
# window pool (a full-pool token is 9 layers x 1 KiB a rank, a window token 39 x 1 KiB): at 0.8 a rank's 8 GB holds
# about 195K tokens of context, at 0.06 about 700K. The running requests need their windows and a prefill chunk in it;
# at step 0 the plugin refuses a ratio below that and names the one that fits (mem_cache/hybrid_swa.check_window_pool).
SWA_RATIO=${SWA_RATIO:-0.06}
# a persistent compile cache: the same path on every start, one directory per image version
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki
export SGLANG_NEURON_HYBRID_SWA_RADIX=${SGLANG_NEURON_HYBRID_SWA_RADIX:-1}
# the device sampler draws with top-k / top-p: a request that names no sampling takes the checkpoint's
# generation_config (temperature 1.0, top_p 0.95), as AgentX's requests do, which the plugin's default greedy sampler
# refuses with a 400
export SGLANG_NEURON_SAMPLER=${SGLANG_NEURON_SAMPLER:-topk_topp}
# past 256K the decode graphs' IO sets are large enough that the runtime's default ring of 32 of them a program leaves
# no HBM for the execution's DMA rings (measured at 1M: "Failed to allocate DEVICE memory" on the first steps); a ring
# of 4, as GLM-5.2's 1M decode, fits
if [ "$CONTEXT" -gt 262144 ]; then
    export NEURON_RT_IO_RING_CACHE_SIZE=${NEURON_RT_IO_RING_CACHE_SIZE:-4}
fi

exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" --served-model-name "$SERVED_NAME" \
    --trust-remote-code --tp-size 64 --ep-size 64 \
    --context-length "$CONTEXT" --chunked-prefill-size "$CHUNK" --max-running-requests "$MAX_RUNNING" \
    --mem-fraction-static "$MEM_FRACTION" --swa-full-tokens-ratio "$SWA_RATIO" \
    --reasoning-parser mimo --tool-call-parser mimo \
    --host "$HOST" --port "$PORT" "$@"
