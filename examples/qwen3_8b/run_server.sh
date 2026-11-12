#!/bin/bash
# Qwen3-8B on one Trn3 node, aggregated (prefill and decode in one server), TP4 on four NeuronCores.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/Qwen3-8B}
# 127.0.0.1: clients in this container only. For clients elsewhere, publish the port (docker run -p
# 127.0.0.1:30000:30000, or an address only they reach) and set HOST=0.0.0.0, as the container guide does.
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
# a persistent compile cache: the same path on every start, one directory per image version
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki

exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
    --tp-size 4 --context-length 4096 --chunked-prefill-size 2048 \
    --max-running-requests 64 --max-total-tokens 266240 --mem-fraction-static 0.6 \
    --host "$HOST" --port "$PORT" "$@"
