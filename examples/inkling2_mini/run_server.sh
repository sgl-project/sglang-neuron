#!/bin/bash
# Inkling-2-Mini on one trn2.3xlarge, aggregated, TP8 (one rank per physical NeuronCore), with the radix cache. The
# model path is the checkpoint's text view (prepare_checkpoint.sh).
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/Inkling-2-Mini-text}
# 127.0.0.1: clients in this container only. For clients elsewhere, publish the port (docker run -p
# 127.0.0.1:30000:30000, or an address only they reach) and set HOST=0.0.0.0, as the container guide does.
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki

# 256-token pages and chunks (the conv-state record's unit), one decode graph of 2 rows, a 16,384-token KV pool: the
# prefill graph's scratchpad needs the rest of each core's HBM beside the weights (docs/model-recipes/inkling2.md)
exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" --served-model-name inkling \
    --tp-size 8 --context-length 16384 --page-size 256 --chunked-prefill-size 256 \
    --max-running-requests 2 --cuda-graph-bs-decode 2 --max-total-tokens 16384 --mem-fraction-static 0.85 \
    --dtype bfloat16 --tool-call-parser inkling --reasoning-parser inkling \
    --host "$HOST" --port "$PORT" "$@"
