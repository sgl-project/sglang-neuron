#!/bin/bash
# Gemma 4 31B (dense, text) on one trn2.3xlarge, aggregated (prefill and decode in one server), TP4 on four NeuronCores.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/gemma-4-31B-it}
# 127.0.0.1: clients on this host only; give an address of the node for clients elsewhere, reachable only by them
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
# a persistent compile cache: the same path on every start, one directory per image version
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki

# one uniform KV pool: --disable-hybrid-swa-memory (the model's launch rules refuse a launch without it)
exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
    --tp-size 4 --context-length 16384 --chunked-prefill-size 512 \
    --max-running-requests 8 --max-total-tokens 20480 --mem-fraction-static 0.9 \
    --disable-hybrid-swa-memory \
    --host "$HOST" --port "$PORT" "$@"
