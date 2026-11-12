#!/bin/bash
# Qwen3-0.6B, prefill/decode on two Trn3 nodes: one DECODE server, TP4 on four NeuronCores of the decode node. The validated
# layout runs five side by side, each on its own cores and port (README.md); run_balancer.sh takes the first as
# DECODE_URL and each other one as one more --decode.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/Qwen3-0.6B}
HOST=${HOST:?set HOST to an address of this node that the prefill node and the balancer reach}
PORT=${PORT:-30001}
CORES=${CORES:-0-3}
# a persistent compile cache: the same path on every start, one directory per image version and node role
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki
export NEURON_RT_VISIBLE_CORES=$CORES

exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
    --tp-size 4 --context-length 4096 --chunked-prefill-size 2048 \
    --max-running-requests 64 --max-total-tokens 266240 --mem-fraction-static 0.4 \
    --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 \
    --disaggregation-mode decode --stream-interval 4 \
    --host "$HOST" --port "$PORT" "$@"
