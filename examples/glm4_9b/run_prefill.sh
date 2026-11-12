#!/bin/bash
# GLM-4-9B-0414, prefill/decode on two Trn3 nodes: the PREFILL server, TP4 on four NeuronCores of the prefill node.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/GLM-4-9B-0414}
HOST=${HOST:?set HOST to an address of this node that the decode node and the balancer reach}
PORT=${PORT:-30000}
# a persistent compile cache: the same path on every start, one directory per image version and node role
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki

exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
    --tp-size 4 --context-length 4096 --chunked-prefill-size 2048 \
    --max-running-requests 64 --max-total-tokens 266240 --mem-fraction-static 0.4 \
    --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 \
    --disaggregation-mode prefill --dp-size 1 \
    --host "$HOST" --port "$PORT" "$@"
