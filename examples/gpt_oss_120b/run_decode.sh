#!/bin/bash
# GPT-OSS-120B, prefill/decode on two Trn3 nodes: the DECODE server, on the decode node. It serves eight DP-attention
# replicas, each attention TP8, with MoE EP64 across them, behind a prefill of five replicas.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/gpt-oss-120b}
HOST=${HOST:?set HOST to an address of this node that the prefill node and the balancer reach}
PORT=${PORT:-30000}
# a persistent compile cache: the same path on every start, one directory per image version and node role
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki

exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
    --context-length 10240 --chunked-prefill-size 8192 --kv-cache-dtype fp8_e4m3 \
    --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 \
    --disaggregation-mode decode --tp-size 64 --dp-size 8 --ep-size 64 \
    --enable-dp-attention --enable-dp-attention-local-control-broadcast \
    --max-running-requests 128 --max-total-tokens 172032 --mem-fraction-static 0.5 --tokenizer-worker-num 4 \
    --reasoning-parser gpt-oss --tool-call-parser gpt-oss \
    --host "$HOST" --port "$PORT" "$@"
