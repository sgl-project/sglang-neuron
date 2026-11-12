#!/bin/bash
# GPT-OSS-120B, prefill/decode on two Trn3 nodes: the PREFILL server, on the prefill node. It serves REPLICAS
# replicas, each attention TP4 with MoE EP2: 5 in front of run_decode.sh, 6 in front of the giga-kernel decode
# (run_decode_giga.sh).
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/gpt-oss-120b}
HOST=${HOST:?set HOST to an address of this node that the decode node and the balancer reach}
PORT=${PORT:-30000}
REPLICAS=${REPLICAS:-5}
# a persistent compile cache: the same path on every start, one directory per image version and node role
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki

exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
    --context-length 10240 --chunked-prefill-size 8192 --kv-cache-dtype fp8_e4m3 \
    --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 \
    --disaggregation-mode prefill --tp-size 4 --ep-size 2 --dp-size "$REPLICAS" \
    --max-running-requests 8 --max-total-tokens 65536 --mem-fraction-static 0.51 \
    --reasoning-parser gpt-oss --tool-call-parser gpt-oss \
    --host "$HOST" --port "$PORT" "$@"
