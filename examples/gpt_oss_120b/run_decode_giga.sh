#!/bin/bash
# GPT-OSS-120B, prefill/decode on two Trn3 nodes: the giga-kernel DECODE server, on the decode node, in place of
# run_decode.sh, behind a prefill of six replicas (REPLICAS=6 run_prefill.sh). It serves 128 ranks, one on each of the
# node's 128 physical NeuronCores (the platform runs this server at one logical core per physical core).
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
    --disaggregation-mode decode --tp-size 128 --dp-size 16 --ep-size 128 \
    --enable-dp-attention --enable-dp-attention-local-control-broadcast \
    --max-running-requests 128 --max-total-tokens 163840 --mem-fraction-static 0.5 --stream-interval 4 \
    --tokenizer-worker-num 4 --detokenizer-worker-num 4 \
    --reasoning-parser gpt-oss --tool-call-parser gpt-oss \
    --host "$HOST" --port "$PORT" "$@"
