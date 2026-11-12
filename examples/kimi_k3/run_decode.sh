#!/bin/bash
# Kimi-K3, prefill/decode on three Trn3 nodes: the DECODE server, one server over two nodes (TP256 as eight
# DP-attention groups of TP32, EP256, DCP32) with DSPARK speculative decoding. Run it inside the image on BOTH decode
# nodes, with NODE_RANK=0 on the first and NODE_RANK=1 on the second; node 0 serves HTTP.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/Kimi-K3}
DRAFT_PATH=${DRAFT_PATH:?set DRAFT_PATH to the DSPARK drafter checkpoint}
HOST=${HOST:?set HOST to an address of this node that the other nodes and the balancer reach}
NODE_RANK=${NODE_RANK:?set NODE_RANK to 0 on the first decode node and 1 on the second}
DIST_INIT_ADDR=${DIST_INIT_ADDR:?set DIST_INIT_ADDR to <decode node 0 address>:29500, the same on both decode nodes}
# draft tokens a step: 3 (four requests a DP group) or 7 (two requests a DP group)
K=${K:-3}
case "$K" in 3|7) ;; *) echo "K=$K: the decode programs are built for K=3 or K=7" >&2; exit 1 ;; esac
PORT=${PORT:-30000}
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki
# how long the decode waits for a request's KV; behind long cold prompts at the prefill it waits longer than
# SGLang's default, the same value as the prefill's bootstrap timeout (run_prefill.sh)
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-3600}

exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" --trust-remote-code \
    --nnodes 2 --node-rank "$NODE_RANK" --dist-init-addr "$DIST_INIT_ADDR" \
    --tp-size 256 --dp-size 8 --enable-dp-attention --ep-size 256 --dcp-size 32 \
    --disaggregation-mode decode --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 \
    --port "$PORT" --speculative-algorithm DSPARK --speculative-dspark-block-size "$K" \
    --speculative-num-draft-tokens "$((K + 1))" --speculative-draft-model-path "$DRAFT_PATH" \
    --enable-dp-lm-head --moe-a2a-backend none --kv-cache-dtype fp8_e4m3 --page-size 128 \
    --disable-radix-cache --context-length 1048576 \
    --enable-dp-attention-local-control-broadcast --scheduler-recv-interval 1 --mem-fraction-static 0.9 \
    --host "$HOST" "$@"
