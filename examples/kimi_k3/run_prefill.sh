#!/bin/bash
# Kimi-K3, prefill/decode on three Trn3 nodes: the PREFILL server, eight pipeline stages of TP16 on one node.
# Run it inside the image on the prefill node.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/Kimi-K3}
DRAFT_PATH=${DRAFT_PATH:?set DRAFT_PATH to the DSPARK drafter checkpoint the decode drafts with}
HOST=${HOST:?set HOST to an address of this node that the decode nodes and the balancer reach}
PORT=${PORT:-30000}
# a persistent compile cache: the same path on every start, one directory per image version and node role
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki
# how long the prefill keeps a request for the decode to bootstrap it: the decode's KV wait (run_decode.sh), so the
# prefill does not fail a request the decode still waits for
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-3600}
# the one layer split the stages serve, and the prefix cache's host tier (about 1,080 GB of host memory)
export SGLANG_PP_LAYER_PARTITION=12,12,12,12,12,12,12,9
export SGLANG_NEURON_KIMI_PROMPT_HOST_SLOTS=${SGLANG_NEURON_KIMI_PROMPT_HOST_SLOTS:-18000}
export SGLANG_NEURON_KIMI_PROMPT_HOST_STATE_SLOTS=${SGLANG_NEURON_KIMI_PROMPT_HOST_STATE_SLOTS:-2000}

exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" --trust-remote-code --skip-tokenizer-init \
    --disaggregation-mode prefill --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 \
    --port "$PORT" --pp-size 8 --tp-size 16 --dcp-size 16 --page-size 128 --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size 4096 --mamba-ssm-dtype bfloat16 --radix-cache-backend kimi-prompt \
    --max-mamba-cache-size 512 --max-total-tokens 524288 --max-running-requests 32 --mem-fraction-static 0.85 \
    --speculative-draft-model-path "$DRAFT_PATH" \
    --host "$HOST" "$@"
