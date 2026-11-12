#!/bin/bash
# GLM-5.2 on Trn3: the DECODE server of the prefill/decode pair, on the decode node, inside the image. One server of
# 128 ranks: the giga-kernel decode at the 1M point (32 sequence slots, any context up to 1,048,448 tokens) with MTP.
# PP names the prefill it pairs with (4 or 16; the same on both nodes): the decode's KV pages match the prefill's.
set -euo pipefail

PP=${PP:-4}
MODEL_PATH=${MODEL_PATH:-/models/GLM-5.2-MXFP4}
WEIGHT_STORE=${WEIGHT_STORE:?set WEIGHT_STORE to the 1M weight store with the latent split of 8, see README.md}
HOST=${HOST:?set HOST to the address of this node that the other node and the balancer reach}
PORT=${PORT:-30000}
CACHE_DIR=${CACHE_DIR:-/cache}
case $PP in
  4) PAGE=1024 ;;
  16) PAGE=256 ;;
  *) echo "PP=$PP: the prefill is PP=4 or PP=16" >&2; exit 1 ;;
esac

export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
  NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki
export SGLANG_NEURON_GLM52_POINT=1m SGLANG_NEURON_GLM52_WEIGHT_STORE=$WEIGHT_STORE SGLANG_NEURON_SAMPLER=topk_topp

exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" --disaggregation-mode decode \
  --tp-size 128 --ep-size 128 --max-running-requests 32 --cuda-graph-bs-decode 32 \
  --disaggregation-decode-extra-slots 0 --context-length 1048448 --max-total-tokens 33554432 \
  --stream-interval 1 --trust-remote-code --speculative-algorithm NEURON_MTP --page-size "$PAGE" \
  --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 \
  --port "$PORT" --host "$HOST" --reasoning-parser glm45 "$@"
