#!/bin/bash
# GLM-5.2 on Trn3: the PREFILL server of the prefill/decode pair, on the prefill node, inside the image. One pipelined
# server over the 78 layers: PP=4 (the default) is 4 stages of 32 ranks, PP=16 is 16 stages of 8 ranks. Give the
# decode the same PP. It sends each prompt's KV cache and the rows of the decode's MTP draft to the decode over EFA.
set -euo pipefail

PP=${PP:-4}
MODEL_PATH=${MODEL_PATH:-/models/GLM-5.2-MXFP4}
HOST=${HOST:?set HOST to the address of this node that the other node and the balancer reach}
PORT=${PORT:-30000}
CACHE_DIR=${CACHE_DIR:-/cache}
case $PP in
  4) SPLIT=18,20,20,20; LAYOUT=(--pp-size 4 --tp-size 32 --ep-size 32 --chunked-prefill-size 4096
                                --cuda-graph-bs-prefill 1024 2048); RUNNING=16 ;;
  16) SPLIT=3,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5; LAYOUT=(--pp-size 16 --tp-size 8 --ep-size 8 --chunked-prefill-size 2048
                                                    --cuda-graph-bs-prefill 1024); RUNNING=48 ;;
  *) echo "PP=$PP: the prefill is PP=4 or PP=16" >&2; exit 1 ;;
esac

export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
  NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki
export SGLANG_NEURON_GLM52_POINT=1m SGLANG_NEURON_GLM52_PD_MTP=1 SGLANG_NEURON_SAMPLER=topk_topp \
  SGLANG_PP_LAYER_PARTITION=$SPLIT

exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" --disaggregation-mode prefill "${LAYOUT[@]}" \
  --context-length 1048448 --max-running-requests "$RUNNING" --max-total-tokens 12582912 \
  --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 \
  --port "$PORT" --host "$HOST" --reasoning-parser glm45 "$@"
