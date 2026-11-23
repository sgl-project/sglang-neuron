#!/bin/bash
# Qwen3.8 27B (text, bf16) on one trn2.3xlarge, aggregated (prefill and decode in one server), TP8: one rank per
# physical NeuronCore (the catalog entry runs the kernels at LNC1).
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/Qwen3.8-27B}
# 127.0.0.1: clients on this host only; give an address of the node for clients elsewhere, reachable only by them
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
# a persistent compile cache: the same path on every start, one directory per image version
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki

# the prefix cache on the chunk-keyed tree, the GDN state kept at each 2048-token chunk's end; RADIX=0 serves
# without it (one less host step per decode step, no reuse of a repeated prefix)
if [ "${RADIX:-1}" = 1 ]; then
    CACHE=(--radix-cache-backend neuron-chunk --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 2048)
else
    CACHE=(--disable-radix-cache)
fi
# the prefill graphs below the 2048-token chunk: a prompt's last piece runs the smallest that holds it. 512 fits beside
# this server's pools with the prefix cache on; the full ladder (512 1024) does not with the float32 GDN state (the
# recipe's "Prefill tiles"). PREFILL_TILES=2048 compiles the chunk alone.
PREFILL_TILES=${PREFILL_TILES:-512}
# paged KV in 512-token pages, prompts prefilled in 2048-token chunks of whole pages
exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
    --tp-size 8 --context-length 16384 --page-size 512 --chunked-prefill-size 2048 \
    --cuda-graph-bs-prefill $PREFILL_TILES \
    --max-running-requests 4 --max-total-tokens 65536 --mem-fraction-static 0.85 \
    "${CACHE[@]}" --dtype bfloat16 \
    --host "$HOST" --port "$PORT" "$@"
