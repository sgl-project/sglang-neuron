#!/bin/bash
# MiMo-V2.6-Flash-RL disaggregated on ONE trn2.48xlarge: a TP32 / EP32 prefill server on logical NeuronCores 0-31 and a
# TP32 / EP32 decode server on 32-63, the KV over NIXL, and the plugin's balancer in front on $PORT.
#
#   bash run_pd_one_node.sh            # starts all three, logs in $LOG_DIR
#   URL=http://127.0.0.1:30000 bash run_client.sh
#
# A bring-up of the PD path on the one node there is to test on (the runtime does not refuse a prefill and a decode on
# disjoint cores of a node, and does not test it either: docs/design/disaggregated-inference.md). On a PD pair:
# - one KV pool for every layer (the window pool is not served on a PD role): hybrid SWA memory and its radix mode off;
# - each half holds 8 of the 256 experts a rank (9.5 GB of FP8), so the pools are about half the aggregated server's.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/MiMo-V2.6-Flash-RL}
SERVED_NAME=${SERVED_NAME:-MiMo-V2.6-Flash-RL}
PORT=${PORT:-30000}
PREFILL_PORT=${PREFILL_PORT:-30001}
DECODE_PORT=${DECODE_PORT:-30002}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
CONTEXT=${CONTEXT:-65536}
CHUNK=${CHUNK:-8192}
MAX_RUNNING=${MAX_RUNNING:-16}
MEM_FRACTION=${MEM_FRACTION:-0.85}
CACHE_DIR=${CACHE_DIR:-/cache}
LOG_DIR=${LOG_DIR:-$CACHE_DIR}
mkdir -p "$LOG_DIR"
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki
export SGLANG_NEURON_HYBRID_SWA_RADIX=0
export SGLANG_NEURON_SAMPLER=${SGLANG_NEURON_SAMPLER:-topk_topp}

COMMON=(--model-path "$MODEL_PATH" --served-model-name "$SERVED_NAME" --trust-remote-code --tp-size 32 --ep-size 32
        --context-length "$CONTEXT" --chunked-prefill-size "$CHUNK" --max-running-requests "$MAX_RUNNING"
        --mem-fraction-static "$MEM_FRACTION" --disable-hybrid-swa-memory
        --disaggregation-transfer-backend nixl --reasoning-parser mimo --tool-call-parser mimo --host 127.0.0.1)

NEURON_RT_VISIBLE_CORES=0-31 nohup python3 -m sglang.launch_server "${COMMON[@]}" \
    --disaggregation-mode prefill --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" --port "$PREFILL_PORT" \
    > "$LOG_DIR/mimo_pd_prefill.log" 2>&1 &
NEURON_RT_VISIBLE_CORES=32-63 nohup python3 -m sglang.launch_server "${COMMON[@]}" \
    --disaggregation-mode decode --port "$DECODE_PORT" \
    > "$LOG_DIR/mimo_pd_decode.log" 2>&1 &
nohup python3 -m sglang_neuron.disaggregation.balancer --prefill "http://127.0.0.1:$PREFILL_PORT" \
    --decode "http://127.0.0.1:$DECODE_PORT" --bootstrap-port "$BOOTSTRAP_PORT" --port "$PORT" > "$LOG_DIR/mimo_pd_balancer.log" 2>&1 &
echo "prefill :$PREFILL_PORT (cores 0-31), decode :$DECODE_PORT (cores 32-63), balancer :$PORT; logs in $LOG_DIR"
wait
