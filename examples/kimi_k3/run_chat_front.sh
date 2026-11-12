#!/bin/bash
# The chat-completions front in front of the balancer. The prefill takes token ids only (--skip-tokenizer-init), so
# this front applies the checkpoint's chat template, sends the ids to the balancer's /generate and returns an OpenAI
# chat completion with the reasoning split from the answer. Run it on the balancer's host once the balancer is up.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/Kimi-K3}
UPSTREAM=${UPSTREAM:-http://127.0.0.1:8000}
PORT=${PORT:-8021}
# the front needs no Neuron device: keep torch from loading the Neuron backend
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# listens on 127.0.0.1 only
exec python3 "$(dirname "$0")/chat_front.py" --upstream "$UPSTREAM" --model "$MODEL_PATH" --port "$PORT" \
    ${LOG:+--log "$LOG"}
