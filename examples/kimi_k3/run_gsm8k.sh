#!/bin/bash
# GSM8K's first questions, five-shot, sampled at temperature 1.0 and top-p 0.95 (the model's recommended sampling),
# through the chat front: SGLang's evaluation client in the image over /v1/chat/completions. It prints the score.
# Run it once the chat front is up.
set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8021}
MODEL_PATH=${MODEL_PATH:-/models/Kimi-K3}
NUM_QUESTIONS=${NUM_QUESTIONS:-100}
# the client downloads GSM8K's test split; on a host without internet access, set GSM8K_DATA to a copy of test.jsonl
ARGS=()
if [ -n "${GSM8K_DATA:-}" ]; then
    [ -f "$GSM8K_DATA" ] || { echo "GSM8K_DATA=$GSM8K_DATA: no such file" >&2; exit 1; }
    ARGS+=(--gsm8k-data-path "$GSM8K_DATA")
fi
curl -sf "http://$HOST:$PORT/health" > /dev/null \
    || { echo "no chat front answers at http://$HOST:$PORT/health: start it first, or set HOST and PORT" >&2; exit 1; }
# the client needs no Neuron device: keep torch from loading the Neuron backend
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# 16 requests at once, as the validation sent them; thinking answers need up to 4,096 tokens
exec python3 -m sglang.test.run_eval --eval-name gsm8k --api chat --base-url "http://$HOST:$PORT" --model "$MODEL_PATH" \
    --num-examples "$NUM_QUESTIONS" --num-threads 16 --temperature 1.0 --top-p 0.95 --max-tokens 4096 \
    ${ARGS[@]+"${ARGS[@]}"}
