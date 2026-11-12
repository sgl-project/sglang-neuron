#!/bin/bash
# GSM8K's first questions, five-shot at temperature 0, through the balancer: SGLang's GSM8K client in the image. It
# prints the accuracy, the share of answers it could not read a number from, and the latency. Run it once the balancer
# is up.
set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
NUM_QUESTIONS=${NUM_QUESTIONS:-100}
# the client downloads GSM8K's test split; on a host without internet access, set GSM8K_DATA to a copy of test.jsonl
ARGS=()
if [ -n "${GSM8K_DATA:-}" ]; then
    [ -f "$GSM8K_DATA" ] || { echo "GSM8K_DATA=$GSM8K_DATA: no such file" >&2; exit 1; }
    ARGS+=(--data-path "$GSM8K_DATA")
fi
# the client needs no Neuron device: keep torch from loading the Neuron backend
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

exec python3 -m sglang.test.few_shot_gsm8k --num-questions "$NUM_QUESTIONS" --host "$HOST" --port "$PORT" \
    ${ARGS[@]+"${ARGS[@]}"}
