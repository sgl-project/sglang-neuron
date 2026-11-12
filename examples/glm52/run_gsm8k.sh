#!/bin/bash
# GLM-5.2 prefill/decode pair: GSM8K through the balancer, with SGLang's few-shot client. It takes no Neuron device.
set -euo pipefail

HOST=${HOST:-127.0.0.1}                 # the balancer
PORT=${PORT:-8000}
NUM_QUESTIONS=${NUM_QUESTIONS:-100}
# GSM8K_DATA: a local copy of grade_school_math/data/test.jsonl (github.com/openai/grade-school-math); without it,
# the client downloads that file
ARGS=()
if [ -n "${GSM8K_DATA:-}" ]; then
    [ -f "$GSM8K_DATA" ] || { echo "GSM8K_DATA=$GSM8K_DATA: no such file" >&2; exit 1; }
    ARGS+=(--data-path "$GSM8K_DATA")
fi
ARGS+=(--parallel 32)                   # at most the decode's 32 sequence slots
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec python3 -m sglang.test.few_shot_gsm8k --num-questions "$NUM_QUESTIONS" --host "$HOST" --port "$PORT" \
    ${ARGS[@]+"${ARGS[@]}"} "$@"
