#!/bin/bash
# One chat request, through the balancer by default, or to the one-node server with URL=http://127.0.0.1:30000.
set -euo pipefail

URL=${URL:-http://127.0.0.1:8000}
MODEL_PATH=${MODEL_PATH:-/models/Qwen3-8B}

curl -sf "$URL/health" > /dev/null || { echo "nothing answers at $URL/health: start it first, or set URL" >&2; exit 1; }

# Qwen3 thinks before it answers unless the chat template is told not to
curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL_PATH"'", "temperature": 0, "max_tokens": 64,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}'
echo
