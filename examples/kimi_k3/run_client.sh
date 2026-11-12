#!/bin/bash
# One chat request through the chat front, once the servers, the balancer and the front are up.
set -euo pipefail

URL=${URL:-http://127.0.0.1:8021}
MODEL_PATH=${MODEL_PATH:-/models/Kimi-K3}

curl -sf "$URL/health" > /dev/null || { echo "no chat front answers at $URL/health: start it first, or set URL" >&2; exit 1; }
echo "chat front: healthy"

curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL_PATH"'", "temperature": 1.0, "top_p": 0.95, "max_tokens": 2048, "seed": 1,
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}'
echo
