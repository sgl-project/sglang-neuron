#!/bin/bash
# One chat request through the balancer (or, with URL=http://127.0.0.1:30000, to the one-node server), once it is up.
set -euo pipefail

URL=${URL:-http://127.0.0.1:8000}
MODEL_PATH=${MODEL_PATH:-/models/DeepSeek-V2-Lite}

curl -sf "$URL/health" > /dev/null || { echo "nothing answers at $URL/health: start it first, or set URL" >&2; exit 1; }

curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL_PATH"'", "temperature": 0, "max_tokens": 256,
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}'
echo
