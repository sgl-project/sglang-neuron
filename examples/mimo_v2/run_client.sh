#!/bin/bash
# One chat request to the server, greedy.
set -euo pipefail

URL=${URL:-http://127.0.0.1:30000}
SERVED_NAME=${SERVED_NAME:-MiMo-V2.6-Flash-RL}

curl -sf "$URL/health" > /dev/null || { echo "nothing answers at $URL/health: start it first, or set URL" >&2; exit 1; }
curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$SERVED_NAME"'", "temperature": 0, "max_tokens": 256,
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}'
echo
