#!/bin/bash
# One chat request to the server; the reasoning parser returns the answer in "content" and the reasoning beside it.
set -euo pipefail

URL=${URL:-http://127.0.0.1:30000}
curl -sf "$URL/health" > /dev/null || { echo "nothing answers at $URL/health: start it first, or set URL" >&2; exit 1; }
curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "inkling", "temperature": 0, "max_tokens": 512,
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}'
echo
