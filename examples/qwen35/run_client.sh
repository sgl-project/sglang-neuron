#!/bin/bash
# One greedy request through the balancer, once both servers and the balancer are up.
set -euo pipefail

URL=${URL:-http://127.0.0.1:8000}

curl -sf "$URL/health" > /dev/null || { echo "no balancer answers at $URL/health: start it first, or set URL" >&2; exit 1; }
echo "balancer: healthy"
curl -s "$URL/generate" -H 'Content-Type: application/json' -d '{
  "text": "Question: What is 17 * 23?\nAnswer:", "sampling_params": {"temperature": 0, "max_new_tokens": 32}}'
echo
