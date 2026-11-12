#!/bin/bash
# GLM-5.2 prefill/decode pair: one chat request through the balancer; prints the answer.
set -euo pipefail

URL=${URL:-http://127.0.0.1:8000}
MODEL_PATH=${MODEL_PATH:-/models/GLM-5.2-MXFP4}

curl -sf "$URL/health" > /dev/null || {
  echo "nothing answers at $URL/health: start the servers and the balancer first" >&2; exit 1; }
curl -sS "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL_PATH"'", "temperature": 0, "max_tokens": 1024,
  "messages": [{"role": "user", "content": "What is 17 times 23? Reply with the number only."}]}' \
  | python3 -c 'import json, sys
body = json.load(sys.stdin)
if "choices" not in body:
    sys.exit(f"no answer: {body}")
print(body["choices"][0]["message"]["content"])'
