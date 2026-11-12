#!/bin/bash
# One chat request and one tool call through the balancer, once both servers and the balancer are up.
set -euo pipefail

URL=${URL:-http://127.0.0.1:8000}
MODEL_PATH=${MODEL_PATH:-/models/gpt-oss-120b}

curl -sf "$URL/health" > /dev/null || { echo "nothing answers at $URL/health: start it first, or set URL" >&2; exit 1; }

curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL_PATH"'", "temperature": 0, "max_tokens": 256,
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}'
echo

curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL_PATH"'", "temperature": 0, "max_tokens": 256,
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}'
echo
