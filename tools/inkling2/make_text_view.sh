#!/usr/bin/env bash
# A text-only view of an Inkling-2-Mini checkpoint for SGLang: config.json is the checkpoint's text_config (model type
# inkling2_loop_text, architecture Inkling2LoopForCausalLM), so SGLang serves the language model without a multimodal
# processor; every other file is a symlink to the checkpoint's. usage: make_text_view.sh SRC DST
set -euo pipefail
SRC=$(cd "$1" && pwd); DST=$2
mkdir -p "$DST"
DST=$(cd "$DST" && pwd)
# relative links, so the view resolves wherever both directories are mounted together (a container's /models)
for f in "$SRC"/*; do
  [ "$(basename "$f")" = config.json ] || ln -sfn "$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$f" "$DST")" "$DST/$(basename "$f")"
done
python3 - "$SRC/config.json" "$DST/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
t = dict(c["text_config"])
t.update(architectures=["Inkling2LoopForCausalLM"], model_type="inkling2_loop_text")
for k in ("eos_token_id", "dtype"):
    if k in c:
        t.setdefault(k, c[k])
json.dump(t, open(sys.argv[2], "w"), indent=2)
PY
echo "text view at $DST"
