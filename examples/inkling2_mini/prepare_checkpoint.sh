#!/bin/bash
# The text view the server loads: the checkpoint's text_config as config.json (architecture Inkling2LoopForCausalLM),
# every other file a relative symlink to the checkpoint's. Run it once, where both paths are mounted.
set -euo pipefail

SRC=${SRC:-/models/Inkling-2-Mini}
DST=${DST:-/models/Inkling-2-Mini-text}
exec bash "$(dirname "$0")/../../tools/inkling2/make_text_view.sh" "$SRC" "$DST"
