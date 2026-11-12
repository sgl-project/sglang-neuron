#!/bin/bash
# Build a Python environment that runs SGLang main from a source checkout on top of
# the Neuron container's torch / torch-neuronx, without modifying them.
# See docs/getting-started/setup-guide.md.
#
# Usage (inside the serving container, with the Python that has torch-neuronx):
#   NEURON_RT_VISIBLE_CORES=<an assigned core> \
#     scripts/setup_sglang_main_env.sh <venv_dir> <sglang_checkout>
# Then launch with <venv_dir>/bin/python and PYTHONPATH=<sglang_checkout>/python
# (the plugin is installed into the venv in editable mode).
set -euo pipefail
VENV=${1:?venv directory (new)}
SGLANG=${2:?SGLang checkout}
PLUGIN=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
# Importing torch auto-loads torch_neuronx; claim only an assigned core.
: "${NEURON_RT_VISIBLE_CORES:?set an assigned NEURON_RT_VISIBLE_CORES before importing Neuron}"

if [ -e "$VENV" ]; then
  echo "refusing to reuse existing $VENV; use a new directory" >&2
  exit 1
fi

# --system-site-packages keeps the image's torch, torch-neuronx, neuronx-cc, nki
# and nkilib visible; the constraint stops pip from replacing torch to satisfy
# SGLang main's torch==2.14.1 pin.
"$PY" -m venv --system-site-packages "$VENV"
PIN=$(mktemp)
REQS=$(mktemp)
trap 'rm -f "$PIN" "$REQS"' EXIT
"$PY" -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > "$PIN"
export PIP_CONSTRAINT=

# SGLang's runtime set at the pins in its pyproject_other.toml, the ones SGLang's own srt_empty
# and runtime_common installs use, read from the checkout without changing it
# (tools/sglang_runtime_requirements.py); only torch is held, at the image's build. On a node
# without PyPI, point pip at a wheelhouse: PIP_NO_INDEX=1 PIP_FIND_LINKS=<dir>.
"$PY" "$PLUGIN/tools/sglang_runtime_requirements.py" "$SGLANG/python" > "$REQS"
"$VENV/bin/pip" install -q -c "$PIN" -r "$REQS"

# SGLang imports torchvision unconditionally (multimodal utils). The CPU wheel
# matching the image's torch, without letting pip touch torch.
TV=$("$PY" - <<'EOF'
import torch
major, minor = (int(x) for x in torch.__version__.split(".")[:2])
print(f"torchvision==0.{minor + 15}.0" if major == 2 else "")
EOF
)
if ! "$VENV/bin/python" -c "import torchvision" 2>/dev/null; then
  "$VENV/bin/pip" install -q --no-deps "$TV" --index-url https://download.pytorch.org/whl/cpu
fi

# SGLang discovers the platform and hook plugins through entry points, so the
# plugin itself must be installed (editable; it declares only `sglang`, which
# runs from the source checkout).
"$VENV/bin/pip" install -q --no-deps -e "$PLUGIN"

"$VENV/bin/python" -c "import torch, importlib.metadata as m; print('torch', torch.__version__, 'torch-neuronx', m.version('torch-neuronx'))"
echo "--- requirements still unmet (the expected ones are listed in docs/getting-started/setup-guide.md):"
"$VENV/bin/python" "$PLUGIN/tools/check_sglang_deps.py" "$SGLANG/python/pyproject.toml"
echo "--- serving entry-point imports:"
PYTHONPATH="$SGLANG/python" "$VENV/bin/python" "$PLUGIN/tools/import_sglang_entrypoints.py"
