#!/bin/bash
# Build a Python environment that runs SGLang Diffusion (the diffusion models of docs/model-recipes/)
# with this plugin on the Neuron container's torch / torch-neuronx: the SGLang main environment
# (scripts/setup_sglang_main_env.sh), plus the packages SGLang Diffusion and the diffusion pipelines
# import. See docs/model-recipes/flux1-dev.md.
#
# Usage (inside the serving container, with the Python that has torch-neuronx):
#   NEURON_RT_VISIBLE_CORES=<an assigned core> \
#     scripts/setup_sglang_diffusion_env.sh [--server-tests] <venv_dir> <sglang_checkout>
# Then launch with <venv_dir>/bin/python and PYTHONPATH=<sglang_checkout>/python, as the
# SGLang main environment does.
#
#   --server-tests  also install pytest, its plugins and the packages SGLang's diffusion server-test
#                   harness imports (scikit-image, opencv). scikit-image needs networkx >= 3, and
#                   the container's neuronx-cc declares networkx ~= 2.6, so pip installs networkx 3
#                   into this venv and `pip check` reports neuronx-cc. Use a separate venv for the
#                   server tests.
set -euo pipefail
EXTRA=none
case "${1:-}" in
  --server-tests) EXTRA=server-tests; shift ;;
esac
VENV=${1:?venv directory (new)}
SGLANG=${2:?SGLang checkout}
HERE=$(cd "$(dirname "$0")" && pwd)

"$HERE/setup_sglang_main_env.sh" "$VENV" "$SGLANG"

# torch stays the container's (the same pin as the SGLang main environment).
PIN=$(mktemp)
"$VENV/bin/python" -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > "$PIN"
export PIP_CONSTRAINT=

# Imported by SGLang Diffusion: docker/diffusion-requirements.txt, the one list of these packages
# and their versions, which the image installs too (through its lock).
"$VENV/bin/pip" install -q -c "$PIN" -r "$HERE/../docker/diffusion-requirements.txt"

# The `sglang` command (`sglang serve`, `sglang generate`, which SGLang's CLI and server tests run)
# exists only when the checkout is installed. --no-deps keeps the packages above; SGLang's build
# requirements pin torch==2.14.1, which the container's own pip constraint rejects, so build
# against the venv's torch (--no-build-isolation, with SGLang's version tool installed here).
# SGLANG_BUILD_RUST_EXTS=none: no Rust extensions (the container has no cargo).
"$VENV/bin/pip" install -q -c "$PIN" setuptools-scm==10.3.4 vcs-versioning==2.6.0
SGLANG_BUILD_RUST_EXTS=none "$VENV/bin/pip" install -q --no-deps --no-build-isolation \
  -e "$SGLANG/python"

if [ "$EXTRA" = server-tests ]; then
  # The container has no pytest; SSIM / PSNR (torchmetrics), per-test timeouts (pytest-timeout),
  # and boto3 for optional S3 checkpoint and reference paths.
  "$VENV/bin/pip" install -q -c "$PIN" pytest==9.2.0 pytest-timeout==2.4.0 torchmetrics==1.9.0 \
    boto3==1.43.123
  # SGLang's server-test helpers (multimodal_gen/test/test_utils.py) import cv2 and skimage.
  "$VENV/bin/pip" install -q -c "$PIN" scikit-image==0.26.0 opencv-python-headless==5.0.0.93
fi
rm -f "$PIN"
