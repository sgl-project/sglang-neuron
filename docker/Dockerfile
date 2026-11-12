# The SGLang-Neuron container images, built by docker/build.sh (public, then qa, then the gate; docker/README.md):
#
#   docker/build.sh
#
# Its context is a `git archive` of HEAD (files 0644, scripts and directories 0755), so the images hold the commit's
# files only. A `docker build -f docker/Dockerfile --target public .` from a working tree would take its uncommitted and
# untracked files too (the dockerignore leaves out only git metadata and build output) with their modes as they are,
# and skip the gate. `public` is the image that serves:
# the Neuron SDK release container plus SGLang, its runtime dependencies, the plugin and the patched NIXL. `qa` is
# `public` plus this checkout and the QA tools, so QA runs against the exact shipped layers, and nothing ever needs
# stripping or flattening.
# Inputs from the context are bind-mounted into the steps that read them, and every pip step writes its cache to a
# tmpfs, so no build input or leftover lands in a layer.

# The Neuron SDK 2.33.0 release container, by digest (sglang_neuron/release/neuron_sdk.toml [image]). It carries the
# pinned neuronx-cc, nki, runtime, collectives and tools, torch 2.14.0+cu130 and torch-neuronx 2.14.3.0.1+b1b6c66 (the
# 2.14.3 line neuron_sdk.toml enforces). Every stage starts from it, so the build tools and the image see one Python,
# one torch and one glibc. Its torch-neuronx is an editable install from /workspace/torch_neuronx, so a volume mounted at
# /workspace hides it (docker/README.md, "Known limitations").
ARG BASE=public.ecr.aws/neuron/torch-neuronx@sha256:b4648437b8c7dbfa2acde18301a4bb4ae9658da5a412da75c59693ed4fd308ec
# The SGLang revision the plugin's SGLang patches are written against and md5-pinned to (README, setup-guide.md), and
# the version its wheel gets: the one SGLang's own versioning gives this commit (its setuptools-scm describe command:
# the newest release tag, v0.5.23, plus the commit's distance from it, 413), which a depth-1 fetch cannot compute.
# Vulnerability scanners compare it with the fixed versions of SGLang's advisories, so it must not be a placeholder.
ARG SGLANG_COMMIT=43881a719e42165ff987fb0273a6bf12eaeb7662
ARG SGLANG_VERSION=0.5.24.dev413+g43881a7
# The plugin commit the context is (the image's revision label); the build script passes `git rev-parse HEAD`.
ARG SGLANG_NEURON_COMMIT=unknown
# When the image was built (the created label; the base's own label is the Ubuntu base's date).
ARG BUILD_DATE=unknown
# SGLang imports torchvision: the CPU build of the release that matches torch 2.14, from PyTorch's CPU index
# (docker/lock_requirements.sh pins it by sha256 in docker/torchvision-lock.txt).
ARG TORCHVISION_VERSION=0.29.0
ARG TORCHVISION_INDEX=https://download.pytorch.org/whl/cpu


# ---- build stage: the patched NIXL 1.5.0 (patches/nixl + tools/build_nixl.sh) ----
FROM ${BASE} AS nixl-build
WORKDIR /build
# The build tools and the nixl 1.5.0 meta wheel at the sha256 pins build_nixl.sh carries (read from the script, so the
# hashes live in one place). The base already has the compilers, binutils, patchelf, git and the EFA headers the build
# needs. The script checks the patched source tree, the three patch markers in the plugin and that nothing links etcd.
# Only the script and the patch series are mounted, so an edit elsewhere in the repository does not rebuild NIXL.
RUN --mount=type=bind,source=tools/build_nixl.sh,target=/build/tools/build_nixl.sh \
    --mount=type=bind,source=patches/nixl,target=/build/patches/nixl \
    sed -n '/^TOOL_WHEELS="/,/"$/p;/^META_WHEEL=/p' tools/build_nixl.sh | sed 's/^[A-Z_]*="//; s/"$//' > nixl_wheels.txt \
 && python3 -m pip download --quiet --no-deps --require-hashes --only-binary=:all: -r nixl_wheels.txt -d /build/wheels \
 && mkdir -p /opt/nixl \
 && NIXL_WHEELS=/build/wheels bash tools/build_nixl.sh /opt/nixl/overlay_cu13


# ---- build stage: SGLang's wheel at the pinned commit, and its runtime requirement list ----
FROM ${BASE} AS sglang-wheel
ARG SGLANG_COMMIT
ARG SGLANG_VERSION
# The wheels' build tools in a throwaway venv: the base's Python is not touched, and they never reach the image.
RUN --mount=type=bind,source=docker/wheel-build-lock.txt,target=/tmp/wheel-build-lock.txt \
    --mount=type=tmpfs,target=/root/.cache \
    python3 -m venv /opt/buildenv \
 && /opt/buildenv/bin/pip install --quiet --no-cache-dir --no-deps --require-hashes -r /tmp/wheel-build-lock.txt
# SGLang at the one commit, unmodified, fetched alone (the test refuses another revision).
RUN git init --quiet /src/sglang \
 && git -C /src/sglang fetch --quiet --depth 1 https://github.com/sgl-project/sglang ${SGLANG_COMMIT} \
 && git -C /src/sglang checkout --quiet --detach FETCH_HEAD \
 && test "$(git -C /src/sglang rev-parse HEAD)" = "${SGLANG_COMMIT}"
# As SGLang's release workflow builds it (README and LICENSE beside pyproject.toml), with no Rust extensions (the
# plugin runs SGLang without them), the version given explicitly, and no build isolation (the CUDA pyproject's build
# requirements name torch==2.14.1, which only its Rust TreeCore build uses).
RUN cd /src/sglang/python && cp ../README.md ../LICENSE . \
 && SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SGLANG_VERSION} \
    /opt/buildenv/bin/pip wheel --quiet --no-deps --no-build-isolation -w /wheels .
# SGLang's runtime set at SGLang's own pins (pyproject_other.toml's runtime_common, less the optional serving modes),
# the list scripts/setup_sglang_main_env.sh installs; the tool uses the standard library only.
RUN --mount=type=bind,source=tools/sglang_runtime_requirements.py,target=/tmp/sglang_runtime_requirements.py \
    python3 /tmp/sglang_runtime_requirements.py /src/sglang/python > /wheels/sglang-runtime-requirements.txt


# ---- build stage: the plugin's wheel, from this checkout (non-editable: the image carries the wheel's files) ----
FROM ${BASE} AS plugin-wheel
COPY --from=sglang-wheel /opt/buildenv /opt/buildenv
# --no-deps: the wheel declares `sglang`, which comes from the wheel above, never from PyPI; no build isolation, so the
# build downloads nothing. The checkout is copied (not mounted): setuptools writes its build tree beside the sources.
COPY . /src/plugin
RUN /opt/buildenv/bin/pip wheel --quiet --no-deps --no-build-isolation -w /wheels /src/plugin


# ---- the torch-neuronx build the image serves with, when it is not the container's own ----
# Empty by default, so `public` keeps the container's torch-neuronx. docker/build.sh --torch-neuronx <wheel> passes a
# directory holding that one wheel as this build context (`docker build --build-context torch-neuronx=<dir>`).
FROM scratch AS torch-neuronx


# ---- target public: the shipped image ----
FROM ${BASE} AS public
ARG SGLANG_COMMIT
ARG TORCHVISION_INDEX
# A torch-neuronx wheel from the torch-neuronx context replaces the container's editable install from
# /workspace/torch_neuronx (README, "Known limitations"); with none given this step changes nothing. --no-deps: torch is
# the container's (its PIP_CONSTRAINT), and check_image.py below refuses the image if the wheel's requirements or the
# plugin's torch-neuronx source pins do not hold.
RUN --mount=type=bind,from=torch-neuronx,target=/torch-neuronx \
    --mount=type=tmpfs,target=/root/.cache \
    set -eu; \
    set -- /torch-neuronx/torch_neuronx-*.whl; \
    if [ ! -e "$1" ]; then exit 0; fi; \
    if [ "$#" -ne 1 ]; then echo "the torch-neuronx context must hold one wheel, it holds $#" >&2; exit 1; fi; \
    python3 -m pip uninstall --yes torch-neuronx; \
    python3 -m pip install --quiet --no-cache-dir --no-deps "$1"
# SGLang's runtime dependencies in the base's own Python (DLC users run plain `python`): exactly the files
# docker/requirements-lock.txt names, by sha256, with nothing resolved at build time (docker/lock_requirements.sh writes
# it from SGLang's runtime set and SGLang Diffusion's packages, docker/diffusion-requirements.txt). torch and
# torch-neuronx are not touched.
RUN --mount=type=bind,source=docker/requirements-lock.txt,target=/tmp/requirements-lock.txt \
    --mount=type=tmpfs,target=/root/.cache \
    python3 -m pip install --quiet --no-cache-dir --no-deps --require-hashes -r /tmp/requirements-lock.txt
# torchvision's CPU build (SGLang imports it), by sha256 and without dependencies, so it cannot bring another torch
RUN --mount=type=bind,source=docker/torchvision-lock.txt,target=/tmp/torchvision-lock.txt \
    --mount=type=tmpfs,target=/root/.cache \
    python3 -m pip install --quiet --no-cache-dir --no-deps --require-hashes --index-url ${TORCHVISION_INDEX} \
    -r /tmp/torchvision-lock.txt
# SGLang and the plugin from their wheels, without dependencies (both are satisfied above)
RUN --mount=type=bind,from=sglang-wheel,source=/wheels,target=/sglang-wheels \
    --mount=type=bind,from=plugin-wheel,source=/wheels,target=/plugin-wheels \
    --mount=type=tmpfs,target=/root/.cache \
    python3 -m pip install --quiet --no-cache-dir --no-deps /sglang-wheels/sglang-*.whl /plugin-wheels/sglang_neuron-*.whl
# The patched NIXL into the base's site-packages, as pip --target laid it out (its packages, its bundled libraries and
# both dist-infos), so it needs no PYTHONPATH and a user's own PYTHONPATH cannot drop it. Its asio headers and pkg-config
# file are left out: nothing in the image builds against NIXL.
RUN --mount=type=bind,from=nixl-build,source=/opt/nixl/overlay_cu13,target=/nixl \
    SP=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
 && cp -a /nixl/nixl /nixl/nixl-1.5.0.dist-info /nixl/nixl_cu13 /nixl/nixl_cu13-1.5.0.dist-info \
          /nixl/.nixl_cu13.mesonpy.libs /nixl/nixl_ep /nixl/nixl_meta_utils.py "$SP"/
# The NIXL notice: where the image's NIXL comes from, its patches and its licenses (shipped on purpose; the audit checks
# them)
COPY patches/nixl/*.patch patches/nixl/README.md /usr/share/doc/sglang-neuron/nixl/
COPY docker/nixl-notice.md /usr/share/doc/sglang-neuron/nixl/NOTICE.md
# The repository's public part beside the install, read-only, at its repository paths: the commit's README.md, LICENSE,
# pyproject.toml, docs/, examples/, the plugin's sources (sglang_neuron/: the installed package's files and the kernels'
# records the wheel leaves out), the NIXL patches with their README, and the one tool an example needs (the GLM-5.2 1M
# weight store's builder, standard library only). Not tests/, docker/, scripts/, the NIXL series' tests or any other
# tool (root-owned, the context's 644 / 755 modes). The check below proves the copy is the commit's files and its
# sglang_neuron/ the installed package; the gate allows nothing else under /opt/sglang-neuron in this image.
COPY README.md LICENSE pyproject.toml /opt/sglang-neuron/
COPY docs /opt/sglang-neuron/docs
COPY examples /opt/sglang-neuron/examples
COPY sglang_neuron /opt/sglang-neuron/sglang_neuron
COPY patches/nixl/README.md patches/nixl/*.patch /opt/sglang-neuron/patches/nixl/
COPY tools/glm52/build_latent_store.py /opt/sglang-neuron/tools/glm52/build_latent_store.py
# SGLang serves on the Neuron platform in this image (the plugin's platform entry point), with no flag from the user
ENV SGLANG_PLATFORM=neuron
# The base's library path less its leading empty entry, which makes the loader search the working directory first
ENV LD_LIBRARY_PATH=/opt/aws/neuron/lib:/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/usr/local/lib:/home/.openmpi/lib/
# The image check (docker/check_image.py; it sets the no-device environment itself): SGLang is installed at the pinned
# commit; the plugin's installed files are the checkout's sglang_neuron/; its entry points load; its torch /
# torch-neuronx source pins hold on the container's builds; NIXL is exactly the patched nixl + nixl-cu13 1.5.0, found
# through the plugin's own loaders (kimi_k3.launch.nixl_problems, the PP seam's 0009 check, the PD progress-delay
# hook); pip finds nothing broken beyond its two known lines, and SGLang's runtime set is satisfied; SGLang's serving
# modules import; the Neuron SDK parts match the release pin and the plugin passes the source guard; /opt/sglang-neuron
# holds the checkout's public part above, byte for byte and root-owned, and its sglang_neuron/ is the installed package.
# Whatever the imports cache (SGLang's inductor cache under $HOME, torch-neuronx's HLO cache under /tmp) goes to a
# tmpfs, so the check leaves nothing in the image.
RUN --mount=type=bind,source=.,target=/src \
    --mount=type=bind,from=sglang-wheel,source=/wheels,target=/sglang-wheels \
    --mount=type=tmpfs,target=/tmp \
    HOME=/tmp/check-home python3 /src/docker/check_image.py --checkout /src --sglang-commit ${SGLANG_COMMIT} \
    --runtime-requirements /sglang-wheels/sglang-runtime-requirements.txt
# The base starts its training-container script and polls a health endpoint the server does not serve: the image
# runs a plain shell (`python -m sglang.launch_server ...` is the user's command) with no health check.
ENTRYPOINT []
CMD ["/bin/bash"]
HEALTHCHECK NONE
# What the image is built from (docker inspect; the release notes). created and description replace the ones the base
# inherits from its Ubuntu image; the base's other labels and environment stay as the Neuron SDK container sets them.
# The commit and the date are declared here, after every RUN, so they key no build step.
ARG SGLANG_NEURON_COMMIT
ARG BUILD_DATE
LABEL org.opencontainers.image.title="sglang-neuron" \
      org.opencontainers.image.description="SGLang with the SGLang-Neuron plugin on the AWS Neuron SDK 2.33.0 container" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.revision="${SGLANG_NEURON_COMMIT}" \
      com.amazonaws.neuron.sdk="2.33.0" \
      com.amazonaws.sglang-neuron.sglang-commit="${SGLANG_COMMIT}"


# ---- target qa: public + the repository checkout and the QA dependencies ----
FROM public AS qa
ARG SGLANG_COMMIT
# The whole checkout, sglang_neuron/ included. A Python started in the checkout imports this sglang_neuron/ ahead of
# the installed one, which is harmless: the check below proves the install equals this copy byte for byte.
COPY . /opt/sglang-neuron/src
# The QA dependencies (pytest and its plugins, lm-eval with its OpenAI-compatible client, torchmetrics and boto3):
# exactly the files docker/qa-requirements-lock.txt names, by sha256, resolved against the public image's packages, so
# QA adds packages and changes none; the step proves it. Two of them publish only source archives, built here with the
# base's setuptools.
RUN --mount=type=bind,source=docker/qa-requirements-lock.txt,target=/tmp/qa-requirements-lock.txt \
    --mount=type=tmpfs,target=/root/.cache \
    --mount=type=tmpfs,target=/tmp/qa \
    python3 -m pip list --format=freeze > /tmp/qa/public.txt \
 && python3 -m pip install --quiet --no-cache-dir --no-deps --require-hashes --no-build-isolation \
    -r /tmp/qa-requirements-lock.txt \
 && python3 -m pip list --format=freeze > /tmp/qa/qa.txt \
 && python3 -c 'public = dict(l.strip().split("==", 1) for l in open("/tmp/qa/public.txt") if "==" in l); \
qa = dict(l.strip().split("==", 1) for l in open("/tmp/qa/qa.txt") if "==" in l); \
changed = {n: (v, qa.get(n)) for n, v in public.items() if qa.get(n) != v}; \
assert not changed, f"the QA step changed packages of the public image: {changed}"; \
print(f"qa: {len(qa) - len(public)} packages added, none of the public image changed")'
# The image check again, with the QA dependencies installed, plus an import of each of them (torchmetrics
# imports torch, so with the device backend's autoload off: the build has no Neuron device)
RUN --mount=type=bind,from=sglang-wheel,source=/wheels,target=/sglang-wheels \
    --mount=type=tmpfs,target=/tmp \
    HOME=/tmp/check-home python3 /opt/sglang-neuron/src/docker/check_image.py --checkout /opt/sglang-neuron/src \
    --sglang-commit ${SGLANG_COMMIT} --runtime-requirements /sglang-wheels/sglang-runtime-requirements.txt \
 && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -c \
    "import pytest, xdist, pytest_timeout, tenacity, lm_eval.models.api_models, torchmetrics, boto3"
LABEL com.amazonaws.sglang-neuron.qa-tests="/opt/sglang-neuron/src"
