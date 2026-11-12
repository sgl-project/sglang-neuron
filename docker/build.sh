#!/usr/bin/env bash
# docker/build.sh: build the SGLang-Neuron public and QA images on this host from one commit of this repository, and
# gate them. It pushes nothing: the images stay in the local Docker image store.
#
#   docker/build.sh [--tag <tag>] [--release --names <file>] [--torch-neuronx <wheel>] [--no-cache] [--out <dir>]
#
#   --tag <tag>                   name the images sglang-neuron:<tag> and sglang-neuron:<tag>-qa. Default
#                                 v<version>-<commit> for --release, v<version>-<commit>-dev otherwise
#   --release                     a release build: needs --names and a clean working tree
#   --names <file>                the internal-name patterns, one regular expression a line (# comments), outside the
#                                 repository: the gate fails on a listed name in our own content. Optional (the
#                                 gate then says it did not check), required with --release
#   --torch-neuronx <wheel>       install this torch_neuronx-*.whl in place of the Neuron SDK container's own build
#   --no-cache                    build every stage again instead of reusing Docker's build cache
#   --out <dir>                   where the logs, the gate's reports and versions.json go; must not exist (default
#                                 build/image-<tag>-<time>/ in the checkout)
#
# A build:
# 1. a clean export of HEAD (`git archive`), so the images hold the commit's files only, never local edits or
#    untracked files;
# 2. `docker build --target public`, the image users run, then `--target qa`, the same layers plus the
#    repository's checkout and the QA dependencies (docker/Dockerfile; its build-time check is docker/check_image.py);
# 3. docker/verify_images.sh, from the same export (the commit's own gate): the QA image is the public image plus
#    additions only, the checkout's sglang_neuron/ is the shipped one byte for byte, no layer of the public image holds
#    the checkout, none of its own layers a build leftover, and, with --names, none of our own content a listed name;
# 4. versions.json in the build directory, written last: the commit, the images the gate passed (by id), the versions
#    the release notes quote, and how the build was made. A build that fails anywhere leaves none.
# Exit status 0 when every step passes.
set -euo pipefail

die() { echo "[build] $*" >&2; exit 1; }
# the usage block of the header: from the first command line to the blank comment line after the options
usage() { sed -n '/^#   docker\/build.sh/,/^# A build:/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit "${1:-0}"; }
readonly TAG_RE='^[A-Za-z0-9_][A-Za-z0-9_.-]{0,124}$'  # leaves room for -qa in docker's 128 characters
readonly REPO=sglang-neuron

# the build and its gate are written for GNU coreutils (realpath -e and -m, sha256sum) and bash 4.2: on macOS or a BSD,
# where realpath has no -e and date no -d, say so here rather than "no such file" for --names or a failure mid-build
if ! { realpath -e / && date -u -d @0; } >/dev/null 2>&1 || (( BASH_VERSINFO[0] * 100 + BASH_VERSINFO[1] < 402 )); then
    die "needs a Linux x86_64 host with GNU coreutils and bash 4.2 or later (docker/README.md lists the rest)"
fi

TAG="" RELEASE=0 NAMES="" TORCH_NEURONX="" NO_CACHE=0 OUT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --release) RELEASE=1; shift ;;
        --no-cache) NO_CACHE=1; shift ;;
        --tag|--names|--torch-neuronx|--out)
            [[ $# -ge 2 && -n $2 && $2 != --* ]] || die "$1 needs a value"
            case $1 in
                --tag) TAG=$2 ;; --out) OUT=$2 ;;
                --names) NAMES=$(realpath -e "$2") || die "--names $2: no such file" ;;
                --torch-neuronx) TORCH_NEURONX=$(realpath -e "$2") || die "--torch-neuronx $2: no such file" ;;
            esac
            shift 2 ;;
        -h|--help) usage ;;
        *) echo "[build] unknown argument $1" >&2; usage 2 >&2 ;;
    esac
done
command -v docker >/dev/null || die "needs docker"
docker info >/dev/null 2>&1 || die "cannot reach the docker daemon (docker info fails)"
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null) \
    || die "build.sh builds from a git checkout of this repository"
COMMIT=$(git -C "${ROOT}" rev-parse HEAD)
readonly ROOT COMMIT
python3 -c 'import tomllib' 2>/dev/null || die "needs Python 3.11 or later on the build host (tomllib)"
# the build passes --build-context, which only buildx's `docker build` (Docker Engine 23.0 on) takes: before 23.0,
# `docker buildx version` passes and `docker build` is still the legacy builder, which rejects the flag. The help is
# read whole, not piped to grep -q, whose early exit could fail the pipeline (pipefail) on docker's SIGPIPE, and with
# DOCKER_BUILDKIT=1 as the build runs: with DOCKER_BUILDKIT=0 in the environment, any Docker shows the legacy help
[[ $(DOCKER_BUILDKIT=1 docker build --help 2>&1) == *--build-context* ]] \
    || die "needs Docker Engine 23.0 or later (docker build is buildx)"
VERSION=$(git -C "${ROOT}" show "${COMMIT}:pyproject.toml" \
    | python3 -c 'import sys, tomllib; print(tomllib.loads(sys.stdin.read())["project"]["version"])')
readonly VERSION
if (( RELEASE )); then
    TAG=${TAG:-v${VERSION}-${COMMIT:0:8}}
    [[ -n ${NAMES} ]] || die "--release needs --names <file>: a release is checked for internal names"
else
    TAG=${TAG:-v${VERSION}-${COMMIT:0:8}-dev}
fi
readonly TAG
[[ ${TAG} =~ ${TAG_RE} ]] || die "--tag ${TAG} is not a valid image tag ([A-Za-z0-9_][A-Za-z0-9_.-]*, at most 125)"
if [[ -n ${TORCH_NEURONX} && ! ${TORCH_NEURONX##*/} =~ ^torch_neuronx-.*\.whl$ ]]; then
    die "--torch-neuronx takes a torch_neuronx-*.whl, not ${TORCH_NEURONX##*/}"
fi
CLEAN=True
if [[ -n $(git -C "${ROOT}" status --porcelain) ]]; then
    CLEAN=False
    # the images are HEAD's either way; a released image must also be a commit nobody has to guess at
    (( RELEASE == 0 )) || die "the working tree has changes: commit them (or stash them) before --release"
    echo "[build] the working tree has changes; the images are built from HEAD ${COMMIT:0:8} only" >&2
fi
readonly PUBLIC="${REPO}:${TAG}" QA="${REPO}:${TAG}-qa"

# one directory per run, so a rebuild of the same commit keeps its own logs and reports
# one clock reading: the label's date (2026-11-11T21:30:00Z) and, without its - and :, the directory's
# (20261111T213000Z)
BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
readonly BUILD_DATE
OUT=$(realpath -m "${OUT:-${ROOT}/build/image-${TAG}-${BUILD_DATE//[-:]/}}")
readonly OUT
[[ ! -e ${OUT} ]] || die "${OUT} exists; give another --out"
mkdir -p "${OUT}"
# two builds of one tag would retag each other's images between their build and their gate
if command -v flock >/dev/null; then
    exec 9>"${TMPDIR:-/tmp}/sglang-neuron-build.$(printf %s "${PUBLIC}" | sha256sum | cut -c1-16).lock"
    flock -n 9 || die "another build of ${PUBLIC} is running"
fi
# the build context, and the torch-neuronx one: that one wheel when given, else empty (the container's own build)
CONTEXT=$(mktemp -d) TORCH_NEURONX_CONTEXT=$(mktemp -d)
readonly CONTEXT TORCH_NEURONX_CONTEXT
trap 'rm -rf "${CONTEXT}" "${TORCH_NEURONX_CONTEXT}"' EXIT
# tar.umask 0022: git archive writes 0664 / 0775 by default (tar.umask 0002), which a root tar (CI, docker-in-docker)
# and a umask-0002 user (Amazon Linux's ec2-user) keep, and the public image's documentation must be 0644 / 0755
git -C "${ROOT}" -c tar.umask=0022 archive "${COMMIT}" | tar -x -C "${CONTEXT}"
# --names is read here, with the gate's own reader, rather than only by the gate after the hour-long build: a directory,
# a pattern that is no regular expression, or a file with no pattern (which would pass as "names checked") stops here
if [[ -n ${NAMES} ]]; then
    count=$(python3 - "${CONTEXT}/tools" "${NAMES}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from audit_release_image import read_names
try:
    print(len(read_names(sys.argv[2])))
except (OSError, ValueError) as error:
    sys.exit(f"[build] {error}")
PY
    ) || die "--names ${NAMES} is not a patterns file the gate can use"
    echo "[build] --names: ${count} patterns" >&2
fi
# one base image: the Dockerfile's must be the release pin's (sglang_neuron/release/neuron_sdk.toml [image])
BASE=$(sed -n 's/^ARG BASE=//p' "${CONTEXT}/docker/Dockerfile")
readonly BASE
PINNED=$(python3 -c 'import sys, tomllib; image = tomllib.load(open(sys.argv[1], "rb"))["image"]
print(image["reference"].rsplit(":", 1)[0] + "@" + image["digest"])' "${CONTEXT}/sglang_neuron/release/neuron_sdk.toml")
[[ ${BASE} == "${PINNED}" ]] || die "Dockerfile BASE ${BASE} is not neuron_sdk.toml's ${PINNED}"
[[ -z ${TORCH_NEURONX} ]] || cp "${TORCH_NEURONX}" "${TORCH_NEURONX_CONTEXT}/"
# the base into the local image store (BuildKit's FROM pull does not put it there, and the gate reads it from there)
if ! docker image inspect "${BASE}" >/dev/null 2>&1; then
    echo "[build] pulling the base image ${BASE}" >&2
    docker pull --platform linux/amd64 "${BASE}" >&2 \
        || die "cannot pull the base image ${BASE} (network, or a stale login: docker logout public.ecr.aws)"
fi
free_gb=$(df -Pk "$(docker info -f '{{.DockerRootDir}}')" 2>/dev/null | awk 'NR == 2 {print int($4 / 1048576)}') || true
[[ -z ${free_gb} || ${free_gb} -ge 60 ]] \
    || echo "[build] warning: ${free_gb} GB free under Docker's root; a build from no cache needs about 60" >&2

# public first, then qa from the cache public just filled: --no-cache applies to the first build only, or qa would
# rebuild public's stages into other layers and fail the gate's "qa is public plus additions"
for target in public qa; do
    image=${PUBLIC}; [[ ${target} == qa ]] && image=${QA}
    echo "[build] ${target} -> ${image}" >&2
    cache=(); [[ ${target} == public ]] && (( NO_CACHE )) && cache=(--no-cache)
    DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -f "${CONTEXT}/docker/Dockerfile" --target "${target}" \
        ${cache[@]+"${cache[@]}"} --build-arg SGLANG_NEURON_COMMIT="${COMMIT}" --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-context torch-neuronx="${TORCH_NEURONX_CONTEXT}" -t "${image}" "${CONTEXT}" 2>&1 \
        | tee "${OUT}/build-${target}.log"
    # the id the image store gives the image (not --iidfile's, the config digest: Docker's containerd image store ids
    # an image by its manifest or index digest, and `docker run` there resolves only that); the lock above keeps any
    # other build of this tag out between the build and this read
    docker image inspect -f '{{.Id}}' "${image}" > "${OUT}/${target}.id"
done
PUBLIC_ID=$(<"${OUT}/public.id") QA_ID=$(<"${OUT}/qa.id")
readonly PUBLIC_ID QA_ID

names=(); [[ -z ${NAMES} ]] || names=(--names "${NAMES}")
"${CONTEXT}/docker/verify_images.sh" ${names[@]+"${names[@]}"} "${QA}" "${PUBLIC}" "${BASE}" "${OUT}/audit" 2>&1 \
    | tee "${OUT}/verify.log"
# the gate read the images by tag: they must still be the ones this run built
for pair in "${PUBLIC} ${PUBLIC_ID}" "${QA} ${QA_ID}"; do
    [[ $(docker image inspect -f '{{.Id}}' "${pair% *}") == "${pair#* }" ]] \
        || die "${pair% *} was retagged during the gate"
done

# the versions the release notes quote, read from the public image itself
docker run --rm --platform linux/amd64 --network none --entrypoint python3 "${PUBLIC_ID}" -c '
import importlib.metadata as m, json
names = ["sglang", "sglang-neuron", "torch", "torch-neuronx", "neuronx-cc", "nki", "nixl", "nixl-cu13",
         "transformers", "tokenizers", "huggingface-hub", "xgrammar", "torchvision"]
print(json.dumps({n: m.version(n) for n in names}, indent=1))' > "${OUT}/packages.json"
python3 - "${OUT}" "${COMMIT}" "${BASE}" "${TAG}" "${PUBLIC_ID}" "${QA_ID}" "${TORCH_NEURONX}" "${NAMES}" \
    "${RELEASE}" "${CLEAN}" "${BUILD_DATE}" <<'PY'
import hashlib, json, os, sys
out, commit, base, tag, public_id, qa_id, torch_neuronx, names, release, clean, date = sys.argv[1:]
sha256 = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
versions = {"sglang_neuron_commit": commit, "release": release == "1", "worktree_clean": clean == "True",
            "built": date, "base": base,
            "public": {"image": f"sglang-neuron:{tag}", "id": public_id},
            "qa": {"image": f"sglang-neuron:{tag}-qa", "id": qa_id},
            "gate": "passed", "names_checked": {"patterns_sha256": sha256(names)} if names else False,
            "packages": json.load(open(f"{out}/packages.json")),
            "torch_neuronx_wheel": {"file": os.path.basename(torch_neuronx), "sha256": sha256(torch_neuronx)}
            if torch_neuronx else "the Neuron SDK container's own"}
json.dump(versions, open(f"{out}/versions.json", "w"), indent=1)
print(json.dumps(versions, indent=1))
PY

echo "[build] done: ${PUBLIC} and ${QA} (${OUT})" >&2
