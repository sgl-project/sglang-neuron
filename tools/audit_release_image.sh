#!/bin/bash
# Audit an image before it ships, against the image it was built on. Nothing is pulled, pushed or changed: both
# images must be local (docker pull them by digest first). `docker save` streams into the checks and the script
# extracts nothing, but on Docker's classic image store the daemon itself first writes the whole export, every layer
# uncompressed, to a directory under its root (DOCKER_TMPDIR, by default <DockerRootDir>/tmp): the audit needs that
# much free space there. The script writes only its results in --out (the lists, the reports, the plugin it copies
# out, the SDK check it copies in); the only containers it starts run with no network and are removed.
#
# usage: tools/audit_release_image.sh <image> [--base <image>] [--out <dir>] [--names <file>] [--forbid <path>]...
#            [--require <path>]... [--own-only <path>]... [--leftover <regex>]... [--allow-source <path>]...
#            [--hash-under <path>]... [--adds-only] [--layer-prefix] [--same-config] [--same-config-except <label>]...
#            [--base-content] [--flattened] [--no-plugin] [--no-nixl]
#
#   --base          the image it was built on; by default the release container sglang_neuron/release/neuron_sdk.toml
#                   pins ([image] reference@digest)
#   --out           a new directory for the results (default ./release_audit_<UTC time>)
#   --names         internal-name patterns, one regular expression a line, from the release process (the repository
#                   holds none); a match in any file the image adds fails
#   --forbid        a path no layer may hold, the base's included, nor delete (the QA checkout a public image drops)
#   --require       a path the image's final file system must hold
#   --own-only      the only paths the image's own layers may add or delete (and their parent directories)
#   --leftover      a regular expression for build leftovers in the image's own layers (wheels, /tmp, caches)
#   --allow-source  a source tree the image ships on purpose (SGLang's, which runs from its checkout)
#   --adds-only     the own layers may not replace or delete a file of the layers below (QA FROM public)
#   --hash-under    report the sha256 of every file under a path (scan/report.json "hashes")
#   --layer-prefix  the base's layers must be the image's first layers, in order (built FROM the base)
#   --same-config   the image's runtime config and labels must equal the base's, but --same-config-except labels
#   --base-content  search the bytes of the layers shared with the base too (slower; the base's own audit)
#   --flattened     the image holds the base's files in its own layers (not built FROM the base): an own-layer file
#                   with the base's path, type and size is the base's, not a rewrite
#   --no-plugin     skip the installed-plugin checks (an image without the plugin, such as the base)
#   --no-nixl       skip the NIXL checks: by default NIXL (nixl, nixl_cu13) must be installed with the Apache-2.0
#                   licence in each dist-info, and /usr/share/doc/sglang-neuron/nixl must hold a README naming
#                   tools/build_nixl.sh's BASE_COMMIT and SOURCE_TREE and this checkout's patches/nixl, byte for byte
#
# Parts, each PASS or FAIL in the summary (tools/audit_release_image.py has the rules):
#   1. layers: every file of every layer, the ones a later layer deletes included (docker save hands out each layer
#      whole); the image's own layers fail on a whiteout leak, a source checkout, a compile cache, a key, a token, an
#      ELF no package owns, an internal name in the plugin or the NIXL notice, a --forbid / --own-only / --leftover /
#      --adds-only / --require breach, and the NIXL licences and notice; what the shared base layers hold is
#      reported. Lists of what the image adds over the base, grouped by directory, and of what it adds outside /root
#      and /home.
#   2. plugin: tools/neuron_release.py guard source over the installed site-packages/sglang_neuron, copied out of the
#      image (no shared object, no GpSimd kernel, no nkilib private package, no module loaded by path).
#   3. versions: versions.json (the SDK packages, torch, torch-neuronx, SGLang, sglang-neuron, NIXL, transformers,
#      the image labels) and this checkout's tools/neuron_release.py check, copied into a container of the image
#      (docker cp, no bind mount: the daemon may be remote), against its neuron_sdk.toml.
#
# Exit status 0 when every part passes, 1 when one fails, 2 on a usage error. tools/audit_release_image.py's docstring
# lists the result files.
# Plain bash 4.2 (Amazon Linux 2's /bin/bash): set -u there calls an empty "${a[@]}" unbound, so an array that may be
# empty expands as ${a[@]+"${a[@]}"}.
set -euo pipefail
export LC_ALL=C

die() { echo "audit_release_image: $*" >&2; exit 2; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HELPER=$REPO/tools/audit_release_image.py
[ $# -ge 1 ] || die "usage: $0 <image> [--base <image>] [--out <dir>] [options]; see the header"
IMAGE=$1; shift
BASE=; OUT=./release_audit_$(date -u +%Y%m%dT%H%M%SZ); PLUGIN=1; NIXL=1; SCAN=()
while [ $# -gt 0 ]; do
  case $1 in
    --base) BASE=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --no-plugin) PLUGIN=0; shift ;;
    --no-nixl) NIXL=0; shift ;;
    --base-content|--adds-only|--layer-prefix|--same-config|--flattened) SCAN+=("$1"); shift ;;
    --names|--forbid|--restrict|--restrict-allow|--require|--own-only|--leftover|--allow-source|--hash-under|\
    --same-config-except)
      [ $# -ge 2 ] || die "$1 needs a value"; SCAN+=("$1" "$2"); shift 2 ;;
    *) die "unknown option $1" ;;
  esac
done
command -v docker > /dev/null || die "no docker on PATH"
command -v python3 > /dev/null || die "no python3 on PATH"
if [ -z "$BASE" ]; then
  BASE=$(python3 - "$REPO/sglang_neuron/release/neuron_sdk.toml" <<'PY'
import sys, tomllib
image = tomllib.load(open(sys.argv[1], "rb"))["image"]
print(f"{image['reference'].rsplit(':', 1)[0]}@{image['digest']}")
PY
)
fi
if [ "$NIXL" = 1 ]; then
  ids=$(sed -nE 's/^(BASE_COMMIT|SOURCE_TREE)=([0-9a-f]{40}).*/\2/p' "$REPO/tools/build_nixl.sh" | paste -sd, -)
  [ "$(tr -cd , <<< "$ids")" = , ] || die "tools/build_nixl.sh names no BASE_COMMIT and SOURCE_TREE"
  SCAN+=(--nixl-patches "$REPO/patches/nixl" --nixl-ids "$ids")
fi
[ ! -e "$OUT" ] || die "$OUT exists; give a new directory"
for ref in "$BASE" "$IMAGE"; do
  docker image inspect "$ref" > /dev/null 2>&1 || die "$ref is not a local image; docker pull it (by digest) first"
done
BASE_ID=$(docker image inspect -f '{{.Id}}' "$BASE")
IMAGE_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE")
BASE_LAYERS=$(docker image inspect -f '{{range $i, $l := .RootFS.Layers}}{{if $i}},{{end}}{{$l}}{{end}}' "$BASE")
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
CTR=
trap '[ -z "$CTR" ] || docker rm -f "$CTR" > /dev/null 2>&1 || true' EXIT
declare -A RESULT
echo "== image $IMAGE ($IMAGE_ID)"
echo "== base  $BASE ($BASE_ID)"

echo "== 1. layers"
refs=("$BASE"); [ "$BASE_ID" = "$IMAGE_ID" ] || refs+=("$IMAGE")
set +e
docker save "${refs[@]}" | python3 "$HELPER" scan --base-id "$BASE_ID" --image-id "$IMAGE_ID" \
  --base-layers "$BASE_LAYERS" --out "$OUT/scan" ${SCAN[@]+"${SCAN[@]}"} | tee "$OUT/scan.txt"
status=("${PIPESTATUS[@]}")
set -e
[ "${status[0]}" = 0 ] || die "docker save failed (status ${status[0]})"
[ "${status[1]}" -le 1 ] || die "the layer scan could not read the saved images (status ${status[1]})"
RESULT[layers]=$([ "${status[1]}" = 0 ] && echo PASS || echo FAIL)

if [ "$PLUGIN" = 1 ]; then
  echo "== 2. plugin"
  mapfile -t INSTALLED < "$OUT/scan/sglang_neuron_path.txt"
  if [ "${#INSTALLED[@]}" != 1 ]; then
    echo "sglang_neuron is installed ${#INSTALLED[@]} times, not once: ${INSTALLED[*]:-nowhere}"
    RESULT[plugin]=FAIL
  else
    CTR=$(docker create --network none --entrypoint "" "$IMAGE" true)
    mkdir -p "$OUT/plugin"
    docker cp -q "$CTR:${INSTALLED[0]}" "$OUT/plugin/"
    docker rm -f "$CTR" > /dev/null; CTR=
    set +e
    python3 "$REPO/tools/neuron_release.py" guard source "$OUT/plugin/sglang_neuron" | tee "$OUT/guard_source.txt"
    guard=${PIPESTATUS[0]}
    set -e
    RESULT[plugin]=$([ "$guard" = 0 ] && echo PASS || echo FAIL)
  fi
fi

echo "== 3. versions"
set +e
docker run --rm -i --network none --entrypoint python3 "$IMAGE" - versions < "$HELPER" > "$OUT/versions_image.json"
versions=$?
set -e
[ "$versions" = 0 ] || die "the versions probe failed in $IMAGE (status $versions)"
# The SDK check is this checkout's code, copied into a container of the image with docker cp, which sends the files
# from this client. A bind mount would name a path on the daemon's host, and a remote DOCKER_HOST or a
# Docker-in-Docker daemon does not have this checkout there: the container would see an empty directory.
# $OUT/sdk_check keeps the files the check ran.
mkdir -p "$OUT/sdk_check/tools" "$OUT/sdk_check/sglang_neuron"
cp "$REPO/tools/neuron_release.py" "$OUT/sdk_check/tools/"
cp "$REPO/sglang_neuron/__init__.py" "$OUT/sdk_check/sglang_neuron/"
cp -R "$REPO/sglang_neuron/release" "$OUT/sdk_check/sglang_neuron/"
CTR=$(docker create --network none --entrypoint python3 "$IMAGE" /audit/tools/neuron_release.py check) \
  || die "could not create a container of $IMAGE"
docker cp -q "$OUT/sdk_check/." "$CTR:/audit" || die "could not copy the SDK check into a container of $IMAGE"
set +e
docker start -a "$CTR" > "$OUT/sdk_check.txt" 2>&1
sdk=$?
set -e
docker rm -f "$CTR" > /dev/null; CTR=
python3 - "$OUT" "$BASE" "$BASE_ID" "$IMAGE" "$IMAGE_ID" "$sdk" <<'PY'
import json, subprocess, sys
out, base, base_id, image, image_id, sdk = sys.argv[1:]
def inspect(ref, field):
    return json.loads(subprocess.run(["docker", "image", "inspect", "-f", "{{json %s}}" % field, ref],
                                     capture_output=True, text=True, check=True).stdout)
doc = {"image": {"ref": image, "id": image_id, "repo_digests": inspect(image, ".RepoDigests") or [],
                 "labels": inspect(image, ".Config.Labels") or {}},
       "base": {"ref": base, "id": base_id, "repo_digests": inspect(base, ".RepoDigests") or []},
       **json.load(open(f"{out}/versions_image.json")),
       "sdk_check": {"status": int(sdk), "output": "sdk_check.txt"}}
json.dump(doc, open(f"{out}/versions.json", "w"), indent=1)
pip = doc["pip"]
for name in ("neuronx-cc", "nki", "torch", "torch-neuronx", "sglang-neuron", "transformers"):
    row = pip.get(name) or {}
    print(f"  {name:14s} {row.get('version', '-')}" + (f"  {row['source']}" if row.get("source") else ""))
print("  sglang         " + (json.dumps(doc["sglang_tree"]) if doc.get("sglang_tree") else "-"))
print("  nixl           " + (", ".join(f"{k} {v['version']}" for k, v in pip.items() if k.startswith("nixl")) or "-"))
print("  apt            " + ", ".join(f"{k} {v}" for k, v in doc["apt"].items()))
print("  labels         " + json.dumps(doc["image"]["labels"])[:300])
PY
tail -4 "$OUT/sdk_check.txt"
RESULT[versions]=$([ "$sdk" = 0 ] && echo PASS || echo FAIL)

echo "== summary ($OUT)"
rc=0
for part in layers plugin versions; do
  [ -n "${RESULT[$part]:-}" ] || continue
  echo "  $part: ${RESULT[$part]}"
  [ "${RESULT[$part]}" = PASS ] || rc=1
done
exit $rc
