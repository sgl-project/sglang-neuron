#!/bin/bash
# Build the NIXL overlay that disaggregated (PD) servers load: upstream NIXL's release/1.5.0 branch at BASE_COMMIT with
# patches/nixl applied in order, built by NIXL's own meson-python build as one wheel (libnixl, its Python bindings and
# the LIBFABRIC plugin) without etcd, and installed with the released nixl 1.5.0 meta wheel into <out dir>.
#
# Why not NIXL's PyPI wheels: their libnixl links etcd-cpp-api, which brings gRPC and a protobuf that collides with the
# one torch-neuronx links statically, so a process importing both crashes. SGLang's KV transfer does not use NIXL's
# etcd metadata exchange. patches/nixl/README.md says what each patch fixes.
#
# usage (on a node of the serving container, which has gcc, the EFA headers and the interpreter the servers run):
#   NIXL_WHEELS=<dir> [NIXL_PACKAGECACHE=<dir>] [NIXL_REPO=<url or path>] bash tools/build_nixl.sh <out dir>
# NIXL_WHEELS holds the build tools and the meta wheel pinned below (pip download them on a machine with PyPI access);
# NIXL_PACKAGECACHE holds NIXL's meson subproject archives for a node that cannot reach their hosts: abseil-cpp, asio,
# liburing, tomlplusplus and taskflow, and the two wrapdb patch archives, as `meson subprojects download` leaves them in
# subprojects/packagecache; with it and NIXL_REPO a local clone that has BASE_COMMIT, the build needs no network. Serve
# the result with <out dir> first on PYTHONPATH and NIXL_PLUGIN_DIR unset. An existing <out dir> is never touched; a
# failed build is kept beside it.
set -euo pipefail
export LC_ALL=C
mapfile -t GIT_LOCAL_ENV < <(git rev-parse --local-env-vars)
unset "${GIT_LOCAL_ENV[@]}"
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CUDA_HOME CUDA_PATH PKG_CONFIG_LIBDIR PKG_CONFIG_SYSROOT_DIR PYTHONPATH

BASE_COMMIT=075b2397029fc67a0b69a34df566c1ae3d00f5ab   # release/1.5.0, the commit the PyPI 1.5.0 wheels were built at
SOURCE_TREE=6c6792fde6bdcc5d6da18de75141daca1b4764d6   # git tree of BASE_COMMIT with patches/nixl applied
NIXL_REPO=${NIXL_REPO:-https://github.com/ai-dynamo/nixl.git}
TOOL_WHEELS="meson==1.6.1 --hash=sha256:3f41f6b03df56bb76836cc33c94e1a404c3584d48b3259540794a60a21fad1f9
pybind11==3.1.0 --hash=sha256:b8488090f8acffbcb6b5d6a85571a6827a0a2981ffb75e5a0b27b87c4a6b7dd0
pkgconf==3.0.7.post0 --hash=sha256:bd3ee4b5294a888d58462324cb83b7c0bc7150cc15b64b36aaf983e4b0ab90d5
meson-python==0.18.0 --hash=sha256:3b0fe051551cc238f5febb873247c0949cd60ded556efa130aa57021804868e2
pyproject-metadata==0.9.1 --hash=sha256:ee5efde548c3ed9b75a354fc319d5afd25e9585fa918a34f62f904cc731973ad
packaging==25.0 --hash=sha256:29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484
ninja==1.13.2 --hash=sha256:65a24341b5ac09fcadcc37082660be40a94174e51a937fabf6e2cae26225fa2c"
META_WHEEL="nixl==1.5.0 --hash=sha256:1be21929dcf331b7637e4daafd538169c70467b0dfb9445621b4036e4831fb38"
# taskflow is the one subproject NIXL takes by git tag ([wrap-git] v3.10.0, no hash): the build rewrites its wrap as the
# same release's archive at a sha256 (a [wrap-file]), so meson checks the bytes it unpacks and NIXL_PACKAGECACHE can
# serve it to an offline build. Its meson overlay (patch_directory) stays NIXL's own.
TASKFLOW_TAG=v3.10.0
TASKFLOW_WRAP="[wrap-file]
directory = taskflow-3.10.0
source_url = https://github.com/taskflow/taskflow/archive/refs/tags/v3.10.0.tar.gz
source_filename = taskflow-3.10.0.tar.gz
source_hash = fe86765da417f6ceaa2d232ffac70c9afaeb3dc0816337d39a7c93e39c2dee0b
patch_directory = taskflow

[provide]
taskflow = taskflow_dep"
EFA=/opt/amazon/efa
# what the overlay must neither need nor export: etcd's client and what it pulls in
FORBIDDEN_LIBS='libetcd|libgrpc|libprotobuf|libcpprest'
FORBIDDEN_SYMS='google::protobuf|grpc::|etcd::'
# a string 0006, 0007 and 0009 each put in the plugin (0008 adds none; the source tree check covers it). The PP seam
# reads 0009's in the plugin it loaded (distributed/pp_seam.py)
MARKERS=("NIXL_LIBFABRIC_EFA_UNSOLICITED_WRITE_RECV" " names no request in use; nothing to fail"
         "Dropping the notification for notif_xfer_id=")

die() { echo "build_nixl: $*" >&2; exit 1; }
[ $# -eq 1 ] || die "usage: NIXL_WHEELS=<dir> bash tools/build_nixl.sh <out dir>"
OUT=${1%/}
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
[ ! -e "$OUT" ] && [ ! -L "$OUT" ] || die "$OUT exists"
[ -d "$(dirname -- "$OUT")" ] || die "$(dirname -- "$OUT") is not a directory"
[ -d "${NIXL_WHEELS:-}" ] || die "NIXL_WHEELS must name the directory of the pinned build wheels"
PATCHES=$(cd "$(dirname "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)/patches/nixl
PY=$(command -v python3) || die "no python3 on PATH"
for t in git gcc g++ patchelf readelf nm; do command -v $t > /dev/null || die "no $t on PATH"; done
[ -f $EFA/include/rdma/fabric.h ] || die "no libfabric headers under $EFA"
! pkg-config --exists etcd-cpp-api 2>/dev/null || die "etcd-cpp-api is installed here, and NIXL would link it"
# NIXL's meta package imports nixl_cu<N>, N the CUDA major of the torch it runs with, so the wheel is named after the
# container torch's CUDA major whatever it links (it links no CUDA)
# (read from torch/version.py: importing torch here would start torch-neuronx)
CUDA_MAJOR=$("$PY" -I -c "
import importlib.util, os, re
path = os.path.join(importlib.util.find_spec('torch').submodule_search_locations[0], 'version.py')
m = re.search(r'^cuda\b.*=\s*[\'\"](\d+)\.', open(path).read(), re.M)
print(m.group(1) if m else '')") || die "cannot read the CUDA version of the container's torch"
[ -n "$CUDA_MAJOR" ] || die "the container's torch is not a CUDA build, so NIXL's meta package loads no variant"
VARIANT=cu$CUDA_MAJOR

shopt -s nullglob
SERIES=("$PATCHES"/[0-9][0-9][0-9][0-9]-*.patch)
[ ${#SERIES[@]} -ge 1 ] || die "no patches in $PATCHES"
W=$(mktemp -d "$OUT.build.XXXXXX")
trap '[ -d "$W" ] && echo "build_nixl: failed; the partial build is in $W" >&2' EXIT

SRC=$W/src
git clone --quiet --no-checkout --template= -- "$NIXL_REPO" "$SRC" || die "cannot clone $NIXL_REPO"
G=(env GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 git -C "$SRC" -c user.name=build_nixl -c user.email=build@nixl)
"${G[@]}" cat-file -e "$BASE_COMMIT^{commit}" 2>/dev/null || die "$NIXL_REPO has no commit $BASE_COMMIT"
"${G[@]}" checkout --quiet --detach "$BASE_COMMIT"
for p in "${SERIES[@]}"; do
  "${G[@]}" apply --index "$p" || die "$(basename "$p") does not apply"
done
GOT=$("${G[@]}" write-tree)
[ "$GOT" = "$SOURCE_TREE" ] \
  || die "the patched source is tree $GOT, not $SOURCE_TREE: patches/nixl and this script disagree"
# the wheel's name, as NIXL's own wheel build sets it (contrib/build-wheel.sh)
[ "$(grep -c '^name = "nixl-cu12"$' "$SRC/pyproject.toml")" = 1 ] || die "pyproject.toml has no single nixl-cu12 name"
sed -i "s/^name = \"nixl-cu12\"$/name = \"nixl-$VARIANT\"/" "$SRC/pyproject.toml"
grep -qx "revision = $TASKFLOW_TAG" "$SRC/subprojects/taskflow.wrap" \
  || die "subprojects/taskflow.wrap is not taskflow $TASKFLOW_TAG: update TASKFLOW_WRAP to its new release"
printf '%s\n' "$TASKFLOW_WRAP" > "$SRC/subprojects/taskflow.wrap"

"$PY" -m venv "$W/bvenv"
BPY=$W/bvenv/bin/python
printf '%s\n' "$TOOL_WHEELS" > "$W/tools.txt"
PIP_FIND_LINKS=$NIXL_WHEELS "$BPY" -m pip install --quiet --no-deps --no-index --require-hashes -r "$W/tools.txt" \
  || die "cannot install the build tools from $NIXL_WHEELS"
mkdir -p "$W/bin"
ln -s "$("$BPY" -c 'import os, pkgconf; print(os.path.join(os.path.dirname(pkgconf.__file__), ".bin", "pkgconf"))')" \
  "$W/bin/pkg-config"
# system tools only: with uv on PATH NIXL's build would resolve its meta wheel's build requirements from PyPI
export PATH=$W/bin:$W/bvenv/bin:/usr/bin:/bin
export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/share/pkgconfig:$EFA/lib/pkgconfig
! command -v uv > /dev/null || die "uv is on the build PATH"

if [ -n "${NIXL_PACKAGECACHE:-}" ]; then
  CACHED=("$NIXL_PACKAGECACHE"/*)
  [ ${#CACHED[@]} -ge 1 ] || die "NIXL_PACKAGECACHE=$NIXL_PACKAGECACHE has no archives"
  mkdir -p "$SRC/subprojects/packagecache" && cp -- "${CACHED[@]}" "$SRC/subprojects/packagecache/"
fi
# every subproject but prometheus-cpp: NIXL's Prometheus telemetry exporter is optional, unused by SGLang, and its
# CMake build installs libraries outside the wheel
WRAPS=()
for w in "$SRC"/subprojects/*.wrap; do n=$(basename "$w" .wrap); [ "$n" = prometheus-cpp ] || WRAPS+=("$n"); done
(cd "$SRC" && meson subprojects download "${WRAPS[@]}" > "$W/subprojects.log" 2>&1) \
  || die "meson subprojects download failed ($W/subprojects.log)"

SETUP=(-Dwheel_variant=$VARIANT -Denable_plugins=LIBFABRIC -Dlibfabric_path=$EFA -Dbuild_tests=false
       -Dbuild_examples=false -Dinstall_headers=false -Dbuild_docs=false --wrap-mode=nodownload)
CS=(); for a in "${SETUP[@]}"; do CS+=("-Csetup-args=$a"); done
"$BPY" -m pip wheel --quiet --no-deps --no-build-isolation --no-index -w "$W/dist" "${CS[@]}" -Cbuild-dir="$W/build" \
  "$SRC" > "$W/wheel.log" 2>&1 || die "the NIXL wheel build failed ($W/wheel.log)"
grep -q "ETCD CPP API library not found" "$W/build/meson-logs/meson-log.txt" || die "meson found etcd"
WHEEL=("$W"/dist/nixl_${VARIANT}-1.5.0-cp*-linux_x86_64.whl)
[ ${#WHEEL[@]} -eq 1 ] || die "expected one nixl_$VARIANT 1.5.0 wheel in $W/dist"

printf '%s\n' "$META_WHEEL" > "$W/meta.txt"
# Two builds lay down the same bytes: the wheel installs by name from the build's own dist directory, not by its path
# (pip records a path install's location in direct_url.json), and without bytecode (each .pyc would carry the random
# build directory as its source path); Python compiles the few NIXL modules where it imports them.
PIP_FIND_LINKS=$NIXL_WHEELS "$BPY" -m pip install --quiet --no-deps --no-index --no-compile --require-hashes \
  --target "$W/overlay" -r "$W/meta.txt" || die "cannot install the nixl 1.5.0 meta wheel from $NIXL_WHEELS"
"$BPY" -m pip install --quiet --no-deps --no-index --no-compile --find-links "$W/dist" --target "$W/overlay" \
  "nixl-$VARIANT==1.5.0" || die "cannot install the wheel"
[ -z "$(find "$W/overlay" -path '*.dist-info/direct_url.json')" ] || die "the overlay records where it was built from"

mapfile -t SOS < <(find "$W/overlay" -name '*.so*' -type f | sort)
[ ${#SOS[@]} -ge 4 ] || die "the overlay has only ${#SOS[@]} shared objects"
for so in "${SOS[@]}"; do
  # captured rather than grep -q: an early grep exit would fail the pipeline on the writer's SIGPIPE
  needed=$(readelf -d "$so" | grep NEEDED | grep -E "$FORBIDDEN_LIBS" || true)
  [ -z "$needed" ] || die "$so needs $needed"
  syms=$(nm -D --defined-only -C "$so" 2>/dev/null | grep -cE "$FORBIDDEN_SYMS" || true)
  [ "${syms:-0}" = 0 ] || die "$so exports $syms protobuf, gRPC or etcd symbols"
done
[ -z "$(find "$W/overlay" -name 'libtelemetry_exporter_prometheus*')" ] || die "the overlay has a Prometheus exporter"
PLUGIN=$W/overlay/.nixl_$VARIANT.mesonpy.libs/plugins/libplugin_LIBFABRIC.so
[ -f "$PLUGIN" ] || die "the overlay has no LIBFABRIC plugin at ${PLUGIN#"$W/overlay/"}"
for m in "${MARKERS[@]}"; do grep -q -a -- "$m" "$PLUGIN" || die "the plugin lacks '$m'"; done
mv -T -- "$W/overlay" "$OUT"
rm -rf -- "$W"
trap - EXIT
echo "build_nixl: $OUT is NIXL ${BASE_COMMIT:0:8} with ${#SERIES[@]} patches (tree ${SOURCE_TREE:0:12})," \
     "without etcd;" \
     "LIBFABRIC plugin md5 $(md5sum < "$OUT/${PLUGIN#"$W/overlay/"}" | cut -c1-32)"
