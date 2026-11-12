#!/usr/bin/env bash
# docker/verify_images.sh: the gate on every image pair docker/build.sh builds. It proves that the public SGLang-Neuron
# image is exactly what the QA image tested, and that no layer of it carries what must not ship.
#
#   docker/verify_images.sh [--names <file>] <qa_image> <public_image> <base_image> [<out_dir>]
#
# docker/Dockerfile builds the `public` target (the image that ships) and the `qa` target FROM it: QA = public + the
# release commit's checkout at /opt/sglang-neuron/src + the QA dependencies + one label. QA removes and replaces
# nothing of public, so nothing of public's survives only in a lower layer (public's own lock install replaces some
# of the base's packages, whose old files stay in the base's layers); this checks that relation with
# tools/audit_release_image.sh, which reads every layer `docker save` streams to it and extracts nothing (on Docker's
# classic image store the daemon itself first writes the whole export under its root, so the gate needs that much free
# space there):
#   1. public against the base: the base's layers open it in order; under /opt/sglang-neuron, in any layer, only the
#      repository's public part (PUBLIC_REPO; docker/check_image.py proves it the commit's files) and never the QA
#      checkout, tests/ or docker/; its sglang_neuron/ is the installed package (step 3); no build leftover, other
#      source checkout, key or token in its own layers, and no listed internal name in our own content there, the
#      repository copy included (a listed name in a third-party package's files is reported); the plugin and NIXL
#      installed as the release ships them;
#   2. QA against public: public's layers open it in order (every layer QA tested ships, by digest); the configs are
#      equal but for the QA label; QA's own layers only ADD, under the checkout, site-packages or /usr/local/bin, and
#      replace or delete nothing of public's;
#   3. the QA checkout's sglang_neuron/ and the public image's repository copy of it are the installed
#      site-packages/sglang_neuron byte for byte: a Python started in the checkout or in /opt/sglang-neuron imports
#      that copy ahead of site-packages, so each must be the shipped code. A record a tree has and the wheel leaves out is listed; a file that differs, one the wheel has
#      and the tree lacks, and a Python source the tree has and the install lacks fail.
#
# --names gives the release process's internal-name patterns (one regex a line; none in the repository). Without it
# that check does not run, and the result says so: docker/build.sh --release requires it. Exit status 0 only when all
# three pass. Plain bash 4.2 (no empty-array expansions).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
readonly REPO
readonly AUDIT="${REPO}/tools/audit_release_image.sh"
# the image's own directory: the public image holds only the repository's public part there, the QA image adds
# the checkout
readonly QA_ROOT=/opt/sglang-neuron
readonly CHECKOUT=/opt/sglang-neuron/src
# the repository's public part the public image ships there, at its repository paths (docker/check_image.py's
# PUBLIC_REPO; a pattern expands against this checkout, the commit's export), and the parts it must never hold
readonly PUBLIC_REPO=(README.md LICENSE pyproject.toml docs examples sglang_neuron patches/nixl/README.md
                      'patches/nixl/*.patch' tools/glm52/build_latent_store.py)
readonly LEFT_OUT=(src tests docker patches/nixl/tests)
# the copy's two parts the audit's source rules take as shipped on purpose, and no other path there: its project file
# and the plugin's sources (a project file or a sglang_neuron/ anywhere else under it, docs/ and examples/ included,
# still fails)
readonly COPY_SOURCES=("${QA_ROOT}/pyproject.toml" "${QA_ROOT}/sglang_neuron")
# the base's Python and its console scripts: where the plugin and the QA test dependencies install
readonly SITE=/usr/local/lib/python3.14/site-packages
readonly SCRIPTS=/usr/local/bin
# the QA image's one config difference (docker/Dockerfile's qa target)
readonly QA_LABEL=com.amazonaws.sglang-neuron.qa-tests
# what a build leaves behind when a step forgets to clean up: wheels, /tmp, pip / uv caches, the NIXL build tree
readonly LEFTOVERS=('\.whl$' '^tmp/.+' '^root/\.cache/' '(^|/)nixl-build(/|$)')

die() { echo "verify_images: $*" >&2; exit 1; }

names=()
names_note="internal names NOT checked (no --names)"
if [[ ${1:-} == --names ]]; then
    [[ -f ${2:-} ]] || die "--names ${2:-}: no such file"
    names=(--names "$2")
    names_note="internal names checked against $2"
    shift 2
fi
[[ ${#names[@]} -gt 0 ]] || echo "verify_images: internal names are not checked (no --names)" >&2
[[ $# -eq 3 || $# -eq 4 ]] \
    || { echo "usage: $0 [--names <file>] <qa_image> <public_image> <base_image> [<out_dir>]" >&2; exit 2; }
qa=$1 public=$2 base=$3
out=${4:-./public_image_check_$(date -u +%Y%m%dT%H%M%SZ)}
[[ ! -e "${out}" ]] || die "${out} exists: give a new directory"
mkdir -p "${out}"
leftovers=()
for pattern in "${LEFTOVERS[@]}"; do leftovers+=(--leftover "${pattern}"); done

sources=()
for path in "${COPY_SOURCES[@]}"; do sources+=(--allow-source "${path}"); done

echo "[1/3] ${public} against ${base}"
copy=(--restrict "${QA_ROOT}" "${sources[@]}")
for entry in "${PUBLIC_REPO[@]}"; do
    for path in "${REPO}"/${entry}; do  # unquoted: a pattern expands; one with no match stays as written and fails
        copy+=(--restrict-allow "${QA_ROOT}/${path#"${REPO}/"}" --require "${QA_ROOT}/${path#"${REPO}/"}")
    done
done
for path in "${LEFT_OUT[@]}"; do copy+=(--forbid "${QA_ROOT}/${path}"); done
"${AUDIT}" "${public}" --base "${base}" --out "${out}/public" --layer-prefix "${copy[@]}" \
    --require "${SITE}/sglang_neuron/__init__.py" "${leftovers[@]}" ${names[@]+"${names[@]}"} \
    --hash-under "${SITE}/sglang_neuron" --hash-under "${QA_ROOT}/sglang_neuron" \
    || die "${public} fails its audit (${out}/public)"

echo "[2/3] ${qa} against ${public}"
"${AUDIT}" "${qa}" --base "${public}" --out "${out}/qa" --layer-prefix --same-config \
    --same-config-except "${QA_LABEL}" --adds-only --own-only "${CHECKOUT}" --own-only "${SITE}" \
    --own-only "${SCRIPTS}" --allow-source "${CHECKOUT}" "${sources[@]}" --no-plugin --no-nixl \
    --hash-under "${CHECKOUT}/sglang_neuron" \
    || die "${qa} is not ${public} plus its checkout and test dependencies (${out}/qa)"

echo "[3/3] ${CHECKOUT}/sglang_neuron and ${QA_ROOT}/sglang_neuron against the installed ${SITE}/sglang_neuron"
python3 - "${out}/public/scan/report.json" "${SITE}/sglang_neuron" "${out}/qa/scan/report.json" \
    "${CHECKOUT}/sglang_neuron" "${out}/public/scan/report.json" "${QA_ROOT}/sglang_neuron" <<'PY' \
    || die "the checkout's or the repository copy's sglang_neuron/ is not the shipped package"
import json, sys

def files(report, root):
    root = root.rstrip("/") + "/"
    # compiled bytecode is pip's, written at install time; it is neither tree's source
    return {p[len(root):]: h for p, h in json.load(open(report))["hashes"].items()
            if p.startswith(root) and "/__pycache__/" not in p and not p.endswith(".pyc")}

installed = files(sys.argv[1], sys.argv[2])
failed = not installed
for report, root in zip(sys.argv[3::2], sys.argv[4::2]):
    tree = files(report, root)
    differ = sorted(p for p in installed.keys() & tree.keys() if installed[p] != tree[p])
    only_installed = sorted(installed.keys() - tree.keys())
    only_tree = sorted(tree.keys() - installed.keys())
    # the wheel ships every Python source (pyproject.toml's packages): one only here is code the install lacks
    unshipped = [p for p in only_tree if p.endswith(".py")]
    print(f"{root}: installed {len(installed)} files, {len(tree)} here: "
          f"{len(installed.keys() & tree.keys()) - len(differ)} identical, {len(differ)} differ, "
          f"{len(only_installed)} only installed, {len(only_tree)} only here"
          + (f" (not in the wheel: {', '.join(only_tree[:10])})" if only_tree else ""))
    for p in differ[:10]:
        print(f"  differs: {p}")
    for p in only_installed[:10]:
        print(f"  only installed: {p}")
    for p in unshipped[:10]:
        print(f"  not installed: {p}")
    failed = failed or bool(differ or only_installed or unshipped or not tree)
sys.exit(1 if failed else 0)
PY
echo "OK: ${public} is what ${qa} tested, without its checkout and test dependencies; ${names_note}; reports in ${out}"
