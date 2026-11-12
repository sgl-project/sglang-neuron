# SPDX-License-Identifier: Apache-2.0
"""Check a built SGLang-Neuron image from the inside: what it installed is what this checkout says it should be.

Run inside the image, with this checkout mounted (the Dockerfile's build-time check) or present at /opt/sglang-neuron
(the QA image), offline. Nothing here needs a Neuron device: on a host without one (a docker build) it sets
NO_DEVICE_ENV first, so torch-neuronx imports without starting its device backend.

    python3 docker/check_image.py --sglang-commit <sha> --runtime-requirements <file>

It checks:
1. SGLang and the plugin are installed packages (not a source tree on the path), SGLang at the pinned commit, and the
   plugin's install is the files of this checkout's sglang_neuron/ its wheel ships (shipped_files), file for file and
   byte for byte;
2. the plugin's entry points load;
3. the torch and torch-neuronx functions the plugin wraps are the sources it pins (backend/torch_pins.py,
   backend/executor_cache.py, backend/direct_submit.py), so its wraps install on this torch-neuronx;
4. the NIXL this image imports is the patched build (tools/build_nixl.sh): one NIXL, the plugin's own launch rule
   passes, the LIBFABRIC plugin a NIXL agent loads carries patches/nixl/0009 (distributed/pp_seam.py), and the PD
   launch's NIXL hook wraps that binding;
5. every installed requirement holds but the two known ones (PIP_CHECK_KNOWN), and SGLang's runtime set holds;
6. SGLang's serving modules import, SGLang Diffusion's and the plugin's diffusion pipelines included
   (tools/import_sglang_entrypoints.py --diffusion);
7. the Neuron SDK parts match the release pin and the installed plugin passes the source guard (tools/neuron_release.py);
8. LD_LIBRARY_PATH has no empty entry;
9. /opt/sglang-neuron holds the checkout's public part (PUBLIC_REPO) byte for byte, root-owned and writable by root
   only, with nothing else there (the QA image's checkout aside), and its sglang_neuron/ is the installed package: the
   files the wheel ships by the copy's own pyproject.toml, byte for byte, beside the records the wheel leaves out.
Exit status 0 when every check passes, 1 otherwise.
"""
import argparse
import hashlib
import importlib
import importlib.metadata as md
import os
import stat
import subprocess
import sys
from pathlib import Path

FAILED = []
# the NIXL distributions PyPI publishes: the meta package and its per-CUDA builds; the patched build is nixl + nixl-cu13
NIXL_NAMES_SEEN_ON_PYPI = {"nixl", "nixl-cu12", "nixl-cu13"}
NIXL_VERSION = "1.5.0"  # tools/build_nixl.sh's release
# the repository's public part, which the public image ships beside the install at its repository paths (docker/
# Dockerfile's public stage; the gate allows nothing else under /opt/sglang-neuron there): files, directories and
# patterns of files, relative to the checkout. Not tests/, docker/, scripts/, the NIXL series' tests or other tools.
PUBLIC_REPO_ROOT = Path("/opt/sglang-neuron")
PUBLIC_REPO = ("README.md", "LICENSE", "pyproject.toml", "docs", "examples", "sglang_neuron", "patches/nixl/README.md",
               "patches/nixl/*.patch", "tools/glm52/build_latent_store.py")
# pip check lines the image is built to have: SGLang's pyproject pins the CUDA stack, which the plugin replaces with
# SGLang's runtime set (checked on its own below); and NIXL's meta package names both its CUDA 12 and CUDA 13 builds
# without a marker, while it imports the one that matches torch (here nixl-cu13), as upstream publishes it
PIP_CHECK_KNOWN = ("sglang ", f"nixl {NIXL_VERSION} requires nixl-cu12, which is not installed.")
# torch-neuronx without a device: no backend autoload, and the target it would otherwise read from the device
NO_DEVICE_ENV = {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn3",
                 "NEURON_LOGICAL_NC_CONFIG": "2"}


def check(name: str, ok: bool, detail="") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + str(detail) if detail != '' else ''}", flush=True)
    if not ok:
        FAILED.append(name)


def digests(base: Path) -> dict:
    """The sha256 of every file under ``base`` by its relative path, without the bytecode an import writes."""
    return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in base.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def site_packages_file(module) -> bool:
    return "site-packages" in Path(module.__file__).resolve().parts


def shipped_files(checkout: Path) -> set:
    """The paths under ``checkout``/sglang_neuron that the plugin's wheel ships, by pyproject.toml's rules: every Python
    source, and every other file package-data takes and exclude-package-data does not leave out (both as file-name
    patterns for every package, ``"*"``); the repository's own records beside the code (the kernels' PROVENANCE.md and
    file manifests) stay out."""
    import fnmatch
    import tomllib

    setuptools = tomllib.loads((checkout / "pyproject.toml").read_text())["tool"]["setuptools"]
    take, leave = setuptools.get("package-data", {}), setuptools.get("exclude-package-data", {})
    if set(take) - {"*"} or set(leave) - {"*"}:
        raise SystemExit("pyproject.toml lists package data per package: shipped_files reads only \"*\" patterns")
    root = checkout / "sglang_neuron"

    def ships(name: str) -> bool:
        if name.endswith(".py"):
            return True
        return any(fnmatch.fnmatchcase(name, p) for p in take.get("*", [])) and \
            not any(fnmatch.fnmatchcase(name, p) for p in leave.get("*", []))

    return {path.relative_to(root).as_posix() for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and ships(path.name)}


def installed_package(checkout: Path, sglang_commit: str) -> Path:
    import sglang
    import sglang_neuron

    check("sglang is an installed package", site_packages_file(sglang), sglang.__file__)
    try:
        version = md.version("sglang")
    except md.PackageNotFoundError:
        version = "no installed distribution"
    # the version the wheel build gave it; the Dockerfile's sglang stage proved the source is that commit
    check("sglang is the pinned commit", version.endswith(f"+g{sglang_commit[:7]}"), version)
    check("sglang_neuron is an installed package", site_packages_file(sglang_neuron), sglang_neuron.__file__)
    root = Path(sglang_neuron.__file__).resolve().parent
    shipped = shipped_files(checkout)
    want = {name: digest for name, digest in digests(checkout / "sglang_neuron").items() if name in shipped}
    have = digests(root)
    check(f"the install holds every file the wheel ships of the checkout's sglang_neuron/ ({len(want)})",
          want.keys() <= have.keys(), sorted(want.keys() - have.keys())[:20] or "")
    check("the install holds nothing the checkout lacks", have.keys() <= want.keys(),
          sorted(have.keys() - want.keys())[:20] or "")
    # byte for byte, so a run from the checkout (Python imports it ahead of the install) runs the shipped code
    differ = sorted(name for name in want.keys() & have.keys() if want[name] != have[name])
    check("every installed file is byte-identical to the checkout's", not differ, differ[:20] or "")
    return root


def public_repo_files(base: Path) -> dict:
    """The sha256 of every file of PUBLIC_REPO under ``base``, by its path relative to ``base``."""
    out = {}
    for name in PUBLIC_REPO:
        for top in sorted(base.glob(name)):
            paths = [top] if top.is_file() else sorted(p for p in top.rglob("*") if p.is_file())
            out.update({str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return out


def public_repo(checkout: Path, root: Path = PUBLIC_REPO_ROOT, owner: tuple = (0, 0)) -> None:
    """``root`` holds the checkout's PUBLIC_REPO byte for byte, with nothing more (the QA image's own checkout under it
    aside), every entry owned by ``owner`` and writable by it only, as the public stage's COPY lays it down: a customer
    reads and runs the docs, examples and sources of exactly the commit the image was built from."""
    skip = checkout.resolve()  # the QA image's checkout, /opt/sglang-neuron/src, is not part of the copy
    have, entries = {}, []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if (here / d).resolve() != skip)
        entries += [here / n for n in dirnames + filenames]
        for name in filenames:
            path = here / name
            digest = "" if path.is_symlink() else hashlib.sha256(path.read_bytes()).hexdigest()
            have[str(path.relative_to(root))] = digest
    want = public_repo_files(checkout)
    check(f"the repository copy holds every file of the checkout's {', '.join(PUBLIC_REPO)} ({len(want)})",
          want.keys() <= have.keys(), sorted(want.keys() - have.keys())[:20] or "")
    check(f"the repository copy under {root} holds nothing else", have.keys() <= want.keys(),
          sorted(have.keys() - want.keys())[:20] or "")
    differ = sorted(name for name in want.keys() & have.keys() if want[name] != have[name])
    check("every repository copy file is byte-identical to the checkout's", not differ, differ[:20] or "")
    bad = []
    for path in [root, *entries]:
        st = path.lstat()
        if path.is_symlink() or (st.st_uid, st.st_gid) != owner or st.st_mode & 0o022:
            bad.append(f"{path} {st.st_uid}:{st.st_gid} {stat.filemode(st.st_mode)}")
    hint = ("; build with docker/build.sh, whose export of the commit gives 0644 / 0755 whatever the umask: a docker "
            "build of a working tree copies its files' modes as they are") if bad else ""
    check(f"the repository copy is owned by {owner[0]}:{owner[1]}, writable by it only, and holds no link", not bad,
          f"{bad[:20]}{hint}" if bad else "")


def public_repo_package(installed: Path, root: Path = PUBLIC_REPO_ROOT) -> None:
    """The copy's sglang_neuron/ is the ``installed`` package: the files its own pyproject.toml has the wheel ship, file
    for file and byte for byte, so ``pip install /opt/sglang-neuron`` gives the plugin the image runs; its other files
    are the records the wheel leaves out (the kernels' manifests, PROVENANCE.md and README.md)."""
    shipped = shipped_files(root)
    copy = digests(root / "sglang_neuron")
    want = {name: digest for name, digest in copy.items() if name in shipped}
    have = digests(installed)
    differ = sorted(name for name in want.keys() & have.keys() if want[name] != have[name])
    check(f"the repository copy's sglang_neuron/ is the installed package ({len(want)} files, "
          f"{len(copy) - len(want)} records the wheel leaves out)", want == have,
          {"only in the copy": sorted(want.keys() - have.keys())[:10],
           "only installed": sorted(have.keys() - want.keys())[:10], "differ": differ[:10]} if want != have else "")


def entry_points() -> None:
    eps = [e for e in md.entry_points() if e.dist is not None and e.dist.name == "sglang-neuron"]
    check("the plugin declares entry points", bool(eps), [f"{e.group}:{e.name}" for e in eps])
    for e in eps:
        try:
            e.load()
            check(f"entry point {e.group}:{e.name} loads", True)
        except Exception as error:  # noqa: BLE001  report every failure, not the first
            check(f"entry point {e.group}:{e.name} loads", False, f"{type(error).__name__}: {error}")


def source_pins() -> None:
    import torch
    from torch_neuronx.neuron_dynamo_backend import backend, cache_artifact
    from torch_neuronx.neuron_dynamo_backend.executor import Executor

    from sglang_neuron.backend import direct_submit, executor_cache, torch_pins

    check("torch is the pinned build", torch.version.git_version == torch_pins.TORCH_GIT_VERSION,
          f"{torch.__version__} {torch.version.git_version[:8]}")
    import torch_neuronx

    # a namespace package (no __file__) is a directory named torch_neuronx found ahead of the install, not torch-neuronx
    check("torch_neuronx is the installed torch-neuronx, not a namespace package",
          getattr(torch_neuronx, "__file__", None) is not None,
          f"{md.version('torch-neuronx')} from {getattr(torch_neuronx, '__file__', None) or list(torch_neuronx.__path__)}")
    wrapped = {"torch.get_device_module": torch.get_device_module,
               "torch.distributed.init_process_group": torch.distributed.init_process_group}
    for name, sources in torch_pins.SOURCES.items():
        check(f"pinned source {name}", torch_pins.source_md5_or_empty(wrapped[name]) in sources)
    for name, want in executor_cache._SOURCES.items():
        check(f"pinned source Executor.{name}", torch_pins.source_md5_or_empty(Executor.__dict__.get(name)) == want)
    found = {"NeuronOutputCode.__call__": cache_artifact.NeuronOutputCode.__dict__.get("__call__"),
             "neuron_backend_fx_compiler": backend.neuron_backend_fx_compiler}
    for name, want in direct_submit._SOURCES.items():
        check(f"pinned source {name}", torch_pins.source_md5_or_empty(found[name]) == want)


def nixl() -> None:
    from sglang_neuron.distributed.pp_seam import loaded_plugin_drops_failed_notifications
    from sglang_neuron.models.kimi_k3.launch import nixl_problems

    import torch

    variant = f"nixl-cu{torch.version.cuda.split('.')[0]}"
    dists = sorted((d.metadata["Name"].lower(), d.version) for d in md.distributions()
                   if (d.metadata["Name"] or "").lower() in NIXL_NAMES_SEEN_ON_PYPI)
    check(f"NIXL is installed once: the patched build's nixl and {variant} {NIXL_VERSION}",
          dists == sorted([("nixl", NIXL_VERSION), (variant, NIXL_VERSION)]), dists)
    # tools/build_nixl.sh installs its wheel by name, so no build directory reaches the image's metadata
    recorded = sorted(d.metadata["Name"] for d in md.distributions()
                      if (d.metadata["Name"] or "").lower() in NIXL_NAMES_SEEN_ON_PYPI and d.read_text("direct_url.json"))
    check("NIXL records no build location (direct_url.json)", recorded == [], recorded)
    # set at all, even empty, NIXL takes it as the plugin directory and loads no plugin of its own
    check("NIXL_PLUGIN_DIR is unset", "NIXL_PLUGIN_DIR" not in os.environ, os.environ.get("NIXL_PLUGIN_DIR", ""))
    problems = nixl_problems()
    check("the plugin's NIXL launch rule passes (the patched build is the one imported)", problems == [], problems or "")
    from nixl._api import nixl_agent, nixl_agent_config

    agent = nixl_agent("sglang-neuron-image-check", nixl_agent_config(backends=[]))
    check("a NIXL agent lists the LIBFABRIC plugin", "LIBFABRIC" in agent.get_plugin_list(), agent.get_plugin_list())
    agent.get_plugin_params("LIBFABRIC")  # loads the plugin into this process
    check("the LIBFABRIC plugin this process loaded carries patches/nixl/0009",
          loaded_plugin_drops_failed_notifications() is True)
    from sglang_neuron.disaggregation import _set_nixl_progress_delay

    _set_nixl_progress_delay(250)  # what a PD role does at launch: wraps the NIXL binding's agent
    api = importlib.import_module(f"{variant.replace('-', '_')}._api")
    check("the PD launch's NIXL hook wraps the patched binding", getattr(api.nixlBind, "_neuron_progress_delay", False),
          api.__file__)


def requirements(runtime_requirements: Path) -> None:
    """Every installed distribution's requirements hold but PIP_CHECK_KNOWN's, and SGLang's runtime set
    (tools/sglang_runtime_requirements.py) holds in their place."""
    from packaging.requirements import Requirement

    out = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True).stdout.splitlines()
    others = [line for line in out if not any(line.startswith(known) for known in PIP_CHECK_KNOWN)
              and "No broken requirements" not in line]
    check("pip check: every requirement holds but the two known ones", not others, others[:20] or "")
    unmet = []
    for line in runtime_requirements.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        req = Requirement(line)
        if req.marker is not None and not req.marker.evaluate():
            continue
        try:
            version = md.version(req.name)
        except md.PackageNotFoundError:
            version = None
        if version is None or not req.specifier.contains(version, prereleases=True):
            unmet.append(f"{line} (installed {version})")
    check(f"SGLang's runtime set holds ({runtime_requirements.name})", not unmet, unmet[:20] or "")


def section(name: str, fn, *args):
    """Run one group of checks; an exception fails the group and the next group still runs."""
    try:
        return fn(*args)
    except Exception as error:  # noqa: BLE001  report every failing group, not only the first
        check(name, False, f"{type(error).__name__}: {error}")
        return None


def environment() -> None:
    entries = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    check("LD_LIBRARY_PATH has no empty entry (the loader would search the working directory)", "" not in entries,
          os.environ.get("LD_LIBRARY_PATH", ""))


def run_tool(name: str, argv: list) -> None:
    out = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
    ok = out.returncode == 0
    check(name, ok, "" if ok else (out.stdout + out.stderr)[-2000:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sglang-commit", required=True, help="the SGLang commit the image must carry")
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="the sglang-neuron checkout the image was built from (default: this file's)")
    parser.add_argument("--runtime-requirements", type=Path, required=True,
                        help="SGLang's runtime set the image installed (tools/sglang_runtime_requirements.py's output)")
    args = parser.parse_args()
    if not Path("/dev/neuron0").exists():
        for name, value in NO_DEVICE_ENV.items():
            os.environ.setdefault(name, value)
        print(f"no Neuron device: {' '.join(f'{k}={os.environ[k]}' for k in NO_DEVICE_ENV)}")
    root = section("the installed packages", installed_package, args.checkout, args.sglang_commit)
    section("the plugin's entry points", entry_points)
    section("the repository copy", public_repo, args.checkout)
    if root is not None:
        section("the repository copy's sglang_neuron/", public_repo_package, root)
    section("the torch / torch-neuronx source pins", source_pins)
    section("the patched NIXL", nixl)
    section("the requirements", requirements, args.runtime_requirements)
    section("the environment", environment)
    run_tool("SGLang's serving modules import",
             [str(args.checkout / "tools/import_sglang_entrypoints.py"), "--diffusion"])
    run_tool("the Neuron SDK parts match the release pin", [str(args.checkout / "tools/neuron_release.py"), "check"])
    if root is not None:
        run_tool("the installed plugin passes the source guard",
                 [str(args.checkout / "tools/neuron_release.py"), "guard", "source", str(root)])
    print("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
