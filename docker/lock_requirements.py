# SPDX-License-Identifier: Apache-2.0
"""Write the image's dependency locks, exact and hashed, so a build installs the same files every time and pip
resolves nothing. Run by docker/lock_requirements.sh inside the Dockerfile's base image, with network access.

- requirements-lock.txt: what SGLang's runtime set (tools/sglang_runtime_requirements.py) and SGLang Diffusion's
  packages (docker/diffusion-requirements.txt) add to the base;
- qa-requirements-lock.txt: what docker/qa-requirements.txt (the QA image's own requirements) adds on top of those;
- torchvision-lock.txt: torchvision's CPU build;
- wheel-build-lock.txt: the tools the two wheels are built with (docker/wheel-build-requirements.txt), resolved in an
  empty venv, as the Dockerfile's sglang-wheel stage installs them.

A lock is regenerated, not edited: every package keeps the version its current lock names, so a rerun resolves only
what changed (a requirement added, or SGLang's own exact pin moved with its commit). --upgrade-package <name> lets that
package move to its newest release the other locked versions allow (name each dependency that must move with it too),
--upgrade every package. The base image's packages are
not touched (pip installs over them only where a requirement needs another version), except the ones BASE_CHANGES names
and explains: any other change of a base package fails here, before it reaches a lock.

Each line is name==version with the sha256 of the one file pip picks for this base; the Dockerfile installs them
with --no-deps --require-hashes.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename

PIP = [sys.executable, "-m", "pip"]
# the base packages the image installs another version of, and why
BASE_CHANGES = {
    "tokenizers": "SGLang pins its own tokenizers",
    "huggingface-hub": "the tokenizers SGLang pins needs a newer huggingface-hub than the container's",
}


def run(*argv) -> None:
    subprocess.run(argv, check=True)


def adds(*install_args, pip=PIP) -> list:
    """name==version of every package `pip install <install_args>` would add or change in ``pip``'s environment (this
    one by default). A resolution that fails ends the run with pip's own explanation and the locked versions that
    block it (resolution_error)."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        out = subprocess.run([*pip, "install", "--dry-run", "--report", str(report), *install_args],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(resolution_error(out.stdout + out.stderr))
        return sorted(f"{p['metadata']['name']}=={p['metadata']['version']}"
                      for p in json.loads(report.read_text())["install"])


def resolution_error(pip_output: str) -> str:
    """What a failed resolution tells the caller: pip's own explanation ("The conflict is caused by: ...", which pip's
    --quiet would drop), and, for each locked version kept() made a constraint that blocks it, the --upgrade-package
    that lets it move. A rerun after moving SGLang's commit fails this way when a new pin needs a newer version of a
    locked package."""
    blocking = sorted(set(re.findall(r"The user requested \(constraint\) ([A-Za-z0-9._-]+)==", pip_output)))
    text = f"lock_requirements: pip cannot resolve the requirements:\n{pip_output.strip()[-6000:]}"
    if blocking:
        text += (f"\nlocked versions block it ({', '.join(blocking)}): rerun with "
                 + " ".join(f"--upgrade-package {name}" for name in blocking))
    return text


def hashed(requirements: list, *download_args) -> list:
    """name==version --hash=sha256:... for the one file pip downloads for each requirement on this platform: its wheel,
    or its source archive where the release publishes no wheel. The version is the file's own, so a local version
    (torchvision's +cpu) is pinned exactly."""
    with tempfile.TemporaryDirectory() as tmp:
        run(*PIP, "download", "--quiet", "--no-deps", "--prefer-binary", "-d", tmp, *download_args, *requirements)
        wheels = {}
        for path in sorted(Path(tmp).iterdir()):
            if path.name.endswith(".whl"):
                name, version, _, _ = parse_wheel_filename(path.name)
            else:
                name, version = parse_sdist_filename(path.name)
            if canonicalize_name(name) in wheels:
                raise SystemExit(f"pip downloaded two files of {name}")
            wheels[canonicalize_name(name)] = (name, version, hashlib.sha256(path.read_bytes()).hexdigest())
    names = {canonicalize_name(r.split("==")[0]) for r in requirements}
    if names != wheels.keys():
        raise SystemExit(f"requirements {sorted(names)} but wheels {sorted(wheels)}")
    return [f"{name}=={version} --hash=sha256:{digest}" for name, version, digest in
            (wheels[n] for n in sorted(wheels))]


def requirement_lines(path: Path) -> list:
    """The requirement lines of a requirements file, without its comments and blank lines."""
    return [line.split("#", 1)[0].strip() for line in path.read_text().splitlines() if line.split("#", 1)[0].strip()]


def locked(lock: Path) -> dict:
    """{name: version} a lock file names; empty when there is none yet."""
    if not lock.is_file():
        return {}
    pins = {}
    for line in lock.read_text().splitlines():
        match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", line)
        if match:
            pins[canonicalize_name(match.group(1))] = match.group(2)
    return pins


def exact(requirement_lines: list) -> set:
    """The names the requirement lines pin exactly (==): their own pin, not the lock's, decides their version."""
    names = set()
    for line in requirement_lines:
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            req = Requirement(line)
            if any(spec.operator == "==" for spec in req.specifier):
                names.add(canonicalize_name(req.name))
    return names


def kept(pins: dict, requirement_lines: list, upgrade: bool, upgrade_packages: set) -> list:
    """The constraint lines that hold each locked package at its locked version, but for the requirements' own exact
    pins, the packages to upgrade, and every package under --upgrade."""
    if upgrade:
        return []
    free = exact(requirement_lines) | {canonicalize_name(n) for n in upgrade_packages}
    return [f"{name}=={version}" for name, version in sorted(pins.items()) if name not in free]


def base_changes(added: list, base: dict) -> list:
    """The lines of ``added`` (name==version) that would install another version of a base package than BASE_CHANGES
    allows, as messages."""
    out = []
    for line in added:
        name, version = line.split("==", 1)
        key = canonicalize_name(name)
        if key in base and key not in BASE_CHANGES:
            out.append(f"{name} {base[key]} -> {version}: the image would replace the base's {name}; if it must, "
                       "add it to BASE_CHANGES with the reason")
    return out


def resolve(requirement_lines: list, lock: Path, upgrade: bool, upgrade_packages: set, *constraints: Path) -> list:
    """name==version of what ``requirement_lines`` add to this environment, under ``constraints`` and the lock's
    kept pins."""
    with tempfile.TemporaryDirectory() as tmp:
        reqs, pins = Path(tmp) / "requirements.txt", Path(tmp) / "kept.txt"
        reqs.write_text("\n".join(requirement_lines) + "\n")
        pins.write_text("\n".join(kept(locked(lock), requirement_lines, upgrade, upgrade_packages)) + "\n")
        args = ["-c", str(pins), "-r", str(reqs)]
        for constraint in constraints:
            args = ["-c", str(constraint), *args]
        return adds(*args)


def sglang_version(sglang: Path, scratch: Path) -> str:
    """The version SGLang's own build gives the checked-out commit: setuptools-scm (the Dockerfile's sglang-wheel
    version) with SGLang's git_describe_command, over the commit's whole history and every tag (fetched without file
    contents). The image's wheel is built from a depth-1 fetch, which cannot compute it, so the Dockerfile states it."""
    run("git", "-C", str(sglang), "fetch", "--quiet", "--filter=blob:none", "--unshallow", "--tags",
        "https://github.com/sgl-project/sglang")
    venv = scratch / "scm"
    run(sys.executable, "-m", "venv", str(venv))
    run(str(venv / "bin" / "pip"), "install", "--quiet", "setuptools-scm==10.3.4")
    return subprocess.run([str(venv / "bin" / "python"), "-m", "setuptools_scm"], cwd=sglang / "python", check=True,
                          capture_output=True, text=True).stdout.strip()


def version_problem(computed: str, declared: str, commit: str):
    """Why the Dockerfile's SGLANG_VERSION is not the one SGLang's versioning computes for ``commit``, or None. The
    release part must be equal; the local part names the commit (setuptools-scm abbreviates it to its own length)."""
    release, _, local = declared.partition("+")
    if release != computed.partition("+")[0] or not (local.startswith("g") and commit.startswith(local[1:])):
        return (f"docker/Dockerfile's SGLANG_VERSION={declared} is not SGLang's own version of {commit[:8]} ({computed}): "
                f"set it to {computed.partition('+')[0]}+g{commit[:7]}")
    return None


def check_sglang_version(sglang: Path, commit: str, declared: str, scratch: Path) -> None:
    problem = version_problem(sglang_version(sglang, scratch), declared, commit)
    if problem:
        raise SystemExit(problem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, required=True, help="the sglang-neuron checkout")
    parser.add_argument("--sglang-commit", required=True)
    parser.add_argument("--sglang-version", required=True, help="the Dockerfile's SGLANG_VERSION, checked here")
    parser.add_argument("--torchvision", required=True, help="torchvision's version")
    parser.add_argument("--torchvision-index", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upgrade", action="store_true", help="resolve every package again, ignoring the locks")
    parser.add_argument("--upgrade-package", action="append", default=[], metavar="NAME",
                        help="let this package move to its newest allowed release (repeatable)")
    args = parser.parse_args()
    docker = args.repo / "docker"
    upgrade_packages = set(args.upgrade_package)
    base = {canonicalize_name(d.metadata["Name"]): d.version for d in md.distributions() if d.metadata["Name"]}
    with tempfile.TemporaryDirectory() as tmp:
        sglang = Path(tmp) / "sglang"
        run("git", "init", "--quiet", str(sglang))
        run("git", "-C", str(sglang), "fetch", "--quiet", "--depth", "1", "https://github.com/sgl-project/sglang",
            args.sglang_commit)
        run("git", "-C", str(sglang), "checkout", "--quiet", "--detach", "FETCH_HEAD")
        check_sglang_version(sglang, args.sglang_commit, args.sglang_version, Path(tmp))
        runtime = subprocess.run(
            [sys.executable, str(args.repo / "tools/sglang_runtime_requirements.py"), str(sglang / "python")],
            check=True, capture_output=True, text=True).stdout.splitlines()
        runtime += requirement_lines(docker / "diffusion-requirements.txt")
        added = resolve(runtime, docker / "requirements-lock.txt", args.upgrade, upgrade_packages)
        problems = base_changes(added, base)
        if problems:
            raise SystemExit("\n".join(problems))
        public = hashed(added)
        lock = args.out / "requirements-lock.txt"
        lock.write_text("\n".join(public) + "\n")
        # the qa target installs on top of the public image: resolve the tests' requirements against the public set,
        # which they may add to but not change
        run(*PIP, "install", "--quiet", "--no-deps", "--require-hashes", "-r", str(lock))
        frozen = Path(tmp) / "public.txt"
        frozen.write_text(subprocess.run([*PIP, "list", "--format=freeze"], check=True, capture_output=True,
                                         text=True).stdout)
        qa = hashed(resolve(requirement_lines(docker / "qa-requirements.txt"), docker / "qa-requirements-lock.txt",
                            args.upgrade, upgrade_packages, frozen))
        (args.out / "qa-requirements-lock.txt").write_text("\n".join(qa) + "\n")
    torchvision = hashed([f"torchvision=={args.torchvision}"], "--index-url", args.torchvision_index)
    (args.out / "torchvision-lock.txt").write_text("\n".join(torchvision) + "\n")
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        run(sys.executable, "-m", "venv", str(venv))
        lines = requirement_lines(docker / "wheel-build-requirements.txt")
        pins = kept(locked(docker / "wheel-build-lock.txt"), lines, args.upgrade, upgrade_packages)
        constraints = Path(tmp) / "kept.txt"
        constraints.write_text("\n".join(pins) + "\n")
        tools = adds("-c", str(constraints), *lines, pip=[str(venv / "bin" / "python"), "-m", "pip"])
    (args.out / "wheel-build-lock.txt").write_text("\n".join(hashed(tools)) + "\n")


if __name__ == "__main__":
    main()
