# SPDX-License-Identifier: Apache-2.0
"""This environment's Neuron SDK against the pin, file by file (``tools/neuron_release.py check [--quick]``).

:func:`run` prints one line per part, ``ok`` or what differs, and returns 1 if any part differs: neuronx-cc and nki by
version, RECORD digest and (unless quick) every file's hash; the runtime and collectives libraries by the file each
soname resolves to; neuron-ls by version and hash; torch-neuronx by its release line. The host driver and the
deviations are printed for information. Run it with the launch's PATH, PYTHONPATH and LD_LIBRARY_PATH; it imports
neither torch nor torch-neuronx, so it can run beside a server."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata as metadata
import importlib.util
import os
import shutil
import subprocess
from typing import Mapping, Optional

from sglang_neuron.release import neuron_sdk

DRIVER_VERSION_FILE = "/sys/module/neuron/version"
# how long a program has to answer --version
VERSION_TIMEOUT_S = 60


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_hash_matches(path: str, record_hash: str) -> bool:
    """Whether the file at ``path`` matches a RECORD hash (``<algorithm>=<urlsafe base64, unpadded>``)."""
    algorithm, _, want = record_hash.partition("=")
    digest = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode() == want


def installed_distribution(wheel: neuron_sdk.Wheel):
    """The distribution of ``wheel.module`` that an import in this environment finds (the first on sys.path that holds
    the module), found without importing it; None when the module is not importable or has no metadata there."""
    spec = importlib.util.find_spec(wheel.module)
    if spec is None:
        return None
    package_dir = spec.submodule_search_locations[0] if spec.submodule_search_locations else spec.origin
    site = os.path.dirname(package_dir)
    return next(iter(metadata.distributions(name=wheel.name, path=[site])), None)


def check_wheel(wheel: neuron_sdk.Wheel, quick: bool, dist=None) -> list:
    """The differences of the installed ``wheel`` from the pin ([] for none), on the distribution an import finds
    (installed_distribution) unless ``dist`` is given."""
    dist = dist or installed_distribution(wheel)
    if dist is None:
        return [f"{wheel.name} is not installed"]
    problems = []
    if dist.version != wheel.version:
        problems.append(f"{wheel.name} {dist.version} is installed, not {wheel.version}")
    record = dist.read_text("RECORD")
    if record is None:
        return problems + [f"{wheel.name} has no RECORD, so its files cannot be identified"]
    rows = neuron_sdk.record_rows(record)
    dist_info = next((row[0].split("/", 1)[0] for row in rows if row and row[0].endswith(".dist-info/RECORD")), None)
    if dist_info is None:
        # pip always lists the RECORD itself; without that row the RECORD is damaged
        return problems + [f"{wheel.name}'s RECORD lists no .dist-info/RECORD entry, so its files cannot be "
                           "identified"]
    if neuron_sdk.content_digest(rows, dist_info) != wheel.content:
        problems.append(f"{wheel.name}'s installed files are not the pinned wheel's ({wheel.url}): their RECORD "
                        "entries differ")
    if not quick:
        site = str(dist.locate_file(""))
        changed = [path for path, digest in neuron_sdk.wheel_entries(rows, dist_info)
                   if not os.path.isfile(os.path.join(site, path))
                   or not _record_hash_matches(os.path.join(site, path), digest)]
        if changed:
            problems.append(f"{len(changed)} of {wheel.name}'s installed files differ from its RECORD "
                            f"(first: {changed[0]})")
    return problems


def check_runtime(deb: neuron_sdk.Deb, environ: Mapping[str, str], release: str) -> list:
    """The differences of the library ``deb`` provides, as ``environ`` resolves it, from the pin: its version
    (neuron_sdk.library_problem) and each pinned library file of the package in the directory it resolves to."""
    problem = neuron_sdk.library_problem(deb, environ, release)
    path = neuron_sdk.resolve_library(deb.library, environ)
    problems = [problem] if problem else []
    if path is None:
        return problems
    directory = os.path.dirname(path)
    for packaged, want in sorted(deb.files.items()):
        installed = os.path.join(directory, os.path.basename(packaged))
        if not os.path.isfile(installed):
            problems.append(f"{deb.name}: no {os.path.basename(packaged)} beside {path}")
        elif file_sha256(installed) != want:
            problems.append(f"{deb.name}: {installed} is not the pinned package's file")
    return problems


def check_program(deb: neuron_sdk.Deb, environ: Mapping[str, str]) -> list:
    """The differences of the program ``deb`` provides, the one PATH resolves first, from the pin. A program that
    hangs or cannot run is named as a difference, and its file is still checked against the pin."""
    found = shutil.which(deb.program, path=environ.get("PATH"))
    if found is None:
        return [f"{deb.name}: no {deb.program} on PATH"]
    problems = []
    try:
        done = subprocess.run([found, "--version"], capture_output=True, text=True, timeout=VERSION_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        problems.append(f"{deb.name}: {found} --version did not finish in {VERSION_TIMEOUT_S} s")
    except OSError as error:
        problems.append(f"{deb.name}: {found} cannot run ({error})")
    else:
        words = (done.stdout or done.stderr).split()
        reported = words[1].split("%", 1)[0] if len(words) > 1 else "nothing"
        if reported != deb.upstream_version:
            problems.append(f"{deb.name}: {found} reports {reported}, not {deb.upstream_version}")
    pinned = [want for packaged, want in deb.files.items() if os.path.basename(packaged) == deb.program]
    if pinned and file_sha256(os.path.realpath(found)) != pinned[0]:
        problems.append(f"{deb.name}: {found} is not the pinned package's {deb.program}")
    return problems


def deviation_note(deviation: neuron_sdk.Deviation, paths: Optional[list] = None) -> str:
    """How the installed package of a deviation (among ``paths``, sys.path when None) compares with the build the
    hardware gates ran: ``the gates' build`` or what is installed instead."""
    where = {} if paths is None else {"path": paths}
    found = sorted({d.version for d in metadata.distributions(name=deviation.name, **where)})
    if found == [deviation.version]:
        return "the build the hardware gates ran"
    return f"{', '.join(found) or 'not installed'} here; the hardware gates ran {deviation.version}"


def driver_version(path: str = DRIVER_VERSION_FILE) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def run(quick: bool = False) -> int:
    """Check every part of the pin and print the result; 1 if any part differs. ``quick`` skips hashing every
    installed file of the wheels."""
    sdk = neuron_sdk.pin()
    environ = os.environ
    print(f"{sdk.release}: {sdk.image_reference}@{sdk.image_digest}")
    results = []
    for wheel in sdk.wheels.values():
        results.append((f"{wheel.name} {wheel.version}", check_wheel(wheel, quick)))
    for deb in sdk.debs.values():
        problems = []
        if deb.library:
            problems += check_runtime(deb, environ, sdk.release)
        if deb.program:
            problems += check_program(deb, environ)
        results.append((f"{deb.name} {deb.version}", problems))
    results.append((f"torch-neuronx {sdk.torch_neuronx_line}", neuron_sdk.torch_neuronx_problems(sdk)))
    for name, problems in results:
        print(f"  {name}: " + ("ok" if not problems else "; ".join(problems)))
    for deviation in sdk.deviations.values():
        print(f"  {deviation.name} (information): {deviation_note(deviation)}")
    driver = driver_version()
    print(f"  host {sdk.driver.name} {sdk.driver.version}: the node runs {driver or 'no Neuron driver'}"
          + (" (the node's, not the container's)" if driver != sdk.driver.version else ""))
    return 1 if any(problems for _, problems in results) else 0
