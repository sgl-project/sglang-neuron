# SPDX-License-Identifier: Apache-2.0
"""The Neuron SDK release the plugin is validated on (``neuron_sdk.toml``): every part of it with the hashes that
identify it, and the two packages the hardware gates ran beside it (``deviations``, reported, not enforced).

A launch checks versions only, cheaply: a model with its own launch rules calls :func:`runtime_problems` for the
runtime and collectives libraries its schedulers would load (a library's version is the suffix of the file its soname
resolves to), and checks its compiler and nki against :func:`pin`. ``tools/neuron_release.py check`` hashes every
installed file once per environment."""

from __future__ import annotations

import csv
import dataclasses
import functools
import hashlib
import os
import tomllib
from typing import Iterable, Mapping, Optional

PIN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuron_sdk.toml")
# where the runtime and collectives packages install their libraries, which the image's ld.so.conf does not always list
DEFAULT_LIBRARY_DIR = "/opt/aws/neuron/lib"
# the dist-info files pip writes or rewrites at install time, which no wheel carries as they are installed
_INSTALL_TIME_FILES = frozenset({"INSTALLER", "REQUESTED", "direct_url.json", "RECORD"})


@dataclasses.dataclass(frozen=True)
class Wheel:
    name: str
    # the import name, whose __version__ torch-neuronx and the toolchain classes read
    module: str
    version: str
    url: str
    sha256: str
    # content_digest of the files the wheel installs into site-packages
    content: str


@dataclasses.dataclass(frozen=True)
class Deb:
    name: str
    # the Debian version, ``<upstream>-<build>``
    version: str
    url: str
    sha256: str
    # the soname a process loads the package's library by, None for a package with no library a launch loads
    library: Optional[str]
    # path in the package -> sha256, of the libraries and programs the offline check hashes
    files: Mapping[str, str]
    # the command a launch runs from the package (neuron-ls), None for a package with none
    program: Optional[str] = None

    @property
    def upstream_version(self) -> str:
        """The version without the build suffix, the one the package's library file names carry."""
        return self.version.split("-", 1)[0]


@dataclasses.dataclass(frozen=True)
class Deviation:
    name: str
    version: str
    reason: str
    # the release's own build of the package, where it has one
    release: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Pin:
    release: str
    notes: str
    image_reference: str
    image_digest: str
    dockerfile: str
    wheels: Mapping[str, Wheel]
    debs: Mapping[str, Deb]
    driver: Deb
    deviations: Mapping[str, Deviation]
    # the torch-neuronx release line (``2.14.3``) a launch takes, whichever build of it is installed
    torch_neuronx_line: str
    # the torch build (``2.14.0+cu130``) and its git revision, which the plugin's in-process wraps are written against
    torch_version: str
    torch_git_version: str

    @property
    def runtime_debs(self) -> tuple:
        """The packages whose library a launch's schedulers load (Deb.library set), in the pin's order."""
        return tuple(deb for deb in self.debs.values() if deb.library)


def load(path: str = PIN_FILE) -> Pin:
    """The pin in the TOML file at ``path``."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    driver = data["driver"]
    return Pin(
        release=data["release"]["name"],
        notes=data["release"]["notes"],
        image_reference=data["image"]["reference"],
        image_digest=data["image"]["digest"],
        dockerfile=data["image"]["dockerfile"],
        wheels={name: Wheel(name=name, **fields) for name, fields in data["wheels"].items()},
        debs={name: Deb(name=name, url=fields["url"], version=fields["version"], sha256=fields["sha256"],
                        library=fields.get("library"), files=dict(fields.get("files", {})),
                        program=fields.get("program"))
              for name, fields in data["debs"].items()},
        driver=Deb(name=driver["package"], version=driver["version"], url=driver["url"], sha256=driver["sha256"],
                   library=None, files={}),
        deviations={name: Deviation(name=name, **fields) for name, fields in data.get("deviations", {}).items()},
        torch_neuronx_line=data["torch-neuronx"]["line"],
        torch_version=data["torch"]["version"],
        torch_git_version=data["torch"]["git_version"],
    )


@functools.cache
def pin() -> Pin:
    """The packaged pin, read once."""
    return load()


def library_dirs(environ: Mapping[str, str]) -> list:
    """The directories a scheduler started with ``environ`` searches for a Neuron library, in order: LD_LIBRARY_PATH's
    (empty entries dropped, as the dynamic loader's are the current directory, which no launch means), then the
    packages' own install directory."""
    dirs = [d for d in environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if d]
    return dirs + ([DEFAULT_LIBRARY_DIR] if DEFAULT_LIBRARY_DIR not in dirs else [])


def resolve_library(soname: str, environ: Mapping[str, str]) -> Optional[str]:
    """The real path of the file ``soname`` resolves to in the first of library_dirs that holds it, None when none
    does."""
    for directory in library_dirs(environ):
        candidate = os.path.join(directory, soname)
        if os.path.exists(candidate):
            return os.path.realpath(candidate)
    return None


def library_version(path: str, soname: str) -> Optional[str]:
    """The version a library file's name carries after its soname's stem: ``2.35.30.0`` for ``libnrt.so.2.35.30.0``
    under ``libnrt.so.1``, and the soname's own ``1`` for a copy named the soname, which says no build; None for a file
    whose name does not start with the stem."""
    stem = soname.split(".so", 1)[0] + ".so."
    name = os.path.basename(path)
    return name[len(stem):] if name.startswith(stem) else None


def library_problem(deb: Deb, environ: Mapping[str, str], release: str) -> Optional[str]:
    """None when ``deb``'s library resolves, under ``environ``, to the pinned version, else the problem: what it found,
    and the pinned package of ``release`` to install."""
    path = resolve_library(deb.library, environ)
    found = None if path is None else library_version(path, deb.library)
    if found == deb.upstream_version:
        return None
    where = (f"no {deb.library} in {os.pathsep.join(library_dirs(environ))}" if path is None
             else f"{deb.library} resolves to {path}")
    return (f"{where}, not the {deb.name} {deb.version} of {release} the release is validated on; install it "
            f"({deb.url}) or put its lib directory first on LD_LIBRARY_PATH")


def runtime_problems(environ: Mapping[str, str], sdk: Optional[Pin] = None) -> list:
    """[] when every runtime library of the pin (Pin.runtime_debs) resolves, under ``environ``, to the pinned version,
    else one problem per library that does not (library_problem)."""
    sdk = sdk or pin()
    return [problem for deb in sdk.runtime_debs if (problem := library_problem(deb, environ, sdk.release))]


def torch_neuronx_problems(sdk: Optional[Pin] = None, paths: Optional[list] = None) -> list:
    """[] when the torch-neuronx this process imports (the first installed among ``paths``, sys.path when None) is a
    build of the pinned release's line, else the one problem naming it: the graphs are traced and compiled through it,
    and every number the release states was measured on that line. Read from the package's metadata, without importing
    it, which would claim the NeuronCores."""
    import importlib.metadata as metadata

    sdk = sdk or pin()
    where = {} if paths is None else {"path": paths}
    found = next(iter(metadata.distributions(name="torch-neuronx", **where)), None)
    if found is not None and found.version.startswith(sdk.torch_neuronx_line + "."):
        return []
    have = "no torch-neuronx is installed" if found is None else f"torch-neuronx {found.version} is installed"
    return [f"{have}, not a build of the torch-neuronx {sdk.torch_neuronx_line} line of {sdk.release} the release is "
            f"validated on; install torch-neuronx {sdk.torch_neuronx_line}"]


def wheel_entries(rows: Iterable[list], dist_info: str) -> list:
    """The ``(path, hash)`` RECORD entries of a distribution that its wheel installs into site-packages unchanged:
    every hashed entry but the files pip writes at install time (_INSTALL_TIME_FILES), compiled bytecode, the
    scripts pip rewrites into bin/ (paths out of site-packages, ``../``) and a wheel's ``.data`` directories."""
    entries = []
    for row in rows:
        if len(row) < 2 or not row[1]:
            continue
        path, digest = row[0], row[1]
        top, _, rest = path.partition("/")
        if path.startswith("../") or top.endswith(".data") or "__pycache__" in path.split("/"):
            continue
        if top == dist_info and rest in _INSTALL_TIME_FILES:
            continue
        entries.append((path, digest))
    return sorted(entries)


def content_digest(rows: Iterable[list], dist_info: str) -> str:
    """sha256 of wheel_entries, one ``path,hash`` line each: the same for a wheel's RECORD and for the RECORD pip
    writes when it installs the wheel."""
    lines = "".join(f"{path},{digest}\n" for path, digest in wheel_entries(rows, dist_info))
    return hashlib.sha256(lines.encode()).hexdigest()


def record_rows(text: str) -> list:
    """The rows of a RECORD file's text."""
    return list(csv.reader(text.splitlines()))
