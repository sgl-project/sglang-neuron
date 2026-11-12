"""Print the requirements scripts/setup_sglang_main_env.sh installs: SGLang's runtime set at SGLang's own pins.

Usage: python tools/sglang_runtime_requirements.py <sglang>/python

The set is ``runtime_common`` in ``pyproject_other.toml``: ``runtime_base``, the torch-free list SGLang's
``srt_empty`` installs on platforms with their own torch, and the few torch-dependent packages SGLang's runtime adds
to it. From that set it drops the packages only optional serving modes import, and adds the packages SGLang imports
that only ``pyproject.toml`` declares and ``dill``, which SGLang imports but does not declare. Each package keeps its
lines from ``pyproject_other.toml``, the pins SGLang's platforms with their own torch install, or ``pyproject.toml``'s
for the packages ``pyproject_other.toml`` does not list; a package neither file pins installs at pip's current release.
Reads only the two files and uses only the standard library, so it runs before anything is installed.
"""
import re
import sys
import tomllib
from pathlib import Path

# Imported only by serving modes the plugin's HTTP server does not load: video input, ModelScope downloads, py-spy
# dumps, the gRPC server, the outlines grammar backend and timm vision towers.
OPTIONAL = {"av", "deepseek-recipe", "modelscope", "py-spy", "smg-grpc-servicer", "outlines", "timm"}
# Imported by SGLang and declared only in pyproject.toml: transformers' tokenizers at SGLang's pin, the TLS
# certificate reload (watchfiles) and zstd request bodies (zstandard).
FROM_PYPROJECT = ("tokenizers", "watchfiles", "zstandard")
# Imported by sglang/srt/sampling/custom_logit_processor.py, declared in neither file.
UNDECLARED = ("dill",)

_REQUIREMENT = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?")


def _name(line):
    """The PEP 503 normalized name and the extras of a PEP 508 requirement line."""
    match = _REQUIREMENT.match(line)
    if match is None:
        raise ValueError(f"not a requirement: {line!r}")
    extras = [e.strip() for e in (match.group(2) or "").split(",") if e.strip()]
    return re.sub(r"[-_.]+", "-", match.group(1)).lower(), extras


def _extra(groups, extra):
    """The requirement lines of ``extra``, with the extras of SGLang it names expanded in place."""
    lines = []
    for line in groups[extra]:
        name, extras = _name(line)
        if name == "sglang":
            for nested in extras:
                lines += _extra(groups, nested)
        else:
            lines.append(line)
    return lines


def _by_name(lines):
    grouped = {}
    for line in lines:
        grouped.setdefault(_name(line)[0], []).append(line)
    return grouped


def requirements(python_dir):
    python_dir = Path(python_dir)
    other = tomllib.loads((python_dir / "pyproject_other.toml").read_text())["project"]
    declared = _by_name(tomllib.loads((python_dir / "pyproject.toml").read_text())["project"]["dependencies"])
    runtime = _by_name(_extra(other["optional-dependencies"], "runtime_common"))
    for name in FROM_PYPROJECT:
        if name not in declared:
            raise SystemExit(f"{name} is no longer in pyproject.toml; check SGLang's imports of it")
    names = (set(runtime) - OPTIONAL) | set(FROM_PYPROJECT) | set(UNDECLARED)
    return [line for name in sorted(names) for line in runtime.get(name) or declared.get(name) or [name]]


if __name__ == "__main__":
    print("\n".join(requirements(sys.argv[1])))
