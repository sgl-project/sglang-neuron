#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The pinned Neuron SDK's release checks (sglang_neuron/release), run from this checkout.

``check [--quick]``: this environment's neuronx-cc, nki, runtime, collectives, tools and torch-neuronx against the pin.
Run it with the launch's PATH, PYTHONPATH and LD_LIBRARY_PATH.

``digest [--show]``: the digest of the neuronx-cc toolchain this environment compiles with, which names a compile
cache directory; ``--show`` prints its fields too.

``guard source [ROOT]`` (sglang_neuron by default) fails on a shared object or ELF file under ROOT, and on Python code
that emits or loads a GpSimd kernel (``load_gpsimd_lib``, ``inline_asm_bytes``), imports nkilib's private package (any
import form, relative, vendored or through importlib) or names it outside a docstring, or loads a module or a shared
object by path (a system soname and ``ctypes.CDLL(None)`` pass). A ROOT that is not a directory, has a directory that
cannot be read, or holds no Python file fails. ``guard neff DIR [DIR ...]`` fails on a NEFF under the directories that
carries a GpSimd kernel table, library, shared object, ELF or ucode member, or a JSON member naming a kernel call or
library; a directory with no NEFF fails.

Exit status 0 when every check passes, 1 when one fails, 2 on a usage error."""

from __future__ import annotations

import ast
import os
import re
import sys
import tarfile
import zlib

ELF_MAGIC = b"\x7fELF"
# the GpSimd kernel loader and the inline custom-op emission
GPSIMD_EMITTERS = frozenset({"load_gpsimd_lib", "inline_asm_bytes"})
# calls that import a module by name: importlib.import_module and importlib.util.find_spec take (name, package),
# __import__ (name, globals, locals, fromlist, level)
IMPORT_CALLS = frozenset({"import_module", "find_spec", "__import__"})
# what loads a module from a file path
MODULE_FILE_LOADERS = frozenset({"spec_from_file_location", "SourceFileLoader", "SourcelessFileLoader",
                                 "ExtensionFileLoader", "zipimporter", "run_path", "load_dynamic", "load_source",
                                 "load_compiled"})
# what loads a shared object into the process
LIBRARY_LOADERS = frozenset({"CDLL", "PyDLL", "WinDLL", "OleDLL", "LoadLibrary", "dlopen", "load_library"})
# ctypes' library loaders that load the library their index names (``ctypes.cdll["/path/lib.so"]``)
LIBRARY_INDEXES = frozenset({"cdll", "pydll", "windll", "oledll"})


def _names_private_package(name: str) -> bool:
    """Whether a text names nkilib's private package: ``nkilib.private``, ``a.nkilib.private.b``,
    ``.../nkilib/private/...``, or one of them inside a larger text (``"from nkilib.private import x"``,
    ``"pkg=nkilib.private"``)."""
    parts = [p for p in re.split(r"[^\w-]+", name) if p]
    return any(a == "nkilib" and b == "private" for a, b in zip(parts, parts[1:]))


def _absolute(name: str, level: int, package: list) -> str:
    """The absolute module of a relative import of ``name`` at ``level`` from a module of ``package`` (its dotted
    parts), as far as the scanned tree reaches."""
    if not level:
        return name
    base = package[:len(package) - level + 1] if level - 1 <= len(package) else []
    return ".".join([*base, *([name] if name else [])])


def _called(func) -> str:
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


def _literal(node):
    return node.value if isinstance(node, ast.Constant) else None


def _call_args(call: ast.Call) -> dict:
    """A call's arguments by position and by keyword."""
    return {**dict(enumerate(call.args)), **{k.arg: k.value for k in call.keywords}}


def _import_call_targets(call: ast.Call, package: list) -> list:
    """The absolute modules an import call with a literal name imports: ``import_module(name, package)`` and
    ``find_spec(name, package)`` resolve leading dots against the package (the calling file's, as ``__package__``
    passes it, or a literal one); ``__import__(name, globals, locals, fromlist, level)`` imports ``name`` and each
    ``fromlist`` entry under it."""
    args = _call_args(call)
    name = _literal(args.get(0, args.get("name")))
    if not isinstance(name, str):
        return []
    if _called(call.func) != "__import__":
        level = len(name) - len(name.lstrip("."))
        anchor = _literal(args.get(1, args.get("package")))
        return [_absolute(name.lstrip("."), level, anchor.split(".") if isinstance(anchor, str) else package)]
    level = _literal(args.get(4, args.get("level")))
    module = _absolute(name, level if isinstance(level, int) else 0, package)
    fromlist = args.get(3, args.get("fromlist"))
    entries = [_literal(e) for e in fromlist.elts] if isinstance(fromlist, (ast.List, ast.Tuple)) else []
    return [module, *(f"{module}.{e}" for e in entries if isinstance(e, str))]


def _import_call_literals(call: ast.Call) -> list:
    """The literals ``_import_call_targets`` builds the targets from: a literal name, the literal package a relative
    name resolves against, and ``__import__``'s ``fromlist`` entries. With a name that is not a literal there are
    none, so every literal argument is left to the string check."""
    args = _call_args(call)
    name = args.get(0, args.get("name"))
    if not isinstance(_literal(name), str):
        return []
    if _called(call.func) == "__import__":
        fromlist = args.get(3, args.get("fromlist"))
        used = [name, *(fromlist.elts if isinstance(fromlist, (ast.List, ast.Tuple)) else ())]
    else:
        used = [name, args.get(1, args.get("package"))] if name.value.startswith(".") else [name]
    return [n for n in used if isinstance(n, ast.Constant)]


def _loads_no_path(node) -> bool:
    """Whether a library loader's argument loads nothing from a path: a system library's soname (a literal name with no
    directory) or None (the process's own symbols)."""
    if not isinstance(node, ast.Constant):
        return False
    value = node.value
    return value is None or isinstance(value, str) and bool(value) and not re.search(r"[/\\]", value)


def python_problems(path: str, source: str | bytes, package: list) -> list:
    """Every GpSimd-kernel load or emission in one Python file (see the module docstring), ``package`` the dotted parts
    of the package it belongs to (relative imports resolve against it)."""
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError) as e:
        return [f"{path}: does not parse, so it cannot be checked ({e})"]
    # strings checked with the node that holds them, or never: a docstring or other bare string runs nothing, an
    # f-string's literal parts are checked as the whole f-string, and the literals an import call resolves as its target
    skip, pathless_loads = set(), set()
    # the library loaders and the names a from-import binds them to (``from ctypes import CDLL as C``)
    loaders = LIBRARY_LOADERS | {a.asname for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                                 for a in node.names if a.name in LIBRARY_LOADERS and a.asname}
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Constant, ast.JoinedStr)):
            skip.add(id(node.value))
        elif isinstance(node, ast.JoinedStr):
            skip.update(id(v) for v in node.values)
        elif isinstance(node, ast.Call) and _called(node.func) in IMPORT_CALLS:
            skip.update(id(n) for n in _import_call_literals(node))
        elif isinstance(node, ast.Call) and _called(node.func) in loaders:
            if node.args and _loads_no_path(node.args[0]):
                pathless_loads.add(id(node.func))
        # a loader named as a type (``def _libc() -> ctypes.CDLL``, ``lib: ctypes.CDLL``) loads nothing
        annotations = [node.returns, *(a.annotation for a in ast.walk(node.args) if isinstance(a, ast.arg))] \
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else \
            [node.annotation] if isinstance(node, ast.AnnAssign) else []
        pathless_loads.update(id(n) for a in annotations if a is not None for n in ast.walk(a))
    problems = []
    for node in ast.walk(tree):
        where = f"{path}:{getattr(node, 'lineno', 0)}"
        if isinstance(node, (ast.Attribute, ast.Name)):
            name = node.attr if isinstance(node, ast.Attribute) else node.id
            if name in GPSIMD_EMITTERS:
                problems.append(f"{where}: {name}")
            elif name in MODULE_FILE_LOADERS:
                problems.append(f"{where}: {name} loads a module from a file path")
            elif name in loaders and id(node) not in pathless_loads:
                problems.append(f"{where}: {name} loads a shared object by path")
        elif isinstance(node, ast.Import):
            problems += [f"{where}: import {a.name}" for a in node.names if _names_private_package(a.name)]
        elif isinstance(node, ast.ImportFrom):
            module = _absolute(node.module or "", node.level, package)
            written = f"from {'.' * node.level}{node.module or ''} import"
            for a in node.names:
                if _names_private_package(f"{module}.{a.name}"):
                    problems.append(f"{where}: {written} {a.name} ({module}.{a.name})")
                elif a.name in GPSIMD_EMITTERS or a.name in MODULE_FILE_LOADERS:
                    problems.append(f"{where}: {written} {a.name}")
        elif isinstance(node, ast.Call) and _called(node.func) in IMPORT_CALLS:
            problems += [f"{where}: {_called(node.func)} of {m}" for m in _import_call_targets(node, package)
                         if _names_private_package(m)]
        elif isinstance(node, ast.Subscript) and _called(node.value) in LIBRARY_INDEXES:
            if not _loads_no_path(node.slice):
                problems.append(f"{where}: {_called(node.value)}[...] loads a shared object by path")
        elif isinstance(node, (ast.Constant, ast.JoinedStr)) and id(node) not in skip:
            if isinstance(node, ast.JoinedStr):  # its literal parts around a placeholder for each value
                text = "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
            elif isinstance(node.value, str):
                text = node.value
            else:
                continue
            # a relative module name, as a vendored copy writes one, resolved against the file's package
            level = len(text) - len(text.lstrip("."))
            relative = level and _names_private_package(_absolute(text[level:], level, package))
            if _names_private_package(text) or relative:
                problems.append(f"{where}: a string naming nkilib's private package ({text!r})")
            elif GPSIMD_EMITTERS & set(re.split(r"[./\\]", text)):
                problems.append(f"{where}: a probe of {text!r}")
    return problems


def source_problems(root: str) -> list:
    """Every shared object / ELF file and every GpSimd-kernel load or emission under ``root``. A root that is not a
    directory, a directory that cannot be read and a tree with no Python file are problems too: they went unchecked."""
    if not os.path.isdir(root):
        return [f"{root}: not a directory"]
    problems, checked = [], 0

    def unreadable(e: OSError):
        problems.append(f"{e.filename}: cannot be read, so it cannot be checked ({e})")

    top = os.path.dirname(os.path.abspath(root))
    for dirpath, _, files in os.walk(root, onerror=unreadable):
        package = os.path.relpath(os.path.abspath(dirpath), top).split(os.sep)
        for name in sorted(files):
            path = os.path.join(dirpath, name)
            if name.endswith(".so") or ".so." in name:
                problems.append(f"{path}: a shared object in the package")
                continue
            try:
                with open(path, "rb") as f:
                    data = f.read() if name.endswith(".py") else f.read(4)
            except OSError as e:
                problems.append(f"{path}: cannot be read, so it cannot be checked ({e})")
                continue
            if name.endswith(".py"):
                problems += python_problems(path, data, package)
                checked += 1
            elif data == ELF_MAGIC:
                problems.append(f"{path}: an ELF binary in the package")
    if not checked:
        problems.append(f"{root}: no Python file under this directory")
    return problems


# A NEFF (a 1024-byte header, then a tar.gz) that carries a custom GpSimd kernel carries it as a member: the runtime
# loads every library a NEFF's kelf names under ``gpsimd_kernels_def`` (each matched against
# NEURON_RT_GPSIMD_LIB_OVERRIDES), and the compiler records a NEFF's custom Pool kernels in pool_kernel_info.json. A
# NEFF of stock NKI holds only the standard members: neff.json, info.json, metrics.json, hlo_stats.json, kelf-N.json,
# sgN/{def, act_info, <engine>N}.json and .bin, the default opcode / datapath / control tables, the activation tables
# and the debug tables.
NEFF_HEADER_BYTES = 1024
FORBIDDEN_MEMBER = re.compile(r"pool_kernel_info|gpsimd_kernels_def|\.so$|\.so\.|\.elf$|gpsimd.*\.bin$|ucode", re.I)
# what names a custom kernel's call, table, library or op in a NEFF's JSON member (the module docstring lists them)
FORBIDDEN_JSON_STRINGS = (b"KernelCall", b"load_gpsimd_lib", b"gpsimd_kernels_def", b"custom_op")


def neff_member_problems(path: str) -> list:
    """Every member of one NEFF that would carry a custom GpSimd kernel: a member named like one (FORBIDDEN_MEMBER), a
    member that is an ELF binary, or a JSON member that names a kernel call or library (FORBIDDEN_JSON_STRINGS). The
    members stream in archive order, so a NEFF of any size is read once."""
    problems = []
    try:
        with open(path, "rb") as f:
            f.seek(NEFF_HEADER_BYTES)
            with tarfile.open(fileobj=f, mode="r|gz") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    if FORBIDDEN_MEMBER.search(member.name):
                        problems.append(f"{path}: member {member.name}")
                        continue
                    body = tf.extractfile(member).read()
                    if body[:4] == ELF_MAGIC:
                        problems.append(f"{path}: member {member.name} is an ELF binary")
                    elif member.name.endswith(".json"):
                        problems += [f"{path}: member {member.name} names {word.decode()}"
                                     for word in FORBIDDEN_JSON_STRINGS if word in body]
    except (tarfile.TarError, EOFError, OSError, zlib.error) as e:
        problems.append(f"{path}: not a readable NEFF, so it cannot be checked ({e})")
    return problems


def neff_problems(dirs) -> tuple:
    """(problems, NEFFs checked) over every NEFF under each directory of ``dirs``; a directory with none is a
    problem."""
    problems, checked = [], 0
    for neff_dir in dirs:
        neffs = sorted(os.path.join(d, f) for d, _, fs in os.walk(neff_dir) for f in fs if f.endswith(".neff"))
        if not neffs:
            problems.append(f"{neff_dir}: no NEFF under this directory")
        for path in neffs:
            problems += neff_member_problems(path)
        checked += len(neffs)
    return problems, checked


def guard(argv) -> int:
    if argv[:1] == ["source"] and len(argv) <= 2:
        root = argv[1] if len(argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                         "sglang_neuron")
        problems = source_problems(root)
        print("\n".join(problems) or f"source: clean ({root})")
        return 1 if problems else 0
    if argv[:1] == ["neff"] and len(argv) > 1:
        problems, n = neff_problems(argv[1:])
        print("\n".join(problems) or f"neff: {n} NEFF(s) clean ({', '.join(argv[1:])})")
        return 1 if problems else 0
    print(__doc__)
    return 2


def main(argv) -> int:
    command, rest = (argv[0], argv[1:]) if argv else (None, [])
    if command == "guard":
        return guard(rest)
    if command not in ("check", "digest") or rest not in ([], ["--quick"] if command == "check" else ["--show"]):
        print(__doc__)
        return 2
    # the package from this checkout, as a launch from it imports it
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if command == "check":
        from sglang_neuron.release import check

        return check.run(quick=rest == ["--quick"])
    import dataclasses
    import json

    from sglang_neuron.release import toolchain

    tc = toolchain.compiler_toolchain()
    if rest:
        print(json.dumps({**dataclasses.asdict(tc), "same_install": tc.same_install,
                          "digest": toolchain.toolchain_digest(tc)}, indent=1))
    else:
        print(toolchain.toolchain_digest(tc))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
