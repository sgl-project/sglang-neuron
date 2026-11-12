# SPDX-License-Identifier: Apache-2.0
"""Source checks for the patch modules' upstream targets.

A patch module pins every upstream symbol it touches by the md5 of its source and refuses to register when the md5
differs. ``with_substitutions`` rebuilds an upstream function from its own source with a few exact text replacements,
for a module that changes only a few lines of a long function."""

from __future__ import annotations

import __future__
import ast
import difflib
import functools
import hashlib
import importlib
import inspect
import json
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Mapping, Tuple

#: The pins of the patch modules that keep theirs in one file rather than inline: ``<module>:<qualname>`` -> md5,
#: written by ``tools/pin_upstream_sources.py --write`` from the SGLang tree the plugin serves.
PINS_FILE = Path(__file__).with_name("upstream_pins.json")
#: The modules whose pins ``PINS_FILE`` holds (this change's patch modules and the GLM-5.2 prefill pool), each with the
#: names of its (module, qualified name) target tuples.
PINNED_MODULES = {
    "sglang_neuron.sglang_patches.dsa_platform_adjustments": ("_TARGETS", "_OVERRIDE_TARGETS", "_SKIPPED_TARGETS"),
    "sglang_neuron.sglang_patches.kernel_kv_handoff": ("DECODE_TARGETS", "PREFILL_TARGETS", "_RELIED_ON"),
    "sglang_neuron.sglang_patches.nixl_await_poll": ("PINNED",),
    "sglang_neuron.sglang_patches.pp_prefill_compute_ahead": ("_WRAPPED", "_RELIED_ON"),
    "sglang_neuron.sglang_patches.spec_v2_host_alloc": ("_TARGETS",),
    "sglang_neuron.models.glm52.prefill.kv_pool": ("_UPSTREAM_TARGETS",),
}


def source_md5(obj) -> str:
    """md5 of ``obj``'s dedented source, as inspect reads it."""
    return hashlib.md5(textwrap.dedent(inspect.getsource(obj)).encode()).hexdigest()


def check(obj, expected_md5: str, name: str, patch: str) -> None:
    """Raise unless ``obj``'s source is the one patch module ``patch`` was written for."""
    got = source_md5(obj)
    if got != expected_md5:
        raise RuntimeError(f"{name}: its source (md5 {got}) is not the upstream SGLang source {patch} wraps "
                           f"(md5 {expected_md5}); re-validate {patch} against this SGLang before serving it")


def resolve(module_name: str, qualname: str):
    """Look up ``qualname`` in module ``module_name``; a method comes back as its plain, unbound function.

    At each class the class's own ``__dict__`` entry is used, so a staticmethod or classmethod resolves to its
    function. A name the class does not define itself, such as an inherited method, goes through getattr."""
    obj = importlib.import_module(module_name)
    for name in qualname.split("."):
        obj = obj.__dict__[name] if isinstance(obj, type) and name in obj.__dict__ else getattr(obj, name)
    return getattr(obj, "__func__", obj)


@functools.lru_cache(maxsize=None)
def pins() -> Mapping[str, str]:
    """``PINS_FILE``'s pins, read once."""
    return json.loads(PINS_FILE.read_text())["targets"]


def pinned(module_name: str, qualname: str) -> str:
    """The md5 ``PINS_FILE`` records for an upstream target."""
    key = f"{module_name}:{qualname}"
    if key not in pins():
        raise RuntimeError(f"{key}: no pin in {PINS_FILE.name}; add the target to its patch module and run "
                           "tools/pin_upstream_sources.py --write against the served SGLang")
    return pins()[key]


def check_pinned(targets: Iterable[Tuple[str, str]], patch: str) -> None:
    """``check_targets`` over (module, qualified name) targets whose md5s ``PINS_FILE`` records."""
    check_targets({target: pinned(*target) for target in targets}, patch)


def check_targets(targets_md5: Mapping[Tuple[str, str], str], patch: str) -> None:
    """Run ``check`` on every (module, qualified name) key of ``targets_md5``; the error names the target."""
    for (module_name, qualname), md5 in targets_md5.items():
        check(resolve(module_name, qualname), md5, f"{module_name}.{qualname}", patch)


def with_substitutions(fn, substitutions, decorators=()):
    """Rebuild ``fn`` from its source, in its own module's namespace, with each (old, new) substitution applied.

    Each ``old`` must occur exactly once in the source. Tracebacks keep pointing at upstream's file: an unchanged line
    keeps its line number, and an added line takes the number of the line before it. Closures and decorated functions
    are refused, because the rebuild would lose a closure's cells and would apply the decorators a second time. A
    caller that applies a decorator to the rebuilt function itself names its exact source line in ``decorators``; the
    rebuild then leaves that line out, so the function comes back undecorated."""
    target = f"{fn.__module__}.{fn.__qualname__}"
    if fn.__code__.co_freevars:
        # a closure, or a method that calls super() with no arguments: rebuilt at module level it would lose its cells
        # and fail only when called
        raise RuntimeError(f"{target}: its code has free variables {fn.__code__.co_freevars}, which a rebuild from "
                           f"source in its module's namespace cannot keep")
    lines, first_line = inspect.getsourcelines(fn)
    source = textwrap.dedent("".join(lines))
    for decorator in decorators:
        if not source.startswith(decorator):
            raise RuntimeError(f"{target}: its source does not start with the decorator {decorator.strip()!r}")
        source = source[len(decorator):]
        first_line += decorator.count("\n")
    if source.lstrip().startswith("@"):
        # the rebuild would apply the decorators a second time (a registering decorator would register twice)
        raise RuntimeError(f"{target}: it is decorated, and a rebuild from its source would decorate it again")
    # origin[i] is the file line number of source line i
    origin = list(range(first_line, first_line + source.count("\n") + 1))
    for old, new in substitutions:
        if source.count(old) != 1:
            raise RuntimeError(f"{target}: the text to replace occurs {source.count(old)} times in its source, not once")
        start = source.count("\n", 0, source.index(old))
        end = start + old.count("\n") + 1
        origin[start:end] = _substituted_lines(old, new, origin[start:end])
        source = source.replace(old, new)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if getattr(node, "end_lineno", None) is None:
            continue
        start, end = origin[node.lineno - 1], origin[node.end_lineno - 1]
        if node.lineno < node.end_lineno and start == end and node.end_col_offset < node.col_offset:
            # a node spanning added lines now sits on one file line, and compile rejects an end column before the start
            # column: end it where it starts
            node.end_col_offset = node.col_offset
        node.lineno, node.end_lineno = start, end
    module = sys.modules[fn.__module__]
    flags = 0
    for name in __future__.all_feature_names:
        feature = getattr(module, name, None)
        if isinstance(feature, __future__._Feature):
            flags |= feature.compiler_flag
    namespace: dict = {}
    # getsourcelines followed __wrapped__ to the source it read, so compile against that function's file
    exec(compile(tree, inspect.getsourcefile(inspect.unwrap(fn)), "exec", flags=flags, dont_inherit=True), vars(module),
         namespace)
    rebuilt = namespace[fn.__name__]
    doc = rebuilt.__doc__
    functools.update_wrapper(rebuilt, fn)
    if inspect.cleandoc(doc or "") != inspect.cleandoc(fn.__doc__ or ""):
        # a substitution changed the docstring: keep the rebuilt one over the one update_wrapper copied from fn
        rebuilt.__doc__ = doc
    return rebuilt


def _substituted_lines(old: str, new: str, lines):
    """Map each line of ``new`` to a file line number, given ``lines``, the numbers of ``old``'s lines.

    An unchanged line keeps its number. A replacing line takes the number of the old line it replaces, and an inserted
    line the number of the line before it. The last line always keeps old's last number, because the source after the
    substitution continues on that line."""
    old_lines, new_lines = old.split("\n"), new.split("\n")
    mapped = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
        if tag == "equal":
            mapped += lines[i1:i2]
        elif tag == "replace":
            mapped += [lines[i1 + min(j - j1, i2 - i1 - 1)] for j in range(j1, j2)]
        elif tag == "insert":
            mapped += [lines[max(i1 - 1, 0)]] * (j2 - j1)
    mapped[-1] = lines[-1]
    return mapped
