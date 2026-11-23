# SPDX-License-Identifier: Apache-2.0
"""Two host-side costs of tracing a graph that change nothing in it, removed (``SGLANG_NEURON_COMPILE_SPEEDUPS``, on by
default; ``0`` keeps both).

Measured with py-spy on a MiMo-V2 1M launch's rank 0 while it traced its decode shapes (22 min for 20 shapes, every
NEFF a cache hit): NKI's own kernel tracing is 0.5% of the time; Dynamo's bytecode tracing and AOTAutograd are 45%; and
two passes that leave the graph as it is take a third:

- **torch-neuronx's f64 demotion** (``DTypeConstantConverter.run``, 21%): it walks every op of the lowered StableHLO
  module from Python to find f64 / complex<f64> results, and every ``op.results[i].type`` goes through the MLIR
  bindings' type casters (importlib lookups). The graphs hold no 64-bit value, so the walk changes nothing: here the
  module's assembly (large constants elided) is checked for ``f64`` first, and the walk runs only when it has one.
- **Python's traceback carets** (12%): Dynamo records a stack trace on every FX node it creates
  (``SubgraphTracer.create_proxy``), and Python 3.13+'s ``StackSummary.format`` parses each frame's source line
  (``ast.parse``) to place its ``^^^`` markers. The markers are cosmetic; without them the traces keep every frame and
  line, without the carets.

Neither changes a traced graph, so a NEFF compiled before keys the same (the HLO is byte for byte the same)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV = "SGLANG_NEURON_COMPILE_SPEEDUPS"
# log the first traced module that holds f64 (what keeps the f64 walk from being skipped)
DEBUG_ENV = "SGLANG_NEURON_COMPILE_SPEEDUPS_DEBUG"
_APPLIED = False
_F64_SEEN = []


def enabled(environ=os.environ) -> bool:
    value = environ.get(ENV, "1").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{ENV}={value!r}: set 1 (skip the two no-op tracing passes) or 0")
    return value == "1"


def _skip_f64_walk_without_f64() -> bool:
    try:
        from torch_neuronx.neuron_dynamo_backend.fx.passes import dtype_conversion
    except ImportError:
        return False
    cls = getattr(dtype_conversion, "DTypeConstantConverter", None)
    if cls is None or getattr(cls.run, "_sglang_neuron_skip", False):
        return False
    original = cls.run

    def run(self, module):
        try:
            asm = module.operation.get_asm(large_elements_limit=1)
        except Exception:  # noqa: BLE001 - any failure: the upstream walk decides
            return original(self, module)
        if "f64" not in asm:
            return module
        if os.environ.get(DEBUG_ENV) == "1" and not _F64_SEEN:
            _F64_SEEN.append(1)
            lines = [line.strip()[:240] for line in asm.splitlines() if "f64" in line]
            logger.warning("[neuron] f64 in a traced module (%d lines), first: %s", len(lines), " || ".join(lines[:6]))
        return original(self, module)

    run._sglang_neuron_skip = True
    cls.run = run
    return True


def _no_traceback_carets() -> bool:
    import traceback

    summary = traceback.StackSummary
    if not hasattr(summary, "_should_show_carets") or getattr(summary._should_show_carets, "_sglang_neuron_skip",
                                                                False):
        return False

    def should_show_carets(self, *args, **kwargs):
        return False

    should_show_carets._sglang_neuron_skip = True
    summary._should_show_carets = should_show_carets
    if hasattr(traceback, "_extract_caret_anchors_from_line_segment"):
        traceback._extract_caret_anchors_from_line_segment = lambda *args, **kwargs: None
    return True


def apply() -> None:
    """Install both, once per process, unless ``SGLANG_NEURON_COMPILE_SPEEDUPS=0``."""
    global _APPLIED
    if _APPLIED or not enabled():
        return
    _APPLIED = True
    done = [name for name, fn in (("f64 walk", _skip_f64_walk_without_f64), ("traceback carets", _no_traceback_carets))
            if fn()]
    if done:
        logger.info("[neuron] tracing speedups: %s", ", ".join(done))
