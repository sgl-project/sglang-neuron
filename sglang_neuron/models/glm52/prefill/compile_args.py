# SPDX-License-Identifier: Apache-2.0
"""neuronx-cc arguments of the GLM-5.2 prefill stage graphs.

The release compiles one list on the released toolchain: the plugin's base arguments (graph_runner COMPILER_ARGS) at
-O2 as a transformer, without the base's verifier-off option, plus the prefill scratchpad page (1344 MiB, enough for the
1M graph). backend/stack.py keeps the model type and adds torch-neuronx's split finder.

HIERARCHY_DEPTH is the op_name depth of the stage's repeated GK launch modules in its HLO, measured from every rank's
StableHLO dump: the depth a hierarchy splitter would cut at to compile each repeated launch once. The released list
takes the split finder instead."""

from __future__ import annotations

# the op_name depth of _GkLaunch[j] in the stage graph's HLO (1 in every rank's dump)
HIERARCHY_DEPTH = 1
SCRATCHPAD_PAGE_MB = 1344
_HLO2TENSORIZER = "--internal-hlo2tensorizer-options="

# The plugin's base arguments (graph_runner COMPILER_ARGS) at -O2 as a transformer, with the BIR verifier on: every
# served stage program passes it, and it changes none of their instruction bytes.
PREFILL_COMPILER_ARGS: tuple = (
    "--auto-cast=none",
    "--verbose=35",
    "-O2",
    f"{_HLO2TENSORIZER}--modular-flow-mac-threshold=10",
    f"--hbm-scratchpad-page-size={SCRATCHPAD_PAGE_MB}",
    "--model-type=transformer",
)


def stage_compiler_args() -> list:
    """The stage graphs' argument list: the released toolchain's (launch refuses any other toolchain)."""
    return list(PREFILL_COMPILER_ARGS)
