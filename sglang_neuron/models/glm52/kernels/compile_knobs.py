# SPDX-License-Identifier: Apache-2.0
"""The environment variables that change a compiled GLM-5.2 kernel.

The kernel copies read their knobs when they are imported or traced, and the NKI compile cache keys a kernel by its
wrapper's source and arguments only, so ``kernel_digest`` adds every set knob classified here to the key. The GLM-5.2
launch refuses one that the served point does not pin (``models/glm52/launch.py``). Stdlib only, so a launcher can
load this file by path.
"""

from collections.abc import Mapping
from typing import Optional

#: A set variable with one of these prefixes is a compile input, unless ``NON_COMPILE_NAMES`` or
#: ``NON_COMPILE_PREFIXES`` names it: the knobs of the kernel copies under ``sglang_neuron/models/glm52/kernels``
#: (``GLM52K_``, ``GLM52_``) and of the prefill giga kernel's variants (``GLM_GK_``), and nkilib's own ``NKILIB_``, so a
#: launch that still sets a copy knob by its old nkilib name is refused instead of ignored. Not ``NKI_``: the toolchain
#: sets ``NKI_*`` values of its own at import (``NKI_ENABLE_TRACE_CACHE``), which would change every key.
COMPILE_PREFIXES = ("GLM52K_", "GLM52_", "GLM_GK_", "NKILIB_")

#: Prefixed variables that change no compiled kernel.
NON_COMPILE_NAMES = frozenset(
    {
        # nkilib's logging and path settings
        "NKILIB_LOG_LEVEL",
        "NKILIB_PATH",
        # nkilib's decode attention switch, which the GPT-OSS decode sets and no GLM-5.2 kernel reads
        "NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP",
        # the prefill MoE's expert placement file, which the weight loaders read (models/glm52/prefill/expert_placements)
        "GLM_GK_MOE_PLACEMENT",
    }
)
NON_COMPILE_PREFIXES = ("NKILIB_LOG_LEVEL_",)


def is_compile_knob(name: str) -> bool:
    """Return True if a set ``name`` can change a compiled kernel.

    Examples:
        >>> is_compile_knob("NKILIB_MOE_X"), is_compile_knob("NKILIB_LOG_LEVEL_SBM")
        (True, False)
    """
    if name in NON_COMPILE_NAMES or name.startswith(NON_COMPILE_PREFIXES):
        return False
    return name.startswith(COMPILE_PREFIXES)


def compile_env(
    environ: Mapping[str, str], defaults: Optional[Mapping[str, str]] = None
) -> dict[str, str]:
    """Return the set compile knobs of ``environ``, sorted by name, leaving out a knob whose value equals its entry in
    ``defaults``.

    Examples:
        >>> compile_env({"NKILIB_X": "1", "NKILIB_PATH": "/x", "HOME": "/r"})
        {'NKILIB_X': '1'}
        >>> compile_env({"GLM52_GIGA_LM_HEAD": "1"}, {"GLM52_GIGA_LM_HEAD": "1"})
        {}
    """
    defaults = defaults or {}
    return {
        name: environ[name]
        for name in sorted(environ)
        if is_compile_knob(name) and defaults.get(name) != environ[name]
    }
