# SPDX-License-Identifier: Apache-2.0
"""The torch and torch-neuronx builds the plugin's in-process wraps are written against, and the torch functions it
wraps, pinned by the md5 of their source.

The runtime wraps two torch functions when it loads: ``torch.get_device_module`` (``sglang_neuron/process_init.py``)
and ``torch.distributed.init_process_group`` (``backend/port_hold.py``). Each wrap first checks here that the function
it wraps is the one it was written against, and refuses another. The direct submit (``backend/direct_submit.py``)
skips dynamo's frame and AOTAutograd's runtime wrapper, whose behaviour is the whole torch build's rather than one
function's, so it checks ``torch.version.git_version`` against TORCH_GIT_VERSION. The torch-neuronx methods the
executor binding cache and the direct submit wrap are pinned by md5 in those modules, against TORCH_NEURONX_BUILD, the
build the release pin names as its torch-neuronx deviation. Both builds are read from the release pin
(release/neuron_sdk.toml), so no version is written here.
"""
from __future__ import annotations

from sglang_neuron.release import neuron_sdk

TORCH_GIT_VERSION = neuron_sdk.pin().torch_git_version
TORCH = f"torch {neuron_sdk.pin().torch_version} ({TORCH_GIT_VERSION[:8]})"
TORCH_NEURONX_BUILD = neuron_sdk.pin().deviations["torch-neuronx"].version

# wrapped function -> {md5 of its dedented source: where that source comes from}
SOURCES = {
    "torch.get_device_module": {"94cad355e960cac99dbcace80f8eb7e4": TORCH},
    "torch.distributed.init_process_group": {"6bc1848ad90d4f42c03ed3294ed18187": TORCH},
}


def source_md5_or_empty(fn) -> str:
    """md5 of ``fn``'s dedented source (a wrapper is read through its ``__wrapped__``), or "" where it has none: None
    (a method the class does not define), a built-in, or a function inspect cannot read. A pin check then reports it
    as changed."""
    from sglang_neuron.sglang_patches.source_guard import source_md5

    try:
        return source_md5(fn)
    except (TypeError, OSError):
        return ""


def require(name: str, fn, wrapper: str) -> None:
    """Refuse to wrap ``fn`` (the current ``name``) unless its source is a pinned one; ``wrapper`` names the plugin code
    that wraps it."""
    from sglang_neuron.sglang_patches.source_guard import source_md5

    sources = SOURCES[name]
    try:
        md5 = source_md5(fn)
    except (OSError, TypeError) as e:
        raise RuntimeError(f"{name} has no readable source ({e}); {wrapper} wraps only the pinned one "
                           f"({', '.join(sources.values())})") from None
    if md5 not in sources:
        raise RuntimeError(f"{name} has source md5 {md5}, not the one {wrapper} was written against "
                           f"({', '.join(f'{m} from {w}' for m, w in sources.items())}); re-validate {wrapper} "
                           "against this torch before wrapping it")
