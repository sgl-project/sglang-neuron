# SPDX-License-Identifier: Apache-2.0
"""torch.compile backend for the torch-neuronx runtime that honors per-graph compiler flags.

The plugin's compile sites pass ``options={"model_name": ..., "compiler_args": [...]}``.
torch-neuronx's ``neuron`` backend reads only ``model_name``; its graph compiler takes neuronx-cc flags from the environment
(``NEURON_CC_FLAGS``, and ``NEURON_COMPILER_OPT_LEVEL`` with default ``-O2``) when the
graph is handed to ``torch_neuronx._C.compile_graph``; the env profiles' launch
``NEURON_CC_FLAGS`` carry ``-O1``, which overrides that default (see :func:`compiler_env`).
That hand-off runs synchronously inside the backend call (AOTAutograd compiles the forward
graph eagerly), so this backend sets both variables for the duration of the call and then
restores them. torch-neuronx has no per-graph compiler-flag option, so the wrapper lives in
the plugin.

Setting ``os.environ`` is process-wide, so this is not thread-safe: a compile on another
thread during the call would see these flags. That does not happen here: a worker calls its
compiled graphs from one thread (the pipeline's forward), and each graph compiles inside its
first call, so two compiles never overlap in one process.
"""

from __future__ import annotations

import contextlib
import os
import shlex

BACKEND_NAME = "sglang_neuron"
#: The -O level of the env profiles' launch NEURON_CC_FLAGS (``DiffusionEnvProfile``,
#: "-O1 --hbm-scratchpad-page-size=2048"), which torch-neuronx places right after
#: NEURON_COMPILER_OPT_LEVEL on every graph's command line. A graph's own level is appended only
#: when it differs, so -O1 graphs keep their command line and cached NEFFs.
_DEFAULT_OPT_LEVEL = "-O1"
_registered = False


def split_compiler_args(compiler_args) -> tuple[str, str | None]:
    """``["-O1", "--foo=1"]`` -> ``("--foo=1", "-O1")``: flags, and the last ``-O`` level."""
    if isinstance(compiler_args, str):
        compiler_args = shlex.split(compiler_args)
    flags: list[str] = []
    opt_level = None
    for arg in compiler_args:
        if arg in ("-O1", "-O2", "-O3"):
            opt_level = arg
        else:
            flags.append(arg)
    return shlex.join(flags), opt_level


def compiler_env(compiler_args) -> dict[str, str | None]:
    """The ``NEURON_CC_FLAGS`` / ``NEURON_COMPILER_OPT_LEVEL`` a graph compiles under.

    torch-neuronx's neuronx-cc command line is ``NEURON_COMPILER_OPT_LEVEL``, then the
    ``NEURON_CC_FLAGS`` the process had at launch (the env profile's, which carry ``-O1``), then
    torch-neuronx's own flags, then ``NEURON_CC_FLAGS`` as this call sets it. The last ``-O`` wins,
    so a graph's own level applies only from the end of its flags; an ``-O1`` graph's flags are
    left as they are.
    """
    flags, opt_level = split_compiler_args(compiler_args)
    if opt_level not in (None, _DEFAULT_OPT_LEVEL):
        flags = f"{flags} {opt_level}".strip()
    return {"NEURON_CC_FLAGS": flags, "NEURON_COMPILER_OPT_LEVEL": opt_level}


@contextlib.contextmanager
def _env(**values: str | None):
    saved = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                continue
            os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def register() -> str:
    """Register the backend with Dynamo once and return its name."""
    global _registered
    if _registered:
        return BACKEND_NAME

    from torch._dynamo import register_backend

    from sglang_neuron.diffusion.neuron_api import neuron_dynamo_backend

    neuron_backend = neuron_dynamo_backend()

    def backend(gm, example_inputs, *, options=None, mode="default"):
        options = dict(options or {})
        compiler_args = options.pop("compiler_args", None)
        if compiler_args is None:
            return neuron_backend(gm, example_inputs, options=options, mode=mode)
        with _env(**compiler_env(compiler_args)):
            return neuron_backend(gm, example_inputs, options=options, mode=mode)

    register_backend(compiler_fn=backend, name=BACKEND_NAME)
    _registered = True
    return BACKEND_NAME


def compile_graph(fn, compiler_args):
    """``fn`` compiled whole, at fixed shapes, by this backend with neuronx-cc ``compiler_args``."""
    import torch

    return torch.compile(
        fn,
        backend=register(),
        options={"compiler_args": list(compiler_args)},
        fullgraph=True,
        dynamic=False,
    )
