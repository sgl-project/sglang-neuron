# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2 NKI kernels the plugin serves (PROVENANCE.md), and the digest that keys their NKI compile cache."""


import functools
import hashlib
import inspect
import os
from pathlib import Path

#: The kernel sources outside the copies that both copies trace: the NKI top-k select of every indexer merge.
SHARED_SOURCES = ("topk_select.py",)


def kernel_digest(here: str) -> str:
    """Return a digest of the files of a kernel copy and of the compile knobs set now.

    The NKI compile cache keys a kernel by its ``wrap_nki`` wrapper's source and
    arguments, and the NKI trace cache sees only the files whose functions ran
    in the trace. Neither reads the kernel files the wrapper calls into (module
    constants, the shared sources), nor the knobs those files read at import. A wrapper
    that takes this digest as an argument, or in its qualname, compiles again
    after any edit of the copy or a knob change instead of loading a stale
    kernel. An edit that leaves the kernel the same (a comment) costs one NKI
    trace; the lowered kernel and the NEFF are the same.

    Every file under ``here`` but the repository's records of it (its README and
    file manifest, ``compile_cache.is_kernel_record``, which the wheel does not
    ship) enters with its relative path and the hash of its bytes, then each of
    ``SHARED_SOURCES``, then each set compile knob (``compile_knobs``). The digest
    does not depend on where the copy lives, nor on whether it is a checkout or
    an install.

    Args:
        here: The directory of the kernel copy.

    Returns:
        str: 16 hex digits.

    Raises:
        OSError: When the copy cannot be read. A digest that silently missed
            files would key a stale kernel.

    Examples:
        >>> kernel_digest(os.path.dirname(__file__))  # doctest: +SKIP
        '3f0c9a1d5be27e48'
    """
    from sglang_neuron.backend.compile_cache import is_kernel_record
    from sglang_neuron.models.glm52.kernels import compile_knobs

    root = Path(here)
    if not root.is_dir():
        raise FileNotFoundError(f"No kernel copy at {here}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts or not path.is_file() or is_kernel_record(path):
            continue
        digest.update(rel.as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    for name in SHARED_SOURCES:
        digest.update(f"../{name}".encode() + b"\0")
        digest.update(hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).digest())
    for name, value in compile_knobs.compile_env(os.environ).items():
        digest.update(f"{name}={value}\n".encode())
    return digest.hexdigest()[:16]


#: The keyword-only argument that ``with_kernel_digest`` adds.
DIGEST_ARG = "kernel_digest"


def with_kernel_digest(kernel, tag: str = ""):
    """Return ``kernel`` with a keyword-only ``kernel_digest`` argument that it ignores.

    For a kernel whose own signature has no digest argument. The NKI caches key
    a kernel by its arguments, so a caller that passes
    ``kernel_digest=kernel_digest(<copy>)`` gets a new entry after every edit
    of the copy and every compile knob. An argument, unlike a longer qualname,
    leaves the lowered kernel as it is: nki names the kernel in its binary by
    its qualname, so a qualname digest would recompile the NEFF after a
    comment-only edit. ``tag`` is appended to the qualname. ``functools.wraps``
    keeps the kernel's name and module, and ``wrap_nki`` and nki read the
    extended signature.

    Examples:
        >>> kernel = with_kernel_digest(glm52_prefill_giga_kernel)  # doctest: +SKIP
        >>> kernel(*args, kernel_digest=kernel_digest(here), **kwargs)  # doctest: +SKIP
    """
    signature = inspect.signature(kernel)

    @functools.wraps(kernel)
    def wrapped(*args, kernel_digest: str = "", **kwargs):
        return kernel(*args, **kwargs)

    digest = inspect.Parameter(
        DIGEST_ARG, inspect.Parameter.KEYWORD_ONLY, default="", annotation=str
    )
    wrapped.__signature__ = signature.replace(
        parameters=[*signature.parameters.values(), digest]
    )
    wrapped.__qualname__ = kernel.__qualname__ + tag
    return wrapped
