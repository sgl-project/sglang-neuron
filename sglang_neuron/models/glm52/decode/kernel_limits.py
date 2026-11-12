# SPDX-License-Identifier: Apache-2.0
"""The limits of a copy of the GLM-5.2 decode giga-kernel (GK) under
``sglang_neuron/models/glm52/kernels``, read from its source.

The decode runs the decode kernel (``models/glm52/kernels/decode``, `DECODE_KERNEL_PACKAGE`). The host calls it with the
arguments of every decode point and passes a 1M option's own arguments only when the geometry sets the option
(``models/glm52/ops/gk_decode.py``), so what the copy serves is bounded by its source: how many indexer keys its
selection ranks on a rank, how many latent stacks its giga-kernel takes, and which 1M options (`KERNEL_1M_OPTIONS`) it
has. `kernel_copy` reads these limits from the SOURCE of the copy's giga-kernel and of its MTP rows chain, and
`check_kernel_copy` refuses a geometry above them when the GK host module is built, before any trace. Nothing here
imports a kernel, so the check gives the same answer on a CPU host and on a device host, and a copy that is
re-synced with other limits is refused at start rather than at trace time.

The copy declares two of its limits as module constants in
``decode/giga_kernel.py``, read here from the source, not imported:

- ``GK_MAX_IDX_KEYS``: the largest ``idx_keys`` (indexer keys of one rank) that the copy serves.
- ``GK_OPTIONS``: the 1M options that the kernel selects from an input's shape or dtype (the
  router's expert columns for ``router_tp``, the leading axis of the MoE stacks for
  ``moe_sparse_slots``, the dtype of the indexer key cache for ``idx_fp8``), so its signature
  does not show them, as a tuple of names, for example ``("router_tp", "idx_fp8")``.

Its signature gives the rest: the latent stacks are the arguments of `LATENT_STACK_ARGS` that the
giga-kernel takes, and a 1M option with its own arguments (`SIGNATURE_OPTIONS`, for example
``latent_replica_group`` for ``latent_tp``) is served when the giga-kernel takes every one of them,
since the host passes them by keyword, and only for the option.

The rows-layout MTP chain of the ``mtp/`` fork (`ROWS_CHAIN_FILE`) has its own attention layer and
indexer. It serves the ``idx_keys`` of the giga-kernel. A fork that asserts ``... < TOPK_MAX_N`` in
its kernel body serves fewer keys (`KernelCopy.rows_chain_max_idx_keys`). The host then refuses the
rows chain at start, before the kernel refuses it at trace time. Of the 1M options, only the fp8
indexer keys reach the chain (`ROWS_CHAIN_OPTIONS`), and the fork declares them in a ``GK_OPTIONS``
of its own (`KernelCopy.rows_chain_options`). A fork whose gathered rows (``rows_gathered``) are
read in whole 128-column k-tiles, with no fallback to the SBUF rows for a narrower ``eh_proj``
column block, serves only o_proj block groups whose ``hidden / oproj`` is a multiple of 128
(`KernelCopy.rows_chain_eh_proj_align`).

The geometry has its own limits too: `gk_kv_cache.latent_stack_sizes` gives the stacks that a
point needs, and `Glm52GkGeometry.idx_topk_chunks` refuses a shard that the chunks do not tile.
`KernelCopy.unmet` checks the copy's limits against the geometry.

Example:
    >>> copy = kernel_copy(DECODE_KERNEL_PACKAGE)  # doctest: +SKIP
    >>> copy.unmet(Glm52GkGeometry())  # doctest: +SKIP
    ''
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import logging
import os
from dataclasses import dataclass

from sglang_neuron.models.glm52.layout.decode.gk_geometry import TOPK_MAX_N, Glm52GkGeometry
from sglang_neuron.models.glm52.layout.decode.gk_kv_cache import latent_stack_sizes

logger = logging.getLogger(__name__)

#: The package of the decode giga kernel, the one copy the decode runs (ops/gk_decode.KERNEL_PACKAGE).
DECODE_KERNEL_PACKAGE = "sglang_neuron.models.glm52.kernels.decode"

#: The geometry options of the 1M points that change the kernel inputs. A copy serves one
#: only when its giga-kernel takes the option's arguments (`SIGNATURE_OPTIONS`) or declares it
#: (`OPTIONS_CONSTANT`). ``rope_split`` is not here: the
#: host builds the RoPE rows of each step from the two small tables, so the kernel gets the
#: same input tensors with or without it. The float32 rows differ by about 1.4e-7
#: (``models/glm52/layout/rope_split.py``). ``oproj_tp`` and ``qproj_tp`` are not here either: the
#: giga-kernel of each copy has the block-sharded exit and query path.
KERNEL_1M_OPTIONS = (
    "idx_fp8",
    "latent_tp",
    "router_tp",
    "dense_tp",
    "shared_tp",
    "moe_sparse_slots",
)

#: The 1M options that have their own giga-kernel arguments, and those arguments. The host
#: passes them by keyword, and only when the option is set (``models/glm52/ops/gk_decode.py``),
#: so a copy whose giga-kernel takes every one of them serves the option.
SIGNATURE_OPTIONS = {
    "latent_tp": ("latent_replica_group",),
    "dense_tp": ("dense_replica_group", "dense_tp_size"),
    "shared_tp": ("shared_replica_group", "shared_tp_size"),
}

#: The 1M options that reach the rows chain of the ``mtp/`` fork. Its MTP layer keeps the
#: token-group shards, the whole router and its one MoE slot (``models/glm52/mtp/draft.py``), and
#: it reads its own indexer key cache, of the dtype of the target's (`gk_kv_cache.idx_key_dtype`).
#: The fork declares the ones it serves in its own `OPTIONS_CONSTANT`.
ROWS_CHAIN_OPTIONS = ("idx_fp8",)

#: The giga-kernel source of a copy, relative to the directory of the copy's package.
GIGA_KERNEL_FILE = "giga_kernel.py"

#: The name of the giga-kernel function in `GIGA_KERNEL_FILE`.
GIGA_KERNEL_NAME = "glm52_decode_giga_kernel"

#: The rows-layout MTP chain of the ``mtp/`` fork, relative to the directory of the copy's package.
ROWS_CHAIN_FILE = os.path.join("mtp", "draft_chain.py")

#: The name of the rows chain kernel function in `ROWS_CHAIN_FILE`.
ROWS_CHAIN_NAME = "glm52_mtp_draft_chain_kernel"

#: The latent stack arguments of the giga-kernel, in layer order (`gk_kv_cache.Glm52GkCaches`). Four stacks keep each
#: one below the device's 2**32-byte limit on a single allocation, which three 4.36 GB stacks of 32 slots at 1M exceed.
LATENT_STACK_ARGS = ("kv_cache_all", "kv_cache_hi", "kv_cache_hi2", "kv_cache_hi3")

#: The module constant that declares the largest ``idx_keys`` of a copy.
MAX_IDX_KEYS_CONSTANT = "GK_MAX_IDX_KEYS"

#: The module constant that declares the 1M options of a copy that its signature does not show (the module
#: docstring's ``GK_OPTIONS``). A copy serves one of those only when it declares it here.
OPTIONS_CONSTANT = "GK_OPTIONS"

#: The largest ``idx_keys`` a rows chain serves that asserts ``<indexer keys> < TOPK_MAX_N`` at its top level: its
#: local selection then ranks the whole shard in one ``nisa.topk``, which takes fewer than `TOPK_MAX_N` keys.
SINGLE_TOPK_MAX_IDX_KEYS = TOPK_MAX_N - 1

#: The k-tile of the rows chain's gathered rows: a fork without the SBUF-rows fallback reads each
#: rank's ``eh_proj`` column block of the gathered rows in whole tiles of this many columns.
ROWS_GATHERED_TILE = 128



@dataclass(frozen=True)
class KernelCopy:
    """The limits of one copy of the giga-kernel.

    Attributes:
        package: The package of the copy (``DECODE_KERNEL_PACKAGE``).
        args: The argument names of the copy's giga-kernel, in order.
        max_idx_keys: The largest ``idx_keys`` (indexer keys of one rank) that the copy serves.
        latent_stacks: The latent stacks that the copy's giga-kernel takes.
        rows_chain_max_idx_keys: The largest ``idx_keys`` that the rows chain of the ``mtp/``
            fork serves. It is not more than `max_idx_keys`.
        options: The 1M options (`KERNEL_1M_OPTIONS`) that the copy serves, in `KERNEL_1M_OPTIONS`
            order: the ones whose arguments its giga-kernel takes (`SIGNATURE_OPTIONS`) and the
            ones that it declares (`OPTIONS_CONSTANT`).
        rows_chain_options: The options of `ROWS_CHAIN_OPTIONS` that the rows chain of the
            ``mtp/`` fork declares in its own `OPTIONS_CONSTANT`.
        rows_chain_eh_proj_align: The column multiple that the rows chain needs of an
            ``eh_proj`` column block (``hidden / oproj``): `ROWS_GATHERED_TILE` for a fork
            that reads the gathered rows in whole k-tiles without a fallback, else 1.
    """

    package: str
    args: tuple[str, ...]
    max_idx_keys: int
    latent_stacks: int
    rows_chain_max_idx_keys: int
    options: tuple[str, ...] = ()
    rows_chain_options: tuple[str, ...] = ()
    rows_chain_eh_proj_align: int = 1

    def unserved_options(self, geo: Glm52GkGeometry) -> list[str]:
        """Return the 1M options that the geometry sets and the copy does not serve.

        Args:
            geo: The decode geometry.

        Returns:
            The option names, in `KERNEL_1M_OPTIONS` order. Empty when the copy serves every one.

        Example:
            >>> kernel_copy(DECODE_KERNEL_PACKAGE).unserved_options(Glm52GkGeometry())  # doctest: +SKIP
            []
        """
        return [
            name
            for name in KERNEL_1M_OPTIONS
            if getattr(geo, name) and name not in self.options
        ]

    def unmet(self, geo: Glm52GkGeometry) -> str:
        """Return the first limit of the copy that the geometry is above, or ``""``.

        Args:
            geo: The decode geometry.

        Returns:
            The limit as text, for an error message. ``""`` when the copy serves the geometry.

        Raises:
            ValueError: If no stack count keeps a latent layer start below the DGE limit
                (`gk_kv_cache.latent_stack_sizes`).
        """
        if geo.idx_keys > self.max_idx_keys:
            return (
                f"idx_keys {geo.idx_keys} is above the largest idx_keys "
                f"{self.max_idx_keys} of the copy"
            )
        stacks = latent_stack_sizes(geo)
        if len(stacks) > self.latent_stacks:
            return (
                f"the latent cache needs {len(stacks)} stacks {stacks}, and the "
                f"giga-kernel of the copy takes {self.latent_stacks}"
            )
        return ""


def _evaluate(node: ast.expr, known: dict[str, object]) -> object:
    """Return the value of a constant expression: an int or str literal, a tuple, int arithmetic, or a name.

    Raises:
        ValueError: If the expression is of another kind, or names an unknown constant.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, str)):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_evaluate(item, known) for item in node.elts)
    if isinstance(node, ast.Name) and node.id in known:
        return known[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _evaluate(node.operand, known)
        if isinstance(value, int):
            return -value
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left, known), _evaluate(node.right, known)
        operations = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Pow: lambda a, b: a**b,
            ast.LShift: lambda a, b: a << b,
        }
        operation = operations.get(type(node.op))
        if operation and isinstance(left, int) and isinstance(right, int):
            return operation(left, right)
    raise ValueError(f"not a constant expression: {ast.dump(node)}")


def _module_constants(tree: ast.Module) -> tuple[dict[str, object], set[str]]:
    """Return the module-level constants of a parsed module, and the names that have no static value.

    A constant can use an earlier constant of the same module. A name that the module imports has
    no static value, so a constant that uses it is in the second set.
    """
    known: dict[str, object] = {}
    unknown: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        try:
            known[target.id] = _evaluate(value, known)
            unknown.discard(target.id)
        except ValueError:
            known.pop(target.id, None)
            unknown.add(target.id)
    return known, unknown


def _kernel_args(tree: ast.Module, path: str) -> tuple[str, ...]:
    """Return the argument names of the giga-kernel function, in order.

    Raises:
        ValueError: If the module has no function `GIGA_KERNEL_NAME` at module level.
    """
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == GIGA_KERNEL_NAME:
            arguments = node.args
            return tuple(
                a.arg
                for a in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                )
            )
    raise ValueError(
        f"{path} has no function {GIGA_KERNEL_NAME}. Re-sync the copy, or check its package."
    )


def _asserts_below_topk_max_n(tree: ast.Module) -> bool:
    """Return True when the rows chain kernel asserts ``<indexer keys> < TOPK_MAX_N`` at its top level.

    This is the refusal of the long context in the ``mtp/`` fork: ``kernel_assert(idx_keys < TOPK_MAX_N,
    ...)``, or ``idx_k_cache.shape[0]`` in the form of the fork generator. Only the body of
    `ROWS_CHAIN_NAME` counts, and only an assert on a name that starts with ``idx_k``. An assert inside
    a branch (for example, the sub-cap branch of a chunked selection) does not count, because it does
    not refuse every shard. An assert in a helper (for example, on one chunk) does not count either.
    """
    for function in tree.body:
        if not (
            isinstance(function, ast.FunctionDef) and function.name == ROWS_CHAIN_NAME
        ):
            continue
        for statement in function.body:
            call = statement.value if isinstance(statement, ast.Expr) else None
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "kernel_assert"
                and call.args
            ):
                continue
            test = call.args[0]
            if not (
                isinstance(test, ast.Compare)
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Lt)
                and isinstance(test.comparators[0], ast.Name)
                and test.comparators[0].id == "TOPK_MAX_N"
            ):
                continue
            names = {
                node.id for node in ast.walk(test.left) if isinstance(node, ast.Name)
            }
            if any(name.startswith("idx_k") for name in names):
                return True
    return False


def _gathered_rows_need_whole_tiles(tree: ast.Module) -> bool:
    """Return True when the rows chain reads its gathered rows with no narrow-block fallback.

    The rows chain picks the draft row out of the column AllGather's rank-major rows
    (``rows_gathered`` in the body of `ROWS_CHAIN_NAME`), and the attention layer asserts whole
    `ROWS_GATHERED_TILE`-column k-tiles per ``eh_proj`` column block there. A fork that takes the
    SBUF rows for a narrower block guards the assignment with ``... % P_MAX == 0`` (or ``% 128``).
    A chain with no ``rows_gathered`` assignment has no gathered rows to refuse.
    """
    for function in tree.body:
        if not (
            isinstance(function, ast.FunctionDef) and function.name == ROWS_CHAIN_NAME
        ):
            continue
        values = [
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "rows_gathered"
                for t in node.targets
            )
        ]
        return any(not _guards_whole_tiles(value) for value in values)
    return False


def _guards_whole_tiles(node: ast.expr) -> bool:
    """Return True when the expression holds a ``... % P_MAX`` (or ``% 128``) test."""
    return any(
        isinstance(sub, ast.BinOp)
        and isinstance(sub.op, ast.Mod)
        and (
            (isinstance(sub.right, ast.Name) and sub.right.id == "P_MAX")
            or (
                isinstance(sub.right, ast.Constant)
                and sub.right.value == ROWS_GATHERED_TILE
            )
        )
        for sub in ast.walk(node)
    )


def _rows_chain_limits(
    path: str | None, max_idx_keys: int, options: tuple[str, ...]
) -> tuple[int, tuple[str, ...], int]:
    """Return the largest ``idx_keys``, the 1M options and the ``eh_proj`` block multiple of the rows chain in ``path``.

    The keys are `max_idx_keys`, or fewer when the chain refuses them. The options are the ones
    of `ROWS_CHAIN_OPTIONS` that the chain declares in `OPTIONS_CONSTANT`. The block multiple is
    `ROWS_GATHERED_TILE` for a chain that reads its gathered rows in whole k-tiles with no
    fallback (`_gathered_rows_need_whole_tiles`), else 1. A copy without the file has no rows
    chain to refuse, so it gets the giga-kernel's keys and ``options``, and no block multiple.
    The import of the chain then fails, and the host reports that.
    """
    if path is None or not os.path.isfile(path):
        return max_idx_keys, tuple(n for n in ROWS_CHAIN_OPTIONS if n in options), 1
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    if _asserts_below_topk_max_n(tree):
        max_idx_keys = min(max_idx_keys, SINGLE_TOPK_MAX_IDX_KEYS)
    known, unknown = _module_constants(tree)
    declared = _declared_options(known, unknown, path, ())
    chain = tuple(n for n in ROWS_CHAIN_OPTIONS if n in (declared or ()))
    align = ROWS_GATHERED_TILE if _gathered_rows_need_whole_tiles(tree) else 1
    return max_idx_keys, chain, align


def _declared_max_idx_keys(known: dict[str, object], unknown: set[str], path: str) -> int:
    """Return the copy's `MAX_IDX_KEYS_CONSTANT`, an int literal (or constant arithmetic) of its giga-kernel module.

    Raises:
        ValueError: If the copy does not declare it as an int.
    """
    value = known.get(MAX_IDX_KEYS_CONSTANT)
    if MAX_IDX_KEYS_CONSTANT in unknown or not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"{path} declares no int {MAX_IDX_KEYS_CONSTANT}, the largest idx_keys its giga-kernel "
            "serves: declare it as an int literal beside the giga-kernel"
        )
    return value


def _takes(args: tuple[str, ...], option: str) -> bool:
    """Return True when the giga-kernel takes every argument of the option (`SIGNATURE_OPTIONS`)."""
    return option in SIGNATURE_OPTIONS and all(
        arg in args for arg in SIGNATURE_OPTIONS[option]
    )


def _declared_options(
    known: dict[str, object], unknown: set[str], path: str, args: tuple[str, ...]
) -> tuple[str, ...] | None:
    """Return the 1M options that the copy declares in `OPTIONS_CONSTANT`, or None when it declares none.

    A name that is not a 1M option, and an option of `SIGNATURE_OPTIONS` whose arguments the
    giga-kernel does not take, are left out, and a
    warning names each one. A constant that is not a tuple of names counts as not declared.
    """
    if OPTIONS_CONSTANT in unknown:
        logger.warning(
            "[glm52 GK] %s sets %s from a value that this check cannot read. The copy then "
            "serves only the options of its signature. Set %s to a tuple of str literals.",
            path,
            OPTIONS_CONSTANT,
            OPTIONS_CONSTANT,
        )
        return None
    value = known.get(OPTIONS_CONSTANT)
    if value is None:
        return None
    if not isinstance(value, tuple) or not all(isinstance(n, str) for n in value):
        logger.warning(
            "[glm52 GK] %s sets %s to %r, which is not a tuple of option names. The copy "
            "then serves only the options of its signature.",
            path,
            OPTIONS_CONSTANT,
            value,
        )
        return None
    options = []
    for name in value:
        if name not in KERNEL_1M_OPTIONS:
            reason = f"is not one of the 1M options {KERNEL_1M_OPTIONS}"
        elif name in SIGNATURE_OPTIONS and not _takes(args, name):
            # The host passes the option's arguments by keyword, so the signature is the hard limit.
            reason = (
                f"needs the giga-kernel arguments {SIGNATURE_OPTIONS[name]}, which the "
                "giga-kernel does not take"
            )
        else:
            options.append(name)
            continue
        logger.warning(
            "[glm52 GK] %s declares %s in %s, and %s %s. The check refuses it.",
            path,
            name,
            OPTIONS_CONSTANT,
            name,
            reason,
        )
    return tuple(options)


def read_kernel_copy(
    package: str, path: str, rows_chain_path: str | None = None
) -> KernelCopy:
    """Read the limits of a copy from the source file of its giga-kernel and of its rows chain.

    Args:
        package: The package of the copy.
        path: The giga-kernel source of the copy.
        rows_chain_path: The rows chain source of the copy (`ROWS_CHAIN_FILE`). None, or a missing
            file, gives the rows chain the limit of the giga-kernel.

    Returns:
        The limits of the copy.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file has no giga-kernel function, or declares no `MAX_IDX_KEYS_CONSTANT`.

    Example:
        >>> read_kernel_copy("pkg", "/path/to/glm52_decode_giga_kernel.py")  # doctest: +SKIP
    """
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    args = _kernel_args(tree, path)
    known, unknown = _module_constants(tree)

    # The stacks the signature takes: kv_cache_all, then kv_cache_hi, kv_cache_hi2 and kv_cache_hi3, in order.
    latent_stacks = 0
    for name in LATENT_STACK_ARGS:
        if name not in args:
            break
        latent_stacks += 1
    max_idx_keys = _declared_max_idx_keys(known, unknown, path)
    # The 1M options: the ones whose arguments the signature takes, and the declared ones.
    declared_options = _declared_options(known, unknown, path, args)
    options = tuple(
        name
        for name in KERNEL_1M_OPTIONS
        if _takes(args, name) or name in (declared_options or ())
    )
    rows_chain_max_idx_keys, rows_chain_options, rows_chain_eh_proj_align = (
        _rows_chain_limits(rows_chain_path, max_idx_keys, options)
    )
    return KernelCopy(
        package=package,
        args=args,
        max_idx_keys=max_idx_keys,
        latent_stacks=latent_stacks,
        rows_chain_max_idx_keys=rows_chain_max_idx_keys,
        options=options,
        rows_chain_options=rows_chain_options,
        rows_chain_eh_proj_align=rows_chain_eh_proj_align,
    )


@functools.lru_cache(maxsize=None)
def kernel_copy(package: str) -> KernelCopy:
    """Return the limits of the kernel copy in ``package``. The result is cached.

    Args:
        package: A package under ``sglang_neuron.models.glm52.kernels`` (``DECODE_KERNEL_PACKAGE``).

    Returns:
        The limits of the copy.

    Raises:
        ModuleNotFoundError: If the package does not exist.
        OSError: If the giga-kernel source of the copy cannot be read.
        ValueError: If that source has no giga-kernel function.

    Example:
        >>> kernel_copy(DECODE_KERNEL_PACKAGE).latent_stacks  # doctest: +SKIP
        4
    """
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(
            f"{package} is not a package of a GK kernel copy under sglang_neuron.models.glm52.kernels"
        )
    directory = list(spec.submodule_search_locations)[0]
    path = os.path.join(directory, GIGA_KERNEL_FILE)
    copy = read_kernel_copy(package, path, os.path.join(directory, ROWS_CHAIN_FILE))
    logger.info(
        "[glm52 GK] kernel copy %s: idx_keys up to %d (rows chain %d), latent stacks %d, "
        "1M options %s (rows chain %s, eh_proj blocks of a multiple of %d columns)",
        package,
        copy.max_idx_keys,
        copy.rows_chain_max_idx_keys,
        copy.latent_stacks,
        list(copy.options),
        list(copy.rows_chain_options),
        copy.rows_chain_eh_proj_align,
    )
    return copy


def check_kernel_copy(geo: Glm52GkGeometry, package: str) -> None:
    """Check that the kernel copy in ``package`` serves this geometry, before anything is traced.

    A copy serves the ``idx_keys``, the latent stacks and the 1M options of its giga-kernel
    (`kernel_copy`), and the chunk-dealt indexer pool (``idx_pool``) only when its giga-kernel takes
    the pool's tables (`idx_pool.CDP_TABLES`). The decode kernel serves both decode points
    (`points.POINTS`).

    Args:
        geo: The decode geometry.
        package: The package of the copy (``DECODE_KERNEL_PACKAGE``).

    Raises:
        ValueError: If the geometry pools the indexer and the copy's giga-kernel
            does not take the pool's tables, if it is above a limit of the copy (`KernelCopy.unmet`),
            or if it sets a 1M option the copy does not serve (`KernelCopy.unserved_options`).

    Example:
        >>> check_kernel_copy(Glm52GkGeometry(), DECODE_KERNEL_PACKAGE)  # doctest: +SKIP
    """
    copy = kernel_copy(package)
    if geo.idx_pool_chunk:
        from sglang_neuron.models.glm52.decode.upstream import CDP_TABLES  # noqa: PLC0415

        missing = [arg for arg in CDP_TABLES if arg not in copy.args]
        if missing:
            raise ValueError(
                f"the giga-kernel of {package} does not take the chunk-dealt indexer pool's tables {missing}: serve a "
                "kernel copy whose giga-kernel takes them."
            )
    unmet = copy.unmet(geo)
    if unmet:
        raise ValueError(
            f"the GK copy {package} does not serve this geometry: {unmet}. Serve a decode point "
            "it serves (SGLANG_NEURON_GLM52_POINT), or a copy that takes this geometry."
        )
    options = copy.unserved_options(geo)
    if options:
        raise ValueError(
            f"the GK copy {package} does not serve {options}. Serve a decode point without them "
            f"(SGLANG_NEURON_GLM52_POINT), or a copy whose giga-kernel takes or declares them "
            f"({OPTIONS_CONSTANT})."
        )
    if geo.idx_topk_chunks > 1 and not geo.idx_pool_chunk:
        raise ValueError(
            f"a rank's {geo.idx_keys} indexer keys are {geo.idx_topk_chunks} top-k chunks, which the GK copy {package} "
            "scores only in the chunk-dealt indexer pool (the decode geometry of a point with pool_deal 'chunk')."
        )
