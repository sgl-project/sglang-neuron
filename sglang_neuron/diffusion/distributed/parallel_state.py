# SPDX-License-Identifier: Apache-2.0
"""Shared distributed helpers for the torch-native Neuron backend.

Provides replica-group registration utilities that any torch-native model
using TP/CP collectives on non-mesh process groups needs for SPMD compilation.
"""

import logging

from sglang_neuron.backend.stack import get_platform_target, normalize_target_family
from sglang_neuron.diffusion.neuron_api import register_process_group_replica_groups
from sglang_neuron.utils.hardware_config import get_logical_nc_config

logger = logging.getLogger(__name__)


def _tp_replica_groups(tp_size: int, world_size: int) -> list[list[int]]:
    """TP partition: contiguous blocks of ``tp_size``, repeated across the whole
    cfg-extended world (each CFG replica has its own TP blocks).

    e.g. tp4, world 32 -> [[0,1,2,3],[4,5,6,7], ..., [28,29,30,31]].
    """
    return [list(range(base, base + tp_size)) for base in range(0, world_size, tp_size)]


def _cp_replica_groups(tp_size: int, cp_size: int, cfg_size: int) -> list[list[int]]:
    """CP partition: strided by ``tp_size`` within each CFG replica, offset by the
    replica base (``cfg_rank * tp*cp``) so the partition spans the whole world.

    e.g. tp4, cp4, cfg2 -> [[0,4,8,12],[1,5,9,13], ..., [16,20,24,28], ...].
    """
    replica = tp_size * cp_size
    return [
        [tp_rank + cp_rank * tp_size + cfg_rank * replica for cp_rank in range(cp_size)]
        for cfg_rank in range(cfg_size)
        for tp_rank in range(tp_size)
    ]


def _cfg_replica_groups(tp_size: int, cp_size: int, cfg_size: int) -> list[list[int]]:
    """CFG partition: pair the corresponding rank of each replica, stride ``tp*cp``.

    e.g. tp4, cp4, cfg2 (replica=16) -> [[0,16],[1,17], ..., [15,31]].
    """
    replica = tp_size * cp_size
    world_size = replica * cfg_size
    return [list(range(base, world_size, replica)) for base in range(replica)]


# ---------------------------------------------------------------------------
# Physical-mesh-aware group construction (the tp_contiguous mapping)
# ---------------------------------------------------------------------------
#
# trn2.48xlarge is 16 chips in a 4x4 2D torus (both axes wrap), 4 logical cores per
# chip at NEURON_LOGICAL_NC_CONFIG=2, so rank r is on chip r // 4. A collective is
# routable only if its member chips form a torus ring (an adjacent pair incl. the wrap,
# a full row/column, or a rectangular block of them) and every chip contributes the same
# core offsets. The arithmetic _*_replica_groups helpers above are mesh-blind and yield
# off-ring groups like CFG [0,32] (chips 0 and 8, two hops apart) that the fabric rejects
# with "no_hier no_mesh". Sharing a mesh row/column is NOT sufficient: column 0 is
# [0,4,16,20,32,36,48,52] = chips [0,1,4,5,8,9,12,13], so 0 and 32 share the column yet
# sit two torus hops apart.
#
# _tp_contiguous_mesh_groups (with _factor_line) places TP/CP/CFG so every group stays on
# a ring; see those for the two layouts.


# The non-contiguous 8x8 tp_contiguous mapping is specific to Trn2. Trn3's switch
# fabric uses the arithmetic TP/CP/CFG groups (_*_replica_groups); those groups are still
# registered below as compiler replica-group metadata.
# Both sets hold target families (``normalize_target_family``).
_PHYSICAL_MESH_PLATFORMS = frozenset({"trn2"})
_ARITHMETIC_GROUP_PLATFORMS = frozenset({"trn3"})


def _supports_physical_mesh(tp_size: int, cp_size: int, cfg_size: int) -> bool:
    """True iff the validated physical-mesh layout applies to this world.

    The Trn2 non-contiguous fabric has a length-8 row axis; a ``world``-rank job
    occupies the first ``world // 8`` physical rows, so the mesh is ``num_rows x 8``
    (``num_rows == 8`` at 64 cores, ``4`` at 32 cores). One logical dim fills a
    length-8 axis and the remaining two factor the other. Two orientations are
    validated:
      - Layout A (``cp==8``, ``tp*cfg==8``): CP = columns; row axis = TP x CFG. This
        needs length-8 columns, so it only applies to the full 64-core mesh.
      - Layout B (``tp==8``, ``cp>1``): TP = rows; the length-``num_rows`` column
        axis = CP x CFG (e.g. tp8/cp4/cfg2 at 64 cores, tp8/cp4/cfg1 at 32 cores).
        ``cp>1`` means CP is parallel.
    It is gated on the *instance type*: only platforms in
    :data:`_PHYSICAL_MESH_PLATFORMS` (currently Trn2) have a confirmed-routable
    tp_contiguous mesh. Everything else, including Trn3, falls back to the
    arithmetic groups.
    """
    # Coerce unset dims (config may leave them None) to 1 so this never crashes.
    tp_size, cp_size, cfg_size = tp_size or 1, cp_size or 1, cfg_size or 1
    layout_a = cp_size == 8 and tp_size * cfg_size == 8
    layout_b = tp_size == 8 and cp_size > 1
    if not (layout_a or layout_b):
        return False

    try:
        target = get_platform_target()
    except (RuntimeError, ImportError):
        # Platform undetectable (e.g. CPU host) -> use mesh-agnostic arithmetic groups.
        return False
    try:
        family = normalize_target_family(target)
    except RuntimeError:
        family = None

    if family not in _PHYSICAL_MESH_PLATFORMS:
        # Trn3 intentionally keeps the arithmetic groups. Warn only for
        # platforms whose topology has not been classified.
        if target and family not in _ARITHMETIC_GROUP_PLATFORMS:
            logger.warning(
                "Physical-mesh CFG layout not enabled for platform '%s'; using "
                "arithmetic groups. tp_contiguous is only validated on %s.",
                target,
                sorted(_PHYSICAL_MESH_PLATFORMS),
            )
        return False

    if get_logical_nc_config() == 1:
        # The tp_contiguous mapping assumes LNC 2 (4 logical cores per chip). At LNC 1 rank r is
        # physical core r on chip r // 8, so an arithmetic TP8 group is one chip and a CP group
        # takes the same core of consecutive chips (whole torus rows). The LNC-2 mapping applied
        # to LNC-1 ranks gives groups such as [4..11] that the runtime rejects ("no_hier no_mesh").
        return False
    return True


def mesh_tp_cp_cfg_groups(
    tp_size: int, cp_size: int, cfg_size: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Return ``(tp_groups, cp_groups, cfg_groups)`` as PHYSICAL-mesh rank lists.

    Uses the validated ``tp_contiguous`` mapping on the 8x8 TRN2 mesh; which axis each
    of TP/CP/CFG maps to depends on the layout (see :func:`_tp_contiguous_mesh_groups`).
    Falls back to the arithmetic (contiguous-stride) partitions when the physical
    mesh does not apply (see :func:`_supports_physical_mesh`).

    Every returned partition covers ``range(tp*cp*cfg)`` exactly once.
    """
    world = tp_size * cp_size * cfg_size

    if not _supports_physical_mesh(tp_size, cp_size, cfg_size):
        tp_g = _tp_replica_groups(tp_size, world)
        cp_g = _cp_replica_groups(tp_size, cp_size, cfg_size)
        cfg_g = _cfg_replica_groups(tp_size, cp_size, cfg_size)
        return tp_g, cp_g, cfg_g

    # _supports_physical_mesh gates this to validated platforms only (trn2 today).
    return _tp_contiguous_mesh_groups(tp_size, cp_size, cfg_size)


# SGLang Diffusion's own groups come from one fixed rank order
# (``initialize_model_parallel``: ``RankGenerator(..., "tp-sp-pp-cfg-dp")``): TP = consecutive
# ranks, SP = stride ``tp``, CFG = stride ``tp*sp``. On trn2 its SP groups are off-ring
# (CP [0,8,...,56] spans chips two torus hops apart). Renaming SGLang's ranks through
# mesh_rank_permutation (the hook in sglang_patches/diffusion_mesh_rank_order.py) makes SGLang build
# exactly the mesh_tp_cp_cfg_groups partitions, so the models use SGLang's groups and the plugin builds
# no second set.
def _sglang_order_groups(axis: int, sizes: tuple[int, int, int]) -> list[list[int]]:
    """SGLang's groups along ``axis`` (0 = TP, 1 = SP, 2 = CFG) for pp = dp = 1."""
    tp, sp, cfg = sizes

    def rank(t: int, s: int, c: int) -> int:
        return t + tp * (s + sp * c)

    others = [(a, b) for b in range(sizes[(axis + 2) % 3]) for a in range(sizes[(axis + 1) % 3])]
    groups = []
    for a, b in others:
        coords = [0, 0, 0]
        coords[(axis + 1) % 3], coords[(axis + 2) % 3] = a, b
        group = []
        for i in range(sizes[axis]):
            coords[axis] = i
            group.append(rank(*coords))
        groups.append(group)
    return groups


def mesh_rank_permutation(tp_size: int, cp_size: int, cfg_size: int) -> list[int] | None:
    """Map SGLang Diffusion's rank order onto :func:`mesh_tp_cp_cfg_groups`.

    Returns ``perm`` (``perm[sglang_rank] = rank``) such that renaming every rank of SGLang's
    TP, SP and CFG groups through it gives exactly the TP, CP and CFG partitions of
    :func:`mesh_tp_cp_cfg_groups`, members in the same order; ``None`` when they already match
    (no renaming). A rank keeps its process and its NeuronCore; only its group membership
    follows the mesh. Rank 0 must stay rank 0: SGLang's scheduler takes each request on rank 0
    and relays it from the first rank of every group. Raises ``ValueError`` if the partitions
    are not a TP x CP x CFG grid of ``range(world)`` or would move rank 0.
    """
    sizes = (tp_size, cp_size, cfg_size)
    partitions = mesh_tp_cp_cfg_groups(tp_size, cp_size, cfg_size)
    world = tp_size * cp_size * cfg_size
    for partition in partitions:
        if sorted(r for g in partition for r in g) != list(range(world)):
            raise ValueError(
                f"a tp{tp_size} cp{cp_size} cfg{cfg_size} partition is not of 0..{world - 1}"
            )
    coords = {r: [-1, -1, -1] for r in range(world)}
    for axis, partition in enumerate(partitions):
        for group in partition:
            for position, r in enumerate(group):
                coords[r][axis] = position
    perm = [-1] * world
    for r, (t, s, c) in coords.items():
        sglang_rank = t + tp_size * (s + cp_size * c)
        if min(t, s, c) < 0 or perm[sglang_rank] != -1:
            raise ValueError(f"the tp{tp_size} cp{cp_size} cfg{cfg_size} groups are not a grid")
        perm[sglang_rank] = r
    for axis, partition in enumerate(partitions):
        renamed = [[perm[r] for r in g] for g in _sglang_order_groups(axis, sizes)]
        if sorted(renamed) != sorted(partition):
            raise ValueError(f"the tp{tp_size} cp{cp_size} cfg{cfg_size} groups are not a grid")
    if perm[0] != 0:
        raise ValueError(
            f"the tp{tp_size} cp{cp_size} cfg{cfg_size} layout does not start every group of rank 0 with it"
        )
    return None if perm == list(range(world)) else perm


def get_replica_groups(
    tp_size: int, cp_size: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Return ``(tp_groups, cp_groups)`` as tuples-of-tuples of GLOBAL ranks.

    :func:`get_cp_replica_groups` wraps it, so the CP partition comes from the same source as
    the TP one.
    """
    from sglang_neuron.diffusion.distributed.groups import get_classifier_free_guidance_world_size

    try:
        cfg_size = get_classifier_free_guidance_world_size()
    except AssertionError:
        cfg_size = 1
    tp_groups, cp_groups, _ = mesh_tp_cp_cfg_groups(tp_size, cp_size, cfg_size)
    return (
        tuple(tuple(g) for g in tp_groups),
        tuple(tuple(g) for g in cp_groups),
    )


def get_cp_replica_groups(tp_size: int, cp_size: int) -> tuple[tuple[int, ...], ...]:
    """Return the CP partition as a tuple-of-tuples of GLOBAL ranks.

    This is the ``replica_groups`` argument the ring-attention kernel needs: every
    CP ring in the world (each SPMD rank resolves its own ring internally), not just
    the caller's ring (``get_cp_group().ranks`` gives only the latter). See
    :func:`get_replica_groups` for the shared source and consistency guarantee.
    """
    return get_replica_groups(tp_size, cp_size)[1]


def _build_2d_mesh(
    ranks: list[int],
    row_size: int,
    num_rows: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """The trn2 64-rank 8 x 8 physical mesh as ``(row_groups, col_groups)``.

    Its only caller passes ``ranks = 0..63``, ``row_size = 8`` and ``num_rows = 8``, after
    :func:`_supports_physical_mesh` chose the physical mesh; any other size is refused.
    """
    if list(ranks) != list(range(64)) or row_size != 8 or num_rows != 8:
        raise ValueError(
            f"the physical mesh is the trn2 64-rank 8 x 8 mesh, not {num_rows} rows of {row_size}"
        )
    mesh = [
        [0, 1, 2, 3, 12, 13, 14, 15],
        [4, 5, 6, 7, 8, 9, 10, 11],
        [16, 17, 18, 19, 28, 29, 30, 31],
        [20, 21, 22, 23, 24, 25, 26, 27],
        [32, 33, 34, 35, 44, 45, 46, 47],
        [36, 37, 38, 39, 40, 41, 42, 43],
        [48, 49, 50, 51, 60, 61, 62, 63],
        [52, 53, 54, 55, 56, 57, 58, 59],
    ]

    row_groups = [list(row) for row in mesh]
    col_groups = [[mesh[r][c] for r in range(num_rows)] for c in range(row_size)]

    return row_groups, col_groups


def _factor_line(
    lines: list[list[int]], n_blocks: int, block_size: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Factor each mesh line into contiguous blocks and the strided groups across them.

    For every ``line`` (length ``n_blocks * block_size``) returns ``(blocks, strides)``:
      - ``blocks``  = the ``n_blocks`` contiguous ``block_size``-slices, and
      - ``strides`` = the ``block_size`` groups of the i-th element of each block
        (``line[i::block_size]``), each of length ``n_blocks``.

    Both mesh axes factor with this primitive but assign CFG to opposite sides so every
    group stays on a torus ring: a row is one torus-adjacent chip pair (blocks and strides
    are both on-ring), while a column walks chips in adjacent pairs, so only the adjacent
    (contiguous) split is on-ring there. Hence Layout A takes blocks=TP/strides=CFG and
    Layout B takes blocks=CFG/strides=CP -- see the callers in _tp_contiguous_mesh_groups.
    """
    blocks: list[list[int]] = []
    strides: list[list[int]] = []
    for line in lines:
        assert len(line) == n_blocks * block_size, (
            f"line {line} does not factor into {n_blocks} x {block_size}"
        )
        for b in range(n_blocks):
            blocks.append(line[b * block_size : (b + 1) * block_size])
        for i in range(block_size):
            strides.append(line[i::block_size])
    return blocks, strides


def _tp_contiguous_mesh_groups(
    tp_size: int, cp_size: int, cfg_size: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """The validated ``tp_contiguous`` mapping on the 8x8 physical mesh.

    One logical dim fills a length-8 mesh axis; the other two factor the second axis
    via :func:`_factor_line`, which assigns CFG to whichever side keeps every group on a
    torus ring (see that helper for the row-vs-column chip layout):
      - Layout A (``cp==8``): CP = columns; row axis factors into TP-contiguous blocks
        and CFG stride-``tp`` groups.
      - Layout B (``tp==8``): TP = rows; column axis factors into CFG-adjacent blocks
        and CP stride-``cfg`` groups (the opposite assignment).

    Derives the physical rank lines from ``_build_2d_mesh``. The caller gates on
    instance type (_PHYSICAL_MESH_PLATFORMS) so only the validated 8x8 fabrics use
    this path.
    """
    world = tp_size * cp_size * cfg_size

    rows, cols = _build_2d_mesh(list(range(64)), row_size=8, num_rows=8)

    if cp_size == 8:
        # Layout A: CP = columns; the row factors into TP blocks + CFG strides.
        tp_groups, cfg_groups = _factor_line(rows, n_blocks=cfg_size, block_size=tp_size)
        return tp_groups, cols, cfg_groups

    # Layout B (tp_size == 8): TP = rows; the column factors into CFG blocks + CP strides.
    # A `world`-rank job occupies the first world // 8 mesh rows, so the column is
    # truncated to that height (cfg=1 / 32 cores -> mesh rows 0-3 = ranks 0-31).
    num_rows = world // 8
    tp_groups = [list(row) for row in rows[:num_rows]]
    cfg_groups, cp_groups = _factor_line(
        [col[:num_rows] for col in cols], n_blocks=cp_size, block_size=cfg_size
    )
    return tp_groups, cp_groups, cfg_groups


def register_replica_groups(tp_size: int, cp_size: int) -> None:
    """Register full replica-group partitions for the TP, CP, and CFG process groups.

    The torch-native Neuron backend lowers c10d collectives to StableHLO and must
    derive HLO ``replica_groups`` that cover *all* ranks the op participates in.
    Its ``collective_legalization`` FX pass reads the collective's group name and
    calls torch-neuronx's mesh registry ``get_all_replica_groups(name)``.
    torch-neuronx fills that registry itself for groups created via a torch ``DeviceMesh``
    and (torch-neuronx 2.14.3) for groups built with ``new_group(...)`` (SGLang's ``GroupCoordinator``):
    its ``new_group`` hook records each partition with every group's ranks sorted
    ascending. SGLang's ``rank_in_group`` follows the order a group was declared in, and the
    trn2 physical-mesh CP groups are declared out of ascending order (for example
    ``[12, 8, 28, 24, 44, 40, 60, 56]``), so this writes each partition in the declared
    order into the registry, replacing torch-neuronx's entry; the validated runs use that
    order. This is purely a compile-time topology hint: the groups, the collectives, and
    the numerics are unchanged.

    The registered partitions MUST match the ranks the groups were built with.
    Trn2's 64-core 8x8 fabric uses the physical-mesh ``tp_contiguous`` mapping;
    Trn3 and other non-mesh layouts use arithmetic striding.
    ``mesh_tp_cp_cfg_groups`` returns whichever applies, keeping compiler metadata
    and live groups consistent.

    Args:
        tp_size: Number of tensor-parallel ranks.
        cp_size: Number of context-parallel ranks.
    """
    # The groups module binds SGLang's groups; it is imported here, not at module
    # load, so the partition helpers above can be used without SGLang.
    from sglang_neuron.diffusion.distributed.groups import (
        get_cfg_group,
        get_classifier_free_guidance_world_size,
        get_cp_group,
        get_tp_group,
    )

    # get_classifier_free_guidance_world_size() asserts a CFG group exists; callers
    # that don't run cfg-parallel (e.g. the cfg=1 golden tests) never create one, so
    # treat an uninitialized CFG group as cfg_size=1 rather than failing.
    try:
        cfg_size = get_classifier_free_guidance_world_size()
    except AssertionError:
        cfg_size = 1
    if tp_size <= 1 and cp_size <= 1 and cfg_size <= 1:
        return

    def _register(pg, groups: list[list[int]]) -> None:
        name = getattr(pg, "group_name", None)
        if name is None:
            logger.warning("Cannot register replica groups: process group has no group_name")
            return
        register_process_group_replica_groups(name, groups)
        logger.debug("Registered replica groups for PG '%s' -> %s", name, groups)

    # Register the SAME partitions the groups were built with: physical-mesh
    # tp_contiguous on the Trn2 64-core 8x8 fabric, else arithmetic striding.
    # mesh_tp_cp_cfg_groups picks the right one, so the registry can't disagree with
    # the live groups (a mismatch re-introduces the "no_hier no_mesh" failure).
    tp_groups, cp_groups, cfg_groups = mesh_tp_cp_cfg_groups(tp_size, cp_size, cfg_size)

    if tp_size > 1:
        _register(get_tp_group().device_group, tp_groups)

    if cp_size > 1:
        # GroupCoordinator.all_gather issues the collective on its device_group.
        _register(get_cp_group().device_group, cp_groups)

    if cfg_size > 1:
        _register(get_cfg_group().device_group, cfg_groups)


def initialize_parallel_groups(
    device,
    *,
    world_size: int,
    tensor_parallel_size: int = 1,
    context_parallel_size: int = 1,
    cfg_parallel_size: int = 1,
) -> list:
    """Bind the models to SGLang's TP / SP / CFG groups, then stamp ranks.

    SGLang's worker built the groups (on the :func:`mesh_tp_cp_cfg_groups` partitions, see
    ``sglang_patches.diffusion_mesh_rank_order``); ``groups.bind_groups`` checks they have exactly these
    partitions. :func:`register_replica_groups` registers the same partitions with the
    compiler.
    """
    from sglang_neuron.diffusion.distributed.groups import bind_groups, stamp_rank_tensors

    tp, cp, cfg = tensor_parallel_size or 1, context_parallel_size or 1, cfg_parallel_size or 1
    if tp * cp * cfg != world_size:
        raise ValueError(f"tp*cp*cfg = {tp}*{cp}*{cfg} != world size {world_size}")
    tp_g, cp_g, cfg_g = mesh_tp_cp_cfg_groups(tp, cp, cfg)
    logger.info("Neuron diffusion groups: tp[0]=%s cp[0]=%s cfg[0]=%s", tp_g[0], cp_g[0], cfg_g[0])
    bind_groups(tp_groups=tp_g, cp_groups=cp_g, cfg_groups=cfg_g)
    return stamp_rank_tensors(device)
