# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kimi-K3 multi-host decode routed experts: the EP x TP split table, the per-rank expert layout and the host re-shard.

The 896 routed experts split over a world of W ranks (128 a host) in G = W / r expert groups of E_L = 896 / G experts,
and each expert splits its intermediate (3,072) in r parts (I-slices) of I_L = 3072 / r. A split is exact when it
keeps the checkpoint's MXFP4 down scales: I_L is a whole number of 32-element MX groups (r divides 96), and G is a
power of 2 that divides 896 (G <= 128). ``ROUTED_COMBOS`` holds the combos of the multi-host decode with their layout
tags (``ep64xtp4-t256``: 64 groups, routed TP 4, narrowest MX tile 256). ``routed_split`` gives the split dict of any
exact combo.

MX tiles (``tile_plan``): the I-slice splits into tiles of 512 columns and one narrower last tile (768 = 512 + 256;
384, 192 and 96 are one tile). A tile of w columns has P = w / 4 rows of x4 words; the MX matmul takes a multiple of
32 partitions, so P pads up to ``tile_partitions(w)`` with zero rows in SBUF (48 -> 64, 24 -> 32; exact). The HBM
tensors hold no padding.

Rank order (``RANK_ORDERS``) places the r ranks of an EP group (its TP set):

- ``group_minor`` (the served EP128 x TP2 and every I-slice of whole 512 tiles): rank g holds group g % G, part
  g // G. This is the order of the served kernel and loader, so the served bytes do not change.
- ``part_minor`` (every I-slice with a narrower tile): rank g holds group g // r, part g % r. A TP set is r consecutive
  ranks: inside one device (8 LNC1 ranks) for r <= 8 and inside one host for r <= 32 (r = 32: exactly one TP32 replica,
  the rank lists of the TP32 group). Both cores of each HBM stack (g, g ^ 1) are in one TP set, so a TP-set collective
  can take a pair-shared HBM destination. A TP-set collective (for example an FP32 reduce of the r I-slice partials
  before the world reduce) stays on the links inside one host, where the runtime also runs SBUF collectives. The world
  routed reduce adds every rank, so the order does not change its cost.

Both orders are bit fields of g (G and r are powers of 2): group = (g >> group_shift) & (G - 1), part =
(g >> part_shift) & (r - 1). The experts of group e are [E_L e, E_L e + E_L) in both orders.

Per-rank expert layout (``STACK_LAYOUTS``), by ``layout_form``:

- ``served`` (whole 512 tiles): the layout of ``moe_tkg_decode``. Gate / up [E_L, 128, 2, H / 512, I_L] with scales
  [E_L, 16, 2, H / 512, I_L]; down [E_L, 128, n_i_tiles, H] with scales [E_L, 16, n_i_tiles, H] (row p of tile t holds
  the x4 word of intermediate 512 t + 4 p .. + 3; scale row s of tile t the group 16 t + s).
- ``slice`` (a narrower tile): gate / up [E_L, 128, H / 512, 2 I_L] with scales [E_L, 16, H / 512, 2 I_L]; down
  [E_L, I_L / 4, H] (row m: the x4 word of intermediate 4 m .. + 3 of the slice) with scales [E_L, I_L / 32, H] (row g:
  the MX group g of the slice). The 2 I_L gate / up columns are chunks of at most ``GATE_UP_CHUNK`` intermediate
  columns, each [gate | up]; a chunk never crosses a tile (``gate_up_chunks``: 768 -> [g256|u256] x 3, 384 ->
  [g256|u256|g128|u128], 192 -> [g192|u192], 96 -> [g96|u96]).

In both forms, row p of hidden tile k of gate / up holds the x4 word of h = 512 k + 4 p .. + 3 and scale row s the group
of h in [512 k + 32 s, + 32). The intermediate columns run in the down's order (``intermediate_order``): column
T + q P + p of a tile that starts at T, of P rows, holds intermediate T + 4 p + q, so the gate / up output of lane q
is a run of P columns.

The host re-shard (``rank_expert_tensors``) turns the checkpoint (per expert: ``w1`` gate and ``w3`` up [I, H / 2]
bytes with scales [I, H / 32], ``w2`` down [H, I / 2] bytes with scales [H, I / 32]; two FP4 values a byte, low nibble
first; E8M0 scales) into one rank's tensors, and ``checkpoint_part_of_rank`` / ``checkpoint_of_tp_set`` turn them back.
Numpy only, for the loader, the tests and the twins. The x4 words are uint16 (the bits of ``float4_e2m1fn_x4``; lane
q is bits 4 q .. 4 q + 3).
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np

NUM_EXPERTS = 896
"""Routed experts of the model (``tkg_moe.NUM_EXPERTS``)."""
EXPERT_INTERMEDIATE = 3072
"""Intermediate size of one routed expert (``tkg_moe.EXPERT_INTERMEDIATE``)."""
LATENT_SIZE = 3584
"""Hidden size H of the routed experts: the latent (``tkg_moe.LATENT_SIZE``)."""
HOST_RANKS = 128
"""LNC1 ranks of one host (``tkg_ffn_world_cc.HOST_RANKS``)."""
DEVICE_RANKS = 8
"""LNC1 ranks of one device: 8 consecutive ranks, 2 on each HBM stack (g and g ^ 1)."""
ATTN_TP = 32
"""Ranks of one attention replica (``tkg_ffn_world_cc.ATTN_TP``): rank g has TP rank g % 32 and DP rank g // 32."""
MX_GROUP = 32
"""Values of one MX scale group (E8M0 scale)."""
FP4_PER_WORD = 4
"""FP4 values of one x4 word (uint16)."""
HIDDEN_TILE = 512
"""Hidden values of one gate / up K tile."""
HIDDEN_ROWS = HIDDEN_TILE // FP4_PER_WORD
"""Rows of x4 words of one hidden tile (128)."""
HIDDEN_SCALE_ROWS = HIDDEN_TILE // MX_GROUP
"""Scale rows of one hidden tile (16)."""
SERVED_MX_TILE = 512
"""Intermediate columns of the served MX tile, and the widest tile."""
MX_PARTITION_STEP = 32
"""The MX matmul takes a multiple of 32 partitions (``tile_partitions``)."""
GATE_UP_CHUNK = 256
"""Intermediate columns of one [gate | up] chunk of the ``slice`` gate / up layout."""
LAYOUT_VERSION = 1
"""Version of the per-rank byte layout (``STACK_LAYOUTS``, ``intermediate_order``, ``gate_up_columns``). A change of
the bytes changes it, so a loader record of another version fails ``check_layout_record``."""

RANK_ORDERS = ("group_minor", "part_minor")
"""The placements of the TP sets: group = g % G (``group_minor``) or group = g // r (``part_minor``)."""
LAYOUT_FORMS = ("served", "slice")
"""The expert tensor layouts: whole 512 tiles (``served``) or an I-slice with a narrower tile (``slice``)."""

ROUTED_COMBOS = {
    "ep128xtp2-t512": (256, 2, "group_minor"),
    "ep64xtp4-t256": (256, 4, "part_minor"),
    "ep32xtp8-t384": (256, 8, "part_minor"),
    "ep8xtp32-t96": (256, 32, "part_minor"),
    "ep128xtp4-t256": (512, 4, "part_minor"),
    "ep64xtp8-t384": (512, 8, "part_minor"),
    "ep32xtp16-t192": (512, 16, "part_minor"),
    "ep16xtp32-t96": (512, 32, "part_minor"),
}
"""Layout tag -> (world, routed_tp, rank order) of the multi-host decode combos. ``ep128xtp2-t512`` is the served
2-host split (``moe_2srv_split(256)``)."""
SERVED_TAG = "ep128xtp2-t512"
"""The tag of the served 2-host split."""

STACK_LAYOUTS = {
    "served": {
        "expert_gate_up_weights": (np.uint16, ("E_L", "P", "GU", "N_H", "I_L")),
        "expert_gate_up_weights_scale": (np.uint8, ("E_L", "S", "GU", "N_H", "I_L")),
        "expert_down_weights": (np.uint16, ("E_L", "P", "N_I", "H")),
        "expert_down_weights_scale": (np.uint8, ("E_L", "S", "N_I", "H")),
    },
    "slice": {
        "expert_gate_up_weights": (np.uint16, ("E_L", "P", "N_H", "GU_I")),
        "expert_gate_up_weights_scale": (np.uint8, ("E_L", "S", "N_H", "GU_I")),
        "expert_down_weights": (np.uint16, ("E_L", "ROWS", "H")),
        "expert_down_weights_scale": (np.uint8, ("E_L", "GROUPS", "H")),
    },
}
"""layout form -> expert tensor -> (dtype, axes). The axes: E_L local experts; P = 128 rows of x4 words (S = 16 scale
rows) of a hidden tile or of a 512 tile; GU = 2 (gate, up); N_H = H / 512 hidden tiles; I_L intermediate columns;
GU_I = 2 I_L ([gate | up] chunks); N_I = I_L / 512 tiles; H hidden; ROWS = I_L / 4; GROUPS = I_L / 32."""
EXPERT_TENSORS = tuple(STACK_LAYOUTS["served"])
"""The four per-rank expert tensors of a layer, in the order of ``moe_2srv_expert_shapes``."""
CHECKPOINT_KEYS = (
    "w1.weight_packed",
    "w1.weight_scale",
    "w3.weight_packed",
    "w3.weight_scale",
    "w2.weight_packed",
    "w2.weight_scale",
)
"""The checkpoint tensors of one expert: gate (w1), up (w3), down (w2), each the FP4 bytes and the E8M0 scales."""
ROUTED_LAYERS_ARGUMENT = "routed_layers"
"""The decode entry argument of a per-layer plan (``check_layer_combos`` ``entry``); None keeps one split."""
MAX_ALT_TAGS = 2
"""Most layout tags of a per-layer plan besides its default tag: each takes its own expert stacks
(``expert_stack_name``), in order of their first layer."""
_RECORD_FIELDS = ("world", "routed_tp", "mx_tile", "rank_order", "layout_form", "layout_tag", "layout_version")
"""The fields of a split that a loader record repeats (``layout_record``, ``check_layout_record``)."""


def _fail(condition: bool, message: str) -> None:
    """Raise ValueError with ``message`` unless ``condition`` (the checks of this module fail closed)."""
    if not condition:
        raise ValueError(message)


def _is_power_of_2(value: int) -> bool:
    return isinstance(value, int) and value >= 1 and value & (value - 1) == 0


# ---- Tiles ----


def tile_plan(local_intermediate: int) -> tuple[int, ...]:
    """Return the MX tile widths of an I-slice: tiles of 512, then the rest (a whole number of MX groups).

    Example:
        >>> tile_plan(1536), tile_plan(768), tile_plan(96)
        ((512, 512, 512), (512, 256), (96,))
    """
    _fail(
        local_intermediate > 0 and local_intermediate % MX_GROUP == 0,
        f"an I-slice of {local_intermediate} columns is not a whole number of MX groups of {MX_GROUP}",
    )
    whole, rest = divmod(local_intermediate, SERVED_MX_TILE)
    return (SERVED_MX_TILE,) * whole + ((rest,) if rest else ())


def tile_partitions(width: int) -> int:
    """Return the partitions of the down tile of ``width`` columns: P = width / 4 rows, padded to a multiple of 32.

    Example:
        >>> tile_partitions(512), tile_partitions(384), tile_partitions(192), tile_partitions(96)
        (128, 96, 64, 32)
    """
    rows = width // FP4_PER_WORD
    return -(-rows // MX_PARTITION_STEP) * MX_PARTITION_STEP


def intermediate_order(local_intermediate: int) -> np.ndarray:
    """Return the intermediate of each gate / up column (the down's order): column T + q P + p of a tile that starts at
    T, of P = w / 4 rows, holds intermediate T + 4 p + q.

    Example:
        >>> intermediate_order(96)[:3].tolist(), intermediate_order(96)[24].item()
        ([0, 4, 8], 1)
    """
    order, start = [], 0
    for width in tile_plan(local_intermediate):
        rows = width // FP4_PER_WORD
        q, p = np.divmod(np.arange(width), rows)
        order.append(start + FP4_PER_WORD * p + q)
        start += width
    return np.concatenate(order)


def gate_up_columns(split: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Return (projection, intermediate) of each column of the last gate / up axis: for ``served`` (I_L columns)
    projection is None and the columns follow ``intermediate_order``; for ``slice`` (2 I_L columns) each tile splits
    into chunks of at most ``GATE_UP_CHUNK`` ordered columns (a chunk never crosses a tile), each its gate columns
    (projection 0) then its up columns (projection 1).

    Example:
        >>> gu, i = gate_up_columns(routed_split(256, 8))
        >>> gu[255:257].tolist(), i[256].item(), gu[512:514].tolist()
        ([0, 1], 0, [0, 0])
    """
    order = intermediate_order(split["local_intermediate"])
    if split["layout_form"] == "served":
        return None, order
    projection, columns, start = [], [], 0
    for width in tile_plan(split["local_intermediate"]):
        for chunk_start in range(start, start + width, GATE_UP_CHUNK):
            chunk = order[chunk_start : min(chunk_start + GATE_UP_CHUNK, start + width)]
            for gu in (0, 1):
                projection.append(np.full(len(chunk), gu))
                columns.append(chunk)
        start += width
    return np.concatenate(projection), np.concatenate(columns)


def gate_up_chunks(local_intermediate: int) -> tuple[tuple[int, int], ...]:
    """Return the [gate | up] chunks (start, width) of the ordered intermediate columns of a ``slice`` I-slice.

    Example:
        >>> gate_up_chunks(768), gate_up_chunks(384), gate_up_chunks(96)
        (((0, 256), (256, 256), (512, 256)), ((0, 256), (256, 128)), ((0, 96),))
    """
    chunks, start = [], 0
    for width in tile_plan(local_intermediate):
        chunks += [(c, min(GATE_UP_CHUNK, start + width - c)) for c in range(start, start + width, GATE_UP_CHUNK)]
        start += width
    return tuple(chunks)


# ---- Splits ----


def layout_tag(groups: int, routed_tp: int, mx_tile: int, rank_order: str, layout_form: str) -> str:
    """Return the layout tag of a split: ``ep{G}xtp{r}-t{mx_tile}``, with ``-{rank_order}`` when the order is not the
    default of the form (``default_rank_order``).

    Example:
        >>> layout_tag(64, 4, 256, "part_minor", "slice"), layout_tag(64, 4, 256, "group_minor", "slice")
        ('ep64xtp4-t256', 'ep64xtp4-t256-group_minor')
    """
    tag = f"ep{groups}xtp{routed_tp}-t{mx_tile}"
    return tag if rank_order == default_rank_order(layout_form) else f"{tag}-{rank_order}"


def default_rank_order(layout_form: str) -> str:
    """Return the default rank order of a layout form: ``group_minor`` for ``served``, else ``part_minor``."""
    return "group_minor" if layout_form == "served" else "part_minor"


def routed_split(world_size: int, routed_tp: int, mx_tile: int = None, rank_order: str = None) -> dict:
    """Return the split dict of an exact EP x TP split of ``world_size`` ranks.

    Args:
        world_size: ranks of the world, a whole number of hosts.
        routed_tp: r, the intermediate parts of each expert.
        mx_tile: None, or the narrowest MX tile of the I-slice (``tile_plan``), which then must agree.
        rank_order: ``RANK_ORDERS``; None gives the order of the combo in ``ROUTED_COMBOS``, else
            ``default_rank_order``.

    Returns:
        dict with ``world``, ``dp`` (world / 32), ``nnodes`` (hosts), ``routed_tp``, ``groups`` (G),
        ``local_experts`` (E_L), ``local_intermediate`` (I_L), ``mx_groups`` (I_L / 32), ``tiles`` (``tile_plan``),
        ``tile_partitions`` (padded partitions of each tile), ``mx_tile`` (the narrowest tile), ``n_i_tiles``,
        ``layout_form`` (``served`` for whole 512 tiles, else ``slice``), ``rank_order``, ``group_shift`` and
        ``part_shift`` (group = (g >> group_shift) & (G - 1), part = (g >> part_shift) & (r - 1)), ``layout_tag`` and
        ``layout_version``.

    Raises:
        ValueError: the split is not exact (I_L not a whole number of MX groups, G not a power of 2 that divides 896,
            r not a power of 2), ``mx_tile`` is not the I-slice's tile, or the world is not whole hosts.

    Example:
        >>> routed_split(256, 4)["layout_tag"], routed_split(256, 4)["tiles"], routed_split(256, 4)["local_experts"]
        ('ep64xtp4-t256', (512, 256), 14)
    """
    _fail(
        isinstance(world_size, int) and world_size > 0 and world_size % HOST_RANKS == 0,
        f"a multi-host world takes whole hosts of {HOST_RANKS} ranks, got {world_size}",
    )
    _fail(
        _is_power_of_2(routed_tp) and world_size % routed_tp == 0,
        f"routed_tp {routed_tp} must be a power of 2 that divides the world {world_size}",
    )
    groups = world_size // routed_tp
    _fail(
        _is_power_of_2(groups) and NUM_EXPERTS % groups == 0,
        f"{groups} expert groups (world {world_size} / routed_tp {routed_tp}) must be a power of 2 that divides "
        f"{NUM_EXPERTS}",
    )
    local_intermediate = EXPERT_INTERMEDIATE // routed_tp
    _fail(
        EXPERT_INTERMEDIATE % routed_tp == 0 and local_intermediate % MX_GROUP == 0,
        f"routed_tp {routed_tp} gives an I-slice of {EXPERT_INTERMEDIATE / routed_tp} columns: an exact split keeps "
        f"whole MX groups of {MX_GROUP} (routed_tp must divide {EXPERT_INTERMEDIATE // MX_GROUP})",
    )
    tiles = tile_plan(local_intermediate)
    _fail(
        mx_tile is None or mx_tile == tiles[-1],
        f"mx_tile {mx_tile}: the I-slice {local_intermediate} of routed_tp {routed_tp} takes the MX tiles {tiles}, "
        f"so its mx_tile is {tiles[-1]}",
    )
    layout_form = "served" if tiles[-1] == SERVED_MX_TILE else "slice"
    if rank_order is None:
        rank_order = default_rank_order(layout_form)
        for combo in ROUTED_COMBOS.values():
            if combo[:2] == (world_size, routed_tp):
                rank_order = combo[2]
    _fail(rank_order in RANK_ORDERS, f"rank_order must be one of {RANK_ORDERS}, got {rank_order!r}")
    group_bits, part_bits = groups.bit_length() - 1, routed_tp.bit_length() - 1
    return {
        "world": world_size,
        "dp": world_size // ATTN_TP,
        "nnodes": world_size // HOST_RANKS,
        "routed_tp": routed_tp,
        "groups": groups,
        "local_experts": NUM_EXPERTS // groups,
        "local_intermediate": local_intermediate,
        "mx_groups": local_intermediate // MX_GROUP,
        "tiles": tiles,
        "tile_partitions": tuple(tile_partitions(width) for width in tiles),
        "mx_tile": tiles[-1],
        "n_i_tiles": len(tiles),
        "layout_form": layout_form,
        "rank_order": rank_order,
        "group_shift": 0 if rank_order == "group_minor" else part_bits,
        "part_shift": group_bits if rank_order == "group_minor" else 0,
        "layout_tag": layout_tag(groups, routed_tp, tiles[-1], rank_order, layout_form),
        "layout_version": LAYOUT_VERSION,
    }


def combo_split(tag: str) -> dict:
    """Return the split dict of a tag of ``ROUTED_COMBOS``.

    Raises:
        ValueError: the tag is not in the table, or its entry does not give the tag.
    """
    _fail(tag in ROUTED_COMBOS, f"layout tag {tag!r} is not one of {tuple(ROUTED_COMBOS)}")
    world, routed_tp, rank_order = ROUTED_COMBOS[tag]
    split = routed_split(world, routed_tp, rank_order=rank_order)
    _fail(split["layout_tag"] == tag, f"ROUTED_COMBOS entry {tag!r} gives the tag {split['layout_tag']!r}")
    return split


def combo_tags(world_size: int) -> tuple[str, ...]:
    """Return the tags of ``ROUTED_COMBOS`` of a world, in table order.

    Example:
        >>> combo_tags(512)[0]
        'ep128xtp4-t256'
    """
    return tuple(tag for tag, combo in ROUTED_COMBOS.items() if combo[0] == world_size)


def check_layout_tag(
    tag: str, world_size: int, routed_tp: int = None, mx_tile: int = None, rank_order: str = None
) -> dict:
    """Return the split of ``tag`` for a world of ``world_size`` ranks; every other argument that is not None must
    agree with it (the layout-tag check of the kernel and the loader). A tag that is not in ``ROUTED_COMBOS`` passes
    only when it is the tag of ``routed_split(world_size, routed_tp, mx_tile, rank_order)``.

    Raises:
        ValueError: the tag is unknown for this world, or an argument differs.

    Example:
        >>> check_layout_tag("ep64xtp4-t256", 256, routed_tp=4)["tiles"]
        (512, 256)
    """
    _fail(isinstance(tag, str) and bool(tag), f"the layout tag must be a non-empty string, got {tag!r}")
    if tag in ROUTED_COMBOS:
        split = combo_split(tag)
    else:
        _fail(routed_tp is not None, f"layout tag {tag!r} is not in the table; give its routed_tp to derive it")
        split = routed_split(world_size, routed_tp, mx_tile, rank_order)
        _fail(split["layout_tag"] == tag, f"layout tag {tag!r} is not the tag {split['layout_tag']!r} of its split")
    given = {"world": world_size, "routed_tp": routed_tp, "mx_tile": mx_tile, "rank_order": rank_order}
    wrong = {key: (value, split[key]) for key, value in given.items() if value is not None and value != split[key]}
    _fail(not wrong, f"layout tag {tag!r} differs in {wrong} (given, table)")
    return split


def check_split(split: Mapping) -> dict:
    """Return the table split of ``split["layout_tag"]`` when every field of ``split`` that the table split has agrees
    (the fail-closed check of a split dict that a kernel body or a loader received).

    Raises:
        ValueError: the split has no tag, the tag is unknown, or a field differs.
    """
    _fail("layout_tag" in split, "the split has no layout_tag: build it with routed_split or combo_split")
    known = check_layout_tag(split["layout_tag"], split["world"], split.get("routed_tp"), None, split.get("rank_order"))
    wrong = {key: (split[key], value) for key, value in known.items() if key in split and split[key] != value}
    _fail(not wrong, f"split {split['layout_tag']!r} differs from the layout table in {wrong} (given, table)")
    return known


def layout_record(split: Mapping) -> dict:
    """Return the record of a split that a loader keeps with the weights it built (``check_layout_record``)."""
    return {key: split[key] for key in _RECORD_FIELDS}


def check_layout_record(record: Mapping, world_size: int) -> dict:
    """Return the split of a loader record (``layout_record``) after checking it against this module's table.

    Raises:
        ValueError: a field is missing, the version differs (a re-shard of another layout), the tag is unknown, a field
            differs, or the world differs.
    """
    missing = [key for key in _RECORD_FIELDS if key not in record]
    _fail(not missing, f"the layout record has no {missing}: the loader of these weights wrote no layout record")
    _fail(
        record["layout_version"] == LAYOUT_VERSION,
        f"the weights have the layout version {record['layout_version']}, and this nkilib reads {LAYOUT_VERSION}: "
        "re-shard the checkpoint with this nkilib",
    )
    _fail(record["world"] == world_size, f"the weights are for a world of {record['world']}, not {world_size}")
    split = check_layout_tag(
        record["layout_tag"], world_size, record["routed_tp"], record["mx_tile"], record["rank_order"]
    )
    _fail(
        record["layout_form"] == split["layout_form"],
        f"the weights have the layout form {record['layout_form']!r}, and {record['layout_tag']!r} is "
        f"{split['layout_form']!r}",
    )
    return split


# ---- Ranks ----


def rank_group_part(split: Mapping, rank: int) -> tuple[int, int]:
    """Return (expert group, intermediate part) of a global rank.

    Example:
        >>> rank_group_part(routed_split(256, 4), 5), rank_group_part(routed_split(256, 2), 129)
        ((1, 1), (1, 1))
    """
    _fail(0 <= rank < split["world"], f"rank {rank} is outside the world of {split['world']} ranks")
    return (
        (rank >> split["group_shift"]) & (split["groups"] - 1),
        (rank >> split["part_shift"]) & (split["routed_tp"] - 1),
    )


def rank_of(split: Mapping, group: int, part: int) -> int:
    """Return the global rank that holds part ``part`` of the experts of ``group``."""
    _fail(0 <= group < split["groups"], f"group {group} is outside the {split['groups']} expert groups")
    _fail(0 <= part < split["routed_tp"], f"part {part} is outside the {split['routed_tp']} parts")
    return (group << split["group_shift"]) | (part << split["part_shift"])


def group_experts(split: Mapping, group: int) -> tuple[int, ...]:
    """Return the expert ids of a group: [E_L group, E_L group + E_L)."""
    _fail(0 <= group < split["groups"], f"group {group} is outside the {split['groups']} expert groups")
    count = split["local_experts"]
    return tuple(range(group * count, (group + 1) * count))


def rank_experts(split: Mapping, rank: int) -> tuple[int, ...]:
    """Return the expert ids of a rank."""
    return group_experts(split, rank_group_part(split, rank)[0])


def part_rows(split: Mapping, part: int) -> slice:
    """Return the intermediate rows [I_L part, I_L part + I_L) of a part."""
    _fail(0 <= part < split["routed_tp"], f"part {part} is outside the {split['routed_tp']} parts")
    size = split["local_intermediate"]
    return slice(part * size, (part + 1) * size)


def tp_set_ranks(split: Mapping, group: int) -> tuple[int, ...]:
    """Return the ranks of the TP set of an expert group, in part order (the rank of part 0 first)."""
    return tuple(rank_of(split, group, part) for part in range(split["routed_tp"]))


def tp_set_lists(split: Mapping) -> list[list[int]]:
    """Return the rank lists of all TP sets, one for each group in group order (the rank lists of a TP-set replica
    group)."""
    return [list(tp_set_ranks(split, group)) for group in range(split["groups"])]


def ep_set_lists(split: Mapping) -> list[list[int]]:
    """Return the rank lists of the EP sets: one for each part, the G ranks that hold that part, in group order."""
    return [[rank_of(split, group, part) for group in range(split["groups"])] for part in range(split["routed_tp"])]


def tp_set_placement(split: Mapping) -> dict:
    """Return where the TP sets lie: ``hosts`` and ``devices`` (the most of either that one TP set spans),
    ``in_replica`` (every TP set inside one TP32 replica) and ``pairs`` (both cores of each HBM stack, g and g ^ 1, in
    one TP set).

    Example:
        >>> tp_set_placement(routed_split(256, 4))["devices"], tp_set_placement(routed_split(256, 2))["hosts"]
        (1, 2)
    """
    sets = tp_set_lists(split)
    return {
        "hosts": max(len({rank // HOST_RANKS for rank in ranks}) for ranks in sets),
        "devices": max(len({rank // DEVICE_RANKS for rank in ranks}) for ranks in sets),
        "in_replica": all(len({rank // ATTN_TP for rank in ranks}) == 1 for ranks in sets),
        "pairs": all(rank ^ 1 in set(ranks) for ranks in sets for rank in ranks),
    }


# ---- Shapes ----


def _axis_sizes(split: Mapping, hidden: int) -> dict:
    i_l = split["local_intermediate"]
    return {
        "E_L": split["local_experts"],
        "P": HIDDEN_ROWS,
        "S": HIDDEN_SCALE_ROWS,
        "GU": 2,
        "N_H": hidden // HIDDEN_TILE,
        "I_L": i_l,
        "GU_I": 2 * i_l,
        "N_I": i_l // SERVED_MX_TILE,
        "H": hidden,
        "ROWS": i_l // FP4_PER_WORD,
        "GROUPS": i_l // MX_GROUP,
    }


def rank_expert_shapes(split: Mapping, hidden: int = LATENT_SIZE) -> dict:
    """Return the shapes of one rank's four expert tensors of one layer (``STACK_LAYOUTS``).

    Example:
        >>> rank_expert_shapes(routed_split(256, 4))["expert_down_weights"]
        (14, 192, 3584)
    """
    _fail(hidden % HIDDEN_TILE == 0, f"the hidden size {hidden} must be a multiple of {HIDDEN_TILE}")
    sizes = _axis_sizes(split, hidden)
    layout = STACK_LAYOUTS[split["layout_form"]]
    return {name: tuple(sizes[axis] for axis in axes) for name, (_, axes) in layout.items()}


def rank_expert_dtypes(split: Mapping) -> dict:
    """Return the numpy dtype of each expert tensor (uint16 holds the bits of ``float4_e2m1fn_x4``)."""
    return {name: dtype for name, (dtype, _) in STACK_LAYOUTS[split["layout_form"]].items()}


def rank_expert_bytes(split: Mapping, hidden: int = LATENT_SIZE) -> int:
    """Return the bytes of one rank's expert tensors of one layer (the same for every exact split of a world)."""
    shapes, dtypes = rank_expert_shapes(split, hidden), rank_expert_dtypes(split)
    return sum(int(np.prod(shape)) * np.dtype(dtypes[name]).itemsize for name, shape in shapes.items())


# ---- Per-layer plans ----

ROUTED_ALT_ARGUMENTS = tuple(f"routed_alt{slot}_{name}" for slot in range(MAX_ALT_TAGS) for name in EXPERT_TENSORS)
"""The decode / verify entry arguments of the alternate expert stacks of a per-layer plan, slot-major
(``expert_stack_name``): ``routed_alt{k}_{tensor}`` is the [layers of the tag, ...] stack of the k-th other tag of the
plan (in order of its first layer), in that tag's per-rank layout (``rank_expert_shapes``), its layers in layer order
(``check_layer_combos`` ``index``)."""


def expert_stack_name(name: str, slot: int = None) -> str:
    """Return the entry stack name of expert tensor ``name`` for a plan slot: the name itself for the default tag
    (slot None), else ``routed_alt{slot}_{name}`` (``MAX_ALT_TAGS`` slots).

    Example:
        >>> expert_stack_name("expert_down_weights"), expert_stack_name("expert_down_weights", 0)
        ('expert_down_weights', 'routed_alt0_expert_down_weights')
    """
    if slot is None:
        return name
    _fail(0 <= slot < MAX_ALT_TAGS, f"a per-layer plan has {MAX_ALT_TAGS} alternate stack slots, got slot {slot}")
    return f"routed_alt{slot}_{name}"


def check_layer_combos(
    world_size: int, moe_layers: Sequence[int], default_tag: str, layer_tags: Mapping = None
) -> dict:
    """Check a per-layer combo plan and return it in the forms of the kernel and the loader (the one validation
    function of a per-layer plan).

    Args:
        world_size: ranks of the world.
        moe_layers: the indices of the MoE layers of the model, in order.
        default_tag: the layout tag of every MoE layer that ``layer_tags`` does not name.
        layer_tags: None, or layer index -> layout tag (for example the hot layers -> ``ep64xtp4-t256``).

    Returns:
        dict with ``tags`` (layer -> tag, every MoE layer), ``splits`` (tag -> split), ``stacks`` (tag -> the MoE layers
        of that tag, in order), ``index`` (layer -> (tag, position of the layer in the stacks of its tag)),
        ``stack_names`` (tag -> expert tensor -> entry stack name, ``expert_stack_name``: the default tag keeps the
        names, the other tags take the alternate slots in order of their first layer) and ``entry`` (the
        ``ROUTED_LAYERS_ARGUMENT`` value: (default tag, ((layer, tag) of each layer with another tag, ...))).

    Raises:
        ValueError: a tag is unknown for the world, a layer of ``layer_tags`` is not a MoE layer, the MoE layers
            repeat or are not in order, or the plan has more than ``MAX_ALT_TAGS`` other tags.

    Example:
        >>> plan = check_layer_combos(256, (1, 2, 3), "ep128xtp2-t512", {2: "ep64xtp4-t256"})
        >>> plan["stacks"]["ep64xtp4-t256"], plan["index"][3]
        ((2,), ('ep128xtp2-t512', 1))
    """
    layers = tuple(int(layer) for layer in moe_layers)
    _fail(bool(layers), "a per-layer plan needs at least one MoE layer")
    _fail(len(set(layers)) == len(layers), f"the MoE layers {layers} repeat an index")
    _fail(layers == tuple(sorted(layers)), f"the MoE layers {layers} must be in order")
    overrides = {int(layer): tag for layer, tag in (layer_tags or {}).items()}
    outside = sorted(set(overrides) - set(layers))
    _fail(not outside, f"layer_tags names the layers {outside}, which are not MoE layers")
    tags = {layer: overrides.get(layer, default_tag) for layer in layers}
    splits = {tag: check_layout_tag(tag, world_size) for tag in sorted(set(tags.values()) | {default_tag})}
    stacks = {tag: tuple(layer for layer in layers if tags[layer] == tag) for tag in splits}
    index = {layer: (tag, stacks[tag].index(layer)) for layer, tag in tags.items()}
    alternates = sorted((tag for tag in splits if tag != default_tag and stacks[tag]), key=lambda tag: stacks[tag][0])
    _fail(
        len(alternates) <= MAX_ALT_TAGS,
        f"a per-layer plan takes at most {MAX_ALT_TAGS} layout tags besides its default, got {alternates}",
    )
    slots = {tag: alternates.index(tag) if tag in alternates else None for tag in splits}
    stack_names = {tag: {name: expert_stack_name(name, slots[tag]) for name in EXPERT_TENSORS} for tag in splits}
    entry = (default_tag, tuple((layer, tag) for layer, tag in tags.items() if tag != default_tag))
    return {
        "tags": tags,
        "splits": splits,
        "stacks": stacks,
        "index": index,
        "stack_names": stack_names,
        "entry": entry,
    }


# ---- Host re-shard (numpy) ----


def _checked(array, dtype, shape: tuple, label: str) -> np.ndarray:
    array = np.asarray(array)
    _fail(array.dtype == dtype, f"{label} must be {np.dtype(dtype)}, got {array.dtype}")
    _fail(tuple(array.shape) == tuple(shape), f"{label} must have the shape {tuple(shape)}, got {tuple(array.shape)}")
    return array


def _words(array: np.ndarray) -> np.ndarray:
    """Pair adjacent bytes of the last axis into little-endian x4 words (byte 2k low, byte 2k + 1 high)."""
    return np.ascontiguousarray(array).view("<u2").astype(np.uint16)


def _bytes(words: np.ndarray) -> np.ndarray:
    """Split the x4 words of the last axis into their two bytes (low first)."""
    return np.ascontiguousarray(words.astype("<u2")).view(np.uint8)


def checkpoint_expert_part(expert: Mapping[str, np.ndarray], split: Mapping, part: int) -> dict:
    """Return the intermediate part ``part`` of one checkpoint expert (``CHECKPOINT_KEYS``): the rows [I_L part, + I_L)
    of w1 and w3 and their scales, the bytes [I_L part / 2, + I_L / 2) and the scale groups [I_L part / 32, + I_L / 32)
    of each w2 row."""
    rows = part_rows(split, part)
    down_bytes = slice(rows.start // 2, rows.stop // 2)
    down_groups = slice(rows.start // MX_GROUP, rows.stop // MX_GROUP)
    return {
        "w1.weight_packed": expert["w1.weight_packed"][rows],
        "w1.weight_scale": expert["w1.weight_scale"][rows],
        "w3.weight_packed": expert["w3.weight_packed"][rows],
        "w3.weight_scale": expert["w3.weight_scale"][rows],
        "w2.weight_packed": expert["w2.weight_packed"][:, down_bytes],
        "w2.weight_scale": expert["w2.weight_scale"][:, down_groups],
    }


def _part_shapes(split: Mapping, hidden: int) -> dict:
    i_l = split["local_intermediate"]
    return {
        "w1.weight_packed": (i_l, hidden // 2),
        "w1.weight_scale": (i_l, hidden // MX_GROUP),
        "w3.weight_packed": (i_l, hidden // 2),
        "w3.weight_scale": (i_l, hidden // MX_GROUP),
        "w2.weight_packed": (hidden, i_l // 2),
        "w2.weight_scale": (hidden, i_l // MX_GROUP),
    }


def _gate_up_layout(canonical: np.ndarray, split: Mapping) -> np.ndarray:
    """[E, rows, N_H, GU, I] (I in intermediate order) -> the gate / up layout of the split's form."""
    projection, columns = gate_up_columns(split)
    if split["layout_form"] == "served":
        return canonical[..., columns].transpose(0, 1, 3, 2, 4)
    return canonical[:, :, :, projection, columns]


def repack_gate_up_weights(gate: np.ndarray, up: np.ndarray, split: Mapping) -> np.ndarray:
    """Return ``expert_gate_up_weights`` (uint16) of the w1 (gate) and w3 (up) FP4 bytes [E, I_L, H / 2] (uint8) of
    the parts of E experts."""
    e_l, i_l, half = np.shape(gate)
    hidden = 2 * half
    _fail(np.shape(up) == np.shape(gate), f"the gate {np.shape(gate)} and up {np.shape(up)} parts differ")
    _fail(
        i_l == split["local_intermediate"],
        f"the parts have {i_l} intermediate rows, the split {split['local_intermediate']}",
    )
    words = np.stack(
        [
            _words(_checked(gate, np.uint8, (e_l, i_l, half), "w1")),
            _words(_checked(up, np.uint8, (e_l, i_l, half), "w3")),
        ],
        axis=1,
    )
    # [E, GU, I, H / 4] -> [E, P, N_H, GU, I] (row p of hidden tile k: h = 512 k + 4 p .. + 3).
    words = words.reshape(e_l, 2, i_l, hidden // HIDDEN_TILE, HIDDEN_ROWS).transpose(0, 4, 3, 1, 2)
    return np.ascontiguousarray(_gate_up_layout(words, split)).astype(np.uint16, copy=False)


def repack_gate_up_scales(gate: np.ndarray, up: np.ndarray, split: Mapping) -> np.ndarray:
    """Return ``expert_gate_up_weights_scale`` (uint8) of the w1 and w3 E8M0 scales [E, I_L, H / 32] (uint8)."""
    e_l, i_l, groups = np.shape(gate)
    hidden = groups * MX_GROUP
    _fail(np.shape(up) == np.shape(gate), f"the gate {np.shape(gate)} and up {np.shape(up)} scales differ")
    _fail(
        i_l == split["local_intermediate"],
        f"the scales have {i_l} intermediate rows, the split {split['local_intermediate']}",
    )
    scales = np.stack(
        [
            _checked(gate, np.uint8, (e_l, i_l, groups), "w1 scale"),
            _checked(up, np.uint8, (e_l, i_l, groups), "w3 scale"),
        ],
        axis=1,
    )
    scales = scales.reshape(e_l, 2, i_l, hidden // HIDDEN_TILE, HIDDEN_SCALE_ROWS).transpose(0, 4, 3, 1, 2)
    return np.ascontiguousarray(_gate_up_layout(scales, split))


def repack_down_weights(down: np.ndarray, split: Mapping) -> np.ndarray:
    """Return ``expert_down_weights`` (uint16) of the w2 FP4 bytes [E, H, I_L / 2] (uint8) of the parts of E experts."""
    e_l, hidden, half = np.shape(down)
    i_l = split["local_intermediate"]
    words = _words(_checked(down, np.uint8, (e_l, hidden, i_l // 2), "w2")).transpose(0, 2, 1)
    if split["layout_form"] == "served":
        words = words.reshape(e_l, i_l // SERVED_MX_TILE, HIDDEN_ROWS, hidden).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(words).astype(np.uint16, copy=False)


def repack_down_scales(down: np.ndarray, split: Mapping) -> np.ndarray:
    """Return ``expert_down_weights_scale`` (uint8) of the w2 E8M0 scales [E, H, I_L / 32] (uint8)."""
    e_l, hidden, _ = np.shape(down)
    i_l = split["local_intermediate"]
    scales = _checked(down, np.uint8, (e_l, hidden, i_l // MX_GROUP), "w2 scale").transpose(0, 2, 1)
    if split["layout_form"] == "served":
        scales = scales.reshape(e_l, i_l // SERVED_MX_TILE, HIDDEN_SCALE_ROWS, hidden).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(scales)


def repack_part_experts(parts: Sequence[Mapping], split: Mapping, hidden: int = LATENT_SIZE) -> dict:
    """Return the four expert tensors (``STACK_LAYOUTS``) of the checkpoint parts of E_L experts
    (``checkpoint_expert_part``, in local expert order). With the ``served`` form these are the bytes of the served
    loader's repack of the part."""
    _fail(
        len(parts) == split["local_experts"],
        f"the split takes {split['local_experts']} experts a rank, got {len(parts)}",
    )
    shapes = _part_shapes(split, hidden)

    def stacked(key: str) -> np.ndarray:
        return np.stack([_checked(part[key], np.uint8, shapes[key], key) for part in parts])

    tensors = {
        "expert_gate_up_weights": repack_gate_up_weights(
            stacked("w1.weight_packed"), stacked("w3.weight_packed"), split
        ),
        "expert_gate_up_weights_scale": repack_gate_up_scales(
            stacked("w1.weight_scale"), stacked("w3.weight_scale"), split
        ),
        "expert_down_weights": repack_down_weights(stacked("w2.weight_packed"), split),
        "expert_down_weights_scale": repack_down_scales(stacked("w2.weight_scale"), split),
    }
    want = rank_expert_shapes(split, hidden)
    wrong = {name: tensor.shape for name, tensor in tensors.items() if tuple(tensor.shape) != want[name]}
    _fail(not wrong, f"the re-shard gave the shapes {wrong}, and the layout has {want}")
    return tensors


def unpack_part_experts(tensors: Mapping[str, np.ndarray], split: Mapping, hidden: int = LATENT_SIZE) -> list:
    """Return the checkpoint parts (``checkpoint_expert_part``) of one rank's expert tensors: the inverse of
    ``repack_part_experts``."""
    shapes, dtypes = rank_expert_shapes(split, hidden), rank_expert_dtypes(split)
    arrays = {name: _checked(tensors[name], dtypes[name], shapes[name], name) for name in shapes}
    e_l, i_l, n_h = split["local_experts"], split["local_intermediate"], hidden // HIDDEN_TILE
    projection, columns = gate_up_columns(split)
    words = np.zeros((e_l, HIDDEN_ROWS, n_h, 2, i_l), dtype=np.uint16)
    scales = np.zeros((e_l, HIDDEN_SCALE_ROWS, n_h, 2, i_l), dtype=np.uint8)
    if split["layout_form"] == "served":
        words[..., columns] = arrays["expert_gate_up_weights"].transpose(0, 1, 3, 2, 4)
        scales[..., columns] = arrays["expert_gate_up_weights_scale"].transpose(0, 1, 3, 2, 4)
        down = arrays["expert_down_weights"].transpose(0, 2, 1, 3).reshape(e_l, i_l // FP4_PER_WORD, hidden)
        down_scales = arrays["expert_down_weights_scale"].transpose(0, 2, 1, 3).reshape(e_l, i_l // MX_GROUP, hidden)
    else:
        words[:, :, :, projection, columns] = arrays["expert_gate_up_weights"]
        scales[:, :, :, projection, columns] = arrays["expert_gate_up_weights_scale"]
        down, down_scales = arrays["expert_down_weights"], arrays["expert_down_weights_scale"]
    # [E, P, N_H, GU, I] -> [E, GU, I, N_H, P] -> [E, GU, I, H / 4].
    words = words.transpose(0, 3, 4, 2, 1).reshape(e_l, 2, i_l, hidden // FP4_PER_WORD)
    scales = scales.transpose(0, 3, 4, 2, 1).reshape(e_l, 2, i_l, hidden // MX_GROUP)
    gate_up = _bytes(words)
    down_bytes = _bytes(np.ascontiguousarray(down.transpose(0, 2, 1)))
    down_scales = down_scales.transpose(0, 2, 1)
    return [
        {
            "w1.weight_packed": np.ascontiguousarray(gate_up[e, 0]),
            "w1.weight_scale": np.ascontiguousarray(scales[e, 0]),
            "w3.weight_packed": np.ascontiguousarray(gate_up[e, 1]),
            "w3.weight_scale": np.ascontiguousarray(scales[e, 1]),
            "w2.weight_packed": np.ascontiguousarray(down_bytes[e]),
            "w2.weight_scale": np.ascontiguousarray(down_scales[e]),
        }
        for e in range(e_l)
    ]


def rank_expert_tensors(
    checkpoint: Callable[[int], Mapping[str, np.ndarray]], split: Mapping, rank: int, hidden: int = LATENT_SIZE
) -> dict:
    """Return one rank's four expert tensors of one layer from the checkpoint (EP-major: ``checkpoint(e)`` gives the
    ``CHECKPOINT_KEYS`` arrays of expert e): the experts of its group, its intermediate part (``rank_group_part``)."""
    group, part = rank_group_part(split, rank)
    parts = [checkpoint_expert_part(checkpoint(e), split, part) for e in group_experts(split, group)]
    return repack_part_experts(parts, split, hidden)


def checkpoint_part_of_rank(tensors: Mapping, split: Mapping, rank: int, hidden: int = LATENT_SIZE) -> dict:
    """Return {expert id: checkpoint part} of one rank's expert tensors (the inverse of ``rank_expert_tensors``)."""
    group, _ = rank_group_part(split, rank)
    return dict(zip(group_experts(split, group), unpack_part_experts(tensors, split, hidden), strict=True))


def checkpoint_of_tp_set(rank_tensors: Mapping, split: Mapping, group: int, hidden: int = LATENT_SIZE) -> dict:
    """Return {expert id: checkpoint arrays} of one EP group from the tensors of its TP set (rank -> tensors): the
    parts of each expert joined in part order."""
    ranks = tp_set_ranks(split, group)
    missing = [rank for rank in ranks if rank not in rank_tensors]
    _fail(not missing, f"the TP set {ranks} of group {group} has no tensors for the ranks {missing}")
    parts = [checkpoint_part_of_rank(rank_tensors[rank], split, rank, hidden) for rank in ranks]
    return {
        expert: {
            key: np.concatenate(
                [by_expert[expert][key] for by_expert in parts], axis=0 if key[:2] in ("w1", "w3") else 1
            )
            for key in CHECKPOINT_KEYS
        }
        for expert in group_experts(split, group)
    }


# ---- Dequantize (numpy: the reference of the tests and the twins) ----

_FP4_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32
)
_E8M0_NAN = 255


def _fp4_bytes(packed: np.ndarray) -> np.ndarray:
    """Return float32 [..., 2 n] of FP4 bytes [..., n] (low nibble first)."""
    low, high = _FP4_VALUES[packed & 0xF], _FP4_VALUES[packed >> 4]
    return np.stack([low, high], axis=-1).reshape(*packed.shape[:-1], -1)


def _e8m0(scales: np.ndarray) -> np.ndarray:
    _fail(not bool((scales == _E8M0_NAN).any()), "an MX scale holds the E8M0 NaN code 255")
    return np.ldexp(np.float32(1.0), scales.astype(np.int32) - 127).astype(np.float32)


def dequantize_checkpoint_expert(expert: Mapping[str, np.ndarray]) -> dict:
    """Return float32 ``gate`` and ``up`` [I, H] and ``down`` [H, I] of one checkpoint expert (or part), exactly (each
    value is an FP4 value times a power of 2)."""
    out = {}
    for label, key in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
        values = _fp4_bytes(np.asarray(expert[f"{key}.weight_packed"]))
        out[label] = values * np.repeat(_e8m0(np.asarray(expert[f"{key}.weight_scale"])), MX_GROUP, axis=-1)
    return out


def dequantize_rank_experts(tensors: Mapping, split: Mapping, hidden: int = LATENT_SIZE) -> dict:
    """Return float32 ``gate`` / ``up`` [E_L, I_L, H] and ``down`` [E_L, H, I_L] of one rank's expert tensors, in
    checkpoint orientation and intermediate order (``unpack_part_experts``, then ``dequantize_checkpoint_expert``)."""
    parts = [dequantize_checkpoint_expert(part) for part in unpack_part_experts(tensors, split, hidden)]
    return {key: np.stack([part[key] for part in parts]) for key in ("gate", "up", "down")}
