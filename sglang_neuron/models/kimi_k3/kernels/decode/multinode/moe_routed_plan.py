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

"""Kimi-K3 multi-server routed experts: the static per-layer EP x TP plan of the decode and verify entries.

A plan (the entry argument ``routed_layers`` = ``(default_tag, ((layer, tag), ...))``,
``moe_routed_layout.ROUTED_LAYERS_ARGUMENT``) fixes the expert split of each MoE layer when the model loads;
nothing switches at run time. Every layer of the default tag reads the base expert stacks (``expert_gate_up_weights``
.. ``expert_down_weights_scale``), one entry for each such layer of the call, in layer order. The layers of each other
tag (at most ``MAX_ALT_TAGS``, in order of their first layer) read the alternate stacks
``routed_alt{slot}_expert_*`` (``expert_stack_name``). ``moe_routed_layout.check_layer_combos`` checks the plan
(the tags of the world, the MoE layers, the alternate count) and gives each layer its tag and position.

The entries hand the decode block one ``LayerStack`` for each expert tensor (MoE layer k of the call: the view of its
tag's stack), and one MoE body (``plan_body``) that runs each layer with the body of its tag. The collective tensors and
the collectives of a MoE layer depend only on the world and the tokens, so every tag takes the same ones.
"""

import re

import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from ..decode_block import NUM_LAYERS, is_dense_layer
from .moe_routed_layout import (
    EXPERT_TENSORS,
    ROUTED_ALT_ARGUMENTS,
    check_layer_combos,
    rank_expert_shapes,
)
from .tkg_moe_2srv import ROUTED_TILE_FORM, moe_2srv_world_split, routed_layers_options
from .tkg_moe_routed_dynamic import csr_gather_of_layers, is_packed_split

ALT_STACK_ARGUMENTS = ROUTED_ALT_ARGUMENTS
"""The entry arguments of the alternate expert stacks (``moe_routed_layout.ROUTED_ALT_ARGUMENTS``): slot 0
then slot 1, each in the order of ``EXPERT_TENSORS``."""
_LAYER_TAG = re.compile(r"L(\d+)(?:_|$)")
"""The layer tag of ``decode_block.decode_block_body`` (``L{layer}``, with a ``_c{chunk}`` suffix in the verify)."""


def moe_layers_of(first_layer: int, num_layers: int) -> tuple:
    """Return the MoE layers of a call of ``num_layers`` layers from ``first_layer``: every layer but the dense one.

    Example:
        >>> moe_layers_of(0, 4)
        (1, 2, 3)
    """
    return tuple(layer for layer in range(first_layer, first_layer + num_layers) if not is_dense_layer(layer))


def call_plan(world_size: int, routed_layers, first_layer: int, num_layers: int) -> dict | None:
    """Return the plan of the call's MoE layers, or None when no layer of the call takes a tag other than the default
    (no plan, or a plan of one tag: the entry's single split).

    The plan of the model (``check_layer_combos`` over every MoE layer of the model, ``NUM_LAYERS``) checks the tags of
    the world and the alternate count, and gives the alternate slots: the k-th other tag of the model in order of its
    first layer takes slot k (``stack_names``), in every call. The call's plan keeps the call's MoE layers: ``tags``
    (layer -> tag), ``splits`` (the default tag and each tag of the call), ``stacks`` (tag -> the call's layers of that
    tag), ``index`` (layer -> (tag, position among the call's layers of that tag): the entry stacks of a call hold its
    own layers) and ``stack_names`` (the model's).

    Raises:
        ValueError: a tag is unknown for the world, a layer of the plan is not a MoE layer of the model, or the model
            takes more than ``MAX_ALT_TAGS`` other tags.
    """
    if routed_layers is None:
        return None
    default_tag, layer_tags = routed_layers
    model = check_layer_combos(world_size, moe_layers_of(0, NUM_LAYERS), default_tag, dict(layer_tags))
    layers = moe_layers_of(first_layer, num_layers)
    tags = {layer: model["tags"][layer] for layer in layers}
    if all(tag == default_tag for tag in tags.values()):
        return None
    stacks = {tag: tuple(layer for layer in layers if tags[layer] == tag) for tag in {default_tag, *tags.values()}}
    return {
        "tags": tags,
        "splits": {tag: model["splits"][tag] for tag in stacks},
        "stacks": stacks,
        "index": {layer: (tag, stacks[tag].index(layer)) for layer, tag in tags.items()},
        "stack_names": model["stack_names"],
    }


def call_csr_gather(
    world_size: int, routed_layers, first_layer: int, num_layers: int, routed_tp: int = None, mx_tile: int = None
) -> str:
    """Return the CSR gathers of the call's dynamic packed MoE layers (``moe_2srv_body`` ``routed_dynamic_csr``):
    ``csr_gather_of_layers`` of their count, so the "staged" form (1 DGE table slot a layer) stays within the table
    of the call's program. Each MoE layer of the call takes the split of its tag in ``routed_layers``, or the call's
    split (``routed_tp``, ``mx_tile``) without a plan; a split of sub-512 MX tiles whose dynamic form packs experts
    (``is_packed_split``) builds the CSR arrays.

    Raises:
        ValueError: as ``call_plan``, for a plan that does not fit the world or the model.
    """
    layers = moe_layers_of(first_layer, num_layers)

    def packed(split: dict) -> bool:
        return split.get("layout_form") == ROUTED_TILE_FORM and is_packed_split(split)

    if routed_layers is None:
        count = len(layers) if packed(moe_2srv_world_split(world_size, routed_tp, mx_tile)) else 0
    else:
        default_tag, layer_tags = routed_layers
        model = check_layer_combos(world_size, moe_layers_of(0, NUM_LAYERS), default_tag, dict(layer_tags))
        count = sum(1 for layer in layers if packed(model["splits"][model["tags"][layer]]))
    return csr_gather_of_layers(count)


def check_plan_cut(routed_layers, layer_cut: bool):
    """Refuse a plan whose MoE layers take other tags in a call of an inline layer cut (the split verify: each layer
    range of the cut would need its own stacks, which the loader does not build). A plan of one tag passes."""
    if routed_layers is None or not layer_cut:
        return
    default_tag, layer_tags = routed_layers
    others = sorted({tag for _, tag in layer_tags} - {default_tag})
    kernel_assert(
        not others,
        f"routed_layers gives some MoE layers other tags {others}, and this verify call is one layer range of an inline "
        "layer cut (attn_res_bank_in / return_attn_res_bank), whose ranges would each need their own stacks. Give every "
        "MoE layer the default tag, or remove the verify layer cut.",
    )


def plan_stack_tensors(plan: dict, default_tag: str, base: tuple, alternates: dict) -> dict:
    """Return tag -> its four expert stacks, after the shape check: the base stacks for the default tag, the alternate
    stacks of its slot (``plan["stack_names"]``) for each other tag. Each stack holds one entry for each of the call's
    layers of its tag (``plan["stacks"]``), in the layout of its split (``rank_expert_shapes``). A tag with no layer in
    the call is not checked."""
    stacks = {}
    for tag, layers in plan["stacks"].items():
        if not layers:
            continue
        if tag == default_tag:
            tensors = tuple(base)
        else:
            names = [plan["stack_names"][tag][name] for name in EXPERT_TENSORS]
            missing = [name for name in names if alternates.get(name) is None]
            kernel_assert(not missing, f"the routed plan's tag {tag!r} needs the entry stacks {missing}")
            tensors = tuple(alternates[name] for name in names)
        shapes = rank_expert_shapes(plan["splits"][tag])
        for name, tensor in zip(EXPERT_TENSORS, tensors, strict=True):
            want = (len(layers), *shapes[name])
            kernel_assert(
                tuple(tensor.shape) == want,
                f"routed plan tag {tag!r}: {name} must be {want} (its {len(layers)} layers of the call), got "
                f"{tuple(tensor.shape)}",
            )
        stacks[tag] = tensors
    return stacks


class LayerStack:
    """One expert tensor of a plan as a per-MoE-layer stack: entry k is MoE layer k of the call, the view of that layer
    in its tag's stack. The decode block reads it as a stacked tensor: ``shape[0]``, ``[k]``, ``[a:b]``. ``_storage``
    is the base stack's (``plan_layer_stacks``); ``_gate_storages`` holds the stack of every entry, so the weight gates of
    the decode block (``ffn_weights_after_attn``) take the loads of every tag's layers, as with one stack."""

    def __init__(self, entries: tuple, base: nl.NkiTensor):
        self._entries = tuple(entries)
        self._base = base
        self._storage = getattr(base, "_storage", None)
        # Every HBM stack of the entries (the base stack and the alternates), for the weight gates of the decode block
        # (``tkg_weight_dma_gate.gate_all_dmas`` ``weights``: ``ffn_weights_after_attn``).
        found = (getattr(entry, "_storage", None) for entry in self._entries)
        self._gate_storages = tuple({id(storage): storage for storage in found if storage is not None}.values())
        self.shape = (len(self._entries), *tuple(base.shape[1:]))
        self.dtype = base.dtype

    def __getitem__(self, key):
        if isinstance(key, slice):
            return LayerStack(self._entries[key], self._base)
        return self._entries[key]

    def __len__(self) -> int:
        return len(self._entries)


def plan_layer_stacks(plan: dict, layers: tuple, stacks: dict, default_tag: str) -> tuple:
    """Return the four ``LayerStack`` of a plan for the call's MoE layers ``layers``, in the order of ``EXPERT_TENSORS``.
    Their base is the default tag's stack (the first layer's tag's when the call has no default-tag layer)."""
    base_tag = default_tag if default_tag in stacks else plan["index"][layers[0]][0]
    out = []
    for index in range(len(EXPERT_TENSORS)):
        entries = []
        for layer in layers:
            tag, position = plan["index"][layer]
            entries.append(stacks[tag][index][position])
        out.append(LayerStack(entries, stacks[base_tag][index]))
    return tuple(out)


SELECTORS_NAME = "k3moe2s_sel"
"""The name prefix of the once-built selectors of the default tag (``tkg_moe_2srv.moe_2srv_selectors``, the served
name)."""


def selectors_name(tag: str, default_tag: str) -> str:
    """Return the name prefix of the once-built selectors of a plan's tag: the served ``SELECTORS_NAME`` for the default
    tag, ``SELECTORS_NAME`` and the tag (``-`` as ``_``) for another tag, so each tag's instructions take their own
    names.

    Example:
        >>> selectors_name("ep64xtp4-t256", "ep128xtp2-t512"), selectors_name("ep128xtp2-t512", "ep128xtp2-t512")
        ('k3moe2s_sel_ep64xtp4_t256', 'k3moe2s_sel')
    """
    return SELECTORS_NAME if tag == default_tag else f"{SELECTORS_NAME}_{tag.replace('-', '_')}"


def plan_selector_splits(world_size: int, routed_tp: int, mx_tile: int, plan: dict | None, default_tag: str) -> list:
    """Return ``(name, routed_tp, mx_tile)`` of the once-built selectors of a call (the entries' ``selectors_once``):
    the default split with the served name, then each other tag of the call's plan with its own name."""
    splits = [(SELECTORS_NAME, routed_tp, mx_tile)]
    if plan is not None:
        splits += [
            (selectors_name(tag, default_tag), *plan_split_options(world_size, tag))
            for tag in sorted(plan["splits"])
            if tag != default_tag
        ]
    return splits


def layer_of_tag(tag: str) -> int:
    """Return the model layer of a decode block layer tag (``L{layer}``, ``L{layer}_c{chunk}``)."""
    match = _LAYER_TAG.match(tag)
    kernel_assert(
        match is not None, f"the routed plan body takes the layer tags L<layer> of the decode block, got {tag!r}"
    )
    return int(match.group(1))


def plan_body(bodies: dict, plan: dict, default_tag: str):
    """Return the MoE body of a plan: each layer runs with the body of its tag (``bodies``: tag -> the MoE body of that
    split, ``tkg_moe_2srv.moe_2srv_body``). The body takes the attributes of the default tag's body and builds the
    selectors of every tag (``build_selectors``). Its ``prefetch`` takes the MoE layer (``layer``): a layer of a
    split of whole 512 tiles starts the expert ring of its tag's body (``expert_prefetch_stream``, the served
    layout); a layer of a sub-512-tile split (``ROUTED_TILE_FORM``) returns None, and its routed-tile loop loads its
    own weights."""
    kernel_assert(default_tag in bodies, f"the routed plan has no body for its default tag {default_tag!r}")
    missing = sorted(set(plan["tags"].values()) - set(bodies))
    kernel_assert(not missing, f"the routed plan has no body for the tags {missing}")

    def body(x_hT, *args, **options):
        layer = layer_of_tag(args[-1])
        kernel_assert(layer in plan["tags"], f"layer {layer} is not a MoE layer of the routed plan")
        return bodies[plan["tags"][layer]](x_hT, *args, **options)

    for key, value in vars(bodies[default_tag]).items():
        setattr(body, key, value)

    def build_selectors(rank_info, rows: int, sb=None):
        return [
            bodies[tag].build_selectors(
                rank_info, rows, sb=sb, **({} if tag == default_tag else {"name": selectors_name(tag, default_tag)})
            )
            for tag in sorted(set(plan["tags"].values()))
        ]

    def prefetch(*weights, layer: int = None, **options):
        kernel_assert(
            layer in plan["tags"],
            f"the expert ring of a routed plan of several tags (moe_expert_prefetch) needs the MoE layer of the call, "
            f"got layer {layer!r}",
        )
        tag = plan["tags"][layer]
        if plan["splits"][tag].get("layout_form") == ROUTED_TILE_FORM:
            return None
        return bodies[tag].prefetch(*weights, **options)

    body.build_selectors = build_selectors
    # The decode block passes the layer to a prefetch that takes it (through the verify's feed-forward wrapper too).
    prefetch.takes_layer = True
    body.prefetch = prefetch
    body.moe_plan = plan
    return body


def check_alternates(alternates: dict, plan: dict | None):
    """Refuse alternate expert stacks that the plan's slots do not name: every given stack must be a stack of a tag
    with layers in the call (``plan["stack_names"]``), and with no plan of several tags none may be given. The plan's
    stacks that are not given fail in ``plan_stack_tensors``."""
    kernel_assert(set(alternates) == set(ALT_STACK_ARGUMENTS), f"the alternate stacks are {ALT_STACK_ARGUMENTS}")
    given = {name for name, tensor in alternates.items() if tensor is not None}
    used = set()
    if plan is not None:
        for tag, layers in plan["stacks"].items():
            if layers:
                used |= {plan["stack_names"][tag][name] for name in EXPERT_TENSORS}
    extra = sorted(given - used)
    kernel_assert(
        not extra,
        f"the alternate expert stacks {extra} are not stacks of the routed_layers plan's tags with layers in the call",
    )


def plan_split_options(world_size: int, tag: str) -> tuple:
    """Return the ``(routed_tp, mx_tile)`` of a tag (``tkg_moe_2srv.routed_layers_options`` of a one-tag plan)."""
    return routed_layers_options(world_size, (tag, ()))


def plan_bodies(plan: dict, default_tag: str, default_body, make_body):
    """Return ``plan_body`` over the default tag's body and one body for each other tag of the call
    (``make_body(routed_tp, mx_tile)``, the entry's MoE body options with that tag's split)."""
    world_size = plan["splits"][default_tag]["world"]
    bodies = {default_tag: default_body}
    for tag in sorted(set(plan["tags"].values()) - {default_tag}):
        bodies[tag] = make_body(*plan_split_options(world_size, tag))
    return plan_body(bodies, plan, default_tag)


def plan_weights(plan: dict, default_tag: str, moe_weights: tuple, alternates: dict, layers: tuple) -> tuple:
    """Return the 7 MoE weight slots of the decode block with the four expert slots (3 .. 6) as ``LayerStack`` of the
    base stacks and the alternate stacks (``plan_stack_tensors``)."""
    stacks = plan_stack_tensors(plan, default_tag, moe_weights[3:7], alternates)
    return (*moe_weights[:3], *plan_layer_stacks(plan, layers, stacks, default_tag))


def plan_moe(
    plan: dict, default_tag: str, default_body, make_body, moe_weights: tuple, alternates: dict, layers: tuple
):
    """Return the MoE body (``plan_bodies``) and the MoE weight slots (``plan_weights``) of a call with a plan."""
    return plan_bodies(plan, default_tag, default_body, make_body), plan_weights(
        plan, default_tag, moe_weights, alternates, layers
    )
