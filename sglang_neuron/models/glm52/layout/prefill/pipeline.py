# SPDX-License-Identifier: Apache-2.0
"""The pipeline-parallel (PP) prefill stages of GLM-5.2: which layers a stage holds, and which top-k picks it hands on.

A PP prefill node runs the decoder layers as stages, stage ``s`` holding the checkpoint layers ``[lo, hi)`` on its own CP
group (``stage_layout`` chooses the split for each served PP size). Inside a stage the layers have local indices
``0 .. hi - lo - 1``; the KV caches keep their global layer ids, so a decode node pairs each of its layers with the stage
that holds it.

GLM-5.2's sparse attention alternates "full" indexer layers, which score every key and pick the top-k, with "shared"
layers, which reuse the picks of the full layer before them. A stage that starts on a shared layer therefore needs the
picks of a layer that lives on the stage before: the stage before hands them over through the PP seam, and the stage
takes them in (``stage_picks``).
"""

from __future__ import annotations

import dataclasses


def stage_config(config, lo: int, hi: int, shared_start: bool = False):
    """The config of the stage that holds the global layers ``[lo, hi)`` of the full model ``config``.

    The per-layer lists keep the types of those layers, ``first_layer_index`` is ``lo``, and the MTP layer keeps its
    checkpoint index. A stage must start on a full indexer layer unless ``shared_start`` says it takes the picks of the
    stage before, which needs a full layer somewhere before it to have picked them.

    Raises:
        ValueError: ``config`` is already a stage, ``[lo, hi)`` is not a non-empty range of its layers, or the stage
            starts on a shared layer it has no picks for.
    """
    if config.first_layer_index:
        raise ValueError(f"the config already holds the stage that starts at layer {config.first_layer_index}; a "
                         "stage is cut from the full model's config")
    if not 0 <= lo < hi <= config.num_hidden_layers:
        raise ValueError(f"the stage's layers [{lo}, {hi}) are not a non-empty range of the model's "
                         f"{config.num_hidden_layers} layers")
    if config.indexer_types[lo] != "full" and not (shared_start and "full" in config.indexer_types[:lo]):
        raise ValueError(f"the stage starts on layer {lo}, a {config.indexer_types[lo]!r} indexer layer, with no full "
                         "layer's picks to take over: a stage must start on a full layer, or after one")
    mtp = config.mtp_layer_index
    return dataclasses.replace(
        config,
        num_hidden_layers=hi - lo,
        first_layer_index=lo,
        indexer_types=list(config.indexer_types[lo:hi]),
        mlp_layer_types=list(config.mlp_layer_types[lo:hi]),
        mtp_layer_index=config.num_hidden_layers if mtp is None else mtp,
    )


def stage_picks(indexer_types: list, lo: int, hi: int) -> tuple[bool, bool]:
    """``(picks_in, picks_out)`` of the stage of global layers ``[lo, hi)``: it takes the picks of the stage before when
    it starts on a shared layer, and hands its own on when the next stage starts on one.

    Example:
        >>> types = ["full"] * 3 + ["shared"] * 3 + ["full"] + ["shared"] * 3
        >>> stage_picks(types, 3, 8), stage_picks(types, 0, 3), stage_picks(types, 6, 10)
        ((True, True), (False, True), (False, False))
    """
    return indexer_types[lo] == "shared", hi < len(indexer_types) and indexer_types[hi] == "shared"
