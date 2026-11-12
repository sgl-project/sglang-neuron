# SPDX-License-Identifier: Apache-2.0
"""Tell GLM-5.2's two Quark exports apart, so that every role serves the MXFP4 one.

The MXFP4 export (``amd/GLM-5.2-MXFP4``) is a Quark export too: bf16 attention and dense MLP, MXFP4 experts. The
Quark AttnFP8 export (``amd/GLM-5.2-Quark-MXFP4-AttnFP8``) stores the attention projections and the indexer's ``wq_b``
and ``wk`` as ``F8_E4M3`` with one ``F32`` scale for each output row, and the dense MLP of layers 0 to 2 and the MTP
layer's experts as MXFP4. No GLM-5.2 role serves it: the prefill's loaders and the decode's weight stores take the
attention and dense weights as the MXFP4 export's bf16 tensors, and the prefill recipe's MXFP8 ``o_proj``
(``GLM_GK_PRJ_OMX=1``) packs its words and scales from the ``o_proj.weight`` rows directly, past any per-row FP8
scale. So a launch of either role refuses the export (`checkpoint_format`, models/glm52/launch.py), and so do the
weight-store builds and loads.

A store records the format of its checkpoint (`FORMAT_KEY`), and a load refuses a store of the other format
(`check_format`).

Example:
    >>> checkpoint_format("/path/to/ckpt")  # doctest: +SKIP
    'mxfp4'
"""

from __future__ import annotations

import json
import os

#: The format of the Quark AttnFP8 export, which no role serves.
QUARK_ATTN_FP8 = "quark-attnfp8"
#: The format of the MXFP4 checkpoint: bf16 attention and dense MLP, MXFP4 experts.
MXFP4 = "mxfp4"
#: The key that records the checkpoint format in a store sentinel. Every store build
#: writes it; a sentinel without it reads as `MXFP4`, the only format a store held
#: before the key was written.
FORMAT_KEY = "checkpoint_format"


def checkpoint_format(ckpt_dir: str, names=None) -> str:
    """Return `QUARK_ATTN_FP8` or `MXFP4` for a checkpoint directory.

    The MXFP4 export is also a Quark export, so ``quant_method`` does not tell the
    two apart. The AttnFP8 export has an FP8 entry for ``*self_attn*`` in
    ``layer_quant_config``. A launcher's view of the checkpoint (the tensors symlinked
    under a config.json of its own) can drop ``quantization_config``, so the tensor
    names decide too (`format_from_names`).

    Args:
        ckpt_dir: The checkpoint directory.
        names: The tensor names. None reads ``model.safetensors.index.json``.

    Example:
        >>> checkpoint_format("/no/such/dir")
        'mxfp4'
    """
    path = os.path.join(ckpt_dir, "config.json")
    if os.path.isfile(path):
        with open(path) as f:
            quant = json.load(f).get("quantization_config") or {}
        for pattern, spec in (quant.get("layer_quant_config") or {}).items():
            weight = (spec or {}).get("weight") or {}
            if "self_attn" in pattern and str(weight.get("dtype", "")).startswith(
                "fp8"
            ):
                return QUARK_ATTN_FP8
    if names is None:
        index = os.path.join(ckpt_dir, "model.safetensors.index.json")
        if not os.path.isfile(index):
            return MXFP4
        with open(index) as f:
            names = json.load(f).get("weight_map", {})
    return format_from_names(names)


def format_from_names(names) -> str:
    """Return the checkpoint format from its tensor names, for a checkpoint with no local ``config.json``."""
    for name in names:
        if name.endswith("self_attn.q_a_proj.weight_scale"):
            return QUARK_ATTN_FP8
    return MXFP4


def store_format(meta: dict) -> str:
    """Return the checkpoint format that a store sentinel records."""
    return str(meta.get(FORMAT_KEY, MXFP4))


def check_format(found: str, expected: str, what: str) -> None:
    """Refuse a store or a file set whose checkpoint format is not the expected one.

    Raises:
        ValueError: If the formats differ.

    Example:
        >>> check_format(MXFP4, MXFP4, "the store")
    """
    if found != expected:
        raise ValueError(
            f"{what} holds the weights of a {found} checkpoint, and this run uses a "
            f"{expected} checkpoint. Build a store from this checkpoint in its own directory."
        )


def refuse_attn_fp8(ckpt_dir: str, what: str) -> None:
    """Refuse the Quark AttnFP8 export for ``what`` (a weight-store build or load), naming the export to serve.

    Raises:
        ValueError: If ``ckpt_dir`` holds the Quark AttnFP8 export.

    Example:
        >>> refuse_attn_fp8("/no/such/dir", "the GK store build")
    """
    if checkpoint_format(ckpt_dir) == QUARK_ATTN_FP8:
        raise ValueError(
            f"{what}: {ckpt_dir} is the Quark AttnFP8 export, whose FP8 attention and MXFP4 dense weights no "
            "GLM-5.2 role reads; build the store from the MXFP4 export."
        )
