# SPDX-License-Identifier: Apache-2.0
"""Match exact request IDs and the numbered copies generated from batched IDs."""

from __future__ import annotations

import re

_IO = "sglang.srt.managers.io_struct"
_TARGETS_MD5 = {
    (_IO, "AbortReq.matches"): "9c6b6361fc69f4fe44a239d5a7b39860",
    (_IO, "AbortReq.__post_init__"): "cd4bcfc9e04f7c9c807f4dde41969104",
    (_IO, "GenerateReqInput._normalize_rid"): "7ec689589057e8b5b4cc320cf1a35664",
}
_EXPANSION = re.compile(r"_[0-9]+")


def reaches(abort, rid: str) -> bool:
    """Whether ``abort`` (an AbortReq) reaches the request ``rid``. Under abort_all it reaches every request. Otherwise
    it reaches the abort's own id and ``<id>_<digits>``, the copies GenerateReqInput._normalize_rid expands that id
    into."""
    if abort.abort_all:
        return True
    return rid == abort.rid or (rid.startswith(abort.rid) and _EXPANSION.fullmatch(rid, len(abort.rid)) is not None)


def register(registry, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_IO}.AbortReq.matches", reaches, replace)
