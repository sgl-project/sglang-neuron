# SPDX-License-Identifier: Apache-2.0
"""Prefill options a stage's model config carries (GlmMoeDsaConfig.options).

The stage graph and its weight tree read them at the served values: the expert-parallel degree, the last stage's MTP
prompt rows, and the formats the stage graph accepts only one value of (fp8 indexer keys, prefill packing, the compact
row). The prefill modules under layout/prefill read none of them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Glm52PrefillOptions:
    """The prefill options of a stage (GlmMoeDsaConfig.options), at the served values.

    A field another value of which the stage graph does not run is refused where it is read: the fp8 keys, packing
    and the compact row by the stage graph."""

    # the MoE's expert-parallel degree: every rank of the stage (EP32 over a PP4 stage's 32 ranks, EP8 at PP16)
    ep_degree: int = 32
    # the MTP layer's prompt rows and their two caches, which only the last stage writes
    glm52_mtp_kv: bool = False
    # decode-format fp8 indexer keys in the pool's key caches, which the decode node reads; the kernel's own key stacks
    # stay bf16
    glm52_idx_fp8_keys: bool = True
    # one sequence per forward: the stage graph runs no packed segment
    glm_prefill_pack: int = 1
    # the 576-byte compact row, which the prefill stack does not write
    glm52_kv_compact: bool = False
