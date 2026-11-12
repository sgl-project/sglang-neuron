# SPDX-License-Identifier: Apache-2.0
"""On a disaggregated prefill of several DP replicas without DP attention, check and register each request under the
replica it was dispatched to, not under its attention DP rank.

Without DP attention every replica's attention DP rank is 0, and upstream's ``CommonKVSender`` reads that rank. So it
fails every request a replica other than 0 serves with "follow_bootstrap_room conflict", and it registers every
replica with the bootstrap server as replica 0. The bootstrap server keys its table by the system DP rank the DP
controller dispatched to.

``CommonKVSender.__init__`` (the check) and ``_register_prefill_dp_rank`` (the registration) are upstream's source with
the attention DP rank replaced by the dispatch rank: the system DP rank when there are several system replicas, else
the attention DP rank.

Upstream: a generic fix for any disaggregated prefill of several DP replicas without DP attention."""

from __future__ import annotations

_COMMON = "sglang.srt.disaggregation.common.conn"
_TARGETS_MD5 = {
    (_COMMON, "CommonKVSender.__init__"): "7dadb846afe3f233cbec620cf1d8eee9",
    (_COMMON, "CommonKVSender._register_prefill_dp_rank"): "e023008cfcb23fcbc2766c56529e8ba0",
}
ATTN_DP_RANK = "self.kv_mgr.attn_dp_rank"
DISPATCH_DP_RANK = "(self.kv_mgr.system_dp_rank if self.kv_mgr.system_dp_size > 1 else self.kv_mgr.attn_dp_rank)"
INIT_SUBSTITUTIONS = (
    # the follow_bootstrap_room check, and its failure message
    (f"{ATTN_DP_RANK}\n            != self.bootstrap_room", f"{DISPATCH_DP_RANK}\n            != self.bootstrap_room"),
    (f'f"{{{ATTN_DP_RANK}}} but bootstrap_room "', f'f"{{{DISPATCH_DP_RANK}}} but bootstrap_room "'),
)
REGISTER_SUBSTITUTIONS = ((f'"dp_rank": {ATTN_DP_RANK},', f'"dp_rank": {DISPATCH_DP_RANK},'),)


def sender_init():
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(source_guard.resolve(_COMMON, "CommonKVSender.__init__"),
                                           INIT_SUBSTITUTIONS)


def register_prefill_dp_rank():
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(source_guard.resolve(_COMMON, "CommonKVSender._register_prefill_dp_rank"),
                                           REGISTER_SUBSTITUTIONS)


def register(registry, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_COMMON}.CommonKVSender.__init__", sender_init(), replace)
    registry.register(f"{_COMMON}.CommonKVSender._register_prefill_dp_rank", register_prefill_dp_rank(), replace)
