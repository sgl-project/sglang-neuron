# SPDX-License-Identifier: Apache-2.0
"""Make the stages of a pipelined PD decode hold and release a failed room's KV pages at the same point of their
admission sequence.

All stages must keep the same free KV pages: a stage that runs short retracts on its own
(``Scheduler.update_running_batch``) and launches a batch the others do not, which the stage seam refuses. Upstream
decides per stage whether to hold a failed room's pages, in ``pop_transferred_with_outcomes``, from rank-local state
(was the destination published yet, ``publish_pending_metadata``; did the ABORT go out), and
sglang_patches/pd_decode_kv_quarantine releases a hold once the room's prefill ranks ack, which each stage learns at its
own time. Either would leave the stages with different free pages.

On a pipelined decode:

1. Every room the release consensus fails is held on every stage: ``pop_transferred_with_outcomes`` is rebuilt with
   the hold decision always true (``SUBSTITUTIONS``). A hold no prefill rank can write into (a destination never
   published, or an ABORT that needs no ack) is releasable at once.
2. Each stage's quarantine agrees with its attention-TP group which holds may go, but releases none
   (pd_decode_kv_quarantine, ``agreed_holds``).
3. Agreed holds ride the release consensus as extra keys (``held_key``: a tuple, which no SGLang rid is, of the hold's
   rid and bootstrap room). ``_pp_pd_get_decode_transferred_ids`` is upstream's intersection along the stages with
   those keys added, so a hold stays in only if every stage agreed.
4. ``process_decode_transfer_queue`` (wrapped) takes the keys out, releases those holds, runs upstream's on the rest,
   and passes the released keys on, so the next stage releases the same holds at the same point. A stage keeps sending
   its agreement until a consensus naming the hold reaches it, so a later consensus can name a hold already released;
   every stage skips it alike.

So the holds match: the consensus fails a room on every stage (a later stage one iteration after the first), and
``pop_transferred_with_outcomes`` is the only place a hold is made. Client aborts land alike too: an abort reaches a
stage only with the requests its previous stage hands on (only stage 0 hears from the tokenizer), at the start of an
iteration and before that iteration's consensus, in upstream's loop as in sglang_patches/pp_decode_overlap_loop
(distributed.pipeline refuses a receive interval above 1). So each stage takes the abort either before the release
naming the room, and its own poll fails the room, or after it, and the abort finishes the admitted request.

Upstream: generic for PP with PD. Upstream's own deferred release frees a hold at a deadline that differs per stage.

Targets, md5-pinned against upstream SGLang 43881a7: ``DecodeTransferQueue.pop_transferred_with_outcomes`` (rebuilt
and replaced), ``SchedulerPPMixin._pp_pd_get_decode_transferred_ids`` (replaced) and
``SchedulerPPMixin.process_decode_transfer_queue`` (wrapped), and the functions of ``_TARGETS_MD5`` they rely on."""

from __future__ import annotations

from typing import Dict, Optional

_PP = "sglang.srt.managers.scheduler_pp_mixin"
_DECODE = "sglang.srt.disaggregation.decode"
POP = f"{_DECODE}.DecodeTransferQueue.pop_transferred_with_outcomes"
TRANSFERRED_IDS = f"{_PP}.SchedulerPPMixin._pp_pd_get_decode_transferred_ids"
PROCESS = f"{_PP}.SchedulerPPMixin.process_decode_transfer_queue"
# (module, qualified name): md5 of the source; the rebuilt, replaced and wrapped ones, then those they rely on
_TARGETS_MD5 = {
    (_DECODE, "DecodeTransferQueue.pop_transferred_with_outcomes"): "eefc7913311dd1a015b87b20faaf3de6",
    (_PP, "SchedulerPPMixin._pp_pd_get_decode_transferred_ids"): "492efd2231275e3582b3377d5cd24db9",
    (_PP, "SchedulerPPMixin.process_decode_transfer_queue"): "e99cc696dba776b5fe5143bc9dedd4a4",
    # the one place a hold is made, and the publish whose rank-local readiness the hold no longer depends on
    (_DECODE, "DecodeTransferQueue._defer_release"): "60935590b6a464dec14c18537964ef16",
    (_DECODE, "DecodeTransferQueue.publish_pending_metadata"): "5655a8445e3667da28b4b5a528f8818f",
    # each stage's (succeeded, failed) polls over its group, which the intersection takes
    (_PP, "SchedulerPPMixin.get_rids"): "4e70b2f59a306baa8f96b1a2af918aa7",
    # the last stage sends the consensus to the first, and every stage hands its outcomes on, unchanged
    (_PP, "SchedulerPPMixin._pp_pd_send_consensus_release_ids"): "d14e06df2f544720282bd7280956f0f4",
    # retracts on the stage's own free pages
    ("sglang.srt.managers.scheduler", "Scheduler.update_running_batch"): "0112f5abf0ea516834bd424c362c8af4",
}
# the upstream text that decides a failed room's hold, and what it becomes (in the dedented source)
SUBSTITUTIONS = (
    ("""                    requires_host_drain
                    or (deferrable and decode_req.kv_receiver.abort_notified),
""", """                    True,  # sglang_neuron.sglang_patches.pp_decode_held_release: held on every stage
"""),
)
# the tag of a consensus key that names a hold every stage agreed releasable; the key is a tuple, so no rid is one
_HELD = "neuron-held"


def held_key(hold) -> tuple:
    """The consensus key of a hold (pd_decode_kv_quarantine.hold_key: its request's rid and bootstrap room)."""
    return (_HELD, *hold)


def split_held(consensus: Optional[Dict]):
    """(the consensus without the holds' keys, the holds named); (None, []) for no consensus."""
    if consensus is None:
        return None, []
    holds = [key[1:] for key in consensus if isinstance(key, tuple) and key[:1] == (_HELD,)]
    if not holds:
        return consensus, []
    return {rid: failed for rid, failed in consensus.items() if not isinstance(rid, tuple)}, holds


def _pp_pd_get_decode_transferred_ids(self) -> Dict[str, bool]:
    """REPLACE SchedulerPPMixin._pp_pd_get_decode_transferred_ids: upstream's {rid: failed} over the transfers every
    stage so far concluded (a failure on any stage wins), plus a key per hold this stage's group agreed releasable."""
    from sglang.srt.disaggregation.base.conn import KVPoll

    from sglang_neuron.sglang_patches import pd_decode_kv_quarantine

    succeeded, failed = self.get_rids(self.disagg_decode_transfer_queue.queue, False, [KVPoll.Success],
                                      [KVPoll.Failed])
    current = dict.fromkeys(succeeded, False)
    current.update(dict.fromkeys(failed, True))
    current.update(dict.fromkeys(map(held_key, pd_decode_kv_quarantine.agreed_holds(self.disagg_decode_transfer_queue)),
                                 False))
    if self.pp_group.is_first_rank:
        return current
    previous = self._pp_recv_pyobj_from_prev_stage()
    return {rid: previous[rid] or rid_failed for rid, rid_failed in current.items() if rid in previous}


def process_decode_transfer_queue(original_fn, self, release_rids):
    """AROUND SchedulerPPMixin.process_decode_transfer_queue: release the holds every stage agreed, run upstream's on
    the rest of the consensus, and pass the released holds' keys on with its outcomes."""
    from sglang_neuron.sglang_patches import pd_decode_kv_quarantine

    release_rids, holds = split_held(release_rids)
    released = pd_decode_kv_quarantine.release_agreed(self.disagg_decode_transfer_queue, holds) if holds else []
    outcomes = original_fn(self, release_rids)
    if released:
        # a consensus with holds is never None, so upstream's outcomes are its pop's own fresh dict
        outcomes.update(dict.fromkeys(map(held_key, released), False))
    return outcomes


def build_pop():
    """Upstream's pop_transferred_with_outcomes, holding every failed room."""
    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(
        source_guard.resolve(_DECODE, "DecodeTransferQueue.pop_transferred_with_outcomes"), SUBSTITUTIONS)


def register(registry, around, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(POP, build_pop(), replace)
    registry.register(TRANSFERRED_IDS, _pp_pd_get_decode_transferred_ids, replace)
    registry.register(PROCESS, process_decode_transfer_queue, around)
