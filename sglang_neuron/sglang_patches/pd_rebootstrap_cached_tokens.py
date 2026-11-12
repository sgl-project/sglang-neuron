# SPDX-License-Identifier: Apache-2.0
"""Keep a request's first-attempt prefix-cache counts across a PD decode rebootstrap.

A retract pause on a PD decode (``Scheduler.pause_generation``, ``mode="retract"``) retracts the running requests and
the transferred ones waiting for a slot, and on continue re-adds each as a rebootstrap
(``DecodeRequest.is_rebootstrap``): its prefill recomputes the prompt plus every output but the last, which is the
decode's next input.

SGLang counts cached tokens once (``Req.reset_for_retract`` keeps ``cached_tokens`` and sets ``retracted_stain``, and
``prepare_for_prebuilt`` adds nothing to a stained request), but the PD commit
(``DecodeTransferQueue._commit_transfer_to_req``) sets ``cached_tokens``, ``already_computed`` and the device, host
and storage parts from the prefill's row on every commit. A rebootstrap would then report the recompute's hit in
``usage.prompt_tokens_details.cached_tokens``, which can exceed the prompt (the prefill's tree holds an earlier
attempt's outputs) or be 0 after a ``/flush_cache``.

The wrapper restores those five fields after a rebootstrap's commit: the cached-token counts stay the first attempt's
(a rebootstrap has always committed once; upstream's commit asserts an output), and ``already_computed`` stays at the
0 the retraction set, which ``prepare_for_prebuilt`` ignores for a stained request. The rest of the commit is
upstream's.

Upstream: a generic accounting fix in the PD commit, with no Neuron dependency.

Targets, md5-pinned against SGLang 43881a7: ``DecodeTransferQueue._commit_transfer_to_req`` (wrapped), and the two the
kept counts rely on, ``Req.reset_for_retract`` and ``ScheduleBatchDisaggregationDecodeMixin.prepare_for_prebuilt``."""

from __future__ import annotations

_DECODE = "sglang.srt.disaggregation.decode"
TARGET = (_DECODE, "DecodeTransferQueue._commit_transfer_to_req")
_TARGETS_MD5 = {
    # sets the five fields from the row on every commit, a rebootstrap's included
    TARGET: "59ff80b249c217d16e8aa06f33331a7e",
    # a retraction keeps the counts and stains the request
    ("sglang.srt.managers.schedule_batch", "Req.reset_for_retract"): "2d5b8589e89922d7d8f4ca3b9726b96c",
    # adds nothing to a stained request's counts
    ("sglang.srt.disaggregation.decode_schedule_batch_mixin",
     "ScheduleBatchDisaggregationDecodeMixin.prepare_for_prebuilt"): "54234b5570697fc3f922520b7a422882",
}
# the request fields the commit sets from the row's prefix-cache counts
ACCOUNTING = ("cached_tokens", "already_computed", "cached_tokens_device", "cached_tokens_host",
              "cached_tokens_storage")


def commit_transfer_to_req(original_fn, self, decode_req, *args, **kwargs):
    """AROUND DecodeTransferQueue._commit_transfer_to_req: a rebootstrap keeps its first attempt's counts."""
    if not decode_req.is_rebootstrap:
        return original_fn(self, decode_req, *args, **kwargs)
    req = decode_req.req
    kept = [getattr(req, name) for name in ACCOUNTING]
    result = original_fn(self, decode_req, *args, **kwargs)
    for name, value in zip(ACCOUNTING, kept):
        setattr(req, name, value)
    return result


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(".".join(TARGET), commit_transfer_to_req, around)
