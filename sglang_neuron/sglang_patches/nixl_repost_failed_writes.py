# SPDX-License-Identifier: Apache-2.0
"""A PD prefill posts a batch's failed NIXL writes again before it fails the room.

When a decode server restarts under a running prefill, the prefill's first RDMA writes to some of the restarted decode
ranks fail at the EFA device: their work completions come back with EFA status 1 (FLUSHED, libfabric's "Operation
canceled", prov_errno 1), the prefill node's ``rdma_write_wr_err`` counters rise with no retransmit timeout or
unresponsive-remote event, and the decode logs nothing. Later writes to the same ranks pass: on Kimi-K3 (prefill on one
trn3 node, decode on two) the probes after a decode-only restart passed 2, 7, 15 and then 16 of 16 requests. The same
signature, with the same libfabric (2.6.0amzn1.0) and EFA installer (1.50.0), is ai-dynamo/nixl#2505. Without this
module each such write fails its room, and the client sees HTTP 500 until enough writes have gone through.

Upstream's transfer worker waits for every handle of a batch to settle (``NixlKVManager._await_handles``) and fails the
room if one failed. It cannot tell a failed handle from a running one: NIXL's Python binding raises
``nixlBackendError('NIXL_ERR_BACKEND')`` from ``check_xfer_state`` once a transfer has completed with an error, which
upstream counts as a state it cannot read, so the batch ends unsettled at once ("left a handle running"). The
LIBFABRIC backend reports a transfer's error only once every one of its requests has completed (``checkXfer``, with
patches/nixl 0007), so that exception is a settled failure. Here, when upstream's wait ends with a failure:

1. the batch is polled again, reading ``NIXL_ERR_BACKEND`` as a settled ERR, for up to ``SETTLE_TIMEOUT_S``;
2. once every handle settled, the ERR handles are posted again, up to ``MAX_REPOSTS`` rounds, each settled the same
   way; a batch whose handles all end DONE passes.

That is safe because a re-post only happens once no write of the batch is still posted, the decode learns the room's
outcome only after the batch, NIXL reposts a settled handle with fresh request counts and error status (the LIBFABRIC
backend's ``postXfer``), and SGLang's KV, state and aux transfers carry no NIXL notification: a re-post writes the same
bytes into the same pages before anyone reads them. A batch that failed before its barrier (``failure_seen``) keeps
upstream's path; so does one with a handle still running after ``SETTLE_TIMEOUT_S``, or with a state that reads as any
other error. A re-post that raises ends the re-posting. Every re-post round is logged.

Targets are md5-pinned against upstream SGLang 43881a7: ``NixlKVManager._await_handles``."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_NIXL = "sglang.srt.disaggregation.nixl.conn"
TARGET = (_NIXL, "NixlKVManager._await_handles")
_TARGETS_MD5 = {TARGET: "316621564891c5ed504bfd66b423d702"}
# rounds of re-posts of a batch's ERR handles: after a decode restart the probes needed up to three requests per decode
# rank before every write passed
MAX_REPOSTS = 4
# how long a failed batch's other handles may still run before the batch counts as unsettled (upstream's own window
# is 5 s)
SETTLE_TIMEOUT_S = 10.0
SETTLE_POLL_S = 0.001
# between rounds, so a device that just failed a write is not handed the same one in the same microsecond
REPOST_BACKOFF_S = 0.01
# the exception text of a transfer that completed with a backend error (a settled ERR)
_SETTLED_ERROR = "NIXL_ERR_BACKEND"


class _Unreadable(Exception):
    pass


def _state(agent, handle) -> str:
    try:
        return agent.check_xfer_state(handle)
    except Exception as error:  # noqa: BLE001 - classified below
        if _SETTLED_ERROR in str(error):
            return "ERR"
        raise _Unreadable(f"{type(error).__name__}: {error}") from error


def _settle(agent, handles):
    """Each handle's state once none runs, or None if one still runs after SETTLE_TIMEOUT_S."""
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while True:
        states = [_state(agent, h) for h in handles]
        if all(s in ("DONE", "ERR") for s in states):
            return states
        if time.monotonic() >= deadline:
            return None
        time.sleep(SETTLE_POLL_S)


def await_with_reposts(original_fn, self, handles, *args, failure_seen: bool = False, **kwargs):
    """AROUND NixlKVManager._await_handles: (settled, any_failed), after up to MAX_REPOSTS rounds of re-posting the
    ERR handles of a batch that settled with a failure."""
    settled, failed = original_fn(self, handles, *args, failure_seen=failure_seen, **kwargs)
    if failure_seen or not failed:
        return settled, failed
    agent = self.agent
    try:
        states = _settle(agent, handles)
        for attempt in range(1, MAX_REPOSTS + 1):
            if states is None:
                logger.warning("NIXL re-post: a handle of the failed batch of %d still runs after %.0fs",
                               len(handles), SETTLE_TIMEOUT_S)
                return False, True
            errored = [h for h, s in zip(handles, states) if s == "ERR"]
            if not errored:
                return True, False
            time.sleep(REPOST_BACKOFF_S)
            for handle in errored:
                agent.transfer(handle)
            states = _settle(agent, handles)
            left = 0 if states is None else sum(s == "ERR" for s in states)
            logger.warning("NIXL re-post round %d of %d: %d of %d handles of the batch had failed; %s", attempt,
                           MAX_REPOSTS, len(errored), len(handles),
                           "the batch passed" if states is not None and not left else
                           f"{left} failed again" if states is not None else "a handle did not settle")
        if states is None:
            return False, True
        return True, any(s == "ERR" for s in states)
    except _Unreadable as error:
        logger.warning("NIXL re-post: a transfer state of the failed batch cannot be read (%s); failing the batch",
                       error)
        return False, True
    except Exception as error:  # noqa: BLE001 - a re-post that raises: upstream's failure path settles the batch
        logger.warning("NIXL re-post: posting a failed handle again raised %s: %s; failing the batch",
                       type(error).__name__, error)
        settled, _ = original_fn(self, handles, *args, failure_seen=True, **kwargs)
        return settled, True


def register(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(".".join(TARGET), await_with_reposts, around)
