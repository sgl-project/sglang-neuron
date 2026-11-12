# SPDX-License-Identifier: Apache-2.0
"""The scheduler's request broadcast to its TP group over SGLang's shared-memory message queue, instead of SGLang's gloo
broadcast_pyobj. general_plugin registers it on a disaggregated decode without DP attention. A decode with DP attention
runs the DP identity, whose control fence does the broadcast. The requests and their order are SGLang's. Only the
transport changes."""
import logging

logger = logging.getLogger(__name__)
_reported_groups = set()


def _tp_group():
    """SGLang's TP group, from its group registry: the receiver holds no group."""
    from sglang.srt.distributed.parallel_state import get_tp_group

    return get_tp_group()


def broadcast_reqs_with_message_queue(original_fn, receiver, recv_reqs, local_reqs=None):
    """AROUND SchedulerRequestReceiver._broadcast_reqs_across_ranks. The source rank's local aborts (``local_reqs``)
    go along with its requests, as in SGLang. A group of one rank, or one without a message queue, keeps SGLang's
    broadcast."""
    group = _tp_group()
    if group.world_size == 1 or getattr(group, "mq_broadcaster", None) is None:
        return original_fn(receiver, recv_reqs, *(() if local_reqs is None else (local_reqs,)))
    if group.unique_name not in _reported_groups:
        logger.info("[neuron] request broadcast over the message queue of %s", group.unique_name)
        _reported_groups.add(group.unique_name)
    return group.broadcast_object(_with_local(recv_reqs, local_reqs))


def _with_local(reqs, local_reqs):
    """The source rank's requests followed by its local ones; ``reqs`` itself when it is None or there are no local
    ones."""
    return [*reqs, *local_reqs] if reqs is not None and local_reqs else reqs
