# SPDX-License-Identifier: Apache-2.0
"""Upstream SGLang's PD prefill PP loop in the overlap order, where the serving policy runs it.

Upstream's ``event_loop_pp_disagg_prefill`` sends a micro-batch's outputs around its launch. With the async last stage
(sglang_patches/pp_last_stage_async) that send first reads the sampled tokens back, so without an async batch depth the
last stage's host waits on a forward with nothing queued behind it, and the first stage waits for those outputs before
handing on its next requests: every stage idles a host round trip per micro-batch.

The overlap order removes that wait. The loop is upstream's, rebuilt from source (``SUBSTITUTIONS``), on every
pipelined PD prefill (distributed.pipeline serves one only without an async batch depth). With N = pp_size and
L = N + 1, iteration i serves micro-batch i % N, and every stage:

1. receives requests, decides the bootstrap queue and polls its transferred requests, as upstream does;
2. applies the release consensus of iteration i - L before scheduling (rank 0 gets it from the last rank, the others
   from their previous stage);
3. schedules and launches; a non-last stage hands its proxy on at once and commits the previous proxy send only after
   the launch;
4. on the last rank, sends rank 0 the consensus of iteration i, then the outputs of iteration i - 1 with iteration i's
   forward already queued; each send is committed L sends of its kind later, once rank 0 has taken it;
5. processes iteration i - L (the last rank from the outputs it sent, the others from the ones they received);
6. on a non-last stage, hands on this iteration's requests, transferred ids and applied consensus.

No host waits on a forward with nothing queued behind it, and every stage applies the same consensus and processes the
same iterations at the same points, so all stages schedule the same batches. A launched batch leaves its ``mbs`` slot
before it is processed, so it lives in ``pp_overlap_records`` (counted by sglang_patches/pp_loop_state's idle and drain
checks) and ``_collect_inflight_batches`` adds it, so an abort still finds its requests. The cost: a request's
last-chunk KV send and first token come one iteration later than in upstream's loop at depth 1.

The departures from upstream are safe because of three Neuron runtime properties:

* The proxy is fresh host descriptor tensors built at every launch (the boundary itself travels on the PP seam), so
  the previous proxy send may complete after the next launch. Upstream commits it first because a CUDA graph's proxy
  outputs are views of buffers the next replay overwrites.
* There is no PP comm stream (``_TARGETS_MD5`` pins the fences that would use one) and the outputs are host tensors,
  so processing needs no receive event or readback, and the last rank processes the outputs it sent instead of getting
  them back round the ring (pp_loop_state.process_step).
* Control and output messages travel on gloo pairs, whose sends complete when the receiver takes them; the last rank's
  commit order (``commit_later``) relies on that.

Upstream: the order is generic but relies on these three properties; an upstream version would keep upstream's commit
order and fences where they do not hold, as on CUDA.

Targets, md5-pinned against upstream SGLang 43881a7: ``SchedulerPPMixin.event_loop_pp_disagg_prefill`` (rebuilt and
replaced), ``Scheduler._collect_inflight_batches`` (wrapped), and the functions of ``_TARGETS_MD5`` the loop relies
on."""

from __future__ import annotations

_MIXIN = "sglang.srt.managers.scheduler_pp_mixin"
_SCHEDULER = "sglang.srt.managers.scheduler"
_PREFILL = "sglang.srt.disaggregation.prefill"
LOOP = f"{_MIXIN}.SchedulerPPMixin.event_loop_pp_disagg_prefill"
# the rebuilt loop's decorator line, which build() applies itself
DECORATOR = "@DynamicGradMode()\n"
INFLIGHT = f"{_SCHEDULER}.Scheduler._collect_inflight_batches"
# (module, qualified name): md5 of the source; the rebuilt and wrapped ones, then those the loop relies on
_TARGETS_MD5 = {
    (_MIXIN, "SchedulerPPMixin.event_loop_pp_disagg_prefill"): "8bbb0377fb4214f8ff289436470f2e7e",
    (_SCHEDULER, "Scheduler._collect_inflight_batches"): "15ee339b1f105825c68ca8a069b7ea2b",
    # pp_size + depth micro-batches, the output queue and the send-work lists
    (_MIXIN, "SchedulerPPMixin.init_pp_loop_state"): "2deba5f7b367977316763050845e0288",
    # the last rank queues one (event, outputs) entry per launch
    (_MIXIN, "SchedulerPPMixin._pp_launch_batch"): "59cebf2d14f04cc82dfbced48d974eff",
    # pops the queue head for a launched target and sends it, which pp_last_stage_async fills first
    (_MIXIN, "SchedulerPPMixin._pp_send_output_to_next_stage"): "e2025eddedf940579e994635b5553252",
    (_MIXIN, "SchedulerPPMixin._pp_send_dict_to_next_stage"): "a68ce1f2295efa4afa71e863298d8f9a",
    (_MIXIN, "SchedulerPPMixin._pp_prepare_tensor_dict"): "d29fe72b57e526eda7c570e320d40051",
    (_MIXIN, "SchedulerPPMixin._pp_commit_comm_work"): "935f872964c601a49544236d3ae8604f",
    (_MIXIN, "SchedulerPPMixin._pp_commit_proxy_send_work"): "6ffd2bd3ac8569b6c41a67d961fb56e3",
    # the proxy commit after the launch rests on these: without a PP comm stream the proxy send's forward fence records
    # nothing (pp_send_done_event stays None), and the launch's forward waits on no such fence
    (_MIXIN, "SchedulerPPMixin._pp_record_comm_event"): "687c0d427ad5df0dc6b9a93aa32d9722",
    (_MIXIN, "SchedulerPPMixin._pp_wait_forward_dependencies"): "1bf85b9db960b2c382ab04d477bb424b",
    (_MIXIN, "SchedulerPPMixin._pp_send_proxy_to_next_stage"): "1c70fc00fe2afb40c94f37811433253a",
    (_MIXIN, "SchedulerPPMixin._pp_recv_proxy_tensors"): "6a8973312df0da42999f2585147e86c4",
    # proxy and output dicts on one pair, demultiplexed by their message type
    (_MIXIN, "SchedulerPPMixin._pp_recv_typed_dict"): "ab12f1b37123b479555077f511092aa2",
    (_MIXIN, "SchedulerPPMixin._pp_recv_dict_from_prev_stage"): "f0958889a27f62eff9f862738f7eb961",
    (_MIXIN, "SchedulerPPMixin._pp_prep_batch_result"): "6b9e7a29b510f1b3d39861653abaeb7d",
    (_MIXIN, "SchedulerPPMixin._pp_process_batch_result"): "f6dfcb7c1de1de98542455b6ffae581c",
    # leader to leader on the world group, the ring wrapping from the last rank to rank 0
    (_MIXIN, "SchedulerPPMixin._pp_send_pyobj_to_next_stage"): "d79ccfa66d862c8bba0470b4fb4e5fc0",
    (_MIXIN, "SchedulerPPMixin._pp_recv_pyobj_from_prev_stage"): "cbfe424507dbff5123a0c09d0af49fb7",
    # the first stage polls its in-flight queue, every later stage intersects with its previous stage's
    (_MIXIN, "SchedulerPPMixin._pp_pd_get_prefill_transferred_ids"): "bbc1e569cc2f034d95a86b27156acb86",
    # the last rank sends its intersection, every other rank forwards the consensus it holds
    (_MIXIN, "SchedulerPPMixin._pp_pd_send_consensus_release_ids"): "d14e06df2f544720282bd7280956f0f4",
    (_MIXIN, "SchedulerPPMixin.get_rids"): "4e70b2f59a306baa8f96b1a2af918aa7",
    (_MIXIN, "SchedulerPPMixin.process_bootstrapped_queue"): "fe06e4c8ce95584afa4e05d0554236e3",
    (_MIXIN, "_pp_can_skip_output_comm"): "b4c70a62406a65d6e31d326f51184e00",
    # releases only the consensus rids still in flight
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue"):
        "5b49bcb8e522b632073c48f2c075fd7f",
    # without the overlap schedule: no middle-chunk send at processing, the last chunk's send there, aborted results
    # retired
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.process_batch_result_disagg_prefill"):
        "27a83c0ccd496600b2d4eca00577a840",
    # the middle-chunk send at the next scheduling step
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.process_prefill_chunk"): "dca76f30fd359b20c4c3b2e861490e34",
    (_PREFILL, "SchedulerDisaggregationPrefillMixin.send_kv_chunk"): "0e5a574d819e4c08526f8751615bc139",
    # the non-overlap branch passes the proxy and returns at the launch
    (_SCHEDULER, "Scheduler.run_batch"): "afa792c782bfe5e3914d892209b15982",
    (_SCHEDULER, "Scheduler.ingest_requests"): "9519de2263bcfd52e322bce6372c0006",
    (_SCHEDULER, "Scheduler.process_batch_result"): "b6ba1c0a36b142e8331927719bbe41ad",
    # an abort marks the in-flight requests _collect_inflight_batches names
    (_SCHEDULER, "Scheduler.abort_request"): "0fed02d75e27c867928d017e68f7b601",
    (_SCHEDULER, "Scheduler.collect_inflight_reqs"): "9efe4e457552a783858421f1ae9ffeec",
    # the idle, flush, weight-update and health checks (pp_loop_state counts the records)
    (_SCHEDULER, "Scheduler._pp_microbatches_drained"): "57001adf8b45f9b89edbc8cd6571539f",
    (_SCHEDULER, "Scheduler.is_fully_idle"): "28e67960ff8d30595eab2437fe29b45e",
    # the record's snapshot of the batch as launched
    ("sglang.srt.managers.schedule_batch", "ScheduleBatch.copy"): "214b0c5e6aac2bd19ee2940e8ebf9a9b",
    # an empty list is still one message
    ("sglang.srt.utils.common", "point_to_point_pyobj"): "7ea53e9802a0b901ae3caa275a586d75",
}

# (old, new) texts of upstream's loop source (dedented: the loop's body at eight spaces, the micro-batch body at
# twelve), applied in order, each old text once in the evolving source
SUBSTITUTIONS = (
    # upstream's docstring lays out upstream's order; the rebuilt loop's points here
    ("    Prefill Server Schedule:\n"
     "    ====================================================================\n"
     "    Stage P\n"
     "    recv ith req from previous stage\n"
     "    pop bootstrapped reqs (PPConsensusStore + pp_sync polls)\n"
     "    recv ith transferred req from previous stage\n"
     "    recv ith proxy from previous stage\n"
     "    run ith batch\n"
     "    recv prev (i+1) % mb_size th release req from previous stage\n"
     "    local consensus on release req\n"
     "    recv prev (i+1) % mb_size th outputs\n"
     "    process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU "
     "computation)\n"
     "    send ith req to next stage\n"
     "    send ith transferred req to next stage\n"
     "    send ith proxy to next stage\n"
     "    send current stage's outputs to next stage (can be stashed and delayed to send later)\n"
     "\n"
     "    the above order can be optimized and reordered to minimize communication-related CPU stall and overhead "
     "bubbles.\n"
     "    ====================================================================\n",
     "    Prefill Server Schedule: the overlap order of sglang_neuron.sglang_patches.pp_prefill_overlap_loop, "
     "whose module\n"
     "    docstring has its steps.\n"),
    # the loop state: the launched, unprocessed batches and the lag (loop_state refuses what the order cannot run),
    # the last rank's deferred sends to rank 0, and the iteration count
    ("    self.init_pp_loop_state()\n",
     "    self.init_pp_loop_state()\n"
     "    from sglang_neuron.sglang_patches import pp_prefill_overlap_loop as _overlap\n"
     "\n"
     "    records, lag = _overlap.loop_state(self)\n"
     "    consensus_sends, output_sends = deque(), deque()\n"
     "    iteration = 0\n"),
    # the exchange's results, which the overlap order does not have
    ("            next_pp_outputs = None\n"
     "            next_release_rids = None\n"
     "            d2h_event = None\n"
     "            next_batch_result = None\n"
     "\n",
     ""),
    # the release consensus of iteration i - lag, applied on every stage before it schedules: rank 0 takes it from the
    # last rank (sent right after that rank's launch of i - lag), every other rank from its previous stage (sent at
    # the end of that stage's iteration i, after its requests and transferred ids)
    ("            tmbs[mb_id] = transferred_rids\n",
     "            tmbs[mb_id] = transferred_rids\n"
     "            release_rids = None\n"
     "            if iteration >= lag:\n"
     "                release_rids = self._pp_recv_pyobj_from_prev_stage()\n"
     "                self.process_disagg_prefill_inflight_queue(release_rids)\n"),
    # no output exchange and no proxy commit before the launch
    ("            if get_parallel().pp_async_batch_depth > 0:\n"
     "                next_pp_outputs, next_batch_result, d2h_event = (\n"
     "                    self._pp_commit_send_output_work_and_preprocess_output_tensors(\n"
     "                        next_first_rank_mb_id,\n"
     "                        next_mb_id,\n"
     "                    )\n"
     "                )\n"
     "            self._pp_commit_proxy_send_work()\n",
     "            # the overlap order: no output wait and no proxy commit before the launch\n"),
    # after the launch: the proxy on at once; the last rank's consensus of this iteration (at pp_loop_size = N,
    # next_first_rank_mb_id is mb_id, whose transferred ids were just set) and the outputs of iteration i - 1 from
    # their host copy while this launch is queued; the processing of iteration i - lag
    ("            if get_parallel().pp_async_batch_depth == 0:\n"
     "                next_pp_outputs, next_batch_result, d2h_event = (\n"
     "                    self._pp_commit_send_output_work_and_preprocess_output_tensors(\n"
     "                        next_first_rank_mb_id,\n"
     "                        next_mb_id,\n"
     "                    )\n"
     "                )\n"
     "            send_release_work, release_rids = (\n"
     "                self._pp_pd_send_consensus_release_ids(\n"
     "                    tmbs, next_first_rank_mb_id, release_rids, transferred_rids\n"
     "                )\n"
     "            )\n"
     "\n"
     "            if tmbs[next_mb_id] is not None:\n"
     "                next_release_rids = self._pp_recv_pyobj_from_prev_stage()\n"
     "            self._pp_commit_comm_work(send_release_work)\n"
     "            # post-process the coming microbatch\n"
     "            if self.mbs[next_mb_id] is not None:\n"
     "                d2h_event.synchronize()\n"
     "                self._pp_process_batch_result(\n"
     "                    self.mbs[next_mb_id],\n"
     "                    next_batch_result,\n"
     "                )\n"
     "                self.last_mbs[next_mb_id] = self.mbs[next_mb_id]\n"
     "\n"
     "            if tmbs[next_mb_id] is not None:\n"
     "                self.process_disagg_prefill_inflight_queue(next_release_rids)\n",
     "            if cur_batch and not self.pp_group.is_last_rank:\n"
     "                # the previous proxy send is committed after this launch, never before it\n"
     "                self._pp_commit_proxy_send_work()\n"
     "                self._pp_send_proxy_to_next_stage(result)\n"
     "            if self.pp_group.is_last_rank:\n"
     "                consensus_work, _ = self._pp_pd_send_consensus_release_ids(\n"
     "                    tmbs, next_first_rank_mb_id, None, transferred_rids\n"
     "                )\n"
     "                _overlap.commit_later(self, consensus_sends, consensus_work, lag)\n"
     "                _overlap.send_outputs(self, records, output_sends, (mb_id - 1) % self.pp_loop_size, lag)\n"
     "            records.append(_overlap.Record(cur_batch, self.mb_metadata[mb_id]) if cur_batch else None)\n"
     "            if len(records) > lag:\n"
     "                _overlap.process(self, records.popleft())\n"
     "            self.last_mbs[mb_id] = cur_batch\n"),
    # a stage that is not the last forwards the release consensus it applied (None before iteration lag) after its
    # requests and transferred ids; its proxy went on after the launch
    ("                if cur_batch:\n"
     "                    self._pp_send_proxy_to_next_stage(result)\n",
     "                self._pp_commit_comm_work(send_release_work)\n"
     "                send_release_work, release_rids = self._pp_pd_send_consensus_release_ids(\n"
     "                    tmbs, next_first_rank_mb_id, release_rids, transferred_rids\n"
     "                )\n"),
    # nothing carries over from the exchange (with the blank line before it)
    ("\n"
     "            self.pp_outputs = next_pp_outputs\n"
     "            release_rids = next_release_rids\n",
     ""),
    ("            self.running_batch.batch_is_full = False\n",
     "            self.running_batch.batch_is_full = False\n"
     "            iteration += 1\n"),
    # idle only once no launched batch is unprocessed
    ("        if server_is_idle:\n",
     "        if server_is_idle and not any(r is not None for r in records):\n"),
    # the next micro-batch, which only the exchange read
    ("            next_mb_id = (mb_id + 1) % self.pp_loop_size\n", ""),
)


class Record:
    """A launched, unprocessed batch: its snapshot at launch (``ScheduleBatch.copy``, since ``process_prefill_chunk``
    may filter the live batch's requests before it is processed), the live batch (to match the last rank's output
    queue), its metadata and, on the last rank, the outputs it sent."""

    __slots__ = ("batch", "live", "metadata", "outputs")

    def __init__(self, batch, metadata):
        self.batch, self.live, self.metadata, self.outputs = batch.copy(), batch, metadata, None


def loop_state(scheduler):
    """The overlap order's state after init_pp_loop_state: the launched, unprocessed batches and the lag, pp_size + 1.
    Refuses an async batch depth, a loop state without the records, and SGLang's skipped pure-chunked output send
    (rank 0 would wait for outputs the last rank never sends)."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel

    pp_size = get_parallel().pp_size
    if scheduler.pp_loop_size != pp_size:
        raise RuntimeError("the overlap-order PD prefill PP loop runs pp_size micro-batches: no --pp-async-batch-depth")
    records = getattr(scheduler, "pp_overlap_records", None)
    if records is None:
        raise RuntimeError("the overlap-order PD prefill PP loop keeps its launched batches in the PP loop state's "
                           "pp_overlap_records (sglang_patches/pp_loop_state), which this loop state lacks")
    if envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get():
        raise RuntimeError("the overlap-order PD prefill PP loop processes every batch from the last rank's outputs: "
                           "SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM would skip their send")
    return records, pp_size + 1


def commit_later(scheduler, sends, work, lag: int) -> None:
    """Keep the last rank's send ``work`` to rank 0 in flight and commit it ``lag`` sends of its kind later. Rank 0
    takes iteration j's consensus and outputs in its iteration j + lag, before handing on that iteration's requests,
    which the last rank receives before either commit, so the commits never wait. Committing the previous send instead
    would deadlock against rank 0's commit of a request send that waits for this rank's next iteration."""
    sends.append(work)
    while len(sends) > lag:
        scheduler._pp_commit_comm_work(sends.popleft())


def send_outputs(scheduler, records, sends, slot: int, lag: int) -> None:
    """The last rank: send rank 0 the outputs of the batch launched last iteration (micro-batch ``slot``) through
    SGLang's ``_pp_send_output_to_next_stage``, filled from its host copy by pp_last_stage_async, and keep them on the
    record for processing. With SGLANG_NEURON_PP_PREFILL_EARLY_LAST_SEND=1 it first fills the outputs and sends the
    batch's last chunks (sglang_patches/pp_prefill_last_send), so rank 0, which answers a request once it processed
    it (sglang_patches/pp_prefill_early_answer), takes the outputs only after every stage's last-chunk send."""
    from sglang_neuron.sglang_patches import pp_last_stage_async, pp_prefill_last_send

    target = scheduler.mbs[slot]
    if target is None:
        return
    record = records[-1] if records else None
    queue = scheduler.last_rank_comm_queue
    # the queue holds the previous batch's entry, then this iteration's if it launched one
    launched = scheduler.mbs[(slot + 1) % scheduler.pp_loop_size] is not None
    if record is None or record.live is not target or len(queue) != 1 + launched:
        raise RuntimeError("the overlap-order PD prefill PP loop's last rank sends the outputs of the previous "
                           "iteration's batch, and its output queue does not hold that batch's entry first")
    event, outputs = queue[0]
    record.outputs = outputs
    if pp_prefill_last_send.enabled():
        # a last chunk's KV and first token go out now, not L iterations later, and before the outputs: fill them from
        # the host copy here, so the send below does not fill them again
        if isinstance(event, pp_last_stage_async.PendingOutputs):
            event.fill(scheduler, outputs.tensors)
            queue[0] = (None, outputs)
        pp_prefill_last_send.send_early(scheduler, record)
    work = scheduler._pp_send_output_to_next_stage(slot, scheduler.mbs, queue, None)
    commit_later(scheduler, sends, work, lag)


def process(scheduler, record) -> None:
    """Process a launched batch's result (``pp_loop_state.process_step``): the last rank from the outputs it sent,
    every other rank from what its previous stage sends, handed on unless the next stage is the last. The ring carries
    host tensors. Then answer the batch's requests whose last chunk went out at their forward
    (sglang_patches/pp_prefill_early_answer)."""
    if record is None:
        return
    import torch

    from sglang.srt.runtime_context import get_parallel

    from sglang_neuron.sglang_patches import pp_loop_state

    is_last = scheduler.pp_group.is_last_rank
    result = None
    if is_last:
        if record.outputs is None:
            raise RuntimeError("the overlap-order PD prefill PP loop's last rank processes a batch before it has sent "
                               "its outputs")
        result = scheduler._pp_prep_batch_result(record.batch, record.metadata, record.outputs)
    with torch.profiler.record_function("process_batch_result"):
        pp_loop_state.process_step(scheduler, record.batch, result, record.metadata, is_last,
                                            get_parallel().pp_rank < get_parallel().pp_size - 2)
    from sglang_neuron.sglang_patches import pp_prefill_early_answer

    pp_prefill_early_answer.answer(scheduler, record.batch)


def build():
    """Upstream's event_loop_pp_disagg_prefill rebuilt in the overlap order, under upstream's ``@DynamicGradMode()``
    (inference or no-grad mode). Keeps its file, line numbers and qualified name; its docstring's schedule points
    here."""
    import inspect

    from sglang.srt.utils import DynamicGradMode

    from sglang_neuron.sglang_patches import source_guard

    upstream = inspect.unwrap(source_guard.resolve(_MIXIN, "SchedulerPPMixin.event_loop_pp_disagg_prefill"))
    return DynamicGradMode()(source_guard.with_substitutions(upstream, SUBSTITUTIONS, decorators=(DECORATOR,)))


def collect_inflight_batches(original_fn, self):
    """AROUND Scheduler._collect_inflight_batches: SGLang's batches plus the overlap loop's launched, unprocessed
    records, whose micro-batch slot was rescheduled before they were processed."""
    batches = original_fn(self)
    records = getattr(self, "pp_overlap_records", None)
    if not records:
        return batches
    return [*batches, *(record.batch for record in records if record is not None)]


def register(registry, replace, around) -> None:
    """Where the serving policy runs the overlap order (general_plugin): the rebuilt loop in place of upstream's, and
    its records added to the in-flight batches."""
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(LOOP, build(), replace)
    registry.register(INFLIGHT, collect_inflight_batches, around)
