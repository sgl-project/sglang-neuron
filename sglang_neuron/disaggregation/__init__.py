# SPDX-License-Identifier: Apache-2.0
"""Disaggregated serving over NIXL: the launch settings of a prefill or decode server and the hooks it runs
(docs/design/disaggregated-inference.md). The platform applies the settings in the launching process, and the
plugin registers the hooks in every process."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Microseconds a NIXL progress thread pauses between iterations. NIXL's Python wrapper creates its agents with no
# pause, so each rank's progress thread would spin on a host core next to the threads that drive the device.
NIXL_PROGRESS_DELAY_US = 250
# The giga-kernel decode's: its 128 ranks already keep every host core busy, and a longer pause delays a request's KV
# arrival by at most that much, which is small against its time to first token.
GIGA_DECODE_NIXL_PROGRESS_DELAY_US = 2000
# Bootstrap-queue requests whose prompts a prefill server re-matches in its radix tree at each waiting-prefix refresh
# (SGLang's SGLANG_WAITING_PREFIX_REFRESH_BOOTSTRAP_MAX, default 512; see apply_launch_settings).
PREFILL_BOOTSTRAP_PREFIX_REFRESH_MAX = 0
WAITING_PREFIX_REFRESH_BOOTSTRAP_MAX_ENV = "SGLANG_WAITING_PREFIX_REFRESH_BOOTSTRAP_MAX"


def apply_launch_settings(server_args) -> None:
    """Fix the settings every scheduler of a prefill or decode server reads, before the schedulers start. A launch
    that sets another value is refused, except for a prefill's bootstrap-queue prefix refresh cap
    (SGLANG_WAITING_PREFIX_REFRESH_BOOTSTRAP_MAX), which a launch may state."""
    from sglang_neuron import serving_policy, sglang_tree
    from sglang_neuron.serving_policy import set_launch_setting

    role = str(getattr(server_args, "disaggregation_mode", "null") or "null")
    if role not in ("prefill", "decode"):
        return
    if server_args.nnodes > 1 and not serving_policy.current().decode_layout:
        # the poll reduce and the decode's rank exchange run through the node's shared memory, which spans nodes only
        # where the model declares a decode layout over them (serving_policy.declared_decode_layout: each attention-TP
        # group inside one node, the DP rendezvous joined by the node leaders)
        raise ValueError(f"--nnodes {server_args.nnodes}: a prefill or decode server runs on one node, except a "
                         "model's declared decode layout; run more servers instead")
    backend = server_args.disaggregation_transfer_backend
    if backend == "fake" and role == "decode" and serving_policy.current().decode_layout:
        # a decode-only run of a model's declared decode layout, with no prefill: SGLang's fake transfer completes each
        # request's receive at once, over the KV pages the decode holds, with a synthetic first token (a benchmark of
        # the decode step; docs/model-recipes/kimi-k3.md, --disaggregation-transfer-backend fake)
        _decode_retraction_backup_none(server_args)
        return
    # SGLang's fake backend moves no KV: a decode measured alone, its requests decoding over the KV its pool holds, or a
    # prefill without a decode peer (sglang_patches.pd_fake_prefill)
    if backend != "nixl" and backend != "fake":
        raise ValueError(f"--disaggregation-transfer-backend {backend}: Neuron moves the KV with NIXL only "
                         "(--disaggregation-transfer-backend nixl); SGLang's fake backend serves a decode alone and a "
                         "prefill without a decode peer")
    # also on a fake-backend role: a pipelined prefill's stage boundaries move over NIXL (distributed.pp_seam)
    set_launch_setting("SGLANG_DISAGGREGATION_NIXL_BACKEND", "LIBFABRIC",
                       "the KV moves over EFA, and a failed transfer reads settled only once LIBFABRIC reports it")
    set_launch_setting("NIXL_LIBFABRIC_EFA_UNSOLICITED_WRITE_RECV", "1",
                       "a decode rank that takes many KV writes at once runs out of posted receives without it")
    set_launch_setting("SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE", "1",
                       "a failed transfer's WRITEs may still land in its decode pages, and a prefill without it acks "
                       "no ABORT")
    if role == "prefill":
        # parse the pipelined prefill's switches now, so a bad value stops the launch before a scheduler reads it
        from sglang_neuron.sglang_patches import pp_prefill_early_answer, pp_prefill_inflight_wait, pp_prefill_last_send

        # (the readers keep their first value: parse uncached here)
        pp_prefill_last_send.enabled.__wrapped__()
        pp_prefill_early_answer.enabled.__wrapped__()
        pp_prefill_inflight_wait.wait_passes()
        set_launch_setting("SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX", "0",
                           "the early send records a CUDA event")
        # SGLang periodically re-matches the prompts of bootstrap-queue requests in the radix tree to keep their
        # prefixes in LRU. A PD prefill's bootstrap queue holds every request waiting for a decode seat, so on long
        # prompts that refresh takes seconds of Python per stage; the waiting queue, which runs next, is still
        # refreshed. A launch may state another value.
        os.environ.setdefault(WAITING_PREFIX_REFRESH_BOOTSTRAP_MAX_ENV, str(PREFILL_BOOTSTRAP_PREFIX_REFRESH_MAX))
        policy = serving_policy.current()
        if (policy.async_execution and policy.pp_size == 1 and sglang_tree.attention_dp_size(server_args) == 1
                and server_args.attn_cp_size == 1):
            # the overlap loop takes requests while the oldest forward's result is pending, which SGLang supports only
            # without attention DP or CP; PP stages take requests at one loop point instead (distributed.pipeline)
            set_launch_setting("SGLANG_ENABLE_DISAGG_PREFILL_CONTINUOUS_INPUT_POLLING", "1",
                               "an asynchronous prefill takes requests while its forwards run")
    else:
        _decode_retraction_backup_none(server_args)


def _decode_retraction_backup_none(server_args) -> None:
    # the Neuron KV pool keeps no host copy: SGLang's "none" aborts a retracted request for its client to retry
    backup = server_args.disaggregation_decode_retraction_backup
    if backup not in (None, "none"):
        raise ValueError(f"--disaggregation-decode-retraction-backup {backup}: the Neuron KV pool keeps no host copy "
                         "of a request's KV, so a retracted request is aborted for its client to retry; launch with "
                         "none or without it")
    server_args.disaggregation_decode_retraction_backup = "none"


def register_hooks(registry, hook_type) -> None:
    """Register the hooks of a prefill or decode server's scheduler processes, by the role the launching process
    resolved (serving_policy). The launching process itself runs none of the code they wrap."""
    from sglang_neuron import serving_policy

    policy = serving_policy.current()
    if policy.role == "aggregated":
        return
    from sglang_neuron.disaggregation import poll_reduce
    from sglang_neuron.sglang_patches import (decode_radix_prefill_chunks, nixl_await_poll, nixl_decode_listener,
                                              nixl_head_major, nixl_peer_refusal, nixl_settle_then_ack,
                                              pd_decode_kv_quarantine, pd_kv_layer_ids, pd_status_lock,
                                              prefill_dispatch_dp_rank)

    around, replace = hook_type.AROUND, hook_type.REPLACE
    nixl_peer_refusal.register(registry, around, replace)
    pd_status_lock.register(registry, around, replace)
    nixl_settle_then_ack.register(registry, around, replace)
    if policy.role == "prefill":
        # the sender's transfer worker, which waits for its WRITEs
        nixl_await_poll.register(registry, replace)
    nixl_decode_listener.register(registry, replace)
    pd_decode_kv_quarantine.register(registry, replace)
    if policy.role == "decode":
        # a room whose metadata row does not land after every prefill status fails, naming the cause (a prefill whose
        # stages before the last send their last chunks with no row, sglang_patches/pp_prefill_last_send)
        from sglang_neuron.sglang_patches import pd_metadata_wait

        pd_metadata_wait.register(registry, around)
    prefill_dispatch_dp_rank.register(registry, replace)
    nixl_head_major.register(registry, around, replace)
    pd_kv_layer_ids.register(registry, around)
    # the recurrent-state PD chain (disaggregation/state_transfer): a prefill refuses, at registration, a decode that
    # cannot take its state components
    from sglang_neuron.sglang_patches import nixl_state_components, pd_kv_cp_layout

    nixl_state_components.register(registry, around)
    # a decode announces its token context-parallel KV layout, and a prefill refuses a decode whose layout is not its
    # own
    pd_kv_cp_layout.register(registry, around)
    from sglang_neuron.models import catalog

    spec = catalog.served_spec()
    if policy.role == "prefill" and spec is not None and spec.pd_prefill_reposts_failed_writes:
        # the first writes to a decode restarted under this prefill fail at the EFA device; a batch's failed writes
        # are posted again before its room fails (a model whose catalog entry asks for it)
        from sglang_neuron.sglang_patches import nixl_repost_failed_writes

        nixl_repost_failed_writes.register(registry, around)
    # DCP: every prefill can send a DCP decode whole pages, and a DCP prefill its stripes; a DCP decode announces its
    # page interleave and pairs on it
    from sglang_neuron.sglang_patches import dcp_page_transfer

    if policy.role == "prefill":
        dcp_page_transfer.register_prefill(registry, around,
                                           stripes=spec is not None and spec.prefill_context_parallel)
    elif policy.dcp_size > 1:
        dcp_page_transfer.register_decode(registry, around, replace)
    # the Qwen3.5 kv_cp decode's hand-off (its decode announces the layout in pd_kv_cp_layout's frame): a whole-page
    # prefill scatters each page's block to the decode rank that attends over it; the prefill-side hooks gate on the
    # peer's layout, the decode-side one on this process's declared layout
    from sglang_neuron.sglang_patches import kv_cp_scatter

    if policy.role == "prefill":
        kv_cp_scatter.register_prefill(registry, around)
    elif policy.role == "decode":
        kv_cp_scatter.register_decode(registry, around)
    # PP: every prefill serves a pipelined decode's stages or refuses another layer partition; a pipelined decode maps
    # its stages behind an unpipelined prefill
    from sglang_neuron.sglang_patches import pp_decode_stage_transfer

    if policy.role == "prefill":
        pp_decode_stage_transfer.register_prefill(registry, around)
    elif policy.pp_size > 1:
        pp_decode_stage_transfer.register_decode(registry, around)
    if policy.role == "prefill" and policy.prefill_compute_ahead:
        # a pipelined prefill computes a request before its decode grants the seat and pushes the KV once the decode
        # publishes, its stages deciding every such request at the bootstrap queue's consensus point (on by default on a
        # pipelined prefill of a model that declares it, serving_policy.prefill_compute_ahead)
        from sglang_neuron.sglang_patches import pp_prefill_compute_ahead

        pp_prefill_compute_ahead.register(registry, around)
    # a model whose decode KV lives in its kernel's own caches hands it off through its own hooks; the prefill side goes
    # after every other hook on the peer registration and the send, so its peer check runs first and its send outermost
    from sglang_neuron.sglang_patches import kernel_kv_handoff

    handoff = kernel_kv_handoff.model_handoff()
    if handoff is not None:
        if policy.role == "prefill":
            kernel_kv_handoff.register_prefill(registry, around, handoff)
        else:
            kernel_kv_handoff.register_decode(registry, around, handoff)
    poll_reduce.register_hooks(registry, around)
    # every prefill advertises its chunk grid and every decode reads it; the chunk-keyed decode pairs on it and keys
    # each request's KV under its prefill's fingerprint
    if policy.role == "prefill":
        decode_radix_prefill_chunks.register_prefill(registry, around)
    else:
        decode_radix_prefill_chunks.register_decode(registry, replace)
    if policy.decode_radix_prefill_chunks:
        decode_radix_prefill_chunks.register_chunk_keyed_decode(registry, around)
    if policy.role == "decode" and (policy.pp_size > 1 or policy.decode_radix_prefill_chunks):
        # a first-contact refusal of one prefill — a pipelined decode's stage pairing
        # (pp_decode_stage_transfer.resolve_rank_mapping) or the chunk-keyed decode's pairing check — raises
        # PrefillIncompatibleError, and this fails just that prefill's queued requests
        from sglang_neuron.sglang_patches import decode_incompatible_prefill

        decode_incompatible_prefill.register(registry, around)
    if policy.role == "decode":
        from sglang_neuron.sglang_patches import decode_state_drain, pd_rebootstrap_cached_tokens

        pd_rebootstrap_cached_tokens.register(registry, around)
        # the decode request table's reset drains a model's recurrent-state pool first
        decode_state_drain.register(registry, around)
        if policy.decode_layout:
            # the pools of a declared decode layout announce their drafter-row form and state page rule, outside the
            # DCP registration registered above
            from sglang_neuron.sglang_patches import decode_pool_frames

            decode_pool_frames.register(registry, around)
    if policy.decode_result_lag:
        from sglang_neuron.sglang_patches import pd_decode_result_lag

        pd_decode_result_lag.register(registry, around, replace)
    if policy.dp_identity:
        from sglang_neuron.managers import dp_identity

        dp_identity.register_hooks(registry, around)
    # a fake-transfer decode (a decode-only benchmark) builds no NIXL agent; a fake-transfer prefill still moves its PP
    # stage boundaries over NIXL (distributed.pp_seam)
    if not (policy.fake_kv_transfer and policy.role == "decode"):
        _set_nixl_progress_delay(GIGA_DECODE_NIXL_PROGRESS_DELAY_US if policy.giga_decode else NIXL_PROGRESS_DELAY_US)


def _set_nixl_progress_delay(delay_us: int) -> None:
    import importlib

    import torch

    # NIXL's meta package imports the variant of the CUDA major the installed torch was built for
    bind = importlib.import_module(f"nixl_cu{torch.version.cuda.split('.')[0]}._api").nixlBind
    if getattr(bind, "_neuron_progress_delay", False):
        return
    agent = bind.nixlAgent

    def agent_with_delay(name, config):
        config.pthrDelay = delay_us
        return agent(name, config)

    bind.nixlAgent = agent_with_delay
    bind._neuron_progress_delay = True
