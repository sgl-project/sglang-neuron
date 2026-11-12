# SPDX-License-Identifier: Apache-2.0
"""One gloo transport device per process, shared by every gloo process group in it.

torch's gloo backend (``distributed_c10d._create_gloo_process_group``) builds each ``ProcessGroupGloo`` on a new
default transport device. Each gloo TCP device runs its own event-loop thread, which wakes every 10 ms to check for
shutdown, whether or not its sockets carry anything. A scheduler holds a device group and a CPU group per SGLang group
coordinator, plus the world group and the plugin's own groups. So a decode node with one scheduler per rank runs
thousands of these threads, on CPUs the schedulers already keep busy. Each wakeup takes a turn on a run queue that a
scheduler's main thread also waits on when its step's result is ready.

Here every gloo process group of the process is built on one device, created with the process's first gloo group, so
there is one loop thread (one device per interface where ``GLOO_SOCKET_IFNAME`` lists several, as torch's creator
builds them). The groups stay separate gloo contexts (their own pairs, tags, timeouts and worker threads),
so their collectives and the order of those collectives are unchanged. Only the event loop that polls their sockets is
shared, so one group's large transfer can delay another group's socket handling in the same process. Under
``TORCH_DISTRIBUTED_DEBUG=DETAIL`` torch also builds a wrapper group for each group directly, on a device of its own,
outside this creator.

``install`` re-registers torch's ``gloo`` backend (``torch.distributed.Backend.register_backend``) with ``create``
below, which does what torch's creator does but puts the shared device in the group's options, with one difference:
the group's worker thread count is the options default (2), where torch's keyword constructor sets 2 per device, so
with several ``GLOO_SOCKET_IFNAME`` interfaces a group gets 2 worker threads rather than 2 per interface.

Two checks guard it. torch's creator is pinned by the md5 of its source (torch 2.14.0), so a torch that builds its
groups another way is refused by name. That source calls the keyword constructor, while ``create`` uses the options
object (``_Options``, ``_devices``) and the constructor that takes it, which the md5 does not cover. So ``install``
also builds a one-rank group through ``create``'s own code on a throwaway device, and refuses the same way if torch's
bindings for it changed."""

from __future__ import annotations

import os
import threading

# md5 of the dedented source of torch.distributed.distributed_c10d._create_gloo_process_group in torch 2.14.0
_TORCH_CREATOR_MD5 = "e4b6c82448eb187512694618a4d2d63d"
_lock = threading.Lock()
_state = {"devices": None, "installed": False}


def _after_fork_in_child() -> None:
    # The parent's device and its event-loop thread do not exist in a forked child, and the lock may have been held by
    # a thread the fork did not copy. The child starts over; the parent's device object is left alone rather than
    # destroyed here, since its destructor would join a thread the child does not have.
    global _lock
    _lock = threading.Lock()
    _state["devices"] = None


os.register_at_fork(after_in_child=_after_fork_in_child)


def shared_devices() -> list:
    """The process's gloo transport devices, created at the first call as torch's creator creates a group's: one for
    each interface GLOO_SOCKET_IFNAME lists, else the host's default device."""
    with _lock:
        if _state["devices"] is None:
            from torch.distributed import ProcessGroupGloo

            interfaces = [name for name in os.environ.get("GLOO_SOCKET_IFNAME", "").split(",") if name]
            _state["devices"] = [ProcessGroupGloo.create_device(interface=name) for name in interfaces] or \
                [ProcessGroupGloo.create_default_device()]
        return _state["devices"]


def create(opts, backend_options):
    """torch's gloo creator (``_create_gloo_process_group``) with the group built on the process's shared device."""
    from torch.distributed import distributed_c10d as c10d

    if not c10d.is_gloo_available():
        raise RuntimeError("Distributed package doesn't have Gloo built in")
    return _build(opts, shared_devices())


def _build(opts, devices):
    from torch.distributed import ProcessGroupGloo

    options = ProcessGroupGloo._Options()
    options._devices = list(devices)
    options._timeout = opts.timeout
    options.enable_reconfigure = opts.enable_reconfigure
    group = ProcessGroupGloo(opts.store, opts.group_rank, opts.group_size, options)
    group.options.global_ranks_in_group = opts.global_ranks_in_group
    group.options.group_name = opts.group_id
    return group


def _check_bindings() -> None:
    """Build and drop a one-rank group through ``_build`` on a device of its own, which goes with the group, so the
    bindings ``create`` uses are proven before any group depends on them."""
    from types import SimpleNamespace

    from torch.distributed import HashStore, ProcessGroupGloo
    from torch.distributed import distributed_c10d as c10d

    opts = SimpleNamespace(store=HashStore(), group_rank=0, group_size=1, timeout=c10d.default_pg_timeout,
                           enable_reconfigure=False, global_ranks_in_group=[0], group_id="sgn_gloo_bindings_check")
    try:
        _build(opts, [ProcessGroupGloo.create_default_device()])
    except (AttributeError, TypeError) as error:
        raise RuntimeError(f"torch's ProcessGroupGloo bindings differ from torch 2.14.0's ({type(error).__name__}: "
                           f"{error}): re-validate sglang_neuron/distributed/gloo_device.py against this torch before "
                           "sharing gloo transport devices") from error


def install() -> None:
    """Build every later gloo process group of this process on one device. Idempotent. Raises if torch's gloo creator
    is not the pinned one, if its bindings for the options object changed, or if another component has already
    re-registered the gloo backend."""
    from torch.distributed import distributed_c10d as c10d

    from sglang_neuron.sglang_patches import source_guard

    if _state["installed"]:
        return
    md5 = source_guard.source_md5(c10d._create_gloo_process_group)
    if md5 != _TORCH_CREATOR_MD5:
        raise RuntimeError(f"torch.distributed.distributed_c10d._create_gloo_process_group has md5 {md5}, not "
                           f"{_TORCH_CREATOR_MD5} (torch 2.14.0): re-validate sglang_neuron/distributed/gloo_device.py "
                           "against this torch before sharing gloo transport devices")
    # torch registers its built-in backends lazily, when a group is created. Load gloo's now, so the check below sees
    # torch's creator and later group creation finds this one already registered.
    c10d.Backend._ensure_backend_registered(c10d.Backend.GLOO)
    plugin = c10d.Backend._plugins.get(c10d.Backend.GLOO.upper())
    # after a reload, an earlier copy of this module may have registered its own create; this copy replaces it
    ours = plugin is not None and getattr(plugin.creator_fn, "__module__", None) == __name__ \
        and getattr(plugin.creator_fn, "__qualname__", None) == create.__qualname__
    if not ours and (plugin is None or plugin.creator_fn is not c10d._create_gloo_process_group
                     or not plugin.extended_api):
        raise RuntimeError(f"torch's gloo backend is registered with {plugin!r}, not torch's own creator: another "
                           "component replaced it, and sharing its transport device would override that component")
    _check_bindings()
    c10d.Backend.register_backend(
        c10d.Backend.GLOO,
        create,
        extended_api=True,
        devices=c10d.Backend.backend_capability[c10d.Backend.GLOO],
        _backend_type=c10d.ProcessGroup.BackendType.GLOO,
    )
    _state["installed"] = True
