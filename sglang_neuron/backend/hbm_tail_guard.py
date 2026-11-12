# SPDX-License-Identifier: Apache-2.0
"""The HBM tail guard (serving_policy.hbm_tail_guard; opt-in, SGLANG_NEURON_HBM_TAIL_GUARD=1): the last MiB of the
runtime's small-allocation arena is held for the life of the process, so no NEFF instruction stream ends there.

The Neuron runtime places small device allocations (up to 512 KiB: per-engine instruction blocks, DMA rings) in an arena
that ends at the top of the logical core's HBM aperture, which is also the engines' program-counter and DMA bound. The
engine sequencers prefetch instructions past the end of an instruction stream; a block that ends within a few KiB of the
bound has its refill DMA run past it, the DMA engine aborts (TX_DATA_AXI_TIMEOUT_ERROR), that engine never reaches its
next collective, and the other ranks report NRT status 1200 ("Stuck in collectives op") at it. Where a block lands
depends only on the arena's fill order, so a launch whose loads fill the arena can hit it on any graph.

install() runs right after the runtime starts (runtime_init.runtime_initialize), before any NEFF loads: it fills the
arena with small allocations, keeps the ones that end within GUARD_BYTES of the highest end it saw (plan_keep), and frees
the rest. Nothing is traced, so no graph or NEFF changes. The fill makes at most MAX_FILL_ALLOCATIONS allocations, a
bounded share of the driver's memory-chunk handle table (DRIVER_HANDLES_PER_DEVICE), which every rank on the device
shares and whose exhaustion takes the device's allocations down ("memchunk handle map out of entries"; no sysfs counter
reports the handles in use). A fill that cannot find the arena's top inside that bound, or a runtime without the
allocation calls, refuses: the launch asked for the guard.
"""
from __future__ import annotations

import ctypes
import logging

logger = logging.getLogger(__name__)

GUARD_BYTES = 1 << 20
FILL_SIZES = (64 << 10, 4 << 10)  # a coarse fill, then the 4 KiB fragments it leaves
# the aperture selector bits of a device physical address: the arena's top is compared without them
ADDR_MASK = (1 << 40) - 1
# the driver's memory-chunk handle table of one device, shared by every rank on it
DRIVER_HANDLES_PER_DEVICE = 512 * 8192
# up to eight ranks (LNC1) share a device's handle table; the fill takes at most 1/64 of it, all freed but the guard's
MAX_FILL_ALLOCATIONS = DRIVER_HANDLES_PER_DEVICE // 64
_NAME = b"sglang_neuron_hbm_tail_guard"
_HELD: list = []  # (handle, bytes) of the guard, never freed


class _AllocationInfo(ctypes.Structure):
    _fields_ = [("physical_address", ctypes.c_uint64), ("size", ctypes.c_size_t), ("hbm_index", ctypes.c_int)]


def plan_keep(allocations, guard_bytes: int = GUARD_BYTES):
    """The indices of ``allocations`` ([(masked physical address, bytes)]) that end within ``guard_bytes`` of the highest
    end among them, and that end; (set(), 0) for none."""
    if not allocations:
        return set(), 0
    top = max(address + size for address, size in allocations)
    return {i for i, (address, size) in enumerate(allocations) if address + size > top - guard_bytes}, top


def _load_libnrt():
    """libnrt with the calls the fill needs; a runtime without them refuses."""
    try:
        lib = ctypes.CDLL("libnrt.so.1")
        lib.nrt_tensor_allocate.restype = ctypes.c_int
        lib.nrt_tensor_allocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_char_p,
                                            ctypes.POINTER(ctypes.c_void_p)]
        lib.nrt_tensor_get_device_allocation_info.restype = ctypes.c_int
        lib.nrt_tensor_get_device_allocation_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AllocationInfo)]
        lib.nrt_tensor_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    except (OSError, AttributeError) as e:
        raise RuntimeError(f"SGLANG_NEURON_HBM_TAIL_GUARD=1: the Neuron runtime library lacks the device allocation "
                           f"calls the guard makes ({e})") from e
    return lib


def _allocate(lib, vnc: int, size: int):
    """(handle, masked physical address, bytes) of one device allocation, or None when the runtime refuses it."""
    handle = ctypes.c_void_p()
    if lib.nrt_tensor_allocate(0, vnc, size, _NAME, ctypes.byref(handle)) != 0:  # 0: NRT_TENSOR_PLACEMENT_DEVICE
        return None
    info = _AllocationInfo()
    if lib.nrt_tensor_get_device_allocation_info(handle, ctypes.byref(info)) != 0:
        lib.nrt_tensor_free(ctypes.byref(handle))
        raise RuntimeError("SGLANG_NEURON_HBM_TAIL_GUARD=1: the runtime does not report a device allocation's address")
    return handle, int(info.physical_address) & ADDR_MASK, int(info.size)


def fill_arena(lib, vnc: int, max_allocations: int = MAX_FILL_ALLOCATIONS):
    """Fill the small-allocation arena of ``vnc``: (handles, [(address, bytes)]). The first allocation marks the
    arena's low end; each fill size runs until an allocation lands below it (the arena is full at that size and the
    runtime took memory elsewhere) or the runtime refuses one. Every handle is freed again on a refusal: no arena
    allocation, or the bound reached before the arena filled."""
    handles, allocations = [], []
    try:
        first = _allocate(lib, vnc, FILL_SIZES[-1])
        if first is None:
            raise RuntimeError("SGLANG_NEURON_HBM_TAIL_GUARD=1: the runtime refused the guard's first 4 KiB allocation")
        handles.append(first[0])
        allocations.append(first[1:])
        arena_low = first[1]
        for size in FILL_SIZES:
            while True:
                if len(handles) >= max_allocations:
                    raise RuntimeError(f"SGLANG_NEURON_HBM_TAIL_GUARD=1: the arena did not fill within "
                                       f"{max_allocations} allocations, so its top is unknown")
                got = _allocate(lib, vnc, size)
                if got is None:
                    break
                handles.append(got[0])
                allocations.append(got[1:])
                if got[1] < arena_low:
                    break
    except BaseException:
        for handle in handles:
            lib.nrt_tensor_free(ctypes.byref(handle))
        raise
    return handles, allocations


def install(lib=None, vnc: int = 0) -> int:
    """Hold the top GUARD_BYTES of ``vnc``'s small-allocation arena (this process's one visible logical core is VNC 0)
    for the life of the process; the bytes held. Idempotent."""
    if _HELD:
        return sum(size for _, size in _HELD)
    lib = _load_libnrt() if lib is None else lib
    handles, allocations = fill_arena(lib, vnc)
    keep, top = plan_keep(allocations)
    for i, handle in enumerate(handles):
        if i in keep:
            _HELD.append((handle, allocations[i][1]))
        else:
            lib.nrt_tensor_free(ctypes.byref(handle))
    held = sum(size for _, size in _HELD)
    logger.info("[neuron] HBM tail guard: holding %d KiB at the small-allocation arena's top (%d blocks, top end "
                "0x%x; the fill made %d allocations)", held >> 10, len(_HELD), top, len(handles))
    return held


def _reset_for_tests() -> None:
    _HELD.clear()
