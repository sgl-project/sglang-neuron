# SPDX-License-Identifier: Apache-2.0
"""The giga-kernel decode graph's DMA-order schedule (``dma_order.json`` beside this module) against the NEFF it
compiled to.

The compiler binds each constraint of the schedule to a backend instruction by name, and a name ends in the scope of
the kernel call in the decode graph, ``inst__I-<K>-0``, where ``K`` is the call's index in the graph: every operation
the graph traces before the kernel moves it (another tail, another embedding path, another input). A constraint whose
name the compiler does not find is skipped without an error, so a schedule written for another graph compiles to an
unscheduled kernel that serves correct output, only slower. Two checks together make sure the served NEFF is the
scheduled one:

- ``compiler_path`` names the schedule by its content in the compiler arguments, which torch-neuronx's compile cache
  keys a NEFF by, and checks that the file at that name holds the schedule's bytes, so a NEFF compiled or served
  under those arguments was compiled with this schedule; a stale cache can't serve one compiled with another.
- ``check_compiled`` reads the warmed decode NEFF and refuses the server when the schedule's kernel scope isn't the
  NEFF's or too few of its constraints name an instruction of the NEFF, so the schedule fits this graph.

Neither proves both: the names exist in a NEFF whether or not the compiler applied the constraints, and the content
path says nothing about whether the names match the graph. ``tools/giga_schedule.py`` rewrites a schedule to a NEFF's
scope.

The schedule was derived for GPT-OSS-120B's decode graph with the fused sampling tail, at DP16 x TP8 and a 10240-token
context (``--context-length 10240``; some constraint names carry it), under the kernel-library flags in
``models/catalog.py``'s giga decode settings.
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import logging
import os
import re
import stat
import tarfile
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEDULE = Path(__file__).with_name("dma_order.json")

SCOPE = re.compile(rb"inst__I-(\d+)-0")
NEFF_HEADER_BYTES = 1024
# A DMA name only the giga-kernel's NEFF carries (layer 0's QKV weight load), which tells its NEFF from the server's
# other cached programs.
GIGA_MARKER = b"-L0_qkv_w"
# The share of the schedule's constraints that must name an instruction of the NEFF. A NEFF's debug names do not list
# every constraint the compiler bound: it folds some small loads into one DMA instruction under one of their names (a
# scale's four quadrant loads, the expert biases), so a correct build names fewer than all of them. Below the share
# the NEFF isn't the kernel build the schedule was derived on (another kernel version); one differing build flag
# (models/catalog.py) moves only a few percent of the names.
MIN_BOUND_SHARE = 0.8


def compiler_path(schedule_path=SCHEDULE) -> str:
    """A copy of the schedule at a path named by its content, for the compiler's ``--dma-order-constraint-file``.

    torch-neuronx's compile cache keys a NEFF by its compiler arguments, not by the files they name. Under the shipped
    file's own path, a changed schedule would keep the arguments, and so the cached NEFF compiled against the old one,
    whose constraints may not bind; ``check_compiled`` can't tell, since the names exist either way. Every rank and
    every restart on a node computes the same path, so they share the NEFF.

    The path is predictable, so its directory must be this user's own (not a symlink, written by no one else), and a
    file already there that isn't this user's regular file holding the schedule's bytes is replaced, through a fresh
    temporary file, before the compiler reads it."""
    data = Path(schedule_path).read_bytes()
    directory = Path(tempfile.gettempdir()) / "sglang_neuron"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = os.lstat(directory)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(f"{directory} holds the giga decode's DMA-order schedule for the compiler, but it is not a "
                           f"directory this user owns: remove it, or point TMPDIR elsewhere")
    if info.st_mode & 0o022:
        os.chmod(directory, info.st_mode & ~0o022)
    target = directory / f"giga_dma_order-{hashlib.md5(data).hexdigest()}.json"
    try:
        info = os.lstat(target)
        same = stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and target.read_bytes() == data
    except OSError:
        same = False
    if not same:
        fd, part = tempfile.mkstemp(dir=directory, prefix=f".{target.name}.")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(part, target)
        except BaseException:
            os.unlink(part)
            raise
    return str(target)


def neff_blob(data: bytes) -> bytes:
    """Every member of a NEFF (a 1024-byte header, then a gzipped tar) concatenated: where instruction names live."""
    blob = []
    with tarfile.open(fileobj=io.BytesIO(data[NEFF_HEADER_BYTES:]), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                blob.append(tar.extractfile(member).read())
    return b"\n".join(blob)


def kernel_scope(blob: bytes) -> int:
    """The scope most instruction names of a NEFF carry: the giga-kernel's, the graph's one large kernel call."""
    counts = collections.Counter(int(m.group(1)) for m in SCOPE.finditer(blob))
    if not counts:
        raise ValueError("the NEFF names no kernel-call scope (inst__I-<K>-0): not a giga-kernel decode NEFF")
    return counts.most_common(1)[0][0]


def schedule_scope(schedule: dict) -> int:
    """The one kernel-call scope a schedule's constraint names carry."""
    scopes = {int(m.group(1)) for op in schedule["constraint_ops"] for m in SCOPE.finditer(op["instr_name"].encode())}
    if len(scopes) != 1:
        raise ValueError(f"the schedule holds {len(scopes)} kernel-call scopes ({sorted(scopes)}), not one")
    return scopes.pop()


def rewrite(schedule: dict, k: int) -> dict:
    """``schedule`` with its one kernel-call scope replaced by ``k``."""
    old, new = f"inst__I-{schedule_scope(schedule)}-0", f"inst__I-{k}-0"
    out = json.loads(json.dumps(schedule))
    for op in out["constraint_ops"]:
        op["instr_name"] = op["instr_name"].replace(old, new)
        if "dbg" in op:
            op["dbg"] = op["dbg"].replace(old, new)
    return out


def bound(schedule: dict, blob: bytes) -> tuple[int, int, collections.Counter]:
    """(constraints, constraints whose instruction the NEFF names, missing names grouped by their prefix)."""
    names = [op["instr_name"] for op in schedule["constraint_ops"]]
    missing = [n for n in names if n.encode() not in blob]
    groups = collections.Counter(re.sub(r"_inst__I-\d+-0$", "", n).rstrip("0123456789_") for n in missing)
    return len(names), len(names) - len(missing), groups


def check_compiled(neffs, schedule_path=SCHEDULE) -> None:
    """Refuse the server unless the giga decode NEFF among ``neffs`` (bytes of the warmed decode programs) carries
    the schedule's kernel-call scope and names at least MIN_BOUND_SHARE of its constraints."""
    with open(schedule_path) as f:
        schedule = json.load(f)
    want = schedule_scope(schedule)
    blobs = [b for b in (neff_blob(data) for data in neffs) if GIGA_MARKER in b]
    if not blobs:
        raise RuntimeError("no warmed decode program is the giga-kernel's, so its DMA-order schedule could not be "
                           "checked against the compiled decode graph")
    for blob in blobs:
        got = kernel_scope(blob)
        if got != want:
            raise RuntimeError(
                f"the giga-kernel DMA-order schedule {schedule_path} names the kernel-call scope I-{want}, but this "
                f"decode graph compiled the kernel at I-{got}: none of its constraints bind, and the kernel would run "
                f"unscheduled. The graph traces a different op sequence before the kernel than the schedule was "
                f"derived on: rewrite it to this scope (python3 tools/giga_schedule.py rewrite {schedule_path} {got} "
                f"{schedule_path})")
        total, ok, missing = bound(schedule, blob)
        if ok < MIN_BOUND_SHARE * total:
            raise RuntimeError(
                f"the giga-kernel DMA-order schedule {schedule_path} binds {ok} of its {total} constraints in this "
                f"decode graph (scope I-{got}): it was derived for another kernel build or kernel flags (missing "
                f"{dict(missing.most_common(6))}). Serve the nkilib build the schedule was derived on, or derive a "
                f"schedule for this build and check it with python3 tools/giga_schedule.py check <schedule> "
                f"<decode.neff>")
        logger.info("[giga] DMA-order schedule: %d of %d constraints name an instruction of the decode NEFF "
                    "(scope I-%d)", ok, total, got)
