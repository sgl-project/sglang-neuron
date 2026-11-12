# SPDX-License-Identifier: Apache-2.0
"""A debug dump of the bytes a Kimi-K3 prefill leaves for one request, for a byte comparison with a decode's.

Part of the Kimi-K3 test dump, which the e2e tests turn on with the logits dump (logits_dump.py): off unless
``SGLANG_NEURON_KIMI_K3_TEST_DUMP_DIR`` names a directory. ``SGLANG_NEURON_KIMI_K3_TEST_DUMP_RID``, a regular
expression, picks the requests by SGLang request id (``re.search``); unset, every request is dumped. The runtime calls
``PrefillDump.after_forward`` after each prefill forward (runtime.KimiK3PrefillRuntime.run_prefill); the dump runs
after the request's last chunk only (its tokens reach SGLang's ``orig_seq_lens``), outside the traced graph: it waits
for this process's launched graphs (async_execution.drain_launches), then copies pool slices to the host.

Each rank of each stage writes ``<dir>/<request>/stage<s>/rank<r>/``: one ``.bin`` per tensor (raw bytes, C order;
bf16 as its uint16 bits) and ``manifest.json`` (the header and one entry per tensor: name, file, shape, dtype, layer,
kind, bytes, sha256). Tensors:

* ``mla/L<layer>.pages`` uint8 ``[pages, 128, 576]``: this rank's block of each of the request's virtual pages, in
  block-table order, for each MLA layer of the stage. Virtual page ``v`` holds the request's stripe
  ``v * world + rank`` (tokens ``stripe * 128 ..``); rows after the request's last token are the fresh-page zeros;
* ``kda/L<layer>.conv`` bf16 ``[3 hc D, K - 1]`` (channel-major, each channel's taps oldest first; channel
  ``j * hc * D + h * D + d`` for ``j`` = q, k, v and this rank's head ``h`` of ``hc``; the header's ``conv_layout``) and ``kda/L<layer>.temporal`` bf16 ``[hc, D_k, D_v]``:
  the request's state slot after the last chunk, for each KDA layer of the stage; this rank holds the heads
  ``rank * hc .. rank * hc + hc - 1``;
* ``draft/D<l>.pages`` uint8 ``[pages, 128, 576]``: the drafter row pool of drafter layer ``l`` at the same blocks, on
  the drafter writer stage. The header gives the RMS of each layer's rows of the request read as FP8
  (``draft_fp8_rms``) and ``draft_rows_suspect`` when one is under ``DRAFT_FP8_RMS_FLOOR`` (logged as a warning).

A request whose first forward follows a cached prefix (a prefix hit) also writes, before that forward
(``PrefillDump.before_forward``), the KDA states its slot starts from, under ``<dir>/<request>/restored/`` in the same
layout (``kda/`` only; ``tokens`` is the cached prefix).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Mapping, Optional

import torch

logger = logging.getLogger(__name__)

DUMP_DIR_ENV = "SGLANG_NEURON_KIMI_K3_TEST_DUMP_DIR"
DUMP_RID_ENV = "SGLANG_NEURON_KIMI_K3_TEST_DUMP_RID"
FORMAT = 1
MANIFEST = "manifest.json"
DRAFT_FP8_RMS_FLOOR = 0.1
"""A drafter layer whose rows, read as FP8 e4m3fn, have a smaller RMS is flagged: a normed latent row has an RMS
near its norm weight's, so bytes far below are not FP8 drafter rows (``draft_rows_suspect``)."""
_UNSAFE = re.compile(r"[^0-9A-Za-z_.-]")


def from_env(environ: Mapping[str, str] = os.environ) -> Optional["PrefillDump"]:
    """The dump the environment asks for, or None (``DUMP_DIR_ENV`` unset or empty)."""
    root = environ.get(DUMP_DIR_ENV)
    if not root:
        return None
    return PrefillDump(root, environ.get(DUMP_RID_ENV) or None)


def request_dir_name(rid: str) -> str:
    """The directory name of a request id: its characters outside ``[0-9A-Za-z_.-]`` become ``_``."""
    return _UNSAFE.sub("_", str(rid)) or "_"


def _first(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].to("cpu"))
    return int(list(value)[0])


def host_pages(pool: torch.Tensor, pages: list) -> torch.Tensor:
    """``pool[pages]`` on the host. A device pool is read one run of consecutive pages at a time (a slice of whole
    pages is one contiguous region), so no gather runs on the device."""
    index = torch.tensor(pages, dtype=torch.int64)
    if pool.device.type == "cpu":
        return pool[index].clone()
    try:
        parts, start = [], 0
        while start < len(pages):
            stop = start + 1
            while stop < len(pages) and pages[stop] == pages[stop - 1] + 1:
                stop += 1
            parts.append(pool.narrow(0, pages[start], stop - start).to("cpu"))
            start = stop
        return torch.cat(parts) if parts else torch.empty((0, *pool.shape[1:]), dtype=pool.dtype)
    except RuntimeError as error:
        # a device whose copy of a slice fails: the whole pool to the host, then the pages
        logger.warning("Kimi-K3 prefill dump: a slice copy failed (%s); copying the whole pool", error)
        return pool.to("cpu")[index].clone()


def host_slot(pool: torch.Tensor, slot: int) -> torch.Tensor:
    """``pool[slot]`` on the host (a slice of one slot is one contiguous region)."""
    return host_pages(pool, [int(slot)])[0]


def fp8_rms(pages: torch.Tensor, rank: int, world: int, tokens: int) -> Optional[float]:
    """The RMS of the FP8 e4m3fn values of this rank's rows of tokens ``0 .. tokens - 1`` in ``pages [n, stripe,
    W]`` uint8 (page ``v`` holds stripe ``v * world + rank``), NaN codes read as 0; None when the rank holds none."""
    count, stripe = pages.shape[0], pages.shape[1]
    start = (torch.arange(count) * world + rank) * stripe
    held = (start[:, None] + torch.arange(stripe)[None, :]) < tokens
    if not bool(held.any()):
        return None
    values = torch.nan_to_num(pages[held].view(torch.float8_e4m3fn).float(), nan=0.0)
    return float(values.square().mean().sqrt())


def _bits(tensor: torch.Tensor) -> tuple:
    """The C-order bytes of a host tensor and its dtype name; bf16 as uint16 bits, FP8 as uint8."""
    t = tensor.detach().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes(), "bfloat16"
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return t.view(torch.uint8).numpy().tobytes(), str(t.dtype).replace("torch.", "")
    return t.numpy().tobytes(), str(t.dtype).replace("torch.", "")


class _Writer:
    def __init__(self, directory: Path, header: dict):
        self.directory, self.header, self.entries = directory, header, []

    def write(self, name: str, tensor: torch.Tensor, *, kind: str, layer: Optional[int] = None) -> None:
        data, dtype = _bits(tensor)
        path = self.directory / f"{name}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        self.entries.append({"name": name, "file": f"{name}.bin", "shape": list(tensor.shape), "dtype": dtype,
                             "layer": layer, "kind": kind, "bytes": len(data),
                             "sha256": hashlib.sha256(data).hexdigest()})

    def flush(self) -> Path:
        path = self.directory / MANIFEST
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"format": FORMAT, "header": self.header, "tensors": self.entries}, indent=1,
                                  sort_keys=True) + "\n")
        os.replace(tmp, path)
        return path


class PrefillDump:
    """The dump of the requests whose id matches ``rid_pattern`` (all when None) under ``root`` (module docstring)."""

    def __init__(self, root, rid_pattern: Optional[str] = None):
        self.root = Path(root)
        self.pattern = None if rid_pattern is None else re.compile(rid_pattern)
        # the requests whose first forward ran here (before_forward) and whose last chunk has not (after_forward)
        self._started: set = set()

    def wants(self, rid) -> bool:
        return rid is not None and (self.pattern is None or self.pattern.search(str(rid)) is not None)

    def before_forward(self, runtime, fb, md) -> Optional[Path]:
        """Dump the KDA states a request starts from when its first forward here follows a cached prefix (a prefix
        hit: the state slot holds the tree's checkpoint, copied in before the forward) and the request is one the dump
        wants; return the manifest path, else None. Written to ``<dir>/<request>/restored/stage<s>/rank<r>/``."""
        from sglang_neuron.models.kimi_k3.runtime import STATE_SLOT_ATTR

        rids = getattr(fb, "rids", None)
        rid = rids[0] if rids else None
        slot = getattr(md, STATE_SLOT_ATTR, None)
        if slot is None or not self.wants(rid) or rid in self._started:
            return None
        self._started.add(rid)
        cached = int(md.cached_seq_len)
        if cached == 0:
            return None
        return self.write(runtime, rid, md, cached, int(slot), part="restored", mla=False)

    def after_forward(self, runtime, fb, md) -> Optional[Path]:
        """Dump the request of ``fb`` (one request's prefill chunk) when the chunk is its last and the request is one
        the dump wants; return the manifest path, else None. A warm-up's metadata (no state slot) dumps nothing."""
        from sglang_neuron.models.kimi_k3.runtime import STATE_SLOT_ATTR

        rids = getattr(fb, "rids", None)
        rid = rids[0] if rids else None
        slot = getattr(md, STATE_SLOT_ATTR, None)
        if slot is None or not self.wants(rid):
            return None
        tokens = int(md.cached_seq_len) + int(md.num_tokens)
        prompt = _first(getattr(fb, "orig_seq_lens", None))
        if prompt is not None and tokens < prompt:
            return None
        if prompt is not None:
            # the request's last chunk ran: a later request under the same id starts again (before_forward)
            self._started.discard(rid)
        return self.write(runtime, rid, md, tokens, int(slot))

    def write(self, runtime, rid: str, md, tokens: int, state_slot: int, *, part: Optional[str] = None,
              mla: bool = True) -> Path:
        """Write this rank's bytes of request ``rid`` (``tokens`` tokens, state slot ``state_slot``) under the
        request's directory, or its ``part`` subdirectory; ``mla`` False writes the KDA states only."""
        from sglang_neuron.backend import device_io
        from sglang_neuron.models.kimi_k3.runtime import stripe_cycle

        if device_io.on_device():
            from sglang_neuron.model_executor.async_execution import drain_launches

            # the forward writes the pools in place; every launched graph ends before a pool is read
            drain_launches()
        graph, layout = runtime.model_graph, runtime.layout
        plan, caches = graph.plan, graph.caches
        pages = -(-tokens // stripe_cycle(layout))
        table = [int(p) for p in md.block_table.reshape(-1)[:pages].to("cpu").tolist()]
        header = {
            "rid": str(rid), "request": request_dir_name(rid), "stage": plan.pp_rank, "pp_size": plan.pp_size,
            "layers": [plan.layers.start, plan.layers.stop], "rank": layout.rank, "world": layout.size,
            "stripe": layout.block, "tokens": tokens, "block_table": table, "state_slot": state_slot,
            "mla_layers": list(plan.mla_layers), "kda_layers": list(plan.kda_layers),
            "draft_layers": len(caches.draft) if graph.drafter_writer else 0,
            "conv_layout": "[3 hc D, K - 1] channel-major; channel j * hc * D + h * D + d, j = q, k, v",
            "temporal_layout": "[hc, D_k, D_v]",
        }
        request = self.root / header["request"]
        directory = (request if part is None else request / part) / f"stage{plan.pp_rank}" / f"rank{layout.rank}"
        writer = _Writer(directory, header)
        for layer in plan.mla_layers if mla else ():
            writer.write(f"mla/L{layer:02d}.pages", host_pages(caches.mla[layer], table), kind="mla_pages",
                         layer=layer)
        for layer in plan.kda_layers:
            conv, temporal = caches.kda[layer]
            writer.write(f"kda/L{layer:02d}.conv", host_slot(conv, state_slot), kind="kda_conv", layer=layer)
            writer.write(f"kda/L{layer:02d}.temporal", host_slot(temporal, state_slot), kind="kda_temporal",
                         layer=layer)
        if graph.drafter_writer and mla:
            rms = []
            for index, pool in enumerate(caches.draft):
                draft_pages = host_pages(pool, table)
                writer.write(f"draft/D{index}.pages", draft_pages, kind="draft_pages", layer=index)
                rms.append(fp8_rms(draft_pages, layout.rank, layout.size, tokens))
            low = [i for i, value in enumerate(rms) if value is not None and value < DRAFT_FP8_RMS_FLOOR]
            header["draft_fp8_rms"] = rms
            header["draft_rows_suspect"] = bool(low)
            if low:
                logger.warning("Kimi-K3 prefill dump: request %s, stage %d rank %d: the rows of drafter layers %s read "
                               "as FP8 have an RMS of %s, under %s: the pools do not hold FP8 drafter rows", rid,
                               plan.pp_rank, layout.rank, low, [round(rms[i], 4) for i in low], DRAFT_FP8_RMS_FLOOR)
        path = writer.flush()
        logger.info("Kimi-K3 prefill dump: request %s, stage %d rank %d, %d tokens, %d pages -> %s", rid,
                    plan.pp_rank, layout.rank, tokens, pages, directory)
        return path
