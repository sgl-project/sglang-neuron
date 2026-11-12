# SPDX-License-Identifier: Apache-2.0
"""A dump of the logits the Kimi-K3 prefill and decode sample from, to check the decode's logits against the
prefill's, teacher forced.

Part of the Kimi-K3 test dump, with the prefill's byte dump (prefill_dump.py): off unless
``SGLANG_NEURON_KIMI_K3_TEST_DUMP_DIR`` names a directory. ``SGLANG_NEURON_KIMI_K3_TEST_DUMP_RID``, a regular
expression, picks the requests by SGLang request id (``re.search``); unset, every request is dumped. With the dump on,
the graphs also hand out the logits they sample from (the prefill's last stage: the sampled row, copied into a buffer
of the graph; the decode's verify program: one more output). These are graphs of their own: a launch with the dump
compiles its own NEFFs, and its perf numbers do not count.

Every rank writes its vocabulary shard of each row, float32, C order: rank ``r`` of a TP group of ``n`` ranks holds the
ids ``r * V / n .. (r + 1) * V / n - 1``. Under ``<dir>/<request>/``:

* ``prefill/rank<r>.bin`` ``[V / n]`` and ``prefill/rank<r>.json``: on the prefill's last stage, after the request's
  last chunk, the logits of its last prompt position (they give its first output token). A later request under the
  same id overwrites them;
* ``decode/rank<r>/row<p>.bin`` ``[a, V / n]`` and ``row<p>.json``: on each rank of the request's decode replica,
  after each step's read, the verify rows of the step's accepted prefix: ``p`` is the step's row 0 (the position of its
  first input token), ``a`` the step's emitted token count, and row ``j`` the row whose argmax (greedy) or draw is the
  step's ``j``-th emitted token. Its context is the prompt and every token emitted before it. The request's first step
  has row 0 = its prompt length, and each step's row 0 is the last one's plus its count, so the steps in row-0 order
  give the output tokens ``1 ..`` in order (token 0 comes from the prefill). A step that runs after the request has its
  tokens (its rows are past the end) is written too; a later request under the same id writes over the same names.

``read_prefill`` and ``read_decode`` put the shards back together; ``compare_rows`` measures a row against a reference
row of the same position.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch

from sglang_neuron.models.kimi_k3 import prefill_dump

logger = logging.getLogger(__name__)

DIR_ENV, RID_ENV = prefill_dump.DUMP_DIR_ENV, prefill_dump.DUMP_RID_ENV
FORMAT = 1
_UNSAFE = re.compile(r"[^0-9A-Za-z_.-]")


def request_dir_name(rid: str) -> str:
    """The directory name of a request id: its characters outside ``[0-9A-Za-z_.-]`` become ``_``."""
    return _UNSAFE.sub("_", str(rid)) or "_"


def from_env(environ: Mapping[str, str] = os.environ) -> Optional["LogitsDump"]:
    """The dump the environment asks for, or None (``DIR_ENV`` unset or empty)."""
    root = environ.get(DIR_ENV)
    if not root:
        return None
    return LogitsDump(root, environ.get(RID_ENV) or None)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _host(tensor: torch.Tensor, event=None) -> torch.Tensor:
    """A device tensor on the host: after ``event`` (a step's launch event) when given, else after every launched
    graph (the prefill writes the buffer in its forward)."""
    if tensor.device.type == "cpu":
        return tensor.detach().clone()
    from sglang_neuron.backend import device_io

    if event is not None:
        return device_io.readback_many([tensor], event)[0]
    from sglang_neuron.model_executor.async_execution import drain_launches

    drain_launches()
    return tensor.to("cpu")


class LogitsDump:
    """The dump of the requests whose id matches ``rid_pattern`` (all when None) under ``root`` (module docstring)."""

    def __init__(self, root, rid_pattern: Optional[str] = None):
        self.root = Path(root)
        self.pattern = None if rid_pattern is None else re.compile(rid_pattern)

    def wants(self, rid) -> bool:
        return rid is not None and (self.pattern is None or self.pattern.search(str(rid)) is not None)

    def request_dir(self, rid) -> Path:
        return self.root / request_dir_name(rid)

    # the prefill

    def after_prefill(self, rid, buffer: torch.Tensor, *, rank: int, world: int, tokens: int,
                      prompt=None) -> Optional[Path]:
        """Write the last stage's logits row ``buffer [1, V / world]`` of request ``rid`` after a chunk that ended at
        ``tokens`` tokens, when the chunk is the request's last (``prompt``: its prompt length, as an int or the first
        entry of SGLang's ``orig_seq_lens``; None: every chunk) and the dump wants the request; return the file, else
        None."""
        if isinstance(prompt, torch.Tensor):
            prompt = int(prompt.reshape(-1)[0].to("cpu"))
        elif prompt is not None and not isinstance(prompt, int):
            prompt = int(list(prompt)[0])
        if not self.wants(rid) or (prompt is not None and tokens < prompt):
            return None
        row = _host(buffer).reshape(-1).to(torch.float32).contiguous()
        directory = self.request_dir(rid) / "prefill"
        path = directory / f"rank{rank}.bin"
        _write(path, row.numpy().tobytes())
        meta = {"format": FORMAT, "rid": str(rid), "rank": rank, "world": world, "shard": int(row.numel()),
                "tokens": tokens, "position": tokens - 1}
        _write(directory / f"rank{rank}.json", (json.dumps(meta, sort_keys=True) + "\n").encode())
        return path

    # the decode

    def after_step(self, rids, logits: torch.Tensor, row0, accepted, tokens, *, queries: int, rank: int, world: int,
                   replica: int, event=None) -> list:
        """Write this rank's verify rows of the accepted prefix of each wanted request of a step: ``rids`` in batch
        order, ``logits [B * Q, V / world]`` request-major on the device (read after ``event``), and the step's host
        ``row0 [B]``, ``accepted [B]`` and ``tokens [B, Q]``. Return the files written."""
        picked = [i for i, rid in enumerate(rids) if self.wants(rid)]
        if not picked:
            return []
        rows = _host(logits, event).to(torch.float32)
        paths = []
        for i in picked:
            rid, count, first = rids[i], int(accepted[i]), int(row0[i])
            block = rows[i * queries: i * queries + count].contiguous()
            directory = self.request_dir(rid) / "decode" / f"rank{rank}"
            path = directory / f"row{first:09d}.bin"
            _write(path, block.numpy().tobytes())
            meta = {"format": FORMAT, "rid": str(rid), "rank": rank, "world": world, "replica": replica,
                    "shard": int(block.shape[-1]), "row0": first, "accepted": count,
                    "tokens": [int(t) for t in list(tokens[i])[:count]]}
            _write(directory / f"row{first:09d}.json", (json.dumps(meta, sort_keys=True) + "\n").encode())
            paths.append(path)
        return paths


# reading a dump back


def _shards(ranks: list, world: int) -> list:
    """The ranks of a TP group of ``world`` in vocabulary order; refused unless ``ranks`` holds each one."""
    if sorted(ranks) != list(range(world)):
        raise ValueError(f"the dump holds the shards of ranks {sorted(ranks)}, not of all {world} ranks")
    return list(range(world))


def read_prefill(request_dir) -> tuple:
    """``(row [V] float32, meta)``: the prefill row of a request, its rank shards put together."""
    directory = Path(request_dir) / "prefill"
    metas = {}
    for path in directory.glob("rank*.json"):
        meta = json.loads(path.read_text())
        metas[int(meta["rank"])] = meta
    if not metas:
        raise FileNotFoundError(f"no prefill logits under {directory}")
    world = int(next(iter(metas.values()))["world"])
    parts = [np.fromfile(directory / f"rank{r}.bin", dtype=np.float32) for r in _shards(list(metas), world)]
    meta = dict(metas[0])
    meta.pop("rank")
    if len({m["tokens"] for m in metas.values()}) != 1:
        raise ValueError(f"the prefill shards under {directory} are of different requests: "
                         f"{sorted({m['tokens'] for m in metas.values()})} tokens")
    return np.concatenate(parts), meta


def read_decode(request_dir) -> tuple:
    """``(rows [n, V] float32, tokens [n], steps)``: the decode rows of a request from its first step on (row ``i``
    gives output token ``i + 1``), its rank shards put together, the emitted tokens, and each step's meta in row-0
    order. The steps are kept while each one's row 0 is the last one's plus its count (``steps[0]["row0"]`` is the
    prompt length of a whole dump)."""
    root = Path(request_dir) / "decode"
    ranks = sorted(int(p.name[len("rank"):]) for p in root.glob("rank*") if p.is_dir())
    if not ranks:
        raise FileNotFoundError(f"no decode logits under {root}")
    world = None
    per_rank = {}
    for rank in ranks:
        steps = {}
        for path in (root / f"rank{rank}").glob("row*.json"):
            meta = json.loads(path.read_text())
            steps[int(meta["row0"])] = (meta, path.with_suffix(".bin"))
        per_rank[rank] = steps
        world = int(next(iter(steps.values()))[0]["world"]) if steps else world
    order = _shards(ranks, world)
    reference = per_rank[0]
    for rank in order:
        if {p: m["tokens"] for p, (m, _) in per_rank[rank].items()} != {p: m["tokens"] for p, (m, _) in
                                                                         reference.items()}:
            raise ValueError(f"rank {rank}'s decode steps under {root} differ from rank 0's")
    chain = []
    for position in sorted(reference):
        meta = reference[position][0]
        if chain and position != chain[-1]["row0"] + chain[-1]["accepted"]:
            break
        chain.append(meta)
    blocks = []
    for meta in chain:
        parts = [np.fromfile(per_rank[rank][meta["row0"]][1], dtype=np.float32) for rank in order]
        blocks.append(np.concatenate([p.reshape(int(meta["accepted"]), -1) for p in parts], axis=1))
    rows = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 0), dtype=np.float32)
    tokens = [t for meta in chain for t in meta["tokens"]]
    return rows, tokens, chain


# measuring a row against a reference row

TOP_K = (5, 50, 1000, None)
TOP_K_LABELS = {5: "K5", 50: "K50", 1000: "K1000", None: "All"}


def _without_shift(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``actual`` less the constant of the least-squares line ``actual = a * reference + b`` over the finite entries:
    a constant added to every logit leaves the softmax as it is."""
    finite = np.isfinite(actual) & np.isfinite(reference)
    x, y = reference[finite].astype(np.float64), actual[finite].astype(np.float64)
    design = np.stack([x, np.ones_like(x)], axis=1)
    (_, shift), *_ = np.linalg.lstsq(design, y, rcond=None)
    return actual - shift


def top_k_errors(actual: np.ndarray, reference: np.ndarray) -> dict:
    """The largest logit error over the reference's top-k ids (k of ``TOP_K``, None: every finite id), relative to the
    largest reference logit magnitude over those ids, after the shift is removed (``_without_shift``)."""
    actual = _without_shift(np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64))
    reference = np.asarray(reference, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(reference) & np.isfinite(actual))
    order = finite[np.argsort(-reference[finite], kind="stable")]
    errors = {}
    for k in TOP_K:
        ids = order if k is None else order[:k]
        scale = float(np.abs(reference[ids]).max())
        errors[TOP_K_LABELS[k]] = float(np.abs(actual[ids] - reference[ids]).max() / scale) if scale else math.inf
    return errors


def _log_softmax(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64)
    top = np.max(row[np.isfinite(row)])
    shifted = row - top
    return shifted - np.log(np.sum(np.exp(shifted[np.isfinite(shifted)])))


def compare_rows(actual: np.ndarray, reference: np.ndarray) -> dict:
    """One row against the reference row of the same position: both argmax tokens, the reference logit gap from its
    top to ``actual``'s token, ``top_k_errors``, KL(reference || actual) and the reference's top-1 probability."""
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError(f"a row of {actual.shape[0]} logits against a reference of {reference.shape[0]}")
    token, ref_token = int(np.nanargmax(actual)), int(np.nanargmax(reference))
    ref_log, log = _log_softmax(reference), _log_softmax(actual)
    ref_prob = np.exp(ref_log)
    both = np.isfinite(ref_log) & np.isfinite(log)
    kl = float(np.sum(ref_prob[both] * (ref_log[both] - log[both])))
    return {"token": token, "reference_token": ref_token, "gap": float(reference[ref_token] - reference[token]),
            **top_k_errors(actual, reference), "kl": kl, "reference_top1_prob": float(ref_prob[ref_token]),
            "nonfinite": int((~np.isfinite(actual)).sum())}
