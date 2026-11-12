# SPDX-License-Identifier: Apache-2.0
"""The DP-attention MLP sync's exchange, kept as an immutable snapshot of the cohort's decision on the batch it ran for
(``ScheduleBatch.mlp_sync_batch_metadata``, carried to ``ForwardBatch.mlp_sync_batch_metadata``).

A model that lowers a whole DP-attention step into one program (a Neuron megakernel host) has to know what every
attention-DP participant runs this step: each one's forward mode, token count, request count and decode-request count.
Upstream's ``prepare_mlp_sync_batch_raw`` already exchanges a row per participant (``MLPSyncBatchInfo``), but it gathers
no request counts, keeps the gathered matrix only as a mutable tensor on a transient object, and hands the batch only
the derived token lists. Here:

- each participant's row gains three columns after upstream's, its request count and decode-request count (``-1``:
  unknown; a fallback/inactive slot reads ``0``): the counts of its local batch, a pure decode batch's requests, a mixed
  batch's decoding tail, else none; and the longest cached prefix of its extend requests (``max_prefix_tokens``, 0
  for any other batch), which a planner sizes a prefill program's prior by (Qwen3.5's prior tiers; any hybrid
  model whose prefill NEFF is keyed by its prior capacity);
- once the exchange has filled ``tp0_info_cpu`` (``finalize_local`` for one participant, ``all_gather`` otherwise), it
  is snapshotted into an ``MLPSyncBatchMetadata`` of Python tuples, without another collective or device copy;
- ``_update_gather_batch`` sets the snapshot on the batch it updates, ``ScheduleBatch.copy`` keeps it, and
  ``ForwardBatch.init_new`` carries it to the forward batch; any batch the exchange has not reached reads ``None``.

The snapshot is the scheduling decision before graph padding and decode-to-extend conversion, not a promise that every
rank runs the same program: a consumer validates it against its own topology. Every rank gathers the same matrix in the
same exchange, so every rank holds the same snapshot (rank agreement). The exchange grows by two int64 columns per
participant (three after this column). A process whose MLP sync does not run upstream's exchange (the platform's DP identity,
``managers/dp_identity``) sets no snapshot.

The hooks are registered at every launch (their sources are checked there) but stay inert, every hooked function
upstream's, until ``enable`` turns the snapshot on: megakernel_platform does that when the served model's megakernel
spec, its built object or its pool layout declares ``mlp_sync_cohort_metadata`` (Qwen3.5's cohort step plan), after
every rank agreed on it, before the first scheduler step. A GPT-OSS launch never turns it on, so its exchange keeps
upstream's columns. ``enable`` refuses a process whose MLP sync is the platform's DP identity (serving_policy's
dp_identity), which runs no upstream exchange and so would leave the consumer without a snapshot.

Upstream: an interface SGLang can take as it is (a source patch of SGLang can add it, with a dataclass for the
snapshot).

Targets, md5-pinned against upstream SGLang 43881a7: ``prepare_mlp_sync_batch_raw`` (rebuilt), and
``MLPSyncBatchInfo._get_local_tensor``, ``_get_fallback_tensor``, ``finalize_local``, ``all_gather``,
``_update_gather_batch``, ``ScheduleBatch.copy`` and ``ForwardBatch.init_new`` (wrapped)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_DP_ATTN = "sglang.srt.managers.scheduler_components.dp_attn"
_SCHEDULE_BATCH = "sglang.srt.managers.schedule_batch"
_FORWARD_BATCH = "sglang.srt.model_executor.forward_batch_info"
RAW = f"{_DP_ATTN}.prepare_mlp_sync_batch_raw"
LOCAL_TENSOR = f"{_DP_ATTN}.MLPSyncBatchInfo._get_local_tensor"
FALLBACK_TENSOR = f"{_DP_ATTN}.MLPSyncBatchInfo._get_fallback_tensor"
FINALIZE_LOCAL = f"{_DP_ATTN}.MLPSyncBatchInfo.finalize_local"
ALL_GATHER = f"{_DP_ATTN}.MLPSyncBatchInfo.all_gather"
UPDATE_GATHER = f"{_DP_ATTN}._update_gather_batch"
COPY = f"{_SCHEDULE_BATCH}.ScheduleBatch.copy"
INIT_NEW = f"{_FORWARD_BATCH}.ForwardBatch.init_new"
ATTR = "mlp_sync_batch_metadata"
# (module, qualified name): md5 of the source
_TARGETS_MD5 = {
    # builds MLPSyncBatchInfo(...) from local_batch, then finalize_local() or all_gather(), then _update_gather_batch
    (_DP_ATTN, "prepare_mlp_sync_batch_raw"): "68b96f570ba15b939509a35f48793127",
    # a participant's row: num_tokens at column 0, local_forward_mode at column 5, 11 columns
    (_DP_ATTN, "MLPSyncBatchInfo._get_local_tensor"): "0670dce643875c52307d551f61a82ee9",
    (_DP_ATTN, "MLPSyncBatchInfo._get_fallback_tensor"): "dd7cb7b4a437da36c48c300d0899f288",
    # each fills tp0_info_cpu, the participants' rows as a CPU matrix
    (_DP_ATTN, "MLPSyncBatchInfo.finalize_local"): "cf07d3f6c249ec1c8206a738c1b42ebd",
    (_DP_ATTN, "MLPSyncBatchInfo.all_gather"): "cabb4cbf311d6eacbf5ff610a1ffbf10",
    (_DP_ATTN, "_update_gather_batch"): "74fa1b927e958161b67d3cec74a0fb9f",
    # the copy process_batch_result reads, built field by field
    (_SCHEDULE_BATCH, "ScheduleBatch.copy"): "214b0c5e6aac2bd19ee2940e8ebf9a9b",
    (_FORWARD_BATCH, "ForwardBatch.init_new"): "30f1ef2894f1a2341f0d3a65dd2cffdd",
}
_TOKENS_COLUMN, _MODE_COLUMN = 0, 5
# whether the snapshot is on in this process (enable); off, every hook is upstream's function
_STATE = {"enabled_by": None}


def enabled() -> bool:
    return _STATE["enabled_by"] is not None


def enable(by: str) -> None:
    """Turn the cohort snapshot on for this process (``by``: who reads it). Refused under the platform's DP identity,
    whose MLP sync runs no upstream exchange."""
    from sglang_neuron import serving_policy

    if serving_policy.current().dp_identity:
        raise ValueError(f"{by} reads the DP-attention MLP sync cohort snapshot, which the platform's DP identity "
                         "(its MLP sync replaces upstream's exchange) never produces; serve it without the DP identity")
    _STATE["enabled_by"] = by


def disable() -> None:
    """Back to upstream's exchange (tests)."""
    _STATE["enabled_by"] = None
# prepare_mlp_sync_batch_raw's MLPSyncBatchInfo construction, and the same with the local batch's request counts set
SUBSTITUTIONS = (
    ("        attention_pairs=_local_attention_pairs(local_batch),\n"
     "    )\n",
     "        attention_pairs=_local_attention_pairs(local_batch),\n"
     "    )\n"
     "    from sglang_neuron.sglang_patches.mlp_sync_cohort_snapshot import cohort_counts, cohort_max_prefix\n"
     "\n"
     "    mlp_sync_info.num_requests, mlp_sync_info.num_decode_requests = cohort_counts(local_batch)\n"
     "    mlp_sync_info.max_prefix_tokens = cohort_max_prefix(local_batch)\n"),
)


@dataclass(frozen=True)
class MLPSyncBatchMetadata:
    """The attention-DP participants' rows of one MLP sync, in the exchange's order.

    ``forward_modes`` holds ForwardMode integer values. Counts precede graph padding, decode-to-extend conversion and
    speculative coordination. A decode-request count is a pure decode batch's requests or a mixed batch's decoding tail.
    ``None`` counts mean the participant did not report request membership (not zero requests). No tensor or device
    lifetime is held."""

    forward_modes: tuple
    num_tokens: tuple
    num_requests: tuple
    num_decode_requests: tuple
    # each participant's longest cached prefix among its extend requests (0: none, or not an extend batch); empty when
    # the exchange did not carry it
    max_prefix_tokens: tuple = ()

    def __post_init__(self):
        fields = (self.forward_modes, self.num_tokens, self.num_requests, self.num_decode_requests)
        if (not self.forward_modes or any(type(values) is not tuple for values in fields)
                or any(len(values) != len(self.forward_modes) for values in fields)):
            raise ValueError("MLP sync metadata requires aligned immutable participant rows")
        if any(type(value) is not int or value < 0 for values in fields[:2] for value in values):
            raise ValueError("MLP sync modes and token counts must be nonnegative integers")
        for requests, decode in zip(self.num_requests, self.num_decode_requests):
            if requests is None and decode is None:
                continue
            if type(requests) is not int or type(decode) is not int or not 0 <= decode <= requests:
                raise ValueError("MLP sync request counts must be known together and ordered")
        if self.max_prefix_tokens != () and (
                type(self.max_prefix_tokens) is not tuple or len(self.max_prefix_tokens) != len(self.forward_modes)
                or any(type(v) is not int or v < 0 for v in self.max_prefix_tokens)):
            raise ValueError("MLP sync prefix lengths must be one nonnegative integer per participant")


def cohort_counts(local_batch) -> tuple:
    """(requests, decode requests) of a participant's local batch: none without a batch; a pure decode batch's requests
    are all decode requests, a mixed batch's decoding tail is, any other batch has none."""
    if local_batch is None:
        return 0, 0
    mode = local_batch.forward_mode
    if mode.is_decode():
        decode = local_batch.batch_size()
    elif mode.is_mixed():
        decode = len(local_batch.decoding_reqs or [])
    else:
        decode = 0
    return local_batch.batch_size(), decode


def cohort_max_prefix(local_batch) -> int:
    """The longest KV history a participant's step reads: its extend requests' cached prefix (SGLang's
    ``prefix_lens``), or a decode batch's longest sequence plus the token it writes; else 0."""
    if local_batch is None:
        return 0
    mode = local_batch.forward_mode
    if mode.is_decode():
        lens = getattr(local_batch, "seq_lens_cpu", None)
        if lens is None:
            lens = getattr(local_batch, "seq_lens", None)
        if lens is None or len(lens) == 0:
            return 0
        return int(max(int(v) for v in lens.tolist())) + 1
    if not (mode.is_extend() or mode.is_mixed()):
        return 0
    return int(max(local_batch.prefix_lens or (), default=0))


def snapshot(tp0_info_cpu) -> MLPSyncBatchMetadata:
    """The snapshot of an exchange's CPU matrix (one row per participant, the three cohort columns last)."""
    rows = tp0_info_cpu.tolist()
    return MLPSyncBatchMetadata(
        forward_modes=tuple(int(row[_MODE_COLUMN]) for row in rows),
        num_tokens=tuple(int(row[_TOKENS_COLUMN]) for row in rows),
        num_requests=tuple(None if row[-3] < 0 else int(row[-3]) for row in rows),
        num_decode_requests=tuple(None if row[-2] < 0 else int(row[-2]) for row in rows),
        max_prefix_tokens=tuple(max(0, int(row[-1])) for row in rows),
    )


def _counts(info) -> list:
    requests = getattr(info, "num_requests", None)
    decode = getattr(info, "num_decode_requests", None)
    return [-1 if requests is None else requests, -1 if decode is None else decode,
            int(getattr(info, "max_prefix_tokens", 0) or 0)]


def local_tensor(original_fn, self, device, dtype=None, **kwargs):
    """AROUND MLPSyncBatchInfo._get_local_tensor: upstream's row, then the request and decode-request counts and the
    longest cached prefix."""
    import torch

    if dtype is not None:
        kwargs["dtype"] = dtype
    row = original_fn(self, device, **kwargs)
    if not enabled():
        return row
    return torch.cat([row, torch.tensor(_counts(self), device=row.device, dtype=row.dtype)])


def fallback_tensor(original_fn, self, device, dtype=None, **kwargs):
    """AROUND MLPSyncBatchInfo._get_fallback_tensor: upstream's inactive row, then zero counts and prefix."""
    import torch

    if dtype is not None:
        kwargs["dtype"] = dtype
    row = original_fn(self, device, **kwargs)
    if not enabled():
        return row
    return torch.cat([row, torch.zeros(3, device=row.device, dtype=row.dtype)])


def finalize_local(original_fn, self, *args, **kwargs):
    """AROUND MLPSyncBatchInfo.finalize_local: upstream's, then the snapshot of the one participant's row."""
    result = original_fn(self, *args, **kwargs)
    if enabled():
        self.cohort_snapshot = snapshot(self.tp0_info_cpu)
    return result


def all_gather(original_fn, self, *args, **kwargs):
    """AROUND MLPSyncBatchInfo.all_gather: upstream's exchange, then the snapshot of the gathered rows."""
    result = original_fn(self, *args, **kwargs)
    if enabled():
        self.cohort_snapshot = snapshot(self.tp0_info_cpu)
    return result


def update_gather_batch(original_fn, batch, mlp_sync_info, *args, **kwargs):
    """AROUND _update_gather_batch: the batch also takes the exchange's snapshot (None when it gathered nothing)."""
    if enabled():
        setattr(batch, ATTR, getattr(mlp_sync_info, "cohort_snapshot", None))
    return original_fn(batch, mlp_sync_info, *args, **kwargs)


def copy(original_fn, self, *args, **kwargs):
    """AROUND ScheduleBatch.copy: the copy keeps the batch's snapshot."""
    result = original_fn(self, *args, **kwargs)
    if enabled():
        setattr(result, ATTR, getattr(self, ATTR, None))
    return result


def init_new(original_fn, cls, batch, *args, **kwargs):
    """AROUND ForwardBatch.init_new: the forward batch carries its schedule batch's snapshot."""
    result = original_fn(cls, batch, *args, **kwargs)
    if enabled():
        setattr(result, ATTR, getattr(batch, ATTR, None))
    return result


def prepare_mlp_sync_batch_raw():
    """prepare_mlp_sync_batch_raw, upstream's with the local batch's request counts on its MLPSyncBatchInfo."""
    import importlib

    from sglang_neuron.sglang_patches import source_guard

    return source_guard.with_substitutions(importlib.import_module(_DP_ATTN).prepare_mlp_sync_batch_raw,
                                           SUBSTITUTIONS)


def raw(*args, **kwargs):
    """REPLACE prepare_mlp_sync_batch_raw: upstream's own function while the snapshot is off, else upstream's rebuilt
    with the local batch's request counts. A REPLACE, so the plugin's other hooks on the exchange (the DP identity,
    the step timing) keep wrapping it."""
    return (_REBUILT["raw"] if enabled() else _REBUILT["upstream"])(*args, **kwargs)


_REBUILT: dict = {}


def register(registry, around, replace) -> None:
    import importlib

    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    # a batch the exchange has not reached reads None
    for module, cls in ((_SCHEDULE_BATCH, "ScheduleBatch"), (_FORWARD_BATCH, "ForwardBatch")):
        owner = getattr(importlib.import_module(module), cls)
        if ATTR in owner.__dict__ and owner.__dict__[ATTR] is not None:
            raise RuntimeError(f"{module}.{cls}.{ATTR} is already defined upstream; {__name__} would shadow it")
        setattr(owner, ATTR, None)
    # rebuilt now so a source the substitution no longer fits refuses the launch, not the first enabled step
    _REBUILT["upstream"] = importlib.import_module(_DP_ATTN).prepare_mlp_sync_batch_raw
    _REBUILT["raw"] = prepare_mlp_sync_batch_raw()
    registry.register(RAW, raw, replace)
    for target, hook in ((LOCAL_TENSOR, local_tensor), (FALLBACK_TENSOR, fallback_tensor),
                         (FINALIZE_LOCAL, finalize_local), (ALL_GATHER, all_gather),
                         (UPDATE_GATHER, update_gather_batch), (COPY, copy), (INIT_NEW, init_new)):
        registry.register(target, hook, around)
