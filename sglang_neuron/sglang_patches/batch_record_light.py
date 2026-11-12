# SPDX-License-Identifier: Apache-2.0
"""Cheaper per-step batch snapshots and copies for the scheduler's overlap loops.

``Scheduler.record_batch_in_overlap`` keeps each launched batch alive for two iterations together with a ``getattr``
snapshot of every ``ScheduleBatch`` field, so tensors a later step rebinds outlive the forward that read them. A copy of
the batch's instance ``__dict__`` holds the same objects in one call. The record the slot held before is retired as
upstream retires it: handed to the scheduler's deferred overlap release where it has one (which drops its last
references off the scheduler thread), else dropped.

Upstream's ``ScheduleBatch.copy`` builds the new batch through the dataclass ``__init__``. Setting the fields it passes
on a bare instance gives the same values, because every other field falls back to an immutable class default and the
class has no default factories or ``__post_init__``. Registration checks all of this, with the field list read from
upstream's source.

Both targets are md5-pinned against upstream SGLang 43881a7.

Upstream: SGLang can take this host-time cut for every platform. It keeps the same objects and field values, with one
dict copy instead of per-field Python loops."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import operator
import textwrap

_SCHEDULER = "sglang.srt.managers.scheduler"
_BATCH = "sglang.srt.managers.schedule_batch"
RECORD = (_SCHEDULER, "Scheduler.record_batch_in_overlap")
COPY = (_BATCH, "ScheduleBatch.copy")
_TARGETS_MD5 = {
    RECORD: "f05d6533e06c099e67eda7ef859adc3e",
    COPY: "214b0c5e6aac2bd19ee2940e8ebf9a9b",
}
_IMMUTABLE_DEFAULTS = (type(None), bool, int, float, str)


def record_batch_in_overlap(self, batch) -> None:
    """Scheduler.record_batch_in_overlap: the batch and a copy of its instance dictionary in the next ring slot; the
    record it replaces goes to the deferred overlap release where the scheduler has one, as upstream's."""
    self.batch_record_ct = (self.batch_record_ct + 1) % 2
    retired = self.batch_record_buf[self.batch_record_ct]
    self.batch_record_buf[self.batch_record_ct] = [batch, batch.__dict__.copy()]
    if retired is not None and self.deferred_overlap_release is not None:
        self.deferred_overlap_release.defer(retired)


def copied_fields(copy_fn) -> tuple[str, ...]:
    """The fields upstream's ``ScheduleBatch.copy`` passes as ``field=self.field``, besides ``reqs=self.reqs[:]``.
    Any other argument form is refused."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(copy_fn)))
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ScheduleBatch")
    fields, reqs = [], None
    for keyword in call.keywords:
        value = ast.unparse(keyword.value)
        if keyword.arg == "reqs":
            reqs = value
        elif value == f"self.{keyword.arg}":
            fields.append(keyword.arg)
        else:
            raise RuntimeError(f"{__name__}: ScheduleBatch.copy passes {keyword.arg}={value}, not self.{keyword.arg}")
    if reqs != "self.reqs[:]" or call.args:
        raise RuntimeError(f"{__name__}: ScheduleBatch.copy builds reqs as {reqs!r} (expected self.reqs[:])")
    return tuple(fields)


def make_copy(batch_cls, fields: tuple[str, ...]):
    """ScheduleBatch.copy: a new instance with ``fields`` and the sliced request list set, every other field its
    class default. Refuses a class whose other fields have a default factory, no default, or a mutable default."""
    copied = set(fields) | {"reqs"}
    for f in dataclasses.fields(batch_cls):
        if f.name in copied:
            continue
        if f.default is dataclasses.MISSING or not isinstance(f.default, _IMMUTABLE_DEFAULTS):
            raise RuntimeError(f"{__name__}: ScheduleBatch.{f.name} has no immutable default "
                               f"({f.default!r}, factory {f.default_factory!r}); its copy would differ from upstream's")
        if batch_cls.__dict__.get(f.name, dataclasses.MISSING) is not f.default:
            raise RuntimeError(f"{__name__}: ScheduleBatch.{f.name}'s class attribute is not its default")
    if hasattr(batch_cls, "__post_init__"):
        raise RuntimeError(f"{__name__}: ScheduleBatch has a __post_init__, which its copy would skip")
    values = operator.attrgetter(*fields)
    new = object.__new__

    def copy(self):
        batch = new(batch_cls)
        state = batch.__dict__
        state.update(zip(fields, values(self)))
        state["reqs"] = self.reqs[:]
        return batch

    copy.__qualname__ = f"{batch_cls.__qualname__}.copy"
    copy.__doc__ = make_copy.__doc__
    return copy


def register(registry, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(".".join(RECORD), record_batch_in_overlap, replace)
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    copy = make_copy(ScheduleBatch, copied_fields(source_guard.resolve(*COPY)))
    registry.register(".".join(COPY), copy, replace)
