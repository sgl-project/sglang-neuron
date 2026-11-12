# SPDX-License-Identifier: Apache-2.0
"""A PD decode whose pools place a request's state and drafter rows by its state slot announces their rules in its
registration (the Kimi-K3 decode, models/kimi_k3/decode/sgl_pools.py):

* ``DecodePreallocQueue._init_kv_manager`` (AROUND, outside dcp_page_transfer's): the manager's args take the draft
  pool's ``neuron_draft_rows`` in place of the platform's form (dcp_stripes.DECODE_DRAFT_ROWS), which the DCP frame
  then announces, and the bytes of the state pool's ``registration_frame()``.
* ``NixlKVManager.build_receiver_registration_message`` (AROUND, outside dcp_page_transfer's): that frame trails the
  registration, after the DCP frame. A prefill reads it by its own prefix; upstream's parser and the DCP frame's
  reader ignore it.

A pool without either attribute changes nothing. Registered on a decode of a model-declared decode layout, after
dcp_page_transfer.register_decode, whose source pins cover both targets."""

from __future__ import annotations

_DECODE = "sglang.srt.disaggregation.decode"
_NIXL = "sglang.srt.disaggregation.nixl.conn"


def init_kv_manager(original_fn, self, *args, **kwargs):
    kv_manager = original_fn(self, *args, **kwargs)
    kv = kv_manager.kv_args
    rows = getattr(getattr(self, "draft_token_to_kv_pool", None), "neuron_draft_rows", None)
    if rows is not None:
        kv.neuron_draft_rows = rows
    frame = getattr(getattr(self.req_to_token_pool, "mamba_pool", None), "registration_frame", None)
    kv.neuron_page_frame = frame() if frame is not None else None
    return kv_manager


def build_receiver_registration_message(original_fn, self, *args, **kwargs):
    message = list(original_fn(self, *args, **kwargs))
    frame = getattr(self.kv_args, "neuron_page_frame", None)
    return message + [frame] if frame else message


def register(registry, around) -> None:
    registry.register(f"{_DECODE}.DecodePreallocQueue._init_kv_manager", init_kv_manager, around)
    registry.register(f"{_NIXL}.NixlKVManager.build_receiver_registration_message", build_receiver_registration_message,
                      around)
