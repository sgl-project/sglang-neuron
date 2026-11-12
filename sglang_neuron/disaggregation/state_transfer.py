# SPDX-License-Identifier: Apache-2.0
"""PD transfer of recurrent-state components: what a prefill and a decode must agree on before the prefill sends a
request's state, and the byte runs one request's state transfer writes.

Upstream SGLang registers a hybrid-linear pool's state as one ``StateType.MAMBA`` component whose entries are the
pool's per-layer tensors, field-major (every conv tensor's layers, then the temporal tensor's layers;
``MambaPool._iter_transfer_state_entries``), each with its global layer id, item length (one slot of one layer) and
sliceable head dimension. At a request's last chunk the prefill pairs its entries with the decode's by layer id (in
order of occurrence, ``build_transfer_entry_pairs``) and, at equal attention TP, WRITEs ``src_item_len`` bytes from
``src_ptr + src_item_len * src_slot`` to ``dst_ptr + src_item_len * dst_slot`` (``NixlKVManager._send_mamba_state``);
at different attention TPs it slices each entry by its head dimension (``_send_mamba_state_slice``).

Upstream checks none of the following at registration, and three of them corrupt state silently instead of failing:

* a paired entry whose decode item length differs from the prefill's (a different state dtype or shape on the decode:
  the WRITE lands with the prefill's stride in the decode's slot grid) - refused;
* a prefill layer the decode registered no entry for (upstream raises at every request's last chunk) - refused;
* different attention TPs with an entry that is not head-sliceable (a kernel layout other than upstream's per-slot
  ``(channels, window)``, which the pool marks with upstream's 0 "replicated" dimension; upstream raises per request)
  - refused (no platform path re-lays such an entry out across attention TPs; Qwen3.5's PD pairs one attention TP on
  both roles);
* a decode that registered fewer state components than the prefill sends (sglang_patches/nixl_state_components).

``state_entries_problem`` states every such refusal in one text; sglang_patches/nixl_state_components raises it when
the decode rank registers (``NixlKVManager._add_remote_peer``), so the pairing fails once, naming both sides, before
any request is admitted. ``slot_runs`` is the equal-TP byte plan upstream's send follows (a test's reference).

KV pairing: the state component travels with the request's last KV chunk (``maybe_send_extra``) and carries the state
after the whole prompt; the decode request becomes runnable only when both have landed. A decode state slot whose
transfer failed or was aborted is held like its KV pages (sglang_patches/pd_decode_kv_quarantine) until no WRITE can
still land in it.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

MAMBA = "mamba"


def _state_name(state_type) -> str:
    return str(getattr(state_type, "value", state_type))


def _upstream_pairs(src_ids, dst_ids, n_src, n_dst, allow_positional_fallback):
    from sglang.srt.disaggregation.utils import build_transfer_entry_pairs

    return build_transfer_entry_pairs(src_ids, dst_ids, n_src, n_dst,
                                      allow_positional_fallback=allow_positional_fallback)


def _at(seq, i):
    return list(seq[i]) if seq is not None and i < len(seq) and seq[i] is not None else []


def state_entries_problem(prefill_args, decode_info, *, prefill_tp: int, pp_size: int = 1,
                          pair: Callable = _upstream_pairs) -> Optional[str]:
    """Why this decode registration cannot take this prefill's recurrent-state entries, or None.

    ``prefill_args``: the prefill's KV args (``state_types``, ``state_item_lens``, ``state_layer_ids``,
    ``state_dim_per_tensor``); ``decode_info``: the decode's ``KVArgsRegisterInfo`` (``dst_state_*``,
    ``decode_tp_size``); ``prefill_tp``: the prefill's state TP size (upstream's ``state_tp_size``)."""
    problems = []
    state_types = list(getattr(prefill_args, "state_types", None) or [])
    for i, st in enumerate(state_types):
        if _state_name(st) != MAMBA:
            continue
        src_lens = _at(getattr(prefill_args, "state_item_lens", None), i)
        src_ids = _at(getattr(prefill_args, "state_layer_ids", None), i)
        src_dims = _at(getattr(prefill_args, "state_dim_per_tensor", None), i)
        dst_lens = _at(getattr(decode_info, "dst_state_item_lens", None), i)
        dst_ids = _at(getattr(decode_info, "dst_state_layer_ids", None), i)
        if not src_lens:
            continue
        if not dst_lens:
            continue  # no decode entries at all: state_components_problem's refusal
        try:
            pairs = pair(src_ids, dst_ids, len(src_lens), len(dst_lens), pp_size == 1)
        except RuntimeError as exc:
            problems.append(f"component {i} ({MAMBA}): {exc}")
            continue
        if int(prefill_tp) == int(decode_info.decode_tp_size):
            bad = [(src_ids[s] if src_ids else s, src_lens[s], dst_lens[d]) for s, d in pairs
                   if int(src_lens[s]) != int(dst_lens[d])]
            if bad:
                shown = ", ".join(f"layer {lid}: {a} != {b} bytes" for lid, a, b in bad[:4])
                problems.append(f"component {i} ({MAMBA}): {len(bad)} paired entries hold a different slot size on "
                                f"the decode ({shown}); the roles store recurrent state in different dtypes or shapes")
        elif 0 in [int(d) for d in src_dims]:
            problems.append(f"component {i} ({MAMBA}): attention TP {prefill_tp} -> {decode_info.decode_tp_size} "
                            "needs every state entry head-sliceable, and this prefill's state pool stores an entry in "
                            "a kernel layout with no head axis upstream can slice; pair a decode of attention TP "
                            f"{prefill_tp}")
    if not problems:
        return None
    return "recurrent state cannot be transferred: " + "; ".join(problems)


def slot_runs(src_ptrs: Sequence[int], src_lens: Sequence[int], src_slot: int, dst_ptrs: Sequence[int],
              dst_slot: int, pairs: Sequence[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """The (src address, dst address, bytes) WRITEs of one request's state at equal TP: one run per paired entry,
    as upstream's ``_send_mamba_state`` builds them. Slot 0 is the padding slot on both roles and never transferred."""
    if src_slot <= 0 or dst_slot <= 0:
        raise ValueError(f"state transfer between slots {src_slot} -> {dst_slot}: slot 0 is padding")
    runs = []
    for s, d in pairs:
        length = int(src_lens[s])
        if length == 0 or not src_ptrs[s] or not dst_ptrs[d]:
            continue
        runs.append((int(src_ptrs[s]) + length * int(src_slot), int(dst_ptrs[d]) + length * int(dst_slot), length))
    return runs
