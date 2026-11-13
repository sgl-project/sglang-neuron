# SPDX-License-Identifier: Apache-2.0
"""PD KV transfer over Mooncake's EFA transport (``--disaggregation-transfer-backend mooncake``).

SGLang's ``MooncakeKVManager`` registers the buffers the KV pools publish (``get_contiguous_buf_infos``: Neuron HBM
device addresses, ``backend.device_io.device_address``) with Mooncake's ``TransferEngine``. Mooncake's EFA transport
(mooncake-transfer-engine-non-cuda 0.3.14 or later) recognizes Neuron HBM by its ``/dev/neuron<N>`` mapping and
registers it with libfabric as ``FI_HMEM_NEURON``, so the prefill writes each page straight from its HBM into the
decode's HBM, as NIXL's LIBFABRIC backend does. The rank runtime maps HBM for that already (``NEURON_RT_MAP_HBM=1``,
``model_executor.runtime_init``), and the image's ``LD_LIBRARY_PATH`` puts the Neuron SDK's and EFA's libraries first,
which libfabric needs to load its Neuron HMEM support.

The plugin's other PD patches are written against ``NixlKVManager`` (the head-major slice transfer, DCP, PP, the
token context-parallel scatter, the kernel-cache hand-offs and the recurrent-state components). None of them applies
to ``MooncakeKVManager``, so a Mooncake launch is served only where upstream's own transfer is correct for the Neuron
pools: a model of ``SERVED_PACKAGES`` without PP, DCP, attention CP or a declared decode layout, and a prefill and a
decode of the same attention TP. The last is checked per peer (``refuse_head_slice``), since a prefill learns its
decode's attention TP only when the decode registers."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

BACKEND = "mooncake"
# Mooncake's transport (SGLang's MOONCAKE_PROTOCOL, default "rdma"): EFA is the only one that moves Neuron HBM. TCP
# would copy from the device addresses on the host, and Trainium hosts have no other RDMA NIC.
PROTOCOL = "efa"
# The model packages whose PD pools upstream's Mooncake transfer moves as they are: one MHA pool of head-major pages,
# K entries then V entries per layer, paired positionally between peers of the same attention TP
SERVED_PACKAGES = (
    "sglang_neuron.models.qwen3",
    "sglang_neuron.models.qwen3_moe",
    "sglang_neuron.models.glm4",
)
# libfabric loads its Neuron HMEM support from the Neuron SDK's libraries
NEURON_LIB_DIR = "/opt/aws/neuron/lib"

_SEND_SLICE = "MooncakeKVManager.send_kvcache_slice"
_CONN = "sglang.srt.disaggregation.mooncake.conn"
# (module, qualified name): md5 of the source the guard wraps
_TARGETS_MD5 = {
    (_CONN, _SEND_SLICE): "91f19f7de0490210cf66bad09a2d0102",
    # where the transfer worker picks the slice path and fails the room on a nonzero return
    (_CONN, "MooncakeKVManager.transfer_worker"): "0b7a713e13d1037299f1ba9443cd506f",
}


def launch_problems(server_args, policy, spec, environ=os.environ) -> list:
    """Every reason a Mooncake prefill or decode launch is not served, in the launching process."""
    problems = []
    package = getattr(spec, "package", "") if spec is not None else ""
    if package not in SERVED_PACKAGES:
        problems.append(f"the model ({package or 'unknown'}) moves its KV with NIXL only; Mooncake serves "
                        + ", ".join(p.rsplit(".", 1)[-1] for p in SERVED_PACKAGES))
    if policy.pp_size > 1:
        problems.append(f"--pp-size {policy.pp_size}: a pipelined server's stage transfers are NIXL's")
    if policy.dcp_size > 1:
        problems.append(f"--dcp-size {policy.dcp_size}: the DCP page transfer is NIXL's")
    if int(getattr(server_args, "attn_cp_size", 1) or 1) > 1:
        problems.append(f"--attn-cp-size {server_args.attn_cp_size}: the context-parallel KV scatter is NIXL's")
    if policy.decode_layout:
        problems.append(f"the declared decode layout {policy.decode_layout} hands its KV off over NIXL")
    if policy.giga_decode:
        problems.append("the giga-kernel decode keeps its KV in its kernel's caches, handed off over NIXL")
    if spec is not None and (spec.kv_handoff or spec.megakernel is not None):
        problems.append("the model's decode keeps its KV in its kernel's caches, handed off over NIXL")
    if environ.get("SGLANG_DISAGG_STAGING_BUFFER", "").strip().lower() in ("1", "true"):
        problems.append("SGLANG_DISAGG_STAGING_BUFFER: the staging buffer is a CUDA buffer; launch without it")
    protocol = environ.get("MOONCAKE_PROTOCOL")
    if protocol is not None and protocol.strip() != PROTOCOL:
        problems.append(f"MOONCAKE_PROTOCOL={protocol}: Neuron HBM moves over Mooncake's EFA transport only "
                        f"(MOONCAKE_PROTOCOL={PROTOCOL})")
    if environ.get("MC_FORCE_TCP") == "1":
        problems.append("MC_FORCE_TCP=1: Mooncake's TCP transport cannot read Neuron HBM")
    if NEURON_LIB_DIR not in environ.get("LD_LIBRARY_PATH", "").split(":"):
        problems.append(f"LD_LIBRARY_PATH has no {NEURON_LIB_DIR}: libfabric then refuses to register Neuron HBM "
                        "(fi_mr_regattr -> ENOSYS)")
    try:
        import mooncake.engine  # noqa: F401
    except ImportError:
        problems.append("mooncake is not installed: pip install mooncake-transfer-engine-non-cuda>=0.3.14")
    return problems


def apply_launch_settings(server_args, set_launch_setting) -> None:
    """Refuse a Mooncake launch that is not served, then fix the transport every scheduler's engine starts with."""
    from sglang_neuron import serving_policy
    from sglang_neuron.models import catalog

    problems = launch_problems(server_args, serving_policy.current(), catalog.served_spec())
    if problems:
        raise ValueError("--disaggregation-transfer-backend mooncake: " + "; ".join(problems))
    set_launch_setting("MOONCAKE_PROTOCOL", PROTOCOL, "Neuron HBM moves over Mooncake's EFA transport only")


def refuse_head_slice(original_fn, self, mooncake_session_id, prefill_kv_indices, dst_kv_ptrs, dst_kv_indices,
                      dst_tp_rank, dst_attn_tp_size, dst_kv_item_len, executor, dst_layer_ids=None):
    """AROUND MooncakeKVManager.send_kvcache_slice: refuse the transfer to a decode of another attention TP.

    Upstream's slice path cuts each page into token rows of head slices, as for token-major pages; a Neuron page is
    head-major (``[heads, page, dim]``), so the slices would land on the wrong bytes. A nonzero return fails the room,
    and the transfer worker marks the session failed."""
    logger.error("[neuron] Mooncake: prefill attention TP %s cannot send to decode attention TP %s (session %s): the "
                 "Neuron KV page is head-major and upstream's Mooncake head-slice transfer is token-major; launch the "
                 "decode at the prefill's attention TP, or use --disaggregation-transfer-backend nixl",
                 self.attn_tp_size, dst_attn_tp_size, mooncake_session_id)
    return 1


def register_hooks(registry, around) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(f"{_CONN}.{_SEND_SLICE}", refuse_head_slice, around)
