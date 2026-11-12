# SPDX-License-Identifier: Apache-2.0
"""What the Kimi-K3 decode serves, as its catalog entry declares it (catalog.ModelSpec.speculative_algorithms and
decode_layouts). Imports only models.capability, since the catalog imports it in the launching process.

The decode runs one server over two trn3 nodes: 256 ranks at LNC1 (each node's 128 physical NeuronCores), eight
DP-attention groups of attention TP 32 (four per node), the experts over all 256 ranks, and the MLA KV in DCP groups of
32, one per attention-TP group. Each step runs DSPARK speculative decoding on the model's own speculative worker
(model_executor.spec_worker), a draft program and a verify program."""

from __future__ import annotations

from sglang_neuron.models.capability import DecodeLaunchLayout

SCRATCHPAD_PAGE_MB = 512
"""The HBM scratchpad page of the decode programs: neuronx-cc's default ``--hbm-scratchpad-page-size``, which the
programs compile with (runtime.SERVED_COMPILER_ARGS names no page size), and the runtime's default
``NEURON_SCRATCHPAD_PAGE_SIZE``. The compiler and the runtime must agree on it."""

RUNTIME_ENV = {
    "NEURON_LOGICAL_NC_CONFIG": "1",
    "NEURON_RT_MAP_HBM": "1",
    "NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS": "32",
    "NEURON_RT_IO_RING_CACHE_SIZE": "16",
    "NEURON_RT_DISABLE_EXECUTION_BARRIER": "1",
    "NEURON_RT_ENABLE_HW_EXECUTION_BARRIER": "1",
    "NEURON_RT_ENABLE_DGE_NOTIFICATIONS": "0",
    "NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION": "0",
    "NEURON_RT_INSTR_FETCH_ON_H2D": "0",
    "NEURON_SCRATCHPAD_PAGE_SIZE": str(SCRATCHPAD_PAGE_MB),
    "NEURON_LIBTORCH_SPMD_STRICT": "1",
    "NKI_USE_TARGET_DEFAULT_LNC": "1",
    "NEURON_ENFORCE_NKI_LNC_CONSISTENCY": "1",
}
"""The runtime settings of one decode rank: the settings that the decode's two programs are validated with. The device
execution barrier is the only cross-rank fence between a rank's verify and draft executions (the host fences nothing),
and the scratchpad page is ``SCRATCHPAD_PAGE_MB`` (at LNC1 the two logical cores of a physical core share one HBM
region, so a larger page takes HBM the pools need). Every rank starts its runtime with them
(runtime_init.runtime_defaults through the layout's ``runtime_env``)."""

TWO_NODE_DP8 = DecodeLaunchLayout(name="Kimi-K3 2-node decode", nnodes=2, tp_size=256, dp_size=8, ep_size=256,
                                  dcp_size=32, logical_nc=1, runtime_env=tuple(sorted(RUNTIME_ENV.items())))

SPECULATIVE_ALGORITHMS = ("DSPARK",)
DECODE_LAYOUTS = (TWO_NODE_DP8,)
