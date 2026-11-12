# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2 decode points the plugin serves: each point's GK geometry and the kernel settings it adds to the GK's
static recipe, one point per launch, named by ``SGLANG_NEURON_GLM52_POINT`` on BOTH PD roles (the prefill writes the
KV formats the decode reads, so the two must name one point).

The points run the decode giga kernel under sglang_neuron/models/glm52/kernels:

| point         | context   | slots | decode sharding                                       | indexer keys | indexer         |
|---------------|-----------|-------|-------------------------------------------------------|--------------|-----------------|
| ``128k``      | 131,072   | 32    | cp 4, t_shard 2, o_proj 16                            | bf16         | per CP group    |
| ``1m``        | 1,048,576 | 32    | cp 4, t_shard 2, o/q-proj 64, latent 8, dense 32,     | fp8          | chunk-dealt     |
|               |           |       | shared 16, router over TP, sparse MoE slots           |              | pools of 8      |

``1m`` is the decode kernel's 32-slot 1M point. It takes no ``paged_indexer``, a runner-side setting and not a
kernel input: the decode kernel reads its indexer cache in position order. Its ``rope_split`` is read only where a graph
keeps a device RoPE table: the MTP verify graph and its draft keep the two-level tables (mtp.graph), while the
decode without MTP builds each step's rows on the host. Its kernel settings are the ones the point needs for HBM at
32 x 1M (the kernel's private scratch, the hardware-DGE score transpose), and it takes the schedule set it was
measured with (SERVE_1M), which the static recipe does not take as a whole: a setting enters the recipe once
measured on the released toolchain."""

from __future__ import annotations

import dataclasses
import os

POINT_ENV = "SGLANG_NEURON_GLM52_POINT"
DEFAULT_POINT = "128k"

#: The decode step's GK split at both points: the 78 layers as three kernel calls, [0, 26), [26, 50) and [50, 78),
#: with a TP all-reduce of zeros between two calls (decode.model.Glm52GkDecodeModel.forward). The runtime gives each
#: function of a program (neuronx-cc starts one only at a torch-level collective) its own block of collective control
#: state: 8 bytes a collective in a 4 KiB on-chip SRAM, and 8 bytes a DMA step (about five a collective) in the
#: 32 KiB TPB instruction memory. A block that does not fit stays in HBM, which adds a delay from trigger to
#: start for every collective and for every collective-core phase. The 1M step as one block has 1,142
#: collectives and 46 KiB of DMA steps; cut at layers 26 and 50 (full indexer layers, so no top-k selection crosses a
#: call), its three blocks have 373, 354 and 415 collectives, inside both memories.
GK_SPLIT = (26, 50)


#: Why each GK kernel setting is a launch setting: the kernel reads it when it is imported and traced.
GK_BUILD_REASON = ("the GLM GK decode's accuracy and performance are measured on the kernel built with this value, "
                   "which it reads when it is imported and traced")


@dataclasses.dataclass(frozen=True)
class Point:
    """One served decode point.

    Attributes:
        name: The value of ``SGLANG_NEURON_GLM52_POINT``.
        knobs: The ``Glm52GkGeometry`` fields that leave the default (the 128k point).
        pool_deal: The decode's pooled indexer (``"chunk"``: whole top-k chunks over pools of as many CP groups as a
            rank holds chunks, ``Glm52GkGeometry.idx_pool_chunk``), or None for the per-group indexer.
        gk_settings: ``(name, value)`` kernel settings of the point's GK build; what every point shares is the
            kernel's code.
        scratchpad_page_mb: The HBM scratchpad page the decode's graphs compile with and the runtime bills in
            (catalog ``scratchpad_page_mb``, NEURON_SCRATCHPAD_PAGE_SIZE). A NEFF's scratchpad takes whole pages, so a
            page well above its extent is HBM no tensor can use.
        store: The name of the weight store the point is measured on, when it is not the point's own build: a latent
            split's side store (tools/glm52/build_latent_store.py). No launch reads it: the launch serves the directory
            ``SGLANG_NEURON_GLM52_WEIGHT_STORE`` names, whatever its name, and the loader checks that store's
            ``geometry.json`` against the point (weight_loaders_mxfp4.check_store_geometry).
            Empty: a store built for the point.
        runtime_settings: ``(name, value, reason)`` Neuron runtime settings the point's decode ranks start with, set
            for the launch beside its kernel settings (catalog ``giga_decode_launch_settings``), so the runtime's
            defaults (runtime_init.runtime_defaults) do not apply to them and a launch that sets another value is
            refused with the reason.
        gk_split: The stack layers where the decode step starts a new GK call (``GK_SPLIT``), each a full indexer
            layer; empty runs the stack as one call. The GK module reads it (decode.model.Glm52GkDecodeModel), and it
            reaches the compiled program as the calls' layer windows.
    """

    name: str
    knobs: dict
    pool_deal: str | None = None
    gk_settings: tuple = ()
    scratchpad_page_mb: int = 512
    store: str = ""
    runtime_settings: tuple = ()
    gk_split: tuple = GK_SPLIT

    def __post_init__(self):
        # the pooled indexer deals whole top-k chunks (Glm52GkGeometry.idx_pool), the one pool the decode serves
        if self.pool_deal not in (None, "chunk"):
            raise ValueError(f"the {self.name} decode point's indexer pool deal {self.pool_deal!r}: the decode runtime "
                             "builds the chunk-dealt pool's tables only (pool_deal 'chunk', or None for the per-group "
                             "indexer)")

    def launch_settings(self) -> tuple:
        """The GK decode's launch settings at this point, as ``(name, value, reason)``: its kernel settings, each with
        the kernel build's reason, then its Neuron runtime settings."""
        return (tuple((name, value, GK_BUILD_REASON) for name, value in self.gk_settings)
                + tuple(self.runtime_settings))

    def geometry(self):
        """The 128-rank geometry of the point without the indexer pool: what the weight store, the KV formats and the
        PD hand-off's byte layout derive from (the pool changes none of them)."""
        from sglang_neuron.models.glm52.layout.decode.gk_geometry import Glm52GkGeometry

        return Glm52GkGeometry(**self.knobs)

    def decode_geometry(self):
        """The geometry the decode serves: ``geometry`` plus the point's indexer pool, pools of ``idx_topk_chunks`` CP
        groups (split ``slots / chunks``), each scoring every spec row of a sequence (`Glm52GkGeometry.idx_pool_tokens`):
        a decode without the MTP draft scores its one real row and seven zero rows, and the same geometry serves the
        draft's eight."""
        base = self.geometry()
        if self.pool_deal is None:
            return base
        from sglang_neuron.models.glm52.layout.decode.gk_geometry import Glm52GkGeometry

        slots = base.num_seqs * base.dp_size
        return Glm52GkGeometry(**self.knobs, idx_pool=True, idx_pool_split=max(1, slots // base.idx_topk_chunks))


#: The serve set of the decode kernel's 32-slot 1M point, the set the point was measured and served with (its GK split is
#: the point's ``gk_split``): the chunk-dealt indexer pool traces only with its indexer
#: knobs (the packed head sum, the hardware-DGE key load, the dynamic chunk loop and its peel, the row-tiled score, the
#: direct candidates, the key prefetch), and the rest is the schedule it was measured with. At 128k the set measured
#: slower, so only this point takes it.
SERVE_1M = (
    ("GLM52K_ROPE_TABLES_ONCE", "1"), ("GLM52K_Q_BLOCK_AG_FIRST", "1"), ("GLM52K_Q_NORM_INDEXER_ONLY", "1"),
    ("GLM52K_IDX_HEADSUM_PACKED", "1"), ("GLM52K_IDX_DYNAMIC_LOOP", "1"),
    ("GLM52K_IDX_CHUNK_LOOP", "1"), ("GLM52K_IDX_CHUNK_PEEL", "1"),
    ("GLM52K_IDX_KEY_HWDGE", "1"), ("GLM52K_Q_BLOCK_SPLIT_EXCHANGE", "1"), ("GLM52K_CP_COMBINE_TWO_STAGE", "1"),
    ("GLM52K_CC_PIN_ATTN", "0"), ("GLM52K_MOE_DOWN_PSUM_SCALE", "last"),
    ("GLM52K_MOE_DOWN_LAST_SPLIT", "4"), ("GLM52K_MOE_EP_AG_EARLY", "1"), ("GLM52K_MOE_AFF_SLICE_LATE", "1"),
    ("GLM52K_ATTN_Q_HOOK", "1"), ("GLM52K_PRIOR_INDEX_AFFINE_TABLE", "-1"),
    ("GLM52K_ATTN_SIB_ONCE", "1"), ("GLM52K_ATTN_EP_SYNC_FIRST", "1"), ("GLM52K_IDX_SCORE_ROWTILE", "1"),
    ("GLM52K_IDX_SELECT_VEC", "1"), ("GLM52K_IDX_CAND_DIRECT", "1"), ("GLM52K_IDX_HEADSUM_LAG", "1"),
    ("GLM52K_IDX_KEY_PREFETCH", "1"), ("GLM52K_IDX_RELU_BY_GROUP", "1"), ("GLM52K_IDX_SELECT_VEC_FP", "1"),
    ("GLM52K_IDX_CAND_HWDGE", "1"), ("GLM52K_IDX_SCORE_KEYPAIR", "1"), ("GLM52K_IDX_SELECT_2Q", "1"),
    ("GLM52K_LAT_ROUTER_AG_FIRST", "1"), ("GLM52K_LAT_QBLOCK_SBUF", "1"),
    ("GLM52K_LAT_DOWNPROJ_CHUNK_LOAD", "6"), ("GLM52K_LAT_DOWNPROJ_AG_SRC", "1"),
    ("GLM52K_LAT_OPROJ_STORE_EACH", "1"), ("GLM52K_LAT_OPROJ_EVICT_ALT", "1"), ("GLM52K_LAT_COMBINE_LSE_FIRST", "1"),
    ("GLM52K_LAT_COMBINE_TS", "1"), ("GLM52K_LAT_QSTAGE_DGE", "sync"), ("GLM52K_PRIOR_INDEX_SNAKE_DIRECT", "1"),
    ("GLM52K_PRIOR_INDEX_KM_SBUF", "1"),
    ("GLM52K_SMALL_LOADS_HWDGE", "sp"),
)


#: The 1M point's attention collective operands pinned (kernels/decode/utils/cc_pin.py ``pin_attn``, each kept for a window of
#: two layers at the point's idx_cache_keys >= 262144 a rank), where the serve set unpins them
#: (``GLM52K_CC_PIN_ATTN=0`` in SERVE_1M). Unpinned, the allocator can give a later tensor the offsets of an
#: attention collective's buffer, and a peer's write into the old occupant then lands in the new one (cc_pin's
#: peer-write class). On the released toolchain, a zero-weight 32 x 1M decode graph with more tensors than the served
#: one corrupts its first decode step in every unpinned run (NaN candidate values in the tail, a token table
#: step_outputs cannot produce), and pinned it serves 32 of 32 requests clean. The served graph shows no corruption
#: unpinned in the same checks, but which buffer a reused offset lands on depends on the whole graph's allocation, so
#: the 1M point keeps the pin rather than rely on its current layout.
PIN_ATTENTION = (("GLM52K_CC_PIN_ATTN", "1"),)


def with_settings(base: tuple, override: tuple) -> tuple:
    """``base`` with each setting of ``override`` in place of the same name's: each name once, in ``base``'s order, and
    an ``override`` name ``base`` lacks appended."""
    values = dict(override)
    return tuple((name, values.pop(name, value)) for name, value in base) + tuple(values.items())


POINTS = {
    # GLM52K_MOE_GU_PREFETCH is a point setting, not the static recipe's, so the 1M point can deepen it; 128k keeps
    # depth 1 (depth 2 measured a shorter step but inflated the p99 there).
    "128k": Point("128k", {}, gk_settings=(("GLM52K_MOE_GU_PREFETCH", "1"),)),
    "1m": Point(
        "1m",
        {
            "oproj_tp": 64,
            "qproj_tp": 64,
            "context": 1048576,
            "idx_fp8": True,
            "router_tp": True,
            # The down-projections (wq_a / wkv_a) over 8 consecutive ranks (2 token slices), a latent split of 8:
            # a shorter step at 127k x 32 than with 64 (real KV), 0.23 GB a core free after the verify graph.
            # A store with the token-group shard serves it (sliced at load; ``store`` below).
            "latent_tp": 8,
            "dense_tp": 32,
            "shared_tp": 16,
            "moe_sparse_slots": True,
            "rope_split": True,
        },
        pool_deal="chunk",
        gk_settings=(
            # 32 x 1M fits a core's HBM only with the kernel's per-layer scratch private (measured at this point: off
            # gives a shorter step, but its per-layer buffers then take ~300 MiB a core the warm-up's collectives need)
            ("GLM52K_GLM52_PRIVATE_SCRATCH", "1"),
            ("GLM52K_GLM52_IDX_SCORE_HWDGE", "1"),
            # the two-stage CP combine's context loads on the ACT queue, not split over Sync (where they wait behind
            # the weight stream) and ACT: a shorter step (64k x 32 live, MTP K7, same node)
            ("GLM52K_ATTN_COMBINE_RECV_QUEUE", "act"),
            # the shared expert's rows out of the EP row gather, not a second 16-rank gather of the same rows ahead of
            # it on the collective stream: a shorter step (64k x 32 live, MTP K7, same node)
            ("GLM52K_MOE_SHARED_ROWS_FROM_EP", "1"),
            # a deeper gate/up weight prefetch: the gate/up weight stream is the 1M point's top exposed kernel wait
            # (at latent 8), a shorter step (32 x 500k, MTP K7, same node; delta
            # uniform p0..p100, none of the 128k point's p99 inflation, so only this point takes 2)
            ("GLM52K_MOE_GU_PREFETCH", "2"),
        ) + with_settings(SERVE_1M, PIN_ATTENTION),
        # the 1M decode's page (NEURON_SCRATCHPAD_PAGE_SIZE=128): the 1M NEFFs'
        # scratchpad extents are 102 MiB (no MTP) and 117 MiB (MTP K7), one 128 MiB page, where a 512 MiB page leaves
        # 384 MiB a core unused, which the MTP graph's collectives need at load
        scratchpad_page_mb=128,
        # latent_tp 8 takes the token-group-shard down-projections, linked from the 128k store:
        # tools/glm52/build_latent_store.py <1m store> <128k store> <this> 8
        store="1m_lat8",
        runtime_settings=(
            ("NEURON_RT_IO_RING_CACHE_SIZE", "4",
             "the 1M decode fits a core's HBM with a ring of 4 IO tensor sets a graph, the ring every 1M gate and "
             "final ran with: the plugin's ring of 32 takes about 0.8 GiB more a core, against the 0.23-0.34 GB the "
             "1M point leaves free; 4 stays above the in-flight bound of 2 (backend.device_io.check_io_ring)"),
        ),
    ),
}


def current(environ=None) -> Point:
    """The launch's point (``SGLANG_NEURON_GLM52_POINT``, default ``128k``)."""
    environ = os.environ if environ is None else environ
    name = environ.get(POINT_ENV, DEFAULT_POINT)
    if name not in POINTS:
        raise ValueError(f"{POINT_ENV}={name!r} is not a served GLM-5.2 decode point; serve one of {sorted(POINTS)}")
    return POINTS[name]
