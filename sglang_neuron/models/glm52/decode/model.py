# SPDX-License-Identifier: Apache-2.0
"""One rank's GLM-5.2 decode giga kernel (GK): the module that holds what the kernel reads, and calls it.

The GK runs a decode step of the whole stack as one SPMD program over the node's 128 ranks, so this module is the
glue around the kernel, not a layer stack. Each rank builds its own: every rank traces the same program over its own
shard of every weight and cache. The module holds, as buffers,

- the stacked weights of every layer (`load_weights`, the rank's shard of a GK weight store);
- the control tensors that every step shares (`STATIC_CONTROL`: the identity block table, the static positions and
  key offsets, the rank's id, and the shared expert's index and unit affinities);
- the KV caches, bound once and written in place by every step (`bind_caches`, `allocate_own_caches`): the indexer's
  key cache and the latent cache, which the 1M point splits over four stacks: the device refuses a single allocation
  of 2**32 bytes or more, and three stacks of the 32 slots' 1M positions would take 4.36 GB each.

`forward` takes one step's tensors (the RoPE rows, the latent and indexer write rows, the residency threshold), the
LM-head tail's inputs and, at a point that pools its indexer, the pool's tables. It runs the stack as the decode
point's split (`points.Point.gk_split`): one kernel call per layer window, the last with the sampling tail. The
embedding of the step's tokens and the tail's weights are the decode graph's (decode/graph.py), which shards both
over the ranks.

The module only decodes: the caches already hold the prompt's keys, written by the PD prefill.

Example:
    >>> model = Glm52GkDecodeModel(points.POINTS["128k"].decode_geometry(), rank_id=0)  # doctest: +SKIP
    >>> model.load_weights("/path/to/store", "/path/to/checkpoint")  # doctest: +SKIP
    >>> model.allocate_own_caches()  # doctest: +SKIP
    >>> hidden = model(**step)  # doctest: +SKIP (step: decode.runtime.rank_step_tensors' tensors for this rank)
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

from sglang_neuron.models.glm52 import points
from sglang_neuron.models.glm52.decode import upstream as gk_upstream
from sglang_neuron.models.glm52.ops import gk_decode as NF
from sglang_neuron.models.glm52.layout.decode.gk_geometry import Glm52GkGeometry

# The checks of the decode kernel are in `decode.kernel_limits`, which imports no kernel, so a host
# without the Neuron SDK can test them.
from sglang_neuron.models.glm52.decode.kernel_limits import DECODE_KERNEL_PACKAGE, check_kernel_copy
from sglang_neuron.models.glm52.layout.decode.gk_kv_cache import (
    Glm52GkCaches,
    allocate_caches,
    idx_key_dtype,
)
from sglang_neuron.models.glm52.layout.decode.weight_loaders_mxfp4 import (
    ALL_FILES,
    load_rank_weights,
)

logger = logging.getLogger(__name__)

# Prefix of the buffers that hold the stacked weights. Buffers, NOT parameters:
# the trace fork's meta-swap binds a parameter stack to the wrong device, and a
# buffer correctly. This follows the GPT-OSS GK model.
GK_STACK_ATTR_PREFIX = "_gk_wstack_"

# Prefix of the buffers that hold the rank's static control tensors.
GK_CTRL_ATTR_PREFIX = "_gk_ctrl_"

# The control tensors that do not change between decode steps.
STATIC_CONTROL = (
    "block_table",
    "positions",
    "key_offset",
    "cp_pos_offset",
    "rank_id",
    "shared_rank_id",
    "shared_affinities",
)

def split_cuts(geo: Glm52GkGeometry, cuts) -> tuple[int, ...]:
    """Return ``cuts`` as the stack layers where a decode step starts a new kernel call, checked against ``geo``.

    Args:
        geo: The sharding point.
        cuts: Increasing layers inside ``(0, num_layers)``, each a full indexer layer, or empty for one call.

    Raises:
        ValueError: If the cuts do not increase inside ``(0, num_layers)``, or a cut is a layer that takes the
            previous layer's top-k selection: a call cannot start there, since no selection crosses a call.

    Example:
        >>> split_cuts(Glm52GkGeometry(), (26, 50))
        (26, 50)
    """
    cuts = tuple(int(cut) for cut in cuts)
    if list(cuts) != sorted(set(cuts)) or (cuts and not (0 < cuts[0] and cuts[-1] < geo.num_layers)):
        raise ValueError(f"GK split {cuts}: the cuts must increase inside (0, {geo.num_layers})")
    shared = [cut for cut in cuts if not geo.is_full(cut)]
    if shared:
        raise ValueError(
            f"GK split {cuts}: layers {shared} take the previous layer's top-k selection, and no selection "
            f"crosses a kernel call; cut at full indexer layers ({', '.join(map(str, geo.full_layers))})"
        )
    return cuts


class Glm52GkDecodeModel(nn.Module):
    """One rank's decode stack: the GK plus the buffers that it reads.

    The module is per-rank, because the GK is an SPMD kernel: every rank traces
    the same graph and reads its own shard of every weight and cache.

    Args:
        geo: The sharding point. It must match the weight store's geometry.
        rank_id: Global rank in ``[0, ranks)``.
        device: Device of the weights, the caches and the step inputs.
        dtype: Residual-stream dtype. The kernel takes bf16.
        split: The layers where a step starts a new kernel call (`split_cuts`), or None for the launch's decode
            point's (`points.Point.gk_split`).

    Attributes:
        geo: The sharding point.
        rank_id: The rank.
        split: The layers where a step starts a new kernel call.
        caches: The two KV caches, once `bind_caches` or `allocate_own_caches`
            has run.

    Raises:
        ValueError: If the split's cuts are not ones the kernel takes (`split_cuts`).
    """

    def __init__(
        self,
        geo: Glm52GkGeometry,
        rank_id: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        split: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        geo.validate()
        check_kernel_copy(geo, DECODE_KERNEL_PACKAGE)
        self.geo = geo
        self.rank_id = rank_id
        self.device_str = str(device)
        self.dtype = dtype
        self.split = split_cuts(geo, points.current().gk_split if split is None else split)
        self.caches: Glm52GkCaches | None = None
        self._meshes = geo.replica_meshes()
        self._scalars = geo.scalar_kwargs()
        self._logged_invoke = False
        self._register_static_control()

    # Buffers
    def _register_static_control(self) -> None:
        """Register the control tensors that every step shares.

        `positions` is deliberately static. It feeds the causal keep test and
        the CP owner test of the sibling keys, and already sits at the compiled
        ``prior_len``, which orders the query after every key that a run can write.

        Every factory call names ``device="cpu"``. A model runner builds the model
        under ``with torch.device("meta")``, and a meta tensor cannot be copied to
        a real device, so an implicit device turns this into "Cannot copy out of
        meta tensor". An explicit device wins over the context.
        """
        geo, rank = self.geo, self.rank_id
        tensors = {
            "block_table": torch.from_numpy(geo.block_table()),
            "positions": torch.from_numpy(geo.positions(rank)),
            "key_offset": torch.from_numpy(geo.key_offset(rank)),
            "cp_pos_offset": torch.from_numpy(geo.cp_pos_offset(rank)),
            "rank_id": torch.tensor([[rank]], dtype=torch.int32, device="cpu"),
            # Zero, not this rank's EP index: the shared expert is expert 0 of a
            # one-column affinity table on every rank.
            "shared_rank_id": torch.tensor([[0]], dtype=torch.int32, device="cpu"),
            # The shared expert's contribution is unweighted, hence all ones.
            "shared_affinities": torch.ones(
                (geo.shared_rows, 1), dtype=torch.bfloat16, device="cpu"
            ),
        }
        for name, tensor in tensors.items():
            self.register_buffer(
                GK_CTRL_ATTR_PREFIX + name, tensor.to(self.device_str), persistent=False
            )

    def load_weights(
        self, store_dir: str, checkpoint_dir: str, device: str | torch.device | None = None
    ) -> None:
        """Load this rank's stacked weights from a GK weight store.

        Args:
            store_dir: Root of a store that `build_rank_store` wrote.
            checkpoint_dir: The checkpoint the server serves; the store must hold a checkpoint of its format.
            device: Where to put the weights. Defaults to the model's device.

        Raises:
            FileNotFoundError: If the store has no ``geometry.json`` or no file for an argument.
            ValueError: If the store was built for a different geometry, or from a checkpoint of another format.
        """
        target = str(device or self.device_str)
        weights = load_rank_weights(store_dir, self.geo, self.rank_id, checkpoint_dir, device=target)
        for name, tensor in weights.as_kernel_kwargs().items():
            self.register_buffer(GK_STACK_ATTR_PREFIX + name, tensor, persistent=False)
        missing = [n for n in ALL_FILES if not hasattr(self, GK_STACK_ATTR_PREFIX + n)]
        if missing:
            raise ValueError(
                f"store {store_dir} gave no tensor for {missing}. Rebuild the "
                "store with build_rank_store."
            )
        logger.info(
            "[glm52] rank %d weights loaded from %s (%d stacks, %d layers)",
            self.rank_id,
            store_dir,
            len(ALL_FILES),
            self.geo.num_layers,
        )

    def bind_caches(self, caches: Glm52GkCaches) -> None:
        """Bind caches that a caller filled, for example from a prefill engine.

        The SAME buffers must serve every decode step: step ``s`` attends to the
        row that step ``s - 1`` wrote.
        """
        # The BLOCK COUNT is the allocator's, not the geometry's: the kernel
        # addresses a block through the table, so a pool that is deeper than one
        # sequence needs is valid. Only the layer count and the row layout are
        # fixed by the geometry.
        expected = (self.geo.num_layers, self.geo.block_size, self.geo.row_bytes)
        shape = tuple(caches.kv_cache_all.shape)
        # A split stack counts its layers over every stack (see gk_kv_cache.latent_stack_sizes).
        got = (caches.num_latent_layers, *shape[3:])
        if got != expected:
            raise ValueError(
                f"latent cache is {shape} and this geometry needs "
                f"[{self.geo.num_layers}, blocks, 1, {self.geo.block_size}, "
                f"{self.geo.row_bytes}]. Allocate it with allocate_caches for this "
                "geometry."
            )
        if caches.kv_cache_all.shape[1] < self.geo.num_seqs_local:
            raise ValueError(
                f"the latent cache has {caches.kv_cache_all.shape[1]} blocks and "
                "cannot hold one block for each sequence."
            )
        # The kernel picks its key path from the indexer cache's dtype alone.
        if caches.idx_k_cache_all.dtype != idx_key_dtype(self.geo):
            raise ValueError(
                f"the indexer cache is {caches.idx_k_cache_all.dtype} and this geometry "
                f"(idx_fp8_keys={self.geo.idx_fp8_keys}) needs {idx_key_dtype(self.geo)}."
            )
        self.caches = caches
        self.register_buffer(
            GK_CTRL_ATTR_PREFIX + "kv_cache_all", caches.kv_cache_all, persistent=False
        )
        self.register_buffer(
            GK_CTRL_ATTR_PREFIX + "idx_k_cache_all",
            caches.idx_k_cache_all,
            persistent=False,
        )
        # The second half of a split latent stack. None keeps the single-stack graph unchanged.
        self.register_buffer(
            GK_CTRL_ATTR_PREFIX + "kv_cache_hi", caches.kv_cache_hi, persistent=False
        )
        self.register_buffer(
            GK_CTRL_ATTR_PREFIX + "kv_cache_hi2", caches.kv_cache_hi2, persistent=False
        )
        self.register_buffer(
            GK_CTRL_ATTR_PREFIX + "kv_cache_hi3", caches.kv_cache_hi3, persistent=False
        )
        # The one greppable line that says this rank will run the GK. It is here
        # and not in `forward`: a compiled forward runs no Python, and a logger
        # call inside the traced region breaks the parallel-trace fork.
        if not self._logged_invoke:
            self._logged_invoke = True
            logger.info(
                "[glm52 GK] decode ready: rank=%d L=%d ranks=%d cp=%d oproj_tp=%d "
                "full_layers=%d blocks=%d",
                self.rank_id,
                self.geo.num_layers,
                self.geo.ranks,
                self.geo.cp_degree,
                self.geo.oproj_tp,
                len(self.geo.full_layers),
                caches.kv_cache_all.shape[1],
            )

    def allocate_own_caches(self, indexer_fill: str = "zeros") -> Glm52GkCaches:
        """Allocate empty caches on the model's device and bind them.

        Returns:
            The caches, so a caller can fill them.
        """
        caches = allocate_caches(
            self.geo, device=self.device_str, indexer_fill=indexer_fill
        )
        self.bind_caches(caches)
        return caches

    # The traced step
    def forward(
        self,
        x_shard: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        idx_cos: torch.Tensor,
        idx_sin: torch.Tensor,
        kv_new_slot_mapping: torch.Tensor,
        idx_new_slot_mapping: torch.Tensor,
        idx_positions: torch.Tensor,
        block_table: torch.Tensor | None = None,
        idx_block_table: torch.Tensor | None = None,
        head: dict[str, torch.Tensor] | None = None,
        idx_pool: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run one decode step of the whole stack.

        Args:
            x_shard: ``[t_shard, hidden]`` entry residual of this rank.
            cos: ``[t_local, rope / 2]`` float32 RoPE cosine at this position.
            sin: ``[t_local, rope / 2]`` float32 RoPE sine.
            idx_cos: ``[spec, rope / 2]`` float32 indexer RoPE cosine.
            idx_sin: ``[spec, rope / 2]`` float32 indexer RoPE sine.
            kv_new_slot_mapping: ``[num_seqs_local, spec]`` int32 latent row.
            idx_new_slot_mapping: ``[spec]`` int32 indexer row.
            idx_positions: ``[spec]`` int32 residency threshold.
            block_table: ``[num_seqs_local, table_width]`` int32 logical to
                physical block map, or None for the rank's static identity
                table. Refused when the kernel is built with
                ``GLM52K_PRIOR_INDEX_AFFINE_TABLE`` (the 1M point), which
                computes the identity table's blocks instead of reading a table.
            idx_block_table: ``[1, idx_blocks + 1]`` int32 logical to physical
                block map of the INDEXER key cache (column 0 a guard, block ``b``
                at column ``b + 1``), or None to use MLA's own row of
                `block_table`. The indexer shares MLA's table and deal, so the two
                must agree column for column. When both tables are None, the decode
                kernel reads the indexer cache in position order, which is what the
                static identity table gives. A caller's own table must allocate
                every live block.
            head: The lm_head tail inputs, or None. Give them when the kernel
                runs the tail (``NF.glm52_gk_decode`` documents the keys).
            idx_pool: The pooled indexer's per-step tables, or None. Required when
                the geometry pools the indexer (`idx_pool_degree`): the chunk-dealt
                pool's (`IdxChunkPoolTables`, `idx_pool.CDP_TABLES`).

        Returns:
            ``[t_shard, hidden]`` post-stack residual of this rank. With `head`,
            a tuple of that residual and the tail's ``[t_shard, 1]`` uint32 tokens.

        Raises:
            RuntimeError: If no cache is bound, or the kernel is unavailable.
            ValueError: If `block_table` is given to a kernel that computes the
                identity table itself (``GLM52K_PRIOR_INDEX_AFFINE_TABLE``).
        """
        # The attention core then resolves block ``j`` of sequence ``s`` to
        # ``s * table_width + j + k`` without reading the table, which is exactly
        # the identity table's entry. Any other table would be ignored and every
        # selected key read from the wrong row, with no error anywhere, so a
        # caller's table is refused here, where it can still be named.
        if block_table is not None and os.environ.get("GLM52K_PRIOR_INDEX_AFFINE_TABLE"):
            raise ValueError(
                "block_table: the decode kernel is built with GLM52K_PRIOR_INDEX_AFFINE_TABLE, which computes the "
                "rank's static identity table instead of reading one; pass block_table=None"
            )
        if self.caches is None:
            raise RuntimeError(
                "no KV cache is bound. Call bind_caches with the prefill engine's "
                "cache, or allocate_own_caches for a self-contained run."
            )
        weights = {
            name: getattr(self, GK_STACK_ATTR_PREFIX + name) for name in ALL_FILES
        }
        control = {
            name: getattr(self, GK_CTRL_ATTR_PREFIX + name) for name in STATIC_CONTROL
        }
        if block_table is not None:
            control["block_table"] = block_table
        # The indexer shares MLA's block table: its keys are dealt the same way, so MLA's row for this
        # rank's (one) sequence serves both caches. A caller may pass that same table explicitly.
        # The decode kernel reads the indexer cache in position order when the caller gives neither table.
        # The static table is the identity (one sequence per rank), so a paged read is a row gather
        # of every key that gives the same rows. That gather cost device time in each full layer.
        if idx_block_table is not None:
            control["idx_block_table"] = idx_block_table
        elif block_table is not None:
            control["idx_block_table"] = control["block_table"][0:1]
        control["kv_cache_all"] = getattr(self, GK_CTRL_ATTR_PREFIX + "kv_cache_all")
        control["idx_k_cache_all"] = getattr(
            self, GK_CTRL_ATTR_PREFIX + "idx_k_cache_all"
        )
        kv_cache_hi = getattr(self, GK_CTRL_ATTR_PREFIX + "kv_cache_hi", None)
        if kv_cache_hi is not None:
            control["kv_cache_hi"] = kv_cache_hi
        kv_cache_hi2 = getattr(self, GK_CTRL_ATTR_PREFIX + "kv_cache_hi2", None)
        if kv_cache_hi2 is not None:
            control["kv_cache_hi2"] = kv_cache_hi2
        kv_cache_hi3 = getattr(self, GK_CTRL_ATTR_PREFIX + "kv_cache_hi3", None)
        if kv_cache_hi3 is not None:
            control["kv_cache_hi3"] = kv_cache_hi3
        control.update(
            {
                "cos": cos,
                "sin": sin,
                "idx_cos": idx_cos,
                "idx_sin": idx_sin,
                "kv_new_slot_mapping": kv_new_slot_mapping,
                "idx_new_slot_mapping": idx_new_slot_mapping,
                "idx_positions": idx_positions,
            }
        )
        if self.geo.idx_pool_degree:
            if idx_pool is None:
                raise ValueError(
                    "the geometry pools the indexer (idx_pool); give the step's idx_pool tables"
                )
            if set(idx_pool) != set(gk_upstream.CDP_TABLES):
                raise ValueError(
                    f"the chunk-dealt pool takes the tables {sorted(gk_upstream.CDP_TABLES)}, got {sorted(idx_pool)}"
                )
            control.update(idx_pool)
        # The split (`self.split`, the point's): layers [0, k1), [k1, k2), ..., [kn, L) as consecutive kernel calls,
        # the last with the sampling tail. Between two calls a one-element all-reduce of exact zeros over the TP
        # group, computed from the previous call's output and added to the next call's input: neuronx-cc starts a new
        # function of the program only at a torch-level collective, and the runtime gives each function its own
        # block of collective control state (points.GK_SPLIT says why one block for the whole stack is too large).
        # x + 0 is exact, so the split changes no number (nan_to_num keeps the fence zero when a residual holds an inf
        # or a NaN). The caches pass from call to call by functionalization: the wrapper returns each cache it writes
        # as an output aliasing its input.
        cuts = self.split
        layer_window = (0, 0)
        for lo, hi in zip([0, *cuts[:-1]], cuts):
            part = NF.glm52_gk_decode(
                x_shard.to(self.dtype),
                weights=weights,
                control=control,
                meshes=self._meshes,
                scalars=self._scalars,
                head=None,
                idx_shard_degree=self.geo.idx_shard_degree,
                idx_block_size=self.geo.idx_block_size,
                layer_window=(lo, hi),
            )
            x_shard = part[0] if isinstance(part, (tuple, list)) else part
            fence = torch.nan_to_num(
                x_shard[0:1, 0:1].to(torch.float32) * 0.0,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            fence = gk_upstream.get_tp_group().all_reduce(fence)
            x_shard = x_shard + fence.to(x_shard.dtype)
            layer_window = (hi, 0)
        out = NF.glm52_gk_decode(
            x_shard.to(self.dtype),
            weights=weights,
            control=control,
            meshes=self._meshes,
            scalars=self._scalars,
            head=head,
            idx_shard_degree=self.geo.idx_shard_degree,
            idx_block_size=self.geo.idx_block_size,
            layer_window=layer_window,
        )
        # The kernel returns the post-stack residual and the caches it wrote, which are aliased inputs, so only the
        # residual is read here; with the tail, (topk_values, topk_indices, tokens) follow, and the later stacks of a
        # split latent cache come last.
        if isinstance(out, (tuple, list)):
            later = [stack for stack in (kv_cache_hi, kv_cache_hi2, kv_cache_hi3) if stack is not None]
            if later:
                out = out[: -len(later)]
            out = out[0] if head is None else (out[0], out[-1])
        return out

