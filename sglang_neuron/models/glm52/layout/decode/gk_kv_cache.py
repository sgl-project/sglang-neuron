# SPDX-License-Identifier: Apache-2.0
"""The two KV caches of the GLM-5.2 decode giga-kernel (GK).

The kernel holds one cache for each of its two attention paths, and they have
different layouts on purpose:

- the **MLA latent cache** is BLOCK-paged, ``[L, num_blocks, 1, block_size,
  row_bytes]`` uint8 (640, or the per-token row's stride). One row is a quantized
  ``[k_pe | kv_latent]`` pair. Block ``b`` of
  a sequence lives on CP rank ``b % cp_degree``, at local index
  ``b // cp_degree``, which the rank's block table maps to a physical block. The
  kernel writes this step's own row through ``kv_new_slot_mapping``.
- the **indexer key cache** has MLA's block form, ``[L_full, idx_blocks, 1,
  idx_block_size, idx_head_dim]`` bf16, or uint8 for ``idx_fp8_keys``: one
  float8_e4m3fn byte per feature (`pack_indexer_keys`). It takes MLA's deal:
  32-key block ``b`` of a sequence lives on key rank ``b % cp_degree`` at local
  block ``b // cp_degree``, so the indexer shares MLA's block table. Only
  ``"full"`` layers have one, which is why the leading axis is the full-layer
  count.

Both are mutated in place by the kernel, and both must be the same buffer across
decode steps: step ``s`` attends to the row that step ``s - 1`` wrote.

This module allocates them, and holds the placement rules that say where an
already-decoded row lands (`deal_latent_blocks`, `indexer_shard_keys`), which the
placement tests read. The decode fills nothing itself: the prefill's KV transfer
writes straight into these caches, which the decode registers as its transfer
regions.

Example:
    >>> geo = Glm52GkGeometry(num_layers=1)
    >>> caches = allocate_caches(geo, device="cpu")
    >>> caches.kv_cache_all.shape
    torch.Size([1, 1024, 1, 32, 640])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from sglang_neuron.models.glm52.kernels.decode.attention.mla.kv_row_layout import (  # noqa: E501
    E4M3FN_MAX,
)
from sglang_neuron.models.glm52.layout.decode.gk_geometry import Glm52GkGeometry


#: Exclusive bound on the byte offset where a latent layer STARTS inside one stacked tensor. The DGE
#: encodes an indirect DMA's static offset in 32 bits, and neuronx-cc refuses a deeper start
#: (NCC_IEIL015 / NCC_IBTN127). Measured at 16 x 1M: layer 51 of one stack compiles, layer 52 does not.
DGE_STATIC_OFFSET_LIMIT = 2**32


#: Exclusive bound on one stack's bytes: the largest single device allocation (the Neuron driver refuses a
#: contiguous allocation of 2**32 bytes or more: 4,294,963,200 allocates, 4,294,967,296 does not).
DEVICE_ALLOCATION_LIMIT = 2**32


#: Latent stacks the kernel takes (``kv_cache_all``, ``kv_cache_hi``, ``kv_cache_hi2``, ``kv_cache_hi3``). Four: at 32 x 1M,
#: three stacks would be 4.36 GB each, past DEVICE_ALLOCATION_LIMIT.
MAX_LATENT_STACKS = 4


def latent_stack_sizes(geo: Glm52GkGeometry) -> tuple[int, ...]:
    """Layers in each latent stack: as few stacks as keep every layer's start below the DGE limit and every stack
    below the device's allocation limit.

    At the 128k point a layer is 21 MB and one 78-layer stack is fine. At 16 x 1M a layer is 84 MB and
    two stacks of 39 are needed; at 32 x 1M a layer is 168 MB, and four stacks (20, 20, 20, 18: 3.36 GB each at
    most) are, where three of 26 would start every layer in range but take 4.36 GB a stack.

    Raises:
        ValueError: If even `MAX_LATENT_STACKS` stacks start a layer at or past `DGE_STATIC_OFFSET_LIMIT` or hold
            `DEVICE_ALLOCATION_LIMIT` bytes or more; the message names the limit that fails.

    Example:
        >>> latent_stack_sizes(Glm52GkGeometry())
        (78,)
    """
    layer_bytes = geo.num_blocks * geo.block_size * geo.row_bytes
    for n in range(1, MAX_LATENT_STACKS + 1):
        size = -(-geo.num_layers // n)
        if (size - 1) * layer_bytes < DGE_STATIC_OFFSET_LIMIT and size * layer_bytes < DEVICE_ALLOCATION_LIMIT:
            return tuple([size] * (n - 1) + [geo.num_layers - size * (n - 1)])
    size = -(-geo.num_layers // MAX_LATENT_STACKS)
    over = []
    if (size - 1) * layer_bytes >= DGE_STATIC_OFFSET_LIMIT:
        over.append(f"start their last layer at byte {(size - 1) * layer_bytes}, past the DGE's static offset limit "
                    f"{DGE_STATIC_OFFSET_LIMIT}")
    if size * layer_bytes >= DEVICE_ALLOCATION_LIMIT:
        over.append(f"hold {size * layer_bytes} bytes each, past the device's allocation limit "
                    f"{DEVICE_ALLOCATION_LIMIT}")
    raise ValueError(
        f"a latent layer is {layer_bytes} bytes, and {MAX_LATENT_STACKS} stacks of {size} layers "
        f"{' and '.join(over)}. Lower the context or the slot count."
    )


@dataclass
class Glm52GkCaches:
    """The kernel's caches for one rank.

    Attributes:
        kv_cache_all: ``[L, num_blocks, 1, block_size, row_bytes]`` uint8 paged
            latent cache, or its first stack's layers when the stack is split
            (`latent_stack_sizes`). Mutated in place.
        idx_k_cache_all: ``[L_full, idx_blocks, 1, idx_block_size, idx_head_dim]``
            paged indexer key cache in MLA's block form, dealt with MLA's rule:
            bf16, or uint8 fp8 bytes for ``idx_fp8_keys``. Mutated in place.
        kv_cache_hi: The next latent layers of a split stack, or None.
        kv_cache_hi2: The third latent stack of a split, or None.
        kv_cache_hi3: The last latent layers of a four-way split, or None.
    """

    kv_cache_all: torch.Tensor
    idx_k_cache_all: torch.Tensor
    kv_cache_hi: torch.Tensor | None = None
    kv_cache_hi2: torch.Tensor | None = None
    kv_cache_hi3: torch.Tensor | None = None

    @property
    def latent_stacks(self) -> tuple[torch.Tensor, ...]:
        """The latent stacks in layer order."""
        return tuple(
            stack
            for stack in (self.kv_cache_all, self.kv_cache_hi, self.kv_cache_hi2, self.kv_cache_hi3)
            if stack is not None
        )

    @property
    def num_latent_layers(self) -> int:
        """Latent layers over every stack."""
        return sum(stack.shape[0] for stack in self.latent_stacks)


def cache_shapes(geo: Glm52GkGeometry) -> dict[str, tuple[int, ...]]:
    """Return the shape of each cache for one geometry.

    Example:
        >>> cache_shapes(Glm52GkGeometry(num_layers=78))["idx_k_cache_all"]
        (21, 1024, 1, 32, 128)
    """
    return {
        "kv_cache_all": (
            geo.num_layers,
            geo.num_blocks,
            1,
            geo.block_size,
            geo.row_bytes,
        ),
        # MLA's block form. The indexer shares MLA's block table, so its cache has MLA's shape; a store
        # file that holds the rows flat is reshaped at load (same bytes).
        "idx_k_cache_all": (
            geo.num_full_layers,
            geo.idx_blocks,
            1,
            geo.idx_block_size,
            geo.idx_head_dim,
        ),
    }


def idx_key_dtype(geo: Glm52GkGeometry) -> torch.dtype:
    """dtype of the indexer key cache: bf16, or uint8 (fp8 bytes) for ``idx_fp8_keys``."""
    return torch.uint8 if geo.idx_fp8_keys else torch.bfloat16


def _zeros(shape, dtype, device) -> torch.Tensor:
    """A zero tensor whose transient memory is one leading slice: on a Neuron device ``torch.zeros`` of a uint8 tensor
    takes 4 bytes an element while it fills (a 3.36 GB stack asks for 13.4 GB), so the stack is allocated empty and
    zeroed slice by slice. On the CPU it is ``torch.zeros``."""
    if torch.device(device).type == "cpu":
        return torch.zeros(shape, dtype=dtype, device=device)
    tensor = torch.empty(shape, dtype=dtype, device=device)
    for i in range(shape[0]):
        tensor[i].zero_()
    return tensor


def allocate_caches(
    geo: Glm52GkGeometry,
    device: str | torch.device = "cpu",
    indexer_fill: str = "zeros",
    seed: int = 42,
) -> Glm52GkCaches:
    """Allocate both caches, zeroed, with the indexer's unwritten rows filled.

    `indexer_fill` is NOT cosmetic. A live candidate whose key row is all zeros
    scores the same as every other zero row, and an exact tie over thousands of
    rows is not something the top-k has to resolve one way: it can return a
    position more than once, so the gather reads one row twice and misses
    another. Distinct rows make the residency threshold the only thing that
    decides the selected set. Measured on a 6-layer stack, zeros moved the error
    from 0.282 to 0.346.

    Rows that the caller fills with real keys overwrite the fill, so this matters
    only for the positions that no key occupies. A served decode allocates zeros
    (the default): its unwritten rows sit past each slot's residency threshold, so
    no zero row is ever a live candidate. The random fill is for a test whose live
    candidates hold no real key.

    Args:
        geo: The sharding point.
        device: Where to allocate.
        indexer_fill: ``"zeros"`` (the default, as served) or ``"random"`` for
            standard normal rows.
        seed: RNG seed of the random fill. Fixed, so two ranks and two runs agree.

    Returns:
        The two caches.

    Raises:
        ValueError: If `indexer_fill` is not one of the two values.
    """
    shapes = cache_shapes(geo)
    full = shapes["kv_cache_all"]
    stacks = [
        _zeros((size, *full[1:]), torch.uint8, device)
        for size in latent_stack_sizes(geo)
    ]
    kv, kv_hi, kv_hi2, kv_hi3 = (stacks + [None, None, None])[:4]
    if indexer_fill == "zeros":
        idx = _zeros(shapes["idx_k_cache_all"], idx_key_dtype(geo), device)
    elif indexer_fill == "random":
        rng = np.random.default_rng(seed)
        # One row set shared by every layer: the fill only has to be distinct, and
        # drawing it once keeps the allocation cheap at 21 layers x 32768 rows.
        one = pack_indexer_keys(
            rng.normal(scale=1.0, size=(geo.idx_keys, geo.idx_head_dim)), geo
        )
        idx = (
            one.reshape(shapes["idx_k_cache_all"][1:])
            .unsqueeze(0)
            .expand(shapes["idx_k_cache_all"])
            .contiguous()
            .to(device)
        )
    else:
        raise ValueError(
            f"indexer_fill must be 'zeros' or 'random', got {indexer_fill!r}. "
            "A served decode allocates zeros; 'random' gives a test's live candidates distinct rows."
        )
    return Glm52GkCaches(
        kv_cache_all=kv, idx_k_cache_all=idx, kv_cache_hi=kv_hi, kv_cache_hi2=kv_hi2, kv_cache_hi3=kv_hi3
    )



def deal_latent_blocks(
    rows: np.ndarray, n_prior: int, geo: Glm52GkGeometry, cp: int
) -> np.ndarray:
    """Give one CP rank its paged latent cache for one layer.

    Global block ``b`` belongs to CP rank ``b % cp_degree`` and sits at that
    rank's local index ``b // cp_degree``. The block table is the identity here,
    so the local index is also the physical block.

    Args:
        rows: ``[n_prior, row_bytes]`` uint8 packed rows, indexed by TRUE
            position.
        n_prior: Prior positions that hold a real key.
        geo: The sharding point.
        cp: The CP position of the rank.

    Returns:
        ``[num_blocks, 1, block_size, row_bytes]`` uint8.
    """
    bs = geo.block_size
    out = np.zeros((geo.num_blocks, 1, bs, geo.row_bytes), dtype=np.uint8)
    for local in range(geo.local_blocks_per_seq):
        block_g = cp + local * geo.cp_degree
        lo = block_g * bs
        if lo >= n_prior:
            break
        take = min(bs, n_prior - lo)
        out[local, 0, :take] = rows[lo : lo + take]
    return out


def pack_indexer_keys(keys: np.ndarray, geo: Glm52GkGeometry) -> torch.Tensor:
    """Convert ``[n, idx_head_dim]`` float keys to the indexer cache's dtype (see `idx_key_dtype`).

    An ``idx_fp8_keys`` cache takes `pack_keys_torch` bytes, the ones the kernel's own key writes hold.
    """
    keys = torch.from_numpy(np.ascontiguousarray(keys, dtype=np.float32))
    return pack_keys_torch(keys) if geo.idx_fp8_keys else keys.to(torch.bfloat16)


def pack_keys_torch(keys: torch.Tensor) -> torch.Tensor:
    """Pack ``[n, 128]`` indexer keys into ``[n, 128]`` uint8 rows of an ``idx_fp8_keys`` fp8 key cache.

    Each byte is one float8_e4m3fn feature of ``clamp(key)``, with no scale: the keys are LayerNorm
    outputs, and the top-k ignores a scale. The decode kernel's ``indexer_kv_score`` writes the same bytes.

    Args:
        keys: Keys, any float dtype.

    Returns:
        ``[n, 128]`` uint8.
    """
    f8 = keys.float().clamp(-E4M3FN_MAX, E4M3FN_MAX).to(torch.float8_e4m3fn)
    return f8.contiguous().view(torch.uint8)


def indexer_shard_keys(
    keys: np.ndarray, n_prior: int, geo: Glm52GkGeometry, key_rank: int
) -> tuple[np.ndarray, int]:
    """Give one key rank its rows of the indexer cache for one layer, in local row order.

    The keys are DEALT with MLA's rule (`Glm52GkGeometry.idx_global_positions`): local row ``i``
    of key rank ``k`` holds global position ``(cp_degree * (i // 32) + k) * 32 + i % 32``. Every
    local row is written. A position past the prior gets a zero row, which the residency
    threshold marks dead.

    Args:
        keys: ``[n_prior, W]`` indexer keys at TRUE positions, in any dtype: ``idx_head_dim``
            values, or the ``idx_key_bytes`` bytes of a packed row.
        n_prior: Prior positions that hold a real key.
        geo: The sharding point.
        key_rank: The key-shard position of the rank.

    Returns:
        The ``[idx_keys, W]`` rows of this rank, in the dtype of ``keys``, and the row count,
        ``idx_keys``.
    """
    positions = geo.idx_global_positions(key_rank)
    rows = np.zeros((geo.idx_keys, keys.shape[1]), dtype=keys.dtype)
    live = positions < n_prior
    rows[live] = keys[positions[live]]
    return rows, geo.idx_keys


