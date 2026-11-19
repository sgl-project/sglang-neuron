# SPDX-License-Identifier: Apache-2.0
"""The paged K/V write of MiMo-V2's attention: its heads are 256 channels (config.KV_HEAD_DIM), and nkilib writes the
cache inside its QKV and decode kernels only up to 128 (``update_cache`` / ``fp8_packed`` need ``d_head <= 128``), so
the graphs write the new rows themselves.

``Tensor.index_put_`` on a cache input lowers to a scatter that materializes a copy of the whole cache; this NKI kernel
DMAs the rows straight into the buffer with an indirect scatter and returns the caches, so the compiler reports the
write as an output alias and the next reader is ordered after it. From the vllm-neuron port (vllm-project/vllm-neuron
#67, ``kernels/kv_cache_write.py``): on the native route it wrote in place, 0.65 ms for two writes whatever the pool
size, as long as no two layers share a buffer, which the Neuron pools never do (one tensor per layer).

The torch ``index_put_`` below is the CPU reference."""

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from nki.isa import oob_mode
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel



def _scatter_rows(cache_flat, new_rows, idx_tile, start, tile_n, d):
    tile = nl.ndarray((tile_n, d), dtype=new_rows.dtype, buffer=nl.sbuf)
    nisa.dma_copy(tile, new_rows[nl.ds(start, tile_n), :])
    nisa.dma_copy(dst=cache_flat.ap(pattern=[[d, tile_n], [1, d]], offset=0, vector_offset=idx_tile, indirect_dim=0),
                  src=tile, oob_mode=oob_mode.skip)


@nki.jit
def _kv_scatter(k_cache, v_cache, k_new, v_new, row_idx):
    """k_new / v_new [N, d] into the [blocks, kv_heads, page, d] caches at flat rows row_idx [N, 1]; the caches are
    flattened here, not by the caller, so the aliasing pass sees the placeholders' own shapes. K and V go to the two
    cores of the logical NeuronCore."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("mimo_kv_write", (0, 1), 2)
    blocks, heads, page, d = k_cache.shape
    k_flat = k_cache.reshape((blocks * heads * page, d))
    v_flat = v_cache.reshape((blocks * heads * page, d))
    n_rows = k_new.shape[0]
    tile_sz = nl.tile_size.pmax
    for start in range(0, n_rows, tile_sz):
        tile_n = min(tile_sz, n_rows - start)
        idx_tile = nl.ndarray((tile_n, 1), dtype=row_idx.dtype, buffer=nl.sbuf)
        nisa.dma_copy(idx_tile, row_idx[nl.ds(start, tile_n)])
        if n_prgs == 1 or prg_id == 0:
            _scatter_rows(v_flat, v_new, idx_tile, start, tile_n, d)
        if n_prgs == 1 or prg_id == 1:
            _scatter_rows(k_flat, k_new, idx_tile, start, tile_n, d)
    return k_cache, v_cache


def write_kv(k_cache: Tensor, v_cache: Tensor, k: Tensor, v: Tensor, slots: Tensor) -> tuple:
    """Write k / v ``[kv_heads, T, d]`` at ``slots`` [T] (the pool's token slots, page = ``k_cache.shape[2]``) into
    the ``[blocks, kv_heads, page, d]`` caches; returns the written caches, which a later reader must take.

    A slot outside the pool (or negative) goes to slot 0, SGLang's reserved padding page, as the dense attention's QKV
    write does (models/transformer/model.Attention.prefill)."""
    blocks, heads, page, d = k_cache.shape
    tokens = slots.shape[0]
    slots = slots.to(torch.int64)
    slots = torch.where((slots < 0) | (slots >= blocks * page), torch.zeros_like(slots), slots)
    k = k.reshape(heads * tokens, d).to(k_cache.dtype)
    v = v.reshape(heads * tokens, d).to(v_cache.dtype)
    if not can_run_kernel(k):
        block, offset = slots // page, slots % page
        head = torch.arange(heads, device=k.device).repeat_interleave(tokens)
        index = (block.repeat(heads), head, offset.repeat(heads))
        k_cache.index_put_(index, k)
        v_cache.index_put_(index, v)
        return k_cache, v_cache
    # flat row of (block, head, offset), head-major over the tokens as k / v are
    base = (slots // page) * (heads * page) + slots % page
    rows = base.repeat(heads) + (torch.arange(heads, device=k.device, dtype=torch.int64) * page).repeat_interleave(
        tokens)
    return wrap_nki(_kv_scatter)[get_logical_nc_config()](k_cache=k_cache, v_cache=v_cache, k_new=k, v_new=v,
                                                          row_idx=rows.to(torch.int32).view(-1, 1))
