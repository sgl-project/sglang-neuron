# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Torch reference for the GLM-5.2 absorbed-MLA decode attention core, in the kernel's six steps:

    0. quantize and pack this step's new rows (pack_kv_rows)
    1. scatter them into the paged cache (_scatter)
    2. resolve each query token's global selection to local rows and dequantize (_gather_token_keys)
    3. scores = scale * (q_absorbed . kv_latent + q_pe . k_pe)
    4. mask: keep a key iff key_pos <= this token's position
    5. softmax, ctx = probs . kv_latent, lse = max + log(sum)

It quantizes exactly as the kernel does (same clamp, fp8 rounding, per-tensor scale and byte order),
so the only kernel-vs-reference difference is fp32 accumulation order. Skipping quantization would
measure the quantization error instead of kernel bugs, and the kernel's dangerous failures (a gather
permutation, a transpose parity swap) stay in range and would survive a loose tolerance.
"""

import ml_dtypes
import numpy as np
import torch

from .kv_row_layout import E4M3FN_MAX, ROPE_BYTES, ROW_BYTES, latent_byte_offset


def mla_decode_attention_torch_ref(
    q_absorbed: torch.Tensor,  # [P_MAX, nC, T, Nh]      feature-major, W_UK already folded in
    q_pe: torch.Tensor,  # [P_MAX, T, Nh]          feature-major, RoPE'd
    kv_cache: torch.Tensor,  # [n_blocks, 1, block_size, ROW_BYTES] uint8, packed rows
    kv_row_new: torch.Tensor,  # [num_seqs, spec, R + C]  bf16, unquantized
    new_slot_mapping: torch.Tensor,  # [num_seqs, spec]  int32, flat local row, or < 0 to skip
    gather_indices: torch.Tensor,  # [num_seqs, spec, num_prior] int32, global, per (seq, spec token)
    block_table: torch.Tensor,  # [num_seqs, max_blocks] int32, local logical to physical block
    positions: torch.Tensor,  # [T] int32, absolute position of each query token
    softmax_scale: float,
    kv_scale: float,
    spec: int,
    kv_lora_rank: int,
    cp_degree: int,
    cp_rank: int,
    # Prebuilt index tables only cache what the kernel derives from gather_indices; the reference
    # always works from the positions.
    prior_snake_rows=None,  # noqa: ARG001
    prior_key_pos=None,  # noqa: ARG001
    prior_natural_rows=None,  # noqa: ARG001
    attend_new_keys: bool = True,
) -> dict[str, torch.Tensor]:
    """
    One CP rank of absorbed-MLA decode. Returns ``ctx`` [T, Nh, C], ``lse`` [T, Nh], ``kv_cache``.

    Shapes use T = num_seqs * spec query tokens, C = kv_lora_rank, R = qk_rope_dim, Nh heads.
    """
    p_max, n_c, tokens, heads = q_absorbed.shape
    c_dim = kv_lora_rank
    r_dim = kv_row_new.shape[2] - c_dim
    num_seqs = block_table.shape[0]
    block_size = kv_cache.shape[2]
    out_dtype = q_absorbed.dtype

    # Undo the feature-major tiling to [T, Nh, feature]: feature f lives at [f % P_MAX, f // P_MAX].
    q_abs = q_absorbed.permute(1, 0, 2, 3).reshape(n_c * p_max, tokens, heads)
    q_abs = q_abs[0:c_dim].permute(1, 2, 0).float()  # [T, Nh, C]
    q_rope = q_pe[0:r_dim].permute(1, 2, 0).float()  # [T, Nh, R]

    # Steps 0 and 1: quantize this step's rows and scatter them, before any read.
    packed_new = pack_kv_rows(kv_row_new.reshape(-1, r_dim + c_dim), kv_scale, r_dim, c_dim)
    cache = _scatter(kv_cache, packed_new, new_slot_mapping)
    cache_rows = cache.reshape(-1, ROW_BYTES).cpu().numpy()

    # Step 2: the selection is [B, S, topk], so each of the T query tokens gathers its own key set.
    num_prior = gather_indices.shape[2]
    n_keys = num_prior + spec
    keys_t = torch.zeros((tokens, n_keys, r_dim + c_dim), dtype=torch.float32)
    key_pos_t = torch.full((tokens, n_keys), float(1 << 30), dtype=torch.float32)
    packed_new_np = packed_new.cpu().numpy()
    for t in range(tokens):
        s, j = divmod(t, spec)
        keys_t[t], key_pos_t[t] = _gather_token_keys(
            gather_indices[s, j],
            block_table[s],
            cache_rows,
            packed_new_np[s * spec : (s + 1) * spec],
            positions[s * spec : (s + 1) * spec],
            kv_scale,
            r_dim,
            c_dim,
            block_size,
            cp_degree,
            cp_rank,
            attend_new_keys,
        )
    k_pe, kv_latent = keys_t[..., :r_dim], keys_t[..., r_dim:]

    # Step 3. The rope half is a separate contraction because it is not absorbed.
    scores = softmax_scale * (
        torch.einsum("thc,tsc->ths", q_abs, kv_latent) + torch.einsum("thr,tsr->ths", q_rope, k_pe)
    )

    # Step 4, non-strict like the kernel and the model: key_pos <= pos keeps a token's own row and its
    # earlier speculative siblings. Non-resident and unselected slots carry 2**30 and always fail.
    keep = key_pos_t <= positions.float().unsqueeze(1)
    scores = scores.masked_fill(~keep.unsqueeze(1), float("-inf"))

    # Step 5: normalized context, plus the log-sum-exp for the CP combine.
    row_max = scores.amax(dim=-1, keepdim=True)
    shifted = torch.exp(scores - row_max)
    row_sum = shifted.sum(dim=-1, keepdim=True)
    ctx = torch.einsum("ths,tsc->thc", shifted / row_sum, kv_latent)
    lse = (row_max.squeeze(-1) + torch.log(row_sum.squeeze(-1))).float()
    # A token with no kept key on this rank (only possible with attend_new_keys=False; its own row is
    # otherwise kept) gets zero context and a -inf lse, which the CP combine weights by 0. Without this
    # the all -inf row would be NaN.
    empty = ~keep.any(dim=-1)  # [T]
    if bool(empty.any()):
        ctx = ctx.masked_fill(empty.view(-1, 1, 1), 0.0)
        lse = lse.masked_fill(empty.view(-1, 1), float("-inf"))

    return {"ctx": ctx.to(out_dtype), "lse": lse, "kv_cache": cache}


def _gather_token_keys(
    sel: torch.Tensor,  # [num_prior] int32 global positions, this query token's top-k
    table: torch.Tensor,  # [max_blocks] int32 this sequence's local block table
    cache_rows: np.ndarray,  # [n_cache_rows, ROW_BYTES] uint8
    own_rows: np.ndarray,  # [spec, ROW_BYTES] uint8, this step's own packed rows
    own_pos: torch.Tensor,  # [spec] int32
    kv_scale: float,
    r_dim: int,
    c_dim: int,
    block_size: int,
    cp_degree: int,
    cp_rank: int,
    attend_new_keys: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resolve one query token's selection to dequantized keys and each key's absolute position.

    As in the kernel, a slot gets position 2**30 (so the mask drops it) when the selector left it
    empty (pos < 0), another CP rank owns its block, or the block is outside or unallocated in this
    rank's table. Every appended sibling is dropped under ``attend_new_keys=False`` (MTP KV-share steps).
    """
    num_prior = sel.shape[0]
    spec = own_rows.shape[0]
    max_blocks = table.shape[0]
    n_rows = cache_rows.shape[0]

    keys = torch.zeros((num_prior + spec, r_dim + c_dim), dtype=torch.float32)
    pos_out = torch.full((num_prior + spec,), float(1 << 30), dtype=torch.float32)

    for k in range(num_prior):
        pos = int(sel[k])
        if pos < 0:
            continue
        block_g, intra = divmod(pos, block_size)
        if block_g % cp_degree != cp_rank:
            continue  # another rank holds this block
        block_l = block_g // cp_degree
        if block_l >= max_blocks:
            continue  # outside this rank's window
        block_id = int(table[block_l])
        if block_id < 0:
            continue  # unallocated
        row = block_id * block_size + intra
        if not (0 <= row < n_rows):
            continue
        keys[k] = torch.from_numpy(unpack_kv_row(cache_rows[row], kv_scale, r_dim, c_dim))
        pos_out[k] = float(pos)

    # This step's own rows, appended rather than gathered and read from the packed bytes, so they carry
    # the same quantization error as a gathered key. They are appended on every CP rank, so they need
    # the same residency test as a gathered key; without it the CP combine would count each one
    # cp_degree times.
    for i in range(spec):
        keys[num_prior + i] = torch.from_numpy(unpack_kv_row(own_rows[i], kv_scale, r_dim, c_dim))
        pos = int(own_pos[i])
        if attend_new_keys and (pos // block_size) % cp_degree == cp_rank:
            pos_out[num_prior + i] = float(pos)
    return keys, pos_out


def _scatter(kv_cache: torch.Tensor, packed_new: torch.Tensor, new_slot_mapping: torch.Tensor) -> torch.Tensor:
    """Write the packed new rows into a copy of the cache. A negative slot skips its row."""
    cache = kv_cache.clone()
    flat = cache.reshape(-1, ROW_BYTES)
    slots = new_slot_mapping.reshape(-1).long()
    for i in range(slots.numel()):
        if 0 <= slots[i] < flat.shape[0]:
            flat[slots[i]] = packed_new[i]
    return cache


def pack_kv_rows(rows: torch.Tensor, kv_scale: float, r_dim: int, c_dim: int) -> torch.Tensor:
    """
    Quantize and pack logical bf16 rows into the cache's byte layout, like ``_quantize_new_rows``.

    Returns [n, ROW_BYTES] uint8: bf16 k_pe at bytes [0, ROPE_BYTES), then the fp8 latent in the
    order of ``kv_row_layout.latent_byte_offset``. The clamp precedes the cast as in the kernel (see
    ``E4M3FN_MAX``). Public so tests build caches with the same packing code.
    """
    n = rows.shape[0]
    vals = rows.float().cpu().numpy()
    out = np.zeros((n, ROW_BYTES), dtype=np.uint8)

    rope = (vals[:, :r_dim] / kv_scale).astype(ml_dtypes.bfloat16)
    out[:, :ROPE_BYTES] = rope.view(np.uint8).reshape(n, r_dim * 2)

    lat = np.clip(vals[:, r_dim : r_dim + c_dim] / kv_scale, -E4M3FN_MAX, E4M3FN_MAX)
    out[:, _latent_byte_map(c_dim)] = lat.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return torch.from_numpy(out)


def unpack_kv_row(row: np.ndarray, kv_scale: float, r_dim: int, c_dim: int) -> np.ndarray:
    """Dequantize one packed uint8 cache row to [r_dim + c_dim] fp32 logical values."""
    rope = row[:ROPE_BYTES].view(ml_dtypes.bfloat16).astype(np.float32) * kv_scale
    lat = row[_latent_byte_map(c_dim)].view(ml_dtypes.float8_e4m3fn).astype(np.float32) * kv_scale
    return np.concatenate([rope, lat]).astype(np.float32)


def _latent_byte_map(c_dim: int) -> np.ndarray:
    """Byte offset of each latent dim, as an index array."""
    return np.array([latent_byte_offset(c) for c in range(c_dim)], dtype=np.int64)
