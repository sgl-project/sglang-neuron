"""The MLA attention of the Kimi-K3 prefill stage: decode-context-parallel segment attention over FP8 cache rows.

The cache stripes the positions over the ranks (``fp8_rows``): rank ``r`` keeps the rows of the stripes it owns, in
consecutive rank-local slots, through its block table. One chunk of ``T`` rows (whole stripe cycles) starts at
``cached_len``, any count of tokens: a prompt-cache hit can end inside a stripe. Per chunk and rank:

1. Gather the normed rows ``[T, hidden]``. Project the queries of the rank's ``hc`` heads (``q_a`` and its norm, then
   ``q_b``) and the cache rows of all ``T`` rows (``kv_a``, the latent norm, the raw rope part, FP8).
2. Write the rows the rank owns at ``write_slots`` (owner only; a pad row or a slot of -1 writes nothing).
3. Per query tile: absorb ``W_K`` into the local queries, gather the queries of all heads, and attend in one softmax
   (a) the rank's prior slots (``fp8_rows.local_prior``: the positions before ``cached_len`` in its stripes, the same
   count on every rank only when ``cached_len`` is whole stripe cycles) through the block table and (b) the chunk rows
   it owns (``fp8_rows.owned_rows_from``: by the absolute stripe of each row's position), with a
   causal bias by position that also hides pad keys. The values are the latent rows; ``W_V`` of every head projects
   each rank's partial before the merge.
4. Merge the partials of all ranks (gather the row max and sum, weight each partial, reduce-scatter by head), apply
   the sigmoid gate and the ``o_proj`` row shard, and reduce-scatter the partials by row.

The model is NoPE: the rope part of the rows is stored and attended raw, never rotated, so ``positions`` does not
enter the math; the masks use ``cached_len`` and the row index (a pad row may repeat a position).

Backends: ``"torch"`` (CPU, fp32 attention) and ``"kernel"`` (the NKI flash and merge kernels, the kernel row write).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from sglang_neuron.models.kimi_k3.prefill import kernel_imports, page_io
from sglang_neuron.models.kimi_k3.prefill.fp8_rows import (
    STRIPE,
    byte_values,
    cache_rows,
    local_prior,
    owned_rows_from,
    quantize_rows,
    rms_norm,
    row_bytes,
)
from sglang_neuron.models.kimi_k3.prefill.kda import zero_pad_rows

_GRID = 1
MARKER = -3.0e4
"""Additive pre-scale score of a hidden cache slot."""
OWN_HIDDEN = -3.0e4
"""Additive post-scale score of a hidden own key (a later position or a pad row)."""
FLASH_BLOCK = 2048
"""Cache slots of one kernel block; the kernel's block table holds whole blocks."""
MAX_Q_TILE = 512


@dataclass
class MlaCache:
    """One MLA layer's cache on this rank.

    ``pool`` ``[num_blocks, STRIPE, width]`` uint8 (FP8 e4m3fn bytes; block 0 reserved); ``block_table`` ``[n]`` int32
    the rank's blocks of the request in order; ``write_slots`` ``[T]`` int32 the pool row of each chunk row this rank
    owns, -1 otherwise; ``cached_len`` ``[1]`` int64 tokens before the chunk (any count of tokens).
    """

    pool: torch.Tensor
    block_table: torch.Tensor
    write_slots: torch.Tensor
    cached_len: torch.Tensor


def prior_marker(prior: torch.Tensor, slots: int, dtype=torch.float32) -> torch.Tensor:
    """``[1, slots]``: 0 for a local slot before ``prior``, ``MARKER`` after."""
    j = torch.arange(slots, device=prior.device)
    zero = torch.zeros((), dtype=dtype, device=prior.device)
    return torch.where(j >= prior.reshape(-1)[:1], torch.full_like(zero, MARKER), zero)[None]


def own_key_bias(key_pos: torch.Tensor, q_pos: torch.Tensor, valid_end: torch.Tensor) -> torch.Tensor:
    """``[Sq, Kc]`` fp32: 0 for a visible own key, ``OWN_HIDDEN`` for a later key or a key at or after ``valid_end``."""
    hidden = (key_pos[None, :] > q_pos[:, None]) | (key_pos >= valid_end.reshape(-1)[:1])[None, :]
    zero = torch.zeros((), dtype=torch.float32, device=key_pos.device)
    return torch.where(hidden, torch.full_like(zero, OWN_HIDDEN), zero)


def flash_partial(q_c, q_r, cache_keys, marker, own_keys, cbias, scale, w_v):
    """The fp32 flash partial of one query tile (the kernel's math, unnormalized).

    Args:
        q_c: ``[H, R, Sq]`` absorbed latent queries (feature-major); q_r ``[H, DR, Sq]``.
        cache_keys: ``[S, R + DR]`` the rank-local logical slots (hidden ones may hold anything finite).
        marker: ``[1, S]`` (``prior_marker``), added before the scale.
        own_keys: ``[Kc, R + DR]``; cbias ``[Sq, Kc]`` added after the scale.
        w_v: ``[H, VD, R]`` value projection of every head.

    Returns:
        ``(acc [H, VD, Sq], m [H, Sq, 1], l [H, Sq, 1])`` fp32: ``acc = sum_j exp(s_j - m) W_V c_j``.
    """
    R = q_c.shape[1]
    q = torch.cat([q_c, q_r], dim=1).float().permute(0, 2, 1)
    k_cache, k_own = cache_keys.float(), own_keys.float()
    s_cache = (torch.einsum("htd,sd->hts", q, k_cache) + marker.float().reshape(1, 1, -1)) * scale
    s_own = torch.einsum("htd,sd->hts", q, k_own) * scale + cbias.float()[None]
    scores = torch.cat([s_cache, s_own], dim=-1)
    m = scores.amax(dim=-1, keepdim=True)
    p = torch.exp(scores - m)
    row_sum = p.sum(dim=-1, keepdim=True)
    acc = torch.einsum("hts,sr->htr", p, torch.cat([k_cache[:, :R], k_own[:, :R]], dim=0))
    return torch.einsum("htr,hdr->hdt", acc, w_v.float()), m, row_sum


def merge_weights(all_stats: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Merge weight ``[H, Sq]`` of this rank: ``exp(m - M) / sum_r l_r exp(m_r - M)`` with ``M`` the max over ranks.

    ``all_stats`` ``[W, 2H, Sq]``: row max then row sum of each rank.
    """
    H = m.shape[0]
    all_m, all_l = all_stats[:, :H], all_stats[:, H:]
    g_max = all_m.amax(dim=0)
    g_sum = (all_l * torch.exp(all_m - g_max[None])).sum(dim=0)
    return torch.exp(m.squeeze(-1) - g_max) / g_sum


def wv_kernel_layout(w_v: torch.Tensor) -> torch.Tensor:
    """``[H, VD, R]`` -> the flash kernel's ``[H, 128, R / 128, VD]``: ``out[h, c, k, d] = w_v[h, d, 128 k + c]``."""
    H, VD, R = w_v.shape
    return w_v.reshape(H, VD, R // 128, 128).permute(0, 3, 2, 1).contiguous()


class MlaAttention(nn.Module):
    """One MLA layer on one rank: ``hc`` query heads, the rank's cache stripes.

    Parameters: ``q_a_proj [Q, hidden]``, ``q_a_layernorm [Q]``, ``kv_a_proj_with_mqa [R + DR, hidden]``,
    ``kv_a_layernorm [R]`` (every rank), ``q_b_proj [hc (N + DR), Q]``, ``g_proj [hc VD, hidden]``, ``o_proj
    [hidden, hc VD]`` (rank shards), ``w_uk [hc, N, R]`` (the rank's ``W_K`` rows of ``kv_b_proj``) and
    ``w_v_all [H, VD, R]`` (``W_V`` of every head, for the projection before the merge).

    The kernel backend can take MXFP8 modules in place of the bf16 linears: ``q_b_proj`` and ``o_proj`` with the
    ``nn.Linear`` call contract (``o_proj.forward_feature_major([hc VD, T]) -> [T, hidden]`` when it has one), and
    ``w_uk_linear([hc, N, M]) -> [hc, R, M]`` for the absorb.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rank: int,
        world: int,
        lora_norm_eps: float = 1e-6,
        stripe: int = STRIPE,
        q_tile: int = MAX_Q_TILE,
        dtype: torch.dtype = torch.bfloat16,
        backend: str = "torch",
    ):
        super().__init__()
        if num_heads % world:
            raise ValueError(f"{num_heads} MLA heads do not divide by the stage width {world}.")
        if backend not in ("torch", "kernel"):
            raise ValueError(f"MLA backend {backend!r} is not known. Use 'torch' or 'kernel'.")
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.rank, self.world, self.hc = rank, world, num_heads // world
        self.q_lora_rank, self.kv_lora_rank = q_lora_rank, kv_lora_rank
        self.nope, self.rope, self.v_head_dim = qk_nope_head_dim, qk_rope_head_dim, v_head_dim
        self.scaling = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5
        self.lora_norm_eps, self.stripe, self.q_tile, self.backend = lora_norm_eps, stripe, q_tile, backend
        hc, R = self.hc, kv_lora_rank

        def linear(n_out, n_in):
            return nn.Linear(n_in, n_out, bias=False, dtype=dtype)

        self.q_a_proj = linear(q_lora_rank, hidden_size)
        self.q_a_layernorm = nn.Parameter(torch.ones(q_lora_rank, dtype=dtype))
        self.q_b_proj = linear(hc * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank)
        self.kv_a_proj_with_mqa = linear(R + qk_rope_head_dim, hidden_size)
        self.kv_a_layernorm = nn.Parameter(torch.ones(R, dtype=dtype))
        self.w_uk = nn.Parameter(torch.zeros(hc, qk_nope_head_dim, R, dtype=dtype))
        self.w_v_all = nn.Parameter(torch.zeros(num_heads, v_head_dim, R, dtype=dtype))
        self.g_proj = linear(hc * v_head_dim, hidden_size)
        self.o_proj = linear(hidden_size, hc * v_head_dim)
        self.w_uk_linear = None
        self._flash_shim = None
        self._merge_kernel = None

    @classmethod
    def from_geometry(cls, geometry, rank: int, world: int, **kwargs) -> "MlaAttention":
        """Build the module from the model geometry (``geometry.KimiK3Geometry``)."""
        return cls(
            hidden_size=geometry.hidden_size,
            num_heads=geometry.num_heads,
            q_lora_rank=geometry.q_lora_rank,
            kv_lora_rank=geometry.kv_lora_rank,
            qk_nope_head_dim=geometry.qk_nope_head_dim,
            qk_rope_head_dim=geometry.qk_rope_head_dim,
            v_head_dim=geometry.v_head_dim,
            lora_norm_eps=geometry.mla_lora_norm_eps,
            rank=rank,
            world=world,
            **kwargs,
        )

    def load_full(self, tensors: Mapping[str, torch.Tensor]) -> None:
        """Copy this rank's shard from the full layer tensors (checkpoint names under ``self_attn.``)."""
        r, hc, H = self.rank, self.hc, self.num_heads
        qd, kvd, VD = self.nope + self.rope, self.nope + self.v_head_dim, self.v_head_dim
        with torch.no_grad():
            self.q_a_proj.weight.copy_(tensors["q_a_proj.weight"])
            self.q_a_layernorm.copy_(tensors["q_a_layernorm.weight"])
            self.q_b_proj.weight.copy_(tensors["q_b_proj.weight"][r * hc * qd : (r + 1) * hc * qd])
            self.kv_a_proj_with_mqa.weight.copy_(tensors["kv_a_proj_with_mqa.weight"])
            self.kv_a_layernorm.copy_(tensors["kv_a_layernorm.weight"])
            kv_b = tensors["kv_b_proj.weight"].view(H, kvd, self.kv_lora_rank)
            self.w_uk.copy_(kv_b[r * hc : (r + 1) * hc, : self.nope])
            self.w_v_all.copy_(kv_b[:, self.nope :])
            self.g_proj.weight.copy_(tensors["g_proj.weight"][r * hc * VD : (r + 1) * hc * VD])
            self.o_proj.weight.copy_(tensors["o_proj.weight"][:, r * hc * VD : (r + 1) * hc * VD])

    def forward(self, x_rows: torch.Tensor, positions: torch.Tensor, cache: MlaCache, valid_len: torch.Tensor,
                comm) -> torch.Tensor:
        """``x_rows [Tl, hidden]`` bf16 -> o_proj rows ``[Tl, hidden]`` bf16; writes the chunk's owned rows into
        ``cache.pool`` in place. ``valid_len`` ``[1]`` int64 counts the valid rows of the chunk."""
        del positions
        x = comm.all_gather(x_rows, dim=0)
        T, device = x.shape[0], x.device
        tile = min(self.q_tile, T)
        if T % (self.stripe * self.world) or T % tile:
            raise ValueError(f"An MLA chunk of {T} rows must tile by stripe x world = {self.stripe * self.world} "
                             f"and by the query tile {tile}.")
        start = cache.cached_len.reshape(-1)[:1].to(torch.int64)
        rank = comm.rank_tensor.to(device)
        # this rank's slots before the chunk: start // world only when start is whole stripe cycles
        prior = local_prior(start, rank, self.world, self.stripe)
        valid_end = start + valid_len.reshape(-1)[:1].to(torch.int64)
        q_pos = start + torch.arange(T, device=device)

        q_a = rms_norm(self.q_a_proj(x), self.q_a_layernorm, self.lora_norm_eps)
        q = self.q_b_proj(q_a).view(T, self.hc, self.nope + self.rope)
        q_nope_t = q[..., : self.nope].permute(1, 2, 0)
        q_rot_t = q[..., self.nope :].permute(1, 2, 0)

        rows = quantize_rows(cache_rows(self.kv_a_proj_with_mqa(x), self.kv_a_layernorm, self.lora_norm_eps,
                                        self.kv_lora_rank))
        own = owned_rows_from(start, T, rank, self.world, self.stripe)
        own_rows = rows.index_select(0, own)
        own_pos = start + own
        keep = own_pos < valid_end
        slots = cache.write_slots.reshape(-1).index_select(0, own).to(torch.int64)
        # A pad own key attends as 0: its bias hides it, and a zero key scores exactly the bias.
        own_keys = torch.where(keep[:, None], own_rows.to(torch.bfloat16), torch.zeros((), dtype=torch.bfloat16,
                                                                                          device=device))
        if self.backend == "kernel":
            o_t = self._attend_kernel(q_nope_t, q_rot_t, cache, own_keys, own_pos, q_pos, prior, valid_end, tile,
                                      comm)
            edge = o_t.reshape(-1)[:1]
            page_io.write_rows_kernel(cache.pool, row_bytes(own_rows), slots, keep,
                                      (edge != edge).to(torch.int32).reshape(1, 1))
        else:
            page_io.write_rows(cache.pool, row_bytes(own_rows), slots, keep)
            o_t = self._attend_torch(q_nope_t, q_rot_t, cache, own_keys, own_pos, q_pos, prior, valid_end, tile,
                                     comm)
        return zero_pad_rows(self._output(o_t, x, comm), valid_len, comm)

    def _absorb(self, q_nope_t: torch.Tensor) -> torch.Tensor:
        """``[hc, N, M]`` -> ``[hc, R, M]``: ``W_K^T q_nope^T`` of each local head."""
        if self.w_uk_linear is not None:
            return self.w_uk_linear(q_nope_t)
        return torch.matmul(self.w_uk.transpose(1, 2), q_nope_t)

    def _tile_queries(self, q_nope_t, q_rot_t, t0, tile, comm):
        q_c = comm.all_gather(self._absorb(q_nope_t[:, :, t0 : t0 + tile].contiguous()), dim=0)
        q_r = comm.all_gather(q_rot_t[:, :, t0 : t0 + tile].contiguous(), dim=0)
        return q_c, q_r

    def _merge(self, acc, m, row_sum, comm, merge_fn=None):
        """Weighted partial of this rank, reduce-scattered by head: ``[hc, VD, Sq]`` fp32."""
        H = m.shape[0]
        stats = torch.cat([m.squeeze(-1), row_sum.squeeze(-1)], dim=0)
        all_stats = comm.all_gather(stats.unsqueeze(0), dim=0).reshape(-1, 2 * H, stats.shape[-1])
        if merge_fn is None:
            weighted = acc * merge_weights(all_stats, m).unsqueeze(1)
        else:
            weighted = merge_fn(all_stats, m, acc)
        return comm.reduce_scatter(weighted, dim=0)

    def _attend_torch(self, q_nope_t, q_rot_t, cache: MlaCache, own_keys, own_pos, q_pos, prior, valid_end, tile,
                      comm):
        view = page_io.gather_pages(cache.pool, cache.block_table)
        slots = view.shape[0]
        visible = torch.arange(slots, device=view.device) < prior
        # A hidden slot reads as 0: the pool may hold bytes of another request or NaN codes there.
        keys = torch.where(visible[:, None], byte_values(view), torch.zeros((), device=view.device))
        marker = prior_marker(prior, slots)
        outs = []
        for t0 in range(0, q_pos.shape[0], tile):
            q_c, q_r = self._tile_queries(q_nope_t, q_rot_t, t0, tile, comm)
            cbias = own_key_bias(own_pos, q_pos[t0 : t0 + tile], valid_end)
            acc, m, row_sum = flash_partial(q_c, q_r, keys, marker, own_keys, cbias, self.scaling, self.w_v_all)
            outs.append(self._merge(acc, m, row_sum, comm).to(self.w_v_all.dtype))
        return torch.cat(outs, dim=-1).reshape(self.hc * self.v_head_dim, -1)

    def _output(self, o_t: torch.Tensor, x: torch.Tensor, comm) -> torch.Tensor:
        """Gate and ``o_proj`` of the feature-major attention output ``[hc VD, T]``, reduce-scattered by row."""
        if hasattr(self.o_proj, "forward_feature_major"):
            gate_t = torch.sigmoid(torch.matmul(self.g_proj.weight, x.t()))
            partial = self.o_proj.forward_feature_major(o_t * gate_t)
        else:
            partial = self.o_proj(o_t.t() * torch.sigmoid(self.g_proj(x)))
        return comm.reduce_scatter(partial, dim=0)

    # kernel backend

    def kernel_buffers(self) -> dict[str, torch.Tensor]:
        """The flash kernel's extra inputs: ``w_v`` in its layout (``wv_kernel_layout``) and a bf16 identity."""
        return {
            "w_v": wv_kernel_layout(self.w_v_all.data.to(torch.bfloat16)),
            "ident": torch.eye(128, dtype=torch.bfloat16, device=self.w_v_all.device),
        }

    def set_kernel_buffers(self, buffers: Mapping[str, torch.Tensor]) -> None:
        """Register the kernel inputs under the names of ``kernel_buffers`` (the loader's kernel forms)."""
        for name, value in buffers.items():
            self.register_buffer(f"kernel_{name}", value, persistent=False)

    def prepare_kernel(self) -> None:
        """Build the flash and merge trace shims, and ``kernel_buffers`` when ``set_kernel_buffers`` has not given
        them; call once after the load, outside any traced forward. The MXFP8 ``q_b_proj``, ``o_proj`` and
        ``w_uk_linear`` come from the loader."""
        if not hasattr(self, "kernel_w_v"):
            self.set_kernel_buffers(self.kernel_buffers())
        self._flash_shim = _flash_shim()
        self._merge_kernel = kernel_imports.mla_dcp_merge()
        page_io.prepare_rows_kernel()

    def _attend_kernel(self, q_nope_t, q_rot_t, cache: MlaCache, own_keys, own_pos, q_pos, prior, valid_end, tile,
                       comm):
        from sglang_neuron.backend import stack

        if self._flash_shim is None:
            raise ValueError("The MLA kernel backend is not prepared. Call prepare_kernel() at load.")
        if cache.pool.shape[-2] != 128 or tile % 128 or tile > MAX_Q_TILE:
            raise ValueError("The MLA flash kernel takes 128-row blocks and a query tile of 128 x k, at most "
                             f"{MAX_Q_TILE}.")
        table = cache.block_table.reshape(1, -1).to(torch.int32).clamp_min(0)
        pad = -table.shape[1] % (FLASH_BLOCK // 128)
        if pad:
            table = torch.cat([table, table.new_zeros(1, pad)], dim=1)
        marker = prior_marker(prior, table.shape[1] * 128, torch.bfloat16)
        kc = quantize_rows(own_keys)
        merge = stack.wrap_nki(self._merge_kernel)[_GRID]
        flash = stack.wrap_nki(self._flash_shim)[_GRID]
        heads = self.num_heads
        group = max(g for g in range(1, min(8, heads) + 1) if heads % g == 0)
        outs = []
        for t0 in range(0, q_pos.shape[0], tile):
            q_c, q_r = self._tile_queries(q_nope_t, q_rot_t, t0, tile, comm)
            cbias = own_key_bias(own_pos, q_pos[t0 : t0 + tile], valid_end)
            # A tile whose first row is a pad row holds pad rows only: it streams no cache block.
            live = (q_pos[t0 : t0 + 1] < valid_end).to(torch.int32)
            acc, m, row_sum = flash(
                q_c.to(torch.bfloat16), q_r.to(torch.bfloat16), cache.pool, table, marker, kc, cbias,
                self.kernel_ident, (prior.to(torch.int32) * live).reshape(1, 1), self.kernel_w_v,
                scale=float(self.scaling), heads_per_group=int(group),
            )
            outs.append(self._merge(acc, m, row_sum, comm, merge_fn=merge).to(torch.bfloat16))
        return torch.cat(outs, dim=-1).reshape(self.hc * self.v_head_dim, -1)


def _flash_shim():
    """The trace shim of the MLA flash kernel: the uint8 pool is viewed as FP8 inside the kernel trace, with ``W_V``
    applied before the merge, MXFP8 latent queries, MX value sums and prescaled FP8 attention weights."""
    import nki
    import nki.language as nl

    kernel = kernel_imports.mla_flash_latent()

    @nki.jit(mode="trace")
    def kimi_k3_mla_flash_u8(q_c, q_r, cache, block_table, marker, kc, cbias, ident, prior_tokens, w_v,
                             scale: float = 1.0, heads_per_group: int = 8):
        return kernel(
            q_c, q_r, cache.view(nl.float8_e4m3fn), block_table, marker, kc, cbias, ident, prior_tokens,
            scale=scale, heads_per_group=heads_per_group, fp8_matmul=True, fp8_query="mx", q_layout="feature",
            mx_values=True, w_v=w_v, p_prescale=True, quad_pages=False,
        )

    return kimi_k3_mla_flash_u8
