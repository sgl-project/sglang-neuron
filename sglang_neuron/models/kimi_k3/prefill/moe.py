# SPDX-License-Identifier: Apache-2.0
"""The latent MoE of the Kimi-K3 prefill on one rank, in the sequence-parallel form.

One MoE layer of a stage of ``W`` ranks, with the row block ``x_rows [T / W, H]`` of this rank:

1. One all-gather gives the input rows of all ranks (the shared experts are intermediate-sharded).
2. The router runs on the row block: fp32 logits ``x @ gate_weight_t``, sigmoid; the top-k choice adds
   ``e_score_correction_bias``, the weights are the unbiased scores of the picks, renormalized to sum 1.
3. One all-gather gives the picks ``[weights | ids as fp32]`` of all rows (ids below 2^24 are exact in fp32).
4. The latent down projection runs on the row block, then one all-gather gives the latent of all rows.
5. The shared experts run on all rows (``shared_gate_up``, SiTU, ``shared_down``) and a reduce-scatter by row gives
   their row block. They run before the routed experts; order operands of the experts call keep that order.
6. The weights of the pad rows (at or after ``valid_len``) become 0, so no pad row goes to an expert.
7. The routed experts of this rank (experts ``e_start .. e_start + e_local - 1``) run on all rows.
8. A reduce-scatter by row sums the routed outputs of all ranks; ``routed_expert_norm`` and the latent up
   projection run on the row block; the sum with the shared output is the layer output.

The routed experts have two backends. ``TorchFp4Experts`` keeps the checkpoint MXFP4 bytes and dequantizes each
picked expert to bf16 at the call. ``BlockwiseExperts`` runs the blockwise MXFP4 kernel on the packed experts
(``fp4_pack``) with the block map of this module; its torch stand-in ``bwmm_reference`` computes the kernel contract
on the packed bytes. The block map is expert-major: each expert takes ``ceil(n_e / 128)`` blocks of 128 slots, its
tokens in token order, pad slots -1.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from sglang_neuron.models.kimi_k3.prefill.dense import situ_and_mul
from sglang_neuron.models.kimi_k3.prefill.fp4_pack import fp4_dequant, kernel_slot_values

BLOCK = 128
"""Token slots of one expert block of the blockwise kernel."""


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm with the normalize in fp32, the cast back to the input dtype, then the weight multiply."""
    dtype = x.dtype
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(dtype)


class RmsNorm(nn.Module):
    """An RMSNorm with the weight ``weight`` (``rms_norm``)."""

    def __init__(self, weight: Tensor, eps: float):
        super().__init__()
        self.register_buffer("weight", weight)
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)


def route(x: Tensor, gate_weight_t: Tensor, bias: Tensor, top_k: int, renormalize: bool) -> tuple[Tensor, Tensor]:
    """Pick the top-k experts of each row of ``x [N, H]``; return ``(ids [N, k] int64, weights [N, k] fp32)``.

    The ids are clamped to the expert range, and an id outside it gets weight 0 (a select: its score can be NaN).
    """
    logits = torch.matmul(x.to(torch.float32), gate_weight_t.to(torch.float32))
    scores = logits.sigmoid()
    choice = scores + bias.to(torch.float32).unsqueeze(0)
    topk_idx = torch.topk(choice, k=top_k, dim=-1, sorted=False)[1]
    num_experts = gate_weight_t.shape[1]
    valid = (topk_idx >= 0) & (topk_idx < num_experts)
    topk_idx = topk_idx.clamp(0, num_experts - 1)
    topk_weight = scores.gather(1, topk_idx)
    topk_weight = torch.where(valid, topk_weight, torch.zeros_like(topk_weight))
    if top_k > 1 and renormalize:
        topk_weight = topk_weight / (topk_weight.sum(-1, keepdim=True) + 1e-20)
    return topk_idx, topk_weight


# ---------------------------------------------------------------------------------------------------------------------
# block map (fp32 integer math: every value is below 2^24)
# ---------------------------------------------------------------------------------------------------------------------


def inclusive_row_count(mask: Tensor, tile: int = 128) -> Tensor:
    """Count the set entries of each column up to and including each row, ``[T, E]`` fp32 exact counts.

    The count runs as fp32 matmuls with triangular constants over the token rows (no integer cumsum, whose integer
    transpose the compiler does not lower).

    Example:
        >>> inclusive_row_count(torch.tensor([[True, False], [True, True], [False, True]])).tolist()
        [[1.0, 0.0], [2.0, 1.0], [2.0, 2.0]]
    """
    t, e = mask.shape
    pad = (-t) % tile
    x = mask.to(torch.float32)
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    n = x.shape[0] // tile
    x = x.reshape(n, tile, e)
    lower = torch.tril(torch.ones(tile, tile, dtype=torch.float32, device=mask.device))
    within = torch.matmul(lower, x)
    strict = torch.tril(torch.ones(n, n, dtype=torch.float32, device=mask.device), diagonal=-1)
    offset = torch.matmul(strict, within[:, -1, :])
    return (within + offset[:, None, :]).reshape(n * tile, e)[:t]


def block_budget(tokens: int, top_k: int, num_local: int, block_size: int = BLOCK) -> int:
    """The static worst-case block count of the block map."""
    total = tokens * min(top_k, num_local)
    cap = (total - (num_local - 1) + block_size - 1) // block_size
    return min(max(cap, 1) + num_local - 1, total)


def local_affinities(topk_idx: Tensor, topk_weight: Tensor, local_expert_ids: Tensor) -> Tensor:
    """Return the dense ``[T, E_L]`` fp32 routing weights of the local experts (0 for a remote or unpicked expert).

    Each pick compares with the local ids in fp32 and the weights sum over the picks; a clamped invalid id can match
    a real pick of the same expert, but its weight is 0, so the sum is exact.
    """
    idx = topk_idx.to(torch.float32)
    local = local_expert_ids.reshape(1, -1).to(torch.float32)
    hit = idx[:, :, None] == local[:, None, :]
    return (hit.to(torch.float32) * topk_weight.to(torch.float32)[:, :, None]).sum(1)


def torch_token_runs(sel: Tensor, start: Tensor, num_slots: int) -> Tensor:
    """The torch form of the token-run kernel: the tokens of expert ``e`` in token order from slot ``start[e]``.

    Slots outside the runs hold 0.
    """
    out = torch.zeros(num_slots, dtype=torch.int32, device=sel.device)
    starts = start.reshape(-1).to(torch.int64).tolist()
    for e, first in enumerate(starts):
        tokens = torch.nonzero(sel[:, e] != 0, as_tuple=True)[0].to(torch.int32)
        out[first:first + tokens.numel()] = tokens
    return out


def grouped_token_runs(fn: Callable[..., Tensor], sel: Tensor, start: Tensor, num_slots: int,
                       max_experts: int = 32) -> Tensor:
    """Run a token-run backend on groups of at most ``max_experts`` experts.

    Expert ``e`` writes only its run from ``start[e]``, and the runs follow the expert order, so a slot at or after
    ``start[g]`` belongs to an expert ``>= g``: each slot takes the value of the last group that starts at or before
    it.
    """
    experts = sel.shape[1]
    if experts <= max_experts:
        return fn(sel, start, num_slots)
    out = fn(sel[:, :max_experts], start[:max_experts], num_slots)
    slots = torch.arange(num_slots, dtype=torch.float32, device=sel.device)
    starts = start.reshape(-1).to(torch.float32)
    for g in range(max_experts, experts, max_experts):
        part = fn(sel[:, g:g + max_experts], start[g:g + max_experts], num_slots)
        out = torch.where(slots >= starts[g], part, out)
    return out


def build_block_mapping(affinities: Tensor, block_size: int, num_blocks: int,
                        token_runs_fn: Callable[..., Tensor] | None = None,
                        pad_id: int = -1) -> tuple[Tensor, Tensor]:
    """Sort the (token, local expert) picks into expert-major blocks of ``block_size`` slots.

    Args:
        affinities: ``[T, E_L]`` fp32, nonzero where token ``t`` routes to local expert ``e``.
        block_size: Slots per block.
        num_blocks: The static block budget (``block_budget``).
        token_runs_fn: ``fn(sel [T, E], start [E], num_slots) -> [num_slots] int32`` that writes the tokens of each
            expert to its run of slots (``torch_token_runs`` or the token-run kernel); None scatters one slot per
            (token, expert) pair.
        pad_id: The token id of a pad slot.

    Returns:
        ``(token_position_to_id [num_blocks * block_size] int32, block_to_expert [num_blocks, 1] int32)``.
    """
    t, e = affinities.shape
    device = affinities.device
    f32 = torch.float32
    sel = affinities != 0
    counts = sel.to(f32).sum(dim=0)
    blocks_per_expert = torch.floor((counts + (block_size - 1)) / block_size)
    block_start = torch.cumsum(blocks_per_expert, 0) - blocks_per_expert
    total = num_blocks * block_size
    block_ids = torch.arange(num_blocks, dtype=f32, device=device)
    block_expert = ((block_ids[:, None] >= block_start[None, :]).to(f32).sum(1) - 1).clamp_(0, e - 1)
    block_to_expert = block_expert.to(torch.int32).reshape(num_blocks, 1)
    if token_runs_fn is not None:
        if block_size % 128:
            raise ValueError(f"The block size {block_size} is not a multiple of 128; the token runs write 128-slot "
                             "chunks.")
        raw = grouped_token_runs(token_runs_fn, sel.to(f32), block_start * block_size, total)
        # Block b holds count[e] - (b - block_start[e]) * block_size tokens of its expert e.
        own = (block_expert[:, None] == torch.arange(e, dtype=f32, device=device)[None, :]).to(f32)
        fill = (own * counts[None, :]).sum(1) - (block_ids - (own * block_start[None, :]).sum(1)) * block_size
        real = (torch.arange(block_size, dtype=f32, device=device)[None, :] < fill[:, None]).reshape(-1)
        tpi = torch.where(real, raw.reshape(-1), torch.full_like(raw.reshape(-1), pad_id))
        return tpi.to(torch.int32), block_to_expert
    position = inclusive_row_count(sel) - 1
    slot = block_start[None, :] * block_size + position
    flat_slot = torch.where(sel, slot, torch.full_like(slot, total)).reshape(-1)
    token_ids = torch.arange(t, device=device)[:, None].expand(t, e).reshape(-1)
    tpi = torch.full((total + 1,), pad_id, dtype=torch.int64, device=device)
    tpi.scatter_(0, flat_slot.to(torch.int64), token_ids)
    return tpi[:total].to(torch.int32), block_to_expert


def build_block_conditions(affinities: Tensor, block_size: int, num_blocks: int) -> Tensor:
    """``[num_blocks + 2]`` int32: 1 for the active blocks (a prefix of the block map), 0 after."""
    counts = (affinities != 0).to(torch.float32).sum(dim=0)
    needed = torch.floor((counts + (block_size - 1)) / block_size).sum()
    block_ids = torch.arange(num_blocks + 2, dtype=torch.float32, device=affinities.device)
    return (block_ids < needed).to(torch.int32)


def bwmm_reference(hidden: Tensor, affinities: Tensor, token_position_to_id: Tensor, block_to_expert: Tensor,
                   conditions: Tensor, gate_up_w: Tensor, down_w: Tensor, gate_up_s: Tensor, down_s: Tensor, *,
                   block_size: int, beta: float, linear_beta: float, top_k: int = 0, ep_degree: int = 0,
                   order_first: Tensor | None = None, order_after: Tensor | None = None) -> Tensor:
    """The torch form of the blockwise MXFP4 kernel contract (``kernel_imports.bwmm_mx``), fp32 sums.

    Slot ``(p, t, q)`` of the gate/up contraction takes hidden position ``q (K / 4) + 128 t + p``, slot ``(p, t, q)``
    of the down contraction takes intermediate position ``512 t + 128 q + p``, and each weight slot is
    ``fp4_pack.kernel_slot_values`` of the packed bytes. Each active block runs its expert on its tokens; a pad slot
    (outside ``[0, T)``) is skipped. Returns ``[2, T, D]`` in the hidden dtype with the sum in plane 0.
    """
    del top_k, ep_degree, order_first, order_after
    tokens, k = hidden.shape
    e_local, intermediate = gate_up_w.shape[0], gate_up_w.shape[-1]
    d_out = down_w.shape[-1]
    x = hidden.to(torch.float32).reshape(tokens, 4, k // 512, 128).permute(0, 1, 3, 2)  # [T, q, p, t]
    weights = affinities.reshape(tokens, e_local).to(torch.float32)
    out = torch.zeros(tokens, d_out, dtype=torch.float32)
    active = int(conditions.to(torch.int64).sum())
    current, slots_of = -1, None
    for b in range(active):
        expert = int(block_to_expert[b, 0])
        slots = token_position_to_id[b * block_size:(b + 1) * block_size].to(torch.int64)
        rows = slots[(slots >= 0) & (slots < tokens)]
        if rows.numel() == 0:
            continue
        if expert != current:
            current = expert
            slots_of = (kernel_slot_values(gate_up_w[expert][:, 0], gate_up_s[expert][:, 0]),
                        kernel_slot_values(gate_up_w[expert][:, 1], gate_up_s[expert][:, 1]),
                        kernel_slot_values(down_w[expert], down_s[expert]))
        gate_v, up_v, down_v = slots_of
        xr = x[rows]
        gate = torch.einsum("rqpt,qptn->rn", xr, gate_v)
        up = torch.einsum("rqpt,qptn->rn", xr, up_v)
        act = situ_and_mul(torch.cat([gate, up], dim=-1), beta, linear_beta)
        a = act.reshape(rows.numel(), intermediate // 512, 4, 128).permute(0, 2, 3, 1)  # [r, q, p, t]
        y = torch.einsum("rqpt,qptn->rn", a, down_v)
        out.index_add_(0, rows, y * weights[rows, expert:expert + 1])
    result = torch.zeros(2, tokens, d_out, dtype=torch.float32)
    result[0] = out
    return result.to(hidden.dtype)


# ---------------------------------------------------------------------------------------------------------------------
# routed experts backends
# ---------------------------------------------------------------------------------------------------------------------

TORCH_EXPERT_BUFFERS = (("dq_w1_codes", "dq_w1_scales"), ("dq_w3_codes", "dq_w3_scales"),
                        ("dq_w2_codes", "dq_w2_scales"))
"""The checkpoint MXFP4 buffers of ``TorchFp4Experts``: (codes, scales) of gate (w1), up (w3) and down (w2)."""

KERNEL_EXPERT_BUFFERS = ("mx_gate_up_weights", "mx_gate_up_scales", "mx_down_weights", "mx_down_scales")
"""The packed buffers of ``BlockwiseExperts``."""


class TorchFp4Experts:
    """Routed experts on the checkpoint MXFP4 bytes: each picked expert dequantizes to bf16 at the call.

    ``source`` holds ``TORCH_EXPERT_BUFFERS`` with a leading local-expert axis (codes ``[E_L, I, D / 2]`` /
    ``[E_L, D, I / 2]``, scales ``[E_L, I, D / 32]`` / ``[E_L, D, I / 32]``), ``e_start`` and ``e_local``. The sum
    runs expert by expert in the latent dtype: for expert ``j`` the picking rows get
    ``(SiTU(x W1^T, x W3^T) W2^T) * w`` added.
    """

    def __init__(self, source: nn.Module, beta: float, linear_beta: float):
        self.source = source
        self.beta, self.linear_beta = float(beta), float(linear_beta)

    def weights(self, j: int, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
        """Return the gate, up and down weights of local expert ``j`` in ``dtype``."""
        src = self.source
        return tuple(fp4_dequant(getattr(src, codes)[j], getattr(src, scales)[j]).to(dtype)
                     for codes, scales in TORCH_EXPERT_BUFFERS)

    def __call__(self, latent: Tensor, topk_idx: Tensor, topk_weight: Tensor, order_first: Tensor | None = None,
                 order_after: Tensor | None = None) -> Tensor:
        src = self.source
        y = torch.zeros_like(latent)
        e_start, e_local = src.e_start, src.e_local
        local = (topk_idx >= e_start) & (topk_idx < e_start + e_local)
        for j in range(e_local):
            rows, slots = ((topk_idx == e_start + j) & local).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            w1, w3, w2 = self.weights(j, latent.dtype)
            xin = latent[rows]
            gate_up = torch.cat([xin @ w1.T, xin @ w3.T], dim=-1)
            out = situ_and_mul(gate_up, self.beta, self.linear_beta) @ w2.T
            weight = topk_weight[rows, slots].to(out.dtype).unsqueeze(-1)
            y.index_add_(0, rows, (out * weight).to(y.dtype))
        return y


class BlockwiseExperts:
    """Routed experts on the blockwise MXFP4 kernel (``kernels="device"``) or its torch form (``kernels="torch"``).

    ``source`` holds ``KERNEL_EXPERT_BUFFERS`` (``fp4_pack.pack_expert_*_bwmm``, stacked over the local experts) and
    ``local_expert_ids [1, E_L]`` int64 (the rank enters as tensor data). The latent comes in the kernel column order:
    the loader permutes the rows of the latent down projection (``fp4_pack.bwmm_hidden_order``).
    """

    def __init__(self, source: nn.Module, *, top_k: int, ep_degree: int, beta: float, linear_beta: float,
                 kernels: str = "device"):
        if kernels not in ("device", "torch"):
            raise ValueError(f"The blockwise experts take kernels 'device' or 'torch', not {kernels!r}.")
        self.source = source
        self.top_k, self.ep_degree = int(top_k), int(ep_degree)
        self.beta, self.linear_beta = float(beta), float(linear_beta)
        self.kernels = kernels

    def _backends(self) -> tuple[Callable[..., Tensor], Callable[..., Tensor]]:
        if self.kernels == "torch":
            return torch_token_runs, bwmm_reference
        from sglang_neuron.models.kimi_k3.prefill import kernel_imports

        return kernel_imports.moe_token_runs, kernel_imports.bwmm_mx

    def routing(self, topk_idx: Tensor, topk_weight: Tensor, tokens: int) -> dict[str, Tensor]:
        """Return the routing operands of the kernel: affinities, token map, block experts and active blocks."""
        token_runs, _ = self._backends()
        src = self.source
        affinities = local_affinities(topk_idx, topk_weight, src.local_expert_ids)
        num_blocks = block_budget(tokens, self.top_k, src.local_expert_ids.shape[1], BLOCK)
        tpi, b2e = build_block_mapping(affinities, BLOCK, num_blocks, token_runs_fn=token_runs, pad_id=-1)
        return dict(affinities=affinities.reshape(-1, 1), token_position_to_id=tpi, block_to_expert=b2e,
                    conditions=build_block_conditions(affinities, BLOCK, num_blocks))

    def __call__(self, latent: Tensor, topk_idx: Tensor, topk_weight: Tensor, order_first: Tensor | None = None,
                 order_after: Tensor | None = None) -> Tensor:
        tokens = latent.shape[0]
        if tokens % 128:
            raise ValueError(f"The blockwise experts take a multiple of 128 rows; got {tokens}.")
        _, bwmm = self._backends()
        src = self.source
        out = bwmm(latent.to(torch.bfloat16), **self.routing(topk_idx, topk_weight, tokens),
                   gate_up_w=src.mx_gate_up_weights, down_w=src.mx_down_weights, gate_up_s=src.mx_gate_up_scales,
                   down_s=src.mx_down_scales, block_size=BLOCK, beta=self.beta, linear_beta=self.linear_beta,
                   top_k=self.top_k, ep_degree=self.ep_degree, order_first=order_first, order_after=order_after)
        return out[0, :tokens].to(latent.dtype)


# ---------------------------------------------------------------------------------------------------------------------
# the MoE layer
# ---------------------------------------------------------------------------------------------------------------------


class LatentMoe(nn.Module):
    """One latent MoE layer on one rank (module docstring).

    Weights: ``gate_weight_t [H, E]`` bf16 (K-major router), ``e_score_correction_bias [E]`` fp32,
    ``routed_expert_down_proj`` / ``routed_expert_up_proj`` / ``shared_gate_up`` / ``shared_down`` (``MxLinear``),
    ``routed_expert_norm.weight [D]``, and the expert buffers of the experts backend (``TORCH_EXPERT_BUFFERS`` or
    ``KERNEL_EXPERT_BUFFERS`` + ``local_expert_ids``). ``experts_fn`` is the routed experts backend.

    Example:
        >>> y_rows = moe(x_rows, comm, valid_len=torch.tensor([1100]))  # [T / W, H] -> [T / W, H]
    """

    def __init__(self, *, gate_weight_t: Tensor, e_score_correction_bias: Tensor, routed_expert_down_proj: nn.Module,
                 routed_expert_up_proj: nn.Module, routed_expert_norm: Tensor, shared_gate_up: nn.Module,
                 shared_down: nn.Module, e_start: int, e_local: int, top_k: int, renormalize: bool, situ_beta: float,
                 situ_linear_beta: float, norm_eps: float):
        super().__init__()
        self.register_buffer("gate_weight_t", gate_weight_t)
        self.register_buffer("e_score_correction_bias", e_score_correction_bias)
        self.routed_expert_down_proj = routed_expert_down_proj
        self.routed_expert_up_proj = routed_expert_up_proj
        self.routed_expert_norm = RmsNorm(routed_expert_norm, norm_eps)
        self.shared_gate_up = shared_gate_up
        self.shared_down = shared_down
        self.e_start, self.e_local = int(e_start), int(e_local)
        self.top_k, self.renormalize = int(top_k), bool(renormalize)
        self.situ_beta, self.situ_linear_beta = float(situ_beta), float(situ_linear_beta)
        self.experts_fn: Callable[..., Tensor] | None = None

    def use_torch_experts(self, buffers: dict[str, Tensor]) -> None:
        """Hold the checkpoint MXFP4 bytes of the local experts and run ``TorchFp4Experts``."""
        for name, value in buffers.items():
            self.register_buffer(name, value)
        self.experts_fn = TorchFp4Experts(self, self.situ_beta, self.situ_linear_beta)

    def use_blockwise_experts(self, buffers: dict[str, Tensor], ep_degree: int, kernels: str = "device") -> None:
        """Hold the packed experts and ``local_expert_ids``, and run ``BlockwiseExperts``."""
        for name, value in buffers.items():
            self.register_buffer(name, value)
        self.register_buffer("local_expert_ids",
                             torch.arange(self.e_start, self.e_start + self.e_local, dtype=torch.int64).unsqueeze(0))
        self.experts_fn = BlockwiseExperts(self, top_k=self.top_k, ep_degree=ep_degree, beta=self.situ_beta,
                                           linear_beta=self.situ_linear_beta, kernels=kernels)

    def route(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """The router of ``x [N, H]``: ``(ids [N, k], weights [N, k] fp32)``."""
        return route(x, self.gate_weight_t, self.e_score_correction_bias, self.top_k, self.renormalize)

    def shared_experts(self, x_full: Tensor) -> Tensor:
        """The shared experts of this rank on all rows: the partial ``[T, H]``."""
        gate_up = self.shared_gate_up(x_full)
        return self.shared_down(situ_and_mul(gate_up, self.situ_beta, self.situ_linear_beta))

    def forward(self, x_rows: Tensor, comm, valid_len: Tensor | None = None, row_sharded: bool = True) -> Tensor:
        """Run the layer on the row block ``x_rows [T / W, H]``; give the row block of the output.

        ``valid_len`` (``[1]`` int, optional) is the count of real rows. ``row_sharded=False`` takes all ``T`` rows on
        every rank and gives all rows; with ``W > 1`` and ``T % W == 0`` each rank still runs the router and the latent
        projections on its own ``T / W`` rows.
        """
        world = comm.world
        tokens = x_rows.shape[0] * world if row_sharded else x_rows.shape[0]
        sharded = row_sharded or (world > 1 and tokens % world == 0)
        if sharded:
            if row_sharded:
                x_local, x_full = x_rows, comm.all_gather(x_rows, 0)
            else:
                x_full = x_rows
                n = tokens // world
                rows = comm.rank_tensor.reshape(-1)[:1].to(torch.int64) * n + torch.arange(n, device=x_rows.device)
                x_local = x_rows.index_select(0, rows)
            idx_local, weight_local = self.route(x_local)
            k = idx_local.shape[1]
            picks = comm.all_gather(torch.cat([weight_local, idx_local.to(torch.float32)], dim=1), 0)
            topk_weight, topk_idx = picks[:, :k], picks[:, k:].to(torch.int64)
            latent = comm.all_gather(self.routed_expert_down_proj(x_local), 0)
            shared = comm.reduce_scatter(self.shared_experts(x_full), 0)
            order = dict(order_first=x_full[:1], order_after=shared)
        else:
            topk_idx, topk_weight = self.route(x_rows)
            latent = self.routed_expert_down_proj(x_rows)
            order = {}
        if valid_len is not None:
            valid = torch.arange(tokens, device=x_rows.device) < valid_len.reshape(-1)[:1].to(torch.int64)
            topk_weight = torch.where(valid[:, None], topk_weight, torch.zeros_like(topk_weight))
        y = self.experts_fn(latent, topk_idx, topk_weight, **order)
        if sharded:
            y = comm.reduce_scatter(y, 0)
            y = self.routed_expert_up_proj(self.routed_expert_norm(y))
            out = y + shared
            return out if row_sharded else comm.all_gather(out, 0)
        y = self.routed_expert_up_proj(self.routed_expert_norm(comm.all_reduce(y)))
        return y + comm.all_reduce(self.shared_experts(x_rows))
