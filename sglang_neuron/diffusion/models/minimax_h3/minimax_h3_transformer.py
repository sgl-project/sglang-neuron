# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 DiT (diffusers 0.40.0 ``MiniMaxH3Transformer3DModel``) for Neuron.

TP on raw ``nn.Parameter`` with the shared loaders and NKI kernels. The block stack runs on each
rank's CP slice of the packed sequence, 1/TP of it per rank under ``tp_sequence_parallel``.

Single-device options (TP4, CP1 on one trn2 device), all off by default:

* ``rope_by_matmul``: the partial rotate-half RoPE as ``x * cos' + (x @ R) * sin'``, bit-exact,
  instead of strided slices and concatenations.
* ``attention_heads_per_call``: at CP1, attention over that many heads per kernel launch.
* ``compile_per_block``: the DiT as a prologue, one shared block graph and an epilogue.
* ``adaln_on_host``: the blocks' AdaLN projections stay on the host; ``block_modulations`` runs
  there.
"""

import copy
import logging
import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3RotaryPosEmbed,
    _apply_rotary_emb,
)
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel as DiffusersMiniMaxH3Transformer3DModel,
)

from sglang_neuron.diffusion.attention.cp_attention import nki_attention, o_proj
from sglang_neuron.diffusion.distributed.groups import (
    get_cp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang_neuron.diffusion.distributed.parallel_state import (
    get_cp_replica_groups,
    register_replica_groups,
)
from sglang_neuron.diffusion.distributed.tp_loaders import (
    cast_weight_loader,
    col_bias_loader,
    col_weight_loader,
    row_weight_loader,
)
from sglang_neuron.diffusion.models.minimax_h3.ring_attention import cp_ring_self_attention
from sglang_neuron.diffusion.models.minimax_h3.sequence_parallel import (
    sp_entry,
    sp_exit,
    sp_padded_len,
)
from sglang_neuron.diffusion.quantization.adaln_kernels import adaln_modulate
from sglang_neuron.diffusion.vn.functional.mlp import mlp
from sglang_neuron.diffusion.vn.utils.hardware_config import get_logical_nc_config
from sglang_neuron.diffusion.vn.utils.shard_loaders import fused_qkv_weight_loader
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import (
    get_weight_loader,
    set_weight_loader,
    with_rank_override,
)

logger = logging.getLogger(__name__)

# Largest score bound ``sqrt(head_dim) * max|g_q| * max|g_k|`` the ring kernel's const max
# tolerates. Blocks whose QK-norm gains exceed it (H3's 45, 48 and 49) attend over gathered K/V.
_RING_MAX_SCORE_NATS = 255.0

_FP32_MODULES = frozenset(DiffusersMiniMaxH3Transformer3DModel._keep_in_fp32_modules)


@dataclass
class MiniMaxH3Config:
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    # Not diffusers config fields.
    tp_sequence_parallel: bool = True
    # Reuse a schedule's block_modulations across requests (NeuronMiniMaxH3Model.denoise).
    adaln_cache: bool = True
    # One device (TP4, CP1): the 50 blocks' AdaLN projections (13B parameters, 6.5 GB per rank at
    # TP4) stay on the host, and block_modulations runs there; the DiT never holds them in HBM.
    adaln_on_host: bool = False
    # One device: the DiT runs as a prologue, one graph per transformer block and an epilogue.
    # The blocks are identical, so they share one NEFF, and the whole-model graph's working set
    # (which does not fit one device at 768p) is never built.
    compile_per_block: bool = False
    # Partial rotate-half RoPE as one full-head-width expression, x * cos' + (x @ R) * sin', with R
    # the signed rotate-half permutation of the rotary channels and cos'/sin' padded with 1/0.
    # Bit-exact against the slice/cat form; the compiler lowers it as a [*, D] x [D, D] matmul and
    # two elementwise ops instead of strided slices and concatenations, which at 768p on one device
    # are ~145K tiny DMAs and ~170 ms of each block. Single device (TP=4, CP=1) only: at TP8 x CP8
    # each core holds 1/16 of those rows x heads and the extra matmul makes denoise 55 % slower.
    rope_by_matmul: bool = False
    # CP=1: attention over this many heads per kernel launch instead of all of them (0 = all). The
    # kernel is Scalar/Vector-bound at 37K rows; smaller launches throttle less.
    attention_heads_per_call: int = 0

    def __post_init__(self):
        assert 2 * MINIMAX_H3_MODALITY_NUM * self.rope_freq_dim <= self.attention_head_dim


class MiniMaxH3SwiGLU(nn.Module):
    """Upstream ``FeedForward(activation_fn="swiglu", bias=False)``, TP-sharded on
    ``vn.functional.mlp``, which computes ``silu(x @ gate) * (x @ up)``."""

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if ffn_dim % tp_size != 0:
            raise ValueError(f"ffn_dim {ffn_dim} is not divisible by TP {tp_size}.")
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        inner_per_rank = ffn_dim // tp_size

        # ff.net.0.proj is [value(ffn); gate(ffn)], i.e. 2*TP column shards: rank r's value
        # (up_proj) is shard r, its gate shard TP + r.
        proj_loader = col_weight_loader(inner_per_rank, 2 * tp_size)
        tp_rank = get_tensor_model_parallel_rank()
        self.gate_proj_weight = nn.Parameter(torch.empty(dim, inner_per_rank))
        set_weight_loader(self.gate_proj_weight, with_rank_override(proj_loader, tp_size + tp_rank))
        self.up_proj_weight = nn.Parameter(torch.empty(dim, inner_per_rank))
        set_weight_loader(self.up_proj_weight, with_rank_override(proj_loader, tp_rank))
        self.down_proj_weight = nn.Parameter(torch.empty(inner_per_rank, dim))
        set_weight_loader(self.down_proj_weight, row_weight_loader(inner_per_rank, tp_size))

    def forward(self, x: torch.Tensor, sp_lengths: tuple[int, int] | None = None) -> torch.Tensor:
        # SP: rows are independent, so the padded gathered rows go through untrimmed.
        if sp_lengths is not None:
            x = sp_entry(x, self.tp_group, self.tp_size, True)
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        num_rows = x.shape[0]
        # The mlp CTE kernel splits the rows over the LNC cores and fails validation when they do
        # not divide, which its eligibility check misses (the refiner's and an unsplit
        # sequence's rows can be odd). Pad, then drop the pad rows.
        pad = -num_rows % get_logical_nc_config()
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        out = mlp(
            x,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )
        out = out[:num_rows].reshape(*shape[:-1], out.shape[-1])
        return sp_exit(out, self.tp_group, self.tp_size, sp_lengths is not None)


#: Rows per RoPE matmul: one flat (rows, D) x (D, D) over every head (530K rows at TP4) overflows
#: an access-pattern count in the backend (NCC_IBTN021).
_ROPE_CHUNK_ROWS = 32768


def _rope_rotate_matrix(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """``R`` with ``x @ R`` = rotate-half of ``x``'s first ``rotary_dim`` channels, 0 elsewhere."""
    half = rotary_dim // 2
    mat = torch.zeros(head_dim, head_dim)
    index = torch.arange(half)
    mat[index + half, index] = -1.0  # out[i]        = -x[i + half]
    mat[index, index + half] = 1.0  # out[i + half] =  x[i]
    return mat


def _rope_by_matmul(hidden_states, cos, sin, rotate):
    """``_apply_rotary_emb`` for ``[..., S, N, D]``, with ``cos``/``sin`` ``(S, rotary_dim)``."""
    head_dim = hidden_states.shape[-1]
    pad = head_dim - cos.shape[-1]
    dtype = hidden_states.dtype
    cos = torch.cat([cos, cos.new_ones(cos.shape[0], pad)], dim=-1).to(dtype)[:, None, :]
    sin = torch.cat([sin, sin.new_zeros(sin.shape[0], pad)], dim=-1).to(dtype)[:, None, :]
    flat = hidden_states.reshape(-1, head_dim)
    rotate = rotate.to(dtype)
    rotated = torch.cat(
        [torch.matmul(piece, rotate) for piece in torch.split(flat, _ROPE_CHUNK_ROWS, dim=0)],
        dim=0,
    ).reshape(hidden_states.shape)
    return hidden_states * cos + rotated * sin


class MiniMaxH3Attention(nn.Module):
    """Upstream ``MiniMaxH3Attention``, TP-sharded (column fused Q/K/V, row o-proj).

    ``use_cp=False`` is for the text refiner, which runs before the CP split.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        use_cp: bool = True,
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(f"num_attention_heads {num_heads} is not divisible by TP {tp_size}.")
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        self.num_heads = num_heads // tp_size
        self.head_dim = head_dim
        tp_inner = self.num_heads * head_dim

        cp_group = get_cp_group()
        self.cp_size = cp_group.world_size if use_cp else 1
        self.cp_group = cp_group if self.cp_size > 1 else None
        self.cp_replica_groups = None
        if self.cp_size > 1:
            self.cp_replica_groups = get_cp_replica_groups(tp_size, self.cp_size)

        self.qkv_split = [tp_inner, 2 * tp_inner]
        self.qkv_proj_weight = nn.Parameter(torch.empty(dim, 3 * tp_inner))
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=tp_inner,
                kv_size=tp_inner,
                shard_dim=1,
                num_shards=tp_size,
                is_storage_transposed=True,
            ),
        )

        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)

        self.o_proj_weight = nn.Parameter(torch.empty(tp_inner, dim))
        set_weight_loader(self.o_proj_weight, row_weight_loader(tp_inner, tp_size))
        # The checkpoint's to_out has no bias; o_proj takes one.
        self.register_buffer("o_proj_bias", torch.zeros(1, dim), persistent=False)

        self.scale = 1.0 / (head_dim**0.5)
        self.ring_bound_holds = True
        self.heads_per_call = 0  # attention_heads_per_call; set by the model
        # rope_by_matmul's rotate matrix (see use_rope_by_matmul); None keeps the slice/cat RoPE.
        self.register_buffer("rope_rotate", None, persistent=False)

    def use_rope_by_matmul(self, rotary_dim: int) -> None:
        """``rope_by_matmul``: a ``(D, D)`` buffer, so every block graph takes it as an input."""
        self.register_buffer(
            "rope_rotate", _rope_rotate_matrix(self.head_dim, rotary_dim), persistent=False
        )

    def update_ring_bound(self) -> None:
        g_q = self.norm_q.weight.detach().to("cpu", torch.float32).abs().max()
        g_k = self.norm_k.weight.detach().to("cpu", torch.float32).abs().max()
        self.ring_bound_holds = float(self.head_dim**0.5 * g_q * g_k) <= _RING_MAX_SCORE_NATS

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb=None,
        kv_valid=None,
        num_pad: int = 0,
        sp_lengths: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        # SP: gather the TP group's rows and drop the SP padding, so pad rows are never keys.
        if sp_lengths is not None:
            hidden_states = sp_entry(
                hidden_states, self.tp_group, self.tp_size, True, sp_lengths[0]
            )
        # A matmul: the fused width (3 x 7168 / TP) does not fit vn.functional.qkv_proj's NKI
        # kernel in SBUF at any TP size the 56 heads allow on 64 cores.
        qkv = torch.matmul(hidden_states, self.qkv_proj_weight)
        q, k, v = (
            t.unflatten(2, (self.num_heads, self.head_dim))
            for t in torch.tensor_split(qkv, self.qkv_split, dim=-1)
        )

        q = self.norm_q(q)
        k = self.norm_k(k)

        if rotary_emb is not None:
            cos, sin = rotary_emb
            if self.rope_rotate is not None:
                q = _rope_by_matmul(q, cos, sin, self.rope_rotate)
                k = _rope_by_matmul(k, cos, sin, self.rope_rotate)
            else:
                q = _apply_rotary_emb(q, cos, sin)
                k = _apply_rotary_emb(k, cos, sin)

        # Both paths return d-major [B, N, D, S], the layout o_proj takes. Without CP they are the
        # same attention, and one path keeps every block on one graph (compile_per_block).
        if self.cp_size == 1 and self.heads_per_call and self.heads_per_call < q.shape[2]:
            # [B, S, N, D] -> [B, N, S, D], one kernel launch per head group.
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            group = self.heads_per_call
            attn = torch.cat(
                [
                    nki_attention(
                        q[:, i : i + group], k[:, i : i + group], v[:, i : i + group], self.scale
                    )
                    for i in range(0, q.shape[1], group)
                ],
                dim=1,
            )
        elif self.ring_bound_holds or self.cp_size == 1:
            attn = cp_ring_self_attention(
                q,
                k,
                v,
                self.scale,
                self.cp_size,
                self.cp_group,
                self.cp_replica_groups,
                kv_valid,
                num_pad,
            )
        else:
            # Keys are rotated before the gather. Token-major, the pad rows end the gathered
            # sequence.
            if self.cp_size > 1:
                k = self.cp_group.all_gather(k.contiguous(), dim=1)
                v = self.cp_group.all_gather(v.contiguous(), dim=1)
            if num_pad:
                k, v = k[:, :-num_pad], v[:, :-num_pad]
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            attn = nki_attention(q, k, v, self.scale)

        out = o_proj(attn, self.o_proj_weight, self.o_proj_bias)
        if sp_lengths is not None:
            return sp_exit(out, self.tp_group, self.tp_size, True, sp_lengths[1])
        return sp_exit(out, self.tp_group, self.tp_size, False)


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    """Upstream ``MiniMaxH3AdaLayerNormModulation``, column-sharded over the TP x CP ranks of one
    replica: it depends on ``temb`` alone, and at 96768 outputs is the block's largest tensor.

    Rank ``(cp_rank, tp_rank)`` holds shard ``cp_rank * tp + tp_rank``, so gathering over TP,
    then CP, restores the column order. The two groups are exactly the ranks that share a
    request's timesteps; the world group would also take in other CFG or DP replicas.
    """

    def __init__(self, time_embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_coord = get_tp_group()
        self.cp_coord = get_cp_group()
        tp_size, cp_size = self.tp_coord.world_size, self.cp_coord.world_size
        num_shards = tp_size * cp_size
        out_features = 6 * hidden_size * MINIMAX_H3_MODALITY_NUM
        assert out_features % num_shards == 0, (
            f"adaln_proj out_features={out_features} not divisible by TP x CP {num_shards}"
        )
        out_per_rank = out_features // num_shards
        shard = self.cp_coord.rank_in_group * tp_size + self.tp_coord.rank_in_group
        self.num_shards = num_shards
        self.weight = nn.Parameter(torch.empty(time_embed_dim, out_per_rank))
        set_weight_loader(
            self.weight, with_rank_override(col_weight_loader(out_per_rank, num_shards), shard)
        )
        self.bias = nn.Parameter(torch.empty(out_per_rank))
        set_weight_loader(
            self.bias, with_rank_override(col_bias_loader(out_per_rank, num_shards), shard)
        )

    def gather(self, local: torch.Tensor, dim: int) -> torch.Tensor:
        """Every shard's ``local`` along ``dim``, in shard order."""
        if self.tp_coord.world_size > 1:
            local = self.tp_coord.all_gather(local, dim=dim)
        if self.cp_coord.world_size > 1:
            local = self.cp_coord.all_gather(local, dim=dim)
        return local

    def local(self, temb: torch.Tensor) -> torch.Tensor:
        """This rank's column shard of the projection."""
        return torch.matmul(F.silu(temb).to(self.weight.dtype), self.weight) + self.bias

    def split(self, temb: torch.Tensor):
        """The gathered projection as the six modulation tensors."""
        temb = temb.reshape(-1, 6 * self.hidden_size)
        return temb.chunk(6, dim=-1)


def _regroup_block_columns(
    gathered: torch.Tensor, num_shards: int, num_blocks: int
) -> tuple[torch.Tensor, ...]:
    """``(shards * T, num_blocks * C)`` -> one ``(T, shards * C)`` tensor per block, columns in
    shard order.

    Built from slices only: gathering along the last dim or permuting is a DRAM-to-DRAM
    transpose, which neuronx-cc fails at a keyframe request's three timestep rows
    ([NCC_IDDT901]).
    """
    rows = gathered.shape[0] // num_shards
    by_rank = gathered.reshape(num_shards, rows, num_blocks, -1)
    return tuple(
        torch.cat([by_rank[:, row, block] for row in range(rows)]).reshape(rows, -1)
        for block in range(num_blocks)
    )


def _local_rows(
    seq_len: int,
    device: torch.device,
    cp_size: int,
    cp_rank: torch.Tensor | int,
    tp_size: int = 1,
    tp_rank: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int] | None]:
    """This rank's CP slice of the packed sequence, the rows it keeps and, under SP (``tp_rank``
    given), the CP slice's ``(real, padded)`` length.

    Under SP a rank keeps 1/TP of the CP slice padded to a multiple of TP; the pad rows repeat
    the slice's last row and never reach Q/K/V.
    """
    if seq_len % cp_size:
        raise ValueError(f"The packed sequence ({seq_len} rows) does not divide CP {cp_size}.")
    cp_len = seq_len // cp_size
    cp_start = cp_rank * cp_len
    cp_rows = cp_start + torch.arange(cp_len, device=device)
    if tp_rank is None:
        return cp_rows, cp_rows, None
    padded_len = sp_padded_len(cp_len, tp_size)
    sp_len = padded_len // tp_size
    sp_rows = (tp_rank * sp_len + torch.arange(sp_len, device=device)).clamp(max=cp_len - 1)
    return cp_rows, cp_start + sp_rows, (cp_len, padded_len)


def _packed_sources(
    text_indices: torch.Tensor, video_indices: torch.Tensor, audio_indices: torch.Tensor
) -> torch.Tensor:
    """Each packed row's index into ``cat([text, video, audio])``: the inverse of upstream's
    ``index_copy`` packing, for any layout, including interleaved ones."""
    order = torch.cat([text_indices, video_indices, audio_indices])
    positions = torch.arange(order.numel(), device=order.device, dtype=order.dtype)
    return torch.zeros_like(order).index_copy_(0, order, positions)


class MiniMaxH3Block(nn.Module):
    """Upstream ``MiniMaxH3TransformerBlock`` built from the TP modules above."""

    def __init__(self, cfg: MiniMaxH3Config):
        super().__init__()
        dim = cfg.hidden_size
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(cfg.time_embed_dim, dim)
        self.norm1 = nn.RMSNorm(dim, eps=cfg.norm_eps)
        self.norm2 = nn.RMSNorm(dim, eps=cfg.norm_eps)
        self.attn = MiniMaxH3Attention(
            dim, cfg.num_attention_heads, cfg.attention_head_dim, eps=cfg.qk_norm_eps
        )
        self.ff = MiniMaxH3SwiGLU(dim, cfg.ffn_dim)

    def forward(
        self,
        hidden_states,
        modulation,
        adaln_indices,
        rotary_emb,
        kv_valid=None,
        num_pad=0,
        sp_lengths=None,
    ):
        """``modulation``: this block's six AdaLN tensors. ``kv_valid`` / ``num_pad``: see
        ``cp_ring_self_attention``. ``sp_lengths``: ``(real, padded)`` CP-slice length under SP,
        else None.

        The norms and modulation run the AdaLN NKI kernel, the attention's gated residual add
        fused into norm2. Both residual adds accumulate in FP32 and round once to BF16 (upstream
        adds in BF16).

        ``modulation`` holds one row per (timestep, modality) -- a handful of rows, not one per
        token -- and ``adaln_indices`` picks each token's row. The kernel gathers those rows per
        token tile (``row_indices``), so the five tensors it consumes are never expanded to the
        full ``(rows, hidden)`` in DRAM. The MLP gate is the exception: its residual add is a
        framework op, so that one row set is still expanded here.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

        h = adaln_modulate(
            hidden_states,
            self._mul(self.norm1, scale_msa),
            shift_msa,
            eps=self.norm1.eps,
            norm_type="rms_norm",
            fold_mul=False,
            row_indices=adaln_indices,
        )
        attn_out = self.attn(h, rotary_emb, kv_valid, num_pad, sp_lengths)
        hidden_states, h = adaln_modulate(
            attn_out,
            self._mul(self.norm2, scale_mlp),
            shift_mlp,
            eps=self.norm2.eps,
            norm_type="rms_norm",
            residual=hidden_states,
            gate=gate_msa,
            row_indices=adaln_indices,
        )
        ff_out = self.ff(h, sp_lengths)
        gate_mlp = gate_mlp.index_select(0, adaln_indices)
        # Not fused: it would carry into the next block's norm1.
        return (hidden_states.float() + gate_mlp.float() * ff_out.float()).type_as(hidden_states)

    @staticmethod
    def _mul(norm, scale):
        """``norm``'s gain folded into ``1 + scale``: the kernel's RMSNorm has no weight."""
        return (norm.weight.float() * (1.0 + scale.float())).type_as(scale)


class _RefinerBlock(nn.Module):
    """Upstream ``MiniMaxH3TokenRefinerBlock`` built from the TP modules above."""

    def __init__(self, cfg: MiniMaxH3Config):
        super().__init__()
        dim = cfg.hidden_size
        self.norm1 = nn.RMSNorm(dim, eps=cfg.norm_eps)
        self.attn = MiniMaxH3Attention(
            dim, cfg.num_attention_heads, cfg.attention_head_dim, eps=cfg.qk_norm_eps, use_cp=False
        )
        self.norm2 = nn.RMSNorm(dim, eps=cfg.norm_eps)
        self.ff = MiniMaxH3SwiGLU(dim, cfg.ffn_dim)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.ff(self.norm2(hidden_states))


class _TokenRefiner(nn.Module):
    """Upstream ``MiniMaxH3TokenRefiner`` built from ``_RefinerBlock``."""

    def __init__(self, cfg: MiniMaxH3Config):
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [_RefinerBlock(cfg) for _ in range(cfg.num_refiner_layers)]
        )
        self.final_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.final_norm_eps)

    def forward(self, hidden_states):
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


#: compile_per_block: block graphs launched before the oldest one's probe is read back (see
#: ``_Probed``). Each launch in flight keeps its input and output hidden states (0.4 GB each at
#: 768p on one device), so stay close to one.
_MAX_BLOCKS_IN_FLIGHT = 2


class _Probed(nn.Module):
    """``block``'s output plus one element of it, for the host to read back (see
    ``MiniMaxH3Transformer3DModel._compile_stages``).

    Takes the block's gathered modulation tensor whole and splits it inside the graph: a graph
    input must be a contiguous device tensor, which ``split``'s chunks are not."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, modulation, *args):
        out = self.block(hidden_states, self.block.adaln_proj.split(modulation), *args)
        return out, out.reshape(-1)[:1] * 1


class MiniMaxH3Transformer3DModel(nn.Module):
    """Neuron MiniMax-H3 DiT (the checkpoint's ``transformer``)."""

    def __init__(self, **kwargs):
        super().__init__()
        # Drop diffusers-only config keys.
        known = {f for f in MiniMaxH3Config.__dataclass_fields__ if not f.startswith("_")}
        cfg_kwargs = {
            k: (tuple(v) if k == "patch_size" else v) for k, v in kwargs.items() if k in known
        }
        self.config = MiniMaxH3Config(**cfg_kwargs)
        cfg = self.config
        dim = cfg.hidden_size

        cp_group = get_cp_group()
        self.cp_size = cp_group.world_size
        self.cp_group = cp_group if self.cp_size > 1 else None
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_coord = get_tp_group()
        self.sequence_parallel = cfg.tp_sequence_parallel and self.tp_size > 1

        register_replica_groups(
            tp_size=get_tensor_model_parallel_world_size(),
            cp_size=self.cp_size,
        )

        video_in = cfg.in_channels * math.prod(cfg.patch_size)
        self.proj_in = nn.Linear(video_in, dim, bias=True)
        self.audio_proj_in = nn.Linear(cfg.audio_in_channels, dim, bias=True)
        self.context_embedder = nn.Linear(cfg.text_dim, dim, bias=True)
        self.time_proj = Timesteps(
            num_channels=cfg.freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=cfg.freq_dim,
            time_embed_dim=cfg.time_embed_hidden_dim,
            out_dim=cfg.time_embed_dim,
        )
        self.rope = MiniMaxH3RotaryPosEmbed(cfg.rope_freq_dim, cfg.rope_theta)

        self.token_refiner = _TokenRefiner(cfg)

        self.transformer_blocks = nn.ModuleList(
            [MiniMaxH3Block(cfg) for _ in range(cfg.num_layers)]
        )
        if cfg.rope_by_matmul:
            rotary_dim = 2 * MINIMAX_H3_MODALITY_NUM * cfg.rope_freq_dim
            for block in self.transformer_blocks:
                block.attn.use_rope_by_matmul(rotary_dim)
        for block in self.transformer_blocks:
            block.attn.heads_per_call = int(cfg.attention_heads_per_call or 0)

        self.norm_out = MiniMaxH3AdaLayerNormOut(
            hidden_size=dim, time_embed_dim=cfg.time_embed_dim, eps=cfg.final_norm_eps
        )
        self.proj_out = nn.Linear(dim, video_in, bias=True)
        self.audio_proj_out = nn.Linear(dim, cfg.audio_in_channels, bias=True)
        # compile_per_block's compiled stages (see _compile_stages).
        self._stages = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        block_modulations: tuple[torch.Tensor, ...],
        return_dict: bool = True,
    ):
        """Upstream ``MiniMaxH3Transformer3DModel.forward``, with the block stack under CP.

        ``block_modulations`` is ``self.block_modulations(timestep)``, run as its own graph.
        Each rank gathers and projects only the input rows it keeps, rather than scattering
        every modality into a full packed sequence.
        """
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"`position_ids` must be (seq_len, 3), got {list(position_ids.shape)}."
            )
        num_pad, sp_lengths = self._static_geometry(position_ids.shape[0])
        if self._stages is not None:
            return self._forward_staged(
                hidden_states,
                audio_hidden_states,
                encoder_hidden_states,
                timestep,
                timestep_indices,
                token_tags,
                position_ids,
                video_indices,
                audio_indices,
                text_indices,
                block_modulations,
                num_pad,
                sp_lengths,
                return_dict,
            )
        hidden_states, temb, adaln_indices, timestep_indices, rotary_emb, kv_valid = self._prologue(
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
        )
        for block, modulation in zip(self.transformer_blocks, block_modulations, strict=True):
            hidden_states = block(
                hidden_states,
                block.adaln_proj.split(modulation),
                adaln_indices,
                rotary_emb,
                kv_valid,
                num_pad,
                sp_lengths,
            )
        return self._finish(
            self._epilogue(
                hidden_states, temb, timestep_indices, video_indices, audio_indices, sp_lengths
            ),
            return_dict,
        )

    def _static_geometry(self, seq_len: int) -> tuple[int, tuple[int, int] | None]:
        """The pad count and, under SP, the CP slice's ``(real, padded)`` length: Python values
        of the sequence length alone, which every stage specializes on."""
        # Pad the packed sequence to a multiple of CP x LNC, so each rank's slice splits evenly
        # over its LNC cores. Rows past seq_len are zero, at position 0, and never keys.
        num_pad = -seq_len % (self.cp_size * (get_logical_nc_config() if self.cp_size > 1 else 1))
        if not self.sequence_parallel:
            return num_pad, None
        cp_len = (seq_len + num_pad) // self.cp_size
        return num_pad, (cp_len, sp_padded_len(cp_len, self.tp_size))

    def _prologue(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
    ):
        """Everything before the block stack: the rows this rank keeps, RoPE, the text refiner,
        the patch projections and ``temb``."""
        seq_len = position_ids.shape[0]
        num_pad, _ = self._static_geometry(seq_len)
        if num_pad:
            position_ids = F.pad(position_ids, (0, 0, 0, num_pad))
        # Rank tensors, not ints, so every rank traces the same graph.
        cp_rank = 0
        if self.cp_size > 1:
            cp_rank = self.cp_group.rank_in_group_tensor.to(torch.int64)
        tp_rank = None
        if self.sequence_parallel:
            tp_rank = self.tp_coord.rank_in_group_tensor.to(torch.int64)
        # RoPE covers the CP slice attention runs on; the blocks carry `rows`, 1/TP of it under SP.
        cp_rows, rows, _ = _local_rows(
            position_ids.shape[0], position_ids.device, self.cp_size, cp_rank, self.tp_size, tp_rank
        )
        kv_valid = None
        if num_pad:
            kv_valid = (cp_rows < seq_len).to(encoder_hidden_states.dtype)[None]
        real_rows = rows.clamp(max=seq_len - 1)

        rotary_emb = self.rope(position_ids.index_select(0, cp_rows))

        text_embeds = self.context_embedder(
            encoder_hidden_states.to(self.context_embedder.weight.dtype)
        )
        text_embeds = self.token_refiner(text_embeds)
        sources = _packed_sources(text_indices, video_indices, audio_indices).index_select(
            0, real_rows
        )
        hidden_states = self._embed_rows(sources, text_embeds, hidden_states, audio_hidden_states)
        if num_pad:
            hidden_states = hidden_states * (rows < seq_len).to(hidden_states.dtype)[None, :, None]

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))

        timestep_indices = timestep_indices.index_select(0, real_rows)
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.index_select(
            0, real_rows
        )
        return hidden_states, temb, adaln_indices, timestep_indices, rotary_emb, kv_valid

    def _epilogue(
        self, hidden_states, temb, timestep_indices, video_indices, audio_indices, sp_lengths=None
    ):
        """``norm_out``, the output heads and the gathers back to the full sequence."""
        # norm_out and the heads are per row, so the ranks gather the heads' outputs rather
        # than the 42x wider hidden states.
        hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(
            self.proj_out.weight.dtype
        )
        output = torch.cat(
            [self.proj_out(hidden_states), self.audio_proj_out(hidden_states)], dim=-1
        )
        if self.sequence_parallel:
            output = sp_entry(output, self.tp_coord.device_group, self.tp_size, True, sp_lengths[0])
        if self.cp_size > 1:
            output = self.cp_group.all_gather(output.contiguous(), dim=1)
        video_output, audio_output = output.split(
            [self.proj_out.out_features, self.audio_proj_out.out_features], dim=-1
        )
        return video_output.index_select(1, video_indices), audio_output.index_select(
            1, audio_indices
        )

    @staticmethod
    def _finish(outputs, return_dict: bool):
        video_output, audio_output = outputs
        if not return_dict:
            return (video_output, audio_output)
        return {"sample": video_output, "audio_sample": audio_output}

    def _compile_stages(self, *args, options=None, **kwargs) -> None:
        """``compile_per_block``: the prologue, each block and the epilogue as their own graphs.

        The blocks are structurally identical and their weights are graph inputs, so all of them
        compile to one NEFF. Each block launch returns a one-element probe the host reads back a
        few launches behind, which bounds how many block launches are in flight at once and so
        how many blocks' hidden states the device holds."""
        import torch._dynamo.config as dynamo_config

        options = options or {}
        model_name = options.get("model_name", "minimax_h3_transformer")
        # One trace per block module; the default cache limit per code object is 8.
        dynamo_config.recompile_limit = max(
            dynamo_config.recompile_limit, 4 * len(self.transformer_blocks)
        )

        def named(fn, suffix):
            return torch.compile(
                fn, *args, options={**options, "model_name": f"{model_name}_{suffix}"}, **kwargs
            )

        self._stages = {
            "prologue": named(self._prologue, "prologue"),
            "blocks": [named(_Probed(block), "block") for block in self.transformer_blocks],
            "epilogue": named(self._epilogue, "epilogue"),
        }

    def _forward_staged(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        block_modulations,
        num_pad,
        sp_lengths,
        return_dict,
    ):
        stages = self._stages
        hidden_states, temb, adaln_indices, timestep_indices, rotary_emb, kv_valid = stages[
            "prologue"
        ](
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
        )
        pending = []
        for run, modulation in zip(stages["blocks"], block_modulations, strict=True):
            hidden_states, probe = run(
                hidden_states,
                modulation,
                adaln_indices,
                rotary_emb,
                kv_valid,
                num_pad,
                sp_lengths,
            )
            pending.append(probe)
            while len(pending) > _MAX_BLOCKS_IN_FLIGHT:
                pending.pop(0).cpu()
        for probe in pending:
            probe.cpu()
        return self._finish(
            stages["epilogue"](
                hidden_states, temb, timestep_indices, video_indices, audio_indices, sp_lengths
            ),
            return_dict,
        )

    def _embed_rows(self, sources, text_embeds, video, audio) -> torch.Tensor:
        """``cat([text_embeds, proj_in(video), audio_proj_in(audio)], dim=1)[:, sources]``,
        each projection run on the gathered rows only."""
        num_text, num_video = text_embeds.shape[1], video.shape[1]
        dtype = text_embeds.dtype
        text = text_embeds.index_select(1, sources.clamp(max=num_text - 1))
        video_rows = (sources - num_text).clamp(0, num_video - 1)
        video = self.proj_in(video.index_select(1, video_rows).to(self.proj_in.weight.dtype))
        audio_rows = (sources - num_text - num_video).clamp(0, audio.shape[1] - 1)
        audio = self.audio_proj_in(
            audio.index_select(1, audio_rows).to(self.audio_proj_in.weight.dtype)
        )
        return torch.where(
            (sources < num_text)[:, None],
            text,
            torch.where(
                (sources < num_text + num_video)[:, None], video.to(dtype), audio.to(dtype)
            ),
        )

    def block_modulations(self, timestep: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Every block's gathered AdaLN projection for ``timestep``: one ``(T, 6 * MOD * dim)``
        tensor per block, from one all-gather.

        Its own graph: inside the DiT graph the gather interleaves with the SP collectives and
        corrupts the result (NaN, no error). One tensor per block: slicing one tensor inside
        the DiT graph gives each block its own offsets, so the blocks do not share one compiled
        body and the NEFF grows ~10x, past what fits in HBM next to the VAE.

        With ``adaln_on_host`` it runs on the host instead (see ``host_block_modulations``) and
        returns host tensors; the caller moves them to the device.
        """
        if self.config.adaln_on_host:
            return self.host_block_modulations(timestep)
        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))
        adalns = [block.adaln_proj for block in self.transformer_blocks]
        local = torch.cat([adaln.local(temb) for adaln in adalns], dim=-1)
        if adalns[0].num_shards > 1:
            gathered = adalns[0].gather(local, dim=0)
            return _regroup_block_columns(gathered, adalns[0].num_shards, len(adalns))
        per_block = local.reshape(local.shape[0], len(adalns), -1)
        # Own buffers, not views of one.
        return tuple(per_block[:, index].contiguous() for index in range(len(adalns)))

    @torch.no_grad()
    def host_block_modulations(self, timestep: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``block_modulations`` on the host, from the AdaLN weights ``adaln_on_host`` keeps there.

        Each rank computes its column shard of every block, in the device graph's precision (the
        activation in ``temb``'s, the projection in BF16), and the shards are all-gathered over
        the TP group's, then the CP group's, gloo group. Returns host tensors in the device
        graph's layout."""
        timestep = timestep.cpu()
        temb = self.time_proj(timestep)
        # ``to`` takes the host copy on a device; on CPU (no move) the module is already there.
        embedder = getattr(self, "_host_time_embedder", self.time_embedder)
        temb = embedder(temb.to(embedder.linear_1.weight.dtype))
        adalns = [block.adaln_proj for block in self.transformer_blocks]
        local = torch.cat([adaln.local(temb) for adaln in adalns], dim=-1)
        if adalns[0].num_shards > 1:
            # The device graph's shard order (over TP, then CP), on each group's gloo group.
            for coord in (adalns[0].tp_coord, adalns[0].cp_coord):
                if coord.world_size > 1:
                    parts = [torch.empty_like(local) for _ in range(coord.world_size)]
                    dist.all_gather(parts, local, group=coord.cpu_group)
                    local = torch.cat(parts)
            return _regroup_block_columns(local, adalns[0].num_shards, len(adalns))
        per_block = local.reshape(local.shape[0], len(adalns), -1)
        return tuple(per_block[:, index].contiguous() for index in range(len(adalns)))

    def to(self, *args, **kwargs):
        """``nn.Module.to``, except that ``adaln_on_host`` leaves the AdaLN projections (and a
        copy of the timestep embedder they read) on the host."""
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if not self.config.adaln_on_host or device is None or device.type == "cpu":
            return super().to(*args, **kwargs)
        if "_host_time_embedder" not in self.__dict__:
            # Not a submodule: it must not follow the module to the device.
            object.__setattr__(self, "_host_time_embedder", copy.deepcopy(self.time_embedder))
        adalns = [block.adaln_proj for block in self.transformer_blocks]
        for block in self.transformer_blocks:
            del block._modules["adaln_proj"]
        try:
            super().to(*args, **kwargs)
        finally:
            for block, adaln in zip(self.transformer_blocks, adalns, strict=True):
                block._modules["adaln_proj"] = adaln
        return self

    def compile(self, *args, **kwargs):
        """``nn.Module.compile``, plus ``block_modulations`` as its own ``<model_name>_adaln``
        graph, whose outputs callers pass to ``forward``.

        ``compile_per_block`` compiles a prologue, the transformer blocks (one graph, shared by
        all of them) and an epilogue instead of the whole model; ``adaln_on_host`` does not
        compile ``block_modulations``, which then runs on the host."""
        kwargs = dict(kwargs)
        options = kwargs.pop("options", None) or {}
        model_name = options.get("model_name", "minimax_h3_transformer")
        if self.config.compile_per_block:
            self._compile_stages(*args, options=options, **kwargs)
        else:
            super().compile(*args, options=options, **kwargs)
        if self.config.adaln_on_host:
            return
        # From the class's method, so compiling twice does not wrap the compiled graph again.
        self.block_modulations = torch.compile(
            type(self).block_modulations.__get__(self),
            *args,
            options={**options, "model_name": f"{model_name}_adaln"},
            **kwargs,
        )

    def load_weights(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Rank-sharded load; parameters named as in the checkpoint need no mapping."""
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        mappings: dict[str, str | list[str]] = {}
        blocks = [
            f"token_refiner.refiner_blocks.{i}" for i in range(self.config.num_refiner_layers)
        ]
        blocks += [f"transformer_blocks.{i}" for i in range(self.config.num_layers)]
        for b in blocks:
            mappings[f"{b}.attn.qkv_proj_weight"] = [
                f"{b}.attn.{proj}.weight" for proj in ("to_q", "to_k", "to_v")
            ]
            mappings[f"{b}.attn.o_proj_weight"] = f"{b}.attn.to_out.0.weight"
            mappings[f"{b}.ff.gate_proj_weight"] = f"{b}.ff.net.0.proj.weight"
            mappings[f"{b}.ff.up_proj_weight"] = f"{b}.ff.net.0.proj.weight"
            mappings[f"{b}.ff.down_proj_weight"] = f"{b}.ff.net.2.weight"
        for i in range(self.config.num_layers):
            for kind in ("weight", "bias"):
                b = f"transformer_blocks.{i}.adaln_proj"
                mappings[f"{b}.{kind}"] = f"{b}.linear.{kind}"

        # Promote the ``_keep_in_fp32_modules`` tensors (rope.inv_freq included) to FP32 before
        # the loaders capture each parameter's dtype.
        for name, tensor in [*self.named_parameters(), *self.named_buffers()]:
            if _FP32_MODULES.intersection(name.split(".")):
                tensor.data = tensor.data.float()

        logged_casts: set = set()
        for name, param in self.named_parameters():
            set_weight_loader(
                param,
                cast_weight_loader(get_weight_loader(param), param.dtype, name, logged_casts),
            )

        checkpoint = SafetensorsCheckpoint(model_name_or_path)
        load_result = checkpoint.load_sharded_pipelined(tp_rank, tp_size, self, mappings, device)
        self.load_state_dict(load_result.state_dict, strict=False, assign=True)

        for block in self.transformer_blocks:
            block.attn.update_ring_bound()
        exact = [i for i, b in enumerate(self.transformer_blocks) if not b.attn.ring_bound_holds]
        if self.cp_size > 1 and exact:
            logger.info("Blocks %s attend over gathered K/V: the ring bound does not hold.", exact)
