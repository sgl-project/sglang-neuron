# SPDX-License-Identifier: Apache-2.0
"""FLUX.1 transformer for Neuron: TP/CP double- and single-stream blocks compiled per block type.

Raw parameters with the shared TP loaders and kernels. The Python ``forward`` only orchestrates five
compiled units (embed, double block, join, single block, final); RoPE tables, CP indices and the
sinusoidal timestep projection are computed on the host with the diffusers modules.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import sglang_neuron.diffusion.vn.functional as NF
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    apply_rotary_emb,
)
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm
from diffusers.models.transformers.transformer_flux import FluxPosEmbed
from sglang_neuron.diffusion.distributed.groups import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from sglang_neuron.diffusion.distributed.groups import get_tp_group
from sglang_neuron.diffusion.vn.utils.shard_loaders import fused_qkv_weight_loader
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    get_weight_loader,
    set_weight_loader,
)
from sglang_neuron.diffusion.distributed.groups import get_cp_group

from sglang_neuron.diffusion.attention.cp_attention import (
    cp_self_attention,
    nki_attention,
    o_proj,
)
from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel
from sglang_neuron.diffusion.distributed.parallel_state import (
    get_cp_replica_groups,
    register_replica_groups,
)
from sglang_neuron.diffusion.distributed.sequence_parallel import (
    sp_all_gather_seq,
    sp_reduce_scatter_seq,
)
from sglang_neuron.diffusion.distributed.tp_loaders import (
    cast_weight_loader,
    col_bias_loader,
    col_weight_loader,
    fused_qkv_bias_loader,
    row_bias_loader,
    row_weight_loader,
)
from sglang_neuron.diffusion.layers.feed_forward import TPFeedForward
from sglang_neuron.diffusion.quantization.adaln_kernels import adaln_modulate
from sglang_neuron.diffusion.kernels.kernel_choice import device_reason, record_kernel_choice

TRANSFORMER_COMPILER_ARGS: tuple[str, ...] = (
    "--model-type=transformer",
    "--auto-cast=none",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
    "--internal-hlo2tensorizer-options=--hierarchy-splitter=false --layers-per-module=0",
)
DOUBLE_BLOCK_COMPILER_ARGS: tuple[str, ...] = (
    *TRANSFORMER_COMPILER_ARGS,
    "--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=4",
)

T5_SEQ_LEN = 512
_NORM_EPS = 1e-6
_BLOCKS_PER_GRAPH = (19, 19)
_CHAIN_MAX_TOKENS = 2112
_NF_QKV_MAX_WIDTH = 2048
_CP_GATHER_MAX_TOKENS = 4608
_QKV_AUTO_ALLOC_MIN_ROWS = 2048


def compile_unit(
    fn: Callable, model_name: str, compiler_args: Sequence[str], *args, **kwargs
) -> Callable:
    """``torch.compile`` one fixed-shape unit under its own NEFF ``model_name``."""
    options = {
        **(kwargs.pop("options", None) or {}),
        "model_name": model_name,
        "compiler_args": list(compiler_args),
    }
    kwargs.setdefault("fullgraph", True)
    return torch.compile(fn, *args, dynamic=False, options=options, **kwargs)


def _packed_row_loader(offset: int, shard: int) -> SafetensorsWeightLoader:
    """Row-parallel shard of one input-column block of a packed ``[out, in]`` weight, transposed."""
    return SafetensorsWeightLoader(
        transform=lambda s, r: (
            s[0][:, offset + r * shard : offset + (r + 1) * shard].t().contiguous()
        )
    )


def _rope_half_perm(head_dim: int) -> torch.Tensor:
    """Per-head channel order that turns Flux's interleaved RoPE pairs into rotate-half halves."""
    return torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])


def _permuted_loader(loader: SafetensorsWeightLoader, index: torch.Tensor, dim: int):
    """Wrap ``loader`` so its output is reordered by ``index`` along ``dim``."""
    return SafetensorsWeightLoader(
        transform=lambda s, r: loader.load(s, r).index_select(dim, index).contiguous()
    )


def latent_image_ids(height: int, width: int) -> torch.Tensor:
    """``(0, row, column)`` position ids of the packed latent tokens, row-major:
    ``[height * width, 3]`` float32 on the CPU."""
    ids = torch.zeros(height, width, 3)
    ids[..., 1] = torch.arange(height)[:, None]
    ids[..., 2] = torch.arange(width)[None, :]
    return ids.reshape(height * width, 3)


@dataclass(frozen=True)
class FluxSequenceLayout:
    """Per-resolution host-built sequence layout for this CP rank, uploaded to the device."""

    image_tokens: int  # full packed image token count N
    local_image_tokens: int  # image tokens on this CP rank, N / cp
    local_text_tokens: int  # text tokens on this CP rank, 512 / cp
    img_index: torch.Tensor  # [L] int64 image rows of this CP rank
    txt_index: torch.Tensor  # [T] int64 text rows of this CP rank
    img_index_tp: torch.Tensor  # [L / tp] int64 image rows of this TP rank (double-block shard)
    txt_rows_tp: torch.Tensor  # [T / tp] int64 rows of the local text this TP rank adds at the join
    joint_image_rows: torch.Tensor  # [L] int64 positions of the local image rows in the joint order
    cos: torch.Tensor  # [T + L, head_dim] fp32 rotate-half RoPE rows, text first
    sin: torch.Tensor  # [T + L, head_dim] fp32 rotate-half RoPE rows, text first
    joint_cos: torch.Tensor  # [T + L, head_dim] cos rows in the joint (TP-rank-interleaved) order
    joint_sin: torch.Tensor  # [T + L, head_dim] sin rows in the joint order


@dataclass(frozen=True)
class FluxConditioning:
    """Step-invariant DiT inputs of one request: text tokens and every step's modulation rows."""

    context: torch.Tensor  # [1, T, dim] embedded text tokens of this CP rank
    double: list[tuple[torch.Tensor, torch.Tensor]]  # per double block (image, text) [n, 6 * dim]
    single: list[torch.Tensor]  # per single block [n, 3 * dim]
    final: torch.Tensor  # norm_out rows [n, 2 * dim]
    steps: list[torch.Tensor]  # [1] int64 device index of each denoise step


def flow_match_euler_step(sample, model_output, dt):
    """Flow-matching Euler update with the diffusers dtype order (bf16 product, fp32 sum)."""
    update = (model_output.float() * dt).to(model_output.dtype).float()
    return (sample.float() + update).to(model_output.dtype)


class FluxModulation(nn.Module):
    """Column-parallel AdaLN-Zero projection ``silu(temb) @ W + b``, all-gathered over TP."""

    def __init__(self, dim: int, num_chunks: int):
        super().__init__()
        self.num_chunks = num_chunks
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_coord = get_tp_group()
        shard = num_chunks * dim // self.tp_size
        self.weight = nn.Parameter(torch.empty(dim, shard))
        set_weight_loader(self.weight, col_weight_loader(shard, self.tp_size))
        self.bias = nn.Parameter(torch.empty(shard))
        set_weight_loader(self.bias, col_bias_loader(shard, self.tp_size))

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        """Return the all-gathered ``[n, num_chunks * dim]`` rows for ``n`` timestep embeddings."""
        out = torch.matmul(F.silu(temb), self.weight) + self.bias
        if self.tp_size > 1:
            out = self.tp_coord.all_gather(out, dim=-1)
        return out

    def select(self, rows: torch.Tensor, step: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Chunks ``[1, 1, dim]`` of the precomputed row of denoise step ``step``."""
        return rows.index_select(0, step).unsqueeze(1).chunk(self.num_chunks, dim=-1)


class _FluxAttentionBlock(nn.Module):
    """Shared head layout, process groups and joint RoPE + CP attention of the Flux blocks."""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        from sglang_neuron.diffusion.kernels.flux.gather_attention import (
            flux_gather_attention_fwd,
        )
        from sglang_neuron.diffusion.kernels.flux.ring_attention import (
            flux_ring_attention_fwd,
        )
        from sglang_neuron.diffusion.neuron_api import wrap_nki_kernel

        self._cp_self_attention = cp_self_attention
        self._ring_kernel = flux_ring_attention_fwd
        self._gather_attention_kernel = flux_gather_attention_fwd
        self._wrap_nki = wrap_nki_kernel
        self._can_run_kernel = can_run_kernel
        self._local_attention = nki_attention
        self._o_proj = o_proj

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group().device_group
        self.head_dim = head_dim
        self.num_heads = num_heads // self.tp_size
        self.inner = self.num_heads * head_dim
        self.qkv_split = [self.inner, 2 * self.inner]
        self._use_nf_qkv_proj = 3 * self.inner <= _NF_QKV_MAX_WIDTH
        self.scale = head_dim**-0.5
        self._head_perm = _rope_half_perm(head_dim)
        qk = torch.cat([h * head_dim + self._head_perm for h in range(2 * self.num_heads)])
        self._qkv_perm = torch.cat([qk, torch.arange(2 * self.inner, 3 * self.inner)])

        cp_coord = get_cp_group()
        self.cp_size = cp_coord.world_size
        self.cp_group = cp_coord if self.cp_size > 1 else None
        self.cp_replica_groups = (
            get_cp_replica_groups(self.tp_size, self.cp_size) if self.cp_size > 1 else None
        )

    def _qkv_params(self, dim: int, prefix: str) -> None:
        """Register a fused column-parallel ``[dim, 3 * inner]`` QKV weight and bias.

        Q and K columns are stored per head in rotate-half order (``_rope_half_perm``); q.k and
        therefore attention are unchanged because both use the same permutation.
        """
        weight = nn.Parameter(torch.empty(dim, 3 * self.inner))
        loader = fused_qkv_weight_loader(
            q_size=self.inner,
            kv_size=self.inner,
            shard_dim=1,
            num_shards=self.tp_size,
            is_storage_transposed=True,
        )
        set_weight_loader(weight, _permuted_loader(loader, self._qkv_perm, 1))
        bias = nn.Parameter(torch.empty(3 * self.inner))
        bias_loader = fused_qkv_bias_loader(self.inner, self.inner)
        set_weight_loader(bias, _permuted_loader(bias_loader, self._qkv_perm, 0))
        setattr(self, f"{prefix}_weight", weight)
        setattr(self, f"{prefix}_bias", bias)

    def _out_params(self, dim: int, prefix: str) -> None:
        """Register a row-parallel ``[inner, dim]`` output weight and its pre-scaled bias."""
        weight = nn.Parameter(torch.empty(self.inner, dim))
        set_weight_loader(weight, row_weight_loader(self.inner, self.tp_size))
        bias = nn.Parameter(torch.empty(dim))
        if self.tp_size > 1:
            set_weight_loader(bias, row_bias_loader(self.tp_size, dim))
        setattr(self, f"{prefix}_weight", weight)
        setattr(self, f"{prefix}_bias", bias)

    def _rms_params(self, *norms: RMSNorm) -> None:
        """Load QK RMSNorm gammas in the same rotate-half channel order as the Q/K columns."""
        for norm in norms:
            set_weight_loader(
                norm.weight,
                SafetensorsWeightLoader(transform=lambda s, r: s[0][:][self._head_perm]),
            )

    def _heads(self, qkv, norm_q, norm_k):
        """Split a fused QKV output into token-major ``[B, S, N, D]`` q, k, v with QK RMSNorm."""
        q, k, v = torch.tensor_split(qkv, self.qkv_split, dim=-1)
        shape = (self.num_heads, self.head_dim)
        return norm_q(q.unflatten(2, shape)), norm_k(k.unflatten(2, shape)), v.unflatten(2, shape)

    @staticmethod
    def _rope(x, cos, sin):
        """Rotate-half RoPE on ``[B, S, N, D]`` with ``[S, D]`` fp32 tables."""
        return apply_rotary_emb(
            x, (cos, sin), use_real=True, use_real_unbind_dim=-2, sequence_dim=1
        )

    def _qkv_rope(self, x, weight, bias, norm_q, norm_k, cos, sin, gamma=None, beta=None):
        """[LayerNorm(gamma, beta)] + QKV + bias + QK RMSNorm + RoPE; q, k, v ``[B, S, N, D]``.

        One NKI ``qkv_proj`` call within its 2048-column budget, else the same math in XLA.
        """
        if not self._use_nf_qkv_proj:
            if gamma is not None:
                x = adaln_modulate(x, gamma, beta, eps=_NORM_EPS, norm_type="layer_norm")
            q, k, v = self._heads(torch.matmul(x, weight) + bias, norm_q, norm_k)
            return self._rope(q, cos, sin), self._rope(k, cos, sin), v
        from nkilib.core.utils.common_types import NormType

        qkv = NF.qkv_proj(
            x,
            weight,
            bias=bias.unsqueeze(0),
            norm_type=NormType.NO_NORM if gamma is None else NormType.LAYER_NORM,
            norm_weights=None if gamma is None else gamma.reshape(1, -1).contiguous(),
            norm_b=None if beta is None else beta.reshape(1, -1).contiguous(),
            eps=_NORM_EPS,
            use_auto_allocation=x.shape[1] >= _QKV_AUTO_ALLOC_MIN_ROWS,
            d_head=self.head_dim,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            cos_cache=cos.to(x.dtype).unsqueeze(0),
            sin_cache=sin.to(x.dtype).unsqueeze(0),
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=_NORM_EPS,
            qk_norm_pre_rope_q_gamma=norm_q.weight.unsqueeze(0),
            qk_norm_pre_rope_k_gamma=norm_k.weight.unsqueeze(0),
        )
        q, k, v = torch.tensor_split(qkv, self.qkv_split, dim=-1)
        shape = (self.num_heads, self.head_dim)
        return q.unflatten(2, shape), k.unflatten(2, shape), v.unflatten(2, shape)

    def _joint_attention(self, q, k, v):
        """CP attention on rotated q, k; returns d-major ``[B, N, D, S]``.

        Up to ``_CP_GATHER_MAX_TOKENS`` joint tokens K/V are all-gathered over CP and attended
        inside one NKI kernel that overlaps the last head's gather with the other heads'
        attention; longer sequences use the const-max ring kernel.
        """
        if self.cp_size > 1 and self._gathered_path(q.shape[1]):
            if self._kernel_gate("flux_gather_attention", v):
                batch, seqlen, heads, head_dim = q.shape
                q_3d = (
                    q.transpose(1, 2).reshape(batch * heads, seqlen, head_dim) * self.scale
                ).contiguous()
                out = self._wrap_nki(self._gather_attention_kernel, supported_lnc=(2,))(
                    q=q_3d,
                    k=k[0].contiguous(),
                    v=v[0].contiguous(),
                    replica_groups=self.cp_replica_groups,
                    num_workers=self.cp_size,
                )
                return out.reshape(batch, heads, head_dim, seqlen)
            kv = self.cp_group.all_gather(torch.stack([k[0], v[0]], dim=1), dim=0)
            kv = kv.permute(1, 2, 0, 3).contiguous()
            return self._local_attention(
                q.transpose(1, 2), kv[0].unsqueeze(0), kv[1].unsqueeze(0), self.scale
            )
        if self.cp_size > 1 and self._kernel_gate("flux_ring_attention", v):
            out = self._wrap_nki(self._ring_kernel, supported_lnc=(2,))(
                q=q,
                k=k,
                v=v,
                replica_groups=self.cp_replica_groups,
                num_workers=self.cp_size,
                softmax_scale=self.scale,
                training=False,
            )
            return out.transpose(2, 3).contiguous()
        return self._cp_self_attention(q, k, v, self.scale, self.cp_size, self.cp_group)

    def _gathered_path(self, joint_tokens: int) -> bool:
        """Whether ``joint_tokens`` text + image tokens per CP rank take the gathered-K/V path."""
        return joint_tokens * self.cp_size <= _CP_GATHER_MAX_TOKENS

    def _kernel_gate(self, name: str, tensor: torch.Tensor) -> bool:
        """Report and return whether the NKI kernel ``name`` can run on ``tensor``'s device."""
        if not self._can_run_kernel(tensor):
            return record_kernel_choice(name, False, device_reason(tensor))
        return record_kernel_choice(name, True, "eligible")

    def _all_reduce(self, tensor):
        if self.tp_size > 1:
            dist.all_reduce(tensor, group=self.tp_group)
        return tensor


class FluxDoubleBlock(_FluxAttentionBlock):
    """Double-stream (MMDiT) block: separate image and text weights, joint attention.

    The image stream is sequence parallel over TP (``1 / tp`` of the image rows per rank, gathered
    before QKV/MLP and reduce-scattered after); the short text stream stays replicated.
    """

    def __init__(self, dim=3072, num_heads=24, head_dim=128, mlp_ratio=4.0):
        super().__init__(num_heads, head_dim)
        self._sp_gather = sp_all_gather_seq
        self._sp_scatter = sp_reduce_scatter_seq

        self.norm1 = FluxModulation(dim, 6)
        self.norm1_context = FluxModulation(dim, 6)
        self._qkv_params(dim, "img_qkv")
        self._qkv_params(dim, "txt_qkv")
        self.norm_q = RMSNorm(head_dim, eps=_NORM_EPS)
        self.norm_k = RMSNorm(head_dim, eps=_NORM_EPS)
        self.norm_added_q = RMSNorm(head_dim, eps=_NORM_EPS)
        self.norm_added_k = RMSNorm(head_dim, eps=_NORM_EPS)
        self._rms_params(self.norm_q, self.norm_k, self.norm_added_q, self.norm_added_k)
        self._out_params(dim, "img_out")
        self._out_params(dim, "txt_out")
        self.ff = TPFeedForward(dim, int(dim * mlp_ratio), dim)
        self.ff_context = TPFeedForward(dim, int(dim * mlp_ratio), dim)

    def forward(self, x, c, mod, mod_context, step, cos, sin, pending=None):
        """One double block; returns ``(x, c, pending)`` with the gated MLP residual not yet added.

        ``pending`` is ``(ff_x, gate_mlp, ff_c, c_gate_mlp)`` of the previous block (``x`` / ``c`` are
        then its pre-MLP residuals); its ``x + gate_mlp * ff_x`` is fused into this block's first
        AdaLN. :meth:`finish` adds the last block's.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1.select(
            mod, step
        )
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context.select(mod_context, step)
        )
        text_tokens = c.shape[1]
        small = self._gathered_path(x.shape[1] * self.tp_size + text_tokens)
        if pending is None:
            nc = adaln_modulate(
                c,
                1 + c_scale_msa,
                c_shift_msa,
                eps=_NORM_EPS,
                norm_type="layer_norm",
                fold_rows=small,
            )
            nx = adaln_modulate(
                x, 1 + scale_msa, shift_msa, eps=_NORM_EPS, norm_type="layer_norm", fold_rows=small
            )
        else:
            ff_x, prev_gate, ff_c, prev_c_gate = pending
            c, nc = adaln_modulate(
                ff_c,
                1 + c_scale_msa,
                c_shift_msa,
                eps=_NORM_EPS,
                norm_type="layer_norm",
                residual=c,
                gate=prev_c_gate,
                fold_rows=small,
                round_gated_product=True,
            )
            x, nx = adaln_modulate(
                ff_x,
                1 + scale_msa,
                shift_msa,
                eps=_NORM_EPS,
                norm_type="layer_norm",
                residual=x,
                gate=prev_gate,
                fold_rows=small,
                round_gated_product=True,
            )
        q_i, k_i, v_i = self._qkv_rope(
            self._gather(nx),
            self.img_qkv_weight,
            self.img_qkv_bias,
            self.norm_q,
            self.norm_k,
            cos[text_tokens:],
            sin[text_tokens:],
        )
        txt_qkv = torch.matmul(nc, self.txt_qkv_weight) + self.txt_qkv_bias
        q_t, k_t, v_t = self._heads(txt_qkv, self.norm_added_q, self.norm_added_k)
        q_t = self._rope(q_t, cos[:text_tokens], sin[:text_tokens])
        k_t = self._rope(k_t, cos[:text_tokens], sin[:text_tokens])
        attn = self._joint_attention(
            torch.cat([q_t, q_i], dim=1),
            torch.cat([k_t, k_i], dim=1),
            torch.cat([v_t, v_i], dim=1),
        )
        a_t = attn[..., :text_tokens].contiguous()
        a_i = attn[..., text_tokens:].contiguous()
        o_i = self._o_proj(a_i, self.img_out_weight, self.img_out_bias.unsqueeze(0))
        o_t = torch.matmul(a_t.flatten(1, 2).transpose(1, 2), self.txt_out_weight)
        o_t = self._all_reduce(o_t + self.txt_out_bias)
        o_i = self._scatter(o_i)

        x, nx2 = adaln_modulate(
            o_i,
            1 + scale_mlp,
            shift_mlp,
            eps=_NORM_EPS,
            norm_type="layer_norm",
            residual=x,
            gate=gate_msa,
            fold_rows=small,
        )
        c, nc2 = adaln_modulate(
            o_t,
            1 + c_scale_mlp,
            c_shift_mlp,
            eps=_NORM_EPS,
            norm_type="layer_norm",
            residual=c,
            gate=c_gate_msa,
            fold_rows=small,
        )
        ff_c = self._all_reduce(self._feed_forward(self.ff_context, nc2))
        ff_x = self._scatter(self._feed_forward(self.ff, self._gather(nx2)))
        if not small:
            return x + gate_mlp * ff_x, c + c_gate_mlp * ff_c, None
        return x, c, (ff_x, gate_mlp, ff_c, c_gate_mlp)

    @staticmethod
    def finish(x, c, pending):
        """Add the gated MLP residual a block returned as ``pending``."""
        if pending is None:
            return x, c
        ff_x, gate_mlp, ff_c, c_gate_mlp = pending
        return x + gate_mlp * ff_x, c + c_gate_mlp * ff_c

    def _gather(self, x):
        return self._sp_gather(x, self.tp_group, self.tp_size) if self.tp_size > 1 else x

    def _scatter(self, x):
        return self._sp_scatter(x, self.tp_group, self.tp_size) if self.tp_size > 1 else x

    @staticmethod
    def _feed_forward(ff, x):
        """GELU-tanh MLP of one stream in XLA matmuls; the TP partial sum (not all-reduced)."""
        up = F.gelu(torch.matmul(x, ff.up_proj_weight) + ff.up_proj_bias, approximate="tanh")
        return torch.matmul(up, ff.down_proj_weight) + ff.down_proj_bias


class FluxSingleBlock(_FluxAttentionBlock):
    """Single-stream block: fused QKV + MLP on the joint sequence, packed ``proj_out`` split in two.

    Sequence parallel over TP: the residual stream holds this TP rank's ``1 / tp`` of the joint rows;
    the modulated input is all-gathered before QKV/MLP and the output reduce-scattered back.
    """

    def __init__(self, dim=3072, num_heads=24, head_dim=128, mlp_ratio=4.0):
        super().__init__(num_heads, head_dim)
        self._sp_gather = sp_all_gather_seq
        self._sp_scatter = sp_reduce_scatter_seq
        mlp_shard = int(dim * mlp_ratio) // self.tp_size
        self.norm = FluxModulation(dim, 3)
        self._qkv_params(dim, "qkv")
        self.norm_q = RMSNorm(head_dim, eps=_NORM_EPS)
        self.norm_k = RMSNorm(head_dim, eps=_NORM_EPS)
        self._rms_params(self.norm_q, self.norm_k)
        self.mlp_up_weight = nn.Parameter(torch.empty(dim, mlp_shard))
        set_weight_loader(self.mlp_up_weight, col_weight_loader(mlp_shard, self.tp_size))
        self.mlp_up_bias = nn.Parameter(torch.empty(mlp_shard))
        set_weight_loader(self.mlp_up_bias, col_bias_loader(mlp_shard, self.tp_size))
        self.out_attn_weight = nn.Parameter(torch.empty(self.inner, dim))
        set_weight_loader(self.out_attn_weight, _packed_row_loader(0, self.inner))
        self.out_mlp_weight = nn.Parameter(torch.empty(mlp_shard, dim))
        set_weight_loader(self.out_mlp_weight, _packed_row_loader(dim, mlp_shard))
        self.out_bias = nn.Parameter(torch.empty(dim))
        if self.tp_size > 1:
            set_weight_loader(self.out_bias, row_bias_loader(self.tp_size, dim))

    def forward(self, h, mod, step, cos, sin):
        shift, scale, gate = self.norm.select(mod, step)
        n = adaln_modulate(
            h,
            1 + scale,
            shift,
            eps=_NORM_EPS,
            norm_type="layer_norm",
            fold_rows=self._gathered_path(h.shape[1] * self.tp_size),
        )
        if self.tp_size > 1:
            n = self._sp_gather(n, self.tp_group, self.tp_size)
        if (
            self.num_heads == 3
            and self.cp_size > 1
            and self._gathered_path(n.shape[1])
            and self._kernel_gate("flux_gather_kv_attention", n)
        ):
            return self._forward_mlp_first(h, n, gate, cos, sin)
        attn = self._joint_attention(
            *self._qkv_rope(n, self.qkv_weight, self.qkv_bias, self.norm_q, self.norm_k, cos, sin)
        )
        up = F.gelu(torch.matmul(n, self.mlp_up_weight) + self.mlp_up_bias, approximate="tanh")
        out = self._o_proj(attn, self.out_attn_weight, self.out_bias.unsqueeze(0))
        out = out + torch.matmul(up, self.out_mlp_weight)
        if self.tp_size > 1:
            out = self._sp_scatter(out, self.tp_group, self.tp_size)
        return h + gate * out

    def _forward_mlp_first(self, h, n, gate, cos, sin):
        """Gathered path with the MLP ordered between the K/V gather and the attention.

        The CP K/V gather and the attention are separate kernels; the MLP output is an unread
        operand of the attention kernel, so the MLP runs while the gather is in flight.
        """
        from sglang_neuron.diffusion.kernels.flux.gather_attention import (
            flux_gathered_kv_attention_fwd,
            flux_gather_kv_fwd,
        )

        up = F.gelu(torch.matmul(n, self.mlp_up_weight) + self.mlp_up_bias, approximate="tanh")
        mlp = torch.matmul(up, self.out_mlp_weight)
        q, k, v = self._qkv_rope(
            n, self.qkv_weight, self.qkv_bias, self.norm_q, self.norm_k, cos, sin
        )
        kv = self._wrap_nki(flux_gather_kv_fwd, supported_lnc=(2,))(
            k=k[0].contiguous(),
            v=v[0].contiguous(),
            replica_groups=self.cp_replica_groups,
            num_workers=self.cp_size,
        )
        batch, seqlen, heads, head_dim = q.shape
        q_3d = (
            q.transpose(1, 2).reshape(batch * heads, seqlen, head_dim) * self.scale
        ).contiguous()
        attn = self._wrap_nki(flux_gathered_kv_attention_fwd, supported_lnc=(2,))(
            q=q_3d,
            k_0=kv[0],
            v_0=kv[1],
            k_1=kv[2],
            v_1=kv[3],
            k_2=kv[4],
            v_2=kv[5],
            after=mlp,
        ).reshape(batch, heads, head_dim, seqlen)
        out = self._o_proj(attn, self.out_attn_weight, self.out_bias.unsqueeze(0)) + mlp
        if self.tp_size > 1:
            out = self._sp_scatter(out, self.tp_group, self.tp_size)
        return h + gate * out


class NeuronFluxTransformer2DModel(nn.Module):
    """FLUX.1 DiT, TP over heads/MLP and CP over the joint sequence; eager orchestration of units."""

    def __init__(
        self,
        num_layers=19,
        num_single_layers=38,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        in_channels=64,
        guidance_embeds=True,
        axes_dims_rope=(16, 56, 56),
        blocks_per_graph=_BLOCKS_PER_GRAPH,
        **unused,
    ):
        super().__init__()
        if not guidance_embeds:
            raise ValueError(
                "NeuronFluxTransformer2DModel requires a guidance-distilled checkpoint"
            )
        dim = num_attention_heads * attention_head_dim
        self.tp_size = get_tensor_model_parallel_world_size()
        self._tp_group = get_tp_group().device_group
        self._sp_gather = sp_all_gather_seq
        self.cp_coord = get_cp_group()
        self.cp_size = self.cp_coord.world_size
        if T5_SEQ_LEN % self.cp_size:
            raise ValueError(f"cp_size {self.cp_size} must divide {T5_SEQ_LEN} text tokens")
        self.local_text_tokens = T5_SEQ_LEN // self.cp_size
        register_replica_groups(tp_size=self.tp_size, cp_size=self.cp_size)

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
            embedding_dim=dim, pooled_projection_dim=pooled_projection_dim
        )
        self.context_embedder = nn.Linear(joint_attention_dim, dim)
        self.x_embedder = nn.Linear(in_channels, dim)
        self.transformer_blocks = nn.ModuleList(
            FluxDoubleBlock(dim, num_attention_heads, attention_head_dim) for _ in range(num_layers)
        )
        self.single_transformer_blocks = nn.ModuleList(
            FluxSingleBlock(dim, num_attention_heads, attention_head_dim)
            for _ in range(num_single_layers)
        )
        self.norm_out = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=_NORM_EPS)
        self.proj_out = nn.Linear(dim, in_channels)

        self.double_per_graph = min(blocks_per_graph[0], num_layers)
        self.single_per_graph = min(blocks_per_graph[1], num_single_layers)
        self._layouts: dict[tuple[int, int], FluxSequenceLayout] = {}
        self._condition_graph = type(self)._condition
        self._embed_graph = type(self)._embed
        self._double_graph = self._double_chain
        self._join_graph = self._join
        self._single_graph = self._single_chain
        self._final_graph = type(self)._final
        self._euler_graph = flow_match_euler_step
        self._head_graph = type(self)._head
        self._tail_graph = type(self)._tail

    @property
    def dtype(self) -> torch.dtype:
        return self.x_embedder.weight.dtype

    def sequence_layout(self, height: int, width: int) -> FluxSequenceLayout:
        """Build (once per resolution) this CP rank's token indices and RoPE rows on the device."""
        key = (height, width)
        if key in self._layouts:
            return self._layouts[key]
        if height % 64 or width % 64 or height < 512 or width < 512:
            raise ValueError(
                f"FLUX.1 on Neuron needs height and width that are multiples of 64 and >= 512, "
                f"got {height}x{width}"
            )
        image_tokens = (height // 16) * (width // 16)
        local_image = image_tokens // self.cp_size
        local_text = self.local_text_tokens
        rank = self.cp_coord.rank_in_group
        img_ids = latent_image_ids(height // 16, width // 16)
        cos, sin = self.pos_embed(torch.cat([torch.zeros(T5_SEQ_LEN, 3), img_ids]))
        txt_index = torch.arange(local_text * rank, local_text * (rank + 1))
        img_index = torch.arange(local_image * rank, local_image * (rank + 1))
        rows = torch.cat([txt_index, T5_SEQ_LEN + img_index])
        tp = self.tp_size
        if local_text % tp or local_image % tp:
            raise ValueError(
                f"tp_size {tp} must divide {local_text} text and {local_image} image tokens per CP rank"
            )
        tp_rank = get_tensor_model_parallel_rank()
        text_span, image_span = local_text // tp, local_image // tp
        joint_order = torch.cat(
            [
                torch.cat(
                    [
                        torch.arange(r * text_span, (r + 1) * text_span),
                        local_text + torch.arange(r * image_span, (r + 1) * image_span),
                    ]
                )
                for r in range(tp)
            ]
        )
        joint_image_rows = torch.argsort(joint_order)[local_text:]
        cos, sin = (
            t[rows][:, _rope_half_perm(t.shape[1])].float().contiguous() for t in (cos, sin)
        )
        device = self.x_embedder.weight.device
        layout = FluxSequenceLayout(
            image_tokens=image_tokens,
            local_image_tokens=local_image,
            local_text_tokens=local_text,
            img_index=img_index.to(device),
            txt_index=txt_index.to(device),
            img_index_tp=img_index[image_span * tp_rank : image_span * (tp_rank + 1)].to(device),
            txt_rows_tp=torch.arange(text_span * tp_rank, text_span * (tp_rank + 1)).to(device),
            joint_image_rows=joint_image_rows.to(device),
            cos=cos.to(device),
            sin=sin.to(device),
            joint_cos=cos[joint_order].contiguous().to(device),
            joint_sin=sin[joint_order].contiguous().to(device),
        )
        self._layouts[key] = layout
        return layout

    def time_projection(self, value: torch.Tensor) -> torch.Tensor:
        """Sinusoidal 256-channel projection of a timestep or guidance value, on the host."""
        return self.time_text_embed.time_proj(value).to(self.dtype)

    def _condition(self, prompt_embeds, txt_index, pooled, t_proj, g_proj):
        c = self.context_embedder(prompt_embeds.index_select(1, txt_index))
        embed = self.time_text_embed
        temb = embed.timestep_embedder(t_proj) + embed.guidance_embedder(g_proj)
        temb = temb + embed.text_embedder(pooled)
        double = [
            (block.norm1(temb), block.norm1_context(temb)) for block in self.transformer_blocks
        ]
        single = [block.norm(temb) for block in self.single_transformer_blocks]
        return c, double, single, self.norm_out.linear(F.silu(temb))

    @torch.inference_mode()
    def condition(self, prompt_embeds, pooled, t_proj, g_proj, layout) -> FluxConditioning:
        """Embed the text tokens and every step's modulation once per request.

        Args:
            prompt_embeds: [1, 512, 4096] T5 hidden states.
            pooled: [1, 768] CLIP pooled output.
            t_proj: [n, 256] sinusoidal projections of the n denoise timesteps.
            g_proj: [1, 256] sinusoidal projection of the guidance value.
            layout: this resolution's ``FluxSequenceLayout``.
        """
        c, double, single, final = self._condition_graph(
            self, prompt_embeds, layout.txt_index, pooled, t_proj, g_proj
        )
        steps = self._step_indices(t_proj.shape[0])
        return FluxConditioning(c, list(double), list(single), final, steps)

    def _step_indices(self, num_steps: int) -> list[torch.Tensor]:
        """The ``[1]`` int64 device index of each denoise step, made once per step count.

        Constants, so no request has to copy them to the device: under torch-neuronx a
        host-to-device copy waits in the stream behind the text encoders and the condition
        graph, so 25 small copies a request sit on the critical path.
        """
        cache = self.__dict__.setdefault("_step_index_cache", {})
        if num_steps not in cache:
            device = self.x_embedder.weight.device
            cache[num_steps] = [torch.tensor([i]).to(device) for i in range(num_steps)]
        return cache[num_steps]

    def _embed(self, latents, img_index):
        return self.x_embedder(latents.index_select(1, img_index))

    @staticmethod
    def _double_chain(blocks, x, c, mods, step, cos, sin):
        pending = None
        for block, (mod, mod_context) in zip(blocks, mods):
            x, c, pending = FluxDoubleBlock.forward(
                block, x, c, mod, mod_context, step, cos, sin, pending
            )
        return FluxDoubleBlock.finish(x, c, pending)

    @staticmethod
    def _single_chain(blocks, h, mods, step, cos, sin):
        for block, mod in zip(blocks, mods):
            h = FluxSingleBlock.forward(block, h, mod, step, cos, sin)
        return h

    @staticmethod
    def _join(c, x, txt_rows_tp):
        """This TP rank's joint rows: its share of the text, then its image shard."""
        return torch.cat([c.index_select(1, txt_rows_tp), x], dim=1)

    def _head(self, blocks, latents, img_index_tp, c, mods, step, cos, sin, txt_rows_tp):
        """Image embedding (TP shard), the double blocks ``blocks`` and the join."""
        x = self.x_embedder(latents.index_select(1, img_index_tp))
        x, c = self._double_chain(blocks, x, c, mods, step, cos, sin)
        return self._join(c, x, txt_rows_tp)

    def _tail(self, blocks, h, mods, final_mod, step, cos, sin, image_rows, latents, dt):
        """The last single blocks and the output projection; with ``dt`` also the Euler update."""
        h = self._single_chain(blocks, h, mods, step, cos, sin)
        out = self._final(h, final_mod, step, image_rows)
        return out if dt is None else flow_match_euler_step(latents, out, dt)

    def _final(self, h, mod, step, image_rows):
        """Gather the joint rows, take the image rows in natural order, final norm and projection."""
        if self.tp_size > 1:
            h = self._sp_gather(h, self._tp_group, self.tp_size)
        scale, shift = mod.index_select(0, step).chunk(2, dim=1)
        h = F.layer_norm(h.index_select(1, image_rows), h.shape[-1:], eps=_NORM_EPS)
        out = self.proj_out(h * (1 + scale)[:, None] + shift[:, None])
        if self.cp_size > 1:
            out = self.cp_coord.all_gather(out.contiguous(), dim=1)
        return out

    def folds(self, layout) -> bool:
        """Whether a step at this layout runs as three graphs (else one graph per block)."""
        return layout.local_text_tokens + layout.local_image_tokens <= _CHAIN_MAX_TOKENS

    def forward(self, latents, cond: FluxConditioning, index: int, layout, dt=None):
        """Denoise step ``index``: the ``[1, N, 64]`` prediction, or with ``dt`` the Euler-updated
        latents."""
        step = cond.steps[index]
        if self.folds(layout):
            return self._folded(latents, cond, step, layout, dt)
        out = self._per_block(latents, cond, step, layout)
        return out if dt is None else self._euler_graph(latents, out, dt)

    def _folded(self, latents, cond, step, layout, dt):
        """Three graphs per step: embed + double blocks + join, single-block chunks, tail."""
        cos, sin = layout.joint_cos, layout.joint_sin
        h = self._head_graph(
            self,
            self.transformer_blocks,
            latents,
            layout.img_index_tp,
            cond.context,
            cond.double,
            step,
            layout.cos,
            layout.sin,
            layout.txt_rows_tp,
        )
        n = self.single_per_graph
        last = len(self.single_transformer_blocks) - n
        for i in range(0, last, n):
            blocks = self.single_transformer_blocks[i : i + n]
            h = self._single_graph(blocks, h, cond.single[i : i + n], step, cos, sin)
        blocks = self.single_transformer_blocks[last:]
        return self._tail_graph(
            self,
            blocks,
            h,
            cond.single[last:],
            cond.final,
            step,
            cos,
            sin,
            layout.joint_image_rows,
            latents,
            dt,
        )

    def _per_block(self, latents, cond, step, layout):
        """One graph per block for sequences too long to chain (4096 px)."""
        cos, sin = layout.cos, layout.sin
        x = self._embed_graph(self, latents, layout.img_index_tp)
        c = cond.context
        for i in range(len(self.transformer_blocks)):
            blocks = self.transformer_blocks[i : i + 1]
            x, c = self._double_graph(blocks, x, c, cond.double[i : i + 1], step, cos, sin)
        h = self._join_graph(c, x, layout.txt_rows_tp)
        cos, sin = layout.joint_cos, layout.joint_sin
        for i in range(len(self.single_transformer_blocks)):
            blocks = self.single_transformer_blocks[i : i + 1]
            h = self._single_graph(blocks, h, cond.single[i : i + 1], step, cos, sin)
        return self._final_graph(self, h, cond.final, step, layout.joint_image_rows)

    def compile(self, *args, **kwargs) -> Self:
        """Compile the per-request conditioning and the per-step units; NEFFs build lazily."""
        args_ = (TRANSFORMER_COMPILER_ARGS, *args)
        self._condition_graph = compile_unit(
            type(self)._condition, "flux_condition", *args_, **kwargs
        )
        self._embed_graph = compile_unit(type(self)._embed, "flux_embed", *args_, **kwargs)
        double_args = (DOUBLE_BLOCK_COMPILER_ARGS, *args)
        self._double_graph = compile_unit(
            self._double_chain, f"flux_double_x{self.double_per_graph}", *double_args, **kwargs
        )
        self._join_graph = compile_unit(self._join, "flux_join", *args_, **kwargs)
        self._single_graph = compile_unit(
            self._single_chain, f"flux_single_x{self.single_per_graph}", *args_, **kwargs
        )
        self._final_graph = compile_unit(type(self)._final, "flux_final", *args_, **kwargs)
        self._euler_graph = compile_unit(flow_match_euler_step, "flux_euler_step", *args_, **kwargs)
        self._head_graph = compile_unit(type(self)._head, "flux_head", *double_args, **kwargs)
        self._tail_graph = compile_unit(type(self)._tail, "flux_tail", *args_, **kwargs)
        return self

    def load_weights(self, path: str) -> None:
        """Load this TP rank's shards from a diffusers ``transformer/`` safetensors directory."""
        mappings: dict[str, str | list[str]] = {}
        for i in range(len(self.transformer_blocks)):
            p = f"transformer_blocks.{i}."
            for name in ("norm1", "norm1_context"):
                mappings[f"{p}{name}.weight"] = f"{p}{name}.linear.weight"
                mappings[f"{p}{name}.bias"] = f"{p}{name}.linear.bias"
            for stream, keys in (
                ("img", ("to_q", "to_k", "to_v")),
                ("txt", ("add_q_proj", "add_k_proj", "add_v_proj")),
            ):
                for kind in ("weight", "bias"):
                    mappings[f"{p}{stream}_qkv_{kind}"] = [f"{p}attn.{k}.{kind}" for k in keys]
            for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
                mappings[f"{p}{name}.weight"] = f"{p}attn.{name}.weight"
            for stream, key in (("img", "to_out.0"), ("txt", "to_add_out")):
                for kind in ("weight", "bias"):
                    mappings[f"{p}{stream}_out_{kind}"] = f"{p}attn.{key}.{kind}"
            for ff in ("ff", "ff_context"):
                for kind in ("weight", "bias"):
                    mappings[f"{p}{ff}.up_proj_{kind}"] = f"{p}{ff}.net.0.proj.{kind}"
                    mappings[f"{p}{ff}.down_proj_{kind}"] = f"{p}{ff}.net.2.{kind}"
        for i in range(len(self.single_transformer_blocks)):
            p = f"single_transformer_blocks.{i}."
            for kind in ("weight", "bias"):
                mappings[f"{p}norm.{kind}"] = f"{p}norm.linear.{kind}"
                mappings[f"{p}qkv_{kind}"] = [
                    f"{p}attn.{k}.{kind}" for k in ("to_q", "to_k", "to_v")
                ]
                mappings[f"{p}mlp_up_{kind}"] = f"{p}proj_mlp.{kind}"
            for name in ("norm_q", "norm_k"):
                mappings[f"{p}{name}.weight"] = f"{p}attn.{name}.weight"
            mappings[f"{p}out_attn_weight"] = f"{p}proj_out.weight"
            mappings[f"{p}out_mlp_weight"] = f"{p}proj_out.weight"
            mappings[f"{p}out_bias"] = f"{p}proj_out.bias"

        logged_casts: set = set()
        for name, param in self.named_parameters():
            set_weight_loader(
                param,
                cast_weight_loader(get_weight_loader(param), param.dtype, name, logged_casts),
            )
        state_dict = (
            SafetensorsCheckpoint(path)
            .load_sharded_pipelined(
                get_tensor_model_parallel_rank(),
                self.tp_size,
                self,
                mappings,
                torch.device("cpu"),
            )
            .state_dict
        )
        missing = sorted(name for name, _ in self.named_parameters() if name not in state_dict)
        if missing:
            raise RuntimeError(f"FLUX.1 checkpoint {path} is missing parameters: {missing[:8]}")
        self.load_state_dict(state_dict, strict=False, assign=True)
