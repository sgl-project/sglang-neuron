# SPDX-License-Identifier: Apache-2.0
"""
WanTransformer3DModel for Neuron: the Wan2.2 DiT with TP, CP (context parallelism) and sequence
parallelism over the TP group.

Raw nn.Parameter with weight loaders (no CPL/RPL modules); the condition embedder is diffusers'.
BF16 only: there are no FP8 (ROW_MX) modules and no Cache-DiT phases.

CP self-attention runs the const-max ring NKI kernel (``ring_attention.ring_cp_self_attention``); where
the kernel cannot run it falls back to all-gather K/V + flash, a capability check, not a setting.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import WanTimeTextImageEmbedding

import sglang_neuron.diffusion.vn.functional as NF
from sglang_neuron.diffusion.attention.cp_attention import (
    nki_attention,
    o_proj,
)
from sglang_neuron.diffusion.distributed.groups import (
    get_classifier_free_guidance_world_size,
    get_cp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    is_cfg_group_initialized,
)
from sglang_neuron.diffusion.distributed.parallel_state import register_replica_groups
from sglang_neuron.diffusion.distributed.sequence_parallel import (
    sp_all_gather_seq,
    sp_entry,
    sp_exit,
    sp_padded_len,
)
from sglang_neuron.diffusion.distributed.tp_loaders import (
    cast_weight_loader,
    col_bias_loader,
    col_weight_loader,
    deinterleaved_fused_qkv_bias_loader,
    deinterleaved_fused_qkv_weight_loader,
    fused_qkv_bias_loader,
    qk_norm_gamma_loader,
    row_bias_loader,
    row_weight_loader,
)
from sglang_neuron.diffusion.kernels.kernel_choice import device_reason, record_kernel_choice
from sglang_neuron.diffusion.kernels.wan.common_types import (
    NormType,
    QKVOutputLayout,
)
from sglang_neuron.diffusion.kernels.wan.qkv_shim import torch_compatible_bf16_qkv
from sglang_neuron.diffusion.layers.feed_forward import TPFeedForward
from sglang_neuron.diffusion.layers.rope import (
    WanRotaryPosEmbed,
    apply_rotary_emb_wan,
)
from sglang_neuron.diffusion.layers.rope_rotate_half import (
    apply_rotate_half_rope,
    rope_rotate_half_caches,
)
from sglang_neuron.diffusion.models.wan2_2.ring_attention import ring_cp_self_attention
from sglang_neuron.diffusion.neuron_api import wrap_nki_kernel
from sglang_neuron.diffusion.quantization.adaln_kernels import adaln_modulate
from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel
from sglang_neuron.diffusion.vn.utils.shard_loaders import (
    fused_qkv_weight_loader,
    sharding_weight_loader,
)
from sglang_neuron.parallel.groups import shard_select
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import get_weight_loader, set_weight_loader

logger = logging.getLogger(__name__)


# ===================================================================
# Config
# ===================================================================

_SUPPORTED_QUANTIZATION = ("bf16",)


def _check_bf16_only(quantization: str | None, modules_to_not_convert) -> str:
    """Return the quantization mode, which must be BF16 (``None`` or ``"bf16"``).

    An FP8 (ROW_MX) checkpoint path is not implemented here, so its arguments are kept only to
    reject them clearly.
    """
    mode = (quantization or "bf16").lower()
    if mode not in _SUPPORTED_QUANTIZATION:
        raise ValueError(
            f"quantization={quantization!r} is not supported; the Neuron Wan2.2 DiT runs BF16 "
            "(quantization=None or 'bf16')."
        )
    if modules_to_not_convert is not None:
        raise ValueError("modules_to_not_convert is only valid with an FP8 quantization")
    return mode


def _check_cp_replica_groups(replica_groups, cp_group: Any, world_size: int) -> None:
    """Raise unless ``replica_groups`` is the CP partition the ring kernel can run on.

    ``ring_cp_self_attention`` takes the all-gather path when it gets no replica groups, so a missing
    partition would silently drop the ring. Each SPMD rank finds its own ring in the partition,
    so it must cover ``range(world_size)`` once, in groups of the CP size, and hold the live CP
    group's ranks (a partition that disagrees with the live groups is the "no_hier no_mesh" /
    "replica id not seen" failure on device).
    """
    if not replica_groups:
        raise ValueError(f"CP size {cp_group.world_size} needs the CP replica groups, got none")
    covered = sorted(rank for group in replica_groups for rank in group)
    if covered != list(range(world_size)) or any(
        len(group) != cp_group.world_size for group in replica_groups
    ):
        raise ValueError(
            f"CP replica groups {replica_groups} are not a partition of the {world_size} ranks "
            f"into groups of {cp_group.world_size}"
        )
    if tuple(cp_group.ranks) not in {tuple(group) for group in replica_groups}:
        raise ValueError(
            f"the live CP group {cp_group.ranks} is not one of the CP replica groups "
            f"{replica_groups}"
        )


def _shard_rope_freqs(freqs: torch.Tensor, lane: int, group: Any) -> torch.Tensor:
    """shard_select a [1, S, 1, D] RoPE table on S, gathering whole rows.

    Wan's tables hold each angle on two adjacent lanes, and the consumers read one lane of each
    pair (cos lanes 0::2, sin lanes 1::2). Sharding the full table lets the compiler fold that
    stride into shard_select's gather, which then moves only one 4-byte value per descriptor and
    is slow. Sharding the distinct angles and duplicating them back
    gathers contiguous rows and gives the same tensor.
    """
    return shard_select(freqs[..., lane::2], 1, group).repeat_interleave(2, dim=-1)


@dataclass
class WanConfig:
    """Configuration for WanTransformer3DModel.

    ``quantization`` must be ``None`` or ``"bf16"`` (BF16 weights); ``modules_to_not_convert``
    must be ``None``. Both stay so that configs which carry these keys still load.

    CP self-attention runs the ring-attention NKI kernel (see :func:`ring_cp_self_attention`),
    which keeps K/V local and rotates it around the CP group. Where the kernel cannot run at
    all (CPU mode, fake-tensor tracing, NKI kernels disabled, LNC 1) it falls back to all-gather +
    flash — a capability check, not a setting.
    """

    patch_size: tuple = (1, 2, 2)
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    eps: float = 1e-6
    image_dim: int | None = None
    added_kv_proj_dim: int | None = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: int | None = None
    quantization: str | None = None
    modules_to_not_convert: list[str] | None = None
    tp_sequence_parallel: bool = False


# ===================================================================
# Model classes
# ===================================================================


def _can_use_wan_bf16_fused_qkv(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    rotary_emb,
    *,
    num_heads: int,
    head_dim: int,
) -> bool:
    """Capability gate for the BF16 fused QKV CTE kernel (projection + Q/K RMSNorm + RoPE).

    The kernel keeps one batch's per-token Q/K sum(x**2) in SBUF, so it takes batch one; Wan2.2
    with CFG parallel 2 presents batch one to each CFG replica.
    """
    if not can_run_kernel(hidden):
        return record_kernel_choice("wan_bf16_qkv", False, device_reason(hidden))
    if rotary_emb is None:
        return record_kernel_choice("wan_bf16_qkv", False, "no rotary embedding")
    if hidden.dtype != torch.bfloat16 or hidden.dim() != 3 or weight.dim() != 2 or bias.dim() != 1:
        return record_kernel_choice("wan_bf16_qkv", False, "unexpected operand dtype or rank")

    batch, sequence, hidden_size = hidden.shape
    qkv_size = weight.shape[1]
    eligible = (
        batch == 1
        and sequence > 0
        and hidden_size == weight.shape[0]
        and hidden_size % 128 == 0
        and qkv_size <= 4096
        and qkv_size == 3 * num_heads * head_dim
        and head_dim == 128
        and bias.shape[0] == qkv_size
        and tuple(q_gamma.shape) == (1, num_heads * head_dim)
        and tuple(k_gamma.shape) == (1, num_heads * head_dim)
    )
    if not eligible:
        return record_kernel_choice("wan_bf16_qkv", False, "shape outside the kernel limits")

    freqs_cos, freqs_sin = rotary_emb
    if not (
        freqs_cos.shape == freqs_sin.shape
        and freqs_cos.shape[0] == batch
        and freqs_cos.shape[1] == sequence
        and freqs_cos.shape[-1] == head_dim
    ):
        return record_kernel_choice("wan_bf16_qkv", False, "rotary table does not match the tokens")
    return record_kernel_choice("wan_bf16_qkv", True, "eligible")


# Built once, at import, like the launchers in cp_attention and feed_forward: a wrap_nki call
# inside the traced forward breaks a warm start from a Dynamo precompile package. It launches at
# the configured LNC, 1 or 2 (the LingBot DiT's self-attention runs it at LNC 1).
_wan_bf16_qkv_launch = wrap_nki_kernel(torch_compatible_bf16_qkv, supported_lnc=(1, 2))


def _wan_bf16_fused_qkv(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    norm_eps: float,
    replica_groups,
) -> torch.Tensor:
    """Run the BF16 QKV projection, across-heads Q/K RMSNorm (TP all-reduce) and RoPE in one kernel."""
    return _wan_bf16_qkv_launch(
        hidden=hidden,
        qkv_weights=weight,
        bias=bias,
        d_head=head_dim,
        num_q_heads=num_heads,
        num_kv_heads=num_heads,
        output_layout=QKVOutputLayout.BSD,
        fused_rope=True,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        qk_norm_pre_rope_q_norm=NormType.RMS_NORM_ACROSS_HEADS,
        qk_norm_pre_rope_k_norm=NormType.RMS_NORM_ACROSS_HEADS,
        qk_norm_pre_rope_eps=norm_eps,
        qk_norm_pre_rope_q_gamma=q_gamma,
        qk_norm_pre_rope_k_gamma=k_gamma,
        replica_groups=replica_groups,
    )


def _install_qk_norm_gamma_loader(
    module: nn.Module, *, deinterleave: bool, num_shards: int | None = None
) -> None:
    """Reshape ``norm_q``/``norm_k`` gamma to the kernel's ``[1, dim]`` row and load it that way.

    The fused QKV kernel (``WanSelfAttentionBF16Fused``, the only caller) takes each qk-norm gamma
    as a ``[1, dim]`` row, de-interleaved per head to match its rotate-half RoPE
    (``deinterleave=True``); ``deinterleave=False`` keeps the checkpoint's order. The transform is
    constant in the loaded weight, so it runs once at checkpoint load through a weight loader
    (replacing ``DistributedRMSNorm``'s plain sharding loader) instead of in every forward.
    Cross-attention does not use the kernel: its Q/K norms stay ``DistributedRMSNorm``.

    ``num_shards`` matches the projection sharding: ``module.tp_size`` (default) shards the gamma
    across the TP head split; ``1`` loads the full-width gamma on every rank.
    """
    shards = module.tp_size if num_shards is None else num_shards
    for norm in (module.norm_q, module.norm_k):
        hidden_size = norm.weight.shape[0]
        norm.weight = nn.Parameter(torch.ones(1, hidden_size), requires_grad=False)
        set_weight_loader(
            norm.weight,
            qk_norm_gamma_loader(hidden_size, shards, module.head_dim, deinterleave),
        )


class DistributedRMSNorm(nn.Module):
    """RMSNorm with global RMS across TP ranks via all-reduce.

    Uses the TP-only process group so that only ranks sharing the same
    sequence tokens participate in the reduction (correct with CP).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        if tp_size > 1:
            set_weight_loader(
                self.weight,
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=hidden_size,
                    num_shards=tp_size,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        local_sum_sq = (x_float**2).sum(dim=-1, keepdim=True)

        if self.tp_size > 1:
            global_sum_sq = local_sum_sq.clone()
            dist.all_reduce(global_sum_sq, group=self.tp_group)
            global_count = x.shape[-1] * self.tp_size
        else:
            global_sum_sq = local_sum_sq
            global_count = x.shape[-1]

        rms = torch.sqrt(global_sum_sq / global_count + self.eps)
        return ((x_float / rms) * self.weight.float()).to(input_dtype)


class WanSelfAttention(nn.Module):
    """Self-attention with fused QKV projection (LLaMA pattern).

    ``project_qkv`` runs the QKV projection (NF.qkv_proj when the per-rank QKV width fits its SBUF
    budget, torch.matmul otherwise), the Q/K DistributedRMSNorm and Wan RoPE;
    :class:`WanSelfAttentionBF16Fused` replaces it with one fused kernel. Attention and the output
    projection use direct NKI-Lib kernels.

    CP self-attention always runs the ring-attention NKI kernel (see
    :func:`ring_cp_self_attention`); there is no knob to select a different path.
    """

    _uses_native_qkv_projection = False

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        tp_sequence_parallel: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        self.sp_enabled = tp_sequence_parallel
        self.sp_real_len: int | None = None
        self.sp_padded_len: int | None = None
        self.num_heads = num_heads // tp_size
        tp_inner_dim = self.num_heads * head_dim

        # CP group
        cp_group = get_cp_group()
        self.cp_size = cp_group.world_size
        self.cp_group = cp_group if self.cp_size > 1 else None

        # Default (ring) path: ring attention needs the GLOBAL ranks of every CP ring
        # (each SPMD rank resolves its own ring internally). get_cp_replica_groups derives
        # them from the same partition source register_replica_groups uses, so the kernel's
        # groups and the registered mesh partition stay consistent.
        self.cp_replica_groups = None
        if self.cp_size > 1:
            from sglang_neuron.diffusion.distributed.parallel_state import get_cp_replica_groups

            self.cp_replica_groups = get_cp_replica_groups(tp_size, self.cp_size)
            cfg_size = (
                get_classifier_free_guidance_world_size() if is_cfg_group_initialized() else 1
            )
            _check_cp_replica_groups(
                self.cp_replica_groups, cp_group, tp_size * self.cp_size * cfg_size
            )

        # Per-rank sizes (MHA: q == k == v)
        q_size = tp_inner_dim
        kv_size = tp_inner_dim
        qkv_size = q_size + 2 * kv_size
        self.qkv_split = [q_size, q_size + kv_size]

        # The NF.qkv_proj NKI kernel exceeds its SBUF budget when the per-rank qkv output dim is
        # above 2048; torch.matmul runs the projection then.
        self._use_nf_qkv_proj = qkv_size <= 2048
        if (
            not self._use_nf_qkv_proj
            and not self._uses_native_qkv_projection
            and get_tensor_model_parallel_rank() == 0
        ):
            logger.warning(
                "NKI QKV kernel disabled: qkv_size=%d exceeds SBUF budget limit of 2048. "
                "Falling back to torch.matmul",
                qkv_size,
            )

        # Fused QKV: [dim, qkv_size_per_rank]
        self.qkv_proj_weight = nn.Parameter(torch.empty(dim, qkv_size))
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=q_size,
                kv_size=kv_size,
                shard_dim=1,
                num_shards=tp_size,
                is_storage_transposed=True,
            ),
        )
        self.qkv_proj_bias = nn.Parameter(torch.empty(qkv_size))
        set_weight_loader(
            self.qkv_proj_bias,
            fused_qkv_bias_loader(q_size=q_size, kv_size=kv_size),
        )

        self.norm_q = DistributedRMSNorm(tp_inner_dim, eps=eps)
        self.norm_k = DistributedRMSNorm(tp_inner_dim, eps=eps)

        # Output projection: [tp_inner_dim, dim]
        self.o_proj_weight = nn.Parameter(torch.empty(tp_inner_dim, dim))
        set_weight_loader(self.o_proj_weight, row_weight_loader(tp_inner_dim, tp_size))
        self.o_proj_bias = nn.Parameter(torch.empty(dim))
        if tp_size > 1:
            set_weight_loader(self.o_proj_bias, row_bias_loader(tp_size, dim))

        self.scale = 1.0 / (head_dim**0.5)
        self._can_run_kernel = can_run_kernel

    def project_qkv(self, hidden_states: torch.Tensor, rotary_emb=None):
        """Project Q/K/V and apply the Q/K norm and Wan RoPE; returns token-major [B, S, N, D]."""
        if self._use_nf_qkv_proj:
            qkv = NF.qkv_proj(
                hidden_states,
                self.qkv_proj_weight,
                bias=self.qkv_proj_bias.unsqueeze(0),
            )
        else:
            qkv = torch.matmul(hidden_states, self.qkv_proj_weight) + self.qkv_proj_bias
        q, k, v = torch.tensor_split(qkv, self.qkv_split, dim=-1)

        query = self.norm_q(q).unflatten(2, (self.num_heads, self.head_dim))
        key = self.norm_k(k).unflatten(2, (self.num_heads, self.head_dim))
        value = v.unflatten(2, (self.num_heads, self.head_dim))

        if rotary_emb is not None:
            freqs_cos, freqs_sin = rotary_emb
            query = apply_rotary_emb_wan(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb_wan(key, freqs_cos, freqs_sin)
        return query, key, value

    def forward(self, hidden_states: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        hidden_states = sp_entry(
            hidden_states, self.tp_group, self.tp_size, self.sp_enabled, self.sp_real_len
        )
        query, key, value = self.project_qkv(hidden_states, rotary_emb)

        # q/k/v are already token-major [B, S, N, D] out of the projection + norm + unflatten
        # + RoPE, which is exactly what the ring kernel wants — no transpose.
        # Ring attention, or all-gather + flash where the kernel cannot run.
        hidden_states = ring_cp_self_attention(
            query,
            key,
            value,
            self.scale,
            self.cp_size,
            self.cp_group,
            self.cp_replica_groups,
        )

        # ring_cp_self_attention returns d-major [B, N, D, S] — already the o-proj layout.
        # Pad the seq dim to sp_padded_len here, on the narrow [B,N,D,S] attention, so the o-proj
        # emits sp_padded_len rows and the wider [B, S, H] sp_exit F.pad self-skips (its
        # ``output.shape[1] < padded_len`` guard is then false). The tail rows are query-only
        # (o-proj has no keys) and are discarded after the block's SP gather.
        if self.sp_enabled and self.sp_padded_len is not None:
            seq_pad = self.sp_padded_len - hidden_states.shape[-1]
            if seq_pad > 0:
                hidden_states = F.pad(hidden_states, (0, seq_pad))
        output = o_proj(
            hidden_states,
            self.o_proj_weight,
            self.o_proj_bias.unsqueeze(0),
        )
        return sp_exit(output, self.tp_group, self.tp_size, self.sp_enabled, self.sp_padded_len)


class WanSelfAttentionBF16Fused(WanSelfAttention):
    """BF16 self-attention with the QKV projection, Q/K RMSNorm and RoPE in one kernel.

    The QKV CTE kernel applies Wan's across-heads Q/K RMSNorm (with its TP all-reduce) and a
    rotate-half RoPE in the same call as the projection. The Q/K
    weight and bias columns and the norm gammas are de-interleaved per head at load time so that
    rotate-half reproduces Wan's interleaved RoPE (Q and K get the same permutation, so Q.K^T is
    unchanged). When the capability gate rejects a runtime shape, the torch fallback keeps that
    de-interleaved convention.
    """

    _uses_native_qkv_projection = True

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        tp_sequence_parallel: bool = False,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            tp_sequence_parallel=tp_sequence_parallel,
        )
        from sglang_neuron.diffusion.distributed.parallel_state import get_replica_groups

        local_width = self.num_heads * self.head_dim
        set_weight_loader(
            self.qkv_proj_weight,
            deinterleaved_fused_qkv_weight_loader(
                local_width, local_width, self.tp_size, self.head_dim
            ),
        )
        set_weight_loader(
            self.qkv_proj_bias,
            deinterleaved_fused_qkv_bias_loader(local_width, local_width, head_dim=self.head_dim),
        )
        # The kernel takes each gamma as a [1, local_width] row, de-interleaved like Q/K.
        _install_qk_norm_gamma_loader(self, deinterleave=True)
        # TP replica groups (global ranks) for the kernel's in-kernel across-heads-norm
        # all-reduce: the RMS is taken over all heads on all TP ranks.
        self.qkv_tp_replica_groups = (
            get_replica_groups(self.tp_size, self.cp_size)[0] if self.tp_size > 1 else None
        )

    def project_qkv(self, hidden_states: torch.Tensor, rotary_emb=None):
        use_kernel = _can_use_wan_bf16_fused_qkv(
            hidden_states,
            self.qkv_proj_weight,
            self.qkv_proj_bias,
            self.norm_q.weight,
            self.norm_k.weight,
            rotary_emb,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        cos_cache = sin_cache = None
        if rotary_emb is not None:
            cos_cache, sin_cache = rope_rotate_half_caches(rotary_emb, self.head_dim)

        if use_kernel:
            batch, sequence = hidden_states.shape[0], hidden_states.shape[1]
            qkv = _wan_bf16_fused_qkv(
                hidden_states,
                self.qkv_proj_weight,
                self.qkv_proj_bias.unsqueeze(0),
                self.norm_q.weight,
                self.norm_k.weight,
                cos_cache.reshape(batch, sequence, self.head_dim).contiguous(),
                sin_cache.reshape(batch, sequence, self.head_dim).contiguous(),
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                norm_eps=self.norm_q.eps,
                replica_groups=self.qkv_tp_replica_groups,
            )
            q, k, v = torch.tensor_split(qkv, self.qkv_split, dim=-1)
            query = q.unflatten(2, (self.num_heads, self.head_dim))
            key = k.unflatten(2, (self.num_heads, self.head_dim))
        else:
            qkv = torch.matmul(hidden_states, self.qkv_proj_weight) + self.qkv_proj_bias
            q, k, v = torch.tensor_split(qkv, self.qkv_split, dim=-1)
            query = self.norm_q(q).unflatten(2, (self.num_heads, self.head_dim))
            key = self.norm_k(k).unflatten(2, (self.num_heads, self.head_dim))
            if rotary_emb is not None:
                query = apply_rotate_half_rope(query, cos_cache, sin_cache)
                key = apply_rotate_half_rope(key, cos_cache, sin_cache)

        value = v.unflatten(2, (self.num_heads, self.head_dim))
        return query, key, value


class WanCrossAttention(nn.Module):
    """Cross-attention (text + optional image) with raw nn.Parameter (LLaMA pattern)."""

    # Whether this layer holds every attention head on every rank instead of a TP slice.
    _replicates_heads = False

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        added_kv_proj_dim: int | None = None,
        tp_sequence_parallel: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        self.sp_enabled = tp_sequence_parallel
        self.sp_real_len: int | None = None
        self.sp_padded_len: int | None = None
        self.allow_padded_sp_path = True
        self.num_heads = num_heads // tp_size
        tp_inner_dim = self.num_heads * head_dim

        # Q/K/V projections (separate — Q from hidden, K/V from encoder)
        self.q_proj_weight = nn.Parameter(torch.empty(dim, tp_inner_dim))
        set_weight_loader(self.q_proj_weight, col_weight_loader(tp_inner_dim, tp_size))
        self.q_proj_bias = nn.Parameter(torch.empty(tp_inner_dim))
        set_weight_loader(self.q_proj_bias, col_bias_loader(tp_inner_dim, tp_size))

        self.k_proj_weight = nn.Parameter(torch.empty(dim, tp_inner_dim))
        set_weight_loader(self.k_proj_weight, col_weight_loader(tp_inner_dim, tp_size))
        self.k_proj_bias = nn.Parameter(torch.empty(tp_inner_dim))
        set_weight_loader(self.k_proj_bias, col_bias_loader(tp_inner_dim, tp_size))

        self.v_proj_weight = nn.Parameter(torch.empty(dim, tp_inner_dim))
        set_weight_loader(self.v_proj_weight, col_weight_loader(tp_inner_dim, tp_size))
        self.v_proj_bias = nn.Parameter(torch.empty(tp_inner_dim))
        set_weight_loader(self.v_proj_bias, col_bias_loader(tp_inner_dim, tp_size))

        self.norm_q = DistributedRMSNorm(tp_inner_dim, eps=eps)
        self.norm_k = DistributedRMSNorm(tp_inner_dim, eps=eps)

        # Optional image K/V projections
        self.added_kv_proj_dim = added_kv_proj_dim
        if added_kv_proj_dim is not None:
            self.add_k_proj_weight = nn.Parameter(torch.empty(added_kv_proj_dim, tp_inner_dim))
            set_weight_loader(self.add_k_proj_weight, col_weight_loader(tp_inner_dim, tp_size))
            self.add_k_proj_bias = nn.Parameter(torch.empty(tp_inner_dim))
            set_weight_loader(self.add_k_proj_bias, col_bias_loader(tp_inner_dim, tp_size))

            self.add_v_proj_weight = nn.Parameter(torch.empty(added_kv_proj_dim, tp_inner_dim))
            set_weight_loader(self.add_v_proj_weight, col_weight_loader(tp_inner_dim, tp_size))
            self.add_v_proj_bias = nn.Parameter(torch.empty(tp_inner_dim))
            set_weight_loader(self.add_v_proj_bias, col_bias_loader(tp_inner_dim, tp_size))

            self.norm_added_k = DistributedRMSNorm(tp_inner_dim, eps=eps)

        # Output projection
        self.o_proj_weight = nn.Parameter(torch.empty(tp_inner_dim, dim))
        set_weight_loader(self.o_proj_weight, row_weight_loader(tp_inner_dim, tp_size))
        self.o_proj_bias = nn.Parameter(torch.empty(dim))
        if tp_size > 1:
            set_weight_loader(self.o_proj_bias, row_bias_loader(tp_size, dim))

        self.scale = 1.0 / (head_dim**0.5)
        self._can_run_kernel = can_run_kernel

    def _use_padded_sp_path(self) -> bool:
        """Whether cross-attention can stay on the SP-padded sequence, skipping the slice/pad.

        Cross-attention's K/V come from the (never SP-padded) encoder context, so an SP pad row
        is only ever a *query* -- it produces a garbage output row that is reduce-scattered and
        finally discarded by ``[:, :S_cp, :]`` after the last block. It never becomes a key, so
        unlike self-attention it cannot dilute a live token's softmax denominator. The Q/o
        projections and the attention kernel already run on the arbitrary (often odd) real
        ``S_cp``, so the even ``sp_padded_len`` needs no extra alignment: staying padded simply
        drops the ``sp_entry`` slice and the ``sp_exit`` F.pad (the latter is already a no-op once
        the tensor is ``sp_padded_len`` long).
        """
        return (
            getattr(self, "allow_padded_sp_path", True)
            and self.sp_enabled
            and self.tp_size > 1
            and self.sp_padded_len is not None
        )

    @property
    def kv_cache_size(self) -> int:
        return 4 if self.added_kv_proj_dim is not None else 2

    def _project_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self.norm_q(torch.matmul(hidden_states, self.q_proj_weight) + self.q_proj_bias)
        return query.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)

    def project_kv(self, encoder_hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        encoder_hidden_states_img = None
        if self.added_kv_proj_dim is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        key = self.norm_k(
            torch.matmul(encoder_hidden_states, self.k_proj_weight) + self.k_proj_bias
        )
        value = torch.matmul(encoder_hidden_states, self.v_proj_weight) + self.v_proj_bias
        key = key.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)
        value = value.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)

        if encoder_hidden_states_img is None:
            return key, value

        key_img = self.norm_added_k(
            torch.matmul(encoder_hidden_states_img, self.add_k_proj_weight) + self.add_k_proj_bias
        )
        value_img = (
            torch.matmul(encoder_hidden_states_img, self.add_v_proj_weight) + self.add_v_proj_bias
        )
        key_img = key_img.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)
        value_img = value_img.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)
        return key, value, key_img, value_img

    def _attend_with_kv(
        self, query: torch.Tensor, kv_cache: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        key, value = kv_cache[:2]
        hidden_states = nki_attention(query, key, value, self.scale)
        if len(kv_cache) == 4:
            key_img, value_img = kv_cache[2:]
            hidden_states = hidden_states + nki_attention(query, key_img, value_img, self.scale)
        return hidden_states

    def _project_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = o_proj(
            hidden_states,
            self.o_proj_weight,
            self.o_proj_bias.unsqueeze(0),
        )
        if self._replicates_heads:
            # Full-contraction o-proj over every head yields the complete local
            # [B, S_loc, H] shard -- there is no partial sum to reduce-scatter.
            return output
        return sp_exit(
            output,
            self.tp_group,
            self.tp_size,
            self.sp_enabled,
            self.sp_padded_len,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_kv_cache: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Attend the local query shard against already-projected encoder K/V.

        This layer never projects K/V itself: cross-attention K/V depend only on the
        encoder context, which is fixed for a whole generation, so they are projected
        once by :meth:`project_kv` in the cache-generating transformer graph and passed
        in here on every later step. Callers that want the one-shot behaviour pass
        ``attn.project_kv(encoder_hidden_states)``.
        """
        if len(cross_attention_kv_cache) != self.kv_cache_size:
            raise ValueError(
                f"expected {self.kv_cache_size} cross-attention K/V tensors, "
                f"got {len(cross_attention_kv_cache)}"
            )
        if not self._replicates_heads:
            # On the padded fast path, pass real_len=None so sp_entry all-gathers without
            # slicing; the closing sp_exit then reduce-scatters without padding (the tensor
            # is already sp_padded_len long).
            hidden_states = sp_entry(
                hidden_states,
                self.tp_group,
                self.tp_size,
                self.sp_enabled,
                None if self._use_padded_sp_path() else self.sp_real_len,
            )
        query = self._project_query(hidden_states)
        return self._project_output(self._attend_with_kv(query, cross_attention_kv_cache))


class WanTransformerBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and FFN (BF16 leaves)."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        cross_attn_norm: bool = False,
        quantization: str | None = None,
        modules_to_not_convert: frozenset[str] = frozenset(),
        tp_sequence_parallel: bool = False,
    ):
        super().__init__()
        if modules_to_not_convert:
            raise ValueError("modules_to_not_convert is only valid with an FP8 quantization")
        _check_bf16_only(quantization, None)
        self_attention_cls = WanSelfAttentionBF16Fused
        cross_attention_cls = WanCrossAttention
        feed_forward_cls = TPFeedForward

        head_dim = dim // num_heads
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = self_attention_cls(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            tp_sequence_parallel=tp_sequence_parallel,
        )
        self.attn2 = cross_attention_cls(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            tp_sequence_parallel=tp_sequence_parallel,
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.ffn = feed_forward_cls(
            dim=dim,
            inner_dim=ffn_dim,
            dim_out=dim,
            tp_sequence_parallel=tp_sequence_parallel,
        )
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        # Offset of this block's K/V inside the flat per-graph cache. Assigned by
        # WanTransformer3DModel.__init__, which owns the layout.
        self.cross_attention_kv_cache_index = 0

    def set_sp_lengths(self, real_len: int, padded_len: int) -> None:
        for mod in (self.attn1, self.attn2, self.ffn):
            mod.sp_real_len = real_len
            mod.sp_padded_len = padded_len

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        *,
        cross_attention_kv_cache,
    ):
        """Run one block, taking the **whole** flat K/V cache and slicing out its own.

        Every block receives byte-identical arguments, the interface of Cache-DiT's
        ``UnifiedBlocks`` wrapper, which replaces ``self.blocks`` and broadcasts one
        argument list to all of the real blocks (``cache_dit`` ``pattern_base``:
        ``block(hidden_states, encoder_hidden_states, *args, **kwargs)``). Cache-DiT is not
        implemented on Neuron (module docstring); the block keeps its interface. Under that
        wrapper a pre-sliced tuple would give every block the first block's K/V, so the
        slicing happens inside the block, keyed off its own index.

        ``encoder_hidden_states`` is unused: cross-attention K/V are projected once per
        generation and arrive via ``cross_attention_kv_cache``. It stays in the signature,
        by name and position, for Cache-DiT, which detects its forward pattern from the
        parameter names and passes this argument positionally.
        """
        # a Cache-DiT wrapper can pass non-contiguous slices; make all inputs contiguous
        hidden_states = hidden_states.contiguous()
        temb = temb.contiguous()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # Fused AdaLN (norm1): LayerNorm(x) * (1 + scale) + shift, via the NKI kernel with an eager
        # fallback (see adaln_modulate), returning the BF16 normalized tensor. norm1 has no
        # preceding residual add, so it takes the plain (single-output) form.
        norm_h = adaln_modulate(
            hidden_states,
            1 + scale_msa,
            shift_msa,
            eps=self.norm1.eps,
            norm_type="layer_norm",
        )
        attn1_out = self.attn1(norm_h, rotary_emb)

        # norm2 with the attn1 gated residual add fused in: adaln computes hidden_states +
        # gate_msa * attn1_out (returned as the bf16 residual) and normalizes it. norm2 is affine
        # LayerNorm when cross_attn_norm is set, else norm-free (Identity has no weight/bias/eps).
        if isinstance(self.norm2, nn.Identity):
            norm2_type, norm2_w, norm2_b, norm2_eps = "none", None, None, self.norm1.eps
        else:
            norm2_type, norm2_w, norm2_b, norm2_eps = (
                "layer_norm",
                self.norm2.weight,
                self.norm2.bias,
                self.norm2.eps,
            )
        hidden_states, norm_h = adaln_modulate(
            attn1_out,
            norm2_w,
            norm2_b,
            eps=norm2_eps,
            norm_type=norm2_type,
            residual=hidden_states,
            gate=gate_msa,
        )
        cache_start = self.cross_attention_kv_cache_index
        attn2_out = self.attn2(
            norm_h,
            cross_attention_kv_cache[cache_start : cache_start + self.attn2.kv_cache_size],
        )

        # Fused AdaLN (norm3), with the attn2 (un-gated) residual add fused in: the kernel computes
        # ``hidden_states + attn2_out`` (gate=None) and normalizes it.
        hidden_states, norm_h = adaln_modulate(
            attn2_out,
            1 + c_scale_msa,
            c_shift_msa,
            eps=self.norm3.eps,
            norm_type="layer_norm",
            residual=hidden_states,
            gate=None,
        )
        # add3 (FFN gated residual) stays in framework: fusing it would cross the block boundary
        # into the next norm1 and break the single-tensor block output that Cache-DiT expects.
        hidden_states = (hidden_states.float() + self.ffn(norm_h).float() * c_gate_msa).type_as(
            hidden_states
        )
        return hidden_states


_WAN_FP32_PARAMETER_MODULES = frozenset(
    {"time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"}
)


class WanTransformer3DModel(nn.Module):
    """Wan Transformer 3D model for Neuron: raw nn.Parameter blocks + diffusers' pure-math modules.

    Supports context parallelism (CP, the ``context_parallel_size`` of the Neuron groups): the
    patch sequence is split across CP ranks before the transformer blocks and gathered back
    after. Each self-attention layer runs the ring kernel over local K/V (all-gather K/V where
    the kernel cannot run), so local Q attends to the full sequence.
    """

    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads=40,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=13824,
        num_layers=40,
        cross_attn_norm=True,
        eps=1e-6,
        image_dim=None,
        added_kv_proj_dim=None,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        quantization=None,
        modules_to_not_convert=None,
        tp_sequence_parallel=False,
    ):
        super().__init__()
        self._quantization_mode = _check_bf16_only(quantization, modules_to_not_convert)

        self.config = WanConfig(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
            quantization=quantization,
            modules_to_not_convert=None,
            tp_sequence_parallel=tp_sequence_parallel,
        )

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # >>> CP: context parallelism over the Neuron CP group <<<
        cp_group = get_cp_group()
        self.cp_size = cp_group.world_size
        self.cp_group = cp_group if self.cp_size > 1 else None

        self.tp_size = get_tensor_model_parallel_world_size()
        self.sp_enabled = bool(tp_sequence_parallel) and self.tp_size > 1
        self.tp_group = get_tp_group() if self.sp_enabled else None

        # Make the TP/CP groups resolvable to their full replica-group partitions
        # so the torch-native Neuron backend can legalize collectives for SPMD
        # compilation (otherwise: "replica id #N not seen in replica groups").
        register_replica_groups(
            tp_size=get_tensor_model_parallel_world_size(),
            cp_size=self.cp_size,
        )

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    eps,
                    added_kv_proj_dim,
                    cross_attn_norm,
                    quantization=self._quantization_mode,
                    tp_sequence_parallel=tp_sequence_parallel,
                )
                for _ in range(num_layers)
            ]
        )
        # Keep non-registering references to the real blocks for K/V prefill and
        # sequence-parallel metadata.
        self._wan_blocks = tuple(self.blocks)
        # Lay out the flat cross-attention K/V cache and tell each block where its slice
        # starts. Assigned here, in one place, so the offsets tile the flat tuple exactly by
        # construction; a block that reads past its slice is caught by the length check in
        # WanCrossAttention.forward.
        cache_index = 0
        for block in self._wan_blocks:
            block.cross_attention_kv_cache_index = cache_index
            cache_index += block.attn2.kv_cache_size
        self.cross_attention_kv_cache_size = cache_index

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embedding.weight.dtype

    @staticmethod
    def _concat_encoder_context(
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None,
    ) -> torch.Tensor:
        """Join the image and text context the way cross-attention expects to see it.

        Image context comes first; ``WanCrossAttention.project_kv`` splits the two back
        apart by taking the trailing 512 text rows, so the order here and the split
        there must stay in sync.
        """
        if encoder_hidden_states_image is None:
            return encoder_hidden_states
        return torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    def project_cross_attention_kv(self, encoder_context: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Project the embedded encoder context into every block's cross-attention K/V.

        Returns one flat tuple (``cross_attention_kv_cache_size`` tensors) because it is a
        graph boundary; each block slices its own K/V out by
        ``cross_attention_kv_cache_index``. Cross-attention K/V depend only on this context,
        so the first-step DiT graph projects them once and the steady-state graph reuses
        them for every later step.
        """
        kv_cache: list[torch.Tensor] = []
        for block in self._wan_blocks:
            kv_cache.extend(block.attn2.project_kv(encoder_context))
        return tuple(kv_cache)

    def _patch_embed_local_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Patch-embed only this CP rank's tokens: [B, C, F, H, W] -> [B, S / cp_size, inner_dim].

        The patch embedding is a Conv3d whose stride equals its kernel, so it is a linear map applied
        to each non-overlapping patch. shard_select takes its offset from a runtime rank tensor (so all
        ranks share one graph), which the compiler cannot fold into the producer: selecting after the
        embedding materializes the full-sequence [B, S, inner_dim] tensor and gathers from it (1.7 GB
        per core at 1080p, enough to overflow trn2 HBM). Selecting the patches first, at
        C * p_t * p_h * p_w values per token, keeps that tensor out of the graph.
        """
        p_t, p_h, p_w = self.config.patch_size
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        patches = hidden_states.reshape(
            batch_size, num_channels, num_frames // p_t, p_t, height // p_h, p_h, width // p_w, p_w
        )
        # Token order matches patch_embedding(x).flatten(2): (frame, row, column); the per-token
        # feature order matches the Conv3d weight: (channel, p_t, p_h, p_w).
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
            batch_size, -1, num_channels * p_t * p_h * p_w
        )
        patches = shard_select(patches, 1, self.cp_group)
        weight = self.patch_embedding.weight.reshape(self.patch_embedding.out_channels, -1)
        return F.linear(patches, weight, self.patch_embedding.bias)

    def forward_and_cache_cross_attention_kv(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = False,
        attention_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Run the first denoise step and return the K/V it generated.

        This is the cache-generating full DiT graph. Keeping K/V projection inside the
        same graph that consumes it preserves the first step's normal transformer
        numerics, while returning the projected tensors lets the steady-state DiT graph
        consume them on later steps.
        """
        return self.forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            cross_attention_kv_cache=None,
            _return_cross_attention_kv_cache=True,
            **kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        cross_attention_kv_cache: tuple[torch.Tensor, ...] | None = None,
        _return_cross_attention_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor | Transformer2DModelOutput:
        p_t, p_h, p_w = self.config.patch_size
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.repeat(batch_size)
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        rotary_emb = self.rope(hidden_states)
        if self.cp_size > 1:
            S = post_patch_num_frames * post_patch_height * post_patch_width
            if S % self.cp_size != 0:
                raise ValueError(
                    f"Sequence length {S} is not divisible by cp_size {self.cp_size}. "
                    f"Choose a resolution/frame count that yields a divisible patch sequence length."
                )
            # >>> CP: each rank embeds only its own share of the sequence <<<
            hidden_states = self._patch_embed_local_tokens(hidden_states)
        else:
            hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
            )
        )

        timestep_proj = timestep_proj.unflatten(1, (6, -1)).contiguous()

        encoder_context = self._concat_encoder_context(
            encoder_hidden_states, encoder_hidden_states_image
        )

        # >>> CP: hidden_states already holds this rank's tokens; shard rotary_emb to match <<<
        if self.cp_size > 1:
            # Shard rotary_emb to match local token positions
            # rotary_emb: (freqs_cos, freqs_sin) each [1, S, 1, D]
            freqs_cos, freqs_sin = rotary_emb
            freqs_cos = _shard_rope_freqs(freqs_cos, 0, self.cp_group)
            freqs_sin = _shard_rope_freqs(freqs_sin, 1, self.cp_group)
            rotary_emb = (freqs_cos, freqs_sin)

        if self.sp_enabled:
            S_cp = hidden_states.shape[1]
            S_cp_padded = sp_padded_len(S_cp, self.tp_size)
            for block in self._wan_blocks:
                block.set_sp_lengths(S_cp, S_cp_padded)
            if S_cp_padded > S_cp:
                hidden_states = F.pad(hidden_states, (0, 0, 0, S_cp_padded - S_cp))
            hidden_states = shard_select(hidden_states, 1, self.tp_group)

        # The cache-generating first-step graph enters with no K/V and returns the
        # projections it creates. The steady-state graph enters with that cache and
        # skips projection.
        if cross_attention_kv_cache is None:
            cross_attention_kv_cache = self.project_cross_attention_kv(encoder_context)
        elif len(cross_attention_kv_cache) != self.cross_attention_kv_cache_size:
            raise ValueError(
                f"expected {self.cross_attention_kv_cache_size} cross-attention K/V "
                f"tensors, got {len(cross_attention_kv_cache)}"
            )

        # Every block gets the same arguments, including the whole flat cache, and slices
        # out its own K/V. See WanTransformerBlock.forward for why the split stays in the
        # block: a Cache-DiT wrapper (not implemented on Neuron) collapses self.blocks into
        # one module that broadcasts a single argument list to all of the real blocks.
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_context,
                timestep_proj,
                rotary_emb,
                cross_attention_kv_cache=cross_attention_kv_cache,
            )

        if self.sp_enabled:
            hidden_states = sp_all_gather_seq(
                hidden_states, self.tp_group.device_group, self.tp_size
            )
            hidden_states = hidden_states[:, :S_cp, :]

        # Output norm with scale/shift from temb
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(
            hidden_states
        )
        hidden_states = self.proj_out(hidden_states)

        if self.cp_size > 1:
            hidden_states = self.cp_group.all_gather(hidden_states.contiguous(), dim=1)

        # Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if _return_cross_attention_kv_cache:
            return (output, *cross_attention_kv_cache)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def load_weights(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Loads a rank-sharded checkpoint to device with pipelined data movement for efficiency.

        Implements a three-stage pipeline:
        1. Cache thread loads files into OS page cache
        2. Main thread opens each cached file; worker threads read this rank's tensor shards
        3. Device thread transfers tensors from CPU to HBM (device memory)

        This pipelining hides I/O latency by overlapping disk reads, CPU processing,
        and host-to-device transfers.

        Args:
            model_name_or_path: local diffusers ``transformer/`` (or ``transformer_2/``)
                directory of ``.safetensors`` files (the plugin's checkpoint loader reads only
                local directories)
            device: Device to load weights to
        """
        # Preserve sensitive checkpoint values: promoting after loading cannot undo an
        # intermediate BF16 rounding.
        for name, param in self.named_parameters():
            if _WAN_FP32_PARAMETER_MODULES.intersection(name.split(".")):
                param.data = param.data.float()

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Build explicit mappings (model param name -> checkpoint key(s))
        mappings: dict[str, str | list[str]] = {}
        for i in range(self.config.num_layers):
            # Self-attention: fused QKV
            mappings[f"blocks.{i}.attn1.qkv_proj_weight"] = [
                f"blocks.{i}.attn1.to_q.weight",
                f"blocks.{i}.attn1.to_k.weight",
                f"blocks.{i}.attn1.to_v.weight",
            ]
            mappings[f"blocks.{i}.attn1.qkv_proj_bias"] = [
                f"blocks.{i}.attn1.to_q.bias",
                f"blocks.{i}.attn1.to_k.bias",
                f"blocks.{i}.attn1.to_v.bias",
            ]
            # Self-attention: output projection (to_out.0 in checkpoint)
            mappings[f"blocks.{i}.attn1.o_proj_weight"] = f"blocks.{i}.attn1.to_out.0.weight"
            mappings[f"blocks.{i}.attn1.o_proj_bias"] = f"blocks.{i}.attn1.to_out.0.bias"
            # Cross-attention: individual projections (name change: *_proj_weight -> to_*.weight)
            mappings[f"blocks.{i}.attn2.q_proj_weight"] = f"blocks.{i}.attn2.to_q.weight"
            mappings[f"blocks.{i}.attn2.q_proj_bias"] = f"blocks.{i}.attn2.to_q.bias"
            mappings[f"blocks.{i}.attn2.k_proj_weight"] = f"blocks.{i}.attn2.to_k.weight"
            mappings[f"blocks.{i}.attn2.k_proj_bias"] = f"blocks.{i}.attn2.to_k.bias"
            mappings[f"blocks.{i}.attn2.v_proj_weight"] = f"blocks.{i}.attn2.to_v.weight"
            mappings[f"blocks.{i}.attn2.v_proj_bias"] = f"blocks.{i}.attn2.to_v.bias"
            mappings[f"blocks.{i}.attn2.o_proj_weight"] = f"blocks.{i}.attn2.to_out.0.weight"
            mappings[f"blocks.{i}.attn2.o_proj_bias"] = f"blocks.{i}.attn2.to_out.0.bias"
            # Cross-attention: optional image projections
            if self.config.added_kv_proj_dim is not None:
                mappings[f"blocks.{i}.attn2.add_k_proj_weight"] = (
                    f"blocks.{i}.attn2.add_k_proj.weight"
                )
                mappings[f"blocks.{i}.attn2.add_k_proj_bias"] = f"blocks.{i}.attn2.add_k_proj.bias"
                mappings[f"blocks.{i}.attn2.add_v_proj_weight"] = (
                    f"blocks.{i}.attn2.add_v_proj.weight"
                )
                mappings[f"blocks.{i}.attn2.add_v_proj_bias"] = f"blocks.{i}.attn2.add_v_proj.bias"
            # FFN (net.0.proj -> up_proj, net.2 -> down_proj in checkpoint)
            mappings[f"blocks.{i}.ffn.up_proj_weight"] = f"blocks.{i}.ffn.net.0.proj.weight"
            mappings[f"blocks.{i}.ffn.up_proj_bias"] = f"blocks.{i}.ffn.net.0.proj.bias"
            mappings[f"blocks.{i}.ffn.down_proj_weight"] = f"blocks.{i}.ffn.net.2.weight"
            mappings[f"blocks.{i}.ffn.down_proj_bias"] = f"blocks.{i}.ffn.net.2.bias"

        checkpoint = SafetensorsCheckpoint(model_name_or_path)
        logged_casts: set = set()
        for name, param in self.named_parameters():
            set_weight_loader(
                param,
                cast_weight_loader(get_weight_loader(param), param.dtype, name, logged_casts),
            )

        load_result = checkpoint.load_sharded_pipelined(
            tp_rank,
            tp_size,
            self,
            mappings,
            device,
        )
        state_dict = load_result.state_dict

        self.load_state_dict(state_dict, strict=False, assign=True)
