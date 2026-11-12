# SPDX-License-Identifier: Apache-2.0
"""LingBot-World-V2 causal-fast DiT for Neuron (BF16, tensor parallel).

Checkpoint: ``robbyant/lingbot-world-v2-14b-causal-fast-diffusers`` ``transformer/``
(``CausalLingBotWorldTransformer3DModel``). The DiT is the Wan2.1/2.2 14B shape (40 layers,
dim 5120, 40 heads x 128, FFN 13824) with three differences, all handled here:

* **Causal chunks over a paged K/V history.** One forward denoises one chunk of 3 latent frames;
  its self-attention attends, without a mask, to the committed history (sink + sliding window) in
  the paged KV pool and to its own K/V
  (:func:`~sglang_neuron.diffusion.attention.paged_attention.attend_over_paged_history`).
  The ``t = 0`` commit pass's self-attention writes the chunk's K/V into the pool.
* **RoPE with a start-frame offset.** The ``(cos, sin)`` tables are a graph input, built per chunk
  (:meth:`LingBotTransformer3DModel.chunk_prep`, equal to :func:`.chunk_prep.lingbot_rotary_tables`)
  for the chunk's start frame, so one graph serves every chunk position.
* **Camera conditioning.** Plücker rays are patch-embedded and refined by two top-level layers;
  every block then applies ``x = (1 + scale) * x + shift`` after the self-attention residual, with
  ``scale``/``shift`` from a per-block injector MLP.

The timestep is uniform over a chunk, so its embedding is computed once at sequence length 1 and
broadcast. Cross-attention K/V of the 512 text tokens are projected once per prompt by
:meth:`LingBotTransformer3DModel.project_cross_attention_kv` and passed to every forward.

Reuses the Neuron Wan2.2 leaves: the fused QKV CTE kernel (projection, across-heads Q/K RMSNorm
and rotate-half RoPE in one launch, Q/K lanes de-interleaved at load like
``WanSelfAttentionBF16Fused``), cross-attention, the NKI MLP and o-proj kernels and the fused AdaLN
kernel. The paged pool holds K in that de-interleaved lane order.

Context parallelism (rank = cp_rank * tp_size + tp_rank): the chunk's video and camera tokens and
the RoPE tables are split contiguously along the sequence over the CP group; per-token modules run
on the shard; self-attention all-gathers the chunk K/V over CP and
attends each query shard to the history plus the full chunk; the projected output is all-gathered
before unpatchify.
"""

import math
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.normalization import FP32LayerNorm
from sglang_neuron.diffusion.distributed.groups import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang_neuron.diffusion.distributed.groups import get_tp_group
from sglang_neuron.parallel.groups import shard_select
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    get_weight_loader,
    set_weight_loader,
)

# Upstream calls the context-parallel group its "sequence-parallel" group.
from sglang_neuron.diffusion.distributed.groups import get_cp_group

from sglang_neuron.diffusion.attention.cp_attention import o_proj
from sglang_neuron.diffusion.attention.paged_attention import (
    PagedAttentionInputs,
    attend_over_paged_history,
)
from sglang_neuron.diffusion.distributed.parallel_state import (
    get_replica_groups,
    register_replica_groups,
)
from sglang_neuron.diffusion.distributed.sequence_parallel import (
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
    row_bias_loader,
    row_weight_loader,
)
from sglang_neuron.diffusion.layers.feed_forward import TPFeedForward
from sglang_neuron.diffusion.layers.rope_rotate_half import apply_rotate_half_rope
from sglang_neuron.diffusion.models.lingbot_world.chunk_prep import (
    assemble_rotary_tables,
    camera_token_grid,
    plucker_patches,
)
from sglang_neuron.diffusion.models.wan2_2.wan2_2_transformer import (
    WanCrossAttention,
    WanSelfAttention,
    _can_use_wan_bf16_fused_qkv,
    _install_qk_norm_gamma_loader,
    _wan_bf16_fused_qkv,
)
from sglang_neuron.diffusion.quantization.adaln_kernels import adaln_modulate


def _load_in_fp32(*params: nn.Parameter) -> None:
    """Mark parameters that :meth:`LingBotTransformer3DModel.load_weights` loads in FP32.

    An attribute on the parameter (like its weight loader): the timestep path and the modulation
    tables it adds to stay FP32 whatever dtype the model is cast to before loading.
    """
    for param in params:
        param.load_in_fp32 = True


def loads_in_fp32(param: nn.Parameter) -> bool:
    """Whether :func:`_load_in_fp32` marked ``param``."""
    return getattr(param, "load_in_fp32", False)


def _fused_col_weight_loader(part_size: int, num_parts: int) -> SafetensorsWeightLoader:
    """Column-parallel loader fusing ``num_parts`` checkpoint ``[out, in]`` weights.

    Each rank takes its ``part_size`` output rows of every part and stores them transposed and
    side by side: ``[in, num_parts * part_size]``.
    """

    def transform(slices, rank):
        assert len(slices) == num_parts
        start = rank * part_size
        parts = [sl[start : start + part_size] for sl in slices]
        return torch.cat(parts, dim=0).t().contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _fused_col_bias_loader(part_size: int, num_parts: int) -> SafetensorsWeightLoader:
    def transform(slices, rank):
        assert len(slices) == num_parts
        start = rank * part_size
        return torch.cat([sl[start : start + part_size] for sl in slices], dim=0)

    return SafetensorsWeightLoader(transform=transform)


def cp_all_gather_seq(
    x: torch.Tensor, cp_group, dim: int = 1, ordered: bool = True
) -> torch.Tensor:
    """All-gather CP sequence shards along ``dim`` back into ``shard_select`` order.

    On the Trn2 physical mesh the order in which ``all_gather`` stacks the CP ranks' blocks is not
    always the ``rank_in_group`` order that ``shard_select`` sharded by: on TP 8 x CP 4 the ranks
    with TP rank 4-7 receive the blocks as ``[1, 0, 3, 2]`` (observed; ranks 0-3 get ``[0, 1, 2,
    3]``). Each rank therefore also gathers the shard index every block came from (its
    ``rank_in_group_tensor``) and moves the blocks back in place with an ``index_select`` (exact).
    ``ordered=False`` skips the fix-up for consumers that do not depend on the order (attention
    over keys).
    """
    gathered = cp_group.all_gather(x.contiguous(), dim=dim)
    if not ordered:
        return gathered
    return cp_reorder_blocks(gathered, cp_group, dim=dim)


def cp_reorder_blocks(gathered: torch.Tensor, cp_group, dim: int) -> torch.Tensor:
    """Put an arrival-order CP all-gather (see :func:`cp_all_gather_seq`) into shard order."""
    cp_size = cp_group.world_size
    source = cp_group.all_gather(cp_group.rank_in_group_tensor.reshape(1).to(torch.int32), dim=0)
    # place[i, j] = 1 if gathered block i is shard j; order[j] = the block holding shard j.
    place = source.reshape(cp_size, 1) == torch.arange(
        cp_size, device=gathered.device, dtype=torch.int32
    )
    order = torch.argmax(place.to(torch.int32), dim=0)
    shape = gathered.shape
    blocks = gathered.reshape(*shape[:dim], cp_size, shape[dim] // cp_size, *shape[dim + 1 :])
    return torch.index_select(blocks, dim, order).reshape(shape)


# ------------------------------------------------------------------------------- block norm
# The residual stream of a block: one gated residual + camera + LayerNorm step per norm site
# (norm1: self-attention AdaLN with the previous block's gated FFN residual folded in; norm3:
# cross-attention affine LayerNorm with the gated self-attention residual and the camera
# modulation folded in; norm2: FFN AdaLN with the cross-attention residual folded in). The steps
# run on Wan2.2's AdaLN kernel (:func:`adaln_modulate`): norm1 and norm2 are one gated-residual
# AdaLN launch each; norm3 is a gated-residual launch with no norm and the per-token camera
# ``(1 + scale, shift)`` as its modulation, then an affine LayerNorm launch. The PyTorch ops (same
# rounding points) are the fallback where the kernel cannot run (CPU mode, fake-tensor tracing,
# kernels disabled, other shapes or dtypes). The camera modulation can be the raw TP all-gather of
# the per-block ``[scale | shift]`` shards (:func:`camera_from_gather`).


def camera_from_gather(cam_raw: torch.Tensor, hidden_dim: int, batch_tokens: int):
    """``(scale, shift)`` ``[1, S, H]`` from the raw ``[T * S, 2 * H / T]`` TP all-gather output."""
    tp = cam_raw.shape[0] // batch_tokens
    local = cam_raw.shape[1] // 2
    out = cam_raw.reshape(tp, 1, batch_tokens, 2, local).permute(1, 2, 3, 0, 4)
    out = out.reshape(1, batch_tokens, 2, tp * local)
    return out[:, :, 0], out[:, :, 1]


class SeqShard:
    """This TP rank's token rows of the residual stream (sequence parallelism over TP).

    The TP ranks of one CP shard all hold the same ``S`` tokens after every TP all-reduce, so the
    residual stream's per-token work (the three :func:`residual_norm` sites per block) would run
    ``T`` times. With a ``SeqShard`` each rank keeps rows ``[r * L, (r + 1) * L)`` of the stream
    (``L = ceil(S / T)``; the last rank's tail repeats row ``S - 1``), runs the norms on them and
    all-gathers only the norm outputs, which the column-parallel matmuls read. Per-token math is
    unchanged, so the result is bit-identical to replicated norms.

    The sub-layers reduce-scatter their outputs onto the same rows of the padded sequence
    (``T * L`` rows; the last rank's tail rows are padding, never gathered back) instead of
    all-reducing them (:meth:`LingBotTransformer3DModel._set_sequence_parallel`).

    The rows come from the TP group's stamped rank tensor (one graph for every rank). The TP
    all-gather stacks the ranks in rank order, as :meth:`LingBotCameraInjector.gather_raw`
    relies on too.
    """

    def __init__(
        self,
        tokens: int,
        tp_size: int,
        tp_group: torch.distributed.ProcessGroup | None,
        device: torch.device | str,
    ):
        """
        Args:
            tokens: the stream's tokens on this CP rank (``S``).
            tp_size: the TP degree (``T``).
            tp_group: the TP device group the norm outputs are all-gathered over.
            device: where the row indices live.
        """
        rank = get_tp_group().rank_in_group_tensor.reshape(()).to(torch.int64)
        self.tokens = tokens
        self.tp_size = tp_size
        self.local = -(-tokens // tp_size)
        self.tp_group = tp_group
        rows = torch.arange(self.local, device=device, dtype=torch.int64) + rank * self.local
        self.rows = rows.clamp(max=tokens - 1)

    def local_rows(self, x: torch.Tensor) -> torch.Tensor:
        """``[1, S, H]`` -> this rank's ``[1, L, H]``."""
        return x.index_select(1, self.rows)

    def camera_rows(self, cam: torch.Tensor) -> torch.Tensor:
        """A camera input of :func:`residual_norm` over this rank's rows: the raw TP gather
        ``[T * S, W]`` -> ``[T * L, W]``."""
        ranks = torch.arange(self.tp_size, device=cam.device, dtype=torch.int64)
        idx = (ranks.reshape(-1, 1) * self.tokens + self.rows.reshape(1, -1)).reshape(-1)
        return cam.index_select(0, idx)



def _residual_norm_torch(hidden, *, eps, residual, gate, cam, mul, add, round_mid=True):
    dtype = hidden.dtype
    stream = hidden
    if residual is not None:
        x = hidden.float()
        if gate is not None:
            x = x * gate.float()
        stream = residual.float() + x
        if round_mid or cam is None:
            stream = stream.to(dtype)
    if cam is not None:
        scale, shift = (
            cam
            if isinstance(cam, tuple)
            else camera_from_gather(cam, hidden.shape[-1], hidden.shape[-2])
        )
        stream = ((1.0 + scale.float()) * stream.float() + shift.float()).to(dtype)
    stream = stream.to(dtype)
    norm = F.layer_norm(stream.float(), (stream.shape[-1],), None, None, eps)
    if mul is not None:
        norm = norm * mul.float()
    if add is not None:
        norm = norm + add.float()
    out = norm.to(dtype)
    if residual is not None or cam is not None:
        return stream, out
    return out


def _can_use_residual_norm_kernel(hidden, residual, cam, round_mid) -> bool:
    """Whether :func:`residual_norm` runs on the AdaLN kernel (kernel choice
    ``lingbot_residual_norm``; the launches themselves report as ``adaln``)."""
    from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel
    from sglang_neuron.diffusion.kernels.kernel_choice import device_reason, record_kernel_choice

    name = "lingbot_residual_norm"
    if not can_run_kernel(hidden):
        return record_kernel_choice(name, False, device_reason(hidden))
    if hidden.dim() != 3 or hidden.shape[0] != 1:
        return record_kernel_choice(name, False, "not a [1, S, H] stream")
    if hidden.dtype != torch.bfloat16 or (residual is not None and residual.dtype != hidden.dtype):
        return record_kernel_choice(name, False, "non-bf16 stream")
    if cam is not None and not round_mid:
        return record_kernel_choice(name, False, "fp32 residual into the camera (round_mid=False)")
    return record_kernel_choice(name, True, "eligible (AdaLN kernel)")


def residual_norm(
    hidden: torch.Tensor,
    *,
    eps: float,
    residual: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    cam: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    mul: torch.Tensor | None = None,
    add: torch.Tensor | None = None,
    round_mid: bool = True,
):
    """``t = bf16(residual + gate * hidden)``; ``t = bf16((1 + scale) * t + shift)``;
    ``out = bf16(LayerNorm(t) * mul + add)``.

    ``hidden`` / ``residual`` are ``[1, S, H]`` bf16. ``gate`` / ``mul`` / ``add`` are per-hidden
    (``[1, 1, H]``, ``[1, H]`` or ``[H]``, any float dtype). ``cam`` is the raw camera TP
    all-gather ``[T * S, 2 * H / T]`` (see :meth:`LingBotCameraInjector.gather_raw`) or an already
    gathered ``(scale, shift)`` pair. Each step is skipped when its inputs are None. Returns
    ``(t, out)`` when ``residual`` or ``cam`` is given, else ``out``.

    On the AdaLN kernel the modulation (``mul``, ``add``, ``1 + scale``, ``shift``) is in bf16 and
    the gate in fp32, as at Wan2.2's AdaLN sites; the PyTorch fallback applies the modulation in
    fp32. The two agree to within about one bf16 rounding.

    ``round_mid=False`` (with ``gate`` and ``cam``) keeps ``residual + gate * hidden`` in fp32
    into the camera modulation, rounding once at the end, as the XLA-lowered PyTorch ops do (the
    PyTorch path only).
    """
    if gate is not None and residual is None:
        raise ValueError("gate requires residual")
    if not _can_use_residual_norm_kernel(hidden, residual, cam, round_mid):
        return _residual_norm_torch(
            hidden,
            eps=eps,
            residual=residual,
            gate=gate,
            cam=cam,
            mul=mul,
            add=add,
            round_mid=round_mid,
        )

    H = hidden.shape[-1]
    dtype = hidden.dtype
    eps = float(eps)

    def _row(v):
        return None if v is None else v.reshape(1, 1, H).to(dtype).contiguous()

    def _gate(v):
        # The gate stays fp32, as the AdaLN kernel takes it (the gated add matches the
        # framework add); the modulation is in the stream's dtype.
        return None if v is None else v.reshape(1, 1, H).float().contiguous()

    if cam is None:
        if residual is None:
            return adaln_modulate(hidden, _row(mul), _row(add), eps=eps, norm_type="layer_norm")
        return adaln_modulate(
            hidden,
            _row(mul),
            _row(add),
            eps=eps,
            norm_type="layer_norm",
            residual=residual,
            gate=_gate(gate),
        )
    # norm3: the gated residual add with the per-token camera modulation and no norm, then the
    # affine LayerNorm of the modulated stream.
    scale, shift = cam if isinstance(cam, tuple) else camera_from_gather(cam, H, hidden.shape[-2])
    cam_mul = (1.0 + scale.float()).to(dtype).contiguous()
    cam_add = shift.to(dtype).contiguous()
    if residual is None:
        stream = adaln_modulate(hidden, cam_mul, cam_add, eps=eps, norm_type="none")
    else:
        _, stream = adaln_modulate(
            hidden,
            cam_mul,
            cam_add,
            eps=eps,
            norm_type="none",
            residual=residual,
            gate=_gate(gate),
        )
    out = adaln_modulate(stream, _row(mul), _row(add), eps=eps, norm_type="layer_norm")
    return stream, out


class _CameraRefineMLP(nn.Module):
    """``x + row(silu(col(x)))``: the residual camera refinement used at the top and per block."""

    def __init__(self, dim: int):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        local = dim // tp_size
        self.layer1_weight = nn.Parameter(torch.empty(dim, local))
        set_weight_loader(self.layer1_weight, col_weight_loader(local, tp_size))
        self.layer1_bias = nn.Parameter(torch.empty(local))
        set_weight_loader(self.layer1_bias, col_bias_loader(local, tp_size))
        self.layer2_weight = nn.Parameter(torch.empty(local, dim))
        set_weight_loader(self.layer2_weight, row_weight_loader(local, tp_size))
        self.layer2_bias = nn.Parameter(torch.empty(dim))
        if tp_size > 1:
            set_weight_loader(self.layer2_bias, row_bias_loader(tp_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(torch.matmul(x, self.layer1_weight) + self.layer1_bias)
        h = torch.matmul(h, self.layer2_weight) + self.layer2_bias
        return x + sp_exit(h, self.tp_group, self.tp_size, sp_enabled=False)  # the TP all-reduce


class LingBotCameraInjector(nn.Module):
    """Per-block camera modulation: ``(scale, shift)`` from the patch-embedded camera tokens.

    Checkpoint: ``cam_injector_layer{1,2}`` (column / row parallel, residual) and
    ``cam_{scale,shift}_layer`` (fused column parallel, gathered across TP ranks).
    """

    def __init__(self, dim: int):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        self.dim = dim
        self.injector = _CameraRefineMLP(dim)
        local = dim // tp_size
        self.scale_shift_weight = nn.Parameter(torch.empty(dim, 2 * local))
        set_weight_loader(self.scale_shift_weight, _fused_col_weight_loader(local, 2))
        self.scale_shift_bias = nn.Parameter(torch.empty(2 * local))
        set_weight_loader(self.scale_shift_bias, _fused_col_bias_loader(local, 2))

    def project(self, camera_hidden_states: torch.Tensor) -> torch.Tensor:
        """This TP rank's ``[B, S, 2 * dim / tp]`` column shard of ``[scale | shift]``.

        All of the injector's compute; what is left (:meth:`gather_raw`) is one TP all-gather.
        """
        features = self.injector(camera_hidden_states)
        return torch.matmul(features, self.scale_shift_weight) + self.scale_shift_bias

    def gather_raw(self, scale_shift_shard: torch.Tensor) -> torch.Tensor:
        """The TP all-gather of a ``[1, S, 2 * dim / tp]`` shard, untransposed: ``[tp * S, 2 * dim / tp]``.

        Row ``r * S + s`` is TP rank ``r``'s ``[scale_r | shift_r]`` of token ``s``; the block
        norm (:func:`residual_norm`) reads it in place.
        """
        B, S, width = scale_shift_shard.shape
        if B != 1:
            raise ValueError("gather_raw expects batch 1")
        flat = scale_shift_shard.reshape(S, width).contiguous()
        if self.tp_size == 1:
            return flat
        out = torch.empty((self.tp_size * S, width), dtype=flat.dtype, device=flat.device)
        dist.all_gather_into_tensor(out, flat, group=self.tp_group)
        return out

class LingBotSelfAttention(WanSelfAttention):
    """Fused-QKV self-attention whose keys are the paged history plus the current chunk.

    :meth:`project_qkv` runs the QKV projection, the across-heads Q/K RMSNorm (TP all-reduce) and
    RoPE in one launch of the BF16 QKV CTE kernel. It uses the gate and the launcher of
    ``WanSelfAttentionBF16Fused``.

    The kernel applies rotate-half RoPE. Thus, as in Wan, the load de-interleaves the weight and
    bias columns of each Q/K head and the norm gammas (``deinterleave_perm`` order), and the RoPE
    tables are rotate-half. Q and K get the same lane permutation, so the scores are those of the
    official interleaved RoPE. Q, K and the K that this layer commits to the paged pool stay in
    the de-interleaved lane order. Code that reads the pool in the checkpoint lane order applies
    the inverse of ``deinterleave_perm``.

    The fallback (CPU, shapes outside the kernel) is the torch path in the same lane order: matmul
    plus bias, then QK-norm and RoPE in FP32 with one rounding. The kernel rounds Q/K to BF16
    after the norm and again after RoPE, so the two paths differ by one BF16 rounding.

    The history is this layer's views of the paged KV pool, read through the pass's block table
    (:class:`PagedAttentionInputs`); on the commit pass the attention also writes the chunk's K/V
    (after QK-norm and RoPE) into the pool.

    Context parallelism: ``hidden_states`` is this CP rank's sequence shard of the chunk. The
    chunk's K/V are all-gathered over the CP group, and each rank
    attends its query shard to the visible history plus the whole chunk -- exact, with no ring
    or softmax merge.
    """

    #: Sequence parallelism (``LingBotTransformer3DModel._set_sequence_parallel``): the padded
    #: length the output is reduce-scattered at (``None``: all-reduce).
    sp_padded_len = None
    # The projection is the fused kernel (or torch.matmul), never NF.qkv_proj.
    _uses_native_qkv_projection = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.num_heads * self.head_dim
        set_weight_loader(
            self.qkv_proj_weight,
            deinterleaved_fused_qkv_weight_loader(width, width, self.tp_size, self.head_dim),
        )
        set_weight_loader(
            self.qkv_proj_bias,
            deinterleaved_fused_qkv_bias_loader(width, width, head_dim=self.head_dim),
        )
        # The kernel takes each gamma as a [1, width] row, de-interleaved like Q/K.
        _install_qk_norm_gamma_loader(self, deinterleave=True)
        # TP replica groups of the kernel's in-kernel across-heads-norm all-reduce.
        self.qkv_tp_replica_groups = (
            get_replica_groups(self.tp_size, self.cp_size)[0] if self.tp_size > 1 else None
        )

    def project_qkv(self, hidden_states: torch.Tensor, rotary_emb):  # type: ignore[override]
        """Q/K/V token-major ``[B, S, N_local, D]``, Q/K normed and rotated, in the de-interleaved
        lane order. ``rotary_emb``: the rotate-half ``(cos, sin)`` ``[B, S, D]`` FP32 tables."""
        cos, sin = rotary_emb
        heads = (self.num_heads, self.head_dim)
        if _can_use_wan_bf16_fused_qkv(
            hidden_states,
            self.qkv_proj_weight,
            self.qkv_proj_bias,
            self.norm_q.weight,
            self.norm_k.weight,
            rotary_emb,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ):
            qkv = _wan_bf16_fused_qkv(
                hidden_states,
                self.qkv_proj_weight,
                self.qkv_proj_bias.unsqueeze(0),
                self.norm_q.weight,
                self.norm_k.weight,
                cos.contiguous(),
                sin.contiguous(),
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                norm_eps=self.norm_q.eps,
                replica_groups=self.qkv_tp_replica_groups,
            )
            q, k, v = torch.tensor_split(qkv, self.qkv_split, dim=-1)
            query, key = q.unflatten(2, heads), k.unflatten(2, heads)
        else:
            qkv = torch.matmul(hidden_states, self.qkv_proj_weight) + self.qkv_proj_bias
            q, k, v = torch.tensor_split(qkv, self.qkv_split, dim=-1)
            # QK-norm and RoPE in FP32 with one rounding at the end (the norm's output feeds RoPE
            # unrounded).
            cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
            query = self.norm_q(q.float()).unflatten(2, heads)
            key = self.norm_k(k.float()).unflatten(2, heads)
            query = apply_rotate_half_rope(query, cos, sin).to(qkv.dtype)
            key = apply_rotate_half_rope(key, cos, sin).to(qkv.dtype)
        return query, key, v.unflatten(2, heads)

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        key_pool: torch.Tensor,
        value_pool: torch.Tensor,
        paged: PagedAttentionInputs,
    ) -> torch.Tensor | None:
        """``key_pool`` / ``value_pool``: this layer's views of the paged KV pool (K in the
        de-interleaved lane order); ``paged``: the pass's block table, mask bounds and, on the
        commit pass, the per-token update indices (the chunk's K/V are then written into the
        pool). Returns the output (``None`` with ``paged.write_only``: no attention, no o-proj,
        no TP collective)."""
        query, key, value = self.project_qkv(hidden_states, rotary_emb)
        attention = attend_over_paged_history(
            query,
            key,
            value,
            key_pool,
            value_pool,
            paged,
            self.scale,
            self.cp_group,
            getattr(self, "cp_replica_groups", ()),
        )
        if getattr(paged, "write_only", False) and paged.kv_cache_update_idx is not None:
            return None  # the commit pass's last layer: only the pool write is needed
        # As Wan2.2's self-attention: with sequence parallelism, pad the narrow d-major
        # [B, N, D, S] attention so the o-proj emits the padded rows (sp_exit's own pad then
        # self-skips), and sp_exit reduce-scatters them onto this rank's rows; without it,
        # sp_exit is the TP all-reduce.
        sp_enabled = self.sp_padded_len is not None
        if sp_enabled and self.tp_size > 1:
            attention = F.pad(attention, (0, self.sp_padded_len - attention.shape[-1]))
        output = o_proj(attention, self.o_proj_weight, self.o_proj_bias.unsqueeze(0))
        return sp_exit(output, self.tp_group, self.tp_size, sp_enabled, self.sp_padded_len)


class LingBotCrossAttention(WanCrossAttention):
    """Wan cross-attention; ``num_local_heads`` (its TP-local heads) for vLLM-Omni's KV cache spec."""

    @property
    def num_local_heads(self) -> int:
        return self.num_heads


class LingBotTransformerBlock(nn.Module):
    """AdaLN self-attention over the history, camera modulation, cross-attention, AdaLN FFN.

    Mirrors ``CausalWanAttentionBlock`` (official) / ``LingBotAttentionBlock`` (vLLM-Omni). The
    checkpoint names its norms ``norm1`` (self-attn, no affine), ``norm3`` (cross-attn, affine) and
    ``norm2`` (FFN, no affine); this module keeps those names.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        head_dim = dim // num_heads
        self.eps = eps
        self.self_attn = LingBotSelfAttention(
            dim=dim, num_heads=num_heads, head_dim=head_dim, eps=eps
        )
        self.camera = LingBotCameraInjector(dim)
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=True)
        self.cross_attn = LingBotCrossAttention(
            dim=dim, num_heads=num_heads, head_dim=head_dim, eps=eps, added_kv_proj_dim=None
        )
        self.ffn = TPFeedForward(dim=dim, inner_dim=ffn_dim, dim_out=dim)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # The modulation table (the FP32 timestep projection is added to it) and norm3 load FP32.
        _load_in_fp32(self.modulation, self.norm3.weight, self.norm3.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        camera_hidden_states: torch.Tensor,
        key_pool: torch.Tensor,
        value_pool: torch.Tensor,
        cross_attention_kv: tuple[torch.Tensor, torch.Tensor],
        paged: PagedAttentionInputs,
        camera_modulation: torch.Tensor | None = None,
        kv_only: bool = False,
        pending_residual: tuple[torch.Tensor, torch.Tensor] | None = None,
        defer_ffn_residual: bool = False,
        seq_shard: SeqShard | None = None,
    ):
        """One block. ``timestep_proj`` is ``[B, 6, dim]`` (uniform over the chunk).

        With ``seq_shard`` the residual stream ``hidden_states`` (in and out, and a pending FFN
        output) holds only this TP rank's token rows (:class:`SeqShard`): the self-attention
        gathers its input, the cross-attention and the FFN all-gather theirs, and every
        sub-layer reduce-scatters its output onto the rows.

        ``key_pool`` / ``value_pool`` / ``paged``: the self-attention's paged history (see
        :meth:`LingBotSelfAttention.forward`). ``kv_only`` (the last block of the commit pass):
        stop after the self-attention, which has written the chunk's K/V into the pool.

        ``camera_modulation`` is this block's precomputed camera ``[scale | shift]`` TP shard
        (:meth:`LingBotCameraInjector.project`) or its packed rows from
        :meth:`LingBotTransformer3DModel.chunk_prep`; when it is given,
        ``camera_hidden_states`` is unused (may be ``None``) and only the TP all-gather runs.

        The residual stream runs one residual + camera + LayerNorm step per norm site
        (:func:`residual_norm`, on the AdaLN kernel).
        ``pending_residual`` is the previous block's ``(ffn_out, gate_ffn)``; its gated FFN
        residual add is folded into this block's norm1. With ``defer_ffn_residual`` the block
        leaves its own FFN residual pending and returns ``(hidden_states, (ffn_out, gate_ffn))``,
        where ``hidden_states`` is the stream before that add. Returns ``(hidden_states, None)``
        otherwise.
        """
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = (
            self.modulation.float() + timestep_proj.float()
        ).chunk(6, dim=1)
        return self._forward_fused(
            hidden_states,
            (shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn),
            rotary_emb,
            camera_hidden_states,
            key_pool,
            value_pool,
            cross_attention_kv,
            paged,
            camera_modulation,
            kv_only,
            pending_residual,
            defer_ffn_residual,
            seq_shard,
        )

    def _camera_for_norm(self, camera_hidden_states, camera_modulation, seq_shard=None):
        """The camera input of the fused norm3: the raw TP all-gather, or a gathered pair.

        A 4-D ``camera_modulation`` is this rank's rows' ``(scale, shift)`` packed ``[2, 1, L, dim]``,
        built once per chunk by :meth:`LingBotTransformer3DModel.chunk_prep`; a 2-D one is the raw
        gather over the stream's rows.
        """
        if isinstance(camera_modulation, torch.Tensor) and camera_modulation.dim() == 4:
            # From chunk_prep: this rank's rows' (scale, shift), packed [2, 1, L, dim].
            return camera_modulation[0], camera_modulation[1]
        if isinstance(camera_modulation, torch.Tensor) and camera_modulation.dim() == 2:
            return camera_modulation
        cam = self._camera_raw(camera_hidden_states, camera_modulation)
        return cam if seq_shard is None else seq_shard.camera_rows(cam)

    def _camera_raw(self, camera_hidden_states, camera_modulation):
        if camera_modulation is None:
            return self.camera.gather_raw(self.camera.project(camera_hidden_states))
        return self.camera.gather_raw(camera_modulation)

    def _forward_fused(
        self,
        hidden_states,
        modulation,
        rotary_emb,
        camera_hidden_states,
        key_pool,
        value_pool,
        cross_attention_kv,
        paged,
        camera_modulation,
        kv_only,
        pending_residual,
        defer_ffn_residual,
        seq_shard=None,
    ):
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = modulation
        sp = seq_shard
        ident = lambda x: x  # noqa: E731
        # With sequence parallelism the sub-layers reduce-scatter their outputs onto the stream's
        # rows already; attn_in: a norm output on the rows -> every token (exactly S), the
        # self-attention's input (the cross-attention and the FFN all-gather their own).
        attn_in = ident if sp is None else lambda x: sp_entry(x, sp.tp_group, sp.tp_size, True, sp.tokens)
        # norm1, with the previous block's gated FFN residual folded in.
        if pending_residual is None:
            norm_h = residual_norm(hidden_states, eps=self.eps, mul=1 + scale_msa, add=shift_msa)
        else:
            ffn_out, prev_gate = pending_residual
            hidden_states, norm_h = residual_norm(
                ffn_out,
                eps=self.eps,
                residual=hidden_states,
                gate=prev_gate,
                mul=1 + scale_msa,
                add=shift_msa,
            )
        attn_out = self.self_attn(attn_in(norm_h), rotary_emb, key_pool, value_pool, paged)
        if kv_only:
            return hidden_states, None
        # norm3: gated self-attention residual + camera modulation + affine LayerNorm. The
        # residual sum is rounded to bf16 before the camera modulation (PyTorch's eager rounding
        # points). Kept in fp32 (one rounding, as the XLA-lowered ops do), it drifts the 40-layer
        # 8-chunk rollout out of the BF16 envelope in the steady state (chunks 5-7); rounded,
        # every chunk stays inside the envelope.
        hidden_states, norm_h = residual_norm(
            attn_out,
            eps=self.norm3.eps,
            residual=hidden_states,
            gate=gate_msa,
            cam=self._camera_for_norm(camera_hidden_states, camera_modulation, sp),
            mul=self.norm3.weight,
            add=self.norm3.bias,
            round_mid=True,
        )
        attn2_out = self.cross_attn(norm_h, cross_attention_kv)
        # norm2: cross-attention residual + AdaLN.
        hidden_states, norm_h = residual_norm(
            attn2_out,
            eps=self.eps,
            residual=hidden_states,
            mul=1 + scale_ffn,
            add=shift_ffn,
        )
        ffn_out = self.ffn(norm_h)
        if defer_ffn_residual:
            return hidden_states, (ffn_out, gate_ffn)
        hidden_states = (hidden_states.float() + ffn_out.float() * gate_ffn).to(hidden_states.dtype)
        return hidden_states, None


class _LingBotHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

        # The modulation table loads FP32: the FP32 timestep embedding is added to it.
        _load_in_fp32(self.modulation)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        pending_residual: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        shift, scale = (self.modulation.float() + temb.float().unsqueeze(1)).chunk(2, dim=1)
        if pending_residual is not None:
            # The last block's gated FFN residual. The output norm stays in fp32 (its precision
            # reaches the flow directly; a bf16 modulation costs L-inf there).
            ffn_out, gate = pending_residual
            hidden_states = (hidden_states.float() + ffn_out.float() * gate).to(hidden_states.dtype)
        normed = self.norm(hidden_states.float()) * (1 + scale) + shift
        return self.head(normed.to(hidden_states.dtype))


def _sinusoidal_embedding(dim: int, timestep: torch.Tensor) -> torch.Tensor:
    """Official ``sinusoidal_embedding_1d`` in FP32: ``[cos(t * f), sin(t * f)]``."""
    half = dim // 2
    exponent = torch.arange(half, dtype=torch.float32, device=timestep.device) / half
    frequencies = torch.pow(10000.0, -exponent)
    phase = timestep.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat((torch.cos(phase), torch.sin(phase)), dim=1)


def _patchify(x: torch.Tensor, patch_size) -> torch.Tensor:
    """``[B, C, F, H, W] -> [B, (F/pt)(H/ph)(W/pw), C*pt*ph*pw]`` in Conv3d token/feature order."""
    p_t, p_h, p_w = patch_size
    B, C, Fr, H, W = x.shape
    x = x.reshape(B, C, Fr // p_t, p_t, H // p_h, p_h, W // p_w, p_w)
    return x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, -1, C * p_t * p_h * p_w)


class LingBotTransformer3DModel(nn.Module):
    """Neuron LingBot-World-V2 causal DiT. See the module docstring."""

    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 36,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        sink_size: int = 9,
        num_frames_per_block: int = 3,
        sliding_window_num_frames: int = 18,
        latent_size: tuple[int, int] = (60, 104),
        **unused_config,
    ):
        super().__init__()
        if not cross_attn_norm:
            raise ValueError("LingBot-World-V2 uses cross_attn_norm=True")
        patch_size = tuple(patch_size)
        dim = num_attention_heads * attention_head_dim
        self.dim = dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.freq_dim = freq_dim
        self.out_channels = out_channels
        self.head_dim = attention_head_dim
        self.sink_size = sink_size
        self.num_frames_per_block = num_frames_per_block
        self.sliding_window_num_frames = sliding_window_num_frames

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group().device_group
        if num_attention_heads % self.tp_size:
            raise ValueError(f"num_heads={num_attention_heads} not divisible by tp={self.tp_size}")
        self.num_local_heads = num_attention_heads // self.tp_size
        cp_group = get_cp_group()
        self.cp_size = cp_group.world_size
        self.cp_group = cp_group if self.cp_size > 1 else None
        register_replica_groups(tp_size=self.tp_size, cp_size=self.cp_size)

        self.patch_embedding = nn.Conv3d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        # Plücker rays: 6 channels (origin xyz, direction xyz) folded 8x8 from pixels to latent cells.
        self.patch_embedding_wancamctrl = nn.Linear(6 * 8 * 8 * math.prod(patch_size), dim)
        self.c2ws_hidden_states = _CameraRefineMLP(dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        # The timestep path stays FP32 (parameters and compute), as the official model's
        # autocast(float32) region.
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        _load_in_fp32(*self.time_embedding.parameters(), *self.time_projection.parameters())
        self.blocks = nn.ModuleList(
            [
                LingBotTransformerBlock(dim, ffn_dim, num_attention_heads, eps)
                for _ in range(num_layers)
            ]
        )
        self.head = _LingBotHead(dim, out_channels * math.prod(patch_size), eps)
        # Sequence parallelism over TP for the chunk's tokens per CP rank at ``latent_size``
        # (latent height, width; 480p by default): see :class:`SeqShard`.
        _, p_h, p_w = patch_size
        chunk_tokens = num_frames_per_block * (latent_size[0] // p_h) * (latent_size[1] // p_w)
        self._set_sequence_parallel(chunk_tokens // self.cp_size)

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        num_layers: int | None = None,
        latent_size: tuple[int, int] = (60, 104),
    ):
        """Build from ``transformer/config.json``, enforcing the checkpoint contract.

        ``latent_size``: the latent (height, width) the DiT runs at (480p by default); it fixes
        the chunk's token count, which the sequence parallelism is set up for."""
        # The checkpoint contract of the causal-fast diffusers transformer (vLLM-Omni
        # enforces the same values in CausalLingBotWorldTransformer3DModel.from_config).
        checkpoint_contract = {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 40,
            "attention_head_dim": 128,
            "in_channels": 36,
            "out_channels": 16,
            "text_dim": 4096,
            "freq_dim": 256,
            "ffn_dim": 13824,
            "cross_attn_norm": True,
            "eps": 1e-6,
            "image_dim": None,
            "added_kv_proj_dim": None,
            "rope_max_seq_len": 1024,
            "pos_embed_seq_len": None,
            "qk_norm": "rms_norm_across_heads",
            "sink_size": 9,
            "num_frames_per_block": 3,
            "sliding_window_num_frames": 18,
            "local_attn_size": -1,
        }
        config = dict(config)
        class_name = config.pop("_class_name", None)
        if class_name not in (None, "CausalLingBotWorldTransformer3DModel"):
            raise ValueError(f"unexpected transformer class {class_name!r}")
        for name, expected in checkpoint_contract.items():
            actual = config.get(name, expected)
            if isinstance(expected, tuple):
                actual = tuple(actual)
            if actual != expected:
                raise ValueError(
                    f"LingBot checkpoint config {name} must be {expected!r}, got {actual!r}"
                )
        kwargs = {k: v for k, v in config.items() if not k.startswith("_")}
        if num_layers is not None:
            kwargs["num_layers"] = num_layers
        model = cls(**kwargs, latent_size=tuple(latent_size))
        # The checkpoint config, as vLLM-Omni's ``transformer.config`` (the pipeline's request
        # validation and cache spec read it).
        resolved = {**checkpoint_contract, "num_layers": 40, **kwargs}
        resolved["patch_size"] = tuple(resolved["patch_size"])
        model.config = SimpleNamespace(**resolved)
        return model

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embedding.weight.dtype

    def _set_sequence_parallel(self, tokens: int | None) -> None:
        """Sequence parallelism over TP for ``tokens`` per CP rank (``None``: off).

        The residual stream stays on each TP rank's 1/TP of the tokens (:class:`SeqShard`): the
        norms run on those rows, the self-attention all-gathers its exact ``tokens`` input, the
        cross-attention and the FFN take the shared Wan layers' padded SP paths (all-gather in,
        reduce-scatter out), and every TP all-reduce becomes a reduce-scatter onto the rows. The
        FFN's partial sums are reduced in fp32 and rounded once; the attentions' in bf16. The
        camera's raw TP gather is built once per chunk in :meth:`chunk_prep`. A no-op at TP 1.
        """
        self._sp_tokens = None if tokens is None or self.tp_size == 1 else int(tokens)
        on = self._sp_tokens is not None
        padded = sp_padded_len(self._sp_tokens, self.tp_size) if on else None
        for block in self.blocks:
            block.self_attn.sp_padded_len = padded
            for layer in (block.cross_attn, block.ffn):
                layer.sp_enabled = on
                layer.sp_real_len = self._sp_tokens
                layer.sp_padded_len = padded
            block.ffn.sp_reduce_dtype = torch.float32 if on else None

    def _seq_shard(self, tokens: int, device) -> SeqShard | None:
        if self._sp_tokens is None:
            return None
        if tokens != self._sp_tokens:
            raise ValueError(
                f"sequence parallelism is set up for {self._sp_tokens} tokens per CP rank "
                f"(latent_size), got {tokens}"
            )
        return SeqShard(tokens, self.tp_size, self.tp_group, device)

    # ------------------------------------------------------------------ graph entry points

    def project_cross_attention_kv(
        self, encoder_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Embed the (512-token, zero-padded) UMT5 context and project every block's text K/V.

        Returns a flat ``(k_0, v_0, k_1, v_1, ...)`` tuple, each ``[B, N_local, 512, D]``.
        Run once per prompt; the result is reused by every chunk and denoising step.
        """
        context = self.text_embedding(encoder_hidden_states)
        kv: list[torch.Tensor] = []
        for block in self.blocks:
            kv.extend(block.cross_attn.project_kv(context))
        return tuple(kv)

    def _local_tokens(self, patches: torch.Tensor) -> torch.Tensor:
        """This CP rank's contiguous sequence shard of ``[B, S, F]`` patches (identity at CP=1).

        Selecting the raw patches before the embedding keeps the full-sequence embedded tensor out
        of the graph (same reasoning as Wan's ``_patch_embed_local_tokens``).
        """
        if self.cp_size == 1:
            return patches
        return shard_select(patches, 1, self.cp_group)

    def embed_camera(self, camera_hidden_states: torch.Tensor) -> torch.Tensor:
        """Patch-embed and refine the folded Plücker rays ``[B, 384, F, H, W]`` to tokens."""
        return self._embed_camera_patches(
            self._local_tokens(_patchify(camera_hidden_states, self.patch_size))
        )

    def _embed_camera_patches(self, patches: torch.Tensor) -> torch.Tensor:
        camera = F.linear(
            patches, self.patch_embedding_wancamctrl.weight, self.patch_embedding_wancamctrl.bias
        )
        return self.c2ws_hidden_states(camera)

    def camera_patches_from_params(
        self, camera_params: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        """This CP rank's FP32 Plücker-ray patches ``[1, S/cp, 1536]`` from the ``[F, 16]`` rows.

        In-graph equivalent of the host rays + fold + :func:`_patchify` + CP shard (see
        :mod:`.chunk_prep`); ``grid_h x grid_w`` is the post-patch token grid of one frame.
        """
        p_t, p_h, p_w = self.patch_size
        frame, origin = camera_token_grid(
            camera_params.shape[0],
            grid_h * p_h,
            grid_w * p_w,
            self.patch_size,
            device=camera_params.device,
        )
        if self.cp_size > 1:
            frame = shard_select(frame, 0, self.cp_group)
            origin = shard_select(origin, 0, self.cp_group)
        return plucker_patches(camera_params, frame, origin, patch_size=self.patch_size)[None]

    def chunk_prep(
        self,
        camera_params: torch.Tensor,
        rope_temporal: torch.Tensor,
        rope_h_cos: torch.Tensor,
        rope_h_sin: torch.Tensor,
        rope_w_cos: torch.Tensor,
        rope_w_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """One static per-chunk graph: ``(freqs_cos, freqs_sin, *camera_modulation)``.

        Inputs are small: the chunk's ``[F, 16]`` camera rows
        (:func:`.chunk_prep.camera_ray_params`) and ``[2, F, 22]`` temporal RoPE factors
        (:func:`.chunk_prep.lingbot_rope_temporal_factors`), plus the session-constant spatial
        RoPE factors (:func:`.chunk_prep.lingbot_rope_spatial_factors`). The outputs are what
        the host used to build and upload: the full-chunk RoPE tables (as
        :func:`.chunk_prep.lingbot_rotary_tables`, bit for bit) and every block's
        :meth:`LingBotCameraInjector.project` of the in-graph Plücker rays. At TP > 1 each camera tensor is instead this rank's stream rows'
        ``(scale, shift)`` from the TP all-gather, packed ``[2, 1, L, dim]`` (see
        :class:`SeqShard`): one gather per chunk rather than one per pass.
        """
        freqs_cos, freqs_sin = assemble_rotary_tables(
            rope_temporal, rope_h_cos, rope_h_sin, rope_w_cos, rope_w_sin
        )
        patches = self.camera_patches_from_params(
            camera_params, rope_h_cos.shape[0], rope_w_cos.shape[0]
        )
        camera = self._embed_camera_patches(
            patches.to(self.patch_embedding_wancamctrl.weight.dtype)
        )
        mods = [block.camera.project(camera) for block in self.blocks]
        shard = self._seq_shard(camera.shape[1], camera.device)
        if shard is not None:
            # The TP gather over this rank's rows and its rearrangement into (scale, shift), once
            # per chunk instead of in every pass. The pass graphs take the pair packed as one
            # [2, 1, L, dim] tensor: rearranging a pass graph's input there (a DRAM-to-DRAM
            # transpose) fails neuronx-cc 2.28 (NCC_IDDT901).
            mods = [
                torch.stack(
                    camera_from_gather(
                        shard.camera_rows(block.camera.gather_raw(m)), self.dim, shard.local
                    )
                ).contiguous()
                for block, m in zip(self.blocks, mods)
            ]
        return (freqs_cos, freqs_sin, *mods)

    def timestep_embedding(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunk-uniform timestep ``[B]`` -> (``temb`` ``[B, dim]``, ``proj`` ``[B, 6, dim]``), FP32."""
        frequency = _sinusoidal_embedding(self.freq_dim, timestep)
        temb = self.time_embedding(frequency.float()).float()
        proj = self.time_projection(temb).float()
        return temb, proj.unflatten(1, (6, self.dim))

    def packed_timestep_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        """``[B, 7, dim]`` FP32: ``temb`` then the 6 modulation projections of ``timestep``.

        The denoising schedule is fixed, so this runs once per schedule and timestep (the DiT
        runtime computes it on the host) and the result is passed to ``forward`` as
        ``timestep_embedding``: the passes skip the replicated FP32 time MLPs (their weights are not read per pass).
        """
        temb, proj = self.timestep_embedding(timestep)
        return torch.cat((temb.unsqueeze(1), proj), dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        camera_hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        kv_pool: tuple[torch.Tensor, ...],
        cross_attention_kv_cache: tuple[torch.Tensor, ...],
        paged: PagedAttentionInputs,
        camera_modulation: tuple[torch.Tensor, ...] | None = None,
        kv_only: bool = False,
        timestep_embedding: torch.Tensor | None = None,
    ):
        """One pass of one chunk at one timestep over the paged history.

        Args:
            hidden_states: ``[B, 36, F, H, W]`` noisy latent (16 ch) concatenated with the image
                condition (4 mask + 16 VAE ch), ``F = num_frames_per_block``.
            timestep: ``[B]`` float32 timestep, uniform over the chunk.
            camera_hidden_states: ``[B, 384, F, H, W]`` folded Plücker rays of the chunk.
            freqs_cos, freqs_sin: rotate-half ``[1, S, D]`` RoPE tables of the chunk
                (:meth:`chunk_prep`, equal to :func:`.chunk_prep.lingbot_rotary_tables`;
                start-frame offset applied).
            kv_pool: the paged KV pool's per-layer views, flat ``(k_0, v_0, k_1, v_1, ...)``.
            cross_attention_kv_cache: flat text K/V from :meth:`project_cross_attention_kv`.
            paged: :class:`PagedAttentionInputs` of this pass (block table, mask bounds and, on
                the commit pass, the per-token update indices).
            kv_only: the ``t = 0`` commit pass: the self-attention writes the chunk's K/V into
                the pool; the last block's self-attention only writes (``write_only``: no
                attention, o-proj or TP collective), the rest of that block and the output head
                are skipped, since nothing after them feeds any layer's K/V. Returns ``()``.
            camera_modulation: the chunk's per-block camera modulation, reused by all passes of
                the chunk: each block's ``[scale | shift]`` TP shard
                (:meth:`LingBotCameraInjector.project`) ``[1, S, 2 * dim / T]``, or, at TP > 1 with sequence parallelism, this rank's stream
                rows' ``(scale, shift)`` packed ``[2, 1, L, dim]`` from :meth:`chunk_prep`. When given, ``camera_hidden_states`` is ignored (may be
                ``None``); otherwise every pass recomputes it.
            timestep_embedding: :meth:`packed_timestep_embedding` of ``timestep``, precomputed
                once per schedule; when given, ``timestep`` is ignored (may be ``None``).

        Returns:
            ``(flow,)`` with ``flow`` ``[B, 16, F, H, W]``; ``()`` with ``kv_only``.
        """
        p_t, p_h, p_w = self.patch_size
        B, _, num_frames, height, width = hidden_states.shape
        if len(kv_pool) != 2 * self.num_layers:
            raise ValueError(f"expected {2 * self.num_layers} KV pool views")
        if len(cross_attention_kv_cache) != 2 * self.num_layers:
            raise ValueError(f"expected {2 * self.num_layers} cross-attention tensors")

        if (num_frames // p_t) * (height // p_h) * (width // p_w) % self.cp_size:
            raise ValueError(f"chunk tokens not divisible by cp_size={self.cp_size}")
        conv_weight = self.patch_embedding.weight.reshape(self.patch_embedding.out_channels, -1)
        hidden = F.linear(
            self._local_tokens(_patchify(hidden_states, self.patch_size)),
            conv_weight,
            self.patch_embedding.bias,
        )
        if self.cp_size > 1:
            freqs_cos = shard_select(freqs_cos, 1, self.cp_group)
            freqs_sin = shard_select(freqs_sin, 1, self.cp_group)
        if camera_modulation is None:
            camera = self.embed_camera(camera_hidden_states)
        else:
            camera = None
            if len(camera_modulation) != self.num_layers:
                raise ValueError(f"expected {self.num_layers} camera modulation tensors")
        if timestep_embedding is None:
            temb, timestep_proj = self.timestep_embedding(timestep)
        else:
            temb, timestep_proj = timestep_embedding[:, 0], timestep_embedding[:, 1:]
        rotary_emb = (freqs_cos, freqs_sin)

        seq_shard = self._seq_shard(hidden.shape[1], hidden.device)
        if seq_shard is not None:
            hidden = seq_shard.local_rows(hidden)
        pending = None
        for index, block in enumerate(self.blocks):
            last_kv_only = kv_only and index == self.num_layers - 1
            hidden, pending = block(
                hidden,
                timestep_proj,
                rotary_emb,
                camera,
                kv_pool[2 * index],
                kv_pool[2 * index + 1],
                cross_attention_kv_cache[2 * index : 2 * index + 2],
                # The commit's last layer feeds nothing but the pool: write only.
                paged._replace(write_only=True)
                if last_kv_only and isinstance(paged, PagedAttentionInputs)
                else paged,
                camera_modulation=None if camera_modulation is None else camera_modulation[index],
                kv_only=last_kv_only,
                pending_residual=pending,
                defer_ffn_residual=not last_kv_only,
                seq_shard=seq_shard,
            )
        if kv_only:
            return ()

        if seq_shard is None:
            hidden = self.head(hidden, temb, pending_residual=pending)
        else:
            # The output head is per token: run it on the rank's rows, gather its narrow output.
            hidden = sp_entry(
                self.head(hidden, temb, pending_residual=pending),
                seq_shard.tp_group,
                seq_shard.tp_size,
                sp_enabled=True,
                real_len=seq_shard.tokens,
            )
        if self.cp_size > 1:
            hidden = cp_all_gather_seq(hidden, self.cp_group, dim=1)
        hidden = hidden.reshape(
            B, num_frames // p_t, height // p_h, width // p_w, p_t, p_h, p_w, -1
        )
        flow = hidden.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return (flow,)

    # ------------------------------------------------------------------ weights

    def checkpoint_mappings(self) -> dict[str, str | list[str]]:
        """Model parameter name -> checkpoint key(s); identical names are omitted."""
        m: dict[str, str | list[str]] = {}
        for i in range(self.num_layers):
            b = f"blocks.{i}"
            sa, ca, cam = f"{b}.self_attn", f"{b}.cross_attn", f"{b}.camera"
            m[f"{sa}.qkv_proj_weight"] = [f"{sa}.q.weight", f"{sa}.k.weight", f"{sa}.v.weight"]
            m[f"{sa}.qkv_proj_bias"] = [f"{sa}.q.bias", f"{sa}.k.bias", f"{sa}.v.bias"]
            m[f"{sa}.o_proj_weight"] = f"{sa}.o.weight"
            m[f"{sa}.o_proj_bias"] = f"{sa}.o.bias"
            for proj in ("q", "k", "v", "o"):
                m[f"{ca}.{proj}_proj_weight"] = f"{ca}.{proj}.weight"
                m[f"{ca}.{proj}_proj_bias"] = f"{ca}.{proj}.bias"
            m[f"{b}.ffn.up_proj_weight"] = f"{b}.ffn.0.weight"
            m[f"{b}.ffn.up_proj_bias"] = f"{b}.ffn.0.bias"
            m[f"{b}.ffn.down_proj_weight"] = f"{b}.ffn.2.weight"
            m[f"{b}.ffn.down_proj_bias"] = f"{b}.ffn.2.bias"
            for layer in (1, 2):
                m[f"{cam}.injector.layer{layer}_weight"] = f"{b}.cam_injector_layer{layer}.weight"
                m[f"{cam}.injector.layer{layer}_bias"] = f"{b}.cam_injector_layer{layer}.bias"
            m[f"{cam}.scale_shift_weight"] = [
                f"{b}.cam_scale_layer.weight",
                f"{b}.cam_shift_layer.weight",
            ]
            m[f"{cam}.scale_shift_bias"] = [
                f"{b}.cam_scale_layer.bias",
                f"{b}.cam_shift_layer.bias",
            ]
        for layer in (1, 2):
            m[f"c2ws_hidden_states.layer{layer}_weight"] = f"c2ws_hidden_states_layer{layer}.weight"
            m[f"c2ws_hidden_states.layer{layer}_bias"] = f"c2ws_hidden_states_layer{layer}.bias"
        return m

    def load_weights(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Load the TP-sharded ``transformer/`` weights (FP32 checkpoint, cast per parameter).

        Every parameter loads in its own dtype, except the ones marked by :func:`_load_in_fp32`
        (the timestep MLPs, the modulation tables, norm3), which load in FP32. So cast the model
        (``.to(dtype)`` / ``.bfloat16()``) before this; device moves are fine either way.
        """
        tp_rank = get_tensor_model_parallel_rank()
        mappings = self.checkpoint_mappings()
        checkpoint = SafetensorsCheckpoint(model_name_or_path)
        logged_casts: set = set()
        # The loader casts every tensor to its parameter's dtype unless dtype_override names
        # it, so the FP32-marked parameters are named there (a cast model holds them in BF16).
        fp32 = {name: torch.float32 for name, p in self.named_parameters() if loads_in_fp32(p)}
        for name, param in self.named_parameters():
            dtype = fp32.get(name, param.dtype)
            set_weight_loader(
                param, cast_weight_loader(get_weight_loader(param), dtype, name, logged_casts)
            )
        result = checkpoint.load_sharded_pipelined(
            tp_rank, self.tp_size, self, mappings, device, dtype_override=fp32
        )
        missing = sorted({n for n, _ in self.named_parameters()} - set(result.state_dict))
        if missing:
            raise RuntimeError(f"LingBot weights not loaded: {missing[:8]}")
        self.load_state_dict(result.state_dict, strict=False, assign=True)
