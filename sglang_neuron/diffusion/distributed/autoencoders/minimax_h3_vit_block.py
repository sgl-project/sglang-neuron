# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 video-decoder transformer block on fused NKI kernels (one tile, one core).

Upstream (diffusers ``MiniMaxH3VideoTransformerBlock``)::

    h = h + attn(rmsnorm1(h)) * scale1        # q/k/v Linear+bias, QK RMSNorm (no affine), partial RoPE
    h = h + ff(rmsnorm2(h)) * scale2          # SwiGLU: proj 2048 -> 2*8192 (+bias), x * silu(gate), 8192 -> 2048

Here, per block:

* ``scale1`` / ``scale2`` are folded into the o-projection and the down projection (exact), so
  both residuals are plain adds.
* q/k/v: ``qkv_proj`` (``qkv_cte``) with the bias, three calls (a fused 3 x 2048 projection is
  past the kernel's 4096 output limit and the wrapper's ``<= H`` rule). Q and K run the kernel's
  pre-RoPE QK RMSNorm (no gamma) and its rotate-half RoPE: the 48-of-64 partial RoPE becomes a
  full-head one by permuting each head's Q/K channels so rotary pair ``(j, j + 24)`` sits at
  ``(j, j + 32)`` and the 16 pass-through channels get cos 1 / sin 0. Q and K share the
  permutation, so ``q . k`` (and the norm) are unchanged; V and the o-projection are untouched.
* attention: ``attention_cte`` with a d-major output, into ``o_proj`` (the o-projection's
  heads read as 16 x 128, the same flat layout as 32 x 64, inside its 17-head limit).
* the SwiGLU: the nkilib MLP kernel over four 2048-wide slices of the intermediate dim (see
  ``MLP_SPLITS``), with the gate/up/down biases; the slices add. ``vn.functional.mlp`` takes no
  bias, so ``_biased_mlp`` calls the kernel itself, with the same fallback rule.
* both RMSNorms: ``torch.rms_norm`` in float32, as upstream.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

ROPE_PAIR = 24  # rotary pairs per head: rope_dim_ratio 0.75 x 64 / 2
#: The SwiGLU's 8192-wide intermediate dim runs as this many MLP kernel calls. At 4096 per call
#: (all 8 PSUM banks of 512) the CTE kernel's output is wrong (rel-L2 5.9e-2 against fp32 on
#: block 10's weights, where bf16 torch is 5.7e-3); at 2048 per call it matches bf16 (5.6e-3).
MLP_SPLITS = 4


def _biased_mlp(hidden, gate_w, up_w, down_w, gate_b, up_b, down_b=None):
    """``(silu(hidden @ gate_w + gate_b) * (hidden @ up_w + up_b)) @ down_w + down_b`` on the
    nkilib MLP kernel where ``vn.functional.mlp`` would run it, else torch. ``hidden`` ``[T, H]``;
    biases ``[1, I]`` (``down_b`` ``[1, H]``)."""
    from nkilib.core.utils.common_types import (
        ActFnType,
        ComputationMode,
        DtypeMode,
        NormType,
        QuantizationType,
    )

    from sglang_neuron.diffusion.vn.functional import mlp as mlp_op
    from sglang_neuron.diffusion.vn.utils.hardware_config import get_logical_nc_config

    if not mlp_op._can_use_kernel(hidden, gate_w):
        out = (F.silu(hidden @ gate_w + gate_b) * (hidden @ up_w + up_b)) @ down_w
        return out if down_b is None else out + down_b
    result = mlp_op.wrapped_mlp()[get_logical_nc_config()](
        hidden_tensor=hidden.unsqueeze(0),
        gate_proj_weights_tensor=gate_w,
        up_proj_weights_tensor=up_w,
        down_proj_weights_tensor=down_w,
        normalization_weights_tensor=None,
        gate_proj_bias_tensor=gate_b,
        up_proj_bias_tensor=up_b,
        down_proj_bias_tensor=down_b,
        normalization_bias_tensor=None,
        fused_add_tensor=None,
        store_fused_add_result=False,
        activation_fn=ActFnType.SiLU,
        normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.NONE,
        gate_w_scale=None,
        up_w_scale=None,
        down_w_scale=None,
        gate_up_in_scale=None,
        down_in_scale=None,
        quant_clipping_bound=0.0,
        output_dtype=None,
        store_output_in_sbuf=False,
        eps=1e-6,
        skip_gate_proj=False,
        use_tkg_gate_up_proj_column_tiling=True,
        use_tkg_down_proj_column_tiling=True,
        use_tkg_down_proj_optimized_layout=False,
        gate_clamp_upper_limit=None,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
        up_clamp_lower_limit=None,
        mode=ComputationMode.AUTO,
        force_cte_mode=False,
        dtype_mode=DtypeMode.AUTO,
    )
    if isinstance(result, (tuple, list)):
        result = result[0]
    return result.squeeze(0)


def _channel_permutation(head_dim: int, rotary_pairs: int) -> torch.Tensor:
    """``perm`` with ``new[c] = old[perm[c]]``: rotary pair ``(j, j + rotary_pairs)`` moves to
    ``(j, j + head_dim // 2)``; the pass-through channels fill the rest of each half."""
    half = head_dim // 2
    rot = rotary_pairs * 2
    passthrough = list(range(rot, head_dim))
    n_pass = len(passthrough) // 2
    first = list(range(rotary_pairs)) + passthrough[:n_pass]
    second = list(range(rotary_pairs, rot)) + passthrough[n_pass:]
    assert len(first) == half and len(second) == half
    return torch.tensor(first + second)


def full_head_rope_tables(cos: torch.Tensor, sin: torch.Tensor, head_dim: int, rotary_pairs: int):
    """Upstream's ``(S, 2 * rotary_pairs)`` duplicated-halves tables as ``(S, head_dim)`` tables
    for the permuted layout: cos 1 / sin 0 on the pass-through channels."""
    half = head_dim // 2
    pad = half - rotary_pairs
    c, s = cos[:, :rotary_pairs], sin[:, :rotary_pairs]
    ones, zeros = c.new_ones(c.shape[0], pad), s.new_zeros(s.shape[0], pad)
    cos_full = torch.cat([c, ones, c, ones], dim=-1)
    sin_full = torch.cat([s, zeros, s, zeros], dim=-1)
    return cos_full, sin_full


class FusedViTBlock(nn.Module):
    """One decoder block's weights, re-laid-out for the NKI kernels; see the module docstring."""

    def __init__(self, block: nn.Module):
        super().__init__()
        attn = block.attn
        heads, head_dim = attn.heads, attn.dim_head
        self.heads, self.head_dim = heads, head_dim
        self.eps_attn = block.norm1.eps
        self.eps_qk = attn.norm_q.eps
        perm = _channel_permutation(head_dim, ROPE_PAIR)
        index = (torch.arange(heads)[:, None] * head_dim + perm[None]).reshape(-1)

        def stored(linear, rows=None):  # nn.Linear [out, in] -> kernel [in, out] (+ [1, out] bias)
            w, b = linear.weight.data, linear.bias.data
            if rows is not None:
                w, b = w[rows], b[rows]
            return nn.Parameter(w.t().contiguous(), requires_grad=False), nn.Parameter(
                b[None].contiguous(), requires_grad=False
            )

        self.q_w, self.q_b = stored(attn.to_q, index)
        self.k_w, self.k_b = stored(attn.to_k, index)
        self.v_w, self.v_b = stored(attn.to_v)
        scale1 = block.scale1.data
        o = attn.to_out[0]
        self.o_w = nn.Parameter(
            (o.weight.data * scale1[:, None]).t().contiguous(), requires_grad=False
        )
        self.o_b = nn.Parameter((o.bias.data * scale1)[None].contiguous(), requires_grad=False)

        proj, down = block.ff.net[0].proj, block.ff.net[2]
        inner = proj.out_features // 2
        self.inner = inner
        # diffusers SwiGLU: hidden, gate = proj(x).chunk(2); hidden * silu(gate).
        up_w, gate_w = proj.weight.data[:inner], proj.weight.data[inner:]
        up_b, gate_b = proj.bias.data[:inner], proj.bias.data[inner:]
        scale2 = block.scale2.data
        down_w = down.weight.data * scale2[:, None]  # [dim, inner]
        down_b = down.bias.data * scale2
        halves = []
        step = inner // MLP_SPLITS
        for lo in range(0, inner, step):
            sl = slice(lo, lo + step)
            halves.append(
                (
                    gate_w[sl].t().contiguous(),
                    up_w[sl].t().contiguous(),
                    down_w[:, sl].t().contiguous(),
                    gate_b[sl][None].contiguous(),
                    up_b[sl][None].contiguous(),
                )
            )
        self.mlp_params = nn.ParameterList(
            [nn.Parameter(t, requires_grad=False) for half in halves for t in half]
        )
        self.down_b = nn.Parameter(down_b[None].contiguous(), requires_grad=False)
        self.norm1_w = nn.Parameter(block.norm1.weight.data.float().clone(), requires_grad=False)
        self.norm2_w = nn.Parameter(block.norm2.weight.data.float().clone(), requires_grad=False)
        self.eps_ff = block.norm2.eps

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """``hidden_states`` ``[1, S, dim]``; ``cos``/``sin`` ``[1, S, head_dim]`` (permuted layout)."""
        from nkilib.core.utils.common_types import NormType

        from sglang_neuron.diffusion.attention.cp_attention import nki_attention
        from sglang_neuron.diffusion.vn.functional.attention.o_proj import o_proj
        from sglang_neuron.diffusion.vn.functional.attention.qkv import qkv_proj

        dtype = hidden_states.dtype
        n, d = self.heads, self.head_dim
        h = F.rms_norm(
            hidden_states.float(), (hidden_states.shape[-1],), self.norm1_w, self.eps_attn
        ).to(dtype)
        S = h.shape[1]
        # The projection kernels split rows over the LNC cores; rows are independent there, so
        # pad to an even count and drop the pad row before attention.
        pad = S % 2
        if pad:
            h = F.pad(h, (0, 0, 0, pad))
            cos = F.pad(cos, (0, 0, 0, pad), value=1.0)
            sin = F.pad(sin, (0, 0, 0, pad))

        def qk(w, b):
            return qkv_proj(
                hidden=h,
                qkv_weights=w,
                bias=b,
                d_head=d,
                num_q_heads=n,
                num_kv_heads=0,
                cos_cache=cos,
                sin_cache=sin,
                qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_eps=self.eps_qk,
            )

        q = qk(self.q_w, self.q_b).view(S + pad, n, d)[:S]
        k = qk(self.k_w, self.k_b).view(S + pad, n, d)[:S]
        v = qkv_proj(hidden=h, qkv_weights=self.v_w, bias=self.v_b).view(S + pad, n, d)[:S]
        # [S, N, D] -> [1, N, S, D]; nki_attention returns d-major [1, N, D, S].
        q, k, v = (t.transpose(0, 1)[None] for t in (q, k, v))
        attn = nki_attention(q, k, v, d**-0.5)
        attn = attn.reshape(1, n * d // 128, 128, S) if (n * d) % 128 == 0 else attn
        out = o_proj(attn, self.o_w, self.o_b)  # [1, S, dim]
        hidden_states = hidden_states + out.to(dtype)

        h = F.rms_norm(
            hidden_states.float(), (hidden_states.shape[-1],), self.norm2_w, self.eps_ff
        ).to(dtype)
        rows = F.pad(h[0], (0, 0, 0, pad)) if pad else h[0]
        p = self.mlp_params
        ff = None
        for i in range(MLP_SPLITS):
            gate_w, up_w, down_w, gate_b, up_b = p[5 * i : 5 * i + 5]
            part = _biased_mlp(
                rows, gate_w, up_w, down_w, gate_b, up_b, self.down_b if i == 0 else None
            )
            ff = part if ff is None else ff + part
        return hidden_states + ff[:S][None].to(dtype)


def fuse_decoder(decoder: nn.Module) -> None:
    """Replace ``decoder``'s transformer blocks with ``FusedViTBlock`` s and its forward with one
    that builds the permuted full-head RoPE tables once per tile."""
    import types

    fused = nn.ModuleList([FusedViTBlock(block) for block in decoder.transformer_blocks])
    del decoder.transformer_blocks
    decoder.fused_blocks = fused
    head_dim = fused[0].head_dim

    def forward(self, hidden_states):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch_size, num_frames * height * width, num_channels
        )
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)
        gt, gh, gw = (
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size)
            - 1.0
            for size in (num_frames, height, width)
        )
        t_ids = gt.repeat_interleave(height * width)
        h_ids = gh.repeat_interleave(width).repeat(num_frames)
        w_ids = gw.repeat(num_frames * height)
        position_ids = torch.stack([t_ids, h_ids, w_ids], dim=-1)[None]
        suffix_ids = position_ids.new_zeros((1, self.num_register_tokens + 1, 3))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        cos, sin = self.rope(position_ids)  # [1, S, 1, 2 * ROPE_PAIR]
        cos, sin = full_head_rope_tables(cos[0, :, 0], sin[0, :, 0], head_dim, ROPE_PAIR)
        cos, sin = cos[None].to(hidden_states.dtype), sin[None].to(hidden_states.dtype)
        for block in self.fused_blocks:
            hidden_states = block(hidden_states, cos, sin)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :num_patches, :]
        p, pt = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(
            batch_size, num_frames, height, width, self.out_channels, pt, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size, self.out_channels, num_frames * pt, height * p, width * p
        )

    decoder.forward = types.MethodType(forward, decoder)
