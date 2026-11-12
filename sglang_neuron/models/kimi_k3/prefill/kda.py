"""The KDA attention block of the Kimi-K3 prefill stage, in the sequence-parallel form.

One rank holds ``hc = num_heads / world`` heads and a row block ``[Tl, hidden]`` of the chunk (``Tl = T / world``,
rows ``rank * Tl .. rank * Tl + Tl - 1``). The block gathers the rows, runs the q, k, v, g, beta and decay
projections of its heads, the causal short convs with SiLU, the gated delta-rule scan, the gated output norm and its
``o_proj`` row shard, and reduce-scatters the ``o_proj`` partials by row.

Rows at or after ``valid_len`` are pad rows: they get ``k = 0``, a zero decay logit and a zero beta, so the scan does
not change the state, and the conv window ends at the last valid row.

Backends: ``"torch"`` (CPU; the token-at-a-time recurrence in fp32) and ``"kernel"`` (the NKI KDA block kernel, one
call per layer, with the MXFP8 q/k/v/g/b projections; ``prepare_kernel`` builds its inputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_neuron.models.kimi_k3.prefill import kernel_imports

_GRID = 1
_FRONT_TILE = 128
_FRONT_GROUP_HEADS = 3
_SCAN_CHUNK = 128
_SCAN_SUB = 8
_SCAN_LEVELS = 7


@dataclass
class KdaState:
    """The state of one KDA layer, one request, one rank (the state pool's slot layout, which the kernels read).

    ``conv`` ``[3 hc D, K - 1]`` bf16, channel-major: channel ``j * hc * D + h * D + d`` for ``j`` in (q, k, v) holds
    its last ``K - 1`` pre-conv projection inputs, oldest first. ``temporal`` ``[hc, D, D]`` bf16: ``S[h]`` is
    ``[D_k, D_v]``.
    """

    conv: torch.Tensor
    temporal: torch.Tensor

    @classmethod
    def zeros(cls, local_heads: int, head_dim: int, conv_size: int = 4, device=None,
              dtype: torch.dtype = torch.bfloat16) -> "KdaState":
        p = local_heads * head_dim
        return cls(
            conv=torch.zeros(3 * p, conv_size - 1, device=device, dtype=dtype),
            temporal=torch.zeros(local_heads, head_dim, head_dim, device=device, dtype=dtype),
        )


def short_conv(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor, n_valid: torch.Tensor):
    """Causal depthwise conv over time, then SiLU.

    Args:
        x: ``[T, C]`` pre-conv rows.
        weight: ``[C, K]`` taps, oldest first.
        state: ``[C, K - 1]`` the last ``K - 1`` inputs of the previous chunk, oldest first.
        n_valid: ``[1]`` int64 count of the valid rows.

    Returns:
        ``(silu(conv) [T, C]`` in the dtype of ``x`` (the conv sums in fp32), ``new_state [C, K - 1])``: the
        ``K - 1`` inputs that end at the last valid row (the old state fills in for a short chunk).
    """
    taps = weight.shape[-1]
    padded = torch.cat([state.transpose(0, 1).to(x.dtype), x], dim=0)
    y = F.conv1d(padded.transpose(0, 1).unsqueeze(0).float(), weight.unsqueeze(1).float(), groups=x.shape[1])
    y = y.squeeze(0).transpose(0, 1)
    rows = n_valid.reshape(-1)[:1].to(torch.int64) + torch.arange(taps - 1, device=x.device)
    new_state = padded.index_select(0, rows).transpose(0, 1)
    return F.silu(y).to(x.dtype), new_state


def zero_pad_rows(y_rows: torch.Tensor, valid_len: torch.Tensor, comm) -> torch.Tensor:
    """Set the pad rows of this rank's row block (global row at or after ``valid_len``) to 0, as the kernels do."""
    rows = y_rows.shape[0]
    first = comm.rank_tensor.reshape(-1)[:1].to(device=y_rows.device, dtype=torch.int64) * rows
    valid = first + torch.arange(rows, device=y_rows.device) < valid_len.reshape(-1)[:1].to(torch.int64)
    return torch.where(valid[:, None], y_rows, torch.zeros((), dtype=y_rows.dtype, device=y_rows.device))


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def kda_scan(q, k, v, g_log, beta, S0):
    """The gated delta rule, one token at a time, in the dtype of the inputs.

    ``S_t = (I - beta_t k_t k_t^T) Diag(exp g_t) S_{t-1} + beta_t k_t v_t^T`` and ``o_t = S_t^T q_t / sqrt(D)``;
    q, k, v, g_log ``[T, H, D]``, beta ``[T, H]``, S0 ``[H, D, D]``.
    """
    T, _, D = q.shape
    scale = D**-0.5
    S = S0.clone()
    o = q.new_zeros(T, q.shape[1], v.shape[-1])
    for t in range(T):
        S = torch.exp(g_log[t])[:, :, None] * S
        err = (v[t] - torch.einsum("hkv,hk->hv", S, k[t])) * beta[t][:, None]
        S = S + k[t][:, :, None] * err[:, None, :]
        o[t] = torch.einsum("hkv,hk->hv", S, q[t] * scale)
    return o, S


def kda_scan_constants(device=None) -> list[torch.Tensor]:
    """The fp32 0/1 constants of the chunked scan kernel, in its argument order and layouts.

    ``[lcat [C, 2C], wcat [C, C + NSUB (NSUB + 1) / 2], mask_a3 [C, NSUB, SUB], mask_b3 [C, NSUB, SUB],
    mask_lvl [C, LEVELS, C], ident [C, C]]`` with ``C = 128`` tokens a chunk, sub-blocks of 8 and 7 doubling levels.
    """
    C, c, nsub = _SCAN_CHUNK, _SCAN_SUB, _SCAN_CHUNK // _SCAN_SUB
    i = torch.arange(C)[:, None]
    j = torch.arange(C)[None, :]
    lcat = torch.cat([(i <= j).float(), (i > j).float()], dim=1)
    end = (j // c) * c + c - 1
    wblocks = [((j < i) & (i <= end)).float()]
    for b in range(nsub):
        for bp in range(b + 1):
            wblocks.append((((bp + 1) * c <= i) & (i < (b + 1) * c)).float()[:, :1].expand(C, 1))
    wcat = torch.cat(wblocks, dim=1)
    mask_a, mask_b = (i < j).float(), (i <= j).float()
    levels = []
    for lv in range(_SCAN_LEVELS):
        n = 1 << lv
        levels.append(((i // (2 * n) == j // (2 * n)) & ((j // n) % 2 == 1) & ((i // n) % 2 == 0)).float())
    mask_lvl = torch.stack(levels)
    out = [
        lcat,
        wcat,
        mask_a.reshape(C, nsub, c),
        mask_b.reshape(C, nsub, c),
        mask_lvl.transpose(1, 2).permute(1, 0, 2).reshape(C, _SCAN_LEVELS, C),
        # explicit: a model loader may build the model under another default dtype
        torch.eye(C, dtype=torch.float32),
    ]
    return [t.contiguous().to(device) for t in out]


def front_head_groups(local_heads: int) -> int:
    """Head groups of the kernel front: groups of 3 heads above 3 local heads, else one group."""
    if local_heads > _FRONT_GROUP_HEADS and local_heads % _FRONT_GROUP_HEADS == 0:
        return local_heads // _FRONT_GROUP_HEADS
    return 1


def _group_major(weights: list[torch.Tensor], groups: int) -> torch.Tensor:
    """Rows of group 0 of each weight, then group 1 of each weight, and so on."""
    return torch.cat([w.chunk(groups)[g] for g in range(groups) for w in weights])


def front_fa_tile(f_a: torch.Tensor, tile: int = _FRONT_TILE) -> torch.Tensor:
    """``f_a [tile, hidden]`` as the kernel's stationary tile ``[1, tile, hidden / tile, tile]``: ``out[0, p, c, j] =
    f_a[j, tile c + p]``."""
    rows, hidden = f_a.shape
    if rows != tile or hidden % tile:
        raise ValueError(f"The KDA f_a weight is [{rows}, {hidden}]; the kernel front needs [{tile}, k * {tile}].")
    return f_a.view(1, tile, hidden // tile, tile).permute(0, 3, 2, 1).contiguous()


class KdaAttention(nn.Module):
    """One KDA layer on one rank (``hc`` heads), SP rows in and out.

    Parameters (rank shards of the checkpoint tensors): ``q_proj``, ``k_proj``, ``v_proj``, ``g_proj`` ``[p, hidden]``,
    ``b_proj`` ``[hc, hidden]``, ``f_a_proj`` ``[D, hidden]`` (every rank), ``f_b_proj`` ``[p, D]``, ``o_proj``
    ``[hidden, p]`` (column shard), and fp32 ``A_log [hc]``, ``dt_bias [p]``, ``conv_weight [3, p, K]``,
    ``o_norm_weight [D]``; ``p = hc D``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        rank: int,
        world: int,
        conv_size: int = 4,
        gate_lower_bound: float = -5.0,
        rms_norm_eps: float = 1e-5,
        dtype: torch.dtype = torch.bfloat16,
        backend: str = "torch",
    ):
        super().__init__()
        if num_heads % world:
            raise ValueError(f"{num_heads} KDA heads do not divide by the stage width {world}.")
        if backend not in ("torch", "kernel"):
            raise ValueError(f"KDA backend {backend!r} is not known. Use 'torch' or 'kernel'.")
        self.hidden_size, self.num_heads, self.head_dim = hidden_size, num_heads, head_dim
        self.rank, self.world, self.hc = rank, world, num_heads // world
        self.conv_size, self.gate_lower_bound, self.o_norm_eps = conv_size, float(gate_lower_bound), rms_norm_eps
        self.backend = backend
        p = self.hc * head_dim

        def linear(n_out, n_in):
            return nn.Linear(n_in, n_out, bias=False, dtype=dtype)

        self.q_proj, self.k_proj, self.v_proj = linear(p, hidden_size), linear(p, hidden_size), linear(p, hidden_size)
        self.g_proj = linear(p, hidden_size)
        self.b_proj = linear(self.hc, hidden_size)
        self.f_a_proj = linear(head_dim, hidden_size)
        self.f_b_proj = linear(p, head_dim)
        self.o_proj = linear(hidden_size, p)
        fp32 = torch.float32
        self.A_log = nn.Parameter(torch.zeros(self.hc, dtype=fp32))
        self.dt_bias = nn.Parameter(torch.zeros(p, dtype=fp32))
        self.conv_weight = nn.Parameter(torch.zeros(3, p, conv_size, dtype=fp32))
        self.o_norm_weight = nn.Parameter(torch.ones(head_dim, dtype=fp32))
        self._block_shim = None

    @classmethod
    def from_geometry(cls, geometry, rank: int, world: int, **kwargs) -> "KdaAttention":
        """Build the module from the model geometry (``geometry.KimiK3Geometry``)."""
        return cls(
            hidden_size=geometry.hidden_size,
            num_heads=geometry.kda_num_heads,
            head_dim=geometry.kda_head_dim,
            conv_size=geometry.kda_conv_kernel,
            gate_lower_bound=geometry.kda_gate_lower_bound,
            rms_norm_eps=geometry.rms_norm_eps,
            rank=rank,
            world=world,
            **kwargs,
        )

    def load_full(self, tensors: Mapping[str, torch.Tensor]) -> None:
        """Copy this rank's shard from the full layer tensors (checkpoint names under ``self_attn.``).

        ``A_log`` may be padded past the head count; the first ``num_heads`` entries are the heads.
        """
        r, hc, D = self.rank, self.hc, self.head_dim
        p0, p1 = r * hc * D, (r + 1) * hc * D
        with torch.no_grad():
            for name in ("q_proj", "k_proj", "v_proj", "g_proj", "f_b_proj"):
                getattr(self, name).weight.copy_(tensors[f"{name}.weight"][p0:p1])
            self.b_proj.weight.copy_(tensors["b_proj.weight"][r * hc : (r + 1) * hc])
            self.f_a_proj.weight.copy_(tensors["f_a_proj.weight"])
            self.o_proj.weight.copy_(tensors["o_proj.weight"][:, p0:p1])
            a_log = tensors["A_log"].reshape(-1)[: self.num_heads]
            self.A_log.copy_(a_log[r * hc : (r + 1) * hc].float())
            self.dt_bias.copy_(tensors["dt_bias"].reshape(-1)[p0:p1].float())
            conv = torch.stack([tensors[f"{n}_conv1d.weight"].reshape(-1, self.conv_size)[p0:p1] for n in "qkv"])
            self.conv_weight.copy_(conv.float())
            self.o_norm_weight.copy_(tensors["o_norm.weight"].float())

    def init_state(self, device=None) -> KdaState:
        return KdaState.zeros(self.hc, self.head_dim, self.conv_size, device=device or self.A_log.device)

    def forward(self, x_rows: torch.Tensor, state: KdaState, valid_len: torch.Tensor, comm) -> tuple[torch.Tensor, KdaState]:
        """``x_rows [Tl, hidden]`` bf16 normed rows of this rank -> ``(o_proj rows [Tl, hidden] bf16, state after the
        last valid row)``. ``valid_len`` ``[1]`` int64 counts the valid rows of the whole chunk."""
        if self.backend == "kernel":
            return self._forward_kernel(x_rows, state, valid_len, comm)
        return self._forward_torch(x_rows, state, valid_len, comm)

    # torch backend

    def _forward_torch(self, x_rows, state: KdaState, valid_len, comm):
        x = comm.all_gather(x_rows, dim=0)
        T, hc, D = x.shape[0], self.hc, self.head_dim
        n_valid = valid_len.reshape(-1)[:1].to(torch.int64)
        valid = torch.arange(T, device=x.device) < n_valid
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        g2 = self.g_proj(x).view(T, hc, D)
        beta_logits = self.b_proj(x)
        z = self.f_b_proj(self.f_a_proj(x)).view(T, hc, D)

        conv_in = state.conv.reshape(3, hc * D, self.conv_size - 1)
        conv_out, conv_state = [], []
        for j, rows in enumerate((q, k, v)):
            y, s = short_conv(rows, self.conv_weight[j], conv_in[j], n_valid)
            conv_out.append(y)
            conv_state.append(s)
        q, k, v = (t.view(T, hc, D) for t in conv_out)

        valid3 = valid[:, None, None]
        pad = torch.finfo(z.dtype).min
        # Selects, so a non-finite pad row cannot reach the scan.
        q = torch.where(valid3, q, torch.zeros_like(q))
        k = torch.where(valid3, k, torch.zeros_like(k))
        v = torch.where(valid3, v, torch.zeros_like(v))
        z = torch.where(valid3, z, torch.full_like(z, pad))
        beta_logits = torch.where(valid[:, None], beta_logits, torch.full_like(beta_logits, pad))

        qf, kf = l2_normalize(q.float()), l2_normalize(k.float())
        a = torch.exp(self.A_log.float())[None, :, None]
        g_log = self.gate_lower_bound * torch.sigmoid(a * (z.float() + self.dt_bias.view(hc, D).float()[None]))
        o, S = kda_scan(qf, kf, v.float(), g_log, torch.sigmoid(beta_logits.float()), state.temporal.float())
        o = o.to(v.dtype).float()

        rstd = torch.rsqrt((o * o).mean(-1, keepdim=True) + self.o_norm_eps)
        o = o * rstd * self.o_norm_weight.float() * torch.sigmoid(g2.float())
        y = comm.reduce_scatter(self.o_proj(o.reshape(T, hc * D).to(x.dtype)), dim=0)
        y = zero_pad_rows(y, valid_len, comm)
        new_state = KdaState(
            conv=torch.stack(conv_state).to(state.conv.dtype).reshape(3 * hc * D, self.conv_size - 1),
            temporal=S.to(state.temporal.dtype),
        )
        return y, new_state

    # kernel backend

    def kernel_buffers(self, mx_quantize, mx_pack) -> dict[str, torch.Tensor]:
        """The kernel backend's weight inputs, built from the loaded torch weights (host work, once per load).

        The q, k, v projections and the g | b projection go to MXFP8 (E8M0 scale per 32 inputs) in the kernel's
        group-major row order (groups of 3 heads); ``mx_quantize(weight [out, in] fp32) -> (bytes, scales)`` and
        ``mx_pack(bytes, scales) -> (data, scale)`` are the kernel package's host quantizer and packer. The f_a
        tile, ``f_b^T``, the fp32 conv taps, ``A_log``, ``dt_bias``, the output norm and ``o_proj^T`` keep their
        dtypes. The scan constants are ``const0`` .. ``const5``.
        """
        import numpy as np

        hc, D = self.hc, self.head_dim
        groups = front_head_groups(hc)

        def to_np(w):
            return w.detach().float().cpu().numpy()

        qkv = _group_major([self.q_proj.weight.data, self.k_proj.weight.data, self.v_proj.weight.data], groups)
        d_qkv, s_qkv = (np.asarray(t, np.uint8) for t in mx_quantize(to_np(qkv)))
        datas, scales = zip(*(mx_pack(d_qkv[t : t + _FRONT_TILE], s_qkv[t : t + _FRONT_TILE])
                              for t in range(0, d_qkv.shape[0], _FRONT_TILE)))
        gb = _group_major([self.g_proj.weight.data, self.b_proj.weight.data], groups)
        gb_data, gb_scale = mx_pack(*(np.asarray(t, np.uint8) for t in mx_quantize(to_np(gb))))
        device = self.A_log.device
        buffers = {
            "w_qkv_mx": torch.from_numpy(np.stack(datas, 0)),
            "w_qkv_scale": torch.from_numpy(np.stack(scales, 0)),
            "w_gb_mx": torch.from_numpy(np.ascontiguousarray(gb_data)),
            "w_gb_scale": torch.from_numpy(np.ascontiguousarray(gb_scale)),
            "w_fa": front_fa_tile(self.f_a_proj.weight.data),
            "w_fb_t": self.f_b_proj.weight.data.t().contiguous(),
            "conv_w": self.conv_weight.data.float().clone(),
            "A_log_2d": self.A_log.data.float().reshape(1, hc).clone(),
            "dt_bias_2d": self.dt_bias.data.float().view(hc, D).clone(),
            "o_norm_w": self.o_norm_weight.data.float().reshape(1, D).clone(),
            "o_proj_w_t": self.o_proj.weight.data.t().contiguous(),
        }
        buffers.update({f"const{i}": value for i, value in enumerate(kda_scan_constants())})
        return {name: value.to(device) for name, value in buffers.items()}

    def set_kernel_buffers(self, buffers: Mapping[str, torch.Tensor]) -> None:
        """Register the kernel inputs under the names of ``kernel_buffers`` (the loader's kernel forms)."""
        for name, value in buffers.items():
            self.register_buffer(f"kernel_{name}", value, persistent=False)

    def prepare_kernel(self, mx_quantize=None, mx_pack=None) -> None:
        """Build the kernel's trace shim, and the kernel inputs (``kernel_buffers``) when ``set_kernel_buffers`` has
        not given them; call once after the load, outside any traced forward. The quantizer and packer default to the
        kernel package's."""
        if not hasattr(self, "kernel_w_qkv_mx"):
            if mx_quantize is None or mx_pack is None:
                mx_quantize, mx_pack = kernel_imports.kda_front_mx_packer()
            self.set_kernel_buffers(self.kernel_buffers(mx_quantize, mx_pack))
        self._block_shim = _kda_block_shim()

    def _forward_kernel(self, x_rows, state: KdaState, valid_len, comm):
        from sglang_neuron.backend import stack

        if self._block_shim is None:
            raise ValueError("The KDA kernel backend is not prepared. Call prepare_kernel() after the load.")
        if x_rows.dtype != torch.bfloat16 or (x_rows.shape[0] * self.world) % _SCAN_CHUNK:
            raise ValueError(f"The KDA block kernel takes bf16 rows, {_SCAN_CHUNK} x k in all; got "
                             f"{x_rows.shape[0]} x {self.world} {x_rows.dtype}.")
        kb = {name[len("kernel_"):]: buf for name, buf in self.named_buffers() if name.startswith("kernel_")}
        groups = getattr(comm, "replica_groups", None) or (tuple(range(comm.world)),)
        y, conv_new, s_new = stack.wrap_nki(self._block_shim)[_GRID](
            x=x_rows,
            w_fa=kb["w_fa"],
            w_fb_t=kb["w_fb_t"],
            conv_w=kb["conv_w"],
            conv_state=state.conv.reshape(3, self.hc * self.head_dim, self.conv_size - 1).to(torch.bfloat16),
            A_log=kb["A_log_2d"],
            dt_bias=kb["dt_bias_2d"],
            S0=state.temporal.float(),
            lcat=kb["const0"],
            wcat=kb["const1"],
            mask_a3=kb["const2"],
            mask_b3=kb["const3"],
            mask_lvl=kb["const4"],
            ident=kb["const5"],
            o_norm_w=kb["o_norm_w"],
            o_proj_w_t=kb["o_proj_w_t"],
            valid_len=valid_len.reshape(-1)[:1].reshape(1, 1).to(torch.int32),
            w_qkv_mx=kb["w_qkv_mx"],
            w_qkv_scale=kb["w_qkv_scale"],
            w_gb_mx=kb["w_gb_mx"],
            w_gb_scale=kb["w_gb_scale"],
            world_ranks=tuple(tuple(int(r) for r in g) for g in groups),
            lower_bound=self.gate_lower_bound,
            eps=float(self.o_norm_eps),
        )
        new_state = KdaState(
            conv=conv_new.to(state.conv.dtype).reshape(3 * self.hc * self.head_dim, self.conv_size - 1),
            temporal=s_new.to(state.temporal.dtype),
        )
        return y.to(x_rows.dtype), new_state


def _kda_block_shim():
    """The trace shim of the KDA block kernel in its f_a-only MXFP8 form (no bf16 ``w_gb``; ``mx=1``)."""
    import nki

    kernel = kernel_imports.kda_block_prefill()

    @nki.jit(mode="trace")
    def kimi_k3_kda_block_fa_mx(
        x, w_fa, w_fb_t, conv_w, conv_state, A_log, dt_bias, S0, lcat, wcat, mask_a3, mask_b3, mask_lvl, ident,
        o_norm_w, o_proj_w_t, valid_len, w_qkv_mx, w_qkv_scale, w_gb_mx, w_gb_scale,
        world_ranks: tuple = (), lower_bound: float = -5.0, eps: float = 1e-5,
    ):
        return kernel(
            x, w_fa, None, w_fb_t, conv_w, conv_state, A_log, dt_bias, S0, lcat, wcat, mask_a3, mask_b3, mask_lvl,
            ident, o_norm_w, o_proj_w_t, valid_len,
            world_ranks=world_ranks, lower_bound=lower_bound, eps=eps, mx=1,
            w_qkv_mx=w_qkv_mx, w_qkv_scale=w_qkv_scale, w_gb_mx=w_gb_mx, w_gb_scale=w_gb_scale,
        )

    return kimi_k3_kda_block_fa_mx
