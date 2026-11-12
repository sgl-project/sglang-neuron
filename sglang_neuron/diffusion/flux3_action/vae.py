# SPDX-License-Identifier: Apache-2.0
"""The FLUX.3 video VAE encoder on Neuron: SGLang's module, its neighborhood attention as tiled dense blocks, compiled
as one graph."""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

KERNEL = 5
TILES = {(136, 184): (2, 46), (68, 92): (4, 23), (34, 46): (2, 46), (17, 23): (17, 23), (88, 184): (2, 46),
         (44, 92): (4, 23)}
PATCH = 4
SPLIT_STAGES = 2  # VAE stages whose canvas rows split over the two chips of a CP pair
DROID_CAMERA_HW = (360, 640)
COMPILER_ARGS = (
    "--model-type=unet-inference",
    "--auto-cast=none",
    "--internal-max-instruction-limit=15000000",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
    "--internal-hlo2tensorizer-options=--hierarchy-splitter=false --layers-per-module=0",
)

_BIAS: dict[tuple[int, int, int], torch.Tensor] = {}


def _block_starts(n: int, tile: int, kernel: int) -> tuple[list[int], int]:
    """First index of the key block of every query tile on one axis, and the block length."""
    span = min(tile + kernel - 1, n)
    return [min(max(t * tile - kernel // 2, 0), n - span) for t in range(n // tile)], span


def neighborhood_bias(height: int, width: int, tile: tuple[int, int], kernel: int = KERNEL) -> torch.Tensor:
    """Additive fp32 mask ``[H/th, W/tw, th*tw, bh*bw]``: 0 where a key is in the query's window."""

    def axis(n: int, t: int) -> torch.Tensor:
        starts, span = _block_starts(n, t, kernel)
        query = torch.arange(n).reshape(n // t, t, 1)
        key = (torch.tensor(starts)[:, None] + torch.arange(span))[:, None, :]
        lo = (query - kernel // 2).clamp(0, n - kernel)
        return (key >= lo) & (key < lo + kernel)

    rows, cols = axis(height, tile[0]), axis(width, tile[1])
    allowed = rows[:, None, :, None, :, None] & cols[None, :, None, :, None, :]
    allowed = allowed.reshape(rows.shape[0], cols.shape[0], tile[0] * tile[1], -1)
    return torch.zeros(allowed.shape, dtype=torch.float32).masked_fill(~allowed, torch.finfo(torch.float32).min)


def prepare(device: torch.device, kernel: int = KERNEL) -> None:
    """Put the masks of the DROID canvas grids on ``device``, so a compiled graph reads them as constants."""
    for (height, width), tile in TILES.items():
        _BIAS[(height, width, kernel)] = neighborhood_bias(height, width, tile, kernel).to(device)


def _key_blocks(x: torch.Tensor, tile: tuple[int, int], kernel: int) -> torch.Tensor:
    """``[B, H, W, N, D]`` -> ``[B, H/th, W/tw, N, bh*bw, D]``: the keys around every query tile."""
    b, h, w, n, d = x.shape
    rows, bh = _block_starts(h, tile[0], kernel)
    cols, bw = _block_starts(w, tile[1], kernel)
    x = torch.stack([x[:, r : r + bh] for r in rows], dim=1)
    x = torch.stack([x[:, :, :, c : c + bw] for c in cols], dim=2)
    return x.permute(0, 1, 2, 5, 3, 4, 6).reshape(b, len(rows), len(cols), n, bh * bw, d)


def neighborhood_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, kernel_size: list[int],
                           is_causal: list[bool] | None = None) -> torch.Tensor:
    """NATTEN ``na2d`` (stride 1, no dilation; NATTEN has no Neuron build) over
    ``(B, H, W, heads, head_dim)``. The VAE's q and k are RMS-normalized without affine and then rotated, so every
    scaled logit is at most sqrt(head_dim) and ``exp`` needs no row max.

    Args:
        q: Queries; ``k`` and ``v`` have the same shape.
        kernel_size: The window, equal on both axes.
        is_causal: Must be unset or all False.
    """
    if q.ndim != 5 or len(set(kernel_size)) != 1 or any(is_causal or ()):
        raise NotImplementedError(f"Neuron neighborhood attention is 2-D, square and non-causal; got {tuple(q.shape)}, "
                                  f"kernel {kernel_size}, causal {is_causal}")
    kernel = kernel_size[0]
    b, h, w, n, d = q.shape
    th, tw = tile = TILES.get((h, w), (h, w))
    bias = _BIAS.get((h, w, kernel))
    if bias is None:
        bias = neighborhood_bias(h, w, tile, kernel).to(q.device)
    qt = (q * d**-0.5).reshape(b, h // th, th, w // tw, tw, n, d).permute(0, 1, 3, 5, 2, 4, 6)
    qt = qt.reshape(b, h // th, w // tw, n, th * tw, d)
    keys = _key_blocks(k, tile, kernel).float().transpose(-1, -2)
    weights = torch.exp(torch.matmul(qt.float(), keys) + bias[:, :, None])
    out = torch.matmul(weights, _key_blocks(v, tile, kernel).float())
    out = out / weights.sum(dim=-1, keepdim=True)
    out = out.to(q.dtype).reshape(b, h // th, w // tw, n, th, tw, d)
    return out.permute(0, 1, 4, 2, 5, 3, 6).reshape(b, h, w, n, d)


class RowParallelLinear(nn.Module):
    """``linear`` over this rank's input ``columns``, summed over the TP group (one all-reduce), then its bias."""

    def __init__(self, linear: nn.Linear, columns: slice):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data[:, columns].contiguous(), requires_grad=False)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from sglang.multimodal_gen.runtime.distributed import get_tp_group

        from sglang_neuron.parallel import collectives

        out = collectives.all_reduce(get_tp_group(), F.linear(x, self.weight))
        return out if self.bias is None else out + self.bias


def _rope_tables(rope, t: int, h: int, w: int, row_offset: int, dtype: torch.dtype, device: torch.device):
    """``RotaryPositionEmbedding3D.cos_sin`` of a grid whose rows start at ``row_offset``, without its memo (in a
    compiled graph the tables fold to constants)."""
    grids = torch.meshgrid(torch.arange(t, device=device, dtype=torch.float32),
                           torch.arange(h, device=device, dtype=torch.float32) + row_offset,
                           torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    pos = torch.stack(grids + (torch.zeros_like(grids[0]),), dim=-1)
    freqs = torch.einsum("...a,af->...af", pos, rope.inv_freq.float()).reshape(1, t, h, w, 1, -1)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _local_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    """``Natten3D.forward`` of one frame on this rank's heads and on rows starting at the module's ``row_offset``:
    SGLang's QK norm and RoPE, then ``neighborhood_attention``; the output keeps the local channel count."""
    b, t, h, w, _ = x.shape
    if t != 1:
        raise NotImplementedError(f"the Neuron VAE encodes one frame at a time, got {t}")
    q, k, v = self.qkv(x).reshape(b, t, h, w, 3, self.num_heads, self.head_dim).unbind(4)
    if self.qk_norm:
        q, k = self.q_norm(q), self.k_norm(k)
    cos, sin = _rope_tables(self.rope, t, h, w, getattr(self, "row_offset", 0), q.dtype, q.device)
    q, k = (z * cos + self.rope._rotate_half(z) * sin for z in (q, k))
    out = neighborhood_attention(q[:, 0], k[:, 0], v[:, 0], kernel_size=self.window_size[1:])
    return self.proj(out.reshape(b, t, h, w, -1))


def _split_attention(module, rank: int, size: int) -> None:
    heads = module.num_heads // size
    local = slice(rank * heads * module.head_dim, (rank + 1) * heads * module.head_dim)
    dim = module.num_heads * module.head_dim
    qkv = nn.Linear(dim, 3 * heads * module.head_dim, dtype=module.qkv.weight.dtype)
    qkv.weight = nn.Parameter(module.qkv.weight.data.view(3, dim, -1)[:, local].reshape(-1, dim).contiguous(),
                              requires_grad=False)
    qkv.bias = nn.Parameter(module.qkv.bias.data.view(3, dim)[:, local].reshape(-1).contiguous(), requires_grad=False)
    module.qkv = qkv
    module.proj = RowParallelLinear(module.proj, local)
    module.num_heads = heads


def _split_mlp(module, rank: int, size: int) -> None:
    hidden = module.down_proj.in_features
    local = slice(rank * hidden // size, (rank + 1) * hidden // size)
    weight = module.gate_up_proj.weight.data.view(2, hidden, -1)[:, local].reshape(-1, module.gate_up_proj.in_features)
    gate_up = nn.Linear(module.gate_up_proj.in_features, weight.shape[0], bias=False, dtype=weight.dtype)
    gate_up.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
    module.gate_up_proj = gate_up
    module.down_proj = RowParallelLinear(module.down_proj, local)


def tensor_parallel(vae: nn.Module, rank: int, size: int) -> None:
    """Split every Swin block of SGLang's VAE over the TP group: this rank's attention heads and MLP channels, the
    output projections summed with one all-reduce each. The TP group is one chip's cores, so the collectives stay on
    the chip. Raises ``ValueError`` when ``size`` does not divide every block's head count and MLP width."""
    from sglang.multimodal_gen.runtime.models.vaes.flux3_video_vae import GLUMLP, Natten3D

    heads = {m.num_heads for m in vae.modules() if isinstance(m, Natten3D)}
    widths = {m.down_proj.in_features for m in vae.modules() if isinstance(m, GLUMLP)}
    if any(n % size for n in heads | widths):
        raise ValueError(f"the VAE's tensor-parallel size {size} must divide its attention head counts {sorted(heads)} "
                         f"and MLP widths {sorted(widths)}")
    for module in list(vae.modules()):
        if isinstance(module, Natten3D):
            if size > 1:
                _split_attention(module, rank, size)
            module.forward = types.MethodType(_local_attention_forward, module)
        elif isinstance(module, GLUMLP) and size > 1:
            _split_mlp(module, rank, size)


def split_rows(height: int, depths: tuple[int, ...]) -> int:
    """Canvas rows each chip of a CP pair encodes through the first ``SPLIT_STAGES`` stages: half the canvas plus the
    window halo, rounded up to 32; ``height`` when that saves nothing."""
    halo = sum(depths[s] * (KERNEL // 2) * (PATCH << s) for s in range(SPLIT_STAGES))
    return min(-(-(height // 2 + halo) // 32) * 32, height)


def _merges(vae: nn.Module) -> list[int]:
    """Indices in the encoder's ``features`` of its spatial patch merges, one after every stage but the last."""
    from sglang.multimodal_gen.runtime.models.vaes.flux3_video_vae import PatchMerging

    return [i for i, m in enumerate(vae.model.encoder.features) if isinstance(m, PatchMerging)]


def row_split(vae: nn.Module, part: int, height: int) -> tuple[int, int, int, int] | None:
    """Set the RoPE row offset of the first ``SPLIT_STAGES`` stages for chip ``part`` (0 or 1) of a CP pair; returns
    ``(start, rows, first, keep)``: its canvas rows ``[start, start + rows)``, and the rows ``[first, first + keep)``
    of its stage-``SPLIT_STAGES`` grid that are exact, or None when the split saves nothing."""
    from sglang.multimodal_gen.runtime.models.vaes.flux3_video_vae import Natten3D

    depths = tuple(len(vae.model.encoder.features[i - 1]) for i in _merges(vae))
    rows = split_rows(height, depths)
    if rows == height:
        return None
    start = part * (height - rows)
    features = vae.model.encoder.features
    end = _merges(vae)[SPLIT_STAGES - 1]
    stage = 0
    for i, module in enumerate(features):
        if i > end:
            break
        for natten in [m for m in module.modules() if isinstance(m, Natten3D)]:
            natten.row_offset = start // (PATCH << stage)
        if i in _merges(vae):
            stage += 1
    keep = height // (PATCH << SPLIT_STAGES) // 2
    return start, rows, part * (rows // (PATCH << SPLIT_STAGES) - keep), keep


def droid_canvas(views: torch.Tensor, canvas_hw: tuple[int, int]) -> torch.Tensor:
    """SGLang's DROID canvas from the ``[1, 3, 360, 640, 3]`` wrist, left and right pixel values (0 to 255): the wrist
    view over the two exterior views halved (bilinear, no antialias: the 2x2 mean), reflect-padded to ``canvas_hw``,
    in [-1, 1] as ``[1, 3, Hc, Wc]``."""
    cams = views[0].float().permute(0, 3, 1, 2) / 255.0
    half = cams[1:].unflatten(2, (-1, 2)).unflatten(4, (-1, 2)).mean(dim=(3, 5))
    composite = torch.cat([cams[0], torch.cat([half[0], half[1]], dim=-1)], dim=-2)
    height, width = composite.shape[-2:]
    pad_w, pad_h = canvas_hw[1] - width, canvas_hw[0] - height
    composite = torch.cat([composite, composite[..., width - 1 - pad_w:width - 1].flip(-1)], dim=-1)
    composite = torch.cat([composite, composite[..., height - 1 - pad_h:height - 1, :].flip(-2)], dim=-2)
    return composite[None] * 2.0 - 1.0


def encode(vae: nn.Module, canvas: torch.Tensor, split: tuple[int, int, int, int] | None) -> torch.Tensor:
    """``ViTNorm.encode`` of the ``[1, 3, H, W]`` canvas; with ``split``, the first ``SPLIT_STAGES`` stages run on
    this chip's canvas rows and their exact rows are gathered over the CP pair."""
    from sglang.multimodal_gen.runtime.distributed import get_sp_group

    from sglang_neuron.parallel import collectives

    model = vae.model
    x = canvas[:, :, None]
    if split is not None:
        start, rows, first, keep = split
        x = x[:, :, :, start:start + rows]
    x = model.encoder.patch_embed(x)
    end = _merges(vae)[SPLIT_STAGES - 1]
    for i, module in enumerate(model.encoder.features):
        x = module(x)
        if split is not None and i == end:
            x = collectives.all_gather(get_sp_group(), x[:, :, first:first + keep], dim=2)
    mu, _ = model.encoder.proj(x).permute(0, 4, 1, 2, 3).chunk(2, dim=-4)
    return model.z_normalizer.normalize(mu)


def tokens(vae: nn.Module, frame: torch.Tensor, canvas_hw: tuple[int, int], latent_hw: tuple[int, int],
           split) -> torch.Tensor:
    """SGLang's observation tokens ``[1, h*w, C]`` of a frame as it came to the device: the uint8 DROID views
    ``[1, 3, 360, 640, 3]`` (the canvas built in the graph) or a canvas ``[1, 3, Hc, Wc]``, cast to the VAE's dtype in
    the graph."""
    canvas = droid_canvas(frame, canvas_hw) if frame.ndim == 5 else frame.float()
    latent = encode(vae, canvas.to(next(vae.parameters()).dtype), split)
    h, w = latent_hw
    return latent[..., :h, :w].flatten(2).transpose(1, 2)
