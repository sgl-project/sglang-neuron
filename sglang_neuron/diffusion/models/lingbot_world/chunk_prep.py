# SPDX-License-Identifier: Apache-2.0
"""Per-chunk LingBot-World inputs built inside a static device graph from a few small tensors.

Each chunk needs (a) the RoPE ``(cos, sin)`` tables of its token positions and (b) the folded,
patchified Plücker rays of its camera trajectory. Built on the host they are megabytes per rank
per chunk (float64 tables, a 480x832 ray grid per frame) plus the upload; here the host provides
only what actually changes between chunks and the graph assembles the rest:

* **RoPE.** Wan's 3D RoPE angle of a token is ``(frame, row, col) * frequency`` per lane group.
  Only the temporal part depends on the chunk (its start frame). The host evaluates the
  ``frames x 22`` temporal ``cos``/``sin`` values in float64 -- the reduced phase, so there is no
  FP32 angle error at any frame index -- and the row/column factors are session constants. The
  graph only broadcasts and concatenates them, so the assembled tables equal the host-built
  :func:`lingbot_rotary_tables` bit for bit (no arithmetic in the graph). The tables are in the
  fused QKV kernel's rotate-half layout, ``[1, S, head_dim]``.
* **Plücker rays.** The host packs one ``[frames, 16]`` FP32 row per frame -- the prepared
  camera-to-world rotation and translation and the pinhole intrinsics (as ``1/fx, 1/fy, cx, cy``)
  -- and the graph evaluates the rays directly in the patch-embedding feature order of this CP
  rank's tokens (the 8x8 pixel fold and the 1x2x2 patchify are index arithmetic on a static
  grid), so neither the full-resolution ray tensor nor its fold permutation is ever built.

Everything is shape-static: one graph per resolution serves every chunk of a session.
"""

import torch

# Packed per-frame camera row: rotation (row-major 3x3), translation, 1/fx, 1/fy, cx, cy.
CAMERA_PARAMS = 16
CAMERA_FOLD = 8  # pixels per latent cell along each axis (the VAE's spatial factor)


# --------------------------------------------------------------------------- RoPE


def _rope_frequencies(dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))


def rope_axis_dims(head_dim: int = 128) -> tuple[int, int, int]:
    """Wan's ``(temporal, height, width)`` rotary lane counts (44, 42, 42 at head_dim 128)."""
    h_dim = w_dim = 2 * (head_dim // 6)
    return head_dim - h_dim - w_dim, h_dim, w_dim


def lingbot_rope_spatial_factors(
    height: int, width: int, *, head_dim: int = 128, theta: float = 10000.0
) -> tuple[torch.Tensor, ...]:
    """Session constants ``(h_cos, h_sin, w_cos, w_sin)``: FP32 ``[height, 21]`` / ``[width, 21]``.

    Evaluated in float64 and rounded once, exactly like :func:`lingbot_rotary_tables`.
    """
    _, h_dim, w_dim = rope_axis_dims(head_dim)
    h = torch.outer(torch.arange(height, dtype=torch.float64), _rope_frequencies(h_dim, theta))
    w = torch.outer(torch.arange(width, dtype=torch.float64), _rope_frequencies(w_dim, theta))
    return h.cos().float(), h.sin().float(), w.cos().float(), w.sin().float()


def lingbot_rope_temporal_factors(
    frames: int, *, start_frame: int, head_dim: int = 128, theta: float = 10000.0
) -> torch.Tensor:
    """Per-chunk ``[2, frames, 22]`` FP32 ``(cos, sin)`` of the chunk's temporal angles.

    The only chunk-dependent RoPE input (~0.5 KB). float64 on the host: exact at any start frame.
    """
    t_dim, _, _ = rope_axis_dims(head_dim)
    positions = torch.arange(start_frame, start_frame + frames, dtype=torch.float64)
    t = torch.outer(positions, _rope_frequencies(t_dim, theta))
    return torch.stack((t.cos(), t.sin())).float()


def lingbot_rotary_tables(
    frames: int,
    height: int,
    width: int,
    *,
    start_frame: int,
    head_dim: int = 128,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host reference: 3D RoPE tables for one chunk whose first post-patch frame is ``start_frame``.

    Same factorization as Wan (temporal ``head_dim - 4*(head_dim//6)``, height/width
    ``2*(head_dim//6)`` each) with the temporal index offset by ``start_frame``, matching the
    official ``causal_rope_apply(..., start_frame)``. Angles are computed in float64 on the host,
    so any start frame is exact (no 1024-frame table limit). :func:`assemble_rotary_tables`
    builds the same tables in the graph.

    Returns ``(cos, sin)`` as float32 ``[1, frames*height*width, head_dim]`` in the rotate-half
    layout of the fused QKV kernel (:mod:`~sglang_neuron.diffusion.layers.rope_rotate_half`):
    the ``head_dim // 2`` angles, then the same angles again. The model's Q/K lanes are
    de-interleaved at load, so rotate-half over them equals the official interleaved RoPE.
    """
    t_dim, h_dim, w_dim = rope_axis_dims(head_dim)

    def angles_of(positions, dim):
        return torch.outer(positions.to(torch.float64), _rope_frequencies(dim, theta))

    t = angles_of(torch.arange(start_frame, start_frame + frames), t_dim)
    h = angles_of(torch.arange(height), h_dim)
    w = angles_of(torch.arange(width), w_dim)
    angles = torch.cat(
        (
            t.view(frames, 1, 1, -1).expand(frames, height, width, -1),
            h.view(1, height, 1, -1).expand(frames, height, width, -1),
            w.view(1, 1, width, -1).expand(frames, height, width, -1),
        ),
        dim=-1,
    ).reshape(1, frames * height * width, -1)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().float(), angles.sin().float()


def assemble_rotary_tables(
    temporal: torch.Tensor,
    h_cos: torch.Tensor,
    h_sin: torch.Tensor,
    w_cos: torch.Tensor,
    w_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-graph: the chunk's rotate-half ``(cos, sin)`` ``[1, F*H*W, head_dim]`` from the factors.

    Pure broadcast + concat: equal to :func:`lingbot_rotary_tables`
    for the same start frame, bit for bit.
    """
    frames = temporal.shape[1]
    height, width = h_cos.shape[0], w_cos.shape[0]
    shape = (frames, height, width)

    def table(t, h, w):
        angles = torch.cat(
            (
                t.reshape(frames, 1, 1, -1).expand(*shape, -1),
                h.reshape(1, height, 1, -1).expand(*shape, -1),
                w.reshape(1, 1, width, -1).expand(*shape, -1),
            ),
            dim=-1,
        )
        angles = angles.reshape(1, frames * height * width, -1)
        return torch.cat((angles, angles), dim=-1)

    return table(temporal[0], h_cos, w_cos), table(temporal[1], h_sin, w_sin)


# --------------------------------------------------------------------------- Plücker rays


def camera_ray_params(poses: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Host: pack prepared poses + intrinsics into the ``[frames, 16]`` FP32 graph input.

    Args:
        poses: ``[F, 3, 4]`` or ``[F, 4, 4]`` camera-to-world (already in the conditioning frame,
            e.g. vLLM-Omni's ``_prepare_framewise_poses``).
        intrinsics: ``[F, 4]`` ``(fx, fy, cx, cy)`` in pixels of the conditioning resolution.
    """
    poses = poses.to(torch.float64)
    intrinsics = intrinsics.to(torch.float64)
    fx, fy, cx, cy = intrinsics.unbind(-1)
    if torch.any(fx == 0) or torch.any(fy == 0):
        raise ValueError("camera focal lengths fx and fy must be non-zero")
    return torch.cat(
        (
            poses[:, :3, :3].reshape(-1, 9),
            poses[:, :3, 3],
            torch.stack((1.0 / fx, 1.0 / fy, cx, cy), dim=-1),
        ),
        dim=-1,
    ).float()


def camera_token_grid(
    frames: int, latent_h: int, latent_w: int, patch_size=(1, 2, 2), device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-graph static grid of the post-patch tokens: ``(frame [S] int64, origin [S, 2] FP32)``.

    ``origin`` is the ``(y0, x0)`` top-left pixel of the token's ``(ph*8) x (pw*8)`` pixel block.
    Built from broadcasts of ``arange`` (no integer division); FP32 is exact (< 2^24).
    """
    p_t, p_h, p_w = patch_size
    if p_t != 1:
        raise ValueError("temporal patch size must be 1")
    gh, gw = latent_h // p_h, latent_w // p_w
    shape = (frames, gh, gw)
    frame = torch.arange(frames, device=device, dtype=torch.int64).reshape(frames, 1, 1)
    y0 = torch.arange(gh, device=device, dtype=torch.float32) * float(p_h * CAMERA_FOLD)
    x0 = torch.arange(gw, device=device, dtype=torch.float32) * float(p_w * CAMERA_FOLD)
    origin = torch.stack(
        (y0.reshape(1, gh, 1).expand(shape), x0.reshape(1, 1, gw).expand(shape)), dim=-1
    )
    return frame.expand(shape).reshape(-1), origin.reshape(-1, 2)


def plucker_patches(
    params: torch.Tensor,
    frame: torch.Tensor,
    origin: torch.Tensor,
    *,
    patch_size=(1, 2, 2),
) -> torch.Tensor:
    """In-graph: Plücker-ray patches ``[n, 6 * 64 * ph * pw]`` FP32 for ``n`` tokens.

    Matches vLLM-Omni ``build_plucker_embedding`` -> ``_fold_camera_embedding`` (pixel-unshuffle
    by 8, channels ``(c, c1, c2)``) -> the patch embedding's ``(C, pt, ph, pw)`` feature order:
    feature ``c * 64*ph*pw + ((c1*8 + c2)*ph + i)*pw + j`` of the token whose pixel block starts
    at ``(y0, x0)`` reads pixel ``(y0 + 8*i + c1, x0 + 8*j + c2)``. Channels ``c`` = ray origin
    xyz, then the unit world direction xyz (checkpoint order). The intrinsics are already in
    pixels of the generated resolution, so pixel centres map to them unscaled.

    Args:
        params: ``[F, 16]`` from :func:`camera_ray_params` (on the device).
        frame, origin: ``[n]`` / ``[n, 2]`` rows of :func:`camera_token_grid` (this rank's tokens).
    """
    _, p_h, p_w = patch_size
    fold = CAMERA_FOLD
    device = params.device
    p = params.index_select(0, frame)[:, :, None]  # [n, 16, 1]: exact per-token camera row
    # In-token pixel offsets in feature order (c1, c2, i, j), FP32 exact.
    c1 = torch.arange(fold, device=device, dtype=torch.float32).reshape(fold, 1, 1, 1)
    c2 = torch.arange(fold, device=device, dtype=torch.float32).reshape(1, fold, 1, 1)
    i = torch.arange(p_h, device=device, dtype=torch.float32).reshape(1, 1, p_h, 1)
    j = torch.arange(p_w, device=device, dtype=torch.float32).reshape(1, 1, 1, p_w)
    shape = (fold, fold, p_h, p_w)
    dy = (c1 + float(fold) * i).expand(shape).reshape(1, -1)
    dx = (c2 + float(fold) * j).expand(shape).reshape(1, -1)
    y = origin[:, 0:1] + dy + 0.5
    x = origin[:, 1:2] + dx + 0.5
    # Pinhole back-projection, normalized: d = ((x - cx)/fx, (y - cy)/fy, 1) / |.|.
    cam_x = (x - p[:, 14]) * p[:, 12]
    cam_y = (y - p[:, 15]) * p[:, 13]
    sq = cam_x * cam_x + cam_y * cam_y + 1.0
    inv = torch.rsqrt(sq)
    # One Newton step: FP32 accuracy even where the device rsqrt is approximate.
    inv = inv * (1.5 - 0.5 * sq * inv * inv)
    dxn, dyn, dzn = cam_x * inv, cam_y * inv, inv
    world = [p[:, 3 * r] * dxn + p[:, 3 * r + 1] * dyn + p[:, 3 * r + 2] * dzn for r in range(3)]
    n, positions = cam_x.shape
    ray_origin = [p[:, 9 + r].expand(n, positions) for r in range(3)]
    return torch.stack((*ray_origin, *world), dim=1).reshape(n, 6 * positions)


def plucker_patches_reference(
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    patch_size=(1, 2, 2),
) -> torch.Tensor:
    """CPU float64 reference: full-resolution rays, 8x8 fold, then patchify -> ``[S, 1536]``."""
    poses = poses.to(torch.float64)
    fx, fy, cx, cy = intrinsics.to(torch.float64).unbind(-1)
    frames = poses.shape[0]
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float64) + 0.5,
        torch.arange(width, dtype=torch.float64) + 0.5,
        indexing="ij",
    )
    cam = torch.stack(
        (
            (xs[None] - cx[:, None, None]) / fx[:, None, None],
            (ys[None] - cy[:, None, None]) / fy[:, None, None],
            torch.ones(frames, height, width, dtype=torch.float64),
        ),
        dim=-1,
    )
    cam = cam / cam.norm(dim=-1, keepdim=True)
    world = torch.einsum("fij,fhwj->fhwi", poses[:, :3, :3], cam)
    origin = poses[:, None, None, :3, 3].expand_as(world)
    rays = torch.cat((origin, world), dim=-1)  # [F, H, W, 6]
    lat_h, lat_w = height // CAMERA_FOLD, width // CAMERA_FOLD
    folded = rays.reshape(frames, lat_h, CAMERA_FOLD, lat_w, CAMERA_FOLD, 6).permute(
        5, 2, 4, 0, 1, 3
    )
    folded = folded.reshape(1, 6 * CAMERA_FOLD**2, frames, lat_h, lat_w)
    _, p_h, p_w = patch_size
    x = folded.reshape(1, -1, frames, 1, lat_h // p_h, p_h, lat_w // p_w, p_w)
    return x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(-1, 6 * CAMERA_FOLD**2 * p_h * p_w)


def assert_static_resolution(height: int, width: int, patch_size=(1, 2, 2)) -> None:
    """The in-graph ray builder needs whole patches of whole 8x8 cells."""
    _, p_h, p_w = patch_size
    if height % (CAMERA_FOLD * p_h) or width % (CAMERA_FOLD * p_w):
        raise ValueError(f"{height}x{width} is not a multiple of {CAMERA_FOLD}x patch")


__all__ = [
    "CAMERA_PARAMS",
    "assemble_rotary_tables",
    "assert_static_resolution",
    "camera_ray_params",
    "camera_token_grid",
    "lingbot_rope_spatial_factors",
    "lingbot_rope_temporal_factors",
    "plucker_patches",
    "plucker_patches_reference",
    "rope_axis_dims",
]
