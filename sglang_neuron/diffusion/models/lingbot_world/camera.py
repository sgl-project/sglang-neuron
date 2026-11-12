# SPDX-License-Identifier: Apache-2.0
"""LingBot-World camera inputs as the DiT's prep graph takes them: prepared poses and intrinsics.

The prep graph builds the ray grid (origin, unit direction) from per-latent-frame
``poses [F, 4, 4]`` (first-relative framewise deltas, translations normalized) and pixel
``intrinsics [F, 4]`` ``(fx, fy, cx, cy)``, both float64. Two sources:

* a recorded trajectory (``poses.npy`` / ``intrinsics.npy`` of a LingBot-World example):
  resampled to the latent frames and prepared once per session, as vLLM-Omni's LingBot
  pipeline does (the interpolation and pose preparation below are copied from
  ``vllm_omni.diffusion.models.lingbot_world.camera``);
* SGLang's per-chunk ``camera_actions`` (``wasd`` / ``ijkl`` per frame): SGLang's own action
  integration and relative poses (``configs/pipeline_configs/lingbot_world.py``), so a Neuron
  session moves as SGLang's GPU pipeline does.
"""

from __future__ import annotations

import math

import msgspec
import numpy as np
import torch

#: The resolution recorded LingBot-World intrinsics refer to.
REFERENCE_HEIGHT = 480
REFERENCE_WIDTH = 832


class CameraTrajectory(msgspec.Struct, frozen=True):
    """Camera-to-world poses ``[F, 4, 4]`` and ``(fx, fy, cx, cy)`` intrinsics ``[F, 4]``."""

    poses: torch.Tensor
    intrinsics: torch.Tensor


def _linear_interpolate(values: torch.Tensor, target_times: np.ndarray) -> torch.Tensor:
    source_times = np.linspace(0.0, 1.0, values.shape[0])
    values_array = values.detach().cpu().double().numpy()
    columns = [
        np.interp(target_times, source_times, values_array[:, column]) for column in range(values_array.shape[1])
    ]
    interpolated = torch.from_numpy(np.stack(columns, axis=1))
    return interpolated.to(device=values.device, dtype=values.dtype)


def _rotation_matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to normalized ``(w, x, y, z)`` quaternions."""

    quaternions: np.ndarray = np.empty((matrices.shape[0], 4), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
                quaternion = np.array(
                    [
                        (matrix[2, 1] - matrix[1, 2]) / scale,
                        0.25 * scale,
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                    ]
                )
            elif axis == 1:
                scale = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
                quaternion = np.array(
                    [
                        (matrix[0, 2] - matrix[2, 0]) / scale,
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        0.25 * scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                    ]
                )
            else:
                scale = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
                quaternion = np.array(
                    [
                        (matrix[1, 0] - matrix[0, 1]) / scale,
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                        0.25 * scale,
                    ]
                )
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
            raise ValueError("camera rotation matrices must describe valid rotations")
        quaternions[index] = quaternion / norm
    return quaternions


def _quaternions_to_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    w, x, y, z = quaternions.T
    matrices: np.ndarray = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)
    return matrices


def _slerp_rotations(matrices: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    quaternions = _rotation_matrices_to_quaternions(matrices)
    interpolated = np.empty((target_times.shape[0], 4), dtype=np.float64)
    for target_index, target_time in enumerate(target_times):
        scaled_time = float(target_time) * (matrices.shape[0] - 1)
        left = min(max(math.floor(scaled_time), 0), matrices.shape[0] - 2)
        fraction = scaled_time - left
        first = quaternions[left]
        second = quaternions[left + 1]
        cosine = float(np.dot(first, second))
        if cosine < 0.0:
            second = -second
            cosine = -cosine
        cosine = min(cosine, 1.0)
        if cosine > 0.9995:
            quaternion = first + fraction * (second - first)
            interpolated[target_index] = quaternion / np.linalg.norm(quaternion)
            continue
        angle = math.acos(cosine)
        denominator = math.sin(angle)
        interpolated[target_index] = (
            math.sin((1.0 - fraction) * angle) / denominator * first + math.sin(fraction * angle) / denominator * second
        )
    return _quaternions_to_rotation_matrices(interpolated)


def interpolate_camera_trajectory(
    trajectory: CameraTrajectory,
    num_frames: int,
) -> CameraTrajectory:
    """Resample translations/intrinsics linearly and rotations with quaternion SLERP."""

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    source_frames = trajectory.poses.shape[0]
    if num_frames == source_frames:
        return trajectory
    if source_frames == 1:
        return CameraTrajectory(
            poses=trajectory.poses.repeat(num_frames, 1, 1),
            intrinsics=trajectory.intrinsics.repeat(num_frames, 1),
        )

    # Rotations use quaternion SLERP; translations and intrinsics are linear.
    target_times = np.linspace(0.0, 1.0, num_frames)
    rotation_matrices = trajectory.poses[:, :3, :3].detach().cpu().double().numpy()
    interpolated_rotations = _slerp_rotations(rotation_matrices, target_times)
    translations = _linear_interpolate(trajectory.poses[:, :3, 3], target_times)
    intrinsics = _linear_interpolate(trajectory.intrinsics, target_times)

    poses = torch.eye(
        4,
        device=trajectory.poses.device,
        dtype=trajectory.poses.dtype,
    ).repeat(num_frames, 1, 1)
    poses[:, :3, :3] = torch.from_numpy(interpolated_rotations).to(
        device=trajectory.poses.device,
        dtype=trajectory.poses.dtype,
    )
    poses[:, :3, 3] = translations
    return CameraTrajectory(poses=poses, intrinsics=intrinsics)


def _invert_camera_poses(poses: torch.Tensor) -> torch.Tensor:
    """Invert rigid transforms using R^-1 = R^T and t^-1 = -R^T t."""

    rotations = poses[:, :3, :3]
    translations = poses[:, :3, 3:]
    inverse_rotations = rotations.transpose(-1, -2)
    inverse = torch.eye(4, device=poses.device, dtype=poses.dtype).repeat(poses.shape[0], 1, 1)
    inverse[:, :3, :3] = inverse_rotations
    inverse[:, :3, 3:] = -torch.bmm(inverse_rotations, translations)
    return inverse


def _prepare_framewise_poses(
    raw_poses: torch.Tensor,
    *,
    translation_scale: float | None = None,
) -> torch.Tensor:
    """Convert raw C2W poses to normalized first-relative framewise deltas."""

    # Remove global placement, then convert the sequence to framewise deltas.
    first_world_to_camera = _invert_camera_poses(raw_poses[:1])
    relative_poses = torch.matmul(first_world_to_camera, raw_poses).clone()
    relative_poses[0] = torch.eye(4, device=raw_poses.device, dtype=raw_poses.dtype)
    if relative_poses.shape[0] > 1:
        relative_poses[1:] = torch.bmm(
            _invert_camera_poses(relative_poses[:-1]),
            relative_poses[1:],
        )

    # Normalize the translation scale. Scripted and realtime input set the scale differently.
    translations = relative_poses[:, :3, 3]
    if translation_scale is None:
        # Scripted action_path input: divide by the largest step so an arbitrary recording scale becomes 1.
        scale = torch.linalg.vector_norm(translations, dim=-1).max()
    else:
        # Realtime interaction input: use the controller translation unit (one WASD step = full speed).
        # It cannot see future motion or normalize globally.
        if translation_scale <= 0:
            raise ValueError(f"translation_scale must be > 0, got {translation_scale}")
        scale = translation_scale
    if float(scale) > 0:
        relative_poses[:, :3, 3] = translations / scale
    return relative_poses


def prepare_recorded_trajectory(
    trajectory: CameraTrajectory, *, num_frames: int, num_latent_frames: int, height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A whole session's prepared ``(poses, intrinsics)`` (float64, one row per latent frame).

    The first ``num_frames`` recorded poses are resampled to the latent frames and prepared as
    one sequence (per-chunk preparation would normalize each chunk's translations by its own
    maximum); chunk ``c`` then takes rows ``[c * F, (c + 1) * F)``.
    """
    available = int(trajectory.poses.shape[0])
    if available < num_frames:
        raise ValueError(
            f"camera trajectory frames must be at least num_frames; got {available} < {num_frames}"
        )
    trajectory = CameraTrajectory(
        poses=trajectory.poses[:num_frames], intrinsics=trajectory.intrinsics[:num_frames]
    )
    trajectory = interpolate_camera_trajectory(trajectory, num_latent_frames)
    poses = _prepare_framewise_poses(trajectory.poses.to(torch.float64))
    intrinsics = trajectory.intrinsics.to(torch.float64).clone()
    intrinsics[:, (0, 2)] *= width / REFERENCE_WIDTH
    intrinsics[:, (1, 3)] *= height / REFERENCE_HEIGHT
    if torch.any(intrinsics[:, :2] == 0):
        raise ValueError("camera focal lengths fx and fy must be non-zero")
    return poses, intrinsics


def prepare_action_history(
    action_history: list[list[str]], *, frames: int, height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The last chunk's prepared ``(poses, intrinsics)`` from a session's ``camera_actions``
    history, with SGLang's semantics (``_build_camera_condition``): the history padded to a
    chunk, integrated into camera-to-world poses, made first-relative and framewise with
    translations normalized over the history, and the chunk's ``frames`` rows taken; pixel
    intrinsics ``(500, 500, W / 2, H / 2)``."""
    from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
        _actions_to_c2ws,
        _pad_actions_to_chunk,
    )
    from sglang.multimodal_gen.runtime.utils.camera_geometry import compute_relative_poses

    history = _pad_actions_to_chunk(action_history, frames)
    c2ws = torch.from_numpy(np.stack(_actions_to_c2ws(history)[1:])).to(torch.float64)
    poses = compute_relative_poses(c2ws, framewise=True)[-frames:].contiguous()
    intrinsics = torch.tensor(
        [[500.0, 500.0, width / 2, height / 2]], dtype=torch.float64
    ).repeat(frames, 1)
    return poses, intrinsics


def load_recorded_trajectory(directory: str) -> CameraTrajectory:
    """``poses.npy`` ``[N, 4, 4]`` and ``intrinsics.npy`` ``[N, 4]`` of a LingBot-World example."""
    import os

    poses = np.load(os.path.join(directory, "poses.npy"), allow_pickle=False)
    intrinsics = np.load(os.path.join(directory, "intrinsics.npy"), allow_pickle=False)
    return trajectory_from_lists(poses, intrinsics)


def trajectory_from_lists(poses, intrinsics, dtype: str | None = None) -> CameraTrajectory:
    """A trajectory from nested lists or arrays (a request's ``camera_trajectory``).

    ``dtype`` (``"float32"`` / ``"float64"``) is the recording's: the resampling runs in it, as
    vLLM-Omni resamples a loaded ``.npy`` in its own dtype. Default: the arrays' floating dtype;
    float64 for lists and for integer arrays (all-integer JSON), so the resampling interpolates.
    """
    poses, intrinsics = np.asarray(poses), np.asarray(intrinsics)
    if dtype is None and not (
        np.issubdtype(poses.dtype, np.floating) and np.issubdtype(intrinsics.dtype, np.floating)
    ):
        dtype = "float64"
    if dtype is not None:
        poses, intrinsics = poses.astype(dtype), intrinsics.astype(dtype)
    poses, intrinsics = torch.as_tensor(poses), torch.as_tensor(intrinsics)
    if poses.ndim != 3 or tuple(poses.shape[1:]) != (4, 4):
        raise ValueError(f"camera_trajectory poses must be [N, 4, 4], got {tuple(poses.shape)}")
    if intrinsics.ndim != 2 or intrinsics.shape[1] != 4 or intrinsics.shape[0] != poses.shape[0]:
        raise ValueError(
            f"camera_trajectory intrinsics must be [N, 4] with N = {poses.shape[0]}, got "
            f"{tuple(intrinsics.shape)}"
        )
    if not (torch.isfinite(poses).all() and torch.isfinite(intrinsics).all()):
        raise ValueError("camera_trajectory values must be finite")
    return CameraTrajectory(poses=poses, intrinsics=intrinsics)
