# SPDX-License-Identifier: Apache-2.0
"""Conversion of the Neuron pipelines' decoded images to SGLang's output format."""

import torch


def to_output_images(images: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` images in [-1, 1] -> float32 in [0, 1] (SGLang's output contract).

    Rank 0 runs this on one CPU thread (workers run single-threaded), so it works in place on
    one float32 copy: that is faster than an out-of-place conversion with
    ``torch.isfinite(images).all()``, and it gives the same values.
    The finiteness check sums the scaled values before the clamp: the sum is non-finite
    when any value is, and finite values overflow it only above about 1e30 in magnitude
    (float32 max / pixel count at 4096 x 4096), far outside a VAE's output range.
    """
    output = images.float()
    if output is images:
        output = output.clone()
    output.mul_(0.5).add_(0.5)
    if not torch.isfinite(output.sum()):
        raise ValueError("Neuron pipeline produced non-finite image values")
    return output.clamp_(0, 1)


def to_output_video(videos: torch.Tensor) -> torch.Tensor:
    """``[B, 3, T, H, W]`` videos in [-1, 1] -> float32 in [0, 1], SGLang's video output contract.

    The same conversion as ``to_output_images`` (in place on one float32 copy; the finiteness
    check sums the scaled values before the clamp), for the Wan2.2 pipelines.
    """
    if videos.dim() != 5:
        raise ValueError(f"expected [B, C, T, H, W] videos, got shape {tuple(videos.shape)}")
    output = videos.float()
    if output is videos:
        output = output.clone()
    output.mul_(0.5).add_(0.5)
    if not torch.isfinite(output.sum()):
        raise ValueError("Neuron pipeline produced non-finite video values")
    return output.clamp_(0, 1)


def to_golden_frames(
    videos: torch.Tensor, dtype: torch.dtype | None = torch.float32
) -> torch.Tensor:
    """The first video of ``[B, 3, T, H, W]`` (in [-1, 1]) as ``[T, 3, H, W]`` in [0, 1].

    The goldens' layout: ``(v * 0.5 + 0.5).clamp(0, 1)`` in the video's own dtype (bf16 from the
    VAE), then ``dtype`` (float32 by default; ``None`` keeps the video's dtype, which loses
    nothing), frames first. The SSIM gate compares these tensors, not mp4s.
    """
    if videos.dim() != 5:
        raise ValueError(f"expected [B, C, T, H, W] videos, got shape {tuple(videos.shape)}")
    frames = (videos[0] * 0.5 + 0.5).clamp(0, 1)
    if dtype is not None:
        frames = frames.to(dtype)
    return frames.permute(1, 0, 2, 3).contiguous()
