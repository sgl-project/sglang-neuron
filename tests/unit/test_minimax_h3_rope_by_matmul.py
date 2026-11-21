# SPDX-License-Identifier: Apache-2.0
"""CPU contract for the MiniMax-H3 DiT's ``rope_by_matmul`` option.

The partial rotate-half RoPE as ``x * cos' + (x @ R) * sin'`` must equal the slice/cat form bit
for bit, and the whole model must give the same outputs with it on.
"""

import pytest
import torch

pytest.importorskip("sglang.multimodal_gen")

from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: E402
    MiniMaxH3PrepareLayoutStep,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (  # noqa: E402
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
)

TINY = {
    "num_attention_heads": 2,
    "attention_head_dim": 128,
    "hidden_size": 256,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "ffn_dim": 512,
    "in_channels": 4,
    "audio_in_channels": 8,
    "text_dim": 64,
    "time_embed_hidden_dim": 128,
    "time_embed_dim": 64,
}


@pytest.fixture(scope="module")
def single_rank_groups():
    from tests.unit.minimax_h3_groups import single_rank_groups as _groups

    with _groups():
        yield


def _model(**overrides):
    from sglang_neuron.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Transformer3DModel,
    )

    torch.manual_seed(0)
    model = MiniMaxH3Transformer3DModel(**{**TINY, **overrides}).eval()
    for param in model.parameters():
        param.data.normal_(0, 0.05)
    return model


def _inputs():
    torch.manual_seed(1)
    num_text, frames, height, width, audio = 6, 2, 4, 4, 3
    position_ids, token_tags, video_indices, audio_indices, text_indices, _, _ = (
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            torch.full((num_text,), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
            frames,
            height,
            width,
            audio,
            (1, 2, 2),
            MINIMAX_H3_AUDIO_CHANNELS,
            MINIMAX_H3_AUDIO_TAG,
            MINIMAX_H3_VIDEO_TAG,
        )
    )
    latents = torch.randn(1, TINY["in_channels"], frames, height, width)
    return {
        "hidden_states": patchify_video_latents(latents, (1, 2, 2))[None],
        "audio_hidden_states": torch.randn(
            1, audio * MINIMAX_H3_AUDIO_CHANNELS, TINY["audio_in_channels"]
        ),
        "encoder_hidden_states": torch.randn(1, num_text, TINY["text_dim"]),
        "timestep": torch.tensor([0.5, 0.75]),
        "timestep_indices": (token_tags == MINIMAX_H3_AUDIO_TAG).long(),
        "token_tags": token_tags,
        "position_ids": position_ids.float(),
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
    }


def _run(model, inputs):
    with torch.no_grad():
        modulations = model.block_modulations(inputs["timestep"])
        return model(**inputs, block_modulations=modulations, return_dict=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_by_matmul_is_bit_exact(dtype):
    """``rope_by_matmul``: x * cos' + (x @ R) * sin' equals the slice/cat rotate-half RoPE."""
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

    from sglang_neuron.diffusion.models.minimax_h3.minimax_h3_transformer import (
        _rope_by_matmul,
        _rope_rotate_matrix,
    )

    torch.manual_seed(0)
    x = torch.randn(1, 37, 3, 128).to(dtype)
    cos, sin = torch.randn(37, 96), torch.randn(37, 96)
    expected = _apply_rotary_emb(x, cos, sin)
    actual = _rope_by_matmul(x, cos, sin, _rope_rotate_matrix(128, 96))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rope_by_matmul_model_matches(single_rank_groups):
    reference = _model()
    by_matmul = _model(rope_by_matmul=True)
    by_matmul.load_state_dict(reference.state_dict(), strict=False)
    inputs = _inputs()
    for expected, actual in zip(_run(reference, inputs), _run(by_matmul, inputs), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
