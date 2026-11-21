# SPDX-License-Identifier: Apache-2.0
"""CPU contract for the MiniMax-H3 DiT's single-device options.

``compile_per_block`` splits ``forward`` into a prologue, the blocks and an epilogue, and
``adaln_on_host`` computes ``block_modulations`` on the host from weights that stay there. In
eager mode on one rank, both must give the same outputs as the whole-model path, bit for bit.
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


def test_staged_forward_matches_the_whole_model(single_rank_groups):
    whole = _model()
    staged = _model(compile_per_block=True)
    staged.load_state_dict(whole.state_dict())
    # The stages without torch.compile: the same calls the compiled graphs trace.
    from sglang_neuron.diffusion.models.minimax_h3.minimax_h3_transformer import _Probed

    staged._stages = {
        "prologue": staged._prologue,
        "blocks": [_Probed(block) for block in staged.transformer_blocks],
        "epilogue": staged._epilogue,
    }
    inputs = _inputs()
    for expected, actual in zip(_run(whole, inputs), _run(staged, inputs), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_host_adaln_matches_the_device_graph_and_stays_on_the_host(single_rank_groups):
    reference = _model()
    host = _model(adaln_on_host=True)
    host.load_state_dict(reference.state_dict())
    timestep = torch.tensor([0.5, 0.75])
    with torch.no_grad():
        expected = reference.block_modulations(timestep)
        actual = host.block_modulations(timestep)
    assert len(actual) == len(expected)
    for e, a in zip(expected, actual, strict=True):
        assert a.device.type == "cpu"
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    inputs = _inputs()
    for e, a in zip(_run(reference, inputs), _run(host, inputs), strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_attention_heads_per_call_matches_one_call(single_rank_groups):
    """``attention_heads_per_call``: attention over head groups, concatenated, is the same
    attention (each head is independent)."""
    reference = _model()
    split = _model(attention_heads_per_call=1)
    split.load_state_dict(reference.state_dict(), strict=False)
    inputs = _inputs()
    for expected, actual in zip(_run(reference, inputs), _run(split, inputs), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
