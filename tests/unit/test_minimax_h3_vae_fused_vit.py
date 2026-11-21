# SPDX-License-Identifier: Apache-2.0
"""CPU contract for ``vae_fused_vit``: the re-laid-out decoder (permuted partial RoPE, folded
residual scales, split SwiGLU) computes upstream's ViT decoder. On CPU the ``NF`` calls take their
torch paths, so this checks the weight transforms and the layout, not the kernels."""

import pytest
import torch

pytest.importorskip("sglang.multimodal_gen")

from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (  # noqa: E402
    MiniMaxH3VideoRotaryPosEmbed,
    MiniMaxH3VideoTransformerBlock,
)

from sglang_neuron.diffusion.distributed.autoencoders.autoencoder_kl_minimax_h3 import (  # noqa: E402
    NeuronMiniMaxH3VideoViTDecoder3d,
)
from sglang_neuron.diffusion.distributed.autoencoders.minimax_h3_vit_block import (  # noqa: E402
    FusedViTBlock,
    _channel_permutation,
    full_head_rope_tables,
    fuse_decoder,
)


def _randomize(module):
    torch.manual_seed(0)
    for param in module.parameters():
        param.data.normal_(0, 0.05)
    for name, param in module.named_parameters():
        if name.endswith(("scale1", "scale2")):
            param.data.uniform_(0.5, 1.5)


def test_channel_permutation_moves_rotary_pairs_to_the_half_split():
    perm = _channel_permutation(64, 24)
    assert sorted(perm.tolist()) == list(range(64))
    for j in range(24):  # pair (j, j + 24) lands at (j, j + 32)
        assert perm[j] == j and perm[j + 32] == j + 24


@pytest.mark.parametrize("seq_len", [37, 38])
def test_fused_block_matches_upstream(seq_len):
    block = MiniMaxH3VideoTransformerBlock(dim=256, heads=4, dim_head=64, ffn_mult=4).eval()
    _randomize(block)
    rope = MiniMaxH3VideoRotaryPosEmbed(48, theta=100.0)
    cos, sin = rope(torch.rand(1, seq_len, 3) * 2 - 1)
    x = torch.randn(1, seq_len, 256)
    with torch.no_grad():
        expected = block(x, (cos, sin))
        fused = FusedViTBlock(block)
        full_cos, full_sin = full_head_rope_tables(cos[0, :, 0], sin[0, :, 0], 64, 24)
        actual = fused(x, full_cos[None], full_sin[None])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_fused_decoder_matches_upstream():
    decoder = NeuronMiniMaxH3VideoViTDecoder3d(
        in_channels=24,
        out_channels=3,
        patch_size=16,
        patch_size_t=4,
        num_layers=2,
        num_attention_heads=4,
        attention_head_dim=64,
        num_register_tokens=4,
        ffn_mult=4,
        rope_theta=100.0,
        rope_dim_ratio=0.75,
    ).eval()
    _randomize(decoder)
    z = torch.randn(1, 24, 2, 3, 3)
    with torch.no_grad():
        expected = decoder(z)
        fuse_decoder(decoder)
        actual = decoder(z)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
