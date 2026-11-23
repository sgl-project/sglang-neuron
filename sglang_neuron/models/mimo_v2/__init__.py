# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 (XiaomiMiMo/MiMo-V2.6-Flash-RL, MiMo-V2.5, ``MiMoV2ForCausalLM``) on Neuron, the text decoder: 48 layers
of hybrid attention (9 full-attention layers, 39 sliding-window layers of window 128 with a learned sink per head;
asymmetric Q/K 192 / V 128 heads and a partial RoPE), a dense MLP in layer 0 and a 256-expert top-8 sigmoid MoE after
it. The checkpoint's 128 x 128 blockwise FP8 and MXFP4 expert weights are dequantized to bf16 as they load."""
