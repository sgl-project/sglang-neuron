# SPDX-License-Identifier: Apache-2.0
"""Hy3 (tencent/Hy3-FP8) on Neuron: the shared dense transformer with Hy3's MoE in place of each MLP after the first
(a sigmoid router with a selection-only expert bias, top-k renormalized and scaled, and one shared expert), its experts
split over the EP ranks. The checkpoint's per-tensor FP8 weights are dequantized to bf16 as they load."""
