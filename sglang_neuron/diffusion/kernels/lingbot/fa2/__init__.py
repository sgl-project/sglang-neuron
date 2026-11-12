# SPDX-License-Identifier: Apache-2.0
"""LingBot-World's paged FlashAttention-2 kernels.

The modules here are a copy of the NKI Library's nkilib/experimental/lingbot/fa2_trn2 (the
Trn2 FlashAttention-2 kernels) and nkilib/experimental/attention/flash_attention2_trn3 (the Trn3
FlashAttention-2 primitive), merged into one primitive with a per-generation engine plan. Tile views
come from the installed NKI Library's nkilib.experimental.neurotile (NKI Library 1.0.24838 or later).
"""
