# SPDX-License-Identifier: Apache-2.0
"""Wan2.2's NKI kernels.

- ``qkv``, ``qkv_cte``, ``qkv_cte_utils``: the BF16 fused QKV kernel (vendored; only the imports
  and comment text changed), and ``qkv_shim``, its torch-compatible entry point. ``common_types``
  adds the one norm type the installed nkilib lacks.
- ``ring_attention_const_max_fwd`` and ``attention_const_max``: the const-max ring attention of the
  CP self-attention, copies of FLUX.1-dev's without its d-major output (``tp_out``) or odd-head split.

The nkilib helpers they use are imported from the installed ``nkilib``; the tile library the
const-max kernel is built on is the vendored ``kernels/neurotile``.
"""
