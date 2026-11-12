# SPDX-License-Identifier: Apache-2.0
"""The GLM MLA latent KV cache row that the GLM prefill giga-kernel stack writes and the GLM-5.2 decode giga-kernel
reads: 640 bytes, one scale for the whole cache (``STATIC_SCALE``), bf16 ``k_pe`` pre-divided by it, then the e4m3fn
latent in the decode kernel's permuted channel order. The prefill writes it directly, so a page moves from the prefill
node to the decode node without translation.
"""

from __future__ import annotations

#: The row's dequantization scale: the decode kernel's ``kv_scale`` (`gk_geometry.KV_SCALE` takes this value).
STATIC_SCALE = 2.0**-6
#: Bytes of one row.
STATIC_ROW_BYTES = 640
