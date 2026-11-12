# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 prefill NKI kernels (MANIFEST.json, ../PROVENANCE.md).

The kernels sit in sub-packages by function (``attention/``, ``attn_res/``, ``kda/``, ``linear/``, ``moe/``,
``utils/``), with ``page_io`` at this package's root. The blockwise MoE keeps the library modules it was built against
under ``moe/`` (``moe/mlp/``, ``moe/quantization/``, ``moe/utils/``). The modules import each other by relative imports.
Nothing here imports the serving image's ``nkilib``.
"""
