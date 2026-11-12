# SPDX-License-Identifier: Apache-2.0
"""Host side of the prefill kernel (``sglang_neuron/models/glm52/kernels/prefill``) that every PP prefill
point runs: ``config`` (the ``GlmMoeDsaConfig`` the stage graphs are traced from), ``weight_loaders`` (checkpoint
tensors as the kernel's per-rank arguments) and ``pipeline`` (a stage's config and the top-k picks it exchanges with its
neighbours). Packing and the MTP prompt rows match the decode's, so ``layout.packing`` and ``layout.mtp_kv`` serve
both.
"""
