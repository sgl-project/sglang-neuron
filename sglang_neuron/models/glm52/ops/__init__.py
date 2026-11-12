# SPDX-License-Identifier: Apache-2.0
"""The kernel calls of the GLM-5.2-family graphs: the PD prefill stage graph's prefill giga kernel wrapper
(prefill_kernel) and the device bitcast its fp8 key and MTP row stores need (bitcast), and the PD decode's
decode giga kernel and MTP draft chain wrappers (gk_decode, mtp_draft).

They live with the model, not in sglang_neuron.ops: that package imports every GPT-OSS kernel wrapper, which import the
image's nkilib, while these need only the released toolchain's nki and their vendored kernel copies
(sglang_neuron/models/glm52/kernels/prefill and decode). Nothing is imported here: each wrapper imports its
kernel copy, which reads its compile knobs, when it is imported, so a process imports it only once the launch's settings
are in its environment."""
