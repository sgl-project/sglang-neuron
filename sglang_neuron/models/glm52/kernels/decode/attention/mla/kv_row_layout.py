# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Byte layout of a packed FP8 latent-KV cache row. This module is the only place that defines it.

A row is one uint8 tensor rather than separate latent / scale / rope tensors because the batched
indirect gather requires it: with a 2-D ``vector_offset`` the source and destination must both be
contiguous and share a dtype, so every gathered field has to span the full width of its own tensor
and lands in SBUF in its HBM dtype. Separate tensors would also mean one indirect DMA per field.

    bytes [  0, 128)  k_pe       64 bfloat16, pre-divided by kv_scale
    bytes [128, 640)  kv_latent  512 float8_e4m3fn (latent / kv_scale), permuted, see latent_byte_offset

640 is a multiple of 32, so a row start is aligned for every view the kernel takes of it, and the
latent starts on a 2-byte boundary, which the bf16-view transpose needs.

There are no MX scales. ``nc_matmul_mx`` reads its scales quadrant-scattered on the contraction
partitions, while a gathered row puts them on the key partition; fixing that up needs a per-(token,
key-tile) transpose plus a quadrant scatter, thousands of extra ops per call. ``quantize_mx`` groups
8 partitions x 4 elements, which on a token-major tile spans eight tokens, so it cannot produce a
per-row scale at all. The cache therefore uses one per-tensor scale, a kernel argument that costs no
instructions. e4m3fn is a floating format, so its relative error does not depend on the scale and a
per-row scale would gain little.
"""

ROPE_BYTES = 128
"""Bytes the k_pe field occupies at the start of a row: qk_rope_dim(64) bfloat16 elements."""

LATENT_BYTES = 512
"""Bytes the kv_latent field occupies: kv_lora_rank(512) float8_e4m3fn elements."""

ROW_BYTES = ROPE_BYTES + LATENT_BYTES
"""Physical width of one packed cache row, in bytes."""

E4M3FN_MAX = 448.0
"""
Largest finite float8_e4m3fn magnitude. The write path clamps to it before casting.

The clamp is required: the simulator's fp32 -> fp8 cast is a plain numpy astype, which yields NaN
for an out-of-range value, and one NaN latent poisons a whole score row and softmax denominator.

It also keeps bf16 NaN/Inf patterns out of the packed row: a bf16 exponent of all ones needs a high
byte of 0x7F or 0xFF, which is exactly the e4m3fn NaN encoding the clamp excludes. bf16 subnormals
remain possible (whenever the exponent-bearing byte of a pair is fp8 +/-0, the partner dim's live
byte becomes the mantissa), which is why the key transpose must be bit-accurate, not merely
value-preserving.
"""


def latent_byte_offset(c: int) -> int:
    """
    Byte offset within a row of latent dimension ``c``.

    The kernel builds the feature-major latent by PE-transposing the gathered rows. A PE transpose
    moves 16-bit words, so the 512 fp8 bytes travel as 256 bf16 words in two 128-column tiles
    instead of four, halving the transposes and their PSUM evictions. Read in plain order, partition
    ``p`` of chunk ``ci = 2j + parity`` would then hold dim ``2 * (128 * j + p) + parity`` instead
    of the ``128 * ci + p`` the query layout expects. The permutation is absorbed here, at write
    time, where the quantize already touches every element, rather than in the query: that is a
    public contract, and a mismatch there is an in-range permutation that can pass a loose tolerance.

    With ``c = 256a + 128b + p`` for ``a, b in {0, 1}`` and ``p in [0, 128)``:

        byte = ROPE_BYTES + 256a + 2p + b

    a bijection from [0, 512) onto [ROPE_BYTES, ROW_BYTES). It is endianness-independent: writer and
    reader address the same bytes. Endianness only decides which byte of a pair holds the bf16
    exponent, which matters solely to the subnormal note in E4M3FN_MAX.
    """
    a, rem = divmod(c, 256)
    b, p = divmod(rem, 128)
    return ROPE_BYTES + 256 * a + 2 * p + b


def latent_write_slices() -> list[tuple[int, int]]:
    """
    The four ``(dst_byte_start, src_latent_start)`` pairs the write path strides over.

    Each pair covers 128 latent dims: bytes ``dst_start, dst_start + 2, ...`` receive dims
    ``src_start .. src_start + 127``. Derived from ``latent_byte_offset`` so the two cannot drift.
    """
    out = []
    for a in range(2):
        for b in range(2):
            src_start = 256 * a + 128 * b
            out.append((latent_byte_offset(src_start), src_start))
    return out
