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

"""Shared hardware, dtype and tiling constants for the MX projection sub-kernels."""

import nki.language as nl

# Hardware config
NUM_QUADRANTS_IN_SBUF = 4
SBUF_QUADRANT_SIZE = 32

# Trn3 quantization block shape for MXFP4/8
_q_height = 8
_q_width = 4

# MX general dtype mapping
MX_PACKED_DTYPES = [nl.float4_e2m1fn_x4, nl.float8_e4m3fn_x4, nl.float8_e5m2_x4]

# QMX config
SUPPORTED_QMX_INPUT_DTYPES = [nl.float16, nl.bfloat16]
MX_SCALE_DTYPE = nl.uint8
SCALE_P_ELEM_PER_QUADRANT = 4

# PACK64 gate/up scale layout: 16 scale rows per SBUF quadrant (at within-quadrant offsets
# {0, 4, 8, 12}) instead of 4, folding the 4_I factor out of the scale's free dim and into
# partitions. Per-DMA shape goes [4 P, wide] -> [16 P, wide/4] at identical bytes, which loads
# substantially faster and reserves 4x less SBUF per partition.
# ISA: mxmem1d_valid_scale_pidx requires scale_pidx % 4 == 0 and < 16, so exactly 4 slots are
# addressable -- exactly a factor of 4 may leave the free dim, making 64P the ceiling.
SCALE_PACK64_P_PER_QUADRANT = SCALE_P_ELEM_PER_QUADRANT * _q_width  # 16 rows per quadrant
SCALE_PACK64_N_PARTITIONS = SCALE_PACK64_P_PER_QUADRANT * NUM_QUADRANTS_IN_SBUF  # 64
# QuantizeMx can accept P in [32, 64, 96, 128], corresponding to unpacked P [128, 256, 384, 512]
ALLOWED_QMX_P_DIM = (32, 64, 96, 128)


def pad_to_valid_qmx_partitions(n: int) -> int:
    """Pad partition count to nearest valid quantize_mx size {32, 64, 96, 128}."""
    for valid in ALLOWED_QMX_P_DIM:
        if valid >= n:
            return valid
    return 128


# MMULMX config
MIN_MATMULT_MX_P_DIM = 8  # Must have at least 1x [8P, 4F] block
MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM = 512  # 128P x 4F

# Gate/up indices
GATE_FUSED_IDX, UP_FUSED_IDX = 0, 1

# Constants for projections (the new NKI FE will implicitly set values to -1 if we use nl API, so hardcode here)
_psum_fmax = 512  # psum max free dim (elts in fp32)
