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

"""MX (Microscaling) quantization of a numpy array into the hardware's x4-packed formats."""

import neuron_dtypes as dt
import nki.language as nl
import numpy as np


def quantize_to_mx(data, out_x4_dtype):
    """Quantize numpy [P, F] to MX x4 format. P%8==0, F%4==0 required.

    Accepts any numeric dtype (bf16, fp16, fp32, etc.) and casts to float32
    internally. Uses numpy because dt.static_cast (the only available packing
    function for MX x4 dtypes) operates on numpy arrays.

    Supports nl.float8_e4m3fn_x4, nl.float8_e5m2_x4, and nl.float4_e2m1fn_x4.

    Args:
        data (np.ndarray): Input array of shape [P, F] in any numeric dtype.
        out_x4_dtype: Target MX x4 dtype (nl.dtype).

    Returns:
        (x4_packed numpy [P, F//4], scale uint8 numpy [P//8, F//4])

    Notes:
        Scale is returned in dense logical layout [P//8, F//4] (one row per
        8-row block), NOT in hardware quadrant layout [P, F//4].
    """
    # Cast to float32 if needed (exponent extraction requires IEEE 754 float32)
    data_f32 = dt.static_cast(data, np.float32) if data.dtype != np.float32 else data

    # Resolve dtype: numpy custom dtypes (.dtype from array) are numpy.dtype objects,
    # not nl.* objects. Convert via str(dtype) to match against known names.
    _STR_TO_NL_DTYPE = {
        'float8_e5m2_x4': nl.float8_e5m2_x4,
        'float8_e4m3fn_x4': nl.float8_e4m3fn_x4,
        'float4_e2m1fn_x4': nl.float4_e2m1fn_x4,
    }
    dtype_str = str(out_x4_dtype)
    if dtype_str in _STR_TO_NL_DTYPE:
        out_x4_dtype = _STR_TO_NL_DTYPE[dtype_str]

    # max exponent and max representable value per MX dtype
    # float8_e5m2:   max_exp=14, max_val=57344
    # float8_e4m3fn: max_exp=7,  max_val=448
    # float4_e2m1fn: max_exp=2,  max_val=6
    _MX_DTYPE_PARAMS = {
        nl.float8_e5m2_x4: (14, 57344.0),
        nl.float8_e4m3fn_x4: (7, 448.0),
        nl.float4_e2m1fn_x4: (2, 6.0),
    }

    if out_x4_dtype not in _MX_DTYPE_PARAMS:
        raise ValueError(f"Unsupported out_x4_dtype: {out_x4_dtype}. Must be one of {list(_MX_DTYPE_PARAMS.keys())}")

    max_exp, max_val = _MX_DTYPE_PARAMS[out_x4_dtype]

    P, F = data_f32.shape
    if P % 8 != 0 or F % 4 != 0:
        raise ValueError(f"Shape ({P}, {F}) must be divisible by (8, 4) for MX block quantization")

    exp_field = ((data_f32.view(np.uint32) >> 23) & 0xFF).astype(np.uint8)
    block_max_exp = exp_field.reshape(P // 8, 8, F // 4, 4).max(axis=(1, 3))
    scale_uint8 = (block_max_exp.astype(np.int32) - max_exp).clip(0, 255).astype(np.uint8)

    scale_factors = np.power(2.0, block_max_exp.astype(np.float64) - max_exp - 127)
    scale_expanded = np.repeat(np.repeat(scale_factors, 8, axis=0), 4, axis=1)
    clipped = np.clip(
        data_f32.astype(np.float64) / np.where(scale_expanded == 0, 1.0, scale_expanded), -max_val, max_val
    )

    return dt.static_cast(np.ascontiguousarray(clipped.astype(np.float32)), out_x4_dtype), scale_uint8
