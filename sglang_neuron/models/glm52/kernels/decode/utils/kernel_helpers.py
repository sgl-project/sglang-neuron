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
General kernel helper functions and utilities for NKI kernels.

This module provides the small set of shared helpers used by the MoE TKG kernels:
activation function mapping, ceiling division and dtype sizes.
"""

import nki.language as nl

from .common_types import ActFnType
from .kernel_assert import kernel_assert


def get_nl_act_fn_from_type(act_fn: ActFnType):
    """
    Convert ActFnType enum to NKI language activation function.

    Maps activation function type enum to corresponding nl.* function.

    Args:
        act_fn (ActFnType): Activation function type enum.

    Returns:
        function: Corresponding NKI language activation function.

    Notes:
        - Supports SiLU, GELU, GELU_Tanh_Approx, Swish, and ReLU
        - Raises assertion error for unsupported types

    Pseudocode:
        if act_fn == SiLU:
            return nl.silu
        elif act_fn == GELU:
            return nl.gelu
        elif act_fn == GELU_Tanh_Approx:
            return nl.gelu_apprx_tanh
        elif act_fn == Swish:
            return nl.gelu_apprx_sigmoid
        elif act_fn == ReLU:
            return nl.relu
        else:
            raise error
    """
    kernel_assert(isinstance(act_fn, ActFnType), f"Unsupported activation function type: {act_fn}")
    if act_fn == ActFnType.SiLU:
        return nl.silu
    elif act_fn == ActFnType.GELU:
        return nl.gelu
    elif act_fn == ActFnType.GELU_Tanh_Approx:
        return nl.gelu_apprx_tanh
    elif act_fn == ActFnType.Swish:
        return nl.gelu_apprx_sigmoid
    elif act_fn == ActFnType.ReLU:
        return nl.relu


def div_ceil(n, d):
    """
    Compute ceiling division.

    Returns the smallest integer greater than or equal to n/d.

    Args:
        n (int): Numerator.
        d (int): Denominator.

    Returns:
        int: Ceiling of n divided by d.

    Notes:
        - Commonly used for tile count calculations

    Pseudocode:
        result = (n + d - 1) // d
        return result
    """
    return (n + d - 1) // d


def sizeinbytes(dtype):
    """Bytes per element of an NKI dtype; the x4 packed MX dtypes report their packed container (2 bytes for
    float4_e2m1fn_x4, 4 for the float8 x4 variants), not one logical element."""
    if str(dtype) == str(nl.float32):
        return 4
    elif (
        str(dtype) == str(nl.bfloat16)
        or str(dtype) == str(nl.float16)
        or str(dtype) == str(nl.uint16)
        or str(dtype) == str(nl.int16)
    ):
        return 2
    elif (
        str(dtype) == str(nl.int8)
        or str(dtype) == str(nl.uint8)
        or str(dtype) in ["float8e4", "float8_e4m3", "float8_e4m3fn", "float8e5", "float8_e5m2", "float8_e8m0fnu"]
    ):
        return 1
    elif str(dtype) == str(nl.int32) or str(dtype) == str(nl.uint32):
        return 4
    elif str(dtype) == str(nl.float4_e2m1fn_x4):
        return 2
    elif str(dtype) == str(nl.float8_e4m3fn_x4) or str(dtype) == str(nl.float8_e5m2_x4):
        return 4
    kernel_assert(False, f"dtype size unknown! {dtype}")
