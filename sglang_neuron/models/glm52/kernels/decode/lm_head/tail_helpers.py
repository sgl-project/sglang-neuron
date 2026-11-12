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

"""Program sharding and ceiling-division helpers shared by the NKI kernels."""

from typing import Optional, Tuple

import nki.language as nl


def get_program_sharding_info() -> Tuple[int, int, int]:
    """
    Get program sharding information for current execution.

    Retrieves grid dimensionality, number of programs, and program ID.

    Args:
        None

    Returns:
        Tuple[int, int, int]: (grid_ndim, n_prgs, prg_id)
            - grid_ndim: Number of dimensions in program grid
            - n_prgs: Total number of programs on axis 0
            - prg_id: Current program ID on axis 0

    Notes:
        - Returns (0, 1, 0) for non-SPMD execution
        - Used for multi-core sharding strategies

    Pseudocode:
        grid_ndim = program_ndim()
        if grid_ndim != 0:
            n_prgs = num_programs(axis=0)
            prg_id = program_id(axis=0)
        else:
            n_prgs = 1
            prg_id = 0
        return (grid_ndim, n_prgs, prg_id)
    """
    grid_ndim = nl.program_ndim()
    n_prgs, prg_id = (nl.num_programs(axes=0), nl.program_id(axis=0)) if grid_ndim != 0 else (1, 0)
    return grid_ndim, n_prgs, prg_id


def get_verified_program_sharding_info(
    kernel_name: str = "",
    allowed_ndims: Optional[Tuple[int, ...]] = None,
    max_sharding: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Get and optionally verify program sharding information.

    Retrieves sharding info and performs optional validation checks.

    Args:
        kernel_name (str): Name of kernel for error messages (optional).
        allowed_ndims (Optional[Tuple[int, ...]]): Allowed grid dimensions (optional).
        max_sharding (Optional[int]): Maximum sharding degree (optional).

    Returns:
        Tuple[int, int, int]: (grid_ndim, n_prgs, prg_id)

    Notes:
        - Currently performs minimal validation
        - Intended for future validation enhancements

    Pseudocode:
        grid_ndim, n_prgs, prg_id = get_program_sharding_info()
        # Optional validation checks
        return (grid_ndim, n_prgs, prg_id)
    """
    grid_ndim, n_prgs, prg_id = get_program_sharding_info()
    return grid_ndim, n_prgs, prg_id


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
