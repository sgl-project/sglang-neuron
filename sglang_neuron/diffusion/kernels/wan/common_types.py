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

"""The nkilib types of the Wan2.2 QKV kernel: the installed nkilib's, plus one norm type.

The installed nkilib has no ``NormType.RMS_NORM_ACROSS_HEADS`` (Wan2.2's across-heads Q/K RMSNorm
with its in-kernel TP all-reduce), so ``NormType`` and the ``QKNormConfig`` that refers to it are
vendored here unchanged. The other types are the installed nkilib's own objects, so the nkilib
helpers the kernel calls compare against the same enums.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import nki.language as nl
from nkilib.core.utils.common_types import (  # noqa: F401 (re-exported for the kernel)
    DtypeMode,
    QKVOutputLayout,
    QKVWeightLayout,
    QuantizationType,
    StridedInputConfig,
)


class NormType(Enum):
    NO_NORM = 0
    RMS_NORM = 1
    LAYER_NORM = 2
    RMS_NORM_SKIP_GAMMA = 3
    # RMSNorm whose sum-of-squares is reduced over the whole Q (or K) vector across
    # all heads on the rank, then all-reduced across the tensor-parallel replica group
    # (one RMS per token applied to every head). Matches Wan2.2's DistributedRMSNorm.
    # Requires replica_groups to be passed for the cross-rank reduce.
    RMS_NORM_ACROSS_HEADS = 4


@dataclass
class QKNormConfig(nl.NKIObject):
    """Configuration for per-head QK-norm on Q and K projections.

    Each head group independently specifies which norm to apply (or None to skip).

    Args:
        q_norm: Normalization type for Q heads, or None to skip Q norm.
            Only RMS_NORM is registered by default.
        k_norm: Normalization type for K heads, or None to skip K norm.
            Only RMS_NORM is registered by default.
        eps: Epsilon for numerical stability. Default matches norm_eps used throughout
            the QKV CTE codebase.
        q_gamma_norm_weights: [1, d_head] gamma weights for Q heads, or None for
            pure RMSNorm without affine scale.
        k_gamma_norm_weights: [1, d_head] gamma weights for K heads, or None for
            pure RMSNorm without affine scale.
        q_beta_norm_weights: [1, d_head] beta weights for Q heads (LayerNorm only).
            Not yet implemented.
        k_beta_norm_weights: [1, d_head] beta weights for K heads (LayerNorm only).
            Not yet implemented.
        gamma_fused_in_rope_caches: When True, gamma weights have been pre-multiplied
            into the cos/sin RoPE caches by the caller. The kernel fuses the rsqrt
            into the RoPE multiply instructions, eliminating the per-head gamma multiply.
            Requires fused_rope=True, q/k_gamma_norm_weights=None, and only valid on
            qk_norm_pre_rope (not qk_norm_post_rope). The sin cache must be [B, S, d_head]
            with sin_lo_fused in [:,:,0:d_half] and sin_hi_fused in [:,:,d_half:d_head].
    """

    q_norm: Optional[NormType] = NormType.RMS_NORM
    k_norm: Optional[NormType] = NormType.RMS_NORM
    eps: float = 1e-6
    q_gamma_norm_weights: Optional[nl.ndarray] = None
    k_gamma_norm_weights: Optional[nl.ndarray] = None
    q_beta_norm_weights: Optional[nl.ndarray] = None
    k_beta_norm_weights: Optional[nl.ndarray] = None
    gamma_fused_in_rope_caches: bool = False
