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

"""Kimi-K3 MXFP4 routed-expert decode kernel."""

from .moe_routed_torch import moe_tkg_decode_torch_ref

# The package exports the kernel moe_tkg_decode of moe_routed under its own name.
from .moe_routed import (
    EXPERT_QUEUES,
    STATIC_ALT_TILE_FORMS,
    STATIC_ALT_TILE_PSUM_CONSUME,
    expert_loop_layer,
    moe_tkg_decode,
    moe_tkg_decode_blocks,
    moe_tkg_decode_per_expert,
    moe_tkg_decode_ragged,
    moe_tkg_decode_token_tiles,
    per_expert_scratch,
    static_expert_stream,
)

__all__ = [
    "EXPERT_QUEUES",
    "STATIC_ALT_TILE_FORMS",
    "STATIC_ALT_TILE_PSUM_CONSUME",
    "expert_loop_layer",
    "moe_tkg_decode",
    "moe_tkg_decode_blocks",
    "moe_tkg_decode_per_expert",
    "moe_tkg_decode_ragged",
    "moe_tkg_decode_token_tiles",
    "moe_tkg_decode_torch_ref",
    "per_expert_scratch",
    "static_expert_stream",
]
