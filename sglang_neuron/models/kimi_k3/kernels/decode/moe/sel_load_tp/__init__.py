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

"""Kimi-K3 TP128 selective-load routed-expert decode kernel."""

from .moe_sel_load_tp import (
    CHUNK_TOKENS,
    MAX_TOKENS,
    PICK_GATHER_MAX_TOKENS,
    SelLoadStart,
    build_down_group_mask,
    moe_sel_load_tp,
    moe_sel_load_tp_body,
    moe_sel_load_tp_finish,
    moe_sel_load_tp_start,
)
from .moe_tp_dense import moe_tp_dense, moe_tp_dense_body

__all__ = [
    "build_down_group_mask",
    "CHUNK_TOKENS",
    "MAX_TOKENS",
    "PICK_GATHER_MAX_TOKENS",
    "moe_sel_load_tp",
    "moe_sel_load_tp_body",
    "moe_sel_load_tp_finish",
    "moe_sel_load_tp_start",
    "SelLoadStart",
    "moe_tp_dense",
    "moe_tp_dense_body",
]
