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

"""AttnRes --- Kimi-K3's cross-layer residual mixture kernels.

A softmax mixture over the current residual stream plus the snapshots banked at earlier layers. ``attn_res_cte.attn_res_cte``
mixes a prefill row block; ``attn_res_bank_scores.attn_res_bank_scores`` scores a prefill stage's bank snapshots once
for all the stage's calls, as the optional ``bank_scores`` of ``attn_res_cte``. The ``*_torch`` modules hold the torch
references. Import a module to get a kernel; this package imports nothing.
"""
