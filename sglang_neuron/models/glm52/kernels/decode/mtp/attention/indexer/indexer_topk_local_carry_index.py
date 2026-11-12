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
GLM-5.2 DSA indexer stage-1 LOCAL, carry-index: refused in the draft layer, whose chunked shard runs
the giga kernel's ``indexer_topk_local_carry_index_packed`` (the wide rounds); a sub-cap shard runs
``indexer_topk_local_packed``.
"""

_REFUSAL = (
    "the draft layer ranks a chunked shard with the giga kernel's indexer_topk_local_carry_index_packed and a "
    "sub-cap shard with indexer_topk_local_packed"
)


def indexer_topk_local_carry_index_packed(index_scores, topk: int, key_offset, shard_degree: int = 1,
                                          block_size: int = 0):
    """The carry-index local of the both-stages hand-off: refused."""
    raise ValueError(_REFUSAL)
