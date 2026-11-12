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


"""GLM-5.2 DSA indexer decode kernels, split at the collective boundaries."""

from .indexer_block_rows import indexer_block_rows
from .indexer_constants import (
    IndexerQProjShape,
    IndexerScoreShape,
    IndexerTopkShape,
    get_indexer_q_proj_shape,
    get_indexer_score_shape,
    get_indexer_topk_shape,
)
from .indexer_kv_score import indexer_kv_score
from .indexer_q_proj import indexer_q_proj
from .indexer_residency_mask import indexer_residency_mask
from .indexer_torch import (
    indexer_block_rows_np,
    indexer_global_positions_np,
    indexer_kv_score_torch_ref,
    indexer_local_index_np,
    indexer_q_proj_torch_ref,
    indexer_residency_keep_np,
    indexer_residency_mask_np,
    indexer_topk_local_torch_ref,
    indexer_topk_merge_torch_ref,
)

__all__ = [
    "IndexerQProjShape",
    "IndexerScoreShape",
    "IndexerTopkShape",
    "get_indexer_q_proj_shape",
    "get_indexer_score_shape",
    "get_indexer_topk_shape",
    "indexer_block_rows",
    "indexer_block_rows_np",
    "indexer_global_positions_np",
    "indexer_kv_score",
    "indexer_kv_score_torch_ref",
    "indexer_local_index_np",
    "indexer_q_proj",
    "indexer_q_proj_torch_ref",
    "indexer_residency_keep_np",
    "indexer_residency_mask",
    "indexer_residency_mask_np",
    "indexer_topk_local_torch_ref",
    "indexer_topk_merge_torch_ref",
]
