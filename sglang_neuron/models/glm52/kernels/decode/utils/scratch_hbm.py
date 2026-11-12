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

"""HBM buffer class for a kernel's own intermediate (scratch) tensors.

At LNC=1 the two cores of an HBM stack run two programs. ``scratch_hbm()`` returns ``shared_hbm``, the class every
scratch tensor of the served build takes, so a collective destination identical on both ranks of a stack keeps one
address; the served points set no knob that changes it.


``layer_scratch_hbm()`` also returns ``private_hbm`` at LNC=1 with ``GLM52K_GLM52_PRIVATE_SCRATCH=1``, for the layer
intermediates no collective touches (the indexer scores and snake tap, the residency mask, the q_kv_up outputs): a
private scratch tensor is a scratchpad variable the compiler can overlap, where a shared one is a separate NEFF buffer
(84 MiB of indexer scores at 1M). Collective operands keep their buffer class, since a reused collective buffer is
the hazard ``cc_pin`` describes. The data is bit-identical; only addresses move.
"""

import os

import nki.language as nl


def scratch_hbm():
    """Buffer class for an HBM scratch tensor whose content is specific to this kernel instance.

    Returns:
        ``nl.shared_hbm``. ``GLM52K_GLM52_PRIVATE_SCRATCH`` does not reach it (see ``layer_scratch_hbm``).
    """
    return nl.shared_hbm


def layer_scratch_hbm():
    """Buffer class for a decode layer's large per-rank intermediate that no collective touches.

    Only upstream's private-scratch sites call it (see the module docstring); everything else calls
    ``scratch_hbm``.

    Returns:
        ``nl.private_hbm`` at LNC=1 when ``GLM52K_GLM52_PRIVATE_SCRATCH=1`` (a scratchpad virtual variable
        instead of a never-reused NEFF tmp-buf); ``scratch_hbm()`` otherwise, so the default is unchanged.
    """
    if os.environ.get("GLM52K_GLM52_PRIVATE_SCRATCH") == "1" and nl.num_programs(0) == 1:
        return nl.private_hbm
    return scratch_hbm()
