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

"""Trace-time records of program forms that a compile option can make wrong.

A kernel cannot see the options that compile its trace. So a kernel records such a form while it traces
(``record_hazard``), and the code that compiles the trace reads the records (``take_hazards``) and decides. A record
adds no instruction, so it does not change the program.

The records are process-wide: clear them with ``take_hazards`` before a trace, and read them after it.
"""

REPEATED_TWO_BUFFER_EXPERT_CALLS = "repeated_two_buffer_expert_calls"
"""The most static expert-loop calls of ``moe_tkg_decode`` with two whole weight buffers in one layer of the trace
(``moe_routed.expert_loop_layer``). Each such call of a layer loads the same expert weights again."""
LOAD_MERGE_MIN_CALLS = 3
"""The fewest such calls in one layer that the load merging of the compiler pass ``dma_optimization_sb`` gave wrong
routed partials for on trn3, with the compiler's own SBUF allocation: from the third call on, a call can read the
weight buffer of an earlier call."""

_HAZARDS = {}


def record_hazard(name: str, count: int):
    """Record ``count`` for the hazard ``name``: the records keep the largest count since the last ``take_hazards``."""
    _HAZARDS[name] = max(_HAZARDS.get(name, 0), count)


def take_hazards() -> dict:
    """Return the records since the last call, and clear them."""
    hazards = dict(_HAZARDS)
    _HAZARDS.clear()
    return hazards
