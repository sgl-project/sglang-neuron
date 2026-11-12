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

"""Unique debug names for tensors allocated more than once in one trace.

The tracer refuses a duplicate ``nl.ndarray(name=...)``, so a kernel that emits the same layer body several times in
one trace (the giga kernel) needs distinct names. A monotonic suffix does it without threading a per-layer tag
through every helper; nothing reads the names back, so a counter left over from an earlier trace is harmless, and
``hidden_normed_shard_7`` still reads well in a profile.
"""

_COUNTER = [0]


def unique_name(base: str) -> str:
    """Return ``base`` with a monotonic suffix, unique within (and across) traces.

    Args:
        base (str): Human-readable stem, e.g. ``"hidden_normed_shard"``.

    Returns:
        str: ``f"{base}_{n}"`` with ``n`` never repeating.
    """
    _COUNTER[0] += 1
    return f"{base}_{_COUNTER[0]}"


__all__ = ["unique_name"]
