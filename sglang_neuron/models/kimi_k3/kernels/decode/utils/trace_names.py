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

"""Call counters for the instruction and tensor names of the Kimi-K3 kernels.

Names must be unique in a kernel, so a body that a kernel calls more than once numbers its calls. The counters belong
to the current trace: each kernel trace, and each core of an LNC=2 kernel, numbers its calls from 0. A kernel then gets
the same names in every process, whatever kernels the process traced before, so its NEFF and its compile-cache key do
not depend on the trace order.
"""

import itertools
import weakref

from nki.compiler._builtins import get_backend

_TRACE_COUNTERS = weakref.WeakKeyDictionary()
_UNTRACED_COUNTERS = {}


def _counters() -> dict:
    """Return the call counters of the current trace (outside a trace: one process-wide set)."""
    try:
        context = get_backend().get_current_context()
    except RuntimeError:
        return _UNTRACED_COUNTERS
    return _TRACE_COUNTERS.setdefault(context, {})


def next_call(kind: str) -> int:
    """Return the next call number of ``kind`` in the current trace: 0, 1, ..."""
    return next(_counters().setdefault(kind, itertools.count()))


def restart_calls():
    """Restart every call counter of the current trace (an entry that another entry's trace calls numbers anew)."""
    _counters().clear()
