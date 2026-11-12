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

"""Common type definitions (enums) shared across NKI Library kernels."""

from enum import Enum


class ActFnType(Enum):
    SiLU = 0
    GELU = 1
    GELU_Tanh_Approx = 2
    Swish = 3
    ReLU = 4


class ExpertAffinityScaleMode(Enum):
    """When to apply expert affinity scaling. POST_SCALE (after the down projection) is the only mode."""

    POST_SCALE = 1
