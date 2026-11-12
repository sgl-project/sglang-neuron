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

"""Kimi-K3 decode NKI kernels and host layout builders (MANIFEST.json, ../PROVENANCE.md).

The model's decode kernels (the verify and draft programs and their layers) sit at this package's root, ``multinode/``
holds the two-node forms, and the sub-packages by function (``moe/``, ``norm/``, ``quantization/``, ``sampling/``,
``utils/``) hold the kernel library modules they call. The modules import each other by relative imports. Nothing here
imports the serving image's ``nkilib``.
"""
