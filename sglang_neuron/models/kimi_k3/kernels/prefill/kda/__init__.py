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

"""Kimi-K3 prefill KDA attention-block kernel (one NKI call per KDA layer, collectives inside).

``kda_block_prefill.kimi_k3_kda_block_prefill`` is the kernel and ``kda_front_mx_torch`` the host MXFP8 packer of its
front weights. Import a module to get it; this package imports
nothing, so the host modules import without ``nki``.
"""
