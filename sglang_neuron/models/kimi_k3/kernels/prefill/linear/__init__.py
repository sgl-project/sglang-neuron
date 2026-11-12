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

"""Kimi-K3 prefill MXFP8 linear (Trn3).

* ``mx_linear.prefill_mx_linear``: ``out = x @ W`` with an MXFP8 weight and an on-chip MXFP8 quantize of x.
* ``mx_head_linear.prefill_mx_head_linear``: ``out[h] = W[h]^T x[h]`` for G heads with K <= 128 (4 heads at
  once in 32-row MX tiles; the MLA absorb).

Import ``prefill_mx_linear`` or ``prefill_mx_head_linear`` to get a kernel. These modules need ``nki``.
"""
