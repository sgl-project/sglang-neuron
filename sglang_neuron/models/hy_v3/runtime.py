# SPDX-License-Identifier: Apache-2.0
"""Hy3's runtime: the platform runtime with its DCP decode contexts aligned for the decode kernel's block reads.

nkilib's attention_block_tkg reads a row's ``s_prior`` cache tokens in blocks of ``gcd(block, s_prior / 256)`` (two
LNC2 programs of P_MAX tokens each; attention_tkg_utils' block resize). The platform's DCP context gives each rank a
share of whole 256-token tiles, so a 262144-token context's share (33280 tokens) reads blocks of 2 tokens: every
decode step's KV comes in two-token DMAs. A share of whole ``LOCAL_ALIGN`` tokens reads blocks of 16."""

from sglang_neuron.distributed.decode_context_parallel import decode_context
from sglang_neuron.model_executor.graph_runner import NeuronModelRuntime

# tokens of a rank's decode context share: 16 x 256, which keeps 16-token blocks of the 128-token pages
LOCAL_ALIGN = 4096


def aligned_context(context: int, size: int, block: int, tile: int) -> int:
    """The platform's DCP decode context for ``context`` tokens, rounded up to whole LOCAL_ALIGN-token shares."""
    tokens = decode_context(context, size, block, tile)
    quantum = size * LOCAL_ALIGN
    return -(-tokens // quantum) * quantum


class HyV3Runtime(NeuronModelRuntime):
    def dcp_decode_context(self, context: int, size: int, block: int, tile: int) -> int:
        return aligned_context(context, size, block, tile)
