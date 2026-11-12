# SPDX-License-Identifier: Apache-2.0
"""The compiled graphs' sizes: the segmented prefill kernel's supported KV segment tiles, and the default ladders of
decode batch sizes and prefill token counts below a launch's largest."""

from typing import List

# KV segment sizes supported by the segmented attention NKI kernel. Capped at 8192 by kernel SBUF capacity: a
# segment's KV tile has to be resident, so the ceiling is a hardware limit, not a tuning choice. The runtime takes the
# prefill token bucket as the tile (graph_runner.build_model_inputs: kv_segment = T), so
# graph_runner.check_prefill_tiles refuses a prefill bucket outside this set and the largest prefill chunk is capped at
# 8192 too.
SUPPORTED_KV_SEGMENT_SIZES = {512, 1024, 2048, 4096, 8192}


def batch_ladder(largest: int) -> List[int]:
    """The powers of two below ``largest`` and ``largest`` itself: a batch pads to the next size up, at most twice
    its own size, rather than to the largest."""
    sizes, size = [], 1
    while size < largest:
        sizes.append(size)
        size *= 2
    return sizes + [largest]
