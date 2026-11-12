# SPDX-License-Identifier: Apache-2.0
"""The pooled indexer's table names and the TP group the GK host modules (decode.model, decode.kernel_limits,
mtp.draft) take from outside the GK layout, in the plugin's own terms:

- ``CDP_TABLES``: the GK inputs of the chunk-dealt pool's per-step tables, which the modules check a pooled geometry's
  tables against: layout.idx_pool's tuple, the names of the giga-kernel arguments IdxChunkPool.tables and
  IdxChunkPoolTables return (all equal). check_kernel_copy
  refuses a copy whose giga-kernel lacks one of them, and the 1M point's chunk-dealt pool passes that set.
- ``get_tp_group``: the TP group the GK takes for its split's fence all-reduce between two kernel calls (the decode
  point's ``gk_split``, models/glm52/points.py) and the MTP draft for its step-0 reduce-scatter: SGLang's TP group
  through the plugin's functional collectives."""

from __future__ import annotations

from sglang_neuron.models.glm52.layout.decode.idx_pool import CDP_TABLES

__all__ = ("CDP_TABLES", "get_tp_group")


class _TpGroup:
    """The TP group handle the GK host modules call: collectives over SGLang's TP group, traced into the decode graph
    as torch.distributed's functional collectives (parallel.collectives), which neuronx-cc lowers as torch-level
    collectives. The GK split's fence all-reduces over it (neuronx-cc cuts the module into functions there), and the
    MTP draft reduce-scatters its step-0 embedding rows over it (mtp_draft)."""

    @staticmethod
    def _group():
        from sglang.srt.distributed.parallel_state import get_tp_group as sglang_tp_group

        return sglang_tp_group()

    def all_reduce(self, tensor):
        from sglang_neuron.parallel.collectives import all_reduce

        return all_reduce(self._group(), tensor)

    def reduce_scatter(self, tensor, dim: int = -1):
        from sglang_neuron.parallel.collectives import reduce_scatter

        return reduce_scatter(self._group(), tensor, dim)


def get_tp_group():
    """The TP group of the GK host modules (decode.model's split fence, mtp.draft's step-0 embedding)."""
    return _TpGroup()
