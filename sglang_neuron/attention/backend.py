# SPDX-License-Identifier: Apache-2.0
"""The runtime's attention backend: the per-forward metadata the models read, built from SGLang's
ForwardBatch. The models run attention, KV write included, inside their compiled graphs, so the backend's own forward
methods are never called.

  block_table[b, j] = req_to_token[req_pool_idx_b, j * page_size] // page_size  (int32, -1 pad)
  slot_mapping[i]   = out_cache_loc[i]                                            (int64, 0 pad)
  cached_seq_len    = the prefill request's cached prefix length

Under --dcp-size D the allocator page is a virtual page of D blocks, one per DCP rank
(distributed.decode_context_parallel). page_size above is that virtual page, the block table is the same on every
rank, and slot_mapping is this rank's slot for each new token, or the reserved slot for a token another rank keeps.

The metadata is exact; the graph runner pads it to the compiled buckets. NeuronModel.forward and the graph runner call
``build_metadata`` themselves, so SGLang's ``init_forward_metadata`` hooks build nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


@dataclass
class NeuronForwardMetadata:
    is_prefill: bool
    batch_size: int  # real number of requests
    num_tokens: int  # real number of tokens in this forward
    block_table: torch.Tensor  # [bs, max_blocks] int32 (CPU), -1 padded
    slot_mapping: torch.Tensor  # [num_tokens] int64 (CPU)
    seq_lens: torch.Tensor  # [bs] int64 (CPU) total lens (prefix + new)
    positions: torch.Tensor  # [num_tokens] int64 (CPU)
    cached_seq_len: int = 0  # the prefill request's radix-cache prefix length (SGLang's extend_prefix_lens)


class NeuronAttnBackend(AttentionBackend):
    def __init__(self, model_runner):
        super().__init__()
        self.model_runner = model_runner
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        # this rank's DCP placement (None without DCP), on the full-attention pool (a hybrid linear-attention pool's
        # leaf); its virtual page is the allocator's page
        from sglang_neuron.mem_cache.memory_pool import full_attention_pool

        self.dcp = getattr(full_attention_pool(self.token_to_kv_pool), "dcp", None)
        self.page_size = model_runner.page_size if self.dcp is None else self.dcp.page
        # (pool tensor, its data_ptr, its numpy view): the block table's source (_req_to_token_np)
        self._r2t_view = None

    # SGLang's runners call these before a forward; the metadata's consumers build it
    # themselves (build_metadata), so building it here too would only repeat the host work.
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        return None

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        return None

    def init_forward_metadata_in_graph(self, forward_batch):  # nothing device-side
        return None

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        return None

    def get_cpu_graph_seq_len_fill_value(self) -> int:
        # SGLang's CPU-device paths (this platform's device type) use this name
        return self.get_cuda_graph_seq_len_fill_value()

    def get_cuda_graph_seq_len_fill_value(self):
        # padded (idle) rows: seq_len 1 so req_to_token[0, 0] (pad page) is read
        return 1

    def build_metadata(self, fb) -> NeuronForwardMetadata:
        """Exact (unpadded) metadata from a ForwardBatch or its replay view."""
        mode: ForwardMode = fb.forward_mode
        # A decode reaches the graph runner unpadded. On the eager path (extend, idle) DP attention's
        # prepare_mlp_sync_batch pads the token dimension, and for an idle or decode batch the batch dimension with
        # batch_size set to the padded size; it keeps the real sizes in _original_batch_size and _original_num_tokens,
        # the fields its post_forward_mlp_sync_batch restores from, and everything here is sliced back to them.
        bs = getattr(fb, "_original_batch_size", None)
        bs = int(bs) if bs is not None else int(fb.batch_size)
        n_tok = getattr(fb, "_original_num_tokens", None)
        n_tok = int(n_tok) if n_tok is not None else int(fb.out_cache_loc.shape[0])
        req_pool_indices = fb.req_pool_indices.to("cpu", torch.long)[:bs]
        seq_lens = (fb.seq_lens_cpu if getattr(fb, "seq_lens_cpu", None) is not None else fb.seq_lens).to(
            "cpu", torch.long
        )[:bs]
        out_cache_loc = fb.out_cache_loc.to("cpu", torch.long)[:n_tok]
        positions = fb.positions.to("cpu", torch.long)[:n_tok]
        is_prefill = mode.is_extend() if not mode.is_idle() else False

        if mode.is_idle() or bs == 0:
            return NeuronForwardMetadata(
                is_prefill=False, batch_size=0, num_tokens=0,
                block_table=torch.full((0, 1), -1, dtype=torch.int32),
                slot_mapping=torch.zeros((0,), dtype=torch.long),
                seq_lens=torch.zeros((0,), dtype=torch.long), positions=positions[:0],
            )

        cached = int(fb.extend_prefix_lens[:bs].max()) if is_prefill else 0
        if self.dcp is not None:
            out_cache_loc = self.dcp.local_slots(out_cache_loc)
        return NeuronForwardMetadata(
            is_prefill=is_prefill, batch_size=bs, num_tokens=int(out_cache_loc.shape[0]),
            block_table=self._block_table(req_pool_indices, seq_lens, bs), slot_mapping=out_cache_loc,
            seq_lens=seq_lens, positions=positions, cached_seq_len=cached,
        )

    def _block_table(self, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor, bs: int) -> torch.Tensor:
        """[bs, max_blocks] int32: each request's page representatives divided by the page, -1 past its own pages. Built
        in numpy over a cached view of req_to_token, since every rank runs it every step: a handful of numpy calls
        instead of a dozen small torch ops. ``req_pool_indices`` and ``seq_lens`` are CPU int64 tensors
        (build_metadata's ``.to("cpu", torch.long)``)."""
        page = self.page_size
        rows = req_pool_indices.numpy()
        lens = seq_lens.numpy()
        max_blocks = (int(lens.max()) + page - 1) // page
        # req_to_token is at least context-wide, so the strided slice has max_blocks columns. For an int32 slot, this
        # division and cast give what torch's int64 floor division and int32 cast would.
        table = (self._req_to_token_np()[rows, : max_blocks * page : page] // page).astype(np.int32, copy=False)
        if bs != 1:  # one request owns the longest length, so all its columns are live
            # column j is live while its first slot j * page lies inside the request (j < ceil(len / page))
            live = np.arange(0, max_blocks * page, page) < lens[:, None]
            table = np.where(live, table, np.int32(-1))
        return torch.from_numpy(table)

    def _req_to_token_np(self) -> np.ndarray:
        """req_to_token as a numpy view of its storage, so SGLang's in-place writes stay visible through it. SGLang
        allocates the table once; the view is taken again when the pool holds another tensor or its tensor moved to
        another storage."""
        r2t = self.req_to_token_pool.req_to_token
        view = self._r2t_view
        if view is None or view[0] is not r2t or view[1] != r2t.data_ptr():
            view = self._r2t_view = (r2t, r2t.data_ptr(), r2t.numpy())
        return view[2]

    # The model computes attention (and writes KV) itself. These are
    # only reached if a model uses SGLang's RadixAttention on this backend.
    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        raise NotImplementedError("NeuronAttnBackend: RadixAttention path not used; model runs fused NKI attention")

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        raise NotImplementedError("NeuronAttnBackend: RadixAttention path not used; model runs fused NKI attention")
