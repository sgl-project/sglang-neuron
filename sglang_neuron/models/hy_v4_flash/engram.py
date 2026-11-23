# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash's Engram memory on attention TP (the reference's HYV4FlashEngramLayer).

**Hashing.** A token's n-grams are itself and the one or two tokens before it in its sequence (the pad id before the
sequence starts). Each token is first mapped through the layer's ``lookup_table`` (a normalized vocabulary), times
the layer's odd multiplier for its offset; the bigram's hash is the XOR of the first two products, the trigram's the
XOR of all three, each taken modulo the 8 hash heads' prime table sizes of its order, and offset into the layer's flat
table (the heads' tables concatenated, 2-gram heads first). The multipliers are drawn the reference's way, from NumPy's
default generator seeded ``engram_seed + 10007 x layer_id``, bounded by the compressed vocabulary (the lookup table's
largest id + 1).

The products are 63-bit and a NeuronCore's int64 arithmetic is not exact (measured: int64 multiply, XOR and remainder
come out wrong), while its int32 arithmetic is. So the hash runs in int32 on 7-bit chunks: at load each token's
product for each offset is split into ``CHUNKS`` 7-bit chunks on the host (``product_chunks``, exact int64 there); the
graph XORs the chunks (XOR is bitwise, so the chunks of the XOR are the XOR of the chunks) and takes each head's
modulus by Horner's rule from the top chunk, ``acc = (acc x 128 + chunk) mod p``, every value below 2^31.

**Earlier tokens.** A decode step and a prefill chunk see only their own tokens, so the graphs keep each KV slot's
token id (``token_history``, one int32 per pool slot, written at the step's slot mapping before it is read), and read
a row's previous tokens through its block table: a radix-cached prefix reads the ids its own forward wrote.

**Tables.** A layer's ~165M rows x 96 split by rows over the attention TP, about 1 GB a rank a layer as the
checkpoint's bf16 rows; under ``engram_fp8_embedding`` the reference reads each row through an FP8 round trip, so the
rows are kept as its FP8 bytes and a scale each (weights.engram_rows), half that. Each rank looks up the ids its shard
holds, zero elsewhere, and one all-reduce sums the ranks' rows (exactly: one rank holds each row). Everything after
the lookup is replicated: the per-head norms, the value and per-stream key projections, the gates."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from sglang_neuron.models.hy_v4_flash import ops, weights
from sglang_neuron.models.hy_v4_flash.config import Hy4FlashConfig
from sglang_neuron.models.hy_v4_flash.fp8 import Memo
from sglang_neuron.models.transformer.config import DTYPE
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader


def hash_multipliers(seed: int, layer_id: int, compressed_vocab: int, max_ngram: int) -> torch.Tensor:
    """The reference's ``draw_hash_multipliers``: odd int64 multipliers, one per n-gram offset."""
    half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab // 2)
    generator = np.random.default_rng(seed + 10007 * layer_id)
    drawn = generator.integers(0, half_bound, size=max_ngram, dtype=np.int64)
    return torch.tensor((drawn * 2 + 1).tolist(), dtype=torch.int64)


# 7-bit chunks of a 63-bit product, least significant first
CHUNK_BITS = 7
CHUNKS = 10


def product_chunks(lookup: torch.Tensor, multipliers: torch.Tensor, pad_id: int) -> torch.Tensor:
    """[V + 1, max_ngram, CHUNKS] int32: each token's compressed id times each offset's multiplier, in 7-bit chunks;
    row V (past the lookup table) is the pad id's, which ids beyond the table read."""
    comp = lookup.to("cpu", torch.int64)
    comp = torch.cat([comp, comp[pad_id:pad_id + 1]])
    prod = comp.unsqueeze(1) * multipliers.to("cpu", torch.int64).unsqueeze(0)
    shifts = torch.arange(CHUNKS, dtype=torch.int64) * CHUNK_BITS
    return ((prod.unsqueeze(-1) >> shifts) & ((1 << CHUNK_BITS) - 1)).to(torch.int32)


def token_slots(block_table: torch.Tensor, block_size: int, positions: torch.Tensor) -> torch.Tensor:
    """[R, ...] int64 KV slots of ``positions`` [R, ...] (clamped into the table) through each row's block table
    [R, blocks]."""
    pos = positions.to(torch.int64).clamp(min=0)
    if block_table.shape[0] == 1 and pos.shape[0] != 1:
        # one request's table (a prefill) serves every row
        return token_slots(block_table, block_size, pos.reshape(1, -1)).reshape(pos.shape)
    blocks = block_table.shape[1]
    col = torch.div(pos, block_size, rounding_mode="floor").clamp(max=blocks - 1)
    flat = col.reshape(col.shape[0], -1)
    page = torch.gather(block_table.to(torch.int64), 1, flat).reshape(col.shape).clamp(min=0)
    return page * block_size + pos % block_size


def ngram_tokens(token_history: torch.Tensor, input_ids: torch.Tensor, positions: torch.Tensor,
                 block_table: torch.Tensor, block_size: int, rows_table, max_ngram: int, pad_id: int) -> torch.Tensor:
    """[T, max_ngram] int64: each token, then the tokens 1 .. max_ngram - 1 before it in its sequence (pad before its
    start), read from the slot history (which holds this step's tokens already). ``rows_table`` [T] picks each token's
    row of ``block_table`` (a decode's row is its own; a prefill's one table serves every token)."""
    cols = [input_ids.to(torch.int64)]
    table = block_table if rows_table is None else block_table.index_select(0, rows_table)
    for k in range(1, max_ngram):
        prev = positions.to(torch.int64) - k
        slots = token_slots(table, block_size, prev.unsqueeze(1)).squeeze(1)
        tok = token_history.index_select(0, slots).to(torch.int64)
        cols.append(torch.where(prev >= 0, tok, torch.full_like(tok, pad_id)))
    return torch.stack(cols, dim=-1)


class Engram(nn.Module):
    """One Engram layer (reference engram id ``engram_id``, read before decoder layer ``engram_id - 1``)."""

    def __init__(self, config: Hy4FlashConfig, index: int):
        super().__init__()
        self.config = config
        self.engram_id = config.engram_ids[index]
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.attn_tp_group.world_size
        sizes = list(config.engram_head_sizes[index])
        self.num_heads = len(sizes)
        self.head_dim = config.engram_head_dim
        self.max_ngram = config.engram_max_ngram
        self.per_ngram = config.engram_heads_per_ngram
        total = sum(sizes)
        self.rows_per_rank = -(-total // tp)
        hidden, width, hc = config.hidden_size, self.num_heads * self.head_dim, config.hc_mult
        self.register_buffer("moduli", torch.tensor(sizes, dtype=torch.int64), persistent=False)
        self.register_buffer("offsets", torch.tensor([0] + list(np.cumsum(sizes[:-1])), dtype=torch.int64),
                             persistent=False)
        # filled from the loaded lookup table (bind_hash)
        self.register_buffer("multipliers", torch.ones(self.max_ngram, dtype=torch.int64), persistent=False)
        self.register_buffer("chunks", torch.zeros(config.engram_lookup_size + 1, self.max_ngram, CHUNKS,
                                                   dtype=torch.int32), persistent=False)
        self.lookup_table = nn.Parameter(torch.empty(config.engram_lookup_size, dtype=torch.int32), requires_grad=False)
        self.fp8 = bool(config.engram_fp8_embedding)
        if self.fp8:
            # engram_fp8_embedding: the reference reads each row through an FP8 round trip, so the rows are kept as
            # its e4m3fn bytes and a bf16 scale each (weights.fp8_rows): half the bf16 table's bytes, decoded through
            # a byte table, so e4m3fn values above Trn2's e4m3 range stay exact
            self.table = nn.Parameter(torch.empty(self.rows_per_rank, self.head_dim, dtype=torch.uint8),
                                      requires_grad=False)
            self.table_scale = nn.Parameter(torch.empty(self.rows_per_rank, 1, dtype=DTYPE), requires_grad=False)
            self.register_buffer("byte_values", weights.e4m3fn_values(), persistent=False)
        else:
            # the rows as the checkpoint stores them, bf16
            self.table = nn.Parameter(torch.empty(self.rows_per_rank, self.head_dim, dtype=DTYPE), requires_grad=False)
        self.head_norms = nn.Parameter(torch.empty(self.num_heads, self.head_dim, dtype=DTYPE), requires_grad=False)
        self.value_w = nn.Parameter(torch.empty(width, hidden, dtype=DTYPE), requires_grad=False)
        self.value_b = nn.Parameter(torch.empty(hidden, dtype=DTYPE), requires_grad=False)
        self.key_w = nn.Parameter(torch.empty(hc, width, hidden, dtype=DTYPE), requires_grad=False)
        self.key_b = nn.Parameter(torch.empty(hc, hidden, dtype=DTYPE), requires_grad=False)
        self.norm_key = nn.Parameter(torch.empty(hc, hidden, dtype=DTYPE), requires_grad=False)
        self.norm_query = nn.Parameter(torch.empty(hc, hidden, dtype=DTYPE), requires_grad=False)
        self.pre_gamma = nn.Parameter(torch.empty(hidden, dtype=DTYPE), requires_grad=False)
        self.hc_scales = nn.Parameter(torch.empty(hc, hidden, dtype=DTYPE), requires_grad=False)
        set_weight_loader(self.lookup_table, weights.vector(torch.int32))
        if self.fp8:
            memo = Memo()
            set_weight_loader(self.table, weights.engram_rows(self.rows_per_rank, "bytes", memo, self.engram_id))
            set_weight_loader(self.table_scale, weights.engram_rows(self.rows_per_rank, "scales", memo,
                                                                    self.engram_id))
        else:
            set_weight_loader(self.table, weights.engram_rows_bf16(self.rows_per_rank))
        set_weight_loader(self.head_norms, weights.stacked(DTYPE))
        set_weight_loader(self.value_w, weights.replicated())
        set_weight_loader(self.value_b, weights.vector(DTYPE))
        set_weight_loader(self.key_w, weights.stacked(DTYPE, transposed=True))
        for p in (self.key_b, self.norm_key, self.norm_query, self.hc_scales):
            set_weight_loader(p, weights.stacked(DTYPE))
        set_weight_loader(self.pre_gamma, weights.vector(DTYPE))

    def weight_mappings(self, src: str, dst: str) -> dict:
        e = f"{src}.engram_layer"
        hc = self.config.hc_mult
        table = f"{e}.multi_head_embedding.embedding.weight"
        m = {f"{dst}.table_scale": table} if self.fp8 else {}
        return {
            **m,
            f"{dst}.lookup_table": f"{e}.lookup_table",
            f"{dst}.table": table,
            f"{dst}.head_norms": [f"{e}.lookup_per_head_norms.{h}.weight" for h in range(self.num_heads)],
            f"{dst}.value_w": f"{e}.value_proj.weight",
            f"{dst}.value_b": f"{e}.value_proj.bias",
            f"{dst}.key_w": [f"{e}.key_projs.{i}.weight" for i in range(hc)],
            f"{dst}.key_b": [f"{e}.key_projs.{i}.bias" for i in range(hc)],
            f"{dst}.norm_key": [f"{e}.norm_key.{i}.weight" for i in range(hc)],
            f"{dst}.norm_query": [f"{e}.norm_query.{i}.weight" for i in range(hc)],
            f"{dst}.pre_gamma": f"{e}.unified_pre_gamma",
            f"{dst}.hc_scales": [f"{e}.unified_per_hc_scales.{i}" for i in range(hc)],
        }

    def bind_hash(self) -> None:
        """Draw the hash multipliers from the loaded lookup table (the reference's ``_init_weights`` order) and build
        the products' chunk table the graph hashes with."""
        lookup = self.lookup_table.detach().to("cpu")
        compressed = int(lookup.max()) + 1
        mult = hash_multipliers(self.config.engram_seed, self.engram_id, compressed, self.max_ngram)
        self.multipliers = mult.to(self.lookup_table.device)
        self.chunks = product_chunks(lookup, mult, self.config.engram_pad_id).to(self.lookup_table.device)

    def hash_ids(self, ngrams: torch.Tensor) -> torch.Tensor:
        """[T, heads] int32 rows of the flat table of ngrams [T, max_ngram] token ids, in exact int32 arithmetic
        (module docstring)."""
        size = self.chunks.shape[0] - 1
        tok = torch.where(ngrams < size, ngrams, torch.full_like(ngrams, size)).clamp(min=0).to(torch.int64)
        # each offset's chunks of its token: whole [max_ngram x CHUNKS] rows gathered, then the offset's columns of
        # the gathered rows (a column slice of the table itself would copy the table every step)
        flat = self.chunks.view(self.chunks.shape[0], -1)
        parts = [flat.index_select(0, tok[:, k])[:, k * CHUNKS:(k + 1) * CHUNKS] for k in range(self.max_ngram)]
        moduli = self.moduli.to(torch.int32)
        ids, mixed = [], parts[0]
        for n in range(2, self.max_ngram + 1):
            mixed = torch.bitwise_xor(mixed, parts[n - 1])
            lo = (n - 2) * self.per_ngram
            p = moduli[lo:lo + self.per_ngram].unsqueeze(0)
            acc = torch.zeros(tok.shape[0], self.per_ngram, dtype=torch.int32, device=tok.device)
            for j in range(CHUNKS - 1, -1, -1):
                acc = torch.remainder(acc * (1 << CHUNK_BITS) + mixed[:, j:j + 1], p)
            ids.append(acc)
        return torch.cat(ids, dim=-1) + self.offsets.to(torch.int32)

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """[T, heads, head_dim] bf16 table rows of ids [T, heads], gathered from the ranks' shards."""
        start = self.attn_tp_group.rank_in_group_tensor.reshape(()).to(torch.int32) * self.rows_per_rank
        local = ids - start
        here = (local >= 0) & (local < self.rows_per_rank)
        flat = torch.where(here, local, torch.zeros_like(local)).reshape(-1).to(torch.int64)
        if self.fp8:
            q = self.table.index_select(0, flat)
            rows = self.byte_values.index_select(0, q.reshape(-1).to(torch.int64)).view(q.shape)
            rows = rows * self.table_scale.index_select(0, flat)
        else:
            rows = self.table.index_select(0, flat)
        rows = rows.view(*ids.shape, self.head_dim) * here.unsqueeze(-1).to(rows.dtype)
        return neuron_collectives.all_reduce(self.attn_tp_group, rows)

    def forward(self, streams: torch.Tensor, ngrams: torch.Tensor) -> torch.Tensor:
        """streams [T, hc, H] plus this layer's Engram output (the reference adds it before the attention)."""
        emb = self.lookup(self.hash_ids(ngrams)).to(streams.dtype)
        emb = torch.stack([ops.torch_rms_norm(emb[:, h], self.head_norms[h]) for h in range(self.num_heads)], dim=1)
        emb = emb.flatten(1)
        hidden = streams.shape[-1]
        gates = []
        for i in range(streams.shape[1]):
            key = ops.torch_rms_norm(emb @ self.key_w[i] + self.key_b[i], self.norm_key[i])
            query = ops.torch_rms_norm(streams[:, i], self.norm_query[i])
            gate = (key * query).sum(dim=-1) * hidden ** -0.5
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gates.append(gate.sigmoid())
        gates = torch.stack(gates, dim=1).unsqueeze(-1)
        main = self.pre_gamma * ops.torch_rms_norm(emb @ self.value_w + self.value_b, None)
        branch = self.hc_scales.unsqueeze(0) * main.unsqueeze(1)
        return streams + (gates * main.unsqueeze(1) + branch)
