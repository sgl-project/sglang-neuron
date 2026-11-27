# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini's role graph (prefill or decode): one tensor-parallel rank's looped decoder over the paged KV pool.

The arithmetic is ``ops.py`` (checked against Hugging Face); this module holds the rank's weights (``weights.py``),
reads and writes the caches, and returns the rank's vocabulary shard of the sampled rows' logits.

Caches. Every layer visit has its own KV layer: 24 layers plus the 12 loop-core layers twice more, 48 in all
(``Inkling2Config.cache_layers``), one KV head of 128 a rank, full-length (the 40 sliding-window visits read the blocks
of their window, ``ops.window_blocks``, and mask past it). The short convolutions' input windows (3 tokens a
convolution, 4 convolutions a cache layer, ``StateLayout``) are a record per request, held in ``state_layers`` more
KV layers: the record is read in the page of the request's last processed token and written in the page of the
step's last token, and a prefill chunk also writes it at the end of every page it fills. So every full page holds the
record of the state after its last token: a radix-cache hit on a prefix (whole pages) reads its last page's record,
and the cached prefix continues exactly as the cold run did.

A prefill step is one request's chunk; a decode step is one token a row.
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn

from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.mem_cache.kv_cache import KVSpec, LayerSpec
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, pad_to_shape, set_weight_loader

from . import ops
from . import weights as W
from .config import Inkling2Config

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16
F32 = torch.float32


def layer_name(i: int) -> str:
    return f"layers.{i}.self_attn"


def _nki(kernel):
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    return wrap_nki(kernel)[get_logical_nc_config()]


class StateLayout:
    """The conv windows of one request: per cache layer, keys ``[3, kv_ch]``, values ``[3, kv_ch]``, attention
    output ``[3, H]``, MoE output ``[3, H]`` (bf16, oldest first), cache layers in order. The record is cut into rows of
    one page's plane (``page * kv_heads * head_dim`` elements): row ``2 j`` in the K plane of state layer ``j``, row
    ``2 j + 1`` in its V plane."""

    def __init__(self, cfg: Inkling2Config, page: int):
        kv_ch, H = cfg.kv_heads_per_rank * cfg.head_dim, cfg.hidden_size
        self.widths = (kv_ch, kv_ch, H, H)
        self.per_layer = ops.CONV_WINDOW * sum(self.widths)
        self.total = cfg.cache_layers * self.per_layer
        self.page = page
        self.plane_block = page * cfg.kv_heads_per_rank * cfg.head_dim
        self.layers = -(-self.total // (2 * self.plane_block))

    def offset(self, c: int, slot: int) -> int:
        return c * self.per_layer + ops.CONV_WINDOW * sum(self.widths[:slot])


def kv_layers(cfg: Inkling2Config, page: int) -> int:
    """The KV layers the pool holds: the layer visits' and the state record's."""
    return cfg.cache_layers + StateLayout(cfg, page).layers


def slot_indices(slot_mapping: torch.Tensor, page: int, num_blocks: int):
    """(block, position) of every slot; a slot outside the pool (none should be) writes SGLang's reserved page 0."""
    s = slot_mapping.to(torch.int64)
    s = torch.where((s >= 0) & (s < num_blocks * page), s, torch.zeros_like(s))
    return torch.div(s, page, rounding_mode="floor"), s % page


class Layer(nn.Module):
    """One model layer's weights on this rank (names as ``ops.decoder_layer`` reads them)."""

    def __init__(self, cfg: Inkling2Config, L: int):
        super().__init__()
        H, d, hq, hkv = cfg.hidden_size, cfg.head_dim, cfg.q_heads_per_rank, cfg.kv_heads_per_rank
        E, I = cfg.num_experts, cfg.inter_per_rank
        P = lambda *shape, dtype=DTYPE: nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)  # noqa
        # every matrix [in, out] (ops.py)
        self.wq, self.q_scale = P(H, hq * d), P(hq, d)
        self.wk, self.wv = P(H, hkv * d), P(H, hkv * d)
        self.wo, self.wgate, self.wsink = P(hq * d, H), P(H, hq * d), P(H, hq)
        self.wr, self.r_scale, self.r_proj = P(H, hq * cfg.d_rel), P(hq, cfg.d_rel), P(cfg.d_rel, cfg.rel_extent)
        self.post_norm, self.mlp_query = P(H), P(1, H, dtype=F32)
        self.post_attn_conv, self.post_mlp_conv = P(H, ops.CONV_TAPS, dtype=F32), P(H, ops.CONV_TAPS, dtype=F32)
        self.router, self.router_bias = P(H, E, dtype=F32), P(E, dtype=F32)
        self.gate_up, self.down = P(E, H, 2 * I), P(E, I, H)
        self.full = cfg.is_full(L)
        if L > 0:
            self.attn_query = P(3 if self.full else 1, H, dtype=F32)
        if self.full:
            self.k_static, self.v_static = P(ops.CONV_TAPS, hkv * d), P(ops.CONV_TAPS, hkv * d)
            self.k_wproj, self.v_wproj = P(H, hkv * ops.CONV_TAPS), P(H, hkv * ops.CONV_TAPS)
            self.q_norm, self.k_norm, self.v_norm = P(H), P(H), P(H)
        else:
            self.k_conv = P(hkv * d, ops.CONV_TAPS, dtype=F32)
            self.v_conv = P(hkv * d, ops.CONV_TAPS, dtype=F32)
            self.in_norm = P(H)
        self.loop = cfg.loop_start <= L < cfg.loop_end
        if self.loop:
            for g in ("q", "k", "v", "mlp"):
                setattr(self, f"gate_{g}_down", P(2 * H, cfg.gate_rank))
                setattr(self, f"gate_{g}_forget", P(cfg.gate_rank, H, dtype=F32))
                setattr(self, f"gate_{g}_gain", P(cfg.gate_rank, H, dtype=F32))

    def params(self) -> dict:
        p = {n: t for n, t in self.named_parameters()}
        p.update(getattr(self, "fused", {}))
        if self.loop:
            for g in ("q", "k", "v", "mlp"):
                p[f"gate_{g}"] = (p[f"gate_{g}_down"], p[f"gate_{g}_forget"], p[f"gate_{g}_gain"])
        return p


class LMHead(nn.Module):
    """This rank's rows of the vocabulary: softcapped logits, with the padded rows past the unpadded vocabulary set to
    the lowest float so they are never sampled. ``gather_output`` (the host clears it for the device sampler) gathers
    the shards."""

    def __init__(self, cfg: Inkling2Config, tp_group):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.hidden_size, cfg.lm_rows_per_rank, dtype=DTYPE),
                                   requires_grad=False)
        self.register_buffer("vocab_mask", torch.zeros(cfg.lm_rows_per_rank, dtype=F32), persistent=False)
        self.cap = cfg.final_cap
        self.gather_output = True
        self.tp_group = tp_group

    def forward(self, h):
        logits = ops.softcap((h @ self.weight).float(), self.cap) + self.vocab_mask
        if self.gather_output:
            logits = neuron_collectives.all_gather(self.tp_group, logits, dim=1)
        return logits


class PagedIO(ops.LayerIO):
    """``ops.LayerIO`` over the paged KV pool for one step: ``B`` rows of ``S`` tokens. Every KV layer reads the same
    block table and slot mapping (one pool)."""

    def __init__(self, graph: "Inkling2Graph", md, positions, valid, reset, B: int, S: int):
        self.g, self.positions, self.B, self.S, self.valid = graph, positions, B, S, valid
        L = graph.state_layout
        page = L.page
        self.slot_mapping = md["slot_mapping"].reshape(-1)
        # padding entries of the block table are -1: read SGLang's reserved page 0 there (its positions are past the
        # request's length, so the causal mask drops them)
        self.table = md["block_table_tensor"][:B].to(torch.int64).clamp(min=0)
        # decode rows the runtime padded (slot 0, the reserved page's first slot): the NKI MoE skips their expert loads
        self.valid_rows = (self.slot_mapping.reshape(B, S)[:, 0] > 0) if S == 1 else None
        q0 = positions[:, 0]
        self.read_pos = (q0 - 1).clamp(min=0)
        self.write_pos = q0 + (valid - 1 if S > 1 else 0)
        rows = []
        for kc, vc in graph.state_caches:
            blk = self._block_of(self.read_pos)
            rows += [kc.index_select(0, blk).reshape(B, -1), vc.index_select(0, blk).reshape(B, -1)]
        rec = torch.cat(rows, dim=1)[:, :L.total]
        if reset is not None:
            rec = rec * reset.to(rec.dtype)
        self.rec = rec
        self.new = {}
        # a prefill chunk also leaves the record at the end of every page it fills (chunks and cached prefixes are
        # whole pages), so a later request whose prefix hits those pages reads it there
        self.fills = list(range(1, S // page + 1)) if S > 1 else []
        self.new_fill = {k: {} for k in self.fills}

    def _block_of(self, pos):
        """The page that holds position ``pos [B]`` in the request's table."""
        j = torch.div(pos, self.g.state_layout.page, rounding_mode="floor").clamp(0, self.table.shape[1] - 1)
        return self.table.gather(1, j.view(-1, 1)).view(-1)

    def kv(self, c, k, v, window=None):
        kc, vc = self.g.kv_caches[c]
        n, _, block, d = kc.shape
        blk, pos = slot_indices(self.slot_mapping, block, n)
        kc[blk, 0, pos] = k.transpose(1, 2).reshape(-1, d)
        vc[blk, 0, pos] = v.transpose(1, 2).reshape(-1, d)
        if window is None:
            blocks, first = self.table, torch.zeros(self.B, dtype=torch.int64, device=k.device)
        else:
            # a sliding layer reads only the blocks of its window (about S + window keys, not the whole context)
            blocks, first = ops.window_blocks(self.table, self.positions[:, 0], self.S, window, block)
        nr = blocks.shape[1]
        k_ctx = kc.index_select(0, blocks.reshape(-1)).view(self.B, 1, nr * block, d)
        v_ctx = vc.index_select(0, blocks.reshape(-1)).view(self.B, 1, nr * block, d)
        kv_pos = first.view(-1, 1) + torch.arange(nr * block, device=k.device).view(1, -1)
        return k_ctx, v_ctx, kv_pos

    def window(self, c, slot):
        L = self.g.state_layout
        o, w = L.offset(c, slot), L.widths[slot]
        return self.rec[:, o:o + ops.CONV_WINDOW * w].view(self.B, ops.CONV_WINDOW, w)

    def put_window(self, c, slot, seq):
        self.new[(c, slot)] = ops.next_window(seq, self.valid).to(DTYPE)
        for k in self.fills:
            self.new_fill[k][(c, slot)] = ops.next_window(seq, k * self.g.state_layout.page).to(DTYPE)

    def moe_decode(self, x, idx, w, gate_up, down):
        """The decode MoE: the NKI kernel (kernels/moe_decode_hw.py) for buckets of up to 2 rows unless
        INKLING2_NKI_MOE=0, else torch (an 8-row graph of the unrolled kernel took the compiler past the host's
        memory on the vLLM port)."""
        from sglang_neuron.utils.neuron_utils import can_run_kernel

        if self.g.moe_kernel and x.shape[0] <= 2 and can_run_kernel(x):
            from .kernels import moe_decode_hw as K2
            return K2.moe_decode_hw(_nki, x, idx, w, gate_up, down, self.valid_rows)
        return ops.moe_gather(x, idx, w, gate_up, down, self.g.config.gate_cap, self.g.config.up_cap)

    def all_reduce(self, x):
        # traceable collectives over SGLang's group (parallel/collectives.py), lowered into the graph
        return neuron_collectives.all_reduce(self.g.tp_group, x)

    def _write_record(self, windows, blk):
        L = self.g.state_layout
        parts = [windows[(c, s)].reshape(self.B, -1) for c in range(self.g.config.cache_layers) for s in range(4)]
        rec = torch.cat(parts, dim=1)
        rec = torch.cat([rec, rec.new_zeros(self.B, 2 * L.layers * L.plane_block - L.total)], dim=1)
        pb = L.plane_block
        for j, (kc, vc) in enumerate(self.g.state_caches):
            kc[blk] = rec[:, (2 * j) * pb:(2 * j + 1) * pb].view(self.B, *kc.shape[1:])
            vc[blk] = rec[:, (2 * j + 1) * pb:(2 * j + 2) * pb].view(self.B, *vc.shape[1:])

    def publish(self):
        """Write the new windows to the page of the step's last token, and in prefill also the windows at the end of
        each page the chunk filled (padding rows and pages the chunk did not fill go to the reserved page 0)."""
        page = self.g.state_layout.page
        for k in self.fills:
            end = self.positions[:, 0] + k * page - 1
            ok = (k * page <= self.valid) & (self.positions[:, 0] % page == 0)
            self._write_record(self.new_fill[k], torch.where(ok, self._block_of(end), torch.zeros_like(end)))
        self._write_record(self.new, self._block_of(self.write_pos))


class Inkling2Graph(nn.Module):
    """One role graph (``phase`` "prefill" or "decode") of the text decoder, over the KV pool's ``kv_layers`` layers."""

    def __init__(self, config: Inkling2Config, phase: str, page: int):
        super().__init__()
        if phase not in ("prefill", "decode"):
            raise ValueError(f"a role graph is a prefill or a decode, not {phase!r}")
        c = self.config = config
        self.phase = phase
        self.tp_group = groups.get_neuron_attn_tp_group()
        self.lm_head_tp_group = self.tp_group
        self.state_layout = StateLayout(c, page)
        # the contract's layers (one per config layer, the keys of attn_metadata): every KV layer reads the same block
        # table and slot mapping, so the graph reads the first; all full attention for the platform (the sliding
        # visits read their window blocks and mask past it themselves)
        self.layer_names = [layer_name(i) for i in range(c.num_layers)]
        self.sliding_windows = [None] * c.num_layers
        # the pool's KV layers (configure_kv_cache builds the pool at this count): the visits', then the record's
        self.kv_names = [f"kv.{i}" for i in range(kv_layers(c, page))]
        self.embed_tokens = VocabDimShardedEmbedding(c.vocab_size, c.hidden_size, dtype=DTYPE,
                                                     tp_group=self.tp_group.device_group)
        P = lambda *shape, dtype=DTYPE: nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)  # noqa
        self.embed_norm, self.norm = P(c.hidden_size), P(c.hidden_size)
        self.final_query = P(1, c.hidden_size, dtype=F32)
        self.message_query = P(c.loop_layers, 4, c.hidden_size, dtype=F32)
        self.layers = nn.ModuleList(Layer(c, L) for L in range(c.num_layers))
        self.lm_head = LMHead(c, self.tp_group)
        self.moe_kernel = os.environ.get("INKLING2_NKI_MOE", "1") != "0"
        self.kv_caches = None
        self.state_caches = None

    # -- KV cache --

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(layers=[LayerSpec(name=n) for n in self.kv_names])

    def bind_kv_cache(self, kv_caches: dict) -> None:
        c, L = self.config, self.state_layout
        caches = [tuple(kv_caches[n]) for n in self.kv_names]
        self.kv_caches, self.state_caches = caches[:c.cache_layers], caches[c.cache_layers:]
        k = self.kv_caches[0][0]
        if k.dim() != 4 or k.shape[1] != c.kv_heads_per_rank or k.shape[2] != L.page or k.shape[3] != c.head_dim:
            raise ValueError(f"Inkling-2-Mini reads KV pages [blocks, {c.kv_heads_per_rank}, {L.page}, {c.head_dim}]; "
                             f"the pool's are {tuple(k.shape)}")

    # -- forward --

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        c = self.config
        md = attn_metadata[self.layer_names[0]]
        prefill = self.phase == "prefill"
        positions = positions.to(torch.int64)
        T = input_ids.numel()
        h = self.embed_tokens(input_ids.reshape(-1), scatter_tokens=False, rank=rank)
        h = ops.rms_norm(h, self.embed_norm, c.eps, DTYPE)
        if prefill:
            B, S = 1, T
            # the chunk's real tokens: the runtime pads a chunk with slot 0 (the reserved page's first slot)
            valid = (md["slot_mapping"] > 0).sum().to(torch.int64).view(1)
            cached = torch.as_tensor(md["cached_seq_len"]).reshape(-1)[:1].to(positions.device)
            reset = (cached > 0).view(1, 1)
        else:
            B, S = T, 1
            valid, reset = 1, None
        io = PagedIO(self, md, positions.reshape(B, S), valid, reset, B, S)
        params = {"layers": [lay.params() for lay in self.layers], "message_query": self.message_query,
                  "final_query": self.final_query, "norm": self.norm}
        h = ops.text_model(params, c, h.view(B, S, -1), io, dense=prefill)
        io.publish()
        h = torch.index_select(h.reshape(B * S, -1), 0, sampling_positions)
        return self.lm_head(h)

    # -- weights --

    def _loaders_and_mappings(self, prefix: str = "model.llm.") -> dict:
        c = self.config
        d, hq, hkv, I = c.head_dim, c.q_heads_per_rank, c.kv_heads_per_rank, c.inter_per_rank
        f32 = lambda t: t.float()  # noqa: E731
        T_ = lambda t: t.t().contiguous()  # noqa: E731 (checkpoint [out, in] -> ops.py's [in, out])
        m = {}
        names = {id(p): n for n, p in self.named_parameters()}

        def add(param, keys, fn):
            set_weight_loader(param, SafetensorsWeightLoader(transform=fn))
            m[names[id(param)]] = keys

        rows_e = self.embed_tokens.vocab_size_per_rank
        add(self.embed_tokens.weight, f"{prefix}embed.weight",
            lambda s, rank: pad_to_shape(s[0][rank * rows_e:(rank + 1) * rows_e], (rows_e, c.hidden_size)))
        add(self.embed_norm, f"{prefix}embed_norm.weight", lambda s, rank: s[0][:])
        add(self.norm, f"{prefix}norm.weight", lambda s, rank: s[0][:])
        add(self.final_query, f"{prefix}attn_res.output_query", lambda s, rank: f32(s[0][:]).view(1, -1))
        add(self.message_query, [f"{prefix}middle_cycle_cross_pass_queries",
                                 f"{prefix}middle_cycle_cross_pass_kv_queries"],
            lambda s, rank: W.message_queries(s[0], s[1]))
        add(self.lm_head.weight, f"{prefix}unembed.weight",
            lambda s, rank: T_(W.lm_head(s[0], rank, c.lm_rows_per_rank, c.unpadded_vocab_size)))
        for L, lay in enumerate(self.layers):
            a = f"{prefix}layers.{L}.attn."
            g = f"{prefix}layers.{L}."
            add(lay.wq, a + "wq_du.weight", lambda s, rank: T_(W.rows(s[0], rank, hq * d)))
            add(lay.q_scale, a + "q_l2_clip.scale", lambda s, rank: W.rows(s[0], rank, hq))
            add(lay.wk, a + "wk_dv.weight", lambda s, rank: T_(W.rows(s[0], rank, hkv * d)))
            add(lay.wv, a + "wv_dv.weight", lambda s, rank: T_(W.rows(s[0], rank, hkv * d)))
            add(lay.wo, a + "wo_ud.weight", lambda s, rank: T_(W.cols(s[0], rank, hq * d)))
            add(lay.wgate, a + "gate_attn.weight", lambda s, rank: T_(W.rows(s[0], rank, hq * d)))
            add(lay.wsink, a + "attn_sink_proj.weight", lambda s, rank: T_(W.rows(s[0], rank, hq)))
            add(lay.wr, a + "wr_du.weight", lambda s, rank: T_(W.rows(s[0], rank, hq * c.d_rel)))
            add(lay.r_scale, a + "rel_l2_clip.scale", lambda s, rank: W.rows(s[0], rank, hq))
            add(lay.r_proj, a + "rel_logits_proj.proj", lambda s, rank: s[0][:])
            add(lay.post_norm, g + "mlp_norm.weight", lambda s, rank: s[0][:])
            add(lay.mlp_query, f"{prefix}attn_res.mlp_queries.{L}", lambda s, rank: f32(s[0][:]).view(1, -1))
            if L > 0:
                n = 3 if lay.full else 1
                add(lay.attn_query, f"{prefix}attn_res.attn_queries.{L - 1}",
                    lambda s, rank, n=n: f32(s[0][:]).view(n, -1))
            add(lay.post_attn_conv, g + "attn_sconv.weight", lambda s, rank: f32(s[0][:]).squeeze(1))
            add(lay.post_mlp_conv, g + "mlp_sconv.weight", lambda s, rank: f32(s[0][:]).squeeze(1))
            add(lay.router, g + "mlp.gate.weight", lambda s, rank: T_(f32(s[0][:])))
            add(lay.router_bias, g + "mlp.gate.bias", lambda s, rank: f32(s[0][:]))
            e = g + "mlp.experts."
            nv = L < 20  # NVFP4 layers carry amax and the outer denominator; MXFP8 ones do not
            keys13 = [e + "w13_weight", e + "w13_weight_scale"] + (
                [e + "w13_weight_amax", e + "w13_weight_outer_scale_denominator"] if nv else [])
            keys2 = [e + "w2_weight", e + "w2_weight_scale"] + (
                [e + "w2_weight_amax", e + "w2_weight_outer_scale_denominator"] if nv else [])
            add(lay.gate_up, keys13, lambda s, rank, nv=nv: W.gate_up(
                s[0], s[1], s[2] if nv else None, s[3] if nv else None, rank, I).transpose(1, 2).contiguous())
            add(lay.down, keys2, lambda s, rank, nv=nv: W.down(
                s[0], s[1], s[2] if nv else None, s[3] if nv else None, rank, I).transpose(1, 2).contiguous())
            if lay.full:
                for x in ("k", "v"):
                    add(getattr(lay, f"{x}_static"), a + f"{x}_sconv.static_w",
                        lambda s, rank: W.cols(s[0], rank, hkv * d))
                    add(getattr(lay, f"{x}_wproj"), a + f"{x}_sconv.weight_proj",
                        lambda s, rank: T_(W.rows(s[0], rank, hkv * ops.CONV_TAPS)))
                    add(getattr(lay, f"{x}_norm"), g + f"{x}_input_norm.weight", lambda s, rank: s[0][:])
                add(lay.q_norm, g + "q_input_norm.weight", lambda s, rank: s[0][:])
            else:
                for x in ("k", "v"):
                    add(getattr(lay, f"{x}_conv"), a + f"{x}_sconv.weight",
                        lambda s, rank: f32(W.rows(s[0], rank, hkv * d)).squeeze(1))
                add(lay.in_norm, g + "attn_norm.weight", lambda s, rank: s[0][:])
            if lay.loop:
                for ours, theirs in (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
                    cg = g + f"controllers.{theirs}."
                    add(getattr(lay, f"gate_{ours}_down"), cg + "down_weight", lambda s, rank: T_(s[0][:]))
                    add(getattr(lay, f"gate_{ours}_forget"), cg + "forget_weight", lambda s, rank: T_(f32(s[0][:])))
                    add(getattr(lay, f"gate_{ours}_gain"), cg + "gain_weight", lambda s, rank: T_(f32(s[0][:])))
        return m

    def hf_weight_mappings(self) -> dict:
        return self._loaders_and_mappings()

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        """Read this rank's share of every text-decoder tensor (``weights.py``). The vision tower, its projector and
        the speculator's weights are never named, so they are never read."""
        mappings = self._loaders_and_mappings()
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        rank, world = self.tp_group.rank_in_group, self.tp_group.world_size
        state = checkpoint.load_sharded_pipelined(rank, world, self, mappings, device).state_dict
        state = {k: (v.to(DTYPE) if v.dtype.is_floating_point and v.dtype not in (DTYPE, F32) else v)
                 for k, v in state.items()}
        missing = set(dict(self.named_parameters())) - set(state)
        if missing:
            raise ValueError(f"Inkling-2-Mini weights not loaded: {sorted(missing)[:8]} ({len(missing)} in all)")
        self.load_state_dict(state, strict=False, assign=True)
        self.finish_load(device)

    def finish_load(self, device) -> None:
        """After the parameters are on the device: the stacked projections (ops.fuse_projections, fewer and larger
        matmuls a visit) and the LM head's padded-row mask."""
        for lay in self.layers:
            f = ops.fuse_projections({n: t for n, t in lay.named_parameters()})
            lay.fused = {k: f[k] for k in ("w_qside", "qside_sizes", "w_kv", "gate_qkv", "kv_conv") if k in f}
        c = self.config
        rank = self.tp_group.rank_in_group
        mask = torch.zeros(c.lm_rows_per_rank, dtype=F32)
        mask[max(0, c.unpadded_vocab_size - rank * c.lm_rows_per_rank):] = torch.finfo(F32).min
        self.lm_head.vocab_mask = mask.to(device)
