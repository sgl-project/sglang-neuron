# SPDX-License-Identifier: Apache-2.0
"""SGLang's FLUX.3 DiT with its blocks on NKI kernels: the same module tree and checkpoint, the block weights
converted once after SGLang loads them (``NeuronFlux3Transformer.prepare``). Tensor parallel only."""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.distributed import get_sp_group, get_tp_group
from sglang.multimodal_gen.runtime.layers.linear import MergedColumnParallelLinear
from sglang.multimodal_gen.runtime.models.dits.flux3 import (
    TEXT_STREAM,
    Flux3Block,
    Flux3SegmentState,
    Flux3Transformer,
    _compute_dtype,
    rope_cos_sin,
    timestep_embedding,
)

from sglang_neuron.diffusion.compile_backend import compile_graph
from sglang_neuron.diffusion.distributed.parallel_state import get_cp_replica_groups
from sglang_neuron.parallel import collectives

QKV_MAX_WIDTH = 4096
O_PROJ_MAX_HEADS = 17
QK_NORM_EPS = 1e-6
JOINT_BLOCKS_PER_GRAPH = 7
RING_ATTENTION = True  # CP attention as the FLUX kernels' const-max ring (else one K/V all-gather per block)
CP_STREAMS = ("video",)  # streams whose mode blocks split their rows over the sequence-parallel group
SP_MIN_ROWS = 128  # a stream of fewer rows keeps its rows on every rank (all-reduce) instead of splitting them over TP
DIT_COMPILER_ARGS = (
    "--model-type=transformer",
    "--auto-cast=none",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
    "--internal-hlo2tensorizer-options=--hierarchy-splitter=false --layers-per-module=0",
)
PARAM_NAMES_MAPPING = {
    r"^dit\.(.*)$": r"\1",
    r"^(.*)\.q_proj\.(.*)$": (r"\1.qkv.\2", 0, 3),
    r"^(.*)\.k_proj\.(.*)$": (r"\1.qkv.\2", 1, 3),
    r"^(.*)\.v_proj\.(.*)$": (r"\1.qkv.\2", 2, 3),
}


class NeuronFlux3Block(Flux3Block):
    """SGLang's FLUX.3 block on NKI kernels: AdaLN, then per group of heads the QKV projection with QK RMSNorm and
    RoPE, attention and the output projection; the SwiGLU MLP and the gated residual in torch; one TP all-reduce.
    q, k, v and the SwiGLU input are two TP layers, ``qkv`` and ``mlp_in`` (as stored), which SGLang's rank-local TP
    loader shards correctly; its merged ``qkv_mlp`` of q, k, v and the fused gate | value it splits evenly."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, quant_config=None,
                 supported_attention_backends=None, prefix: str = ""):
        super().__init__(hidden_size, num_heads, mlp_ratio, quant_config=quant_config,
                         supported_attention_backends=supported_attention_backends, prefix=prefix)
        del self.qkv_mlp
        self.qkv = MergedColumnParallelLinear(hidden_size, [hidden_size] * 3, bias=False, gather_output=False,
                                              quant_config=quant_config, prefix=f"{prefix}.qkv")
        self.mlp_in = MergedColumnParallelLinear(hidden_size, [self.mlp_hidden_dim] * 2, bias=False,
                                                 gather_output=False, quant_config=quant_config,
                                                 prefix=f"{prefix}.mlp_in")

    def prepare(self) -> None:
        """Convert SGLang's loaded weights once into the kernels' layout and drop them.

        Each head's q and k rows and the QK-norm scales go to rotate-half channel order (even channels, then odd),
        which turns SGLang's interleaved RoPE pairs into the kernel's halves and leaves q.k unchanged. The heads are
        split into groups of ``g``, the most that fit one qkv call (``3 * g * head_dim <= QKV_MAX_WIDTH``) and one
        output projection (``g <= O_PROJ_MAX_HEADS``): ``qkv_weight`` ``[G, H, 3 * g * d]`` (q | k | v of the group),
        ``attn_out_weight`` ``[G, g * d, H]``, ``mlp_in_weight`` ``[H, 2 * mlp]`` (gate | value), ``mlp_out_weight``
        ``[mlp, H]``, ``q_scale`` / ``k_scale`` ``[1, d]``. With sequence parallelism it keeps the ring kernel's
        ``replica_groups``, every CP group of the plugin's layout (``parallel_state.get_cp_replica_groups``).
        """
        heads, d = self.local_heads, self.head_dim
        weight = self.qkv.weight
        perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)]).to(weight.device)
        q, k, v = weight.split(self.local_hidden)
        mlp_in = self.mlp_in.weight
        group = max(g for g in range(1, heads + 1)
                    if heads % g == 0 and 3 * g * d <= QKV_MAX_WIDTH and g <= O_PROJ_MAX_HEADS)
        qkv = torch.stack([q.unflatten(0, (heads, d))[:, perm], k.unflatten(0, (heads, d))[:, perm],
                           v.unflatten(0, (heads, d))])
        converted = {
            "qkv_weight": qkv.unflatten(1, (-1, group)).permute(1, 4, 0, 2, 3).flatten(2),
            "attn_out_weight": self.attn_out.weight.t().unflatten(0, (-1, group * d)),
            "mlp_in_weight": mlp_in.t(),
            "mlp_out_weight": self.mlp_out.weight.t(),
            "q_scale": self.norm.query_norm.scale[perm][None],
            "k_scale": self.norm.key_norm.scale[perm][None],
        }
        del self.qkv, self.mlp_in, self.attn_out, self.mlp_out, self.norm
        for name, tensor in converted.items():
            setattr(self, name, nn.Parameter(tensor.contiguous(), requires_grad=False))
        sp = get_sp_group().world_size
        self.cp_replica_groups = get_cp_replica_groups(self.tp_size, sp) if sp > 1 else None

    def forward(self, x: torch.Tensor, rope: torch.Tensor, mods: list, lengths: list[int] | None = None,
                sp=None) -> torch.Tensor:
        """``Flux3Block.forward``: ``run`` with the key length of SGLang's sequence shard ``sp``, if any."""
        return self.run(x, rope, mods, lengths, None if sp is None else sp.shard.orig_len)

    def run(self, x: torch.Tensor, rope: torch.Tensor, mods: list, lengths: list[int] | None = None,
            kv_len: int | None = None) -> torch.Tensor:
        """The block on the converted weights (``run_chain`` of this block alone)."""
        return run_chain([self], x, rope, mods, lengths, kv_len)

    def partial(self, modulated: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_len: int | None = None,
                shard_start: int | None = None) -> torch.Tensor:
        """This rank's attention and MLP output of the modulated rows, before the TP sum. ``cos`` and ``sin`` are the
        kernel's rotate-half tables. With ``kv_len``, the rows are this rank's sequence-parallel shard: the keys and
        values are gathered over the sequence-parallel group and cut to ``kv_len``, dropping SGLang's tail pad; given
        the shard's first row ``shard_start`` and ``RING_ATTENTION``, they stay local, the pad keys and values are
        zeroed and the ring kernel attends over every rank's."""
        ops = _kernels()
        qkvs = [ops.qkv_proj(modulated, self.qkv_weight[g], cos, sin, self.q_scale, self.k_scale, eps=QK_NORM_EPS)
                for g in range(self.qkv_weight.shape[0])]
        if kv_len is not None and shard_start is not None and RING_ATTENTION:
            return self._ring_partial(modulated, qkvs, kv_len, shard_start, ops)
        kvs = [qkv[..., qkv.shape[-1] // 3:] for qkv in qkvs]
        if kv_len is not None:
            kvs = [collectives.all_gather(get_sp_group(), kv.transpose(0, 1).contiguous(), dim=0)[:kv_len]
                   .transpose(0, 1) for kv in kvs]
        mlp_gate, mlp_value = torch.matmul(modulated, self.mlp_in_weight).chunk(2, dim=-1)
        out = torch.matmul(F.silu(mlp_gate) * mlp_value, self.mlp_out_weight)
        for g, (qkv, kv) in enumerate(zip(qkvs, kvs)):
            q = qkv[..., :qkv.shape[-1] // 3].unflatten(-1, (-1, self.head_dim)).transpose(1, 2)
            k, v = kv.unflatten(-1, (2, -1, self.head_dim)).permute(2, 0, 3, 1, 4).unbind(0)
            out = out + ops.o_proj(ops.attention(q, k, v, self.head_dim**-0.5), self.attn_out_weight[g])
        return out


    def _ring_partial(self, modulated, qkvs, kv_len, shard_start, ops):
        """``partial`` with the attention as the ring kernel over local token-major q, k, v (``[B, S, N, d]``)."""
        rows = modulated.shape[1]
        valid = ((torch.arange(rows, device=modulated.device) + shard_start) < kv_len).to(modulated.dtype)
        mlp_gate, mlp_value = torch.matmul(modulated, self.mlp_in_weight).chunk(2, dim=-1)
        out = torch.matmul(F.silu(mlp_gate) * mlp_value, self.mlp_out_weight)
        for g, qkv in enumerate(qkvs):
            q, k, v = qkv.unflatten(-1, (3, -1, self.head_dim)).unbind(2)
            k, v = k * valid[:, None, None], v * valid[:, None, None]
            attn = ops.ring_attention(q, k, v, self.head_dim**-0.5, kv_len, self.cp_replica_groups)
            out = out + ops.o_proj(attn, self.attn_out_weight[g])
        return out

    def sp_forward(self, x: torch.Tensor, n: torch.Tensor, mul: torch.Tensor, add: torch.Tensor, gate: torch.Tensor,
                   cos: torch.Tensor, sin: torch.Tensor, rows: int, cp: bool, shard_start: int):
        """The block on this rank's residual rows ``x`` and modulated rows ``n``, 1/tp of its
        sequence-parallel shard: ``n`` is gathered over TP for the projections, the output is reduce-scattered
        back, and one AdaLN launch adds the gated residual and modulates the next block's input. ``rows`` is the
        sequence's real length: without CP the gathered rows are cut to it, with CP the keys past it are dropped.
        ``shard_start`` is the shard's first row."""
        ops = _kernels()
        full = _tp_gather_rows(n)
        if not cp:
            full, cos, sin = full[:, :rows], cos[:, :rows], sin[:, :rows]
        out = self.partial(full, cos, sin, rows if cp else None, shard_start)
        padded = n.shape[1] * self.tp_size
        if out.shape[1] < padded:
            out = F.pad(out, (0, 0, 0, padded - out.shape[1]))
        return ops.adaln_modulate(_tp_scatter_rows(out), mul, add, eps=self.pre_norm.eps, residual=x, gate=gate)


def _kernels():
    """The block's NKI kernels (``kernels/flux3.py``), imported at first use: the module needs the image's nkilib."""
    from sglang_neuron.diffusion.kernels import flux3

    return flux3


def _tp_gather_rows(t: torch.Tensor) -> torch.Tensor:
    """``[1, S, H]`` rows of every TP rank, in rank order."""
    return collectives.all_gather(get_tp_group(), t[0], dim=0)[None]


def _tp_scatter_rows(t: torch.Tensor) -> torch.Tensor:
    """This TP rank's ``1/tp`` of the rows of the TP sum of ``t`` ``[1, S, H]``."""
    return collectives.reduce_scatter(get_tp_group(), t[0], dim=0)[None]


def rows_layout(rows: int, cp: bool) -> tuple[int, int, int, int, int]:
    """``(total, shard_start, shard, local_start, local)`` of a ``rows``-row sequence padded to a multiple of tp x cp
    rows: this rank's sequence-parallel shard (every row without CP) and its TP rank's rows of it."""
    tp, sp = get_tp_group(), get_sp_group()
    cp_size, cp_rank = (sp.world_size, sp.rank_in_group) if cp else (1, 0)
    total = -(-rows // (tp.world_size * cp_size)) * tp.world_size * cp_size
    shard = total // cp_size
    local = shard // tp.world_size
    return total, cp_rank * shard, shard, cp_rank * shard + tp.rank_in_group * local, local


def segment_rows(mods: list, lengths: list[int], start: int, count: int, total: int) -> tuple:
    """Rows ``[start, start + count)`` of the per-segment ``(shift, scale, gate)``; the pad rows past the last
    segment take its modulation."""
    bounds = [*lengths[:-1], lengths[-1] + total - sum(lengths)]
    pieces, pos = ([], [], []), 0
    for mod, length in zip(mods, bounds):
        lo, hi = max(pos, start), min(pos + length, start + count)
        if lo < hi:
            for piece, t in zip(pieces, mod):
                piece.append(t.expand(-1, hi - lo, -1))
        pos += length
    return tuple(torch.cat(piece, dim=1) for piece in pieces)


def rotate_half_tables(rope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """SGLang's ``rope`` ``[cos | sin]`` of the angles as the kernel's rotate-half ``[cos | cos]``, ``[sin | sin]``."""
    cos, sin = rope.chunk(2, dim=-1)
    return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)


def run_sp_chain(blocks: list, state: tuple, rope: torch.Tensor, mods: list, lengths: list[int], rows: int, cp: bool,
                 first: bool, last: bool):
    """``blocks`` with the rows split over TP: each rank holds 1/tp of its sequence-parallel shard. ``state`` is
    the full ``[1, rows, H]`` sequence for the first chain, else ``(x, n)`` of the previous one; the last chain
    gathers the rows back on every rank."""
    total, shard_start, shard, local_start, local = rows_layout(rows, cp)
    shift, scale, gate = segment_rows(mods, lengths, local_start, local, total)
    mul = 1 + scale
    padded_rope = torch.cat([rope, rope[:, -1:].expand(-1, total - rows, -1)], dim=1)
    cos, sin = rotate_half_tables(padded_rope[:, shard_start:shard_start + shard])
    ops = _kernels()
    if first:
        x = F.pad(state[0], (0, 0, 0, total - rows))[:, local_start:local_start + local]
        n = ops.adaln_modulate(x, mul, shift, eps=blocks[0].pre_norm.eps)
    else:
        x, n = state
    for block in blocks:
        x, n = block.sp_forward(x, n, mul, shift, gate, cos, sin, rows, cp, shard_start)
    if not last:
        return x, n
    full = _tp_gather_rows(x)
    if cp:
        full = collectives.all_gather(get_sp_group(), full[0], dim=0)[None]
    return full[:, :rows]

def run_chain(blocks: list, x: torch.Tensor, rope: torch.Tensor, mods: list, lengths: list[int] | None = None,
              kv_len: int | None = None) -> torch.Tensor:
    """``blocks`` in turn on the residual rows ``x``, all with the same per-segment modulations ``mods`` (one
    ``(shift, scale, gate)`` per segment of ``lengths``, or one for every row), whose expansion to rows is built once
    per chain. SGLang's ``rope`` ``[cos | sin]`` of
    the angles becomes the kernel's rotate-half tables ``[cos | cos]`` and ``[sin | sin]``."""
    if lengths is None:
        shift, scale, gate = mods[0]
    else:
        shift, scale, gate = (torch.cat([m[i].expand(-1, length, -1) for m, length in zip(mods, lengths)], dim=1)
                              for i in range(3))
    cos, sin = rope.chunk(2, dim=-1)
    cos, sin = torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)
    ops = _kernels()
    eps = blocks[0].pre_norm.eps
    modulated = ops.adaln_modulate(x, 1 + scale, shift, eps=eps)
    for i, block in enumerate(blocks):
        out = block.partial(modulated, cos, sin, kv_len)
        if block.tp_size > 1:
            out = collectives.all_reduce(get_tp_group(), out)
        x = x + out * gate
        if i + 1 < len(blocks):
            modulated = ops.adaln_modulate(x, 1 + scale, shift, eps=eps)
    return x


class NeuronFlux3Transformer(Flux3Transformer):
    """SGLang's FLUX.3 DiT built from ``NeuronFlux3Block``; ``prepare`` once after SGLang has loaded the weights."""

    param_names_mapping = PARAM_NAMES_MAPPING

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_names_mapping = PARAM_NAMES_MAPPING
        self._chains: dict = {}
        self._ropes: dict = {}

    def rope(self, ids: torch.Tensor) -> torch.Tensor:
        """``Flux3Transformer.rope`` in float64 on the host (the device has none), cached on the device by the ids'
        content. The Neuron stages pass the host ids they build, so a lookup copies nothing from the device."""
        host = ids if ids.device.type == "cpu" else ids.cpu()
        key = (tuple(host.shape), hashlib.sha1(host.contiguous().numpy().tobytes()).digest())
        table = self._ropes.get(key)
        if table is None:
            table = self._ropes[key] = rope_cos_sin(host, self.axes_dim, self.theta).to(next(self.parameters()).device)
        return table

    def _vector(self, timesteps: torch.Tensor) -> torch.Tensor:
        """``Flux3Transformer._timestep_vector`` with no vector input, without its autocast context (a compiled graph
        takes none)."""
        vec = self.time_in(timestep_embedding(timesteps.reshape(-1, 1)).to(_compute_dtype(self.time_in.in_layer)))
        vec = vec.to(self.dtype)
        if self.vector_in is not None:
            zeros = torch.zeros(vec.shape[0], self.vector_in.in_layer.in_features, device=vec.device, dtype=self.dtype)
            vec = vec + self.vector_in(zeros)[:, None, :]
        return vec

    def _stream_graph(self, name: str):
        """One stream's timestep vector, modulations, input embedding and mode blocks as one compiled graph. A stream of
        at least ``SP_MIN_ROWS`` rows splits its rows over TP (``run_sp_chain``), over the CP group too
        for ``CP_STREAMS``; a shorter one keeps its rows on every rank."""
        graph = self._chains.get(("stream", name))
        if graph is None:
            blocks = list(self.content_mode_blocks[name])

            def run(x, timesteps, rope):
                rows = x.shape[1]
                vec = self._vector(timesteps)
                early = self.early_stream_modulations[name](vec)
                hidden = self.emb_in[name](x.to(self.dtype))
                if rows < SP_MIN_ROWS:
                    hidden = run_chain(blocks, hidden, rope, [early])
                else:
                    cp = name in CP_STREAMS and get_sp_group().world_size > 1
                    hidden = run_sp_chain(blocks, (hidden,), rope, [tuple(early)], [rows], rows, cp, True, True)
                return (hidden, vec, *self.single_stream_modulations[name](vec))

            graph = self._chains[("stream", name)] = compile_graph(run, DIT_COMPILER_ARGS)
        return graph

    def encode_context(self, ctx, ctx_ids, timesteps=None, vector=None):
        """``Flux3Transformer.encode_context`` as one graph: the timestep vector, the text modulations, ``txt_in`` and
        the text mode blocks."""
        if vector is not None:
            raise NotImplementedError("NeuronFlux3Transformer.encode_context takes no vector input")
        if timesteps is None:
            timesteps = torch.zeros(ctx.shape[0]).to(ctx.device)
        rope = self.rope(ctx_ids)
        hidden, vec, *joint_mod = self._text_graph()(ctx, timesteps, rope)
        return Flux3SegmentState(name=TEXT_STREAM, hidden=hidden, rope=rope, vec=vec, joint_mod=tuple(joint_mod))

    def _text_graph(self):
        """The text stream's ``encode_context`` work as one compiled graph."""
        graph = self._chains.get(("text",))
        if graph is None:
            blocks = list(self.txt_mode_blocks)

            def run(ctx, timesteps, rope):
                vec = self._vector(timesteps)
                early = self.early_stream_modulations[TEXT_STREAM](vec)
                hidden = run_chain(blocks, self.txt_in(ctx.to(self.dtype)), rope, [early])
                return (hidden, vec, *self.single_stream_modulations[TEXT_STREAM](vec))

            graph = self._chains[("text",)] = compile_graph(run, DIT_COMPILER_ARGS)
        return graph

    def encode_stream(self, name, x, ids, timesteps, vector=None, rope=None):
        """``Flux3Transformer.encode_stream`` as one graph per stream (``_stream_graph``)."""
        if vector is not None:
            raise NotImplementedError("NeuronFlux3Transformer.encode_stream takes no vector input")
        rope = self.rope(ids) if rope is None else rope
        timesteps = torch.as_tensor(timesteps, dtype=torch.float32, device=x.device)
        hidden, vec, *joint_mod = self._stream_graph(name)(x, timesteps, rope)
        return Flux3SegmentState(name=name, hidden=hidden, rope=rope, vec=vec, joint_mod=tuple(joint_mod))

    def _joint(self, context, streams, heads: tuple = ()):
        """``Flux3Transformer._joint`` with the joint blocks as chains of ``JOINT_BLOCKS_PER_GRAPH``, each rank on
        1/tp of its sequence-parallel shard (``run_sp_chain``); the first chain joins the segments, and with ``heads``
        (``(segment, stream name)`` pairs) the last one applies those streams' output heads and returns them."""
        segments = [context, *streams]
        lengths = [s.length for s in segments]
        rows, cp = sum(lengths), get_sp_group().world_size > 1
        ropes = [s.rope for s in segments]
        flat = [t for s in segments for t in s.joint_mod]
        vecs = [segments[i].vec for i, _ in heads]
        state = tuple(s.hidden for s in segments)
        blocks = list(self.single_blocks)
        for i in range(0, len(blocks), JOINT_BLOCKS_PER_GRAPH):
            first, last = i == 0, i + JOINT_BLOCKS_PER_GRAPH >= len(blocks)
            chain = self._sp_chain(("joint", i, tuple(lengths), heads), blocks[i:i + JOINT_BLOCKS_PER_GRAPH], lengths,
                                   rows, cp, first, last, heads)
            state = chain(*state, *ropes, *flat, *(vecs if last else ()))
            state = state if isinstance(state, tuple) else (state,)
        if heads:
            return list(state)
        return list(torch.split(state[0], lengths, dim=1)[1:])

    def _sp_chain(self, key, blocks, lengths, rows, cp, first, last, heads=()):
        """``run_sp_chain`` of ``blocks`` as one compiled graph (cached by ``key``)."""
        chain = self._chains.get(key)
        if chain is None:
            count = len(lengths)

            def run(*args):
                taken = count if first else 2
                state, ropes = args[:taken], args[taken:taken + count]
                flat = args[taken + count:taken + 4 * count]
                vecs = args[taken + 4 * count:]
                mods = [tuple(flat[j:j + 3]) for j in range(0, len(flat), 3)]
                if first:
                    state = (torch.cat(state, dim=1),)
                out = run_sp_chain(list(blocks), state, torch.cat(ropes, dim=1), mods, list(lengths), rows, cp, first,
                                   last)
                if not (last and heads):
                    return out
                segments = torch.split(out, list(lengths), dim=1)
                return tuple(self.final_layer[name](segments[i], vec) for (i, name), vec in zip(heads, vecs))

            chain = self._chains[key] = compile_graph(run, DIT_COMPILER_ARGS)
        return chain

    def denoise(self, context, streams, targets):
        """``Flux3Transformer.denoise``: the joint blocks, then the output heads of ``targets`` inside the last joint
        graph."""
        heads = tuple((i + 1, s.name) for i, s in enumerate(streams) if s.name in targets)
        outputs = dict(zip((name for _, name in heads), self._joint(context, streams, heads)))
        return {name: outputs[name] for name in targets}

    def _make_block(self, prefix: str) -> NeuronFlux3Block:
        """``Flux3Transformer._make_block`` with the Neuron block."""
        return NeuronFlux3Block(self.hidden_size, self.num_attention_heads, self.arch.mlp_ratio,
                                quant_config=self.quant_config,
                                supported_attention_backends=self._supported_attention_backends, prefix=prefix)

    def prepare(self) -> None:
        """Convert every block's weights into the kernels' layout (``NeuronFlux3Block.prepare``)."""
        for block in [m for m in self.modules() if isinstance(m, NeuronFlux3Block)]:
            block.prepare()
        self._chains.clear()
        self._ropes.clear()
