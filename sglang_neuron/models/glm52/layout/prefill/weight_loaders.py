# SPDX-License-Identifier: Apache-2.0
"""Weight loaders and HuggingFace key mappings for the GLM-MoE-DSA prefill giga kernel (GK).

The loaders put each checkpoint tensor in the kernel's layout on the host, at load time:

- The kernels read ``[in, out]`` and HuggingFace stores ``[out, in]``, so most loaders transpose. ``wqkv_a`` fuses
  ``q_a_proj`` with ``kv_a_proj_with_mqa``; ``wuk`` and ``wuv`` both split from ``kv_b_proj``.
- The latent axis of ``wqkv_a``, ``kv_norm``, ``wuk`` and ``wuv`` is in the channel order of the decode's static KV row
  (``glm_kv_row``), so the QKV kernel writes the latent contiguously; scores and context are sums over that axis and do
  not change. ``kv_norm`` also carries ``glm_kv_balance``'s power-of-two gain and ``wuk`` / ``wuv`` divide by it, as on
  the decode node, so the cache rows hold the latent in the decode's units.
- The attention up-projections are 1/world rank-major chunks that the kernel all-gathers; o_proj and the dense MLP are
  shards of the TP x CP groups.
- The MXFP4 experts are tiled for ``block_moe_mx`` with the MX scales packed; the shared expert uses the routed
  layout with one expert, and the router weight the fused norm-router's swizzle-H order.
- Each per-layer weight loads into the kernel's ``[L, ...]`` stack (``stacked_loader``).

``attach_causal_lm_loaders`` and ``causal_lm_hf_mappings`` load a whole model; a prefill stage reuses their per-layer
mappings and loaders, so it reads exactly the bytes a whole-model load gives.
"""

import logging
import os

import torch

from sglang_neuron.models.glm52.layout.glm_kv_balance import with_static_kv_balance
from sglang_neuron.models.glm52.layout.prefill.mx_tiles import swizzle_h_index, tile_gate_up_blocks, tile_gate_up_scale
from sglang_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from sglang_neuron.models.glm52.layout.prefill.mtp_kv import mtp_hf_mappings

logger = logging.getLogger(__name__)

# One x4-element holds 4 fp4 values. One E8M0 scale covers 32 input values.
_MX_TILE = 512  # input values of one matmul contraction tile
_PMAX = 128  # SBUF partitions, which is the x4-element count of one tile
_MX_GROUP = 32  # input values of one MX scale

# The attention weights that the kernel all-gathers from 1/world chunks.
CHUNKED_ATTN_WEIGHTS = ("wqkv_a", "wq_b", "wuk", "wuv")


def prj_omx() -> bool:
    """`GLM_GK_PRJ_OMX=1`: the o_proj row shard is stored MXFP8 (fp8x4 rows + e8m0 scales, `wo_mx_pack`) and the
    GK runs the head-TP V-up + o_proj on the MX PE path (`glm_oproj_mx`). Approximate (tolerance-gated), compile-keyed."""
    return os.environ.get("GLM_GK_PRJ_OMX", "0").strip() == "1"


def wo_mx_pack(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8-quantize a `[K, N]` o_proj row shard (K = heads * v_head_dim, a multiple of 512) for `nc_matmul_mx`.

    Returns `(q, s)`: `q` `[K / 4, N]` int32, the fp8x4 words of the moving operand (row `128 t + p`, byte `j` = fp8 of row
    `512 t + 128 j + p`, little-endian); `s` `[K / 32, N]` uint8 e8m0 (bias 127), row `16 t + g` = the scale of the hardware
    group `{512 t + 128 j + 8 g + r : j < 4, r < 8}` of each column. The scale is the smallest power of two that keeps the
    group's amax <= 448 (no element saturates), which any e8m0 encoding allows.
    """
    K, N = w.shape
    if K % _MX_TILE:
        raise ValueError(f"wo_mx_pack: the o_proj row shard [{K}, {N}] has {K} rows, not a multiple of {_MX_TILE}")
    nt = K // _MX_TILE
    g = w.float().reshape(
        nt, 4, 16, 8, N
    )  # [t, j, g, r, n] = W[512 t + 128 j + 8 g + r, n]
    amax = g.abs().amax(dim=(1, 3), keepdim=True)  # [t, 1, g, 1, n]
    _, e = torch.frexp(torch.where(amax > 0, amax, torch.ones_like(amax)))
    e = e.to(torch.int32) - 1 - 8  # floor(log2 amax) - 8
    e = e + (
        torch.where(amax > 0, amax, torch.ones_like(amax)) / torch.exp2(e.float())
        > 448.0
    ).to(torch.int32)
    e = e.clamp(-127, 127)
    q = (g / torch.exp2(e.float())).clamp(-448.0, 448.0)
    q = torch.where(amax > 0, q, torch.zeros_like(q)).to(torch.float8_e4m3fn)
    s = (
        torch.where(amax > 0, e + 127, torch.zeros_like(e))
        .to(torch.uint8)
        .reshape(nt * 16, N)
    )  # row 16 t + g
    qb = (
        q.reshape(nt, 4, _PMAX, N).permute(0, 2, 3, 1).contiguous().view(torch.uint8)
    )  # [t, p, n, j] bytes
    words = (
        qb[..., 0].to(torch.int64)
        | (qb[..., 1].to(torch.int64) << 8)
        | (qb[..., 2].to(torch.int64) << 16)
        | (qb[..., 3].to(torch.int64) << 24)
    )
    words = torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32)
    return words.reshape(nt * _PMAX, N).contiguous(), s.contiguous()


def _multi_group(world: int, wo_group: int) -> bool:
    """Whether the world holds more than one whole wo group (the layout the head-TP paths need)."""
    return world > wo_group and world % wo_group == 0


def hybrid_tp(world: int, wo_group: int) -> bool:
    """Head tensor parallel over the wo groups for the attention projections.

    Each rank stores a 1/(world / wo group) chunk of its wo position's slice of `wqkv_a`
    (a column block), `wq_b`, `wuk` and `wuv` (whole heads), so each layer gathers only
    that slice, over the ranks that share the position. The weights still take 1/world
    of each rank's memory. On when the world holds more than one whole wo group; otherwise
    each layer gathers the full attention weights over the world.
    """
    return _multi_group(world, wo_group)


#: The variable that keeps each rank's whole wo position slice of the attention weights resident (`attn_resident`);
#: unset or 0 (default) keeps the 1/(world / wo group) chunks.
ATTN_RESIDENT_ENV = "GLM_GK_ATTN_RESIDENT"


def attn_resident(world: int, wo_group: int) -> frozenset:
    """The attention weights each rank stores as its whole wo position slice (`GLM_GK_ATTN_RESIDENT`).

    With `hybrid_tp` a rank stores 1/(world / wo_group) of its position's slice and the kernel all-gathers the rest one
    layer ahead. A resident weight skips that gather at the cost of the slice's other bytes per rank (+1.18 GB at PP1
    for all four); the values are the same. `1`: all four (`CHUNKED_ATTN_WEIGHTS`); a comma list names weights; unset
    or `0`: none. Needs `hybrid_tp`.
    """
    raw = os.environ.get(ATTN_RESIDENT_ENV, "").strip()
    if raw in ("", "0"):
        return frozenset()
    names = (
        frozenset(CHUNKED_ATTN_WEIGHTS)
        if raw == "1"
        else frozenset(x.strip() for x in raw.split(",") if x.strip())
    )
    if not names <= frozenset(CHUNKED_ATTN_WEIGHTS):
        raise ValueError(
            f"{ATTN_RESIDENT_ENV}={raw!r}: use 1, 0 or a comma list of {CHUNKED_ATTN_WEIGHTS}."
        )
    if not hybrid_tp(world, wo_group):
        raise ValueError(
            f"{ATTN_RESIDENT_ENV}={raw!r} needs the hybrid head TP (more than one wo group)."
        )
    return names


def position_slice_shape(name: str, full: tuple, config, wo_group: int) -> tuple:
    """Shape of one wo position's slice of the full attention weight `name` (see `hybrid_tp`)."""
    rows, cols = full
    if name in ("wqkv_a", "wq_b", "wuk"):
        return (rows, cols // wo_group)
    return (rows // wo_group, cols)  # wuv: head-major rows


def _position_slice(
    name: str, w: torch.Tensor, p: int, config, wo_group: int
) -> torch.Tensor:
    """Wo position `p`'s slice of the full attention weight `name`.

    `wqkv_a`: the q block `[p * q / g, (p + 1) * q / g)` then the kv block of its kv columns,
    so the slices in position order rebuild each half. `wq_b`, `wuk`: the columns of heads
    `[p * n / g, (p + 1) * n / g)`. `wuv`: their rows.
    """
    g = wo_group
    if name == "wqkv_a":
        q = config.q_lora_rank
        qw, kw = q // g, (w.shape[1] - q) // g
        return torch.cat(
            [w[:, p * qw : (p + 1) * qw], w[:, q + p * kw : q + (p + 1) * kw]], dim=1
        )
    if name in ("wq_b", "wuk"):
        n = w.shape[1] // g
        return w[:, p * n : (p + 1) * n]
    n = w.shape[0] // g
    return w[p * n : (p + 1) * n]


def _position_chunk_loader(
    inner, name: str, config, world: int, wo_group: int
) -> SafetensorsWeightLoader:
    """Wrap `inner` to keep this rank's chunk of its wo position's slice (see `hybrid_tp`).

    The rank sits at position `rank % wo_group` of its wo group and holds the chunk
    `rank // wo_group` of the slice, so the ranks of a position gather it in rank order.
    """
    n_chunks = world // wo_group

    def chunk(w, rank):
        sl = _position_slice(name, w, rank % wo_group, config, wo_group).contiguous()
        shape = world_chunk_shape(sl.shape[0], sl.shape[1], n_chunks)
        n = shape[0] * shape[1]
        i = rank // wo_group
        return sl.reshape(-1)[i * n : (i + 1) * n].reshape(shape)

    return _wrapped_loader(inner, chunk)


def _position_slice_loader(
    inner, name: str, config, wo_group: int
) -> SafetensorsWeightLoader:
    """Wrap `inner` to keep this rank's WHOLE wo position slice (`attn_resident`).

    The chunks of `_position_chunk_loader` in group order are this slice, flattened.
    """

    def keep(w, rank):
        return _position_slice(name, w, rank % wo_group, config, wo_group)

    return _wrapped_loader(inner, keep)


# Ranks of one tensor-parallel shared-expert block by default (see `shared_tp`).
_SHARED_TP_DEFAULT = 4


def shared_tp(world: int, wo_group: int) -> int:
    """Run the shared expert tensor parallel over blocks of m consecutive ranks.

    Each rank holds the intermediate slice `rank % m` of the shared expert (1/m of its memory) and
    runs it for the tokens of its block; the MoE reduce-scatter sums the partials. The default is
    4 when the world holds more than one wo group, else 1 (the replicated expert).
    `GLM_GK_SHARED_TP=<m>` sets m; 1 keeps the replicated expert.
    """
    raw = os.environ.get("GLM_GK_SHARED_TP", "").strip()
    if not raw:
        return _SHARED_TP_DEFAULT if _multi_group(world, wo_group) else 1
    m = int(raw)
    if m < 1 or m & (m - 1) or world % m:
        raise ValueError(
            f"GLM_GK_SHARED_TP={m}: use a power of two that divides {world}."
        )
    return m


# Generic loaders
def _replicated_loader(dtype: torch.dtype | None = None):
    """Load a weight as it is. Cast it to `dtype` if `dtype` is set."""

    def transform(slices, rank):
        tensor = torch.as_tensor(slices[0][:])
        return tensor if dtype is None else tensor.to(dtype)

    return SafetensorsWeightLoader(transform=transform)


def _transpose_loader():
    """Transpose a HuggingFace `[out, in]` linear weight to the kernel `[in, out]`."""

    def transform(slices, rank):
        return torch.as_tensor(slices[0][:]).transpose(0, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _wrapped_loader(inner: SafetensorsWeightLoader, fn) -> SafetensorsWeightLoader:
    """Load through `inner`, then apply `fn` to the loaded tensor."""

    def transform(slices, rank):
        return fn(inner.load(slices, rank), rank).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def world_chunk_shape(rows: int, cols: int, world: int) -> tuple:
    """Shape of one rank's chunk of a row-major `[rows, cols]` weight split `world` ways.

    Whole rows when they divide, else one flat row of `rows * cols / world` elements.
    Either way, the chunks in rank order make the flat weight.
    """
    if rows % world == 0:
        return (rows // world, cols)
    if (rows * cols) % world:
        raise ValueError(f"world_chunk_shape: a [{rows}, {cols}] weight does not split {world} ways")
    return (1, rows * cols // world)


def _world_chunk_loader(inner, world: int) -> SafetensorsWeightLoader:
    """Wrap `inner` to keep only this rank's rank-major chunk (see `world_chunk_shape`)."""

    def chunk(w, rank):
        shape = world_chunk_shape(w.shape[0], w.shape[1], world)
        n = shape[0] * shape[1]
        return w.reshape(-1)[rank * n : (rank + 1) * n].reshape(shape)

    return _wrapped_loader(inner, chunk)


def stacked_loader(
    loaders: list, key_counts: list[int], shape: tuple, dtype: torch.dtype
) -> SafetensorsWeightLoader:
    """Load a `[n, *shape]` stack: layer i through `loaders[i]`, on its own slices.

    The slices are the per-layer key lists, concatenated in layer order;
    `key_counts[i]` is the length of the list of layer i. Each layer is written into
    one preallocated host buffer, so the host holds the stack once.
    """

    def transform(slices, rank):
        out = torch.empty((len(loaders), *shape), dtype=dtype, device="cpu")
        lo = 0
        for i, (loader, n) in enumerate(zip(loaders, key_counts)):
            w = loader.load(slices[lo : lo + n], rank)
            lo += n
            if tuple(w.shape) != shape:
                raise ValueError(
                    f"Stack layer {i} loaded {tuple(w.shape)}, expected {shape}."
                )
            out[i].copy_(w)
        return out

    return SafetensorsWeightLoader(transform=transform)


# Attention
def decode_row_latent_order(kv_lora_rank: int) -> torch.Tensor:
    """Model latent dim held by latent byte `k` of the decode kernel's static KV row.

    The decode kernel stores dim `c = 256a + 128b + p` (a, b in {0, 1}, p in [0, 128)) at
    byte `256a + 2p + b`. So byte `k` holds `256 (k // 256) + 128 (k % 2) + (k % 256) // 2`.
    """
    if kv_lora_rank != 512:
        raise ValueError(
            f"The decode KV row is defined for kv_lora_rank 512, got {kv_lora_rank}."
        )
    k = torch.arange(kv_lora_rank)
    return 256 * (k // 256) + 128 * (k % 2) + (k % 256) // 2


def _wqkv_a_fused_loader():
    """Fuse `q_a_proj` and `kv_a_proj_with_mqa`, then transpose to `[H, out]`.

    The two HF weights are `[q_lora, H]` and `[kv_lora + rope, H]`. Concatenate them
    on the output dim, then transpose, so that the hidden dim is the contraction dim.
    """

    def transform(slices, rank):
        q_a = torch.as_tensor(slices[0][:])  # [q_lora, H]
        kv_a = torch.as_tensor(slices[1][:])  # [kv_lora + rope, H]
        fused = torch.cat([q_a, kv_a], dim=0)  # [q_lora + kv_lora + rope, H]
        return fused.transpose(0, 1).contiguous()  # [H, q_lora + kv_lora + rope]

    return SafetensorsWeightLoader(transform=transform)


def _wuk_loader(num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank):
    """Extract the nope half of `kv_b_proj` as the absorption weight `[nope, H*L]`.

    HF `kv_b_proj.weight` is `[num_heads*(nope + v), kv_lora]`. Take the nope rows of
    each head, then lay them out so that head h owns the columns `[h*L, (h+1)*L)`.
    """

    def transform(slices, rank):
        w = torch.as_tensor(slices[0][:]).view(
            num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
        )
        nope = w[:, :qk_nope_head_dim, :]  # [H, nope, L]
        nope = nope.permute(1, 0, 2).reshape(qk_nope_head_dim, num_heads * kv_lora_rank)
        return nope.contiguous()  # [nope, H*L]

    return SafetensorsWeightLoader(transform=transform)


def _wuv_loader(num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank):
    """Extract the V half of `kv_b_proj` as the V-up weight `[H*L, d_v]`.

    HF `kv_b_proj.weight` is `[num_heads*(nope + v), kv_lora]`. Take the V rows of
    each head, transpose each head to `[L, d_v]`, then reshape head-major.
    """

    def transform(slices, rank):
        w = torch.as_tensor(slices[0][:]).view(
            num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
        )
        v = w[:, qk_nope_head_dim:, :]  # [H, d_v, L]
        v = v.transpose(1, 2).reshape(num_heads * kv_lora_rank, v_head_dim)
        return v.contiguous()  # [H*L, d_v]

    return SafetensorsWeightLoader(transform=transform)


def _latent_permutations(config) -> dict:
    """The decode-row permutation of each full attention weight that touches the latent axis."""
    L, q0 = config.kv_lora_rank, config.q_lora_rank
    n_heads, nope, v_dim = (
        config.num_attention_heads,
        config.qk_nope_head_dim,
        config.v_head_dim,
    )
    perm = decode_row_latent_order(L)
    return {
        "wqkv_a": lambda w, rank: torch.cat(
            [w[:, :q0], w[:, q0 : q0 + L].index_select(1, perm), w[:, q0 + L :]], dim=1
        ),
        "kv_norm_weight": lambda w, rank: w.index_select(0, perm),
        "wuk": lambda w, rank: w.reshape(nope, n_heads, L)
        .index_select(2, perm)
        .reshape(w.shape),
        "wuv": lambda w, rank: w.reshape(n_heads, L, v_dim)
        .index_select(1, perm)
        .reshape(w.shape),
    }


def _wo_row_shard_loader(num_shards: int) -> SafetensorsWeightLoader:
    """Load this rank's row shard of o_proj in the `[Nh * v_head_dim, H]` layout.

    The shard index is the rank's position in its group of `num_shards` consecutive
    ranks. The rows are head-major, so a shard holds whole heads.
    """

    def transform(slices, rank):
        w = torch.as_tensor(slices[0][:]).transpose(0, 1)  # [Nh * v_head_dim, H]
        shard = w.shape[0] // num_shards
        start = (rank % num_shards) * shard
        return w[start : start + shard].contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _weights_proj_t_loader(dtype: torch.dtype):
    """Load the indexer `weights_proj` TRANSPOSED, as `[hidden_size, n_heads]`, in `dtype`.

    The kernel takes this K-major layout, so it loads `weights_proj` with one strided copy
    instead of one DMA transpose for each 128-row tile.
    """

    def transform(slices, rank):
        w = torch.as_tensor(slices[0][:])  # [n_heads, hidden_size]
        return w.transpose(0, 1).contiguous().to(dtype)

    return SafetensorsWeightLoader(transform=transform)


def attach_attention_loaders(attn, config, wo_group_size: int) -> None:
    """Attach the loaders of one `GlmMoeDsaAttention` module and its indexer.

    `wqkv_a`, `wq_b`, `wuk` and `wuv` keep this rank's 1/world chunk of the full weight,
    after the decode-row permutation under the static KV row. o_proj keeps the row shard
    of the rank's position in its wo group.
    """
    shape = (
        config.num_attention_heads,
        config.qk_nope_head_dim,
        config.v_head_dim,
        config.kv_lora_rank,
    )
    full = {
        "wqkv_a": _wqkv_a_fused_loader(),
        "wq_b": _transpose_loader(),
        "wuk": _wuk_loader(*shape),
        "wuv": _wuv_loader(*shape),
        "kv_norm_weight": _replicated_loader(),
    }
    for name, fn in _latent_permutations(config).items():
        full[name] = _wrapped_loader(full[name], fn)
    resident = attn_resident(attn.world_size, wo_group_size)
    for name in CHUNKED_ATTN_WEIGHTS:
        if name in resident:
            full[name] = _position_slice_loader(full[name], name, config, wo_group_size)
            continue
        full[name] = (
            _position_chunk_loader(
                full[name], name, config, attn.world_size, wo_group_size
            )
            if hybrid_tp(attn.world_size, wo_group_size)
            else _world_chunk_loader(full[name], attn.world_size)
        )
    # The latent gain of the static row, the same as on the decode node (`glm_kv_balance`).
    # The gain multiplies `kv_norm`, and the latent projections divide by it.
    full["kv_norm_weight"] = with_static_kv_balance(full["kv_norm_weight"])
    for name in ("wuk", "wuv"):
        full[name] = with_static_kv_balance(full[name], projection=True)
    for name, loader in full.items():
        set_weight_loader(getattr(attn, name), loader)
    set_weight_loader(attn.q_norm_weight, _replicated_loader())
    wo_loader = _wo_row_shard_loader(wo_group_size)
    if getattr(attn, "wo_scale", None) is not None:
        # GLM_GK_PRJ_OMX: the MXFP8 words and their scales, both from the bf16 row shard.
        set_weight_loader(
            attn.wo, _wrapped_loader(wo_loader, lambda w, rank: wo_mx_pack(w)[0])
        )
        set_weight_loader(
            attn.wo_scale, _wrapped_loader(wo_loader, lambda w, rank: wo_mx_pack(w)[1])
        )
    else:
        set_weight_loader(attn.wo, wo_loader)

    indexer = attn.indexer
    if indexer is not None:
        set_weight_loader(indexer.wq_b, _transpose_loader())
        set_weight_loader(indexer.wk, _transpose_loader())
        set_weight_loader(
            indexer.weights_proj, _weights_proj_t_loader(indexer.weights_proj.dtype)
        )
        # The checkpoint stores the LayerNorm statistics in bf16; the kernel takes fp32.
        for param in (indexer.k_norm_weight, indexer.k_norm_bias):
            set_weight_loader(param, _replicated_loader(param.dtype))


def attention_hf_mappings(
    config, layer_idx: int, param_prefix: str = "", ckpt_idx: int | None = None
) -> dict:
    """Return `{param name: HF key}` for one attention layer and its indexer.

    `param_prefix` prefixes the model-side names. A "shared" indexer layer reuses the
    top-k of a previous full layer, so it holds no indexer weight. `ckpt_idx` is the
    checkpoint layer of model layer `layer_idx` (default: the same index).
    """
    a = f"model.layers.{layer_idx if ckpt_idx is None else ckpt_idx}.self_attn"
    mappings = {
        f"{param_prefix}wqkv_a": [
            f"{a}.q_a_proj.weight",
            f"{a}.kv_a_proj_with_mqa.weight",
        ],
        f"{param_prefix}q_norm_weight": f"{a}.q_a_layernorm.weight",
        f"{param_prefix}wq_b": f"{a}.q_b_proj.weight",
        f"{param_prefix}kv_norm_weight": f"{a}.kv_a_layernorm.weight",
        f"{param_prefix}wo": f"{a}.o_proj.weight",
    }
    if prj_omx():
        mappings[f"{param_prefix}wo_scale"] = f"{a}.o_proj.weight"
    # The latent projections also read `kv_norm` for the static row's latent gain.
    latent = [f"{a}.kv_b_proj.weight", f"{a}.kv_a_layernorm.weight"]
    for name in ("wuk", "wuv"):
        mappings[f"{param_prefix}{name}"] = list(latent)
    if config.indexer_types[layer_idx] != "shared":
        idx, p = f"{a}.indexer", f"{param_prefix}indexer."
        mappings.update(
            {
                f"{p}wq_b": f"{idx}.wq_b.weight",
                f"{p}wk": f"{idx}.wk.weight",
                f"{p}weights_proj": f"{idx}.weights_proj.weight",
                f"{p}k_norm_weight": f"{idx}.k_norm.weight",
                f"{p}k_norm_bias": f"{idx}.k_norm.bias",
            }
        )
    return mappings


# Dense MLP (tensor parallel over the wo group)
def _mlp_tp_shard_loader(num_shards: int, shard_dim: int):
    """Load this rank's tensor-parallel shard of a dense MLP weight, in `[in, out]`.

    Gate and up are column shards `[H, I / n]` (`shard_dim=1`), and down is a row shard
    `[I / n, H]` (`shard_dim=0`). The shard index is the rank's position in its group
    of `num_shards` consecutive ranks.
    """

    def transform(slices, rank):
        w = torch.as_tensor(slices[0][:]).transpose(0, 1)  # [in, out]
        size = w.shape[shard_dim] // num_shards
        return w.narrow(shard_dim, (rank % num_shards) * size, size).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def attach_mlp_loaders(mlp, group_size: int) -> None:
    """Attach the loaders of one dense `GlmMoeDsaMLP`."""
    set_weight_loader(mlp.gate_proj, _mlp_tp_shard_loader(group_size, 1))
    set_weight_loader(mlp.up_proj, _mlp_tp_shard_loader(group_size, 1))
    set_weight_loader(mlp.down_proj, _mlp_tp_shard_loader(group_size, 0))
    set_weight_loader(mlp.post_attention_layernorm_weight, _replicated_loader())


def mlp_hf_mappings(config, layer_idx: int, param_prefix: str = "") -> dict:
    """Return `{param name: HF key}` for one dense MLP layer."""
    layer = f"model.layers.{layer_idx}"
    return {
        f"{param_prefix}gate_proj": f"{layer}.mlp.gate_proj.weight",
        f"{param_prefix}up_proj": f"{layer}.mlp.up_proj.weight",
        f"{param_prefix}down_proj": f"{layer}.mlp.down_proj.weight",
        f"{param_prefix}post_attention_layernorm_weight": (
            f"{layer}.post_attention_layernorm.weight"
        ),
    }


# MXFP4 MoE
MOE_PLACEMENT_ENV = "GLM_GK_MOE_PLACEMENT"
_PLACEMENT_CACHE: dict = {}


#: The correction bias of an empty slot (GLM_GK_MOE_PLACEMENT with `e_local`): far below every real score (sigmoid in
#: (0, 1) plus a bias of ~+-20), so NOAUX_TC never selects it.
EMPTY_SLOT_BIAS = -1000.0


def moe_placement(n_experts: int = 256) -> dict:
    """`GLM_GK_MOE_PLACEMENT=<file.json>`: a static expert placement, `{checkpoint layer: [physical slot -> expert]}`.

    The file holds `{"layers": {"<checkpoint layer>": [ints]}}` and optionally `"e_local": n`. Without `e_local` each
    list is a permutation of `range(E)`. With it, each list has `ep * e_local` entries, every expert once and `-1` for
    an empty slot, whose router column is zero with bias `EMPTY_SLOT_BIAS` so it is never selected. That lets a hot
    expert's rank hold fewer experts. Router rows, correction bias and experts load in slot order, so the kernel runs
    unchanged (EP rank `q` owns slots `[q * e_local, (q + 1) * e_local)`); a token routes to the same experts, only the
    combine's summation order changes. Unlisted layers, and an unset variable, keep the identity.

    Returns:
        `{layer: slot list}`, with `"e_local"` holding `e_local` (None without it). The file is re-checked for each
        `n_experts` it is called with.

    Raises:
        ValueError: If the file has no `"layers"` map or has a `"replicas"` section, `e_local` is not an int >= 1, a
            key or entry is not a plain int, or a list is not a valid placement.
    """
    path = os.environ.get(MOE_PLACEMENT_ENV, "").strip()
    if not path:
        return {}
    key = (path, n_experts)
    if key not in _PLACEMENT_CACHE:
        import json

        with open(path) as f:
            raw = json.load(f)
        if not isinstance(raw, dict) or not isinstance(raw.get("layers"), dict):
            raise ValueError(
                f'{MOE_PLACEMENT_ENV}={path}: the file has no "layers" map of checkpoint layer -> slot list.'
            )
        if "replicas" in raw:
            # a replica slot copies a hot expert onto another EP rank; the stage graph has no path for one, so the
            # file would load as a placement other than the one it states
            raise ValueError(
                f'{MOE_PLACEMENT_ENV}={path}: the file has a "replicas" section, and the stage graph serves no expert '
                "replica slots; remove the section."
            )
        e_local = raw.get("e_local")
        if e_local is not None and (
            isinstance(e_local, bool) or not isinstance(e_local, int) or e_local < 1
        ):
            raise ValueError(
                f'{MOE_PLACEMENT_ENV}={path}: "e_local" is {e_local!r}, not a slot count of at least 1 per EP rank.'
            )
        out = {"e_local": e_local}
        n_slots = None
        for k, v in raw["layers"].items():
            layer = _placement_layer(path, k)
            if not isinstance(v, list):
                raise ValueError(f"{MOE_PLACEMENT_ENV}={path}: layer {k} maps to {v!r}, not a slot list.")
            perm = [_placement_int(path, f"layer {k} slot {j}", x) for j, x in enumerate(v)]
            real = sorted(x for x in perm if x >= 0)
            ok = real == list(range(n_experts)) and (e_local or len(perm) == n_experts)
            if e_local:
                n_slots = n_slots or len(perm)
                ok = (
                    ok
                    and len(perm) == n_slots
                    and n_slots % e_local == 0
                    and all(x >= -1 for x in perm)
                )
            if not ok:
                raise ValueError(
                    f"{MOE_PLACEMENT_ENV}={path}: layer {k} is not a placement of range({n_experts})"
                    + (f" over {n_slots} slots" if e_local else "")
                    + "."
                )
            out[layer] = perm
        logger.info(
            "[glm_moe_dsa] %s=%s: %d MoE layers placed, e_local %s.",
            MOE_PLACEMENT_ENV,
            path,
            sum(isinstance(k, int) for k in out),
            out["e_local"],
        )
        _PLACEMENT_CACHE[key] = out
    return _PLACEMENT_CACHE[key]


def placement_e_local(n_experts: int, ep_degree: int) -> int:
    """Slots per EP rank: the placement file's `e_local`, else `n_experts // ep_degree`."""
    e_local = moe_placement(n_experts).get("e_local")
    if e_local is None:
        return n_experts // ep_degree
    if e_local * ep_degree < n_experts:
        raise ValueError(
            f"{MOE_PLACEMENT_ENV}: e_local {e_local} x ep {ep_degree} slots hold fewer than {n_experts} experts."
        )
    return e_local


def _placement_int(path, where: str, x) -> int:
    """A slot entry of the placement file, which must be a JSON integer.

    `int()` would take a bool (`true` is slot 1), truncate a float (`2.7` is slot 2) and parse a string, so a
    malformed file would load as another placement instead of being refused."""
    if isinstance(x, bool) or not isinstance(x, int):
        raise ValueError(f"{MOE_PLACEMENT_ENV}={path}: {where} is {x!r}, not a plain int.")
    return x


def _placement_layer(path, k: str) -> int:
    """A `"layers"` key: a JSON object key is a string, so a checkpoint layer index is its decimal
    digits alone. `int()` would also take `" 3"`, `"+3"`, `"3_0"` and `"03"`, and two spellings of one layer would
    then load as whichever the file lists last."""
    if not (k.isascii() and k.isdigit() and str(int(k)) == k):
        raise ValueError(f"{MOE_PLACEMENT_ENV}={path}: the layer key {k!r} is not a plain int.")
    return int(k)


def _layer_placement(layer_idx, n_experts: int, ep_degree: int = None):
    """The slot -> expert list (-1 = empty) of checkpoint layer `layer_idx`, or None for the plain identity."""
    if layer_idx is None:
        return None
    pl = moe_placement(n_experts)
    perm = pl.get(int(layer_idx))
    e_local = pl.get("e_local")
    if e_local is not None and ep_degree is not None:
        if perm is None:
            el0 = n_experts // ep_degree
            perm = [
                el0 * q + e if e < el0 else -1
                for q in range(ep_degree)
                for e in range(e_local)
            ]
        if len(perm) != ep_degree * e_local:
            raise ValueError(
                f"{MOE_PLACEMENT_ENV}: layer {layer_idx} has {len(perm)} slots, not {ep_degree} x {e_local}."
            )
        return perm
    return None if perm is None or perm == list(range(n_experts)) else perm


def _slot_rows(t: torch.Tensor, perm, fill: float) -> torch.Tensor:
    """Rows of `t` [E, ...] in slot order; an empty slot (-1) gets `fill`."""
    idx = torch.as_tensor(perm, dtype=torch.int64)
    out = torch.full((len(perm),) + tuple(t.shape[1:]), fill, dtype=t.dtype)
    keep = idx >= 0
    out[keep] = t.index_select(0, idx[keep].to(t.device))
    return out.contiguous()


def _permuted_fp32_loader(perm):
    """Load the router correction bias as fp32, in slot order (`perm`: slot -> expert, -1 = empty)."""

    def transform(slices, rank):
        b = torch.as_tensor(slices[0][:]).to(torch.float32)
        return _slot_rows(b, perm, EMPTY_SLOT_BIAS)

    return SafetensorsWeightLoader(transform=transform)


def _swizzle_router_weight_loader(hidden_size: int, perm=None):
    """Load the router weight as `[H, E]` bf16 in the swizzle-H order of the kernel.

    `perm` (GLM_GK_MOE_PLACEMENT): the slot -> expert order of the E columns; None keeps the expert order.

    The fused `rmsnorm_mx_prefill` router matmuls against the swizzle-transposed
    activation, whose partition axis is `h_swz = h512*512 + 4p + q`. So row h of the
    natural weight must be at swizzle slot `h_swz`. NOAUX_TC also needs the tile and
    partition axes transposed: the kernel reads this weight as `reshape(128, H/128, E)`,
    so row `p * (H/128) + t` must hold swizzle slot `t * 128 + p`.
    """

    def transform(slices, rank):
        w = torch.as_tensor(slices[0][:])  # [E, H] natural H order
        if perm is not None:
            w = _slot_rows(w, perm, 0.0)  # [slots, H] slot order, empty slots zero
        hperm = swizzle_h_index(hidden_size).to(w.device)
        swizzled = w.index_select(1, hperm).transpose(0, 1)  # [H, E] swizzle order
        tiles = hidden_size // _PMAX
        # [tile, partition, E] -> [partition, tile, E], the (p, t, e) byte order.
        return (
            swizzled.reshape(tiles, _PMAX, -1)
            .permute(1, 0, 2)
            .contiguous()
            .reshape(swizzled.shape)
        )

    return SafetensorsWeightLoader(transform=transform)


def _stack_local_experts(slices, rank, ep_degree, tp_degree, num_local, per_expert):
    """Read only THIS rank's experts out of a per-expert slice list.

    `slices` holds `per_expert` entries for every global expert, in expert order. EP
    rank k owns the experts `[k*num_local, (k+1)*num_local)`, and the EP position of a
    global rank is `rank // tp_degree`. The 256 experts of one GLM layer are about
    3 GB of fp4 bytes, and 4 local experts are about 50 MB.

    Returns one list for each of the `per_expert` positions, each of `num_local` tensors.
    """
    ep_rank = (rank // tp_degree) % ep_degree if ep_degree > 1 else 0
    first = ep_rank * num_local
    out = [[] for _ in range(per_expert)]
    for local_e in range(num_local):
        base = (first + local_e) * per_expert
        for pos in range(per_expert):
            out[pos].append(torch.as_tensor(slices[base + pos][:]))
    return out


def _tp_rank(rank: int, tp_degree: int) -> int:
    return rank % tp_degree if tp_degree > 1 else 0


#: `GLM_GK_MOE_WRELAY=1|2` stores the routed experts' gate/up weights I-tile-major in HBM (see
#: `_gate_up_tile_major`); the prefill giga kernel (`_moe_wstream`) reads the same variable and streams them in long runs.
MOE_WRELAY_ENV = "GLM_GK_MOE_WRELAY"


def moe_wrelay() -> int:
    """`GLM_GK_MOE_WRELAY`: 0 (unset, default) = the default `[E, 128, 2, H/512, I]` layout; 1 / 2 = the
    routed gate/up weights in the I-tile-major byte order of `_gate_up_tile_major` (the kernel side differs: 1 keeps the
    H-chunked weight streams, 2 loads each 512-wide I tile with one DMA). A GLM_GK_ name, so it keys the compile cache.

    Raises:
        ValueError: If the variable is not 0, 1 or 2.
    """
    raw = os.environ.get(MOE_WRELAY_ENV, "").strip()
    if raw not in ("", "0", "1", "2"):
        raise ValueError(f"{MOE_WRELAY_ENV}={raw!r}: use 1, 2, 0 or leave it unset.")
    return int(raw) if raw else 0


def _gate_up_tile_major(w: torch.Tensor) -> torch.Tensor:
    """`[E, 128, 2, n_h512, I]` gate/up weights -> the same shape, bytes in the order
    `[E][I / 512][128][2][n_h512][512]` (I-tile-major).

    The kernel's weight loads move one 512-wide I tile of one expert at a time into an SBUF buffer
    `[128, 2, n_h512, 512]`. In the default order a partition's part of that tile is `2 * n_h512` runs of 1 KiB, 4 KiB apart
    (one per gate/up and H512 row), so every DMA packet gathers 1 KiB pieces. In this order the tile is one contiguous
    `[128][2][n_h512][512]` block: a partition's part is one 24 KiB run (n_h512 12) and a 3-H512-tile stream chunk one
    3 KiB run. The kernel (bwmm `_gup_tile_src`) views the tensor as `[E, I / 512, 128, 2, n_h512, 512]`, so every SBUF
    byte it loads is the byte it loads from the default layout. `I <= 512` (one tile) is the identity.
    """
    e, p, two, nh, inter = w.shape
    if inter <= _MX_TILE:
        return w
    if inter % _MX_TILE:
        raise ValueError(
            f"_gate_up_tile_major: I {inter} of the gate/up weights {tuple(w.shape)} is not whole {_MX_TILE}-wide tiles"
        )
    nt = inter // _MX_TILE
    return (
        w.reshape(e, p, two, nh, nt, _MX_TILE)
        .permute(0, 4, 1, 2, 3, 5)
        .contiguous()
        .reshape(e, p, two, nh, inter)
    )


def _moe_gate_up_weight_loader(
    ep_degree, tp_degree, num_local_experts, inter, tile_major=False
):
    """Load fp4 `gate`/`up` weights as `[E_local, 128, 2, H/512, I_rank]` uint16.

    The slice order is `[gate_0, up_0, gate_1, up_1, ...]`, and each tensor is
    `[I, H/2]` uint8. Shard I, concatenate gate and then up on the I axis, pack byte
    pairs into uint16 x4-elements, transpose so that H leads, then tile.
    `tile_major` (GLM_GK_MOE_WRELAY, the routed experts only): the same shape in the I-tile-major byte order of
    `_gate_up_tile_major`.
    """

    def transform(slices, rank):
        gate_list, up_list = _stack_local_experts(
            slices, rank, ep_degree, tp_degree, num_local_experts, per_expert=2
        )
        lo = _tp_rank(rank, tp_degree) * inter
        gate = torch.stack([g[lo : lo + inter] for g in gate_list])  # [E, I_rank, H/2]
        up = torch.stack([u[lo : lo + inter] for u in up_list])
        blocks = torch.cat([gate, up], dim=1)  # [E_local, 2*I_rank, H/2]
        # 2 fp4 bytes make one uint16 x4-element, so H/2 bytes give H/4 elements.
        blocks = blocks.contiguous().view(torch.uint16)  # [E_local, 2*I_rank, H/4]
        blocks = blocks.transpose(1, 2).contiguous()  # [E_local, H/4, 2*I_rank]
        w = tile_gate_up_blocks(blocks).contiguous()
        return _gate_up_tile_major(w) if tile_major else w

    return SafetensorsWeightLoader(transform=transform)


def _moe_gate_up_scale_loader(ep_degree, tp_degree, num_local_experts, inter):
    """Load `gate`/`up` scales in the packed layout `[E_local, 128, ceil(H/512/4), 2, I_rank]`.

    The slice order is `[gate_scale_0, up_scale_0, ...]`, and each tensor is `[I, H/32]`.
    """

    def transform(slices, rank):
        gate_list, up_list = _stack_local_experts(
            slices, rank, ep_degree, tp_degree, num_local_experts, per_expert=2
        )
        lo = _tp_rank(rank, tp_degree) * inter
        gate = torch.stack([g[lo : lo + inter] for g in gate_list])  # [E, I_rank, H/32]
        up = torch.stack([u[lo : lo + inter] for u in up_list])
        scales = torch.cat([gate, up], dim=1).transpose(1, 2).contiguous()
        std = tile_gate_up_scale(scales)  # [E_local, 16, 2, H/512, I_rank]
        # The packed layout puts the tile axis ahead of the gate/up axis.
        return _pack_mx_scale_tiles(std.permute(0, 1, 3, 2, 4).contiguous(), 2)

    return SafetensorsWeightLoader(transform=transform)


def _moe_down_weight_loader(ep_degree, tp_degree, num_local_experts, inter):
    """Load fp4 `down` weights as `[E_local, 128, I_rank/512, H]` uint16, H in natural order.

    Each tensor is `[H, I/2]` uint8. The I shard cuts the byte axis at `I_rank/2`.
    """

    def transform(slices, rank):
        (down_list,) = _stack_local_experts(
            slices, rank, ep_degree, tp_degree, num_local_experts, per_expert=1
        )
        lo = _tp_rank(rank, tp_degree) * inter // 2
        blocks = torch.stack([d[:, lo : lo + inter // 2] for d in down_list])
        e, h, _ = blocks.shape
        # [E, H, I_rank/4, 2] -> [E, I_rank/4, H, 2] -> pack -> [E, I_rank/4, H]
        blocks = blocks.contiguous().view(e, h, -1, 2).transpose(1, 2).contiguous()
        blocks = blocks.view(torch.uint16).view(e, -1, h)
        return blocks.reshape(e, -1, _PMAX, h).permute(0, 2, 1, 3).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _moe_down_scale_loader(ep_degree, tp_degree, num_local_experts, inter):
    """Load `down` scales in the packed layout `[E_local, 128, ceil(I_rank/512/4), H]`.

    Each tensor is `[H, I/32]` uint8, so the I shard cuts at `I_rank/32`.
    """
    groups = _MX_TILE // _MX_GROUP  # 16 scale rows for each 512-wide I tile

    def transform(slices, rank):
        (down_list,) = _stack_local_experts(
            slices, rank, ep_degree, tp_degree, num_local_experts, per_expert=1
        )
        lo = _tp_rank(rank, tp_degree) * inter // _MX_GROUP
        scales = torch.stack([d[:, lo : lo + inter // _MX_GROUP] for d in down_list])
        scales = scales.transpose(1, 2).contiguous()  # [E, I_rank/32, H]
        e, i_div_32, h = scales.shape
        std = (
            scales.reshape(e, i_div_32 // groups, groups, h)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # [E, 16, I_rank/512, H]
        return _pack_mx_scale_tiles(std, 2)

    return SafetensorsWeightLoader(transform=transform)


def _pack_mx_scale_tiles(std: torch.Tensor, tile_dim: int) -> torch.Tensor:
    """Repack standard-layout MX scales `[E, 16, ...]` into the packed `[E, 128, ...]` layout.

    The host twin of `_repack_scales_to_packed` in `block_moe_mx`. In the standard
    layout, scale row `4q + r` of contraction tile `k` belongs at SBUF partition
    `32q + r`. The packed layout holds 4 tiles per buffer: tile `k = 4b + s` is in
    packed buffer `b` at partition `32q + 4s + r`. `tile_dim` is the tile axis of `std`,
    which becomes the packed-buffer axis. The partitions that the layout does not use
    stay zero; the matmul does not read them.
    """
    rows = std.shape[1]
    if rows != 16:
        raise ValueError(
            f"_pack_mx_scale_tiles: the standard MX scales {tuple(std.shape)} have {rows} rows per expert, not 16"
        )
    n_tiles = std.shape[tile_dim]
    shape = list(std.shape)
    shape[1], shape[tile_dim] = _PMAX, -(-n_tiles // 4)
    packed = torch.zeros(shape, dtype=std.dtype)
    for q in range(4):
        for r in range(4):
            for k in range(n_tiles):
                b, s = divmod(k, 4)
                src = std.select(1, 4 * q + r).select(tile_dim - 1, k)
                packed.select(1, 32 * q + 4 * s + r).select(tile_dim - 1, b).copy_(src)
    return packed.contiguous()


def attach_moe_loaders(moe, config, layer_idx=None) -> None:
    """Attach the loaders of one `GlmMoeDsaMoE` module.

    The routed experts read the EP x TP mesh of the module. The shared expert reuses
    the routed-expert loaders with one expert, EP 1 and TP 1, so it gets the same
    tiling. Those loaders read a per-expert key list, which `moe_hf_mappings` supplies.
    `layer_idx`: the checkpoint layer, for GLM_GK_MOE_PLACEMENT (router rows and bias in
    slot order; `moe_hf_mappings` lists the experts in the same order).
    """
    perm = _layer_placement(layer_idx, config.n_routed_experts, moe.ep_degree)
    set_weight_loader(moe.post_attn_norm_weight, _replicated_loader())
    set_weight_loader(
        moe.router_weight, _swizzle_router_weight_loader(config.hidden_size, perm)
    )
    set_weight_loader(
        moe.e_score_correction_bias,
        _replicated_loader(torch.float32) if perm is None else _permuted_fp32_loader(perm),
    )
    routed = (
        moe.ep_degree,
        moe.tp_degree,
        moe.num_local_experts,
        moe.moe_inter_per_rank,
    )
    # With `shared_tp` the shared expert is an I-sharded one-expert MoE (slice rank % m).
    shared = (1, moe.shared_tp, 1, moe.shared_inter // moe.shared_tp)
    for mesh, prefix in ((routed, ""), (shared, "shared_")):
        for name, make in (
            ("gate_up_weight", _moe_gate_up_weight_loader),
            ("gate_up_scale", _moe_gate_up_scale_loader),
            ("down_weight", _moe_down_weight_loader),
            ("down_scale", _moe_down_scale_loader),
        ):
            # GLM_GK_MOE_WRELAY: the routed gate/up weights I-tile-major (the shared expert keeps its layout).
            tile_major = name == "gate_up_weight" and prefix == "" and moe_wrelay() > 0
            loader = make(*mesh, tile_major=True) if tile_major else make(*mesh)
            set_weight_loader(getattr(moe, f"{prefix}{name}"), loader)


def moe_hf_mappings(
    config, layer_idx: int, param_prefix: str = "", ep_degree: int = None
) -> dict:
    """Return `{param name: HF key}` for one MXFP4 MoE layer.

    The routed experts give one key for every global expert, in expert order. The
    loaders read only the experts of this rank out of the list, so the full list costs
    no memory. The shared expert maps as one expert.
    """
    layer = f"model.layers.{layer_idx}"
    m = f"{layer}.mlp"
    p = param_prefix
    mappings = {
        # The post-attention norm is on the decoder layer in HuggingFace. The fused
        # kernel owns it here, so it loads into the MoE module.
        f"{p}post_attn_norm_weight": f"{layer}.post_attention_layernorm.weight",
        f"{p}router_weight": f"{m}.gate.weight",
        f"{p}e_score_correction_bias": f"{m}.gate.e_score_correction_bias",
    }
    # GLM_GK_MOE_PLACEMENT: the experts in slot order (the loaders take the EP rank's slots). An empty slot (-1) loads
    # the first expert of its rank's slots (never read: its router column is never selected).
    order = _layer_placement(layer_idx, config.n_routed_experts, ep_degree) or range(
        config.n_routed_experts
    )
    if any(e < 0 for e in order):
        if ep_degree is None:
            raise ValueError(f"{MOE_PLACEMENT_ENV}: layer {layer_idx} has empty slots, whose experts come from their EP "
                             "rank's own slots: pass the stage's EP degree")
        n_slots = len(order)
        el = n_slots // ep_degree
        first_real = next(x for x in order if x >= 0)
        order = [
            e
            if e >= 0
            else next(
                (x for x in order[(j // el) * el : (j // el + 1) * el] if x >= 0),
                first_real,
            )
            for j, e in enumerate(order)
        ]
    for prefix, experts in (
        ("", [f"{m}.experts.{e}" for e in order]),
        ("shared_", [f"{m}.shared_experts"]),
    ):
        mappings[f"{p}{prefix}gate_up_weight"] = [
            f"{x}.{proj}.weight" for x in experts for proj in ("gate_proj", "up_proj")
        ]
        mappings[f"{p}{prefix}gate_up_scale"] = [
            f"{x}.{proj}.weight_scale"
            for x in experts
            for proj in ("gate_proj", "up_proj")
        ]
        mappings[f"{p}{prefix}down_weight"] = [f"{x}.down_proj.weight" for x in experts]
        mappings[f"{p}{prefix}down_scale"] = [
            f"{x}.down_proj.weight_scale" for x in experts
        ]
    return mappings


# Decoder layers and the causal LM
def attach_causal_lm_loaders(model) -> None:
    """Attach the weight loaders of a full `GlmMoeDsaForCausalLM`.

    The embedding and the lm_head shard the vocabulary over the TP group, and each
    loader reads only the rows of this rank. The final norm is replicated.
    """
    embed = model.model.embed_tokens
    # The embedding rounds the vocabulary up to a multiple of the TP size, so the last
    # shard can be short. Pad it to the full shard size.
    set_weight_loader(
        embed.weight,
        sharding_weight_loader(
            shard_dim=0,
            shard_size=embed.vocab_size_per_rank,
            num_shards=embed.tp_size,
            pad_shard=True,
        ),
    )
    lm_head = model.lm_head
    set_weight_loader(
        lm_head.weight,
        sharding_weight_loader(
            shard_dim=0,
            shard_size=lm_head.out_features_per_rank,
            num_shards=lm_head.tp_size,
        ),
    )
    set_weight_loader(model.model.norm_weight, _replicated_loader())
    mtp = model.model.mtp_kv
    if mtp is not None:
        # The MTP layer's prompt-row weights. HuggingFace stores a linear weight as [out, in].
        for param in (mtp.enorm_weight, mtp.hnorm_weight, mtp.input_norm_weight):
            set_weight_loader(param, _replicated_loader())
        set_weight_loader(mtp.kv_norm_weight, _replicated_loader())
        for param in (mtp.eh_proj, mtp.wkv_a, mtp.idx_wk):
            set_weight_loader(param, _transpose_loader())
        set_weight_loader(mtp.idx_k_norm_weight, _replicated_loader(torch.float32))
        set_weight_loader(mtp.idx_k_norm_bias, _replicated_loader(torch.float32))
    for layer in model.model.layers:
        attach_attention_loaders(layer.self_attn, model.config, layer.wo_group_size)
        if layer.is_moe:
            attach_moe_loaders(
                layer.mlp,
                model.config,
                layer_idx=getattr(model.config, "first_layer_index", 0)
                + layer.layer_idx,
            )
        else:
            attach_mlp_loaders(layer.mlp, layer.wo_group_size)
        set_weight_loader(layer.input_layernorm_weight, _replicated_loader())


def causal_lm_hf_mappings(config, mtp: bool = False, ep_degree: int = None) -> dict:
    """Return `{param name: HF key}` for a full `GlmMoeDsaForCausalLM`.

    `mtp` adds the weights of the MTP layer (checkpoint layer `mtp_layer_index`) that its
    prompt rows need (`mtp_kv`).
    """
    mappings = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm_weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    # `first_layer_index`: model layer i loads checkpoint layer first_layer_index + i.
    first = getattr(config, "first_layer_index", 0)
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}."
        mappings[f"{prefix}input_layernorm_weight"] = (
            f"model.layers.{first + i}.input_layernorm.weight"
        )
        mappings.update(
            attention_hf_mappings(config, i, f"{prefix}self_attn.", ckpt_idx=first + i)
        )
        if config.mlp_layer_types[i] == "sparse":
            mappings.update(
                moe_hf_mappings(config, first + i, f"{prefix}mlp.", ep_degree=ep_degree)
            )
        else:
            mappings.update(mlp_hf_mappings(config, first + i, f"{prefix}mlp."))
    if mtp:
        mappings.update(mtp_hf_mappings(config.mtp_layer_index))
    return mappings
