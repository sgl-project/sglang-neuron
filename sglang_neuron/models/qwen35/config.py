# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 model dimensions and the parallel layout the giga kernels run in under SGLang.

``Qwen35Dims`` reads the HF config (the text config of a multimodal checkpoint). ``Qwen35Layout`` maps SGLang's
parallelism onto the kernels' groups. SGLang serves the model with DP attention: ``--tp-size world --dp-size dp
--enable-dp-attention --ep-size world``. A DP replica is ``replica = world / dp`` consecutive ranks (rank
``d * replica + t`` is position ``t`` of replica ``d``), and SGLang gives every replica its own scheduler, request table,
KV pool and recurrent-state pool. The kernels' groups are chosen so that every request's state lives inside its own
replica:

- GDN TP = attention TP = the replica (``tp_group`` and ``atp_group`` of ``qwen35_hybrid_multistep`` are both the
  replica's ranks). A GDN layer's heads shard over the replica (``num_v_heads / replica`` value heads, the GQA key
  heads with them), so a request's recurrent state is on its replica's ranks only, which is what SGLang's per-replica
  state pool holds. Attention keeps one KV head per rank (the 2 KV heads replicate over the replica).
- MoE EP = every rank (``ep_group``): rank ``r`` owns experts ``r * E / world ...``; the router runs on the rank's own
  token rows.
- Token rows are sequence-parallel inside a replica: a decode step of ``rows`` rows per replica gives each rank
  ``rows / replica`` rows (``tok_shard``), DP-major over the world (the EP-gathered order).

The kernels' layouts of an earlier standalone engine (GDN TP 64 and attention TP 8 over 128 ranks) put one request's
GDN state on ranks of 8 attention replicas; SGLang's replica-local request and state pools cannot express that, so it is
not served here.

The KV cache is slot-contiguous: one SGLang page of ``slot_tokens`` tokens per request (``--page-size`` = the slot),
whose page id is the kernel's KV slot (``kv_slot``), K stored transposed (``[d, S]`` per slot) and V ``[S, d]``.
With a context longer than the page (``context``), the same tensors are a page pool (``paged``): a request owns the
pages SGLang's paged allocator gave it, the decode reads them through a block table and the prefill writes and reads
them by page (attn_decode_tkg's "dp" pages; docs/design/recurrent-state.md), so the radix cache shares them.

On the PD decode role only, ``kv_cp`` (``SGLANG_NEURON_QWEN35_KV_CP``) context-parallels the KV of the paged decode
over the replica: every rank holds all ``num_kv_heads`` KV heads of a ``slot_tokens / kv_cp``-token block of each page
(block ``r`` of every page on rank ``r``; the page id is the same on every rank), the decode all-gathers the query
heads over the replica and merges its ranks' split-K partials in the kernel, and a prefill without it sends the decode
rank its block of each page (sglang_patches/kv_cp_scatter). Block tables, the allocator and the radix cache are
identical on every rank (the whole page is still the allocation unit; a rank only stores and reads less of it).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

P = 128
# the kernel hard-codes the hybrid period: layer i is gated full attention when i % 4 == 3, else Gated DeltaNet
FULL_ATTENTION_PERIOD = 4
# attn_decode_tkg's flat new-token K/V write addresses the whole layer-major pool with one int32 element offset
FLAT_WRITE_ELEMENTS = 1 << 31


def text_config(hf_config):
    """The language model's config: ``text_config`` of a multimodal Qwen3.5 config, else the config itself."""
    return getattr(hf_config, "text_config", None) or hf_config


def _get(cfg, name, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


@dataclass(frozen=True)
class Qwen35Dims:
    hidden: int
    num_layers: int
    layer_types: Tuple[str, ...]
    num_k_heads: int  # GDN key / query heads
    num_v_heads: int  # GDN value heads
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int
    num_q_heads: int  # gated attention
    num_kv_heads: int
    head_dim: int
    num_experts: int
    top_k: int
    moe_intermediate: int
    shared_intermediate: int
    vocab: int
    rope_theta: float
    rotary_dim: int
    rms_norm_eps: float
    ssm_dtype: str

    @classmethod
    def from_hf(cls, hf_config) -> "Qwen35Dims":
        t = text_config(hf_config)
        n = int(_get(t, "num_hidden_layers"))
        interval = int(_get(t, "full_attention_interval", FULL_ATTENTION_PERIOD) or FULL_ATTENTION_PERIOD)
        types = _get(t, "layer_types") or [
            "full_attention" if i % interval == interval - 1 else "linear_attention" for i in range(n)]
        rope = _get(t, "rope_parameters") or {}
        head_dim = int(_get(t, "head_dim"))
        partial = float(_get(rope, "partial_rotary_factor", _get(t, "partial_rotary_factor", 0.25)))
        return cls(
            hidden=int(_get(t, "hidden_size")), num_layers=n, layer_types=tuple(types),
            num_k_heads=int(_get(t, "linear_num_key_heads")), num_v_heads=int(_get(t, "linear_num_value_heads")),
            head_k_dim=int(_get(t, "linear_key_head_dim")), head_v_dim=int(_get(t, "linear_value_head_dim")),
            conv_kernel=int(_get(t, "linear_conv_kernel_dim")),
            num_q_heads=int(_get(t, "num_attention_heads")), num_kv_heads=int(_get(t, "num_key_value_heads")),
            head_dim=head_dim, num_experts=int(_get(t, "num_experts")), top_k=int(_get(t, "num_experts_per_tok")),
            moe_intermediate=int(_get(t, "moe_intermediate_size")),
            shared_intermediate=int(_get(t, "shared_expert_intermediate_size")),
            vocab=int(_get(t, "vocab_size")),
            rope_theta=float(_get(rope, "rope_theta", _get(t, "rope_theta", 1e7))),
            rotary_dim=int(head_dim * partial), rms_norm_eps=float(_get(t, "rms_norm_eps", 1e-6)),
            ssm_dtype=str(_get(t, "mamba_ssm_dtype", "float32") or "float32"),
        )

    @property
    def gdn_layers(self) -> Tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.layer_types) if kind == "linear_attention")

    @property
    def attention_layers(self) -> Tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.layer_types) if kind == "full_attention")

    @property
    def conv_dim(self) -> int:
        """GDN conv channels over all heads: q | k | v."""
        return 2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim

    def problems(self) -> list:
        """What the giga kernels cannot run for this model (empty: they can)."""
        out = []
        for i, kind in enumerate(self.layer_types):
            want = "full_attention" if i % FULL_ATTENTION_PERIOD == FULL_ATTENTION_PERIOD - 1 else "linear_attention"
            if kind != want:
                out.append(f"layer {i} is {kind}; the giga kernels hard-code {want} there (every 4th layer full)")
                break
        if self.num_layers % FULL_ATTENTION_PERIOD:
            out.append(f"{self.num_layers} layers are not whole 3 GDN + 1 attention groups")
        if self.head_k_dim != P or self.head_v_dim != P:
            out.append(f"GDN heads are {self.head_k_dim}/{self.head_v_dim} wide; the kernels take 128")
        if self.hidden % 512:
            out.append(f"hidden {self.hidden} is not a multiple of 512")
        if self.head_dim & (self.head_dim - 1) or self.head_dim % P:
            out.append(f"attention head_dim {self.head_dim} is not a power of two multiple of 128")
        if self.moe_intermediate % 512:
            out.append(f"expert intermediate {self.moe_intermediate} is not a multiple of 512")
        if self.conv_kernel != 4:
            out.append(f"GDN conv kernel {self.conv_kernel}; the kernels keep a 3-token window")
        if self.ssm_dtype not in ("float32", "torch.float32"):
            out.append(f"mamba_ssm_dtype {self.ssm_dtype}; the kernels keep the recurrent state in float32")
        return out


# the launch's decode context-parallel degree (module docstring): 1 without the flag
KV_CP_ENV = "SGLANG_NEURON_QWEN35_KV_CP"


def pool_context_problems(context: int, page: int, max_total_tokens) -> list:
    """What a paged launch's KV pool cannot serve at its context (empty: it can serve it). SGLang's KV allocator keeps
    one page back and gives a request only the tokens left after its paged prompt and that page
    (``BaseTokenToKVPoolAllocator.max_new_tokens_for_memory``), so a pool under the context plus two pages answers a
    prompt near the context with no tokens, or never schedules it. ``context``: --context-length of a paged launch (0
    for the slot pool); ``page``: the KV page; ``max_total_tokens``: --max-total-tokens (a replica's pool), None when
    SGLang sizes the pool."""
    if not context or not max_total_tokens or page < 1:
        return []
    need = (-(-int(context) // int(page)) + 2) * int(page)
    if int(max_total_tokens) < need:
        return [f"--max-total-tokens {int(max_total_tokens)} (a replica's KV pool) cannot hold a request at "
                f"--context-length {int(context)}: the allocator keeps one page back, and a prompt near the context "
                f"needs its pages plus that page; set --max-total-tokens to at least {need} (the context plus two "
                f"pages of {int(page)}), or lower --context-length"]
    return []


def kv_cp_launch() -> int:
    """The decode context-parallel degree the launch asks for (``KV_CP_ENV``; 1 = the whole page per rank)."""
    raw = os.environ.get(KV_CP_ENV)
    if raw is None or raw == "":
        return 1
    try:
        degree = int(raw)
    except ValueError:
        return -1
    return degree


# opt-in token-major K pages on the kv_cp=1 PD prefill (docs/design/context-parallelism.md): the kv_cp>1 decode's pool
# always stores K token-major ([.., block, head_dim], like V); a prefill serving its scatter opts its own pool's K
# into the same layout so a rank's K block of a page is one contiguous run per layer on BOTH roles
# (sglang_patches/kv_cp_scatter). Refused on every other role: the kv_cp=1 decode's attention kernel reads the
# dim-major layout, and a whole-page PD transfer is layout-blind.
KV_TM_ENV = "SGLANG_NEURON_QWEN35_KV_TM"


def kv_tm_launch() -> bool:
    """The launch stores the kv_cp=1 prefill pool's K pages token-major (``KV_TM_ENV``, 0 or 1)."""
    from sglang_neuron import env_registry

    return env_registry.zero_or_one(KV_TM_ENV)


# opt-in HiCache host (L2) tier on the PD decode role (docs/design/context-parallelism.md): refused without the flag
HICACHE_ENV = "SGLANG_NEURON_QWEN35_HICACHE"


def hicache_launch() -> bool:
    """The launch opts the decode's KV and state pools into HiCache's host tier (``HICACHE_ENV``, 0 or 1)."""
    from sglang_neuron import env_registry

    return env_registry.zero_or_one(HICACHE_ENV)


@dataclass(frozen=True)
class Qwen35Layout:
    world: int  # SGLang --tp-size: every rank of the model
    dp: int  # SGLang attention DP replicas
    slot_tokens: int  # tokens of one request's KV slot (SGLang --page-size), or of one KV page when paged
    context: int = 0  # the longest context served (SGLang --context-length); 0: the slot
    kv_cp: int = 1  # decode context parallelism over the replica (module docstring); 1: every rank holds the page

    @property
    def paged(self) -> bool:
        """The KV pool is pages of a request's context, read through block tables (module docstring)."""
        return self.context > self.slot_tokens

    @property
    def block_tokens(self) -> int:
        """The tokens of one page one rank holds: the whole page without KV CP, its ``kv_cp``-th block with it (what the
        kernel's flat K/V write and the PD transfer address)."""
        if self.kv_cp > 1 and self.slot_tokens % self.kv_cp == 0:
            return self.slot_tokens // self.kv_cp
        return self.slot_tokens

    @property
    def rope_len(self) -> int:
        """Positions the RoPE tables cover: the context (whole pages)."""
        if not self.paged:
            return self.slot_tokens
        return -(-self.context // self.slot_tokens) * self.slot_tokens

    @property
    def pages_per_row(self) -> int:
        """Block-table columns of a decode row (MP): the context's pages."""
        return self.rope_len // self.slot_tokens

    @property
    def replica(self) -> int:
        """Ranks per DP replica: the GDN TP and attention TP group size."""
        return self.world // self.dp

    @property
    def ep(self) -> int:
        return self.world

    def replica_of(self, rank: int) -> int:
        return rank // self.replica

    def position_of(self, rank: int) -> int:
        """The rank's position in its replica: its GDN / attention TP rank."""
        return rank % self.replica

    def tp_mesh(self) -> tuple:
        """The GDN and attention TP partition: one cell of consecutive ranks per replica."""
        r = self.replica
        return tuple(tuple(range(d * r, (d + 1) * r)) for d in range(self.dp))

    def ep_mesh(self) -> tuple:
        return (tuple(range(self.world)),)

    def rows_per_rank(self, rows: int) -> int:
        """``tok_shard`` rows of a decode step of ``rows`` rows per replica."""
        if rows % self.replica:
            raise ValueError(f"{rows} decode rows per replica do not shard over the {self.replica}-rank replica")
        return rows // self.replica

    def problems(self, dims: Qwen35Dims, *, ep_size: Optional[int] = None, attention_layers: Optional[int] = None,
                 slots: Optional[int] = None) -> list:
        """What this layout cannot serve for ``dims`` (empty: it can). ``slots``: the KV pool's pages per layer
        (padding page included), checked against the flat K/V write's int32 offsets."""
        out = []
        r = self.replica
        if self.world < 1 or self.dp < 1 or self.world % self.dp:
            return [f"--dp-size {self.dp} does not divide --tp-size {self.world}"]
        if ep_size is not None and ep_size != self.world:
            out.append(f"--ep-size {ep_size}: the giga MoE runs expert parallelism over every rank ({self.world})")
        if dims.num_v_heads % r or dims.num_k_heads % r and r % dims.num_k_heads:
            out.append(f"GDN heads ({dims.num_k_heads} key, {dims.num_v_heads} value) do not shard over the "
                       f"{r}-rank replica")
        else:
            vh = dims.num_v_heads // r
            gqa = dims.num_v_heads // dims.num_k_heads
            if vh % gqa and gqa % vh:
                out.append(f"{vh} GDN value heads per rank do not align with the {gqa}:1 grouped heads")
        if dims.num_q_heads % r or r % dims.num_kv_heads:
            out.append(f"attention ({dims.num_q_heads} query, {dims.num_kv_heads} KV heads) over the {r}-rank "
                       "replica: the kernel needs whole query heads and one KV head per rank")
        if dims.num_experts % self.world or dims.shared_intermediate % self.world or dims.vocab % self.world:
            out.append(f"experts {dims.num_experts}, shared intermediate {dims.shared_intermediate} and vocabulary "
                       f"{dims.vocab} must divide over {self.world} ranks")
        if self.slot_tokens % P or self.slot_tokens < P:
            out.append(f"KV slot of {self.slot_tokens} tokens (--page-size) is not a multiple of 128")
        elif self.paged and self.slot_tokens & (self.slot_tokens - 1):
            out.append(f"KV page of {self.slot_tokens} tokens (--page-size): the paged attention takes a power of two "
                       "of at least 128")
        block = self.block_tokens
        if self.kv_cp != 1:
            if self.kv_cp < 1 or not isinstance(self.kv_cp, int):
                out.append(f"{KV_CP_ENV}={self.kv_cp}: the decode context-parallel degree is a positive integer")
            else:
                if not self.paged:
                    out.append(f"{KV_CP_ENV}={self.kv_cp}: decode context parallelism runs the paged attention (a "
                               "context past the page); the slot pool has no block table")
                if self.kv_cp != r:
                    out.append(f"{KV_CP_ENV}={self.kv_cp}: decode context parallelism spans the {r}-rank replica "
                               "(one KV block per rank of every page)")
                elif self.slot_tokens % self.kv_cp:
                    out.append(f"KV page of {self.slot_tokens} tokens does not split into {self.kv_cp} blocks")
                elif block % P:
                    out.append(f"a KV CP block of {block} tokens is not a multiple of 128 (the attention kernel's "
                               "CP segment)")
        if dims.head_dim * block >= FLAT_WRITE_ELEMENTS:
            out.append(f"a KV slot of {block} tokens exceeds the kernel's per-slot int32 offsets")
        if attention_layers is not None and slots is not None:
            elements = attention_layers * slots * dims.head_dim * block
            if elements > FLAT_WRITE_ELEMENTS:
                out.append(f"{slots} KV slots x {attention_layers} attention layers x {dims.head_dim} x "
                           f"{block} = {elements} elements exceed the flat K/V write's 2^31 (lower "
                           "--max-total-tokens or the slot)")
        return out

    def row_problems(self, dims: Qwen35Dims, rows: int, *, kv_slots: Optional[int] = None,
                     state_slots: Optional[int] = None) -> list:
        """What a decode step of ``rows`` rows per replica cannot run as (empty: it can). ``kv_slots`` / ``state_slots``:
        the pools' rows per layer (the KV pages with the reserved page; the state slots with slot 0 and the sink), which
        the kernel requires to hold at least one row per step row."""
        out = []
        if rows < 1 or rows % self.replica:
            return [f"{rows} decode rows per replica do not shard over the {self.replica}-rank replica"]
        for what, slots in (("KV pages", kv_slots), ("GDN state slots", state_slots)):
            if slots is not None and slots < rows:
                out.append(f"{slots} {what} per layer are fewer than the step's {rows} rows")
        if rows > P:
            out.append(f"{rows} GDN rows per replica exceed 128 (the router-fold path is not hosted)")
        q_local = dims.num_q_heads // self.replica
        if rows * q_local > P:
            out.append(f"{rows} attention rows x {q_local} query heads exceed the 128 partitions of attn_decode_tkg")
        return out
