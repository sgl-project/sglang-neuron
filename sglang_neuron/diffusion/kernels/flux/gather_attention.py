# SPDX-License-Identifier: Apache-2.0
"""CP K/V all-gather fused with attention_cte for the FLUX.1 gathered-K/V path.

Each local head's K and V token slice is copied to a contiguous shared-HBM buffer (heads split
between the cores) and all heads are all-gathered over the CP ring inside the kernel in one
coalesced collective, straight into per-head
[cp * seqlen, d] tensors (the layout attention_cte reads), so no separate permute runs. Each core then
runs attention_cte's body on its own heads (head i on core i), and the remainder head of an odd head
count is split by query-row groups across both cores, which attention_cte itself does only from
1024 query rows; each row's computation is the same either way.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
import nkilib.core.attention.attention_cte as _attention_cte_module
from nki.collectives import ReplicaGroup
from nkilib.core.attention.attention_cte import _attention_cte

from nkilib.core.utils.kernel_assert import kernel_assert


def _single_core_sharding(*args, **kwargs):
    """Program sharding info of a one-core launch: (grid_ndim, num_programs, program_id)."""
    return 1, 1, 0


def _attention_cte_body(q, k_all, v_all, o_out, split_rows=False):
    """attention_cte body (non-causal, tp_q / tp_k / tp_out, fp32 softmax).

    On this core only, or with ``split_rows`` the single head's query-row groups split across the
    launch's cores (attention_cte's remainder-head sharding with its minimum length lifted).

    nkilib's attention_cte takes its core split from the module's
    ``get_verified_program_sharding_info`` and splits a head's query rows across cores only from
    ``_MIN_SEQLEN_FOR_LNC2_SHARDING`` rows, with no argument for either. So this calls its
    private body ``_attention_cte`` with one of those module globals replaced: the sharding
    function by a one-core answer, or the minimum by 0. The replacement lasts only for this
    call, which runs while NKI traces the kernel, and the ``finally`` restores both. The assert
    fails the trace if an nkilib update removes either global.
    """
    module = _attention_cte_module
    kernel_assert(
        hasattr(module, "get_verified_program_sharding_info")
        and hasattr(module, "_MIN_SEQLEN_FOR_LNC2_SHARDING"),
        "nkilib attention_cte no longer has the sharding hooks that _attention_cte_body overrides",
    )
    sharding = module.get_verified_program_sharding_info
    min_rows = module._MIN_SEQLEN_FOR_LNC2_SHARDING
    if split_rows:
        module._MIN_SEQLEN_FOR_LNC2_SHARDING = 0
    else:
        module.get_verified_program_sharding_info = _single_core_sharding
    try:
        _attention_cte(
            q=q,
            k=k_all,
            v=v_all,
            scale=1.0,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            tp_out=True,
            cache_softmax=False,
            softmax_dtype=nl.float32,
            mm_out_dtype=nl.float32,
            out_o_hbm=o_out,
        )
    finally:
        module.get_verified_program_sharding_info = sharding
        module._MIN_SEQLEN_FOR_LNC2_SHARDING = min_rows


@nki.jit
def flux_gather_attention_fwd(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    v: nl.NkiTensor,
    replica_groups: tuple = None,
    num_workers: int = 1,
):
    """Gather K/V over CP into head-major per-head tensors and run attention.

    Args:
        q (nl.NkiTensor): Pre-scaled query [heads, seqlen_q, d] (all CP-local query rows).
        k (nl.NkiTensor): Local key, token-major [seqlen, heads, d].
        v (nl.NkiTensor): Local value, token-major [seqlen, heads, d].
        replica_groups (tuple): Replica groups of the CP rings.
        num_workers (int): CP degree.

    Returns:
        o (nl.NkiTensor): [heads, d, seqlen_q] attention output (d-major).
    """
    heads, seqlen_q, head_dim = q.shape
    seqlen, _, _ = k.shape
    replica_group = ReplicaGroup(replica_groups)
    k = k.reshape((seqlen, heads * head_dim))
    v = v.reshape((seqlen, heads * head_dim))
    num_shards = nl.num_programs(0) if nl.program_ndim() == 1 else 1

    k_loc = []
    v_loc = []
    for head in range(heads):
        k_loc.append(
            nl.ndarray(
                (seqlen, head_dim), dtype=k.dtype, buffer=nl.shared_hbm, name=f"fga_k_loc{head}"
            )
        )
        v_loc.append(
            nl.ndarray(
                (seqlen, head_dim), dtype=v.dtype, buffer=nl.shared_hbm, name=f"fga_v_loc{head}"
            )
        )
    k_all = []
    v_all = []
    for head in range(heads):
        k_all.append(
            nl.ndarray(
                (num_workers * seqlen, head_dim),
                dtype=k.dtype,
                buffer=nl.shared_hbm,
                name=f"fga_k_all{head}",
            )
        )
        v_all.append(
            nl.ndarray(
                (num_workers * seqlen, head_dim),
                dtype=v.dtype,
                buffer=nl.shared_hbm,
                name=f"fga_v_all{head}",
            )
        )
    o = nl.ndarray((heads, head_dim, seqlen_q), dtype=q.dtype, buffer=nl.shared_hbm)
    shard_id = nl.program_id(0) if num_shards > 1 else 0

    if num_shards > 1:
        for head in range(heads):
            nisa.core_barrier(k_loc[head], (0, 1))
            nisa.core_barrier(v_loc[head], (0, 1))

    row_stride = heads * head_dim
    for head in range(shard_id, heads, num_shards):
        nisa.dma_copy(
            dst=k_loc[head],
            src=k.ap(pattern=[[row_stride, seqlen], [1, head_dim]], offset=head * head_dim),
        )
        nisa.dma_copy(
            dst=v_loc[head],
            src=v.ap(pattern=[[row_stride, seqlen], [1, head_dim]], offset=head * head_dim),
        )

    if num_shards > 1:
        for head in range(heads):
            nisa.core_barrier(k_loc[head], (0, 1))
            nisa.core_barrier(v_loc[head], (0, 1))

    srcs = []
    dsts = []
    for head in range(heads):
        srcs.append(k_loc[head])
        srcs.append(v_loc[head])
        dsts.append(k_all[head])
        dsts.append(v_all[head])
    ncc.all_gather(srcs=srcs, dsts=dsts, replica_group=replica_group, collective_dim=0)
    if num_shards > 1:
        for head in range(heads):
            nisa.core_barrier(k_all[head], (0, 1))
            nisa.core_barrier(v_all[head], (0, 1))

    paired = (heads // num_shards) * num_shards
    for head in range(heads):
        if head >= paired or head % num_shards == shard_id:
            kv_shape = (1, num_workers * seqlen, head_dim)
            _attention_cte_body(
                q[head : head + 1],
                k_all[head].reshape(kv_shape),
                v_all[head].reshape(kv_shape),
                o[head : head + 1],
                split_rows=head >= paired and num_shards > 1,
            )
    if num_shards > 1:
        nisa.core_barrier(o, (0, 1))
    return o


@nki.jit
def flux_gather_kv_fwd(
    k: nl.NkiTensor,
    v: nl.NkiTensor,
    replica_groups: tuple = None,
    num_workers: int = 1,
):
    """The K/V half of flux_gather_attention_fwd: copy-in and one coalesced CP all-gather.

    Args:
        k (nl.NkiTensor): Local key, token-major [seqlen, heads, d].
        v (nl.NkiTensor): Local value, token-major [seqlen, heads, d].
        replica_groups (tuple): Replica groups of the CP rings.
        num_workers (int): CP degree.

    Returns:
        Per head h, k_all[h] and v_all[h], each [cp * seqlen, d], as (k_0, v_0, k_1, v_1, ...).
    """
    seqlen, heads, head_dim = k.shape
    replica_group = ReplicaGroup(replica_groups)
    k = k.reshape((seqlen, heads * head_dim))
    v = v.reshape((seqlen, heads * head_dim))
    num_shards = nl.num_programs(0) if nl.program_ndim() == 1 else 1
    shard_id = nl.program_id(0) if num_shards > 1 else 0

    loc = []
    out = []
    for head in range(heads):
        for name, src_dtype in (("k", k.dtype), ("v", v.dtype)):
            loc.append(
                nl.ndarray(
                    (seqlen, head_dim),
                    dtype=src_dtype,
                    buffer=nl.shared_hbm,
                    name=f"fgkv_{name}_loc{head}",
                )
            )
            out.append(
                nl.ndarray((num_workers * seqlen, head_dim), dtype=src_dtype, buffer=nl.shared_hbm)
            )
    if num_shards > 1:
        for buf in loc:
            nisa.core_barrier(buf, (0, 1))

    row_stride = heads * head_dim
    for head in range(shard_id, heads, num_shards):
        nisa.dma_copy(
            dst=loc[2 * head],
            src=k.ap(pattern=[[row_stride, seqlen], [1, head_dim]], offset=head * head_dim),
        )
        nisa.dma_copy(
            dst=loc[2 * head + 1],
            src=v.ap(pattern=[[row_stride, seqlen], [1, head_dim]], offset=head * head_dim),
        )
    if num_shards > 1:
        for buf in loc:
            nisa.core_barrier(buf, (0, 1))

    ncc.all_gather(srcs=loc, dsts=out, replica_group=replica_group, collective_dim=0)
    if num_shards > 1:
        for buf in out:
            nisa.core_barrier(buf, (0, 1))
    return tuple(out)


@nki.jit
def flux_gathered_kv_attention_fwd(
    q: nl.NkiTensor,
    k_0: nl.NkiTensor,
    v_0: nl.NkiTensor,
    k_1: nl.NkiTensor,
    v_1: nl.NkiTensor,
    k_2: nl.NkiTensor,
    v_2: nl.NkiTensor,
    after: nl.NkiTensor = None,
):
    """The attention half of flux_gather_attention_fwd for 3 heads on flux_gather_kv_fwd's outputs.

    The remainder head's query-row groups are split across both cores.

    Args:
        q (nl.NkiTensor): Pre-scaled query [3, seqlen_q, d].
        k_0, v_0, k_1, v_1, k_2, v_2 (nl.NkiTensor): Gathered per-head K and V, [cp * seqlen, d].
        after (nl.NkiTensor): Optional operand that orders its producer before this kernel; not read.

    Returns:
        o (nl.NkiTensor): [3, d, seqlen_q] attention output (d-major).
    """
    heads, seqlen_q, head_dim = q.shape
    num_shards = nl.num_programs(0) if nl.program_ndim() == 1 else 1
    k_all = [k_0, k_1, k_2]
    v_all = [v_0, v_1, v_2]
    o = nl.ndarray((heads, head_dim, seqlen_q), dtype=q.dtype, buffer=nl.shared_hbm)
    shard_id = nl.program_id(0) if num_shards > 1 else 0
    paired = (heads // num_shards) * num_shards
    for head in range(heads):
        if head >= paired or head % num_shards == shard_id:
            kv_shape = (1, k_all[head].shape[0], head_dim)
            _attention_cte_body(
                q[head : head + 1],
                k_all[head].reshape(kv_shape),
                v_all[head].reshape(kv_shape),
                o[head : head + 1],
                split_rows=head >= paired and num_shards > 1,
            )
    if num_shards > 1:
        nisa.core_barrier(o, (0, 1))
    return o
