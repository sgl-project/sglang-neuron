# SPDX-License-Identifier: Apache-2.0
"""The GPT-OSS decode giga-kernel: every decoder layer of a decode step in one NKI kernel call.

``gpt_oss_decode_mega_kernel`` (nkilib ``models.gpt_oss.c128_giga_kernel``, from the image's kernel library) runs the
whole stack for this rank's sequence-parallel token shard: per layer, the TP attention with its in-kernel packed-FP8 KV
write, then the EP MoE over all the ranks of the expert mesh. It takes:

- ``x_shard``, a ``[B/(DP*TP), H]`` placeholder for this rank's token rows, whose shape and dtype the kernel takes:
  with ``embed_weight`` and ``embed_ids`` it embeds layer 0's input itself from this rank's vocabulary shard;
- every per-layer weight stacked on a leading ``[L, ...]`` axis (NKI takes no Python lists; ``W[i]`` is layer ``i``);
- the KV caches in the kernel's paired-buffer form: buffer ``j`` holds layers 2j and 2j + 1 (``[2 * blocks, kv_heads,
  block // 2, d, 2]`` for packed FP8 K), the odd layer's blocks in the second half;
- the attention control: the RoPE tables, the full and the window-trimmed block tables with their positions, each
  layer's window start (``[L, B, S]``, zero on the full layers), the active-token mask, and the KV write slots of the
  full and the sliding layers;
- the TP and EP rank meshes.

It returns the post-stack token shard and the K / V caches it wrote in place (the aliases that carry the write back to
the pool). There is no torch fallback: the giga-kernel decode graph is only built where the kernel runs.
"""
from torch import Tensor

from sglang_neuron.backend.stack import resolve_full_mesh, wrap_nki

try:
    import nki
    import nki.collectives as ncc
    import nki.language as nl
    from nkilib.models.gpt_oss.c128_giga_kernel.experimental.gpt_oss.gpt_oss_decode_layer_dp_tp_ep import (
        gpt_oss_decode_mega_kernel as _gpt_oss_decode_mega_kernel,
    )
except Exception as _err:  # noqa: BLE001 -- an image without the giga kernel; the decode graph refuses to build
    _gpt_oss_decode_mega_kernel = None
    _IMPORT_ERROR = _err
else:
    _IMPORT_ERROR = None


def available() -> bool:
    """Whether the image's kernel library ships the GPT-OSS decode giga-kernel."""
    return _gpt_oss_decode_mega_kernel is not None


def require() -> None:
    if not available():
        raise RuntimeError(
            "the GPT-OSS giga-kernel decode needs nkilib.models.gpt_oss.c128_giga_kernel from the image's kernel "
            f"library, which does not import here: {_IMPORT_ERROR!r}"
        )


def scale_pack64() -> bool:
    """Whether the image's giga kernel reads its gate/up MX scales in the PACK64 layout: the value its MoE froze when it
    was imported (``gate_up_projection_mx._SCALE_PACK64``), which is what the kernel traces with."""
    from nkilib.models.gpt_oss.c128_giga_kernel.core.moe.moe_tkg import gate_up_projection_mx

    return bool(gate_up_projection_mx._SCALE_PACK64)


def replica_mesh(group) -> tuple:
    """A group's full partition as static nested tuples: every cell, the same on every rank (one SPMD NEFF)."""
    return tuple(tuple(int(r) for r in cell) for cell in resolve_full_mesh(group))


if _gpt_oss_decode_mega_kernel is not None:

    @nki.jit
    def _giga_decode_jit(
        x_shard,
        input_layernorm_weight,
        W_qkv,
        bias_qkv,
        W_out,
        bias_out,
        sink,
        K_cache,
        V_cache,
        post_attn_layernorm_weight,
        router_weights,
        router_bias,
        expert_gate_up_weights,
        expert_down_weights,
        expert_gate_up_weights_scale,
        expert_down_weights_scale,
        expert_gate_up_bias,
        expert_down_bias,
        swa_start_pos_ids,
        cos,
        sin,
        active_blocks_table,
        active_blocks_table_swa,
        attention_mask,
        kv_cache_update_idx,
        pos_ids,
        pos_ids_swa,
        rank_id,
        tp_mesh: tuple = (),
        ep_mesh: tuple = (),
        num_layers: int = 0,
        tp_size: int = 0,
        ep_size: int = 0,
        B_attn: int = 0,
        S_tkg: int = 0,
        top_k: int = 0,
        eps: float = 1e-5,
        hidden_actual: int = 0,
        softmax_scale: float = 1.0,
        gate_clamp_upper_limit: float = 0.0,
        up_clamp_upper_limit: float = 0.0,
        up_clamp_lower_limit: float = 0.0,
        k_scale=None,
        v_scale=None,
        kv_cache_update_idx_swa=None,
        x_first_full=None,
        embed_weight=None,
        embed_ids=None,
        embed_x_shard_ids=None,
        inv_freq=None,
        rope_concentration: float = 1.0,
        K_cache_0=None,
        K_cache_1=None,
        K_cache_2=None,
        K_cache_3=None,
        K_cache_4=None,
        K_cache_5=None,
        K_cache_6=None,
        K_cache_7=None,
        K_cache_8=None,
        K_cache_9=None,
        K_cache_10=None,
        K_cache_11=None,
        K_cache_12=None,
        K_cache_13=None,
        K_cache_14=None,
        K_cache_15=None,
        K_cache_16=None,
        K_cache_17=None,
        V_cache_0=None,
        V_cache_1=None,
        V_cache_2=None,
        V_cache_3=None,
        V_cache_4=None,
        V_cache_5=None,
        V_cache_6=None,
        V_cache_7=None,
        V_cache_8=None,
        V_cache_9=None,
        V_cache_10=None,
        V_cache_11=None,
        V_cache_12=None,
        V_cache_13=None,
        V_cache_14=None,
        V_cache_15=None,
        V_cache_16=None,
        V_cache_17=None,
    ):
        """Rebuild the replica groups inside the trace (a ReplicaGroup cannot cross the HOP boundary; NKI's tracer
        rejects comprehensions, hence the loops) and view the uint16 MXFP4 blocks as the MX dtype."""
        if expert_gate_up_weights.dtype == nl.uint16:
            expert_gate_up_weights = expert_gate_up_weights.view(nl.float4_e2m1fn_x4)
        if expert_down_weights.dtype == nl.uint16:
            expert_down_weights = expert_down_weights.view(nl.float4_e2m1fn_x4)
        tp_cells = []
        for cell in tp_mesh:
            tp_cells.append(list(cell))
        ep_cells = []
        for cell in ep_mesh:
            ep_cells.append(list(cell))
        return _gpt_oss_decode_mega_kernel(
            x_shard,
            input_layernorm_weight,
            W_qkv,
            bias_qkv,
            W_out,
            bias_out,
            sink,
            K_cache,
            V_cache,
            post_attn_layernorm_weight,
            router_weights,
            router_bias,
            expert_gate_up_weights,
            expert_down_weights,
            expert_gate_up_weights_scale,
            expert_down_weights_scale,
            expert_gate_up_bias,
            expert_down_bias,
            swa_start_pos_ids,
            cos,
            sin,
            active_blocks_table,
            active_blocks_table_swa,
            attention_mask,
            kv_cache_update_idx,
            pos_ids,
            pos_ids_swa,
            rank_id,
            ncc.ReplicaGroup(tp_cells),
            ncc.ReplicaGroup(ep_cells),
            num_layers,
            tp_size,
            ep_size,
            B_attn,
            S_tkg,
            top_k,
            eps,
            hidden_actual,
            softmax_scale,
            gate_clamp_upper_limit,
            up_clamp_upper_limit,
            up_clamp_lower_limit,
            k_scale,
            v_scale,
            kv_cache_update_idx_swa,
            x_first_full,
            # by keyword: the host-mask inputs take the next two positions
            embed_weight=embed_weight,
            embed_ids=embed_ids,
            embed_x_shard_ids=embed_x_shard_ids,
            inv_freq=inv_freq,
            rope_concentration=rope_concentration,
            # the paired KV buffers, last in the kernel's signature
            K_cache_0=K_cache_0,
            K_cache_1=K_cache_1,
            K_cache_2=K_cache_2,
            K_cache_3=K_cache_3,
            K_cache_4=K_cache_4,
            K_cache_5=K_cache_5,
            K_cache_6=K_cache_6,
            K_cache_7=K_cache_7,
            K_cache_8=K_cache_8,
            K_cache_9=K_cache_9,
            K_cache_10=K_cache_10,
            K_cache_11=K_cache_11,
            K_cache_12=K_cache_12,
            K_cache_13=K_cache_13,
            K_cache_14=K_cache_14,
            K_cache_15=K_cache_15,
            K_cache_16=K_cache_16,
            K_cache_17=K_cache_17,
            V_cache_0=V_cache_0,
            V_cache_1=V_cache_1,
            V_cache_2=V_cache_2,
            V_cache_3=V_cache_3,
            V_cache_4=V_cache_4,
            V_cache_5=V_cache_5,
            V_cache_6=V_cache_6,
            V_cache_7=V_cache_7,
            V_cache_8=V_cache_8,
            V_cache_9=V_cache_9,
            V_cache_10=V_cache_10,
            V_cache_11=V_cache_11,
            V_cache_12=V_cache_12,
            V_cache_13=V_cache_13,
            V_cache_14=V_cache_14,
            V_cache_15=V_cache_15,
            V_cache_16=V_cache_16,
            V_cache_17=V_cache_17,
        )


# The kernel's paired-buffer KV form: buffer j holds layers 2j and 2j + 1 (up to 18 pairs, 36 layers).
MAX_KV_PAIRS = 18


def gpt_oss_giga_decode(x_shard: Tensor, *, weights: dict, k_pairs: list, v_pairs: list, control: dict,
                        rank_id: Tensor, tp_mesh: tuple, ep_mesh: tuple, scalars: dict, embed_weight: Tensor,
                        embed_ids: Tensor, k_scale=None, v_scale=None):
    """Run the stack. ``weights`` holds the ``[L, ...]`` stacks by the kernel's names, ``k_pairs`` / ``v_pairs`` the
    paired KV buffers (pair ``j``: layer 2j's blocks, then layer 2j + 1's), ``control`` the attention control,
    ``scalars`` the trace-time constants. Returns ``(x_shard_out, k_pairs_out, v_pairs_out)``: the KV outputs are the
    in-place-written buffers, which keep the write live so the graph's aliasing threads it back to the pool."""
    n = len(k_pairs)
    w, c = weights, control
    kv = {f"K_cache_{j}": t for j, t in enumerate(k_pairs)}
    kv.update({f"V_cache_{j}": t for j, t in enumerate(v_pairs)})
    out = wrap_nki(_giga_decode_jit)[1](
        x_shard,
        w["input_layernorm_weight"],
        w["W_qkv"],
        w["bias_qkv"],
        w["W_out"],
        w["bias_out"],
        w["sink"],
        None,
        None,
        w["post_attn_layernorm_weight"],
        w["router_weights"],
        w["router_bias"],
        w["expert_gate_up_weights"],
        w["expert_down_weights"],
        w["expert_gate_up_weights_scale"],
        w["expert_down_weights_scale"],
        w["expert_gate_up_bias"],
        w["expert_down_bias"],
        c["swa_start_pos_ids"],
        c["cos"],
        c["sin"],
        c["active_blocks_table"],
        c["active_blocks_table_swa"],
        c["attention_mask"],
        c["kv_cache_update_idx"],
        c["pos_ids"],
        c["pos_ids_swa"],
        rank_id,
        tp_mesh=tp_mesh,
        ep_mesh=ep_mesh,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_cache_update_idx_swa=c["kv_cache_update_idx_swa"],
        embed_weight=embed_weight,
        embed_ids=embed_ids,
        **kv,
        **scalars,
    )
    if not isinstance(out, (tuple, list)) or len(out) != 1 + 2 * n:
        raise RuntimeError("the giga-kernel returned no KV aliases: its build doesn't write the KV cache in place "
                           "(the KV_UPDATE build, on by default), and only those aliases carry the write to the pool")
    return out[0], list(out[1:1 + n]), list(out[1 + n:])
