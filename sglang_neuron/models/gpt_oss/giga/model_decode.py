# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS MXFP4 decode graph on the giga-kernel: the decoder stack of a decode step in one kernel call.

The graph serves the PD decode at DP16 x attention-TP8 with EP128 (one expert per rank), one logical core per physical
core. A step:

1. builds this rank's vocabulary-shard token ids and the layer-invariant attention control;
2. runs ``ops.gpt_oss_giga_decode`` over all the layers: per layer, TP attention over the packed-FP8 cache (written in
   place), then the EP MoE over the expert mesh; the kernel embeds layer 0's input itself and returns this rank's
   sequence-parallel shard of the post-stack rows;
3. gathers every rank's rows over the world (DP-major: rank ``d * tp + t`` holds rows of DP group ``d``) and runs the
   fused sampling tail (``ops.gpt_oss_giga_tail``): final norm, this rank's vocabulary slice of the LM head (sharded
   over the world, ``vocab / world`` per rank), the logits exchange and the greedy pick, which give every row's token;
   the graph keeps this DP group's rows and returns them as its sampled output.

The weights of every layer are stacked on a leading ``[L, ...]`` axis at load (``_STACKS``), in the per-layer kernels'
layouts with the giga-kernel's differences applied on the host: the PACK64 gate/up scale, the V scale folded into
``W_out`` and the router in float16. The KV pool allocates each layer pair's K and V as one tensor of twice the blocks
(``kv_layout.KVLayout.paired_layers``): layer 2j, a sliding-window layer, addresses the first half and layer 2j + 1, a
full layer, the second, so the full layers' block ids and KV slots are offset by the pool's block count.
"""
import logging

import torch
from sglang.srt.distributed.parallel_state import get_attn_tp_group

from sglang_neuron.layers import ColumnParallelLinear
from sglang_neuron.models.gpt_oss.config import DTYPE
from sglang_neuron.models.gpt_oss.mxfp4 import model_decode as per_layer
from sglang_neuron.models.gpt_oss.weight_loaders_mxfp4 import Q_WIDTH, shuffling_weight_loader
from sglang_neuron.ops import gpt_oss_giga_decode as giga
from sglang_neuron.ops import gpt_oss_giga_tail as giga_tail
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.parallel.state import STATE
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import ShardSpec, set_weight_loader, with_rank_override

from .scale_pack64 import pack64_gate_up_scale

logger = logging.getLogger(__name__)

# Kernel argument -> the per-layer parameter it stacks (module path below ``model.layers.<i>``), in kernel order.
_STACKS = {
    "input_layernorm_weight": "input_layernorm.weight",
    "W_qkv": "self_attn.qkv_proj_weight",
    "bias_qkv": "self_attn.qkv_proj_bias",
    "W_out": "self_attn.o_proj_weight",
    "bias_out": "self_attn.o_proj_bias",
    "sink": "self_attn.sinks",
    "post_attn_layernorm_weight": "mlp.experts.post_attention_layernorm.weight",
    "router_weights": "mlp.experts.router_weight",
    "router_bias": "mlp.experts.router_bias",
    "expert_gate_up_weights": "mlp.experts.gate_up_proj_weight",
    "expert_down_weights": "mlp.experts.down_proj_weight",
    "expert_gate_up_weights_scale": "mlp.experts.gate_up_proj_scale",
    "expert_down_weights_scale": "mlp.experts.down_proj_scale",
    "expert_gate_up_bias": "mlp.experts.gate_up_proj_bias",
    "expert_down_bias": "mlp.experts.down_proj_bias",
}
_STACK_ATTR = "giga_"
# the expert weights tile the per-rank intermediate dimension in tiles of at most this many columns, as the MXFP4
# loaders do (weight_loaders_mxfp4._get_i_tiling_shard_i)
_I_TILE = 512
# the giga-kernel attends with the sliding-window layers on the even ids and the full layers on the odd ids
_SWA_LAYER, _FULL_LAYER = "layers.0.self_attn", "layers.1.self_attn"
# parameters only the prefill graph reads (the fused prefill kernel's router and gamma copies)
_PREFILL_ONLY = ("router_weight_prefill", "gamma_prefill")


class GptOssModel(per_layer.GptOssModel):
    def __init__(self, config):
        super().__init__(config)
        layers = self.layers
        if len(layers) < 2 or len(layers) % 2:
            raise ValueError(f"the giga-kernel decode pairs sliding and full layers: {len(layers)} layers")
        if len(layers) > 2 * giga.MAX_KV_PAIRS:
            raise ValueError(f"the giga-kernel takes at most {giga.MAX_KV_PAIRS} KV layer pairs; {len(layers)} layers")
        experts0 = layers[0].mlp.experts
        if experts0.ep_degree <= 1 or experts0.num_local_experts != 1 or experts0.tp_degree != 1:
            raise ValueError(
                f"the GPT-OSS giga-kernel decode runs one whole expert per rank (expert parallelism over every rank); "
                f"got {experts0.num_local_experts} expert(s) per rank at expert TP {experts0.tp_degree}")
        self.tp_group = get_attn_tp_group()
        self.ep_group = groups.get_neuron_ep_group()
        self.tp_mesh = giga.replica_mesh(self.tp_group)
        self.ep_mesh = giga.replica_mesh(self.ep_group)
        for layer in layers:
            for attr in _PREFILL_ONLY:
                layer.mlp.experts._parameters.pop(attr, None)
        # bound by bind_kv_pairs: each layer pair's K and V tensor, and the blocks of one half
        self.giga_k_pairs = None
        self.giga_v_pairs = None
        self.giga_pair_blocks = None
        # SGLang builds the KV pool after the model, so the pool sees this before it allocates
        from sglang_neuron.model_executor import kv_layout

        kv_layout.declare(kv_layout.KVLayout(paired_layers=True))

    # weights
    def stack_layer_weights(self, host_state: dict, device) -> None:
        """Stack every layer's loaded host tensor of each kernel argument into one ``[L, ...]`` device buffer, popping
        the per-layer entries from ``host_state``, and release the per-layer parameters."""
        layers = self.layers
        attn0 = layers[0].self_attn
        for layer in layers:
            if (layer.self_attn.k_scale_float, layer.self_attn.v_scale_float) != (attn0.k_scale_float,
                                                                                   attn0.v_scale_float):
                raise ValueError("the giga-kernel takes one KV scale for every layer; layer "
                                 f"{layer.self_attn.layer_idx} has its own")
        # the kernel reads its gate/up scales in the PACK64 layout, its build's default; the launch refuses a giga flag
        # it does not pin, so only another kernel build could have frozen them unpacked
        if not giga.scale_pack64():
            raise RuntimeError("the image's giga kernel reads its gate/up MX scales unpacked: this decode lays them "
                               "out in the PACK64 order only (scale_pack64), the kernel build it was validated on")
        e0 = layers[0].mlp.experts
        i_per_rank = e0.gate_up_proj_weight.shape[-1]
        n_i_blocks = -(-i_per_rank // _I_TILE)
        i_block = i_per_rank // n_i_blocks
        v_scale = attn0.v_scale_float

        def per_layer_layout(key, t):
            if key == "router_weights":
                # the giga-kernel's MoE runs the router matmul in float16 (its router_mm_dtype)
                return t.to(torch.float16)
            if key == "router_bias":
                return t.reshape(1, -1).to(torch.float16)
            if key in ("input_layernorm_weight", "bias_qkv", "bias_out", "post_attn_layernorm_weight"):
                return t.reshape(1, -1)
            if key == "sink":
                return t.reshape(-1, 1)
            if key == "W_out":
                # the V dequantization folded into the output projection (a unit scale passes the weight as it is)
                return t if v_scale == 1.0 else (t.float() / v_scale).to(t.dtype)
            if key == "expert_gate_up_weights_scale":
                return pack64_gate_up_scale(t)
            if key == "expert_gate_up_bias":
                # the tiled layout the MoE kernel reshapes against the packed weight's I tiling
                return t.reshape(e0.num_local_experts, i_block // Q_WIDTH, 2, n_i_blocks, Q_WIDTH)
            return t

        for key, path in _STACKS.items():
            names = [f"model.layers.{i}.{path}" for i in range(len(layers))]
            missing = [n for n in names if n not in host_state]
            if missing:
                raise RuntimeError(f"giga-kernel stack {key}: the checkpoint load gave no {missing[:3]}")
            stack = torch.stack([per_layer_layout(key, host_state.pop(n)) for n in names], dim=0).contiguous()
            self.register_buffer(_STACK_ATTR + key, stack.to(device), persistent=False)
        for layer in layers:
            for path in _STACKS.values():
                mod_path, _, attr = path.rpartition(".")
                layer.get_submodule(mod_path)._parameters.pop(attr)
        logger.info("[giga] stacked %d layers x %d weights", len(layers), len(_STACKS))

    # decode step
    def forward(self, input_ids, positions, attn_metadata, rank):
        if self.giga_k_pairs is None:
            raise RuntimeError("the giga-kernel decode graph has no paired KV cache bound (bind_kv_pairs)")
        layers = self.layers
        attn0, experts0 = layers[0].self_attn, layers[0].mlp.experts
        tokens = input_ids.numel()
        tp_size = self.tp_group.world_size
        if tokens % tp_size:
            raise ValueError(f"the giga-kernel decode shards {tokens} decode tokens over TP {tp_size}")
        shard = tokens // tp_size
        # The fused embedding: this rank's vocabulary shard's row of each token, -1 for a token of another shard (the
        # kernel gathers, reduce-scatters and all-gathers over TP; each token lives in exactly one shard).
        embed = self.embed_tokens
        per_rank = embed.vocab_size_per_rank
        local = input_ids.reshape(tokens).to(torch.int32) - rank.reshape(()).to(torch.int32) * per_rank
        embed_ids = torch.where((local >= 0) & (local < per_rank), local, torch.full_like(local, -1))
        x_shard = torch.zeros((shard, embed.weight.shape[1]), dtype=DTYPE, device=input_ids.device)
        control = self._attention_control(positions, attn_metadata, attn0)
        weights = {key: getattr(self, _STACK_ATTR + key) for key in _STACKS}
        out_shard, _k, _v = giga.gpt_oss_giga_decode(
            x_shard,
            weights=weights,
            k_pairs=self.giga_k_pairs,
            v_pairs=self.giga_v_pairs,
            control=control,
            rank_id=self.ep_group.rank_in_group_tensor.reshape(1, 1).to(torch.int32),
            tp_mesh=self.tp_mesh,
            ep_mesh=self.ep_mesh,
            scalars=dict(
                num_layers=len(layers),
                tp_size=tp_size,
                ep_size=self.ep_group.world_size,
                B_attn=control.pop("B_local"),
                S_tkg=control.pop("S_decode"),
                top_k=experts0.num_experts_per_token,
                eps=self.config.rms_norm_eps,
                hidden_actual=attn0.unpadded_hidden_size,
                # the K dequantization folded into the softmax scale, the V one into W_out (stack_layer_weights)
                softmax_scale=attn0.scaling / attn0.k_scale_float,
                gate_clamp_upper_limit=experts0.limit,
                up_clamp_upper_limit=experts0.limit + 1,
                up_clamp_lower_limit=-experts0.limit + 1,
            ),
            k_scale=attn0.k_scale,
            v_scale=attn0.v_scale,
            embed_weight=embed.weight,
            embed_ids=embed_ids,
        )
        return out_shard

    @staticmethod
    def _second_half(index, offset: int):
        """Block ids or slots of a full layer in its pair's tensor: ``offset`` past the sliding-window layer's. A -1
        block-table entry (no block) stays -1; a padding row's slot 0 moves to the second half's reserved page."""
        return torch.where(index >= 0, index + offset, index)

    def _attention_control(self, positions, attn_metadata, attn0) -> dict:
        """The layer-invariant attention inputs: RoPE tables, the full and the window-trimmed block tables with their
        positions, the per-layer window starts, the active-token mask, the full and the sliding layers' KV slots."""
        full, swa = attn_metadata[_FULL_LAYER], attn_metadata[_SWA_LAYER]
        full_bt = full["block_table_tensor"]
        b_local = full_bt.shape[0]
        s_decode = positions.shape[0] // b_local
        half_d = attn0.head_dim // 2
        cos, sin = self.rotary_emb(positions, dtype=DTYPE)

        def rope_layout(t):
            return t[:, :half_d].reshape(b_local, s_decode, half_d).permute(2, 0, 1).contiguous().to(DTYPE)

        pos_flat = positions.reshape(1, b_local * s_decode)
        offset = swa.get("swa_kv_pos_offset")
        swa_pos = pos_flat
        if offset is not None:
            # the trimmed table's frame: each row's position past its offset (a position is never below it)
            swa_pos = torch.clamp(pos_flat - offset.reshape(b_local, 1).expand(b_local, s_decode).reshape(1, -1), min=0)
        pos_ids_swa = swa_pos.reshape(b_local, s_decode).to(torch.float32)
        window = self.config.sliding_window
        start_swa = (torch.clamp(pos_ids_swa - window + 1, min=0) if window is not None
                     else torch.zeros_like(pos_ids_swa))
        zeros = torch.zeros_like(pos_ids_swa)
        swa_start_pos_ids = torch.stack(
            [start_swa if layer.self_attn.sliding_window is not None else zeros for layer in self.layers], dim=0)
        q_heads = attn0.num_attention_heads_per_rank
        causal = torch.tril(torch.ones(s_decode, s_decode, dtype=torch.float32, device=positions.device))
        # the kernel builds its masks itself but still takes this active-token overlay, [S, B, heads, S]
        attention_mask = (
            causal.unsqueeze(0).unsqueeze(0).expand(b_local, q_heads, s_decode, s_decode)
            .permute(3, 0, 1, 2).contiguous().to(torch.uint8)
        )
        return {
            "cos": rope_layout(cos),
            "sin": rope_layout(sin),
            "pos_ids": positions.reshape(b_local, s_decode).to(torch.float32),
            "pos_ids_swa": pos_ids_swa,
            "active_blocks_table": self._second_half(full_bt, self.giga_pair_blocks),
            "active_blocks_table_swa": swa["block_table_tensor"],
            "swa_start_pos_ids": swa_start_pos_ids,
            "attention_mask": attention_mask,
            # the full layers address the second half of each pair's tensor, a padding row its reserved page 0
            "kv_cache_update_idx": self._second_half(full["slot_mapping"], self.giga_pair_blocks * full["block_size"])
            .reshape(b_local, s_decode).to(torch.uint32),
            "kv_cache_update_idx_swa": swa["slot_mapping"].reshape(b_local, s_decode).to(torch.uint32),
            "B_local": b_local,
            "S_decode": s_decode,
        }


class GptOssForCausalLM(per_layer.GptOssForCausalLM):
    phase = "decode"
    model_type = GptOssModel
    # the forward returns this DP group's sampled tokens itself (the fused tail), so no sampler follows it
    samples_in_forward = True

    def __init__(self, config):
        super().__init__(config)
        giga.require()
        # The LM head over the world: rank r holds vocabulary slice r of vocab / world, and reads every DP group's rows.
        world = groups.get_wide_ep_group()
        self.lm_head_tp_group = world
        self.dp_group = groups.get_dp_group()
        self.dp_size = STATE.dp_size
        cfg = self.config
        self.lm_head = ColumnParallelLinear(cfg.hidden_size, cfg.vocab_size, dtype=DTYPE, gather_output=True,
                                            tp_group=world.device_group)
        per_rank = self.lm_head.out_features_per_rank
        set_weight_loader(self.lm_head.weight, with_rank_override(shuffling_weight_loader(
            shuffle_dim=1, expected_shard_shape=(per_rank, cfg.hidden_size),
            shard_spec=ShardSpec(dim=0, size=per_rank, num_shards=self.lm_head.tp_size),
        ), rank=world.rank_in_group))
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        giga_tail.require(
            vocab=cfg.vocab_size, world=world.world_size, lnc=get_logical_nc_config(), hidden=cfg.hidden_size,
            hidden_actual=cfg.unpadded_hidden_size, per_rank=per_rank, mesh=giga.replica_mesh(world),
            compiler=giga_tail.compiler_version(),
        )
        self.tail_hops = giga_tail.HOPS
        logger.info("[giga] GPT-OSS giga-kernel decode: %d layers, attention TP %d x DP %d, EP %d, LM head %d / rank "
                    "over %d, fused tail (exchange hops %s)", len(self.model.layers), self.model.tp_group.world_size,
                    self.dp_size, self.model.ep_group.world_size, per_rank, world.world_size, giga_tail.HOPS)

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        shard = self.model(input_ids, positions.to(torch.int32), attn_metadata, rank)
        # every rank's rows, DP-major (rank d * tp + t holds DP group d's rows t * shard ...)
        rows = neuron_collectives.all_gather(self.lm_head_tp_group, shard, dim=0)
        # every row's token, in the order of rows
        tokens = giga_tail.batch_tokens(
            rows, group=self.lm_head_tp_group, gamma=self.tail_norm_weight, lm_head_weight_t=self.tail_lm_head_t,
            hops=self.tail_hops, eps=self.model.norm.variance_epsilon, hidden_actual=self.config.unpadded_hidden_size,
        )
        return {"token_ids": groups.shard_select(tokens, 0, self.dp_group, n_shards=self.dp_size)}

    # weights
    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        """Load this rank's shard of every parameter on the host, stack the layers' weights onto the device, build the
        fused tail's operands, and place the rest (embedding, final norm, LM head) on the device."""
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        host_state = checkpoint.load_sharded_pipelined(
            self.tp_group.rank_in_group, self.world_size, self, self.hf_weight_mappings(), torch.device("cpu")
        ).state_dict
        self._load_kv_cache_scales(checkpoint, device)
        self.model.stack_layer_weights(host_state, device)
        self.load_fused_tail(host_state, device)
        self.load_state_dict({name: t.to(device) for name, t in host_state.items()}, strict=False, assign=True)

    def load_fused_tail(self, host_state: dict, device) -> None:
        """The fused tail's operands, built once from the loaded host tensors: the LM-head shard as the [H, V_s] bf16
        C-contiguous mirror the kernel reads and the final norm weight as [1, H] float32 (both in the shuffled hidden
        order of the rows)."""
        missing = [n for n in ("lm_head.weight", "model.norm.weight") if n not in host_state]
        if missing:
            raise RuntimeError(f"the giga-kernel fused tail: the checkpoint load gave no {missing}")
        self.register_buffer("tail_lm_head_t", giga_tail.lm_head_mirror(host_state["lm_head.weight"]).to(device),
                             persistent=False)
        gamma = host_state["model.norm.weight"].detach().reshape(1, -1).to(torch.float32)
        self.register_buffer("tail_norm_weight", gamma.to(device), persistent=False)

    # KV cache
    def bind_kv_pairs(self, layer_pairs) -> None:
        """Bind the per-pair K and V tensors the pool allocated the layers' caches in (``pool.layer_pairs``), after
        ``bind_kv_cache`` bound each layer's half."""
        if layer_pairs is None:
            raise RuntimeError("the giga-kernel decode needs the KV pool's paired layout (kv_layout.paired_layers)")
        k_pairs, v_pairs = layer_pairs
        layers = self.model.layers
        if len(k_pairs) * 2 != len(layers) or len(v_pairs) * 2 != len(layers):
            raise RuntimeError(f"{len(k_pairs)} KV layer pairs for a {len(layers)}-layer graph")
        nb = layers[0].self_attn.k_cache.shape[0]
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            for half, pair in ((attn.k_cache, k_pairs[i // 2]), (attn.v_cache, v_pairs[i // 2])):
                if (half.shape[0] != nb or pair.shape[0] != 2 * nb or half.shape[1:] != pair.shape[1:]
                        or half.storage_offset() != (i % 2) * nb * pair.stride(0)
                        or half.untyped_storage().data_ptr() != pair.untyped_storage().data_ptr()):
                    raise RuntimeError(f"layer {attn.layer_idx}'s KV cache is not half {i % 2} of pair {i // 2}")
        self.model.giga_k_pairs, self.model.giga_v_pairs = list(k_pairs), list(v_pairs)
        self.model.giga_pair_blocks = nb
