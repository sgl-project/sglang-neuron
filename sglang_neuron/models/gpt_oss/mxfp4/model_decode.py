# SPDX-License-Identifier: Apache-2.0
"""The GPT-OSS MXFP4 decode graph: one fused attention kernel and one fused MoE kernel per layer, the masks built once
per step, and under DP attention with EP the tokens of every replica exchanged around the experts."""

import torch

from sglang_neuron.models.gpt_oss.attention_masks import decode_attention_masks
from sglang_neuron.models.gpt_oss.config import DTYPE
from sglang_neuron.ops.attention.attention_decode import (
    attention_decode,
    packed_fp8_viable_for_bucket,
    swizzle_packed_k,
    unswizzle_packed_k,
)
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel.groups import shard_select

from . import common


class GptOssAttention(common.GptOssAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        if self.num_key_value_heads_per_rank != 1:
            raise ValueError(f"the GPT-OSS decode attends one KV head per rank: serve it at an attention TP of at "
                             f"least {config.num_key_value_heads}, not {self.world_size}")

    def forward(self, hidden_states, position_embeddings, attn_metadata, attn_mask, norm, rsag_dispatch: bool):
        """Attention of this rank's heads, with the layer's input norm ``norm`` fused in. Returns the output reduced
        over attention TP: summed on every rank, or with ``rsag_dispatch`` reduce-scattered to a token shard."""
        md = attn_metadata[f"layers.{self.layer_idx}.self_attn"]
        block_table = md["block_table_tensor"]
        page = md["block_size"]
        rows = block_table.shape[0]
        s_active = hidden_states.shape[0] // rows
        cos, sin = position_embeddings
        half = self.head_dim // 2
        # [T, half] -> [half, B, S]
        cos = cos[:, :half].view(rows, s_active, half).permute(2, 0, 1).contiguous().to(DTYPE)
        sin = sin[:, :half].view(rows, s_active, half).permute(2, 0, 1).contiguous().to(DTYPE)

        # Where the kernel cannot read the packed FP8 K cache at this bucket, it reads an unpacked copy, and the K it
        # writes is packed back into the cache. The copy is a graph temporary, so its write comes back as an output.
        # (under DCP the kernel attends the group's gathered query heads)
        unpack = not packed_fp8_viable_for_bucket(
            block_len=page, bs=rows, q_head=self.num_attention_heads_per_rank * self.dcp_size, s_active=s_active,
            s_prior=md["max_blocks_per_seq"] * page,
        )
        result = attention_decode(
            hidden_states.to(DTYPE).view(rows, s_active, -1),
            X_hidden_dim_actual=self.unpadded_hidden_size,
            rmsnorm_X_gamma=norm.weight.unsqueeze(0).to(torch.float32),
            rmsnorm_X_eps=norm.variance_epsilon,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=self.qkv_proj_bias.unsqueeze(0),
            cos=cos,
            sin=sin,
            active_blocks_table=block_table,
            K_cache=unswizzle_packed_k(self.k_cache) if unpack else self.k_cache,
            V_cache=self.v_cache,
            attention_mask=attn_mask,
            # the cache holds K times its inverted scale; dequantization folds into the softmax scale and, for V,
            # into the output projection
            softmax_scale=self.scaling / self.k_scale_float,
            sink=self.sinks.unsqueeze(1),
            kv_cache_update_idx=md["slot_mapping"].view(rows, s_active).to(torch.uint32),
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            fp8_packed=not unpack,
            W_out=self.o_proj_weight if self.v_scale_float == 1.0 else self.o_proj_weight / self.v_scale_float,
            bias_out=self.o_proj_bias.unsqueeze(0),
            return_k_cache=unpack,
            dcp_group=self.dcp_group,
        )
        if unpack:
            out, k_written = result
            self.k_cache.copy_(swizzle_packed_k(k_written))
        else:
            out = result
        if rsag_dispatch and self.world_size > 1:
            return neuron_collectives.reduce_scatter(self.attn_tp_group, out, dim=0)
        return neuron_collectives.all_reduce(self.attn_tp_group, out)


class GptOssExperts(common.GptOssExperts):
    def forward(self, hidden_states, rsag_dispatch: bool):
        """The MoE of this replica's tokens. Under cross-DP EP the replicas' tokens are gathered first and the results
        combined after: by AllGather[DP] and AllReduce over the world then this replica's slice, or, with
        ``rsag_dispatch``, by AllGather and ReduceScatter over the world, which leaves a token shard."""
        if self.cross_dp_ep:
            group = self.wide_ep_group if rsag_dispatch else self.dp_group
            hidden_states = neuron_collectives.all_gather(group, hidden_states, dim=0)
        out = self._run_moe_block_tkg(hidden_states)
        if not self.cross_dp_ep:
            return neuron_collectives.all_reduce(self.moe_group, out) if self.moe_group.world_size > 1 else out
        if rsag_dispatch:
            return neuron_collectives.reduce_scatter(self.wide_ep_group, out, dim=0)
        return shard_select(self._wide_ep_all_reduce(out), 0, self.dp_group, n_shards=self.dp_size)


class GptOssMLP(common.GptOssMLP):
    experts_type = GptOssExperts

    def forward(self, hidden_states, rsag_dispatch: bool):
        return self.experts(hidden_states.to(DTYPE), rsag_dispatch)


class GptOssDecoderLayer(common.GptOssDecoderLayer):
    attention_type = GptOssAttention
    mlp_type = GptOssMLP

    def forward(self, hidden_states, position_embeddings, attn_metadata, attn_mask):
        # attention and the MoE agree on one DP-to-EP transition
        rsag_dispatch = self._use_rsag_dispatch(hidden_states)
        residual = hidden_states
        attn_out = self.self_attn(hidden_states, position_embeddings, attn_metadata, attn_mask,
                                  self.input_layernorm, rsag_dispatch)
        if rsag_dispatch:
            # attention left a token shard: take the residual's matching rows (a slice, since the residual is the same
            # on every TP rank)
            residual = shard_select(residual, 0, self.tp_group)
        hidden_states = residual + attn_out
        hidden_states = hidden_states + self.mlp(hidden_states, rsag_dispatch)
        if rsag_dispatch:
            hidden_states = neuron_collectives.all_gather(self.tp_group, hidden_states, dim=0)
        return hidden_states

    def _use_rsag_dispatch(self, hidden_states) -> bool:
        """Whether the cross-DP EP transition is ReduceScatter[TP] + AllGather[world] rather than AllReduce[TP] +
        AllGather[DP]: with at least DECODE_RSAG_TOKEN_THRESHOLD tokens over the fleet, sharding evenly over it."""
        experts = self.mlp.experts
        if not experts.cross_dp_ep:
            return False
        fleet_tokens = hidden_states.shape[0] * experts.dp_size
        return (fleet_tokens >= common.DECODE_RSAG_TOKEN_THRESHOLD
                and fleet_tokens % (self.world_size * experts.dp_size) == 0)


class GptOssModel(common.GptOssModel):
    layer_type = GptOssDecoderLayer

    def __init__(self, config, stage=None):
        super().__init__(config, stage)
        attns = [layer.self_attn for layer in self.layers]
        # the masks' shapes come from one layer of each attention kind
        self.swa_layer = next((f"layers.{a.layer_idx}.self_attn" for a in attns if a.sliding_window is not None), None)
        self.full_layer = next((f"layers.{a.layer_idx}.self_attn" for a in attns if a.sliding_window is None), None)

    def forward(self, input_ids, positions, attn_metadata, rank, pipeline_hidden_states=None,
                pipeline_hidden_states_out=None):
        """The final-normed hidden states; on a stage before the last, the last layer's output (TP-replicated, a row
        per sequence) written into ``pipeline_hidden_states_out`` and returned. The runtime checks the write landed in
        place, so no device copy runs between stages. Later stages start from ``pipeline_hidden_states``."""
        if not self.is_last_stage and pipeline_hidden_states_out is None:
            raise ValueError("a decode pipeline stage before the last writes its boundary into the seam slot it is "
                             "handed (pipeline_hidden_states_out)")
        if self.is_first_stage:
            hidden_states = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        elif pipeline_hidden_states is None:
            raise ValueError("a decode pipeline stage after the first needs pipeline_hidden_states")
        else:
            hidden_states = pipeline_hidden_states
        position_embeddings = self.rotary_emb(positions, dtype=hidden_states.dtype)
        attn = self.layers[0].self_attn
        swa_mask, full_mask = decode_attention_masks(
            attn_metadata, positions, sliding_window=self.config.sliding_window,
            q_heads=attn.num_attention_heads_per_rank * attn.dcp_size,
            swa_layer=self.swa_layer, full_layer=self.full_layer,
        )
        for layer in self.layers:
            mask = swa_mask if layer.self_attn.sliding_window is not None else full_mask
            hidden_states = layer(hidden_states, position_embeddings, attn_metadata, mask)
        if not self.is_last_stage:
            if (tuple(pipeline_hidden_states_out.shape) != tuple(hidden_states.shape)
                    or pipeline_hidden_states_out.dtype != hidden_states.dtype):
                raise ValueError(f"the decode boundary is {tuple(hidden_states.shape)} {hidden_states.dtype}, its seam "
                                 f"slot {tuple(pipeline_hidden_states_out.shape)} {pipeline_hidden_states_out.dtype}")
            pipeline_hidden_states_out.copy_(hidden_states)
            return {"pipeline_hidden_states": pipeline_hidden_states_out}
        return self.norm(hidden_states)


class GptOssForCausalLM(common.GptOssForCausalLM):
    phase = "decode"
    model_type = GptOssModel

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank, pipeline_hidden_states=None,
                pipeline_hidden_states_out=None):
        """The sampled rows' logits, or on a stage before the last its boundary (written into the send slot)."""
        hidden_states = self.model(input_ids, positions.to(torch.int32), attn_metadata, rank, pipeline_hidden_states,
                                   pipeline_hidden_states_out)
        if not self.is_last_stage:
            return hidden_states
        return self.lm_head(hidden_states.index_select(0, sampling_positions).to(DTYPE))
