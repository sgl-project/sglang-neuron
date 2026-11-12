# SPDX-License-Identifier: Apache-2.0
"""The GPT-OSS MXFP4 prefill graph: one request per forward, sequence-parallel over attention TP between the layers,
attention segmented over the paged KV cache with the K/V write inside the kernels."""

import torch

import sglang_neuron.ops as NF
from sglang_neuron.models.gpt_oss.config import DTYPE
from sglang_neuron.models.gpt_oss.sequence_parallel import select_sampled_rows
from sglang_neuron.ops.attention.o_proj import MAX_HEADS
from sglang_neuron.ops.attention.qkv import MAX_FUSED_QKV
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel.groups import shard_select

from . import common


class GptOssAttention(common.GptOssAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # the prefill kernels' limits on one rank's share of the attention
        width, limit = self.qkv_proj_weight.shape[1], min(MAX_FUSED_QKV, self.hidden_size)
        if width > limit or self.num_attention_heads_per_rank > MAX_HEADS:
            raise ValueError(f"at an attention TP of {self.world_size} a GPT-OSS prefill rank holds "
                             f"{self.num_attention_heads_per_rank} heads and a QKV of {width} columns, and the "
                             f"kernels take at most {MAX_HEADS} and {limit}: serve the prefill at a larger attention "
                             "TP")

    def forward(self, hidden_states, position_embeddings, attn_metadata):
        md = attn_metadata[f"layers.{self.layer_idx}.self_attn"]
        if self.world_size > 1:
            hidden_states = neuron_collectives.all_gather(self.tp_group, hidden_states, dim=0)
        hidden_states = hidden_states.to(DTYPE)
        tokens = hidden_states.shape[0]
        page = md["block_size"]
        block_table = md["block_table_tensor"]
        # the kernels' RoPE takes cos/sin over the whole head, the halves repeated
        cos, sin = position_embeddings
        cos = torch.cat([cos, cos], dim=-1).unsqueeze(0)
        sin = torch.cat([sin, sin], dim=-1).unsqueeze(0)
        # The prefill kernels quantize by dividing by the scale they are given; k_scale is the inverted checkpoint
        # scale the decode multiplies by, so they get its inverse and write the same bytes.
        k_scale, v_scale = 1.0 / self.k_scale, 1.0 / self.v_scale

        if self.fuse_swa:
            # QKV, RoPE, windowed attention over the prior with sinks, the K/V write and the output projection
            out, _, _ = NF.swa_fused_attention(
                hidden_states=hidden_states.unsqueeze(0),
                qkv_weight=self.qkv_proj_weight,
                op_weight=self.o_proj_weight,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table.to(torch.int32),
                cos_cache=cos,
                sin_cache=sin,
                sink=self.sinks.reshape(1, self.num_attention_heads_per_rank),
                prior_tokens=md["cached_seq_len"].to(torch.int32).reshape(1, 1),
                qkv_bias=self.qkv_proj_bias.unsqueeze(0),
                op_bias=self.o_proj_bias.unsqueeze(0),
                # the kernel dequantizes the cache itself, so this is the plain softmax scale
                scale=self.scaling,
                sliding_window=self.sliding_window,
                block_size=page,
                num_q_heads=self.num_attention_heads_per_rank,
                num_kv_heads=self.num_key_value_heads_per_rank,
                d_head=self.head_dim,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            out = out.squeeze(0)
        else:
            # The runtime pads with slot 0, on SGLang's reserved page. The QKV kernel writes K/V at its slots as DMA
            # offsets with no bounds check, so any slot outside the pool goes to slot 0 too instead of past the cache.
            slots = md["slot_mapping"]
            slots = torch.where((slots < 0) | (slots >= self.k_cache.shape[0] * page), torch.zeros_like(slots), slots)
            q, _, _ = NF.qkv_proj(
                hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=self.qkv_proj_bias.unsqueeze(0),
                d_head=self.head_dim,
                cos_cache=cos,
                sin_cache=sin,
                num_q_heads=self.num_attention_heads_per_rank,
                num_kv_heads=self.num_key_value_heads_per_rank,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                k_scale=k_scale,
                v_scale=v_scale,
                block_size=page,
                slot_mapping=slots.to(torch.int32).reshape(1, tokens),
            )
            q = q.squeeze(0).view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
            # The cache holds K and V times their (inverted) scales; the K scale folds into the softmax scale and the
            # V scale is divided out of the output.
            out = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table,
                prior_tokens=md["cached_seq_len"],
                block_size=page,
                kv_segment_size=md["kv_segment_size"],
                scale=self.scaling / self.k_scale_float,
                sliding_window=self.sliding_window,
                sink=self.sinks.unsqueeze(1),
            )
            if self.v_scale_float != 1.0:
                out = out / self.v_scale_float
            out = NF.o_proj(out.unsqueeze(0), self.o_proj_weight, self.o_proj_bias.unsqueeze(0)).squeeze(0)
        if self.world_size > 1:
            out = neuron_collectives.reduce_scatter(self.tp_group, out, dim=0)
        return out.contiguous()


class GptOssExperts(common.GptOssExperts):
    def forward(self, hidden_states, positions, residual):
        """The MoE of this rank's sequence shard; returns ``(out, residual + hidden_states)``."""
        # One kernel adds the residual, norms, MX-quantizes and routes. Its row packs [hidden | scale | affinities] as
        # bf16, so the gathers below carry the affinities with the tokens.
        packed, _, summed = NF.rmsnorm_mx_prefill(
            hidden_states.to(DTYPE),
            residual=residual,
            gamma=self.gamma_prefill,
            router_weights=self.router_weight_prefill.T,
            router_bias=self.router_bias,
            eps=self.rms_norm_eps,
            top_k=self.num_experts_per_token,
            unpadded_hidden_size=self.unpadded_hidden_size,
        )
        if self.tp_group.world_size > 1:
            packed = neuron_collectives.all_gather(self.tp_group, packed, dim=0)
        # Padding trails the request's tokens, at the last token's position; it is masked here because the replicas'
        # positions are not ordered once concatenated. argmax may return any index of that tie, so the mask keeps every
        # token and at most some padding, whose expert outputs nothing reads.
        real = torch.arange(positions.shape[0], device=positions.device) <= torch.argmax(positions)
        if self.cross_dp_ep:
            packed = neuron_collectives.all_gather(self.dp_group, packed, dim=0)
            real = neuron_collectives.all_gather(self.dp_group, real, dim=0)

        affinities = NF.slice_affinities_from_packed(packed, hidden_size=self.hidden_size,
                                                     num_experts=self.total_num_experts)
        # moe_cte reads the packed affinities by global expert, from the local expert and the EP rank tensor, which it
        # needs even without EP
        ep_rank = self._ep_rank_id()
        token_position_to_id, block_to_expert, conditions = self._blockwise_mapping(
            self._local_affinities(affinities, ep_rank), real)
        out = NF.moe_cte(
            packed,
            gate_up_proj_weight=self.gate_up_proj_weight,
            down_proj_weight=self.down_proj_weight,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            block_size=self.block_size,
            top_k=self.num_experts_per_token,
            ep_degree=self.ep_degree,
            ep_rank=ep_rank,
            conditions=torch.cat([conditions, torch.zeros(2, device=conditions.device)]),
            gate_and_up_proj_bias=self._gate_up_bias_tiles(),
            down_proj_bias=self.down_proj_bias,
            gate_up_proj_scale=self.gate_up_proj_scale,
            down_proj_scale=self.down_proj_scale,
            **self._clamps(),
        )
        return self._combine(out), summed

    def _local_affinities(self, affinities, ep_rank):
        """This rank's experts' [T, E_local] columns of the [T, E] affinities."""
        if self.ep_degree == 1:
            return affinities
        # a 1-D rank: dynamo folds a 0-D one into a per-rank constant, which would compile one graph per rank
        local = (torch.arange(self.num_local_experts, device=affinities.device, dtype=torch.int32)
                 + ep_rank.reshape(1) * self.num_local_experts)
        return affinities.index_select(1, local)

    def _blockwise_mapping(self, affinities, real):
        """moe_cte's token-to-block mapping of the ``real`` tokens over this rank's experts, from [T, E_local]
        affinities."""
        return NF.build_blockwise_mapping(
            affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.num_experts_per_token,
            block_size=self.block_size,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
            padding_mask=real,
        )

    def _combine(self, out):
        """Sum the gathered tokens' expert outputs over the expert ranks, back to this rank's sequence shard."""
        if self.cross_dp_ep:
            # back to this replica's tokens, then this rank's sequence shard; the offsets come from the groups' rank
            # tensors, so every rank traces one graph
            out = self._wide_ep_all_reduce(out)
            out = shard_select(out, 0, self.dp_group, n_shards=self.dp_size)
            return shard_select(out, 0, self.moe_group)
        if self.moe_group.world_size > 1:
            return neuron_collectives.reduce_scatter(self.moe_group, out, dim=0)
        return out


class GptOssMLP(common.GptOssMLP):
    experts_type = GptOssExperts

    def forward(self, hidden_states, positions, residual):
        return self.experts(hidden_states, positions, residual)


class GptOssDecoderLayer(common.GptOssDecoderLayer):
    attention_type = GptOssAttention
    mlp_type = GptOssMLP

    def forward(self, hidden_states, residual, positions, position_embeddings, attn_metadata):
        """Returns ``(mlp_out, residual)`` without adding them: the next layer's input norm, or the final norm, fuses
        the add. ``residual`` is None for the first layer."""
        norm = self.input_layernorm
        normed, summed = NF.rmsnorm_residual_prefill(
            hidden_states, gamma=norm.weight, residual=residual, eps=norm.variance_epsilon,
            unpadded_hidden_size=norm.unpadded_hidden_size,
        )
        # the first layer has no add to fuse: its input is the residual
        residual = hidden_states if residual is None else summed
        attn_out = self.self_attn(normed, position_embeddings, attn_metadata)
        return self.mlp(attn_out, positions, residual)


class GptOssModel(common.GptOssModel):
    layer_type = GptOssDecoderLayer

    def forward(self, input_ids, positions, attn_metadata, rank, pipeline_hidden_states=None, pipeline_residual=None):
        """The final-normed hidden states; on a stage before the last, the last layer's output and residual, unadded
        (the next stage's first input norm fuses the add). Later stages start from the previous stage's boundary."""
        if self.is_first_stage:
            # sequence-parallel from the embedding on: each attention TP rank holds a contiguous shard of the tokens
            hidden_states = self.embed_tokens(input_ids, scatter_tokens=True, rank=rank)
            residual = None
        else:
            if pipeline_hidden_states is None or pipeline_residual is None:
                raise ValueError("a prefill pipeline stage after the first needs pipeline_hidden_states and "
                                 "pipeline_residual")
            hidden_states, residual = pipeline_hidden_states, pipeline_residual
        position_embeddings = self.rotary_emb(positions, dtype=hidden_states.dtype)
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, positions, position_embeddings, attn_metadata)
        if not self.is_last_stage:
            return {"pipeline_hidden_states": hidden_states, "pipeline_residual": residual}
        hidden_states, _ = NF.rmsnorm_residual_prefill(
            hidden_states, gamma=self.norm.weight, residual=residual, eps=self.norm.variance_epsilon,
            unpadded_hidden_size=self.norm.unpadded_hidden_size,
        )
        return hidden_states


class GptOssForCausalLM(common.GptOssForCausalLM):
    phase = "prefill"
    model_type = GptOssModel

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank, pipeline_hidden_states=None,
                pipeline_residual=None, pipeline_hidden_states_out=None, pipeline_residual_out=None):
        """The sampled rows' logits, or on a stage before the last its boundary written in place into the ``*_out``
        slots and returned; the runtime checks the write landed in place (NeuronModelRuntime._check_boundary_in_slots)
        and sends the slots on."""
        positions = positions.to(torch.int32)
        hidden_states = self.model(input_ids, positions, attn_metadata, rank, pipeline_hidden_states,
                                   pipeline_residual)
        if not self.is_last_stage:
            if pipeline_hidden_states_out is None or pipeline_residual_out is None:
                raise ValueError("a prefill pipeline stage before the last writes its boundary into the seam slots "
                                 "it is handed (pipeline_hidden_states_out, pipeline_residual_out)")
            for name, slot in (("pipeline_hidden_states", pipeline_hidden_states_out),
                               ("pipeline_residual", pipeline_residual_out)):
                # copy_ would silently convert the dtype or broadcast the shape
                boundary = hidden_states[name]
                if tuple(slot.shape) != tuple(boundary.shape) or slot.dtype != boundary.dtype:
                    raise ValueError(f"the prefill boundary's {name} is {tuple(boundary.shape)} {boundary.dtype}, its "
                                     f"seam slot {tuple(slot.shape)} {slot.dtype}")
                slot.copy_(boundary)
            return {"pipeline_hidden_states": pipeline_hidden_states_out, "pipeline_residual": pipeline_residual_out}
        if self.world_size > 1:
            sampled = select_sampled_rows(hidden_states, sampling_positions, self.tp_group)
        else:
            sampled = hidden_states.index_select(0, sampling_positions)
        return self.lm_head(sampled.to(DTYPE))
