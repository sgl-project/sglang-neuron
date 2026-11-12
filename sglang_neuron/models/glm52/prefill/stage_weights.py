# SPDX-License-Identifier: Apache-2.0
"""Weights of one GLM-5.2 prefill pipeline stage on the prefill giga kernel.

The tree keeps the ``GlmMoeDsaForCausalLM`` attribute names (``.config``, ``.model.layers``, ``.model.embed_tokens``,
``.model.norm_weight``, ``.model.mtp_kv``, ``.lm_head``) for the stage's layers, so the prefill causal-LM loaders
(layout/prefill/weight_loaders) run on it unedited and each rank loads the same bytes those loaders give the whole model.

The module classes hold only weights and the static facts their loaders and the stage graph read (``is_moe``,
``layer_idx``, the stage's world and the MoE mesh); the graph takes its scalars from the stage config. Parameter
shapes come from the same recipe knobs the loaders read
(``GLM_GK_ATTN_RESIDENT``, ``GLM_GK_PRJ_OMX``, ``GLM_GK_SHARED_TP``, ``GLM_GK_MOE_PLACEMENT``, ...). Process groups,
compute methods and the paged key slots are not here: the stage graph builds the first two from the ``StageLayout``
and the KV pool owns the slots. The world is the stage's CP degree and the EP degree is ``options.ep_degree``, passed
in rather than read from a process group. ``stack_weight_params`` stacks each launch group's layers so a launch takes
one tensor per kernel argument.

A stage holds only the extra pieces its layout uses: the embedding when it embeds or writes the MTP rows, the final
norm and lm_head when it samples, the MTP layer when it writes the MTP rows (about 119 MB a rank saved on stages 1
and 2).

Every parameter is built on the meta device: 128 ranks share one host, so a host copy of each stage would cost host
memory and a second copy at load. ``load_state_dict(assign=True)`` replaces each with its device tensor.
"""

from __future__ import annotations

import torch
from torch import nn

from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.layers.linear import ColumnParallelLinear
from sglang_neuron.models.glm52.layout.prefill.mtp_kv import GlmMoeDsaMtpKv
from sglang_neuron.models.glm52.prefill.expert_placements import check_placement
from sglang_neuron.models.glm52.layout.prefill.weight_loaders import (
    CHUNKED_ATTN_WEIGHTS,
    attn_resident,
    hybrid_tp,
    placement_e_local,
    position_slice_shape,
    prj_omx,
    shared_tp,
    stacked_loader,
    world_chunk_shape,
)
from sglang_neuron.utils.weight_loader import get_weight_loader, set_weight_loader

# Consecutive ranks of one row-parallel o_proj / tensor-parallel dense-MLP group: o_proj fits SBUF at 8, not 64 or 128.
WO_GROUP_SIZE = 8
# The MX expert kernel's SBUF partitions and contraction tile.
_PMAX = 128
_MX_TILE = 512


def _param(*shape, dtype) -> nn.Parameter:
    """An uninitialized frozen parameter (on the meta device inside Glm52StageWeights)."""
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


class GlmMoeDsaIndexer(nn.Module):
    """The weights of the DSA indexer of one full layer.

    The index heads are replicated, so every rank picks the same top-k. The indexer's ``k_cache`` slot is not here: the
    kernel's bf16 key stacks and the fp8 key caches belong to the KV pool.
    """

    def __init__(self, config):
        super().__init__()
        n_heads, head_dim, dtype = config.index_n_heads, config.index_head_dim, config.torch_dtype
        hidden = config.hidden_size
        self.wq_b = _param(config.q_lora_rank, n_heads * head_dim, dtype=dtype)
        self.wk = _param(hidden, head_dim, dtype=dtype)
        # The kernel takes the LayerNorm statistics of K in fp32.
        self.k_norm_weight = _param(head_dim, dtype=torch.float32)
        self.k_norm_bias = _param(head_dim, dtype=torch.float32)
        # Stored transposed, [H, n_heads]: the kernel loads it without transposes.
        self.weights_proj = _param(hidden, n_heads, dtype=dtype)


class GlmMoeDsaAttention(nn.Module):
    """The weights of the attention of one layer.

    ``wqkv_a``, ``wq_b``, ``wuk`` and ``wuv`` are this rank's piece of its wo position's slice (``hybrid_tp``), or the
    whole slice for the names ``GLM_GK_ATTN_RESIDENT`` keeps resident; ``wo`` is the row shard of the rank's wo
    position, MXFP8 words and their e8m0 scales under ``GLM_GK_PRJ_OMX``. A "shared" layer reuses the top-k of the
    previous full layer and has no indexer. ``world_size`` is the stage's CP degree, the size of its TP group under PP.
    """

    def __init__(self, config, layer_idx: int, wo_group_size: int, world_size: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.world_size = world_size
        shared = config.indexer_types[layer_idx] == "shared"
        self.indexer = None if shared else GlmMoeDsaIndexer(config)

        hidden, n_heads, dtype = config.hidden_size, config.num_attention_heads, config.torch_dtype
        q_lora, kv_lora = config.q_lora_rank, config.kv_lora_rank
        full = {
            "wqkv_a": (hidden, q_lora + kv_lora + config.qk_rope_head_dim),
            "wq_b": (q_lora, n_heads * config.qk_head_dim),
            "wuk": (config.qk_nope_head_dim, n_heads * kv_lora),
            "wuv": (n_heads * kv_lora, config.v_head_dim),
        }
        resident = attn_resident(world_size, wo_group_size)
        for name in CHUNKED_ATTN_WEIGHTS:
            if name in resident:
                # GLM_GK_ATTN_RESIDENT: the whole slice of this rank's wo position (no in-kernel gather).
                shape = position_slice_shape(name, full[name], config, wo_group_size)
            elif hybrid_tp(world_size, wo_group_size):
                # This rank's chunk of its wo position's slice (see hybrid_tp).
                piece = position_slice_shape(name, full[name], config, wo_group_size)
                shape = world_chunk_shape(*piece, world_size // wo_group_size)
            else:
                shape = world_chunk_shape(*full[name], world_size)
            setattr(self, name, _param(*shape, dtype=dtype))
        # the rank's 1/wo_group_size row shard of o_proj
        wo_rows = n_heads * config.v_head_dim // wo_group_size
        if prj_omx():
            # GLM_GK_PRJ_OMX: MXFP8 fp8x4 words and e8m0 scales (weight_loaders.wo_mx_pack), the head-TP o_proj only.
            if not hybrid_tp(world_size, wo_group_size):
                raise ValueError("GLM_GK_PRJ_OMX=1 needs the head-TP o_proj: a stage of more than one o_proj group.")
            self.wo = _param(wo_rows // 4, hidden, dtype=torch.int32)
            self.wo_scale = _param(wo_rows // 32, hidden, dtype=torch.uint8)
        else:
            self.wo = _param(wo_rows, hidden, dtype=dtype)
        self.q_norm_weight = _param(q_lora, dtype=dtype)
        self.kv_norm_weight = _param(kv_lora, dtype=dtype)


class GlmMoeDsaMLP(nn.Module):
    """The weights of the dense SwiGLU MLP of one layer, with its post-attention norm.

    The MLP is tensor-parallel over the wo group: gate and up are [H, I / g], down is [I / g, H].
    """

    def __init__(self, config, group_size: int):
        super().__init__()
        hidden, inter, dtype = config.hidden_size, config.intermediate_size, config.torch_dtype
        if inter % group_size:
            raise ValueError(f"intermediate_size {inter} must be a multiple of the group size {group_size}.")
        self.gate_proj = _param(hidden, inter // group_size, dtype=dtype)
        self.up_proj = _param(hidden, inter // group_size, dtype=dtype)
        self.down_proj = _param(inter // group_size, hidden, dtype=dtype)
        self.post_attention_layernorm_weight = _param(hidden, dtype=dtype)


class GlmMoeDsaMoE(nn.Module):
    """The weights of one MXFP4 MoE block and its mesh: norm, router, routed experts, shared expert (the ``mlp`` of a
    sparse layer).

    The routed experts use an EP x TP mesh over the stage, in the layout of ``block_moe_mx`` with packed MX
    scales; with ``GLM_GK_MOE_PLACEMENT``'s ``e_local`` a rank holds more slots than experts per rank (the uneven
    placement: an empty slot's router column is zero and its bias ``EMPTY_SLOT_BIAS``). The shared expert is a
    one-expert blockwise-MX MoE, tensor-parallel over ``shared_tp`` ranks.
    """

    def __init__(self, config, world_size: int, ep_degree: int):
        super().__init__()
        self.shared_tp = shared_tp(world_size, min(WO_GROUP_SIZE, world_size))
        self.n_routed_experts = config.n_routed_experts
        # EP x TP mesh over the stage: ep_degree is every rank of the stage with expert parallel on, 1 without it.
        if ep_degree < 1 or world_size % ep_degree:
            raise ValueError(f"The expert-parallel degree {ep_degree} must divide the stage's {world_size} ranks.")
        self.ep_degree, self.tp_degree = ep_degree, world_size // ep_degree
        if self.n_routed_experts % self.ep_degree:
            raise ValueError(f"The routed expert count {self.n_routed_experts} must be a multiple of the "
                             f"expert-parallel degree {self.ep_degree}.")
        # GLM_GK_MOE_PLACEMENT with "e_local": more slots than experts per EP rank (empty slots: an uneven placement).
        self.num_local_experts = placement_e_local(self.n_routed_experts, self.ep_degree)
        self.n_slots = self.num_local_experts * self.ep_degree
        if config.moe_intermediate_size % self.tp_degree:
            raise ValueError(f"The MoE intermediate size {config.moe_intermediate_size} must be a multiple of the MoE "
                             f"tensor-parallel degree {self.tp_degree}.")
        self.moe_inter_per_rank = config.moe_intermediate_size // self.tp_degree
        self.shared_inter = config.n_shared_experts * config.moe_intermediate_size
        hidden = config.hidden_size
        for size in (hidden, self.moe_inter_per_rank, self.shared_inter // self.shared_tp):
            if size % _MX_TILE:
                raise ValueError(f"The MX expert kernel needs sizes that are multiples of {_MX_TILE}, got {size}. "
                                 "Raise the expert-parallel degree.")

        self.post_attn_norm_weight = _param(hidden, dtype=config.torch_dtype)
        # The kernel reads the router weight in the swizzle-H order, which the loader applies.
        self.router_weight = _param(hidden, self.n_slots, dtype=torch.bfloat16)
        self.e_score_correction_bias = _param(self.n_slots, dtype=torch.float32)

        n_h512 = hidden // _MX_TILE
        n_pk_h = -(-n_h512 // 4)
        u8, u16 = torch.uint8, torch.uint16
        for prefix, experts, inter in (("", self.num_local_experts, self.moe_inter_per_rank),
                                       ("shared_", 1, self.shared_inter // self.shared_tp)):
            n_i512 = inter // _MX_TILE
            setattr(self, f"{prefix}gate_up_weight", _param(experts, _PMAX, 2, n_h512, inter, dtype=u16))
            setattr(self, f"{prefix}gate_up_scale", _param(experts, _PMAX, n_pk_h, 2, inter, dtype=u8))
            setattr(self, f"{prefix}down_weight", _param(experts, _PMAX, n_i512, hidden, dtype=u16))
            setattr(self, f"{prefix}down_scale", _param(experts, _PMAX, -(-n_i512 // 4), hidden, dtype=u8))


class GlmMoeDsaDecoderLayer(nn.Module):
    """The weights of one decoder layer, dense or sparse (``config.mlp_layer_types``).

    ``layer_idx`` is the stage-local index: the stage config's per-layer lists start at its first layer, and the
    loaders add ``config.first_layer_index`` to reach the checkpoint layer.
    """

    def __init__(self, config, layer_idx: int, world_size: int, ep_degree: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_moe = config.mlp_layer_types[layer_idx] == "sparse"
        self.wo_group_size = min(WO_GROUP_SIZE, world_size)
        if self.wo_group_size <= 1 or world_size % self.wo_group_size:
            raise NotImplementedError(f"GLM-MoE-DSA needs a stage ({world_size} ranks) that splits into o_proj groups "
                                      f"of {self.wo_group_size} > 1 ranks.")
        self.input_layernorm_weight = _param(config.hidden_size, dtype=config.torch_dtype)
        self.self_attn = GlmMoeDsaAttention(config, layer_idx, self.wo_group_size, world_size)
        self.mlp = (GlmMoeDsaMoE(config, world_size, ep_degree) if self.is_moe
                    else GlmMoeDsaMLP(config, self.wo_group_size))


def _stack_members(layers: list) -> dict[str, list[tuple[nn.Module, str]]]:
    """The ``(module, attribute)`` of each per-layer weight of each kernel stack.

    Per kind: attention [L], indexer [L_full], MoE [L_moe], dense MLP [L_dense]. The keys are the GK stack's argument
    names, so a launch passes its stacks by keyword.
    """
    attn = [layer.self_attn for layer in layers]
    full = [a.indexer for a in attn if a.indexer is not None]
    moe = [layer.mlp for layer in layers if layer.is_moe]
    dense = [layer.mlp for layer in layers if not layer.is_moe]
    st = {
        "input_norm_w": [(x, "input_layernorm_weight") for x in layers],
        "wqkv_a": [(a, "wqkv_a") for a in attn],
        "wq_b": [(a, "wq_b") for a in attn],
        "q_norm_w": [(a, "q_norm_weight") for a in attn],
        "kv_norm_w": [(a, "kv_norm_weight") for a in attn],
        "wuk": [(a, "wuk") for a in attn],
        "wuv": [(a, "wuv") for a in attn],
        "wo_shard": [(a, "wo") for a in attn],
        # GLM_GK_PRJ_OMX: the e8m0 scales of the MXFP8 wo (empty list = no stack, see stack_weight_params).
        "wo_scale": [(a, "wo_scale") for a in attn if getattr(a, "wo_scale", None) is not None],
        "post_norm_w": [(x.mlp, "post_attn_norm_weight" if x.is_moe else "post_attention_layernorm_weight")
                        for x in layers],
    }
    for key, name in (("idx_wq_b", "wq_b"), ("idx_wk", "wk"), ("idx_k_norm_w", "k_norm_weight"),
                      ("idx_k_norm_b", "k_norm_bias"), ("idx_weights_proj", "weights_proj")):
        st[key] = [(i, name) for i in full]
    if moe:
        for key, name in (("router_w", "router_weight"), ("router_bias", "e_score_correction_bias"),
                          ("gate_up_w", "gate_up_weight"), ("gate_up_scale", "gate_up_scale"),
                          ("down_w", "down_weight"), ("down_scale", "down_scale"),
                          ("shared_gate_up_w", "shared_gate_up_weight"),
                          ("shared_gate_up_scale", "shared_gate_up_scale"),
                          ("shared_down_w", "shared_down_weight"), ("shared_down_scale", "shared_down_scale")):
            st[key] = [(m, name) for m in moe]
    if dense:
        for key, name in (("mlp_gate_w", "gate_proj"), ("mlp_up_w", "up_proj"), ("mlp_down_w", "down_proj")):
            st[key] = [(m, name) for m in dense]
    return st


class GlmMoeDsaStageModel(nn.Module):
    """The backbone's weights of one stage: the decoder layers, and the embedding, the final norm and the MTP layer
    where the layout holds them (``None`` otherwise).

    ``_groups`` are the stage-local ``[lo, hi)`` of the GK launches (the layout's ``LaunchGroup``s), over which
    ``stack_weight_params`` stacks the per-layer weights; ``_stack_names`` lists the stacks, ``_stk{j}_{key}``.
    """

    def __init__(self, config, layout, *, ep_degree: int, tp_group):
        super().__init__()
        self.config = config
        self.first_layer = config.first_layer_index
        world = layout.cp
        self.layers = nn.ModuleList([GlmMoeDsaDecoderLayer(config, i, world, ep_degree)
                                     for i in range(config.num_hidden_layers)])
        self.embed_tokens = (VocabDimShardedEmbedding(config.vocab_size, config.hidden_size, dtype=config.torch_dtype,
                                                      tp_group=tp_group)
                             if layout.holds_embedding else None)
        self.norm_weight = _param(config.hidden_size, dtype=config.torch_dtype) if layout.samples else None
        # The prompt rows of the MTP layer, for a decode node that runs the MTP draft (layout/prefill/mtp_kv).
        self.mtp_kv = GlmMoeDsaMtpKv(config) if layout.mtp_rows else None
        self._groups = tuple((group.lo, group.hi) for group in layout.groups)
        bounds = [0] + [hi for _, hi in self._groups]
        tiles = all(lo == start and hi > lo for (lo, hi), start in zip(self._groups, bounds))
        if not self._groups or not tiles or bounds[-1] != len(self.layers):
            raise ValueError(f"The launch groups {list(self._groups)} do not tile the stage's {len(self.layers)} "
                             "layers in order, so some layer's weights would reach no launch or two.")
        self._stack_names: tuple[str, ...] = ()

    def stack_weight_params(self, mappings: dict) -> None:
        """Replace the per-layer weights with one stacked parameter for each kernel stack.

        Each stack gets the checkpoint keys of its layers in ``mappings`` and a loader that runs the per-layer loaders.
        Run it after the per-layer loaders are attached and before the load. Every per-layer weight moves into a stack
        (the groups tile the layers), so no parameter is left under ``layers``.
        """
        names = {id(p): f"model.{n}" for n, p in self.named_parameters()}
        layers = list(self.layers)
        keys = []
        for j, (lo, hi) in enumerate(self._groups):
            for key, members in _stack_members(layers[lo:hi]).items():
                if not members:
                    # A launch of shared layers only has no indexer stacks.
                    continue
                self._stack_weight(f"_stk{j}_{key}", members, names, mappings)
                keys.append(f"_stk{j}_{key}")
        self._stack_names = tuple(keys)
        left = [n for n, _ in self.layers.named_parameters()]
        if left:
            raise RuntimeError(f"stack_weight_params left per-layer weights no stack holds: {left[:8]}")

    def _stack_weight(self, attr: str, members: list, names: dict, mappings: dict) -> None:
        """Register the stacked parameter ``attr`` of ``members`` and its checkpoint keys."""
        params = [getattr(m, a) for m, a in members]
        shape, dtype = tuple(params[0].shape), params[0].dtype
        if not all(tuple(p.shape) == shape and p.dtype == dtype for p in params):
            raise ValueError(f"The layers of stack {attr} differ in shape or dtype.")
        ckpt = []
        for p in params:
            k = mappings.pop(names[id(p)])
            ckpt.append(k if isinstance(k, list) else [k])
        stack = nn.Parameter(torch.empty((len(params), *shape), dtype=dtype, device="meta"), requires_grad=False)
        set_weight_loader(stack, stacked_loader([get_weight_loader(p) for p in params], [len(k) for k in ckpt], shape,
                                                dtype))
        self.register_parameter(attr, stack)
        mappings[f"model.{attr}"] = [k for layer_keys in ckpt for k in layer_keys]
        for m, a in members:
            del m._parameters[a]

    def stacks_to_buffers(self) -> None:
        """Turn the loaded stacks into buffers.

        Buffers, not plain attributes: Dynamo lifts a buffer as a graph input, where a plain tensor attribute could be
        baked into the graph as a constant.
        """
        for attr in self._stack_names:
            self.register_buffer(attr, self._parameters.pop(attr).data, persistent=False)


class Glm52StageWeights(nn.Module):
    """The ``GlmMoeDsaForCausalLM`` attribute tree of one stage (``.config``, ``.model``, ``.lm_head``), so the prefill
    loaders apply by name; every parameter on the meta device until the load (weights_load).

    ``lm_head`` is the vocab-sharded ``ColumnParallelLinear`` of a sampling stage, without the gather: the stage graph
    selects its first token over each rank's vocabulary shard. ``layout`` is the stage's ``StageLayout``
    (stage_layout).
    """

    def __init__(self, config, layout, options, groups):
        super().__init__()
        self.config = config
        self.layout = layout
        tp_group = groups.cp.device_group
        with torch.device("meta"):
            self.model = GlmMoeDsaStageModel(config, layout, ep_degree=options.ep_degree, tp_group=tp_group)
            self.lm_head = (ColumnParallelLinear(config.hidden_size, config.vocab_size, dtype=config.torch_dtype,
                                                 gather_output=False, tp_group=tp_group)
                            if layout.samples else None)
        # The vocabulary shards over the stage's CP group: a group of another size would load the wrong rows.
        for name, module in (("model.embed_tokens", self.model.embed_tokens), ("lm_head", self.lm_head)):
            if module is not None and module.tp_size != layout.cp:
                raise ValueError(f"{name} shards over a group of {module.tp_size} ranks, and the stage has "
                                 f"{layout.cp}: pass the stage's CP group (StageGroups.cp).")

    def launch_weights(self, j: int) -> dict[str, torch.Tensor]:
        """The stacked weights of GK launch ``j`` by kernel argument name (the keys of ``_stack_members`` that the
        launch's layers have), which the stage graph passes to the launch by keyword; device buffers after the load.
        """
        model = self.model
        if not model._stack_names:
            raise RuntimeError("The stage weights are not stacked yet: load them first (weights_load).")
        if not 0 <= j < len(model._groups):
            raise IndexError(f"Launch {j} is not one of the stage's {len(model._groups)} launches.")
        prefix = f"_stk{j}_"
        return {attr[len(prefix):]: getattr(model, attr) for attr in model._stack_names if attr.startswith(prefix)}


def build_stage_weights(config, layout, options, groups) -> Glm52StageWeights:
    """The weights tree of one stage on the meta device.

    ``config`` is the stage's ``GlmMoeDsaConfig`` (layout/prefill/pipeline.stage_config of the full config: ``first_layer_index``
    is the stage's first layer), ``layout`` its ``StageLayout``, ``options`` its ``Glm52PrefillOptions`` and ``groups``
    its ``StageGroups``. Refuses a config, layout and options that do not describe the same stage, since each decides a
    part of the tree. The MoE slots come from ``GLM_GK_MOE_PLACEMENT`` (``placement_e_local``) at construction, so the
    pinned placement must be in the environment before this runs (placements.check_placement).
    """
    first, count = int(config.first_layer_index), int(config.num_hidden_layers)
    if (first, first + count) != (layout.start_layer, layout.end_layer):
        raise ValueError(f"The config holds the layers [{first}, {first + count}) and the layout "
                         f"[{layout.start_layer}, {layout.end_layer}): build the config with stage_config.")
    if bool(options.glm52_mtp_kv) != bool(layout.mtp_rows):
        raise ValueError(f"options.glm52_mtp_kv is {options.glm52_mtp_kv} and the layout's mtp_rows "
                         f"{layout.mtp_rows}: only the stage that writes the MTP rows holds the MTP weights.")
    if config.tie_word_embeddings:
        raise NotImplementedError("GLM-MoE-DSA does not support tied word embeddings. Use a checkpoint with a "
                                  "separate lm_head.")
    if any(t == "sparse" for t in config.mlp_layer_types):
        check_placement(pp_size=layout.pp_size)
    return Glm52StageWeights(config, layout, options, groups)
