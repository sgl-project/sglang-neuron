# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2's MoE block: Qwen3-MoE's expert-parallel experts (models/qwen3_moe/moe.py) behind a sigmoid router with a
selection-only bias (``route``, the same rule as Hy3's, kept as MiMo's own copy), and no shared expert.

The HF gate (``MiMoV2MoEGate``, noaux_tc with one group) is, in fp32: scores = sigmoid(x W^T); the top-k experts of
scores + ``e_score_correction_bias``; their weights the chosen scores renormalized (``norm_topk_prob``) and times
``routed_scaling_factor``. With ``n_group = topk_group = 1`` its group step keeps every expert, so it is exactly
route. At EP64 a rank holds 4 of the 256 experts; the decode runs nkilib's bf16 moe_tkg over them, the prefill
nkilib's hybrid I-sharded moe_cte (experts_cte.py, MiMo's own copy), which skips the blocks the routing leaves empty.

The checkpoint's experts (MXFP4 on V2.6, 128 x 128 FP8 blocks on V2.5) load as Trn2 FP8 with a power-of-two scale per
output row, exactly for MXFP4 (experts_fp8.py); a bf16 checkpoint's experts load as they are."""

import torch
from torch import nn

from sglang_neuron.models.mimo_v2 import experts_fp8, weights
from sglang_neuron.models.qwen3_moe.moe import SparseMoE
from sglang_neuron.ops.moe.moe_tkg_bf16 import moe_experts
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.utils.neuron_utils import can_run_kernel
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, get_weight_loader, set_weight_loader

def route(x: torch.Tensor, router_weight: torch.Tensor, expert_bias: torch.Tensor, top_k: int, route_norm: bool,
          scaling: float) -> tuple:
    """Dense [T, E] fp32 affinities (zero off the chosen experts) and the [T, K] chosen experts: scores =
    sigmoid(x W_r^T) in fp32, the top k chosen on scores + expert_bias (the bias selects only), their weights the
    chosen scores, renormalized to sum to one (``route_norm``) and scaled by ``scaling``."""
    scores = torch.sigmoid(x.float() @ router_weight.float().t())
    _, experts = torch.topk(scores + expert_bias, top_k, dim=-1)
    chosen = scores.gather(1, experts)
    if route_norm:
        chosen = chosen / (chosen.sum(dim=-1, keepdim=True) + 1e-20)
    affinities = torch.zeros_like(scores).scatter(1, experts, chosen * scaling)
    return affinities, experts

class MiMoV2MoE(SparseMoE):
    # under DP attention (a PD decode) the experts span the replicas: each step gathers the DP groups' tokens, runs this
    # rank's experts over all of them, sums the outputs over the world and keeps its group's rows (models/gpt_oss/mxfp4's
    # cross-DP EP)
    serves_dp_attention = True

    def __init__(self, *, hidden_size: int, num_experts: int, top_k: int, intermediate_size: int, norm_topk_prob: bool,
                 scaling: float, expert_scale_suffix: str, prefill: bool, layer_key: str = ""):
        super().__init__(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k,
                         intermediate_size=intermediate_size, norm_topk_prob=norm_topk_prob, prefill=prefill)
        self.scaling = scaling
        from sglang_neuron.parallel import groups as neuron_groups
        from sglang_neuron.parallel.state import STATE

        self.dp_size = STATE.dp_size
        if self.dp_size > 1:
            if prefill:
                raise ValueError("MiMo-V2's experts take DP attention on a PD decode only")
            self.dp_group = neuron_groups.get_dp_group()
            self.wide_ep_group = neuron_groups.get_wide_ep_group()
        self.expert_bias = nn.Parameter(torch.empty(num_experts, dtype=torch.float32))
        # "" for a bf16 checkpoint, else the experts' scale key (weight_scale: MXFP4, weight_scale_inv: FP8 blocks)
        self.expert_scale_suffix = expert_scale_suffix
        self.expert_tp_rank = _expert_tp_rank(self)
        self.fp8_experts = bool(expert_scale_suffix) and experts_fp8.enabled()
        self.fused = False
        if self.fp8_experts:
            experts_fp8.make_fp8(self, layer_key)
            if not prefill and not (self.top_k <= 8 and self.down_weight.shape[1] % 256 == 0):
                raise ValueError(f"the decode MoE kernel (kernels/moe_half.py) takes top_k <= 8 and an expert "
                                 f"shard of a multiple of 256 rows (top_k {self.top_k}, {self.down_weight.shape[1]})")
            # the kernel's router reads the router weight transposed, [H, E]; the prefill graph holds it too,
            # unused, since an aggregated server's two graphs share one device copy of the weights
            self.router_wT = nn.Parameter(torch.empty(hidden_size, num_experts, dtype=torch.bfloat16),
                                          requires_grad=False)
            set_weight_loader(self.router_wT, SafetensorsWeightLoader(
                transform=lambda slices, rank: slices[0][...].to(torch.bfloat16).t().contiguous()))
            self.fused = not prefill
        elif expert_scale_suffix:
            for p in (self.gate_up_weight, self.down_weight):
                set_weight_loader(p, weights.dequantizing(get_weight_loader(p)))

    def prefill_residual(self, x, normed, real=None, positions=None):
        """A prefill chunk's residual stream after the MoE block, x + MoE(normed): on a device the routing and
        mapping (kernels/prefill.prefill_route), the FP8 experts, and the all-reduce with the add
        (kernels/prefill.prefill_residual). ``real`` [T] marks the chunk's real tokens on the CPU reference;
        ``positions`` [T, 1] int32 its positions (the kernels tell the padding by its repeated position)."""
        if not (self.fp8_experts and self.dp_size == 1 and can_run_kernel(x)):
            return x + self(normed, real=real)
        from sglang_neuron.backend.stack import resolve_full_mesh, wrap_nki
        from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_residual
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        mesh = tuple(tuple(int(r) for r in cell) for cell in resolve_full_mesh(self.attn_tp_group))
        return wrap_nki(prefill_residual)[get_logical_nc_config()](x, self._prefill_kernels(normed, positions), mesh)

    def forward(self, x, real=None):
        x = self._gather(x)
        affinities, experts = route(x, self.router_weight, self.expert_bias, self.top_k, self.norm_topk_prob,
                                        self.scaling)
        # the kernels take fp32 affinities; rounded through bf16 as the HF model's bf16 expert sum weighs them
        affinities = affinities.to(x.dtype).float()
        if self.prefill:
            from sglang_neuron.models.mimo_v2 import experts_cte

            local = (torch.arange(self.num_local_experts, device=x.device, dtype=torch.int32)
                     + self._ep_rank_id().reshape(1) * self.num_local_experts)
            if real is None:
                real = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            scales = (self.gate_up_scale, self.down_scale) if self.fp8_experts else (None, None)
            out = experts_cte.prefill_experts(x, self.gate_up_weight, self.down_weight,
                                              affinities.index_select(1, local), self.top_k, real, *scales)
        elif self.fp8_experts:
            out = experts_fp8.decode_experts(self, x, affinities, experts.to(torch.int32))
        else:
            out = moe_experts(x, gate_up=self.gate_up_weight, down=self.down_weight, affinities=affinities,
                              index=experts.to(torch.int32), rank_id=self._ep_rank_id())
        return self._reduce(out)

    def _prefill_kernels(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """This rank's routed-expert output [T, H] for a prefill chunk: the routing and the blockwise mapping in one
        kernel (kernels/prefill.prefill_route), the experts in the FP8 hybrid moe_cte (kernels/moe_cte_fp8w.py)."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.mimo_v2.experts_cte import BLOCK_SIZE
        from sglang_neuron.models.mimo_v2.kernels.moe_cte_fp8w import moe_cte_fp8w
        from sglang_neuron.models.mimo_v2.kernels.prefill import prefill_route
        from sglang_neuron.ops.moe.moe_blockwise import _num_blocks
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        tokens = x.shape[0]
        experts, intermediate, hidden = self.down_weight.shape
        if experts % 2 or tokens % 128 or positions is None:
            raise ValueError(f"the prefill routing kernel takes an even local expert count and whole 128-token "
                             f"tiles with the chunk's positions ({experts} experts, {tokens} tokens)")
        blocks = _num_blocks(tokens, self.top_k, experts, BLOCK_SIZE)
        grid = get_logical_nc_config()
        aff, token_position_to_id, block_to_expert, conditions = wrap_nki(prefill_route)[grid](
            x, self.router_wT, self.expert_bias.view(1, -1), self._ep_rank_id(), positions, top_k=self.top_k,
            route_scale=float(self.scaling), n_local=experts, block_size=BLOCK_SIZE, num_blocks=blocks)
        out = wrap_nki(moe_cte_fp8w)[grid](
            conditions, x, aff, self.gate_up_weight, self.down_weight, token_position_to_id, block_to_expert,
            self.gate_up_scale.view(experts, 1, 2 * intermediate), self.down_scale.view(experts, 1, hidden),
            block_size=BLOCK_SIZE, num_static_block=experts, skip_token=True)
        return (out[:, 0] if out.dim() == 3 else out)[:tokens]

    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        """The rows every DP group decodes this step (DP attention), else ``x``. Over the world, then each group's first
        rank's copy: on a Trn2 64-rank layout the DP columns (ranks 8 apart) form no collective group the runtime can
        build ("no_hier no_mesh")."""
        if self.dp_size == 1:
            return x
        rows, tp = x.shape[0], self.attn_tp_group.world_size
        every = neuron_collectives.all_gather(self.wide_ep_group, x, dim=0).view(self.dp_size * tp, rows, -1)
        leaders = torch.arange(self.dp_size, device=x.device, dtype=torch.int64) * tp
        return every.index_select(0, leaders).reshape(self.dp_size * rows, -1)

    def _reduce(self, out: torch.Tensor) -> torch.Tensor:
        """This rank's partial expert output summed over the ranks holding experts: the attention TP group, or under DP
        attention the world, and this rank's group's rows of it."""
        if self.dp_size == 1:
            return neuron_collectives.all_reduce(self.attn_tp_group, out)
        from sglang_neuron.parallel.groups import shard_select

        out = neuron_collectives.all_reduce(self.wide_ep_group, out)
        return shard_select(out, 0, self.dp_group, n_shards=self.dp_size)

    def weight_mappings(self, src: str, dst: str) -> dict:
        mlp, experts = f"{src}.mlp", f"{src}.mlp.experts"
        local = self.local_experts
        gate_up = [f"{experts}.{e}.gate_proj.weight" for e in local] + [f"{experts}.{e}.up_proj.weight" for e in local]
        down = [f"{experts}.{e}.down_proj.weight" for e in local]
        if self.expert_scale_suffix:
            gate_up = weights.with_scales(gate_up, self.expert_scale_suffix)
            down = weights.with_scales(down, self.expert_scale_suffix)
        mappings = {
            f"{dst}.router_weight": f"{mlp}.gate.weight",
            f"{dst}.expert_bias": f"{mlp}.gate.e_score_correction_bias",
            f"{dst}.gate_up_weight": gate_up,
            f"{dst}.down_weight": down,
        }
        if hasattr(self, "router_wT"):
            mappings[f"{dst}.router_wT"] = f"{mlp}.gate.weight"
        if self.fp8_experts:
            # a scale comes out of its weight's requantization (experts_fp8.make_fp8)
            mappings[f"{dst}.gate_up_scale"] = gate_up
            mappings[f"{dst}.down_scale"] = down
        return mappings


def _expert_tp_rank(moe) -> int:
    """This rank's place among the ranks an expert's intermediate dimension splits over (SparseMoE's layout)."""
    from sglang_neuron.parallel import groups
    from sglang_neuron.parallel.state import STATE

    tp, ep = moe.attn_tp_group.world_size, STATE.ep_degree
    if ep == 1:
        return moe.attn_tp_group.rank_in_group
    return groups.get_neuron_ep_tp_group().rank_in_group if tp // ep > 1 else 0
