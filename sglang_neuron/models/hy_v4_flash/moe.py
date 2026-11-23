# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash's LatentMoE block (the reference's HYV4FlashMoE) and its dense layer-0 MLP, on attention TP with the
routed experts over EP.

    out = combine_output_norm(post_expert_proj(experts(pre_expert_proj(x))) + shared_expert(x))

The router (sigmoid, selection bias, renormalized, times ``routed_scaling_factor``) and the latent projections are
replicated; each rank runs its ``E / ep`` experts on the latent rows (a one-token decode only the local experts it
chose, moe_hits.py; a larger decode all of them in moe_tkg's FP8 row mode; a prefill in moe_cte, experts_cte.py), applies ``post_expert_proj`` to its partial sum (linear, so the sum over ranks commutes
with it), adds its TP
shard of the shared expert, and one all-reduce sums the ranks before the norm.

The experts load as Trn2 FP8 (e4m3, at most 240) with a power-of-two scale per output row, requantized from the
checkpoint's 128 x 128 blocks (fp8.requant_rows), in moe_tkg's layout: gate/up [E, H, 2, I]
with scales [E, 2, I], down [E, I, H] with scales [E, H]. Their SwiGLU clamps the gate above and the up both ways at
``swiglu_limit``."""

from __future__ import annotations

import nki
import torch
from torch import nn

from sglang_neuron.models.hy_v4_flash import ops, weights
from sglang_neuron.models.hy_v4_flash.config import Hy4FlashConfig
from sglang_neuron.models.hy_v4_flash.fp8 import FP8, Memo, dequant_blocks, fp8_t, requant_rows
from sglang_neuron.models.transformer.config import DTYPE
from sglang_neuron.models.transformer.model import RMSNorm
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.parallel.state import STATE
from sglang_neuron.utils.neuron_utils import can_run_kernel
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

MAX_TOKENS = 128


class DenseMLP(nn.Module):
    """Layer 0's SiLU MLP, its intermediate dimension over the attention TP, then ``post_mlp_layernorm``."""

    def __init__(self, config: Hy4FlashConfig):
        super().__init__()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.attn_tp_group.world_size
        rows = config.intermediate_size // tp
        self.gate_up = nn.Parameter(torch.empty(config.hidden_size, 2 * rows, dtype=DTYPE), requires_grad=False)
        self.down = nn.Parameter(torch.empty(rows, config.hidden_size, dtype=DTYPE), requires_grad=False)
        set_weight_loader(self.gate_up, weights.mlp_gate_up(rows))
        set_weight_loader(self.down, weights.mlp_down(rows))
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def weight_mappings(self, src: str, dst: str) -> dict:
        return {f"{dst}.gate_up": [f"{src}.mlp.gate_proj.weight", f"{src}.mlp.up_proj.weight"],
                f"{dst}.down": f"{src}.mlp.down_proj.weight",
                f"{dst}.post_norm.weight": f"{src}.mlp.post_fc2_norm.weight"}

    def forward(self, x, real=None):
        gate, up = (x @ self.gate_up).chunk(2, dim=-1)
        out = neuron_collectives.all_reduce(self.attn_tp_group, (ops.swiglu(gate, up, 0.0) @ self.down))
        return self.post_norm(out)


def _expert_parts(slices, local: range, inter: int):
    """[E, H, 2, I] FP8, [E, 2, I] and [E, I, H] FP8, [E, H] fp32 of the local experts, from the fused checkpoint
    tensors gate_up [E, 2 I, H] and down [E, H, I] with their block scales."""
    gu_w, gu_s, dn_w, dn_s = slices
    out = {"gate_up": [], "gate_up_scale": [], "down": [], "down_scale": []}
    for e in local:
        w = dequant_blocks(gu_w[e:e + 1][0], gu_s[e:e + 1][0])
        pair_w, pair_s = [], []
        for part in (w[:inter], w[inter:]):
            q, sc = requant_rows(part)
            pair_w.append(fp8_t(q).view(torch.uint8))
            pair_s.append(sc)
        out["gate_up"].append(torch.stack(pair_w, dim=1))
        out["gate_up_scale"].append(torch.stack(pair_s))
        w = dequant_blocks(dn_w[e:e + 1][0], dn_s[e:e + 1][0])
        q, sc = requant_rows(w)
        out["down"].append(fp8_t(q).view(torch.uint8))
        out["down_scale"].append(sc)
    return {"gate_up": torch.stack(out["gate_up"]).view(FP8), "gate_up_scale": torch.stack(out["gate_up_scale"]),
            "down": torch.stack(out["down"]).view(FP8), "down_scale": torch.stack(out["down_scale"])}


@nki.jit
def _moe_tkg_fp8(hidden, gate_up, down, affinities, index, rank_id, gate_up_scale, down_scale, limit):
    from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
    from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

    return moe_tkg(hidden, gate_up, down, affinities, index, is_all_expert=True, rank_id=rank_id,
                   expert_gate_up_weights_scale=gate_up_scale, expert_down_weights_scale=down_scale,
                   activation_fn=ActFnType.SiLU, mask_unselected_experts=True,
                   expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
                   gate_clamp_upper_limit=limit, up_clamp_upper_limit=limit, up_clamp_lower_limit=-limit)


class LatentMoE(nn.Module):
    def __init__(self, config: Hy4FlashConfig, phase: str):
        super().__init__()
        self.config = config
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.attn_tp_group.world_size
        ep = STATE.ep_degree
        if ep != tp:
            raise ValueError(f"Hy4-Flash's experts run whole on their rank: --ep-size must equal --tp-size ({tp})")
        if config.num_experts % ep:
            raise ValueError(f"{config.num_experts} experts do not split over --ep-size {ep}")
        self.decode = phase == "decode"
        self.num_local = config.num_experts // ep
        self.ep_group = groups.get_neuron_ep_group() if ep > 1 else None
        ep_rank = self.ep_group.rank_in_group if ep > 1 else 0
        self.local = range(ep_rank * self.num_local, (ep_rank + 1) * self.num_local)
        hidden, latent, inter = config.hidden_size, config.routed_hidden_size, config.moe_intermediate_size
        self.router_w = nn.Parameter(torch.empty(config.num_experts, hidden, dtype=DTYPE), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(config.num_experts, dtype=torch.float32), requires_grad=False)
        self.pre = nn.Parameter(torch.empty(hidden, latent, dtype=DTYPE), requires_grad=False)
        self.post = nn.Parameter(torch.empty(latent, hidden, dtype=DTYPE), requires_grad=False)
        e = self.num_local
        self.gate_up = nn.Parameter(torch.empty(e, latent, 2, inter, dtype=FP8), requires_grad=False)
        self.gate_up_scale = nn.Parameter(torch.empty(e, 2, inter, dtype=torch.float32), requires_grad=False)
        self.down = nn.Parameter(torch.empty(e, inter, latent, dtype=FP8), requires_grad=False)
        self.down_scale = nn.Parameter(torch.empty(e, latent, dtype=torch.float32), requires_grad=False)
        # one conversion per layer and rank shared by the four expert parameters, whose loaders may run on the
        # checkpoint loader's threads at once
        memo, local = Memo(), self.local
        for name in ("gate_up", "gate_up_scale", "down", "down_scale"):
            set_weight_loader(getattr(self, name), SafetensorsWeightLoader(
                transform=lambda slices, rank, name=name: memo.take(rank, name, lambda: _expert_parts(
                    slices, local, inter))))
        rows = config.shared_intermediate_size // tp
        self.shared_gate_up = nn.Parameter(torch.empty(hidden, 2 * rows, dtype=DTYPE), requires_grad=False)
        self.shared_down = nn.Parameter(torch.empty(rows, hidden, dtype=DTYPE), requires_grad=False)
        set_weight_loader(self.shared_gate_up, weights.mlp_gate_up(rows))
        set_weight_loader(self.shared_down, weights.mlp_down(rows))
        set_weight_loader(self.router_w, weights.replicated(transposed=False))
        set_weight_loader(self.bias, weights.vector(torch.float32))
        set_weight_loader(self.pre, weights.replicated())
        set_weight_loader(self.post, weights.replicated())
        self.combine_norm = RMSNorm(hidden, config.rms_norm_eps)

    def weight_mappings(self, src: str, dst: str) -> dict:
        m = f"{src}.mlp"
        return {
            f"{dst}.router_w": f"{m}.gate.weight",
            f"{dst}.bias": f"{m}.gate.e_score_correction_bias",
            # the checkpoint's "down" projection maps into the latent space (the reference's pre_expert_proj)
            f"{dst}.pre": f"{m}.routed_expert_down_proj.weight",
            f"{dst}.post": f"{m}.routed_expert_up_proj.weight",
            **{f"{dst}.{name}": [f"{m}.experts.gate_up_proj", f"{m}.experts.gate_up_proj_scale_inv",
                                 f"{m}.experts.down_proj", f"{m}.experts.down_proj_scale_inv"]
               for name in ("gate_up", "gate_up_scale", "down", "down_scale")},
            f"{dst}.shared_gate_up": [f"{m}.shared_experts.gate_proj.weight", f"{m}.shared_experts.up_proj.weight"],
            f"{dst}.shared_down": f"{m}.shared_experts.down_proj.weight",
            f"{dst}.combine_norm.weight": f"{m}.combine_output_norm.weight",
        }

    def _rank_id(self, device) -> torch.Tensor:
        if self.ep_group is None:
            return torch.zeros((1, 1), dtype=torch.int32, device=device)
        return self.ep_group.rank_in_group_tensor.reshape(1, 1).to(torch.int32)

    def _experts_torch(self, x: torch.Tensor, affinities: torch.Tensor) -> torch.Tensor:
        """[T, latent] fp32 this rank's routed output: every local expert over every row, weighted (zero off the
        experts a row chose)."""
        limit = self.config.swiglu_limit
        first = self._rank_id(x.device).reshape(()).to(torch.int64) * self.num_local
        local_aff = affinities.index_select(1, first + torch.arange(self.num_local, device=x.device))
        out = torch.zeros(x.shape[0], self.down.shape[-1], dtype=torch.float32, device=x.device)
        if x.device.type == "cpu":
            # the CPU reference runs each expert on its own tokens only (data-dependent shapes)
            for e in range(self.num_local):
                rows = torch.nonzero(local_aff[:, e] > 0).flatten()
                if rows.numel() == 0:
                    continue
                gu = self.gate_up[e].to(DTYPE)
                xe = x.index_select(0, rows)
                gate = (xe @ gu[:, 0]).float() * self.gate_up_scale[e, 0]
                up = (xe @ gu[:, 1]).float() * self.gate_up_scale[e, 1]
                h = ops.swiglu(gate, up, limit).to(DTYPE)
                y = (h @ self.down[e].to(DTYPE)).float() * self.down_scale[e]
                out.index_add_(0, rows, y * local_aff.index_select(0, rows)[:, e:e + 1])
            return out
        for e in range(self.num_local):
            gu = self.gate_up[e].to(DTYPE)
            gate = (x @ gu[:, 0]).float() * self.gate_up_scale[e, 0]
            up = (x @ gu[:, 1]).float() * self.gate_up_scale[e, 1]
            h = ops.swiglu(gate, up, limit).to(DTYPE)
            y = (h @ self.down[e].to(DTYPE)).float() * self.down_scale[e]
            out = out + y * local_aff[:, e:e + 1]
        return out

    def _experts_hits(self, x: torch.Tensor, affinities: torch.Tensor) -> torch.Tensor:
        """A one-token step's routed output from only the local experts it chose (moe_hits.py): the token's local
        experts packed first with their affinities, and their count."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.hy_v4_flash.moe_hits import moe_tkg_hits
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        e = self.num_local
        first = self._rank_id(x.device).reshape(()).to(torch.int64) * e
        local = affinities.index_select(1, first + torch.arange(e, device=x.device))
        hit = local > 0
        # the chosen experts first, in expert order
        key = hit.to(torch.int32) * (2 * e) + (e - torch.arange(e, device=x.device, dtype=torch.int32))
        order = torch.topk(key, e, dim=1).indices
        slots = order.to(torch.int32)
        slot_aff = local.gather(1, order).float()
        count = hit.sum(dim=1, keepdim=True).to(torch.int32)
        kernel = wrap_nki(moe_tkg_hits)[get_logical_nc_config()]
        return kernel(x, self.gate_up, self.down, slots, slot_aff, count, self.gate_up_scale, self.down_scale,
                      float(self.config.swiglu_limit)).float()

    def _experts_kernel(self, x: torch.Tensor, affinities: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        if x.shape[0] == 1:
            # about half an expert a rank: the all-expert kernel would read all of them
            return self._experts_hits(x, affinities)
        kernel = wrap_nki(_moe_tkg_fp8)[get_logical_nc_config()]
        rank = self._rank_id(x.device)
        idx = chosen.to(torch.int32)
        tiles = [kernel(x[s:s + MAX_TOKENS], self.gate_up, self.down, affinities[s:s + MAX_TOKENS],
                        idx[s:s + MAX_TOKENS], rank, self.gate_up_scale, self.down_scale,
                        float(self.config.swiglu_limit)) for s in range(0, x.shape[0], MAX_TOKENS)]
        out = tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)
        return out.float()

    def forward(self, x: torch.Tensor, real: torch.Tensor | None = None) -> torch.Tensor:
        """The block's output for x [T, H]; ``real`` [T] bool marks a prefill chunk's real rows (the prefill
        kernel's mapping leaves the padding out)."""
        cfg = self.config
        affinities, chosen = ops.route(x, self.router_w, self.bias, cfg.top_k, cfg.norm_topk_prob,
                                       cfg.routed_scaling_factor)
        latent = x @ self.pre
        if self.decode and can_run_kernel(x):
            routed = self._experts_kernel(latent, affinities, chosen)
        elif can_run_kernel(x):
            from sglang_neuron.models.hy_v4_flash.experts_cte import prefill_experts

            first = self._rank_id(x.device).reshape(()).to(torch.int64) * self.num_local
            local = affinities.index_select(1, first + torch.arange(self.num_local, device=x.device))
            if real is None:
                real = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            routed = prefill_experts(latent, self.gate_up, self.down, self.gate_up_scale, self.down_scale, local,
                                     cfg.top_k, real, cfg.swiglu_limit).float()
        else:
            routed = self._experts_torch(latent, affinities)
        out = routed.to(DTYPE) @ self.post
        gate, up = (x @ self.shared_gate_up).chunk(2, dim=-1)
        out = out + ops.swiglu(gate, up, 0.0) @ self.shared_down
        return self.combine_norm(neuron_collectives.all_reduce(self.attn_tp_group, out))
