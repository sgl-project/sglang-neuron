# SPDX-License-Identifier: Apache-2.0
"""One GLM-5.2 prefill pipeline stage on the prefill giga kernel.

Each launch group of the stage layout is one giga-kernel launch, with every collective inside the kernel. Around the
launches the graph does the stage's host side: the first stage embeds the tokens; every stage builds its RoPE rows, KV
slots and the kernel's static arguments; launches with full indexer layers store their new key rows as the decode's
fp8 keys; the last stage writes the MTP prompt rows and picks the first token over its vocabulary shards. Earlier
stages write their (h, r) shard into the seam slots they are handed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from torch import nn

from sglang_neuron.models.glm52.prefill import inputs as pi
from sglang_neuron.models.glm52.layout import glm_kv_row, rope_split
from sglang_neuron.models.glm52.layout.prefill.mtp_kv import pack_keys, pack_latent_rows
from sglang_neuron.models.glm52.layout.prefill.weight_loaders import hybrid_tp, shared_tp
from sglang_neuron.models.glm52.prefill.stage_layout import CP_INTERLEAVE, StageLayout
from sglang_neuron.models.glm52.ops import prefill_kernel as gk
from sglang_neuron.models.glm52.ops.bitcast import bitcast

if TYPE_CHECKING:
    from sglang_neuron.models.glm52.prefill.stage_groups import StageGroups

#: Ranks of one TP x CP o_proj / dense-MLP group: the kernel's row-parallel o_proj fits SBUF at 8.
WO_GROUP_SIZE = 8
#: Tokens of one routed-expert block, and the most tokens an expert gets on average for which the kernel takes
#: 128-token blocks instead.
MOE_BLOCK_SIZE = 256
MOE_BLOCK_LIMIT = 128


def moe_block_size(pairs_per_expert: int) -> int:
    """The routed experts' block size: 128-token blocks pad less when experts get few tokens on average."""
    return 128 if pairs_per_expert <= MOE_BLOCK_LIMIT else MOE_BLOCK_SIZE


def softmax_scale(config) -> float:
    """``1 / sqrt(qk_head_dim)``. A scaled RoPE is refused: the stage's RoPE tables are unscaled."""
    rope = getattr(config, "rope_parameters", None) or {}
    if rope.get("rope_type", "default") != "default" or float(rope.get("factor", 1.0)) != 1.0:
        raise ValueError(f"rope_parameters {rope}: the GLM-5.2 prefill stage serves the checkpoint's default RoPE only, "
                         "with no rope scaling")
    return config.qk_head_dim ** -0.5


class _GkLaunch(nn.Module):
    """One giga-kernel launch as its own submodule, so its op names give the compiler's splitter a boundary at each
    launch and a repeated launch compiles once."""

    def forward(self, hidden, residual, st: dict, n_layers: int, **kwargs):
        st = dict(st)
        # the per-layer norm gammas and the router bias take a leading [1, ...] row
        for key in ("input_norm_w", "post_norm_w", "q_norm_w", "kv_norm_w"):
            st[key] = st[key].reshape(n_layers, 1, -1)
        if "router_bias" in st:
            st["router_bias"] = st["router_bias"].reshape(st["router_bias"].shape[0], 1, -1)
        return gk.prefill_giga_kernel(hidden, residual=residual, **st, **kwargs)


class _Rotary(nn.Module):
    """The decode's RoPE rows at the given positions (prefill.inputs.rope_rows), from host-built buffers the load
    moves to the device."""

    def __init__(self, theta: float, dim: int):
        super().__init__()
        for name, table in pi.rope_tables(float(theta), int(dim)).items():
            self.register_buffer(name, table, persistent=False)

    def forward(self, positions):
        return pi.rope_rows({name: getattr(self, name) for name in rope_split.TABLE_NAMES}, positions)


class Glm52PrefillStage(nn.Module):
    """One prefill stage: its layout, stage config, weight tree (on the meta device until load) and CP / EP groups.

    Raises:
        ValueError: A logical-core degree, options or weight tree the graph was not built for.
    """

    # the last stage samples inside its forward, so the host runs it without the device sampler's wrapper
    samples_in_forward = True
    sampler_modes = ("greedy", "topk_topp")

    def __init__(self, layout: StageLayout, config, weights: nn.Module, groups: "StageGroups"):
        super().__init__()
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        problems = []
        lnc = get_logical_nc_config()
        if lnc != 1:
            problems.append(f"NEURON_LOGICAL_NC_CONFIG={lnc}: the giga kernel is an LNC1 kernel")
        options = config.options
        if int(getattr(options, "glm_prefill_pack", 1) or 1) != 1 or getattr(options, "glm52_kv_compact", False):
            problems.append("options: the stage graph runs one sequence per forward (glm_prefill_pack 1) on the "
                            "640-byte row (no glm52_kv_compact)")
        if not getattr(options, "glm52_idx_fp8_keys", False):
            problems.append("options.glm52_idx_fp8_keys: the pool's indexer key caches hold the decode node's "
                            "fp8 keys")
        if bool(getattr(options, "glm52_mtp_kv", False)) != layout.mtp_rows:
            problems.append(f"options.glm52_mtp_kv={getattr(options, 'glm52_mtp_kv', False)}, but stage "
                            f"{layout.pp_rank} {'writes' if layout.mtp_rows else 'writes no'} MTP prompt rows")
        if (int(config.first_layer_index), int(config.num_hidden_layers)) != (layout.start_layer, layout.layers):
            first = int(config.first_layer_index)
            problems.append(f"the stage config holds layers [{first}, {first + config.num_hidden_layers}), the layout "
                            f"[{layout.start_layer}, {layout.end_layer})")
        if config.tie_word_embeddings:
            problems.append("tied word embeddings: GLM-5.2 keeps a separate LM head")
        if layout.cp % WO_GROUP_SIZE:
            problems.append(f"a stage of {layout.cp} ranks does not split into whole o_proj groups of "
                            f"{WO_GROUP_SIZE}")
        model = getattr(weights, "model", None)
        held = ((model, "embed_tokens", layout.holds_embedding), (model, "norm_weight", layout.samples),
                (model, "mtp_kv", layout.mtp_rows), (weights, "lm_head", layout.samples))
        missing = [("" if owner is weights else "model.") + attr for owner, attr, needed in held
                   if needed and getattr(owner, attr, None) is None]
        if missing:
            problems.append(f"stage {layout.pp_rank}'s weight tree holds no {missing}")
        if problems:
            raise ValueError("the GLM-5.2 prefill stage graph cannot run:\n  - " + "\n  - ".join(problems))
        self.layout, self.config, self.groups = layout, config, groups
        # children in this order and with these names: the launches' module path is what a hierarchy splitter cuts at
        # (compile_args.HIERARCHY_DEPTH, measured; the released list takes the split finder)
        self.weights = weights
        self.rotary = _Rotary(config.rope_parameters["rope_theta"], config.rope_head_dim)
        self.launches = nn.ModuleList([_GkLaunch() for _ in layout.groups])
        # the runtime's contract names every layer of the model, and slices the stage's (graph_runner._init_stage)
        self.layer_names = tuple(f"layers.{i}.self_attn" for i in range(layout.model_layers))
        self.sliding_windows = (None,) * layout.model_layers
        cp, g = layout.cp, WO_GROUP_SIZE
        # the prefill loaders' layout choices, read once (the weight tree's shapes read them too)
        self.hybrid_tp = hybrid_tp(cp, g)
        self.shared_tp = shared_tp(cp, g)
        self.ep_degree = int(options.ep_degree)
        self.softmax_scale = softmax_scale(config)
        self.kv_row_words = glm_kv_row.STATIC_ROW_BYTES // 2
        # the kernel's replica groups: static tuples, the NEFF's topology
        stage_cells = [tuple(range(cp))]
        wo_cells = [tuple(range(i, i + g)) for i in range(0, cp, g)]
        position_cells = [tuple(range(p, cp, g)) for p in range(g)]
        self.replica = {"cp_replica_groups": layout.replica_groups(stage_cells),
                        "moe_replica_groups": layout.replica_groups(stage_cells),
                        "wo_replica_groups": layout.replica_groups(wo_cells)}
        if self.hybrid_tp:
            self.replica["attn_tp_gather_groups"] = layout.replica_groups(position_cells)
        # the logit test (models/glm52/logit_probe.py): the last stage also returns its sampled rows' full-vocabulary
        # logits; never on a served launch
        from sglang_neuron.models.glm52 import logit_probe

        self.probe_logits = bool(layout.samples and logit_probe.test_dir() is not None)
        if self.probe_logits:
            logit_probe.warn_on("the prefill's last stage all-gathers its sampled row's logits over its CP ranks at "
                                "every step, and its CP rank 0 reads them back and writes them")
        if layout.mtp_rows:
            # the carry state of the MTP rows: the final hidden row of the last forward, the global
            # position after it and the cache slot of that row
            h = config.hidden_size
            self.register_buffer("_mtp_carry_h", torch.zeros((1, h), dtype=config.torch_dtype), persistent=False)
            self.register_buffer("_mtp_carry_end", torch.full((1,), -1, dtype=torch.int32), persistent=False)
            self.register_buffer("_mtp_carry_slot", torch.full((1,), -1, dtype=torch.int32), persistent=False)
        self._pool = None

    @property
    def lm_head(self):
        """The vocabulary-sharded LM head of a sampling stage, which the stage samples over shard by shard."""
        return self.weights.lm_head if self.layout.samples else None

    def get_kv_spec(self):
        """The stage's KV regions by layer name, in the decode node's order (StageLayout.kv_layer_names)."""
        return SimpleNamespace(layers=self.layout.kv_layer_names())

    def load_weights(self, path: str, device) -> None:
        """This rank's stage weights from the checkpoint at ``path``, straight into the launches' stacks."""
        from sglang_neuron.models.glm52.prefill import weights_load

        weights_load.load_stage_weights(self.weights, path, device, rank=int(self.groups.cp.rank_in_group),
                                        world=self.layout.cp)

    # the KV pool's caches
    def bind_pool(self, pool) -> None:
        """Bind the pool's caches as non-persistent buffers, which dynamo lifts as graph inputs, for the kernel and the
        stores to write in place: each launch's KV and indexer stacks, the fp8 key caches and the last stage's MTP
        caches.

        Raises:
            ValueError: A cache of another shape or dtype, or a pool whose written tensors are not the bound ones.
        """
        layout, d = self.layout, int(self.config.index_head_dim)
        bound, slots, mtp_drafts = {}, set(), False

        def check(name, tensor, layers, width, dtype):
            # a launch's stack [layers, n_slots, width], or with layers None a flat [n_slots, width] cache
            want = (width,) if layers is None else (layers, width)
            ok = (tensor.dtype == dtype and tensor.dim() == len(want) + 1 and tensor.shape[-1] == width
                  and (layers is None or tensor.shape[0] == layers))
            if not ok:
                lead = "" if layers is None else f"{layers}, "
                raise ValueError(f"the pool's {name} is {tuple(tensor.shape)} {tensor.dtype}; the stage graph writes "
                                 f"[{lead}n_slots, {width}] {dtype}")
            slots.add(int(tensor.shape[-2]))
            bound[name] = tensor

        for group in layout.groups:
            j = group.index
            kv, idx, keys = pool.launch_caches(j)
            check(f"_kv_stack{j}", kv, group.layers, self.kv_row_words, torch.bfloat16)
            if group.full:
                check(f"_idx_stack{j}", idx, len(group.full), d, torch.bfloat16)
            elif idx.shape[0]:
                raise ValueError(f"the pool gives launch {j}, which holds no full indexer layer, a key stack of "
                                 f"{idx.shape[0]} layers")
            if len(keys) != len(group.full):
                raise ValueError(f"the pool gives launch {j} {len(keys)} fp8 key caches for its {len(group.full)} full "
                                 "layers")
            for k, words in enumerate(keys):
                check(f"_fp8_keys{j}_{k}", words, None, d // 2, torch.bfloat16)
        if layout.mtp_rows:
            latent, keys = pool.mtp_caches()
            check("_mtp_latent", latent, None, self.kv_row_words, torch.bfloat16)
            check("_mtp_keys", keys, None, d // 2, torch.bfloat16)
            # a decode that drafts (pd.placement.PD_MTP_ENV) reads the prompt's MTP rows: the step's last one takes the
            # first sampled token (forward)
            mtp_drafts = bool(getattr(pool, "pd_mtp", False))
            pd_keys = pool.mtp_pd_keys()
            if pd_keys is not None:
                # the MTP rows' bf16 keys for a drafting decode at a bf16-key point
                check("_mtp_pd_keys", pd_keys, None, d, torch.bfloat16)
        if len(slots) != 1 or next(iter(slots)) % layout.kv_page_rows:
            raise ValueError(f"the pool's caches hold {sorted(slots)} slots; every cache of the stage holds the same "
                             f"whole pages of {layout.kv_page_rows} rows")
        # the warm-up proves the pool's graph-written tensors in place, so they must be exactly the bound caches
        written = dict(pool.graph_written_tensors())
        ids = {id(t) for t in bound.values()}
        unbound = sorted(n for n, t in written.items() if id(t) not in ids)
        undeclared = len(ids - {id(t) for t in written.values()})
        if unbound or undeclared:
            raise ValueError(f"the pool's graph-written tensors are not the caches the stage binds: {unbound} not "
                             f"bound, {undeclared} bound cache(s) not declared")
        for name, tensor in bound.items():
            self.register_buffer(name, tensor, persistent=False)
        self._mtp_drafts = mtp_drafts
        self._pool = pool

    def in_place_targets(self) -> dict:
        """Every tensor the graph writes in place outside its outputs, by name: the pool's caches it binds and the MTP
        carry buffers. The warm-up proves each written in place by the NEFF (graph_runner.check_in_place)."""
        if self._pool is None:
            raise RuntimeError("the stage graph's caches are not bound: bind_pool first")
        targets = dict(self._pool.graph_written_tensors())
        if self.layout.mtp_rows:
            targets.update(mtp_carry_h=self._mtp_carry_h, mtp_carry_end=self._mtp_carry_end,
                           mtp_carry_slot=self._mtp_carry_slot)
        return targets

    # the forward
    @torch.no_grad()
    def forward(self, input_ids, positions, slot_mapping, block_table, cached_seq_len, sampling_positions, rank,
                pipeline_hidden_states=None, pipeline_residual=None, pipeline_hidden_states_out=None,
                pipeline_residual_out=None, sampling_temperatures=None, sampling_seeds=None, sampling_filters=None,
                pipeline_picks=None, pipeline_picks_out=None):
        """One step of one request: an earlier stage returns its boundary in the seam slots, the last stage the first
        token and its logprobs."""
        layout, block = self.layout, CP_INTERLEAVE
        cp, seq_len = layout.cp, int(input_ids.shape[0])
        if seq_len % (cp * block):
            raise ValueError(f"the prefill bucket {seq_len} is not a multiple of {cp} ranks x {block}")
        t_local = seq_len // cp
        positions = positions.to(torch.int32)
        q_pos = pi.query_positions(positions)
        order = pi.rank_major_order(seq_len, block, cp, input_ids.device)
        model = self.weights.model
        if layout.embeds:
            # the reduce-scatter of the reordered tokens gives each rank its own shard
            hidden = model.embed_tokens(input_ids.index_select(0, order), scatter_tokens=True, rank=rank)
            residual = None
        else:
            if pipeline_hidden_states is None or pipeline_residual is None:
                raise ValueError("a prefill stage after the first takes the stage before's boundary "
                                 "(pipeline_hidden_states, pipeline_residual)")
            hidden, residual = pipeline_hidden_states, pipeline_residual
        if layout.picks_in and pipeline_picks is None:
            raise ValueError(f"stage {layout.pp_rank} starts on a shared indexer layer and takes the stage before's "
                             "top-k picks (pipeline_picks)")
        # at PP16 every stage carries the picks (sglang_model's boundary): a stage that takes none ignores them
        cp_rank = self.groups.cp.rank_in_group_tensor
        owned = pi.owned_index(seq_len, block, cp, cp_rank.reshape(()), input_ids.device)
        own_positions = positions.index_select(0, owned)
        rope = self.rotary(own_positions)
        slots, prefix = pi.kv_context(slot_mapping, block_table, cached_seq_len, t_local=t_local, cp=cp,
                                      cp_rank=cp_rank, block=block, page=layout.kv_page_rows)
        static = self._static_args(t_local, prefix, cp_rank, input_ids.device)
        if self.hybrid_tp:
            static.update(pi.group_tables(positions, slots, seq_len=seq_len, block=block, cp=cp, cp_rank=cp_rank,
                                          wo_group=WO_GROUP_SIZE, rope=self.rotary))
        out = self._run_layers(hidden, residual, rope, q_pos.index_select(0, owned), slots, cp_rank, t_local, static,
                               picks=pipeline_picks if layout.picks_in else None)
        if not layout.samples:
            return self._write_boundary(out, pipeline_hidden_states_out, pipeline_residual_out, pipeline_picks_out)
        drafts = layout.mtp_rows and getattr(self, "_mtp_drafts", False)
        if layout.mtp_rows and not drafts:
            self._write_mtp_rows(input_ids, order, rank, out, own_positions, rope, slots, prefix)
        rows = pi.select_sampled_rows(out, sampling_positions, block=block, cp=cp, cp_rank=cp_rank,
                                      group=self.groups.cp)
        logits = self.weights.lm_head(rows)
        result = pi.sample_first_token(logits, self.groups.cp, cp_rank, int(self.config.vocab_size),
                                       positions=positions, sampling_positions=sampling_positions,
                                       sampling_temperatures=sampling_temperatures, sampling_seeds=sampling_seeds,
                                       sampling_filters=sampling_filters)
        if self.probe_logits:
            from sglang_neuron.parallel import collectives

            # the vocabulary shards in CP rank order (pi._token_logprobs' ids): the [R, V] logits on every rank
            result["probe_logits"] = collectives.all_gather(self.groups.cp, logits, dim=-1)
        if drafts:
            # for a drafting decode the last position's MTP row takes the sampled token, so the decode need not
            # make it; on an earlier chunk the next forward's carry rewrites that row
            self._write_mtp_rows(input_ids, order, rank, out, own_positions, rope, slots, prefix,
                                 first_token=result["token_ids"], sampling_positions=sampling_positions)
        return result

    def _static_args(self, t_local: int, prefix: dict, cp_rank, device) -> dict:
        """The kernel arguments every launch of the step shares."""
        config, layout = self.config, self.layout
        cp = layout.cp
        keys = cp * prefix["kv_ctx_local"]
        if keys <= config.index_topk:
            raise NotImplementedError(f"the giga kernel covers the sparse DSA path only ({keys} keys, index_topk "
                                      f"{config.index_topk}): use a longer context")
        moe = {"ep_rank": torch.zeros((1, 1), dtype=torch.int32, device=device)}
        if any(group_type == "sparse" for group in layout.groups for group_type in group.mlp_types):
            moe = {
                "ep_rank": (moe["ep_rank"] if self.ep_degree <= 1
                            else self.groups.ep.rank_in_group_tensor.reshape(1, 1).to(torch.int32)),
                "top_k": config.num_experts_per_tok,
                "n_group": config.n_group,
                "topk_group": config.topk_group,
                "routed_scaling_factor": config.routed_scaling_factor,
                "moe_block_size": moe_block_size(cp * t_local * config.num_experts_per_tok // config.n_routed_experts),
                "ep_degree": self.ep_degree,
            }
            if self.shared_tp > 1:
                moe["shared_tp"] = self.shared_tp
        cp_rank = cp_rank.to(torch.int32).reshape(1, 1)
        return dict(
            n_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            index_topk=config.index_topk,
            rms_norm_eps=config.rms_norm_eps,
            mla_norm_eps=config.mla_norm_eps,
            softmax_scale=self.softmax_scale,
            cp_degree=cp,
            cp_interleave=CP_INTERLEAVE,
            wo_group_rank=(cp_rank % WO_GROUP_SIZE).to(torch.int32).reshape(1, 1),
            **self.replica,
            **prefix,
            **moe,
        )

    def _run_layers(self, hidden, residual, rope, q_pos, slots, cp_rank, t_local: int, static: dict, picks=None):
        """Run the stage's launches in order, storing each full-layer launch's fp8 keys. Picks come in from the stage
        before and go out to the next where the layout says so. Returns the normed rows on a sampling stage, else
        (h, r), plus the picks it hands on."""
        layout, model = self.layout, self.weights.model
        cos, sin = rope
        cp_rank = cp_rank.to(torch.int32).reshape(1, 1)
        h, r = hidden, residual
        for group, launch in zip(layout.groups, self.launches):
            j = group.index
            last_group = j == len(layout.groups) - 1
            last = layout.samples and last_group
            has_full = bool(group.full)
            pick_args = {}
            if j == 0 and picks is not None:
                pick_args["topk_in"] = picks.reshape(t_local, -1).to(torch.int32)
            emit_picks = layout.picks_out and last_group and not last
            if emit_picks:
                pick_args["emit_picks"] = True
            outs = launch(
                h, r, self.weights.launch_weights(j), group.layers,
                cos=cos.to(torch.float32),
                sin=sin.to(torch.float32),
                q_pos=q_pos.reshape(-1).to(torch.float32),
                slot_mapping=slots,
                cp_rank=cp_rank,
                kv_caches=getattr(self, f"_kv_stack{j}"),
                idx_k_caches=getattr(self, f"_idx_stack{j}") if has_full else None,
                final_norm_w=model.norm_weight.reshape(1, -1) if last else None,
                layer_mlp_types=group.mlp_types,
                layer_indexer_types=group.indexer_types,
                **({"emit_key_rows": True} if has_full else {}),
                **pick_args,
                **static,
            )
            if has_full:
                # (normed | h, r[, picks]), the in-place caches, then the launch's [F, T, d] new key rows in slot order
                self._store_fp8_keys(group, slots, outs[-1])
            if last:
                return outs[0].reshape(t_local, -1)
            h, r = outs[0], outs[1]
            if emit_picks:
                return h.reshape(t_local, -1), r.reshape(t_local, -1), outs[2].reshape(t_local, -1)
        return h.reshape(t_local, -1), r.reshape(t_local, -1)

    def _store_fp8_keys(self, group, slots, new_rows) -> None:
        """Write the launch's new key rows at ``slots`` into its full layers' fp8 key caches: the decode
        node's e4m3fn bytes, bound as bf16 words."""
        index = slots.reshape(-1).to(torch.int64)
        for k in range(len(group.full)):
            rows = bitcast(pack_keys(new_rows[k], bitcast, True).contiguous(), torch.bfloat16)
            getattr(self, f"_fp8_keys{group.index}_{k}").index_put_((index,), rows)

    def _put_mtp_rows(self, slots, k_pe, c_kv, keys, keep=None) -> None:
        """Write packed MTP rows at ``slots``. Rows where ``keep`` is True go to the null page instead, which leaves
        the cached rows alone without reading them back (a read-then-write of the cache does not compile)."""
        rows = pack_latent_rows(k_pe, c_kv, bitcast, self._mtp_latent.shape[1])
        key_rows = pack_keys(keys, bitcast, True)
        if self._mtp_keys.dtype != key_rows.dtype:
            key_rows = bitcast(key_rows.contiguous(), self._mtp_keys.dtype)
        index = slots.to(torch.int64)
        if keep is not None:
            index = torch.where(keep.reshape(index.shape), torch.zeros_like(index), index)
        self._mtp_latent.index_put_((index,), rows)
        self._mtp_keys.index_put_((index,), key_rows)
        if hasattr(self, "_mtp_pd_keys"):
            # the same keys in bf16, the decode draft indexer's format (the target layers' key regions)
            self._mtp_pd_keys.index_put_((index,), pack_keys(keys, bitcast, False))

    def _write_mtp_rows(self, input_ids, order, rank, hidden, positions, rope, slots, prefix, first_token=None,
                        sampling_positions=None) -> None:
        """Write this rank's MTP prompt rows: row t takes h_t and the embedding of token t + 1. The last row of a
        forward needs the next forward's first token, so the next forward rewrites it from the carried h when the
        carry belongs to the same request. The prompt's final row needs the sampled token: ``first_token`` writes it
        here, otherwise the decode does."""
        layout, model = self.layout, self.weights.model
        mtp, embed, cp = model.mtp_kv, model.embed_tokens, layout.cp
        cos, sin = rope
        nxt = torch.cat([input_ids[1:], input_ids[-1:]])
        if first_token is not None:
            at = torch.arange(nxt.shape[0], dtype=torch.int64, device=nxt.device)
            last = sampling_positions.reshape(1).to(torch.int64)
            nxt = torch.where(at == last, first_token.reshape(1).to(nxt.dtype), nxt)
        embeds = embed(nxt.index_select(0, order), scatter_tokens=True, rank=rank)
        k_pe, c_kv, keys = mtp.rows(embeds, hidden, positions, cos, sin)
        # rank cp - 1 owns position start - 1, at its local row prior_local - 1
        prior_local = prefix["prior_tokens"].reshape(())
        start = prior_local * cp
        pos = (start - 1).clamp(min=0).reshape(1).to(positions.dtype)
        first = embed(input_ids[:1], scatter_tokens=False, rank=rank)
        cos1, sin1 = self.rotary(pos)
        k1, c1, key1 = mtp.rows(first, self._mtp_carry_h, pos, cos1, sin1)
        local = (prior_local - 1).clamp(min=0)
        page, table = prefix["kv_block_size"], prefix["block_table"]
        # the page of row ``local`` by a one-hot sum over the table, not a gather at a device index: the image's
        # neuronx-cc 2.28 tensorizer cannot split that gather's use at the 1M bucket's table (NCC_ILSM901)
        cols = torch.arange(table.shape[0], dtype=torch.int64, device=table.device)
        hit = cols == pi.nonneg_div(local, page).to(torch.int64).reshape(())
        page_id = torch.where(hit, table.to(torch.int64), torch.zeros_like(cols)).sum().reshape(1)
        slot = (page_id * page + pi.nonneg_mod(local, page).to(torch.int64)).to(table.dtype)
        cp_rank = self.groups.cp.rank_in_group_tensor.reshape(())
        owner = ((cp_rank == cp - 1) & (self._mtp_carry_end.reshape(()) == start)
                 & (self._mtp_carry_slot.reshape(()) == slot.reshape(()).to(torch.int32)) & (prior_local > 0))
        # the carry row and the forward's rows in one store: a separate one-row store at the carry's slot does not
        # compile at the 1M bucket
        n = int(slots.shape[0])
        keep = torch.cat([(~owner).reshape(1, 1), torch.zeros((n, 1), dtype=torch.bool, device=slots.device)])
        self._put_mtp_rows(torch.cat([slot.reshape(1).to(slots.dtype), slots]), torch.cat([k1, k_pe]),
                           torch.cat([c1, c_kv]), torch.cat([key1, keys]), keep=keep)
        self._mtp_carry_h.copy_(hidden[-1:])
        self._mtp_carry_slot.copy_(slots[-1:].reshape(1).to(torch.int32))
        end = prior_local * cp + input_ids.shape[0]
        self._mtp_carry_end.copy_(end.reshape(1).to(torch.int32))

    def _write_boundary(self, out, hidden_slot, residual_slot, picks_slot=None) -> dict:
        """Write an earlier stage's (h, r) shard, and its picks when the next stage needs them, into the seam slots."""
        if hidden_slot is None or residual_slot is None or (self.layout.picks_out and picks_slot is None):
            raise ValueError("a prefill pipeline stage before the last writes its boundary into the seam slots it is "
                             "handed (pipeline_hidden_states_out, pipeline_residual_out"
                             f"{', pipeline_picks_out' if self.layout.picks_out else ''})")
        named = [("pipeline_hidden_states", hidden_slot, out[0]), ("pipeline_residual", residual_slot, out[1])]
        if self.layout.picks_out:
            named.append(("pipeline_picks", picks_slot, out[2]))
        elif picks_slot is not None:
            # a stage whose next starts on a full layer hands no picks: its slot of the uniform boundary holds zeros
            named.append(("pipeline_picks", picks_slot, torch.zeros_like(picks_slot)))
        for name, slot, value in named:
            # copy_ would convert a dtype and broadcast a shape without a word
            if tuple(slot.shape) != tuple(value.shape) or slot.dtype != value.dtype:
                raise ValueError(f"the prefill boundary's {name} is {tuple(value.shape)} {value.dtype}, its seam slot "
                                 f"{tuple(slot.shape)} {slot.dtype}")
            slot.copy_(value)
        return {name: slot for name, slot, _ in named}
