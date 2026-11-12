# SPDX-License-Identifier: Apache-2.0
"""Static-graph chunk runtime of the Neuron LingBot-World DiT over the paged KV pool.

One AR chunk (3 latent frames) is 4 denoising passes and one ``t = 0`` commit pass over the
session's cached history. This module owns the compiled graphs and the per-chunk launch order;
the KV pool and its session paging are :mod:`.kv_pool`'s: the pool's per-layer views are bound once
(:meth:`LingBotDiTRuntime.bind_kv_cache`) and every pass takes them as persistent inputs, plus
that pass's paged inputs (any object with the fields of
:class:`~sglang_neuron.diffusion.attention.paged_attention.PagedAttentionInputs` and a
``commit`` flag: per-head block table, mask bounds, and on the commit pass the per-token update
indices; optionally ``host_bounds``, the bounds as host ints, for the host-side plan check
:meth:`LingBotDiTRuntime.check_paged_plan`).

What is fixed (no switches):

* **Paged cache attention** on every layer, warm-up and steady state alike: the paged FA2 kernel
  reads the listed history from the pool through the block table, masks it with
  ``(sink_end, window_start, kv_len)`` and attends it with the chunk's K/V, gathered over CP in
  the kernel (where the kernel does not run, nkilib ``attention_cte`` over the visible history
  gathered into a dense prefix); the commit pass writes the chunk's K/V into the pool through
  ``kv_cache_update_idx`` (in place). Probes write nothing. Once the history is full, the
  steady-state twin graphs attend without the mask (bit-identical).
* **In-graph chunk prep.** One prep graph per chunk builds the RoPE tables (host float64
  temporal factors + device-constant spatial factors, bit-exact) and the per-block camera
  modulation from in-graph Plücker rays of a ``[3, 16]`` pose/intrinsics tensor.
  :meth:`LingBotDiTRuntime.launch_prep` is called for chunk ``n + 1`` while chunk ``n`` runs
  (prefetch). The text cross-attention K/V are projected once per prompt.
* **Host timestep embedding.** The packed FP32 timestep embedding of the session's timesteps
  (and the commit's ``t = 0``) is computed once per schedule on the host and uploaded; no graph.
* **Sampler in the pass graphs, one host sync per chunk.** Latent, noise and the DMD update stay
  on the device, with no sampler graph: every pass graph starts with the previous step's update
  (on the latent and flow the previous pass returned) and builds its own input from it; the
  commit pass applies the last step's update and returns the clean latent. Launches are bounded
  (:attr:`LingBotDiTRuntime.max_inflight`): the runtime's execution queue overflows at ~10.

Graphs (per resolution; shared by every chunk and every session): text K/V, chunk prep, denoise
step and commit, plus the steady-state twins of the last two (:meth:`LingBotDiTRuntime.graph_for`)
-- 6 in total.
"""

from __future__ import annotations

import collections
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from sglang_neuron.diffusion.models.lingbot_world.chunk_prep import (
    camera_ray_params,
    lingbot_rope_spatial_factors,
    lingbot_rope_temporal_factors,
)

#: neuronx-cc flags of every DiT graph (the set the model is validated with).
DIT_COMPILER_ARGS = (
    "-O1",
    "--hbm-scratchpad-page-size=2048",
    "--internal-hlo2tensorizer-options=--hierarchy-splitter=false",
)
#: Outstanding graph launches before the driver waits for the oldest one. The runtime holds a
#: bounded number of unfinished executions per core ("Execution Queue Full" at 10 in flight).
#: 8 lets a serving step queue the next chunk whole (its prep, 4 denoise steps and the commit)
#: across the chunk boundary, so the device does not idle while the host returns the chunk.
DEFAULT_MAX_INFLIGHT = 8


def _await_device_tensor(tensor: torch.Tensor) -> None:
    """Block the host until ``tensor`` is computed (one element comes back to the host)."""
    tensor.reshape(-1)[:1].cpu()


# ------------------------------------------------------------------------------- graph modules


class _TextKV(nn.Module):
    """``[1, 512, 4096]`` UMT5 context -> every block's text K/V (once per prompt)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, context):
        return self.model.project_cross_attention_kv(context)


class _Temb(nn.Module):
    """``[1]`` timestep -> packed ``[1, 7, dim]`` FP32 embedding (host; once per schedule)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, timestep):
        return self.model.packed_timestep_embedding(timestep)


def _step_scalars(timesteps: Sequence[float]) -> torch.Tensor:
    """Per step float32 ``[sigma, 1 - next_sigma, next_sigma]`` (``next_sigma = 0`` at the end).

    ``sigma = t / 1000``. The float32 rounding of the Python floats is the host loop's, so the
    update of :class:`_Pass` is bit-identical to vLLM-Omni's ``apply_transition`` on CPU.
    """
    rows = []
    for k, t in enumerate(timesteps):
        s_next = timesteps[k + 1] / 1000.0 if k + 1 < len(timesteps) else 0.0
        rows.append([t / 1000.0, 1.0 - s_next, s_next])
    return torch.tensor(rows, dtype=torch.float32)


class _Prep(nn.Module):
    """``(cos, sin, *camera_modulation)`` of one chunk from its small per-chunk tensors."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, camera_params, rope_temporal, h_cos, h_sin, w_cos, w_sin):
        return self.model.chunk_prep(camera_params, rope_temporal, h_cos, h_sin, w_cos, w_sin)


class _Pass(nn.Module):
    """One DiT pass over the paged history (a denoise step or the commit), after the previous
    step's sampler update.

    Inputs: the previous latent ``x_prev`` and flow ``flow_prev`` (fp32 / bf16 ``[1, 16, F, H,
    W]``), the re-noise draw ``noise`` (fp32), the update's ``[3]`` fp32 step scalars
    (:func:`_step_scalars`; ``[0, 1, 0]`` on the first pass: ``x_t = x_prev``), ``condition``
    (fp32 ``[1, 20, F, H, W]``), packed ``temb``, RoPE ``cos``/``sin``, the pass's per-head
    ``block_table`` and ``bounds``, on the commit pass ``kv_cache_update_idx``, then the pool
    views (``k_0, v_0, ...``), text K/V and camera modulation. A denoise step returns
    ``(flow, x_t)``; the commit pass (K/V-only, it writes the pool in place) returns one element
    read back from the last layer's V pool (a sync on it orders after every write) and the clean
    latent ``x_t``.

    The update is the DMD transition of vLLM-Omni's ``LingBotDMDBlockRunner.apply_transition``:
    the clean prediction ``x0 = x_prev - sigma * flow_prev``, re-noised at the next sigma,
    ``(1 - next_sigma) * x0 + next_sigma * noise``. Upstream takes Python-float sigmas and draws
    the noise inside; here the scalars are a tensor input and the noise is passed in, so one
    static graph serves every step (the last step's ``[sigma, 1, 0]`` gives ``x0``). Updating
    at the start of the next pass rather than at the end of this one keeps the flow a plain
    graph output, which runs faster than the end-of-pass form.
    """

    def __init__(self, model, commit: bool, num_history_blocks: int, all_visible: bool = False):
        super().__init__()
        self.model = model
        self.commit = commit
        self.num_history_blocks = num_history_blocks
        self.all_visible = all_visible

    def forward(self, x_prev, flow_prev, noise, step, condition, temb, cos, sin, *rest):
        from sglang_neuron.diffusion.attention.paged_attention import PagedAttentionInputs

        x_t = step[1] * (x_prev - step[0] * flow_prev.float()) + step[2] * noise
        hidden = torch.cat((x_t, condition), 1).to(self.model.dtype)
        layers = self.model.num_layers
        block_table, bounds, rest = rest[0], rest[1], rest[2:]
        kv_idx = None
        if self.commit:
            kv_idx, rest = rest[0], rest[1:]
        pool = rest[: 2 * layers]
        cross_kv = rest[2 * layers : 4 * layers]
        camera_mod = rest[4 * layers :]
        paged = PagedAttentionInputs(
            block_table, bounds, kv_idx, self.num_history_blocks, self.all_visible
        )
        out = self.model(
            hidden,
            None,
            None,
            cos,
            sin,
            tuple(pool),
            tuple(cross_kv),
            paged,
            camera_modulation=tuple(camera_mod),
            kv_only=self.commit,
            timestep_embedding=temb,
        )
        if self.commit:
            return pool[-1].reshape(-1)[:1].float() + 0.0, x_t
        return out[0], x_t


# ------------------------------------------------------------------------------- host state


@dataclass
class ChunkPrep:
    """One chunk's launched device inputs (prep outputs not yet synced)."""

    start_frame: int
    cos: torch.Tensor
    sin: torch.Tensor
    camera_modulation: tuple[torch.Tensor, ...]
    x_t: torch.Tensor  # fp32 noise_0
    condition: torch.Tensor  # fp32 [1, 20, F, H, W]
    noises: tuple[torch.Tensor, ...]  # fp32 re-noise draws for steps 2..4


@dataclass
class SessionConstants:
    """Per-schedule device constants: the timestep embeddings and the sampler's step scalars."""

    timesteps: tuple[float, ...]
    temb: tuple[torch.Tensor, ...]  # one per step, then the commit (t = 0)
    coefficients: tuple[torch.Tensor, ...]  # one [3] fp32 per step (_step_scalars)
    first_pass: torch.Tensor  # [0, 1, 0]: the first pass starts from the initial latent


class LingBotDiTRuntime:
    """Compiled-graph driver of one Neuron LingBot DiT (this rank's TP x CP shard)."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | str,
        latent_height: int = 60,
        latent_width: int = 104,
        frames_per_chunk: int = 3,
        compile_graphs: bool = True,
        compiler_args: Sequence[str] = DIT_COMPILER_ARGS,
        max_inflight: int = DEFAULT_MAX_INFLIGHT,
        steady_graphs: bool = True,
    ) -> None:
        if max_inflight < 2:
            raise ValueError("max_inflight must be at least 2")
        self.model = model
        self.device = torch.device(device)
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.frames_per_chunk = frames_per_chunk
        self.compile_graphs = compile_graphs
        self.max_inflight = max_inflight
        # Steady-state twins of the pass graphs (2 more graphs; see graph_for).
        self.steady_graphs = bool(steady_graphs)
        self._kv: tuple[torch.Tensor, ...] | None = None
        self._opts = None
        if compile_graphs:
            from sglang_neuron.diffusion.compile_backend import register as get_compile_backend_name

            self._opts = dict(
                backend=get_compile_backend_name(),
                fullgraph=True,
                dynamic=False,
                options={"compiler_args": list(compiler_args)},
            )
        # Headroom for the graphs sharing a code object (and test-only variants).
        dynamo = torch._dynamo.config
        dynamo.recompile_limit = max(dynamo.recompile_limit, 64)
        dynamo.accumulated_recompile_limit = max(dynamo.accumulated_recompile_limit, 512)
        self._graphs: dict[object, object] = {}
        # One entry per unsynced launch (device only): that launch's output tensors.
        self._inflight: collections.deque[tuple[torch.Tensor, ...]] = collections.deque()
        #: Host time in ``_launch`` since the last :meth:`take_launch_stats`: waiting for a
        #: launch slot (the device) vs the graph call itself (host dispatch).
        self.launch_stats = {"wait_ms": 0.0, "call_ms": 0.0, "launches": 0}
        _, p_h, p_w = model.patch_size
        self.grid_h, self.grid_w = latent_height // p_h, latent_width // p_w
        self._rope_hw = tuple(
            t.to(self.device) for t in lingbot_rope_spatial_factors(self.grid_h, self.grid_w)
        )
        self._session_constants: dict[tuple[float, ...], SessionConstants] = {}
        self._zero_flows: dict[tuple[int, ...], torch.Tensor] = {}
        self._zero_camera: dict = {}  # without_camera's zero modulation, per shape

    # -- graphs ------------------------------------------------------------------------

    def bind_kv_cache(self, pool_views: Sequence[torch.Tensor]) -> None:
        """Bind the paged pool's per-layer views ``(k_0, v_0, ...)`` (once, at load)."""
        if len(pool_views) != 2 * self.model.num_layers:
            raise ValueError(
                f"expected {2 * self.model.num_layers} pool views, got {len(pool_views)}"
            )
        self._kv = tuple(pool_views)

    @property
    def kv_views(self) -> tuple[torch.Tensor, ...]:
        if self._kv is None:
            raise RuntimeError("the paged KV pool is not bound (bind_kv_cache)")
        return self._kv

    def _graph(self, key, factory):
        graph = self._graphs.get(key)
        if graph is None:
            module = factory().eval()
            graph = torch.compile(module, **self._opts) if self._opts is not None else module
            self._graphs[key] = graph
        return graph

    def expected_graph_keys(self, num_history_blocks: int) -> list:
        """Every graph a full session compiles (warm-up and steady chunks), for the
        static-graph checks: compare ``sorted(map(str, ...))`` with :meth:`graph_keys`."""
        keys = ["text_kv", "prep"]
        for kind in ("denoise", "commit"):
            keys.append((kind, num_history_blocks))
            if self.steady_graphs:
                keys.append((kind, num_history_blocks, "steady"))
        return keys

    @staticmethod
    def steady_state(paged) -> bool:
        """Whether every listed history position of this pass is visible (the plan's
        ``history_full``; False when the plan does not say): the pass then runs a steady-state
        twin graph (:meth:`graph_for`)."""
        return bool(getattr(paged, "history_full", False))

    def graph_keys(self) -> list:
        """Graphs built so far (for the static-graph check)."""
        return list(self._graphs)

    def runs_unmasked(self, paged) -> bool:
        """Whether this pass runs a steady-state twin (:meth:`graph_for`), which attends without
        the history mask: with ``steady_graphs`` on, its plan is ``history_full`` or sets
        ``all_visible``. :meth:`check_paged_plan` then verifies that its bounds mask nothing."""
        return self.steady_graphs and (
            bool(getattr(paged, "all_visible", False)) or self.steady_state(paged)
        )

    def graph_for(self, kind: str, paged):
        """The ``"denoise"`` or ``"commit"`` graph for one pass's paged inputs.

        Each has a steady-state twin (key suffix ``"steady"``), picked once every listed history
        position is visible (:meth:`steady_state`, from chunk 5 at 480p): its attention runs
        without the mask (``PagedAttentionInputs.all_visible``; bit-identical).
        """
        steady = self.runs_unmasked(paged)
        nb = paged.num_history_blocks
        key = (kind, nb) + (("steady",) if steady else ())
        commit = {"denoise": False, "commit": True}[kind]
        return self._graph(key, lambda: _Pass(self.model, commit, nb, all_visible=steady))

    def pass_args(self, paged, text_kv, prep) -> tuple:
        """The paged, pool, text K/V and camera inputs of one pass (after the per-step inputs)."""
        head = (paged.block_table, paged.bounds)
        if paged.commit:
            head = head + (paged.kv_cache_update_idx,)
        return (*head, *self.kv_views, *text_kv, *prep.camera_modulation)

    # -- launches ----------------------------------------------------------------------

    def take_launch_stats(self) -> dict:
        """The launch stats since the last call (rounded), then reset."""
        stats = {k: round(v, 1) if isinstance(v, float) else v for k, v in self.launch_stats.items()}
        self.launch_stats = {"wait_ms": 0.0, "call_ms": 0.0, "launches": 0}
        return {f"launch_{k}": v for k, v in stats.items()}

    def _launch(self, graph, *args):
        """Enqueue one graph; first waits for the oldest launch past ``max_inflight``."""
        t0 = time.perf_counter()
        while len(self._inflight) >= self.max_inflight:
            _await_device_tensor(self._inflight.popleft()[0])
        t1 = time.perf_counter()
        out = graph(*args)
        self.launch_stats["wait_ms"] += (t1 - t0) * 1e3
        self.launch_stats["call_ms"] += (time.perf_counter() - t1) * 1e3
        self.launch_stats["launches"] += 1
        if self.device.type != "cpu":
            self._inflight.append(tuple(out) if isinstance(out, (tuple, list)) else (out,))
        return out

    def synchronize(self, tensor: torch.Tensor) -> None:
        """Wait for ``tensor`` (any output of a launch, e.g. ``x0``) on the host.

        Every tracked launch up to and including the one that produced ``tensor`` is then done
        and leaves the in-flight queue; later launches stay. A tensor no tracked launch
        produced is awaited but changes nothing in the queue.
        """
        _await_device_tensor(tensor)
        producer = next(
            (i for i, outs in enumerate(self._inflight) if any(t is tensor for t in outs)), None
        )
        if producer is not None:
            for _ in range(producer + 1):
                self._inflight.popleft()

    def free_launch_slots(self) -> int:
        """Launches that can be enqueued before the next one waits for the oldest:
        ``max_inflight`` minus the tracked launches not yet synced (a finished but unsynced
        launch still counts). The pipeline sizes its early launches of the next chunk by it."""
        return self.max_inflight - len(self._inflight)

    # -- per prompt / session ----------------------------------------------------------

    def text_kv(self, prompt_embeds: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Every block's text K/V from the ``[1, 512, 4096]`` UMT5 embedding (one graph call)."""
        context = prompt_embeds.to(device=self.device, dtype=self.model.dtype)
        out = self._launch(self._graph("text_kv", lambda: _TextKV(self.model)), context)
        return tuple(out)

    def session_constants(self, timesteps: Sequence[float]) -> SessionConstants:
        """The schedule's packed timestep embeddings (+ the commit's t = 0) and step scalars.

        Computed on the host (FP32, from host copies of the time MLPs' weights) once per
        schedule and uploaded.
        """
        key = tuple(float(t) for t in timesteps)
        cached = self._session_constants.get(key)
        if cached is not None:
            return cached
        model = self.model
        host_params = {
            f"model.{name}.{p}": t.detach().cpu()
            for name in ("time_embedding", "time_projection")
            for p, t in getattr(model, name).named_parameters()
        }
        with torch.no_grad():
            temb = tuple(
                torch.func.functional_call(
                    _Temb(model), host_params, (torch.tensor([t], dtype=torch.float32),)
                ).to(self.device)
                for t in (*key, 0.0)
            )
        scalars = _step_scalars(key)
        coefficients = tuple(scalars[k].contiguous().to(self.device) for k in range(len(key)))
        first = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).to(self.device)
        cached = SessionConstants(
            timesteps=key, temb=temb, coefficients=coefficients, first_pass=first
        )
        self._session_constants[key] = cached
        return cached

    # -- per chunk ---------------------------------------------------------------------

    def launch_prep(
        self,
        *,
        start_frame: int,
        poses: torch.Tensor,
        intrinsics: torch.Tensor,
        noise: torch.Tensor,
        condition: torch.Tensor,
    ) -> ChunkPrep:
        """Upload chunk inputs and launch its prep graph (no sync; call ahead as a prefetch).

        ``poses`` ``[F, 3|4, 4]`` / ``intrinsics`` ``[F, 4]``: the chunk's prepared (framewise,
        conditioning-frame) camera; ``noise`` ``[4, 1, 16, F, H, W]``: the initial latent and
        the three re-noise draws; ``condition`` ``[1, 20, F, H, W]`` (mask + image latent).
        """
        if noise.shape[0] < 1:
            raise ValueError("noise needs the initial latent draw")
        params = camera_ray_params(poses, intrinsics).to(self.device)
        rope_t = lingbot_rope_temporal_factors(self.frames_per_chunk, start_frame=start_frame)
        return self.launch_prep_uploaded(
            start_frame=start_frame,
            params=params,
            rope_t=rope_t.to(self.device),
            x_t=noise[0].float().to(self.device),
            noises=tuple(noise[k].float().to(self.device) for k in range(1, noise.shape[0])),
            condition=condition.float().to(self.device),
        )

    def launch_prep_uploaded(
        self, *, start_frame: int, params, rope_t, x_t, noises, condition
    ) -> ChunkPrep:
        """:meth:`launch_prep` with its inputs already on the device (no upload): the camera ray
        parameters, the temporal RoPE factors, the initial latent, the re-noise draws and the
        condition."""
        out = self._launch(
            self._graph("prep", lambda: _Prep(self.model)), params, rope_t, *self._rope_hw
        )
        return ChunkPrep(
            start_frame=start_frame,
            cos=out[0],
            sin=out[1],
            camera_modulation=tuple(out[2:]),
            x_t=x_t,
            condition=condition,
            noises=tuple(noises),
        )

    def without_camera(self, prep: ChunkPrep) -> ChunkPrep:
        """``prep`` with no camera condition: zero camera ``(scale, shift)``, so every block's
        ``(1 + scale) * x + shift`` is ``x``, as SGLang's pipeline skips the injection. The zeros
        are allocated once per shape and reused (the pass graphs only read them)."""
        mods = prep.camera_modulation
        key = tuple((tuple(m.shape), m.dtype) for m in mods)
        zeros = self._zero_camera.get(key)
        if zeros is None:
            zeros = tuple(torch.zeros(m.shape, dtype=m.dtype).to(self.device) for m in mods)
            self._zero_camera[key] = zeros
        return replace(prep, camera_modulation=zeros)

    def _zero_flow(self, like: torch.Tensor) -> torch.Tensor:
        """The first pass's ``flow_prev`` (zeros in the model dtype; never written, shared by
        every chunk)."""
        zero = self._zero_flows.get(tuple(like.shape))
        if zero is None:
            zero = torch.zeros(like.shape, dtype=self.model.dtype).to(self.device)
            self._zero_flows[tuple(like.shape)] = zero
        return zero

    def _update_inputs(self, prep, constants, k):
        """``(noise, step scalars)`` of the update pass ``k`` starts with (``k = steps``: commit).

        Pass ``k >= 1`` applies step ``k - 1``'s update (re-noise draw ``k - 1``); the first pass
        takes the initial latent as it is, and the commit's ``next_sigma = 0`` ignores its noise.
        """
        if k == 0:
            return prep.noises[0], constants.first_pass
        return prep.noises[min(k - 1, len(prep.noises) - 1)], constants.coefficients[k - 1]

    def check_paged_plan(self, paged) -> None:
        """Raise ``ValueError`` before launch if the pass's plan cannot run on this path.

        * A commit plan must cover whole pool blocks: the paged FA2 kernel writes the commit
          block by block. The plan states it (``paged.commit_block_aligned``) or carries a CPU
          ``kv_cache_update_idx`` to check; a device-resident index without the flag is refused.
        * When the paged attention compacts the history block-wise (``attention_cte``), the
          plan's ``sink_end`` and ``window_start`` must be multiples of the pool's block size.
        * When the pass attends without the mask (:meth:`runs_unmasked`: a steady-state twin, or
          a plan that sets ``all_visible``), its bounds must mask no listed history position
          (``p < num_history_blocks * block``): ``p < sink_end`` or
          ``window_start <= p < kv_len`` for every one.

        Host-side, no device sync: it reads ``paged.host_bounds`` (the plan's bounds as host
        ints) when given, else a CPU ``paged.bounds``; when a check applies, device-resident
        bounds without a host copy are refused (the plan must carry ``host_bounds``).
        """
        from sglang_neuron.diffusion.attention.paged_attention import (
            commit_block_aligned,
            uses_block_aligned_history,
        )

        key_pool, value_pool = self.kv_views[0], self.kv_views[1]
        if paged.commit:
            aligned = getattr(paged, "commit_block_aligned", None)
            if aligned is None:
                idx = paged.kv_cache_update_idx
                if idx.device.type != "cpu":
                    raise ValueError(
                        "the commit's kv_cache_update_idx is on the device: the plan must carry "
                        "commit_block_aligned"
                    )
                nb, n, block, _ = value_pool.shape
                aligned = commit_block_aligned(idx, block, nb * n * block)
            if not aligned:
                raise ValueError(
                    "the commit does not cover whole pool blocks (frame-sized blocks and "
                    "frame-aligned chunks are required)"
                )
        block_aligned = uses_block_aligned_history(key_pool)
        all_visible = self.runs_unmasked(paged)
        if not (block_aligned or all_visible):
            return
        bounds = getattr(paged, "host_bounds", None)
        if bounds is None:
            if paged.bounds.device.type != "cpu":
                path = "the attention_cte path" if block_aligned else "all_visible"
                raise ValueError(
                    f"paged plan bounds are on the device: the plan must carry host_bounds for "
                    f"{path}"
                )
            bounds = paged.bounds.reshape(-1).tolist()
        sink_end, window_start, kv_len = (int(b) for b in bounds[:3])
        # The block size from V (token-major [num_blocks, N, block, D]); K may be d-major.
        block = value_pool.shape[2]
        if all_visible:
            history_tokens = int(paged.num_history_blocks) * block
            # Positions [max(sink_end, 0), history_tokens) must all lie in [window_start, kv_len).
            first_unpinned = max(sink_end, 0)
            masks_nothing = first_unpinned >= history_tokens or (
                window_start <= first_unpinned and kv_len >= history_tokens
            )
            if not masks_nothing:
                raise ValueError(
                    f"paged plan bounds {(sink_end, window_start, kv_len)} mask listed history "
                    f"positions (of {history_tokens}), but the pass attends without the mask "
                    "(all_visible)"
                )
        if block_aligned and (sink_end % block or window_start % block):
            raise ValueError(
                f"paged plan bounds {tuple(int(b) for b in bounds)}: sink_end and window_start "
                f"must be multiples of the block size {block} for the attention_cte path"
            )

    def _launch_pass(self, graph, prep, paged, text_kv, constants, k, x_prev, flow_prev):
        self.check_paged_plan(paged)
        noise, step = self._update_inputs(prep, constants, k)
        return self._launch(
            graph,
            x_prev,
            flow_prev,
            noise,
            step,
            prep.condition,
            constants.temb[k],
            prep.cos,
            prep.sin,
            *self.pass_args(paged, text_kv, prep),
        )

    def launch_denoise(self, prep, paged, text_kv, constants, k, x_prev=None, flow_prev=None):
        """Enqueue denoise step ``k`` after step ``k - 1``'s update; returns ``(flow, x_t)``.

        ``x_prev`` / ``flow_prev``: the previous step's ``(x_t, flow)`` (default for ``k = 0``:
        the chunk's initial latent and a zero flow).
        """
        if k == 0:
            x_prev, flow_prev = prep.x_t, self._zero_flow(prep.x_t)
        graph = self.graph_for("denoise", paged)
        return self._launch_pass(graph, prep, paged, text_kv, constants, k, x_prev, flow_prev)

    def launch_commit(self, prep, paged, text_kv, constants, x_prev, flow_prev):
        """Enqueue the commit pass after the last step's update (writes the chunk's K/V into the
        pool); returns ``(done, x0)``: the sync handle and the clean latent (device fp32)."""
        if not paged.commit:
            raise RuntimeError("the commit pass needs a committing context (update indices)")
        graph = self.graph_for("commit", paged)
        steps = len(constants.coefficients)
        return self._launch_pass(graph, prep, paged, text_kv, constants, steps, x_prev, flow_prev)

    def run_chunk(
        self,
        prep: ChunkPrep,
        paged_for,
        text_kv: Sequence[torch.Tensor],
        constants: SessionConstants,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enqueue the chunk: 4 denoise steps and the commit pass (each after its update).

        ``paged_for(commit: bool)`` returns the next pass's paged inputs; it is called before
        every pass (4 denoise steps, then the commit). Returns ``(x0, done)``: the clean latent
        (device fp32, the commit graph's ``out[1]``) and the commit graph's sync handle
        (``out[0]``); :meth:`synchronize` on either one is the chunk's one host sync.
        """
        steps = len(constants.coefficients)
        if len(prep.noises) < steps - 1:
            raise ValueError(f"{steps} steps need {steps - 1} re-noise draws")
        flow, x_t = self.launch_denoise(prep, paged_for(False), text_kv, constants, 0)
        for k in range(1, steps):
            flow, x_t = self.launch_denoise(
                prep, paged_for(False), text_kv, constants, k, x_t, flow
            )
        done, x0 = self.launch_commit(prep, paged_for(True), text_kv, constants, x_t, flow)
        return x0, done


__all__ = [
    "DEFAULT_MAX_INFLIGHT",
    "DIT_COMPILER_ARGS",
    "ChunkPrep",
    "LingBotDiTRuntime",
    "SessionConstants",
]
