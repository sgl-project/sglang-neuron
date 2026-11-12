# SPDX-License-Identifier: Apache-2.0
"""SGLang's ``cosmos_unipc`` (bh2 UniPC, predict-x0, at the scheduler's order) with each step's CFG combine and
update as one compiled device graph: the coefficients come from the sigmas and config of SGLang's scheduler on the
host and reach the graph as a tensor input with the guidance, and the solver state stays on the device in fp32."""

from __future__ import annotations

import copy

import torch

from sglang_neuron.diffusion.compile_backend import compile_graph
from sglang_neuron.parallel import collectives

NUM_TRAIN_TIMESTEPS = 1000
COMPILER_ARGS = ("--model-type=transformer", "--auto-cast=none", "-O1")

_GRAPHS: dict = {}
_TIMESTEPS: dict = {}
_COEFFICIENTS: dict = {}
_GUIDANCE: dict = {}
UPDATES = ("corrector", "predictor")


def _lam(sigma: torch.Tensor) -> torch.Tensor:
    return torch.log(1.0 - sigma) - torch.log(sigma)


def _bh(h: torch.Tensor, rks: list[torch.Tensor], order: int, corrector: bool):
    """bh2 ``(h_phi_1, b_h, rhos)`` of one predictor or corrector update (predict_x0)."""
    hh = -h
    h_phi_1 = torch.expm1(hh)
    h_phi_k = h_phi_1 / hh - 1.0
    b_h = torch.expm1(hh)
    rks_t = torch.stack([*rks, torch.ones_like(h)])
    rows, rhs, factorial = [], [], 1
    for i in range(1, order + 1):
        rows.append(torch.pow(rks_t, i - 1))
        rhs.append(h_phi_k * factorial / b_h)
        factorial *= i + 1
        h_phi_k = h_phi_k / hh - 1.0 / factorial
    if corrector:
        rhos = [0.5] if order == 1 else torch.linalg.solve(torch.stack(rows), torch.stack(rhs)).tolist()
    else:
        rhos = [0.5] if order == 2 else []
    return float(h_phi_1), float(b_h), rhos


def coefficients(sigmas: torch.Tensor, solver_order: int) -> list[dict]:
    """Per step: ``sigma``, and the ``corrector`` (None at step 0) and ``predictor`` terms ``a``, ``b``, ``c``, ``rk``
    and ``rho``, from the solver's ``n_steps + 1`` fp32 sigmas (the last 0) at ``solver_order`` 1 or 2."""
    sigmas = sigmas.double()
    n_steps = len(sigmas) - 1
    steps, lower, order = [], 0, 1
    for step in range(n_steps):
        coef = {"sigma": float(sigmas[step]), "corrector": None}
        if step > 0:
            sigma_t, sigma_s = sigmas[step], sigmas[step - 1]
            h = _lam(sigma_t) - _lam(sigma_s)
            rks = [(_lam(sigmas[step - (i + 1)]) - _lam(sigma_s)) / h for i in range(1, order)]
            h_phi_1, b_h, rhos = _bh(h, rks, order, corrector=True)
            coef["corrector"] = {"a": float(sigma_t / sigma_s), "b": float((1 - sigma_t) * h_phi_1),
                                 "c": float((1 - sigma_t) * b_h), "rk": [float(r) for r in rks], "rho": rhos}
        order = min(solver_order, n_steps - step, lower + 1)
        sigma_t, sigma_s = sigmas[step + 1], sigmas[step]
        h = _lam(sigma_t) - _lam(sigma_s) if sigma_t > 0 else torch.tensor(float("inf"), dtype=sigmas.dtype)
        rks = [(_lam(sigmas[step - i]) - _lam(sigma_s)) / h for i in range(1, order)]
        h_phi_1, b_h, rhos = _bh(h, rks, order, corrector=False)
        coef["predictor"] = {"a": float(sigma_t / sigma_s), "b": float((1 - sigma_t) * h_phi_1),
                             "c": float((1 - sigma_t) * b_h), "rk": [float(r) for r in rks], "rho": rhos}
        lower = min(lower + 1, solver_order)
        steps.append(coef)
    return steps


def update(samples: dict, velocity: dict, last: dict | None, prev_x0: dict | None, older_x0: dict | None,
           coef: dict) -> tuple[dict, dict, dict]:
    """One UniPC step: the x0 of ``velocity``, the corrector of ``samples`` from ``last`` and the previous x0s, then the
    predictor; returns ``(next, corrected, x0)``, all fp32."""
    x0 = {k: samples[k].float() - coef["sigma"] * velocity[k].float() for k in samples}
    c = coef["corrector"]
    if c is not None:
        hist = [{k: (older_x0[k] - prev_x0[k]) / rk for k in samples} for rk in c["rk"]]
        samples = {k: c["a"] * last[k] - c["b"] * prev_x0[k]
                   - c["c"] * (sum(rho * d[k] for rho, d in zip(c["rho"], hist)) + c["rho"][-1] * (x0[k] - prev_x0[k]))
                   for k in samples}
    p = coef["predictor"]
    hist = [{k: (prev_x0[k] - x0[k]) / rk for k in samples} for rk in p["rk"]]
    predicted = {k: p["a"] * samples[k].float() - p["b"] * x0[k]
                 - p["c"] * sum(rho * d[k] for rho, d in zip(p["rho"], hist)) for k in samples}
    return predicted, {k: v.float() for k, v in samples.items()}, x0


def velocity(preds: list[dict], guidance: dict | None) -> dict:
    """SGLang's velocity of one step's predictions: the single pass, or the CFG combine
    ``uncond + guidance * (cond - uncond)`` of the conditional and unconditional passes, in fp32."""
    if len(preds) == 1:
        return {k: v.float() for k, v in preds[0].items()}
    cond, uncond = preds
    return {k: (uncond[k] + guidance[k] * (cond[k] - uncond[k])).float() for k in cond}


def layout(coef: dict) -> tuple:
    """The structure of one step's coefficients: for the corrector and the predictor, None when it does not run, else
    its ``(rk, rho)`` counts."""
    return tuple(None if coef[u] is None else (len(coef[u]["rk"]), len(coef[u]["rho"])) for u in UPDATES)


def pack(coef: dict) -> list[float]:
    """One step's coefficients as a flat list in ``layout`` order: ``sigma``, then ``a``, ``b``, ``c``, ``rk`` and
    ``rho`` of each update that runs."""
    values = [coef["sigma"]]
    for u in UPDATES:
        if coef[u] is not None:
            values += [coef[u]["a"], coef[u]["b"], coef[u]["c"], *coef[u]["rk"], *coef[u]["rho"]]
    return values


def unpack(values: torch.Tensor, structure: tuple) -> dict:
    """The coefficient dict of ``pack`` from its flat tensor, each value a 0-d tensor."""
    coef, i = {"sigma": values[0]}, 1
    for u, counts in zip(UPDATES, structure):
        if counts is None:
            coef[u] = None
            continue
        n_rk, n_rho = counts
        coef[u] = {"a": values[i], "b": values[i + 1], "c": values[i + 2],
                   "rk": [values[i + 3 + j] for j in range(n_rk)],
                   "rho": [values[i + 3 + n_rk + j] for j in range(n_rho)]}
        i += 3 + n_rk + n_rho
    return coef


def _step_graph(keys: tuple, structure: tuple, passes: int, guided: bool, have: tuple[bool, bool, bool],
                cfg_gather: bool):
    """``velocity`` and ``update`` of one step as a compiled graph over the flattened stream dicts, then the step's
    packed coefficients and, when ``guided``, the per-stream guidance as tensor inputs; cached by the graph's structure
    alone, so every step count, shift and guidance shares its graphs. With ``cfg_gather`` the one pass is this CFG
    rank's branch, and the graph all-gathers the branches over the CFG group in rank order (SGLang's
    ``run_cfg_parallel`` puts branch ``i`` on CFG rank ``i``)."""
    key = (keys, structure, passes, guided, have, cfg_gather)
    graph = _GRAPHS.get(key)
    if graph is None:
        def run(*flat):
            from sglang.multimodal_gen.runtime.distributed.parallel_state import get_cfg_group

            n = len(keys)
            groups = [dict(zip(keys, flat[i * n:(i + 1) * n])) for i in range(1 + passes)]
            i = (1 + passes) * n
            history = []
            for present in have:
                history.append(dict(zip(keys, flat[i:i + n])) if present else None)
                i += n if present else 0
            coef = unpack(flat[i], structure)
            guidance = dict(zip(keys, flat[i + 1])) if guided else None
            preds = groups[1:]
            if cfg_gather:
                both = {k: collectives.all_gather(get_cfg_group(), v[None], dim=0) for k, v in preds[0].items()}
                preds = [{k: v[0] for k, v in both.items()}, {k: v[1] for k, v in both.items()}]
            out = update(groups[0], velocity(preds, guidance), *history, coef)
            return tuple(t for d in out for t in d.values())

        graph = _GRAPHS[key] = compile_graph(run, COMPILER_ARGS)
    return graph


def _device_values(cache: dict, key: tuple, values: list[float], device: torch.device) -> torch.Tensor:
    """``values`` as an fp32 tensor on ``device``, kept in ``cache`` under ``key`` across requests."""
    t = cache.get(key)
    if t is None:
        t = cache[key] = torch.tensor(values, dtype=torch.float32).to(device)
    return t


def _timestep(n_steps: int, shift: float, step: int, tick, device: torch.device) -> torch.Tensor:
    """The model timestep ``[1]`` of one solver step on ``device``, as SGLang computes it (``tick / 1000`` in fp32);
    kept on the device across requests, since it depends only on the schedule."""
    key = (n_steps, shift, step, str(device))
    t = _TIMESTEPS.get(key)
    if t is None:
        t = (torch.tensor(float(tick), dtype=torch.float32) / NUM_TRAIN_TIMESTEPS).reshape(1)
        t = _TIMESTEPS[key] = t.to(device)
    return t


def device_unipc(samples: dict, predict, *, scheduler, n_steps: int, shift: float, guidance: dict | None = None,
                 device: torch.device, cfg_gather: bool = False):
    """SGLang's ``cosmos_unipc`` solve: SGLang's scheduler gives the sigmas and the model ticks, ``predict(samples,
    timestep)`` the raw predictions of each pass (one, CFG's conditional and unconditional, or with ``cfg_gather``
    this CFG rank's branch), and every step's combine and update is one device graph. Raises ``NotImplementedError``
    for a scheduler config other than bh2, predict-x0, order 1 or 2, ``lower_order_final`` and no disabled corrector."""
    config = scheduler.config
    if ((config.solver_type, config.predict_x0, config.lower_order_final, tuple(config.disable_corrector))
            != ("bh2", True, True, ()) or config.solver_order not in (1, 2)):
        raise NotImplementedError(f"the device UniPC implements bh2 predict-x0 UniPC of order 1 or 2 with "
                                  f"lower_order_final and every corrector, not {dict(config)}")
    schedule = copy.deepcopy(scheduler)
    schedule.set_timesteps(n_steps, device="cpu", shift=shift)
    coefs = coefficients(schedule.sigmas.float(), config.solver_order)
    keys = tuple(samples)
    last = prev_x0 = older_x0 = None
    for step, (tick, coef) in enumerate(zip(schedule.timesteps, coefs)):
        preds = predict(samples, _timestep(n_steps, shift, step, tick, device))
        have = (last is not None, prev_x0 is not None, older_x0 is not None)
        flat = [d[k] for d in (samples, *preds, last, prev_x0, older_x0) if d is not None for k in keys]
        flat.append(_device_values(_COEFFICIENTS, (n_steps, shift, config.solver_order, step, str(device)), pack(coef),
                                   device))
        guided = len(preds) > 1 or cfg_gather
        if guided:
            values = [float(guidance[k]) for k in keys]
            flat.append(_device_values(_GUIDANCE, (keys, tuple(values), str(device)), values, device))
        out = _step_graph(keys, layout(coef), len(preds), guided, have, cfg_gather)(*flat)
        n = len(keys)
        samples, last, x0 = (dict(zip(keys, out[i * n:(i + 1) * n])) for i in range(3))
        older_x0, prev_x0 = prev_x0, x0
    return samples
