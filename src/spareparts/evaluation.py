"""Évaluation répétée des politiques et réglage de la politique P1 (§ 7.1 et § 7.3)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from scipy import stats

from spareparts.config import ScenarioConfig
from spareparts.policies import FixedThresholdPolicy, Policy
from spareparts.sim import FleetSimulation


def replicate(
    cfg: ScenarioConfig,
    policy_factory: Callable[[], Policy],
    seeds: Iterable[int],
    days: int | None = None,
) -> list[dict[str, float]]:
    """Simule une politique sur plusieurs graines (nombres aléatoires communs entre politiques)."""
    out = []
    for seed in seeds:
        sim = FleetSimulation(cfg, seed=seed, log_events=False)
        out.append(sim.run(policy_factory(), days=days).kpis)
    return out


def summarize(runs: list[dict[str, float]], level: float = 0.95) -> dict[str, dict[str, float]]:
    """Moyenne et intervalle de confiance de Student pour chaque indicateur."""
    summary = {}
    n = len(runs)
    for key in runs[0]:
        values = np.array([r[key] for r in runs], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary[key] = {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
            continue
        mean = float(finite.mean())
        if n > 1 and finite.size > 1:
            half = float(stats.t.ppf(0.5 + level / 2, finite.size - 1) * stats.sem(finite))
        else:
            half = 0.0
        summary[key] = {"mean": mean, "low": mean - half, "high": mean + half}
    return summary


def optimize_fixed_threshold(
    cfg: ScenarioConfig,
    n_trials: int = 40,
    seeds: Iterable[int] = (101, 102, 103),
    rng_seed: int = 0,
    progress: Callable[[int, float, float], None] | None = None,
) -> tuple[dict, float, list[dict]]:
    """Recherche aléatoire des seuils (s, S) et des facteurs d'intervalle de maintenance.

    Le premier essai correspond à la politique P0 ; chaque candidat est évalué sur les mêmes
    graines afin que les écarts de coût tiennent à la politique et non au hasard.
    """
    seeds = list(seeds)
    rng = np.random.default_rng(rng_seed)
    s0 = np.array([p.s for p in cfg.parts])
    S0 = np.array([p.S for p in cfg.parts])
    K = cfg.n_components

    def sample() -> dict:
        s = np.array([rng.integers(0, 3 * a + 3) for a in s0])
        width = np.array(
            [rng.integers(1, max(3, 3 * (b - a)) + 1) for a, b in zip(s0, S0, strict=True)]
        )
        return {
            "s": s.tolist(),
            "S": (s + width).tolist(),
            "pm_factor": np.round(rng.uniform(0.5, 1.6, K), 3).tolist(),
        }

    candidates = [{"s": s0.tolist(), "S": S0.tolist(), "pm_factor": [1.0] * K}]
    candidates += [sample() for _ in range(n_trials - 1)]
    history = []
    best, best_cost = candidates[0], float("inf")
    for i, params in enumerate(candidates):
        runs = replicate(cfg, lambda p=params: FixedThresholdPolicy(name="P1", **p), seeds)
        cost = float(np.mean([r["cout_total"] for r in runs]))
        history.append({**params, "cout_total": cost})
        if cost < best_cost:
            best, best_cost = params, cost
        if progress is not None:
            progress(i, cost, best_cost)
    return best, best_cost, history
