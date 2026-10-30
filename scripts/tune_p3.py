"""Règle les seuils de la politique prédictive P3 sur une grille, comme P1 (§ 7.1).

Exemple : uv run python scripts/tune_p3.py
Résultat : configs/policies/p3_s1_nominal.yaml
"""

from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

from spareparts.config import load_scenario
from spareparts.policies import TUNED_DIR, _predictor_path
from spareparts.policies.predictive import PredictivePolicy
from spareparts.sim import FleetSimulation

# Grille élargie après un premier passage dont l'optimum touchait les bornes
# (seuil 0,05 ; facteur d'âge 1,5 ; marge 0,5).
GRID = {
    "pm_threshold": [0.02, 0.05, 0.1],
    "age_factor": [1.5, 2.0, 3.0],
    "z": [0.0, 0.5],
}


def _cost(args) -> float:
    scenario, params, seed = args
    path, arch = _predictor_path(scenario)
    sim = FleetSimulation(load_scenario(scenario), seed=seed, log_events=False)
    return sim.run(PredictivePolicy(path, arch, **params)).kpis["cout_total"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    args = parser.parse_args()

    combos = [dict(zip(GRID, values, strict=True)) for values in itertools.product(*GRID.values())]
    jobs = [(args.scenario, p, s) for p in combos for s in args.seeds]
    with ProcessPoolExecutor() as pool:
        costs = list(pool.map(_cost, jobs))
    n = len(args.seeds)
    means = [float(np.mean(costs[i * n : (i + 1) * n])) for i in range(len(combos))]
    for params, cost in sorted(zip(combos, means, strict=True), key=lambda x: x[1])[:5]:
        print(f"{cost:10.1f}  {params}")
    best = combos[int(np.argmin(means))]
    path = TUNED_DIR / f"p3_{args.scenario}.yaml"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Seuils de P3 retenus sur une grille de {len(combos)} combinaisons.\n")
        fh.write(f"# Coût moyen sur les graines {args.seeds} : {min(means):.1f} UR\n")
        yaml.safe_dump(best, fh, allow_unicode=True, sort_keys=False)
    print(f"enregistré dans {path}")


if __name__ == "__main__":
    main()
