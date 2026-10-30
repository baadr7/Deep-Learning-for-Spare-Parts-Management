"""Choisit, pour chaque agent PPO, entre son action la plus probable et le tirage de ses actions.

Le choix se fait sur les graines de réglage de P1 et P3, jamais sur celles de l'évaluation
finale, et il est enregistré avec les autres paramètres réglés.

Exemple : uv run python scripts/choose_agent_mode.py
Résultat : configs/policies/p4_s1_nominal.yaml et p5_s1_nominal.yaml
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

from spareparts.config import load_scenario
from spareparts.policies import POLICY_LABELS, TUNED_DIR
from spareparts.policies.learned import LEVEL_OF, LearnedPolicy, model_path
from spareparts.sim import FleetSimulation


def _cost(args) -> float:
    scenario, code, deterministic, seed = args
    policy = LearnedPolicy(
        model_path(code, scenario),
        LEVEL_OF[code],
        name=code,
        label=POLICY_LABELS[code],
        deterministic=deterministic,
    )
    sim = FleetSimulation(load_scenario(scenario), seed=seed, log_events=False)
    return sim.run(policy).kpis["cout_total"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--agents", nargs="+", default=["P4", "P5"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    args = parser.parse_args()

    agents = [c for c in args.agents if model_path(c, args.scenario).exists()]
    jobs = [(args.scenario, c, d, s) for c in agents for d in (True, False) for s in args.seeds]
    with ProcessPoolExecutor() as pool:
        costs = list(pool.map(_cost, jobs))
    n = len(args.seeds)
    for i, code in enumerate(agents):
        start = 2 * i * n
        mode = float(np.mean(costs[start : start + n]))
        sampled = float(np.mean(costs[start + n : start + 2 * n]))
        deterministic = mode <= sampled
        print(f"{code} : mode {mode:.0f} UR, tirage {sampled:.0f} UR")
        path = TUNED_DIR / f"{code.lower()}_{args.scenario}.yaml"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"# Mode de jeu de {code} choisi sur les graines {args.seeds}.\n")
            fh.write(f"# Coût moyen : mode {mode:.1f} UR, tirage {sampled:.1f} UR\n")
            yaml.safe_dump({"deterministic": deterministic}, fh, sort_keys=False)
        print(f"enregistré dans {path}")


if __name__ == "__main__":
    main()
