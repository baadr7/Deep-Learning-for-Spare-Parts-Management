"""Règle la politique P1 par recherche aléatoire et enregistre ses paramètres.

Exemple : uv run python scripts/optimize_p1.py --scenario s1_nominal --trials 40
"""

from __future__ import annotations

import argparse

import yaml

from spareparts.config import load_scenario
from spareparts.evaluation import optimize_fixed_threshold
from spareparts.policies import TUNED_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    args = parser.parse_args()

    cfg = load_scenario(args.scenario)

    def progress(i, cost, best):
        line = f"essai {i + 1:3d}/{args.trials}  coût {cost:10.1f}  meilleur {best:10.1f}"
        print(line, flush=True)

    best, best_cost, _ = optimize_fixed_threshold(
        cfg, n_trials=args.trials, seeds=args.seeds, progress=progress
    )
    TUNED_DIR.mkdir(parents=True, exist_ok=True)
    path = TUNED_DIR / f"p1_{args.scenario}.yaml"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Paramètres de P1 obtenus par recherche aléatoire ({args.trials} essais).\n")
        fh.write(f"# Coût moyen sur les graines {args.seeds} : {best_cost:.1f} UR\n")
        yaml.safe_dump(best, fh, allow_unicode=True, sort_keys=False)
    print(f"enregistré dans {path}")


if __name__ == "__main__":
    main()
