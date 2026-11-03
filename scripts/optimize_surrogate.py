"""Règle P1 par optimisation bayésienne sur le modèle de substitution (§ 6, amélioration 8).

Exemple : uv run python scripts/optimize_surrogate.py --trials 300

Le modèle de substitution (§ 5.4) prédit le coût d'un scénario en quelques millisecondes, là où
le simulateur demande une année entière de calcul. On remplace donc la recherche aléatoire de P1
par une optimisation bayésienne (échantillonneur TPE d'Optuna) qui n'interroge que le modèle de
substitution, puis on vérifie le candidat retenu sur le simulateur, aux mêmes graines que la
recherche aléatoire, et on ne le conserve que s'il fait mieux.

Le modèle de substitution ne représente que la politique à seuils fixes, paramétrée par deux
facteurs globaux — intervalle de maintenance et niveau de stock —, à l'image de
``surrogate.doe.evaluate`` : la politique prédictive P3 en est absente, et sa grille de 18
combinaisons reste explorée exhaustivement.

Résultat : reports/metrics/optimisation_p1.json ; avec ``--write``, les paramètres sont en outre
écrits dans configs/policies/p1_<scénario>.yaml lorsqu'ils font mieux que la recherche aléatoire.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml

from spareparts.config import ROOT, load_scenario
from spareparts.evaluation import replicate
from spareparts.policies import TUNED_DIR, FixedThresholdPolicy
from spareparts.surrogate import MODEL_PATH
from spareparts.surrogate.doe import PARAMS, thresholds_from_factor
from spareparts.surrogate.model import DeepEnsemble

# Système nominal : la recherche bayésienne ne porte que sur les deux facteurs de politique de P1,
# les paramètres de flotte étant fixés à leur valeur du scénario pour isoler l'effet de la
# politique. Le facteur de délai fournisseur et celui de taux de panne valent 1 par construction.
NOMINAL = {"technicians": 2, "lead_time_factor": 1.0, "failure_rate_factor": 1.0}
PM_FACTOR = PARAMS["pm_factor"][:2]
STOCK_FACTOR = PARAMS["stock_factor"][:2]


def p1_from_factors(cfg, pm_factor: float, stock_factor: float) -> dict:
    """Paramètres P1 correspondant aux deux facteurs globaux du modèle de substitution.

    La conversion est celle de ``surrogate.doe`` : le facteur de stock multiplie les seuils
    constructeur, le facteur de maintenance leur intervalle. Le modèle de substitution prédit
    donc exactement le coût de la politique ainsi reconstruite, ce qui rend la vérification sur le
    simulateur directement comparable.
    """
    s, S = thresholds_from_factor(cfg, stock_factor)
    return {"s": s, "S": S, "pm_factor": round(float(pm_factor), 3)}


def optimize(model: DeepEnsemble, n_trials: int, seed: int = 0) -> tuple[dict, float]:
    """Optimisation bayésienne du coût prédit sur les deux facteurs de politique.

    L'objectif est déterministe et presque lisse : le TPE d'Optuna y converge en quelques
    centaines d'évaluations, toutes instantanées puisqu'elles n'appellent que le réseau.
    """
    import optuna

    def objective(trial) -> float:
        row = pd.DataFrame(
            [
                {
                    **NOMINAL,
                    "pm_factor": trial.suggest_float("pm_factor", *PM_FACTOR),
                    "stock_factor": trial.suggest_float("stock_factor", *STOCK_FACTOR),
                }
            ]
        )
        mean, _ = model.predict(row)
        return float(mean["cout_total"].iloc[0])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


def verified_cost(cfg, params: dict, seeds) -> float:
    """Coût annuel moyen d'une politique P1 sur les graines de réglage."""
    runs = replicate(
        cfg, lambda p=params: FixedThresholdPolicy(name="P1", **p), seeds
    )
    return float(np.mean([r["cout_total"] for r in runs]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument(
        "--write",
        action="store_true",
        help="enregistre les paramètres dans configs/policies/p1_<scénario>.yaml s'ils font mieux",
    )
    args = parser.parse_args()

    cfg = load_scenario(args.scenario)
    model, _ = DeepEnsemble.load(MODEL_PATH)

    best, predicted = optimize(model, args.trials)
    params = p1_from_factors(cfg, best["pm_factor"], best["stock_factor"])
    print(f"optimum du modèle de substitution : coût prédit {predicted:.1f} UR  {params}")

    cost_surrogate = verified_cost(cfg, params, args.seeds)
    cost_p0 = verified_cost(cfg, {}, args.seeds)
    current = TUNED_DIR / f"p1_{args.scenario}.yaml"
    cost_random = None
    if current.exists():
        random_params = yaml.safe_load(current.read_text(encoding="utf-8"))
        cost_random = verified_cost(cfg, random_params, args.seeds)

    print(f"P0                      : {cost_p0:10.1f} UR")
    if cost_random is not None:
        print(f"P1, recherche aléatoire : {cost_random:10.1f} UR")
    print(f"P1, bayésien (subrogé)  : {cost_surrogate:10.1f} UR")

    payload = {
        "scenario": args.scenario,
        "trials": args.trials,
        "seeds": args.seeds,
        "facteurs": {"pm_factor": best["pm_factor"], "stock_factor": best["stock_factor"]},
        "cout_predit": predicted,
        "cout_p0": cost_p0,
        "cout_recherche_aleatoire": cost_random,
        "cout_surrogate": cost_surrogate,
        "parametres": params,
    }
    out = ROOT / "reports" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"optimisation_p1_{args.scenario}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"résultats enregistrés dans {path}")

    if args.write and (cost_random is None or cost_surrogate < cost_random):
        TUNED_DIR.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Paramètres de P1 obtenus par optimisation bayésienne sur le modèle de substitution "
            f"({args.trials} évaluations).",
            f"# Coût moyen sur les graines {args.seeds} : {cost_surrogate:.1f} UR",
        ]
        body = yaml.safe_dump(params, allow_unicode=True, sort_keys=False)
        current.write_text("\n".join(lines) + "\n" + body, encoding="utf-8")
        print(f"paramètres enregistrés dans {current}")
    elif args.write:
        print("la recherche aléatoire reste meilleure : paramètres conservés")


if __name__ == "__main__":
    main()
