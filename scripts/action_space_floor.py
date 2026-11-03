"""Plancher de représentation des espaces d'action — décomposition de l'objectif O4.

Le critère O4 demande qu'un agent fasse **mieux** que la meilleure politique à seuils fixes
(P1). Un tel verdict n'a de sens que si l'agent *peut* battre P1, c'est-à-dire si son espace
d'action contient au moins une politique aussi bonne. On appelle **plancher de représentation**
d'un espace ``A`` le meilleur coût atteignable par une politique **constante** de cet espace,
évaluée sur les mêmes graines que l'évaluation publiée :

    c*(A) = min_{a ∈ A} c(a)

Le plancher sépare alors l'écart observé d'un agent à P1 en deux parts additives :

    c(agent) − c(P1)  =  [ c(agent) − c*(A) ]  +  [ c*(A) − c(P1) ]
                         \\_ écart d'apprentissage _/     \\_ écart de représentation _/

- **écart d'apprentissage** : ce que l'agent n'a pas su tirer d'un répertoire qui le lui
  permettait ; il se réduit par plus d'entraînement ou un meilleur réglage. Les politiques
  constantes formant un sous-ensemble de celles de l'espace, c'est un **minorant** du déficit
  réel de l'agent ;
- **écart de représentation** : distance du meilleur réglage **fixe** de l'espace à P1 ; c'est un
  **majorant** de ce que l'espace peut imputer. Un signe **négatif** est décisif : le meilleur coût
  de l'espace passe alors sous P1, donc l'espace **contient** une politique qui bat P1 et la
  représentation ne peut plus être invoquée pour expliquer l'écart de l'agent.

Deux espaces sont mesurés ici :

- ``reduced`` (36 actions, celui du DQN) est **énuméré exhaustivement** : le plancher y est exact ;
- ``n1`` (6⁸ = 1 679 616 actions, celui de P4) est exploré par **montée de coordonnée à départs
  multiples** ; le meilleur point trouvé **majore** le plancher exact.

Le script lit les coûts des agents dans ``reports/evaluation_s1_nominal.json`` afin que la
décomposition porte sur les valeurs publiées, non sur un nouveau tirage.

Exemple : uv run python scripts/action_space_floor.py
Sortie : reports/metrics/plancher_representation.json
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from spareparts.config import ROOT, load_scenario
from spareparts.envs.codecs import PM_LEVELS, STOCK_LEVELS, Context, make_codec
from spareparts.evaluation import summarize
from spareparts.policies.base import Policy
from spareparts.sim import FleetSimulation

# Espace d'action de chaque agent publié à espace paramétrique. P5 et le BDQ décident machine
# par machine : leur espace n'est pas une famille de seuils, et un plancher « constant » n'y a
# pas le même sens — ils sont traités par les bornes, voir ``structural_bounds``.
AGENT_SPACE = {"DQN": "reduced", "P4": "n1"}
N1_DIM = 8  # 4 seuils de maintenance (par composant) + 4 niveaux de stock (par référence)


class CodecFloorPolicy(Policy):
    """Membre constant d'un espace d'action : rejoue chaque jour la même action du codec.

    C'est l'objet dont le script cherche le meilleur représentant. À la différence d'un agent
    entraîné, elle n'observe rien : elle fixe une fois pour toutes les seuils et les niveaux de
    stock que l'espace autorise.
    """

    name = "plancher"
    label = "Plancher de représentation"

    def __init__(self, level: str, action):
        self.level = level
        self.action = np.asarray(action, dtype=int)

    def reset(self, sim: FleetSimulation) -> None:
        super().reset(sim)
        horizon = getattr(sim.cfg, "horizon_days", 365)
        self.codec = make_codec(self.level, Context.from_sim(sim, int(horizon)))

    def decide(self, obs):
        return self.codec.decide(self.action, obs)


def _run(job) -> dict[str, float]:
    """Simule une action constante sur une graine et renvoie ses indicateurs."""
    scenario, level, action, seed, days = job
    cfg = load_scenario(scenario)
    sim = FleetSimulation(cfg, seed=seed, log_events=False)
    return sim.run(CodecFloorPolicy(level, action), days=days).kpis


def reduced_actions() -> list[tuple[int, ...]]:
    """Les 36 actions de l'espace réduit, dans l'ordre du codec (seuil × multiplicateur)."""
    return [(a,) for a in range(len(PM_LEVELS) * len(STOCK_LEVELS))]


def n1_starts(n_random: int = 3, seed: int = 0) -> list[np.ndarray]:
    """Points de départ de la montée de coordonnée : le point constructeur et des tirages.

    Le premier départ fixe les seuils de maintenance et les niveaux de stock aux valeurs
    nominales (indice 2, soit 1,0 × l'intervalle et 1,0 × S_base) ; les suivants sont tirés au
    hasard avec une graine fixée, pour que la campagne soit reproductible.
    """
    starts = [np.full(N1_DIM, 2, dtype=int)]
    rng = np.random.default_rng(seed)
    starts += [rng.integers(0, 6, N1_DIM) for _ in range(n_random)]
    return starts


class Evaluator:
    """Évalue des actions constantes sur un jeu de graines, en cache et en parallèle."""

    def __init__(self, scenario: str, seeds: list[int], days: int, workers: int | None):
        self.scenario = scenario
        self.seeds = seeds
        self.days = days
        self.workers = workers
        self._cache: dict[tuple, dict] = {}

    def cost(self, pool, level: str, action) -> dict:
        """Moyenne, IC et traces d'une action ; mémoïse les actions déjà évaluées."""
        key = (level, tuple(int(a) for a in np.atleast_1d(action)))
        if key in self._cache:
            return self._cache[key]
        jobs = [(self.scenario, level, key[1], s, self.days) for s in self.seeds]
        runs = list(pool.map(_run, jobs))
        result = {"runs": runs, "summary": summarize(runs), "cout_total": float(
            np.mean([r["cout_total"] for r in runs])
        )}
        self._cache[key] = result
        return result


def floor_reduced(evaluator: Evaluator, pool) -> dict:
    """Énumère les 36 actions de l'espace réduit : le plancher obtenu est exact."""
    ranked = []
    for action in reduced_actions():
        res = evaluator.cost(pool, "reduced", action)
        ranked.append({"action": action[0], "cout_total": res["cout_total"]})
    ranked.sort(key=lambda r: r["cout_total"])
    best = ranked[0]
    pm, stock = divmod(best["action"], len(STOCK_LEVELS))
    return {
        "methode": "énumération exhaustive des 36 actions",
        "n_actions": len(reduced_actions()),
        "exhaustif": True,
        "plancher": {
            "action": best["action"],
            "pm_index": pm,
            "stock_index": stock,
            "pm_level": float(PM_LEVELS[pm]),
            "stock_level": float(STOCK_LEVELS[stock]),
            "cout_total": best["cout_total"],
        },
        "classement": ranked,
    }


def floor_n1(evaluator: Evaluator, pool, starts: int, sweeps: int) -> dict:
    """Monte de coordonnée sur les 6⁸ actions du niveau N1, depuis plusieurs départs.

    À chaque balayage, on essaie les six valeurs de chaque dimension en gardant le meilleur
    voisin. L'optimum trouvé n'est pas certifié — l'espace est trop vaste pour être énuméré —
    mais tout point atteint est une politique réalisable : le coût retenu **majore** le
    plancher exact.
    """
    best, best_cost = None, float("inf")
    evaluations = 0
    for start in n1_starts(starts):
        current = start.copy()
        current_cost = evaluator.cost(pool, "n1", current)["cout_total"]
        evaluations += 1
        for _ in range(sweeps):
            improved = False
            for dim in range(N1_DIM):
                for value in range(len(PM_LEVELS)):
                    if value == current[dim]:
                        continue
                    candidate = current.copy()
                    candidate[dim] = value
                    cost = evaluator.cost(pool, "n1", candidate)["cout_total"]
                    evaluations += 1
                    if cost < current_cost:
                        current, current_cost = candidate, cost
                        improved = True
            if not improved:
                break
        if current_cost < best_cost:
            best, best_cost = current, current_cost
    return {
        "methode": f"montée de coordonnée, {starts} départs × {sweeps} balayages",
        "n_actions": len(PM_LEVELS) ** N1_DIM,
        "exhaustif": False,
        "evaluations": evaluations,
        "plancher": {
            "action": [int(a) for a in best],
            "pm_index": [int(a) for a in best[: N1_DIM // 2]],
            "stock_index": [int(a) for a in best[N1_DIM // 2 :]],
            "pm_levels": [float(PM_LEVELS[a]) for a in best[: N1_DIM // 2]],
            "stock_levels": [float(STOCK_LEVELS[a]) for a in best[N1_DIM // 2 :]],
            "cout_total": best_cost,
        },
    }


def published_costs(scenario: str) -> dict[str, float]:
    """Coûts publiés par politique, lus dans l'évaluation de référence (graines 9000-9029).

    La décomposition porte ainsi sur les valeurs **publiées**, non sur un nouveau tirage. ``P1``
    est la cible du critère O4, ``P2`` la meilleure politique à décision individuelle (donc
    exprimable au niveau N2, qui décide machine par machine) et ``Oracle`` une borne supérieure
    absolue ; ``DQN`` et ``P4`` sont les agents dont l'espace d'action est décomposé.
    """
    path = ROOT / "reports" / f"evaluation_{scenario}.json"
    if not path.exists():
        return {}
    nominal = json.loads(path.read_text(encoding="utf-8"))["nominal"]
    return {code: float(s["cout_total"]["mean"]) for code, s in nominal.items()}


def structural_bounds(costs: dict[str, float]) -> dict[str, float]:
    """Restriction des coûts publiés aux bornes structurelles lues au rapport."""
    return {code: costs[code] for code in ("P1", "P2", "P3", "Oracle") if code in costs}


def decomposition(spaces: dict, costs: dict[str, float]) -> dict:
    """Scinde l'écart de chaque agent à P1 en écart d'apprentissage et écart de représentation."""
    reference = costs.get("P1")
    out = {}
    for code, space in AGENT_SPACE.items():
        floor = spaces[space]["plancher"]["cout_total"]
        agent = costs.get(code)
        if agent is None or reference is None:
            continue
        total = agent - reference
        out[code] = {
            "espace": space,
            "cout_agent": agent,
            "cout_plancher": floor,
            "cout_p1": reference,
            "ecart_observe": total,
            "ecart_apprentissage": agent - floor,
            "ecart_representation": floor - reference,
            "part_apprentissage": (agent - floor) / total if total else float("nan"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument(
        "--starts", type=int, default=3, help="départs aléatoires de la montée de coordonnée (N1)"
    )
    parser.add_argument(
        "--sweeps", type=int, default=2, help="balayages au plus par départ (N1)"
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    workers = args.workers or os.cpu_count()
    seeds = list(range(9000, 9000 + args.replications))
    evaluator = Evaluator(args.scenario, seeds, args.days, workers)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(f"plancher de l'espace réduit ({len(reduced_actions())} actions)…")
        reduced = floor_reduced(evaluator, pool)
        print(f"  plancher : {reduced['plancher']['cout_total']:.0f} UR")
        print(f"plancher de l'espace N1 (montée de coordonnée, {args.starts} départs)…")
        n1 = floor_n1(evaluator, pool, args.starts, args.sweeps)
        cout = n1["plancher"]["cout_total"]
        print(f"  plancher : {cout:.0f} UR ({n1['evaluations']} évaluations)")

    spaces = {"reduced": reduced, "n1": n1}
    costs = published_costs(args.scenario)
    report = {
        "scenario": args.scenario,
        "replications": args.replications,
        "days": args.days,
        "seeds": seeds,
        "reference": "P1",
        "bornes": structural_bounds(costs),
        "couts_publies": costs,
        "espaces": spaces,
        "decomposition": decomposition(spaces, costs),
    }

    out = ROOT / "reports" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "plancher_representation.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float),
                    encoding="utf-8")
    print(f"\nrésultats enregistrés dans {path}")
    for code, dec in report["decomposition"].items():
        print(
            f"{code} ({dec['espace']}) : écart à P1 {dec['ecart_observe']:+.0f} UR = "
            f"apprentissage {dec['ecart_apprentissage']:+.0f} + "
            f"représentation {dec['ecart_representation']:+.0f}"
        )


if __name__ == "__main__":
    main()
