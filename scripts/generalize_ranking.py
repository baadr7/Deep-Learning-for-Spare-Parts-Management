"""Généralisation du classement à plusieurs compositions de flotte (note § 6, item 12).

Le classement des politiques est établi sur une seule composition nominale (100 machines,
deux familles, deux techniciens). Ce script rejoue le protocole sur les compositions tirées
par ``scripts/generate_fleets.py`` et mesure la **stabilité du classement** :

- **P1 et P3 sont réglés composition par composition** (graines de réglage 101-103) : leurs
  seuils dépendent des lois de panne, donc de la part de chaque famille dans la flotte ;
- les autres politiques — P0, P2, Oracle et les quatre agents — sont **transférées** telles
  quelles. Réentraîner les agents par composition reste hors budget : au niveau N1, 300 000
  pas coûtent ≈ 53 min par graine (mesuré), soit plusieurs heures par composition et par
  agent. Le transfert est signalé dans la sortie et dans la synthèse ;
- chaque composition est évaluée sur les **mêmes graines** (9000-9029) que l'évaluation
  publiée : l'écart de classement tient donc à la composition, non au tirage.

La stabilité est mesurée par le tau de Kendall du classement de chaque composition contre le
classement nominal, et par la conservation de l'ordre attendu par le cahier des charges
(``P2 < P3 < P1 < agents``, au sens du coût).

Exemple :
  uv run python scripts/generate_fleets.py --n 5 --seed 0
  uv run python scripts/generalize_ranking.py --replications 30
Résultats : reports/metrics/generalisation_flotte.json et reports/generalisation_flotte.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from spareparts.config import ROOT, load_scenario
from spareparts.evaluation import optimize_fixed_threshold
from spareparts.policies import (
    POLICY_LABELS,
    _predictor_path,
    available_policies,
    make_policy,
)
from spareparts.policies.classic import FixedThresholdPolicy
from spareparts.policies.predictive import PredictivePolicy
from spareparts.sim import FleetSimulation
from spareparts.sim.fleets import (
    FleetComposition,
    composition_overrides,
    sample_compositions,
    write_compositions,
)

TUNE_SEEDS = [101, 102, 103]
# Les agents retenus dans le classement publié ; l'ordre attendu les compare à P1.
AGENT_CODES = ("P4", "P5", "RPPO", "DQN", "BDQ")
# Ordre attendu par le cahier des charges (§ 7.3), au sens du coût : la maintenance
# conditionnelle bat la prédictive, qui bat les seuils optimisés, eux-mêmes battus par les
# agents. C'est une hypothèse à éprouver, non un résultat acquis.
CDC_CHAIN = ("P2", "P3", "P1")


def read_compositions(directory: Path, base: str) -> list[FleetComposition]:
    """Compositions écrites par ``scripts/generate_fleets.py``, dans l'ordre des indices."""
    out = []
    for path in sorted(Path(directory).glob(f"{base}_composition_*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        out.append(
            FleetComposition(
                index=int(path.stem.rsplit("_", 1)[1]),
                seed=int(raw["seed"]),
                technicians=int(raw["technicians"]),
                family_share={k: float(v) for k, v in raw["family_share"].items()},
            )
        )
    return out


def mean_costs(by_policy: dict[str, list[dict]]) -> dict[str, float]:
    """Coût total annuel moyen de chaque politique sur les graines d'évaluation."""
    return {
        code: float(np.mean([r["cout_total"] for r in runs])) for code, runs in by_policy.items()
    }


def ranking(means: dict[str, float]) -> dict[str, float]:
    """Rang de chaque politique (1 = coût le plus faible) ; ex æquo = rang moyen."""
    order = sorted(means, key=lambda code: means[code])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and means[order[j + 1]] == means[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def kendall_vs(reference: dict[str, float], other: dict[str, float]) -> float:
    """Tau de Kendall entre deux classements, calculé sur les politiques communes."""
    common = sorted(set(reference) & set(other))
    if len(common) < 2:
        return float("nan")
    return float(
        stats.kendalltau([reference[c] for c in common], [other[c] for c in common]).statistic
    )


def cdc_order(means: dict[str, float]) -> dict:
    """L'ordre attendu ``P2 < P3 < P1 < agents`` est-il conservé, au sens du coût ?"""
    checks: dict[str, bool] = {}
    for left, right in itertools.pairwise(CDC_CHAIN):
        if left in means and right in means:
            checks[f"{left}<{right}"] = bool(means[left] < means[right])
    agents = [c for c in AGENT_CODES if c in means]
    if agents and "P1" in means:
        checks["P1<agents"] = bool(means["P1"] < min(means[c] for c in agents))
    return {"conserve": bool(checks) and all(checks.values()), "controles": checks}


def chunks(items: list, count: int) -> list[list]:
    """Tranches contiguës de taille comparable, une par processus de travail.

    Les politiques n'ont pas le même coût (P3 et le DQN durent une vingtaine de fois plus
    longtemps que P0) : découper les graines évite qu'un seul processus porte à lui seul
    l'essentiel de la charge.
    """
    if count <= 1:
        return [list(items)]
    size = -(-len(items) // count)
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _build(code: str, base: str, kind: str, params: dict | None):
    """Politique du cas : P1 ou P3 réglés sur place, les autres lues dans le registre."""
    if kind == "p1":
        return FixedThresholdPolicy(name="P1", **params)
    if kind == "p3":
        path, arch = _predictor_path(base)
        return PredictivePolicy(path, arch, **params)
    return make_policy(code, base)


def _run_case(job: tuple) -> tuple[str, list[dict]]:
    """Rejoue une politique sur un lot de graines, dans un processus de travail.

    La politique est construite une fois par lot afin que les modèles appris ne soient
    chargés qu'une fois ; ``FleetSimulation.run`` la réinitialise à chaque graine.
    """
    key, base, overrides, code, kind, params, seeds, days = job
    cfg = load_scenario(base, **overrides)
    policy = _build(code, base, kind, params)
    runs = [
        FleetSimulation(cfg, seed=seed, log_events=False).run(policy, days=days).kpis
        for seed in seeds
    ]
    return key, runs


def tune_p1(base: str, overrides: dict, trials: int, rng_seed: int) -> tuple[dict, float]:
    """Recherche aléatoire des seuils de P1 sur la composition, graines 101-103."""
    cfg = load_scenario(base, **overrides)
    best, cost, _ = optimize_fixed_threshold(
        cfg, n_trials=trials, seeds=TUNE_SEEDS, rng_seed=rng_seed
    )
    return best, cost


def tune_p3(pool, base: str, overrides: dict, grid: dict, days: int) -> tuple[dict, float]:
    """Grille des seuils de P3 sur la composition, graines 101-103.

    Un travail par couple (jeu de seuils, graine) : la grille se répartit ainsi sur tous les
    processus, au lieu de laisser un seul processus porter les dix-huit jeux.
    """
    combos = [dict(zip(grid, values, strict=True)) for values in itertools.product(*grid.values())]
    jobs = [
        (str(i), base, overrides, "P3", "p3", combo, [seed], days)
        for i, combo in enumerate(combos)
        for seed in TUNE_SEEDS
    ]
    costs: dict[str, list[float]] = {}
    for key, runs in pool.map(_run_case, jobs):
        costs.setdefault(key, []).extend(r["cout_total"] for r in runs)
    means = [float(np.mean(costs[str(i)])) for i in range(len(combos))]
    return combos[int(np.argmin(means))], min(means)


def markdown(report: dict) -> str:
    """Synthèse lisible : une ligne par composition, puis le détail des classements."""
    lines = [
        "# Généralisation du classement — compositions de flotte",
        "",
        f"Référence : {report['base']} · {report['replications']} réplications "
        f"(graines {report['seeds'][0]}-{report['seeds'][-1]}) · {report['days']} jours.",
        "",
        "P1 et P3 sont réglés composition par composition (graines 101-103) ; P0, P2, Oracle et "
        "les agents sont transférés tels quels, faute de réentraînement par composition.",
        "",
        "## Synthèse",
        "",
        "| Composition | Techniciens | Part U | τ Kendall vs nominal | Ordre CDC | Coût P1 "
        "| Coût P2 | Meilleur agent |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        means = case["cout_moyen"]
        agents = {c: means[c] for c in AGENT_CODES if c in means}
        best = min(agents, key=agents.get) if agents else None
        best_agent = f"{best} ({agents[best]:.0f})" if best else "—"
        part_u = case["part_U"]
        part_u_txt = "—" if part_u != part_u else f"{part_u:.2f}"
        lines.append(
            f"| {case['nom']} | {case['techniciens']} | {part_u_txt} "
            f"| {case['kendall_vs_nominal']:.3f} | "
            f"{'oui' if case['ordre_cdc']['conserve'] else 'non'} "
            f"| {means.get('P1', float('nan')):.0f} | {means.get('P2', float('nan')):.0f} "
            f"| {best_agent} |"
        )
    lines += ["", "## Détail des classements", ""]
    for case in report["cases"]:
        means = case["cout_moyen"]
        ranks = case["rangs"]
        lines += [
            f"### {case['nom']}",
            "",
            "| Rang | Politique | Coût total moyen (UR) |",
            "| --- | --- | --- |",
        ]
        for code in sorted(means, key=lambda c: means[c]):
            lines.append(
                f"| {ranks[code]:.0f} | {code} {POLICY_LABELS.get(code, '')} | {means[code]:.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="s1_nominal")
    parser.add_argument("--compositions", type=Path, default=ROOT / "configs" / "fleets")
    parser.add_argument(
        "--draw", type=int, default=0, help="tire N compositions si aucune n'existe"
    )
    parser.add_argument("--seed", type=int, default=0, help="graine du tirage des compositions")
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--p1-trials", type=int, default=40)
    # Chaque processus charge NumPy, SciPy et PyTorch : en lancer autant que de cœurs épuise
    # le fichier d'échange de la machine (erreur « insufficient system resources »). Huit
    # processus suffisent à saturer un poste courant sans dépasser la mémoire disponible.
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tune_p3 import GRID as P3_GRID

    comps = read_compositions(args.compositions, args.base)
    if not comps:
        comps = sample_compositions(args.base, args.draw or 5, args.seed)
        write_compositions(comps, args.base, args.compositions)
        print(f"{len(comps)} compositions tirées dans {args.compositions}")

    seeds = list(range(9000, 9000 + args.replications))
    batches = chunks(seeds, args.workers)
    policies = available_policies(args.base)
    cases = [("nominal", {}, None)] + [
        (comp.name, composition_overrides(comp, args.base), comp) for comp in comps
    ]

    cases_out = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, overrides, comp in cases:
            tuned: dict[str, dict] = {}
            if comp is not None:
                p1, cost_p1 = tune_p1(args.base, overrides, args.p1_trials, comp.seed)
                p3, cost_p3 = tune_p3(pool, args.base, overrides, P3_GRID, args.days)
                tuned = {"P1": p1, "P3": p3}
                print(
                    f"{name} : P1 réglé ({cost_p1:.0f} UR), P3 réglé ({cost_p3:.0f} UR)",
                    flush=True,
                )
            jobs = [
                (
                    code,
                    args.base,
                    overrides,
                    code,
                    "p1" if code in tuned and code == "P1" else "p3" if code in tuned else "file",
                    tuned.get(code),
                    batch,
                    args.days,
                )
                for code in policies
                for batch in batches
            ]
            by_policy: dict[str, list[dict]] = {}
            for code, runs in pool.map(_run_case, jobs):
                by_policy.setdefault(code, []).extend(runs)
            means = mean_costs(by_policy)
            cases_out.append(
                {
                    "nom": name,
                    "techniciens": (
                        comp.technicians if comp else load_scenario(args.base).technicians
                    ),
                    "part_U": comp.family_share.get("U", float("nan")) if comp else float("nan"),
                    "family_share": comp.family_share if comp else {},
                    "reglages": tuned,
                    "cout_moyen": means,
                    "rangs": ranking(means),
                    "runs": by_policy,
                }
            )
            print(
                f"{name} : "
                + ", ".join(f"{c} {means[c]:.0f}" for c in sorted(means, key=means.get)),
                flush=True,
            )

    reference = cases_out[0]["rangs"]
    for case in cases_out:
        case["kendall_vs_nominal"] = (
            1.0 if case["nom"] == "nominal" else kendall_vs(reference, case["rangs"])
        )
        case["ordre_cdc"] = cdc_order(case["cout_moyen"])

    report = {
        "base": args.base,
        "replications": args.replications,
        "seeds": seeds,
        "days": args.days,
        "policies": policies,
        "transfert_agents": True,
        "cases": cases_out,
    }
    (args.out / "metrics").mkdir(parents=True, exist_ok=True)
    (args.out / "metrics" / "generalisation_flotte.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    (args.out / "generalisation_flotte.md").write_text(markdown(report), encoding="utf-8")
    taus = [c["kendall_vs_nominal"] for c in cases_out[1:]]
    print(f"\ntau de Kendall vs nominal : {np.round(taus, 3).tolist()}")
    print(f"ordre CDC conservé : {[c['ordre_cdc']['conserve'] for c in cases_out]}")
    print(f"résultats enregistrés dans {args.out}")


if __name__ == "__main__":
    main()
