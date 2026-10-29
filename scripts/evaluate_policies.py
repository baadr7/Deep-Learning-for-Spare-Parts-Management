"""Évaluation finale des politiques selon le protocole du cahier des charges (§ 7.3).

Chaque politique est simulée sur les mêmes graines (nombres aléatoires communs). Le script
calcule moyennes et intervalles de confiance à 95 %, teste les écarts à la référence par un
test de Wilcoxon apparié, puis mesure la robustesse dans trois situations dégradées.

Exemple : uv run python scripts/evaluate_policies.py --replications 30 --robustness
Ajout d'une politique entraînée après coup : ... --add P5
Résultats : reports/evaluation_<scénario>.json et reports/evaluation_<scénario>.md
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from spareparts.config import ROOT, load_scenario
from spareparts.evaluation import summarize
from spareparts.policies import POLICY_LABELS, available_policies, make_policy
from spareparts.sim import FleetSimulation

KEYS = [
    "cout_total",
    "trs",
    "disponibilite",
    "frequence_ruptures",
    "attente_piece_moyenne",
    "pannes",
]
NOMINAL = "nominal"
ROBUSTNESS = {
    "délai fournisseur doublé": {"lead_time_factor": 2.0},
    "taux de panne + 30 %": {"failure_rate_factor": 1.3},
    "un technicien de moins": {"technicians": 1},
}


def _run(args) -> dict:
    scenario, policy, seed, days, overrides = args
    cfg = load_scenario(scenario, **overrides)
    sim = FleetSimulation(cfg, seed=seed, log_events=False)
    return sim.run(make_policy(policy, scenario), days=days).kpis


def evaluate(pool, scenario, policies, seeds, days, overrides) -> dict:
    out = {}
    for code in policies:
        jobs = [(scenario, code, s, days, overrides) for s in seeds]
        runs = list(pool.map(_run, jobs))
        out[code] = {"runs": runs, "summary": summarize(runs)}
    return out


def wilcoxon(results: dict, reference: str, key: str) -> dict[str, float]:
    ref = np.array([r[key] for r in results[reference]["runs"]])
    pvalues = {}
    for code, res in results.items():
        if code == reference:
            continue
        other = np.array([r[key] for r in res["runs"]])
        diff = other - ref
        pvalues[code] = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    return pvalues


def markdown(results: dict, pvalues: dict, reference: str) -> str:
    lines = [
        "| Politique | Coût total (UR) | TRS | Disponibilité | Ruptures | p (coût, vs "
        + reference
        + ") |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for code, res in results.items():
        s = res["summary"]
        p = pvalues.get(code)
        lines.append(
            f"| {code} {POLICY_LABELS.get(code, '')} | {s['cout_total']['mean']:.0f} "
            f"[{s['cout_total']['low']:.0f} ; {s['cout_total']['high']:.0f}] "
            f"| {100 * s['trs']['mean']:.2f} % | {100 * s['disponibilite']['mean']:.2f} % "
            f"| {100 * s['frequence_ruptures']['mean']:.1f} % | {'' if p is None else f'{p:.3g}'} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--reference", default="P1")
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument(
        "--add",
        nargs="+",
        metavar="POLITIQUE",
        help="évalue ces politiques et les ajoute au rapport existant sans refaire les autres",
    )
    args = parser.parse_args()

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    json_path = out / f"evaluation_{args.scenario}.json"
    runs: dict[str, dict[str, list]] = {}
    if args.add:
        # Mêmes graines et mêmes situations que l'évaluation existante, dont on garde les années.
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        args.replications = previous["replications"]
        args.days = previous.get("days", args.days)
        args.robustness = "robustesse" in previous
        runs = previous["runs"]
        policies = args.add
    else:
        policies = args.policies or available_policies(args.scenario)
    seeds = list(range(9000, 9000 + args.replications))
    cases = {NOMINAL: {}} | (ROBUSTNESS if args.robustness else {})

    with ProcessPoolExecutor() as pool:
        for name, overrides in cases.items():
            res = evaluate(pool, args.scenario, policies, seeds, args.days, overrides)
            runs.setdefault(name, {}).update({c: r["runs"] for c, r in res.items()})
            print(f"{name} : {', '.join(policies)} évaluées")

    known = available_policies(args.scenario)
    present = list(runs[NOMINAL])
    order = [c for c in known if c in present] + [c for c in present if c not in known]
    report = {
        "scenario": args.scenario,
        "replications": args.replications,
        "days": args.days,
        "policies": order,
    }
    md = [f"# Évaluation des politiques, scénario {args.scenario}", ""]
    for name, by_policy in runs.items():
        results = {
            c: {"runs": by_policy[c], "summary": summarize(by_policy[c])}
            for c in order
            if c in by_policy
        }
        pvalues = wilcoxon(results, args.reference, "cout_total")
        summaries = {c: r["summary"] for c, r in results.items()}
        table = markdown(results, pvalues, args.reference)
        if name == NOMINAL:
            report["nominal"] = summaries
            report["wilcoxon_cout"] = pvalues
            md += ["## Scénario nominal", "", table, ""]
        else:
            report.setdefault("robustesse", {})[name] = summaries
            report.setdefault("robustesse_wilcoxon", {})[name] = pvalues
            md += [f"## Robustesse : {name}", "", table, ""]
        print(f"\n{name}\n{table}")
    report["runs"] = runs

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    (out / f"evaluation_{args.scenario}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nrésultats enregistrés dans {out}")


if __name__ == "__main__":
    main()
