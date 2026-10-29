"""Évaluation finale des politiques selon le protocole du cahier des charges (§ 7.3).

Chaque politique est simulée sur les mêmes graines (nombres aléatoires communs). Le script
calcule moyennes et intervalles de confiance à 95 %, teste les écarts à la référence par un
test de Wilcoxon apparié, puis mesure la robustesse dans quatre situations dégradées.

Exemple : uv run python scripts/evaluate_policies.py --replications 30 --robustness
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
    args = parser.parse_args()

    policies = args.policies or available_policies(args.scenario)
    seeds = list(range(9000, 9000 + args.replications))
    report = {"scenario": args.scenario, "replications": args.replications, "policies": policies}
    md = [f"# Évaluation des politiques, scénario {args.scenario}", ""]

    with ProcessPoolExecutor() as pool:
        nominal = evaluate(pool, args.scenario, policies, seeds, args.days, {})
        pvalues = wilcoxon(nominal, args.reference, "cout_total")
        report["nominal"] = {c: r["summary"] for c, r in nominal.items()}
        report["wilcoxon_cout"] = pvalues
        md += ["## Scénario nominal", "", markdown(nominal, pvalues, args.reference), ""]
        print(markdown(nominal, pvalues, args.reference))

        if args.robustness:
            report["robustesse"] = {}
            cases = dict(ROBUSTNESS)
            for name, overrides in cases.items():
                res = evaluate(pool, args.scenario, policies, seeds, args.days, overrides)
                p = wilcoxon(res, args.reference, "cout_total")
                report["robustesse"][name] = {c: r["summary"] for c, r in res.items()}
                md += [f"## Robustesse : {name}", "", markdown(res, p, args.reference), ""]
                print(f"\n{name}\n" + markdown(res, p, args.reference))

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / f"evaluation_{args.scenario}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    (out / f"evaluation_{args.scenario}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nrésultats enregistrés dans {out}")


if __name__ == "__main__":
    main()
