"""Évaluation finale des politiques selon le protocole du cahier des charges (§ 7.3).

Chaque politique est simulée sur les mêmes graines (nombres aléatoires communs). Le script
calcule moyennes et intervalles de confiance à 95 %, teste les écarts à la référence par un
test de Wilcoxon apparié, vérifie l'équivalence à la référence par un test TOST à marge
paramétrable, puis mesure la robustesse dans quatre situations dégradées.

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
# Chaque situation dégradée décrit les surcharges appliquées au scénario nominal. La clé
# « scenario » bascule sur un autre fichier de scénario (flotte homogène S0) ; les agents
# restent ceux entraînés sur le scénario nominal.
ROBUSTNESS = {
    "délai fournisseur doublé": {"lead_time_factor": 2.0},
    "taux de panne + 30 %": {"failure_rate_factor": 1.3},
    "un technicien de moins": {"technicians": 1},
    "flotte homogène S0": {"scenario": "s0_homogene"},
}
# P3 s'appuie sur un réseau de survie dont l'entrée encode la famille de chaque machine. Un
# modèle entraîné pour le scénario nominal ne peut donc pas scorer une flotte qui n'a pas le
# même nombre de familles : P3 n'est pas applicable à la situation S0. Les agents de
# l'apprentissage par renforcement rejoignent cet ensemble dès qu'ils observent le risque prédit
# (``use_risk``), recalculé par ce même réseau de survie ; l'observation enrichie est commune aux
# quatre agents (PPO, DQN, BDQ), de sorte que leur comparaison porte sur l'algorithme et la
# représentation d'action, et non sur l'information disponible.
FAMILY_BOUND = ("P3",)
RL_AGENTS = ("P4", "P5", "RPPO", "DQN", "BDQ")


def _agent_uses_risk(code: str, scenario: str) -> bool:
    if code == "BDQ":
        meta = ROOT / "models" / "rl" / f"bdq_{scenario}.json"
    else:
        meta = ROOT / "models" / "rl" / f"{code.lower()}_{scenario}.json"
    if not meta.exists():
        return False
    return bool(json.loads(meta.read_text(encoding="utf-8")).get("use_risk"))


def family_bound_policies(scenario: str) -> set[str]:
    """Politiques dont le modèle encode les familles, donc inapplicables à une autre flotte.

    P3 en fait toujours partie. Chaque agent d'apprentissage par renforcement rejoint cet
    ensemble dès que ses métadonnées signalent ``use_risk`` : l'observation enrichie du risque
    passe par le réseau de survie du scénario nominal, dont l'entrée dépend du nombre de familles.
    """
    bound = set(FAMILY_BOUND)
    bound.update(code for code in RL_AGENTS if _agent_uses_risk(code, scenario))
    return bound


def applicable(policies: list[str], scenario: str, overrides: dict) -> list[str]:
    """Écarte les politiques liées à la flotte nominale si la situation change ses familles."""
    target = overrides.get("scenario", scenario)
    if target == scenario:
        return list(policies)
    same_fleets = len(load_scenario(scenario).families) == len(load_scenario(target).families)
    if same_fleets:
        return list(policies)
    bound = family_bound_policies(scenario)
    return [c for c in policies if c not in bound]


def _run(args) -> dict:
    scenario, policy, seed, days, overrides = args
    overrides = dict(overrides)
    target = overrides.pop("scenario", scenario)
    cfg = load_scenario(target, **overrides)
    sim = FleetSimulation(cfg, seed=seed, log_events=False)
    return sim.run(make_policy(policy, scenario), days=days).kpis


def evaluate(pool, scenario, policies, seeds, days, overrides) -> dict:
    out = {}
    for code in applicable(policies, scenario, overrides):
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


def tost(results: dict, reference: str, key: str, margin: float, level: float = 0.05) -> dict:
    """Test d'équivalence TOST (deux tests unilatéraux) contre la politique de référence.

    Le test de Wilcoxon dit si une politique *diffère* de P1 ; le TOST dit si elle lui est
    *équivalente* à une marge près, ou si elle est franchement meilleure. Les différences sont
    appariées graine par graine (nombres aléatoires communs). La marge est une fraction du coût
    moyen de la référence : à ``margin = 0,02``, une politique est équivalente si son écart moyen
    reste dans $\pm 2~\%$ du coût de P1.

    Renvoie, par politique, l'écart moyen signé, la marge en unités relatives, les deux valeurs
    unilatérales et le verdict (``équivalent``, ``meilleur``, ``moins bon`` ou ``non concluant``).
    """
    ref = np.array([r[key] for r in results[reference]["runs"]], dtype=float)
    delta = float(margin * ref.mean())
    out = {}
    for code, res in results.items():
        if code == reference:
            continue
        other = np.array([r[key] for r in res["runs"]], dtype=float)
        diff = other - ref
        n = diff.size
        mean = float(diff.mean())
        se = float(stats.sem(diff)) if n > 1 else 0.0
        if se == 0:
            p_pas_pire = 0.0 if mean < delta else 1.0
            p_pas_meilleur = 0.0 if mean > -delta else 1.0
        else:
            # H1 « pas pire » : écart < +marge ; H1 « pas meilleur » : écart > -marge.
            p_pas_pire = float(stats.t.cdf((mean - delta) / se, n - 1))
            p_pas_meilleur = float(stats.t.sf((mean + delta) / se, n - 1))
        if p_pas_pire < level and p_pas_meilleur < level:
            verdict = "équivalent"
        elif p_pas_pire < level:
            verdict = "meilleur"
        elif p_pas_meilleur < level:
            verdict = "moins bon"
        else:
            verdict = "non concluant"
        out[code] = {
            "ecart_moyen": mean,
            "marge": delta,
            "marge_relative": margin,
            "p_pas_pire": p_pas_pire,
            "p_pas_meilleur": p_pas_meilleur,
            "verdict": verdict,
        }
    return out


def markdown(results: dict, pvalues: dict, reference: str, tost_res: dict | None = None) -> str:
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
    if tost_res:
        first = tost_res[next(iter(tost_res))]
        parts = [f"{code} {v['verdict']}" for code, v in tost_res.items()]
        marge = f"±{100 * first['marge_relative']:.0f} % du coût de {reference}"
        lines += ["", f"Test d'équivalence TOST ({marge}) : " + " ; ".join(parts) + "."]
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
        "--workers",
        type=int,
        default=None,
        help="nombre de processus de simulation ; par défaut, tous les cœurs disponibles",
    )
    parser.add_argument(
        "--add",
        nargs="+",
        metavar="POLITIQUE",
        help="évalue ces politiques et les ajoute au rapport existant sans refaire les autres",
    )
    parser.add_argument(
        "--tost-margin",
        type=float,
        default=0.02,
        help="marge d'équivalence du test TOST, en fraction du coût moyen de la référence "
        "(0,02 = ±2 %%)",
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
        # Les politiques réévaluées remplacent leurs anciennes traces dans toutes les
        # situations, y compris celles où elles ne sont plus applicables (par exemple un agent
        # devenu lié aux familles par l'observation du risque prédit) : sans quoi des valeurs
        # d'un modèle précédent subsisteraient.
        for by_policy in runs.values():
            for code in args.add:
                by_policy.pop(code, None)
        policies = args.add
    else:
        policies = args.policies or available_policies(args.scenario)
    seeds = list(range(9000, 9000 + args.replications))
    cases = {NOMINAL: {}} | (ROBUSTNESS if args.robustness else {})

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, overrides in cases.items():
            res = evaluate(pool, args.scenario, policies, seeds, args.days, overrides)
            runs.setdefault(name, {}).update({c: r["runs"] for c, r in res.items()})
            skipped = [c for c in policies if c not in res]
            suffix = f" (hors {', '.join(skipped)} : modèle lié aux familles)" if skipped else ""
            print(f"{name} : {', '.join(res)} évaluées{suffix}")

    known = available_policies(args.scenario)
    present = list(runs[NOMINAL])
    order = [c for c in known if c in present] + [c for c in present if c not in known]
    report = {
        "scenario": args.scenario,
        "replications": args.replications,
        "days": args.days,
        "policies": order,
        "tost_margin": args.tost_margin,
    }
    md = [f"# Évaluation des politiques, scénario {args.scenario}", ""]
    for name, by_policy in runs.items():
        results = {
            c: {"runs": by_policy[c], "summary": summarize(by_policy[c])}
            for c in order
            if c in by_policy
        }
        pvalues = wilcoxon(results, args.reference, "cout_total")
        tost_res = tost(results, args.reference, "cout_total", args.tost_margin)
        summaries = {c: r["summary"] for c, r in results.items()}
        table = markdown(results, pvalues, args.reference, tost_res)
        if name == NOMINAL:
            report["nominal"] = summaries
            report["wilcoxon_cout"] = pvalues
            report["tost_cout"] = tost_res
            md += ["## Scénario nominal", "", table, ""]
        else:
            report.setdefault("robustesse", {})[name] = summaries
            report.setdefault("robustesse_wilcoxon", {})[name] = pvalues
            report.setdefault("robustesse_tost", {})[name] = tost_res
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
