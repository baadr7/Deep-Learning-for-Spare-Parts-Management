"""Valide le simulateur sur le jeu Azure : pannes par composant à 10 % près (objectif O2).

1. Rejeu de la maintenance planifiée d'Azure avec les lois ajustées directement.
2. Correction de l'échelle des lois sur le premier semestre 2015.
3. Validation de la correction sur le second semestre 2015, non utilisé pour la calibrer.

Prérequis : scripts/calibrate_azure.py exécuté (configs/calibration/azure_weibull.yaml).
Exemple : uv run python scripts/validate_azure.py --seeds 8
Résultats : configs/calibration/azure_weibull_corrige.yaml et
reports/metrics/validation_azure.json
"""

from __future__ import annotations

import argparse
import json

import yaml

from spareparts.config import ROOT
from spareparts.data import azure
from spareparts.validation import (
    azure_failures_by_period,
    calibrate_scale,
    compare,
    planned_schedule,
    simulated_failures,
)


def show(title: str, result: dict) -> None:
    print(title)
    for comp, r in result.items():
        print(
            f"  {comp}: Azure {r['azure']:4d}  simulé {r['simulees']:6.1f} ± {r['ecart_type']:4.1f}"
            f"  écart {100 * r['ecart_relatif']:+5.1f} %"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()

    calib_dir = ROOT / "configs" / "calibration"
    calib = calib_dir / "azure_weibull.yaml"
    if not calib.exists():
        raise SystemExit("lancer d'abord scripts/calibrate_azure.py")
    weibull = yaml.safe_load(calib.read_text(encoding="utf-8"))
    comps = sorted(weibull)
    seeds = range(args.seeds)

    failures, maint = azure.load("failures"), azure.load("maint")
    schedule = planned_schedule(maint, failures, comps)
    observed = azure_failures_by_period(failures)

    raw = simulated_failures(weibull, schedule, seeds)
    raw_s2 = compare(raw["S2"], observed["S2"], comps)
    show("Lois ajustées directement, second semestre 2015 :", raw_s2)

    adjusted = calibrate_scale(weibull, schedule, observed["S1"], seeds=range(4))
    corrected = simulated_failures(adjusted, schedule, seeds)
    fit_s1 = compare(corrected["S1"], observed["S1"], comps)
    test_s2 = compare(corrected["S2"], observed["S2"], comps)
    show("\nLois corrigées, premier semestre (calibration) :", fit_s1)
    show("\nLois corrigées, second semestre (validation) :", test_s2)

    year_obs = {c: observed["S1"].get(c, 0) + observed["S2"].get(c, 0) for c in comps}
    year = compare(corrected["S1"] + corrected["S2"], year_obs, comps)
    show("\nLois corrigées, année 2015 complète :", year)

    ok = all(abs(r["ecart_relatif"]) <= 0.10 for r in year.values())
    ok_s2 = all(abs(r["ecart_relatif"]) <= 0.10 for r in test_s2.values())
    print(f"\ncritère O2 sur l'année 2015 : {'atteint' if ok else 'non atteint'}")
    print(f"critère O2 sur le seul second semestre : {'atteint' if ok_s2 else 'non atteint'}")
    for c in comps:
        factor = adjusted[c]["eta"] / weibull[c]["eta"]
        before, after = weibull[c]["eta"], adjusted[c]["eta"]
        print(f"  {c}: eta {before:.0f} h corrigé en {after:.0f} h (× {factor:.2f})")

    with (calib_dir / "azure_weibull_corrige.yaml").open("w", encoding="utf-8") as fh:
        fh.write("# Lois corrigées par simulation sur le premier semestre 2015 (heures).\n")
        yaml.safe_dump(adjusted, fh, allow_unicode=True, sort_keys=True)
    out = ROOT / "reports" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "lois_brutes_s2": raw_s2,
        "calibration_s1": fit_s1,
        "validation_s2": test_s2,
        "annee_2015": year,
        "lois_corrigees": adjusted,
        "critere_atteint": ok,
        "critere_atteint_s2": ok_s2,
        "graines": args.seeds,
    }
    (out / "validation_azure.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
