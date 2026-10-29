"""Ajuste les lois de Weibull des composants sur le jeu Azure (lot L1, § 5.1).

Prérequis : les cinq CSV du jeu Azure dans data/raw.
Exemple : uv run python scripts/calibrate_azure.py

Les paramètres sont écrits dans configs/calibration/azure_weibull.yaml. Un paramètre de forme
supérieur à 1 indique une usure, et donc l'intérêt d'une maintenance préventive (§ 3.2).
"""

from __future__ import annotations

import yaml

from spareparts.config import ROOT
from spareparts.data import azure
from spareparts.reliability.weibull import fit_aft, fit_by_component, lifetimes_from_azure


def main() -> None:
    if not azure.is_complete():
        missing = [f for f, ok in azure.available().items() if not ok]
        raise SystemExit(f"fichiers manquants dans {azure.RAW_DIR} : {', '.join(missing)}")
    maint, failures, machines = azure.load("maint"), azure.load("failures"), azure.load("machines")
    lifetimes = lifetimes_from_azure(maint, failures)
    fits = fit_by_component(lifetimes)
    for comp, f in fits.items():
        verdict = "usure, maintenance préventive utile" if f["beta"] > 1 else "pannes aléatoires"
        print(
            f"{comp}: beta {f['beta']:.2f}  eta {f['eta']:.0f} h  "
            f"({f['pannes']} pannes / {f['n']} durées)  {verdict}"
        )
    print()
    print(fit_aft(lifetimes, machines)[["coef", "exp(coef)", "p"]].round(3))

    out = ROOT / "configs" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "azure_weibull.yaml"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            "# Lois de Weibull ajustées sur le jeu Azure (heures calendaires, machines 24 h/24).\n"
        )
        yaml.safe_dump(fits, fh, allow_unicode=True, sort_keys=True)
    print(f"\nparamètres enregistrés dans {path}")


if __name__ == "__main__":
    main()
