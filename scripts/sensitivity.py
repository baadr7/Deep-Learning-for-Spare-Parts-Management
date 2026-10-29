"""Analyse de sensibilité globale par indices de Sobol, sur le modèle de substitution (§ 5.4).

Les indices sont estimés par l'échantillonnage de Saltelli : plusieurs dizaines de milliers
d'évaluations, impossibles avec le simulateur, instantanées avec le modèle de substitution.
L'indice de premier ordre S1 mesure la part de variance expliquée par un paramètre seul, l'indice
total ST y ajoute ses interactions avec les autres.

Exemple : uv run python scripts/sensitivity.py
Résultat : reports/metrics/sobol.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample

from spareparts.config import ROOT
from spareparts.surrogate import MODEL_PATH
from spareparts.surrogate.doe import PARAMS
from spareparts.surrogate.model import DeepEnsemble

OUTPUTS = ["cout_total", "trs", "frequence_ruptures"]


def main() -> None:
    model, _ = DeepEnsemble.load(MODEL_PATH)
    names = list(PARAMS)
    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": [[PARAMS[n][0], PARAMS[n][1]] for n in names],
    }
    X = sobol_sample.sample(problem, 2048, calc_second_order=False, seed=0)
    mean, _ = model.predict(pd.DataFrame(X, columns=names))
    result = {}
    for out in OUTPUTS:
        si = sobol.analyze(problem, mean[out].to_numpy(), calc_second_order=False, seed=0)
        result[out] = {
            n: {"S1": float(si["S1"][i]), "ST": float(si["ST"][i])} for i, n in enumerate(names)
        }
        ranked = sorted(names, key=lambda n: -result[out][n]["ST"])
        print(out, "  ".join(f"{n} {result[out][n]['ST']:.2f}" for n in ranked))
    path = ROOT / "reports" / "metrics" / "sobol.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{len(X)} évaluations ; indices enregistrés dans {path}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
