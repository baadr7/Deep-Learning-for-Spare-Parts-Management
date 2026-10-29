"""Construit le plan d'expériences puis entraîne le modèle de substitution (§ 5.4).

Exemple : uv run python scripts/train_surrogate.py --scenarios 2000 --replications 2

Les simulations sont mises en cache dans data/processed ; 20 % des scénarios servent à
mesurer le R² et l'erreur relative du modèle, qui est ensuite réentraîné sur l'ensemble.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from spareparts.config import ROOT
from spareparts.surrogate import MODEL_PATH
from spareparts.surrogate.doe import OUTPUTS, PARAMS, build
from spareparts.surrogate.model import DeepEnsemble, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--scenarios", type=int, default=2000)
    parser.add_argument("--replications", type=int, default=2)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    cache = ROOT / "data" / "processed" / f"surrogate_{args.scenario}_{args.scenarios}.parquet"
    if cache.exists() and not args.rebuild:
        df = pd.read_parquet(cache)
    else:
        start = time.perf_counter()
        df = build(args.scenario, args.scenarios, args.replications, args.days)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        print(f"{len(df)} scénarios simulés en {time.perf_counter() - start:.0f} s")

    rng = np.random.default_rng(0)
    test = rng.random(len(df)) < 0.2
    ens = DeepEnsemble()
    ens.fit(df[~test][list(PARAMS)], df[~test][OUTPUTS], epochs=args.epochs)
    mean, _ = ens.predict(df[test][list(PARAMS)])
    metrics = score(df[test][OUTPUTS], mean)
    for k, v in metrics.items():
        print(f"{k:24s} R² {v['r2']:.3f}  MAPE {v['mape']:.3f}")

    final = DeepEnsemble()
    final.fit(df[list(PARAMS)], df[OUTPUTS], epochs=args.epochs)
    final.save(MODEL_PATH, metrics={"test": metrics, "scenarios": len(df), "days": args.days})
    report = ROOT / "reports" / "metrics"
    report.mkdir(parents=True, exist_ok=True)
    (report / f"surrogate_{args.scenario}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"modèle enregistré dans {MODEL_PATH}")


if __name__ == "__main__":
    main()
