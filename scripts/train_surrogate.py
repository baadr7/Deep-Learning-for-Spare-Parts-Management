"""Construit le plan d'expériences puis entraîne le modèle de substitution (§ 5.4).

Exemple : uv run python scripts/train_surrogate.py --scenarios 2000 --replications 5

Les simulations sont mises en cache dans data/processed ; 20 % des scénarios servent à
mesurer le R² et l'erreur relative du modèle, qui est ensuite réentraîné sur l'ensemble.
Un processus gaussien et LightGBM fournissent les deux points de comparaison du § 5.4.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from spareparts import tracking
from spareparts.config import ROOT
from spareparts.surrogate import MODEL_PATH
from spareparts.surrogate.doe import OUTPUTS, PARAMS, build

# Le processus gaussien coûte O(n³) en la taille de l'échantillon : on l'entraîne, comme le veut
# l'usage, sur un sous-échantillon ; LightGBM passe sur l'ensemble.
GP_MAX_SAMPLES = 600


def _lightgbm(X, Y, X_test, Y_test) -> dict:
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    from spareparts.surrogate.model import score

    model = MultiOutputRegressor(
        LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)
    )
    model.fit(X, Y)
    pred = pd.DataFrame(model.predict(X_test), columns=OUTPUTS, index=X_test.index)
    return score(Y_test, pred)


def _gaussian_process(X, Y, X_test, Y_test, seed: int = 0) -> dict:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    from spareparts.surrogate.model import score

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(GP_MAX_SAMPLES, len(X)), replace=False)
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(1e-3)
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6)
    model.fit(X.iloc[idx], Y.iloc[idx])
    pred = pd.DataFrame(model.predict(X_test), columns=OUTPUTS, index=X_test.index)
    return score(Y_test, pred)


def compare_baselines(X, Y, X_test, Y_test) -> dict:
    """R² et erreur relative des deux modèles de comparaison du § 5.4."""
    out = {}
    for name, fit in (("lightgbm", _lightgbm), ("processus_gaussien", _gaussian_process)):
        out[name] = fit(X, Y, X_test, Y_test)
        r2 = out[name]["cout_total"]["r2"]
        print(f"  {name:20s} coût : R² {r2:.3f}")
    return out


def main() -> None:
    from spareparts.surrogate.model import DeepEnsemble, score

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--scenarios", type=int, default=2000)
    parser.add_argument("--replications", type=int, default=5)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="nombre de processus de simulation ; par défaut, tous les cœurs disponibles",
    )
    args = parser.parse_args()

    cache = (
        ROOT
        / "data"
        / "processed"
        / f"surrogate_{args.scenario}_{args.scenarios}_{args.replications}.parquet"
    )
    if cache.exists() and not args.rebuild:
        df = pd.read_parquet(cache)
    else:
        start = time.perf_counter()
        df = build(
            args.scenario, args.scenarios, args.replications, args.days, workers=args.workers
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        print(f"{len(df)} scénarios simulés en {time.perf_counter() - start:.0f} s")

    rng = np.random.default_rng(0)
    test = rng.random(len(df)) < 0.2
    x_train, y_train = df[~test][list(PARAMS)], df[~test][OUTPUTS]
    x_test, y_test = df[test][list(PARAMS)], df[test][OUTPUTS]

    params = {
        "scenario": args.scenario,
        "scenarios": args.scenarios,
        "replications": args.replications,
        "days": args.days,
        "epochs": args.epochs,
    }
    report = ROOT / "reports" / "metrics"
    report.mkdir(parents=True, exist_ok=True)

    with tracking.tracked("surrogate", params):
        ens = DeepEnsemble()
        ens.fit(x_train, y_train, epochs=args.epochs)
        mean, _ = ens.predict(x_test)
        metrics = score(y_test, mean)
        for k, v in metrics.items():
            print(f"{k:24s} R² {v['r2']:.3f}  MAPE {v['mape']:.3f}")

        print("modèles de comparaison")
        comparison = compare_baselines(x_train, y_train, x_test, y_test)

        final = DeepEnsemble()
        final.fit(df[list(PARAMS)], df[OUTPUTS], epochs=args.epochs)
        final.save(
            MODEL_PATH,
            metrics={
                "test": metrics,
                "comparaison": comparison,
                "scenarios": len(df),
                "days": args.days,
                "replications": args.replications,
            },
        )
        payload = {
            "ensemble": metrics,
            "comparaison": comparison,
            "scenarios": len(df),
            "days": args.days,
            "replications": args.replications,
        }
        path = report / f"surrogate_{args.scenario}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tracking.log(
            {f"ensemble_{k}_r2": v["r2"] for k, v in metrics.items()}
            | {f"{name}_{k}_r2": v["r2"] for name, m in comparison.items() for k, v in m.items()}
        )
        tracking.artifact(path)
    print(f"modèle enregistré dans {MODEL_PATH}")


if __name__ == "__main__":
    main()
