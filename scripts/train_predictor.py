"""Entraîne et compare les modèles de prédiction de pannes sur données simulées (§ 5.2).

Exemple : uv run python scripts/train_predictor.py --models lightgbm gru tcn lstm --epochs 15

Le jeu de données est simulé une fois (graines 1 à 5 : apprentissage 1 à 3, validation 4,
test 5) puis mis en cache dans data/processed. Les métriques vont dans reports/metrics.
"""

from __future__ import annotations

import argparse
import json
import time

from spareparts.config import ROOT
from spareparts.predict.dataset import PredictionData, simulate_dataset
from spareparts.predict.training import split_by_group, train_lightgbm, train_network


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--models", nargs="+", default=["lightgbm", "gru", "tcn", "lstm"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    cache = ROOT / "data" / "processed" / f"predict_{args.scenario}_{args.days}j.npz"
    if cache.exists() and not args.rebuild:
        data = PredictionData.load(cache)
    else:
        start = time.perf_counter()
        data = simulate_dataset(args.scenario, seeds=(1, 2, 3, 4, 5), days=args.days)
        cache.parent.mkdir(parents=True, exist_ok=True)
        data.save(cache)
        print(f"{len(data)} exemples simulés en {time.perf_counter() - start:.0f} s")

    train, val, test = split_by_group(data, val_groups=[4], test_groups=[5])
    print(f"apprentissage {len(train)}, validation {len(val)}, test {len(test)}")

    results = {}
    for name in args.models:
        start = time.perf_counter()
        if name == "lightgbm":
            metrics = train_lightgbm(train, val, test)
        else:
            import torch

            model, metrics = train_network(name, train, val, test, epochs=args.epochs)
            out = ROOT / "models" / "predict"
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / f"{name}_{args.scenario}.pt")
        metrics["duree_s"] = round(time.perf_counter() - start, 1)
        results[name] = metrics
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})

    report = ROOT / "reports" / "metrics"
    report.mkdir(parents=True, exist_ok=True)
    path = report / f"prediction_{args.scenario}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"métriques enregistrées dans {path}")


if __name__ == "__main__":
    main()
