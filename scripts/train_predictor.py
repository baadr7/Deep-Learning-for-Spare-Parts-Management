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
from spareparts.predict.training import (
    evaluate_network,
    split_by_group,
    train_lightgbm,
    train_network,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--models", nargs="+", default=["lightgbm", "gru", "tcn", "lstm"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--source",
        choices=["simulation", "azure"],
        default="simulation",
        help="azure : données réelles, découpage temporel (janvier-août, septembre-octobre, "
        "novembre-décembre)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="évalue les réseaux déjà entraînés sans les réentraîner (LightGBM est réentraîné)",
    )
    args = parser.parse_args()

    if args.source == "azure":
        args.scenario = "azure"
        cache = ROOT / "data" / "processed" / "predict_azure.npz"
    else:
        cache = ROOT / "data" / "processed" / f"predict_{args.scenario}_{args.days}j.npz"
    if cache.exists() and not args.rebuild:
        data = PredictionData.load(cache)
    else:
        start = time.perf_counter()
        if args.source == "azure":
            from spareparts.predict.azure_dataset import build_azure_dataset

            data = build_azure_dataset()
        else:
            data = simulate_dataset(args.scenario, seeds=(1, 2, 3, 4, 5), days=args.days)
        cache.parent.mkdir(parents=True, exist_ok=True)
        data.save(cache)
        print(f"{len(data)} exemples construits en {time.perf_counter() - start:.0f} s")

    if args.source == "azure":
        # Découpage temporel : la variable « group » contient le mois.
        train, val, test = split_by_group(data, val_groups=[9, 10], test_groups=[11, 12])
    else:
        train, val, test = split_by_group(data, val_groups=[4], test_groups=[5])
    print(f"apprentissage {len(train)}, validation {len(val)}, test {len(test)}")

    results = {}
    for name in args.models:
        start = time.perf_counter()
        out = ROOT / "models" / "predict"
        if name == "lightgbm":
            metrics = train_lightgbm(train, val, test)
        elif args.evaluate_only:
            import torch

            from spareparts.predict.dataset import HORIZON
            from spareparts.predict.models import ARCHITECTURES

            model = ARCHITECTURES[name](5, train.static.shape[1], train.event_bin.shape[1], HORIZON)
            model.load_state_dict(torch.load(out / f"{name}_{args.scenario}.pt"))
            metrics = evaluate_network(model, val, test)
        else:
            import torch

            model, metrics = train_network(name, train, val, test, epochs=args.epochs)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / f"{name}_{args.scenario}.pt")
        metrics["duree_s"] = round(time.perf_counter() - start, 1)
        results[name] = metrics
        week = metrics["a_7_jours"]
        print(
            f"{name:9s} 24 h : rappel {metrics['rappel']:.3f}  F1 {metrics['f1']:.3f}  "
            f"PR-AUC {metrics['pr_auc']:.3f} | 7 j : rappel {week['rappel']:.3f}  "
            f"F1 {week['f1']:.3f}  PR-AUC {week['pr_auc']:.3f}"
        )

    report = ROOT / "reports" / "metrics"
    report.mkdir(parents=True, exist_ok=True)
    path = report / f"prediction_{args.scenario}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"métriques enregistrées dans {path}")


if __name__ == "__main__":
    main()
