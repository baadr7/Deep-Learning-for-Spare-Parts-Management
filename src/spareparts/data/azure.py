"""Chargement du jeu de données Microsoft Azure Predictive Maintenance (§ 4.1).

Les cinq fichiers CSV téléchargés depuis Kaggle doivent être placés dans ``data/raw``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from spareparts.config import ROOT

RAW_DIR = ROOT / "data" / "raw"
FILES = {
    "telemetry": "PdM_telemetry.csv",
    "errors": "PdM_errors.csv",
    "maint": "PdM_maint.csv",
    "failures": "PdM_failures.csv",
    "machines": "PdM_machines.csv",
}
SIGNAL_COLUMNS = ["volt", "rotate", "pressure", "vibration"]


def available() -> dict[str, bool]:
    return {name: (RAW_DIR / fname).exists() for name, fname in FILES.items()}


def is_complete() -> bool:
    return all(available().values())


def load(name: str) -> pd.DataFrame:
    path = RAW_DIR / FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : placer les CSV Azure dans {RAW_DIR}")
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
    return df


@lru_cache(maxsize=1)
def summary() -> dict:
    """Indicateurs descriptifs utilisés par la page « Données » de l'application."""
    if not is_complete():
        return {"available": available(), "complete": False}
    tel = load("telemetry")
    errors = load("errors")
    maint = load("maint")
    failures = load("failures")
    machines = load("machines")

    def span(df: pd.DataFrame) -> list[str]:
        return [df["datetime"].min().isoformat(), df["datetime"].max().isoformat()]

    monthly = failures.set_index("datetime").groupby("failure").resample("MS").size()
    months = sorted({d.strftime("%Y-%m") for d in monthly.index.get_level_values(1)})
    by_month = {
        comp: [int(monthly.get((comp, pd.Timestamp(m + "-01")), 0)) for m in months]
        for comp in sorted(failures["failure"].unique())
    }
    hist, edges = np.histogram(machines["age"], bins=range(0, 22, 2))
    merged = failures.merge(machines, on="machineID")
    return {
        "available": available(),
        "complete": True,
        "rows": {
            "telemetry": len(tel),
            "errors": len(errors),
            "maint": len(maint),
            "failures": len(failures),
            "machines": len(machines),
        },
        "period": {"telemetry": span(tel), "maint": span(maint), "failures": span(failures)},
        "signals": {
            col: {
                "mean": float(tel[col].mean()),
                "std": float(tel[col].std()),
                "p05": float(tel[col].quantile(0.05)),
                "p95": float(tel[col].quantile(0.95)),
            }
            for col in SIGNAL_COLUMNS
        },
        "failures_by_component": failures["failure"].value_counts().sort_index().to_dict(),
        "failures_by_model": merged["model"].value_counts().sort_index().to_dict(),
        "failures_by_month": {"months": months, "series": by_month},
        "machines_by_model": machines["model"].value_counts().sort_index().to_dict(),
        "age_histogram": {"edges": [int(e) for e in edges], "counts": hist.tolist()},
        "errors_by_type": errors["errorID"].value_counts().sort_index().to_dict(),
        "maint_by_component": maint["comp"].value_counts().sort_index().to_dict(),
    }
