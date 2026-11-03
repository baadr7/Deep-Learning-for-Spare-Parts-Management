"""Tests du chargement du jeu Azure (§ 4.1).

Le jeu complet — plusieurs gigaoctets — est absent du dépôt. Les tests couvrent donc les deux
régimes : le message d'erreur quand un CSV manque, et le calcul des indicateurs descriptifs
sur un jeu minimal mais conforme au schéma des cinq fichiers officiels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from spareparts.data import azure


@pytest.fixture(autouse=True)
def _clear_summary_cache():
    """``summary`` est mémorisé : on repart d'un cache vide avant et après chaque test."""
    azure.summary.cache_clear()
    yield
    azure.summary.cache_clear()


def _write_azure_csvs(raw: Path) -> None:
    """Écrit un jeu Azure minimal : deux machines, un mois de télémétrie, deux pannes."""
    t = pd.date_range("2015-01-01", periods=40, freq="D")
    pd.DataFrame(
        {
            "datetime": t,
            "machineID": 1,
            "volt": 100.0,
            "rotate": 400.0,
            "pressure": 10.0,
            "vibration": 40.0,
        }
    ).to_csv(raw / "PdM_telemetry.csv", index=False)
    pd.DataFrame(
        {"machineID": [1, 2], "model": ["model1", "model2"], "age": [5, 10]}
    ).to_csv(raw / "PdM_machines.csv", index=False)
    pd.DataFrame(
        {"datetime": [t[3]], "machineID": [1], "errorID": ["error1"]}
    ).to_csv(raw / "PdM_errors.csv", index=False)
    pd.DataFrame(
        {"datetime": [t[10], t[20]], "machineID": [1, 2], "failure": ["comp1", "comp2"]}
    ).to_csv(raw / "PdM_failures.csv", index=False)
    pd.DataFrame(
        {"datetime": [t[5], t[6]], "machineID": [1, 1], "comp": ["comp1", "comp2"]}
    ).to_csv(raw / "PdM_maint.csv", index=False)


def test_load_reports_the_missing_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(azure, "RAW_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="introuvable"):
        azure.load("telemetry")


def test_calibration_exposes_the_committed_weibull_laws():
    calib = azure.calibration()
    assert calib is not None
    assert calib["lois"]
    # Le rapport de validation est enregistré à côté des lois ajustées.
    assert "validation" in calib


def test_summary_signals_the_missing_raw_files():
    body = azure.summary()
    assert body["complete"] is False
    assert set(body["available"]) == set(azure.FILES)


def test_summary_computes_the_descriptive_indicators(tmp_path, monkeypatch):
    _write_azure_csvs(tmp_path)
    monkeypatch.setattr(azure, "RAW_DIR", tmp_path)
    body = azure.summary()
    assert body["complete"] is True
    assert body["rows"] == {
        "telemetry": 40,
        "errors": 1,
        "maint": 2,
        "failures": 2,
        "machines": 2,
    }
    assert set(body["signals"]) == set(azure.SIGNAL_COLUMNS)
    assert set(body["failures_by_component"]) == {"comp1", "comp2"}
    assert body["failures_by_model"] == {"model1": 1, "model2": 1}
    assert body["machines_by_model"] == {"model1": 1, "model2": 1}
    assert body["errors_by_type"] == {"error1": 1}
    assert sum(body["age_histogram"]["counts"]) == 2
    assert body["failures_by_month"]["months"] == ["2015-01"]
