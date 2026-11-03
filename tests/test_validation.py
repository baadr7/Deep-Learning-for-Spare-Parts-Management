"""Tests du rejeu Azure et de la calibration des lois de Weibull (objectif O2, § 5.1).

Le jeu Azure complet — plusieurs gigaoctets — est absent du dépôt : on reconstitue donc de
petits journaux de maintenance et de pannes conformes au schéma des CSV officiels, et l'on
vérifie la mécanique du rejeu (filtrage des remplacements consécutifs à une panne, découpage en
semestres, décomposition de l'écart, correction d'échelle). La simulation complète n'est lancée
qu'une fois, sur une unique graine, pour borner le coût du test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spareparts import validation
from spareparts.validation import (
    END,
    MID,
    START,
    YEAR_START,
    azure_failures_by_period,
    compare,
    planned_schedule,
)


def test_planned_schedule_skips_failure_repairs_and_rows_before_the_horizon():
    # Trois motifs d'écart : une ligne planifiée, une ligne qui n'est que la conséquence d'une
    # panne (même machine, même instant, même composant) et une ligne antérieure au 1er juin 2014.
    maint = pd.DataFrame(
        {
            "datetime": [
                START + pd.Timedelta(days=2),
                START + pd.Timedelta(days=5),
                START - pd.Timedelta(days=1),
                START + pd.Timedelta(days=10),
            ],
            "machineID": [1, 1, 2, 2],
            "comp": ["comp1", "comp2", "comp1", "comp2"],
        }
    )
    failures = pd.DataFrame(
        {
            "datetime": [START + pd.Timedelta(days=5)],
            "machineID": [1],
            "failure": ["comp2"],
        }
    )
    schedule = planned_schedule(maint, failures, ["comp1", "comp2"])
    # Les machines sont renumérotées à partir de 0 et les composants indexés par leur rang.
    assert schedule == {2: [(0, 0)], 10: [(1, 1)]}


def test_azure_failures_are_counted_per_component_and_period():
    failures = pd.DataFrame(
        {
            "datetime": [
                YEAR_START,
                YEAR_START + pd.Timedelta(days=1),
                MID,
                END,  # borne haute exclue : janvier 2016 n'appartient à aucun des deux semestres
            ],
            "failure": ["comp1", "comp1", "comp1", "comp2"],
        }
    )
    out = azure_failures_by_period(failures)
    assert out["S1"] == {"comp1": 2}
    assert out["S2"] == {"comp1": 1}


def test_compare_reports_the_mean_gap_and_its_spread():
    sims = np.array([[10.0, 20.0], [12.0, 18.0], [14.0, 22.0]])
    out = compare(sims, {"comp1": 10, "comp2": 20}, ["comp1", "comp2"])
    assert out["comp1"]["azure"] == 10
    assert out["comp1"]["simulees"] == pytest.approx(12.0)
    assert out["comp1"]["ecart_relatif"] == pytest.approx(0.2)
    assert out["comp1"]["ecart_type"] > 0.0
    # Avec une seule graine, l'écart-type d'échantillon n'est pas défini : il est ramené à zéro.
    single = compare(np.array([[10.0, 20.0]]), {"comp1": 10, "comp2": 20}, ["comp1", "comp2"])
    assert single["comp1"]["ecart_type"] == 0.0


def test_calibrate_scale_moves_eta_towards_the_observed_failure_count(monkeypatch):
    # La simulation est remplacée : deux fois trop de pannes au premier semestre doivent doubler
    # l'échelle en deux itérations, puisque eta croît comme (simulé / observé)^(1/beta).
    weibull = {"comp1": {"beta": 2.0, "eta": 1000.0}}
    calls = {"n": 0}

    def fake_simulated(adjusted, schedule, seeds):
        calls["n"] += 1
        return {"S1": np.array([[2.0 * 50.0]])}

    monkeypatch.setattr(validation, "simulated_failures", fake_simulated)
    adjusted = validation.calibrate_scale(weibull, {}, {"comp1": 50}, seeds=[0], iterations=2)
    assert calls["n"] == 2
    assert adjusted["comp1"]["eta"] == pytest.approx(2000.0, rel=1e-9)
    # L'entrée d'origine reste intacte : l'ajustement se fait sur une copie.
    assert weibull["comp1"]["eta"] == 1000.0


def test_simulated_failures_returns_one_row_per_seed_and_one_column_per_component():
    weibull = {f"comp{i}": {"beta": 1.8, "eta": 5000.0} for i in range(1, 3)}
    schedule = {0: [(0, 0)]}
    sims = validation.simulated_failures(weibull, schedule, seeds=[0])
    assert set(sims) == {"S1", "S2"}
    assert sims["S1"].shape == (1, 2)
    assert sims["S2"].shape == (1, 2)
    assert (sims["S1"] >= 0).all() and (sims["S2"] >= 0).all()
