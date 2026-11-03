"""Tests du plan d'expériences du modèle de substitution (§ 5.4).

Le plan couvre le budget d'évaluation du modèle de substitution : chaque point est un scénario
simulé sur plusieurs graines, résumé par la moyenne de ses indicateurs. On vérifie que les
tirages respectent les bornes déclarées et que l'évaluation d'un point produit bien tous les
indicateurs attendus.
"""

from __future__ import annotations

import numpy as np

from spareparts.surrogate.doe import OUTPUTS, PARAMS, _job, evaluate, sample


def test_sample_stays_within_the_declared_bounds():
    design = sample(12, seed=3)
    assert list(design.columns) == list(PARAMS)
    for name, (lo, hi, integer) in PARAMS.items():
        assert design[name].between(lo, hi).all()
        if integer:
            assert np.all(design[name].to_numpy() == design[name].round())
    # Un hypercube latin place un point par tranche : aucun doublon sur douze tirages.
    assert len(design.drop_duplicates()) == 12


def test_sample_is_reproducible_for_a_given_seed():
    assert sample(8, seed=1).equals(sample(8, seed=1))
    assert not sample(8, seed=1).equals(sample(8, seed=2))


def test_evaluate_averages_every_indicator_over_the_seeds():
    params = {
        "technicians": 2,
        "lead_time_factor": 1.0,
        "failure_rate_factor": 1.0,
        "pm_factor": 1.0,
        "stock_factor": 1.0,
    }
    kpis = evaluate("s1_nominal", params, seeds=[1, 2], days=30)
    assert set(kpis) == set(OUTPUTS)
    assert all(isinstance(value, float) for value in kpis.values())
    assert kpis["cout_total"] > 0.0
    # Le facteur de stock à l'unité redonne les seuils constructeur : cohérent avec P0.
    assert 0.0 <= kpis["disponibilite"] <= 1.0


def test_job_is_the_parallel_entry_point_of_evaluate():
    params = {
        "technicians": 2,
        "lead_time_factor": 1.0,
        "failure_rate_factor": 1.0,
        "pm_factor": 1.0,
        "stock_factor": 1.0,
    }
    args = ("s1_nominal", params, [1], 30)
    assert _job(args) == evaluate(*args)


def test_thresholds_from_factor_keeps_the_reorder_point_below_the_target():
    from spareparts.config import load_scenario
    from spareparts.surrogate.doe import thresholds_from_factor

    cfg = load_scenario("s1_nominal")
    for factor in (0.5, 1.0, 3.0):
        s, S = thresholds_from_factor(cfg, factor)
        assert all(0 <= reorder < target for reorder, target in zip(s, S, strict=True))
        assert all(target >= 1 for target in S)
