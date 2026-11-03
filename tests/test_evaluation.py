"""Évaluation statistique des politiques : écart apparié et taille d'effet (§ 7.3)."""

from __future__ import annotations

import numpy as np

from spareparts.evaluation import paired_effects


def _runs(base: list[float], degraded: list[float]) -> dict:
    """Bloc ``runs`` minimal au format du fichier d'évaluation, sur des graines communes."""
    return {
        "nominal": {"P1": [{"cout_total": value} for value in base]},
        "délais doublés": {"P1": [{"cout_total": value} for value in degraded]},
    }


def _d_z(differences: np.ndarray) -> float:
    """Taille d'effet appariée de Cohen reproduite indépendamment de l'implémentation."""
    return float(differences.mean() / differences.std(ddof=1))


def test_paired_effect_is_the_per_seed_mean_difference():
    # Situations appariées : l'écart se calcule graine par graine, pas entre les moyennes.
    runs = _runs([100.0, 200.0, 300.0], [110.0, 190.0, 330.0])
    effect = paired_effects(runs)["délais doublés"]["P1"]
    assert effect["moyenne"] == np.mean([10.0, -10.0, 30.0])
    # La variation rapporte l'écart au coût moyen de la référence (200 UR ici).
    assert effect["variation"] == effect["moyenne"] / 200.0


def test_paired_effect_brackets_the_mean_and_reports_the_effect_size():
    runs = _runs([100.0, 200.0, 300.0], [110.0, 190.0, 330.0])
    effect = paired_effects(runs)["délais doublés"]["P1"]
    assert effect["low"] < effect["moyenne"] < effect["high"]
    assert effect["variation_low"] < effect["variation"] < effect["variation_high"]
    differences = np.array([10.0, -10.0, 30.0])
    # d_z rapporte l'écart moyen à l'écart-type des écarts, et garde son signe.
    assert effect["d_z"] == _d_z(differences)
    assert effect["d_z"] > 0


def test_paired_effect_ignores_a_policy_absent_from_the_reference():
    # Une politique présente seulement dans une situation dégradée n'a pas de contrepartie.
    runs = _runs([100.0, 200.0, 300.0], [110.0, 190.0, 330.0])
    runs["délais doublés"]["P9"] = [{"cout_total": 1.0}]
    effects = paired_effects(runs)
    assert "P9" not in effects.get("délais doublés", {})


def test_reference_is_not_compared_to_itself():
    runs = _runs([100.0, 200.0, 300.0], [110.0, 190.0, 330.0])
    effects = paired_effects(runs)
    assert "nominal" not in effects


def test_single_seed_yields_a_zero_interval():
    # Avec une seule graine, l'écart ponctuel reste exact mais sans dispersion mesurable.
    runs = _runs([100.0], [120.0])
    effect = paired_effects(runs)["délais doublés"]["P1"]
    assert effect["moyenne"] == 20.0
    assert effect["low"] == effect["high"] == 20.0
    assert effect["d_z"] == 0.0


def difference_ratio(differences: np.ndarray) -> float:
    """Taille d'effet appariée de Cohen reproduite indépendamment de l'implémentation."""
    return float(differences.mean() / differences.std(ddof=1))


def test_summarize_brackets_the_mean_and_leaves_undefined_indicators_unset():
    from spareparts.evaluation import summarize

    runs = [
        {"cout_total": 10.0, "ratio": float("nan")},
        {"cout_total": 20.0, "ratio": float("nan")},
    ]
    out = summarize(runs)
    assert out["cout_total"]["mean"] == 15.0
    assert out["cout_total"]["low"] < 15.0 < out["cout_total"]["high"]
    # Un indicateur sans aucune valeur finie reste indéfini au lieu de faire échouer le résumé.
    assert all(np.isnan(out["ratio"][bound]) for bound in ("mean", "low", "high"))


def test_summarize_of_a_single_run_has_no_dispersion():
    from spareparts.evaluation import summarize

    assert summarize([{"cout_total": 42.0}])["cout_total"] == {
        "mean": 42.0,
        "low": 42.0,
        "high": 42.0,
    }


def test_replicate_uses_common_random_numbers_across_policies():
    from spareparts.config import load_scenario
    from spareparts.evaluation import replicate
    from spareparts.policies import make_policy

    cfg = load_scenario("s1_nominal")
    runs = replicate(cfg, lambda: make_policy("P0"), seeds=[1, 1], days=30)
    assert len(runs) == 2
    # Deux tirages sur la même graine donnent exactement le même coût : le bruit est commun.
    assert runs[0]["cout_total"] == runs[1]["cout_total"]


def test_optimize_fixed_threshold_evaluates_the_baseline_first():
    from spareparts.config import load_scenario
    from spareparts.evaluation import optimize_fixed_threshold

    cfg = load_scenario("s1_nominal")
    seen: list[int] = []
    _, best_cost, history = optimize_fixed_threshold(
        cfg, n_trials=3, seeds=[101], progress=lambda i, cost, best: seen.append(i)
    )
    assert len(history) == 3
    assert seen == [0, 1, 2]
    # Le premier candidat est la politique constructeur : facteurs d'intervalle à l'unité.
    assert history[0]["pm_factor"] == [1.0] * cfg.n_components
    assert best_cost == min(trial["cout_total"] for trial in history)


def test_stock_factor_scales_the_reorder_thresholds():
    # Paramétrisation de politique du plan d'expériences, réutilisée par l'optimisation
    # bayésienne de P1 : un facteur unité redonne les seuils constructeur, un facteur double
    # multiplie les niveaux cibles en gardant un point de commande strictement inférieur.
    from spareparts.config import load_scenario
    from spareparts.surrogate.doe import thresholds_from_factor

    cfg = load_scenario("s1_nominal")
    s, S = thresholds_from_factor(cfg, 1.0)
    assert s == [p.s for p in cfg.parts]
    assert S == [p.S for p in cfg.parts]

    s2, S2 = thresholds_from_factor(cfg, 2.0)
    assert S2 == [max(1, round(2 * p.S)) for p in cfg.parts]
    assert all(0 <= s_i < S_i for s_i, S_i in zip(s2, S2, strict=True))
