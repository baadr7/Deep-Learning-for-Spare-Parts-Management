"""Tests du plancher de représentation des espaces d'action (décomposition de l'objectif O4).

Ces tests ne lancent pas la campagne — elle coûte plusieurs milliers de simulations annuelles —
mais vérifient ce qui la rendrait fausse : l'inventaire des actions, la reproductibilité des
départs, l'arithmétique de la décomposition et la cohérence du rapport publié.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from spareparts.config import ROOT
from spareparts.envs.codecs import PM_LEVELS, STOCK_LEVELS

_SPEC = importlib.util.spec_from_file_location(
    "action_space_floor", ROOT / "scripts" / "action_space_floor.py"
)
plancher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plancher)

RAPPORT = ROOT / "reports" / "metrics" / "plancher_representation.json"


def test_reduced_space_lists_the_thirty_six_actions():
    actions = plancher.reduced_actions()
    assert len(actions) == len(PM_LEVELS) * len(STOCK_LEVELS) == 36
    assert len(set(actions)) == 36
    assert actions == [(a,) for a in range(36)]


def test_n1_starts_begin_at_the_nominal_point_and_are_reproducible():
    starts = plancher.n1_starts(n_random=3, seed=0)
    assert len(starts) == 4
    assert all(start.shape == (plancher.N1_DIM,) for start in starts)
    # Le premier départ fixe les seuils et les stocks à leur valeur nominale (indice 2).
    assert list(starts[0]) == [2] * plancher.N1_DIM
    again = plancher.n1_starts(n_random=3, seed=0)
    assert all(np.array_equal(a, b) for a, b in zip(starts, again, strict=True))


def test_floor_reduced_returns_the_cheapest_of_the_thirty_six_actions():
    class _FakeEvaluator:
        def cost(self, pool, level, action):
            return {"cout_total": 100.0 - action[0]}

    space = plancher.floor_reduced(_FakeEvaluator(), pool=None)
    assert space["exhaustif"] is True
    assert space["n_actions"] == 36
    # Le coût décroît avec l'indice : le plancher est la dernière action, seuil et stock au
    # maximum de leurs grilles.
    assert space["plancher"]["action"] == 35
    assert space["plancher"]["pm_index"] == 5
    assert space["plancher"]["stock_index"] == 5
    assert space["plancher"]["pm_level"] == float(PM_LEVELS[5])
    assert space["plancher"]["stock_level"] == float(STOCK_LEVELS[5])
    assert [r["cout_total"] for r in space["classement"]] == sorted(
        r["cout_total"] for r in space["classement"]
    )


def test_decomposition_splits_the_gap_additively():
    spaces = {
        "reduced": {"plancher": {"cout_total": 54.0}},
        "n1": {"plancher": {"cout_total": 53.0}},
    }
    bounds = {"P1": 52.0, "DQN": 55.0, "P4": 55.5}
    dec = plancher.decomposition(spaces, bounds)
    dqn = dec["DQN"]
    assert dqn["espace"] == "reduced"
    assert dqn["ecart_observe"] == pytest.approx(3.0)
    assert dqn["ecart_apprentissage"] == pytest.approx(1.0)
    assert dqn["ecart_representation"] == pytest.approx(2.0)
    assert dqn["ecart_apprentissage"] + dqn["ecart_representation"] == pytest.approx(
        dqn["ecart_observe"]
    )
    assert dqn["part_apprentissage"] == pytest.approx(1 / 3)
    assert dec["P4"]["espace"] == "n1"


def test_floor_policy_runs_a_short_simulation():
    """La politique constante est utilisable telle quelle par le simulateur."""
    from spareparts.config import load_scenario
    from spareparts.sim import FleetSimulation

    sim = FleetSimulation(load_scenario("s1_nominal"), seed=0, log_events=False)
    result = sim.run(plancher.CodecFloorPolicy("reduced", (2,)), days=5)
    assert np.isfinite(result.kpis["cout_total"])


def test_published_floor_report_is_consistent():
    # Le rapport du plancher est un artefact publié : ``tableau_plancher.tex`` en est tiré et lu par
    # le rapport. Son absence est une régression, pas une raison de ne rien vérifier.
    assert RAPPORT.exists(), f"rapport du plancher absent : {RAPPORT}"
    report = json.loads(RAPPORT.read_text(encoding="utf-8"))
    assert report["reference"] == "P1"
    assert set(report["espaces"]) == {"reduced", "n1"}
    for space in report["espaces"].values():
        assert {"methode", "n_actions", "exhaustif", "plancher"} <= set(space)
        assert space["n_actions"] > 0
    # L'espace réduit est contenu dans N1 : un seuil commun et un stock commun sont un cas
    # particulier des seuils et stocks par composant. Le plancher de N1 ne peut donc pas être
    # plus cher que celui de l'espace réduit.
    reduced = report["espaces"]["reduced"]["plancher"]["cout_total"]
    n1 = report["espaces"]["n1"]["plancher"]["cout_total"]
    assert n1 <= reduced + 1e-6
    # Les deux agents à espace paramétrique sont décomposés, et l'espace réduit est bien celui du
    # DQN (le niveau N1 celui de P4).
    assert set(report["decomposition"]) == set(plancher.AGENT_SPACE)
    for code, space in plancher.AGENT_SPACE.items():
        assert report["decomposition"][code]["espace"] == space
    for dec in report["decomposition"].values():
        assert dec["ecart_observe"] == pytest.approx(
            dec["ecart_apprentissage"] + dec["ecart_representation"], abs=1e-6
        )
