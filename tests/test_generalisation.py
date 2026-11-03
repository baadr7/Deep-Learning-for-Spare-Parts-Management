"""Tests de la campagne de généralisation du classement (note de synthèse, § 6, item 12)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from spareparts.sim.fleets import sample_compositions, write_compositions

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "generalize_ranking", ROOT / "scripts" / "generalize_ranking.py"
)
assert _SPEC is not None and _SPEC.loader is not None
generalize = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(generalize)


def test_ranking_orders_by_cost_and_averages_ties():
    ranks = generalize.ranking({"a": 30.0, "b": 10.0, "c": 20.0, "d": 20.0})
    assert ranks["b"] == 1
    assert ranks["c"] == ranks["d"] == 2.5
    assert ranks["a"] == 4


def test_kendall_is_one_for_identical_and_minus_one_for_reversed():
    reference = {"P0": 3.0, "P1": 2.0, "P2": 1.0}
    assert generalize.kendall_vs(reference, reference) == pytest.approx(1.0)
    reversed_ = {"P0": 1.0, "P1": 2.0, "P2": 3.0}
    assert generalize.kendall_vs(reference, reversed_) == pytest.approx(-1.0)


def test_kendall_ignores_policies_absent_from_one_ranking():
    reference = {"P0": 2.0, "P1": 1.0}
    assert np.isnan(generalize.kendall_vs(reference, {"P0": 1.0}))


def test_cdc_order_holds_when_chain_and_agents_are_beaten():
    means = {"P2": 10.0, "P3": 20.0, "P1": 30.0, "P4": 40.0, "DQN": 50.0}
    assert generalize.cdc_order(means)["conserve"] is True


def test_cdc_order_fails_when_an_agent_beats_p1():
    means = {"P2": 10.0, "P3": 20.0, "P1": 30.0, "DQN": 25.0}
    report = generalize.cdc_order(means)
    assert report["conserve"] is False
    assert report["controles"]["P1<agents"] is False


def test_chunks_cover_every_seed_exactly_once():
    seeds = list(range(30))
    batches = generalize.chunks(seeds, 8)
    assert sorted(seed for batch in batches for seed in batch) == seeds
    assert len(batches) == 8  # 30 graines, huit tranches de quatre au plus
    assert all(len(batch) <= 4 for batch in batches)


def test_mean_costs_average_over_replications():
    by_policy = {"P1": [{"cout_total": 10.0}, {"cout_total": 20.0}]}
    assert generalize.mean_costs(by_policy) == {"P1": pytest.approx(15.0)}


def test_read_compositions_roundtrip(tmp_path):
    comps = sample_compositions("s1_nominal", 3, seed=4)
    write_compositions(comps, "s1_nominal", tmp_path)
    back = generalize.read_compositions(tmp_path, "s1_nominal")
    assert [c.index for c in back] == [c.index for c in comps]
    assert [c.technicians for c in back] == [c.technicians for c in comps]
    for restored, original in zip(back, comps, strict=True):
        assert restored.family_share == pytest.approx(original.family_share)
