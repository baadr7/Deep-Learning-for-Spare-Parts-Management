"""Tests du générateur de compositions de flotte (note de synthèse, § 6, amélioration 12)."""

import numpy as np
import pytest
import yaml

from spareparts.config import load_scenario
from spareparts.sim import FleetSimulation
from spareparts.sim.fleets import (
    apply_composition,
    family_counts,
    sample_compositions,
    write_compositions,
)

BASE = "s1_nominal"


def test_sampling_is_deterministic():
    a = sample_compositions(BASE, 4, seed=7)
    b = sample_compositions(BASE, 4, seed=7)
    assert [c.family_share for c in a] == [c.family_share for c in b]
    assert [c.technicians for c in a] == [c.technicians for c in b]
    assert [c.seed for c in a] == [c.seed for c in b]


def test_different_seeds_differ():
    a = sample_compositions(BASE, 6, seed=0)
    b = sample_compositions(BASE, 6, seed=1)
    assert [c.family_share for c in a] != [c.family_share for c in b]


def test_shares_sum_to_one_and_respect_floor():
    comps = sample_compositions(BASE, 8, seed=3, min_share=0.15)
    for comp in comps:
        values = np.array(list(comp.family_share.values()))
        assert values.sum() == pytest.approx(1.0)
        assert values.min() >= 0.15 - 1e-12


def test_machine_count_is_constant():
    n = load_scenario(BASE).fleet.n_machines
    for comp in sample_compositions(BASE, 5, seed=2, min_share=0.2):
        counts = family_counts(comp.family_share, n)
        assert sum(counts.values()) == n
        cfg = apply_composition(BASE, comp)
        assert cfg.fleet.n_machines == n


def test_technicians_stay_within_bounds():
    comps = sample_compositions(BASE, 10, seed=5, technicians=(1, 3))
    assert all(1 <= c.technicians <= 3 for c in comps)


def test_infeasible_floor_raises():
    with pytest.raises(ValueError):
        sample_compositions(BASE, 3, seed=0, min_share=0.6)


def test_apply_composition_preserves_base_structure():
    base = load_scenario(BASE)
    comp = sample_compositions(BASE, 1, seed=11)[0]
    cfg = apply_composition(BASE, comp)
    assert set(cfg.families) == set(base.families)
    assert cfg.n_components == base.n_components
    assert cfg.technicians == comp.technicians
    assert cfg.fleet.family_share == pytest.approx(comp.family_share)


def test_composition_matches_simulator_repartition():
    comp = sample_compositions(BASE, 1, seed=13)[0]
    cfg = apply_composition(BASE, comp)
    sim = FleetSimulation(cfg, seed=0)
    seen = {code: int((sim.family == i).sum()) for i, code in enumerate(sim.family_codes)}
    assert seen == family_counts(comp.family_share, cfg.fleet.n_machines)


def test_write_compositions_roundtrip(tmp_path):
    comps = sample_compositions(BASE, 3, seed=9)
    paths = write_compositions(comps, BASE, tmp_path)
    assert len(paths) == 3
    for path, comp in zip(paths, comps, strict=True):
        assert path.exists()
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["base"] == BASE
        assert payload["technicians"] == comp.technicians
        assert payload["family_share"] == pytest.approx(comp.family_share)
