import time

import numpy as np
import pytest

from spareparts.config import list_scenarios, load_scenario
from spareparts.policies import make_policy
from spareparts.sim import Decisions, FleetSimulation, MachineState


@pytest.fixture(scope="module")
def cfg():
    return load_scenario("s1_nominal")


def test_all_scenarios_load():
    for name in list_scenarios():
        assert load_scenario(name).fleet.n_machines > 0


def test_same_seed_gives_same_result(cfg):
    a = FleetSimulation(cfg, seed=3).run(make_policy("P0"), days=60)
    b = FleetSimulation(cfg, seed=3).run(make_policy("P0"), days=60)
    assert a.kpis == b.kpis


def test_invariants_hold_every_day(cfg):
    sim = FleetSimulation(cfg, seed=5)
    policy = make_policy("P2")
    policy.reset(sim)
    for _ in range(120):
        sim.apply(policy.decide(sim.observe()))
        record = sim.advance_day()
        assert (record.stock >= 0).all()
        assert record.states.min() >= 0 and record.states.max() < len(MachineState)
        assert record.running_hours <= record.open_hours
        assert record.defects <= record.produced
        assert sim.techs.count <= cfg.technicians
    assert sim.n_corrective <= sim.n_failures


def test_kpis_are_consistent(cfg):
    k = FleetSimulation(cfg, seed=2).run(make_policy("P0"), days=120).kpis
    assert 0.0 <= k["disponibilite"] <= 1.0
    assert k["trs"] <= k["disponibilite"] + 1e-12
    assert 0.0 <= k["frequence_ruptures"] <= 1.0
    assert k["ruptures"] <= k["demandes_pieces"]
    assert k["cout_total"] > 0


def test_preventive_is_refused_on_a_stopped_machine(cfg):
    sim = FleetSimulation(cfg, seed=1)
    sim.state[0] = MachineState.REPAIR
    rejected = sim.apply(Decisions(preventive=[(0, 0)]))
    assert rejected and rejected[0]["reason"] == "machine indisponible"


def test_preventive_consumes_a_part(cfg):
    sim = FleetSimulation(cfg, seed=1)
    p = int(sim.part_of[0, 0])
    before = sim.stock[p]
    assert sim.apply(Decisions(preventive=[(0, 0)])) == []
    assert sim.stock[p] == before - 1
    assert sim.state[0] == MachineState.PREVENTIVE


def test_orders_arrive_and_refill_stock(cfg):
    sim = FleetSimulation(cfg, seed=1, log_events=True)
    sim.apply(Decisions(urgent={0: 5}))
    for _ in range(4):
        sim.advance_day()
    assert any(e["type"] == "reception" and e["part"] == 0 for e in sim.events)


def test_higher_failure_rate_means_more_failures():
    base = load_scenario("s1_nominal")
    worse = load_scenario("s1_nominal", failure_rate_factor=2.0)
    a = FleetSimulation(base, seed=11).run(make_policy("P0"), days=180).kpis["pannes"]
    b = FleetSimulation(worse, seed=11).run(make_policy("P0"), days=180).kpis["pannes"]
    assert b > a


def test_oracle_costs_less_than_hand_tuned_policy(cfg):
    costs = {}
    for code in ("P0", "Oracle"):
        runs = [FleetSimulation(cfg, seed=s).run(make_policy(code)) for s in (1, 2)]
        costs[code] = np.mean([r.kpis["cout_total"] for r in runs])
    assert costs["Oracle"] < costs["P0"]


def test_one_year_runs_fast_enough(cfg):
    start = time.perf_counter()
    FleetSimulation(cfg, seed=4, log_events=False).run(make_policy("P0"))
    assert time.perf_counter() - start < 3.0
