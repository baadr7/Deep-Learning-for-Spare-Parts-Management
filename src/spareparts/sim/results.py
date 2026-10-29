"""Indicateurs de performance et séries journalières d'une simulation (§ 7.2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spareparts.sim.engine import COST_KEYS, SIGNALS, FleetSimulation
from spareparts.sim.states import STATE_LABELS, MachineState


@dataclass
class SimulationResult:
    scenario: str
    policy: str
    seed: int
    days: int
    kpis: dict[str, float]
    costs: dict[str, float]
    daily: dict[str, np.ndarray]
    events: list[dict]
    meta: dict = field(default_factory=dict)

    def to_summary(self) -> dict:
        return {
            "scenario": self.scenario,
            "policy": self.policy,
            "seed": self.seed,
            "days": self.days,
            "kpis": self.kpis,
            "costs": self.costs,
        }


def compute_kpis(sim: FleetSimulation) -> dict[str, float]:
    hours = max(sim.env.now, 1.0)
    availability = sim.c_running / sim.c_open if sim.c_open else 1.0
    quality = 1.0 - sim.defects / sim.produced if sim.produced else 1.0
    performance = 1.0
    demands = int(sim.demands.sum())
    total_cost = float(sum(sim.costs.values()))
    return {
        "trs": availability * performance * quality,
        "disponibilite": availability,
        "qualite": quality,
        "taux_defauts": 1.0 - quality,
        "cout_total": total_cost,
        "cout_annuel": total_cost * 365.0 / max(sim.day, 1),
        "frequence_ruptures": float(sim.stockouts.sum() / demands) if demands else 0.0,
        "attente_piece_moyenne": sim.wait_part_hours / sim.n_corrective
        if sim.n_corrective
        else 0.0,
        "attente_technicien_moyenne": (
            sim.wait_tech_hours / sim.n_corrective if sim.n_corrective else 0.0
        ),
        "stock_immobilise": sim.stock_value_hours / hours,
        "mtbf": sim.c_running / sim.n_failures if sim.n_failures else float("inf"),
        "mttr": sim.corrective_downtime / sim.n_corrective if sim.n_corrective else 0.0,
        "pannes": float(sim.n_failures),
        "interventions_correctives": float(sim.n_corrective),
        "maintenances_preventives": float(sim.n_preventive),
        "revisions": float(sim.n_overhauls),
        "echecs_test": float(sim.test_failures),
        "demandes_pieces": float(demands),
        "ruptures": float(sim.stockouts.sum()),
        "pieces_produites": float(sim.produced),
        "pieces_defectueuses": float(sim.defects),
        "charge_techniciens": sim.tech_busy_hours / (sim.cfg.technicians * hours),
    }


def build_result(sim: FleetSimulation, policy) -> SimulationResult:
    h = sim.history
    days = len(h["states"])
    daily_costs = {k: np.array([c[k] for c in h["costs"]]) for k in COST_KEYS}
    open_hours = np.array(h["open_hours"])
    running = np.array(h["running_hours"])
    produced = np.array(h["produced"], dtype=float)
    defects = np.array(h["defects"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        availability = np.where(open_hours > 0, running / open_hours, np.nan)
        quality = np.where(produced > 0, 1.0 - defects / produced, np.nan)
    daily = {
        "states": np.array(h["states"]),
        "day_state": np.array(h["day_state"]),
        "stock": np.array(h["stock"]),
        "on_order": np.array(h["on_order"]),
        "backorders": np.array(h["backorders"]),
        "costs": daily_costs,
        "availability": availability,
        "quality": quality,
        "trs": availability * quality,
        "produced": produced,
        "defects": defects,
        "defect_rate": np.array(h["defect_rate"]),
        "signal_mean": np.array(h["signal_mean"]),
        "alarms": np.array(h["alarms"]),
    }
    meta = {
        "families": [sim.family_codes[f] for f in sim.family],
        "family_labels": {c: sim.cfg.families[c].label for c in sim.family_codes},
        "shifts": sim.shifts.tolist(),
        "age_years": sim.age_years.round(1).tolist(),
        "parts": [p.model_dump() for p in sim.cfg.parts],
        "components": sim.component_names,
        "signals": list(SIGNALS),
        "states": {int(s): STATE_LABELS[MachineState(s)] for s in MachineState},
        "technicians": sim.cfg.technicians,
        "failures_by_component": sim.failures_by_component.tolist(),
        "stockouts_by_part": sim.stockouts.tolist(),
        "demands_by_part": sim.demands.tolist(),
        "state_hours": sim.state_hours.tolist(),
        "down_hours": sim.down_hours.round(1).tolist(),
    }
    if hasattr(policy, "s") and hasattr(policy, "S"):
        meta["policy_params"] = {
            "s": np.asarray(policy.s).tolist(),
            "S": np.asarray(policy.S).tolist(),
        }
    return SimulationResult(
        scenario=sim.cfg.name,
        policy=getattr(policy, "name", type(policy).__name__),
        seed=sim.seed,
        days=days,
        kpis=compute_kpis(sim),
        costs={k: float(v) for k, v in sim.costs.items()},
        daily=daily,
        events=sim.events,
        meta=meta,
    )
