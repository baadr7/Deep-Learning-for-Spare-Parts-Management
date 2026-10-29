"""Conversion des résultats de simulation en structures JSON."""

from __future__ import annotations

import math

import numpy as np

from spareparts.sim import FleetSimulation, SimulationResult
from spareparts.sim.engine import COST_KEYS, DayRecord
from spareparts.sim.states import N_STATES

LIVE_EVENT_TYPES = {
    "panne",
    "rupture",
    "maintenance_preventive",
    "commande",
    "reception",
    "remise_en_service",
    "revision",
    "echec_test",
}


def jsonable(value):
    """Remplace les types numpy et les valeurs non finies par des équivalents JSON."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run_payload(run_id: str, result: SimulationResult) -> dict:
    d = result.daily
    states = d["states"]
    state_counts = np.stack([(states == s).sum(axis=1) for s in range(N_STATES)], axis=1)
    cumulative = {k: np.cumsum(d["costs"][k]) for k in COST_KEYS}
    daily_total = sum(d["costs"][k] for k in COST_KEYS)
    return jsonable(
        {
            "run_id": run_id,
            **result.to_summary(),
            "meta": result.meta,
            "daily": {
                "trs": d["trs"],
                "availability": d["availability"],
                "quality": d["quality"],
                "stock": d["stock"],
                "on_order": d["on_order"],
                "backorders": d["backorders"],
                "state_counts": state_counts,
                "cost_total": daily_total,
                "cost_cumulative": cumulative,
                "produced": d["produced"],
                "defects": d["defects"],
            },
        }
    )


def machine_payload(result: SimulationResult, m: int) -> dict:
    d = result.daily
    events = [e for e in result.events if e.get("machine") == m]
    return jsonable(
        {
            "machine": m,
            "family": result.meta["families"][m],
            "shifts": result.meta["shifts"][m],
            "age_years": result.meta["age_years"][m],
            "states": d["states"][:, m],
            "signal_mean": d["signal_mean"][:, m, :],
            "defect_rate": d["defect_rate"][:, m],
            "alarms": d["alarms"][:, m],
            "events": events,
            "signals": result.meta["signals"],
            "components": result.meta["components"],
        }
    )


def live_frame(sim: FleetSimulation, record: DayRecord) -> dict:
    total_cost = float(sum(sim.costs.values()))
    availability = sim.c_running / sim.c_open if sim.c_open else 1.0
    quality = 1.0 - sim.defects / sim.produced if sim.produced else 1.0
    events = [e for e in record.events if e["type"] in LIVE_EVENT_TYPES][-40:]
    return jsonable(
        {
            "type": "day",
            "day": record.day,
            "states": record.states,
            "stock": record.stock,
            "on_order": record.on_order,
            "backorders": record.backorders,
            "techs_busy": sim.techs.count,
            "techs_queue": len(sim.techs.queue),
            "trs": availability * quality,
            "availability": availability,
            "quality": quality,
            "cost_total": total_cost,
            "cost_day": float(sum(record.costs.values())),
            "failures": sim.n_failures,
            "stockouts": int(sim.stockouts.sum()),
            "events": events,
        }
    )
