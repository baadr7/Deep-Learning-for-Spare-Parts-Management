"""Validation du simulateur sur le jeu Azure (objectif O2, § 5.1).

Le simulateur rejoue la politique observée dans Azure : chaque remplacement planifié du
journal de maintenance (hors remplacements consécutifs à une panne) est appliqué à sa date,
machine par machine et composant par composant, du 1er juin 2014 au 1er janvier 2016. Les
pannes, elles, ne proviennent que des lois de Weibull. Les 100 machines tournent 24 heures sur
24, comme dans Azure.

Les lois ajustées directement sur les durées (censure supposée indépendante) sous-estiment les
pannes d'environ 30 % : les remplacements planifiés d'Azure visent probablement des composants
déjà usés, ce qui rend la censure informative. L'échelle de chaque loi est donc corrigée par
simulation sur le premier semestre 2015, puis la correction est validée sur le second semestre,
que la calibration n'a pas vu.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from spareparts.config import ComponentSpec, FamilySpec, Lognormal, PartSpec, load_scenario
from spareparts.policies.base import Policy
from spareparts.sim import FleetSimulation
from spareparts.sim.engine import Decisions, Observation

START = pd.Timestamp("2014-06-01 06:00:00")
YEAR_START = pd.Timestamp("2015-01-01 06:00:00")
MID = pd.Timestamp("2015-07-01 06:00:00")
END = pd.Timestamp("2016-01-01 06:00:00")
PERIODS = {"S1": (YEAR_START, MID), "S2": (MID, END)}
SIGNALS = ["volt", "rotate", "pressure", "vibration"]


def azure_config(weibull: dict[str, dict], init_age_h: float = 1200.0):
    """Scénario à une famille, 24 h/24, avec les lois de Weibull ajustées sur Azure."""
    base = load_scenario("s1_nominal")
    comps = sorted(weibull)
    components = [
        ComponentSpec(
            name=c,
            part=i,
            regime="calendar",
            weibull_beta=weibull[c]["beta"],
            weibull_eta=weibull[c]["eta"],
            pm_interval=init_age_h,
            repair=Lognormal(mean=6.0, cv=0.3),
            signal=SIGNALS[i % 4],
            drift=2.5,
            defect_type="Non précisé",
            defect_weight=0.0,
        )
        for i, c in enumerate(comps)
    ]
    family = FamilySpec(
        label="Machines Azure", shifts=[3], cadence=40, margin_per_part=0.025, components=components
    )
    parts = [
        PartSpec(
            code=f"P{i + 1}",
            name=c,
            nature="Rejeu",
            unit_cost=1,
            lead_days=1,
            urgent_lead_days=1,
            s=200,
            S=500,
            initial_stock=1000,
        )
        for i, c in enumerate(comps)
    ]
    return base.model_copy(
        update={
            "name": "Rejeu Azure",
            "technicians": 50,
            "fleet": base.fleet.model_copy(update={"family_share": {"A": 1.0}}),
            "families": {"A": family},
            "parts": parts,
        }
    )


def planned_schedule(maint: pd.DataFrame, failures: pd.DataFrame, comps: list[str]):
    """Jour de simulation vers liste (machine, composant) des remplacements planifiés."""
    fail_keys = set(
        zip(failures["machineID"], failures["datetime"], failures["failure"], strict=True)
    )
    schedule: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in maint.itertuples(index=False):
        if (row.machineID, row.datetime, row.comp) in fail_keys:
            continue
        day = (row.datetime - START).days
        if day >= 0:
            schedule[day].append((int(row.machineID) - 1, comps.index(row.comp)))
    return dict(schedule)


class AzureTracePolicy(Policy):
    """Applique les remplacements planifiés du journal Azure à leur date."""

    name = "Rejeu Azure"

    def __init__(self, schedule: dict[int, list[tuple[int, int]]]):
        self.schedule = schedule

    def decide(self, obs: Observation) -> Decisions:
        return Decisions(preventive=list(self.schedule.get(obs.day, [])))


def azure_failures_by_period(failures: pd.DataFrame) -> dict[str, dict[str, int]]:
    out = {}
    for name, (a, b) in PERIODS.items():
        sel = failures[(failures["datetime"] >= a) & (failures["datetime"] < b)]
        out[name] = sel["failure"].value_counts().to_dict()
    return out


def simulated_failures(weibull, schedule, seeds) -> dict[str, np.ndarray]:
    """Pannes simulées par période : {période: [graines, composants]}."""
    comps = sorted(weibull)
    cfg = azure_config(weibull)
    bounds = {n: ((a - START).days, (b - START).days) for n, (a, b) in PERIODS.items()}
    counts = {n: [] for n in PERIODS}
    for seed in seeds:
        sim = FleetSimulation(cfg, seed=seed, log_events=True)
        sim.run(AzureTracePolicy(schedule), days=(END - START).days)
        per = {n: np.zeros(len(comps)) for n in PERIODS}
        for e in sim.events:
            if e["type"] != "panne":
                continue
            for n, (a, b) in bounds.items():
                if a <= e["day"] < b:
                    per[n][e["component"]] += 1
        for n in PERIODS:
            counts[n].append(per[n])
    return {n: np.array(v) for n, v in counts.items()}


def compare(sims: np.ndarray, azure: dict[str, int], comps: list[str]) -> dict:
    out = {}
    for i, c in enumerate(comps):
        mean = float(sims[:, i].mean())
        out[c] = {
            "azure": int(azure.get(c, 0)),
            "simulees": mean,
            "ecart_type": float(sims[:, i].std(ddof=1)) if len(sims) > 1 else 0.0,
            "ecart_relatif": (mean - azure[c]) / azure[c],
        }
    return out


def calibrate_scale(
    weibull: dict[str, dict], schedule, azure_s1: dict[str, int], seeds, iterations: int = 3
) -> dict[str, dict]:
    """Corrige l'échelle eta de chaque loi pour reproduire les pannes du premier semestre.

    Pour une loi de Weibull, le nombre de pannes varie à peu près comme eta^(-beta) : on
    multiplie eta par (simulé / observé)^(1/beta), puis on recommence.
    """
    comps = sorted(weibull)
    adjusted = {c: dict(weibull[c]) for c in comps}
    for _ in range(iterations):
        sims = simulated_failures(adjusted, schedule, seeds)["S1"].mean(axis=0)
        for i, c in enumerate(comps):
            ratio = max(sims[i], 1.0) / azure_s1[c]
            adjusted[c]["eta"] = float(adjusted[c]["eta"] * ratio ** (1.0 / adjusted[c]["beta"]))
    return adjusted
