"""Plan d'expériences du modèle de substitution (§ 5.4).

Chaque scénario combine des paramètres du système (techniciens, délais, taux de panne) et de
la politique à seuils fixes (facteur d'intervalle de maintenance, facteur de stock). Il est
simulé sur plusieurs graines et résumé par la moyenne de ses indicateurs.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import qmc

from spareparts.config import load_scenario
from spareparts.policies import FixedThresholdPolicy
from spareparts.sim import FleetSimulation

PARAMS: dict[str, tuple[float, float, bool]] = {
    "technicians": (1, 4, True),
    "lead_time_factor": (0.5, 2.5, False),
    "failure_rate_factor": (0.6, 1.8, False),
    "pm_factor": (0.5, 1.6, False),
    "stock_factor": (0.5, 3.0, False),
}
OUTPUTS = [
    "cout_total",
    "trs",
    "disponibilite",
    "frequence_ruptures",
    "attente_piece_moyenne",
    "pannes",
]


def sample(n: int, seed: int = 0) -> pd.DataFrame:
    unit = qmc.LatinHypercube(d=len(PARAMS), seed=seed).random(n)
    cols = {}
    for j, (name, (lo, hi, integer)) in enumerate(PARAMS.items()):
        v = lo + unit[:, j] * (hi - lo)
        cols[name] = np.rint(v).astype(int) if integer else np.round(v, 4)
    return pd.DataFrame(cols)


def evaluate(scenario: str, params: dict, seeds, days: int) -> dict[str, float]:
    cfg = load_scenario(
        scenario,
        technicians=int(params["technicians"]),
        lead_time_factor=float(params["lead_time_factor"]),
        failure_rate_factor=float(params["failure_rate_factor"]),
    )
    f = float(params["stock_factor"])
    S = [max(1, round(p.S * f)) for p in cfg.parts]
    s = [min(max(0, round(p.s * f)), big - 1) for p, big in zip(cfg.parts, S, strict=True)]
    runs = []
    for seed in seeds:
        policy = FixedThresholdPolicy(s=s, S=S, pm_factor=float(params["pm_factor"]), name="DOE")
        runs.append(FleetSimulation(cfg, seed=seed, log_events=False).run(policy, days=days).kpis)
    return {k: float(np.mean([r[k] for r in runs])) for k in OUTPUTS}


def _job(args):
    scenario, params, seeds, days = args
    return evaluate(scenario, params, seeds, days)


def build(
    scenario: str, n: int, replications: int = 2, days: int = 180, workers: int | None = None
) -> pd.DataFrame:
    design = sample(n)
    seeds = list(range(5000, 5000 + replications))
    jobs = [(scenario, row._asdict(), seeds, days) for row in design.itertuples(index=False)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_job, jobs, chunksize=4))
    return pd.concat([design, pd.DataFrame(results)], axis=1)
