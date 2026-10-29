"""Ajustement des lois de Weibull des composants (calibration du simulateur, § 5.1).

Une durée de vie s'étend entre deux remplacements successifs d'un même composant. Elle est
complète si le remplacement qui la termine fait suite à une panne, et censurée à droite s'il
s'agit d'un remplacement planifié ou de la fin de l'historique : le composant aurait encore
duré. Ignorer cette censure ferait croire à des durées de vie plus courtes qu'en réalité.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lifetimes_from_azure(maint: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """Durées entre remplacements (heures) avec indicateur de panne, par machine et composant."""
    fail_keys = set(
        zip(failures["machineID"], failures["datetime"], failures["failure"], strict=True)
    )
    end = maint["datetime"].max()
    rows = []
    for (machine, comp), grp in maint.sort_values("datetime").groupby(["machineID", "comp"]):
        times = grp["datetime"].tolist()
        for start, stop in zip(times[:-1], times[1:], strict=True):
            rows.append(
                {
                    "machineID": machine,
                    "comp": comp,
                    "duration": (stop - start).total_seconds() / 3600.0,
                    "event": (machine, stop, comp) in fail_keys,
                }
            )
        rows.append(
            {
                "machineID": machine,
                "comp": comp,
                "duration": (end - times[-1]).total_seconds() / 3600.0,
                "event": False,
            }
        )
    out = pd.DataFrame(rows)
    return out[out["duration"] > 0].reset_index(drop=True)


def fit_weibull(durations: np.ndarray, events: np.ndarray) -> dict[str, float]:
    """Paramètres de forme (beta) et d'échelle (eta) par maximum de vraisemblance censurée."""
    from lifelines import WeibullFitter

    wf = WeibullFitter().fit(np.asarray(durations, float), event_observed=np.asarray(events, bool))
    return {
        "beta": float(wf.rho_),
        "eta": float(wf.lambda_),
        "n": int(len(durations)),
        "pannes": int(np.sum(events)),
    }


def fit_by_component(lifetimes: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        comp: fit_weibull(grp["duration"].to_numpy(), grp["event"].to_numpy())
        for comp, grp in lifetimes.groupby("comp")
    }


def fit_aft(lifetimes: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    """Modèle de Weibull à temps de vie accéléré : effet du modèle et de l'âge de la machine."""
    from lifelines import WeibullAFTFitter

    df = lifetimes.merge(machines, on="machineID")
    design = pd.get_dummies(df[["model"]], drop_first=True, dtype=float)
    design["age"] = df["age"].astype(float)
    design["duration"] = df["duration"]
    design["event"] = df["event"].astype(int)
    aft = WeibullAFTFitter().fit(design, duration_col="duration", event_col="event")
    return aft.summary
