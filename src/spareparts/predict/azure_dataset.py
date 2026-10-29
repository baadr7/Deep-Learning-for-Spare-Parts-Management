"""Jeu d'apprentissage construit sur les données réelles d'Azure (§ 5.2).

Même forme que le jeu simulé, pour réutiliser les mêmes réseaux :

- fenêtre de 48 heures : les quatre signaux normalisés, et le nombre d'erreurs de l'heure à la
  place de l'indicateur de marche (les machines Azure tournent en continu) ;
- attributs statiques : modèle de la machine, âge, temps écoulé depuis le dernier remplacement
  de chaque composant (en milliers d'heures) ;
- cible de survie sur 14 jours, censurée par un remplacement planifié ou la fin de l'historique.

Le découpage est temporel, comme le prévoit le protocole : ``group`` contient le mois.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spareparts.data import azure
from spareparts.predict.dataset import HORIZON, WINDOW, PredictionData

T0 = pd.Timestamp("2015-01-01 06:00:00")
SIGNALS = azure.SIGNAL_COLUMNS


def _hours(ts: pd.Series) -> np.ndarray:
    return ((ts - T0) / pd.Timedelta(hours=1)).to_numpy()


def build_azure_dataset() -> PredictionData:
    tel = azure.load("telemetry").sort_values(["machineID", "datetime"])
    machines = azure.load("machines").sort_values("machineID")
    errors, maint, failures = azure.load("errors"), azure.load("maint"), azure.load("failures")

    ids = machines["machineID"].to_numpy()
    M = len(ids)
    hours = int(_hours(tel["datetime"]).max()) + 1
    values = tel[SIGNALS].to_numpy(np.float32)
    z = (values - values.mean(axis=0)) / values.std(axis=0)
    signals = np.zeros((hours, M, 4), np.float32)
    signals[_hours(tel["datetime"]).astype(int), tel["machineID"].to_numpy() - 1] = z

    err = np.zeros((hours, M), np.float32)
    e_h = _hours(errors["datetime"]).astype(int)
    keep = (e_h >= 0) & (e_h < hours)
    np.add.at(err, (e_h[keep], errors["machineID"].to_numpy()[keep] - 1), 1.0)

    comps = sorted(maint["comp"].unique())
    K = len(comps)
    fail_keys = set(
        zip(failures["machineID"], failures["datetime"], failures["failure"], strict=True)
    )
    fail_times = [[[] for _ in range(K)] for _ in range(M)]
    planned = [[[] for _ in range(K)] for _ in range(M)]
    replaced = [[[] for _ in range(K)] for _ in range(M)]
    for row in maint.itertuples(index=False):
        m, k, h = row.machineID - 1, comps.index(row.comp), _hours(pd.Series([row.datetime]))[0]
        replaced[m][k].append(h)
        if (row.machineID, row.datetime, row.comp) not in fail_keys:
            planned[m][k].append(h)
    for row in failures.itertuples(index=False):
        fail_times[row.machineID - 1][comps.index(row.failure)].append(
            _hours(pd.Series([row.datetime]))[0]
        )
    fail_times = [[np.sort(x) for x in row] for row in fail_times]
    planned = [[np.sort(x) for x in row] for row in planned]
    replaced = [[np.sort(x) for x in row] for row in replaced]

    model_onehot = pd.get_dummies(machines["model"]).to_numpy(np.float32)
    age = (machines["age"].to_numpy(np.float32) / 20.0)[:, None]

    seq, static, ev, obs, group, day_idx, mach = [], [], [], [], [], [], []
    last_day = hours // 24
    for d in range(WINDOW // 24, last_day):
        t = d * 24
        window = np.concatenate([signals[t - WINDOW : t], err[t - WINDOW : t, :, None]], axis=2)
        month = (T0 + pd.Timedelta(hours=t)).month
        for m in range(M):
            since = []
            e_row, o_row = [], []
            for k in range(K):
                r = replaced[m][k]
                # Strictement avant t : un remplacement daté de t est la conséquence d'une panne
                # enregistrée au même instant, le compter révélerait la cible.
                i = np.searchsorted(r, t, side="left") - 1
                since.append(min((t - r[i]) / 1000.0, 3.0) if i >= 0 else 3.0)
                f = fail_times[m][k]
                nf = f[np.searchsorted(f, t)] if np.searchsorted(f, t) < len(f) else np.inf
                p = planned[m][k]
                nr = p[np.searchsorted(p, t)] if np.searchsorted(p, t) < len(p) else np.inf
                if nf < nr and nf - t < HORIZON * 24:
                    b = int((nf - t) // 24)
                    e_row.append(b)
                    o_row.append(b)
                else:
                    e_row.append(-1)
                    end = min(nr, hours)
                    o_row.append(int(min(HORIZON, (end - t) // 24)))
            seq.append(window[:, m, :])
            static.append(np.concatenate([model_onehot[m], age[m], since]).astype(np.float32))
            ev.append(e_row)
            obs.append(o_row)
            group.append(month)
            day_idx.append(d)
            mach.append(m)
    return PredictionData(
        seq=np.asarray(seq, np.float32),
        static=np.asarray(static, np.float32),
        event_bin=np.asarray(ev, np.int16),
        observed=np.asarray(obs, np.int16),
        group=np.asarray(group, np.int32),
        day=np.asarray(day_idx, np.int16),
        machine=np.asarray(mach, np.int16),
    )
