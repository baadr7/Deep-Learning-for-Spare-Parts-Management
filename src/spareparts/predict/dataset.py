"""Construction des jeux d'apprentissage du module de prédiction (§ 5.2).

Chaque exemple correspond à une machine, observée au début d'une journée :

- une fenêtre des ``WINDOW`` dernières heures : quatre signaux normalisés et l'indicateur de
  marche (un signal absent, machine arrêtée, vaut 0) ;
- des attributs statiques : famille, nombre de postes, avancement de chaque composant par
  rapport à son intervalle de maintenance (information connue en pratique) ;
- une cible de survie en temps discret par composant, sur ``HORIZON`` jours : le jour de la
  panne si elle survient dans l'horizon, sinon le nombre de jours observés sans panne
  (censure par l'horizon ou par un remplacement préventif).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spareparts.config import load_scenario
from spareparts.policies import make_policy
from spareparts.sim import FleetSimulation

WINDOW = 48
HORIZON = 14


@dataclass
class PredictionData:
    seq: np.ndarray  # [N, WINDOW, 5] float32
    static: np.ndarray  # [N, S] float32
    event_bin: np.ndarray  # [N, K] int, -1 si pas de panne dans l'horizon
    observed: np.ndarray  # [N, K] int, nombre de jours observés sans panne
    group: np.ndarray  # [N] graine de simulation
    day: np.ndarray  # [N]
    machine: np.ndarray  # [N]

    def __len__(self) -> int:
        return len(self.day)

    def subset(self, mask: np.ndarray) -> PredictionData:
        return PredictionData(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))

    def label_24h(self) -> np.ndarray:
        """Panne de chaque composant dans les 24 heures : [N, K] booléen."""
        return self.event_bin == 0

    def label_days(self, days: int) -> np.ndarray:
        """Panne de chaque composant dans les ``days`` prochains jours : [N, K] booléen."""
        return (self.event_bin >= 0) & (self.event_bin < days)

    def save(self, path) -> None:
        np.savez_compressed(path, **{f: getattr(self, f) for f in self.__dataclass_fields__})

    @classmethod
    def load(cls, path) -> PredictionData:
        with np.load(path) as z:
            return cls(**{f: z[f] for f in cls.__dataclass_fields__})


def _events_by_component(events: list[dict], M: int, K: int, usage: np.ndarray):
    """Instants de panne et de remplacement préventif, par (machine, composant)."""
    fails = [[[] for _ in range(K)] for _ in range(M)]
    renew = [[[] for _ in range(K)] for _ in range(M)]
    for e in events:
        m, k = e.get("machine"), e.get("component")
        if e["type"] == "panne":
            fails[m][k].append(e["t"])
        elif e["type"] == "maintenance_preventive":
            renew[m][k].append(e["t"])
        elif e["type"] == "fin_revision":
            for kk in range(K):
                if not usage[m, kk]:
                    renew[m][kk].append(e["t"])
    return fails, renew


def simulate_dataset(
    scenario: str = "s1_nominal", seeds=(1,), policy: str = "P0", days: int = 365
) -> PredictionData:
    cfg = load_scenario(scenario)
    parts = []
    for seed in seeds:
        sim = FleetSimulation(cfg, seed=seed, log_events=True, log_telemetry=True)
        static_days = []
        pol = make_policy(policy, scenario)
        pol.reset(sim)
        for _ in range(days):
            obs = sim.observe()
            ratio = np.clip(obs.age_unit / obs.pm_interval, 0, 3)
            static_days.append(ratio.astype(np.float32))
            sim.apply(pol.decide(obs))
            sim.advance_day()
        parts.append(_build(sim, np.array(static_days), seed))
    return PredictionData(
        *(
            np.concatenate([getattr(p, f) for p in parts])
            for f in PredictionData.__dataclass_fields__
        )
    )


def _build(sim: FleetSimulation, ratios: np.ndarray, seed: int) -> PredictionData:
    M, K = sim.M, sim.K
    tel = np.stack(sim.telemetry_log)  # [T, M, 4]
    running = ~np.isnan(tel[..., 0])
    z = (tel - sim.sig_mean) / sim.sig_sd
    z = np.nan_to_num(z, nan=0.0).astype(np.float32)
    fails, renew = _events_by_component(sim.events, M, K, sim.usage)
    n_fam = len(sim.family_codes)

    seq, static, ev, obs_days, day_idx, mach = [], [], [], [], [], []
    days = ratios.shape[0]
    for d in range(WINDOW // 24, days - 1):
        t0 = d * 24
        window = np.concatenate(
            [z[t0 - WINDOW : t0], running[t0 - WINDOW : t0, :, None].astype(np.float32)], axis=2
        )  # [W, M, 5]
        for m in range(M):
            fam = np.zeros(n_fam, dtype=np.float32)
            fam[sim.family[m]] = 1.0
            static.append(
                np.concatenate([fam, [sim.shifts[m] / 3.0], ratios[d, m]]).astype(np.float32)
            )
            seq.append(window[:, m, :])
            e_row, o_row = [], []
            for k in range(K):
                nf = next((t for t in fails[m][k] if t >= t0), np.inf)
                nr = next((t for t in renew[m][k] if t >= t0), np.inf)
                if nf < nr and nf - t0 < HORIZON * 24:
                    b = int((nf - t0) // 24)
                    e_row.append(b)
                    o_row.append(b)
                else:
                    e_row.append(-1)
                    o_row.append(int(min(HORIZON, (min(nr, (days - d) * 24 + t0) - t0) // 24)))
            ev.append(e_row)
            obs_days.append(o_row)
            day_idx.append(d)
            mach.append(m)
    return PredictionData(
        seq=np.asarray(seq, dtype=np.float32),
        static=np.asarray(static, dtype=np.float32),
        event_bin=np.asarray(ev, dtype=np.int16),
        observed=np.asarray(obs_days, dtype=np.int16),
        group=np.full(len(day_idx), seed, dtype=np.int32),
        day=np.asarray(day_idx, dtype=np.int16),
        machine=np.asarray(mach, dtype=np.int16),
    )


def tabular_features(data: PredictionData) -> np.ndarray:
    """Variables glissantes pour LightGBM : moyennes, écarts-types et dernières valeurs."""
    s = data.seq
    run = s[..., 4:5]
    sig = s[..., :4]
    feats = []
    for hours in (3, 24, WINDOW):
        part = sig[:, -hours:]
        w = run[:, -hours:]
        n = np.maximum(w.sum(axis=1), 1.0)
        mean = (part * w).sum(axis=1) / n
        var = ((part - mean[:, None]) ** 2 * w).sum(axis=1) / n
        feats += [mean, np.sqrt(var)]
    feats.append(run[:, -24:].mean(axis=1))
    feats.append(data.static)
    return np.concatenate(feats, axis=1).astype(np.float32)
