"""Traduction entre l'état du simulateur et les vecteurs manipulés par un agent (§ 5.3).

Un « codec » fournit l'observation vectorielle, l'espace d'actions, les masques d'actions
invalides et la conversion d'une action en décisions. Les mêmes codecs servent à
l'entraînement (environnement Gymnasium) et à l'exécution d'un agent entraîné dans le
simulateur ou dans l'application web.

Niveaux d'action :

- ``reduced`` : une seule action discrète parmi 36 (seuil de maintenance commun à tous les
  composants, multiplicateur de stock commun à toutes les références), pour le DQN classique ;
- ``n1`` : un seuil de maintenance par composant et un niveau de stock par référence ;
- ``n2`` : quantités commandées, commandes urgentes et maintenance machine par machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spareparts.policies.base import inventory_position, reorder_up_to, schedule_preventive
from spareparts.sim.engine import Decisions, FleetSimulation, Observation
from spareparts.sim.states import MachineState

PM_LEVELS = np.array([0.6, 0.8, 1.0, 1.2, 1.5, np.inf])
STOCK_LEVELS = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
URGENT_MAX = 2
OBS_CLIP = 5.0
# Au niveau N2, un composant n'est remplaçable préventivement qu'à partir de la moitié de son
# intervalle de maintenance : en deçà, le remplacement gaspille une pièce presque neuve.
PM_MIN_RATIO = 0.5
# Biais initial des logits de l'option « ne rien faire » (voir ``inaction_prior``).
PRIOR_ORDER = 4.0
PRIOR_URGENT = 4.0
PRIOR_MACHINE = 5.0


@dataclass
class Context:
    """Grandeurs fixes d'un épisode, utilisées pour normaliser les observations."""

    S_base: np.ndarray
    capacity: int
    horizon: int
    n_machines: int
    n_components: int
    n_parts: int

    @classmethod
    def from_sim(cls, sim: FleetSimulation, horizon: int) -> Context:
        return cls(
            S_base=np.array([max(p.S, 1) for p in sim.cfg.parts], dtype=float),
            capacity=sim.cfg.technicians,
            horizon=horizon,
            n_machines=sim.M,
            n_components=sim.K,
            n_parts=sim.P,
        )


def fleet_features(obs: Observation, ctx: Context) -> np.ndarray:
    """Résumé de la flotte et du magasin, indépendant du nombre de machines."""
    S = ctx.S_base
    ip = inventory_position(obs)
    parts = np.concatenate([obs.stock / S, obs.on_order / S, obs.backorders / 5.0, ip / S])

    operational = obs.machine_state == MachineState.OPERATIONAL
    ratio = obs.age_unit / obs.pm_interval
    r = ratio[operational] if operational.any() else ratio
    comps = np.concatenate(
        [r.mean(axis=0), (r >= 0.8).mean(axis=0), (r >= 1.0).mean(axis=0), r.max(axis=0) / 2.0]
    )
    if obs.risk is not None:
        comps = np.concatenate([comps, obs.risk.mean(axis=0) * 10.0, obs.risk.max(axis=0)])

    states = obs.machine_state
    rate = np.nan_to_num(obs.defect_rate_24h, nan=0.0)
    fleet = np.array(
        [
            operational.mean(),
            (states == MachineState.WAITING_PART).mean() * 10.0,
            (states == MachineState.WAITING_TECH).mean() * 10.0,
            obs.techs_free / max(ctx.capacity, 1),
            rate.mean() * 20.0,
            obs.alarms_24h.mean() / 5.0,
            obs.day / max(ctx.horizon, 1),
        ]
    )
    return np.clip(np.concatenate([parts, comps, fleet]), -OBS_CLIP, OBS_CLIP).astype(np.float32)


def machine_features(obs: Observation) -> np.ndarray:
    """Pour chaque machine : en service ou non, avancement de chaque composant."""
    operational = (obs.machine_state == MachineState.OPERATIONAL).astype(float)[:, None]
    ratio = np.clip(obs.age_unit / obs.pm_interval, 0.0, 3.0) / 2.0
    cols = [operational, ratio]
    if obs.risk is not None:
        cols.append(np.clip(obs.risk * 10.0, 0.0, OBS_CLIP))
    return np.concatenate(cols, axis=1).ravel().astype(np.float32)


def _threshold_preventive(obs: Observation, thresholds: np.ndarray) -> list[tuple[int, int]]:
    ratio = obs.age_unit / obs.pm_interval
    score = np.where(ratio >= thresholds[None, :], ratio, np.nan)
    return schedule_preventive(obs, score, keep=1)


def _stock_rule(obs: Observation, ctx: Context, multipliers: np.ndarray) -> dict[int, int]:
    S = np.maximum(np.rint(ctx.S_base * multipliers), 1).astype(int)
    s = np.maximum(np.rint(0.4 * S), 0).astype(int)
    return reorder_up_to(obs, s, S)


class ReducedCodec:
    """36 actions : 6 seuils de maintenance communs × 6 multiplicateurs de stock communs."""

    level = "reduced"

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.n = len(PM_LEVELS) * len(STOCK_LEVELS)
        self.obs_size = len(fleet_features_template(ctx))

    def observe(self, obs: Observation) -> np.ndarray:
        return fleet_features(obs, self.ctx)

    def masks(self, obs: Observation) -> np.ndarray:
        return np.ones(self.n, dtype=bool)

    def decide(self, action, obs: Observation) -> Decisions:
        a = int(np.asarray(action).item())
        pm, stock = divmod(a, len(STOCK_LEVELS))
        thresholds = np.full(self.ctx.n_components, PM_LEVELS[pm])
        multipliers = np.full(self.ctx.n_parts, STOCK_LEVELS[stock])
        return Decisions(
            orders=_stock_rule(obs, self.ctx, multipliers),
            preventive=_threshold_preventive(obs, thresholds),
        )


class Level1Codec:
    """N1 : un seuil de maintenance par composant, un niveau de stock par référence."""

    level = "n1"

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.nvec = np.array(
            [len(PM_LEVELS)] * ctx.n_components + [len(STOCK_LEVELS)] * ctx.n_parts
        )
        self.obs_size = len(fleet_features_template(ctx))

    def observe(self, obs: Observation) -> np.ndarray:
        return fleet_features(obs, self.ctx)

    def masks(self, obs: Observation) -> np.ndarray:
        return np.ones(int(self.nvec.sum()), dtype=bool)

    def decide(self, action, obs: Observation) -> Decisions:
        a = np.asarray(action, dtype=int)
        K = self.ctx.n_components
        return Decisions(
            orders=_stock_rule(obs, self.ctx, STOCK_LEVELS[a[K:]]),
            preventive=_threshold_preventive(obs, PM_LEVELS[a[:K]]),
        )


class Level2Codec:
    """N2 : quantités commandées, commandes urgentes et maintenance machine par machine."""

    level = "n2"

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.q_max = (2 * ctx.S_base).astype(int)
        self.nvec = np.concatenate(
            [
                self.q_max + 1,
                np.full(ctx.n_parts, URGENT_MAX + 1),
                np.full(ctx.n_machines, ctx.n_components + 1),
            ]
        ).astype(int)
        per_machine = 1 + ctx.n_components
        self.obs_size = len(fleet_features_template(ctx)) + ctx.n_machines * per_machine

    def observe(self, obs: Observation) -> np.ndarray:
        return np.concatenate([fleet_features(obs, self.ctx), machine_features(obs)])

    def masks(self, obs: Observation) -> np.ndarray:
        ctx = self.ctx
        parts = [np.ones(q + 1, dtype=bool) for q in self.q_max]
        urgent = [np.ones(URGENT_MAX + 1, dtype=bool) for _ in range(ctx.n_parts)]
        operational = obs.machine_state == MachineState.OPERATIONAL
        in_stock = obs.stock[obs.part_of] > 0
        worn = obs.age_unit / obs.pm_interval >= PM_MIN_RATIO
        can = operational[:, None] & in_stock & worn & (obs.techs_free > 0)
        machines = np.concatenate([np.ones((ctx.n_machines, 1), dtype=bool), can], axis=1)
        return np.concatenate(parts + urgent + [machines.ravel()])

    def decide(self, action, obs: Observation) -> Decisions:
        a = np.asarray(action, dtype=int)
        P, M = self.ctx.n_parts, self.ctx.n_machines
        orders = {p: int(a[p]) for p in range(P) if a[p] > 0}
        urgent = {p: int(a[P + p]) for p in range(P) if a[P + p] > 0}
        choice = a[2 * P : 2 * P + M]
        preventive = [(int(m), int(choice[m] - 1)) for m in np.flatnonzero(choice > 0)]
        return Decisions(orders=orders, urgent=urgent, preventive=preventive)


def inaction_prior(codec: Level2Codec) -> np.ndarray:
    """Biais à ajouter aux sorties d'un agent N2 non entraîné, une valeur par option.

    Un agent initialisé au hasard choisit presque uniformément entre les options : il commande
    chaque jour des quantités aléatoires et envoie en maintenance quatre machines sur cinq. Ces
    années absurdes coûtent dix fois le coût normal et ralentissent fortement la simulation.
    Le biais rend l'option 0 (rien à commander, rien à faire) nettement plus probable au départ ;
    l'apprentissage peut ensuite s'en écarter librement.
    """
    ctx = codec.ctx
    first = np.concatenate(
        [
            np.full(ctx.n_parts, PRIOR_ORDER),
            np.full(ctx.n_parts, PRIOR_URGENT),
            np.full(ctx.n_machines, PRIOR_MACHINE),
        ]
    )
    prior = np.zeros(int(codec.nvec.sum()))
    starts = np.concatenate([[0], np.cumsum(codec.nvec)[:-1]])
    prior[starts] = first
    return prior


def fleet_features_template(ctx: Context) -> np.ndarray:
    """Taille de ``fleet_features`` sans risque prédit (4 P + 4 K + 7)."""
    return np.zeros(4 * ctx.n_parts + 4 * ctx.n_components + 7)


CODECS = {"reduced": ReducedCodec, "n1": Level1Codec, "n2": Level2Codec}


def make_codec(level: str, ctx: Context):
    try:
        return CODECS[level](ctx)
    except KeyError as exc:
        raise ValueError(f"niveau d'action inconnu : {level}") from exc
