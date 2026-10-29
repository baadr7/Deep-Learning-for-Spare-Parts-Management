"""Interface commune des politiques de gestion et briques réutilisables."""

from __future__ import annotations

import numpy as np

from spareparts.sim.engine import Decisions, FleetSimulation, Observation
from spareparts.sim.states import MachineState


class Policy:
    """Une politique reçoit une observation chaque matin et renvoie ses décisions."""

    name = "base"
    label = ""

    def reset(self, sim: FleetSimulation) -> None:
        self.parts = sim.cfg.parts
        self.unit_cost = sim.unit_cost
        self.lead_hours = np.array([p.lead_days * 24.0 for p in sim.cfg.parts])
        self.lead_hours = self.lead_hours * sim.cfg.lead_time_factor

    def decide(self, obs: Observation) -> Decisions:
        raise NotImplementedError


def inventory_position(obs: Observation) -> np.ndarray:
    return obs.stock + obs.on_order - obs.backorders


def reorder_up_to(obs: Observation, s, S) -> dict[int, int]:
    """Règle (s, S) : si la position de stock passe sous s, on recomplète jusqu'à S."""
    ip = inventory_position(obs)
    s, S = np.asarray(s), np.asarray(S)
    return {int(p): int(S[p] - ip[p]) for p in range(len(ip)) if ip[p] <= s[p] and S[p] > ip[p]}


def schedule_preventive(
    obs: Observation, score: np.ndarray, keep: np.ndarray | int = 0
) -> list[tuple[int, int]]:
    """Retient les couples (machine, composant) au score le plus élevé, sous contraintes.

    ``score`` vaut NaN pour les couples non candidats. Une seule intervention par machine,
    dans la limite des techniciens libres et des pièces en stock ; ``keep`` est le nombre de
    pièces de chaque référence réservées aux pannes, que la maintenance préventive ne consomme
    pas.
    """
    keep = np.broadcast_to(np.asarray(keep), obs.stock.shape)
    score = np.where(obs.machine_state[:, None] == MachineState.OPERATIONAL, score, np.nan)
    flat = np.flatnonzero(~np.isnan(score))
    if flat.size == 0 or obs.techs_free <= 0:
        return []
    order = flat[np.argsort(-score.ravel()[flat])]
    stock = obs.stock.copy()
    chosen, used = [], set()
    K = score.shape[1]
    for idx in order:
        m, k = divmod(int(idx), K)
        p = obs.part_of[m, k]
        if m in used or stock[p] <= keep[p]:
            continue
        chosen.append((m, k))
        used.add(m)
        stock[p] -= 1
        if len(chosen) >= obs.techs_free:
            break
    return chosen
