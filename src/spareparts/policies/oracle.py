"""Politique oracle : connaît la durée de vie réelle de chaque composant (§ 7.1).

Elle remplace chaque composant lorsqu'il a consommé ``target`` de sa durée de vie réelle, ce
qui évite la panne et limite la hausse du taux de défauts en fin de vie, et commande les pièces
dont elle sait qu'elles seront nécessaires avant la prochaine livraison. Aucune politique
réaliste ne dispose de cette information : l'oracle sert de borne de comparaison.
"""

from __future__ import annotations

import numpy as np

from spareparts.policies.base import Policy, inventory_position, schedule_preventive
from spareparts.sim.engine import Decisions, FleetSimulation, Observation


class OraclePolicy(Policy):
    name = "Oracle"
    label = "Connaissance des pannes futures"

    def __init__(self, target: float = 0.7, safety: int = 1):
        self.target = target
        self.safety = safety

    def reset(self, sim: FleetSimulation) -> None:
        super().reset(sim)
        self.n_parts = sim.P

    def _hours_to_target(self, obs: Observation) -> np.ndarray:
        hid = obs.hidden
        remaining = np.maximum(self.target * hid.life - hid.unit_age, 0.0)
        # Une heure de marche correspond en moyenne à 1 / duty heures calendaires.
        return np.where(obs.usage, remaining / hid.duty[:, None], remaining)

    def decide(self, obs: Observation) -> Decisions:
        left = self._hours_to_target(obs)
        score = np.where(left <= 24.0, -left, np.nan)
        preventive = schedule_preventive(obs, score)

        orders = {}
        ip = inventory_position(obs)
        for p in range(self.n_parts):
            horizon = self.lead_hours[p] + 7 * 24.0
            need = int(((left <= horizon) & (obs.part_of == p)).sum()) + self.safety
            if ip[p] < need:
                orders[p] = need - int(ip[p])
        return Decisions(orders=orders, preventive=preventive)
