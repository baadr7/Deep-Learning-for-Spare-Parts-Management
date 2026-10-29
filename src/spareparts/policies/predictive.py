"""P3 : politique prédictive sans apprentissage par renforcement (§ 7.1).

Elle isole l'apport de la prédiction : le réseau de survie estime pour chaque composant la
probabilité de panne dans les jours à venir, et deux règles fixes s'en servent.

- Maintenance : un composant est remplacé lorsque sa probabilité de panne sous
  ``pm_days`` jours dépasse ``pm_threshold``.
- Stock : chaque référence est recomplétée jusqu'à la demande attendue sur le délai
  fournisseur, augmentée d'une marge de sécurité de ``z`` écarts-types (loi de Poisson).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from spareparts.policies.base import Policy, inventory_position, schedule_preventive
from spareparts.sim.engine import Decisions, FleetSimulation, Observation


class PredictivePolicy(Policy):
    name = "P3"
    label = "Prédictive sans apprentissage par renforcement"

    def __init__(
        self,
        model_path: Path,
        arch: str,
        pm_threshold: float = 0.15,
        pm_days: int = 3,
        z: float = 1.65,
    ):
        self.model_path, self.arch = Path(model_path), arch
        self.pm_threshold, self.pm_days, self.z = pm_threshold, pm_days, z

    def reset(self, sim: FleetSimulation) -> None:
        # Imports locaux : predict.dataset dépend lui-même du registre des politiques.
        from spareparts.predict.dataset import HORIZON
        from spareparts.predict.inference import FailurePredictor, window_hours

        super().reset(sim)
        self.horizon = HORIZON
        if sim.telemetry_window() is None:
            sim.enable_window(window_hours())
        self.sim = sim
        self.predictor = FailurePredictor(self.model_path, self.arch, sim)
        self.lead_days = np.maximum(np.rint(self.lead_hours / 24.0).astype(int), 1)

    def decide(self, obs: Observation) -> Decisions:
        cum = self.predictor.cumulative(self.sim.telemetry_window(), obs)  # [M, K, H]
        risk_pm = cum[:, :, self.pm_days - 1]
        score = np.where(risk_pm >= self.pm_threshold, risk_pm, np.nan)
        preventive = schedule_preventive(obs, score, keep=1)

        orders = {}
        ip = inventory_position(obs)
        for p in range(len(self.parts)):
            mask = obs.part_of == p
            days = min(int(self.lead_days[p]), self.horizon)
            expected = float(cum[:, :, days - 1][mask].sum())
            # Au-delà de l'horizon du réseau, la demande est prolongée à rythme constant.
            expected *= self.lead_days[p] / days
            target = math.ceil(expected + self.z * math.sqrt(expected) + 1)
            if ip[p] < target:
                orders[p] = target - int(ip[p])
        return Decisions(orders=orders, preventive=preventive)
