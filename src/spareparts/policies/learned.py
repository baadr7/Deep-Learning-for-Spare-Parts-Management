"""Politiques P4 et P5 : agents entraînés par apprentissage par renforcement (§ 5.3).

Le modèle est chargé depuis ``models/rl`` ; le codec qui a servi à l'entraînement convertit
l'observation du simulateur en vecteur et l'action de l'agent en décisions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spareparts.config import ROOT
from spareparts.policies.base import Policy
from spareparts.sim.engine import Decisions, FleetSimulation, Observation

MODEL_DIR = ROOT / "models" / "rl"
LEVEL_OF = {"DQN": "reduced", "P4": "n1", "P5": "n2"}


def model_path(code: str, scenario: str) -> Path:
    return MODEL_DIR / f"{code.lower()}_{scenario}.zip"


class LearnedPolicy(Policy):
    def __init__(self, path: Path, level: str, name: str, label: str, horizon: int = 365):
        self.path = Path(path)
        self.level = level
        self.name = name
        self.label = label
        self.horizon = horizon
        self._model = None

    def _load(self):
        if self._model is None:
            # Le DQN du niveau réduit n'a pas d'actions à masquer.
            if self.level == "reduced":
                from stable_baselines3 import DQN as algo
            else:
                from sb3_contrib import MaskablePPO as algo

            self._model = algo.load(self.path, device="cpu")
        return self._model

    def reset(self, sim: FleetSimulation) -> None:
        from spareparts.envs.codecs import Context, make_codec

        super().reset(sim)
        self.codec = make_codec(self.level, Context.from_sim(sim, self.horizon))
        self._load()

    def decide(self, obs: Observation) -> Decisions:
        vector = self.codec.observe(obs)
        masks = self.codec.masks(obs)
        if self.level == "reduced":
            action, _ = self._model.predict(vector, deterministic=True)
        else:
            action, _ = self._model.predict(vector, action_masks=masks, deterministic=True)
        return self.codec.decide(np.asarray(action), obs)
