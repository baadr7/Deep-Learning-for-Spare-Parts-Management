"""Politiques P4 et P5 : agents entraînés par apprentissage par renforcement (§ 5.3).

Le modèle est chargé depuis ``models/rl`` ; le codec qui a servi à l'entraînement convertit
l'observation du simulateur en vecteur et l'action de l'agent en décisions.

Lorsque l'agent a été entraîné avec le risque prédit (``use_risk``) ou la dérive du signal
(``use_drift``), voir les métadonnées enregistrées à côté du modèle, la politique reconstruit
à l'identique l'observation de l'entraînement avant de l'encoder : le risque de panne est
recalculé par le réseau de survie à chaque matin, la dérive se lit directement dans
l'observation du simulateur.

``algo`` sélectionne la famille d'algorithme. Les agents publiés sont des MaskablePPO et un DQN ;
``recurrent_ppo`` charge la variante à mémoire (``RecurrentPPO``, politique ``MlpLstmPolicy``) du
§ 5.3, dont l'état récurrent est réinitialisé au début de chaque épisode par ``reset``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spareparts.config import ROOT
from spareparts.policies.base import Policy
from spareparts.sim.engine import Decisions, FleetSimulation, Observation

MODEL_DIR = ROOT / "models" / "rl"
# Le PPO récurrent joue le même niveau N1 que P4 : seule la mémoire de la politique change,
# ce qui isole l'effet de l'observabilité partielle (§ 6, amélioration 13).
LEVEL_OF = {"DQN": "reduced", "P4": "n1", "RPPO": "n1", "P5": "n2"}


def model_path(code: str, scenario: str) -> Path:
    return MODEL_DIR / f"{code.lower()}_{scenario}.zip"


def model_meta(code: str, scenario: str) -> dict:
    """Métadonnées d'entraînement enregistrées à côté du modèle, si elles existent."""
    path = model_path(code, scenario).with_suffix(".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


class LearnedPolicy(Policy):
    """Agent entraîné, joué de façon déterministe ou en tirant ses actions.

    Une politique PPO est une loi sur les actions. Son mode, action la plus probable de chaque
    dimension, peut s'en écarter beaucoup : au niveau N2, « ne rien commander » est l'option la
    plus probable de chaque référence, et le mode ne commande donc jamais. Le choix entre les
    deux usages se fait sur les graines de réglage (``configs/policies``), et le tirage est
    initialisé par la graine de la simulation pour rester reproductible.
    """

    def __init__(
        self,
        path: Path,
        level: str,
        name: str,
        label: str,
        horizon: int = 365,
        deterministic: bool = True,
        scenario: str = "s1_nominal",
        use_risk: bool = False,
        use_drift: bool = False,
        risk_min: float | None = None,
        algo: str = "maskable_ppo",
    ):
        self.path = Path(path)
        self.level = level
        self.name = name
        self.label = label
        self.horizon = horizon
        self.deterministic = deterministic
        self.scenario = scenario
        self.use_risk = use_risk
        self.use_drift = use_drift
        self.risk_min = risk_min
        self.algo = algo
        self._model = None
        self._predictor = None
        # État récurrent (RecurrentPPO uniquement) : réinitialisé à chaque épisode par ``reset``.
        self._state = None
        self._episode_start = np.array([True])

    def _load(self):
        if self._model is None:
            # Le DQN du niveau réduit n'a pas d'actions à masquer, la variante à mémoire a sa
            # propre classe ; les autres agents sont des MaskablePPO.
            if self.algo == "recurrent_ppo":
                from sb3_contrib import RecurrentPPO as algo
            elif self.level == "reduced":
                from stable_baselines3 import DQN as algo
            else:
                from sb3_contrib import MaskablePPO as algo

            self._model = algo.load(self.path, device="cpu")
        return self._model

    def reset(self, sim: FleetSimulation) -> None:
        from spareparts.envs.codecs import Context, make_codec

        super().reset(sim)
        self.sim = sim
        self.codec = make_codec(
            self.level,
            Context.from_sim(
                sim,
                self.horizon,
                use_risk=self.use_risk,
                use_drift=self.use_drift,
                risk_min=self.risk_min,
            ),
        )
        if self.use_risk:
            from spareparts.predict.inference import (
                make_predictor,
                risk_horizon_days,
                window_hours,
            )

            if sim.telemetry_window() is None:
                sim.enable_window(window_hours())
            self._predictor = make_predictor(self.scenario, sim)
            self._risk_days = risk_horizon_days(self.scenario)
        self._load()
        if self.algo == "recurrent_ppo":
            self._state = None
            self._episode_start = np.array([True])
        if not self.deterministic:
            import torch

            torch.manual_seed(sim.seed)

    def decide(self, obs: Observation) -> Decisions:
        if self.use_risk:
            from spareparts.predict.inference import observed_risk

            # Le risque observé à l'entraînement est recalculé ici : l'observation reçue du
            # simulateur ne le porte pas. ``run`` le récupère ensuite pour l'historique.
            obs.risk = observed_risk(self._predictor, self.sim, obs, self._risk_days)
        vector = self.codec.observe(obs)
        if self.algo == "recurrent_ppo":
            # La mémoire du LSTM se propage d'un jour à l'autre ; ``episode_start`` marque le
            # premier jour, où l'état est neuf.
            action, self._state = self._model.predict(
                vector,
                state=self._state,
                episode_start=self._episode_start,
                deterministic=self.deterministic,
            )
            self._episode_start = np.array([False])
        elif self.level == "reduced":
            action, _ = self._model.predict(vector, deterministic=True)
        else:
            action, _ = self._model.predict(
                vector, action_masks=self.codec.masks(obs), deterministic=self.deterministic
            )
        return self.codec.decide(np.asarray(action), obs)
