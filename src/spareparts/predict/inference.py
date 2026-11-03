"""Utilisation d'un réseau de prédiction entraîné pendant une simulation (§ 5.2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spareparts.config import ROOT
from spareparts.predict.dataset import HORIZON, WINDOW
from spareparts.sim.engine import FleetSimulation, Observation

MODEL_DIR = ROOT / "models" / "predict"
METRICS_DIR = ROOT / "reports" / "metrics"


def best_architecture(scenario: str) -> str | None:
    """Réseau qui classe le mieux les pannes à 7 jours (aire sous la courbe précision-rappel).

    La politique P3 décide sur un risque à quelques jours : le classement à 7 jours compte plus
    que la détection à 24 heures.
    """
    trained = [p.stem.split("_")[0] for p in MODEL_DIR.glob(f"*_{scenario}.pt")]
    if not trained:
        return None
    metrics_file = METRICS_DIR / f"prediction_{scenario}.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))

        def score(arch: str) -> float:
            m = metrics.get(arch, {})
            return m.get("a_7_jours", {}).get("pr_auc", m.get("pr_auc", -1.0))

        return max(trained, key=score)
    return trained[0]


class FailurePredictor:
    """Probabilités cumulées de panne de chaque composant sur l'horizon de prédiction."""

    def __init__(self, path: Path, arch: str, sim: FleetSimulation):
        import torch

        from spareparts.predict.models import ARCHITECTURES

        self.n_families = len(sim.family_codes)
        static_size = self.n_families + 1 + sim.K
        self.model = ARCHITECTURES[arch](5, static_size, sim.K, HORIZON)
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()
        self.family = sim.family
        self.shifts = sim.shifts
        self.arch = arch

    def static(self, obs: Observation) -> np.ndarray:
        fam = np.eye(self.n_families, dtype=np.float32)[self.family]
        ratio = np.clip(obs.age_unit / obs.pm_interval, 0, 3).astype(np.float32)
        return np.concatenate([fam, (self.shifts / 3.0)[:, None].astype(np.float32), ratio], axis=1)

    def cumulative(self, window: np.ndarray, obs: Observation) -> np.ndarray:
        """[M, K, HORIZON] : probabilité de panne avant la fin de chaque jour à venir."""
        import torch

        with torch.no_grad():
            logits = self.model(torch.from_numpy(window), torch.from_numpy(self.static(obs)))
            log_surv = torch.cumsum(torch.nn.functional.logsigmoid(-logits), dim=-1)
        return (1.0 - torch.exp(log_surv)).numpy()


def make_predictor(scenario: str, sim: FleetSimulation) -> FailurePredictor:
    """Charge le meilleur réseau de prédiction du scénario pour une simulation."""
    arch = best_architecture(scenario)
    if not arch:
        raise FileNotFoundError(
            f"aucun modèle de prédiction pour {scenario} : lancer scripts/train_predictor.py"
        )
    return FailurePredictor(MODEL_DIR / f"{arch}_{scenario}.pt", arch, sim)


def risk_horizon_days(scenario: str) -> int:
    """Horizon du risque observé par les agents, aligné sur la politique P3 (3 jours par défaut)."""
    from spareparts.policies import tuned_parameters

    return int((tuned_parameters("p3", scenario) or {}).get("pm_days", 3))


def observed_risk(
    predictor: FailurePredictor, sim: FleetSimulation, obs: Observation, days: int
) -> np.ndarray:
    """Probabilité cumulée de panne à ``days`` jours de chaque composant : [M, K].

    C'est exactement la grandeur qui déclenche la maintenance de la politique P3 : les agents de
    l'apprentissage par renforcement l'observent pour arbitrer entre prédiction et adaptation.
    """
    cum = predictor.cumulative(sim.telemetry_window(), obs)
    return cum[:, :, min(days, cum.shape[-1]) - 1]


def window_hours() -> int:
    return WINDOW
