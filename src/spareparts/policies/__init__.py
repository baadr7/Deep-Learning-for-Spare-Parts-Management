"""Registre des politiques disponibles."""

from __future__ import annotations

from pathlib import Path

import yaml

from spareparts.config import ROOT
from spareparts.policies.base import Policy
from spareparts.policies.classic import ConditionalPolicy, FixedThresholdPolicy
from spareparts.policies.learned import LEVEL_OF, LearnedPolicy, model_path
from spareparts.policies.oracle import OraclePolicy

TUNED_DIR = ROOT / "configs" / "policies"

POLICY_LABELS = {
    "P0": "Seuils fixes réglés à la main",
    "P1": "Seuils fixes optimisés",
    "P2": "Maintenance conditionnelle",
    "P3": "Prédictive sans apprentissage par renforcement",
    "P4": "Agent DRL, niveau N1",
    "P5": "Agent DRL, niveau N2",
    "DQN": "DQN, niveau d'action réduit",
    "BDQ": "Branching Dueling Q-Network, niveau N2",
    "Oracle": "Connaissance des pannes futures",
}


def tuned_parameters(policy: str, scenario: str) -> dict | None:
    """Paramètres optimisés enregistrés pour une politique et un scénario, s'ils existent."""
    path = TUNED_DIR / f"{policy.lower()}_{scenario}.yaml"
    if not path.exists():
        return None
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def available_policies(scenario: str = "s1_nominal") -> list[str]:
    """Politiques utilisables : les agents n'apparaissent qu'une fois entraînés."""
    codes = ["P0", "P1", "P2"]
    if _predictor_path(scenario):
        codes.append("P3")
    codes += [c for c in LEVEL_OF if model_path(c, scenario).exists()]
    if _bdq_path(scenario).exists():
        codes.append("BDQ")
    return codes + ["Oracle"]


def _bdq_path(scenario: str) -> Path:
    return ROOT / "models" / "rl" / f"bdq_{scenario}.pt"


def _predictor_path(scenario: str):
    from spareparts.predict.inference import MODEL_DIR, best_architecture

    arch = best_architecture(scenario)
    return (MODEL_DIR / f"{arch}_{scenario}.pt", arch) if arch else None


def make_policy(code: str, scenario: str = "s1_nominal") -> Policy:
    code = "Oracle" if code.lower() == "oracle" else code.upper()
    if code == "P0":
        return FixedThresholdPolicy(name="P0")
    if code == "P1":
        params = tuned_parameters("p1", scenario) or {}
        return FixedThresholdPolicy(
            s=params.get("s"), S=params.get("S"), pm_factor=params.get("pm_factor", 1.0), name="P1"
        )
    if code == "P2":
        params = tuned_parameters("p2", scenario) or {}
        return ConditionalPolicy(**params)
    if code == "P3":
        found = _predictor_path(scenario)
        if not found:
            raise ValueError(
                f"aucun modèle de prédiction pour {scenario} : lancer scripts/train_predictor.py"
            )
        from spareparts.policies.predictive import PredictivePolicy

        return PredictivePolicy(*found, **(tuned_parameters("p3", scenario) or {}))
    if code in LEVEL_OF:
        path = model_path(code, scenario)
        if not path.exists():
            raise ValueError(
                f"agent {code} non entraîné pour {scenario} : lancer scripts/train_rl.py"
            )
        return LearnedPolicy(path, LEVEL_OF[code], name=code, label=POLICY_LABELS[code])
    if code == "BDQ":
        from spareparts.rl.bdq import BDQPolicy, load_bdq

        if not _bdq_path(scenario).exists():
            raise ValueError(
                f"réseau BDQ non entraîné pour {scenario} : lancer scripts/train_bdq.py"
            )
        return BDQPolicy(load_bdq(_bdq_path(scenario)))
    if code == "Oracle":
        return OraclePolicy()
    raise ValueError(f"politique inconnue : {code}")


__all__ = [
    "POLICY_LABELS",
    "ConditionalPolicy",
    "FixedThresholdPolicy",
    "OraclePolicy",
    "Policy",
    "available_policies",
    "make_policy",
]
