"""Registre des politiques disponibles."""

from __future__ import annotations

from pathlib import Path

import yaml

from spareparts.config import ROOT
from spareparts.policies.base import Policy
from spareparts.policies.classic import ConditionalPolicy, FixedThresholdPolicy
from spareparts.policies.oracle import OraclePolicy

TUNED_DIR = ROOT / "configs" / "policies"


def tuned_parameters(policy: str, scenario: str) -> dict | None:
    """Paramètres optimisés enregistrés pour une politique et un scénario, s'ils existent."""
    path = TUNED_DIR / f"{policy.lower()}_{scenario}.yaml"
    if not path.exists():
        return None
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def make_policy(code: str, scenario: str = "s1_nominal") -> Policy:
    code = code.upper() if code.lower() != "oracle" else "Oracle"
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
    if code == "Oracle":
        return OraclePolicy()
    raise ValueError(f"politique inconnue : {code}")


POLICY_LABELS = {
    "P0": "Seuils fixes réglés à la main",
    "P1": "Seuils fixes optimisés",
    "P2": "Maintenance conditionnelle",
    "Oracle": "Connaissance des pannes futures",
}

__all__ = [
    "POLICY_LABELS",
    "ConditionalPolicy",
    "FixedThresholdPolicy",
    "OraclePolicy",
    "Policy",
    "make_policy",
]
