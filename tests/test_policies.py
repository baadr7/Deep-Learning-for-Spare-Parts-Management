"""Tests du registre des politiques (§ 6) : résolution d'un code, cas dégradés.

Les agents entraînés ne sont pas dans le dépôt (``models/`` est ignoré par Git) : les tests
portent donc sur les politiques qui se construisent sans fichier de modèle — P0, P1, P2,
Oracle — et sur les messages d'erreur des codes invalides.
"""

from __future__ import annotations

import json

import pytest

from spareparts.policies import (
    FixedThresholdPolicy,
    OraclePolicy,
    available_policies,
    make_policy,
    tuned_parameters,
)


def test_make_policy_accepts_lower_case_and_resolves_the_references():
    assert isinstance(make_policy("p0"), FixedThresholdPolicy)
    assert make_policy("P1").name == "P1"
    assert isinstance(make_policy("oracle"), OraclePolicy)
    assert isinstance(make_policy("Oracle"), OraclePolicy)


def test_make_policy_rejects_an_unknown_code():
    with pytest.raises(ValueError, match="politique inconnue"):
        make_policy("P9")


def test_tuned_parameters_returns_none_when_no_file_is_recorded():
    assert tuned_parameters("p8", "s1_nominal") is None
    # P1 a été réglée sur le scénario nominal : ses paramètres doivent être retrouvés.
    tuned = tuned_parameters("p1", "s1_nominal")
    assert tuned and {"s", "S"} <= tuned.keys()


def test_available_policies_always_expose_the_references():
    codes = available_policies("s1_nominal")
    assert codes[:3] == ["P0", "P1", "P2"]
    assert codes[-1] == "Oracle"
    assert len(codes) == len(set(codes))


def test_bdq_metadata_is_read_when_present_and_empty_otherwise(monkeypatch, tmp_path):
    from spareparts import policies

    monkeypatch.setattr(policies, "ROOT", tmp_path)
    assert policies._bdq_meta("s1_nominal") == {}  # aucun fichier latéral : dictionnaire vide

    meta_dir = tmp_path / "models" / "rl"
    meta_dir.mkdir(parents=True)
    (meta_dir / "bdq_s1_nominal.json").write_text(
        json.dumps({"use_risk": True, "risk_min": 0.2}), encoding="utf-8"
    )
    assert policies._bdq_meta("s1_nominal") == {"use_risk": True, "risk_min": 0.2}
