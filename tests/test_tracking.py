"""Tests du suivi d'expériences MLflow (§ 8 du cahier des charges).

MLflow est simulé par un objet factice : les tests vérifient ce que le module demande au
journal (paramètres, métriques, fichiers, versions de modèle) sans ouvrir de vraie base
``mlflow.db``. Le chemin « MLflow absent » est éprouvé en masquant le module, ce qui doit
laisser toutes les fonctions sans effet — un simple ``uv sync`` suffit alors à faire tourner
le projet.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from spareparts import tracking


class _FakeClient:
    """Registre minimal : refuse un modèle déjà déclaré, comme l'API MLflow."""

    def __init__(self) -> None:
        self.registered: list[str] = []
        self.versions: list[tuple[str, str, str]] = []

    def create_registered_model(self, name: str) -> None:
        if name in self.registered:
            raise RuntimeError("modèle déjà déclaré")
        self.registered.append(name)

    def create_model_version(self, name: str, source: str, run_id: str) -> None:
        self.versions.append((name, source, run_id))


class _FakeMlflow:
    """Journal en mémoire reproduisant la surface de MLflow utilisée par le module."""

    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment: str | None = None
        self.params: dict = {}
        self.metrics: list[tuple[dict, int | None]] = []
        self.artifacts: list[tuple[str, str | None]] = []
        self.ended = 0
        self.active: object | None = None
        self.client = _FakeClient()

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def active_run(self):
        return self.active

    def end_run(self) -> None:
        self.ended += 1
        self.active = None

    @contextmanager
    def start_run(self, run_name: str | None = None):
        previous, self.active = self.active, SimpleNamespace(
            info=SimpleNamespace(run_id="run-abc"), run_name=run_name
        )
        try:
            yield self.active
        finally:
            self.active = previous

    def log_params(self, params: dict) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self.metrics.append((dict(metrics), step))

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((path, artifact_path))

    def MlflowClient(self) -> _FakeClient:
        return self.client


@pytest.fixture
def fake(monkeypatch):
    module = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return module


def test_scalars_keeps_only_loggable_values():
    out = tracking._scalars(
        {
            "entier": 3,
            "reel": 1.5,
            "texte": "P4",
            "vecteur": [1, 2, 3],
            "trop_long": list(range(21)),  # au-delà de 20 valeurs : ignoré
            "imbrique": {"a": 1},  # MLflow ne journalise pas les dictionnaires
        }
    )
    assert out == {"entier": 3, "reel": 1.5, "texte": "P4", "vecteur": "1, 2, 3"}


def test_scalars_of_an_absent_mapping_is_empty():
    assert tracking._scalars(None) == {}


def test_tracking_is_inert_without_mlflow(monkeypatch):
    # Masquer le module force l'import à échouer, comme sur un poste sans l'extra « ml ».
    monkeypatch.setitem(sys.modules, "mlflow", None)
    assert tracking._mlflow() is None
    with tracking.tracked("course", {"jours": 365}):
        tracking.log({"cout": 1.0}, step=0)
        tracking.artifact(Path("absent.json"))
        tracking.register("P4", Path("absent.zip"))  # aucune exception ne doit remonter


def test_helpers_do_nothing_outside_an_active_run(fake):
    assert fake.active_run() is None
    tracking.log({"cout": 1.0}, step=0)
    tracking.artifact(Path("x.json"))
    tracking.register("P4", Path("x.zip"))
    assert fake.metrics == [] and fake.artifacts == [] and fake.client.versions == []


def test_tracked_records_params_metrics_artifacts_and_model_version(fake, tmp_path):
    artifact_file = tmp_path / "resultats.json"
    artifact_file.write_text("{}", encoding="utf-8")
    model_file = tmp_path / "agent.zip"
    model_file.write_bytes(b"modele")

    # Une exécution laissée ouverte est refermée avant d'ouvrir la nouvelle.
    fake.active = SimpleNamespace(info=SimpleNamespace(run_id="precedente"))
    with tracking.tracked("course", {"jours": 365, "graines": [1, 2]}):
        tracking.log({"cout": 12.5}, step=7)
        tracking.artifact(artifact_file)
        tracking.register("P4", model_file)

    assert fake.tracking_uri == tracking.TRACKING_URI
    assert fake.experiment == tracking.EXPERIMENT
    assert fake.ended == 1
    assert fake.params == {"jours": 365, "graines": "1, 2"}
    assert fake.metrics == [({"cout": 12.5}, 7)]
    assert (str(artifact_file), None) in fake.artifacts
    assert (str(model_file), "model") in fake.artifacts
    assert fake.client.registered == ["P4"]
    assert fake.client.versions == [("P4", "runs:/run-abc/model/agent.zip", "run-abc")]


def test_register_adds_a_version_to_an_already_declared_model(fake, tmp_path):
    model_file = tmp_path / "agent.zip"
    model_file.write_bytes(b"modele")
    with tracking.tracked("course"):
        tracking.register("P4", model_file)
        tracking.register("P4", model_file)  # le modèle existe déjà : seule une version s'ajoute
    assert fake.client.registered == ["P4"]
    assert len(fake.client.versions) == 2
