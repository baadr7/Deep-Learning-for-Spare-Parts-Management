"""Suivi des expériences avec MLflow (§ 8 du cahier des charges).

MLflow enregistre, pour chaque entraînement, les paramètres, les métriques et les fichiers de
résultats dans la base ``mlflow.db`` : c'est le complément de TensorBoard, qui trace les courbes
pas à pas. Une base SQLite remplace le magasin de fichiers, en maintenance dans MLflow 3. Le
module reste sans effet si MLflow n'est pas installé, de sorte qu'un simple ``uv sync`` suffit à
faire tourner le projet.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from spareparts.config import ROOT

EXPERIMENT = "spareparts"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def _mlflow():
    """Renvoie le module MLflow configuré, ou ``None`` s'il n'est pas installé."""
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


@contextmanager
def tracked(run_name: str, params: dict | None = None):
    """Ouvre une expérience MLflow nommée ; ne fait rien si MLflow est absent."""
    mlflow = _mlflow()
    if mlflow is None:
        yield
        return
    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_scalars(params))
        yield


def log(metrics: dict, step: int | None = None) -> None:
    """Enregistre des métriques scalaires dans l'expérience en cours."""
    mlflow = _mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    mlflow.log_metrics(_scalars(metrics), step=step)


def artifact(path: Path) -> None:
    """Téléverse un fichier de résultats dans l'expérience en cours."""
    mlflow = _mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    mlflow.log_artifact(str(path))


def register(name: str, path: Path) -> None:
    """Ajoute une version du modèle ``path`` au registre MLflow (§ 8 du cahier des charges).

    Le journal MLflow conserve métriques et fichiers par exécution ; le registre y ajoute une
    version numérotée du fichier de modèle, rattachée à l'exécution : les runs (P4, P5, TCN)
    deviennent comparables et promouvoables sans dépendre des fichiers posés sur le disque. Sans
    effet si MLflow est absent ou si aucune exécution n'est ouverte.
    """
    mlflow = _mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    try:
        run_id = mlflow.active_run().info.run_id
        mlflow.log_artifact(str(path), artifact_path="model")
        client = mlflow.MlflowClient()
        try:
            client.create_registered_model(name)
        except Exception:
            pass  # modèle déjà déclaré : on se contente d'ajouter une version
        client.create_model_version(
            name=name, source=f"runs:/{run_id}/model/{path.name}", run_id=run_id
        )
    except Exception as exc:  # pragma: no cover - dépend de la version de MLflow installée
        print(f"registre MLflow indisponible pour {name} : {exc}")


def _scalars(values: dict | None) -> dict:
    """Ne garde que les valeurs que MLflow sait journaliser telles quelles."""
    out = {}
    for key, value in (values or {}).items():
        if isinstance(value, (int, float, str)):
            out[key] = value
        elif isinstance(value, (list, tuple)) and len(value) <= 20:
            out[key] = ", ".join(str(v) for v in value)
    return out
