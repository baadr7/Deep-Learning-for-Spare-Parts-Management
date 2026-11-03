"""Garde-fous du rapport : les sources LaTeX et les données qu'elles lisent restent cohérentes.

Ces tests ne compilent pas le rapport ; ils vérifient ce qui le rendrait faux ou non compilable :
tout fichier référencé existe, chaque tableau produit reste inclus et décrit les mêmes politiques,
et l'export des courbes d'apprentissage restitue les conventions du rapport.
"""

from __future__ import annotations

import collections
import importlib.util
import re

import pytest

from spareparts.config import ROOT

RAPPORT = ROOT / "docs" / "rapport"
DATA = RAPPORT / "data"
# Références à un fichier de données ou à une figure, quelle que soit la commande qui les porte
# (``\lignes``, ``\includegraphics``, l'argument d'un ``table`` de pgfplots). L'extension les
# distingue des chemins cités en prose, comme \texttt{data/raw}, qui ne sont pas lus par LaTeX.
REFERENCE = re.compile(r"\{((?:data|figures)/[\w./-]+\.\w{2,4})\}")
INCLUDE = re.compile(r"\\input\{([^}]+)\}")
# Vérification statique de la cohérence LaTeX : labels, références croisées et environnements. Le
# commentaire (un ``%`` non échappé jusqu'à la fin de ligne) est retiré avant analyse, sinon un
# ``\ref`` cité en commentaire serait pris pour une référence pendante.
COMMENT = re.compile(r"(?<!\\)%.*")
LABEL = re.compile(r"\\label\{([^}]+)\}")
CROSSREF = re.compile(r"\\(?:ref|eqref|autoref|pageref)\{([^}]+)\}")
BEGIN = re.compile(r"\\begin\{([^}]+)\}")
END = re.compile(r"\\end\{([^}]+)\}")


def _stripped_sources() -> dict[str, str]:
    """Sources LaTeX du rapport, commentaires retirés, indexées par nom de fichier."""
    return {
        tex.name: COMMENT.sub("", tex.read_text(encoding="utf-8"))
        for tex in sorted(RAPPORT.glob("*.tex"))
    }


def _load_export():
    spec = importlib.util.spec_from_file_location(
        "export_report_data", ROOT / "scripts" / "export_report_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = _load_export()


def _first_column(path):
    """Codes de politique de la première colonne d'un tableau généré."""
    return {
        line.split("&", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_every_file_referenced_by_the_report_exists():
    # Un \lignes ou un \includegraphics pointant vers un fichier absent arrête la compilation ;
    # un tableau lisant un CSV absent produit une figure vide sans le dire.
    missing = []
    for tex in sorted(RAPPORT.glob("*.tex")):
        text = tex.read_text(encoding="utf-8")
        for relative in REFERENCE.findall(text):
            if not (RAPPORT / relative).exists():
                missing.append(f"{tex.name} → {relative}")
        for name in INCLUDE.findall(text):
            target = name if name.endswith(".tex") else f"{name}.tex"
            if not (RAPPORT / target).exists():
                missing.append(f"{tex.name} → {target}")
    assert not missing, "fichiers référencés par le rapport et absents : " + ", ".join(missing)


def test_every_generated_table_is_included_in_the_report():
    # Les tableaux du chapitre d'évaluation et l'analyse de sensibilité sont le corps du rapport :
    # les produire sans les inclure laisserait le lecteur devant une affirmation non étayée.
    text = "\n".join(tex.read_text(encoding="utf-8") for tex in RAPPORT.glob("*.tex"))
    for name in (
        "tableau_evaluation.tex",
        "tableau_robustesse.tex",
        "tableau_robustesse_effets.tex",
        "tableau_tost.tex",
        "tableau_sobol.tex",
        "tableau_generalisation.tex",
        "tableau_plancher.tex",
    ):
        assert f"data/{name}" in text, f"{name} produit mais non inclus dans le rapport"


def test_tost_table_lists_every_policy_except_the_reference():
    evaluated = _first_column(DATA / "tableau_evaluation.tex")
    equivalent = _first_column(DATA / "tableau_tost.tex")
    assert equivalent < evaluated
    # Le TOST compare chaque politique à P1 : la référence ne se compare pas à elle-même.
    assert equivalent == evaluated - {"P1"}


def test_smooth_episodes_annualises_the_episode_cost(tmp_path):
    path = tmp_path / "bdq_s1_nominal_episodes_s0.csv"
    path.write_text("pas,cout_annuel\n90,25\n180,50\n270,75\n", encoding="utf-8")
    rows = export.smooth_episodes(path)
    assert [step for step, _ in rows] == [90, 180, 270]
    # Un épisode de 90 jours : 25 UR d'épisode valent 25 × 365 / 90 UR par an.
    first, second, third = (value for _, value in rows)
    assert first == pytest.approx(25 * 365 / 90, abs=0.05)
    assert second == pytest.approx((25 + 50) * 365 / 90 / 2, abs=0.05)
    assert third == pytest.approx((25 + 50 + 75) * 365 / 90 / 3, abs=0.05)


def test_smooth_episodes_smooths_on_the_last_twenty_episodes(tmp_path):
    path = tmp_path / "bdq_s1_nominal_episodes_s0.csv"
    path.write_text(
        "pas,cout_annuel\n" + "\n".join(f"{100 * k},{k}" for k in range(1, 26)) + "\n",
        encoding="utf-8",
    )
    values = [value for _, value in export.smooth_episodes(path)]
    assert len(values) == 25
    # Aux vingt derniers épisodes, la fenêtre de lissage est pleine ; elle ne l'est pas encore au
    # troisième, où la moyenne ne porte que sur les épisodes déjà joués.
    assert values[-1] == pytest.approx(sum(365 * k / 100 for k in range(6, 26)) / 20, abs=0.05)
    assert values[2] == pytest.approx((1 + 2 + 3) * 365 / 100 / 3, abs=0.05)


def test_generalisation_table_lists_every_composition(tmp_path, monkeypatch):
    # Une ligne par composition, la nominale en tête ; la part de famille n'a pas de sens pour
    # elle, et le meilleur agent se lit parmi les seuls agents du classement.
    report = {
        "cases": [
            {
                "nom": "nominal",
                "techniciens": 2,
                "part_U": float("nan"),
                "cout_moyen": {"P1": 52_136.0, "P2": 47_135.0, "P4": 52_799.0, "DQN": 54_932.0},
                "kendall_vs_nominal": 1.0,
                "ordre_cdc": {"conserve": True},
            },
            {
                "nom": "composition_00",
                "techniciens": 1,
                "part_U": 0.42,
                "cout_moyen": {"P1": 52_642.0, "P2": 53_061.0, "P4": 52_954.0, "DQN": 54_446.0},
                "kendall_vs_nominal": 0.667,
                "ordre_cdc": {"conserve": False},
            },
        ]
    }
    monkeypatch.setattr(export, "OUT", tmp_path)
    export.generalisation_table(report)

    rows = (tmp_path / "tableau_generalisation.tex").read_text(encoding="utf-8").splitlines()
    attendu_nominal = "Nominale & 2 & --- & 1,000 & oui & 52\\,136 & 47\\,135 & P4 (52\\,799)"
    attendu_c0 = "C0 & 1 & 0,42 & 0,667 & non & 52\\,642 & 53\\,061 & P4 (52\\,954)"
    assert rows[0].startswith(attendu_nominal)
    assert rows[1].startswith(attendu_c0)

    # Le CSV de la figure ne porte que les compositions tirées : le nominal est la référence.
    lines = (tmp_path / "generalisation.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "composition,part_u,tau"
    assert [line.split(",")[0] for line in lines[1:]] == ["C0"]


def test_learning_curves_annualise_short_curriculum_episodes(tmp_path, monkeypatch):
    pytest.importorskip("tensorboard")
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    run = tmp_path / "runs" / "maskable_ppo_n1_s1_nominal_s1"
    run.mkdir(parents=True)
    writer = EventFileWriter(str(run))
    # Sous curriculum, l'agent joue d'abord des épisodes courts : leur coût brut vaut une fraction
    # d'année et doit être ramené à l'année, sinon la courbe monterait avec l'horizon des épisodes.
    for step, (reward, days) in enumerate([(-0.5, 365.0), (-0.5, 90.0)]):
        writer.add_event(
            Event(
                wall_time=float(step),
                step=step,
                summary=Summary(
                    value=[
                        Summary.Value(tag="rollout/ep_rew_mean", simple_value=reward),
                        Summary.Value(tag="rollout/ep_len_mean", simple_value=days),
                    ]
                ),
            )
        )
    writer.flush()
    writer.close()

    monkeypatch.setattr(export, "tensorboard_runs", lambda: {run.name: run})
    monkeypatch.setattr(export, "OUT", tmp_path / "data")
    monkeypatch.setattr(export, "METRICS", tmp_path / "metrics")
    export.learning_curves()

    lines = (tmp_path / "data" / f"courbe_{run.name}.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "pas,cout_annuel"
    # 0,5 × 150 UR par pas annulé sur un épisode de 365 jours ; le même épisode de 90 jours,
    # ramené à l'année, vaut 365 / 90 fois davantage.
    assert lines[1].split(",")[1] == "75.0"
    assert lines[2].split(",")[1] == "304.2"


def test_no_two_labels_share_the_same_name():
    # Un label répété ne fait pas échouer la compilation mais fait pointer \ref vers la mauvaise
    # cible ; le doublon doit être détecté ici plutôt que lu par hasard dans le PDF.
    labels = [name for text in _stripped_sources().values() for name in LABEL.findall(text)]
    duplicated = sorted({name for name in labels if labels.count(name) > 1})
    assert not duplicated, "labels dupliqués dans le rapport : " + ", ".join(duplicated)


def test_every_reference_points_to_an_existing_label():
    # Un \ref sans label compile en silence et imprime « ?? » : la référence pendante, comme celle
    # d'une annexe ajoutée sans son label, doit être signalée.
    sources = _stripped_sources()
    labels = {name for text in sources.values() for name in LABEL.findall(text)}
    dangling = sorted(
        {name for text in sources.values() for name in CROSSREF.findall(text) if name not in labels}
    )
    assert not dangling, "références sans label dans le rapport : " + ", ".join(dangling)


def test_environments_open_and_close_in_balance():
    # Un \begin non refermé arrête la compilation : mieux vaut le voir ici que sur le job LaTeX.
    for name, text in _stripped_sources().items():
        assert collections.Counter(BEGIN.findall(text)) == collections.Counter(END.findall(text)), (
            f"environnements \\begin/\\end déséquilibrés dans {name}"
        )

