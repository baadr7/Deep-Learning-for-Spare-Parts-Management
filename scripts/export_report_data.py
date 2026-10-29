"""Exporte les résultats en fichiers CSV lus par les graphiques du rapport (docs/rapport/data).

Exemple : uv run python scripts/export_report_data.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from spareparts.config import ROOT

METRICS = ROOT / "reports" / "metrics"
OUT = ROOT / "docs" / "rapport" / "data"
REWARD_SCALE = 150.0
SOBOL_LABELS = {
    "technicians": "Techniciens",
    "pm_factor": "Intervalle de maintenance",
    "stock_factor": "Niveau de stock",
    "lead_time_factor": "Délai fournisseur",
    "failure_rate_factor": "Taux de panne",
}


def write(name: str, header: list[str], rows: list[list]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name} : {len(rows)} lignes")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fr(value: float, decimals: int = 0) -> str:
    """Nombre au format français de LaTeX : virgule décimale, espace fine entre milliers."""
    text = f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")
    return text.replace(" ", "\\,")


def pvalue(p: float | None) -> str:
    if p is None:
        return "réf."
    return "$<$\\,0,001" if p < 0.001 else fr(p, 3)


def write_tex(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{name} : {len(lines)} lignes")


def evaluation_tables(evaluation: dict) -> None:
    """Lignes des tableaux du chapitre d'évaluation, incluses telles quelles par \\input."""
    nominal = evaluation["nominal"]
    pvalues = evaluation.get("wilcoxon_cout", {})
    reference = next((c for c in nominal if c not in pvalues), None)
    rows = []
    for code, s in nominal.items():
        cost = s["cout_total"]
        rows.append(
            f"{code} & {fr(cost['mean'])} & {fr(cost['low'])} à {fr(cost['high'])} "
            f"& {fr(100 * s['trs']['mean'], 2)}~\\% "
            f"& {fr(100 * s['disponibilite']['mean'], 2)}~\\% "
            f"& {fr(100 * s['frequence_ruptures']['mean'], 1)}~\\% "
            f"& {fr(s['attente_piece_moyenne']['mean'], 1)} "
            f"& {pvalue(None if code == reference else pvalues.get(code))} \\\\"
        )
    write_tex("tableau_evaluation.tex", rows)

    cases = evaluation.get("robustesse", {})
    if cases:
        rows = []
        for code in nominal:
            base = nominal[code]["cout_total"]["mean"]
            cells = [fr(base / 1000, 1)]
            for res in cases.values():
                if code in res:
                    cost = res[code]["cout_total"]["mean"]
                    change = 100 * (cost / base - 1)
                    sign = "$+$" if change >= 0 else "$-$"
                    cells.append(f"{fr(cost / 1000, 1)} ({sign}{fr(abs(change))}~\\%)")
                else:
                    cells.append("")
            rows.append(f"{code} & " + " & ".join(cells) + " \\\\")
        write_tex("tableau_robustesse.tex", rows)


def learning_curves() -> None:
    """Coût annuel moyen des épisodes d'entraînement, lu dans les journaux TensorBoard."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    for run in sorted((ROOT / "runs" / "rl").glob("*")):
        acc = EventAccumulator(str(run))
        acc.Reload()
        if "rollout/ep_rew_mean" not in acc.Tags().get("scalars", []):
            continue
        rows = [
            [e.step, round(-e.value * REWARD_SCALE, 1)] for e in acc.Scalars("rollout/ep_rew_mean")
        ]
        write(f"courbe_{run.name}.csv", ["pas", "cout_annuel"], rows)

    # Le BDQ n'écrit pas dans TensorBoard : moyenne glissante sur 20 épisodes, comme ep_rew_mean
    # porte sur les derniers épisodes de PPO.
    episodes = METRICS / "bdq_s1_nominal_episodes.csv"
    if episodes.exists():
        with episodes.open(encoding="utf-8") as fh:
            data = [(int(r["pas"]), float(r["cout_annuel"])) for r in csv.DictReader(fh)]
        rows = [
            [n, round(sum(c for _, c in data[max(0, i - 19) : i + 1]) / min(i + 1, 20), 1)]
            for i, (n, _) in enumerate(data)
        ]
        write("courbe_bdq.csv", ["pas", "cout_annuel"], rows)


def main() -> None:
    evaluation = load(ROOT / "reports" / "evaluation_s1_nominal.json")
    if evaluation:
        rows = []
        for code, s in evaluation["nominal"].items():
            rows.append(
                [
                    code,
                    round(s["cout_total"]["mean"], 1),
                    round(s["cout_total"]["low"], 1),
                    round(s["cout_total"]["high"], 1),
                    round(100 * s["trs"]["mean"], 3),
                    round(100 * s["trs"]["low"], 3),
                    round(100 * s["trs"]["high"], 3),
                    round(100 * s["frequence_ruptures"]["mean"], 2),
                    round(s["pannes"]["mean"], 1),
                    round(s["cout_total"]["mean"] / 1000, 2),
                    round((s["cout_total"]["high"] - s["cout_total"]["mean"]) / 1000, 2),
                ]
            )
        # Étiquettes du nuage coût, TRS : au-dessus du point, sauf si une politique de coût voisin
        # (moins de 1 500 UR d'écart) a un TRS nettement plus élevé ; l'étiquette passe alors
        # au-dessous, pour ne pas chevaucher le point ou la barre de sa voisine.
        rows.sort(key=lambda r: r[1])
        for row in rows:
            crowded = any(
                other is not row and abs(other[1] - row[1]) < 1500 and other[4] > row[4] + 0.05
                for other in rows
            )
            row.append("north" if crowded else "south")
        write(
            "evaluation.csv",
            [
                "politique",
                "cout",
                "cout_bas",
                "cout_haut",
                "trs",
                "trs_bas",
                "trs_haut",
                "ruptures",
                "pannes",
                "cout_k",
                "ic_k",
                "ancre",
            ],
            rows,
        )
        evaluation_tables(evaluation)

    for source, name in (("simulation", "s1_nominal"), ("azure", "azure")):
        pred = load(METRICS / f"prediction_{name}.json")
        if pred:
            rows = [
                [
                    m,
                    round(v["f1"], 3),
                    round(v["pr_auc"], 3),
                    round(v["a_7_jours"]["f1"], 3),
                    round(v["a_7_jours"]["pr_auc"], 3),
                ]
                for m, v in pred.items()
            ]
            write(
                f"prediction_{source}.csv",
                ["modele", "f1_24h", "prauc_24h", "f1_7j", "prauc_7j"],
                rows,
            )

    sobol = load(METRICS / "sobol.json")
    if sobol:
        # Du moins influent au plus influent sur le coût : le graphique place la première ligne
        # en bas et lit les libellés dans le même ordre.
        params = sorted(next(iter(sobol.values())), key=lambda p: sobol["cout_total"][p]["ST"])
        rows = [
            [p, SOBOL_LABELS.get(p, p)] + [round(sobol[o][p]["ST"], 3) for o in sobol]
            for p in params
        ]
        write("sobol.csv", ["parametre", "libelle"] + list(sobol), rows)

    learning_curves()


if __name__ == "__main__":
    main()
