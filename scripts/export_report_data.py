"""Exporte les résultats en fichiers CSV lus par les graphiques du rapport (docs/rapport/data).

Exemple : uv run python scripts/export_report_data.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from spareparts.config import ROOT
from spareparts.evaluation import paired_effects

METRICS = ROOT / "reports" / "metrics"
OUT = ROOT / "docs" / "rapport" / "data"
REWARD_SCALE = 150.0
# Stable-Baselines3 suffixe le nom d'un journal TensorBoard d'un compteur (``_1``, ``_2``,
# \dots) chaque fois qu'un apprentissage est relancé sous le même nom. On retire ce compteur
# pour que le CSV produit — et donc la référence du rapport — ne dépende que du nom logique de
# l'exécution (<algorithme>_<niveau>_<scénario>_s<graine>).
RUN_COUNTER = re.compile(r"^(.+_s\d+)(?:_\d+)?$")
# Numéro de graine d'un nom logique d'exécution (<algo>_<niveau>_<scénario>_s<graine>).
SEED_SUFFIX = re.compile(r"_s(\d+)(?:_\d+)?$")
# Graines de la campagne définitive : seules celles-ci alimentent les figures d'apprentissage, afin
# qu'aucun essai exploratoire (s77, s99, …) ne vienne fausser une courbe, une moyenne ou une
# enveloppe min-max. Les graines de réglage et de sélection restent hors des figures publiées.
CAMPAGNE_SEEDS = (0, 1, 2)
WEIBULL_CORRIGE = ROOT / "configs" / "calibration" / "azure_weibull_corrige.yaml"
# Ordre des situations de robustesse, identique à celui de scripts/evaluate_policies.py.
ROBUSTESSE_ORDRE = [
    "délai fournisseur doublé",
    "taux de panne + 30 %",
    "un technicien de moins",
    "flotte homogène S0",
]

SOBOL_LABELS = {
    "technicians": "Techniciens",
    "pm_factor": "Intervalle de maintenance",
    "stock_factor": "Niveau de stock",
    "lead_time_factor": "Délai fournisseur",
    "failure_rate_factor": "Taux de panne",
}

# Agents du classement publié (P4 et P5 sont les agents MaskablePPO des niveaux N1 et N2, le DQN
# celui du niveau réduit, le BDQ le Branching Dueling Q-Network du niveau N2). Sert à désigner,
# pour chaque composition de flotte, le meilleur agent au regard du coût.
AGENTS = ("P4", "P5", "DQN", "BDQ")


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
                    # Politique dont le modèle est lié aux familles du scénario nominal : elle
                    # ne s'applique pas à la situation correspondante.
                    cells.append("---")
            rows.append(f"{code} & " + " & ".join(cells) + " \\\\")
        write_tex("tableau_robustesse.tex", rows)

        # Variation de coût par rapport au nominal, pour le graphique de robustesse (figure).
        rows = []
        for code in nominal:
            base = nominal[code]["cout_total"]["mean"]
            cells = []
            for name in ROBUSTESSE_ORDRE:
                res = cases.get(name, {})
                if code in res:
                    cells.append(round(100 * (res[code]["cout_total"]["mean"] / base - 1), 1))
                else:
                    cells.append("nan")
            rows.append([code] + cells)
        write("robustesse.csv", ["politique", "delais", "pannes", "technicien", "s0"], rows)

    robustesse_effects(evaluation)


def robustesse_effects(evaluation: dict) -> None:
    """Variation appariée du coût de chaque politique, avec son IC et sa taille d'effet.

    Le tableau de robustesse donne le coût et sa variation ; celui-ci ajoute l'intervalle de
    confiance de cette variation — les politiques affrontent les mêmes années dans la situation
    nominale et dans la situation dégradée, l'écart se calcule donc graine par graine — et la
    taille d'effet appariée ``d_z``, qui rapporte l'écart moyen à sa propre dispersion. Une
    politique dont le modèle dépend des familles du scénario nominal reste sans valeur en flotte
    S0 ; sa cellule porte alors ``---``, comme dans le tableau des coûts.
    """
    runs = evaluation.get("runs")
    if not runs or "nominal" not in runs:
        return
    effects = paired_effects(runs)
    rows = []
    for code in runs["nominal"]:
        cells = []
        for name in ROBUSTESSE_ORDRE:
            effect = effects.get(name, {}).get(code)
            if not effect:
                cells.append("---")
                continue
            change = 100 * effect["variation"]
            sign = "$+$" if change >= 0 else "$-$"
            interval = (
                f"{fr(100 * effect['variation_low'], 1)} à {fr(100 * effect['variation_high'], 1)}"
            )
            cells.append(
                f"\\makecell{{{sign}{fr(abs(change), 1)} ({interval}) \\\\ "
                f"\\scriptsize $d_z={fr(effect['d_z'], 1)}$}}"
            )
        rows.append(f"{code} & " + " & ".join(cells) + " \\\\")
    write_tex("tableau_robustesse_effets.tex", rows)


def tost_table(evaluation: dict) -> None:
    """Verdicts du test d'équivalence TOST au coût, contre la politique de référence.

    Le test de Wilcoxon dit si une politique diffère de la référence ; le TOST dit si elle lui
    est équivalente à une marge près, ou franchement meilleure. Les deux se lisent ensemble.
    """
    nominal = evaluation["nominal"]
    tost_res = evaluation.get("tost_cout")
    if not tost_res:
        return
    reference = next((c for c in nominal if c not in tost_res), None)
    ref_cost = nominal[reference]["cout_total"]["mean"]
    rows = []
    for code in nominal:
        if code not in tost_res:
            continue
        r = tost_res[code]
        change = 100 * r["ecart_moyen"] / ref_cost
        sign = "$+$" if change >= 0 else "$-$"
        rows.append(
            f"{code} & {fr(r['ecart_moyen'], 0)} & {sign}{fr(abs(change), 1)}~\\% "
            f"& {r['verdict']} \\\\"
        )
    write_tex("tableau_tost.tex", rows)


# Libellés des espaces d'action des agents, pour la colonne du tableau du plancher.
ESPACE_LABELS = {"reduced": "Réduit (36 actions)", "n1": "$N1$ ($6^8$)"}


def signed(value: float, decimals: int = 0) -> str:
    """Nombre signé au format français, avec un signe toujours explicite."""
    sign = "$+$" if value >= 0 else "$-$"
    return f"{sign}{fr(abs(value), decimals)}"


def plancher_table(report: dict) -> None:
    """Plancher de représentation de chaque espace d'action, et décomposition de l'écart à P1.

    Le plancher est le meilleur coût atteignable par une politique constante de l'espace d'action —
    exact pour l'espace réduit, majoré pour N1. Il scinde l'écart de coût de chaque agent à P1 en
    une part d'apprentissage, que davantage d'entraînement peut réduire, et une part de
    représentation, que seul un changement d'espace d'action peut réduire. La dernière ligne
    rappelle la cible du critère O4.
    """
    reference = report["bornes"]["P1"]
    rows = []
    for code, dec in report["decomposition"].items():
        apprentissage = dec["ecart_apprentissage"]
        representation = dec["ecart_representation"]
        rows.append(
            f"{code} & {ESPACE_LABELS.get(dec['espace'], dec['espace'])} "
            f"& {fr(dec['cout_agent'])} & {fr(dec['cout_plancher'])} "
            f"& {signed(apprentissage)} ({signed(100 * apprentissage / reference, 1)}~\\%) "
            f"& {signed(representation)} ({signed(100 * representation / reference, 1)}~\\%) \\\\"
        )
    rows.append(f"P1 & --- & {fr(reference)} & --- & --- & --- \\\\")
    write_tex("tableau_plancher.tex", rows)


def sobol_table(sobol: dict) -> None:
    """Indices de Sobol totaux, du paramètre le plus influent sur le coût au moins influent.

    Le tableau se lit indépendamment de la figure : celle-ci empile, pour chaque paramètre, les
    trois indicateurs ; celui-là donne les mêmes indices en regard, colonne par colonne.
    """
    outputs = list(sobol)
    params = sorted(sobol["cout_total"], key=lambda p: sobol["cout_total"][p]["ST"], reverse=True)
    rows = [
        f"{SOBOL_LABELS.get(p, p)} & "
        + " & ".join(fr(sobol[o][p]["ST"], 3) for o in outputs)
        + " \\\\"
        for p in params
    ]
    write_tex("tableau_sobol.tex", rows)


def composition_label(name: str) -> str:
    """Nom court d'une composition de flotte : « Nominale », « C0 », « C1 », \dots"""
    return "Nominale" if name == "nominal" else f"C{int(name.rsplit('_', 1)[1])}"


def generalisation_table(report: dict) -> None:
    """Stabilité du classement selon la composition de la flotte (item 12 du cahier des charges).

    Une ligne par composition, la nominale servant de référence. Le tau de Kendall mesure la
    concordance de son classement avec celui du scénario nominal ; la colonne « ordre attendu »
    dit si la chaîne ``P2 < P3 < P1 < agents`` y est conservée. Les coûts se limitent à P1 et P2,
    qui encadrent cette chaîne avec P3. Un CSV accompagne le tableau pour la figure : les seules
    compositions tirées, le nominal étant la référence à laquelle elles se comparent.
    """
    rows = []
    for case in report["cases"]:
        means = case["cout_moyen"]
        agents = {c: means[c] for c in AGENTS if c in means}
        best = min(agents, key=agents.get) if agents else None
        best_cell = f"{best} ({fr(agents[best])})" if best else "---"
        part_u = case["part_U"]
        rows.append(
            f"{composition_label(case['nom'])} & {case['techniciens']} "
            f"& {'---' if part_u != part_u else fr(part_u, 2)} "
            f"& {fr(case['kendall_vs_nominal'], 3)} "
            f"& {'oui' if case['ordre_cdc']['conserve'] else 'non'} "
            f"& {fr(means['P1'])} & {fr(means['P2'])} & {best_cell} \\\\"
        )
    write_tex("tableau_generalisation.tex", rows)

    write(
        "generalisation.csv",
        ["composition", "part_u", "tau"],
        [
            [
                composition_label(case["nom"]),
                round(float(case["part_U"]), 2),
                round(float(case["kendall_vs_nominal"]), 3),
            ]
            for case in report["cases"]
            if case["nom"] != "nominal"
        ],
    )


def calibration_curves() -> None:
    """Courbes de fiabilité probabiliste à 7 jours : probabilité prédite contre fréquence observée.

    Un modèle parfaitement calibré place ses points sur la diagonale ; l'écart à celle-ci mesure
    la fiabilité des probabilités annoncées. Un fichier par modèle : les points d'une même courbe
    se suivent, et le graphique les trace sans les relier à ceux d'un autre modèle.
    """
    for source, name in (("simulation", "s1_nominal"), ("azure", "azure")):
        pred = load(METRICS / f"prediction_{name}.json")
        if not pred:
            continue
        for model, metrics in pred.items():
            # Le LightGBM range ses points sous « a_7_jours », les réseaux à la racine.
            points = metrics.get("calibration_7j") or metrics.get("a_7_jours", {}).get(
                "calibration_7j"
            )
            if points:
                write(f"calibration_{source}_{model}.csv", ["proba", "observe", "effectif"], points)


def weibull_curves() -> None:
    """Fiabilité R(t) = exp(-(t/eta)^beta) des quatre composants, lois corrigées du simulateur."""
    import math

    import yaml

    if not WEIBULL_CORRIGE.exists():
        return
    lois = yaml.safe_load(WEIBULL_CORRIGE.read_text(encoding="utf-8"))
    comps = sorted(lois)
    rows = []
    for heures in range(0, 12001, 250):
        row = [heures]
        for comp in comps:
            beta, eta = lois[comp]["beta"], lois[comp]["eta"]
            row.append(round(math.exp(-((heures / eta) ** beta)), 4))
        rows.append(row)
    write("weibull.csv", ["heures"] + comps, rows)


def seed_of(name: str) -> int | None:
    """Numéro de graine d'un nom logique d'exécution, ou ``None`` s'il n'en porte pas."""
    match = SEED_SUFFIX.search(name)
    return int(match.group(1)) if match else None


def tensorboard_runs() -> dict[str, Path]:
    """Journal TensorBoard retenu pour chaque exécution logique.

    Stable-Baselines3 ouvre un nouveau journal à chaque relance sous le même nom, et une relance
    repart du début : les pas des différents journaux se recouvrent, si bien qu'on ne peut pas les
    concaténer. On retient donc l'essai **le plus avancé** — le pas final le plus grand — et, à
    égalité, le plus récent. Ce critère écarte les essais interrompus (un journal d'un seul point,
    par exemple) au profit de la campagne définitive menée jusqu'à 300 000 pas.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    best: dict[str, tuple[int, float, Path]] = {}
    for run in (ROOT / "runs" / "rl").iterdir():
        events = list(run.glob("events.out.tfevents.*")) if run.is_dir() else []
        if not events:
            continue
        acc = EventAccumulator(str(run))
        acc.Reload()
        if "rollout/ep_rew_mean" not in acc.Tags().get("scalars", []):
            continue
        final = acc.Scalars("rollout/ep_rew_mean")[-1].step
        stamp = max(p.stat().st_mtime for p in events)
        name = RUN_COUNTER.sub(r"\1", run.name)
        if name not in best or (final, stamp) > best[name][:2]:
            best[name] = (final, stamp, run)
    return {name: run for name, (_, _, run) in best.items()}


def learning_curves() -> None:
    """Coût annuel moyen des épisodes d'entraînement, lu dans les journaux TensorBoard.

    Un épisode est ramené à une année : avec le curriculum, l'agent joue d'abord des épisodes de
    90 jours, dont le coût brut vaut le quart d'une année et croîtrait mécaniquement jusqu'à
    l'horizon complet. On divise donc le coût de l'épisode par sa durée (``rollout/ep_len_mean``)
    puis on multiplie par 365, ce qui isole la progression de la politique de l'allongement des
    épisodes.

    Une courbe par graine de la campagne définitive (graines ``CAMPAGNE_SEEDS``) : les essais
    exploratoires portent d'autres graines et sont écartés des figures publiées. La dispersion
    ainsi révélée d'un entraînement à l'autre est résumée par ``dispersion_curves`` sous la forme
    d'une moyenne et d'une enveloppe min-max. La courbe « P4 sans normalisation »
    (``courbe_maskable_ppo_n1_sans_normalisation.csv``) n'est plus régénérable : le journal
    TensorBoard de cet essai a été archivé hors de ``runs/rl``. Le CSV correspondant, produit
    pendant l'étude, est conservé tel quel dans ``data/``.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    for name, run in sorted(tensorboard_runs().items()):
        if seed_of(name) not in CAMPAGNE_SEEDS:
            continue
        acc = EventAccumulator(str(run))
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        if "rollout/ep_rew_mean" not in tags:
            continue
        lengths = (
            {e.step: e.value for e in acc.Scalars("rollout/ep_len_mean")}
            if "rollout/ep_len_mean" in tags
            else {}
        )
        rows = []
        for e in acc.Scalars("rollout/ep_rew_mean"):
            days = lengths.get(e.step) or 365.0
            rows.append([e.step, round(-e.value * REWARD_SCALE * 365.0 / days, 1)])
        write(f"courbe_{name}.csv", ["pas", "cout_annuel"], rows)

    # Le BDQ n'écrit pas dans TensorBoard : une courbe par graine. L'écart entre deux numéros de
    # pas consécutifs est la durée de l'épisode en jours, ce qui permet la même remise à l'année
    # que pour PPO, suivie d'un lissage sur 20 épisodes comme ``ep_rew_mean`` pour PPO.
    for path in sorted(METRICS.glob("bdq_*_episodes_s*.csv")):
        seed = int(path.stem.rsplit("_s", 1)[-1])
        if seed not in CAMPAGNE_SEEDS:
            continue
        write(f"courbe_bdq_s{seed}.csv", ["pas", "cout_annuel"], smooth_episodes(path))


def smooth_episodes(path: Path) -> list[list]:
    """Coût annuel de chaque épisode (remis à l'année) lissé sur 20 épisodes consécutifs."""
    with path.open(encoding="utf-8") as fh:
        data = [(int(r["pas"]), float(r["cout_annuel"])) for r in csv.DictReader(fh)]
    annual, previous = [], 0
    for step, cost in data:
        annual.append(cost * 365.0 / max(1, step - previous))
        previous = step
    return [
        [step, round(sum(annual[max(0, i - 19) : i + 1]) / min(i + 1, 20), 1)]
        for i, (step, _) in enumerate(data)
    ]


# Préfixe des courbes par graine de chaque agent, pour le résumé de dispersion.
DISPERSION_AGENTS = {
    "p4": "maskable_ppo_n1_s1_nominal",
    "p5": "maskable_ppo_n2_s1_nominal",
    "dqn": "dqn_reduced_s1_nominal",
    "bdq": "bdq",
}


def dispersion_curves() -> None:
    """Courbe moyenne et enveloppe min-max de chaque agent, sur ses trois graines.

    La campagne définitive entraîne chaque agent sur les graines ``CAMPAGNE_SEEDS`` (0, 1, 2) : la
    dispersion d'un entraînement à l'autre est une information à part entière, au même titre que la
    moyenne. Les graines sont nommées explicitement — et non captées par un motif ``s*`` — pour
    qu'aucun essai exploratoire (s77, s99, …) ne vienne élargir l'enveloppe. Les séries sont d'abord
    rééchantillonnées sur une grille de pas commune, car les épisodes du BDQ ne tombent pas sur les
    mêmes pas que les relevés de PPO et du DQN.
    """
    import numpy as np

    for agent, prefix in DISPERSION_AGENTS.items():
        series = []
        for seed in CAMPAGNE_SEEDS:
            path = OUT / f"courbe_{prefix}_s{seed}.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                rows = [(int(r["pas"]), float(r["cout_annuel"])) for r in csv.DictReader(fh)]
            series.append(([r[0] for r in rows], [r[1] for r in rows]))
        if not series:
            continue
        last = min(steps[-1] for steps, _ in series)
        grid = np.arange(0, last + 1, max(1, last // 200))
        stacked = np.vstack([np.interp(grid, steps, values) for steps, values in series])
        rows = [
            [int(p), round(float(m), 1), round(float(lo), 1), round(float(hi), 1)]
            for p, m, lo, hi in zip(
                grid, stacked.mean(axis=0), stacked.min(axis=0), stacked.max(axis=0), strict=True
            )
        ]
        write(f"courbe_{agent}.csv", ["pas", "cout_moy", "cout_min", "cout_max"], rows)


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
        tost_table(evaluation)

    plancher = load(METRICS / "plancher_representation.json")
    if plancher and plancher.get("decomposition"):
        plancher_table(plancher)

    generalisation = load(METRICS / "generalisation_flotte.json")
    if generalisation:
        generalisation_table(generalisation)

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
        sobol_table(sobol)

    learning_curves()
    dispersion_curves()
    weibull_curves()
    calibration_curves()


if __name__ == "__main__":
    main()
