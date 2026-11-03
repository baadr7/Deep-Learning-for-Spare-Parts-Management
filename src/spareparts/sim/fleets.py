"""Compositions de flotte nominales (note de synthèse, § 6, amélioration 12).

Le classement des politiques est établi sur une seule composition : 100 machines, deux
familles, deux techniciens. Pour le généraliser, il faut tirer plusieurs compositions
nominales **à nombre de machines constant** — proportions des familles (modèles de machine)
autour du nominal et effectif technique borné — puis rejouer les campagnes sur chacune.

Ce module fournit la brique manquante : le tirage reproductible des compositions et leur
application comme surcharge du scénario de base. Il ne lance aucune campagne ; il prépare
les entrées que ``scripts/tune_p3.py``, ``scripts/optimize_p1.py`` et
``scripts/train_rl.py --randomize`` consomment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from spareparts.config import ROOT, SCENARIO_DIR, ScenarioConfig, load_scenario

DEFAULT_OUT_DIR = ROOT / "configs" / "fleets"


def _raw_scenario(base: str | Path) -> dict:
    """Contenu brut du scénario de base, pour en dériver des surcharges."""
    path = Path(base)
    if not path.suffix:
        path = SCENARIO_DIR / f"{base}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _base_id(base: str | Path) -> str:
    path = Path(base)
    return path.stem if path.suffix else str(base)


@dataclass(frozen=True)
class FleetComposition:
    """Une composition nominale : proportions des familles et effectif technique."""

    index: int
    seed: int
    technicians: int
    family_share: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"composition_{self.index:02d}"


def family_counts(family_share: dict[str, float], n_machines: int) -> dict[str, int]:
    """Répartit ``n_machines`` entre les familles, exactement comme le simulateur.

    Reproduit ``FleetSimulation._setup_fleet`` : les familles de part strictement positive
    reçoivent ``floor(part / somme * n)`` machines, le reliquat allant à la plus grande part.
    """
    codes = [c for c, share in family_share.items() if share > 0]
    if not codes:
        raise ValueError("aucune famille de part strictement positive")
    shares = np.array([family_share[c] for c in codes], dtype=float)
    counts = np.floor(shares / shares.sum() * n_machines).astype(int)
    counts[int(np.argmax(shares))] += n_machines - int(counts.sum())
    return {c: int(n) for c, n in zip(codes, counts, strict=True)}


def sample_compositions(
    base: str | Path,
    n: int,
    seed: int,
    *,
    min_share: float = 0.1,
    technicians: tuple[int, int] = (1, 3),
    concentration: float = 2.0,
) -> list[FleetComposition]:
    """Tire ``n`` compositions à nombre de machines constant autour du scénario de base.

    Les proportions suivent une loi de Dirichlet centrée sur les parts nominales, puis sont
    mélangées à la loi uniforme ``alpha = min_share`` fois le nombre de familles : chaque
    famille conserve donc au moins ``min_share`` et aucune ne disparaît (ce qui épargne le
    cas des politiques liées à la famille). L'effectif technique est tiré uniformément dans
    ``technicians`` (bornes incluses). Le tirage est **déterministe** : une même graine
    redonne les mêmes compositions.
    """
    raw = _raw_scenario(base)
    families = list(raw["families"])
    k = len(families)
    if n <= 0:
        raise ValueError("n doit être strictement positif")
    if not 0.0 <= min_share < 1.0 / k:
        raise ValueError(
            f"min_share doit rester sous 1/{k} = {1 / k:.3f} pour que chaque famille existe"
        )
    if technicians[0] < 1 or technicians[1] < technicians[0]:
        raise ValueError("bornes de techniciens invalides")

    nominal = np.array(
        [raw["fleet"]["family_share"].get(f, 0.0) for f in families], dtype=float
    )
    if nominal.sum() <= 0:
        nominal = np.ones(k)
    nominal /= nominal.sum()

    rng = np.random.default_rng(seed)
    alpha = min_share * k
    out: list[FleetComposition] = []
    for i in range(n):
        draw = rng.dirichlet(nominal * concentration)
        share = (1.0 - alpha) * draw + alpha / k
        share /= share.sum()
        tech = int(rng.integers(technicians[0], technicians[1] + 1))
        out.append(
            FleetComposition(
                index=i,
                seed=int(rng.integers(0, 2**31 - 1)),
                technicians=tech,
                family_share={f: float(s) for f, s in zip(families, share, strict=True)},
            )
        )
    return out


def composition_overrides(
    comp: FleetComposition, base: str | Path
) -> dict:
    """Surcharges ``load_scenario`` appliquant la composition au scénario de base.

    ``load_scenario`` met à jour les clés de premier niveau : la flotte entière est donc
    remplacée, sa part de familles portant la composition.
    """
    raw = _raw_scenario(base)
    fleet = dict(raw.get("fleet", {}))
    fleet["family_share"] = dict(comp.family_share)
    return {"fleet": fleet, "technicians": comp.technicians}


def apply_composition(base: str | Path, comp: FleetComposition) -> ScenarioConfig:
    """Scénario de base rejoué avec la composition ``comp`` appliquée."""
    return load_scenario(base, **composition_overrides(comp, base))


def write_compositions(
    comps: list[FleetComposition], base: str | Path, out_dir: str | Path = DEFAULT_OUT_DIR
) -> list[Path]:
    """Écrit chaque composition en YAML, sous ``out_dir`` (hors ``configs/scenarios``).

    Les fichiers restent hors de ``configs/scenarios`` pour que ``list_scenarios`` — et donc
    l'application — ne les présente pas comme des scénarios autonomes : une composition est
    une **surcharge** du scénario de base, pas un scénario complet.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    base_id = _base_id(base)
    paths: list[Path] = []
    for comp in comps:
        payload = {
            "name": f"{base_id} / {comp.name}",
            "base": base_id,
            "seed": comp.seed,
            "technicians": comp.technicians,
            "family_share": comp.family_share,
        }
        path = out_path / f"{base_id}_{comp.name}.yaml"
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        paths.append(path)
    return paths
