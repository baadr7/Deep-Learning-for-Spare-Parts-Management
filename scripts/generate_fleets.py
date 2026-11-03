"""Génère des compositions de flotte nominales (note de synthèse, § 6, amélioration 12).

Tire N compositions à nombre de machines constant — proportions des familles (modèles de
machine) autour du nominal et effectif technique borné — puis écrit une surcharge par
composition. Chaque surcharge alimente ensuite une campagne de réglage (P1, P3) et
d'entraînement (``--randomize``), dont on mesure la stabilité du classement (corrélation de
rang de Kendall, conservation de l'ordre attendu P2 < P3 < P1 < agents).

Exemple : uv run python scripts/generate_fleets.py --base s1_nominal --n 5 --seed 0 --smoke
Résultat : configs/fleets/<base>_composition_*.yaml (compositions, réutilisables en surcharge)
"""

from __future__ import annotations

import argparse

from spareparts.config import load_scenario
from spareparts.policies import make_policy
from spareparts.sim import FleetSimulation
from spareparts.sim.fleets import (
    DEFAULT_OUT_DIR,
    apply_composition,
    family_counts,
    sample_compositions,
    write_compositions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="s1_nominal", help="scénario nominal de référence")
    parser.add_argument("--n", type=int, default=5, help="nombre de compositions à tirer")
    parser.add_argument("--seed", type=int, default=0, help="graine du tirage")
    parser.add_argument("--min-share", type=float, default=0.1, help="part plancher par famille")
    parser.add_argument("--tech-min", type=int, default=1, help="effectif technique minimal")
    parser.add_argument("--tech-max", type=int, default=3, help="effectif technique maximal")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="dossier des compositions")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="exécute 30 jours de P1 par composition pour valider le chargement",
    )
    args = parser.parse_args()

    base_cfg = load_scenario(args.base)
    n_machines = base_cfg.fleet.n_machines
    families = sorted(base_cfg.families)
    print(f"Base {args.base} — {n_machines} machines, familles {families}")

    comps = sample_compositions(
        args.base,
        args.n,
        args.seed,
        min_share=args.min_share,
        technicians=(args.tech_min, args.tech_max),
    )

    for comp in comps:
        counts = family_counts(comp.family_share, n_machines)
        parts = "  ".join(f"{f}={comp.family_share[f]:.2f} ({counts[f]})" for f in families)
        line = f"  {comp.name} — techniciens {comp.technicians} — {parts}"
        if args.smoke:
            cfg = apply_composition(args.base, comp)
            kpis = FleetSimulation(cfg, seed=comp.seed).run(make_policy("P1"), days=30).kpis
            line += f"  [coût 30 j {kpis['cout_total']:.0f} UR]"
        print(line)

    paths = write_compositions(comps, args.base, args.out)
    print(f"{len(paths)} compositions écrites dans {args.out}")


if __name__ == "__main__":
    main()
