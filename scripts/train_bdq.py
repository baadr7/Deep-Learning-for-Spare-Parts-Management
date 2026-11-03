"""Entraîne le Branching Dueling Q-Network au niveau N2 puis l'évalue (§ 5.3).

Exemples :
  uv run python scripts/train_bdq.py --steps 200000
  uv run python scripts/train_bdq.py --risk --drift --curriculum --seeds 0 1 2
  uv run python scripts/train_bdq.py --risk --drift --curriculum --branch-reward --randomize

Avec ``--seeds``, chaque graine est entraînée et enregistrée séparément dans
``models/rl``, puis la meilleure est retenue comme modèle de la politique après évaluation sur les
graines de validation : la variabilité de l'apprentissage est ainsi séparée de la méthode, comme
pour P4 et P5. Le réseau est enregistré dans ``models/rl/bdq_<scénario>.pt`` ; l'évaluation de
scripts/evaluate_policies.py utilise les mêmes graines pour rester comparable aux autres
politiques. Les graines sont entraînées en parallèle (``--workers``) : l'entraînement N2 n'occupe
qu'un cœur, et trois processus indépendants terminent la campagne en un tiers du temps.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from spareparts import tracking
from spareparts.config import ROOT, load_scenario
from spareparts.envs.fleet_env import RANDOMIZATION_RANGES, FleetEnv
from spareparts.evaluation import summarize
from spareparts.rl.bdq import BDQConfig, BDQPolicy, load_bdq, save_bdq, train_bdq
from spareparts.sim import FleetSimulation

VALIDATION_SEEDS = list(range(8000, 8005))


def canonical_path(args) -> Path:
    if args.output:
        return Path(args.output)
    return ROOT / "models" / "rl" / f"bdq_{args.scenario}.pt"


def seed_path(canonical: Path, seed: int, n_seeds: int) -> Path:
    if n_seeds == 1:
        return canonical
    return canonical.with_name(f"{canonical.stem}_s{seed}{canonical.suffix}")


def score_agents(args, paths: list[Path], seeds: list[int]) -> dict[int, float]:
    """Coût annuel moyen de chaque graine sur les graines de validation."""
    cfg = load_scenario(args.scenario)
    scores = {}
    for seed, path in zip(seeds, paths, strict=True):
        policy = BDQPolicy(
            load_bdq(path),
            scenario=args.scenario,
            use_risk=args.risk,
            use_drift=args.drift,
            risk_min=args.risk_min,
        )
        runs = [
            FleetSimulation(cfg, seed=s, log_events=False).run(policy).kpis
            for s in VALIDATION_SEEDS
        ]
        scores[seed] = float(np.mean([r["cout_total"] for r in runs]))
        print(f"graine {seed} : coût de validation {scores[seed]:.0f} UR")
    return scores


def _train_seed(job) -> tuple[int, list[tuple[int, float]]]:
    """Entraîne une graine dans son propre processus et enregistre le réseau.

    L'entraînement au niveau N2 est mono-cœur : le simulateur joue une seule journée à la fois,
    et l'évaluation des 108 dimensions de l'action n'occupe qu'un cœur. Lancer les graines en
    parallèle, chacune dans son environnement, divise donc le temps de la campagne sans rien
    changer à ce que chaque graine apprend : le réseau reste initialisé par sa graine et le
    tirage d'exploration par son propre générateur.
    """
    scenario, steps, warmup, risk, drift, risk_min, branch_reward, randomize, seed, path = job
    # Un fil par processus : sans cela, trois processus réclameraient chacun tous les cœurs pour
    # les quelques multiplications du réseau, sans profit, et se gêneraient mutuellement.
    torch.set_num_threads(1)
    env = FleetEnv(
        scenario=scenario,
        level="n2",
        use_risk=risk,
        use_drift=drift,
        episode_days=warmup or None,
        risk_min=risk_min,
        branch_reward=branch_reward,
        randomize=RANDOMIZATION_RANGES if randomize else None,
    )
    history: list[tuple[int, float]] = []
    # La courbe d'épisodes de la graine est écrite au fil de l'entraînement : les graines
    # tournant dans des processus fils, cette écriture est le seul moyen de suivre leur
    # progression, et elle alimente directement la figure de dispersion du rapport.
    curve = ROOT / "reports" / "metrics" / f"bdq_{scenario}_episodes_s{seed}.csv"
    curve.parent.mkdir(parents=True, exist_ok=True)
    with curve.open("w", encoding="utf-8") as fh:
        fh.write("pas,cout_annuel\n")

        def note(_message: str) -> None:
            step, cost = history[-1]
            fh.write(f"{step},{cost:.1f}\n")
            fh.flush()

        net = train_bdq(
            env,
            BDQConfig(
                total_steps=steps,
                curriculum_days=warmup,
                seed=seed,
                branch_reward=branch_reward,
            ),
            log=note,
            history=history,
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_bdq(net, env.observation_space.shape[0], path)
    return seed, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--eval-replications", type=int, default=10)
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="graines d'entraînement ; trois au moins rendent compte de la variabilité (§ 5.3)",
    )
    parser.add_argument(
        "--risk",
        action="store_true",
        help="observe aussi la probabilité de panne prédite par le réseau de survie",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="observe aussi la dérive normalisée du signal de chaque composant (comme P2)",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="joue d'abord des épisodes courts (90 jours) allongés jusqu'à l'horizon complet",
    )
    parser.add_argument(
        "--risk-min",
        type=float,
        default=None,
        help="seuil de risque en deçà duquel la maintenance préventive est masquée "
        "(masquage guidé par le risque, § 6)",
    )
    parser.add_argument(
        "--branch-reward",
        action="store_true",
        help="entraîne chaque branche du BDQ sur la récompense de sa propre décision (§ 6)",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="tire délais, taux de panne et effectifs à chaque épisode (entraînement "
        "multi-scénarios, § 6)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="graines entraînées en parallèle ; par défaut une par graine, selon les cœurs",
    )
    args = parser.parse_args()

    seeds = args.seeds or [args.seed]
    if args.output and len(seeds) > 1:
        raise SystemExit("--output ne se combine pas avec plusieurs graines")

    warmup = 90 if args.curriculum else 0
    canonical = canonical_path(args)
    params = {
        "scenario": args.scenario,
        "steps": args.steps,
        "level": "n2",
        "seeds": seeds,
        "use_risk": args.risk,
        "use_drift": args.drift,
        "curriculum": args.curriculum,
        "risk_min": args.risk_min,
        "branch_reward": args.branch_reward,
        "randomize": args.randomize,
    }
    with tracking.tracked("bdq", params):
        jobs = [
            (
                args.scenario,
                args.steps,
                warmup,
                args.risk,
                args.drift,
                args.risk_min,
                args.branch_reward,
                args.randomize,
                seed,
                seed_path(canonical, seed, len(seeds)),
            )
            for seed in seeds
        ]
        workers = args.workers or min(len(jobs), os.cpu_count() or 1)
        if len(jobs) > 1 and workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_train_seed, jobs))
        else:
            results = [_train_seed(job) for job in jobs]
        paths = [Path(job[-1]) for job in jobs]
        histories = dict(results)

        if len(seeds) > 1:
            scores = score_agents(args, paths, seeds)
            best = min(scores, key=lambda s: scores[s])
            shutil.copyfile(paths[seeds.index(best)], canonical)
            tracking.log({f"validation_s{seed}": value for seed, value in scores.items()})
            print(f"graine retenue : {best} ; modèle enregistré dans {canonical}")
        else:
            best = seeds[0]

        # Métadonnées lues à l'exécution : elles indiquent à la politique si le risque et la
        # dérive étaient observés, afin qu'elle rejoue exactement l'observation de l'entraînement.
        canonical.with_suffix(".json").write_text(
            json.dumps(
                {
                    "scenario": args.scenario,
                    "level": "n2",
                    "use_risk": args.risk,
                    "use_drift": args.drift,
                    "risk_min": args.risk_min,
                    "branch_reward": args.branch_reward,
                    "randomize": args.randomize,
                    "seeds": seeds,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Évaluation et courbe d'apprentissage de la graine retenue.
        cfg = load_scenario(args.scenario)
        eval_seeds = range(9000, 9000 + args.eval_replications)
        policy = BDQPolicy(
            load_bdq(canonical),
            scenario=args.scenario,
            use_risk=args.risk,
            use_drift=args.drift,
            risk_min=args.risk_min,
        )
        runs = [FleetSimulation(cfg, seed=s, log_events=False).run(policy).kpis for s in eval_seeds]
        summary = summarize(runs)
        keys = ("cout_total", "trs", "frequence_ruptures")
        print({k: round(summary[k]["mean"], 3) for k in keys})
        out = ROOT / "reports" / "metrics"
        out.mkdir(parents=True, exist_ok=True)
        # Les courbes de chaque graine ont été écrites pendant l'entraînement ; on conserve en
        # plus celle de la graine retenue, sous le nom attendu par les autres politiques.
        lines = ["pas,cout_annuel"] + [f"{n},{c:.1f}" for n, c in histories[best]]
        curve = out / f"bdq_{args.scenario}_episodes.csv"
        curve.write_text("\n".join(lines) + "\n", encoding="utf-8")
        metrics_path = out / f"bdq_{args.scenario}.json"
        metrics_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
        )
        tracking.log({f"eval_{k}": v["mean"] for k, v in summary.items()})
        tracking.artifact(metrics_path)

        # Registre MLflow : le réseau retenu rejoint la trace de l'exécution.
        tracking.register("bdq", canonical)

    print(f"modèle enregistré dans {canonical}")


if __name__ == "__main__":
    main()
