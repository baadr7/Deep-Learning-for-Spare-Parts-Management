"""Entraîne un agent d'apprentissage par renforcement sur le simulateur (§ 5.3).

Exemples :
  uv run python scripts/train_rl.py --level n1 --timesteps 300000
  uv run python scripts/train_rl.py --level n2 --algo maskable_ppo --envs 8
  uv run python scripts/train_rl.py --level reduced --algo dqn
  uv run python scripts/train_rl.py --level n1 --seeds 0 1 2
  uv run python scripts/train_rl.py --level n1 --risk --drift --curriculum
  uv run python scripts/train_rl.py --level n2 --risk --drift --curriculum --randomize

Le § 5.3 demande au moins trois graines par configuration. Avec ``--seeds``, chaque graine est
entraînée et enregistrée séparément dans ``models/rl``, puis la meilleure est retenue comme modèle
de la politique après évaluation sur les graines de validation, ce qui rend compte de la
variabilité de l'apprentissage. ``--workers`` entraîne ces graines en parallèle : le simulateur,
et non le réseau, est le goulot d'étranglement, si bien que trois graines menées de front
terminent la campagne en un tiers du temps sans rien changer à ce que chacune apprend (même
raisonnement que pour le BDQ, ``scripts/train_bdq.py``). Le modèle MaskablePPO est enregistré dans
models/rl/p4_<scénario>.zip (niveau n1) ou p5_<scénario>.zip (niveau n2), le DQN dans
dqn_<scénario>.zip, la variante à mémoire dans rppo_<scénario>.zip (§ 6, amélioration 13) ;
les courbes d'apprentissage vont dans runs/rl (TensorBoard).
"""

from __future__ import annotations

import argparse
import json
import shutil
from functools import partial
from pathlib import Path

import numpy as np

from spareparts import tracking
from spareparts.config import ROOT, load_scenario
from spareparts.envs.fleet_env import FleetEnv
from spareparts.policies.learned import MODEL_DIR
from spareparts.sim import FleetSimulation

VALIDATION_SEEDS = list(range(8000, 8005))


def make_env(
    scenario: str,
    level: str,
    horizon: int,
    rank: int,
    use_risk: bool = False,
    use_drift: bool = False,
    episode_days: int | None = None,
    risk_min: float | None = None,
    randomize: bool = False,
):
    from stable_baselines3.common.monitor import Monitor

    from spareparts.envs.fleet_env import RANDOMIZATION_RANGES

    env = FleetEnv(
        scenario=scenario,
        level=level,
        horizon_days=horizon,
        use_risk=use_risk,
        use_drift=use_drift,
        episode_days=episode_days,
        risk_min=risk_min,
        randomize=RANDOMIZATION_RANGES if randomize else None,
    )
    env.reset(seed=rank)
    return Monitor(env)


def build_model(args, vec, seed: int, ent_coef: float, log_dir: Path):
    """Construit l'algorithme demandé ; ``--resume`` recharge un MaskablePPO existant."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import DQN, PPO

    common = {"verbose": 1, "seed": seed, "tensorboard_log": str(log_dir), "device": "cpu"}
    if args.resume:
        if args.algo != "maskable_ppo":
            raise SystemExit("--resume ne s'applique qu'à MaskablePPO")
        return MaskablePPO.load(args.resume, env=vec, device="cpu", tensorboard_log=str(log_dir))
    if args.algo == "dqn":
        if args.level != "reduced":
            raise SystemExit("le DQN classique ne s'applique qu'au niveau « reduced »")
        return DQN("MlpPolicy", vec, learning_rate=3e-4, buffer_size=200_000, **common)
    if args.algo == "ppo":
        return PPO("MlpPolicy", vec, n_steps=365, batch_size=730, gamma=0.995, **common)
    if args.algo == "recurrent_ppo":
        # Variante à mémoire (§ 5.3) : utile si l'observabilité partielle pénalise l'agent.
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO(
            "MlpLstmPolicy", vec, n_steps=365, batch_size=730, gamma=0.995, **common
        )
    # Au niveau N2, la loi masquée de sb3-contrib coûte 80 ms par décision ; la version
    # vectorisée donne les mêmes probabilités cinquante fois plus vite.
    from spareparts.rl.padded import PaddedMaskablePolicy

    return MaskablePPO(
        PaddedMaskablePolicy if args.level == "n2" else "MlpPolicy",
        vec,
        n_steps=365,
        batch_size=730,
        learning_rate=3e-4,
        gamma=0.995,
        ent_coef=ent_coef,
        **common,
    )


def learner_level(args) -> str:
    """Niveau du codec à utiliser pour rejouer l'agent entraîné."""
    return "reduced" if args.algo == "dqn" else args.level


def registry_name(args) -> str:
    """Nom du modèle dans le registre MLflow : politique retenue (P4, P5, DQN, PPO récurrent)."""
    if args.algo == "maskable_ppo" and args.level in ("n1", "n2"):
        return "p4" if args.level == "n1" else "p5"
    if args.algo == "dqn":
        return "dqn"
    if args.algo == "recurrent_ppo":
        return "rppo"
    return f"{args.algo}_{args.level}"


def canonical_path(args) -> Path:
    if args.output:
        return Path(args.output)
    if args.algo == "maskable_ppo" and args.level in ("n1", "n2"):
        code = "p4" if args.level == "n1" else "p5"
    elif args.algo == "dqn":
        code = "dqn"
    elif args.algo == "recurrent_ppo":
        code = "rppo"
    else:
        code = f"{args.algo}_{args.level}"
    return MODEL_DIR / f"{code}_{args.scenario}.zip"


def seed_path(canonical: Path, seed: int, n_seeds: int) -> Path:
    if n_seeds == 1:
        return canonical
    return canonical.with_name(f"{canonical.stem}_s{seed}.zip")


def train_one(args, seed: int, n_seeds: int, ent_coef: float, log_dir: Path) -> Path:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    # Curriculum : la première année est d'abord jouée par tranches courtes, ce qui multiplie les
    # fins d'épisode et donc le signal d'apprentissage, avant de revenir à l'horizon complet.
    warmup = 90 if args.curriculum else None
    factories = [
        partial(
            make_env,
            args.scenario,
            args.level,
            args.horizon,
            seed * 100 + i,
            args.risk,
            args.drift,
            warmup,
            risk_min=args.risk_min,
            randomize=args.randomize,
        )
        for i in range(args.envs)
    ]
    vec = SubprocVecEnv(factories) if args.envs > 1 else DummyVecEnv(factories)
    if args.algo == "maskable_ppo":
        # Sans normalisation, le critique de PPO n'apprenait rien (variance expliquée nulle aux
        # niveaux N1 et N2) : les retours, de l'ordre de -200, varient fortement d'une année à
        # l'autre. La normalisation des récompenses ramène ses cibles vers l'unité ; les
        # observations restent inchangées, si bien que l'agent s'exécute ensuite sans ce wrapper.
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True, gamma=0.995)

    model = build_model(args, vec, seed, ent_coef, log_dir)

    if args.level == "n2" and args.algo == "maskable_ppo" and not args.resume:
        import torch

        from spareparts.envs.codecs import inaction_prior

        # Point de départ raisonnable : l'acteur commence par ne presque rien faire.
        prior = inaction_prior(FleetEnv(scenario=args.scenario, level="n2").codec)
        with torch.no_grad():
            model.policy.action_net.bias.add_(torch.as_tensor(prior, dtype=torch.float32))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = seed_path(canonical_path(args), seed, n_seeds)
    run_name = f"{args.algo}_{args.level}_{args.scenario}_s{seed}"

    # L'apprentissage peut progresser puis se dégrader : on garde le modèle au meilleur coût
    # annuel plutôt que le dernier. Le critère est annualisé, sans quoi le curriculum
    # retiendrait le tout premier tirage, d'épisodes courts donc artificiellement « moins
    # coûteux » (voir ``best_model_saver``).
    callbacks = []
    if args.algo == "maskable_ppo":
        keep_best = best_model_saver(path, args.horizon)
        callbacks.append(keep_best)
    else:
        keep_best = None
    if args.curriculum:
        callbacks.append(curriculum_scheduler(args.horizon, warmup))
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=run_name,
        progress_bar=False,
        callback=callbacks or None,
        # Un apprentissage prolongé continue la même courbe dans TensorBoard.
        reset_num_timesteps=not args.resume,
    )
    if keep_best and keep_best.saved:
        print(f"graine {seed} — meilleur coût annuel : {-keep_best.best * 150:.0f} UR")
    else:
        model.save(path)
    vec.close()
    return path


def _train_seed_job(job) -> Path:
    """Entraîne une graine dans son propre processus (usage de ``--workers`` > 1).

    Un fil par processus : sans cela, trois graines menées de front réclameraient chacune tous les
    cœurs pour les quelques multiplications du réseau, sans profit, et se gêneraient mutuellement.
    """
    import torch

    torch.set_num_threads(1)
    return train_one(*job)


def score_agents(args, paths: list[Path], seeds: list[int]) -> dict[int, float]:
    """Coût annuel moyen de chaque graine sur les graines de validation.

    Chaque graine est jouée dans les deux modes — action la plus probable et tirage de la
    loi — et l'on retient son meilleur des deux. Le classement des graines ne dépend alors
    plus du mode que ``choose_agent_mode.py`` publiera ensuite : au niveau N2, où le mode ne
    commande jamais (cf. ``LearnedPolicy``), scorer le seul mode déterministe classerait les
    trois graines d'après un coût dégénéré, sans rapport avec la politique publiée.
    """
    from spareparts.policies.learned import LearnedPolicy

    cfg = load_scenario(args.scenario)
    level = learner_level(args)
    # Le DQN du niveau réduit joue toujours de façon déterministe (``LearnedPolicy.decide``) :
    # inutile de doubler son évaluation.
    modes = (True,) if level == "reduced" else (True, False)
    scores = {}
    for seed, path in zip(seeds, paths, strict=True):
        costs = {}
        for deterministic in modes:
            policy = LearnedPolicy(
                path,
                level,
                name="candidat",
                label=f"graine {seed}",
                deterministic=deterministic,
                scenario=args.scenario,
                use_risk=args.risk,
                use_drift=args.drift,
                risk_min=args.risk_min,
                algo=args.algo,
            )
            runs = [
                FleetSimulation(cfg, seed=s, log_events=False).run(policy).kpis
                for s in VALIDATION_SEEDS
            ]
            costs[deterministic] = float(np.mean([r["cout_total"] for r in runs]))
        scores[seed] = min(costs.values())
        detail = ", ".join(
            f"{'mode' if d else 'tirage'} {c:.0f} UR" for d, c in costs.items()
        )
        print(f"graine {seed} : {detail} → retenu {scores[seed]:.0f} UR")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--level", choices=["reduced", "n1", "n2"], default="n1")
    parser.add_argument(
        "--algo", choices=["maskable_ppo", "ppo", "recurrent_ppo", "dqn"], default="maskable_ppo"
    )
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="graines entraînées en parallèle ; chacune garde ses propres environnements, le "
        "simulateur étant le goulot d'étranglement (même mécanisme que le BDQ)",
    )
    parser.add_argument("--horizon", type=int, default=365)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="graines d'entraînement ; trois au moins rendent compte de la variabilité (§ 5.3)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        help="bonus d'entropie de PPO (par défaut 0,01, et 0 au niveau n2)",
    )
    parser.add_argument("--output", help="chemin du modèle (par défaut models/rl/...)")
    parser.add_argument(
        "--risk",
        action="store_true",
        help="observe aussi la probabilité de panne prédite par le réseau de survie (§ 5.2)",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="observe aussi la dérive normalisée du signal de chaque composant, la grandeur "
        "sur laquelle décide la maintenance conditionnelle P2 (§ 5.3)",
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
        "--randomize",
        action="store_true",
        help="tire délais, taux de panne et effectifs à chaque épisode (entraînement "
        "multi-scénarios, § 6)",
    )
    parser.add_argument(
        "--resume",
        help="modèle MaskablePPO à prolonger ; --timesteps compte alors les pas supplémentaires",
    )
    args = parser.parse_args()

    seeds = args.seeds or [args.seed]
    if args.resume and len(seeds) > 1:
        raise SystemExit("--resume ne se combine pas avec plusieurs graines")
    if args.output and len(seeds) > 1:
        raise SystemExit("--output ne se combine pas avec plusieurs graines")

    # Au niveau N2, le bonus d'entropie se somme sur 108 dimensions et ramène l'agent vers des
    # actions uniformes, à l'opposé du biais initial : l'agent se dégradait au fil des mises à jour.
    ent_coef = args.ent_coef if args.ent_coef is not None else (0.0 if args.level == "n2" else 0.01)
    log_dir = ROOT / "runs" / "rl"
    params = {
        "scenario": args.scenario,
        "level": args.level,
        "algo": args.algo,
        "timesteps": args.timesteps,
        "envs": args.envs,
        "seeds": seeds,
        "ent_coef": ent_coef,
        "use_risk": args.risk,
        "use_drift": args.drift,
        "curriculum": args.curriculum,
        "risk_min": args.risk_min,
        "randomize": args.randomize,
    }

    with tracking.tracked(f"{args.algo}_{args.level}", params):
        if len(seeds) > 1 and args.workers > 1:
            # Les graines sont indépendantes : le suivi MLflow et le choix de la meilleure restent
            # dans le processus principal, les fils n'exécutent que l'entraînement.
            from concurrent.futures import ProcessPoolExecutor

            jobs = [(args, seed, len(seeds), ent_coef, log_dir) for seed in seeds]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                paths = list(pool.map(_train_seed_job, jobs))
        else:
            paths = [train_one(args, seed, len(seeds), ent_coef, log_dir) for seed in seeds]
        if len(seeds) > 1:
            scores = score_agents(args, paths, seeds)
            best = min(scores, key=lambda s: scores[s])
            target = canonical_path(args)
            shutil.copyfile(paths[seeds.index(best)], target)
            tracking.log({f"validation_s{seed}": value for seed, value in scores.items()})
            print(f"graine retenue : {best} ; modèle enregistré dans {target}")
        else:
            target = paths[0]
            print(f"modèle enregistré dans {target}")
        tracking.register(registry_name(args), target)

    # Métadonnées lues à l'exécution : elles indiquent au codec si le risque et la dérive étaient
    # observés, afin que la politique rejoue exactement l'observation de l'entraînement.
    canonical = Path(args.output) if args.output else canonical_path(args)
    canonical.with_suffix(".json").write_text(
        json.dumps(
            {
                "scenario": args.scenario,
                "level": learner_level(args),
                "algo": args.algo,
                "use_risk": args.risk,
                "use_drift": args.drift,
                "risk_min": args.risk_min,
                "randomize": args.randomize,
                "seeds": seeds,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def curriculum_scheduler(horizon: int, start: int):
    """Allonge progressivement l'épisode de ``start`` jours jusqu'à ``horizon``.

    Une année complète ne rend qu'un retour par épisode : l'agent apprend lentement au début.
    Épisode court, l'agent enchaîne les fins d'épisode, donc les retours, et les compétences
    acquises se conservent lorsque l'horizon grandit.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _Curriculum(BaseCallback):
        def __init__(self):
            super().__init__()
            self._last = start

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            fraction = min(1.0, self.num_timesteps / max(self.model._total_timesteps, 1))
            days = int(round(start + (horizon - start) * fraction))
            if days != self._last:
                self._last = days
                self.training_env.env_method("set_episode_days", days)

    return _Curriculum()


def best_model_saver(path: Path, horizon: int = 365):
    """Rappel qui enregistre le modèle lorsque le meilleur coût annuel s'améliore.

    La récompense est l'opposé du coût cumulé sur l'épisode : elle est proportionnelle au
    nombre de jours joués. Sous curriculum, où l'épisode s'allonge de 90 à ``horizon`` jours,
    comparer les récompenses brutes reviendrait à préférer systématiquement l'épisode le plus
    court — donc le tout premier tirage, avant tout apprentissage. On rapporte donc chaque
    épisode à une année (``r / l * horizon``) pour comparer les tirages à durée égale : le
    critère redevient le coût annuel, celui-là même que publie le rapport.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _KeepBest(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best, self.saved = -np.inf, False

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            episodes = self.model.ep_info_buffer
            if len(episodes) < 30:
                return
            # Coût annuel équivalent de chaque épisode puis moyenne : le classement reste juste
            # même lorsque les environnements n'ont pas tous la même durée au même pas.
            annual = [e["r"] / max(e["l"], 1) * horizon for e in episodes]
            mean = float(np.mean(annual))
            if mean > self.best:
                self.best, self.saved = mean, True
                self.model.save(path)

    return _KeepBest()


if __name__ == "__main__":
    main()
