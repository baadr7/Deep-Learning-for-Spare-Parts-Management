"""Entraîne un agent d'apprentissage par renforcement sur le simulateur (§ 5.3).

Exemples :
  uv run python scripts/train_rl.py --level n1 --timesteps 300000
  uv run python scripts/train_rl.py --level n2 --algo maskable_ppo --envs 8
  uv run python scripts/train_rl.py --level reduced --algo dqn

Le modèle MaskablePPO est enregistré dans models/rl/p4_<scénario>.zip (niveau n1) ou
p5_<scénario>.zip (niveau n2), le DQN dans dqn_<scénario>.zip ; les courbes d'apprentissage
vont dans runs/rl (TensorBoard).
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from spareparts.config import ROOT
from spareparts.envs.fleet_env import FleetEnv
from spareparts.policies.learned import MODEL_DIR


def make_env(scenario: str, level: str, horizon: int, rank: int):
    from stable_baselines3.common.monitor import Monitor

    env = FleetEnv(scenario=scenario, level=level, horizon_days=horizon)
    env.reset(seed=rank)
    return Monitor(env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--level", choices=["reduced", "n1", "n2"], default="n1")
    parser.add_argument(
        "--algo", choices=["maskable_ppo", "ppo", "recurrent_ppo", "dqn"], default="maskable_ppo"
    )
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=365)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ent-coef",
        type=float,
        help="bonus d'entropie de PPO (par défaut 0,01, et 0 au niveau n2)",
    )
    parser.add_argument("--output", help="chemin du modèle (par défaut models/rl/...)")
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    factories = [
        partial(make_env, args.scenario, args.level, args.horizon, args.seed * 100 + i)
        for i in range(args.envs)
    ]
    vec = SubprocVecEnv(factories) if args.envs > 1 else DummyVecEnv(factories)
    if args.algo == "maskable_ppo":
        from stable_baselines3.common.vec_env import VecNormalize

        # Sans normalisation, le critique de PPO n'apprenait rien (variance expliquée nulle aux
        # niveaux N1 et N2) : les retours, de l'ordre de -200, varient fortement d'une année à
        # l'autre. La normalisation des récompenses ramène ses cibles vers l'unité ; les
        # observations restent inchangées, si bien que l'agent s'exécute ensuite sans ce wrapper.
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True, gamma=0.995)
    log_dir = ROOT / "runs" / "rl"
    common = {"verbose": 1, "seed": args.seed, "tensorboard_log": str(log_dir), "device": "cpu"}

    # Au niveau N2, le bonus d'entropie se somme sur 108 dimensions et ramène l'agent vers des
    # actions uniformes, à l'opposé du biais initial : l'agent se dégradait au fil des mises à jour.
    ent_coef = args.ent_coef if args.ent_coef is not None else (0.0 if args.level == "n2" else 0.01)

    if args.algo == "dqn":
        if args.level != "reduced":
            raise SystemExit("le DQN classique ne s'applique qu'au niveau « reduced »")
        model = DQN("MlpPolicy", vec, learning_rate=3e-4, buffer_size=200_000, **common)
    elif args.algo == "ppo":
        model = PPO("MlpPolicy", vec, n_steps=365, batch_size=730, gamma=0.995, **common)
    elif args.algo == "recurrent_ppo":
        # Variante à mémoire (§ 5.3) : utile si l'observabilité partielle pénalise l'agent.
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO(
            "MlpLstmPolicy", vec, n_steps=365, batch_size=730, gamma=0.995, **common
        )
    else:
        # Au niveau N2, la loi masquée de sb3-contrib coûte 80 ms par décision ; la version
        # vectorisée donne les mêmes probabilités cinquante fois plus vite.
        from spareparts.rl.padded import PaddedMaskablePolicy

        model = MaskablePPO(
            PaddedMaskablePolicy if args.level == "n2" else "MlpPolicy",
            vec,
            n_steps=365,
            batch_size=730,
            learning_rate=3e-4,
            gamma=0.995,
            ent_coef=ent_coef,
            **common,
        )

    if args.level == "n2" and args.algo == "maskable_ppo":
        import torch

        from spareparts.envs.codecs import inaction_prior

        # Point de départ raisonnable : l'acteur commence par ne presque rien faire.
        prior = inaction_prior(FleetEnv(scenario=args.scenario, level="n2").codec)
        with torch.no_grad():
            model.policy.action_net.bias.add_(torch.as_tensor(prior, dtype=torch.float32))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.algo}_{args.level}_{args.scenario}_s{args.seed}"
    if args.output:
        path = Path(args.output)
    elif args.algo == "maskable_ppo" and args.level in ("n1", "n2"):
        code = "p4" if args.level == "n1" else "p5"
        path = MODEL_DIR / f"{code}_{args.scenario}.zip"
    elif args.algo == "dqn":
        path = MODEL_DIR / f"dqn_{args.scenario}.zip"
    else:
        path = MODEL_DIR / f"{run_name}.zip"

    # L'apprentissage peut progresser puis se dégrader : on garde le modèle dont la moyenne des
    # derniers épisodes d'apprentissage est la meilleure plutôt que le dernier.
    keep_best = best_model_saver(path) if args.algo == "maskable_ppo" else None
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=run_name,
        progress_bar=False,
        callback=keep_best,
    )
    if keep_best and keep_best.saved:
        print(f"meilleur coût annuel moyen : {-keep_best.best * 150:.0f} UR")
    else:
        model.save(path)
    print(f"modèle enregistré dans {path}")


def best_model_saver(path: Path):
    """Rappel qui enregistre le modèle chaque fois que la moyenne des épisodes s'améliore."""
    import numpy as np
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
            mean = float(np.mean([e["r"] for e in episodes]))
            if mean > self.best:
                self.best, self.saved = mean, True
                self.model.save(path)

    return _KeepBest()


if __name__ == "__main__":
    main()
