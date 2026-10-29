"""Entraîne un agent d'apprentissage par renforcement sur le simulateur (§ 5.3).

Exemples :
  uv run python scripts/train_rl.py --level n1 --timesteps 300000
  uv run python scripts/train_rl.py --level n2 --algo maskable_ppo --envs 8
  uv run python scripts/train_rl.py --level reduced --algo dqn

Le modèle MaskablePPO est enregistré dans models/rl/p4_<scénario>.zip (niveau n1) ou
p5_<scénario>.zip (niveau n2) ; les courbes d'apprentissage vont dans runs/rl (TensorBoard).
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
    log_dir = ROOT / "runs" / "rl"
    common = {"verbose": 1, "seed": args.seed, "tensorboard_log": str(log_dir), "device": "cpu"}

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
        model = MaskablePPO(
            "MlpPolicy",
            vec,
            n_steps=365,
            batch_size=730,
            learning_rate=3e-4,
            gamma=0.995,
            ent_coef=0.01,
            **common,
        )

    run_name = f"{args.algo}_{args.level}_{args.scenario}_s{args.seed}"
    model.learn(total_timesteps=args.timesteps, tb_log_name=run_name, progress_bar=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        path = Path(args.output)
    elif args.algo == "maskable_ppo" and args.level in ("n1", "n2"):
        code = "p4" if args.level == "n1" else "p5"
        path = MODEL_DIR / f"{code}_{args.scenario}.zip"
    else:
        path = MODEL_DIR / f"{run_name}.zip"
    model.save(path)
    print(f"modèle enregistré dans {path}")


if __name__ == "__main__":
    main()
