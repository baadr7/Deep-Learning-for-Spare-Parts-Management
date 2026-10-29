"""Entraîne le Branching Dueling Q-Network au niveau N2 puis l'évalue (§ 5.3).

Exemple : uv run python scripts/train_bdq.py --steps 200000
Le réseau est enregistré dans models/rl/bdq_<scénario>.pt ; l'évaluation utilise les graines
de scripts/evaluate_policies.py pour rester comparable aux autres politiques.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spareparts.config import ROOT, load_scenario
from spareparts.envs.fleet_env import FleetEnv
from spareparts.evaluation import summarize
from spareparts.rl.bdq import BDQConfig, BDQPolicy, BranchingQNet, train_bdq
from spareparts.sim import FleetSimulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="s1_nominal")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--eval-replications", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()

    env = FleetEnv(scenario=args.scenario, level="n2")
    net = train_bdq(env, BDQConfig(total_steps=args.steps))
    path = Path(args.output) if args.output else ROOT / "models" / "rl" / f"bdq_{args.scenario}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"state": net.state_dict(), "sizes": net.sizes, "obs": env.observation_space.shape[0]}
    torch.save(blob, path)

    cfg = load_scenario(args.scenario)
    seeds = range(9000, 9000 + args.eval_replications)
    runs = [FleetSimulation(cfg, seed=s, log_events=False).run(BDQPolicy(net)).kpis for s in seeds]
    summary = summarize(runs)
    print({k: round(summary[k]["mean"], 3) for k in ("cout_total", "trs", "frequence_ruptures")})
    out = ROOT / "reports" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"bdq_{args.scenario}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(f"modèle enregistré dans {path}")


def load(path) -> BranchingQNet:
    blob = torch.load(path, map_location="cpu")
    net = BranchingQNet(blob["obs"], np.array(blob["sizes"]))
    net.load_state_dict(blob["state"])
    return net


if __name__ == "__main__":
    main()
