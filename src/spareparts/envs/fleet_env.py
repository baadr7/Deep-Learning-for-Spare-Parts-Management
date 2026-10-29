"""Environnement Gymnasium : une étape correspond à une journée de la flotte (§ 5.3)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from spareparts.config import load_scenario
from spareparts.envs.codecs import Context, make_codec
from spareparts.sim import FleetSimulation


class FleetEnv(gym.Env):
    """Récompense : opposé du coût de la journée, divisé par ``reward_scale``.

    Le coût journalier regroupe stockage, achats, urgences, interventions, arrêts et
    non-qualité (cahier des charges, § 4.3). Chaque épisode tire une nouvelle graine, sauf si
    ``reset`` en reçoit une.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "s1_nominal",
        level: str = "n1",
        horizon_days: int = 365,
        reward_scale: float = 150.0,
        overrides: dict | None = None,
    ):
        super().__init__()
        self.cfg = load_scenario(scenario, **(overrides or {}))
        self.level = level
        self.horizon = horizon_days
        self.reward_scale = reward_scale
        self.sim = FleetSimulation(self.cfg, seed=0, log_events=False)
        self.codec = make_codec(level, Context.from_sim(self.sim, horizon_days))
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.codec.obs_size,), dtype=np.float32
        )
        if level == "reduced":
            self.action_space = spaces.Discrete(self.codec.n)
        else:
            self.action_space = spaces.MultiDiscrete(self.codec.nvec)
        self._obs = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        sim_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.sim = FleetSimulation(self.cfg, seed=sim_seed, log_events=False)
        self._obs = self.sim.observe()
        return self.codec.observe(self._obs), {"seed": sim_seed}

    def step(self, action):
        decisions = self.codec.decide(action, self._obs)
        rejected = self.sim.apply(decisions)
        record = self.sim.advance_day()
        cost = float(sum(record.costs.values()))
        self._obs = self.sim.observe()
        truncated = self.sim.day >= self.horizon
        info = {"cost": cost, "rejected": [r["reason"] for r in rejected]}
        if truncated:
            info["kpis"] = {
                "trs": self.sim.c_running
                / self.sim.c_open
                * (1 - self.sim.defects / self.sim.produced)
                if self.sim.c_open and self.sim.produced
                else 0.0,
                "cout_total": float(sum(self.sim.costs.values())),
            }
        return self.codec.observe(self._obs), -cost / self.reward_scale, False, truncated, info

    def action_masks(self) -> np.ndarray:
        """Masque des actions valides, au format attendu par MaskablePPO."""
        return self.codec.masks(self._obs)
