"""Environnement Gymnasium : une étape correspond à une journée de la flotte (§ 5.3)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from spareparts.config import load_scenario
from spareparts.envs.codecs import Context, make_codec
from spareparts.sim import FleetSimulation, Observation

# Bornes de la randomisation multi-scénarios (§ 6, amélioration 7) : elles couvrent les
# situations dégradées de l'évaluation de robustesse (délai doublé, taux de panne +30 %,
# technicien en moins) afin que l'agent rencontre ces régimes pendant l'apprentissage.
RANDOMIZATION_RANGES: dict[str, tuple[float, float]] = {
    "lead_time_factor": (0.5, 2.0),
    "failure_rate_factor": (0.6, 1.8),
    "technicians": (1.0, 4.0),
}


class FleetEnv(gym.Env):
    """Récompense : opposé du coût de la journée, divisé par ``reward_scale``.

    Le coût journalier regroupe stockage, achats, urgences, interventions, arrêts et
    non-qualité (cahier des charges, § 4.3). Chaque épisode tire une nouvelle graine, sauf si
    ``reset`` en reçoit une.

    Trois options étendent l'environnement sans modifier son comportement par défaut :

    - ``risk_min`` restreint la maintenance préventive aux composants dont le risque prédit
      dépasse ce seuil (masquage guidé par le risque, § 6, amélioration 6) ;
    - ``randomize`` tire les paramètres de l'épisode (délais, taux de panne, effectifs) dans les
      bornes fournies, ce qui entraîne des politiques robustes par construction (§ 6,
      amélioration 7) ;
    - ``branch_reward`` place dans ``info`` une récompense par branche d'action, dont la somme
      égale la récompense globale, pour l'apprentissage du BDQ (§ 6, amélioration 5).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "s1_nominal",
        level: str = "n1",
        horizon_days: int = 365,
        reward_scale: float = 150.0,
        overrides: dict | None = None,
        use_risk: bool = False,
        use_drift: bool = False,
        episode_days: int | None = None,
        risk_min: float | None = None,
        randomize: dict[str, tuple[float, float]] | None = None,
        branch_reward: bool = False,
    ):
        super().__init__()
        self.scenario = scenario
        self.overrides = dict(overrides or {})
        self.level = level
        self.horizon = horizon_days
        self.reward_scale = reward_scale
        self.use_risk = use_risk
        self.use_drift = use_drift
        self.risk_min = risk_min
        self.randomize = randomize
        # La récompense par branche suppose un espace d'action multidimensionnel : elle se
        # restreint donc au niveau N2.
        self.branch_reward = bool(branch_reward) and level == "n2"
        # Durée effective d'un épisode : le curriculum l'allonge progressivement, tandis que
        # ``horizon`` reste la référence de normalisation de l'observation (jour / horizon).
        self.episode_days = horizon_days if episode_days is None else episode_days
        self.cfg = load_scenario(scenario, **self.overrides)
        self.sim = FleetSimulation(self.cfg, seed=0, log_events=False)
        self.codec = self._make_codec()
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.codec.obs_size,), dtype=np.float32
        )
        if level == "reduced":
            self.action_space = spaces.Discrete(self.codec.n)
        else:
            self.action_space = spaces.MultiDiscrete(self.codec.nvec)
        self._predictor = None
        self._risk_days = 1
        if use_risk:
            self._build_risk()
        self._obs = None

    def _make_codec(self):
        return make_codec(
            self.level,
            Context.from_sim(
                self.sim,
                self.horizon,
                use_risk=self.use_risk,
                use_drift=self.use_drift,
                risk_min=self.risk_min,
            ),
        )

    def set_episode_days(self, days: int) -> None:
        """Fixe la durée effective d'un épisode (curriculum), sans changer sa normalisation."""
        self.episode_days = int(days)

    def _build_risk(self) -> None:
        """Charge le réseau de survie et fixe l'horizon du risque observé.

        Le réseau est indépendant de la graine : il est chargé une fois, puis la fenêtre de
        télémétrie est réinitialisée à chaque épisode sur la nouvelle simulation.
        """
        from spareparts.predict.inference import make_predictor, risk_horizon_days

        self._predictor = make_predictor(self.scenario, self.sim)
        self._risk_days = risk_horizon_days(self.scenario)

    def _sample_overrides(self) -> dict:
        """Tire les paramètres du scénario d'un épisode (entraînement multi-scénarios).

        Les délais fournisseur, le taux de panne et l'effectif sont échantillonnés uniformément
        dans les bornes fournies ; l'effectif est arrondi à l'entier le plus proche. Le tirage
        suit le générateur de l'environnement, donc la graine de ``reset``.
        """
        if not self.randomize:
            return {}
        out = {}
        for key, (low, high) in self.randomize.items():
            value = float(self.np_random.uniform(low, high))
            out[key] = int(np.rint(value)) if key == "technicians" else value
        return out

    def _observe(self) -> Observation:
        """Observation du matin, enrichie du risque prédit lorsque ``use_risk`` est actif."""
        obs = self.sim.observe()
        if self.use_risk:
            from spareparts.predict.inference import observed_risk

            obs.risk = observed_risk(self._predictor, self.sim, obs, self._risk_days)
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        overrides = self._sample_overrides()
        if overrides:
            self.cfg = load_scenario(self.scenario, **{**self.overrides, **overrides})
        sim_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.sim = FleetSimulation(self.cfg, seed=sim_seed, log_events=False)
        if overrides:
            # Le codec dépend des effectifs et le risque observé de la flotte : on les
            # reconstruit sur la simulation de l'épisode.
            self.codec = self._make_codec()
        if self.use_risk:
            from spareparts.predict.inference import window_hours

            self.sim.enable_window(window_hours())
            if overrides:
                self._build_risk()
        self._obs = self._observe()
        info = {"seed": sim_seed}
        if overrides:
            info["overrides"] = overrides
        return self.codec.observe(self._obs), info

    def _branch_reward(self, record, total: float) -> np.ndarray:
        """Récompense de chaque branche d'action, de somme égale à la récompense globale.

        Chaque branche reçoit le coût de sa propre décision : les branches de commande portent
        la tenue de stock et l'achat de la référence, les branches urgentes la surprime
        d'urgence, les branches machine l'arrêt, la main-d'œuvre et la réparation. Le coût non
        attribuable (rebut, retouche) est réparti également sur l'ensemble des branches.
        """
        attributed = np.concatenate(
            [
                record.costs_order + record.costs_holding,
                record.costs_urgent,
                record.costs_machine,
            ]
        )
        residual = total - float(attributed.sum())
        return (-(attributed + residual / attributed.size) / self.reward_scale).astype(np.float32)

    def step(self, action):
        decisions = self.codec.decide(action, self._obs)
        rejected = self.sim.apply(decisions)
        record = self.sim.advance_day(risk=self._obs.risk)
        cost = float(sum(record.costs.values()))
        self._obs = self._observe()
        truncated = self.sim.day >= self.episode_days
        info = {"cost": cost, "rejected": [r["reason"] for r in rejected]}
        if self.branch_reward:
            info["reward_branch"] = self._branch_reward(record, cost)
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
