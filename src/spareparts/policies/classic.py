"""Politiques de référence P0, P1 et P2 (cahier des charges, § 7.1)."""

from __future__ import annotations

import numpy as np

from spareparts.policies.base import Policy, reorder_up_to, schedule_preventive
from spareparts.sim.engine import Decisions, FleetSimulation, Observation


class FixedThresholdPolicy(Policy):
    """Stock géré en (s, S) et maintenance préventive systématique.

    ``pm_factor`` multiplie l'intervalle de maintenance de chaque composant : 1 correspond aux
    intervalles constructeur (P0), d'autres valeurs résultent de l'optimisation (P1).
    """

    def __init__(self, s=None, S=None, pm_factor=1.0, reserve=1, name="P0", label=None):
        self.s_init, self.S_init, self.pm_factor_init = s, S, pm_factor
        self.reserve = reserve
        self.name = name
        self.label = label or (
            "Seuils fixes réglés à la main" if name == "P0" else "Seuils fixes optimisés"
        )

    def reset(self, sim: FleetSimulation) -> None:
        super().reset(sim)
        parts = sim.cfg.parts
        self.s = np.array(self.s_init if self.s_init is not None else [p.s for p in parts])
        self.S = np.array(self.S_init if self.S_init is not None else [p.S for p in parts])
        self.pm_factor = np.broadcast_to(np.asarray(self.pm_factor_init, float), (sim.K,))

    def decide(self, obs: Observation) -> Decisions:
        ratio = obs.age_unit / (obs.pm_interval * self.pm_factor)
        score = np.where(ratio >= 1.0, ratio, np.nan)
        return Decisions(
            orders=reorder_up_to(obs, self.s, self.S),
            preventive=schedule_preventive(obs, score, keep=self.reserve),
        )


class ConditionalPolicy(Policy):
    """P2 : maintenance déclenchée par la dérive d'une mesure ou par le taux de défauts.

    Un composant est remplacé lorsque le signal qui lui est associé s'écarte de plus de
    ``z_threshold`` écarts-types dans le sens de sa dérive, ou lorsque le taux de défauts de la
    machine dépasse ``defect_threshold`` (le composant le plus âgé est alors choisi). Un
    remplacement systématique à ``safety_factor`` fois l'intervalle sert de garde-fou.
    """

    name = "P2"
    label = "Maintenance conditionnelle"

    def __init__(
        self,
        s=None,
        S=None,
        z_threshold=1.0,
        defect_threshold=0.03,
        safety_factor=1.5,
        reserve=1,
    ):
        self.s_init, self.S_init = s, S
        self.reserve = reserve
        self.z_threshold = z_threshold
        self.defect_threshold = defect_threshold
        self.safety_factor = safety_factor

    def reset(self, sim: FleetSimulation) -> None:
        super().reset(sim)
        parts = sim.cfg.parts
        self.s = np.array(self.s_init if self.s_init is not None else [p.s for p in parts])
        self.S = np.array(self.S_init if self.S_init is not None else [p.S for p in parts])

    def decide(self, obs: Observation) -> Decisions:
        M = obs.machine_state.shape[0]
        z = obs.signal_z_24h[np.arange(M)[:, None], obs.signal_of] * obs.drift_sign
        ratio = obs.age_unit / obs.pm_interval
        drifting = z >= self.z_threshold
        rate = np.nan_to_num(obs.defect_rate_24h, nan=0.0)
        oldest = ratio == ratio.max(axis=1, keepdims=True)
        poor_quality = (rate >= self.defect_threshold)[:, None] & oldest & (ratio >= 0.5)
        safety = ratio >= self.safety_factor
        candidate = drifting | poor_quality | safety
        score = np.where(candidate, z + ratio, np.nan)
        return Decisions(
            orders=reorder_up_to(obs, self.s, self.S),
            preventive=schedule_preventive(obs, score, keep=self.reserve),
        )
