"""Simulateur à événements discrets de la flotte (cahier des charges, § 5.1).

Le temps est exprimé en heures et ``t = 0`` correspond au début du premier poste du jour 0 :
une machine à ``n`` postes est ouverte pendant les ``8 n`` premières heures de chaque jour.

Le vieillissement des composants est suivi heure par heure, de façon vectorisée sur toute la
flotte. La production, les défauts, la télémétrie et les alarmes dépendent de l'usure, qui varie
de moins de 1 % par jour : leurs espérances sont donc évaluées par blocs de six heures (et juste
avant chaque remplacement de composant), puis les quantités sont tirées une fois par jour. La
loi des quantités journalières reste celle d'un tirage horaire, pour un coût de calcul bien
moindre. Les pannes, interventions et livraisons sont des processus SimPy qui se partagent le
stock et l'équipe de techniciens.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import simpy

from spareparts.config import Lognormal, ScenarioConfig, Triangular
from spareparts.sim.states import N_STATES
from spareparts.sim.states import MachineState as S

SIGNALS = ("volt", "rotate", "pressure", "vibration")
COST_KEYS = (
    "stockage",
    "commandes",
    "urgences",
    "main_oeuvre",
    "corrective",
    "arret",
    "rebut",
    "retouche",
    "revision",
)
HOURS_PER_YEAR = 8760.0
FLUSH_EVERY_HOURS = 6
_OP = int(S.OPERATIONAL)


@dataclass
class Decisions:
    """Décisions prises par une politique au début d'une journée."""

    orders: dict[int, int] = field(default_factory=dict)
    urgent: dict[int, int] = field(default_factory=dict)
    preventive: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Hidden:
    """Informations inaccessibles en pratique, réservées à la politique oracle."""

    life: np.ndarray
    unit_age: np.ndarray
    duty: np.ndarray


@dataclass
class Observation:
    day: int
    stock: np.ndarray
    on_order: np.ndarray
    backorders: np.ndarray
    pipeline: list[dict]
    machine_state: np.ndarray
    family: np.ndarray
    age_unit: np.ndarray
    usage: np.ndarray
    pm_interval: np.ndarray
    part_of: np.ndarray
    signal_of: np.ndarray
    drift_sign: np.ndarray
    defect_rate_24h: np.ndarray
    signal_z_24h: np.ndarray
    alarms_24h: np.ndarray
    techs_free: int
    hidden: Hidden
    risk: np.ndarray | None = None


@dataclass
class DayRecord:
    day: int
    states: np.ndarray
    stock: np.ndarray
    on_order: np.ndarray
    backorders: np.ndarray
    costs: dict[str, float]
    produced: int
    defects: int
    open_hours: float
    running_hours: float
    events: list[dict]


def _lognormal(rng: np.random.Generator, mean: float, cv: float) -> float:
    if cv <= 0:
        return float(mean)
    s2 = math.log1p(cv * cv)
    return float(rng.lognormal(math.log(mean) - s2 / 2, math.sqrt(s2)))


def _gamma(rng: np.random.Generator, mean: float, cv: float) -> float:
    if cv <= 0:
        return float(mean)
    shape = 1.0 / (cv * cv)
    return float(rng.gamma(shape, mean / shape))


class FleetSimulation:
    def __init__(
        self,
        config: ScenarioConfig,
        seed: int | None = None,
        *,
        log_events: bool = True,
        log_telemetry: bool = False,
    ) -> None:
        self.cfg = config
        self.seed = config.seed if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.log_events = log_events
        self.log_telemetry = log_telemetry
        self._setup_fleet()
        self._setup_state()
        self.env = simpy.Environment()
        self.techs = simpy.PriorityResource(self.env, capacity=config.technicians)
        self.env.process(self._clock())

    # ------------------------------------------------------------------ construction

    def _setup_fleet(self) -> None:
        cfg, rng = self.cfg, self.rng
        M = cfg.fleet.n_machines
        codes = [c for c in cfg.families if cfg.fleet.family_share.get(c, 0) > 0]
        shares = np.array([cfg.fleet.family_share[c] for c in codes], dtype=float)
        counts = np.floor(shares / shares.sum() * M).astype(int)
        counts[int(np.argmax(shares))] += M - counts.sum()
        self.family_codes = codes
        self.family = np.repeat(np.arange(len(codes)), counts)
        fams = [cfg.families[codes[f]] for f in self.family]

        self.M, self.K, self.P = M, cfg.n_components, len(cfg.parts)
        self.shifts = np.array([rng.choice(f.shifts) for f in fams])
        self.open_per_day = 8.0 * self.shifts
        self.duty = self.open_per_day / 24.0
        self.cadence = np.array([f.cadence for f in fams], dtype=float)
        self.margin = np.array([f.margin_per_part for f in fams], dtype=float)
        self.hourly_margin = self.cadence * self.margin
        self.age_years = rng.uniform(*cfg.fleet.age_years, size=M)

        comps = [f.components for f in fams]

        def grid(attr, dtype=float):
            return np.array([[getattr(c, attr) for c in row] for row in comps], dtype=dtype)

        self.usage = np.array([[c.regime == "usage" for c in row] for row in comps])
        self._use_inc = self.usage.astype(float)
        self._cal_inc = 1.0 - self._use_inc
        self.beta = grid("weibull_beta")
        # Multiplier le taux de panne par c revient à diviser l'échelle de Weibull par c^(1/beta).
        self.eta = grid("weibull_eta") * cfg.failure_rate_factor ** (-1.0 / self.beta)
        self.pm_interval = grid("pm_interval")
        self.repair_mean = np.array([[c.repair.mean for c in row] for row in comps])
        self.repair_cv = np.array([[c.repair.cv for c in row] for row in comps])
        self.pm_ratio = grid("pm_ratio")
        self.part_of = grid("part", int)
        self.signal_of = np.array([[SIGNALS.index(c.signal) for c in row] for row in comps])
        self.drift = grid("drift")
        self.defect_weight = grid("defect_weight")
        self.component_names = [c.name for c in fams[0].components]

        self._onehot = np.zeros((M, self.K, len(SIGNALS)))
        rows, cols = np.indices((M, self.K))
        self._onehot[rows, cols, self.signal_of] = 1.0

        tel = cfg.telemetry
        self.sig_mean = np.array([tel.mean[s] for s in SIGNALS])
        self.sig_sd = np.array([tel.sd[s] for s in SIGNALS])
        self.sig_base = self.sig_mean + rng.normal(0, tel.machine_offset, (M, 4)) * self.sig_sd

        # Révisions générales (famille C) positionnées à partir de l'âge de la machine.
        self.next_overhaul = np.full(M, np.inf)
        self.overhaul_interval = np.full(M, np.inf)
        self._overhaul_spec: list[tuple[Lognormal, float] | None] = []
        for m, f in enumerate(fams):
            if f.overhaul_years and f.overhaul_days:
                years = float(rng.choice(f.overhaul_years))
                nxt = math.ceil(self.age_years[m] / years) * years
                if nxt <= self.age_years[m]:
                    nxt += years
                self.next_overhaul[m] = (nxt - self.age_years[m]) * HOURS_PER_YEAR
                self.overhaul_interval[m] = years * HOURS_PER_YEAR
                self._overhaul_spec.append((f.overhaul_days, f.overhaul_cost))
            else:
                self._overhaul_spec.append(None)
        self._next_overhaul_min = float(self.next_overhaul.min())

    def _setup_state(self) -> None:
        cfg, rng = self.cfg, self.rng
        M, K, P = self.M, self.K, self.P
        self.state = np.zeros(M, dtype=np.int8)
        self.life = self.eta * rng.weibull(self.beta)
        self.drift_amp = self._draw_drift_amp((M, K))
        self._sig_drift = self._onehot * (self.drift * self.drift_amp)[:, :, None]

        # Départ « en régime » : chaque composant a déjà consommé une part de son intervalle.
        # L'âge est compté dans l'unité du régime : heures de marche ou heures calendaires.
        init = rng.uniform(0, 1, (M, K)) * self.pm_interval
        self.uage = np.minimum(init, 0.85 * self.life)

        self.stock = np.array([p.initial_stock for p in cfg.parts], dtype=int)
        self.unit_cost = np.array([p.unit_cost for p in cfg.parts], dtype=float)
        self.pipeline: list[dict] = []
        self.backorders: list[deque] = [deque() for _ in range(P)]
        self.events: list[dict] = []
        self._events_mark = 0
        self.costs = dict.fromkeys(COST_KEYS, 0.0)

        self.day = 0
        self.c_open = 0.0
        self.c_running = 0.0
        self.state_hours = np.zeros(N_STATES)
        self.produced = 0
        self.defects = 0
        self.scrapped = 0
        self.reworked = 0
        self.stock_value_hours = 0.0
        self.n_failures = 0
        self.n_corrective = 0
        self.n_preventive = 0
        self.n_overhauls = 0
        self.test_failures = 0
        self.demands = np.zeros(P, dtype=int)
        self.stockouts = np.zeros(P, dtype=int)
        self.wait_part_hours = 0.0
        self.wait_tech_hours = 0.0
        self.corrective_downtime = 0.0
        self.tech_busy_hours = 0.0
        self.failures_by_component = np.zeros(K, dtype=int)
        self.down_hours = np.zeros(M)
        self._rows = np.arange(M)

        self._reset_day_accumulators()
        self.last_defect_rate = np.full(M, np.nan)
        self.last_signal_z = np.zeros((M, 4))
        self.last_signal_mean = np.tile(self.sig_mean, (M, 1))
        self.last_alarms = np.zeros(M)
        self.history: dict[str, list] = {
            k: []
            for k in (
                "states",
                "day_state",
                "stock",
                "on_order",
                "backorders",
                "costs",
                "produced",
                "defects",
                "open_hours",
                "running_hours",
                "defect_rate",
                "signal_mean",
                "alarms",
                "risk",
            )
        }
        self.telemetry_log: list[np.ndarray] = []

    def _draw_drift_amp(self, size) -> np.ndarray:
        cv = self.cfg.telemetry.drift_cv
        s2 = math.log1p(cv * cv)
        return self.rng.lognormal(-s2 / 2, math.sqrt(s2), size)

    def _reset_day_accumulators(self) -> None:
        M, K = self.M, self.K
        self._blk_run = np.zeros(M)
        self._d_state_h = np.zeros((M, N_STATES))
        self._d_prod = np.zeros(M)
        self._d_expdef = np.zeros(M)
        self._d_run = np.zeros(M)
        self._d_D3 = np.zeros((M, K))
        self._d_lam = np.zeros(M)
        self._d_open = 0.0
        self._d_running = 0.0
        self._d_stock_value = 0.0

    # ------------------------------------------------------------------ outils

    def _tri(self, t: Triangular) -> float:
        if t.high <= t.low:
            return float(t.mode)
        return float(self.rng.triangular(t.low, t.mode, t.high))

    def _log(self, kind: str, machine: int | None = None, component: int | None = None, **info):
        if not self.log_events:
            return
        self.events.append(
            {
                "t": round(float(self.env.now), 3),
                "day": int(self.env.now // 24),
                "type": kind,
                "machine": None if machine is None else int(machine),
                "component": None if component is None else int(component),
                **info,
            }
        )

    def unit_age(self) -> np.ndarray:
        return self.uage

    def life_fraction(self) -> np.ndarray:
        return np.minimum(self.uage / self.life, 1.0)

    def defect_probability(self, life_fraction: np.ndarray, machine: int | None = None):
        """Probabilité qu'une pièce soit défectueuse (§ 3.3), pour la flotte ou une machine."""
        q = self.cfg.quality
        weights = self.defect_weight if machine is None else self.defect_weight[machine]
        powered = life_fraction**q.gamma
        return np.minimum(q.p0 + (weights * powered).sum(axis=-1), 0.5)

    # ------------------------------------------------------------------ horloge

    def _clock(self):
        while True:
            self._tick()
            yield self.env.timeout(1.0)

    def _tick(self) -> None:
        """Avance d'une heure le vieillissement et les compteurs de la journée."""
        t = self.env.now
        open_ = (t % 24.0) < self.open_per_day
        op = self.state == _OP
        running = op & open_
        self.uage += self._cal_inc + self._use_inc * running[:, None]

        changed = False
        over = self.uage >= self.life
        if over.any():
            over &= op[:, None]
            for m in np.flatnonzero(over.any(axis=1)):
                k = int(np.argmax(np.where(over[m], self.uage[m] / self.life[m], -1.0)))
                self.state[m] = S.FAILED
                self.env.process(self._corrective(int(m), k))
                changed = True
        if t >= self._next_overhaul_min:
            due = (self.state == _OP) & (t >= self.next_overhaul)
            for m in np.flatnonzero(due):
                self.state[m] = S.PREVENTIVE
                self.next_overhaul[m] = np.inf
                self.env.process(self._overhaul(int(m)))
                changed = True
            self._next_overhaul_min = float(self.next_overhaul.min())
        if changed:
            running = (self.state == _OP) & open_

        n_open = float(np.count_nonzero(open_))
        n_run = float(np.count_nonzero(running))
        self.c_open += n_open
        self.c_running += n_run
        self._d_open += n_open
        self._d_running += n_run
        self.state_hours += np.bincount(self.state[open_], minlength=N_STATES)
        self._d_state_h[self._rows, self.state] += 1.0
        down = open_ & ~running
        self.down_hours += down
        self.costs["arret"] += float(self.hourly_margin @ down)
        self._blk_run += running
        self._d_stock_value += float(self.stock @ self.unit_cost)

        if self.log_telemetry:
            D = np.minimum(self.uage / self.life, 1.0)
            tel = self.cfg.telemetry
            drift = np.einsum("mk,mks->ms", D**3, self._sig_drift)
            noise = self.rng.normal(0.0, tel.noise, (self.M, 4))
            signal = (self.sig_base + (drift + noise) * self.sig_sd).astype(np.float32)
            signal[~running] = np.nan
            self.telemetry_log.append(signal)
        if t % FLUSH_EVERY_HOURS == FLUSH_EVERY_HOURS - 1:
            self._flush()

    def _flush(self) -> None:
        """Convertit les heures de marche du bloc en espérances de production et de défauts."""
        run = self._blk_run
        if not run.any():
            return
        D = np.minimum(self.uage / self.life, 1.0)
        D3 = D * D * D
        produced = self.cadence * run
        self._d_prod += produced
        self._d_expdef += produced * self.defect_probability(D)
        self._d_run += run
        self._d_D3 += D3 * run[:, None]
        tel = self.cfg.telemetry
        lam = tel.alarm_base_per_hour + tel.alarm_slope_per_hour * D.max(axis=1) ** 2
        self._d_lam += lam * run
        self._blk_run = np.zeros(self.M)

    # ------------------------------------------------------------------ processus

    def _renew(self, m: int, k: int) -> None:
        self._flush()
        self.uage[m, k] = 0.0
        self.life[m, k] = self.eta[m, k] * self.rng.weibull(self.beta[m, k])
        self.drift_amp[m, k] = self._draw_drift_amp(None)
        self._sig_drift[m, k] = self._onehot[m, k] * self.drift[m, k] * self.drift_amp[m, k]

    def _trial_batch_ok(self, m: int) -> bool:
        D = np.minimum(self.uage[m] / self.life[m], 1.0)
        p = float(self.defect_probability(D, machine=m))
        return bool(self.rng.random() < (1.0 - p) ** self.cfg.quality.trial_batch)

    def _intervene(self, m: int, k: int, corrective: bool):
        """Réparation ou maintenance préventive suivie du test, technicien mobilisé."""
        env, cfg = self.env, self.cfg
        work_state = S.REPAIR if corrective else S.PREVENTIVE
        self.state[m] = work_state
        base = _lognormal(self.rng, self.repair_mean[m, k], self.repair_cv[m, k])
        if not corrective:
            base *= self.pm_ratio[m, k]
        hours = self._tri(cfg.durations.store_issue) + base
        yield env.timeout(hours)
        self._renew(m, k)
        while True:
            self.state[m] = S.TEST
            test = self._tri(cfg.durations.test)
            yield env.timeout(test)
            hours += test
            if self._trial_batch_ok(m):
                break
            self.test_failures += 1
            self._log("echec_test", m, k)
            self.state[m] = work_state
            rework = base * cfg.durations.rework_fraction
            yield env.timeout(rework)
            hours += rework
        labor = hours * cfg.costs.labor_per_hour
        self.costs["main_oeuvre"] += labor
        if corrective:
            part_cost = self.unit_cost[self.part_of[m, k]]
            self.costs["corrective"] += (cfg.costs.corrective_ratio - 1.0) * (labor + part_cost)
        self.tech_busy_hours += hours

    def _corrective(self, m: int, k: int):
        env = self.env
        t_fail = env.now
        self.n_failures += 1
        self.failures_by_component[k] += 1
        self._log("panne", m, k)
        diag = self.cfg.durations.diagnosis
        yield env.timeout(_lognormal(self.rng, diag.mean, diag.cv))
        p = int(self.part_of[m, k])
        self.demands[p] += 1
        if self.stock[p] > 0:
            self.stock[p] -= 1
        else:
            self.stockouts[p] += 1
            self.state[m] = S.WAITING_PART
            self._log("rupture", m, k, part=p)
            arrival = env.event()
            self.backorders[p].append(arrival)
            t0 = env.now
            yield arrival
            self.wait_part_hours += env.now - t0
        self.state[m] = S.WAITING_TECH
        t0 = env.now
        with self.techs.request(priority=0) as req:
            yield req
            self.wait_tech_hours += env.now - t0
            yield from self._intervene(m, k, corrective=True)
        self.n_corrective += 1
        self.corrective_downtime += env.now - t_fail
        self.state[m] = S.OPERATIONAL
        self._log("remise_en_service", m, k)

    def _preventive(self, m: int, k: int):
        with self.techs.request(priority=1) as req:
            yield req
            yield from self._intervene(m, k, corrective=False)
        self.n_preventive += 1
        self.state[m] = S.OPERATIONAL
        self._log("fin_preventive", m, k)

    def _overhaul(self, m: int):
        spec = self._overhaul_spec[m]
        assert spec is not None
        days, cost = spec
        self._log("revision", m)
        with self.techs.request(priority=1) as req:
            yield req
            duration = _lognormal(self.rng, days.mean, days.cv) * 24.0
            yield self.env.timeout(duration)
        for k in range(self.K):
            if not self.usage[m, k]:
                self._renew(m, k)
        labor_hours = duration / 3.0
        self.costs["revision"] += cost + labor_hours * self.cfg.costs.labor_per_hour
        self.tech_busy_hours += labor_hours
        self.n_overhauls += 1
        self.next_overhaul[m] = self.env.now + self.overhaul_interval[m]
        self._next_overhaul_min = float(self.next_overhaul.min())
        self.state[m] = S.OPERATIONAL
        self._log("fin_revision", m)

    def _order(self, p: int, qty: int, urgent: bool):
        env, cfg = self.env, self.cfg
        part = cfg.parts[p]
        lead_days = (part.urgent_lead_days if urgent else part.lead_days) * cfg.lead_time_factor
        dur = cfg.durations
        delay = dur.order_processing + _gamma(self.rng, lead_days, part.lead_cv) * 24.0
        delay += self._tri(dur.reception)
        record = {
            "part": p,
            "qty": qty,
            "urgent": urgent,
            "placed": float(env.now),
            "expected": float(
                env.now + dur.order_processing + lead_days * 24.0 + dur.reception.mode
            ),
        }
        self.pipeline.append(record)
        self.costs["commandes"] += cfg.costs.order_fixed + qty * part.unit_cost
        if urgent:
            self.costs["urgences"] += qty * part.unit_cost * (cfg.costs.urgent_factor - 1.0)
        self._log("commande", part=p, qty=qty, urgent=urgent)
        yield env.timeout(delay)
        self.pipeline.remove(record)
        self.stock[p] += qty
        self._log("reception", part=p, qty=qty)
        while self.stock[p] > 0 and self.backorders[p]:
            self.stock[p] -= 1
            self.backorders[p].popleft().succeed()

    # ------------------------------------------------------------------ pilotage

    @property
    def techs_free(self) -> int:
        return self.techs.capacity - self.techs.count - len(self.techs.queue)

    def on_order(self) -> np.ndarray:
        out = np.zeros(self.P, dtype=int)
        for rec in self.pipeline:
            out[rec["part"]] += rec["qty"]
        return out

    def observe(self, risk: np.ndarray | None = None) -> Observation:
        unit_age = self.uage.copy()
        return Observation(
            day=self.day,
            stock=self.stock.copy(),
            on_order=self.on_order(),
            backorders=np.array([len(b) for b in self.backorders]),
            pipeline=[dict(r) for r in self.pipeline],
            machine_state=self.state.copy(),
            family=self.family,
            age_unit=unit_age,
            usage=self.usage,
            pm_interval=self.pm_interval,
            part_of=self.part_of,
            signal_of=self.signal_of,
            drift_sign=np.sign(self.drift),
            defect_rate_24h=self.last_defect_rate.copy(),
            signal_z_24h=self.last_signal_z.copy(),
            alarms_24h=self.last_alarms.copy(),
            techs_free=self.techs_free,
            hidden=Hidden(life=self.life.copy(), unit_age=unit_age, duty=self.duty),
            risk=risk,
        )

    def apply(self, decisions: Decisions) -> list[dict]:
        """Exécute les décisions ; renvoie les maintenances refusées et leur motif."""
        for p, qty in decisions.orders.items():
            if qty > 0:
                self.env.process(self._order(int(p), int(qty), urgent=False))
        for p, qty in decisions.urgent.items():
            if qty > 0:
                self.env.process(self._order(int(p), int(qty), urgent=True))
        rejected = []
        free = self.techs_free
        for m, k in decisions.preventive:
            p = int(self.part_of[m, k])
            reason = None
            if self.state[m] != S.OPERATIONAL:
                reason = "machine indisponible"
            elif self.stock[p] <= 0:
                reason = "pièce absente du stock"
            elif free <= 0:
                reason = "aucun technicien libre"
            if reason:
                rejected.append({"machine": int(m), "component": int(k), "reason": reason})
                continue
            free -= 1
            self.stock[p] -= 1
            self.state[m] = S.PREVENTIVE
            self._log("maintenance_preventive", m, k)
            self.env.process(self._preventive(int(m), int(k)))
        return rejected

    def _close_day(self) -> tuple[int, int]:
        """Tire les quantités journalières à partir des espérances cumulées."""
        self._flush()
        rng, cfg = self.rng, self.cfg
        prod = np.rint(self._d_prod).astype(np.int64)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(prod > 0, self._d_expdef / np.maximum(self._d_prod, 1e-9), 0.0)
        p = np.clip(p, 0.0, 1.0)
        defects = rng.binomial(prod, p)
        inspected = prod // cfg.quality.inspection_every
        observed = rng.binomial(inspected, p)
        n_def = int(defects.sum())
        scrapped = int(rng.binomial(n_def, 1.0 - cfg.costs.rework_share)) if n_def else 0
        self.produced += int(prod.sum())
        self.defects += n_def
        self.scrapped += scrapped
        self.reworked += n_def - scrapped
        self.costs["rebut"] += scrapped * cfg.costs.scrap_cost
        self.costs["retouche"] += (n_def - scrapped) * cfg.costs.rework_cost

        with np.errstate(invalid="ignore", divide="ignore"):
            self.last_defect_rate = np.where(inspected > 0, observed / inspected, np.nan)
        n = self._d_run
        seen = n > 0
        if seen.any():
            tel = cfg.telemetry
            drift = np.einsum("mk,mks->ms", self._d_D3, self._sig_drift)
            n_safe = np.maximum(n, 1.0)[:, None]
            noise = rng.normal(0.0, 1.0, (self.M, 4)) * tel.noise / np.sqrt(n_safe)
            z = drift / n_safe + noise
            mean = self.sig_base + z * self.sig_sd
            self.last_signal_mean = np.where(seen[:, None], mean, self.last_signal_mean)
            self.last_signal_z = np.where(seen[:, None], z, self.last_signal_z)
        self.last_alarms = rng.poisson(self._d_lam).astype(float)

        self.stock_value_hours += self._d_stock_value
        self.costs["stockage"] += self._d_stock_value * cfg.costs.holding_rate / HOURS_PER_YEAR
        return int(prod.sum()), n_def

    def advance_day(self, risk: np.ndarray | None = None) -> DayRecord:
        """Simule une journée complète et renvoie son bilan."""
        costs_before = dict(self.costs)
        # Les décisions du matin sont journalisées avant l'appel : la journée commence au
        # premier événement qui suit la journée précédente.
        n_events = self._events_mark
        self.env.run(until=(self.day + 1) * 24.0)
        produced, defects = self._close_day()

        on_order = self.on_order()
        backorders = np.array([len(b) for b in self.backorders])
        day_costs = {k: self.costs[k] - costs_before[k] for k in COST_KEYS}
        record = DayRecord(
            day=self.day,
            states=self.state.copy(),
            stock=self.stock.copy(),
            on_order=on_order,
            backorders=backorders,
            costs=day_costs,
            produced=produced,
            defects=defects,
            open_hours=self._d_open,
            running_hours=self._d_running,
            events=self.events[n_events:],
        )
        self._events_mark = len(self.events)
        # État dominant de la journée : l'état hors service le plus long s'il a duré au moins
        # une heure, sinon « opérationnelle ».
        stopped = self._d_state_h[:, 1:]
        day_state = np.where(stopped.sum(axis=1) >= 1.0, stopped.argmax(axis=1) + 1, 0)
        h = self.history
        h["states"].append(record.states)
        h["day_state"].append(day_state.astype(np.int8))
        h["stock"].append(record.stock)
        h["on_order"].append(on_order)
        h["backorders"].append(backorders)
        h["costs"].append(day_costs)
        h["produced"].append(produced)
        h["defects"].append(defects)
        h["open_hours"].append(record.open_hours)
        h["running_hours"].append(record.running_hours)
        h["defect_rate"].append(self.last_defect_rate.copy())
        h["signal_mean"].append(self.last_signal_mean.copy())
        h["alarms"].append(self.last_alarms.copy())
        h["risk"].append(risk)
        self._reset_day_accumulators()
        self.day += 1
        return record

    def run(self, policy, days: int | None = None, on_day=None):
        """Simule ``days`` jours (par défaut l'horizon du scénario) sous une politique."""
        from spareparts.sim.results import build_result

        days = self.cfg.horizon_days if days is None else days
        policy.reset(self)
        for _ in range(days):
            obs = self.observe()
            self.apply(policy.decide(obs))
            record = self.advance_day()
            if on_day is not None:
                on_day(record)
        return build_result(self, policy)
