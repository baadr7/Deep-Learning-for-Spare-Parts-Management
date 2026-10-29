"""Paramètres d'un scénario de simulation (cahier des charges, chapitre 4).

Un scénario est décrit dans un fichier YAML de ``configs/scenarios`` puis validé par les
modèles ci-dessous. Les durées sont en heures et les coûts en unités relatives (UR).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = ROOT / "configs" / "scenarios"

Regime = Literal["usage", "calendar"]
Signal = Literal["volt", "rotate", "pressure", "vibration"]


class Lognormal(BaseModel):
    """Loi lognormale décrite par sa moyenne et son coefficient de variation."""

    mean: float = Field(gt=0)
    cv: float = Field(default=0.3, ge=0)


class Triangular(BaseModel):
    low: float = Field(ge=0)
    mode: float = Field(ge=0)
    high: float = Field(ge=0)


class ComponentSpec(BaseModel):
    name: str
    part: int = Field(ge=0, description="indice de la référence de pièce consommée")
    regime: Regime
    weibull_beta: float = Field(gt=0)
    weibull_eta: float = Field(gt=0, description="heures de marche (usage) ou calendaires")
    pm_interval: float = Field(gt=0, description="même unité que weibull_eta")
    repair: Lognormal
    pm_ratio: float = Field(default=0.6, gt=0, description="durée préventive / corrective")
    signal: Signal
    drift: float = Field(description="dérive du signal en fin de vie, en écarts-types")
    defect_type: str
    defect_weight: float = Field(ge=0, description="contribution a_k au taux de défauts")


class FamilySpec(BaseModel):
    label: str
    shifts: list[int] = Field(min_length=1)
    cadence: float = Field(gt=0, description="pièces produites par heure de marche")
    margin_per_part: float = Field(ge=0)
    overhaul_years: list[float] | None = None
    overhaul_days: Lognormal | None = None
    overhaul_cost: float = 0.0
    components: list[ComponentSpec] = Field(min_length=1)


class PartSpec(BaseModel):
    code: str
    name: str
    nature: str
    unit_cost: float = Field(gt=0)
    lead_days: float = Field(gt=0)
    lead_cv: float = Field(default=0.3, ge=0)
    urgent_lead_days: float = Field(gt=0)
    s: int = Field(ge=0, description="seuil de commande")
    S: int = Field(ge=0, description="niveau de recomplètement")
    initial_stock: int = Field(ge=0)


class FleetSpec(BaseModel):
    n_machines: int = Field(default=100, gt=0)
    family_share: dict[str, float]
    age_years: tuple[float, float] = (0.0, 20.0)


class CostSpec(BaseModel):
    holding_rate: float = 0.25
    order_fixed: float = 2.0
    urgent_factor: float = 2.5
    labor_per_hour: float = 0.5
    corrective_ratio: float = 2.0
    scrap_cost: float = 0.1
    rework_cost: float = 0.03
    rework_share: float = Field(default=0.7, ge=0, le=1)


class QualitySpec(BaseModel):
    p0: float = Field(default=0.01, ge=0, le=1)
    gamma: float = Field(default=3.0, gt=0)
    inspection_every: int = Field(default=20, gt=0)
    trial_batch: int = Field(default=5, gt=0)


class DurationSpec(BaseModel):
    diagnosis: Lognormal = Lognormal(mean=1.0, cv=0.5)
    store_issue: Triangular = Triangular(low=0.25, mode=0.5, high=1.0)
    test: Triangular = Triangular(low=1.0, mode=1.5, high=2.0)
    rework_fraction: float = 0.3
    order_processing: float = 4.0
    reception: Triangular = Triangular(low=1.0, mode=2.0, high=3.0)


class TelemetrySpec(BaseModel):
    """Niveaux de base inspirés des moyennes et écarts-types du jeu Azure."""

    mean: dict[Signal, float] = Field(
        default_factory=lambda: {
            "volt": 170.0,
            "rotate": 446.0,
            "pressure": 100.0,
            "vibration": 40.0,
        }
    )
    sd: dict[Signal, float] = Field(
        default_factory=lambda: {"volt": 15.0, "rotate": 52.0, "pressure": 11.0, "vibration": 5.0}
    )
    noise: float = 0.6
    machine_offset: float = 0.3
    drift_cv: float = Field(
        default=0.4, ge=0, description="variabilité de la dérive d'un composant"
    )
    alarm_base_per_hour: float = 0.003
    alarm_slope_per_hour: float = 0.03


class ScenarioConfig(BaseModel):
    name: str
    description: str = ""
    horizon_days: int = Field(default=365, gt=0)
    seed: int = 42
    technicians: int = Field(default=3, gt=0)
    failure_rate_factor: float = Field(default=1.0, gt=0, description="multiplie le taux de panne")
    lead_time_factor: float = Field(
        default=1.0, gt=0, description="multiplie les délais fournisseur"
    )
    fleet: FleetSpec
    families: dict[str, FamilySpec]
    parts: list[PartSpec] = Field(min_length=1)
    costs: CostSpec = CostSpec()
    quality: QualitySpec = QualitySpec()
    durations: DurationSpec = DurationSpec()
    telemetry: TelemetrySpec = TelemetrySpec()

    @model_validator(mode="after")
    def _check_consistency(self) -> ScenarioConfig:
        sizes = {len(f.components) for f in self.families.values()}
        if len(sizes) != 1:
            raise ValueError("toutes les familles doivent avoir le même nombre de composants")
        for code, family in self.families.items():
            for comp in family.components:
                if comp.part >= len(self.parts):
                    raise ValueError(f"famille {code} : référence {comp.part} inexistante")
        unknown = set(self.fleet.family_share) - set(self.families)
        if unknown:
            raise ValueError(f"familles inconnues dans family_share : {sorted(unknown)}")
        return self

    @property
    def n_components(self) -> int:
        return len(next(iter(self.families.values())).components)


def list_scenarios() -> dict[str, str]:
    """Identifiant de fichier vers nom lisible, pour chaque scénario disponible."""
    out = {}
    for path in sorted(SCENARIO_DIR.glob("*.yaml")):
        with path.open(encoding="utf-8") as fh:
            out[path.stem] = yaml.safe_load(fh).get("name", path.stem)
    return out


def load_scenario(name_or_path: str | Path, **overrides) -> ScenarioConfig:
    """Charge un scénario par identifiant (``s1_nominal``) ou par chemin, avec surcharges."""
    path = Path(name_or_path)
    if not path.suffix:
        path = SCENARIO_DIR / f"{name_or_path}.yaml"
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return ScenarioConfig.model_validate(raw)
