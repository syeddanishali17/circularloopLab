"""Central configuration: every synthetic assumption, threshold and default lives here.

All numbers in this module are **illustrative assumptions for a synthetic demonstrator**.
They are chosen to be plausible in order of magnitude, not taken from any validated
engineering standard, LCA database or industrial partner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
REFERENCE_YEAR: int = 2026
N_COMPONENTS: int = 500
TEST_SIZE: float = 0.25
CV_FOLDS: int = 5
SENSITIVITY_SEED: int = 7
SENSITIVITY_DRAWS: int = 400
SENSITIVITY_CONCENTRATION: float = 30.0

HOURS_PER_YEAR: float = 8_760.0

# ---------------------------------------------------------------------------
# Component types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentTypeSpec:
    """Illustrative design parameters of one component family."""

    design_life_years: float
    nominal_hours_per_year: float
    load_cycles_per_hour: float
    vibration_base_mm_s: float
    vibration_alarm_mm_s: float
    temperature_nominal_c: float
    temperature_limit_c: float
    mass_range_kg: tuple[float, float]
    cost_per_kg_eur: float
    materials: Mapping[str, float] = field(default_factory=dict)

    @property
    def design_life_months(self) -> float:
        return self.design_life_years * 12.0


COMPONENT_TYPES: Mapping[str, ComponentTypeSpec] = MappingProxyType(
    {
        "Rail bogie frame": ComponentTypeSpec(
            design_life_years=30,
            nominal_hours_per_year=4_000,
            load_cycles_per_hour=30,
            vibration_base_mm_s=1.6,
            vibration_alarm_mm_s=3.5,
            temperature_nominal_c=22,
            temperature_limit_c=60,
            mass_range_kg=(1_400, 3_000),
            cost_per_kg_eur=38,
            materials={"High-strength steel": 0.85, "Stainless steel": 0.15},
        ),
        "Industrial gearbox": ComponentTypeSpec(
            design_life_years=20,
            nominal_hours_per_year=6_000,
            load_cycles_per_hour=15,
            vibration_base_mm_s=2.4,
            vibration_alarm_mm_s=4.5,
            temperature_nominal_c=62,
            temperature_limit_c=80,
            mass_range_kg=(250, 2_500),
            cost_per_kg_eur=24,
            materials={"Cast iron": 0.55, "High-strength steel": 0.45},
        ),
        "Drive shaft": ComponentTypeSpec(
            design_life_years=15,
            nominal_hours_per_year=5_000,
            load_cycles_per_hour=25,
            vibration_base_mm_s=2.0,
            vibration_alarm_mm_s=4.5,
            temperature_nominal_c=42,
            temperature_limit_c=70,
            mass_range_kg=(60, 450),
            cost_per_kg_eur=45,
            materials={"High-strength steel": 0.80, "Stainless steel": 0.20},
        ),
        "Steel structural profile": ComponentTypeSpec(
            design_life_years=40,
            nominal_hours_per_year=8_000,
            load_cycles_per_hour=2,
            vibration_base_mm_s=0.6,
            vibration_alarm_mm_s=1.8,
            temperature_nominal_c=18,
            temperature_limit_c=55,
            mass_range_kg=(150, 1_500),
            cost_per_kg_eur=9,
            materials={"High-strength steel": 0.75, "Stainless steel": 0.25},
        ),
        "Electric motor housing": ComponentTypeSpec(
            design_life_years=25,
            nominal_hours_per_year=5_000,
            load_cycles_per_hour=8,
            vibration_base_mm_s=1.8,
            vibration_alarm_mm_s=3.5,
            temperature_nominal_c=52,
            temperature_limit_c=75,
            mass_range_kg=(40, 600),
            cost_per_kg_eur=30,
            materials={"Aluminium alloy": 0.50, "Cast iron": 0.40, "Stainless steel": 0.10},
        ),
    }
)


@dataclass(frozen=True)
class MaterialSpec:
    """Illustrative material properties (order-of-magnitude, not LCA data)."""

    corrosion_rate_per_year: float  # corrosion-score points per year in a neutral environment
    co2_kg_per_kg: float  # cradle-to-gate factor incl. processing overhead (illustrative)
    cost_multiplier: float
    recyclability_range_pct: tuple[float, float]


MATERIALS: Mapping[str, MaterialSpec] = MappingProxyType(
    {
        "High-strength steel": MaterialSpec(0.30, 2.6, 1.00, (88, 97)),
        "Cast iron": MaterialSpec(0.33, 2.2, 0.85, (85, 95)),
        "Stainless steel": MaterialSpec(0.08, 6.2, 1.60, (86, 95)),
        "Aluminium alloy": MaterialSpec(0.12, 11.0, 1.40, (78, 93)),
    }
)

# ---------------------------------------------------------------------------
# Model features (economic / identity fields are deliberately excluded)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: tuple[str, ...] = (
    "age_years",
    "operating_hours",
    "load_cycles",
    "vibration_rms",
    "temperature_mean_c",
    "corrosion_score",
    "residual_thickness_pct",
    "maintenance_count",
    "failure_events",
    "last_maintenance_months_ago",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("component_type", "material")
MODEL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET: str = "rul_months"

MODEL_LINEAR = "Linear Regression"
MODEL_GBR = "Gradient Boosting"
MODEL_NAMES: tuple[str, ...] = (MODEL_LINEAR, MODEL_GBR)
MODEL_GBR_UNCONSTRAINED = "Gradient Boosting (unconstrained, ablation)"

GBR_PARAMS: Mapping[str, object] = MappingProxyType(
    {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_samples_leaf": 10,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": RANDOM_SEED,
    }
)
# Engineering priors encoded as monotonic constraints (−1: RUL can only fall as the
# feature rises; +1: can only rise; omitted: unconstrained). They prevent the model
# from learning confounded patterns such as "longer since maintenance → longer life",
# which arise because young components are rarely maintained.
MONOTONIC_CONSTRAINTS: Mapping[str, int] = MappingProxyType(
    {
        "age_years": -1,
        "operating_hours": -1,
        "load_cycles": -1,
        "vibration_rms": -1,
        "temperature_mean_c": -1,
        "corrosion_score": -1,
        "residual_thickness_pct": 1,
        "maintenance_count": 1,
        "failure_events": -1,
        "last_maintenance_months_ago": -1,
    }
)
PREDICTION_INTERVAL: tuple[float, float] = (0.10, 0.90)
# Residuals are divided by (prediction + offset) so interval width scales with predicted life.
INTERVAL_SCALE_OFFSET_MONTHS: float = 12.0

# ---------------------------------------------------------------------------
# Decision layer
# ---------------------------------------------------------------------------
REUSE, REPAIR, REFURBISH, REMANUFACTURE, RECYCLE = (
    "Reuse",
    "Repair",
    "Refurbish",
    "Remanufacture",
    "Recycle",
)
STRATEGIES: tuple[str, ...] = (REUSE, REPAIR, REFURBISH, REMANUFACTURE, RECYCLE)

STRATEGY_DEFINITIONS: Mapping[str, str] = MappingProxyType(
    {
        REUSE: "Return to service as-is after inspection and requalification.",
        REPAIR: "Targeted correction of a specific defect; wall thickness is not restored.",
        REFURBISH: "Restore to good working condition (surface treatment, wear parts); not to as-new specification.",
        REMANUFACTURE: "Disassemble and restore the core to as-new specification.",
        RECYCLE: "Recover the material; the function is provided by a new replacement component.",
    }
)

CRITERIA: tuple[str, ...] = ("feasibility", "co2", "cost", "life", "circularity")
CRITERION_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "feasibility": "Technical feasibility",
        "co2": "CO₂ per service-year",
        "cost": "Cost per service-year",
        "life": "Service life delivered",
        "circularity": "Circularity",
    }
)
DEFAULT_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {"feasibility": 0.30, "co2": 0.20, "cost": 0.20, "life": 0.20, "circularity": 0.10}
)


@dataclass(frozen=True)
class StrategyEconomics:
    """Intervention cost/CO2 as fractions of the new component, plus life model."""

    cost_fraction: float
    co2_fraction: float
    life_mode: str  # "rul" (as is), "restore" (recover part of consumed life), "fixed" (share of design life)
    life_fraction: float  # restore: share of consumed life recovered; fixed: share of design life
    circularity: float


STRATEGY_ECONOMICS: Mapping[str, StrategyEconomics] = MappingProxyType(
    {
        REUSE: StrategyEconomics(0.04, 0.01, "rul", 0.0, 1.00),
        REPAIR: StrategyEconomics(0.12, 0.05, "restore", 0.25, 0.90),
        REFURBISH: StrategyEconomics(0.28, 0.12, "restore", 0.55, 0.80),
        REMANUFACTURE: StrategyEconomics(0.55, 0.28, "fixed", 0.85, 0.65),
        RECYCLE: StrategyEconomics(1.00, 1.00, "fixed", 1.00, 0.40),
    }
)
REPAIR_NO_SPARES_MULTIPLIER: float = 1.6
REPAIR_COST_PER_FAILURE: float = 0.05
REFURBISH_COST_PER_CORROSION_POINT: float = 0.04
SCRAP_CREDIT_FRACTION: float = 0.05  # of replacement cost, scaled by recyclability
RECYCLED_CONTENT_CO2_CREDIT: float = 0.30  # of embodied CO2, scaled by recyclability
NORMALISATION_RATIO_CAP: float = 2.0  # annualised ratio vs. new at which score reaches 0


@dataclass(frozen=True)
class GateThresholds:
    """Hard technical gates. A violated gate blocks a strategy regardless of weights."""

    reuse_min_thickness_pct: float = 80.0
    reuse_max_corrosion: float = 5.0
    reuse_min_conservative_rul_months: float = 12.0
    reuse_max_failures: int = 2
    reuse_max_vibration_alarm_ratio: float = 1.5
    repair_min_thickness_pct: float = 65.0
    repair_max_corrosion: float = 7.0
    repair_max_failures: int = 4
    refurbish_min_thickness_pct: float = 60.0
    refurbish_max_corrosion: float = 8.5
    remanufacture_min_thickness_pct: float = 45.0
    remanufacture_max_corrosion: float = 9.5
    min_feasibility: float = 0.20


DEFAULT_GATES = GateThresholds()

# Strategy-specific weights of condition sub-scores in graded technical feasibility.
FEASIBILITY_WEIGHTS: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        REUSE: {"thickness": 0.30, "corrosion": 0.25, "rul": 0.20, "failures": 0.15, "vibration": 0.10},
        REPAIR: {"spares": 0.30, "thickness": 0.20, "corrosion": 0.20, "failures": 0.20, "rul": 0.10},
        REFURBISH: {"thickness": 0.40, "corrosion": 0.25, "failures": 0.15, "spares": 0.10, "rul": 0.10},
        REMANUFACTURE: {"thickness": 0.55, "corrosion": 0.30, "failures": 0.15},
    }
)
# Sub-score scales: (value at which the sub-score is 0, value at which it is 1).
SUBSCORE_SCALES: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "failures": (6.0, 0.0),  # failure events
        "vibration": (2.0, 0.5),  # vibration / alarm level
        "rul": (0.0, 0.30),  # conservative RUL / design life
    }
)
# Thickness and corrosion are scored as the *margin to each strategy's own gate*:
# zero at the gate limit, one where the condition is comfortably inside it.
STRATEGY_CONDITION_SCALES: Mapping[str, Mapping[str, tuple[float, float]]] = MappingProxyType(
    {
        REUSE: {"thickness": (80.0, 97.0), "corrosion": (5.0, 0.5)},
        REPAIR: {"thickness": (65.0, 92.0), "corrosion": (7.0, 1.0)},
        REFURBISH: {"thickness": (60.0, 88.0), "corrosion": (8.5, 2.0)},
        REMANUFACTURE: {"thickness": (45.0, 78.0), "corrosion": (9.5, 3.0)},
    }
)

# Decision confidence
CONFIDENCE_HIGH_STABILITY: float = 0.80
CONFIDENCE_LOW_STABILITY: float = 0.55
CONFIDENCE_LOW_QUALITY: float = 60.0

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
CRITICAL_FIELDS: tuple[str, ...] = (
    "component_type",
    "material",
    "age_years",
    "operating_hours",
    "corrosion_score",
    "residual_thickness_pct",
    "replacement_cost_eur",
    "embodied_co2_kg",
)
QUALITY_WARNING_PENALTY: float = 8.0
QUALITY_IMPUTATION_PENALTY: float = 5.0  # per imputed non-critical field, on top of completeness
QUALITY_SUFFICIENT: float = 85.0
CRITICAL_FIELD_WEIGHT: float = 3.0

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
SCENARIO_CURRENT = "Current state"
SCENARIO_IMPROVED = "Improved maintenance"
SCENARIO_SEVERE = "Severe degradation"
SCENARIOS: tuple[str, ...] = (SCENARIO_CURRENT, SCENARIO_IMPROVED, SCENARIO_SEVERE)
