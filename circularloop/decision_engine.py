"""Transparent circular-strategy decision layer.

For each strategy the engine estimates intervention cost, CO₂, and service life
delivered; derives a graded technical feasibility; applies hard technical gates;
normalises all criteria against a fixed "new component" reference; and ranks the
strategies with a weighted sum (MCDA). It never uses a trained classifier.

Two mechanisms keep unsafe options from winning:

1. **Hard gates** block a strategy outright (score 0), independent of weights.
2. **Graded feasibility** is a weighted criterion that rewards technically
   comfortable options among those that pass the gates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from circularloop.config import (
    COMPONENT_TYPES,
    CONFIDENCE_HIGH_STABILITY,
    CONFIDENCE_LOW_QUALITY,
    CONFIDENCE_LOW_STABILITY,
    CRITERIA,
    DEFAULT_GATES,
    DEFAULT_WEIGHTS,
    FEASIBILITY_WEIGHTS,
    NORMALISATION_RATIO_CAP,
    RECYCLE,
    RECYCLED_CONTENT_CO2_CREDIT,
    REFURBISH,
    REFURBISH_COST_PER_CORROSION_POINT,
    REPAIR,
    REPAIR_COST_PER_FAILURE,
    REPAIR_NO_SPARES_MULTIPLIER,
    REUSE,
    SCRAP_CREDIT_FRACTION,
    SENSITIVITY_CONCENTRATION,
    SENSITIVITY_DRAWS,
    SENSITIVITY_SEED,
    STRATEGIES,
    STRATEGY_CONDITION_SCALES,
    STRATEGY_ECONOMICS,
    SUBSCORE_SCALES,
    GateThresholds,
)

MIN_LIFE_MONTHS = 1.0  # floor to keep annualised values finite


@dataclass(frozen=True)
class StrategyAssessment:
    """Everything computed for one strategy."""

    strategy: str
    cost_eur: float
    co2_kg: float
    life_months: float
    cost_per_year: float
    co2_per_year: float
    feasibility: float
    circularity: float
    criteria: Mapping[str, float]  # normalised 0–1 scores
    contributions: Mapping[str, float]  # weight × score × 100
    score: float  # 0–100, 0 when blocked
    blocked: bool
    block_reasons: tuple[str, ...]


@dataclass
class DecisionResult:
    """Ranked outcome of the MCDA for one component."""

    assessments: list[StrategyAssessment]  # ranked best first
    weights: dict[str, float]
    recommended: str
    runner_up: str | None
    margin: float
    win_shares: dict[str, float] = field(default_factory=dict)

    @property
    def stability(self) -> float:
        return self.win_shares.get(self.recommended, 0.0)

    def get(self, strategy: str) -> StrategyAssessment:
        return next(a for a in self.assessments if a.strategy == strategy)

    def to_frame(self) -> pd.DataFrame:
        """Ranked comparison table (one row per strategy)."""
        rows = []
        for rank, a in enumerate(self.assessments, start=1):
            rows.append(
                {
                    "Rank": rank,
                    "Strategy": a.strategy,
                    "Cost (EUR)": a.cost_eur,
                    "CO₂ (kg)": a.co2_kg,
                    "Service life (months)": a.life_months,
                    "Cost / service-year (EUR)": a.cost_per_year,
                    "CO₂ / service-year (kg)": a.co2_per_year,
                    "Feasibility": a.feasibility,
                    "Circularity": a.circularity,
                    "Score": a.score,
                    "Status": "Blocked" if a.blocked else "Feasible",
                    "Blocking reason": "; ".join(a.block_reasons),
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def normalize_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    """Return non-negative weights over all criteria summing to 1 (defaults if all zero)."""
    raw = {c: max(0.0, float((weights or {}).get(c, 0.0))) for c in CRITERIA}
    total = sum(raw.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {c: v / total for c, v in raw.items()}


# ---------------------------------------------------------------------------
# Condition and feasibility
# ---------------------------------------------------------------------------


def _scale(value: float, zero_at: float, one_at: float) -> float:
    """Linear map so that ``zero_at`` → 0 and ``one_at`` → 1, clipped to [0, 1]."""
    if one_at == zero_at:
        return 1.0
    return float(np.clip((value - zero_at) / (one_at - zero_at), 0.0, 1.0))


def condition_subscores(record: Mapping[str, Any], conservative_rul: float, strategy: str) -> dict[str, float]:
    """Condition indicators mapped to 0 (poor / at the gate) … 1 (comfortable) for one strategy."""
    spec = COMPONENT_TYPES[record["component_type"]]
    scales = {**SUBSCORE_SCALES, **STRATEGY_CONDITION_SCALES.get(strategy, {})}
    values = {
        "thickness": float(record["residual_thickness_pct"]),
        "corrosion": float(record["corrosion_score"]),
        "failures": float(record["failure_events"]),
        "vibration": float(record["vibration_rms"]) / spec.vibration_alarm_mm_s,
        "rul": conservative_rul / spec.design_life_months,
    }
    scores = {k: _scale(v, *scales[k]) for k, v in values.items() if k in scales}
    scores["spares"] = 1.0 if bool(record["spare_parts_available"]) else 0.3
    return scores


def technical_feasibility(strategy: str, record: Mapping[str, Any], conservative_rul: float) -> float:
    """Graded feasibility (0–1) as a weighted mean of relevant condition sub-scores."""
    if strategy == RECYCLE:
        return 1.0
    subscores = condition_subscores(record, conservative_rul, strategy)
    weights = FEASIBILITY_WEIGHTS[strategy]
    return float(sum(w * subscores[k] for k, w in weights.items()) / sum(weights.values()))


@dataclass(frozen=True)
class GateCheck:
    """One evaluated hard-gate rule."""

    strategy: str
    criterion: str
    observed: float
    limit: float
    kind: str  # "min" (observed must be ≥ limit) or "max" (observed must be ≤ limit)
    unit: str

    @property
    def passed(self) -> bool:
        return self.observed >= self.limit if self.kind == "min" else self.observed <= self.limit

    def describe(self) -> str:
        op = "<" if self.kind == "min" else ">"
        return f"{self.criterion} {_fmt(self.observed, self.unit)} {op} {_fmt(self.limit, self.unit)}"


def _fmt(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.0f}%"
    if unit == "months":
        return f"{value:.0f} months"
    if unit == "×":
        return f"{value:.1f}×"
    if unit == "count":
        return f"{value:.0f}"
    return f"{value:.2f}" if unit == "score" else f"{value:.1f}"


def gate_checks(
    strategy: str,
    record: Mapping[str, Any],
    conservative_rul: float,
    feasibility: float,
    gates: GateThresholds = DEFAULT_GATES,
) -> list[GateCheck]:
    """All hard-gate rules for one strategy, evaluated on the record (Recycle has none)."""
    if strategy == RECYCLE:
        return []
    thk = float(record["residual_thickness_pct"])
    corr = float(record["corrosion_score"])
    fails = float(record["failure_events"])
    vib_ratio = float(record["vibration_rms"]) / COMPONENT_TYPES[record["component_type"]].vibration_alarm_mm_s

    def g(criterion: str, observed: float, limit: float, kind: str, unit: str) -> GateCheck:
        return GateCheck(strategy, criterion, observed, limit, kind, unit)

    prefix = strategy.lower()
    rules: list[GateCheck] = [
        g("residual thickness", thk, getattr(gates, f"{prefix}_min_thickness_pct"), "min", "%"),
        g("corrosion", corr, getattr(gates, f"{prefix}_max_corrosion"), "max", "/10"),
    ]
    if strategy == REUSE:
        rules += [
            g("conservative RUL", conservative_rul, gates.reuse_min_conservative_rul_months, "min", "months"),
            g("failure events", fails, gates.reuse_max_failures, "max", "count"),
            g("vibration vs. alarm level", vib_ratio, gates.reuse_max_vibration_alarm_ratio, "max", "×"),
        ]
    elif strategy == REPAIR:
        rules.append(g("failure events", fails, gates.repair_max_failures, "max", "count"))
    rules.append(g("technical feasibility", feasibility, gates.min_feasibility, "min", "score"))
    return rules


def gate_violations(
    strategy: str,
    record: Mapping[str, Any],
    conservative_rul: float,
    feasibility: float,
    gates: GateThresholds = DEFAULT_GATES,
) -> list[str]:
    """Human-readable list of hard-gate violations (empty → strategy allowed)."""
    return [c.describe() for c in gate_checks(strategy, record, conservative_rul, feasibility, gates) if not c.passed]


# ---------------------------------------------------------------------------
# Cost, CO2, life
# ---------------------------------------------------------------------------


def strategy_estimates(strategy: str, record: Mapping[str, Any], rul_point: float) -> tuple[float, float, float]:
    """Illustrative (cost EUR, CO₂ kg, service life months) of one strategy."""
    econ = STRATEGY_ECONOMICS[strategy]
    spec = COMPONENT_TYPES[record["component_type"]]
    design = spec.design_life_months
    replacement = float(record["replacement_cost_eur"])
    embodied = float(record["embodied_co2_kg"])
    recyclability = float(record["recyclability_pct"]) / 100.0

    cost = econ.cost_fraction * replacement
    co2 = econ.co2_fraction * embodied
    if strategy == REPAIR:
        if not bool(record["spare_parts_available"]):
            cost *= REPAIR_NO_SPARES_MULTIPLIER
        cost *= 1.0 + REPAIR_COST_PER_FAILURE * float(record["failure_events"])
    elif strategy == REFURBISH:
        cost *= 1.0 + REFURBISH_COST_PER_CORROSION_POINT * float(record["corrosion_score"])
    elif strategy == RECYCLE:
        cost -= SCRAP_CREDIT_FRACTION * replacement * recyclability
        co2 *= 1.0 - RECYCLED_CONTENT_CO2_CREDIT * recyclability

    if econ.life_mode == "rul":
        life = rul_point
    elif econ.life_mode == "restore":
        # An intervention can only recover life that has actually been consumed.
        consumed = max(design - rul_point, 0.0)
        life = rul_point + econ.life_fraction * consumed
    else:
        life = econ.life_fraction * design
    life = float(np.clip(life, MIN_LIFE_MONTHS, design))
    return float(cost), float(co2), life


def _circularity(strategy: str, record: Mapping[str, Any]) -> float:
    base = STRATEGY_ECONOMICS[strategy].circularity
    if strategy == RECYCLE:
        return base * float(record["recyclability_pct"]) / 100.0
    return base


def _ratio_score(value_per_year: float, reference_per_year: float) -> float:
    """1 at zero, 0.5 at parity with a new component, 0 at ``NORMALISATION_RATIO_CAP`` × new."""
    if reference_per_year <= 0:
        return 0.0
    return float(np.clip(1.0 - (value_per_year / reference_per_year) / NORMALISATION_RATIO_CAP, 0.0, 1.0))


# ---------------------------------------------------------------------------
# MCDA
# ---------------------------------------------------------------------------


def _assess_strategy(
    strategy: str,
    record: Mapping[str, Any],
    rul_point: float,
    rul_conservative: float,
    weights: Mapping[str, float],
    gates: GateThresholds,
) -> StrategyAssessment:
    design = COMPONENT_TYPES[record["component_type"]].design_life_months
    years_new = design / 12.0
    cost, co2, life = strategy_estimates(strategy, record, rul_point)
    cost_py, co2_py = cost / (life / 12.0), co2 / (life / 12.0)

    feas = technical_feasibility(strategy, record, rul_conservative)
    circ = _circularity(strategy, record)
    criteria = {
        "feasibility": feas,
        "cost": _ratio_score(cost_py, float(record["replacement_cost_eur"]) / years_new),
        "co2": _ratio_score(co2_py, float(record["embodied_co2_kg"]) / years_new),
        "life": life / design,
        "circularity": circ,
    }
    reasons = tuple(gate_violations(strategy, record, rul_conservative, feas, gates))
    blocked = bool(reasons)
    contributions = {c: 0.0 if blocked else 100.0 * weights[c] * criteria[c] for c in CRITERIA}
    return StrategyAssessment(
        strategy=strategy,
        cost_eur=cost,
        co2_kg=co2,
        life_months=life,
        cost_per_year=cost_py,
        co2_per_year=co2_py,
        feasibility=feas,
        circularity=circ,
        criteria=criteria,
        contributions=contributions,
        score=sum(contributions.values()),
        blocked=blocked,
        block_reasons=reasons,
    )


def _rank(assessments: list[StrategyAssessment]) -> list[StrategyAssessment]:
    """Feasible first by score; ties broken by the resource-retention order of STRATEGIES."""
    order = {s: i for i, s in enumerate(STRATEGIES)}
    return sorted(assessments, key=lambda a: (a.blocked, -round(a.score, 9), order[a.strategy]))


def weight_sensitivity(
    assessments: list[StrategyAssessment],
    weights: Mapping[str, float],
    draws: int = SENSITIVITY_DRAWS,
    concentration: float = SENSITIVITY_CONCENTRATION,
    seed: int = SENSITIVITY_SEED,
) -> dict[str, float]:
    """Share of perturbed weight vectors (Dirichlet around ``weights``) won by each strategy."""
    rng = np.random.default_rng(seed)
    alpha = np.maximum(np.array([weights[c] for c in CRITERIA]) * concentration, 1e-3)
    sampled = rng.dirichlet(alpha, size=draws)  # draws × criteria
    matrix = np.array([[a.criteria[c] for c in CRITERIA] for a in assessments])  # strategies × criteria
    scores = sampled @ matrix.T
    blocked = np.array([a.blocked for a in assessments])
    scores[:, blocked] = -np.inf
    winners = scores.argmax(axis=1)
    counts = np.bincount(winners, minlength=len(assessments))
    return {a.strategy: float(counts[i] / draws) for i, a in enumerate(assessments)}


def evaluate_strategies(
    record: Mapping[str, Any],
    rul_point: float,
    rul_conservative: float,
    weights: Mapping[str, float] | None = None,
    gates: GateThresholds = DEFAULT_GATES,
) -> DecisionResult:
    """Evaluate and rank all strategies for one complete (validated/imputed) record."""
    w = normalize_weights(weights if weights is not None else DEFAULT_WEIGHTS)
    assessed = [_assess_strategy(s, record, rul_point, rul_conservative, w, gates) for s in STRATEGIES]
    ranked = _rank(assessed)
    best = ranked[0]
    second = next((a for a in ranked[1:] if not a.blocked), None)
    return DecisionResult(
        assessments=ranked,
        weights=w,
        recommended=best.strategy,
        runner_up=second.strategy if second else None,
        margin=best.score - second.score if second else best.score,
        win_shares=weight_sensitivity(ranked, w),
    )


def decision_confidence(stability: float, quality_score: float, quality_sufficient: bool) -> str:
    """Combine weight-robustness and data quality into High / Medium / Low."""
    if stability < CONFIDENCE_LOW_STABILITY or quality_score < CONFIDENCE_LOW_QUALITY:
        return "Low"
    if quality_sufficient and stability >= CONFIDENCE_HIGH_STABILITY:
        return "High"
    return "Medium"
