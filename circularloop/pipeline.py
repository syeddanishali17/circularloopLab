"""Single orchestration point: lifecycle record → validation → RUL → decision → explanation.

Every UI view calls :func:`assess_component`, so business logic is never duplicated
in the frontend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from circularloop.config import DEFAULT_GATES, GateThresholds
from circularloop.decision_engine import DecisionResult, decision_confidence, evaluate_strategies
from circularloop.explanation import Explanation, explain_decision
from circularloop.record_schema import is_missing, to_structured_record
from circularloop.rul_model import ModelBundle, RulPrediction, predict_rul
from circularloop.validation import STATUS_SUFFICIENT, ValidationReport, validate_record

INSUFFICIENT_MESSAGE = "Insufficient data for a reliable recommendation"


@dataclass
class Assessment:
    """Full result for one component under one set of settings."""

    record: dict[str, Any]  # as supplied
    record_used: dict[str, Any]  # after imputation of non-critical gaps
    validation: ValidationReport
    rul: RulPrediction | None
    decision: DecisionResult | None
    confidence: str | None
    explanation: Explanation | None

    @property
    def has_recommendation(self) -> bool:
        return self.decision is not None

    @property
    def recommended(self) -> str | None:
        return self.decision.recommended if self.decision else None


def impute_record(record: Mapping[str, Any], defaults: Mapping[str, Any], fields: list[str]) -> dict[str, Any]:
    """Fill the listed missing fields from training-fleet defaults."""
    out = dict(record)
    for name in fields:
        if is_missing(out.get(name)) and name in defaults:
            out[name] = defaults[name]
    return out


def assess_component(
    record: Mapping[str, Any],
    bundle: ModelBundle,
    weights: Mapping[str, float] | None = None,
    model: str | None = None,
    gates: GateThresholds = DEFAULT_GATES,
) -> Assessment:
    """Run the full pipeline. Records with critical issues get no RUL and no recommendation."""
    report = validate_record(record)
    if not report.is_usable:
        return Assessment(dict(record), dict(record), report, None, None, None, None)

    used = impute_record(record, bundle.feature_medians, report.imputed_fields)
    rul = predict_rul(bundle, used, model)
    decision = evaluate_strategies(used, rul.point, rul.lower, weights, gates)
    confidence = decision_confidence(decision.stability, report.quality_score, report.status == STATUS_SUFFICIENT)
    explanation = explain_decision(used, rul, decision, report)
    return Assessment(dict(record), used, report, rul, decision, confidence, explanation)


def decision_support_block(assessment: Assessment) -> dict[str, Any]:
    """The ``decision_support`` section of the lifecycle-record export."""
    if not assessment.has_recommendation:
        return {
            "status": INSUFFICIENT_MESSAGE,
            "data_quality_score": assessment.validation.quality_score,
            "critical_issues": [i.message for i in assessment.validation.critical],
        }
    rul, decision = assessment.rul, assessment.decision
    assert rul is not None and decision is not None
    return {
        "status": "Recommendation available (synthetic, illustrative)",
        "rul_model": rul.model,
        "predicted_rul_months": round(rul.point, 1),
        "rul_interval_80pct_months": [round(rul.lower, 1), round(rul.upper, 1)],
        "recommended_strategy": decision.recommended,
        "decision_score": round(decision.get(decision.recommended).score, 1),
        "runner_up": decision.runner_up,
        "decision_margin": round(decision.margin, 1),
        "weight_stability": round(decision.stability, 2),
        "decision_confidence": assessment.confidence,
        "data_quality_score": assessment.validation.quality_score,
        "mcda_weights": {k: round(v, 3) for k, v in decision.weights.items()},
        "blocked_strategies": {a.strategy: list(a.block_reasons) for a in decision.assessments if a.blocked},
    }


def assessment_to_record(assessment: Assessment) -> dict[str, Any]:
    """Sectioned lifecycle-record JSON including the decision-support section."""
    return to_structured_record(assessment.record, decision_support_block(assessment))
