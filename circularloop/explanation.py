"""Deterministic, template-based explanations built from actual values and scores.

No language model is involved: every sentence is assembled from the component
record, the RUL prediction and the per-strategy MCDA results, so the text changes
whenever the data or the weights change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from circularloop.config import (
    COMPONENT_TYPES,
    CRITERIA,
    RECYCLE,
    REFURBISH,
    REMANUFACTURE,
    REPAIR,
    REUSE,
)
from circularloop.decision_engine import DecisionResult, StrategyAssessment
from circularloop.rul_model import RulPrediction
from circularloop.validation import ValidationReport

NOUNS: Mapping[str, str] = {
    REUSE: "direct reuse",
    REPAIR: "repair",
    REFURBISH: "refurbishment",
    REMANUFACTURE: "remanufacturing",
    RECYCLE: "recycling with replacement",
}
CLOSE_CALL_POINTS = 3.0
NOTABLE_ALTERNATIVE_SHARE = 0.15


@dataclass
class Explanation:
    """Structured explanation; ``paragraph`` is the plain-English summary."""

    headline: str
    paragraph: str
    strengths: list[str] = field(default_factory=list)
    trade_offs: list[str] = field(default_factory=list)
    others: list[str] = field(default_factory=list)
    robustness: str = ""
    data_note: str = ""


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _criterion_phrase(criterion: str, a: StrategyAssessment, ref: StrategyAssessment | None, design_m: float) -> str:
    """Phrase describing why ``a`` does well on ``criterion`` (optionally compared to ``ref``)."""
    vs = f" for {NOUNS[ref.strategy]}" if ref else ""
    if criterion == "feasibility":
        return f"it has the higher technical feasibility ({a.feasibility:.2f}" + (
            f" vs. {ref.feasibility:.2f}{vs})" if ref else ")"
        )
    if criterion == "life":
        share = a.life_months / design_m
        return f"it delivers about {_months(a.life_months)} of service ({share:.0%} of a new part's design life)"
    if criterion == "cost":
        return f"its cost per service-year is lower (≈ €{a.cost_per_year:,.0f}" + (
            f" vs. €{ref.cost_per_year:,.0f}{vs})" if ref else ")"
        )
    if criterion == "co2":
        return f"its carbon per service-year is lower (≈ {a.co2_per_year:,.0f} kg CO₂e" + (
            f" vs. {ref.co2_per_year:,.0f} kg{vs})" if ref else ")"
        )
    return f"it retains more of the existing product (circularity {a.circularity:.2f}" + (
        f" vs. {ref.circularity:.2f}{vs})" if ref else ")"
    )


def _months(value: float) -> str:
    n = round(value)
    return f"{n} month" if n == 1 else f"{n} months"


def _contribution_gaps(a: StrategyAssessment, b: StrategyAssessment) -> dict[str, float]:
    return {c: a.contributions[c] - b.contributions[c] for c in CRITERIA}


def _condition_sentence(record: Mapping[str, Any], rul: RulPrediction) -> str:
    design_y = COMPONENT_TYPES[record["component_type"]].design_life_years
    share = rul.point / (design_y * 12)
    level = "substantial" if share >= 0.4 else "moderate" if share >= 0.15 else "limited"
    return (
        f"The {record['component_type'].lower()} is predicted to have about {_months(rul.point)} of remaining "
        f"useful life (approx. 80% range {rul.lower:.0f}–{rul.upper:.0f}), which is {level} relative to a "
        f"{design_y:.0f}-year design life. Corrosion is {float(record['corrosion_score']):.1f}/10 and residual "
        f"thickness {float(record['residual_thickness_pct']):.0f}%."
    )


def _blocked_sentence(a: StrategyAssessment) -> str:
    return f"{_cap(NOUNS[a.strategy])} is ruled out by hard technical gates: {', '.join(a.block_reasons)}."


def _weaker_sentence(a: StrategyAssessment, winner: StrategyAssessment, design_m: float) -> str:
    gaps = _contribution_gaps(winner, a)
    worst = max(gaps, key=gaps.get)  # criterion where it loses most to the winner
    detail = {
        "feasibility": f"lower technical feasibility ({a.feasibility:.2f})",
        "life": f"shorter service life (≈ {_months(a.life_months)})",
        "cost": f"higher cost per service-year (≈ €{a.cost_per_year:,.0f})",
        "co2": f"higher carbon per service-year (≈ {a.co2_per_year:,.0f} kg)",
        "circularity": f"lower resource retention (circularity {a.circularity:.2f})",
    }[worst]
    return f"{_cap(NOUNS[a.strategy])} scores {a.score:.0f}/100, held back mainly by {detail}."


def explain_decision(
    record: Mapping[str, Any], rul: RulPrediction, decision: DecisionResult, report: ValidationReport
) -> Explanation:
    """Build the explanation for a ranked decision."""
    design_m = COMPONENT_TYPES[record["component_type"]].design_life_months
    winner = decision.get(decision.recommended)
    runner = decision.get(decision.runner_up) if decision.runner_up else None

    # Strengths: where the winner gains most weighted points over the runner-up.
    if runner is not None:
        gaps = _contribution_gaps(winner, runner)
        top = [c for c in sorted(gaps, key=gaps.get, reverse=True) if gaps[c] > 0.5][:2]
        losses = [c for c in sorted(gaps, key=gaps.get) if gaps[c] < -0.5][:1]
    else:
        top = sorted(CRITERIA, key=lambda c: winner.contributions[c], reverse=True)[:2]
        losses = []
    strengths = [_criterion_phrase(c, winner, runner, design_m) for c in top]
    trade_offs = (
        [
            f"{_cap(NOUNS[runner.strategy])} would be better on {_criterion_label(c)}, "
            f"but not by enough to outweigh these advantages."
            for c in losses
        ]
        if runner is not None
        else []
    )

    blocked = [a for a in decision.assessments if a.blocked]
    feasible_others = [a for a in decision.assessments if not a.blocked and a.strategy != winner.strategy]
    others = [_blocked_sentence(a) for a in blocked] + [_weaker_sentence(a, winner, design_m) for a in feasible_others]

    headline = f"{winner.strategy} ranks highest ({winner.score:.0f}/100)"
    because = " and ".join(strengths) if strengths else "it is the only option that passes the technical gates"
    paragraph = f"{_cap(NOUNS[winner.strategy])} ranks highest because {because}."
    if blocked:
        items = [f"{NOUNS[a.strategy]} ({a.block_reasons[0]})" for a in blocked]
        listed = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
        verb = "is" if len(items) == 1 else "are"
        paragraph += f" {_cap(listed)} {verb} blocked by hard technical gates."
    if trade_offs:
        paragraph += " " + trade_offs[0]

    robustness = _robustness_sentence(decision)
    data_note = _data_note(report)
    return Explanation(
        headline=headline,
        paragraph=_condition_sentence(record, rul) + " " + paragraph,
        strengths=strengths,
        trade_offs=trade_offs,
        others=others,
        robustness=robustness,
        data_note=data_note,
    )


def _criterion_label(criterion: str) -> str:
    return {
        "feasibility": "technical feasibility",
        "life": "service life",
        "cost": "cost per service-year",
        "co2": "carbon per service-year",
        "circularity": "resource retention",
    }[criterion]


def _robustness_sentence(decision: DecisionResult) -> str:
    share = decision.stability
    parts = []
    if decision.runner_up is None:
        parts.append("All other strategies are blocked, so the recommendation does not depend on the weights.")
    elif decision.margin < CLOSE_CALL_POINTS:
        parts.append(
            f"This is a close call: {decision.recommended.lower()} leads {decision.runner_up.lower()} by only "
            f"{decision.margin:.1f} points."
        )
    else:
        parts.append(f"{decision.recommended} leads {decision.runner_up.lower()} by {decision.margin:.1f} points.")
    if decision.runner_up is not None:
        parts.append(f"It stays on top in {share:.0%} of randomly perturbed weight sets.")
        alternatives = [
            f"{s.lower()} ({v:.0%})"
            for s, v in sorted(decision.win_shares.items(), key=lambda kv: -kv[1])
            if s != decision.recommended and v >= NOTABLE_ALTERNATIVE_SHARE
        ]
        if alternatives:
            parts.append(f"Plausible alternatives under other weightings: {', '.join(alternatives)}.")
    return " ".join(parts)


def _data_note(report: ValidationReport) -> str:
    if report.imputed_fields:
        return (
            f"{len(report.imputed_fields)} non-critical field(s) were missing and imputed from fleet values; "
            "treat the result with additional caution."
        )
    if report.warnings:
        return f"{len(report.warnings)} data-quality warning(s) reduce confidence in this result."
    return ""
