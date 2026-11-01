"""MCDA ranking, weight handling and hard technical gates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from circularloop.config import (
    COMPONENT_TYPES,
    CRITERIA,
    DEFAULT_WEIGHTS,
    RECYCLE,
    REFURBISH,
    REPAIR,
    REUSE,
    STRATEGIES,
)
from circularloop.decision_engine import evaluate_strategies, normalize_weights, strategy_estimates


def _design(record: dict[str, Any]) -> float:
    return COMPONENT_TYPES[record["component_type"]].design_life_months


def test_weights_normalise_to_one_and_clip_negatives() -> None:
    w = normalize_weights({"feasibility": 3, "co2": 1, "cost": -5, "life": 0, "circularity": 0})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["cost"] == 0.0
    assert w["feasibility"] == pytest.approx(0.75)
    assert normalize_weights({c: 0 for c in CRITERIA}) == dict(DEFAULT_WEIGHTS)
    assert sum(normalize_weights({"life": 7}).values()) == pytest.approx(1.0)


def test_all_strategies_returned_and_ranking_sorted(healthy_record: dict[str, Any]) -> None:
    result = evaluate_strategies(healthy_record, 150.0, 110.0)
    assert sorted(a.strategy for a in result.assessments) == sorted(STRATEGIES)
    feasible = [a.score for a in result.assessments if not a.blocked]
    assert feasible == sorted(feasible, reverse=True)
    blocked_flags = [a.blocked for a in result.assessments]
    assert blocked_flags == sorted(blocked_flags), "blocked strategies must rank last"
    assert result.recommended == result.assessments[0].strategy


def test_severe_degradation_blocks_reuse_and_repair(severe_record: dict[str, Any]) -> None:
    result = evaluate_strategies(severe_record, 4.0, 0.0)
    assert result.get(REUSE).blocked and result.get(REPAIR).blocked
    assert result.get(REUSE).score == 0.0
    assert result.recommended not in (REUSE, REPAIR)


def test_gates_hold_even_when_weights_favour_reuse(severe_record: dict[str, Any]) -> None:
    """Zero feasibility weight and all weight on CO₂ / circularity must not revive unsafe reuse."""
    for weights in (
        {"co2": 1.0},
        {"circularity": 1.0},
        {"co2": 0.5, "circularity": 0.5, "feasibility": 0.0},
    ):
        result = evaluate_strategies(severe_record, 4.0, 0.0, weights)
        assert result.recommended != REUSE
        assert result.win_shares[REUSE] == 0.0


def test_reuse_never_wins_for_severe_record_under_random_weights(severe_record: dict[str, Any]) -> None:
    rng = np.random.default_rng(0)
    for w in rng.dirichlet(np.ones(len(CRITERIA)), size=100):
        result = evaluate_strategies(severe_record, 4.0, 0.0, dict(zip(CRITERIA, w)))
        assert result.recommended not in (REUSE, REPAIR)


def test_healthy_component_favours_high_retention_strategy(healthy_record: dict[str, Any]) -> None:
    result = evaluate_strategies(healthy_record, 150.0, 110.0)
    assert result.recommended in (REUSE, REPAIR, REFURBISH)
    assert not result.get(REUSE).blocked


def test_recycle_is_always_available(severe_record: dict[str, Any]) -> None:
    worst = {**severe_record, "corrosion_score": 10.0, "residual_thickness_pct": 30.0, "failure_events": 9}
    result = evaluate_strategies(worst, 0.0, 0.0)
    assert not result.get(RECYCLE).blocked
    assert result.recommended == RECYCLE
    assert result.runner_up is None or not result.get(result.runner_up).blocked


def test_estimates_are_finite_and_life_bounded(healthy_record: dict[str, Any], severe_record: dict[str, Any]) -> None:
    for record, rul in ((healthy_record, 500.0), (severe_record, 0.0)):
        for a in evaluate_strategies(record, rul, rul).assessments:
            assert 0 < a.life_months <= _design(record)
            assert all(np.isfinite([a.cost_eur, a.co2_kg, a.cost_per_year, a.co2_per_year, a.score]))
            assert all(0.0 <= v <= 1.0 for v in a.criteria.values())


def test_interventions_cannot_add_life_that_was_not_consumed(healthy_record: dict[str, Any]) -> None:
    design = _design(healthy_record)
    _, _, life_reuse = strategy_estimates(REUSE, healthy_record, design)
    _, _, life_refurb = strategy_estimates(REFURBISH, healthy_record, design)
    assert life_refurb == pytest.approx(life_reuse)
    _, _, mid_refurb = strategy_estimates(REFURBISH, healthy_record, design / 2)
    assert mid_refurb > design / 2


def test_missing_spare_parts_make_repair_more_expensive(healthy_record: dict[str, Any]) -> None:
    with_parts = strategy_estimates(REPAIR, healthy_record, 100.0)[0]
    without = strategy_estimates(REPAIR, {**healthy_record, "spare_parts_available": False}, 100.0)[0]
    assert without > with_parts


def test_weights_change_the_ranking_transparently(healthy_record: dict[str, Any]) -> None:
    life_heavy = evaluate_strategies(healthy_record, 30.0, 20.0, {"life": 1.0})
    cost_heavy = evaluate_strategies(healthy_record, 30.0, 20.0, {"cost": 1.0})
    assert life_heavy.recommended != cost_heavy.recommended


def test_decision_is_deterministic(healthy_record: dict[str, Any]) -> None:
    a = evaluate_strategies(healthy_record, 80.0, 60.0)
    b = evaluate_strategies(healthy_record, 80.0, 60.0)
    assert a.to_frame().equals(b.to_frame())
    assert a.win_shares == b.win_shares
