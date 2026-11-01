"""What-if scenarios applied to a component record.

Material loss is treated as irreversible: better maintenance can slow corrosion
and reduce vibration, but cannot restore residual thickness.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from circularloop.config import SCENARIO_CURRENT, SCENARIO_IMPROVED, SCENARIO_SEVERE, SCENARIOS
from circularloop.record_schema import is_missing


@dataclass(frozen=True)
class ScenarioSpec:
    """Description and transformation of one scenario."""

    name: str
    description: str
    transform: Callable[[dict[str, Any]], dict[str, Any]]


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _apply(record: Mapping[str, Any], updates: Mapping[str, Callable[[float], float]]) -> dict[str, Any]:
    """Apply per-field updates, leaving missing values missing."""
    out = dict(record)
    for name, fn in updates.items():
        if not is_missing(out.get(name)):
            out[name] = fn(float(out[name]))
    return out


def _current(record: dict[str, Any]) -> dict[str, Any]:
    return dict(record)


def _improved(record: dict[str, Any]) -> dict[str, Any]:
    return _apply(
        record,
        {
            "vibration_rms": lambda v: round(v * 0.7, 2),
            "failure_events": lambda v: max(0, int(v) - 1),
            "corrosion_score": lambda v: round(_clip(v - 1.0, 0, 10), 1),
            "last_maintenance_months_ago": lambda v: min(v, 2.0),
            "temperature_mean_c": lambda v: round(v - 5.0, 1),
        },
    )


def _severe(record: dict[str, Any]) -> dict[str, Any]:
    return _apply(
        record,
        {
            "corrosion_score": lambda v: round(_clip(v + 3.0, 0, 10), 1),
            "residual_thickness_pct": lambda v: round(_clip(v - 15.0, 0, 100), 1),
            "failure_events": lambda v: int(v) + 2,
            "vibration_rms": lambda v: round(v * 1.6, 2),
            "temperature_mean_c": lambda v: round(v + 10.0, 1),
        },
    )


SCENARIO_SPECS: Mapping[str, ScenarioSpec] = {
    SCENARIO_CURRENT: ScenarioSpec(SCENARIO_CURRENT, "The lifecycle record as stored.", _current),
    SCENARIO_IMPROVED: ScenarioSpec(
        SCENARIO_IMPROVED,
        "Condition-based maintenance: vibration −30%, one fewer failure, corrosion −1 (protective treatment), "
        "recent service, temperature −5 °C. Thickness unchanged (material loss is irreversible).",
        _improved,
    ),
    SCENARIO_SEVERE: ScenarioSpec(
        SCENARIO_SEVERE,
        "Aggressive environment and overload: corrosion +3, thickness −15 pp, two more failures, "
        "vibration +60%, temperature +10 °C.",
        _severe,
    ),
}


def apply_scenario(record: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    """Return a modified copy of ``record`` for the named scenario."""
    return SCENARIO_SPECS[scenario].transform(dict(record))


def all_scenarios(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Records for every scenario, in display order."""
    return {name: apply_scenario(record, name) for name in SCENARIOS}
