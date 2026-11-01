"""Validation rules, severities and the quality score."""

from __future__ import annotations

from typing import Any

import pytest

from circularloop.validation import (
    STATUS_INSUFFICIENT,
    STATUS_REDUCED,
    STATUS_SUFFICIENT,
    validate_record,
)


def _critical_fields(record: dict[str, Any]) -> set[str]:
    return {i.field for i in validate_record(record).critical}


def test_valid_record_is_sufficient_with_full_score(healthy_record: dict[str, Any]) -> None:
    report = validate_record(healthy_record)
    assert report.status == STATUS_SUFFICIENT
    assert report.quality_score == 100.0
    assert report.is_usable


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recyclability_pct", 130.0),
        ("residual_thickness_pct", -4.0),
        ("corrosion_score", 14.0),
        ("operating_hours", -1_200.0),
        ("replacement_cost_eur", -500.0),
        ("embodied_co2_kg", -10.0),
        ("temperature_mean_c", 400.0),
    ],
)
def test_impossible_values_are_critical(healthy_record: dict[str, Any], field: str, value: float) -> None:
    healthy_record[field] = value
    assert field in _critical_fields(healthy_record)
    assert validate_record(healthy_record).status == STATUS_INSUFFICIENT


def test_missing_critical_field_blocks_recommendation(healthy_record: dict[str, Any]) -> None:
    healthy_record["residual_thickness_pct"] = None
    report = validate_record(healthy_record)
    assert not report.is_usable
    assert "residual_thickness_pct" in {i.field for i in report.critical}


def test_missing_non_critical_fields_are_imputed_with_reduced_confidence(healthy_record: dict[str, Any]) -> None:
    for field in ("vibration_rms", "load_cycles", "maintenance_count"):
        healthy_record[field] = float("nan")
    report = validate_record(healthy_record)
    assert report.is_usable
    assert set(report.imputed_fields) == {"vibration_rms", "load_cycles", "maintenance_count"}
    assert report.status == STATUS_REDUCED
    assert report.quality_score < 85


def test_operating_hours_cannot_exceed_calendar_time(healthy_record: dict[str, Any]) -> None:
    healthy_record["operating_hours"] = healthy_record["age_years"] * 8_760 * 1.5
    assert "operating_hours" in _critical_fields(healthy_record)


def test_implausible_but_possible_values_are_warnings(healthy_record: dict[str, Any]) -> None:
    healthy_record["temperature_mean_c"] = 135.0
    healthy_record["manufacture_year"] = 1990  # inconsistent with age 3
    report = validate_record(healthy_record)
    assert report.is_usable
    assert {"temperature_mean_c", "manufacture_year"} <= {i.field for i in report.warnings}
    assert report.quality_score < 100


def test_unknown_category_is_critical(healthy_record: dict[str, Any]) -> None:
    healthy_record["material"] = "Unobtainium"
    assert "material" in _critical_fields(healthy_record)


def test_quality_score_is_bounded(healthy_record: dict[str, Any]) -> None:
    for key in list(healthy_record):
        if key != "component_id":
            healthy_record[key] = None
    report = validate_record(healthy_record)
    assert 0 <= report.quality_score <= 100
    assert report.completeness_pct == 0
