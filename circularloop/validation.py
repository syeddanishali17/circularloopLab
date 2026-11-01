"""Data-validation layer for component lifecycle records.

Each record is checked for completeness, physically possible ranges and internal
consistency. Issues carry a severity:

* ``critical``: the record must not produce a recommendation;
* ``warning``: a recommendation is possible but with reduced confidence;
* ``info``: noteworthy, no effect on confidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from circularloop.config import (
    COMPONENT_TYPES,
    CRITICAL_FIELD_WEIGHT,
    CRITICAL_FIELDS,
    HOURS_PER_YEAR,
    MATERIALS,
    QUALITY_IMPUTATION_PENALTY,
    QUALITY_SUFFICIENT,
    QUALITY_WARNING_PENALTY,
    REFERENCE_YEAR,
)
from circularloop.record_schema import ALL_FIELD_NAMES, FIELD_BY_NAME, is_missing

Severity = Literal["critical", "warning", "info"]

STATUS_SUFFICIENT = "Sufficient"
STATUS_REDUCED = "Reduced confidence"
STATUS_INSUFFICIENT = "Insufficient"

# Fields that are optional for identification and do not affect completeness.
_COMPLETENESS_EXCLUDED: frozenset[str] = frozenset({"component_id"})

# Plausible-but-unusual limits (warnings, not errors).
TEMP_WARNING_MAX_C = 120.0
TEMP_WARNING_MIN_C = -40.0
VIBRATION_WARNING_MAX = 20.0
CYCLES_PER_HOUR_WARNING_MAX = 3_600.0


@dataclass(frozen=True)
class Issue:
    """One validation finding."""

    field: str
    severity: Severity
    message: str


@dataclass
class ValidationReport:
    """Result of validating a single record."""

    issues: list[Issue] = field(default_factory=list)
    completeness_pct: float = 100.0
    quality_score: float = 100.0
    imputed_fields: list[str] = field(default_factory=list)

    @property
    def critical(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "info"]

    @property
    def is_usable(self) -> bool:
        """True when a recommendation may be produced."""
        return not self.critical

    @property
    def status(self) -> str:
        if self.critical:
            return STATUS_INSUFFICIENT
        if self.quality_score >= QUALITY_SUFFICIENT:
            return STATUS_SUFFICIENT
        return STATUS_REDUCED


def _num(record: Mapping[str, Any], name: str) -> float | None:
    """Return a field as float, or None when missing / non-numeric."""
    value = record.get(name)
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_missing(record: Mapping[str, Any]) -> tuple[list[Issue], list[str]]:
    issues: list[Issue] = []
    imputed: list[str] = []
    for name in ALL_FIELD_NAMES:
        if name in _COMPLETENESS_EXCLUDED or not is_missing(record.get(name)):
            continue
        label = FIELD_BY_NAME[name].label
        if name in CRITICAL_FIELDS:
            issues.append(Issue(name, "critical", f"{label} is missing; it is required for a reliable assessment."))
        else:
            issues.append(Issue(name, "warning", f"{label} is missing and was imputed from the fleet median."))
            imputed.append(name)
    return issues, imputed


def _check_categories(record: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    ctype, material = record.get("component_type"), record.get("material")
    if not is_missing(ctype) and ctype not in COMPONENT_TYPES:
        issues.append(Issue("component_type", "critical", f"Unknown component type '{ctype}'."))
    if not is_missing(material) and material not in MATERIALS:
        issues.append(Issue("material", "critical", f"Unknown material '{material}'."))
    return issues


def _check_ranges(record: Mapping[str, Any]) -> list[Issue]:
    """Physically impossible values are critical."""
    issues: list[Issue] = []
    for name in ALL_FIELD_NAMES:
        spec = FIELD_BY_NAME[name]
        if spec.kind not in ("float", "int"):
            continue
        value = _num(record, name)
        if value is None:
            if not is_missing(record.get(name)):
                issues.append(Issue(name, "critical", f"{spec.label} is not a number."))
            continue
        too_low = spec.valid_min is not None and value < spec.valid_min
        too_high = spec.valid_max is not None and value > spec.valid_max
        if too_low or too_high:
            bounds = (
                f"{spec.valid_min if spec.valid_min is not None else '−∞'}–"
                f"{spec.valid_max if spec.valid_max is not None else '∞'}"
            )
            issues.append(
                Issue(
                    name, "critical", f"{spec.label} = {value:g} is outside the possible range ({bounds} {spec.unit})."
                )
            )
    for name in ("replacement_cost_eur", "embodied_co2_kg"):
        value = _num(record, name)
        if value is not None and value == 0:
            issues.append(Issue(name, "critical", f"{FIELD_BY_NAME[name].label} of zero is not meaningful."))
    return issues


def _check_plausibility(record: Mapping[str, Any]) -> list[Issue]:
    """Possible but unusual values, and cross-field consistency."""
    issues: list[Issue] = []
    age = _num(record, "age_years")
    hours = _num(record, "operating_hours")
    cycles = _num(record, "load_cycles")
    temp = _num(record, "temperature_mean_c")
    vib = _num(record, "vibration_rms")
    last_maint = _num(record, "last_maintenance_months_ago")
    year = _num(record, "manufacture_year")
    maint = _num(record, "maintenance_count")
    failures = _num(record, "failure_events")

    if temp is not None and (temp > TEMP_WARNING_MAX_C or temp < TEMP_WARNING_MIN_C):
        issues.append(
            Issue("temperature_mean_c", "warning", f"Mean temperature of {temp:g} °C is implausible for this duty.")
        )
    if vib is not None and vib > VIBRATION_WARNING_MAX:
        issues.append(
            Issue("vibration_rms", "warning", f"Vibration of {vib:g} mm/s is implausibly high; check the sensor.")
        )
    if age is not None and hours is not None and age >= 0 and hours > age * HOURS_PER_YEAR * 1.001:
        issues.append(
            Issue(
                "operating_hours",
                "critical",
                f"Operating hours ({hours:,.0f} h) exceed the calendar hours since commissioning "
                f"({age * HOURS_PER_YEAR:,.0f} h).",
            )
        )
    if hours is not None and cycles is not None and hours > 0 and cycles / hours > CYCLES_PER_HOUR_WARNING_MAX:
        issues.append(
            Issue(
                "load_cycles", "warning", "More than one major load cycle per second on average, which is implausible."
            )
        )
    if age is not None and last_maint is not None and last_maint > age * 12 + 1:
        issues.append(
            Issue("last_maintenance_months_ago", "warning", "Last maintenance is dated before commissioning.")
        )
    if age is not None and year is not None and abs((REFERENCE_YEAR - year) - age) > 1.5:
        issues.append(Issue("manufacture_year", "warning", "Manufacture year is inconsistent with the stated age."))
    if maint is not None and failures is not None and failures > maint + 1:
        issues.append(
            Issue(
                "failure_events",
                "warning",
                "More failures than maintenance interventions; the history may be incomplete.",
            )
        )
    ctype = record.get("component_type")
    if age is not None and ctype in COMPONENT_TYPES and age > COMPONENT_TYPES[ctype].design_life_years:
        issues.append(Issue("age_years", "info", "Component is beyond its nominal design life."))
    return issues


def _completeness(record: Mapping[str, Any]) -> float:
    total = present = 0.0
    for name in ALL_FIELD_NAMES:
        if name in _COMPLETENESS_EXCLUDED:
            continue
        weight = CRITICAL_FIELD_WEIGHT if name in CRITICAL_FIELDS else 1.0
        total += weight
        present += 0.0 if is_missing(record.get(name)) else weight
    return 100.0 * present / total


def validate_record(record: Mapping[str, Any]) -> ValidationReport:
    """Validate one flat component record."""
    missing_issues, imputed = _check_missing(record)
    issues = missing_issues + _check_categories(record) + _check_ranges(record) + _check_plausibility(record)
    completeness = _completeness(record)
    n_warnings = sum(1 for i in issues if i.severity == "warning" and i.field not in imputed)
    quality = completeness - QUALITY_WARNING_PENALTY * n_warnings - QUALITY_IMPUTATION_PENALTY * len(imputed)
    if any(i.severity == "critical" for i in issues):
        quality = min(quality, 40.0)
    return ValidationReport(
        issues=issues,
        completeness_pct=round(completeness, 1),
        quality_score=round(max(0.0, min(100.0, quality)), 1),
        imputed_fields=imputed,
    )


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate every row; returns one summary row per record."""
    rows = []
    for record in df.to_dict(orient="records"):
        report = validate_record(record)
        rows.append(
            {
                "component_id": record.get("component_id"),
                "status": report.status,
                "quality_score": report.quality_score,
                "n_critical": len(report.critical),
                "n_warnings": len(report.warnings),
            }
        )
    return pd.DataFrame(rows, index=df.index)


def usable_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of records without critical issues (used to filter training data)."""
    return validate_frame(df)["n_critical"] == 0
