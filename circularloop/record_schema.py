"""Field catalogue and JSON export for the simplified lifecycle record.

This is a conceptual record structure for the demonstrator. It is **not** an
implementation of, or compliant with, any formal product-data standard.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from circularloop.config import CRITICAL_FIELDS

SCHEMA_ID = "circularloop-lifecycle-record/0.3"

SECTIONS: tuple[str, ...] = ("identity", "usage", "condition", "maintenance", "sustainability")


@dataclass(frozen=True)
class FieldSpec:
    """Description of one lifecycle-record field."""

    name: str
    section: str
    label: str
    unit: str
    kind: str  # "str", "float", "int", "bool"
    description: str
    valid_min: float | None = None
    valid_max: float | None = None

    @property
    def critical(self) -> bool:
        return self.name in CRITICAL_FIELDS


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("component_id", "identity", "Component ID", "–", "str", "Unique identifier of the physical item."),
    FieldSpec("component_type", "identity", "Component type", "–", "str", "Product family of the component."),
    FieldSpec("material", "identity", "Material", "–", "str", "Dominant structural material."),
    FieldSpec(
        "manufacture_year",
        "identity",
        "Manufacture year",
        "year",
        "int",
        "Year of manufacture / commissioning.",
        1900,
        2100,
    ),
    FieldSpec("age_years", "usage", "Age", "years", "float", "Time in service since commissioning.", 0, 80),
    FieldSpec("operating_hours", "usage", "Operating hours", "h", "float", "Cumulative hours under load.", 0, None),
    FieldSpec(
        "load_cycles",
        "usage",
        "Major load cycles",
        "count",
        "float",
        "Counted major load events (start/stop, load reversal).",
        0,
        None,
    ),
    FieldSpec(
        "vibration_rms",
        "condition",
        "Vibration (RMS velocity)",
        "mm/s",
        "float",
        "Latest condition-monitoring vibration level.",
        0,
        50,
    ),
    FieldSpec(
        "temperature_mean_c",
        "condition",
        "Mean operating temperature",
        "°C",
        "float",
        "Average temperature during operation.",
        -60,
        250,
    ),
    FieldSpec(
        "corrosion_score",
        "condition",
        "Corrosion score",
        "0–10",
        "float",
        "Inspection-based corrosion severity (0 none, 10 severe).",
        0,
        10,
    ),
    FieldSpec(
        "residual_thickness_pct",
        "condition",
        "Residual wall thickness",
        "% of nominal",
        "float",
        "Measured remaining wall/section thickness.",
        0,
        100,
    ),
    FieldSpec(
        "maintenance_count",
        "maintenance",
        "Maintenance interventions",
        "count",
        "int",
        "Number of recorded maintenance interventions.",
        0,
        None,
    ),
    FieldSpec(
        "failure_events",
        "maintenance",
        "Failure events",
        "count",
        "int",
        "Number of recorded functional failures.",
        0,
        None,
    ),
    FieldSpec(
        "last_maintenance_months_ago",
        "maintenance",
        "Months since last maintenance",
        "months",
        "float",
        "Time since last intervention (or commissioning).",
        0,
        None,
    ),
    FieldSpec(
        "spare_parts_available",
        "maintenance",
        "Spare parts available",
        "yes/no",
        "bool",
        "Whether repair parts can currently be sourced.",
    ),
    FieldSpec(
        "recyclability_pct",
        "sustainability",
        "Recyclability",
        "%",
        "float",
        "Share of mass recoverable as secondary material.",
        0,
        100,
    ),
    FieldSpec(
        "embodied_co2_kg",
        "sustainability",
        "Embodied CO₂ of new part",
        "kg CO₂e",
        "float",
        "Illustrative cradle-to-gate footprint of a new equivalent.",
        0,
        None,
    ),
    FieldSpec(
        "replacement_cost_eur",
        "sustainability",
        "Replacement cost",
        "EUR",
        "float",
        "Price of a new equivalent component.",
        0,
        None,
    ),
)

FIELD_BY_NAME: Mapping[str, FieldSpec] = {f.name: f for f in FIELDS}
ALL_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in FIELDS)


def is_missing(value: Any) -> bool:
    """Return True for None, NaN and empty strings."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        import pandas as pd  # local import keeps module light for pure-JSON use

        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_jsonable(value: Any, kind: str) -> Any:
    """Convert numpy/pandas scalars to plain JSON types; missing → None."""
    if is_missing(value):
        return None
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(round(float(value)))
    if kind == "float":
        return round(float(value), 2)
    return str(value)


def to_structured_record(
    record: Mapping[str, Any], decision_support: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Convert a flat component record into the sectioned lifecycle-record structure."""
    structured: dict[str, Any] = {section: {} for section in SECTIONS}
    for spec in FIELDS:
        structured[spec.section][spec.name] = _to_jsonable(record.get(spec.name), spec.kind)
    structured["decision_support"] = dict(decision_support or {})
    structured["metadata"] = {
        "schema": SCHEMA_ID,
        "data_origin": "synthetic",
        "standards_compliance": "none; simplified conceptual record for research demonstration",
        "disclaimer": "Illustrative synthetic record. Not an official or regulatory product-data record.",
    }
    return structured


def flatten_structured_record(structured: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a sectioned lifecycle-record structure back into a flat record."""
    record: dict[str, Any] = {}
    for section in SECTIONS:
        record.update(structured.get(section, {}))
    return record


def record_to_json(structured: Mapping[str, Any]) -> str:
    """Serialise a lifecycle-record structure to pretty JSON."""
    return json.dumps(structured, indent=2, ensure_ascii=False)


def field_catalogue_rows() -> list[dict[str, Any]]:
    """Rows describing each field, for documentation tables."""
    return [
        {
            "Section": f.section,
            "Field": f.name,
            "Label": f.label,
            "Unit": f.unit,
            "Critical": "yes" if f.critical else "",
            "Description": f.description,
        }
        for f in FIELDS
    ]
