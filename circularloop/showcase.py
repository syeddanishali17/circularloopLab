"""Hand-defined demonstration cases.

Only the *input condition* is specified here. The resulting RUL and
recommendation are whatever the pipeline produces; they are not tuned to reach
a predetermined answer. These records carry no RUL label and are never used
for training.
"""

from __future__ import annotations

from typing import Any

from circularloop.config import REFERENCE_YEAR


def _case(cid: str, title: str, **values: Any) -> dict[str, Any]:
    age = values["age_years"]
    record = {
        "component_id": cid,
        "manufacture_year": int(REFERENCE_YEAR - int(age)),
        **values,
    }
    record["_title"] = title
    return record


SHOWCASE_CASES: tuple[dict[str, Any], ...] = (
    _case(
        "SHOW-A",
        "Healthy drive shaft, early in life",
        component_type="Drive shaft",
        material="High-strength steel",
        age_years=3.5,
        operating_hours=17_200.0,
        load_cycles=420_000.0,
        vibration_rms=1.6,
        temperature_mean_c=43.0,
        corrosion_score=0.9,
        residual_thickness_pct=97.0,
        maintenance_count=2,
        failure_events=0,
        last_maintenance_months_ago=7.0,
        spare_parts_available=True,
        recyclability_pct=93.0,
        embodied_co2_kg=780.0,
        replacement_cost_eur=11_800.0,
    ),
    _case(
        "SHOW-B",
        "Mid-life gearbox, moderate wear",
        component_type="Industrial gearbox",
        material="High-strength steel",
        age_years=11.0,
        operating_hours=66_500.0,
        load_cycles=1_050_000.0,
        vibration_rms=4.2,
        temperature_mean_c=66.0,
        corrosion_score=4.6,
        residual_thickness_pct=71.0,
        maintenance_count=5,
        failure_events=1,
        last_maintenance_months_ago=14.0,
        spare_parts_available=True,
        recyclability_pct=94.0,
        embodied_co2_kg=3_700.0,
        replacement_cost_eur=28_000.0,
    ),
    _case(
        "SHOW-C",
        "Heavily corroded bogie frame",
        component_type="Rail bogie frame",
        material="High-strength steel",
        age_years=27.0,
        operating_hours=112_000.0,
        load_cycles=3_600_000.0,
        vibration_rms=4.1,
        temperature_mean_c=27.0,
        corrosion_score=8.2,
        residual_thickness_pct=57.0,
        maintenance_count=9,
        failure_events=3,
        last_maintenance_months_ago=20.0,
        spare_parts_available=False,
        recyclability_pct=95.0,
        embodied_co2_kg=6_300.0,
        replacement_cost_eur=96_000.0,
    ),
    _case(
        "SHOW-D",
        "Partially documented motor housing",
        component_type="Electric motor housing",
        material="Aluminium alloy",
        age_years=9.0,
        operating_hours=43_000.0,
        load_cycles=None,
        vibration_rms=None,
        temperature_mean_c=58.0,
        corrosion_score=1.8,
        residual_thickness_pct=90.0,
        maintenance_count=None,
        failure_events=0,
        last_maintenance_months_ago=None,
        spare_parts_available=True,
        recyclability_pct=88.0,
        embodied_co2_kg=2_400.0,
        replacement_cost_eur=9_500.0,
    ),
    _case(
        "SHOW-E",
        "Record without thickness measurement",
        component_type="Steel structural profile",
        material="High-strength steel",
        age_years=22.0,
        operating_hours=171_000.0,
        load_cycles=345_000.0,
        vibration_rms=0.9,
        temperature_mean_c=21.0,
        corrosion_score=5.5,
        residual_thickness_pct=None,
        maintenance_count=4,
        failure_events=0,
        last_maintenance_months_ago=30.0,
        spare_parts_available=True,
        recyclability_pct=96.0,
        embodied_co2_kg=1_900.0,
        replacement_cost_eur=6_400.0,
    ),
)


def showcase_records() -> list[dict[str, Any]]:
    """Copies of the showcase records without the display title."""
    return [{k: v for k, v in case.items() if k != "_title"} for case in SHOWCASE_CASES]


def showcase_titles() -> dict[str, str]:
    """Mapping component_id → short title."""
    return {case["component_id"]: case["_title"] for case in SHOWCASE_CASES}
