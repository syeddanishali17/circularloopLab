"""Shared fixtures. Model training is session-scoped because it takes a few seconds."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from circularloop.rul_model import ModelBundle, train_and_evaluate
from circularloop.synthetic_data import build_dataset, generate_fleet


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    return build_dataset()


@pytest.fixture(scope="session")
def fleet() -> pd.DataFrame:
    return generate_fleet()


@pytest.fixture(scope="session")
def bundle(dataset: pd.DataFrame) -> ModelBundle:
    return train_and_evaluate(dataset)


@pytest.fixture()
def healthy_record() -> dict[str, Any]:
    return {
        "component_id": "T-HEALTHY",
        "component_type": "Drive shaft",
        "material": "High-strength steel",
        "manufacture_year": 2023,
        "age_years": 3.0,
        "operating_hours": 15_000.0,
        "load_cycles": 380_000.0,
        "vibration_rms": 1.5,
        "temperature_mean_c": 42.0,
        "corrosion_score": 0.8,
        "residual_thickness_pct": 97.0,
        "maintenance_count": 2,
        "failure_events": 0,
        "last_maintenance_months_ago": 6.0,
        "spare_parts_available": True,
        "recyclability_pct": 93.0,
        "embodied_co2_kg": 800.0,
        "replacement_cost_eur": 12_000.0,
    }


@pytest.fixture()
def severe_record() -> dict[str, Any]:
    return {
        "component_id": "T-SEVERE",
        "component_type": "Rail bogie frame",
        "material": "High-strength steel",
        "manufacture_year": 1998,
        "age_years": 28.0,
        "operating_hours": 115_000.0,
        "load_cycles": 3_700_000.0,
        "vibration_rms": 6.0,
        "temperature_mean_c": 30.0,
        "corrosion_score": 9.0,
        "residual_thickness_pct": 52.0,
        "maintenance_count": 8,
        "failure_events": 4,
        "last_maintenance_months_ago": 30.0,
        "spare_parts_available": False,
        "recyclability_pct": 95.0,
        "embodied_co2_kg": 6_000.0,
        "replacement_cost_eur": 90_000.0,
    }
