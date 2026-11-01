"""Synthetic-data constraints, plausibility and determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd

from circularloop.config import COMPONENT_TYPES, HOURS_PER_YEAR, N_COMPONENTS, REFERENCE_YEAR
from circularloop.synthetic_data import COLUMN_ORDER, defective_records, generate_fleet, hidden_rul_signal
from circularloop.validation import validate_frame


def test_fleet_has_expected_rows_columns_and_unique_ids(fleet: pd.DataFrame) -> None:
    assert len(fleet) == N_COMPONENTS
    assert list(fleet.columns) == list(COLUMN_ORDER)
    assert fleet["component_id"].is_unique
    assert set(fleet["component_type"]) == set(COMPONENT_TYPES)


def test_fleet_values_are_physically_possible(fleet: pd.DataFrame) -> None:
    assert fleet["corrosion_score"].between(0, 10).all()
    assert fleet["residual_thickness_pct"].between(0, 100).all()
    assert fleet["recyclability_pct"].between(0, 100).all()
    for col in ("operating_hours", "load_cycles", "maintenance_count", "failure_events", "vibration_rms"):
        assert (fleet[col] >= 0).all(), col
    assert (fleet["replacement_cost_eur"] > 0).all()
    assert (fleet["embodied_co2_kg"] > 0).all()
    assert not fleet.drop(columns=["rul_months"]).isna().any().any()


def test_lifecycle_values_are_internally_consistent(fleet: pd.DataFrame) -> None:
    assert (fleet["operating_hours"] <= fleet["age_years"] * HOURS_PER_YEAR).all()
    assert (fleet["last_maintenance_months_ago"] <= fleet["age_years"] * 12 + 1e-6).all()
    assert ((REFERENCE_YEAR - fleet["manufacture_year"]) - fleet["age_years"]).abs().max() <= 1.0
    for ctype, spec in COMPONENT_TYPES.items():
        materials = set(fleet.loc[fleet["component_type"] == ctype, "material"])
        assert materials <= set(spec.materials), ctype


def test_rul_target_is_non_negative_finite_and_mostly_positive(fleet: pd.DataFrame) -> None:
    rul = fleet["rul_months"].to_numpy()
    assert np.isfinite(rul).all()
    assert (rul >= 0).all()
    assert (rul == 0).mean() < 0.05


def test_condition_variables_co_evolve_plausibly(fleet: pd.DataFrame) -> None:
    corr = fleet.corr(numeric_only=True)
    assert corr.loc["corrosion_score", "residual_thickness_pct"] < -0.5
    assert corr.loc["age_years", "corrosion_score"] > 0.3
    assert corr.loc["residual_thickness_pct", "rul_months"] > 0
    assert corr.loc["corrosion_score", "rul_months"] < 0


def test_generation_is_deterministic_for_a_fixed_seed() -> None:
    pd.testing.assert_frame_equal(generate_fleet(80, seed=3), generate_fleet(80, seed=3))
    assert not generate_fleet(80, seed=3).equals(generate_fleet(80, seed=4))


def test_hidden_signal_has_nonlinear_corrosion_effect(fleet: pd.DataFrame) -> None:
    """Each additional corrosion point costs more life than the previous one (convex loss)."""
    row = fleet.iloc[[0]].copy()
    ones = np.ones(1)
    values = []
    for c in (0.0, 3.0, 6.0, 9.0):
        row["corrosion_score"] = c
        values.append(float(hidden_rul_signal(row, ones, ones)[0]))
    drops = -np.diff(values)
    assert (drops > 0).all()
    assert drops[2] > drops[0]


def test_generated_fleet_passes_validation_and_defects_do_not(fleet: pd.DataFrame) -> None:
    assert (validate_frame(fleet)["n_critical"] == 0).all()
    defects = validate_frame(defective_records())
    assert (defects["n_critical"] > 0).all()
