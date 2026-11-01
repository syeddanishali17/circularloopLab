"""RUL model training, evaluation, prediction sanity and leakage guards."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from circularloop.config import MODEL_FEATURES, MODEL_GBR, MODEL_LINEAR, MODEL_NAMES, N_COMPONENTS, TARGET
from circularloop.rul_model import ModelBundle, predict_rul, response_curve, train_and_evaluate
from circularloop.synthetic_data import generate_fleet


def test_both_models_train_with_finite_plausible_metrics(bundle: ModelBundle) -> None:
    for name in MODEL_NAMES:
        row = bundle.metrics.loc[name]
        assert np.isfinite(row.to_numpy(dtype=float)).all()
        assert row["MAE (months)"] > 0
        assert 0.5 < row["R²"] < 0.99, "latent factors and noise must prevent near-perfect reconstruction"
    assert bundle.default_model in MODEL_NAMES
    assert bundle.default_model == bundle.metrics.loc[list(MODEL_NAMES), "CV MAE (train, months)"].idxmin()


def test_invalid_records_are_excluded_from_training(bundle: ModelBundle, dataset: pd.DataFrame) -> None:
    assert bundle.n_train + bundle.n_test == N_COMPONENTS
    assert bundle.n_excluded == len(dataset) - N_COMPONENTS


def test_model_features_exclude_target_identity_and_economics() -> None:
    forbidden = {
        TARGET,
        "component_id",
        "manufacture_year",
        "replacement_cost_eur",
        "embodied_co2_kg",
        "recyclability_pct",
        "spare_parts_available",
    }
    assert forbidden.isdisjoint(MODEL_FEATURES)


def test_predictions_are_finite_non_negative_with_ordered_interval(
    bundle: ModelBundle, severe_record: dict[str, Any], healthy_record: dict[str, Any]
) -> None:
    extreme = {**severe_record, "corrosion_score": 10.0, "residual_thickness_pct": 30.0, "failure_events": 10}
    for record in (healthy_record, severe_record, extreme):
        for name in MODEL_NAMES:
            pred = predict_rul(bundle, record, name)
            assert all(np.isfinite([pred.lower, pred.point, pred.upper]))
            assert 0.0 <= pred.lower <= pred.point <= pred.upper


def test_healthy_component_has_longer_predicted_life_than_severe(
    bundle: ModelBundle, severe_record: dict[str, Any], healthy_record: dict[str, Any]
) -> None:
    for name in MODEL_NAMES:
        assert predict_rul(bundle, healthy_record, name).point > predict_rul(bundle, severe_record, name).point


def test_boosting_is_monotone_in_degradation_indicators(bundle: ModelBundle, healthy_record: dict[str, Any]) -> None:
    for feature, values, direction in (
        ("corrosion_score", np.linspace(0, 10, 25), -1),
        ("vibration_rms", np.linspace(0.2, 10, 25), -1),
        ("residual_thickness_pct", np.linspace(40, 100, 25), 1),
        ("last_maintenance_months_ago", np.linspace(0, 120, 25), -1),
    ):
        curve = response_curve(bundle, healthy_record, feature, values)[MODEL_GBR].to_numpy()
        assert (direction * np.diff(curve) >= -1e-9).all(), feature


def test_missing_non_critical_inputs_still_predict(bundle: ModelBundle, healthy_record: dict[str, Any]) -> None:
    healthy_record["vibration_rms"] = None
    pred = predict_rul(bundle, healthy_record, MODEL_LINEAR)
    assert np.isfinite(pred.point)


def test_training_is_deterministic() -> None:
    small = generate_fleet(150, seed=11)
    a, b = train_and_evaluate(small), train_and_evaluate(small)
    pd.testing.assert_frame_equal(a.metrics, b.metrics)
    pd.testing.assert_frame_equal(a.test_frame, b.test_frame)
