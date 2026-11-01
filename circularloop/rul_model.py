"""Remaining-Useful-Life regression: training, evaluation, explainability, prediction.

Two models are compared on identical held-out data:

* Linear Regression: interpretable baseline;
* Gradient Boosting: nonlinear comparison model.

The model used for downstream decisions is chosen by cross-validated MAE on the
*training* split, so the test split stays a clean, reported-only benchmark.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from circularloop.config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    GBR_PARAMS,
    INTERVAL_SCALE_OFFSET_MONTHS,
    MODEL_FEATURES,
    MODEL_GBR,
    MODEL_GBR_UNCONSTRAINED,
    MODEL_LINEAR,
    MODEL_NAMES,
    MONOTONIC_CONSTRAINTS,
    NUMERIC_FEATURES,
    PREDICTION_INTERVAL,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
)
from circularloop.record_schema import FIELDS
from circularloop.validation import usable_mask


@dataclass(frozen=True)
class RulPrediction:
    """Point prediction and approximate 80 % interval (months, non-negative)."""

    point: float
    lower: float
    upper: float
    model: str


@dataclass
class ModelBundle:
    """Trained models plus everything needed to report and explain them."""

    models: dict[str, Pipeline]
    metrics: pd.DataFrame  # index: model name
    test_frame: pd.DataFrame  # actual + predictions per model on the test split
    residual_quantiles: dict[str, tuple[float, float]]  # scaled residual quantiles (P10, P90)
    permutation_importance: pd.DataFrame  # columns: feature, model, importance, std
    linear_coefficients: pd.DataFrame  # columns: feature, coefficient
    default_model: str
    n_train: int
    n_test: int
    n_excluded: int
    feature_medians: dict[str, Any] = field(default_factory=dict)


def _preprocessor(scale_numeric: bool, drop_first: bool = False) -> ColumnTransformer:
    """Imputation + encoding. The linear model drops one reference category per variable
    so its category coefficients are identifiable (read relative to that reference)."""
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first" if drop_first else None),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", Pipeline(numeric_steps), list(NUMERIC_FEATURES)), ("cat", categorical, list(CATEGORICAL_FEATURES))],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _gradient_boosting(constrained: bool) -> Pipeline:
    params = dict(GBR_PARAMS)
    if constrained:
        params["monotonic_cst"] = dict(MONOTONIC_CONSTRAINTS)
    return Pipeline([("prep", _preprocessor(scale_numeric=False)), ("model", HistGradientBoostingRegressor(**params))])


def build_models() -> dict[str, Pipeline]:
    """Unfitted pipelines for both compared models."""
    return {
        MODEL_LINEAR: Pipeline(
            [("prep", _preprocessor(scale_numeric=True, drop_first=True)), ("model", LinearRegression())]
        ),
        MODEL_GBR: _gradient_boosting(constrained=True),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE (months)": float(mean_absolute_error(y_true, y_pred)),
        "RMSE (months)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²": float(r2_score(y_true, y_pred)),
    }


def _feature_frame(records: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
    """Select model features in fixed order; missing columns become NaN."""
    df = pd.DataFrame([records]) if isinstance(records, Mapping) else records
    out = df.reindex(columns=list(MODEL_FEATURES)).copy()
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype(object)
    return out


def train_and_evaluate(df: pd.DataFrame, seed: int = RANDOM_SEED) -> ModelBundle:
    """Train both models on validated records and evaluate them on a held-out split."""
    labelled = df[df[TARGET].notna()]
    mask = usable_mask(labelled)
    clean = labelled[mask]
    X = _feature_frame(clean)
    y = clean[TARGET].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)

    models = build_models()
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    rows: dict[str, dict[str, float]] = {}
    residual_q: dict[str, tuple[float, float]] = {}
    test_frame = pd.DataFrame({"actual": y_test}, index=X_test.index)
    importances: list[pd.DataFrame] = []

    for name, pipe in models.items():
        oof = cross_val_predict(pipe, X_train, y_train, cv=cv)
        # Residuals scaled by prediction size: degradation noise grows with remaining life.
        scaled = (y_train - oof) / (np.maximum(oof, 0.0) + INTERVAL_SCALE_OFFSET_MONTHS)
        residual_q[name] = tuple(float(q) for q in np.quantile(scaled, PREDICTION_INTERVAL))  # type: ignore[assignment]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        test_frame[name] = y_pred
        rows[name] = {**_metrics(y_test, y_pred), "CV MAE (train, months)": float(mean_absolute_error(y_train, oof))}

        perm = permutation_importance(
            pipe, X_test, y_test, scoring="neg_mean_absolute_error", n_repeats=5, random_state=seed
        )
        importances.append(
            pd.DataFrame(
                {
                    "feature": list(MODEL_FEATURES),
                    "model": name,
                    "importance": perm.importances_mean,
                    "std": perm.importances_std,
                }
            )
        )

    metrics = pd.DataFrame(rows).T.loc[list(MODEL_NAMES)]
    default_model = str(metrics["CV MAE (train, months)"].idxmin())
    metrics.loc[MODEL_GBR_UNCONSTRAINED] = _ablation_metrics(X_train, y_train, X_test, y_test, cv)

    medians = _fleet_defaults(clean.loc[X_train.index])

    return ModelBundle(
        models=models,
        metrics=metrics,
        test_frame=test_frame,
        residual_quantiles=residual_q,
        permutation_importance=pd.concat(importances, ignore_index=True),
        linear_coefficients=_linear_coefficients(models[MODEL_LINEAR]),
        default_model=default_model,
        n_train=len(X_train),
        n_test=len(X_test),
        n_excluded=int((~mask).sum()) + int(df[TARGET].isna().sum()),
        feature_medians=medians,
    )


def _ablation_metrics(
    X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray, cv: KFold
) -> dict[str, float]:
    """Same boosting model without monotonic constraints; reported, never used for decisions."""
    pipe = _gradient_boosting(constrained=False)
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv)
    pipe.fit(X_train, y_train)
    return {
        **_metrics(y_test, pipe.predict(X_test)),
        "CV MAE (train, months)": float(mean_absolute_error(y_train, oof)),
    }


def _fleet_defaults(train_rows: pd.DataFrame) -> dict[str, Any]:
    """Training-fleet medians/modes used to impute missing non-critical record fields."""
    defaults: dict[str, Any] = {}
    for spec in FIELDS:
        if spec.name not in train_rows or spec.kind == "str":
            continue
        if spec.kind == "bool":
            defaults[spec.name] = False  # conservative: assume spare parts are not available
        else:
            defaults[spec.name] = float(pd.to_numeric(train_rows[spec.name], errors="coerce").median())
    return defaults


def _linear_coefficients(pipe: Pipeline) -> pd.DataFrame:
    """Linear-model coefficients: numeric per 1 SD, categories relative to the reference level."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_
    df = pd.DataFrame({"feature": names, "coefficient": coefs})
    df["kind"] = np.where(df["feature"].isin(NUMERIC_FEATURES), "per 1 SD increase", "vs. reference category")
    return df.reindex(df["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def predict_rul(bundle: ModelBundle, record: Mapping[str, Any], model: str | None = None) -> RulPrediction:
    """Predict RUL for one record with an approximate 80 % interval (clipped at 0).

    The interval applies cross-validated, scale-normalised residual quantiles to the
    point prediction (a simple normalised split-conformal style approximation).
    """
    name = model or bundle.default_model
    X = _feature_frame(record)
    point = max(0.0, float(bundle.models[name].predict(X)[0]))
    lo_q, hi_q = bundle.residual_quantiles[name]
    scale = point + INTERVAL_SCALE_OFFSET_MONTHS
    return RulPrediction(
        point=point,
        lower=max(0.0, point + lo_q * scale),
        upper=max(0.0, point + hi_q * scale),
        model=name,
    )


def response_curve(bundle: ModelBundle, record: Mapping[str, Any], feature: str, values: np.ndarray) -> pd.DataFrame:
    """Predicted RUL of both models as one feature varies, others held at the record's values."""
    base = _feature_frame(record)
    grid = pd.concat([base] * len(values), ignore_index=True)
    grid[feature] = values
    out = pd.DataFrame({feature: values})
    for name, pipe in bundle.models.items():
        out[name] = np.maximum(0.0, pipe.predict(grid))
    return out
