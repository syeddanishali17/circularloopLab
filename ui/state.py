"""Cached resources, sidebar controls and the per-run application context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from circularloop.config import CRITERIA, CRITERION_LABELS, DEFAULT_WEIGHTS, MODEL_GBR, MODEL_LINEAR, MODEL_NAMES
from circularloop.data_io import load_components
from circularloop.decision_engine import normalize_weights
from circularloop.pipeline import Assessment, assess_component
from circularloop.rul_model import ModelBundle, train_and_evaluate
from circularloop.showcase import showcase_records, showcase_titles

AUTO_MODEL = "Auto (best CV)"
MODEL_SHORT = {AUTO_MODEL: "Auto", MODEL_LINEAR: "Linear", MODEL_GBR: "Boosting"}
WEIGHT_KEY = "w_{}"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Fleet dataset (created deterministically on first run)."""
    return load_components()


@st.cache_resource(show_spinner="Training both RUL models on the synthetic fleet…")
def load_bundle() -> ModelBundle:
    """Trained models and benchmark results, shared across sessions."""
    return train_and_evaluate(load_data())


@dataclass
class AppContext:
    """Everything a view needs for the current rerun."""

    df: pd.DataFrame
    bundle: ModelBundle
    component_id: str
    title: str
    record: dict[str, Any]
    weights: dict[str, float]
    model: str
    assessment: Assessment

    def assess(self, record: dict[str, Any]) -> Assessment:
        """Run the shared pipeline on another record with the current settings."""
        return assess_component(record, self.bundle, self.weights, self.model)


def _records_by_id(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    fleet = {r["component_id"]: r for r in df.drop(columns=["rul_months"]).to_dict(orient="records")}
    return {**{r["component_id"]: r for r in showcase_records()}, **fleet}


def _option_label(cid: str, records: dict[str, dict[str, Any]]) -> str:
    titles = showcase_titles()
    if cid in titles:
        return f"{cid} · {titles[cid]}"
    rec = records[cid]
    if cid.startswith("DQ-"):
        return f"{cid} · defective record (validation demo)"
    return f"{cid} · {rec['component_type']}, {float(rec['age_years']):.0f} y"


def _reset_weights() -> None:
    for c in CRITERIA:
        st.session_state[WEIGHT_KEY.format(c)] = DEFAULT_WEIGHTS[c]


def sidebar(df: pd.DataFrame, bundle: ModelBundle) -> AppContext:
    """Render sidebar controls and build the context for this run."""
    records = _records_by_id(df)
    options = list(records.keys())
    for c in CRITERIA:
        st.session_state.setdefault(WEIGHT_KEY.format(c), DEFAULT_WEIGHTS[c])

    with st.sidebar:
        st.subheader("Component")
        cid = st.selectbox(
            "Component record",
            options,
            index=options.index("SHOW-B"),
            format_func=lambda x: _option_label(x, records),
            key="component_id",
            help="Showcase cases first, then the synthetic fleet and a few deliberately defective records.",
        )

        st.subheader("RUL model")
        choice = st.segmented_control(
            "Model used for decisions",
            [AUTO_MODEL, *MODEL_NAMES],
            default=AUTO_MODEL,
            required=True,
            format_func=MODEL_SHORT.get,
            key="model_choice",
            help="Auto selects the model with the lower 5-fold cross-validated MAE on the training split.",
        )
        model = bundle.default_model if choice == AUTO_MODEL else str(choice)
        st.caption(f"Decisions use **{model}**.")

        st.subheader("MCDA weights")
        raw = {c: st.slider(CRITERION_LABELS[c], 0.0, 1.0, step=0.05, key=WEIGHT_KEY.format(c)) for c in CRITERIA}
        weights = normalize_weights(raw)
        st.caption("Normalised: " + " · ".join(f"{CRITERION_LABELS[c].split()[0]} {weights[c]:.2f}" for c in CRITERIA))
        st.button("Reset to default weights", on_click=_reset_weights, icon=":material/restart_alt:", width="stretch")

        st.space("medium")
        st.caption(
            "Independent synthetic research demonstrator. All data, thresholds and cost/carbon factors are "
            "illustrative assumptions, not validated for real industrial decisions."
        )

    record = records[cid]
    title = showcase_titles().get(cid, record.get("component_type") or "Component")
    assessment = assess_component(record, bundle, weights, model)
    return AppContext(df, bundle, cid, title, record, weights, model, assessment)
