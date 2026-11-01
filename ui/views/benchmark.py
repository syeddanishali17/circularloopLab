"""Tab 5: RUL model benchmark and explainability."""

from __future__ import annotations

import numpy as np
import streamlit as st

from circularloop.config import MODEL_GBR, MODEL_LINEAR
from circularloop.record_schema import FIELD_BY_NAME, is_missing
from circularloop.rul_model import response_curve
from ui import charts
from ui.components import show_chart
from ui.state import AppContext

CURVE_RANGES = {
    "corrosion_score": (0.0, 10.0),
    "residual_thickness_pct": (40.0, 100.0),
    "vibration_rms": (0.2, 10.0),
    "failure_events": (0.0, 8.0),
    "age_years": (0.5, 45.0),
    "temperature_mean_c": (0.0, 110.0),
}


def render(ctx: AppContext) -> None:
    b = ctx.bundle
    st.info(
        "Regression predicts remaining life. It does **not** by itself decide the circular strategy; "
        "that is the job of the transparent decision layer.",
        icon=":material/info:",
    )
    st.caption(
        f"Trained on {b.n_train} and tested on {b.n_test} synthetic components (75/25 split, seed 42). "
        f"{b.n_excluded} records were excluded because they failed validation or carry no label. "
        "All results describe the synthetic demonstrator only."
    )

    metrics = b.metrics.copy()
    best_test = metrics.loc[[MODEL_LINEAR, MODEL_GBR], "MAE (months)"].idxmin()
    cols = st.columns(2, gap="medium")
    for col, name in zip(cols, (MODEL_LINEAR, MODEL_GBR)):
        m = metrics.loc[name]
        with col, st.container(border=True):
            st.markdown(f"**{name}**" + ("  :blue-badge[used by default]" if name == b.default_model else ""))
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{m['MAE (months)']:.1f} mo")
            c2.metric("RMSE", f"{m['RMSE (months)']:.1f} mo")
            c3.metric("R²", f"{m['R²']:.3f}")
            st.caption(f"5-fold CV MAE on training split: {m['CV MAE (train, months)']:.1f} months")

    other = MODEL_LINEAR if best_test == MODEL_GBR else MODEL_GBR
    gap = metrics.loc[other, "MAE (months)"] - metrics.loc[best_test, "MAE (months)"]
    st.markdown(
        f"On the held-out set, **{best_test}** has the lower MAE by {gap:.1f} months. "
        "The default model is chosen by cross-validated MAE on the training split, so the test set is never "
        "used for model selection. The boosting model uses monotonic constraints (e.g. more corrosion can never "
        "increase predicted life); the unconstrained variant is reported below as an ablation."
    )
    with st.expander("Full metrics table (including ablation)", icon=":material/table_chart:"):
        st.dataframe(metrics.style.format("{:.3f}"), alt="Model metrics")

    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True, height="stretch"):
        st.markdown("**Predicted vs. actual RUL (held-out set)**")
        show_chart(charts.predicted_vs_actual(b.test_frame), alt="Scatter of predicted against actual RUL")
    with c2, st.container(border=True, height="stretch"):
        st.markdown("**Prediction error distribution**")
        show_chart(charts.residual_hist(b.test_frame), alt="Histogram of prediction errors")

    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True, height="stretch"):
        st.markdown("**Top factors influencing RUL predictions**")
        show_chart(charts.importance_bar(b.permutation_importance), alt="Permutation importance per model")
        st.caption("Permutation importance on the held-out set: feature importance within the synthetic demonstrator.")
    with c2, st.container(border=True, height="stretch"):
        st.markdown("**Linear regression coefficients**")
        show_chart(charts.coefficient_bar(b.linear_coefficients), alt="Linear regression coefficients")
        st.caption(
            "Numeric features: change in predicted RUL per one standard deviation. Categories: difference to the "
            "reference level. Correlated features (age, hours, cycles) share effects, so read signs with care."
        )

    with st.container(border=True):
        st.markdown("**Model response for the selected component**")
        record = ctx.assessment.record_used
        if not ctx.assessment.has_recommendation:
            st.caption("Not available: the selected record failed validation.")
            return
        feature = st.segmented_control(
            "Vary one input, keep all others at the component's values",
            list(CURVE_RANGES.keys()),
            default="corrosion_score",
            required=True,
            format_func=lambda f: FIELD_BY_NAME[f].label,
            key="curve_feature",
        )
        lo, hi = CURVE_RANGES[feature]
        values = np.linspace(lo, hi, 60)
        if feature == "failure_events":
            values = np.arange(0, 9)
        curve = response_curve(b, record, feature, values)
        current = None if is_missing(record.get(feature)) else float(record[feature])
        show_chart(charts.response_curve_chart(curve, feature, current), alt="Response curve")
        st.caption(
            "Linear regression responds as a straight line; boosting can capture thresholds and saturation. "
            "Caution: corrosion and residual thickness are strongly correlated in the fleet, so moving one while "
            "holding the other fixed leaves the region the models were trained on. This is one reason the "
            "decision layer "
            "relies on hard gates rather than on the RUL prediction alone."
        )
