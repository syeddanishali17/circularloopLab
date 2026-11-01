"""Tab 4: research sandbox: edit condition values and watch RUL and ranking respond."""

from __future__ import annotations

from typing import Any

import streamlit as st

from circularloop.config import CRITICAL_FIELDS
from circularloop.record_schema import FIELD_BY_NAME, is_missing
from ui import charts
from ui.components import STRATEGY_ICONS, confidence_badge, insufficient_panel, issues_list, quality_badge, show_chart
from ui.state import AppContext

# (field, min, max, step, fallback); the fallback is used when the record has no value.
SLIDERS: tuple[tuple[str, float, float, float, float], ...] = (
    ("corrosion_score", 0.0, 10.0, 0.1, 3.0),
    ("residual_thickness_pct", 30.0, 100.0, 0.5, 85.0),
    ("vibration_rms", 0.1, 15.0, 0.05, 2.0),
    ("failure_events", 0.0, 10.0, 1.0, 0.0),
    ("operating_hours", 0.0, 400_000.0, 500.0, 50_000.0),
    ("temperature_mean_c", -20.0, 130.0, 0.5, 40.0),
    ("recyclability_pct", 0.0, 100.0, 0.5, 90.0),
    ("replacement_cost_eur", 500.0, 250_000.0, 500.0, 20_000.0),
    ("embodied_co2_kg", 50.0, 20_000.0, 50.0, 2_000.0),
)
MISSABLE = (
    "vibration_rms",
    "load_cycles",
    "temperature_mean_c",
    "maintenance_count",
    "last_maintenance_months_ago",
    "residual_thickness_pct",
    "corrosion_score",
)


def _key(ctx: AppContext, name: str) -> str:
    return f"sandbox_{ctx.component_id}_{name}"


def _initial(ctx: AppContext, name: str, lo: float, hi: float, fallback: float) -> float:
    value = ctx.record.get(name)
    return fallback if is_missing(value) else min(hi, max(lo, float(value)))


def _reset(ctx: AppContext) -> None:
    for name, lo, hi, _, fallback in SLIDERS:
        st.session_state[_key(ctx, name)] = _initial(ctx, name, lo, hi, fallback)
    st.session_state[_key(ctx, "missing")] = []


def _inputs(ctx: AppContext) -> dict[str, Any]:
    edited = dict(ctx.record)
    left, right = st.columns(2)
    for i, (name, lo, hi, step, fallback) in enumerate(SLIDERS):
        spec = FIELD_BY_NAME[name]
        key = _key(ctx, name)
        st.session_state.setdefault(key, _initial(ctx, name, lo, hi, fallback))
        with left if i % 2 == 0 else right:
            value = st.slider(f"{spec.label} ({spec.unit})", lo, hi, step=step, key=key)
        edited[name] = int(value) if spec.kind == "int" else value
    missing = st.multiselect(
        "Treat as missing (simulate an incomplete record)",
        MISSABLE,
        format_func=lambda f: FIELD_BY_NAME[f].label + (" · critical" if f in CRITICAL_FIELDS else ""),
        key=_key(ctx, "missing"),
    )
    for name in missing:
        edited[name] = None
    st.button("Reset to record values", on_click=_reset, args=(ctx,), icon=":material/restart_alt:")
    return edited


@st.fragment
def render(ctx: AppContext) -> None:
    st.caption(
        "Change condition or economic values to explore how the predicted RUL and the strategy ranking respond. "
        "Changes stay in this tab; the stored record itself is not modified."
    )
    original = ctx.assessment
    inputs_col, result_col = st.columns([1, 1.25], gap="large")
    with inputs_col:
        with st.container(border=True):
            st.markdown("**Edit record values**")
            edited = _inputs(ctx)
    modified = ctx.assess(edited)

    with result_col:
        with st.container(border=True):
            st.markdown("**Result for the modified record**")
            quality_badge(modified.validation)
            if not modified.has_recommendation:
                insufficient_panel(modified.validation)
                issues_list(modified.validation, include_info=False)
                return
            c1, c2 = st.columns(2)
            rul_delta = None
            if original.rul is not None:
                rul_delta = f"{modified.rul.point - original.rul.point:+.0f} months vs. record"
            c1.metric("Predicted RUL", f"{modified.rul.point:.0f} months", delta=rul_delta)
            c1.caption(f"Approx. 80% range {modified.rul.lower:.0f}–{modified.rul.upper:.0f} months")
            changed = original.recommended is not None and original.recommended != modified.recommended
            c2.metric(
                "Recommendation",
                f"{modified.recommended}",
                delta=f"was {original.recommended}" if changed else None,
                delta_color="off",
            )
            c2.markdown(
                f"{STRATEGY_ICONS[modified.recommended]} score "
                f"{modified.decision.get(modified.recommended).score:.0f}/100"
            )
            confidence_badge(modified.confidence)
            if original.decision is not None:
                show_chart(
                    charts.comparison_scores(original.decision, modified.decision),
                    alt="Strategy scores before and after editing",
                )
            else:
                show_chart(charts.ranking_bar(modified.decision), alt="Strategy ranking for the edited record")
            st.markdown(modified.explanation.paragraph)
            if modified.validation.issues:
                with st.expander("Validation findings for the modified record"):
                    issues_list(modified.validation)
