"""Reusable Streamlit building blocks (cards, badges, tables)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from circularloop.config import COMPONENT_TYPES, STRATEGY_DEFINITIONS
from circularloop.decision_engine import DecisionResult
from circularloop.pipeline import INSUFFICIENT_MESSAGE, Assessment
from circularloop.record_schema import is_missing
from circularloop.validation import STATUS_INSUFFICIENT, STATUS_SUFFICIENT, ValidationReport
from ui import charts

STRATEGY_ICONS: Mapping[str, str] = {
    "Reuse": ":material/autorenew:",
    "Repair": ":material/build:",
    "Refurbish": ":material/format_paint:",
    "Remanufacture": ":material/precision_manufacturing:",
    "Recycle": ":material/recycling:",
}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def fmt(value: Any, digits: int = 0, unit: str = "") -> str:
    """Format a possibly-missing number for display."""
    if is_missing(value):
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{number:,.{digits}f}"
    return f"{text} {unit}".strip()


CHART_CONFIG: Mapping[str, Any] = {"displayModeBar": False}


def show_chart(fig: Any, alt: str) -> None:
    """Render a Plotly figure with the shared configuration."""
    st.plotly_chart(fig, config=dict(CHART_CONFIG), alt=alt)


# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------


def quality_badge(report: ValidationReport) -> None:
    if report.status == STATUS_SUFFICIENT:
        st.badge(f"Data quality: {report.status}", icon=":material/check_circle:", color="green")
    elif report.status == STATUS_INSUFFICIENT:
        st.badge(f"Data quality: {report.status}", icon=":material/error:", color="red")
    else:
        st.badge(f"Data quality: {report.status}", icon=":material/warning:", color="orange")


def confidence_badge(confidence: str | None) -> None:
    if confidence is None:
        return
    color = {"High": "green", "Medium": "orange", "Low": "red"}[confidence]
    icon = {"High": ":material/verified:", "Medium": ":material/help:", "Low": ":material/report:"}[confidence]
    st.badge(f"Decision confidence: {confidence}", icon=icon, color=color)


def synthetic_badges() -> None:
    with st.container(horizontal=True, gap="small", horizontal_alignment="right"):
        st.badge("Synthetic data", icon=":material/science:", color="gray")
        st.badge("Research demo", icon=":material/school:", color="blue")
        st.badge("Not for operational use", icon=":material/do_not_disturb_on:", color="orange")


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def identity_header(record: Mapping[str, Any], component_id: str, title: str) -> None:
    ctype = record.get("component_type")
    spec = COMPONENT_TYPES.get(ctype) if isinstance(ctype, str) else None
    st.subheader(f"{component_id} · {title}")
    with st.container(horizontal=True, gap="small"):
        st.badge(str(ctype or "unknown type"), icon=":material/settings:", color="gray")
        st.badge(str(record.get("material") or "unknown material"), icon=":material/layers:", color="gray")
        st.badge(f"In service {fmt(record.get('age_years'), 1)} y", icon=":material/schedule:", color="gray")
        if spec:
            st.badge(
                f"Design life {spec.design_life_years:.0f} y (illustrative)", icon=":material/timelapse:", color="gray"
            )


def condition_metrics(record: Mapping[str, Any]) -> None:
    """Lifecycle and condition data as a compact metric grid."""
    ctype = record.get("component_type")
    alarm = COMPONENT_TYPES[ctype].vibration_alarm_mm_s if ctype in COMPONENT_TYPES else None
    vib = record.get("vibration_rms")
    vib_help = f"Illustrative alarm level for this type: {alarm} mm/s" if alarm else None
    items = [
        ("Operating hours", fmt(record.get("operating_hours"), 0, "h"), None),
        ("Load cycles", fmt(record.get("load_cycles"), 0), "Counted major start/stop and load-reversal events."),
        ("Vibration RMS", fmt(vib, 2, "mm/s"), vib_help),
        ("Temperature", fmt(record.get("temperature_mean_c"), 1, "°C"), "Mean operating temperature."),
        (
            "Corrosion",
            f"{fmt(record.get('corrosion_score'), 1)} / 10",
            "Inspection-based, 0 = none, 10 = severe.",
        ),
        (
            "Wall thickness",
            fmt(record.get("residual_thickness_pct"), 0, "%"),
            "Remaining wall/section thickness vs. nominal.",
        ),
        (
            "Failures",
            f"{fmt(record.get('failure_events'))} / {fmt(record.get('maintenance_count'))}",
            "Recorded failure events / maintenance interventions.",
        ),
        ("Spare parts", fmt(record.get("spare_parts_available")), None),
    ]
    for row in (items[:4], items[4:]):
        cols = st.columns(4)
        for col, (label, value, help_text) in zip(cols, row):
            col.metric(label, value, help=help_text)


def issues_list(report: ValidationReport, include_info: bool = True) -> None:
    icons = {
        "critical": ":red[:material/error:]",
        "warning": ":orange[:material/warning:]",
        "info": ":blue[:material/info:]",
    }
    shown = [i for i in report.issues if include_info or i.severity != "info"]
    if not shown:
        st.caption(":material/check: No validation issues found.")
        return
    st.markdown("\n".join(f"- {icons[i.severity]} {i.message}" for i in shown))


def quality_panel(report: ValidationReport) -> None:
    with st.container(border=True, height="stretch"):
        st.markdown("**Data quality**")
        quality_badge(report)
        c1, c2 = st.columns(2)
        c1.metric(
            "Quality",
            f"{report.quality_score:.0f} / 100",
            help="Weighted completeness (critical fields ×3) minus penalties for warnings and imputed fields.",
        )
        c2.metric("Completeness", f"{report.completeness_pct:.0f}%")
        issues_list(report)


def insufficient_panel(report: ValidationReport) -> None:
    st.error(
        "The record contains critical gaps or impossible values, so no RUL prediction or strategy "
        "recommendation is produced. Correct the issues listed under Data quality, "
        "or try values in the Research sandbox.",
        title=INSUFFICIENT_MESSAGE,
        icon=":material/block:",
    )


def rul_panel(assessment: Assessment) -> None:
    rul = assessment.rul
    assert rul is not None
    design = COMPONENT_TYPES[assessment.record_used["component_type"]].design_life_months
    with st.container(border=True, height="stretch"):
        st.markdown("**Remaining useful life**")
        st.metric(
            "Point estimate",
            f"{rul.point:.0f} months",
            help="Synthetic-data model prediction; the range is an approximate 80% interval from "
            "cross-validated residuals.",
        )
        st.caption(f"Approx. 80% range **{rul.lower:.0f}–{rul.upper:.0f} months** · model: {rul.model}")
        show_chart(charts.rul_bullet(rul, design), alt="Predicted RUL with interval compared to design life")
        st.caption(
            "Marker: point estimate · teal band: approx. 80% range · grey: design life of a new part. "
            "Reuse gates use the conservative lower bound."
        )


def recommendation_panel(assessment: Assessment) -> None:
    decision = assessment.decision
    assert decision is not None
    winner = decision.get(decision.recommended)
    with st.container(border=True, height="stretch"):
        st.markdown("**Recommended strategy**")
        st.markdown(f"## {STRATEGY_ICONS[winner.strategy]} {winner.strategy}")
        st.caption(STRATEGY_DEFINITIONS[winner.strategy])
        c1, c2 = st.columns(2)
        c1.metric("Decision score", f"{winner.score:.0f} / 100")
        lead = f"+{decision.margin:.1f}" if decision.runner_up else "only option"
        c2.metric(
            "Lead",
            lead,
            help=f"Runner-up: {decision.runner_up}" if decision.runner_up else "All alternatives are blocked.",
        )
        confidence_badge(assessment.confidence)
        st.caption(f"Stays first in {decision.stability:.0%} of perturbed weight sets.")


def strategy_table(decision: DecisionResult) -> None:
    """Per-strategy comparison with absolute and annualised values."""
    df = decision.to_frame()
    st.dataframe(
        df,
        hide_index=True,
        column_order=[
            "Rank",
            "Strategy",
            "Status",
            "Score",
            "Feasibility",
            "Circularity",
            "Service life (months)",
            "Cost (EUR)",
            "Cost / service-year (EUR)",
            "CO₂ (kg)",
            "CO₂ / service-year (kg)",
            "Blocking reason",
        ],
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            "Feasibility": st.column_config.ProgressColumn("Feasibility", min_value=0, max_value=1, format="%.2f"),
            "Circularity": st.column_config.ProgressColumn("Circularity", min_value=0, max_value=1, format="%.2f"),
            "Service life (months)": st.column_config.NumberColumn(format="%.0f"),
            "Cost (EUR)": st.column_config.NumberColumn(format="€%,.0f"),
            "Cost / service-year (EUR)": st.column_config.NumberColumn(format="€%,.0f"),
            "CO₂ (kg)": st.column_config.NumberColumn(format="%,.0f"),
            "CO₂ / service-year (kg)": st.column_config.NumberColumn(format="%,.0f"),
            "Blocking reason": st.column_config.TextColumn(width="large"),
        },
        alt="Comparison of all five circular strategies",
    )


def frame_download(df: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", icon=":material/download:"
    )
