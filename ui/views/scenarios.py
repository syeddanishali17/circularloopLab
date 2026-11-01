"""Tab 3: scenario comparison (current / improved maintenance / severe degradation)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from circularloop.config import SCENARIO_CURRENT, SCENARIOS
from circularloop.record_schema import FIELD_BY_NAME
from circularloop.scenarios import SCENARIO_SPECS, all_scenarios
from ui import charts
from ui.components import STRATEGY_ICONS, confidence_badge, fmt, insufficient_panel, show_chart
from ui.state import AppContext

COMPARED_FIELDS = (
    "corrosion_score",
    "residual_thickness_pct",
    "vibration_rms",
    "failure_events",
    "temperature_mean_c",
    "last_maintenance_months_ago",
)


def render(ctx: AppContext) -> None:
    if not ctx.assessment.has_recommendation:
        insufficient_panel(ctx.assessment.validation)
        return

    st.caption(
        "The same pipeline is re-run on modified copies of the lifecycle record. "
        "This shows how sensitive the recommendation is to plausible changes in condition."
    )
    records = all_scenarios(ctx.record)
    results = {name: ctx.assess(rec) for name, rec in records.items()}
    base_rul = results[SCENARIO_CURRENT].rul

    cols = st.columns(len(SCENARIOS), gap="medium")
    for col, name in zip(cols, SCENARIOS):
        res = results[name]
        with col, st.container(border=True, height="stretch"):
            st.markdown(f"**{name}**")
            st.caption(SCENARIO_SPECS[name].description)
            if not res.has_recommendation:
                st.warning("Insufficient data for a reliable recommendation.")
                continue
            delta = None if name == SCENARIO_CURRENT else f"{res.rul.point - base_rul.point:+.0f} months"
            st.metric(
                "Predicted RUL",
                f"{res.rul.point:.0f} months",
                delta=delta,
                delta_description="vs. current" if delta else None,
            )
            st.markdown(f"#### {STRATEGY_ICONS[res.recommended]} {res.recommended}")
            st.caption(f"Score {res.decision.get(res.recommended).score:.0f}/100 · lead {res.decision.margin:.1f}")
            confidence_badge(res.confidence)

    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1:
        with st.container(border=True, height="stretch"):
            st.markdown("**Strategy scores per scenario**")
            show_chart(
                charts.scenario_scores({n: r.decision for n, r in results.items()}),
                alt="Grouped bar chart of strategy scores per scenario",
            )
            st.caption("A score of 0 means the strategy is blocked by a hard gate in that scenario.")
    with c2:
        with st.container(border=True, height="stretch"):
            st.markdown("**RUL with approximate 80% range**")
            show_chart(charts.scenario_rul({n: r.rul for n, r in results.items()}), alt="Predicted RUL per scenario")

    with st.container(border=True):
        st.markdown("**Changed inputs**")
        table = pd.DataFrame(
            {name: [fmt(records[name].get(f), 2) for f in COMPARED_FIELDS] for name in SCENARIOS},
            index=[FIELD_BY_NAME[f].label for f in COMPARED_FIELDS],
        )
        st.dataframe(table, alt="Input values per scenario")

    blocked_rows = []
    for name in SCENARIOS:
        res = results[name]
        if res.decision:
            for a in res.decision.assessments:
                if a.blocked:
                    blocked_rows.append(
                        {"Scenario": name, "Strategy": a.strategy, "Blocked because": "; ".join(a.block_reasons)}
                    )
    if blocked_rows:
        with st.expander("Which gates block which strategies in each scenario", icon=":material/block:"):
            st.dataframe(pd.DataFrame(blocked_rows), hide_index=True)
