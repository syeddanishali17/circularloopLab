"""Tab 2: why this recommendation won."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from circularloop.config import CRITERION_LABELS, STRATEGIES
from circularloop.decision_engine import gate_checks
from ui import charts
from ui.components import frame_download, insufficient_panel, show_chart, strategy_table
from ui.state import AppContext


def _gate_frame(ctx: AppContext) -> pd.DataFrame:
    a = ctx.assessment
    rows = []
    for s in STRATEGIES:
        feas = a.decision.get(s).feasibility
        for check in gate_checks(s, a.record_used, a.rul.lower, feas):
            rows.append(
                {
                    "Strategy": s,
                    "Rule": check.criterion,
                    "Observed": check.observed,
                    "Limit": f"{'≥' if check.kind == 'min' else '≤'} {check.limit:g}",
                    "Result": "pass" if check.passed else "BLOCKS",
                }
            )
    return pd.DataFrame(rows)


def render(ctx: AppContext) -> None:
    a = ctx.assessment
    if not a.has_recommendation:
        insufficient_panel(a.validation)
        return
    exp, decision = a.explanation, a.decision

    st.subheader(exp.headline)
    st.markdown(exp.paragraph)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True, height="stretch"):
            st.markdown("**What drives the result**")
            st.markdown("\n".join(f"- {s[0].upper() + s[1:]}." for s in exp.strengths) or "- n/a")
            if exp.trade_offs:
                st.markdown("\n".join(f"- {t}" for t in exp.trade_offs))
            st.caption(exp.robustness)
            if exp.data_note:
                st.caption(f":orange[:material/warning:] {exp.data_note}")
    with c2:
        with st.container(border=True, height="stretch"):
            st.markdown("**Other strategies**")
            st.markdown("\n".join(f"- {o}" for o in exp.others))

    st.markdown("#### Strategy comparison")
    st.caption(
        "Cost and CO₂ are illustrative estimates. Because strategies deliver different amounts of service, "
        "they are compared **per service-year**; *Recycle* includes the new replacement component."
    )
    strategy_table(decision)
    frame_download(decision.to_frame(), f"{ctx.component_id}_strategy_comparison.csv", "Download comparison (CSV)")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True, height="stretch"):
            st.markdown("**How each criterion contributes to the score**")
            show_chart(charts.contribution_stack(decision), alt="Stacked weighted criterion contributions")
            st.caption("Bar length = weight × normalised score × 100. Blocked strategies receive no points.")
    with c2:
        with st.container(border=True, height="stretch"):
            st.markdown("**Normalised criteria (0 = poor, 1 = best) and weights**")
            show_chart(charts.criteria_heatmap(decision), alt="Heatmap of normalised criterion scores")
            st.caption(
                "Weights: "
                + " · ".join(f"{charts.CRITERION_SHORT[c]} {w:.2f}" for c, w in decision.weights.items())
                + ". Cost and CO₂ are scored against a new component: 0.5 = same cost/carbon per service-year as "
                "new, 1 = free, 0 = twice as expensive. Life = share of a new part's design life."
            )

    c1, c2 = st.columns([1, 1.3], gap="medium")
    with c1:
        with st.container(border=True, height="stretch"):
            st.markdown("**Sensitivity to the weights**")
            show_chart(charts.sensitivity_bar(decision), alt="Share of perturbed weight sets won by each strategy")
            st.caption("400 random weight vectors drawn around the current weights (Dirichlet, fixed seed).")
    with c2:
        with st.container(border=True, height="stretch"):
            st.markdown("**Hard technical gates**")
            gates = _gate_frame(ctx)
            st.dataframe(
                gates,
                hide_index=True,
                height=300,
                column_config={"Observed": st.column_config.NumberColumn(format="%.2f")},
                alt="Hard gate checks per strategy",
            )
            st.caption("A single failed gate blocks a strategy regardless of the weights. Recycle has no gates.")

    with st.expander("Current weights and criterion definitions", icon=":material/tune:"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Criterion": [CRITERION_LABELS[c] for c in decision.weights],
                    "Weight": list(decision.weights.values()),
                }
            ),
            hide_index=True,
            column_config={"Weight": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")},
        )
