"""Tab 1: component decision overview."""

from __future__ import annotations

import streamlit as st

from ui import charts
from ui.components import (
    condition_metrics,
    identity_header,
    insufficient_panel,
    quality_panel,
    recommendation_panel,
    rul_panel,
    show_chart,
)
from ui.state import AppContext


def render(ctx: AppContext) -> None:
    a = ctx.assessment
    identity_header(ctx.record, ctx.component_id, ctx.title)

    left, right = st.columns([2.2, 1], gap="medium")
    with left:
        with st.container(border=True, height="stretch"):
            st.markdown("**Lifecycle and condition data** · from the record")
            condition_metrics(ctx.record)
    with right:
        quality_panel(a.validation)

    if not a.has_recommendation:
        insufficient_panel(a.validation)
        return

    c1, c2, c3 = st.columns([1, 1, 1.3], gap="medium")
    with c1:
        rul_panel(a)
    with c2:
        recommendation_panel(a)
    with c3:
        with st.container(border=True, height="stretch"):
            st.markdown("**Strategy ranking**")
            show_chart(charts.ranking_bar(a.decision), alt="Decision score of each circular strategy")
            st.caption("Grey bars are feasible alternatives; *blocked* strategies failed a hard technical gate.")

    exp = a.explanation
    with st.container(border=True):
        st.markdown(f"**{exp.headline}**")
        st.markdown(exp.paragraph)
        st.caption(exp.robustness + (" " + exp.data_note if exp.data_note else ""))
        st.caption(":material/arrow_forward: Open *Why this recommendation?* for the full trade-off analysis.")
