"""CircularLoop Lab: Streamlit entry point (layout and routing only).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="CircularLoop Lab",
    page_icon=":material/recycling:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.components import synthetic_badges  # noqa: E402  (page config must come first)
from ui.state import load_bundle, load_data, sidebar  # noqa: E402
from ui.views import about, benchmark, decision, record, research, sandbox, scenarios, why  # noqa: E402

TABS = (
    (":material/info: About", about.render),
    (":material/fact_check: Decision", decision.render),
    (":material/psychology_alt: Why this recommendation?", why.render),
    (":material/compare_arrows: Scenarios", scenarios.render),
    (":material/tune: Sandbox", sandbox.render),
    (":material/monitoring: Benchmark", benchmark.render),
    (":material/data_object: Record", record.render),
    (":material/menu_book: Method", research.render),
)

title_col, badge_col = st.columns([3, 2], vertical_alignment="bottom")
with title_col:
    st.title("CircularLoop Lab")
    st.markdown("**Circular strategy decision support from lifecycle data**")
with badge_col:
    synthetic_badges()
st.caption(
    "Independent synthetic research demonstrator inspired by publicly described circular lifecycle "
    "decision-support challenges  ·  lifecycle record → validation → RUL prediction → feasibility → "
    "cost / CO₂ / life → MCDA → recommendation"
)

df = load_data()
bundle = load_bundle()
ctx = sidebar(df, bundle)

tabs = st.tabs([label for label, _ in TABS], on_change="rerun", key="active_tab")
for tab, (_, render) in zip(tabs, TABS):
    if tab.open:
        with tab:
            render(ctx)
