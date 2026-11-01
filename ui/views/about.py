"""Tab 0, About: the problem, the approach and how to explore the demonstrator."""

from __future__ import annotations

import streamlit as st

from circularloop.config import COMPONENT_TYPES, MODEL_NAMES, N_COMPONENTS, STRATEGIES
from circularloop.record_schema import FIELDS
from ui.state import AppContext

APPROACH = (
    (
        ":material/fact_check:",
        "1 · Check the record",
        "Every lifecycle record is validated first. Missing or impossible values are flagged; if critical "
        "information is missing, the tool says so and does not recommend anything.",
    ),
    (
        ":material/monitoring:",
        "2 · Estimate remaining life",
        "A regression model predicts remaining useful life from usage and condition data, with an "
        "approximate uncertainty range. A simple linear baseline is benchmarked against gradient boosting.",
    ),
    (
        ":material/rule:",
        "3 · Screen technical feasibility",
        "Hard engineering gates rule out unsafe options (for example, no direct reuse with heavy corrosion "
        "or thin walls), no matter how attractive they look on cost or carbon.",
    ),
    (
        ":material/balance:",
        "4 · Compare strategies openly",
        "The remaining options are compared on cost, carbon, service life, feasibility and resource "
        "retention, with adjustable weights, a visible score breakdown and a robustness check.",
    ),
)

DESIGN_CHOICES = (
    "**Prediction and decision are kept apart.** The model estimates how long a component can last; it never "
    "outputs a strategy. The choice is made by a transparent, inspectable decision layer.",
    "**Fair comparison.** Strategies deliver very different amounts of service, so cost and carbon are compared "
    "per year of service delivered, and recycling includes the new part that replaces the old one.",
    "**Safety cannot be traded away.** Hard gates block unsafe options regardless of how the weights are set.",
    "**Uncertainty is shown, not hidden.** Remaining life comes with a range, the reuse check uses its "
    "conservative end, and every recommendation reports its margin and how stable it is under other weightings.",
    "**Engineering knowledge constrains the model.** Monotonic constraints stop the model from learning "
    "implausible patterns, e.g. that more corrosion could extend life.",
    "**Knowing when not to answer.** Incomplete or contradictory records lead to an explicit "
    "“insufficient data” result instead of a confident guess.",
)

EXPLORE = (
    ("SHOW-B", "a mid-life gearbox: a genuinely close call between remanufacturing and recycling"),
    ("SHOW-C", "a heavily corroded frame: most options are blocked by technical gates"),
    ("SHOW-A", "a healthy, young drive shaft: direct reuse comes out on top"),
    ("SHOW-E", "a record without a thickness measurement: the tool declines to recommend"),
)


def render(ctx: AppContext) -> None:
    st.subheader("Turning lifecycle data into transparent circular decisions")
    st.markdown(
        "Durable industrial components such as gearboxes, drive shafts, rail bogie frames, structural "
        "profiles and motor housings eventually reach a decision point: **reuse, repair, refurbish, remanufacture or "
        "recycle?** The choice affects safety, cost, carbon emissions and how much material stays in use. "
        "In practice it is often made from scattered records, experience and fixed rules of thumb."
    )
    st.markdown(
        "At the same time, structured digital lifecycle records for products are becoming more common: usage, "
        "condition monitoring, maintenance history and material data that travel with a component. This project "
        "explores a simple question: **how could such data support better, explainable end-of-use decisions?**"
    )

    stats = (
        ("Components", N_COMPONENTS),
        ("Component types", len(COMPONENT_TYPES)),
        ("Record fields", len(FIELDS)),
        ("RUL models", len(MODEL_NAMES)),
        ("Strategies", len(STRATEGIES)),
    )
    for col, (label, value) in zip(st.columns(len(stats)), stats):
        col.metric(label, f"{value}", border=True)

    st.markdown("#### Why this is hard")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(
            "- Lifecycle records are often **incomplete or inconsistent**.\n"
            "- **Remaining life is uncertain**, and real failure data are scarce.\n"
            "- Objectives **conflict**: cost, carbon, safety and resource retention rarely agree."
        )
    with c2:
        st.markdown(
            "- Some constraints are **non-negotiable**: safety cannot be offset by savings.\n"
            "- Engineers need to see **why** a recommendation was made, not just a score.\n"
            "- Weightings are a matter of **judgement** and should be open to discussion."
        )

    st.markdown("#### The approach")
    cols = st.columns(len(APPROACH), gap="medium")
    for col, (icon, title, text) in zip(cols, APPROACH):
        with col, st.container(border=True, height="stretch"):
            st.markdown(f"**{icon} {title}**")
            st.caption(text)

    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        st.markdown("#### Design choices")
        st.markdown("\n".join(f"- {item}" for item in DESIGN_CHOICES))
    with c2:
        st.markdown("#### How to explore")
        with st.container(border=True):
            st.markdown(
                "Pick a case in the sidebar, then open **Decision** and **Why this recommendation?** "
                "(outcomes below are at the default weights):"
            )
            st.markdown("\n".join(f"- **{cid}**: {text}" for cid, text in EXPLORE))
            st.markdown(
                "Then try **Scenarios** to see the effect of better maintenance or faster degradation, the "
                "**Sandbox** to change condition values yourself, and **Benchmark** for the prediction side."
            )

    st.markdown("#### Scope")
    st.caption(
        "This is an independent research demonstrator built on synthetic data. Thresholds, cost and carbon "
        "factors are illustrative assumptions chosen to be plausible, not validated engineering values, and the "
        "record format is a simplified concept rather than a regulatory standard. It is intended to make the "
        "modelling choices visible and discussable. The **Method** tab lists the assumptions, limitations and "
        "the open research questions the prototype raises."
    )
