"""Tab 7: method summary, assumptions, open research questions and limitations."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from circularloop.config import (
    COMPONENT_TYPES,
    DEFAULT_GATES,
    STRATEGIES,
    STRATEGY_DEFINITIONS,
    STRATEGY_ECONOMICS,
)
from ui.state import AppContext

PIPELINE_DOT = """
digraph {
  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.3;
  node [shape=box, style="rounded,filled", fillcolor="#f1f4f7", color="#c9d1d9",
        fontname="Helvetica", fontsize=11, margin="0.15,0.08"];
  edge [color="#8a96a3", arrowsize=0.6];
  rec0 [label="Lifecycle\\ndata record"];
  val [label="Data\\nvalidation"];
  rul [label="RUL prediction\\n(LR vs. boosting)"];
  feas [label="Technical\\nfeasibility + gates"];
  eval [label="Cost / CO₂ /\\nservice life"];
  mcda [label="MCDA\\n(weights)"];
  rec [label="Strategy\\nrecommendation", fillcolor="#dcefee", color="#0f6a6e"];
  rec0 -> val -> rul -> feas -> eval -> mcda -> rec;
  val -> stop [style=dashed];
  stop [label="Insufficient data:\\nno recommendation", fillcolor="#fbeaea", color="#d9a0a0"];
}
"""

RESEARCH_QUESTIONS = (
    "Which lifecycle-record attributes are actually required for circular decisions, and which are nice to have?",
    "How should missing or uncertain lifecycle data affect recommendations: imputation, wider intervals, abstention?",
    "How accurately can remaining useful life be estimated from real industrial data, which is sparse and censored?",
    "How should lifecycle cost and lifecycle carbon be estimated, and allocated between successive life cycles?",
    "Who should set MCDA weights (engineers, lifecycle-cost models, policy targets), and how is disagreement handled?",
    "How should recommendations be validated against expert decisions and real outcomes?",
    "How can lifecycle records be exchanged across organisations without losing semantics or exposing sensitive data?",
    "How should prediction uncertainty enter strategy selection beyond a conservative bound on safety gates?",
)

LIMITATIONS = (
    "All data are synthetic; the hidden degradation function is an assumption, not an empirical model.",
    "Metrics describe how well models recover that synthetic function; they say nothing about real components.",
    "Cost, CO₂ and life-extension factors are illustrative orders of magnitude, not LCC/LCA results.",
    "Gate thresholds are plausible placeholders, not engineering or safety standards.",
    "Maintenance-history features are confounded with age in observational data; monotonic constraints "
    "encode a prior but do not establish causality.",
    "The record structure is conceptual and not compliant with any regulatory product-data, AAS or EPCIS spec.",
)


def render(ctx: AppContext) -> None:
    st.subheader("How the demonstrator works")
    st.graphviz_chart(PIPELINE_DOT)
    st.caption(
        "Prediction (how long can it last?) is kept separate from decision support (what should we do with it?). "
        "The ML model never outputs a strategy."
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True, height="stretch"):
        st.markdown("**This demonstrator is**")
        st.markdown(
            "- an independent synthetic research prototype\n"
            "- inspired by publicly described circular lifecycle decision-support challenges\n"
            "- a way to make modelling assumptions explicit and discussable"
        )
        st.markdown("**This demonstrator is not**")
        st.markdown(
            "- an official implementation of any research project, university or company\n"
            "- compliant with any regulatory product-data standard\n"
            "- validated for real industrial or safety-relevant decisions"
        )
    with c2, st.container(border=True, height="stretch"):
        st.markdown("**Questions this prototype raises**")
        st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(RESEARCH_QUESTIONS, start=1)))

    with st.container(border=True):
        st.markdown("**Known limitations**")
        st.markdown("\n".join(f"- {item}" for item in LIMITATIONS))
        st.markdown(
            "**Potential future interoperability layer.** A real implementation could map this simplified record "
            "towards Asset Administration Shell submodels, GS1 EPCIS lifecycle events, OPC UA condition data and "
            "emerging EU product-data requirements under the Ecodesign for Sustainable Products Regulation (ESPR). "
            "No such mapping is implemented here."
        )

    st.markdown("#### Illustrative assumptions")
    st.caption("All values live in `circularloop/config.py` and are documented in `docs/methodology.md`.")
    t1, t2, t3 = st.tabs(["Strategies", "Hard gates", "Component types"])
    with t1:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Strategy": s,
                        "Definition": STRATEGY_DEFINITIONS[s],
                        "Cost (× new)": STRATEGY_ECONOMICS[s].cost_fraction,
                        "CO₂ (× new)": STRATEGY_ECONOMICS[s].co2_fraction,
                        "Life model": {
                            "rul": "predicted RUL",
                            "restore": f"RUL + {STRATEGY_ECONOMICS[s].life_fraction:.0%} of consumed life",
                            "fixed": f"{STRATEGY_ECONOMICS[s].life_fraction:.0%} of design life",
                        }[STRATEGY_ECONOMICS[s].life_mode],
                        "Circularity": STRATEGY_ECONOMICS[s].circularity,
                    }
                    for s in STRATEGIES
                ]
            ),
            hide_index=True,
            column_config={"Definition": st.column_config.TextColumn(width="large")},
        )
        st.caption(
            "Recycle cost and CO₂ refer to the new replacement component, minus scrap and recycled-content credits."
        )
    with t2:
        st.dataframe(
            pd.DataFrame([{"Threshold": k.replace("_", " "), "Value": v} for k, v in asdict(DEFAULT_GATES).items()]),
            hide_index=True,
        )
    with t3:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Type": name,
                        "Design life (y)": spec.design_life_years,
                        "Nominal h/year": spec.nominal_hours_per_year,
                        "Vibration alarm (mm/s)": spec.vibration_alarm_mm_s,
                        "Temperature limit (°C)": spec.temperature_limit_c,
                        "Materials": ", ".join(f"{m} {p:.0%}" for m, p in spec.materials.items()),
                    }
                    for name, spec in COMPONENT_TYPES.items()
                ]
            ),
            hide_index=True,
        )
