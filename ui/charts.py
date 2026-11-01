"""Plotly figure builders with one consistent, restrained visual style."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from circularloop.config import CRITERIA, CRITERION_LABELS, MODEL_GBR, MODEL_LINEAR, STRATEGIES
from circularloop.decision_engine import DecisionResult
from circularloop.record_schema import FIELD_BY_NAME
from circularloop.rul_model import RulPrediction

# Categorical slots (validated adjacent-order palette); colour follows the entity.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
# Neutrals and the single accent colour mirror .streamlit/config.toml.
PRIMARY = "#0f6a6e"
PRIMARY_LIGHT = "#b9dedc"
NEUTRAL = "#b3bcc6"
BLOCKED = "#e4e8ec"
INK = "#17212b"
MUTED = "#5a6673"
GRID = "#e6e9ed"
FONT_FAMILY = "Inter, 'Source Sans 3', sans-serif"

CRITERION_COLORS: Mapping[str, str] = dict(zip(CRITERIA, SLOTS))
CRITERION_SHORT: Mapping[str, str] = {
    "feasibility": "Feasibility",
    "co2": "CO₂ / yr",
    "cost": "Cost / yr",
    "life": "Life",
    "circularity": "Circularity",
}
MODEL_COLORS: Mapping[str, str] = {MODEL_LINEAR: SLOTS[0], MODEL_GBR: SLOTS[1]}
# Current = neutral ink, improved = blue, severe = orange (warning connotation without status red).
SCENARIO_COLORS: tuple[str, ...] = (MUTED, SLOTS[0], SLOTS[1])


def _style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=13, color=INK),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT_FAMILY, size=12, color=INK)),
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(color=MUTED, size=12))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(color=MUTED, size=12))
    return fig


# ---------------------------------------------------------------------------
# Decision charts
# ---------------------------------------------------------------------------


def ranking_bar(decision: DecisionResult) -> go.Figure:
    """Horizontal ranking; recommended strategy highlighted, blocked strategies greyed."""
    rows = list(reversed(decision.assessments))
    colors = [BLOCKED if a.blocked else PRIMARY if a.strategy == decision.recommended else NEUTRAL for a in rows]
    labels = ["blocked" if a.blocked else f"{a.score:.0f}" for a in rows]
    fig = go.Figure(
        go.Bar(
            x=[max(a.score, 1.2) if a.blocked else a.score for a in rows],
            y=[a.strategy for a in rows],
            orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=labels,
            textposition="outside",
            textfont=dict(color=MUTED),
            cliponaxis=False,
            customdata=[("; ".join(a.block_reasons) or "passes all gates") for a in rows],
            hovertemplate="<b>%{y}</b><br>Score %{x:.1f}/100<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 108], title="Decision score (0–100)")
    return _style(fig, height=250, legend=False)


def contribution_stack(decision: DecisionResult) -> go.Figure:
    """Weighted contribution of each criterion to every strategy's score."""
    rows = list(reversed(decision.assessments))
    fig = go.Figure()
    for c in CRITERIA:
        fig.add_bar(
            name=CRITERION_LABELS[c],
            y=[a.strategy for a in rows],
            x=[a.contributions[c] for a in rows],
            orientation="h",
            marker=dict(color=CRITERION_COLORS[c], line=dict(color="white", width=2)),
            hovertemplate=f"<b>%{{y}}</b><br>{CRITERION_LABELS[c]}: %{{x:.1f}} pts<extra></extra>",
        )
    for a in rows:
        if a.blocked:
            fig.add_annotation(
                x=1,
                y=a.strategy,
                text="blocked by hard gate",
                showarrow=False,
                xanchor="left",
                font=dict(color=MUTED, size=12),
            )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 100], title="Weighted points")
    return _style(fig, height=300)


def criteria_heatmap(decision: DecisionResult) -> go.Figure:
    """Normalised criterion scores (0–1) per strategy, in fixed strategy order."""
    ordered = [decision.get(s) for s in STRATEGIES]
    z = [[a.criteria[c] for c in CRITERIA] for a in ordered]
    x = [CRITERION_SHORT[c] for c in CRITERIA]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=[a.strategy + (" (blocked)" if a.blocked else "") for a in ordered],
            zmin=0,
            zmax=1,
            colorscale=[[0, "#f2f7f7"], [0.5, "#7cc2bf"], [1, "#0f5357"]],
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(size=12),
            hovertemplate="%{y} · %{x}<br>normalised score %{z:.2f}<extra></extra>",
            showscale=False,
            xgap=2,
            ygap=2,
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=0, side="top", tickfont=dict(size=12))
    return _style(fig, height=300, legend=False).update_layout(margin=dict(l=8, r=8, t=48, b=8))


def sensitivity_bar(decision: DecisionResult) -> go.Figure:
    """Share of perturbed weight sets in which each strategy ranks first."""
    shares = [(s, decision.win_shares.get(s, 0.0)) for s in STRATEGIES]
    fig = go.Figure(
        go.Bar(
            x=[s for s, _ in shares],
            y=[v * 100 for _, v in shares],
            marker=dict(color=[PRIMARY if s == decision.recommended else NEUTRAL for s, _ in shares], cornerradius=4),
            text=[f"{v:.0%}" if v > 0 else "" for _, v in shares],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>wins in %{y:.0f}% of weight draws<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 112], title="Wins (% of draws)")
    return _style(fig, height=260, legend=False)


def rul_bullet(rul: RulPrediction, design_months: float) -> go.Figure:
    """Point RUL with its approximate 80 % interval against the design life."""
    upper_axis = max(design_months, rul.upper) * 1.05
    fig = go.Figure()
    fig.add_bar(
        x=[design_months],
        y=[""],
        orientation="h",
        marker=dict(color="#eef1f4"),
        hovertemplate=f"Design life of a new part: {design_months:.0f} months<extra></extra>",
        width=0.55,
    )
    fig.add_bar(
        x=[rul.upper - rul.lower],
        base=[rul.lower],
        y=[""],
        orientation="h",
        marker=dict(color=PRIMARY_LIGHT),
        width=0.55,
        hovertemplate=f"Approx. 80% range: {rul.lower:.0f}–{rul.upper:.0f} months<extra></extra>",
    )
    fig.add_scatter(
        x=[rul.point],
        y=[""],
        mode="markers",
        marker=dict(symbol="line-ns", size=26, line=dict(width=4, color=PRIMARY)),
        hovertemplate=f"Predicted RUL: {rul.point:.0f} months<extra></extra>",
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(range=[0, upper_axis], title="Months")
    fig.update_yaxes(showticklabels=False)
    return _style(fig, height=120, legend=False).update_layout(margin=dict(l=8, r=16, t=8, b=8))


# ---------------------------------------------------------------------------
# Scenario charts
# ---------------------------------------------------------------------------


def scenario_scores(decisions: Mapping[str, DecisionResult | None]) -> go.Figure:
    """Grouped bars: strategy scores under each scenario (blocked = 0)."""
    fig = go.Figure()
    for color, (name, decision) in zip(SCENARIO_COLORS, decisions.items()):
        if decision is None:
            continue
        values = [decision.get(s).score for s in STRATEGIES]
        fig.add_bar(
            name=name,
            x=list(STRATEGIES),
            y=values,
            marker=dict(color=color, cornerradius=4, line=dict(color="white", width=2)),
            customdata=[("blocked" if decision.get(s).blocked else "") for s in STRATEGIES],
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.1f}} %{{customdata}}<extra></extra>",
        )
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_yaxes(range=[0, 100], title="Decision score")
    return _style(fig, height=320)


def scenario_rul(ruls: Mapping[str, RulPrediction | None]) -> go.Figure:
    """Point RUL with interval for each scenario."""
    names = [n for n, r in ruls.items() if r is not None]
    pts = [ruls[n] for n in names]
    fig = go.Figure()
    for color, name, r in zip(SCENARIO_COLORS, names, pts):
        fig.add_scatter(
            x=[r.point],
            y=[name],
            mode="markers",
            name=name,
            marker=dict(size=12, color=color, line=dict(color="white", width=2)),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[r.upper - r.point],
                arrayminus=[r.point - r.lower],
                color=color,
                thickness=2,
                width=6,
            ),
            hovertemplate=f"<b>{name}</b><br>RUL %{{x:.0f}} months ({r.lower:.0f}–{r.upper:.0f})<extra></extra>",
        )
    fig.update_xaxes(title="Predicted RUL (months)", rangemode="tozero")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, height=220, legend=False)


# ---------------------------------------------------------------------------
# Model benchmark charts
# ---------------------------------------------------------------------------


def predicted_vs_actual(test_frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    hi = float(np.nanmax(test_frame[["actual", MODEL_LINEAR, MODEL_GBR]].to_numpy())) * 1.05
    fig.add_scatter(
        x=[0, hi],
        y=[0, hi],
        mode="lines",
        line=dict(color=NEUTRAL, dash="dot", width=1.5),
        name="Perfect prediction",
        hoverinfo="skip",
    )
    for name in (MODEL_LINEAR, MODEL_GBR):
        fig.add_scatter(
            x=test_frame["actual"],
            y=test_frame[name],
            mode="markers",
            name=name,
            marker=dict(size=8, color=MODEL_COLORS[name], opacity=0.75, line=dict(color="white", width=1)),
            hovertemplate=f"<b>{name}</b><br>actual %{{x:.0f}} · predicted %{{y:.0f}} months<extra></extra>",
        )
    fig.update_xaxes(title="Synthetic actual RUL (months)", range=[0, hi])
    fig.update_yaxes(title="Predicted RUL (months)", range=[0, hi])
    return _style(fig, height=360)


def residual_hist(test_frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name in (MODEL_LINEAR, MODEL_GBR):
        fig.add_histogram(
            x=test_frame[name] - test_frame["actual"],
            name=name,
            opacity=0.65,
            marker=dict(color=MODEL_COLORS[name], line=dict(color="white", width=1)),
            xbins=dict(size=10),
            hovertemplate=f"{name}<br>error %{{x}} months: %{{y}} parts<extra></extra>",
        )
    fig.add_vline(x=0, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Prediction error (predicted − actual, months)")
    fig.update_yaxes(title="Components")
    return _style(fig, height=360)


def importance_bar(perm: pd.DataFrame) -> go.Figure:
    order = perm.groupby("feature")["importance"].max().sort_values().index.tolist()
    fig = go.Figure()
    for name in (MODEL_LINEAR, MODEL_GBR):
        sub = perm[perm["model"] == name].set_index("feature").loc[order]
        fig.add_bar(
            y=[FIELD_BY_NAME[f].label for f in order],
            x=sub["importance"],
            name=name,
            orientation="h",
            error_x=dict(type="data", array=sub["std"], color=MUTED, thickness=1),
            marker=dict(color=MODEL_COLORS[name], cornerradius=3),
            hovertemplate=f"<b>{name}</b><br>%{{y}}: MAE +%{{x:.1f}} months when shuffled<extra></extra>",
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Increase in test MAE when the feature is shuffled (months)")
    return _style(fig, height=460)


def _encoded_label(feature: str) -> str:
    """Readable label for raw or one-hot-encoded feature names."""
    if feature in FIELD_BY_NAME:
        return FIELD_BY_NAME[feature].label
    for prefix, label in (("component_type_", "Type"), ("material_", "Material")):
        if feature.startswith(prefix):
            return f"{label}: {feature[len(prefix) :]}"
    return feature


def coefficient_bar(coefs: pd.DataFrame, top: int = 14) -> go.Figure:
    sub = coefs.head(top).iloc[::-1]
    labels = [_encoded_label(f) for f in sub["feature"]]
    fig = go.Figure(
        go.Bar(
            y=labels,
            x=sub["coefficient"],
            orientation="h",
            marker=dict(color=[SLOTS[0] if v >= 0 else SLOTS[1] for v in sub["coefficient"]], cornerradius=3),
            customdata=sub["kind"],
            hovertemplate="<b>%{y}</b><br>%{x:+.1f} months (%{customdata})<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title="Coefficient (months)")
    return _style(fig, height=440, legend=False)


def response_curve_chart(curve: pd.DataFrame, feature: str, current: float | None) -> go.Figure:
    fig = go.Figure()
    for name in (MODEL_LINEAR, MODEL_GBR):
        fig.add_scatter(
            x=curve[feature],
            y=curve[name],
            mode="lines",
            name=name,
            line=dict(color=MODEL_COLORS[name], width=2.5),
            hovertemplate=f"<b>{name}</b><br>%{{x:.2f}} → %{{y:.0f}} months<extra></extra>",
        )
    if current is not None:
        fig.add_vline(
            x=current,
            line=dict(color=MUTED, width=1, dash="dot"),
            annotation_text="current value",
            annotation_font_color=MUTED,
        )
    spec = FIELD_BY_NAME[feature]
    fig.update_xaxes(title=f"{spec.label} ({spec.unit})")
    fig.update_yaxes(title="Predicted RUL (months)", rangemode="tozero")
    return _style(fig, height=340)


def comparison_scores(before: DecisionResult, after: DecisionResult) -> go.Figure:
    """Original vs. modified strategy scores (sandbox)."""
    fig = go.Figure()
    for color, name, d in ((NEUTRAL, "Original record", before), (PRIMARY, "Modified record", after)):
        fig.add_bar(
            name=name,
            x=list(STRATEGIES),
            y=[d.get(s).score for s in STRATEGIES],
            marker=dict(color=color, cornerradius=4, line=dict(color="white", width=2)),
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.1f}}<extra></extra>",
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 100], title="Decision score")
    return _style(fig, height=300)
