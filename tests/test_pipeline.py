"""End-to-end pipeline: insufficient data, scenarios, record export, explanations, branding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from circularloop.config import REUSE, SCENARIO_CURRENT, SCENARIO_IMPROVED, SCENARIO_SEVERE
from circularloop.pipeline import INSUFFICIENT_MESSAGE, assess_component, assessment_to_record
from circularloop.record_schema import SECTIONS, flatten_structured_record, record_to_json
from circularloop.rul_model import ModelBundle
from circularloop.scenarios import all_scenarios, apply_scenario
from circularloop.showcase import showcase_records

ROOT = Path(__file__).resolve().parent.parent


def test_insufficient_data_produces_no_recommendation(bundle: ModelBundle, healthy_record: dict[str, Any]) -> None:
    healthy_record["corrosion_score"] = None
    result = assess_component(healthy_record, bundle)
    assert result.rul is None and result.decision is None and not result.has_recommendation
    structured = assessment_to_record(result)
    assert structured["decision_support"]["status"] == INSUFFICIENT_MESSAGE


def test_imputed_record_lowers_confidence(bundle: ModelBundle, healthy_record: dict[str, Any]) -> None:
    complete = assess_component(healthy_record, bundle)
    partial = assess_component({**healthy_record, "vibration_rms": None, "load_cycles": None}, bundle)
    assert partial.has_recommendation
    assert partial.validation.quality_score < complete.validation.quality_score
    assert "imputed" in partial.explanation.data_note


def test_scenarios_do_not_mutate_input_and_keep_thickness(healthy_record: dict[str, Any]) -> None:
    original = dict(healthy_record)
    improved = apply_scenario(healthy_record, SCENARIO_IMPROVED)
    assert healthy_record == original
    assert improved["residual_thickness_pct"] == original["residual_thickness_pct"]
    assert improved["corrosion_score"] <= original["corrosion_score"]
    severe = apply_scenario(healthy_record, SCENARIO_SEVERE)
    assert severe["residual_thickness_pct"] < original["residual_thickness_pct"]
    assert 0 <= severe["corrosion_score"] <= 10


def test_scenario_rul_ordering_across_fleet(bundle: ModelBundle, fleet: pd.DataFrame) -> None:
    """Improved maintenance never lowers predicted RUL; severe degradation never raises it."""
    for record in fleet.drop(columns=["rul_months"]).head(60).to_dict(orient="records"):
        results = {n: assess_component(r, bundle) for n, r in all_scenarios(record).items()}
        current = results[SCENARIO_CURRENT].rul.point
        assert results[SCENARIO_IMPROVED].rul.point >= current - 1e-6
        assert results[SCENARIO_SEVERE].rul.point <= current + 1e-6


def test_severe_scenario_never_recommends_direct_reuse(bundle: ModelBundle, fleet: pd.DataFrame) -> None:
    for record in fleet.drop(columns=["rul_months"]).head(120).to_dict(orient="records"):
        severe = assess_component(apply_scenario(record, SCENARIO_SEVERE), bundle)
        assert severe.recommended != REUSE


def test_record_export_has_sections_round_trips_and_is_labelled(
    bundle: ModelBundle, healthy_record: dict[str, Any]
) -> None:
    result = assess_component(healthy_record, bundle)
    structured = assessment_to_record(result)
    assert set(SECTIONS) | {"decision_support", "metadata"} == set(structured)
    parsed = json.loads(record_to_json(structured))
    assert parsed["metadata"]["data_origin"] == "synthetic"
    assert "not" in parsed["metadata"]["disclaimer"].lower()
    assert parsed["decision_support"]["recommended_strategy"] == result.recommended
    flat = flatten_structured_record(parsed)
    for key in ("corrosion_score", "residual_thickness_pct", "component_type"):
        assert flat[key] == healthy_record[key]


def test_explanation_reflects_actual_values(bundle: ModelBundle, severe_record: dict[str, Any]) -> None:
    result = assess_component(severe_record, bundle)
    text = result.explanation.paragraph
    assert result.recommended.lower()[:6] in text.lower()
    assert "direct reuse" in text.lower() and "blocked" in text.lower()
    assert "9.0" in text  # the observed corrosion value appears in the text
    other = assess_component({**severe_record, "corrosion_score": 7.5}, bundle).explanation.paragraph
    assert other != text
    assert assess_component(severe_record, bundle).explanation == result.explanation


def test_showcase_cases_run_through_pipeline(bundle: ModelBundle) -> None:
    results = {r["component_id"]: assess_component(r, bundle) for r in showcase_records()}
    assert results["SHOW-E"].has_recommendation is False
    assert all(results[k].has_recommendation for k in ("SHOW-A", "SHOW-B", "SHOW-C", "SHOW-D"))
    assert results["SHOW-C"].recommended != REUSE


def test_application_code_does_not_claim_institutional_affiliation() -> None:
    """No institution, partner or project name appears in the code or documentation."""
    forbidden = ("tu wien", "circloops", "siemens", "welser", "passport", "official implementation of circ")
    sources = [ROOT / "app.py", *(ROOT / "circularloop").rglob("*.py"), *(ROOT / "ui").rglob("*.py")]
    docs = [*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md")]
    for path in sources + docs:
        text = path.read_text(encoding="utf-8").lower()
        for word in forbidden:
            assert word not in text, f"'{word}' found in {path.name}"
    images = [
        p
        for p in ROOT.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".gif", ".ico"} and ".venv" not in p.parts
    ]
    assert images == [], "no logos or image assets are shipped"
