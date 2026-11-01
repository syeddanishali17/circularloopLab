"""Regenerate the bundled data files:  python -m circularloop"""

from __future__ import annotations

from circularloop.data_io import COMPONENTS_CSV, SAMPLE_RECORD_JSON, load_components
from circularloop.pipeline import assess_component, assessment_to_record
from circularloop.record_schema import record_to_json
from circularloop.rul_model import train_and_evaluate
from circularloop.showcase import showcase_records

SAMPLE_COMPONENT = "SHOW-B"


def main() -> None:
    df = load_components(regenerate=True)
    bundle = train_and_evaluate(df)
    record = next(r for r in showcase_records() if r["component_id"] == SAMPLE_COMPONENT)
    SAMPLE_RECORD_JSON.write_text(
        record_to_json(assessment_to_record(assess_component(record, bundle))) + "\n", encoding="utf-8"
    )
    print(f"Wrote {COMPONENTS_CSV.name} ({len(df)} records) and {SAMPLE_RECORD_JSON.name}.")
    print(bundle.metrics.round(3).to_string())


if __name__ == "__main__":
    main()
