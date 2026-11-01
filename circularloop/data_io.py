"""Loading and persisting the synthetic dataset (paths are relative to the repository)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from circularloop.synthetic_data import build_dataset

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
COMPONENTS_CSV: Path = DATA_DIR / "sample_components.csv"
SAMPLE_RECORD_JSON: Path = DATA_DIR / "sample_lifecycle_record.json"


def load_components(path: Path = COMPONENTS_CSV, regenerate: bool = False) -> pd.DataFrame:
    """Load the component dataset, creating it deterministically if it is missing."""
    if regenerate or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        build_dataset().to_csv(path, index=False)
    df = pd.read_csv(path)
    df["spare_parts_available"] = df["spare_parts_available"].map(
        lambda v: v if pd.isna(v) else str(v).strip().lower() in {"true", "1", "yes"}
    )
    return df
