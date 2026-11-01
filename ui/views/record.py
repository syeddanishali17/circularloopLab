"""Tab 6: lifecycle record explorer and export."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from circularloop.pipeline import assessment_to_record
from circularloop.record_schema import field_catalogue_rows, record_to_json
from ui.state import AppContext


def render(ctx: AppContext) -> None:
    st.caption(
        "A simplified, conceptual lifecycle-record structure for this demonstrator. It is **not** an "
        "implementation of, "
        "or compliant with, any formal product-data standard."
    )
    structured = assessment_to_record(ctx.assessment)
    left, right = st.columns([1, 1.2], gap="medium")
    with left, st.container(border=True):
        st.markdown(f"**Lifecycle record · {ctx.component_id}**")
        st.json(structured, expanded=2)
        st.download_button(
            "Download record JSON",
            record_to_json(structured).encode("utf-8"),
            file_name=f"{ctx.component_id}_lifecycle_record.json",
            mime="application/json",
            icon=":material/download:",
        )
    with right, st.container(border=True):
        st.markdown("**Field catalogue**")
        st.dataframe(
            pd.DataFrame(field_catalogue_rows()),
            hide_index=True,
            height=560,
            column_order=["Section", "Label", "Unit", "Critical", "Description", "Field"],
            column_config={
                "Critical": st.column_config.TextColumn(width="small"),
                "Description": st.column_config.TextColumn(width="large"),
            },
            alt="Description of each record field",
        )
        st.caption("Critical fields must be present and valid; otherwise no recommendation is produced.")
