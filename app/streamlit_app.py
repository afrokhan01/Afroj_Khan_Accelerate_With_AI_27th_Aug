from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from agents.orchestrator import (
    run_bronze_to_silver_sttm,
    run_gold_and_report,
    run_silver_to_gold_sttm,
    run_until_bronze_sttm,
)
from core.audit import AuditLogger
from core.config import LANDING_DIR

PHASES = [
    "1. Upload & Intent",
    "2. Bronze STTM",
    "3. Silver STTM",
    "4. Gold STTM",
    "5. Report",
]

PHASE_STATUS_MAP = {
    "profiling": 0,
    "awaiting_bronze_approval": 1,
    "awaiting_silver_approval": 2,
    "awaiting_gold_approval": 3,
    "complete": 4,
}


def render_progress_banner(current_phase_index: int) -> None:
    st.markdown(
        """
        <style>
        .idamp-stepper { display: flex; justify-content: space-between; margin-bottom: 1.5rem; }
        .idamp-step { flex: 1; text-align: center; padding: 0.5rem; border-bottom: 4px solid #d0d5dd;
                      color: #667085; font-size: 0.85rem; }
        .idamp-step.active { border-bottom-color: #2563eb; color: #2563eb; font-weight: 600; }
        .idamp-step.done { border-bottom-color: #16a34a; color: #16a34a; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    html_steps = []
    for i, phase in enumerate(PHASES):
        css_class = "idamp-step"
        if i < current_phase_index:
            css_class += " done"
        elif i == current_phase_index:
            css_class += " active"
        html_steps.append(f'<div class="{css_class}">{phase}</div>')
    st.markdown(f'<div class="idamp-stepper">{"".join(html_steps)}</div>', unsafe_allow_html=True)


def render_audit_sidebar(run_id: str | None) -> None:
    with st.sidebar:
        st.header("Audit Trail")
        if not run_id:
            st.caption("No run started yet.")
            return
        logger = AuditLogger(run_id)
        logs = logger.get_logs()
        if not logs:
            st.caption("No audit events yet.")
            return
        for entry in reversed(logs[-30:]):
            st.text(f"[{entry.get('timestamp', '')[:19]}] {entry.get('agent')} · {entry.get('action')}")


def render_sttm_editor(sttm_path: str, label: str) -> pd.DataFrame | None:
    if not sttm_path or not Path(sttm_path).exists():
        st.warning(f"No {label} STTM file found yet.")
        return None
    frame = pd.read_csv(sttm_path)
    st.subheader(f"{label} STTM Rules")
    edited = st.data_editor(frame, num_rows="dynamic", use_container_width=True, key=f"editor_{label}")
    return edited


def render_phase1() -> None:
    st.header("Phase 1 · Upload & Business Intent")
    uploaded = st.file_uploader(
        "Upload one or more CSV files", type=["csv"], accept_multiple_files=True
    )
    business_intent = st.text_area(
        "Business question", placeholder="e.g., Which category had the highest sales last month?"
    )
    use_sample = st.checkbox("Use bundled sample data (data/landing)", value=not uploaded)

    if st.button("Start Pipeline", type="primary"):
        file_paths: list[str] = []
        if use_sample:
            file_paths = [str(p) for p in LANDING_DIR.glob("*.csv")]
        elif uploaded:
            LANDING_DIR.mkdir(parents=True, exist_ok=True)
            for file in uploaded:
                dest = LANDING_DIR / file.name
                dest.write_bytes(file.getbuffer())
                file_paths.append(str(dest))

        if not file_paths:
            st.error("Please upload files or use the sample data.")
            return
        if not business_intent:
            st.error("Please describe your business question.")
            return

        with st.spinner("Profiling data and generating Bronze STTM rules..."):
            state = run_until_bronze_sttm(file_paths, business_intent)
        st.session_state["pipeline_state"] = state
        st.rerun()


def render_phase2() -> None:
    st.header("Phase 2 · Review Bronze STTM")
    state = st.session_state["pipeline_state"]
    render_sttm_editor(state.get("sttm_bronze_path", ""), "Bronze")

    col1, col2 = st.columns(2)
    if col1.button("Approve & Continue to Silver STTM", type="primary"):
        with st.spinner("Ingesting Bronze and generating Silver STTM rules..."):
            state = run_bronze_to_silver_sttm(state)
        st.session_state["pipeline_state"] = state
        st.rerun()
    if col2.button("Reject / Start Over"):
        st.session_state.pop("pipeline_state", None)
        st.rerun()


def render_phase3() -> None:
    st.header("Phase 3 · Review Silver STTM")
    state = st.session_state["pipeline_state"]
    render_sttm_editor(state.get("sttm_silver_path", ""), "Silver")

    col1, col2 = st.columns(2)
    if col1.button("Approve & Continue to Gold STTM", type="primary"):
        with st.spinner("Cleansing Silver data and generating Gold STTM rules..."):
            state = run_silver_to_gold_sttm(state)
        st.session_state["pipeline_state"] = state
        st.rerun()
    if col2.button("Reject / Start Over", key="reject3"):
        st.session_state.pop("pipeline_state", None)
        st.rerun()


def render_phase4() -> None:
    st.header("Phase 4 · Review Gold STTM")
    state = st.session_state["pipeline_state"]
    render_sttm_editor(state.get("sttm_gold_path", ""), "Gold")

    col1, col2 = st.columns(2)
    if col1.button("Approve & Generate Report", type="primary"):
        with st.spinner("Aggregating Gold data and generating the report..."):
            state = run_gold_and_report(state)
        st.session_state["pipeline_state"] = state
        st.rerun()
    if col2.button("Reject / Start Over", key="reject4"):
        st.session_state.pop("pipeline_state", None)
        st.rerun()


def render_phase5() -> None:
    st.header("Phase 5 · Report")
    state = st.session_state["pipeline_state"]
    report_path = state.get("report_path", "")
    if report_path and Path(report_path).exists():
        html = Path(report_path).read_text(encoding="utf-8")
        st.components.v1.html(html, height=800, scrolling=True)
        st.download_button(
            "Download Report", data=html, file_name=Path(report_path).name, mime="text/html"
        )
    else:
        st.error("Report not found.")

    if st.button("Start a New Run"):
        st.session_state.pop("pipeline_state", None)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="IDAMP · Retail Agentic Medallion Pipeline", layout="wide")
    st.title("Retail Agentic Medallion Pipeline")

    state = st.session_state.get("pipeline_state")
    phase_index = PHASE_STATUS_MAP.get(state.get("status"), 0) if state else 0
    render_progress_banner(phase_index)
    render_audit_sidebar(state.get("run_id") if state else None)

    if not state:
        render_phase1()
    elif state["status"] == "awaiting_bronze_approval":
        render_phase2()
    elif state["status"] == "awaiting_silver_approval":
        render_phase3()
    elif state["status"] == "awaiting_gold_approval":
        render_phase4()
    elif state["status"] == "complete":
        render_phase5()
    else:
        render_phase1()


if __name__ == "__main__":
    main()
