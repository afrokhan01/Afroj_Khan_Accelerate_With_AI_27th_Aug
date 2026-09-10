from __future__ import annotations

from pathlib import Path

import streamlit as st

from agents.orchestrator import RetailMedallionOrchestrator
from core.config import load_config


st.set_page_config(page_title="Retail Medallion Pipeline", layout="wide")
st.title("Retail Co Agentic Medallion Pipeline")

env = st.selectbox("Environment", ["dev", "staging", "prod"], index=0)
config = load_config(environment=env)
orchestrator = RetailMedallionOrchestrator(config=config)

uploaded = st.file_uploader("Upload raw sales CSV", type=["csv"])
approved = st.toggle("Approve inferred mapping (required for low confidence)", value=False)

if uploaded is not None:
	raw_path = config.paths.raw_input / uploaded.name
	raw_path.parent.mkdir(parents=True, exist_ok=True)
	raw_path.write_bytes(uploaded.getvalue())

	if st.button("Run Pipeline", type="primary"):
		try:
			result = orchestrator.run_file(raw_path, approval_override=approved)
			st.success("Pipeline execution complete")
			st.write(
				{
					"bronze": str(result.bronze_file),
					"silver": str(result.silver_file),
					"gold": str(result.gold_file),
					"report": str(result.report_file),
					"approval_required": result.approval_required,
					"approved": result.approved,
				}
			)
		except PermissionError as exc:
			st.error(str(exc))

st.subheader("Queued Raw Files")
queued_files = sorted(config.paths.raw_input.glob("*.csv"))
if queued_files:
	st.dataframe({"file": [str(file) for file in queued_files]})
else:
	st.info("No raw files are queued")
