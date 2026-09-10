from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.audit import AuditLogger
from core.config import BRONZE_DIR, LLM_PROVIDER
from core.logger import get_logger
from core.observability import AgentTrace
from tools.helpers import safe_stem

BRONZE_AGENT_PROMPT = """You MUST call both tools in order: inspect_task_tool then bronze_ingestion_tool.
Do not just describe a plan - actually execute it by calling bronze_ingestion_tool.
Planning without executing is FAILURE.
"""


def _inspect_task(input_files: list[str], sttm_path: str) -> dict:
    previews = {}
    for file_path in input_files:
        frame = pd.read_csv(file_path, nrows=20)
        previews[str(file_path)] = {"shape": list(frame.shape), "columns": list(frame.columns)}
    sttm_rows = []
    if sttm_path and Path(sttm_path).exists():
        sttm_rows = pd.read_csv(sttm_path).to_dict(orient="records")
    return {"previews": previews, "sttm_rows": sttm_rows}


def _apply_bronze_rules(input_files: list[str], sttm_path: str, run_id: str) -> list[str]:
    sttm = pd.read_csv(sttm_path) if sttm_path and Path(sttm_path).exists() else pd.DataFrame()
    output_paths: list[str] = []

    for file_path in input_files:
        frame = pd.read_csv(file_path)

        if not sttm.empty:
            file_rules = sttm[sttm["source_table"].astype(str) == Path(file_path).stem]
            rename_map = {
                row["source_column"]: row["target_column"]
                for _, row in file_rules.iterrows()
                if pd.notna(row.get("source_column")) and pd.notna(row.get("target_column"))
            }
            if rename_map:
                frame = frame.rename(columns=rename_map)

        frame["_load_timestamp"] = datetime.now(timezone.utc).isoformat()
        frame["_source_file"] = str(file_path)

        BRONZE_DIR.mkdir(parents=True, exist_ok=True)
        out_file = BRONZE_DIR / f"{Path(file_path).stem}_bronze.parquet"
        frame.to_parquet(out_file, index=False)
        output_paths.append(str(out_file))

    return output_paths


def _make_bronze_tools(input_files: list[str], sttm_path: str, run_id: str) -> list:
    from langchain_core.tools import tool

    @tool
    def inspect_task_tool() -> str:
        """Preview input CSV shapes/columns and the Bronze STTM rules."""
        return json.dumps(_inspect_task(input_files, sttm_path))

    @tool
    def bronze_ingestion_tool() -> str:
        """Apply Bronze STTM rules to the input files and write Bronze Parquet outputs."""
        paths = _apply_bronze_rules(input_files, sttm_path, run_id)
        return json.dumps({"bronze_output_paths": paths})

    return [inspect_task_tool, bronze_ingestion_tool]


def _make_llm():
    from core.llm_factory import make_llm

    return make_llm()


def execute_bronze(input_files: list[str], sttm_path: str, run_id: str, task_description: str) -> list[str]:
    from langgraph.prebuilt import create_react_agent

    trace = AgentTrace("bronze_agent", run_id).set_input(input_files=input_files, task=task_description)
    audit = AuditLogger(run_id)
    audit.log("bronze_agent", "start", task=task_description)

    human_task = (
        f"Business context (background only, do not answer it here): {task_description}\n\n"
        "Your only job right now is to ingest the given files into the Bronze layer. "
        "Call inspect_task_tool, then call bronze_ingestion_tool. "
        "Only use the tools provided to you; do not invent or call any other tool."
    )

    try:
        llm = _make_llm()
        tools = _make_bronze_tools(input_files, sttm_path, run_id)
        agent = create_react_agent(llm, tools)
        result = agent.invoke(
            {"messages": [("system", BRONZE_AGENT_PROMPT), ("human", human_task)]}
        )
        trace.extract_from_messages(result.get("messages", []))

        output_paths = _apply_bronze_rules(input_files, sttm_path, run_id)
        trace.set_output(bronze_output_paths=output_paths).complete()
        audit.log("bronze_agent", "complete", output_paths=output_paths)
        return output_paths
    except Exception as exc:
        trace.fail(str(exc))
        audit.log("bronze_agent", "failed", error=str(exc))
        raise


@dataclass
class BronzeResult:
    source_file: Path
    bronze_file: Path
    profile: dict[str, object]
    dataframe: pd.DataFrame


class BronzeAgent:
    def __init__(self) -> None:
        self.logger = get_logger("bronze_agent")

    def ingest(self, source_file: Path, bronze_dir: Path) -> BronzeResult:
        self.logger.info("Ingesting raw file: %s", source_file)
        frame = pd.read_csv(source_file)
        profile = self._profile(frame)

        bronze_dir.mkdir(parents=True, exist_ok=True)
        bronze_file = bronze_dir / f"{safe_stem(source_file)}_bronze.parquet"
        frame.to_parquet(bronze_file, index=False)

        return BronzeResult(
            source_file=source_file,
            bronze_file=bronze_file,
            profile=profile,
            dataframe=frame,
        )

    def _profile(self, frame: pd.DataFrame) -> dict[str, object]:
        return {
            "rows": int(len(frame)),
            "columns": [str(col) for col in frame.columns],
            "missing_pct": {
                str(col): float(frame[col].isna().mean()) for col in frame.columns
            },
            "dtypes": {str(col): str(dtype) for col, dtype in frame.dtypes.items()},
        }
