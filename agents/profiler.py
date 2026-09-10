from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.config import LLM_PROVIDER, PROFILES_DIR
from core.observability import AgentTrace

PROFILER_AGENT_PROMPT = """You are a Data Analyst specializing in data profiling.
Follow this sequence: THINK -> INSPECT -> PLAN -> ACT -> VERIFY.
First call inspect_files_tool to preview the files.
Then call profiler_tool to get full statistics.
Return a JSON object with keys: semantic_meanings, join_keys, quality_notes.
"""


def _inspect_files(file_paths: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for file_path in file_paths:
        frame = pd.read_csv(file_path, nrows=50)
        result[str(file_path)] = {
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "columns": list(frame.columns),
            "dtypes": {str(col): str(dtype) for col, dtype in frame.dtypes.items()},
            "samples": {
                str(col): frame[col].dropna().astype(str).head(3).tolist()
                for col in frame.columns
            },
        }
    return result


def _compute_stats(file_paths: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for file_path in file_paths:
        frame = pd.read_csv(file_path)
        columns_stats: dict[str, object] = {}
        for col in frame.columns:
            series = frame[col]
            stat: dict[str, object] = {
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "null_pct": float(series.isna().mean()),
                "unique_count": int(series.nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(series):
                stat["min"] = float(series.min()) if series.notna().any() else None
                stat["max"] = float(series.max()) if series.notna().any() else None
                stat["mean"] = float(series.mean()) if series.notna().any() else None
            else:
                stat["sample_values"] = series.dropna().astype(str).head(5).tolist()
            columns_stats[str(col)] = stat
        result[str(file_path)] = {
            "rows": int(len(frame)),
            "columns": columns_stats,
        }
    return result


def _make_profiler_tools(file_paths: list[str], run_id: str) -> list:
    from langchain_core.tools import tool

    @tool
    def inspect_files_tool() -> str:
        """Preview the shape, columns, dtypes, and sample values of each input file."""
        return json.dumps(_inspect_files(file_paths))

    @tool
    def profiler_tool() -> str:
        """Compute full column statistics (nulls, uniques, min/max/mean, samples) for each input file."""
        return json.dumps(_compute_stats(file_paths))

    return [inspect_files_tool, profiler_tool]


def _make_llm():
    from core.llm_factory import make_llm

    return make_llm()


def profile_dataset(file_path: str, run_id: str, task_description: str) -> str:
    return profile_multiple_datasets([file_path], run_id, task_description)


def profile_multiple_datasets(file_paths: list[str], run_id: str, task_description: str) -> str:
    from langgraph.prebuilt import create_react_agent

    trace = AgentTrace("profiler", run_id).set_input(file_paths=file_paths, task=task_description)

    human_task = (
        f"Business context (background only, do not attempt to answer it yourself): {task_description}\n\n"
        "Your only job right now is to profile the given files. "
        "Call inspect_files_tool, then call profiler_tool. "
        "Only use the tools provided to you; do not invent or call any other tool, and do not search the internet."
    )

    try:
        llm = _make_llm()
        tools = _make_profiler_tools(file_paths, run_id)
        agent = create_react_agent(llm, tools)
        result = agent.invoke(
            {"messages": [("system", PROFILER_AGENT_PROMPT), ("human", human_task)]}
        )
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)

        combined = {
            "inspect": _inspect_files(file_paths),
            "stats": _compute_stats(file_paths),
        }
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        out_file = PROFILES_DIR / f"profile_combined_{timestamp}.json"
        out_file.write_text(json.dumps(combined, default=str, indent=2), encoding="utf-8")

        trace.set_output(profile_path=str(out_file)).complete()
        return str(out_file)
    except Exception as exc:
        trace.fail(str(exc))
        raise


class DataProfiler:
	def write_profile(self, frame: pd.DataFrame, out_file: Path) -> Path:
		out_file.parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"rows": int(len(frame)),
			"columns": list(frame.columns),
			"missing_pct": {
				str(col): float(frame[col].isna().mean()) for col in frame.columns
			},
			"describe": frame.describe(include="all")
			.fillna("")
			.to_dict(),
		}
		out_file.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
		return out_file
