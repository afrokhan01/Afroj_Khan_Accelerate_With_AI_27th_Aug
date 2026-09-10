from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.config import LLM_PROVIDER, STTM_DIR
from core.observability import AgentTrace

STTM_AGENT_PROMPT = """You generate STTM (Source-to-Target Mapping) rules for the Bronze, Silver, or Gold layer.
First call inspect_context_tool to understand the current context.
Then call the appropriate generation tool exactly ONCE.
Bronze: map every source column, and add metadata columns _load_timestamp and _source_file.
Silver: add a surrogate key first, then cleansing rules (nulls, dedup, types, dates, text normalization).
Gold: add a surrogate key first, then join/aggregation rules driven by the business intent.
"""

STTM_COLUMNS = [
    "source_schema",
    "source_table",
    "source_column",
    "target_schema",
    "target_table",
    "target_column",
    "transformation_type",
    "transformation_logic",
]


def _prepare_bronze_context(profile_path: str) -> dict:
    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_silver_context(bronze_paths: list[str], bronze_sttm_path: str) -> dict:
    approved_columns: set[str] = set()
    if bronze_sttm_path and Path(bronze_sttm_path).exists():
        sttm = pd.read_csv(bronze_sttm_path)
        approved_columns = set(sttm["target_column"].astype(str))

    context: dict[str, object] = {}
    for bronze_path in bronze_paths:
        frame = pd.read_parquet(bronze_path)
        columns = [col for col in frame.columns if not approved_columns or col in approved_columns]
        context[Path(bronze_path).name] = {
            "columns": columns,
            "dtypes": {str(col): str(dtype) for col, dtype in frame.dtypes.items()},
        }
    return context


def _prepare_gold_context(silver_paths: list[str], silver_sttm_path: str) -> dict:
    approved_columns: set[str] = set()
    if silver_sttm_path and Path(silver_sttm_path).exists():
        sttm = pd.read_csv(silver_sttm_path)
        approved_columns = set(sttm["target_column"].astype(str))

    context: dict[str, object] = {}
    for silver_path in silver_paths:
        frame = pd.read_parquet(silver_path)
        columns = [col for col in frame.columns if not approved_columns or col in approved_columns]
        context[Path(silver_path).name] = {
            "columns": columns,
            "dtypes": {str(col): str(dtype) for col, dtype in frame.dtypes.items()},
        }
    return context


def _rows_to_csv(rows: list[dict], out_file: Path) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=STTM_COLUMNS)
    frame.to_csv(out_file, index=False)
    return out_file


def _valid_sttm_rows(rows: object) -> bool:
    """True if rows is a non-empty list of dicts using at least one real STTM_COLUMNS key."""
    if not isinstance(rows, list) or not rows:
        return False
    if not all(isinstance(row, dict) for row in rows):
        return False
    return any(any(row.get(col) not in (None, "") for col in STTM_COLUMNS) for row in rows)


def _extract_sttm_rows(messages: list) -> list[dict]:
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            continue
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            continue
        if _valid_sttm_rows(parsed):
            return parsed
    return []


def _parse_rows_json(rows_json: str) -> list[dict]:
    """Parse a (possibly fenced/truncated) JSON array of row objects from model output."""
    text = (rows_json or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"rows_json is not valid JSON: {text[:200]!r}") from None
        parsed = json.loads(text[start : end + 1])
    if not _valid_sttm_rows(parsed):
        raise ValueError(
            "rows_json must be a JSON array of objects using exactly these keys: "
            + ", ".join(STTM_COLUMNS)
        )
    return parsed


def _make_llm():
    from core.llm_factory import make_llm

    return make_llm()


def _make_sttm_tools(
    layer: str,
    context_fn,
    out_file: Path,
    run_id: str,
    business_intent: str = "",
) -> list:
    from langchain_core.tools import tool

    @tool
    def inspect_context_tool() -> str:
        """Return context (columns, dtypes, profile) needed to generate STTM rules for this layer."""
        return json.dumps(context_fn())

    @tool
    def generate_bronze_sttm_tool(rows_json: str) -> str:
        """Generate Bronze STTM rules and save to CSV.

        rows_json must be a JSON array of objects, each using exactly these keys:
        source_schema, source_table, source_column, target_schema, target_table,
        target_column, transformation_type, transformation_logic.
        """
        try:
            rows = _parse_rows_json(rows_json)
        except ValueError as exc:
            return f"Error: {exc}"
        _rows_to_csv(rows, out_file)
        return json.dumps({"sttm_path": str(out_file), "rows": len(rows)})

    @tool
    def generate_silver_sttm_tool(rows_json: str) -> str:
        """Generate Silver STTM rules and save to CSV.

        rows_json must be a JSON array of objects, each using exactly these keys:
        source_schema, source_table, source_column, target_schema, target_table,
        target_column, transformation_type, transformation_logic.
        """
        try:
            rows = _parse_rows_json(rows_json)
        except ValueError as exc:
            return f"Error: {exc}"
        _rows_to_csv(rows, out_file)
        return json.dumps({"sttm_path": str(out_file), "rows": len(rows)})

    @tool
    def generate_gold_sttm_tool(rows_json: str) -> str:
        """Generate Gold STTM rules (using the business intent) and save to CSV.

        rows_json must be a JSON array of objects, each using exactly these keys:
        source_schema, source_table, source_column, target_schema, target_table,
        target_column, transformation_type, transformation_logic.
        """
        try:
            rows = _parse_rows_json(rows_json)
        except ValueError as exc:
            return f"Error: {exc}"
        _rows_to_csv(rows, out_file)
        return json.dumps({"sttm_path": str(out_file), "rows": len(rows), "business_intent": business_intent})

    tool_map = {
        "bronze": generate_bronze_sttm_tool,
        "silver": generate_silver_sttm_tool,
        "gold": generate_gold_sttm_tool,
    }
    return [inspect_context_tool, tool_map[layer]]


def _run_sttm_agent(layer: str, tools: list, run_id: str, task_description: str, out_file: Path) -> str:
    from langgraph.prebuilt import create_react_agent

    trace = AgentTrace(f"sttm_generator_{layer}", run_id).set_input(task=task_description)
    human_task = (
        f"Business context (background only, do not attempt to answer it yourself): {task_description}\n\n"
        f"Your only job right now is to generate {layer} STTM rules. "
        "Call inspect_context_tool first, then call the appropriate generation tool exactly once. "
        "Each row object in rows_json MUST use exactly these keys: " + ", ".join(STTM_COLUMNS) + ". "
        "Only use the tools provided to you; do not invent or call any other tool, and do not search the internet."
    )
    try:
        llm = _make_llm()
        agent = create_react_agent(llm, tools)
        result = agent.invoke(
            {"messages": [("system", STTM_AGENT_PROMPT), ("human", human_task)]}
        )
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)

        if not out_file.exists():
            rows = _extract_sttm_rows(messages)
            _rows_to_csv(rows, out_file)

        trace.set_output(sttm_path=str(out_file)).complete()
        return str(out_file)
    except Exception as exc:
        trace.fail(str(exc))
        raise


def generate_bronze_sttm(profile_path: str, run_id: str, task_description: str) -> str:
    out_file = STTM_DIR / f"bronze_sttm_{run_id[:8]}.csv"
    tools = _make_sttm_tools("bronze", lambda: _prepare_bronze_context(profile_path), out_file, run_id)
    return _run_sttm_agent("bronze", tools, run_id, task_description, out_file)


def generate_silver_sttm(bronze_output_paths: list[str], bronze_sttm_path: str, run_id: str, task_description: str) -> str:
    out_file = STTM_DIR / f"silver_sttm_{run_id[:8]}.csv"
    tools = _make_sttm_tools(
        "silver",
        lambda: _prepare_silver_context(bronze_output_paths, bronze_sttm_path),
        out_file,
        run_id,
    )
    return _run_sttm_agent("silver", tools, run_id, task_description, out_file)


def generate_gold_sttm(
    silver_output_paths: list[str],
    silver_sttm_path: str,
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    out_file = STTM_DIR / f"gold_sttm_{run_id[:8]}.csv"
    tools = _make_sttm_tools(
        "gold",
        lambda: _prepare_gold_context(silver_output_paths, silver_sttm_path),
        out_file,
        run_id,
        business_intent=business_intent,
    )
    return _run_sttm_agent("gold", tools, run_id, task_description, out_file)
