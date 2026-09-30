from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.audit import AuditLogger
from core.config import GOLD_DIR, LLM_PROVIDER
from core.logger import get_logger
from core.observability import AgentTrace

GOLD_AGENT_PROMPT = """You MUST call inspect_silver_tool then gold_aggregation_tool.
Apply the Gold STTM rules exactly: surrogate key first, then joins/aggregations driven by the business intent.
Planning without executing is FAILURE.
"""


def _inspect_silver(silver_paths: list[str], sttm_path: str) -> dict:
    previews = {}
    for path in silver_paths:
        frame = pd.read_parquet(path)
        previews[str(path)] = {"shape": list(frame.shape), "columns": list(frame.columns)}
    sttm_rows = []
    if sttm_path and Path(sttm_path).exists():
        sttm_rows = pd.read_csv(sttm_path).to_dict(orient="records")
    return {"previews": previews, "sttm_rows": sttm_rows}


def _apply_gold_rules(silver_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    def _norm(value: object) -> str:
        return str(value).strip().upper() if pd.notna(value) else ""

    def _extract_logic_col(logic: str) -> str:
        match = re.search(r"\(([^)]+)\)", logic or "")
        return match.group(1).strip() if match else ""

    def _build_sttm_plan(frame: pd.DataFrame, file_rules: pd.DataFrame) -> tuple[list[str], list[tuple[str, str, str]]]:
        group_cols: list[str] = []
        agg_specs: list[tuple[str, str, str]] = []

        agg_map = {"SUM": "sum", "COUNT": "count", "AVG": "mean", "MEAN": "mean"}
        if file_rules.empty:
            return group_cols, agg_specs

        for _, row in file_rules.iterrows():
            transformation_type = _norm(row.get("transformation_type"))
            transformation_logic = str(row.get("transformation_logic", "") or "").strip()
            source_col = str(row.get("source_column", "") or "").strip()
            target_col = str(row.get("target_column", "") or "").strip()

            if not transformation_type and transformation_logic:
                upper_logic = transformation_logic.upper()
                for candidate in ("GROUP_BY", "SUM", "COUNT", "AVG", "MEAN"):
                    if upper_logic.startswith(candidate):
                        transformation_type = candidate
                        break

            if transformation_type == "GROUP_BY":
                selected_col = ""
                for col in (target_col, source_col, _extract_logic_col(transformation_logic)):
                    if col in frame.columns:
                        selected_col = col
                        break
                if selected_col and selected_col not in group_cols:
                    group_cols.append(selected_col)
                continue

            if transformation_type in agg_map:
                agg_fn = agg_map[transformation_type]
                selected_col = ""
                for col in (source_col, target_col, _extract_logic_col(transformation_logic)):
                    if col in frame.columns:
                        selected_col = col
                        break
                if not selected_col:
                    continue
                output_col = target_col or f"{agg_fn}_{selected_col}"
                agg_specs.append((output_col, selected_col, agg_fn))

        return group_cols, agg_specs

    output_paths: list[str] = []
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    sttm = pd.read_csv(sttm_path) if sttm_path and Path(sttm_path).exists() else pd.DataFrame()

    for path in silver_paths:
        frame = pd.read_parquet(path)
        table_name = Path(path).stem

        file_rules = pd.DataFrame()
        if not sttm.empty and "source_table" in sttm.columns:
            file_rules = sttm[sttm["source_table"].astype(str) == table_name]
            if file_rules.empty and "target_table" in sttm.columns:
                file_rules = sttm[sttm["target_table"].astype(str) == table_name]

        group_cols, agg_specs = _build_sttm_plan(frame, file_rules)

        if group_cols and agg_specs:
            named_aggs = {out_col: (src_col, agg_fn) for out_col, src_col, agg_fn in agg_specs}
            agg = frame.groupby(group_cols, dropna=False, as_index=False).agg(**named_aggs)
        elif group_cols and not agg_specs:
            agg = frame[group_cols].drop_duplicates().reset_index(drop=True)
        elif agg_specs and not group_cols:
            row: dict[str, object] = {}
            for out_col, src_col, agg_fn in agg_specs:
                value = frame[src_col].count() if agg_fn == "count" else frame[src_col].agg(agg_fn)
                row[out_col] = value
            agg = pd.DataFrame([row])
        else:
            default_group_cols = [
                col
                for col in ("category", "store_id", "region", "product_id")
                if col in frame.columns
            ]
            default_numeric_cols = [col for col in ("quantity", "total_amount") if col in frame.columns]
            if default_group_cols and default_numeric_cols:
                agg = frame.groupby(default_group_cols, as_index=False)[default_numeric_cols].sum()
            else:
                agg = frame.copy()

        pk_col = "pk_gold_id"
        agg.insert(0, pk_col, [str(uuid.uuid4()) for _ in range(len(agg))])

        out_file = GOLD_DIR / f"{table_name}_gold.parquet"
        agg.to_parquet(out_file, index=False)
        output_paths.append(str(out_file))

    return output_paths


def _make_gold_tools(silver_paths: list[str], sttm_path: str, run_id: str) -> list:
    from langchain_core.tools import tool

    @tool
    def inspect_silver_tool() -> str:
        """Preview Silver parquet files and the Gold STTM rules."""
        return json.dumps(_inspect_silver(silver_paths, sttm_path))

    @tool
    def gold_aggregation_tool() -> str:
        """Apply Gold STTM rules to join/aggregate Silver data and write Gold Parquet outputs."""
        paths = _apply_gold_rules(silver_paths, sttm_path, run_id)
        return json.dumps({"gold_output_paths": paths})

    return [inspect_silver_tool, gold_aggregation_tool]


def _make_llm():
    from core.llm_factory import make_llm

    return make_llm()


def execute_gold(silver_output_paths: list[str], sttm_path: str, run_id: str, task_description: str) -> list[str]:
    from langgraph.prebuilt import create_react_agent

    trace = AgentTrace("gold_agent", run_id).set_input(silver_paths=silver_output_paths, task=task_description)
    audit = AuditLogger(run_id)
    audit.log("gold_agent", "start", task=task_description)

    human_task = (
        f"Business context (background only, do not answer it here): {task_description}\n\n"
        "Your only job right now is to aggregate the given Silver data into the Gold layer. "
        "Call inspect_silver_tool, then call gold_aggregation_tool. "
        "Only use the tools provided to you; do not invent or call any other tool."
    )

    try:
        llm = _make_llm()
        tools = _make_gold_tools(silver_output_paths, sttm_path, run_id)
        agent = create_react_agent(llm, tools)
        result = agent.invoke(
            {"messages": [("system", GOLD_AGENT_PROMPT), ("human", human_task)]}
        )
        trace.extract_from_messages(result.get("messages", []))

        output_paths = _apply_gold_rules(silver_output_paths, sttm_path, run_id)
        trace.set_output(gold_output_paths=output_paths).complete()
        audit.log("gold_agent", "complete", output_paths=output_paths)
        return output_paths
    except Exception as exc:
        trace.fail(str(exc))
        audit.log("gold_agent", "failed", error=str(exc))
        raise


@dataclass
class GoldResult:
	gold_file: Path
	dataframe: pd.DataFrame


class GoldAgent:
	def __init__(self) -> None:
		self.logger = get_logger("gold_agent")

	def aggregate(self, silver_frame: pd.DataFrame, gold_dir: Path, source_name: str) -> GoldResult:
		frame = silver_frame.copy()
		frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce")
		frame["order_month"] = frame["order_date"].dt.to_period("M").astype(str)

		gold = (
			frame.groupby(["order_month", "store_id"], dropna=False)
			.agg(
				transactions=("product_id", "count"),
				units_sold=("quantity", "sum"),
				gross_sales=("gross_sales", "sum"),
				net_sales=("net_sales", "sum"),
			)
			.reset_index()
		)

		gold_dir.mkdir(parents=True, exist_ok=True)
		gold_file = gold_dir / f"{source_name}_gold.parquet"
		gold.to_parquet(gold_file, index=False)

		self.logger.info("Gold aggregation complete. rows=%s", len(gold))
		return GoldResult(gold_file=gold_file, dataframe=gold)
