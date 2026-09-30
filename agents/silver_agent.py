from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.audit import AuditLogger
from core.config import ApprovalConfig, BusinessRules, LLM_PROVIDER, SILVER_DIR
from core.logger import get_logger
from core.memory import AgentMemory, MappingMemoryRecord
from core.observability import AgentTrace
from tools.helpers import canonicalize, parse_date, parse_numeric, source_signature

SILVER_AGENT_PROMPT = """You MUST call inspect_bronze_tool then silver_cleansing_tool.
Apply the Silver STTM rules exactly: surrogate key, null handling, date standardization, dedup, text normalization.
Planning without executing is FAILURE.
"""


def _inspect_bronze(bronze_paths: list[str], sttm_path: str) -> dict:
    previews = {}
    for path in bronze_paths:
        frame = pd.read_parquet(path)
        previews[str(path)] = {"shape": list(frame.shape), "columns": list(frame.columns)}
    sttm_rows = []
    if sttm_path and Path(sttm_path).exists():
        sttm_rows = pd.read_csv(sttm_path).to_dict(orient="records")
    return {"previews": previews, "sttm_rows": sttm_rows}


def _apply_silver_rules(bronze_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
	def _norm(value: object) -> str:
		return str(value).strip().upper() if pd.notna(value) else ""

	def _target_column(frame: pd.DataFrame, source_col: str, target_col: str) -> str:
		if target_col in frame.columns:
			return target_col
		if source_col in frame.columns:
			return source_col
		return ""

	def _fill_with_logic(series: pd.Series, logic: str) -> pd.Series:
		text = (logic or "").strip()
		upper = text.upper()

		if upper.startswith("VALUE:"):
			raw = text.split(":", 1)[1].strip()
			if raw.upper() in {"NULL", "NONE"}:
				value = None
			elif raw.replace(".", "", 1).isdigit() or (
				raw.startswith("-") and raw[1:].replace(".", "", 1).isdigit()
			):
				value = float(raw) if "." in raw else int(raw)
			else:
				value = raw
			return series.fillna(value)

		if upper in {"MEDIAN", "FILL_MEDIAN"} and pd.api.types.is_numeric_dtype(series):
			return series.fillna(series.median())
		if upper in {"MEAN", "FILL_MEAN"} and pd.api.types.is_numeric_dtype(series):
			return series.fillna(series.mean())
		if upper in {"MODE", "FILL_MODE"}:
			mode = series.mode(dropna=True)
			return series.fillna(mode.iloc[0] if not mode.empty else "Unknown")
		if upper in {"ZERO", "0", "FILL_ZERO"}:
			return series.fillna(0)
		if upper in {"UNKNOWN", "FILL_UNKNOWN"}:
			return series.fillna("Unknown")
		if upper in {"FFILL", "FORWARD_FILL"}:
			return series.ffill()
		if upper in {"BFILL", "BACKWARD_FILL"}:
			return series.bfill()

		if pd.api.types.is_numeric_dtype(series):
			return series.fillna(series.median())
		return series.fillna("Unknown")

	sttm = pd.read_csv(sttm_path) if sttm_path and Path(sttm_path).exists() else pd.DataFrame()
	output_paths: list[str] = []

	for path in bronze_paths:
		frame = pd.read_parquet(path)
		table_name = Path(path).stem

		file_rules = pd.DataFrame()
		if not sttm.empty:
			file_rules = sttm[sttm["source_table"].astype(str) == table_name]
			if file_rules.empty and "target_table" in sttm.columns:
				file_rules = sttm[sttm["target_table"].astype(str) == table_name]

		if not file_rules.empty:
			rename_map = {
				row["source_column"]: row["target_column"]
				for _, row in file_rules.iterrows()
				if pd.notna(row.get("source_column")) and pd.notna(row.get("target_column"))
			}
			if rename_map:
				frame = frame.rename(columns=rename_map)

		dedup_subsets: list[str] = []
		fill_null_columns: set[str] = set()
		date_columns: set[str] = set()

		if not file_rules.empty:
			for _, row in file_rules.iterrows():
				transformation_type = _norm(row.get("transformation_type"))
				transformation_logic = str(row.get("transformation_logic", "") or "").strip()
				source_col = str(row.get("source_column", "") or "").strip()
				target_col = str(row.get("target_column", "") or "").strip()
				selected_col = _target_column(frame, source_col, target_col)

				if transformation_type == "DATE_FORMAT" and selected_col:
					frame[selected_col] = frame[selected_col].map(parse_date)
					date_columns.add(selected_col)
				elif transformation_type in {"NUMERIC_CAST", "TO_NUMERIC", "CAST_NUMERIC"} and selected_col:
					frame[selected_col] = frame[selected_col].map(parse_numeric)
				elif transformation_type == "TEXT_NORMALIZE" and selected_col:
					frame[selected_col] = frame[selected_col].astype("string").str.strip().str.lower()
				elif transformation_type == "FILL_NULL" and selected_col:
					frame[selected_col] = _fill_with_logic(frame[selected_col], transformation_logic)
					fill_null_columns.add(selected_col)
				elif transformation_type == "DEDUP":
					if selected_col:
						dedup_subsets.append(selected_col)
					elif transformation_logic:
						for col_name in [c.strip() for c in transformation_logic.split(",") if c.strip()]:
							if col_name in frame.columns:
								dedup_subsets.append(col_name)

		if dedup_subsets:
			frame = frame.drop_duplicates(subset=list(dict.fromkeys(dedup_subsets)))
		else:
			frame = frame.drop_duplicates()

		for col in frame.columns:
			if col in fill_null_columns:
				continue
			if pd.api.types.is_datetime64_any_dtype(frame[col]) or col in date_columns:
				continue
			if pd.api.types.is_numeric_dtype(frame[col]):
				frame[col] = frame[col].fillna(frame[col].median())
			else:
				frame[col] = frame[col].fillna("Unknown")

		pk_col = f"pk_{table_name}_silver_id"
		frame.insert(0, pk_col, [str(uuid.uuid4()) for _ in range(len(frame))])

		SILVER_DIR.mkdir(parents=True, exist_ok=True)
		out_file = SILVER_DIR / f"{table_name}_silver.parquet"
		frame.to_parquet(out_file, index=False)
		output_paths.append(str(out_file))

	return output_paths


def _make_silver_tools(bronze_paths: list[str], sttm_path: str, run_id: str) -> list:
    from langchain_core.tools import tool

    @tool
    def inspect_bronze_tool() -> str:
        """Preview Bronze parquet files and the Silver STTM rules."""
        return json.dumps(_inspect_bronze(bronze_paths, sttm_path))

    @tool
    def silver_cleansing_tool() -> str:
        """Apply Silver STTM rules to cleanse Bronze data and write Silver Parquet outputs."""
        paths = _apply_silver_rules(bronze_paths, sttm_path, run_id)
        return json.dumps({"silver_output_paths": paths})

    return [inspect_bronze_tool, silver_cleansing_tool]


def _make_llm():
    from core.llm_factory import make_llm

    return make_llm()


def execute_silver(bronze_output_paths: list[str], sttm_path: str, run_id: str, task_description: str) -> list[str]:
    from langgraph.prebuilt import create_react_agent

    trace = AgentTrace("silver_agent", run_id).set_input(bronze_paths=bronze_output_paths, task=task_description)
    audit = AuditLogger(run_id)
    audit.log("silver_agent", "start", task=task_description)

    human_task = (
        f"Business context (background only, do not answer it here): {task_description}\n\n"
        "Your only job right now is to cleanse the given Bronze data into the Silver layer. "
        "Call inspect_bronze_tool, then call silver_cleansing_tool. "
        "Only use the tools provided to you; do not invent or call any other tool."
    )

    try:
        llm = _make_llm()
        tools = _make_silver_tools(bronze_output_paths, sttm_path, run_id)
        agent = create_react_agent(llm, tools)
        result = agent.invoke(
            {"messages": [("system", SILVER_AGENT_PROMPT), ("human", human_task)]}
        )
        trace.extract_from_messages(result.get("messages", []))

        output_paths = _apply_silver_rules(bronze_output_paths, sttm_path, run_id)
        trace.set_output(silver_output_paths=output_paths).complete()
        audit.log("silver_agent", "complete", output_paths=output_paths)
        return output_paths
    except Exception as exc:
        trace.fail(str(exc))
        audit.log("silver_agent", "failed", error=str(exc))
        raise

TARGET_COLUMNS = [
	"order_date",
	"store_id",
	"product_id",
	"quantity",
	"unit_price",
	"discount",
	"currency",
]

ALIASES = {
	"order_date": ["order_date", "date", "sale_date", "txn_date"],
	"store_id": ["store_id", "store", "shop_id", "branch"],
	"product_id": ["product_id", "product", "sku", "item_id"],
	"quantity": ["quantity", "qty", "units", "count"],
	"unit_price": ["unit_price", "price", "amount", "unit_amount"],
	"discount": ["discount", "disc", "markdown", "rebate"],
	"currency": ["currency", "ccy", "curr"],
}


@dataclass
class MappingProposal:
	source_signature: str
	mapping: dict[str, str]
	missing_targets: list[str]
	confidence: float
	requires_approval: bool


@dataclass
class SilverResult:
	silver_file: Path
	dataframe: pd.DataFrame
	proposal: MappingProposal


class SilverAgent:
	def __init__(self, memory: AgentMemory) -> None:
		self.memory = memory
		self.logger = get_logger("silver_agent")

	def propose_mapping(
		self,
		source_columns: list[str],
		approval: ApprovalConfig,
	) -> MappingProposal:
		signature = source_signature(source_columns)
		stored = self.memory.find_mapping(signature)
		if stored:
			missing = [col for col in TARGET_COLUMNS if col not in stored.mapping]
			requires_approval = not approval.auto_approve and stored.confidence < approval.confidence_threshold
			return MappingProposal(
				source_signature=signature,
				mapping=stored.mapping,
				missing_targets=missing,
				confidence=stored.confidence,
				requires_approval=requires_approval,
			)

		canonical_lookup = {canonicalize(col): col for col in source_columns}
		mapping: dict[str, str] = {}
		hits = 0

		for target in TARGET_COLUMNS:
			for alias in ALIASES[target]:
				if alias in canonical_lookup:
					mapping[target] = canonical_lookup[alias]
					hits += 1
					break

		missing = [target for target in TARGET_COLUMNS if target not in mapping]
		confidence = hits / len(TARGET_COLUMNS)
		requires_approval = (confidence < approval.confidence_threshold) or (not approval.auto_approve)

		return MappingProposal(
			source_signature=signature,
			mapping=mapping,
			missing_targets=missing,
			confidence=confidence,
			requires_approval=requires_approval,
		)

	def execute(
		self,
		frame: pd.DataFrame,
		proposal: MappingProposal,
		silver_dir: Path,
		source_name: str,
		rules: BusinessRules,
		approved: bool,
	) -> SilverResult:
		if proposal.requires_approval and not approved:
			raise PermissionError("Silver mapping requires explicit approval before execution")

		out = pd.DataFrame()
		for target, source_col in proposal.mapping.items():
			out[target] = frame[source_col]

		for missing_target in proposal.missing_targets:
			out[missing_target] = None

		out["order_date"] = out["order_date"].map(parse_date)
		out["quantity"] = out["quantity"].map(parse_numeric)
		out["unit_price"] = out["unit_price"].map(parse_numeric)
		out["discount"] = out["discount"].map(parse_numeric).fillna(0.0)
		out["currency"] = out["currency"].fillna("USD")

		out = out.dropna(subset=["order_date"]) if "order_date" in out.columns else out

		if rules.drop_rows_missing_keys:
			key_columns = [key for key in rules.required_keys if key in out.columns]
			out = out.dropna(subset=key_columns)

		out = out[out["quantity"].fillna(0) >= rules.min_quantity]
		out = out[out["discount"].fillna(0) <= rules.discount_max]
		out["gross_sales"] = out["quantity"].fillna(0) * out["unit_price"].fillna(0)
		out["net_sales"] = out["gross_sales"] * (1 - out["discount"].fillna(0))

		silver_dir.mkdir(parents=True, exist_ok=True)
		silver_file = silver_dir / f"{source_name}_silver.parquet"
		out.to_parquet(silver_file, index=False)

		self.memory.remember(
			MappingMemoryRecord(
				source_signature=proposal.source_signature,
				mapping=proposal.mapping,
				confidence=proposal.confidence,
			)
		)

		self.logger.info(
			"Silver transformation complete. rows_in=%s rows_out=%s",
			len(frame),
			len(out),
		)
		return SilverResult(silver_file=silver_file, dataframe=out, proposal=proposal)
