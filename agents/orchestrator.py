from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from agents.bronze_agent import BronzeAgent, execute_bronze
from agents.gold_agent import GoldAgent, execute_gold
from agents.profiler import DataProfiler, profile_multiple_datasets
from agents.reporter import ReportingAgent, generate_report
from agents.silver_agent import SilverAgent, execute_silver
from agents.sttm_generator import generate_bronze_sttm, generate_gold_sttm, generate_silver_sttm
from core.audit import AuditLogger
from core.config import LLM_PROVIDER, PipelineConfig
from core.logger import get_logger
from core.memory import AgentMemory
from core.observability import EventTracker
from tools.helpers import safe_stem


class PipelineState(TypedDict, total=False):
	run_id: str
	uploaded_files: list[str]
	business_intent: str
	profile_path: str
	sttm_bronze_path: str
	sttm_silver_path: str
	sttm_gold_path: str
	bronze_output_paths: list[str]
	silver_output_paths: list[str]
	gold_output_paths: list[str]
	report_path: str
	status: str
	error: str | None


def _make_llm():
	from core.llm_factory import make_llm

	return make_llm()


def _make_phase1_tools(state: PipelineState) -> list:
	from langchain_core.tools import tool

	def _do_profile() -> str:
		profile_path = profile_multiple_datasets(
			state["uploaded_files"], state["run_id"], state["business_intent"]
		)
		state["profile_path"] = profile_path
		return profile_path

	@tool
	def profile_tool() -> str:
		"""Profile the uploaded files and store the profile path in the pipeline state."""
		return json.dumps({"profile_path": _do_profile()})

	@tool
	def bronze_sttm_tool() -> str:
		"""Generate the Bronze STTM rules from the profile and store the STTM path in state."""
		# LLM tool-call order isn't guaranteed, so ensure the prerequisite ran first.
		if "profile_path" not in state:
			_do_profile()
		sttm_path = generate_bronze_sttm(
			state["profile_path"], state["run_id"], state["business_intent"]
		)
		state["sttm_bronze_path"] = sttm_path
		return json.dumps({"sttm_bronze_path": sttm_path})

	return [profile_tool, bronze_sttm_tool]


def _make_phase2_tools(state: PipelineState) -> list:
	from langchain_core.tools import tool

	def _do_bronze_ingest() -> list:
		paths = execute_bronze(
			state["uploaded_files"], state["sttm_bronze_path"], state["run_id"], state["business_intent"]
		)
		state["bronze_output_paths"] = paths
		return paths

	@tool
	def bronze_ingest_tool() -> str:
		"""Ingest the uploaded files into Bronze using the approved Bronze STTM rules."""
		return json.dumps({"bronze_output_paths": _do_bronze_ingest()})

	@tool
	def silver_sttm_tool() -> str:
		"""Generate the Silver STTM rules from the Bronze outputs and store the STTM path in state."""
		if "bronze_output_paths" not in state:
			_do_bronze_ingest()
		sttm_path = generate_silver_sttm(
			state["bronze_output_paths"], state["sttm_bronze_path"], state["run_id"], state["business_intent"]
		)
		state["sttm_silver_path"] = sttm_path
		return json.dumps({"sttm_silver_path": sttm_path})

	return [bronze_ingest_tool, silver_sttm_tool]


def _make_phase3_tools(state: PipelineState) -> list:
	from langchain_core.tools import tool

	def _do_silver_cleanse() -> list:
		paths = execute_silver(
			state["bronze_output_paths"], state["sttm_silver_path"], state["run_id"], state["business_intent"]
		)
		state["silver_output_paths"] = paths
		return paths

	@tool
	def silver_cleanse_tool() -> str:
		"""Cleanse the Bronze outputs into Silver using the approved Silver STTM rules."""
		return json.dumps({"silver_output_paths": _do_silver_cleanse()})

	@tool
	def gold_sttm_tool() -> str:
		"""Generate the Gold STTM rules from the Silver outputs and business intent."""
		if "silver_output_paths" not in state:
			_do_silver_cleanse()
		sttm_path = generate_gold_sttm(
			state["silver_output_paths"],
			state["sttm_silver_path"],
			state["business_intent"],
			state["run_id"],
			state["business_intent"],
		)
		state["sttm_gold_path"] = sttm_path
		return json.dumps({"sttm_gold_path": sttm_path})

	return [silver_cleanse_tool, gold_sttm_tool]


def _make_phase4_tools(state: PipelineState) -> list:
	from langchain_core.tools import tool

	def _do_gold_aggregate() -> list:
		paths = execute_gold(
			state["silver_output_paths"], state["sttm_gold_path"], state["run_id"], state["business_intent"]
		)
		state["gold_output_paths"] = paths
		return paths

	@tool
	def gold_aggregate_tool() -> str:
		"""Aggregate the Silver outputs into Gold using the approved Gold STTM rules."""
		return json.dumps({"gold_output_paths": _do_gold_aggregate()})

	@tool
	def report_tool() -> str:
		"""Generate the final HTML report answering the business intent from the Gold outputs."""
		if "gold_output_paths" not in state:
			_do_gold_aggregate()
		report_path = generate_report(
			state["gold_output_paths"], state["business_intent"], state["run_id"], state["business_intent"]
		)
		state["report_path"] = report_path
		return json.dumps({"report_path": report_path})

	return [gold_aggregate_tool, report_tool]


def _run_supervisor(tools: list, phase_goal: str, phase_name: str, run_id: str) -> None:
	from langgraph.prebuilt import create_react_agent

	audit = AuditLogger(run_id)
	audit.log("orchestrator", f"{phase_name}_start", goal=phase_goal)
	try:
		llm = _make_llm()
		agent = create_react_agent(llm, tools)
		agent.invoke({"messages": [("human", phase_goal)]}, config={"recursion_limit": 50})
		audit.log("orchestrator", f"{phase_name}_complete")
	except Exception as exc:
		audit.log("orchestrator", f"{phase_name}_agent_error", error=str(exc))
		# Tool-calling on smaller/free-tier models is unreliable; the last tool in each
		# phase is self-healing (it runs its prerequisite if missing), so falling back to
		# a direct, deterministic call guarantees the phase still completes.
		try:
			tools[-1].invoke({})
			audit.log("orchestrator", f"{phase_name}_complete_fallback")
		except Exception as fallback_exc:
			audit.log("orchestrator", f"{phase_name}_failed", error=str(fallback_exc))
			raise



def run_until_bronze_sttm(uploaded_files: list[str], business_intent: str, run_id: str | None = None) -> PipelineState:
	state: PipelineState = {
		"run_id": run_id or str(uuid.uuid4()),
		"uploaded_files": uploaded_files,
		"business_intent": business_intent,
		"status": "profiling",
	}
	tools = _make_phase1_tools(state)
	_run_supervisor(
		tools,
		f"Profile these files and generate Bronze STTM rules: {uploaded_files}. Business intent: {business_intent}",
		"phase1_bronze_sttm",
		state["run_id"],
	)
	state["status"] = "awaiting_bronze_approval"
	return state


def run_bronze_to_silver_sttm(state: PipelineState) -> PipelineState:
	tools = _make_phase2_tools(state)
	_run_supervisor(
		tools,
		"Ingest files into Bronze using the approved STTM rules, then generate Silver STTM rules.",
		"phase2_silver_sttm",
		state["run_id"],
	)
	state["status"] = "awaiting_silver_approval"
	return state


def run_silver_to_gold_sttm(state: PipelineState) -> PipelineState:
	tools = _make_phase3_tools(state)
	_run_supervisor(
		tools,
		"Cleanse Bronze data into Silver using the approved STTM rules, then generate Gold STTM rules.",
		"phase3_gold_sttm",
		state["run_id"],
	)
	state["status"] = "awaiting_gold_approval"
	return state


def run_gold_and_report(state: PipelineState) -> PipelineState:
	tools = _make_phase4_tools(state)
	_run_supervisor(
		tools,
		f"Aggregate Silver data into Gold using approved STTM rules, then generate the final report for: {state['business_intent']}",
		"phase4_report",
		state["run_id"],
	)
	state["status"] = "complete"
	return state


@dataclass
class PipelineRunResult:
	source_file: Path
	bronze_file: Path
	silver_file: Path
	gold_file: Path
	report_file: Path
	approval_required: bool
	approved: bool


class RetailMedallionOrchestrator:
	def __init__(self, config: PipelineConfig) -> None:
		self.config = config
		self.logger = get_logger("orchestrator")
		self.events = EventTracker(config.paths.audit_logs / "pipeline_events.jsonl")
		self.memory = AgentMemory(config.paths.audit_logs / "mapping_memory.json")

		self.bronze_agent = BronzeAgent()
		self.silver_agent = SilverAgent(memory=self.memory)
		self.gold_agent = GoldAgent()
		self.reporting_agent = ReportingAgent()
		self.profiler = DataProfiler()

	def run_file(self, source_file: Path, approval_override: bool | None = None) -> PipelineRunResult:
		self.logger.info("Starting medallion flow for %s", source_file)

		bronze = self.bronze_agent.ingest(source_file=source_file, bronze_dir=self.config.paths.bronze)
		self.events.record("bronze", "ingest_complete", bronze.profile)

		proposal = self.silver_agent.propose_mapping(
			source_columns=list(bronze.dataframe.columns),
			approval=self.config.approval,
		)
		self.events.record(
			"silver",
			"mapping_proposed",
			{
				"mapping": proposal.mapping,
				"confidence": proposal.confidence,
				"missing_targets": proposal.missing_targets,
				"requires_approval": proposal.requires_approval,
			},
		)

		approved = approval_override if approval_override is not None else not proposal.requires_approval
		if proposal.requires_approval and not approved:
			self.events.record(
				"silver",
				"approval_pending",
				{"source_file": str(source_file), "confidence": proposal.confidence},
			)
			raise PermissionError(
				"Mapping requires approval. Re-run with approval_override=True to execute."
			)

		silver = self.silver_agent.execute(
			frame=bronze.dataframe,
			proposal=proposal,
			silver_dir=self.config.paths.silver,
			source_name=safe_stem(source_file),
			rules=self.config.rules,
			approved=approved,
		)
		self.events.record("silver", "transform_complete", {"rows": len(silver.dataframe)})

		gold = self.gold_agent.aggregate(
			silver_frame=silver.dataframe,
			gold_dir=self.config.paths.gold,
			source_name=safe_stem(source_file),
		)
		self.events.record("gold", "aggregate_complete", {"rows": len(gold.dataframe)})

		report = self.reporting_agent.build_report(
			source_file=source_file,
			bronze_profile=bronze.profile,
			silver_frame=silver.dataframe,
			gold_frame=gold.dataframe,
			reports_dir=self.config.paths.reports,
		)
		self.events.record("report", "report_published", report.summary)

		self.profiler.write_profile(
			silver.dataframe,
			self.config.paths.profiles / f"{safe_stem(source_file)}_silver_profile.json",
		)
		self.profiler.write_profile(
			gold.dataframe,
			self.config.paths.profiles / f"{safe_stem(source_file)}_gold_profile.json",
		)

		return PipelineRunResult(
			source_file=source_file,
			bronze_file=bronze.bronze_file,
			silver_file=silver.silver_file,
			gold_file=gold.gold_file,
			report_file=report.report_file,
			approval_required=proposal.requires_approval,
			approved=approved,
		)
