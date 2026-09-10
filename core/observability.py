from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRACES_DIR = Path("data/traces")
TRACES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Event:
	stage: str
	action: str
	payload: dict[str, Any]
	ts_utc: str


class EventTracker:
	def __init__(self, log_file: Path) -> None:
		self.log_file = log_file
		self.log_file.parent.mkdir(parents=True, exist_ok=True)

	def record(self, stage: str, action: str, payload: dict[str, Any]) -> Event:
		event = Event(
			stage=stage,
			action=action,
			payload=payload,
			ts_utc=datetime.now(UTC).isoformat(),
		)
		with self.log_file.open("a", encoding="utf-8") as handle:
			handle.write(json.dumps(event.__dict__, default=str) + "\n")
		return event


class AgentTrace:
	def __init__(self, agent_name: str, run_id: str) -> None:
		self.agent_name = agent_name
		self.run_id = run_id
		self._start = time.time()
		self.trace: dict[str, Any] = {
			"agent": agent_name,
			"run_id": run_id,
			"started_at": datetime.now(UTC).isoformat(),
			"input": {},
			"plan": "",
			"tool_calls": [],
			"reasoning_steps": [],
			"output": {},
			"duration_seconds": None,
			"status": "in_progress",
		}

	def set_input(self, **kwargs: Any) -> AgentTrace:
		self.trace["input"] = kwargs
		return self

	def set_plan(self, plan_str: str) -> AgentTrace:
		self.trace["plan"] = plan_str
		return self

	def set_output(self, **kwargs: Any) -> AgentTrace:
		self.trace["output"] = kwargs
		return self

	def extract_from_messages(self, messages: list[Any]) -> AgentTrace:
		for message in messages:
			msg_type = type(message).__name__
			if msg_type == "HumanMessage":
				self.trace["reasoning_steps"].append(
					{"type": "task_input", "content": getattr(message, "content", "")}
				)
			elif msg_type == "AIMessage":
				tool_calls = getattr(message, "tool_calls", None) or []
				if tool_calls:
					self.trace["tool_calls"].extend(tool_calls)
				content = getattr(message, "content", "")
				if content:
					self.trace["reasoning_steps"].append({"type": "ai_reasoning", "content": content})
					if not self.trace["plan"]:
						self.trace["plan"] = content
			elif msg_type == "ToolMessage":
				self.trace["reasoning_steps"].append(
					{"type": "tool_result", "content": getattr(message, "content", "")}
				)
		return self

	def _save(self) -> Path:
		out_file = TRACES_DIR / f"trace_{self.agent_name}_{self.run_id[:8]}.json"
		out_file.write_text(json.dumps(self.trace, default=str, indent=2), encoding="utf-8")
		return out_file

	def complete(self, status: str = "success") -> AgentTrace:
		self.trace["duration_seconds"] = round(time.time() - self._start, 3)
		self.trace["status"] = status
		out_file = self._save()
		print(f"[trace] {self.agent_name} ({self.run_id[:8]}) {status} in {self.trace['duration_seconds']}s -> {out_file}")
		return self

	def fail(self, error: str) -> AgentTrace:
		self.trace["error"] = str(error)
		return self.complete(status="failed")
