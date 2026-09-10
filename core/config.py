from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

# Static project-wide paths and provider settings (used by AI agent tooling).
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
PROFILES_DIR = DATA_DIR / "profiles"
STTM_DIR = DATA_DIR / "sttm"
BRONZE_DIR = DATA_DIR / "bronze_layer"
SILVER_DIR = DATA_DIR / "silver_layer"
GOLD_DIR = DATA_DIR / "gold_layer"
REPORTS_DIR = BASE_DIR / "reports"
AUDIT_DIR = BASE_DIR / "audit_logs"
CHROMA_DIR = BASE_DIR / ".chroma"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini")
GITHUB_BASE_URL = "https://models.inference.ai.azure.com"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")


def _detect_llm_provider() -> str:
    provider = os.getenv("LLM_PROVIDER")
    if provider:
        return provider.lower()
    if GITHUB_TOKEN:
        return "github"
    if OPENAI_API_KEY:
        return "openai"
    if GROQ_API_KEY:
        return "groq"
    if GOOGLE_API_KEY:
        return "gemini"
    return ""


LLM_PROVIDER = _detect_llm_provider()

for _directory in (
    DATA_DIR,
    LANDING_DIR,
    PROFILES_DIR,
    STTM_DIR,
    BRONZE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    REPORTS_DIR,
    AUDIT_DIR,
    CHROMA_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelinePaths:
	raw_input: Path
	bronze: Path
	silver: Path
	gold: Path
	reports: Path
	profiles: Path
	audit_logs: Path


@dataclass
class ApprovalConfig:
	auto_approve: bool = True
	confidence_threshold: float = 0.75


@dataclass
class BusinessRules:
	discount_max: float = 1.0
	min_quantity: int = 1
	drop_rows_missing_keys: bool = True
	required_keys: list[str] = field(
		default_factory=lambda: ["order_date", "store_id", "product_id"]
	)


@dataclass
class PipelineConfig:
	environment: str
	paths: PipelinePaths
	approval: ApprovalConfig = field(default_factory=ApprovalConfig)
	rules: BusinessRules = field(default_factory=BusinessRules)


def _merge_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
	for key, value in updates.items():
		if hasattr(instance, key):
			setattr(instance, key, value)
	return instance


def _resolve_paths(root: Path) -> PipelinePaths:
	data_dir = root / "data"
	return PipelinePaths(
		raw_input=data_dir / "bronze",
		bronze=data_dir / "bronze",
		silver=data_dir / "silver",
		gold=data_dir / "gold",
		reports=data_dir / "reports",
		profiles=data_dir / "profiles",
		audit_logs=root / "audit_logs",
	)


def load_config(environment: str = "dev", project_root: Path | None = None) -> PipelineConfig:
	root = (project_root or Path(__file__).resolve().parents[1]).resolve()
	config_path = root / "config" / f"{environment}.yaml"

	config = PipelineConfig(environment=environment, paths=_resolve_paths(root))

	if config_path.exists() and config_path.stat().st_size > 0:
		with config_path.open("r", encoding="utf-8") as handle:
			payload = yaml.safe_load(handle) or {}

		if "approval" in payload:
			config.approval = _merge_dataclass(config.approval, payload["approval"])
		if "rules" in payload:
			config.rules = _merge_dataclass(config.rules, payload["rules"])
		if "paths" in payload:
			current = config.paths
			raw = payload["paths"]
			config.paths = PipelinePaths(
				raw_input=root / raw.get("raw_input", str(current.raw_input.relative_to(root))),
				bronze=root / raw.get("bronze", str(current.bronze.relative_to(root))),
				silver=root / raw.get("silver", str(current.silver.relative_to(root))),
				gold=root / raw.get("gold", str(current.gold.relative_to(root))),
				reports=root / raw.get("reports", str(current.reports.relative_to(root))),
				profiles=root / raw.get("profiles", str(current.profiles.relative_to(root))),
				audit_logs=root / raw.get("audit_logs", str(current.audit_logs.relative_to(root))),
			)

	for directory in [
		config.paths.bronze,
		config.paths.silver,
		config.paths.gold,
		config.paths.reports,
		config.paths.profiles,
		config.paths.audit_logs,
	]:
		directory.mkdir(parents=True, exist_ok=True)

	return config
