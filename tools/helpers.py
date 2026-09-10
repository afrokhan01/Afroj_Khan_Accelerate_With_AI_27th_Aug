from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


def canonicalize(name: str) -> str:
	cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
	return re.sub(r"_+", "_", cleaned).strip("_")


def parse_numeric(value: object) -> float | None:
	if value is None or pd.isna(value):
		return None
	text = str(value).strip().replace(",", "")
	text = text.replace("$", "").replace("USD", "").replace("usd", "")
	if text == "":
		return None
	try:
		return float(text)
	except ValueError:
		return None


def parse_date(value: object) -> pd.Timestamp | None:
	if value is None or pd.isna(value):
		return None
	date = pd.to_datetime(value, errors="coerce", utc=False)
	if pd.isna(date):
		return None
	return date


def safe_stem(path: Path) -> str:
	return canonicalize(path.stem)


def source_signature(columns: list[str]) -> str:
	joined = "|".join(sorted(canonicalize(col) for col in columns))
	return hashlib.sha256(joined.encode("utf-8")).hexdigest()
