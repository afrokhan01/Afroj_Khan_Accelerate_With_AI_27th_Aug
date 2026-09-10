from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationResult:
	valid: bool
	errors: list[str]


def validate_silver(frame: pd.DataFrame) -> ValidationResult:
	required = [
		"order_date",
		"store_id",
		"product_id",
		"quantity",
		"unit_price",
		"discount",
		"gross_sales",
		"net_sales",
	]
	errors: list[str] = []

	for column in required:
		if column not in frame.columns:
			errors.append(f"Missing required column: {column}")

	if "discount" in frame.columns and (frame["discount"].fillna(0) > 1).any():
		errors.append("Discount values must be <= 1")
	if "quantity" in frame.columns and (frame["quantity"].fillna(0) < 1).any():
		errors.append("Quantity values must be >= 1")

	return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_gold(frame: pd.DataFrame) -> ValidationResult:
	required = ["order_month", "store_id", "transactions", "units_sold", "net_sales"]
	errors = [f"Missing required column: {column}" for column in required if column not in frame.columns]
	return ValidationResult(valid=len(errors) == 0, errors=errors)
