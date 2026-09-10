from __future__ import annotations

import argparse
from pathlib import Path

from agents.orchestrator import RetailMedallionOrchestrator
from core.config import load_config


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Retail Medallion agentic pipeline")
	parser.add_argument("--env", default="dev", help="Environment config file to use")
	parser.add_argument("--file", default=None, help="Optional CSV file path to run")
	parser.add_argument(
		"--approve",
		action="store_true",
		help="Approve pending inferred mapping for execution",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	config = load_config(environment=args.env)
	orchestrator = RetailMedallionOrchestrator(config=config)

	if args.file:
		files = [Path(args.file)]
	else:
		files = sorted(config.paths.raw_input.glob("*.csv"))

	if not files:
		print("No CSV files found to process.")
		return 0

	for file in files:
		result = orchestrator.run_file(file, approval_override=args.approve)
		print(
			f"Processed {file.name}: bronze={result.bronze_file.name}, "
			f"silver={result.silver_file.name}, gold={result.gold_file.name}, "
			f"report={result.report_file.name}"
		)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
