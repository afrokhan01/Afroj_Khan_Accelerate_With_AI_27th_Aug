"""Quick import sanity check for IDAMP entry points.

Run with: python test_import.py
Exits non-zero if any import fails, so it can be used as a fast pre-flight check
before installing an LLM API key or launching the Streamlit app.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from agents.orchestrator import (
            run_bronze_to_silver_sttm,
            run_gold_and_report,
            run_silver_to_gold_sttm,
            run_until_bronze_sttm,
        )
        from core.config import LLM_PROVIDER
        from core.state import PipelineState

        print("All IDAMP entry points imported successfully.")
        print(f"Detected LLM provider: {LLM_PROVIDER or '(none configured)'}")
        print("Entry points:", [
            run_until_bronze_sttm.__name__,
            run_bronze_to_silver_sttm.__name__,
            run_silver_to_gold_sttm.__name__,
            run_gold_and_report.__name__,
        ])
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level sanity check
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
