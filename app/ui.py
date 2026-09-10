from __future__ import annotations

import subprocess
import sys


def run_ui() -> int:
	return subprocess.call([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])


if __name__ == "__main__":
	raise SystemExit(run_ui())
