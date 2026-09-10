from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from random import Random

import pandas as pd


def generate_sample_sales_csv(out_file: Path, rows: int = 100, seed: int = 42) -> Path:
	rng = Random(seed)
	start = date(2026, 1, 1)
	payload: list[dict[str, object]] = []

	for idx in range(rows):
		payload.append(
			{
				"Sale Date": (start + timedelta(days=idx % 120)).isoformat(),
				"Store": f"S{(idx % 5) + 1}",
				"SKU": f"P{(idx % 20) + 1}",
				"Qty": rng.randint(1, 5),
				"Amount": round(rng.uniform(5, 200), 2),
				"Disc": round(rng.uniform(0, 0.3), 2),
				"Curr": "USD",
			}
		)

	frame = pd.DataFrame(payload)
	out_file.parent.mkdir(parents=True, exist_ok=True)
	frame.to_csv(out_file, index=False)
	return out_file
