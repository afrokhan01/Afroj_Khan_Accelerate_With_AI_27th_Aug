from __future__ import annotations

import pandas as pd

from agents.silver_agent import _apply_silver_rules, _inspect_bronze


def _write_bronze_parquet(tmp_path, name: str) -> str:
    path = tmp_path / name
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 2],
            "category": ["Electronics", None, None],
            "amount": [10.0, None, None],
        }
    )
    frame.to_parquet(path, index=False)
    return str(path)


def test_inspect_bronze_previews_parquet_files(tmp_path):
    file_path = _write_bronze_parquet(tmp_path, "bronze.parquet")
    result = _inspect_bronze([file_path], "")

    assert file_path in result["previews"]
    assert result["previews"][file_path]["columns"] == ["order_id", "category", "amount"]
    assert result["sttm_rows"] == []


def test_apply_silver_rules_deduplicates_fills_nulls_and_adds_surrogate_key(tmp_path, monkeypatch):
    silver_dir = tmp_path / "silver_layer"
    monkeypatch.setattr("agents.silver_agent.SILVER_DIR", silver_dir)

    file_path = _write_bronze_parquet(tmp_path, "bronze")
    output_paths = _apply_silver_rules([file_path], "", "run-1")

    assert len(output_paths) == 1
    out_frame = pd.read_parquet(output_paths[0])

    assert len(out_frame) == 2  # duplicate row removed
    pk_columns = [col for col in out_frame.columns if col.startswith("pk_")]
    assert len(pk_columns) == 1
    assert out_frame["category"].isna().sum() == 0
    assert out_frame["amount"].isna().sum() == 0
