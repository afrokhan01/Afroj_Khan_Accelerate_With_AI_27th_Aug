from __future__ import annotations

import pandas as pd

from agents.bronze_agent import _apply_bronze_rules, _inspect_task


def _write_csv(tmp_path, name: str) -> str:
    path = tmp_path / name
    pd.DataFrame({"order_id": [1, 2], "store": ["S01", "S02"]}).to_csv(path, index=False)
    return str(path)


def test_inspect_task_previews_files_without_sttm(tmp_path):
    file_path = _write_csv(tmp_path, "sales.csv")
    result = _inspect_task([file_path], "")

    assert file_path in result["previews"]
    assert result["previews"][file_path]["columns"] == ["order_id", "store"]
    assert result["sttm_rows"] == []


def test_apply_bronze_rules_adds_metadata_columns(tmp_path, monkeypatch):
    bronze_dir = tmp_path / "bronze_layer"
    monkeypatch.setattr("agents.bronze_agent.BRONZE_DIR", bronze_dir)

    file_path = _write_csv(tmp_path, "sales.csv")
    output_paths = _apply_bronze_rules([file_path], "", "run-1")

    assert len(output_paths) == 1
    out_frame = pd.read_parquet(output_paths[0])
    assert "_load_timestamp" in out_frame.columns
    assert "_source_file" in out_frame.columns
    assert out_frame["_source_file"].iloc[0] == file_path


def test_apply_bronze_rules_renames_columns_using_sttm(tmp_path, monkeypatch):
    bronze_dir = tmp_path / "bronze_layer"
    monkeypatch.setattr("agents.bronze_agent.BRONZE_DIR", bronze_dir)

    file_path = _write_csv(tmp_path, "sales.csv")
    sttm_path = tmp_path / "bronze_sttm.csv"
    pd.DataFrame(
        [
            {
                "source_schema": "",
                "source_table": "sales",
                "source_column": "order_id",
                "target_schema": "",
                "target_table": "sales",
                "target_column": "sales_order_id",
                "transformation_type": "rename",
                "transformation_logic": "",
            }
        ]
    ).to_csv(sttm_path, index=False)

    output_paths = _apply_bronze_rules([file_path], str(sttm_path), "run-2")
    out_frame = pd.read_parquet(output_paths[0])

    assert "sales_order_id" in out_frame.columns
    assert "order_id" not in out_frame.columns
