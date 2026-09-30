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


def test_apply_silver_rules_applies_sttm_date_and_fill_null_directives(tmp_path, monkeypatch):
    silver_dir = tmp_path / "silver_layer"
    monkeypatch.setattr("agents.silver_agent.SILVER_DIR", silver_dir)

    bronze_file = tmp_path / "orders"
    frame = pd.DataFrame(
        {
            "sale_date": ["2024-01-01", "01/02/2024", None],
            "amount": [10.0, None, 5.0],
            "category": [None, "Electronics", None],
        }
    )
    frame.to_parquet(bronze_file, index=False)

    sttm_path = tmp_path / "silver_sttm.csv"
    sttm = pd.DataFrame(
        [
            {
                "source_schema": "bronze",
                "source_table": "orders",
                "source_column": "sale_date",
                "target_schema": "silver",
                "target_table": "orders_silver",
                "target_column": "order_date",
                "transformation_type": "DATE_FORMAT",
                "transformation_logic": "YYYY-MM-DD",
            },
            {
                "source_schema": "bronze",
                "source_table": "orders",
                "source_column": "amount",
                "target_schema": "silver",
                "target_table": "orders_silver",
                "target_column": "amount",
                "transformation_type": "FILL_NULL",
                "transformation_logic": "VALUE:0",
            },
            {
                "source_schema": "bronze",
                "source_table": "orders",
                "source_column": "category",
                "target_schema": "silver",
                "target_table": "orders_silver",
                "target_column": "category",
                "transformation_type": "FILL_NULL",
                "transformation_logic": "VALUE:Missing",
            },
        ]
    )
    sttm.to_csv(sttm_path, index=False)

    output_paths = _apply_silver_rules([str(bronze_file)], str(sttm_path), "run-2")
    out_frame = pd.read_parquet(output_paths[0])

    assert "order_date" in out_frame.columns
    assert out_frame["order_date"].notna().sum() >= 2
    assert out_frame["amount"].isna().sum() == 0
    assert 0 in out_frame["amount"].tolist()
    assert "Missing" in out_frame["category"].tolist()
