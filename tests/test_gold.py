from __future__ import annotations

import pandas as pd

from agents.gold_agent import _apply_gold_rules, _inspect_silver


def _write_silver_parquet(tmp_path, name: str) -> str:
    path = tmp_path / name
    frame = pd.DataFrame(
        {
            "category": ["Electronics", "Electronics", "Apparel"],
            "store_id": ["S01", "S01", "S02"],
            "quantity": [1, 2, 1],
            "total_amount": [100.0, 200.0, 50.0],
        }
    )
    frame.to_parquet(path, index=False)
    return str(path)


def test_inspect_silver_previews_parquet_files(tmp_path):
    file_path = _write_silver_parquet(tmp_path, "silver.parquet")
    result = _inspect_silver([file_path], "")

    assert file_path in result["previews"]
    assert "category" in result["previews"][file_path]["columns"]


def test_apply_gold_rules_aggregates_and_adds_surrogate_key(tmp_path, monkeypatch):
    gold_dir = tmp_path / "gold_layer"
    monkeypatch.setattr("agents.gold_agent.GOLD_DIR", gold_dir)

    file_path = _write_silver_parquet(tmp_path, "silver")
    output_paths = _apply_gold_rules([file_path], "", "run-1")

    assert len(output_paths) == 1
    out_frame = pd.read_parquet(output_paths[0])

    assert "pk_gold_id" in out_frame.columns
    assert len(out_frame) == 2  # grouped by category+store_id
    electronics_row = out_frame[out_frame["category"] == "Electronics"].iloc[0]
    assert electronics_row["quantity"] == 3
    assert electronics_row["total_amount"] == 300.0


def test_apply_gold_rules_uses_sttm_group_and_aggregation_definitions(tmp_path, monkeypatch):
    gold_dir = tmp_path / "gold_layer"
    monkeypatch.setattr("agents.gold_agent.GOLD_DIR", gold_dir)

    silver_file = _write_silver_parquet(tmp_path, "sales_silver")
    sttm_path = tmp_path / "gold_sttm.csv"
    sttm = pd.DataFrame(
        [
            {
                "source_schema": "silver",
                "source_table": "sales_silver",
                "source_column": "store_id",
                "target_schema": "gold",
                "target_table": "sales_gold",
                "target_column": "store_id",
                "transformation_type": "GROUP_BY",
                "transformation_logic": "GROUP_BY(store_id)",
            },
            {
                "source_schema": "silver",
                "source_table": "sales_silver",
                "source_column": "quantity",
                "target_schema": "gold",
                "target_table": "sales_gold",
                "target_column": "units_sold",
                "transformation_type": "SUM",
                "transformation_logic": "SUM(quantity)",
            },
            {
                "source_schema": "silver",
                "source_table": "sales_silver",
                "source_column": "quantity",
                "target_schema": "gold",
                "target_table": "sales_gold",
                "target_column": "txn_count",
                "transformation_type": "COUNT",
                "transformation_logic": "COUNT(quantity)",
            },
        ]
    )
    sttm.to_csv(sttm_path, index=False)

    output_paths = _apply_gold_rules([silver_file], str(sttm_path), "run-2")
    out_frame = pd.read_parquet(output_paths[0])

    assert "pk_gold_id" in out_frame.columns
    assert "store_id" in out_frame.columns
    assert "units_sold" in out_frame.columns
    assert "txn_count" in out_frame.columns
    assert len(out_frame) == 2

    s01 = out_frame[out_frame["store_id"] == "S01"].iloc[0]
    assert s01["units_sold"] == 3
    assert s01["txn_count"] == 2
