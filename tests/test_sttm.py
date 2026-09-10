from __future__ import annotations

import json

import pandas as pd

from agents.sttm_generator import (
    STTM_COLUMNS,
    _extract_sttm_rows,
    _prepare_bronze_context,
    _prepare_gold_context,
    _prepare_silver_context,
    _rows_to_csv,
)


def test_prepare_bronze_context_loads_profile_json(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"inspect": {}, "stats": {}}), encoding="utf-8")

    context = _prepare_bronze_context(str(profile_path))
    assert "inspect" in context
    assert "stats" in context


def test_prepare_bronze_context_missing_file_raises(tmp_path):
    missing_path = tmp_path / "missing.json"
    try:
        _prepare_bronze_context(str(missing_path))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_prepare_silver_context_reads_bronze_columns(tmp_path):
    bronze_path = tmp_path / "sales_bronze.parquet"
    pd.DataFrame({"order_id": [1], "category": ["Electronics"]}).to_parquet(bronze_path, index=False)

    context = _prepare_silver_context([str(bronze_path)], "")
    assert bronze_path.name in context
    assert "order_id" in context[bronze_path.name]["columns"]


def test_prepare_gold_context_reads_silver_columns(tmp_path):
    silver_path = tmp_path / "sales_silver.parquet"
    pd.DataFrame({"category": ["Electronics"], "total_amount": [100.0]}).to_parquet(silver_path, index=False)

    context = _prepare_gold_context([str(silver_path)], "")
    assert silver_path.name in context
    assert "total_amount" in context[silver_path.name]["columns"]


def test_rows_to_csv_writes_sttm_columns(tmp_path):
    out_file = tmp_path / "sttm.csv"
    rows = [
        {
            "source_schema": "",
            "source_table": "sales",
            "source_column": "order_id",
            "target_schema": "",
            "target_table": "sales",
            "target_column": "order_id",
            "transformation_type": "passthrough",
            "transformation_logic": "",
        }
    ]
    _rows_to_csv(rows, out_file)

    frame = pd.read_csv(out_file)
    assert list(frame.columns) == STTM_COLUMNS
    assert len(frame) == 1


def test_extract_sttm_rows_parses_json_array_from_message_content():
    class FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    rows = [{"source_column": "a", "target_column": "b"}]
    message = FakeMessage("Here are the rules:\n" + json.dumps(rows))

    extracted = _extract_sttm_rows([message])
    assert extracted == rows


def test_extract_sttm_rows_returns_empty_when_no_json_array():
    class FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    extracted = _extract_sttm_rows([FakeMessage("no json here")])
    assert extracted == []
