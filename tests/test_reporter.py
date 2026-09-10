from __future__ import annotations

import pandas as pd

from agents.reporter import _build_html_report, _query_gold, generate_chart_from_spec


def _write_gold_parquet(tmp_path, name: str) -> str:
    path = tmp_path / name
    frame = pd.DataFrame(
        {
            "category": ["Electronics", "Apparel"],
            "total_amount": [500.0, 150.0],
        }
    )
    frame.to_parquet(path, index=False)
    return str(path)


def test_query_gold_runs_default_query(tmp_path):
    file_path = _write_gold_parquet(tmp_path, "gold.parquet")
    result = _query_gold([file_path])

    assert result["columns"] == ["category", "total_amount"]
    assert len(result["rows"]) == 2


def test_query_gold_runs_custom_sql(tmp_path):
    file_path = _write_gold_parquet(tmp_path, "gold.parquet")
    table_name = "gold"
    sql = f"SELECT category, SUM(total_amount) AS total FROM {table_name} GROUP BY category ORDER BY category"
    result = _query_gold([file_path], sql=sql)

    assert result["columns"] == ["category", "total"]
    assert len(result["rows"]) == 2


def test_generate_chart_from_spec_returns_html():
    data = pd.DataFrame({"category": ["Electronics", "Apparel"], "total_amount": [500.0, 150.0]})
    chart_html = generate_chart_from_spec(data, "bar", "category", "total_amount", "Sales by Category")

    assert "<div" in chart_html or "plotly" in chart_html.lower()


def test_build_html_report_includes_business_intent_and_data():
    query_result = {"columns": ["category"], "rows": [{"category": "Electronics"}]}
    html = _build_html_report("Which category sold the most?", query_result, "<div>chart</div>", "run-1234")

    assert "Which category sold the most?" in html
    assert "Electronics" in html
    assert "<div>chart</div>" in html
