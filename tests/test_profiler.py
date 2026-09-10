from __future__ import annotations

import pandas as pd

from agents.profiler import _compute_stats, _inspect_files


def _write_csv(tmp_path, name: str) -> str:
    path = tmp_path / name
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, None],
            "category": ["Electronics", "Apparel", None, "Electronics"],
            "amount": [10.5, 20.0, 30.25, 5.0],
        }
    )
    frame.to_csv(path, index=False)
    return str(path)


def test_inspect_files_returns_shape_and_columns(tmp_path):
    file_path = _write_csv(tmp_path, "sample.csv")
    result = _inspect_files([file_path])

    assert file_path in result
    assert result[file_path]["shape"] == [4, 3]
    assert result[file_path]["columns"] == ["id", "category", "amount"]
    assert "samples" in result[file_path]


def test_compute_stats_reports_nulls_and_numeric_summary(tmp_path):
    file_path = _write_csv(tmp_path, "sample2.csv")
    result = _compute_stats([file_path])

    stats = result[file_path]
    assert stats["rows"] == 4
    assert stats["columns"]["id"]["null_count"] == 1
    assert stats["columns"]["amount"]["min"] == 5.0
    assert stats["columns"]["amount"]["max"] == 30.25
    assert "sample_values" in stats["columns"]["category"]


def test_compute_stats_handles_multiple_files(tmp_path):
    file1 = _write_csv(tmp_path, "a.csv")
    file2 = _write_csv(tmp_path, "b.csv")
    result = _compute_stats([file1, file2])

    assert set(result.keys()) == {file1, file2}
