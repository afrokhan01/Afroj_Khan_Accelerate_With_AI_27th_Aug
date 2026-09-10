from __future__ import annotations

from core.audit import AuditLogger


def test_audit_logger_creates_log_file(tmp_path, monkeypatch):
    monkeypatch.setattr("core.audit.AUDIT_DIR", tmp_path)
    logger = AuditLogger(run_id="run-123")
    assert logger.log_path == tmp_path / "run-123.jsonl"


def test_audit_logger_log_and_get_logs(tmp_path, monkeypatch):
    monkeypatch.setattr("core.audit.AUDIT_DIR", tmp_path)
    logger = AuditLogger(run_id="run-456")

    logger.log("bronze_agent", "start", task="ingest files")
    logger.log("bronze_agent", "complete", output_paths=["a.parquet"])

    logs = logger.get_logs()
    assert len(logs) == 2
    assert logs[0]["agent"] == "bronze_agent"
    assert logs[0]["action"] == "start"
    assert logs[1]["action"] == "complete"
    assert logs[1]["output_paths"] == ["a.parquet"]


def test_audit_logger_generates_run_id_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("core.audit.AUDIT_DIR", tmp_path)
    logger = AuditLogger()
    assert logger.run_id
    assert logger.log_path.name == f"{logger.run_id}.jsonl"


def test_audit_logger_get_logs_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("core.audit.AUDIT_DIR", tmp_path)
    logger = AuditLogger(run_id="never-logged")
    assert logger.get_logs() == []
