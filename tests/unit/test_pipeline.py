from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agents.orchestrator import RetailMedallionOrchestrator
from agents.silver_agent import SilverAgent
from core.config import ApprovalConfig, load_config
from core.memory import AgentMemory


def test_silver_mapping_inference_from_aliases(tmp_path: Path) -> None:
    memory = AgentMemory(tmp_path / "mapping_memory.json")
    agent = SilverAgent(memory=memory)

    proposal = agent.propose_mapping(
        source_columns=["Sale Date", "Store", "SKU", "Qty", "Amount", "Disc", "Curr"],
        approval=ApprovalConfig(auto_approve=True, confidence_threshold=0.7),
    )

    assert proposal.mapping["order_date"] == "Sale Date"
    assert proposal.mapping["store_id"] == "Store"
    assert proposal.mapping["product_id"] == "SKU"
    assert proposal.confidence == 1.0
    assert proposal.requires_approval is False


def test_silver_requires_approval_on_low_confidence(tmp_path: Path) -> None:
    memory = AgentMemory(tmp_path / "mapping_memory.json")
    agent = SilverAgent(memory=memory)

    proposal = agent.propose_mapping(
        source_columns=["date", "store", "qty"],
        approval=ApprovalConfig(auto_approve=True, confidence_threshold=0.9),
    )

    assert proposal.confidence < 0.9
    assert proposal.requires_approval is True


def test_orchestrator_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "data" / "bronze").mkdir(parents=True)
    (root / "data" / "silver").mkdir(parents=True)
    (root / "data" / "gold").mkdir(parents=True)
    (root / "data" / "reports").mkdir(parents=True)
    (root / "data" / "profiles").mkdir(parents=True)
    (root / "audit_logs").mkdir(parents=True)
    (root / "config" / "dev.yaml").write_text("", encoding="utf-8")

    source = root / "data" / "bronze" / "retail_sales.csv"
    frame = pd.DataFrame(
        [
            {
                "Sale Date": "2026-01-01",
                "Store": "S1",
                "SKU": "P1",
                "Qty": 2,
                "Amount": 100,
                "Disc": 0.1,
                "Curr": "USD",
            },
            {
                "Sale Date": "2026-01-02",
                "Store": "S1",
                "SKU": "P2",
                "Qty": 1,
                "Amount": 30,
                "Disc": 0.0,
                "Curr": "USD",
            },
        ]
    )
    frame.to_csv(source, index=False)

    config = load_config(environment="dev", project_root=root)
    orchestrator = RetailMedallionOrchestrator(config=config)
    result = orchestrator.run_file(source)

    assert result.bronze_file.exists()
    assert result.silver_file.exists()
    assert result.gold_file.exists()
    assert result.report_file.exists()


def test_orchestrator_blocks_without_approval(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "data" / "bronze").mkdir(parents=True)
    (root / "data" / "silver").mkdir(parents=True)
    (root / "data" / "gold").mkdir(parents=True)
    (root / "data" / "reports").mkdir(parents=True)
    (root / "data" / "profiles").mkdir(parents=True)
    (root / "audit_logs").mkdir(parents=True)
    (root / "config" / "dev.yaml").write_text(
        "approval:\n  auto_approve: false\n  confidence_threshold: 0.95\n",
        encoding="utf-8",
    )

    source = root / "data" / "bronze" / "unknown_schema.csv"
    pd.DataFrame([{"weird": "x", "other": "y"}]).to_csv(source, index=False)

    config = load_config(environment="dev", project_root=root)
    orchestrator = RetailMedallionOrchestrator(config=config)

    with pytest.raises(PermissionError):
        orchestrator.run_file(source, approval_override=False)