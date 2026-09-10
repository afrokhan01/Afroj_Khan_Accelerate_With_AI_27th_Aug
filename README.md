# Retail Medallion Pipeline & IDAMP

This repository contains **two** pipelines that share the same Medallion (Bronze/Silver/Gold) shape but solve the problem differently:

1. **Retail Medallion Pipeline (original)** — a fast, deterministic, rule-based pandas pipeline. No LLM required. Fully tested (`tests/unit/`).
2. **IDAMP — Intent-Driven Agentic Medallion Pipeline (new)** — an LLM-agent-driven pipeline built with LangChain/LangGraph. Agents infer STTM (Source-to-Target Mapping) rules from your data and a plain-English business question, with a human-in-the-loop (HITL) approval gate between every layer.

Both live side by side in this codebase without conflicting: IDAMP's agent logic was added as new functions/files alongside the original classes, so the original pipeline and its test suite keep working unmodified.

## Table of Contents

- [Architecture](#architecture)
- [Medallion Layers](#medallion-layers)
- [IDAMP Agents](#idamp-agents)
- [Setup](#setup)
- [Running the Original Pipeline](#running-the-original-pipeline)
- [Running IDAMP](#running-idamp)
- [Tests](#tests)
- [FAQ](#faq)

## Architecture

```
                 ┌────────────────────────┐
                 │   Streamlit UI (HITL)  │
                 │   app/streamlit_app.py │
                 └───────────┬────────────┘
                             │
                 ┌───────────▼────────────┐
                 │   Supervisor Agent      │
                 │  agents/orchestrator.py │
                 └───────────┬────────────┘
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                     ▼
 ┌─────────────┐     ┌──────────────┐      ┌───────────────┐
 │ Profiler /  │ --> │ Bronze / STTM│ -->  │ Silver / STTM │ --> Gold / STTM --> Reporter
 │ STTM Agent  │     │   Agent      │      │    Agent      │
 └─────────────┘     └──────────────┘      └───────────────┘
        │                    │                     │
        ▼                    ▼                     ▼
  data/profiles        data/bronze_layer      data/silver_layer  ->  data/gold_layer  ->  reports/*.html
```

Every agent call is captured by `core/observability.py`'s `AgentTrace` (saved to `data/traces/`) and every pipeline action is recorded by `core/audit.py`'s `AuditLogger` (saved to `audit_logs/<run_id>.jsonl`), which the Streamlit sidebar reads to show a live audit trail.

## Medallion Layers

| Layer  | Purpose | Original Pipeline | IDAMP |
|--------|---------|--------------------|-------|
| Bronze | Raw ingestion + light metadata | `agents/bronze_agent.py` → `BronzeAgent.ingest()` | `execute_bronze()` — applies LLM-generated Bronze STTM rules, adds `_load_timestamp`/`_source_file` |
| Silver | Cleansing, dedup, mapping | `SilverAgent.execute()` (alias-based mapping) | `execute_silver()` — applies Silver STTM rules, adds a `pk_{table}_silver_id` surrogate key, null handling |
| Gold   | Aggregation for reporting | `GoldAgent.aggregate()` (fixed groupby) | `execute_gold()` — STTM-driven aggregation with a `pk_gold_id` surrogate key |
| Report | Summaries / analytics | `ReportingAgent.build_report()` (JSON) | `generate_report()` — DuckDB SQL + Plotly chart + styled HTML report |

## IDAMP Agents

| File | Responsibility |
|------|-----------------|
| `agents/profiler.py` (`profile_dataset`, `profile_multiple_datasets`) | Inspects and computes statistics for uploaded files |
| `agents/sttm_generator.py` | Generates Bronze/Silver/Gold STTM rule CSVs from profiles + business intent |
| `agents/bronze_agent.py` (`execute_bronze`) | Ingests raw files into Bronze using approved STTM rules |
| `agents/silver_agent.py` (`execute_silver`) | Cleanses Bronze data into Silver using approved STTM rules |
| `agents/gold_agent.py` (`execute_gold`) | Aggregates Silver data into Gold using approved STTM rules |
| `agents/reporter.py` (`generate_report`) | Answers the business question with SQL + a chart + an HTML report |
| `agents/orchestrator.py` | LangGraph supervisor exposing 4 HITL-gated phase functions: `run_until_bronze_sttm`, `run_bronze_to_silver_sttm`, `run_silver_to_gold_sttm`, `run_gold_and_report` |

Supporting modules: `core/config.py` (paths + LLM provider detection), `core/audit.py` (`AuditLogger`), `core/observability.py` (`AgentTrace`), `core/memory.py` (ChromaDB semantic memory), `core/state.py` (Pydantic `PipelineState`).

## Setup

1. Create/activate a virtual environment, then install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in **one** LLM provider's credentials (Groq, Google Gemini, OpenAI, or GitHub Models). IDAMP auto-detects the provider based on which key is present, or you can set `LLM_PROVIDER` explicitly.

3. Verify everything imports correctly (no API key required for this step):

   ```powershell
   python test_import.py
   ```

## Running the Original Pipeline

```powershell
python -m app.main --env dev --file data/bronze/your_file.csv
streamlit run streamlit_app.py
```

## Running IDAMP

```powershell
streamlit run app/streamlit_app.py
```

1. **Phase 1 — Upload & Intent**: upload CSVs (or use the bundled sample data in `data/landing/`) and type a business question, e.g. *"Which category had the highest sales?"*.
2. **Phase 2 — Bronze STTM**: review/edit the LLM-generated Bronze mapping rules, then approve.
3. **Phase 3 — Silver STTM**: review/edit the Silver cleansing rules, then approve.
4. **Phase 4 — Gold STTM**: review/edit the Gold aggregation rules, then approve.
5. **Phase 5 — Report**: view and download the generated HTML report (chart + data table answering your question).

The sidebar shows a live audit trail for the current run, sourced from `audit_logs/<run_id>.jsonl`.

## Tests

```powershell
pytest -q
```

This runs both suites:
- `tests/unit/` — original rule-based pipeline (4 tests, no external dependencies).
- `tests/test_*.py` — IDAMP's pure-Python agent helpers (profiler stats, STTM CSV generation, Bronze/Silver/Gold transform logic, DuckDB queries, chart/report generation, audit logging). These test the deterministic logic behind each tool, not the LLM call itself, so they run without any API key.

## FAQ

**Do I need an API key to run the tests?** No. All IDAMP tests exercise the pure-Python helper functions directly (`_apply_bronze_rules`, `_query_gold`, etc.), not the LLM-calling `execute_*`/`generate_*` wrappers.

**Which LLM providers are supported?** Groq, Google Gemini, OpenAI, and GitHub Models. Set the corresponding key in `.env`; `core/config.py` picks the first one found (or respects `LLM_PROVIDER` if set explicitly).

**Where do generated files go?** `data/landing` (uploads), `data/profiles` (profiles), `data/sttm` (STTM CSVs), `data/bronze_layer` / `data/silver_layer` / `data/gold_layer` (Parquet outputs), `reports/` (HTML reports), `data/traces` (per-agent JSON traces), `audit_logs/` (JSONL audit trail).

**Can I still use the original, non-agentic pipeline?** Yes — it is unchanged and fully tested; use `app/main.py` or the root `streamlit_app.py`.

