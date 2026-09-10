# API

## Orchestrator

- `RetailMedallionOrchestrator(config)`
	- `run_file(source_file: Path, approval_override: bool | None = None)`

`run_file` executes Bronze, Silver, Gold, and Reporting stages and returns artifact paths.

## Agents

- `BronzeAgent.ingest(source_file, bronze_dir)`
- `SilverAgent.propose_mapping(source_columns, approval)`
- `SilverAgent.execute(frame, proposal, silver_dir, source_name, rules, approved)`
- `GoldAgent.aggregate(silver_frame, gold_dir, source_name)`
- `ReportingAgent.build_report(...)`

## Command line

Run all files in `data/bronze`:

```bash
python -m app.main --env dev
```

Run a single file:

```bash
python -m app.main --env dev --file data/bronze/new_sales.csv
```

Approve a pending low-confidence mapping:

```bash
python -m app.main --env staging --file data/bronze/new_sales.csv --approve
```
