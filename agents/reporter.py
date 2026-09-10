from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.audit import AuditLogger
from core.config import LLM_PROVIDER, REPORTS_DIR
from core.observability import AgentTrace

REPORTER_AGENT_PROMPT = """You MUST call query_gold_tool to answer the business intent with SQL over the Gold Parquet files,
then call chart_spec_tool to describe a chart, then call html_report_tool to write the final report.
Planning without executing is FAILURE.
"""


def _query_gold(gold_files: list[str], sql: str | None = None) -> dict:
    import duckdb

    con = duckdb.connect()
    for path in gold_files:
        table_name = Path(path).stem
        con.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{path}')")

    if not sql:
        first_table = Path(gold_files[0]).stem
        sql = f"SELECT * FROM {first_table} LIMIT 100"

    result = con.execute(sql).fetchdf()
    return {"columns": list(result.columns), "rows": json.loads(result.to_json(orient="records"))}


def generate_chart_from_spec(data: pd.DataFrame, chart_type: str, x: str, y: str, title: str) -> str:
    import plotly.express as px

    chart_fns = {
        "bar": px.bar,
        "line": px.line,
        "pie": px.pie,
        "scatter": px.scatter,
    }
    fn = chart_fns.get(chart_type, px.bar)
    if chart_type == "pie":
        fig = fn(data, names=x, values=y, title=title)
    else:
        fig = fn(data, x=x, y=y, title=title)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _build_html_report(business_intent: str, query_result: dict, chart_html: str, run_id: str) -> str:
    rows_html = pd.DataFrame(query_result["rows"]).to_html(index=False, classes="data-table")
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>IDAMP Report - {run_id[:8]}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2rem; background: #f7f8fa; color: #1a1a1a; }}
  h1 {{ color: #2c3e50; }}
  .intent {{ background: #eef2f7; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; }}
  .data-table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
  .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  .data-table th {{ background: #2c3e50; color: white; }}
</style>
</head>
<body>
  <h1>IDAMP Pipeline Report</h1>
  <div class="intent"><strong>Business intent:</strong> {business_intent}</div>
  <h2>Chart</h2>
  {chart_html}
  <h2>Data</h2>
  {rows_html}
</body>
</html>"""
    return html


def _make_reporter_tools(gold_files: list[str], business_intent: str, run_id: str) -> list:
    from langchain_core.tools import tool

    state: dict[str, object] = {}
    table_names = [Path(p).stem for p in gold_files]

    @tool
    def query_gold_tool(sql: str) -> str:
        """Run a DuckDB SQL query over the Gold Parquet files to answer the business intent."""
        try:
            result = _query_gold(gold_files, sql)
        except Exception as exc:
            return f"Error: {exc}. Available table names are exactly: {table_names}. Retry with corrected SQL."
        state["query_result"] = result
        return json.dumps(result)

    @tool
    def chart_spec_tool(chart_type: str, x: str, y: str, title: str) -> str:
        """Generate a Plotly chart (bar, line, pie, scatter) from the last query result."""
        query_result = state.get("query_result")
        if not query_result:
            return "Error: no successful query result yet. Call query_gold_tool with valid SQL first."
        data = pd.DataFrame(query_result["rows"])
        if x not in data.columns or y not in data.columns:
            return (
                f"Error: x={x!r} or y={y!r} is not a column in the query result. "
                f"Available columns are exactly: {list(data.columns)}. "
                "Retry with x/y matching those column names exactly (use the SELECT alias, e.g. `AS total_sales`)."
            )
        try:
            chart_html = generate_chart_from_spec(data, chart_type, x, y, title)
        except Exception as exc:
            return f"Error generating chart: {exc}. Available columns: {list(data.columns)}."
        state["chart_html"] = chart_html
        return json.dumps({"chart_generated": True})

    @tool
    def html_report_tool() -> str:
        """Assemble and write the final HTML report using the query results and chart."""
        query_result = state.get("query_result") or _query_gold(gold_files)
        chart_html = state.get("chart_html", "")
        html = _build_html_report(business_intent, query_result, chart_html, run_id)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = REPORTS_DIR / f"report_{run_id[:8]}.html"
        out_file.write_text(html, encoding="utf-8")
        state["report_path"] = str(out_file)
        return json.dumps({"report_path": str(out_file)})

    return [query_gold_tool, chart_spec_tool, html_report_tool], state


def _make_llm():
    from core.llm_factory import make_llm

    return make_llm()


def generate_report(gold_files: list[str], business_intent: str, run_id: str, task_description: str) -> str:
    from langgraph.prebuilt import create_react_agent

    trace = AgentTrace("reporter", run_id).set_input(gold_files=gold_files, business_intent=business_intent)
    audit = AuditLogger(run_id)
    audit.log("reporter", "start", task=task_description)

    table_names = [Path(p).stem for p in gold_files]
    human_task = (
        f"Business intent to answer using ONLY the local Gold Parquet files: {task_description}\n\n"
        f"The only queryable table names are exactly: {table_names} (one per Gold file, no other name exists). "
        "Call query_gold_tool with a DuckDB SQL query using one of those exact table names, giving every "
        "SELECT expression a plain alias (e.g. `SUM(sales) AS total_sales`). "
        "Then call chart_spec_tool using x/y values that exactly match the column names/aliases returned by "
        "query_gold_tool (not the raw SQL expression), then call html_report_tool. "
        "Only use the tools provided to you; do not invent or call any other tool, and do not search the internet."
    )

    tools, state = _make_reporter_tools(gold_files, business_intent, run_id)
    try:
        llm = _make_llm()
        agent = create_react_agent(llm, tools)
        result = agent.invoke(
            {"messages": [("system", REPORTER_AGENT_PROMPT), ("human", human_task)]},
            config={"recursion_limit": 50},
        )
        trace.extract_from_messages(result.get("messages", []))
    except Exception as exc:
        # The ReAct loop can fail to converge (e.g. GraphRecursionError) on small/free-tier
        # models; fall through to the deterministic fallback below using whatever state
        # (query_result/chart_html) the tools already gathered before it gave up.
        audit.log("reporter", "agent_error", error=str(exc))

    try:
        report_path = state.get("report_path")
        if not report_path:
            query_result = state.get("query_result") or _query_gold(gold_files)
            data = pd.DataFrame(query_result["rows"])
            x = data.columns[0] if len(data.columns) > 0 else "x"
            y = data.columns[1] if len(data.columns) > 1 else x
            chart_html = state.get("chart_html") or generate_chart_from_spec(
                data, "bar", x, y, business_intent or "Report"
            )
            html = _build_html_report(business_intent, query_result, chart_html, run_id)
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            out_file = REPORTS_DIR / f"report_{run_id[:8]}.html"
            out_file.write_text(html, encoding="utf-8")
            report_path = str(out_file)

        trace.set_output(report_path=report_path).complete()
        audit.log("reporter", "complete", report_path=report_path)
        return str(report_path)
    except Exception as exc:
        trace.fail(str(exc))
        audit.log("reporter", "failed", error=str(exc))
        raise


@dataclass
class ReportResult:
	report_file: Path
	summary: dict[str, object]


class ReportingAgent:
	def build_report(
		self,
		source_file: Path,
		bronze_profile: dict[str, object],
		silver_frame: pd.DataFrame,
		gold_frame: pd.DataFrame,
		reports_dir: Path,
	) -> ReportResult:
		summary = {
			"source_file": str(source_file),
			"bronze_rows": bronze_profile.get("rows", 0),
			"silver_rows": int(len(silver_frame)),
			"gold_rows": int(len(gold_frame)),
			"silver_columns": list(silver_frame.columns),
			"gold_columns": list(gold_frame.columns),
			"total_net_sales": float(gold_frame["net_sales"].sum()) if not gold_frame.empty else 0.0,
		}

		reports_dir.mkdir(parents=True, exist_ok=True)
		report_file = reports_dir / f"{source_file.stem}_report.json"
		report_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
		return ReportResult(report_file=report_file, summary=summary)
