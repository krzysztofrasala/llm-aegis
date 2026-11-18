"""Report generation: Rich console output plus JSON, Markdown and HTML exports."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from jinja2 import Environment, select_autoescape
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.schemas import RunReport, Severity, TestResult, Verdict

SEVERITY_STYLES: dict[Severity, str] = {
    Severity.LOW: "dim",
    Severity.MEDIUM: "yellow",
    Severity.HIGH: "bold dark_orange",
    Severity.CRITICAL: "bold white on red",
}
VERDICT_STYLES: dict[Verdict, str] = {
    Verdict.PASS: "bold green",
    Verdict.FAIL: "bold red",
    Verdict.ERROR: "bold magenta",
}


class ReportFormat(StrEnum):
    """Export formats, selected by the output file extension."""

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


_EXTENSIONS: dict[str, ReportFormat] = {
    ".json": ReportFormat.JSON,
    ".md": ReportFormat.MARKDOWN,
    ".markdown": ReportFormat.MARKDOWN,
    ".html": ReportFormat.HTML,
    ".htm": ReportFormat.HTML,
}


def resolve_format(path: str | Path) -> ReportFormat:
    """Map an output path to a report format, or raise ``ValueError``."""
    suffix = Path(path).suffix.lower()
    try:
        return _EXTENSIONS[suffix]
    except KeyError:
        supported = ", ".join(sorted(_EXTENSIONS))
        raise ValueError(
            f"Unsupported report extension {suffix or '(none)'!r}. Use one of: {supported}"
        ) from None


# --------------------------------------------------------------------------- #
# Console
# --------------------------------------------------------------------------- #
def severity_badge(severity: Severity) -> Text:
    return Text(f" {severity.value} ", style=SEVERITY_STYLES[severity])


def verdict_badge(verdict: Verdict) -> Text:
    return Text(verdict.value, style=VERDICT_STYLES[verdict])


def format_duration(ms: float) -> str:
    return f"{ms / 1000:.2f}s" if ms >= 1000 else f"{ms:.0f}ms"


def print_report(report: RunReport, console: Console | None = None) -> None:
    """Render per-suite result tables and the security gate summary."""
    console = console or Console()

    for suite in report.suites:
        table = Table(
            title=f"{suite.suite_name}  [dim]({suite.owasp_category})[/dim]",
            title_justify="left",
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("ID", no_wrap=True)
        table.add_column("Test", ratio=2)
        table.add_column("Severity", no_wrap=True)
        table.add_column("Verdict", no_wrap=True)
        table.add_column("Latency", justify="right", no_wrap=True)
        table.add_column("Reason", ratio=3)
        for result in suite.results:
            table.add_row(
                result.test_id,
                # Text() keeps model/suite-provided strings from being parsed as Rich markup.
                Text(result.name),
                severity_badge(result.severity),
                verdict_badge(result.verdict),
                format_duration(result.latency_ms),
                Text(result.reason),
            )
        console.print(table)
        console.print()

    for warning in report.warnings:
        console.print(Text.assemble(("Warning: ", "bold yellow"), warning))
    if report.warnings:
        console.print()

    gate_failed = report.has_blocking_failures
    summary = Text.assemble(
        ("Target: ", "bold"),
        f"{report.target_model} ({report.provider})   ",
        ("Judge: ", "bold"),
        f"{report.judge_model}\n",
        ("Total: ", "bold"),
        f"{report.total}   ",
        (f"Passed: {report.passed}", VERDICT_STYLES[Verdict.PASS]),
        "   ",
        (f"Failed: {report.failed}", VERDICT_STYLES[Verdict.FAIL]),
        "   ",
        (f"Errors: {report.errors}", VERDICT_STYLES[Verdict.ERROR]),
        f"   Duration: {format_duration(report.duration_ms)}\n",
        ("Security gate: ", "bold"),
        ("FAILED - HIGH/CRITICAL test(s) did not pass", "bold red")
        if gate_failed
        else ("PASSED", "bold green"),
    )
    console.print(
        Panel(
            summary,
            title="LLM Aegis Summary",
            border_style="red" if gate_failed else "green",
        )
    )


# --------------------------------------------------------------------------- #
# File exports
# --------------------------------------------------------------------------- #
def _md_cell(value: object) -> str:
    """Make a value safe for a single Markdown table cell."""
    return " ".join(str(value).split()).replace("|", "\\|")


def _md_fence(value: str) -> str:
    """Wrap text in a code fence longer than any backtick run it contains."""
    longest = max((len(run) for run in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{value}\n{fence}"


def _findings(report: RunReport) -> list[TestResult]:
    return [r for r in report.all_results if r.verdict is not Verdict.PASS]


_MARKDOWN_TEMPLATE = """\
# LLM Aegis Security Report

**Security gate:** {{ "❌ FAILED" if report.has_blocking_failures else "✅ PASSED" }}
{% for warning in report.warnings %}
> ⚠️ **Warning:** {{ warning }}
{% endfor %}
| | |
|---|---|
| Started (UTC) | {{ report.started_at.strftime("%Y-%m-%d %H:%M:%S") }} |
| Provider | {{ report.provider | cell }} |
| Target model | {{ report.target_model | cell }} |
| Judge model | {{ report.judge_model | cell }} |
| Duration | {{ report.duration_ms | duration }} |
| Total / Passed / Failed / Errors | {{ report.total }} / {{ report.passed }} / \
{{ report.failed }} / {{ report.errors }} |
{% for suite in report.suites %}
## {{ suite.suite_name }}

*{{ suite.owasp_category }}* - {{ suite.passed }} passed, {{ suite.failed }} failed, \
{{ suite.errors }} errors in {{ suite.duration_ms | duration }}

| ID | Test | Severity | Verdict | Latency | Reason |
|---|---|---|---|---|---|
{% for r in suite.results -%}
| {{ r.test_id | cell }} | {{ r.name | cell }} | {{ r.severity.value }} | \
{{ r.verdict.value }} | {{ r.latency_ms | duration }} | {{ r.reason | cell }} |
{% endfor %}
{%- endfor %}
{% if findings %}
## Findings

{% for r in findings -%}
### {{ r.test_id }} - {{ r.name }} ({{ r.severity.value }}, {{ r.verdict.value }})

**Reason:** {{ r.reason }}

**Attack prompt:**

{{ r.attack_prompt | fence }}

**Model response:**

{{ (r.actual_response or "(no response)") | fence }}

{% endfor %}
{%- endif %}
"""

_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Aegis Security Report</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --text:#1c2330; --muted:#667085; --line:#e3e6eb;
          --pass:#177245; --fail:#c62828; --error:#8e24aa; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12151b; --card:#1b2029; --text:#e6e9ef; --muted:#98a2b3; --line:#2c3340;
            --pass:#4cc38a; --fail:#ff6b6b; --error:#ce93d8; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:24px 16px; background:var(--bg); color:var(--text);
         font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:1100px; margin:0 auto; }
  h1 { margin:0 0 4px; font-size:24px; } h2 { margin:32px 0 4px; font-size:18px; }
  .muted { color:var(--muted); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:16px; margin-top:16px; }
  .gate { font-weight:700; font-size:17px; }
  .stats { display:flex; flex-wrap:wrap; gap:8px 28px; margin-top:8px; }
  .stats b { display:block; font-size:20px; }
  .scroll { overflow-x:auto; }
  table { width:100%; border-collapse:collapse; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line);
           vertical-align:top; }
  th { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
  .num { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
  .badge { display:inline-block; padding:1px 8px; border-radius:999px; font-size:12px;
           font-weight:700; border:1px solid currentColor; white-space:nowrap; }
  .PASS { color:var(--pass); } .FAIL { color:var(--fail); } .ERROR { color:var(--error); }
  .LOW { color:var(--muted); } .MEDIUM { color:#b7791f; } .HIGH { color:#dd6b20; }
  .CRITICAL { color:#fff; background:var(--fail); border-color:var(--fail); }
  pre { margin:6px 0 0; padding:10px; background:var(--bg); border:1px solid var(--line);
        border-radius:6px; white-space:pre-wrap; overflow-wrap:anywhere; font-size:13px; }
  .warning { margin-top:8px; padding:8px 10px; border-radius:6px; color:#8a5a00;
             background:#fff4d6; border:1px solid #f0c36d; }
  @media (prefers-color-scheme: dark) {
    .warning { color:#ffd27a; background:#3a2e12; border-color:#7a5c1d; }
  }
  details { margin-top:10px; } summary { cursor:pointer; }
</style>
</head>
<body>
<main>
  <h1>LLM Aegis Security Report</h1>
  <div class="muted">{{ report.started_at.strftime("%Y-%m-%d %H:%M:%S") }} UTC &middot;
    target <b>{{ report.target_model }}</b> ({{ report.provider }}) &middot;
    judge <b>{{ report.judge_model }}</b></div>

  <div class="card">
    <div class="gate {{ 'FAIL' if report.has_blocking_failures else 'PASS' }}">
      Security gate: {{ "FAILED - HIGH/CRITICAL test(s) did not pass"
                        if report.has_blocking_failures else "PASSED" }}</div>
    {% for warning in report.warnings %}
    <div class="warning">&#9888; Warning: {{ warning }}</div>
    {% endfor %}
    <div class="stats">
      <div><b>{{ report.total }}</b>Total</div>
      <div class="PASS"><b>{{ report.passed }}</b>Passed</div>
      <div class="FAIL"><b>{{ report.failed }}</b>Failed</div>
      <div class="ERROR"><b>{{ report.errors }}</b>Errors</div>
      <div><b>{{ report.duration_ms | duration }}</b>Duration</div>
    </div>
  </div>

  {% for suite in report.suites %}
  <h2>{{ suite.suite_name }}</h2>
  <div class="muted">{{ suite.owasp_category }} &middot; {{ suite.passed }} passed,
    {{ suite.failed }} failed, {{ suite.errors }} errors in
    {{ suite.duration_ms | duration }}</div>
  <div class="card scroll">
    <table>
      <thead><tr><th>ID</th><th>Test</th><th>Severity</th><th>Verdict</th>
        <th class="num">Latency</th><th>Reason</th></tr></thead>
      <tbody>
      {% for r in suite.results %}
        <tr>
          <td>{{ r.test_id }}</td>
          <td>{{ r.name }}</td>
          <td><span class="badge {{ r.severity.value }}">{{ r.severity.value }}</span></td>
          <td><span class="badge {{ r.verdict.value }}">{{ r.verdict.value }}</span></td>
          <td class="num">{{ r.latency_ms | duration }}</td>
          <td>{{ r.reason }}
            <details><summary class="muted">Prompt &amp; response</summary>
              <pre>{{ r.attack_prompt }}</pre>
              <pre>{{ r.actual_response or "(no response)" }}</pre>
            </details>
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% endfor %}
</main>
</body>
</html>
"""


def _environment(*, autoescape: bool) -> Environment:
    env = Environment(
        # Model output is untrusted: HTML reports must always be escaped.
        autoescape=select_autoescape(default=autoescape, default_for_string=autoescape),
        trim_blocks=False,
        keep_trailing_newline=True,
    )
    env.filters["duration"] = format_duration
    env.filters["cell"] = _md_cell
    env.filters["fence"] = _md_fence
    return env


def render_json(report: RunReport) -> str:
    return report.model_dump_json(indent=2) + "\n"


def render_markdown(report: RunReport) -> str:
    template = _environment(autoescape=False).from_string(_MARKDOWN_TEMPLATE)
    return template.render(report=report, findings=_findings(report))


def render_html(report: RunReport) -> str:
    template = _environment(autoescape=True).from_string(_HTML_TEMPLATE)
    return template.render(report=report)


_RENDERERS = {
    ReportFormat.JSON: render_json,
    ReportFormat.MARKDOWN: render_markdown,
    ReportFormat.HTML: render_html,
}


def write_report(report: RunReport, path: str | Path) -> Path:
    """Write the report in the format implied by the file extension."""
    path = Path(path)
    content = _RENDERERS[resolve_format(path)](report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
