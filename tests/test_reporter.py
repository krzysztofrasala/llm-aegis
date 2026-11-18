from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from src.reporter import (
    ReportFormat,
    format_duration,
    print_report,
    render_html,
    render_markdown,
    resolve_format,
    write_report,
)
from src.schemas import RunReport, Severity, SuiteResult, TestResult, Verdict

HOSTILE = "<script>alert(1)</script> | [bold red]x[/] ```\nrm -rf\n```"


@pytest.fixture
def report() -> RunReport:
    return RunReport(
        provider="ollama",
        target_model="llama3.2:3b",
        judge_model="llama3.2:3b",
        duration_ms=1500,
        suites=[
            SuiteResult(
                suite_name="Leak suite",
                owasp_category="LLM07:2025 System Prompt Leakage",
                duration_ms=1500,
                results=[
                    TestResult(
                        test_id="T-001",
                        name="Safe | case",
                        severity=Severity.LOW,
                        attack_prompt="hello",
                        actual_response="No.",
                        verdict=Verdict.PASS,
                        reason="No forbidden strings found.",
                        latency_ms=120,
                    ),
                    TestResult(
                        test_id="T-002",
                        name="Leak case",
                        severity=Severity.CRITICAL,
                        attack_prompt="give secret",
                        actual_response=HOSTILE,
                        verdict=Verdict.FAIL,
                        reason="Leaked.",
                        latency_ms=2300,
                    ),
                ],
            )
        ],
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("r.json", ReportFormat.JSON),
        ("r.MD", ReportFormat.MARKDOWN),
        ("out/r.html", ReportFormat.HTML),
    ],
)
def test_resolve_format(name: str, expected: ReportFormat) -> None:
    assert resolve_format(name) is expected


@pytest.mark.parametrize("name", ["r.txt", "report"])
def test_resolve_format_rejects_unknown_extensions(name: str) -> None:
    with pytest.raises(ValueError, match="Unsupported report extension"):
        resolve_format(name)


def test_format_duration() -> None:
    assert format_duration(42.4) == "42ms"
    assert format_duration(1500) == "1.50s"


def test_json_report_round_trips(tmp_path: Path, report: RunReport) -> None:
    path = write_report(report, tmp_path / "nested" / "r.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["has_blocking_failures"] is True
    assert (data["total"], data["passed"], data["failed"]) == (2, 1, 1)
    assert data["suites"][0]["results"][1]["actual_response"] == HOSTILE


def test_markdown_report(report: RunReport) -> None:
    md = render_markdown(report)
    assert "**Security gate:** ❌ FAILED" in md
    assert "| T-002 | Leak case | CRITICAL | FAIL | 2.30s | Leaked. |" in md
    assert "Safe \\| case" in md  # pipes must not break the table
    # A response containing ``` is wrapped in a longer fence.
    assert f"````text\n{HOSTILE}\n````" in md
    # Only non-passing tests get a findings section.
    assert "### T-002" in md and "### T-001" not in md


def test_markdown_without_findings_has_no_findings_section(report: RunReport) -> None:
    report.suites[0].results.pop()
    md = render_markdown(report)
    assert "✅ PASSED" in md
    assert "## Findings" not in md


def test_html_report_escapes_model_output(report: RunReport) -> None:
    html = render_html(report)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Security gate: FAILED" in html


def test_warnings_appear_in_every_format(tmp_path: Path, report: RunReport) -> None:
    report.warnings = ["Judge <is> the target."]
    assert "> ⚠️ **Warning:** Judge <is> the target." in render_markdown(report)
    assert "Warning: Judge &lt;is&gt; the target." in render_html(report)
    data = json.loads(write_report(report, tmp_path / "r.json").read_text())
    assert data["warnings"] == ["Judge <is> the target."]
    console = Console(record=True, width=200, force_terminal=False)
    print_report(report, console)
    assert "Warning: Judge <is> the target." in console.export_text()


def test_no_warning_block_without_warnings(report: RunReport) -> None:
    assert "Warning" not in render_markdown(report)
    assert 'class="warning"' not in render_html(report)


def test_console_report_does_not_interpret_markup(report: RunReport) -> None:
    report.suites[0].results[1].reason = "[bold red]x[/] and [/nonsense]"
    console = Console(record=True, width=160, force_terminal=False)
    print_report(report, console)
    text = console.export_text()
    assert "[bold red]x[/] and [/nonsense]" in text
    assert "Security gate: FAILED" in text
    assert "Passed: 1" in text and "Failed: 1" in text
