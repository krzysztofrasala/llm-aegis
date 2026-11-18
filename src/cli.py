"""Typer CLI entry point: ``python -m src.cli [run|validate]``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from src.config import ProviderType, get_settings
from src.evaluator import Evaluator
from src.providers import create_client
from src.reporter import print_report, resolve_format, verdict_badge, write_report
from src.runner import Runner
from src.schemas import (
    AssertionType,
    SuiteLoadError,
    TestCase,
    TestResult,
    TestSuite,
    discover_suite_files,
    load_suite,
)

EXIT_OK = 0
EXIT_SECURITY_GATE_FAILED = 1
EXIT_USAGE_ERROR = 2

SELF_JUDGE_WARNING = (
    "The judge model is the same as the target model ({model}). A model grading "
    "its own output is unreliable and can miss leaks; use a separate, stronger "
    "judge via --judge-model or JUDGE_MODEL."
)

app = typer.Typer(
    name="llm-aegis",
    help="Automated LLM security testing & red teaming (OWASP Top 10 for LLM Applications).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

SuiteOption = Annotated[
    Path,
    typer.Option("--suite", "-s", help="Suite YAML file or a directory of suites."),
]


def _fail(message: str) -> typer.Exit:
    err_console.print(Text.assemble(("Error: ", "bold red"), message))
    return typer.Exit(EXIT_USAGE_ERROR)


def _suite_files(path: Path) -> list[Path]:
    if not path.exists():
        raise _fail(f"Suite path does not exist: {path}")
    files = discover_suite_files(path)
    if not files:
        raise _fail(f"No .yaml/.yml suite files found in: {path}")
    return files


@app.command()
def run(
    suite: SuiteOption = Path("suites"),
    provider: Annotated[
        ProviderType | None,
        typer.Option(help="Target provider: local 'ollama' or cloud 'api'. [default: ollama]"),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Target model (system under test).")] = None,
    api_key: Annotated[
        str | None,
        typer.Option(help="API key for 'api' mode (prefer LLM_API_KEY in .env)."),
    ] = None,
    base_url: Annotated[str | None, typer.Option(help="OpenAI-compatible base URL.")] = None,
    judge_provider: Annotated[
        ProviderType | None,
        typer.Option(help="Judge provider. [default: same as target]"),
    ] = None,
    judge_model: Annotated[
        str | None, typer.Option(help="Judge model. [default: same as target]")
    ] = None,
    judge_api_key: Annotated[str | None, typer.Option(help="API key for the judge.")] = None,
    judge_base_url: Annotated[str | None, typer.Option(help="Base URL for the judge.")] = None,
    output: Annotated[
        list[Path] | None,
        typer.Option(
            "--output",
            "-o",
            help="Report file (.json, .md or .html). Repeat for several formats.",
        ),
    ] = None,
) -> None:
    """Run security suites against a target model.

    Exits with code 1 if any HIGH or CRITICAL test does not pass.
    """
    outputs = output or []
    # Validate cheap inputs first so a long scan is never wasted on a typo.
    for path in outputs:
        try:
            resolve_format(path)
        except ValueError as exc:
            raise _fail(str(exc)) from exc

    suites: list[TestSuite] = []
    for file in _suite_files(suite):
        try:
            suites.append(load_suite(file))
        except SuiteLoadError as exc:
            raise _fail(str(exc)) from exc

    needs_judge = any(t.assertion.type is AssertionType.LLM_JUDGE for s in suites for t in s.tests)
    try:
        settings = get_settings()
        target_config = settings.target_config(
            provider=provider, model=model, api_key=api_key, base_url=base_url
        )
        # Only resolve judge settings when a suite actually needs a judge.
        judge_config = (
            settings.judge_config(
                provider=judge_provider,
                model=judge_model,
                api_key=judge_api_key,
                base_url=judge_base_url,
                target=target_config,
            )
            if needs_judge
            else None
        )
    except ValidationError as exc:
        details = "; ".join(err["msg"].removeprefix("Value error, ") for err in exc.errors())
        raise _fail(f"Invalid configuration: {details}") from exc

    warnings: list[str] = []
    if judge_config and (judge_config.model, judge_config.base_url) == (
        target_config.model,
        target_config.base_url,
    ):
        warnings.append(SELF_JUDGE_WARNING.format(model=target_config.model))

    target = create_client(target_config)
    judge = create_client(judge_config) if judge_config else None
    runner = Runner(target, Evaluator(judge))

    total = sum(len(s.tests) for s in suites)
    console.print(
        f"[bold]LLM Aegis[/bold] - {len(suites)} suite(s), {total} test(s) against "
        f"[cyan]{target_config.model}[/cyan] ({target_config.provider.value})\n"
    )
    for warning in warnings:
        console.print(Text.assemble(("Warning: ", "bold yellow"), warning, "\n"))

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Starting...", total=total)

        def on_test_start(_: TestSuite, test: TestCase) -> None:
            progress.update(task, description=Text(f"{test.id} {test.name}").markup)

        def on_result(_: TestSuite, result: TestResult) -> None:
            progress.console.print(
                Text.assemble(verdict_badge(result.verdict), f"  {result.test_id}  {result.name}")
            )
            progress.advance(task)

        report = runner.run(
            suites,
            judge_model=judge_config.model if judge_config else "n/a",
            on_test_start=on_test_start,
            on_result=on_result,
        )
    report.warnings = warnings

    console.print()
    print_report(report, console)

    for path in outputs:
        try:
            written = write_report(report, path)
        except OSError as exc:
            raise _fail(f"Cannot write report {path}: {exc.strerror or exc}") from exc
        console.print(f"Report written: [cyan]{written}[/cyan]")

    raise typer.Exit(EXIT_SECURITY_GATE_FAILED if report.has_blocking_failures else EXIT_OK)


@app.command()
def validate(suite: SuiteOption = Path("suites")) -> None:
    """Validate suite YAML files against the schema without calling any LLM."""
    table = Table(header_style="bold cyan", expand=True)
    table.add_column("File", ratio=2, overflow="fold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Details", ratio=3, overflow="fold")

    invalid = 0
    for file in _suite_files(suite):
        try:
            loaded = load_suite(file)
        except SuiteLoadError as exc:
            invalid += 1
            table.add_row(str(file), Text("INVALID", style="bold red"), Text(exc.message))
        else:
            table.add_row(
                str(file),
                Text("VALID", style="bold green"),
                Text(f"{loaded.owasp_category} - {len(loaded.tests)} test(s)"),
            )

    console.print(table)
    if invalid:
        err_console.print(f"[bold red]{invalid} invalid suite file(s).[/bold red]")
        raise typer.Exit(1)
    console.print("[bold green]All suite files are valid.[/bold green]")


if __name__ == "__main__":
    app()
