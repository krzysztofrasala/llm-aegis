"""Pydantic models for YAML test suites and test results."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    computed_field,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Severity(StrEnum):
    """Impact level of a failed test, ordered from least to most severe."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def blocks_pipeline(self) -> bool:
        """Failures at this severity must produce a non-zero CLI exit code."""
        return self in {Severity.HIGH, Severity.CRITICAL}


class AssertionType(StrEnum):
    """Supported evaluation strategies."""

    NOT_CONTAINS = "not_contains"
    REGEX_NOT_MATCH = "regex_not_match"
    LLM_JUDGE = "llm_judge"


class Verdict(StrEnum):
    """Outcome of a single test."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"  # The test could not be evaluated (provider failure, etc.)


# --------------------------------------------------------------------------- #
# Suite definition (YAML)
# --------------------------------------------------------------------------- #
class Assertion(BaseModel):
    """How a model response is judged. Fields required depend on ``type``.

    For ``llm_judge`` assertions, ``patterns`` is optional: when given, the
    strings are checked case-insensitively *before* the judge is called and any
    hit is an immediate FAIL. This keeps known secrets from slipping past a weak
    or fooled judge model.
    """

    model_config = ConfigDict(extra="forbid")

    type: AssertionType
    patterns: list[NonEmptyStr] = Field(default_factory=list)
    judge_criteria: str | None = None

    @model_validator(mode="after")
    def _check_fields_for_type(self) -> Assertion:
        if self.type is AssertionType.LLM_JUDGE:
            if not (self.judge_criteria and self.judge_criteria.strip()):
                raise ValueError("'judge_criteria' is required for 'llm_judge' assertions")
            return self

        if not self.patterns:
            raise ValueError(f"'patterns' must be non-empty for '{self.type.value}' assertions")
        if self.type is AssertionType.REGEX_NOT_MATCH:
            for pattern in self.patterns:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return self


class TestCase(BaseModel):
    """A single adversarial test."""

    __test__ = False  # Prevent pytest from collecting this model as a test class.

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    name: NonEmptyStr
    severity: Severity
    attack_prompt: NonEmptyStr
    assertion: Assertion


class TestSuite(BaseModel):
    """A collection of tests sharing one simulated target application."""

    __test__ = False

    model_config = ConfigDict(extra="forbid")

    suite_name: NonEmptyStr
    owasp_category: NonEmptyStr
    target_system_prompt: NonEmptyStr
    tests: Annotated[list[TestCase], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique_test_ids(self) -> TestSuite:
        seen: set[str] = set()
        duplicates = {t.id for t in self.tests if t.id in seen or seen.add(t.id)}
        if duplicates:
            raise ValueError(f"duplicate test ids: {', '.join(sorted(duplicates))}")
        return self


class SuiteLoadError(Exception):
    """Raised when a suite file cannot be read, parsed or validated."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def load_suite(path: str | Path) -> TestSuite:
    """Load and validate a single YAML suite file."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SuiteLoadError(path, f"cannot read file ({exc.strerror or exc})") from exc
    except yaml.YAMLError as exc:
        raise SuiteLoadError(path, f"invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise SuiteLoadError(path, "top-level YAML value must be a mapping")
    try:
        return TestSuite.model_validate(raw)
    except ValidationError as exc:
        raise SuiteLoadError(path, _format_validation_error(exc)) from exc


def discover_suite_files(path: str | Path) -> list[Path]:
    """Return suite files for a file or a directory (``*.yaml`` / ``*.yml``, sorted)."""
    path = Path(path)
    if path.is_dir():
        return sorted([*path.glob("*.yaml"), *path.glob("*.yml")])
    return [path]


def _format_validation_error(exc: ValidationError) -> str:
    lines = [
        f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()
    ]
    return "; ".join(lines)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
class EvaluationResult(BaseModel):
    """Verdict produced by the evaluator for one response."""

    verdict: Verdict
    reason: str


class TestResult(BaseModel):
    """Recorded outcome of one executed test."""

    __test__ = False

    test_id: str
    name: str
    severity: Severity
    attack_prompt: str
    actual_response: str = ""
    verdict: Verdict
    reason: str
    latency_ms: float = Field(ge=0)

    @property
    def is_blocking_failure(self) -> bool:
        return self.verdict is not Verdict.PASS and self.severity.blocks_pipeline


class SuiteResult(BaseModel):
    """Aggregated results for one suite run."""

    suite_name: str
    owasp_category: str
    results: list[TestResult] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> int:
        return sum(r.verdict is Verdict.PASS for r in self.results)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed(self) -> int:
        return sum(r.verdict is Verdict.FAIL for r in self.results)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def errors(self) -> int:
        return sum(r.verdict is Verdict.ERROR for r in self.results)


class RunReport(BaseModel):
    """Top-level report covering every suite in a single CLI invocation."""

    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provider: str
    target_model: str
    judge_model: str
    suites: list[SuiteResult] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0, ge=0)
    warnings: list[str] = Field(default_factory=list)

    @property
    def all_results(self) -> list[TestResult]:
        return [r for s in self.suites for r in s.results]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> int:
        return len(self.all_results)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> int:
        return sum(s.passed for s in self.suites)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed(self) -> int:
        return sum(s.failed for s in self.suites)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def errors(self) -> int:
        return sum(s.errors for s in self.suites)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_blocking_failures(self) -> bool:
        """True if any HIGH/CRITICAL test did not pass (drives CLI exit code 1)."""
        return any(r.is_blocking_failure for r in self.all_results)
