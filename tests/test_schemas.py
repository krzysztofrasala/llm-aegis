from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.schemas import (
    Assertion,
    RunReport,
    Severity,
    SuiteLoadError,
    SuiteResult,
    TestResult,
    TestSuite,
    Verdict,
    discover_suite_files,
    load_suite,
)
from tests.conftest import SUITES_DIR


def _result(severity: Severity, verdict: Verdict) -> TestResult:
    return TestResult(
        test_id="X",
        name="x",
        severity=severity,
        attack_prompt="a",
        verdict=verdict,
        reason="r",
        latency_ms=1,
    )


class TestAssertion:
    def test_llm_judge_requires_criteria(self) -> None:
        with pytest.raises(ValidationError, match="judge_criteria"):
            Assertion(type="llm_judge")
        with pytest.raises(ValidationError, match="judge_criteria"):
            Assertion(type="llm_judge", judge_criteria="   ")

    @pytest.mark.parametrize("kind", ["not_contains", "regex_not_match"])
    def test_deterministic_requires_patterns(self, kind: str) -> None:
        with pytest.raises(ValidationError, match="patterns"):
            Assertion(type=kind)

    def test_invalid_regex_is_rejected_at_load_time(self) -> None:
        with pytest.raises(ValidationError, match="invalid regex"):
            Assertion(type="regex_not_match", patterns=["("])

    def test_not_contains_patterns_are_not_compiled_as_regex(self) -> None:
        assert Assertion(type="not_contains", patterns=["("]).patterns == ["("]

    def test_llm_judge_accepts_optional_pre_check_patterns(self) -> None:
        assertion = Assertion(type="llm_judge", judge_criteria="c", patterns=["SECRET"])
        assert assertion.patterns == ["SECRET"]
        assert Assertion(type="llm_judge", judge_criteria="c").patterns == []

    def test_unknown_type_and_extra_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Assertion(type="contains", patterns=["x"])
        with pytest.raises(ValidationError):
            Assertion(type="not_contains", patterns=["x"], pattern="typo")


class TestSuiteModel:
    def test_valid_suite(self, suite: TestSuite) -> None:
        assert [t.id for t in suite.tests] == ["T-001", "T-002", "T-003"]
        assert suite.tests[0].severity is Severity.HIGH

    def test_duplicate_ids_are_rejected(self, suite_dict: dict) -> None:
        suite_dict["tests"][1]["id"] = "T-001"
        with pytest.raises(ValidationError, match="duplicate test ids: T-001"):
            TestSuite.model_validate(suite_dict)

    def test_empty_tests_are_rejected(self, suite_dict: dict) -> None:
        suite_dict["tests"] = []
        with pytest.raises(ValidationError):
            TestSuite.model_validate(suite_dict)

    def test_invalid_severity_is_rejected(self, suite_dict: dict) -> None:
        suite_dict["tests"][0]["severity"] = "URGENT"
        with pytest.raises(ValidationError):
            TestSuite.model_validate(suite_dict)

    def test_blank_attack_prompt_is_rejected(self, suite_dict: dict) -> None:
        suite_dict["tests"][0]["attack_prompt"] = "  "
        with pytest.raises(ValidationError):
            TestSuite.model_validate(suite_dict)


class TestLoading:
    def test_load_valid_file(self, tmp_path: Path, suite_dict: dict) -> None:
        path = tmp_path / "s.yaml"
        path.write_text(yaml.safe_dump(suite_dict), encoding="utf-8")
        assert load_suite(path).suite_name == "Demo suite"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteLoadError, match="cannot read file"):
            load_suite(tmp_path / "nope.yaml")

    def test_broken_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "s.yaml"
        path.write_text("suite_name: [unclosed", encoding="utf-8")
        with pytest.raises(SuiteLoadError, match="invalid YAML"):
            load_suite(path)

    def test_non_mapping_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "s.yaml"
        path.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(SuiteLoadError, match="must be a mapping"):
            load_suite(path)

    def test_schema_error_names_the_field(self, tmp_path: Path, suite_dict: dict) -> None:
        del suite_dict["tests"][0]["assertion"]["patterns"]
        path = tmp_path / "s.yaml"
        path.write_text(yaml.safe_dump(suite_dict), encoding="utf-8")
        with pytest.raises(SuiteLoadError, match=r"tests\.0\.assertion") as excinfo:
            load_suite(path)
        assert excinfo.value.path == path

    def test_discover_directory_is_sorted_and_filtered(self, tmp_path: Path) -> None:
        for name in ["b.yaml", "a.yml", "notes.txt"]:
            (tmp_path / name).write_text("x", encoding="utf-8")
        assert [p.name for p in discover_suite_files(tmp_path)] == ["a.yml", "b.yaml"]

    def test_discover_single_file(self, tmp_path: Path) -> None:
        assert discover_suite_files(tmp_path / "one.yaml") == [tmp_path / "one.yaml"]

    def test_bundled_suites_are_valid(self) -> None:
        files = discover_suite_files(SUITES_DIR)
        assert files, "repository must ship at least one suite"
        for file in files:
            load_suite(file)


class TestResults:
    @pytest.mark.parametrize(
        ("severity", "verdict", "blocking"),
        [
            (Severity.CRITICAL, Verdict.FAIL, True),
            (Severity.HIGH, Verdict.FAIL, True),
            (Severity.HIGH, Verdict.ERROR, True),
            (Severity.HIGH, Verdict.PASS, False),
            (Severity.MEDIUM, Verdict.FAIL, False),
            (Severity.LOW, Verdict.ERROR, False),
        ],
    )
    def test_blocking_failure_matrix(
        self, severity: Severity, verdict: Verdict, blocking: bool
    ) -> None:
        assert _result(severity, verdict).is_blocking_failure is blocking

    def test_report_aggregates(self) -> None:
        report = RunReport(
            provider="ollama",
            target_model="m",
            judge_model="j",
            suites=[
                SuiteResult(
                    suite_name="a",
                    owasp_category="c",
                    results=[
                        _result(Severity.LOW, Verdict.PASS),
                        _result(Severity.MEDIUM, Verdict.FAIL),
                    ],
                ),
                SuiteResult(
                    suite_name="b",
                    owasp_category="c",
                    results=[_result(Severity.LOW, Verdict.ERROR)],
                ),
            ],
        )
        assert (report.total, report.passed, report.failed, report.errors) == (
            3,
            1,
            1,
            1,
        )
        assert report.has_blocking_failures is False
        report.suites[1].results.append(_result(Severity.CRITICAL, Verdict.FAIL))
        assert report.has_blocking_failures is True
        assert report.model_dump()["has_blocking_failures"] is True
