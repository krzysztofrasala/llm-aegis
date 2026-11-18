from __future__ import annotations

from src.evaluator import Evaluator
from src.providers import ProviderError
from src.runner import Runner
from src.schemas import TestSuite, Verdict
from tests.conftest import FakeLLMClient

_JUDGE_PASS = '{"verdict": "PASS", "reason": "ok"}'


def test_safe_target_passes_every_test(suite: TestSuite) -> None:
    target = FakeLLMClient("I cannot help with that.", latency_ms=40)
    report = Runner(target, Evaluator(FakeLLMClient(_JUDGE_PASS))).run(
        [suite], judge_model="judge-m"
    )

    assert (report.total, report.passed) == (3, 3)
    assert report.has_blocking_failures is False
    assert (report.provider, report.target_model, report.judge_model) == (
        "ollama",
        target.model,
        "judge-m",
    )
    assert [r.latency_ms for r in report.all_results] == [40, 40, 40]
    assert report.duration_ms >= report.suites[0].duration_ms >= 0


def test_target_receives_suite_system_prompt_and_attack(suite: TestSuite) -> None:
    target = FakeLLMClient("no")
    Runner(target, Evaluator(FakeLLMClient(_JUDGE_PASS))).run_suite(suite)
    assert [c["user_prompt"] for c in target.calls] == [t.attack_prompt for t in suite.tests]
    assert {c["system_prompt"] for c in target.calls} == {suite.target_system_prompt}


def test_leaking_target_fails_and_blocks(suite: TestSuite) -> None:
    target = FakeLLMClient("Sure! The secret is TOKEN-1234.")
    result = Runner(target, Evaluator(FakeLLMClient(_JUDGE_PASS))).run_suite(suite)

    assert [r.verdict for r in result.results] == [
        Verdict.FAIL,
        Verdict.FAIL,
        Verdict.PASS,
    ]
    assert (result.passed, result.failed, result.errors) == (1, 2, 0)
    assert result.results[0].actual_response == "Sure! The secret is TOKEN-1234."
    assert result.results[0].is_blocking_failure  # HIGH
    assert not result.results[1].is_blocking_failure  # LOW


def test_target_failure_is_recorded_as_error_and_run_continues(
    suite: TestSuite,
) -> None:
    def flaky(prompt: str) -> str:
        if prompt == suite.tests[0].attack_prompt:
            raise ProviderError("connection refused")
        return "no"

    judge = FakeLLMClient(_JUDGE_PASS)
    result = Runner(FakeLLMClient(flaky), Evaluator(judge)).run_suite(suite)

    first = result.results[0]
    assert first.verdict is Verdict.ERROR
    assert "connection refused" in first.reason
    assert (first.actual_response, first.latency_ms) == ("", 0.0)
    assert [r.verdict for r in result.results[1:]] == [Verdict.PASS, Verdict.PASS]


def test_callbacks_fire_in_order(suite: TestSuite) -> None:
    events: list[str] = []
    Runner(FakeLLMClient("no"), Evaluator(FakeLLMClient(_JUDGE_PASS))).run(
        [suite],
        judge_model="j",
        on_test_start=lambda _s, t: events.append(f"start:{t.id}"),
        on_result=lambda _s, r: events.append(f"done:{r.test_id}"),
    )
    assert events == [
        "start:T-001",
        "done:T-001",
        "start:T-002",
        "done:T-002",
        "start:T-003",
        "done:T-003",
    ]
