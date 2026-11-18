"""Test execution engine: sends attacks to the target model and records verdicts."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from src.evaluator import Evaluator
from src.providers import LLMClient, ProviderError
from src.schemas import (
    RunReport,
    SuiteResult,
    TestCase,
    TestResult,
    TestSuite,
    Verdict,
)

ResultCallback = Callable[[TestSuite, TestResult], None]
TestStartCallback = Callable[[TestSuite, TestCase], None]


class Runner:
    """Runs suites against a target model and evaluates every response."""

    def __init__(self, target: LLMClient, evaluator: Evaluator) -> None:
        self._target = target
        self._evaluator = evaluator

    def run_test(self, suite: TestSuite, test: TestCase) -> TestResult:
        """Execute one test. Provider failures become ERROR results, never exceptions."""
        try:
            completion = self._target.complete(
                test.attack_prompt, system_prompt=suite.target_system_prompt
            )
        except ProviderError as exc:
            return TestResult(
                test_id=test.id,
                name=test.name,
                severity=test.severity,
                attack_prompt=test.attack_prompt,
                verdict=Verdict.ERROR,
                reason=f"Target call failed: {exc}",
                latency_ms=0.0,
            )

        evaluation = self._evaluator.evaluate(test, completion.text)
        return TestResult(
            test_id=test.id,
            name=test.name,
            severity=test.severity,
            attack_prompt=test.attack_prompt,
            actual_response=completion.text,
            verdict=evaluation.verdict,
            reason=evaluation.reason,
            latency_ms=completion.latency_ms,
        )

    def run_suite(
        self,
        suite: TestSuite,
        *,
        on_test_start: TestStartCallback | None = None,
        on_result: ResultCallback | None = None,
    ) -> SuiteResult:
        """Execute every test in a suite sequentially."""
        started = time.perf_counter()
        results: list[TestResult] = []
        for test in suite.tests:
            if on_test_start:
                on_test_start(suite, test)
            result = self.run_test(suite, test)
            results.append(result)
            if on_result:
                on_result(suite, result)
        return SuiteResult(
            suite_name=suite.suite_name,
            owasp_category=suite.owasp_category,
            results=results,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def run(
        self,
        suites: Sequence[TestSuite],
        *,
        judge_model: str,
        on_test_start: TestStartCallback | None = None,
        on_result: ResultCallback | None = None,
    ) -> RunReport:
        """Execute all suites and assemble the final report."""
        report = RunReport(
            provider=self._target.config.provider.value,
            target_model=self._target.model,
            judge_model=judge_model,
        )
        started = time.perf_counter()
        for suite in suites:
            report.suites.append(
                self.run_suite(suite, on_test_start=on_test_start, on_result=on_result)
            )
        report.duration_ms = (time.perf_counter() - started) * 1000
        return report
