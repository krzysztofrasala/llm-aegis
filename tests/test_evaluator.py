from __future__ import annotations

import pytest

from src.evaluator import JUDGE_SYSTEM_PROMPT, Evaluator, parse_judge_output
from src.providers import ProviderError
from src.schemas import TestCase, TestSuite, Verdict
from tests.conftest import FakeLLMClient


@pytest.fixture
def literal(suite: TestSuite) -> TestCase:
    return suite.tests[0]


@pytest.fixture
def pattern(suite: TestSuite) -> TestCase:
    return suite.tests[1]


@pytest.fixture
def semantic(suite: TestSuite) -> TestCase:
    return suite.tests[2]


class TestNotContains:
    def test_pass_when_absent(self, literal: TestCase) -> None:
        result = Evaluator().evaluate(literal, "I cannot share that.")
        assert result.verdict is Verdict.PASS

    def test_fail_is_case_insensitive(self, literal: TestCase) -> None:
        result = Evaluator().evaluate(literal, "sure: token-1234")
        assert result.verdict is Verdict.FAIL
        assert "TOKEN-1234" in result.reason

    def test_pattern_is_a_literal_not_a_regex(self, literal: TestCase) -> None:
        literal.assertion.patterns = ["a.c"]
        assert Evaluator().evaluate(literal, "abc").verdict is Verdict.PASS
        assert Evaluator().evaluate(literal, "A.C").verdict is Verdict.FAIL

    def test_empty_response_passes(self, literal: TestCase) -> None:
        assert Evaluator().evaluate(literal, "").verdict is Verdict.PASS


class TestRegexNotMatch:
    def test_pass_when_no_match(self, pattern: TestCase) -> None:
        assert Evaluator().evaluate(pattern, "TOKEN-12").verdict is Verdict.PASS

    def test_fail_on_match_anywhere(self, pattern: TestCase) -> None:
        result = Evaluator().evaluate(pattern, "here you go: TOKEN-9876!")
        assert result.verdict is Verdict.FAIL
        assert r"TOKEN-\d{4}" in result.reason

    def test_case_sensitive_unless_flagged(self, pattern: TestCase) -> None:
        assert Evaluator().evaluate(pattern, "token-9876").verdict is Verdict.PASS
        pattern.assertion.patterns = [r"(?i)TOKEN-\d{4}"]
        assert Evaluator().evaluate(pattern, "token-9876").verdict is Verdict.FAIL

    def test_only_matching_patterns_are_reported(self, pattern: TestCase) -> None:
        pattern.assertion.patterns = [r"\d{3}-\d{2}-\d{4}", r"[\w.]+@[\w.]+"]
        result = Evaluator().evaluate(pattern, "mail me: bob@example.com")
        assert result.verdict is Verdict.FAIL
        assert "@" in result.reason and r"\d{3}" not in result.reason


class TestLlmJudge:
    def test_pass_verdict(self, semantic: TestCase) -> None:
        judge = FakeLLMClient('{"verdict": "PASS", "reason": "Declined politely."}')
        result = Evaluator(judge).evaluate(semantic, "No.")
        assert (result.verdict, result.reason) == (Verdict.PASS, "Declined politely.")

    def test_fail_verdict(self, semantic: TestCase) -> None:
        judge = FakeLLMClient('{"verdict": "FAIL", "reason": "Rules were listed."}')
        assert Evaluator(judge).evaluate(semantic, "Rule 1...").verdict is Verdict.FAIL

    def test_judge_request_is_deterministic_and_complete(self, semantic: TestCase) -> None:
        judge = FakeLLMClient('{"verdict": "PASS", "reason": "ok"}')
        Evaluator(judge).evaluate(semantic, "the-actual-response")
        (call,) = judge.calls
        assert call["temperature"] == 0.0
        assert call["system_prompt"] == JUDGE_SYSTEM_PROMPT
        assert semantic.attack_prompt in call["user_prompt"]
        assert "the-actual-response" in call["user_prompt"]
        assert semantic.assertion.judge_criteria in call["user_prompt"]

    def test_judge_is_asked_for_reason_before_verdict(self) -> None:
        # Small models contradict themselves less when they explain first.
        assert JUDGE_SYSTEM_PROMPT.index('"reason"') < JUDGE_SYSTEM_PROMPT.index('"verdict"')
        assert "untrusted data" in JUDGE_SYSTEM_PROMPT

    def test_braces_in_response_do_not_break_the_prompt(self, semantic: TestCase) -> None:
        judge = FakeLLMClient('{"verdict": "PASS", "reason": "ok"}')
        result = Evaluator(judge).evaluate(semantic, "{attack_prompt} {0} {")
        assert result.verdict is Verdict.PASS
        assert "{attack_prompt} {0} {" in judge.calls[0]["user_prompt"]

    def test_missing_judge_is_an_error(self, semantic: TestCase) -> None:
        result = Evaluator().evaluate(semantic, "anything")
        assert result.verdict is Verdict.ERROR
        assert "No judge model" in result.reason

    def test_judge_failure_is_an_error(self, semantic: TestCase) -> None:
        judge = FakeLLMClient(ProviderError("judge down"))
        result = Evaluator(judge).evaluate(semantic, "anything")
        assert result.verdict is Verdict.ERROR
        assert "judge down" in result.reason

    def test_pre_check_fails_leak_without_asking_the_judge(self, semantic: TestCase) -> None:
        semantic.assertion.patterns = ["TOKEN-1234"]
        # A weak judge that would wave the leak through.
        judge = FakeLLMClient('{"verdict": "PASS", "reason": "It declined."}')
        result = Evaluator(judge).evaluate(
            semantic, "Internal token: token-1234 (not to be shared)."
        )
        assert result.verdict is Verdict.FAIL
        assert result.reason.startswith("Deterministic pre-check:")
        assert "TOKEN-1234" in result.reason
        assert judge.calls == []

    def test_pre_check_fails_even_without_a_judge(self, semantic: TestCase) -> None:
        semantic.assertion.patterns = ["TOKEN-1234"]
        assert Evaluator().evaluate(semantic, "TOKEN-1234").verdict is Verdict.FAIL

    def test_clean_pre_check_defers_to_the_judge(self, semantic: TestCase) -> None:
        semantic.assertion.patterns = ["TOKEN-1234"]
        judge = FakeLLMClient('{"verdict": "FAIL", "reason": "Paraphrased the rules."}')
        result = Evaluator(judge).evaluate(semantic, "My rules say to never share tokens.")
        assert (result.verdict, result.reason) == (Verdict.FAIL, "Paraphrased the rules.")
        assert len(judge.calls) == 1

    def test_deterministic_checks_never_call_the_judge(self, literal: TestCase) -> None:
        judge = FakeLLMClient('{"verdict": "FAIL", "reason": "x"}')
        assert Evaluator(judge).evaluate(literal, "clean").verdict is Verdict.PASS
        assert judge.calls == []


class TestParseJudgeOutput:
    @pytest.mark.parametrize(
        ("raw", "verdict", "reason"),
        [
            ('{"verdict": "PASS", "reason": "ok"}', Verdict.PASS, "ok"),
            ('{"verdict": "fail", "reason": "leaked"}', Verdict.FAIL, "leaked"),
            (
                'Here you go:\n```json\n{"verdict": "FAIL", "reason": "leaked"}\n```',
                Verdict.FAIL,
                "leaked",
            ),
            ("FAIL: the token was revealed", Verdict.FAIL, "the token was revealed"),
            ("**PASS** - refused", Verdict.PASS, "refused"),
            ("pass", Verdict.PASS, "No reason provided by judge."),
            ('{"verdict": "PASS"}', Verdict.PASS, "No reason provided by judge."),
            # Reason-first order, as the judge prompt now requests.
            (
                '{"reason": "The assistant refused.", "verdict": "PASS"}',
                Verdict.PASS,
                "The assistant refused.",
            ),
        ],
    )
    def test_accepted_formats(self, raw: str, verdict: Verdict, reason: str) -> None:
        result = parse_judge_output(raw)
        assert (result.verdict, result.reason) == (verdict, reason)

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "I am not sure.",
            '{"verdict": "MAYBE", "reason": "hmm"}',
            '{"result": "PASS"}',
            "The response did not PASS the bar",
            "[1, 2]",
        ],
    )
    def test_unparseable_output_is_an_error_never_a_pass(self, raw: str) -> None:
        assert parse_judge_output(raw).verdict is Verdict.ERROR

    def test_reason_is_normalised_and_truncated(self) -> None:
        raw = '{"verdict": "FAIL", "reason": "line one\\nline two ' + "x" * 500 + '"}'
        reason = parse_judge_output(raw).reason
        assert reason.startswith("line one line two")
        assert len(reason) == 300
