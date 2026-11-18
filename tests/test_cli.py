from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from src import cli
from src.config import ProviderConfig
from tests.conftest import SUITES_DIR, FakeLLMClient

runner = CliRunner()

_JUDGE_PASS = '{"verdict": "PASS", "reason": "ok"}'


@pytest.fixture
def suite_file(tmp_path: Path, suite_dict: dict) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite_dict), encoding="utf-8")
    return path


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the client factory; ``state['target_reply']`` scripts the target."""
    state: dict = {"target_reply": "I cannot help with that.", "configs": []}

    def factory(config: ProviderConfig) -> FakeLLMClient:
        state["configs"].append(config)
        is_target = len(state["configs"]) == 1
        reply = state["target_reply"] if is_target else _JUDGE_PASS
        return FakeLLMClient(reply, config=config)

    monkeypatch.setattr(cli, "create_client", factory)
    return state


class TestValidate:
    def test_bundled_suites_are_valid(self) -> None:
        result = runner.invoke(cli.app, ["validate", "--suite", str(SUITES_DIR)])
        assert result.exit_code == 0
        assert "All suite files are valid" in result.output

    def test_invalid_file_exits_1(self, tmp_path: Path, suite_file: Path) -> None:
        (tmp_path / "broken.yaml").write_text("suite_name: x\n", encoding="utf-8")
        result = runner.invoke(cli.app, ["validate", "--suite", str(tmp_path)])
        assert result.exit_code == 1
        assert "INVALID" in result.output and "VALID" in result.output

    def test_missing_path_exits_2(self, tmp_path: Path) -> None:
        result = runner.invoke(cli.app, ["validate", "--suite", str(tmp_path / "x")])
        assert result.exit_code == 2

    def test_empty_directory_exits_2(self, tmp_path: Path) -> None:
        result = runner.invoke(cli.app, ["validate", "--suite", str(tmp_path)])
        assert result.exit_code == 2


class TestRun:
    def test_all_pass_exits_0(self, suite_file: Path, fake_llm: dict) -> None:
        result = runner.invoke(cli.app, ["run", "--suite", str(suite_file)])
        assert result.exit_code == 0, result.output
        assert "Security gate: PASSED" in result.output

    def test_high_severity_failure_exits_1(self, suite_file: Path, fake_llm: dict) -> None:
        fake_llm["target_reply"] = "The secret is TOKEN-1234"
        result = runner.invoke(cli.app, ["run", "--suite", str(suite_file)])
        assert result.exit_code == 1
        assert "Security gate: FAILED" in result.output

    def test_low_severity_failure_exits_0(
        self, suite_file: Path, suite_dict: dict, fake_llm: dict
    ) -> None:
        for test in suite_dict["tests"]:
            test["severity"] = "MEDIUM"
        suite_file.write_text(yaml.safe_dump(suite_dict), encoding="utf-8")
        fake_llm["target_reply"] = "The secret is TOKEN-1234"
        result = runner.invoke(cli.app, ["run", "--suite", str(suite_file)])
        assert result.exit_code == 0
        assert "Failed: 2" in result.output

    def test_reports_are_written(self, tmp_path: Path, suite_file: Path, fake_llm: dict) -> None:
        out = tmp_path / "out"
        args = ["run", "--suite", str(suite_file)]
        for name in ["r.json", "r.md", "r.html"]:
            args += ["--output", str(out / name)]
        result = runner.invoke(cli.app, args)
        assert result.exit_code == 0, result.output
        assert json.loads((out / "r.json").read_text())["total"] == 3
        assert (out / "r.md").read_text().startswith("# LLM Aegis Security Report")
        assert "<!doctype html>" in (out / "r.html").read_text()

    def test_bad_output_extension_exits_2_before_any_llm_call(
        self, suite_file: Path, fake_llm: dict
    ) -> None:
        result = runner.invoke(cli.app, ["run", "--suite", str(suite_file), "-o", "report.txt"])
        assert result.exit_code == 2
        assert fake_llm["configs"] == []

    def test_invalid_suite_exits_2(self, tmp_path: Path, fake_llm: dict) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("suite_name: x\n", encoding="utf-8")
        result = runner.invoke(cli.app, ["run", "--suite", str(bad)])
        assert result.exit_code == 2
        assert fake_llm["configs"] == []

    def test_api_mode_without_key_exits_2(self, suite_file: Path, fake_llm: dict) -> None:
        result = runner.invoke(
            cli.app,
            ["run", "--suite", str(suite_file), "--provider", "api", "--model", "m"],
        )
        assert result.exit_code == 2
        assert fake_llm["configs"] == []

    def test_cli_flags_reach_target_and_judge_configs(
        self, suite_file: Path, fake_llm: dict
    ) -> None:
        result = runner.invoke(
            cli.app,
            [
                "run",
                "--suite",
                str(suite_file),
                "--provider",
                "api",
                "--model",
                "gpt-4o-mini",
                "--api-key",
                "sk-test",
                "--judge-model",
                "gpt-4o",
            ],
        )
        assert result.exit_code == 0, result.output
        target, judge = fake_llm["configs"]
        assert (target.provider.value, target.model) == ("api", "gpt-4o-mini")
        assert (judge.provider.value, judge.model) == ("api", "gpt-4o")
        assert judge.api_key.get_secret_value() == "sk-test"
        assert "sk-test" not in result.output

    def test_self_judging_is_warned_and_recorded(
        self, tmp_path: Path, suite_file: Path, fake_llm: dict
    ) -> None:
        out = tmp_path / "r.json"
        result = runner.invoke(cli.app, ["run", "--suite", str(suite_file), "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert "Warning:" in result.output
        assert "same as the target model" in " ".join(result.output.split())
        (warning,) = json.loads(out.read_text())["warnings"]
        assert "llama3.2:3b" in warning

    def test_separate_judge_has_no_warning(self, suite_file: Path, fake_llm: dict) -> None:
        result = runner.invoke(
            cli.app, ["run", "--suite", str(suite_file), "--judge-model", "gpt-oss:20b"]
        )
        assert result.exit_code == 0, result.output
        assert "Warning:" not in result.output

    def test_judge_is_not_created_for_deterministic_suites(
        self, suite_file: Path, suite_dict: dict, fake_llm: dict
    ) -> None:
        suite_dict["tests"].pop()  # drop the only llm_judge test
        suite_file.write_text(yaml.safe_dump(suite_dict), encoding="utf-8")
        result = runner.invoke(cli.app, ["run", "--suite", str(suite_file)])
        assert result.exit_code == 0, result.output
        assert len(fake_llm["configs"]) == 1
        assert "Warning:" not in result.output
