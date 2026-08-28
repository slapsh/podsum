"""CLI wiring, exercised end to end with offline backends."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest
from typer.testing import CliRunner

from podsum.cli import app
from podsum.config import Config
from podsum.errors import ConfigError

runner = CliRunner()

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


@pytest.fixture
def episode(tmp_path):
    path = tmp_path / "episode.mp3"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=300:duration=5", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_run_writes_transcript_and_summary(tmp_path, episode, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["run", str(episode), "--asr", "stub", "--llm", "stub", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "episode.transcript.json").is_file()
    assert (out / "episode.transcript.txt").is_file()
    summary = (out / "episode.summary.md").read_text(encoding="utf-8")
    assert summary.startswith("#")
    assert "## Кратко" in summary


@needs_ffmpeg
def test_second_run_reuses_the_cached_transcript(tmp_path, episode):
    out = tmp_path / "out"
    args = ["run", str(episode), "--asr", "stub", "--llm", "stub", "--out", str(out)]
    runner.invoke(app, args)
    transcript_path = out / "episode.transcript.json"
    marker = json.loads(transcript_path.read_text(encoding="utf-8"))
    marker["segments"][0]["text"] = "МАРКЕР КЕША"
    transcript_path.write_text(json.dumps(marker, ensure_ascii=False), encoding="utf-8")

    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "МАРКЕР КЕША" in transcript_path.read_text(encoding="utf-8")

    forced = runner.invoke(app, args + ["--no-cache"])
    assert forced.exit_code == 0, forced.output
    assert "МАРКЕР КЕША" not in transcript_path.read_text(encoding="utf-8")


@needs_ffmpeg
def test_transcribe_then_summarize_as_separate_steps(tmp_path, episode):
    out = tmp_path / "out"
    first = runner.invoke(app, ["transcribe", str(episode), "--asr", "stub", "--out", str(out)])
    assert first.exit_code == 0, first.output
    assert not (out / "episode.summary.md").exists()

    second = runner.invoke(
        app,
        ["summarize", str(out / "episode.transcript.json"), "--llm", "stub", "--out", str(out)],
    )
    assert second.exit_code == 0, second.output
    assert (out / "episode.summary.md").is_file()


@needs_ffmpeg
def test_stub_can_replay_a_recorded_transcript(tmp_path, episode, monkeypatch):
    from tests.conftest import FIXTURES

    monkeypatch.setenv("PODSUM_STUB_TRANSCRIPT", str(FIXTURES / "transcript_ru.json"))
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["run", str(episode), "--asr", "stub", "--llm", "stub", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    text = (out / "episode.transcript.txt").read_text(encoding="utf-8")
    assert "монолита на микросервисы" in text


def test_missing_api_key_is_a_clean_error_not_a_traceback(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # keep a developer .env out of the way
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"")
    result = runner.invoke(app, ["run", str(audio), "--profile", "cloud"])
    assert result.exit_code == 1
    assert "OPENROUTER_API_KEY" in result.output
    assert "Traceback" not in result.output


def test_unknown_backend_is_rejected():
    cfg = Config.from_profile("cloud", asr_backend="whisperfish")
    with pytest.raises(ConfigError):
        cfg.validate()


def test_unknown_profile_is_rejected():
    with pytest.raises(ConfigError):
        Config.from_profile("hybrid")


def test_profiles_pick_matching_backends():
    cloud = Config.from_profile("cloud")
    local = Config.from_profile("local")
    assert (cloud.asr_backend, cloud.llm_backend) == ("openrouter", "openrouter")
    assert (local.asr_backend, local.llm_backend) == ("local", "ollama")


def test_dotenv_does_not_override_a_real_environment_variable(tmp_path, monkeypatch):
    from podsum.config import load_dotenv

    monkeypatch.setenv("OPENROUTER_API_KEY", "from-environment")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-file\n")
    monkeypatch.chdir(tmp_path)
    load_dotenv()
    assert Config.from_profile("cloud").openrouter_api_key == "from-environment"


def test_dotenv_is_read_when_the_variable_is_absent(tmp_path, monkeypatch):
    from podsum.config import load_dotenv

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text('OPENROUTER_API_KEY="from-file"\n# comment\n\n')
    monkeypatch.chdir(tmp_path)
    load_dotenv()
    assert Config.from_profile("cloud").openrouter_api_key == "from-file"


def test_doctor_reports_without_failing():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ffmpeg" in result.output


def test_version_prints_the_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "podsum" in result.output
