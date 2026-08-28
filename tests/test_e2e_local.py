"""Real inference over synthesized Russian speech.

Marked slow: it downloads Whisper `tiny` on first run. Everything else in the
suite is offline, but this is the only test that proves ffmpeg normalization,
the local backend, timestamp handling, and rendering work on actual audio
rather than on fixtures.

Run with: pytest -m slow
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.slow

SENTENCES = [
    "Всем привет, это подкаст про разработку.",
    "Сегодня мы обсуждаем микросервисы и наблюдаемость.",
    "Главный вывод такой: сначала метрики, потом трейсинг.",
]

needs_tools = pytest.mark.skipif(
    shutil.which("espeak-ng") is None or shutil.which("ffmpeg") is None,
    reason="espeak-ng and ffmpeg are required to synthesize the test clip",
)


def synthesize(path, text: str):
    """Render Russian speech with espeak-ng, then mux to MP3 like a real episode.

    The rate, amplitude, and word-gap settings are deliberate: at espeak's
    defaults the output sits close enough to Whisper's no-speech threshold that
    small checkpoints intermittently return nothing at all.
    """
    wav = path.with_suffix(".raw.wav")
    subprocess.run(
        ["espeak-ng", "-v", "ru", "-s", "130", "-p", "45", "-a", "200", "-g", "8",
         "-w", str(wav), text],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
         "-ar", "44100", "-ac", "2", "-b:a", "128k", str(path)],
        check=True,
    )
    return path


def has_cyrillic(text: str) -> bool:
    return any("\u0400" <= ch <= "\u04ff" for ch in text)


@needs_tools
def test_local_whisper_transcribes_russian_audio(tmp_path):
    pytest.importorskip("faster_whisper", reason="install the 'local' extra")

    from podsum.asr.local_whisper import FasterWhisperASR
    from podsum.audio import normalize, probe
    from podsum.config import Config

    episode = synthesize(tmp_path / "episode.mp3", " ".join(SENTENCES))
    info = probe(episode)
    assert info.duration > 5

    normalized = normalize(episode, tmp_path / "episode.wav")
    assert probe(normalized).sample_rate == 16_000
    assert probe(normalized).channels == 1

    cfg = Config.from_profile("local", asr_model="tiny", device="cpu", compute_type="int8")
    progress: list[tuple[float, float]] = []
    transcript = FasterWhisperASR(cfg).transcribe(
        normalized, lambda done, total: progress.append((done, total))
    )

    assert transcript.segments, "recognizer returned nothing"
    assert has_cyrillic(transcript.text), f"expected Russian output, got: {transcript.text!r}"
    assert transcript.language == "ru"
    assert progress and progress[-1][0] == pytest.approx(progress[-1][1])

    starts = [s.start for s in transcript.segments]
    assert starts == sorted(starts)
    assert transcript.segments[-1].end <= info.duration + 1.0


@needs_tools
def test_full_pipeline_from_mp3_to_summary(tmp_path):
    """The whole chain on real audio, with the LLM stage stubbed out."""
    pytest.importorskip("faster_whisper", reason="install the 'local' extra")

    from typer.testing import CliRunner

    from podsum.cli import app

    episode = synthesize(tmp_path / "episode.mp3", " ".join(SENTENCES))
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        ["run", str(episode), "--profile", "local", "--asr-model", "tiny",
         "--device", "cpu", "--llm", "stub", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output

    timed = (out / "episode.transcript.txt").read_text(encoding="utf-8")
    assert timed.startswith("[00:00:")
    assert has_cyrillic(timed)

    summary = (out / "episode.summary.md").read_text(encoding="utf-8")
    assert "## Кратко" in summary
    assert "`tiny` (local)" in summary
