"""Local Whisper backend behaviour, with the model itself faked out."""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from podsum.asr.local_whisper import FasterWhisperASR, resolve_compute_type, resolve_device
from podsum.glossary import build_prompt, load_terms

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


class FakeWhisper:
    """Records how it was called and can pretend a prompt silenced it."""

    def __init__(self, empty_with_prompt: bool = False):
        self.calls: list[dict] = []
        self.empty_with_prompt = empty_with_prompt

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        info = SimpleNamespace(language="ru", duration=6.0)
        if self.empty_with_prompt and kwargs.get("initial_prompt"):
            return iter([]), info
        segments = [
            SimpleNamespace(start=0.0, end=3.0, text=" Привет, это Kubernetes."),
            SimpleNamespace(start=3.0, end=6.0, text="   "),  # whitespace-only
            SimpleNamespace(start=6.0, end=9.0, text=" И Kafka."),
        ]
        return iter(segments), info


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=6:sample_rate=16000", str(path)],
        check=True,
    )
    return path


def backend_with(config, model):
    backend = FasterWhisperASR(config)
    backend._model = model
    return backend


class TestRussianDefaults:
    @needs_ffmpeg
    def test_decoding_options_guard_against_known_failure_modes(self, local_config, clip):
        model = FakeWhisper()
        backend_with(local_config, model).transcribe(clip)
        kwargs = model.calls[0]
        assert kwargs["language"] == "ru"
        assert kwargs["condition_on_previous_text"] is False  # repetition loop guard
        assert kwargs["vad_filter"] is True
        assert "Kubernetes" in kwargs["initial_prompt"]

    @needs_ffmpeg
    def test_blank_segments_are_dropped_and_text_is_trimmed(self, local_config, clip):
        transcript = backend_with(local_config, FakeWhisper()).transcribe(clip)
        assert [s.text for s in transcript.segments] == ["Привет, это Kubernetes.", "И Kafka."]

    @needs_ffmpeg
    def test_a_glossary_that_silences_the_model_is_retried_without_it(self, local_config, clip):
        model = FakeWhisper(empty_with_prompt=True)
        transcript = backend_with(local_config, model).transcribe(clip)
        assert len(model.calls) == 2
        assert model.calls[0]["initial_prompt"]
        assert model.calls[1]["initial_prompt"] is None
        assert transcript.segments
        assert transcript.usage["glossary_prompt"] is False

    @needs_ffmpeg
    def test_progress_reaches_the_full_duration(self, local_config, clip):
        seen: list[tuple[float, float]] = []
        backend_with(local_config, FakeWhisper()).transcribe(clip, lambda d, t: seen.append((d, t)))
        assert seen[-1][0] == pytest.approx(seen[-1][1])
        assert seen[-1][1] == pytest.approx(6.0, abs=0.2)

    @needs_ffmpeg
    def test_metadata_records_how_it_ran(self, local_config, clip):
        transcript = backend_with(local_config, FakeWhisper()).transcribe(clip)
        assert transcript.backend == "local"
        assert transcript.model == local_config.asr_model
        assert transcript.usage["device"] in {"cpu", "cuda"}
        assert transcript.duration == pytest.approx(6.0, abs=0.2)


class TestDeviceResolution:
    def test_explicit_choices_are_respected(self):
        assert resolve_device("cpu") == "cpu"
        assert resolve_compute_type("float32", "cpu") == "float32"

    def test_auto_picks_a_compute_type_that_matches_the_device(self):
        assert resolve_compute_type("auto", "cuda") == "int8_float16"
        assert resolve_compute_type("auto", "cpu") == "int8"

    def test_auto_device_returns_something_usable(self):
        assert resolve_device("auto") in {"cpu", "cuda"}


class TestGlossary:
    def test_default_list_covers_both_scripts(self):
        terms = load_terms()
        assert "Kubernetes" in terms
        assert "техдолг" in terms

    def test_comments_and_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "g.txt"
        path.write_text("# комментарий\n\nKafka\n  Redis  \n", encoding="utf-8")
        assert load_terms(path) == ["Kafka", "Redis"]

    def test_prompt_is_trimmed_to_what_whisper_will_read(self):
        prompt = build_prompt()
        assert len(prompt) <= 851
        assert prompt.startswith("Подкаст об IT")
        assert prompt.endswith(".")

    def test_an_empty_glossary_yields_no_prompt(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("# только комментарий\n", encoding="utf-8")
        assert build_prompt(path) == ""
