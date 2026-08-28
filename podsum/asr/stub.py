"""Deterministic ASR backend used by tests and `--asr stub` dry runs.

Reads a canned transcript instead of touching a model, so the chunking,
summarization, and rendering stages can be exercised without GPUs or API keys.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import Config
from ..errors import ConfigError
from .base import ProgressCB, Segment, Transcript

FIXTURE_ENV = "PODSUM_STUB_TRANSCRIPT"

_FALLBACK = [
    (0.0, 8.0, "Всем привет, это подкаст про разработку, сегодня говорим про наблюдаемость."),
    (8.0, 18.0, "Мы выкатили OpenTelemetry в продакшен и собрали все грабли, какие можно."),
    (18.0, 30.0, "Главный вывод: сначала метрики и логи, трейсинг подключайте последним."),
]


class StubASR:
    name = "stub"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.fixture = os.environ.get(FIXTURE_ENV)

    def transcribe(
        self, audio_path: Path, on_progress: ProgressCB | None = None
    ) -> Transcript:
        if self.fixture:
            path = Path(self.fixture)
            if not path.is_file():
                raise ConfigError(f"{FIXTURE_ENV} points at a missing file: {path}")
            transcript = Transcript.load(path)
            transcript.backend = self.name
            transcript.source = str(audio_path)
            if on_progress:
                on_progress(transcript.duration, transcript.duration)
            return transcript
        segments = [Segment(start=s, end=e, text=t) for s, e, t in _FALLBACK]
        if on_progress:
            on_progress(segments[-1].end, segments[-1].end)
        return Transcript(
            segments=segments,
            language="ru",
            duration=segments[-1].end,
            backend=self.name,
            model="stub",
            source=str(audio_path),
        )
