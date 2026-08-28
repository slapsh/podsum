from __future__ import annotations

import json
from pathlib import Path

import pytest

from podsum.asr.base import Segment, Transcript
from podsum.config import Config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Retry backoff is real seconds; tests assert on attempts, not on waiting."""
    import podsum.http

    monkeypatch.setattr(podsum.http.time, "sleep", lambda _: None)


@pytest.fixture
def cloud_config() -> Config:
    cfg = Config.from_profile("cloud")
    cfg.openrouter_api_key = "test-key"
    return cfg


@pytest.fixture
def local_config() -> Config:
    return Config.from_profile("local")


@pytest.fixture
def transcript() -> Transcript:
    data = json.loads((FIXTURES / "transcript_ru.json").read_text(encoding="utf-8"))
    return Transcript.from_dict(data)


@pytest.fixture
def long_transcript() -> Transcript:
    """A transcript big enough to force map-reduce at a small context budget."""
    segments = [
        Segment(
            start=float(i * 10),
            end=float(i * 10 + 9),
            text=(
                f"Фрагмент {i}: обсуждаем миграцию сервиса на Kubernetes, "
                "нагрузочное тестирование и то, как команда справлялась с техдолгом."
            ),
        )
        for i in range(120)
    ]
    return Transcript(
        segments=segments,
        duration=1200.0,
        backend="stub",
        model="stub",
        source="episode.mp3",
    )
