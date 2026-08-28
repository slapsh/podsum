"""Transcript data model shared by every ASR backend."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

SCHEMA_VERSION = 1

# Called with (seconds_done, seconds_total) so the CLI can draw a progress bar.
ProgressCB = Callable[[float, float], None]


def format_timestamp(seconds: float) -> str:
    """Render seconds as HH:MM:SS, the form used in the summary anchors."""
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class Segment:
    """One utterance-sized span of recognized speech, in absolute episode time."""

    start: float
    end: float
    text: str
    speaker: str | None = None  # reserved for diarization

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)

    def shifted(self, offset: float) -> "Segment":
        return Segment(
            start=self.start + offset,
            end=self.end + offset,
            text=self.text,
            speaker=self.speaker,
        )


@dataclass
class Transcript:
    """A full episode transcript plus the provenance needed to reproduce it."""

    segments: list[Segment]
    language: str = "ru"
    duration: float = 0.0
    backend: str = ""
    model: str = ""
    source: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def timed_text(self) -> str:
        """Plain-text rendering with a leading timestamp per segment."""
        return "\n".join(
            f"[{s.timestamp}] {s.text.strip()}" for s in self.segments if s.text.strip()
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = SCHEMA_VERSION
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transcript":
        payload = {k: v for k, v in data.items() if k != "schema_version"}
        payload["segments"] = [Segment(**s) for s in payload.get("segments", [])]
        return cls(**payload)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "Transcript":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def merge(cls, parts: Iterable["Transcript"], **meta: Any) -> "Transcript":
        """Concatenate per-chunk transcripts that already hold absolute times."""
        segments: list[Segment] = []
        usage: dict[str, Any] = {}
        duration = 0.0
        for part in parts:
            segments.extend(part.segments)
            duration = max(duration, part.duration)
            for key, value in part.usage.items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
        segments.sort(key=lambda s: s.start)
        if segments:
            duration = max(duration, segments[-1].end)
        return cls(segments=segments, duration=duration, usage=usage, **meta)


class ASRBackend(Protocol):
    """Turns an audio file into a Transcript with absolute timestamps."""

    name: str

    def transcribe(
        self, audio_path: Path, on_progress: ProgressCB | None = None
    ) -> Transcript: ...
