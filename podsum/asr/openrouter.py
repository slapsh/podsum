"""Cloud transcription through OpenRouter's `/audio/transcriptions` endpoint."""

from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx

from ..audio import probe, split_audio
from ..config import Config
from ..errors import BackendError, ConfigError
from ..glossary import build_prompt
from ..http import json_body, request_with_retry
from .base import ProgressCB, Segment, Transcript

# Multipart is cheaper on bandwidth but capped at 25 MB; base64 JSON handles the
# rest. Chunks stay far below either limit at 16 kHz mono Opus.
MULTIPART_MAX_BYTES = 20 * 1024 * 1024

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def synthesize_segments(text: str, start: float, duration: float) -> list[Segment]:
    """Spread a plain-text result over the chunk's span, proportional to length.

    Some providers ignore `verbose_json` and return only `text`. Sentence-level
    estimates keep the summary's timestamp anchors roughly right instead of
    collapsing an entire ten-minute chunk onto one instant.
    """
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return []
    total_chars = sum(len(s) for s in sentences)
    segments: list[Segment] = []
    cursor = start
    for sentence in sentences:
        span = duration * (len(sentence) / total_chars) if total_chars else 0.0
        segments.append(Segment(start=cursor, end=cursor + span, text=sentence))
        cursor += span
    return segments


class OpenRouterASR:
    """Uploads audio chunk by chunk and stitches the results back together."""

    name = "openrouter"

    def __init__(self, cfg: Config, client: httpx.Client | None = None) -> None:
        if not cfg.openrouter_api_key and client is None:
            raise ConfigError(
                "OPENROUTER_API_KEY is required for cloud transcription.",
                hint="Set the key, or use --profile local to stay on-premise.",
            )
        self.cfg = cfg
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.cfg.openrouter_base_url,
                timeout=httpx.Timeout(self.cfg.timeout_seconds, connect=30.0),
                headers=self._headers(),
            )
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.cfg.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/podsum",
            "X-Title": "podsum",
        }
        headers.update(self.cfg.extra_headers)
        return headers

    def transcribe(
        self, audio_path: Path, on_progress: ProgressCB | None = None
    ) -> Transcript:
        total = probe(audio_path).duration
        fmt = self.cfg.cloud_audio_format
        with tempfile.TemporaryDirectory(prefix="podsum-chunks-") as tmp:
            chunks = split_audio(
                audio_path,
                Path(tmp),
                target_seconds=self.cfg.chunk_seconds,
                fmt=fmt,
                duration=total,
            )
            parts: list[Transcript] = []
            for chunk in chunks:
                parts.append(self._transcribe_chunk(chunk.path, chunk.offset, chunk.duration, fmt))
                if on_progress:
                    on_progress(min(chunk.offset + chunk.duration, total), total)
        merged = Transcript.merge(
            parts,
            language=self.cfg.language,
            backend=self.name,
            model=self.cfg.asr_model,
            source=str(audio_path),
        )
        merged.duration = total or merged.duration
        return merged

    def _transcribe_chunk(
        self, path: Path, offset: float, duration: float, fmt: str
    ) -> Transcript:
        size = path.stat().st_size
        if size <= MULTIPART_MAX_BYTES:
            data = self._post_multipart(path, fmt)
        else:
            data = self._post_base64(path, fmt)
        return self._parse(data, offset, duration)

    def _common_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "model": self.cfg.asr_model,
            "language": self.cfg.language,
            "response_format": "verbose_json",
            "temperature": 0,
        }
        if prompt := build_prompt(self.cfg.glossary_path):
            # Provider passthrough: Groq-backed Whisper accepts a biasing prompt.
            fields["provider"] = {"prompt": prompt}
        return fields

    def _post_multipart(self, path: Path, fmt: str) -> dict[str, Any]:
        fields = self._common_fields()
        data = {k: v for k, v in fields.items() if not isinstance(v, dict)}
        data["timestamp_granularities[]"] = "segment"
        response = request_with_retry(
            self._get_client(),
            "POST",
            "/audio/transcriptions",
            what="OpenRouter transcription",
            files={"file": (path.name, path.read_bytes(), f"audio/{fmt}")},
            data={k: str(v) for k, v in data.items()},
        )
        return json_body(response, "OpenRouter transcription")

    def _post_base64(self, path: Path, fmt: str) -> dict[str, Any]:
        payload = self._common_fields()
        payload["input_audio"] = {
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            "format": fmt,
        }
        payload["timestamp_granularities"] = ["segment"]
        response = request_with_retry(
            self._get_client(),
            "POST",
            "/audio/transcriptions",
            what="OpenRouter transcription",
            json=payload,
        )
        return json_body(response, "OpenRouter transcription")

    def _parse(self, data: dict[str, Any], offset: float, duration: float) -> Transcript:
        raw_segments = data.get("segments") or []
        segments: list[Segment] = []
        for raw in raw_segments:
            text = (raw.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                Segment(
                    start=float(raw.get("start", 0.0)) + offset,
                    end=float(raw.get("end", raw.get("start", 0.0))) + offset,
                    text=text,
                )
            )
        if not segments:
            text = (data.get("text") or "").strip()
            if not text:
                raise BackendError(
                    "OpenRouter returned an empty transcription.",
                    hint="The chunk may be silent, or the model may not support this language.",
                )
            segments = synthesize_segments(text, offset, duration)
        return Transcript(
            segments=segments,
            language=data.get("language") or self.cfg.language,
            duration=offset + duration,
            backend=self.name,
            model=self.cfg.asr_model,
            usage=_usage(data),
        )

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None


def _usage(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    keep = ("cost", "total_tokens", "input_tokens", "seconds", "duration")
    return {k: usage[k] for k in keep if isinstance(usage.get(k), (int, float))}
