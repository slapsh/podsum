"""OpenRouter transcription backend, driven by recorded response shapes."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess

import httpx
import pytest

from podsum.asr.base import Transcript
from podsum.asr.openrouter import OpenRouterASR, synthesize_segments
from podsum.errors import BackendError

VERBOSE_JSON = {
    "text": "Сегодня про Kubernetes.",
    "language": "ru",
    "duration": 30.0,
    "segments": [
        {"id": 0, "start": 0.0, "end": 12.0, "text": " Сегодня разбираем Kubernetes."},
        {"id": 1, "start": 12.0, "end": 25.0, "text": " И заодно Kafka в проде."},
    ],
    "usage": {"cost": 0.0031, "seconds": 30},
}

TEXT_ONLY = {"text": "Первое предложение. Второе предложение! Третье?"}


def mock_client(handler, base_url="https://openrouter.ai/api/v1"):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=base_url)


class TestSynthesizeSegments:
    def test_spreads_sentences_across_the_span(self):
        segments = synthesize_segments("Раз. Два. Три.", start=100.0, duration=30.0)
        assert len(segments) == 3
        assert segments[0].start == 100.0
        assert segments[-1].end == pytest.approx(130.0)
        assert all(a.end == pytest.approx(b.start) for a, b in zip(segments, segments[1:]))

    def test_empty_text_yields_nothing(self):
        assert synthesize_segments("   ", 0.0, 10.0) == []


class TestParsing:
    def test_segments_are_shifted_into_absolute_time(self, cloud_config):
        backend = OpenRouterASR(cloud_config, client=mock_client(lambda r: httpx.Response(200)))
        transcript = backend._parse(VERBOSE_JSON, offset=600.0, duration=30.0)
        assert [s.start for s in transcript.segments] == [600.0, 612.0]
        assert transcript.segments[0].text == "Сегодня разбираем Kubernetes."
        assert transcript.usage["cost"] == pytest.approx(0.0031)

    def test_text_only_response_falls_back_to_estimated_timings(self, cloud_config):
        backend = OpenRouterASR(cloud_config, client=mock_client(lambda r: httpx.Response(200)))
        transcript = backend._parse(TEXT_ONLY, offset=60.0, duration=30.0)
        assert len(transcript.segments) == 3
        assert transcript.segments[0].start == 60.0
        assert transcript.segments[-1].end == pytest.approx(90.0)

    def test_empty_response_is_an_error_not_a_silent_gap(self, cloud_config):
        backend = OpenRouterASR(cloud_config, client=mock_client(lambda r: httpx.Response(200)))
        with pytest.raises(BackendError):
            backend._parse({"text": ""}, offset=0.0, duration=30.0)


class TestRequests:
    def test_small_chunks_are_sent_as_multipart_with_a_language_hint(self, cloud_config, tmp_path):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["content_type"] = request.headers.get("content-type", "")
            seen["body"] = request.content
            return httpx.Response(200, json=VERBOSE_JSON)

        audio = tmp_path / "chunk.opus"
        audio.write_bytes(b"fake audio bytes")
        backend = OpenRouterASR(cloud_config, client=mock_client(handler))
        backend._transcribe_chunk(audio, offset=0.0, duration=30.0, fmt="opus")

        assert seen["url"].endswith("/audio/transcriptions")
        assert "multipart/form-data" in seen["content_type"]
        assert b'name="language"' in seen["body"]
        assert b"ru" in seen["body"]
        assert b"verbose_json" in seen["body"]

    def test_large_chunks_switch_to_base64_json(self, cloud_config, tmp_path, monkeypatch):
        import podsum.asr.openrouter as module

        monkeypatch.setattr(module, "MULTIPART_MAX_BYTES", 4)
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json=VERBOSE_JSON)

        audio = tmp_path / "chunk.opus"
        audio.write_bytes(b"a much larger blob of audio")
        backend = OpenRouterASR(cloud_config, client=mock_client(handler))
        backend._transcribe_chunk(audio, offset=0.0, duration=30.0, fmt="opus")

        payload = captured["payload"]
        assert payload["input_audio"]["format"] == "opus"
        assert base64.b64decode(payload["input_audio"]["data"]) == audio.read_bytes()
        assert payload["model"] == cloud_config.asr_model

    def test_rate_limits_are_retried(self, cloud_config, tmp_path):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(429, json={"error": {"message": "slow down"}})
            return httpx.Response(200, json=VERBOSE_JSON)

        audio = tmp_path / "chunk.opus"
        audio.write_bytes(b"bytes")
        backend = OpenRouterASR(cloud_config, client=mock_client(handler))
        transcript = backend._transcribe_chunk(audio, 0.0, 30.0, "opus")

        assert attempts["n"] == 3
        assert transcript.segments

    def test_authentication_failure_explains_itself(self, cloud_config, tmp_path):
        audio = tmp_path / "chunk.opus"
        audio.write_bytes(b"bytes")
        backend = OpenRouterASR(
            cloud_config,
            client=mock_client(lambda r: httpx.Response(401, json={"error": "bad key"})),
        )
        with pytest.raises(BackendError) as excinfo:
            backend._transcribe_chunk(audio, 0.0, 30.0, "opus")
        assert "OPENROUTER_API_KEY" in (excinfo.value.hint or "")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_full_episode_is_chunked_and_stitched(cloud_config, tmp_path):
    """Every chunk hits the API once and the merged transcript is monotonic."""
    src = tmp_path / "episode.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=12:sample_rate=16000", str(src)],
        check=True,
    )
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "text": "фрагмент",
                "segments": [{"start": 0.0, "end": 3.0, "text": f"фрагмент {calls['n']}"}],
            },
        )

    cloud_config.chunk_seconds = 4
    backend = OpenRouterASR(cloud_config, client=mock_client(handler))
    transcript = backend.transcribe(src)

    assert calls["n"] > 1
    assert isinstance(transcript, Transcript)
    starts = [s.start for s in transcript.segments]
    assert starts == sorted(starts)
    assert starts[-1] > 0  # later chunks were offset, not stacked at zero
    assert transcript.duration == pytest.approx(12.0, abs=0.5)
