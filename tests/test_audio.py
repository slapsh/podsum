from __future__ import annotations

import shutil
import subprocess

import pytest

from podsum.audio import (
    detect_silences,
    format_duration,
    normalize,
    plan_cut_points,
    probe,
    split_audio,
)
from podsum.errors import PodsumError

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is required for audio tests"
)


def make_audio(path, spec="sine=frequency=440:duration=6", rate=44100):
    """Render a synthetic clip so tests do not ship binary fixtures."""
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"{spec}:sample_rate={rate}",
            "-ac", "2", str(path),
        ],
        check=True,
    )
    return path


class TestCutPoints:
    def test_short_audio_is_not_split(self):
        assert plan_cut_points(300.0, [(100.0, 101.0)], target=600.0) == []

    def test_cut_snaps_to_nearest_silence(self):
        silences = [(595.0, 599.0), (700.0, 701.0)]
        cuts = plan_cut_points(1500.0, silences, target=600.0, tolerance=30.0)
        assert cuts[0] == pytest.approx(597.0)

    def test_hard_cut_when_no_silence_is_close(self):
        cuts = plan_cut_points(1500.0, [(100.0, 101.0)], target=600.0, tolerance=30.0)
        assert cuts[0] == pytest.approx(600.0)

    def test_cuts_advance_and_leave_no_tiny_tail(self):
        silences = [(float(t), float(t) + 1) for t in range(50, 3600, 50)]
        cuts = plan_cut_points(3600.0, silences, target=600.0)
        assert cuts == sorted(cuts)
        assert all(b - a > 1.0 for a, b in zip(cuts, cuts[1:]))
        assert 3600.0 - cuts[-1] >= 20.0


class TestProbeAndNormalize:
    def test_probe_reports_duration_and_layout(self, tmp_path):
        src = make_audio(tmp_path / "clip.wav")
        info = probe(src)
        assert info.duration == pytest.approx(6.0, abs=0.2)
        assert info.channels == 2
        assert info.sample_rate == 44100

    def test_probe_rejects_a_file_without_audio(self, tmp_path):
        broken = tmp_path / "not-audio.mp3"
        broken.write_bytes(b"definitely not audio")
        with pytest.raises(PodsumError):
            probe(broken)

    def test_normalize_produces_16k_mono(self, tmp_path):
        src = make_audio(tmp_path / "clip.mp3")
        dest = normalize(src, tmp_path / "norm.wav")
        info = probe(dest)
        assert info.channels == 1
        assert info.sample_rate == 16_000

    def test_normalize_extracts_a_span(self, tmp_path):
        src = make_audio(tmp_path / "clip.wav")
        dest = normalize(src, tmp_path / "span.wav", start=2.0, duration=2.0)
        assert probe(dest).duration == pytest.approx(2.0, abs=0.2)

    def test_opus_is_much_smaller_than_the_source(self, tmp_path):
        src = make_audio(tmp_path / "clip.wav")
        dest = normalize(src, tmp_path / "clip.opus", fmt="opus")
        assert dest.stat().st_size < src.stat().st_size / 4


class TestSplitting:
    def test_detects_the_silent_middle(self, tmp_path):
        loud = make_audio(tmp_path / "loud.wav", "sine=frequency=440:duration=3")
        quiet = make_audio(tmp_path / "quiet.wav", "anullsrc=r=44100:cl=stereo:duration=2")
        listing = tmp_path / "list.txt"
        listing.write_text(f"file '{loud}'\nfile '{quiet}'\nfile '{loud}'\n")
        joined = tmp_path / "joined.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
             "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)],
            check=True,
        )
        silences = detect_silences(joined)
        assert any(start >= 2.5 and end <= 5.5 for start, end in silences)

    def test_chunks_cover_the_episode_with_absolute_offsets(self, tmp_path):
        src = make_audio(tmp_path / "long.wav", "sine=frequency=440:duration=12")
        chunks = split_audio(src, tmp_path / "chunks", target_seconds=4.0)
        assert len(chunks) > 1
        assert chunks[0].offset == 0.0
        assert [c.index for c in chunks] == list(range(len(chunks)))
        assert sum(c.duration for c in chunks) == pytest.approx(12.0, abs=0.5)
        for previous, following in zip(chunks, chunks[1:]):
            assert following.offset == pytest.approx(previous.offset + previous.duration, abs=0.01)
        for chunk in chunks:
            assert chunk.path.is_file()


def test_format_duration_reads_naturally():
    assert format_duration(45) == "0м 45с"
    assert format_duration(3725) == "1ч 02м 05с"
