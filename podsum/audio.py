"""Audio preparation: probing, normalization, and silence-aware chunking.

Everything here shells out to ffmpeg/ffprobe, which handles every podcast
container we care about (mp3, m4a/aac, opus, ogg, wav, flac, and mp4/mkv video).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .errors import MissingDependency, PodsumError

SAMPLE_RATE = 16_000
SILENCE_NOISE_DB = -32
SILENCE_MIN_DURATION = 0.45

# Formats ffmpeg produces for us, mapped to the OpenRouter `input_audio.format`
# string and the encoder settings that keep speech intelligible while small.
_ENCODERS = {
    "wav": ["-c:a", "pcm_s16le"],
    "opus": ["-c:a", "libopus", "-b:a", "24k", "-application", "voip"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "64k"],
    "flac": ["-c:a", "flac"],
}


@dataclass
class AudioInfo:
    path: Path
    duration: float
    codec: str
    sample_rate: int
    channels: int
    size_bytes: int


@dataclass
class AudioChunk:
    """A slice of the episode plus the offset needed to restore absolute time."""

    path: Path
    offset: float
    duration: float
    index: int


def ensure_ffmpeg() -> None:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise MissingDependency(
            f"{' and '.join(missing)} not found on PATH.",
            hint=(
                "Install it: `apt install ffmpeg` (Debian/Ubuntu), "
                "`brew install ffmpeg` (macOS), or `choco install ffmpeg` (Windows)."
            ),
        )


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
        raise PodsumError(
            f"{Path(cmd[0]).name} failed: {' / '.join(tail) or 'no output'}"
        )
    return proc


def probe(path: Path) -> AudioInfo:
    """Read duration and stream layout without decoding the whole file."""
    ensure_ffmpeg()
    path = Path(path)
    if not path.is_file():
        raise PodsumError(f"Audio file not found: {path}")
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise PodsumError(
            f"No audio stream in {path.name}.",
            hint="Pass a podcast audio file (mp3, m4a, opus, wav, flac) or a video with sound.",
        )
    stream = streams[0]
    duration = float(data.get("format", {}).get("duration") or 0.0)
    return AudioInfo(
        path=path,
        duration=duration,
        codec=stream.get("codec_name", "unknown"),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        size_bytes=path.stat().st_size,
    )


def normalize(
    src: Path,
    dest: Path,
    fmt: str = "wav",
    sample_rate: int = SAMPLE_RATE,
    start: float | None = None,
    duration: float | None = None,
) -> Path:
    """Transcode to mono `sample_rate` audio, optionally extracting one span.

    Speech models all resample to 16 kHz mono internally, so doing it once up
    front avoids repeating the work per chunk and shrinks cloud uploads.
    """
    ensure_ffmpeg()
    if fmt not in _ENCODERS:
        raise PodsumError(
            f"Unsupported intermediate format {fmt!r}.",
            hint=f"Choose one of: {', '.join(_ENCODERS)}.",
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sample_rate)]
    cmd += _ENCODERS[fmt]
    cmd += [str(dest)]
    _run(cmd)
    return dest


_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silences(
    path: Path,
    noise_db: int = SILENCE_NOISE_DB,
    min_duration: float = SILENCE_MIN_DURATION,
) -> list[tuple[float, float]]:
    """Return (start, end) pairs for every silent stretch ffmpeg can find."""
    ensure_ffmpeg()
    # Not check=True: a failure here only costs us silence-aligned boundaries,
    # and falling back to fixed-length cuts beats aborting the episode.
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_duration}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    silences: list[tuple[float, float]] = []
    pending: float | None = None
    for line in proc.stderr.splitlines():
        if match := _SILENCE_START.search(line):
            pending = float(match.group(1))
        elif match := _SILENCE_END.search(line):
            end = float(match.group(1))
            start = pending if pending is not None else end
            silences.append((max(0.0, start), end))
            pending = None
    return silences


def plan_cut_points(
    duration: float,
    silences: list[tuple[float, float]],
    target: float,
    tolerance: float | None = None,
    min_tail: float | None = None,
) -> list[float]:
    """Pick chunk boundaries at ~`target` seconds, snapped to nearby silence.

    Falls back to a hard cut when no silence is within `tolerance`; a mid-word
    cut costs one garbled word, whereas an oversized chunk can exceed a
    provider's upload limit outright.
    """
    if duration <= target:
        return []
    tolerance = tolerance if tolerance is not None else max(20.0, target * 0.2)
    # Never leave a sliver of a final chunk, but scale the rule to the target so
    # short targets still split.
    min_tail = min_tail if min_tail is not None else min(20.0, target * 0.25)
    midpoints = sorted((s + e) / 2 for s, e in silences)
    cuts: list[float] = []
    position = target
    while position < duration - min_tail:
        window = [
            m for m in midpoints if abs(m - position) <= tolerance and m > (cuts[-1] if cuts else 0.0)
        ]
        cut = min(window, key=lambda m: abs(m - position)) if window else position
        if cuts and cut - cuts[-1] < 1.0:
            cut = position
        cuts.append(round(cut, 3))
        position = cut + target
    return cuts


def split_audio(
    src: Path,
    out_dir: Path,
    target_seconds: float,
    fmt: str = "wav",
    duration: float | None = None,
    overlap: float = 0.0,
) -> list[AudioChunk]:
    """Cut the episode into chunks, each tagged with its absolute start offset."""
    info_duration = duration if duration is not None else probe(src).duration
    cuts = plan_cut_points(info_duration, detect_silences(src), target_seconds)
    bounds = [0.0, *cuts, info_duration]
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[AudioChunk] = []
    for index, (start, end) in enumerate(zip(bounds, bounds[1:])):
        # Overlap only extends the tail, so offsets stay exact.
        span_end = min(info_duration, end + overlap) if index < len(bounds) - 2 else end
        dest = out_dir / f"chunk_{index:03d}.{fmt}"
        normalize(src, dest, fmt=fmt, start=start, duration=span_end - start)
        chunks.append(
            AudioChunk(path=dest, offset=start, duration=span_end - start, index=index)
        )
    return chunks


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}ч {minutes:02d}м {secs:02d}с"
    return f"{minutes}м {secs:02d}с"
