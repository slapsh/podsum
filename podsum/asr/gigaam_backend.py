"""Russian-first CPU transcription with GigaAM v3 RNN-T via the `gigastt` wheel.

Whisper turbo does not keep up with real time on CPU; GigaAM does, at
comparable Russian accuracy and a ~225 MB footprint. The trade-off is that the
engine returns a flat word stream, so segments are reconstructed from pauses.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..audio import probe
from ..config import Config
from ..errors import ConfigError, MissingDependency
from .base import ProgressCB, Segment, Transcript

# A pause longer than this, or a span longer than the cap, starts a new segment.
PAUSE_SPLIT_SECONDS = 0.8
MAX_SEGMENT_SECONDS = 14.0


def group_words(words, pause: float = PAUSE_SPLIT_SECONDS) -> list[Segment]:
    """Fold a word stream into utterance-sized segments on pause boundaries."""
    segments: list[Segment] = []
    buffer: list[str] = []
    start = end = 0.0
    for word in words:
        text = (getattr(word, "text", "") or "").strip()
        if not text:
            continue
        w_start = float(getattr(word, "start_s", 0.0))
        w_end = float(getattr(word, "end_s", w_start))
        if not buffer:
            start, end = w_start, w_end
            buffer = [text]
            continue
        too_long = w_end - start > MAX_SEGMENT_SECONDS
        if w_start - end > pause or too_long:
            segments.append(Segment(start=start, end=end, text=" ".join(buffer)))
            start, end, buffer = w_start, w_end, [text]
        else:
            buffer.append(text)
            end = w_end
    if buffer:
        segments.append(Segment(start=start, end=end, text=" ".join(buffer)))
    return segments


class GigaAMASR:
    """Wraps the side-loaded GigaAM engine.

    `--asr-model` carries the model *directory* for this backend (the weights
    are not downloaded automatically), falling back to $GIGASTT_MODEL_DIR.
    """

    name = "gigaam"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        candidate = cfg.asr_model
        if candidate in ("", "large-v3-turbo", "large-v3"):
            candidate = os.environ.get("GIGASTT_MODEL_DIR", "")
        if not candidate:
            raise ConfigError(
                "GigaAM needs a local model directory.",
                hint=(
                    "Download the GigaAM v3 weights, then pass "
                    "`--asr-model /path/to/gigastt/models` or set GIGASTT_MODEL_DIR."
                ),
            )
        self.model_dir = Path(candidate).expanduser()

    def transcribe(
        self, audio_path: Path, on_progress: ProgressCB | None = None
    ) -> Transcript:
        try:
            import gigastt_uniffi as gigastt
        except ImportError as exc:
            raise MissingDependency(
                "gigastt is not installed.",
                hint="Install the CPU Russian extra: `pip install 'podsum[gigaam]'`.",
            ) from exc
        if not self.model_dir.is_dir():
            raise ConfigError(f"GigaAM model directory not found: {self.model_dir}")

        total = probe(audio_path).duration
        engine = gigastt.Engine(str(self.model_dir))
        result = engine.transcribe_file(str(audio_path))
        segments = group_words(getattr(result, "words", []) or [])
        if not segments and getattr(result, "text", ""):
            segments = [Segment(start=0.0, end=total, text=result.text.strip())]
        if on_progress and total:
            on_progress(total, total)
        return Transcript(
            segments=segments,
            language="ru",
            duration=total or float(getattr(result, "duration_s", 0.0)),
            backend=self.name,
            model=f"gigaam-v3 ({self.model_dir.name})",
            source=str(audio_path),
        )
