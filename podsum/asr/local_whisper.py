"""On-premise transcription with faster-whisper (CTranslate2)."""

from __future__ import annotations

from pathlib import Path

from ..audio import probe
from ..config import Config
from ..errors import MissingDependency
from ..glossary import build_prompt
from .base import ProgressCB, Segment, Transcript


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:  # pragma: no cover - depends on local hardware
        pass
    return "cpu"


def resolve_compute_type(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    return "int8_float16" if device == "cuda" else "int8"


class FasterWhisperASR:
    """Whisper-family models run locally; no data leaves the machine.

    Russian-specific defaults matter more than the model choice here:
    `condition_on_previous_text=False` prevents the repetition loop that
    corrupts hour-long Russian audio, and the glossary prompt keeps English
    product names in Latin script.
    """

    name = "local"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.compute_type = resolve_compute_type(cfg.compute_type, self.device)
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MissingDependency(
                "faster-whisper is not installed.",
                hint=(
                    "Install the on-premise extra: `pip install 'podsum[local]'` "
                    "(or `uv sync --extra local`)."
                ),
            ) from exc
        self._model = WhisperModel(
            self.cfg.asr_model, device=self.device, compute_type=self.compute_type
        )
        return self._model

    def _decode(
        self,
        model,
        audio_path: Path,
        prompt: str | None,
        total: float,
        on_progress: ProgressCB | None,
    ) -> tuple[list[Segment], object]:
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            initial_prompt=prompt,
            word_timestamps=False,
        )
        segments: list[Segment] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                segments.append(Segment(start=seg.start, end=seg.end, text=text))
            if on_progress:
                on_progress(min(seg.end, total or seg.end), total or seg.end)
        return segments, info

    def transcribe(
        self, audio_path: Path, on_progress: ProgressCB | None = None
    ) -> Transcript:
        model = self._load()
        total = probe(audio_path).duration
        prompt = build_prompt(self.cfg.glossary_path) or None
        segments, info = self._decode(model, audio_path, prompt, total, on_progress)
        used_glossary = bool(prompt)
        if not segments and prompt:
            # A long biasing prompt can derail smaller Whisper checkpoints into
            # emitting nothing at all. A transcript with phonetic product names
            # beats an empty one, so drop the hint and decode again.
            segments, info = self._decode(model, audio_path, None, total, on_progress)
            used_glossary = False
        if on_progress and total:
            on_progress(total, total)
        return Transcript(
            segments=segments,
            language=getattr(info, "language", self.cfg.language) or self.cfg.language,
            duration=total or (segments[-1].end if segments else 0.0),
            backend=self.name,
            model=self.cfg.asr_model,
            source=str(audio_path),
            usage={
                "device": self.device,
                "compute_type": self.compute_type,
                "glossary_prompt": used_glossary,
            },
        )
