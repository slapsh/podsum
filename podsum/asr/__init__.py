"""Speech-to-text backends."""

from __future__ import annotations

from ..config import Config
from ..errors import ConfigError
from .base import ASRBackend, Segment, Transcript

__all__ = ["ASRBackend", "Segment", "Transcript", "build_asr"]


def build_asr(cfg: Config) -> ASRBackend:
    """Instantiate the ASR backend named by the config."""
    if cfg.asr_backend == "openrouter":
        from .openrouter import OpenRouterASR

        return OpenRouterASR(cfg)
    if cfg.asr_backend == "local":
        from .local_whisper import FasterWhisperASR

        return FasterWhisperASR(cfg)
    if cfg.asr_backend == "gigaam":
        from .gigaam_backend import GigaAMASR

        return GigaAMASR(cfg)
    if cfg.asr_backend == "stub":
        from .stub import StubASR

        return StubASR(cfg)
    raise ConfigError(f"Unknown ASR backend {cfg.asr_backend!r}.")
