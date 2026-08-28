"""Errors that carry an actionable message for the CLI to print."""

from __future__ import annotations


class PodsumError(Exception):
    """Base class for failures we can explain to the user."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class MissingDependency(PodsumError):
    """An optional package or external binary is not installed."""


class ConfigError(PodsumError):
    """The requested combination of backends/credentials cannot work."""


class BackendError(PodsumError):
    """A transcription or LLM backend returned an unusable response."""
