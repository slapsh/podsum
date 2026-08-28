"""Chat backends used for summarization."""

from __future__ import annotations

from ..config import Config
from ..errors import ConfigError
from .base import ChatClient, ChatResult

__all__ = ["ChatClient", "ChatResult", "build_llm"]


def build_llm(cfg: Config) -> ChatClient:
    if cfg.llm_backend == "openrouter":
        from .openrouter import OpenRouterChat

        return OpenRouterChat(cfg)
    if cfg.llm_backend == "ollama":
        from .ollama import OllamaChat

        return OllamaChat(cfg)
    if cfg.llm_backend == "stub":
        from .stub import StubChat

        return StubChat(cfg)
    raise ConfigError(f"Unknown LLM backend {cfg.llm_backend!r}.")
