"""Summarization through OpenRouter's chat completions endpoint."""

from __future__ import annotations

from typing import Any

import httpx

from ..config import Config
from ..errors import BackendError, ConfigError
from ..http import json_body, request_with_retry
from .base import ChatResult, strip_reasoning


class OpenRouterChat:
    name = "openrouter"

    def __init__(self, cfg: Config, client: httpx.Client | None = None) -> None:
        if not cfg.openrouter_api_key and client is None:
            raise ConfigError(
                "OPENROUTER_API_KEY is required for cloud summarization.",
                hint="Set the key, or run summarization on Ollama with --llm ollama.",
            )
        self.cfg = cfg
        self.model = cfg.llm_model
        self._client = client
        self._owns_client = client is None
        self._context_limit: int | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            headers = {
                "Authorization": f"Bearer {self.cfg.openrouter_api_key}",
                "HTTP-Referer": "https://github.com/podsum",
                "X-Title": "podsum",
            }
            headers.update(self.cfg.extra_headers)
            self._client = httpx.Client(
                base_url=self.cfg.openrouter_base_url,
                timeout=httpx.Timeout(self.cfg.timeout_seconds, connect=30.0),
                headers=headers,
            )
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens or self.cfg.max_output_tokens,
            "usage": {"include": True},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = request_with_retry(
            self._get_client(),
            "POST",
            "/chat/completions",
            what="OpenRouter chat",
            json=payload,
        )
        data = json_body(response, "OpenRouter chat")
        choices = data.get("choices") or []
        if not choices:
            raise BackendError("OpenRouter chat returned no choices.")
        message = choices[0].get("message") or {}
        text = strip_reasoning(message.get("content") or "")
        if not text:
            raise BackendError(
                "OpenRouter chat returned empty content.",
                hint=(
                    f"finish_reason={choices[0].get('finish_reason')!r}. "
                    "A reasoning model may have spent the whole budget thinking; "
                    "raise --max-output-tokens or pick another model."
                ),
            )
        return ChatResult(text=text, model=data.get("model", self.model), usage=data.get("usage") or {})

    def context_limit(self) -> int:
        """Ask the catalog how much context this slug actually has."""
        if self._context_limit is not None:
            return self._context_limit
        fallback = self.cfg.context_budget("openrouter")
        try:
            response = self._get_client().get("/models", timeout=30.0)
            models = response.json().get("data", [])
            for entry in models:
                if entry.get("id") == self.model:
                    self._context_limit = int(entry.get("context_length") or fallback)
                    return self._context_limit
        except Exception:
            pass
        self._context_limit = fallback
        return fallback

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None


def list_models(cfg: Config, transcription: bool = False) -> list[dict[str, Any]]:
    """Fetch the live catalog. Works without a key; the endpoint is public."""
    url = f"{cfg.openrouter_base_url}/models"
    params = {"output_modalities": "transcription"} if transcription else None
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json().get("data", [])
