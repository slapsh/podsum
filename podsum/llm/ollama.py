"""Summarization on a local Ollama server."""

from __future__ import annotations

from typing import Any

import httpx

from ..config import Config
from ..errors import BackendError
from ..http import json_body, request_with_retry
from .base import ChatResult, strip_reasoning


class OllamaChat:
    """Chat against `POST /api/chat` on a self-hosted Ollama instance.

    `options.num_ctx` is always sent: Ollama defaults to a few thousand tokens
    and silently truncates the prompt, which would quietly summarize the first
    ten minutes of an episode and drop the rest.
    """

    name = "ollama"

    def __init__(self, cfg: Config, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self.model = cfg.llm_model
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.cfg.ollama_base_url,
                timeout=httpx.Timeout(self.cfg.timeout_seconds, connect=10.0),
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
            "stream": False,
            "options": {
                "num_ctx": self.cfg.num_ctx,
                "temperature": self.cfg.temperature,
                "num_predict": max_tokens or self.cfg.max_output_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        try:
            response = request_with_retry(
                self._get_client(),
                "POST",
                "/api/chat",
                what="Ollama chat",
                json=payload,
            )
        except BackendError as exc:
            raise BackendError(
                exc.message,
                hint=exc.hint
                or (
                    f"Is Ollama running at {self.cfg.ollama_base_url}? "
                    f"Start it with `ollama serve` and pull the model: "
                    f"`ollama pull {self.model}`."
                ),
            ) from exc
        data = json_body(response, "Ollama chat")
        message = data.get("message") or {}
        text = strip_reasoning(message.get("content") or "")
        if not text:
            raise BackendError(
                "Ollama returned empty content.",
                hint=f"Check that `{self.model}` is pulled and fits in memory.",
            )
        usage = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
            "num_ctx": self.cfg.num_ctx,
        }
        return ChatResult(
            text=text,
            model=data.get("model", self.model),
            usage={k: v for k, v in usage.items() if v is not None},
        )

    def context_limit(self) -> int:
        return self.cfg.num_ctx

    def is_available(self) -> tuple[bool, str]:
        """Used by `podsum doctor` to explain an unreachable server."""
        try:
            response = self._get_client().get("/api/tags", timeout=5.0)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
            if self.model not in names:
                return False, (
                    f"Ollama is up but `{self.model}` is not pulled. "
                    f"Run: ollama pull {self.model}"
                )
            return True, f"Ollama is up with {self.model} available."
        except Exception as exc:
            return False, f"Cannot reach Ollama at {self.cfg.ollama_base_url}: {exc}"

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None
