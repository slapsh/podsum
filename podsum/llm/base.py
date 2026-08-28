"""Common chat interface plus JSON extraction shared by the backends."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..errors import BackendError

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class ChatResult:
    text: str
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


class ChatClient(Protocol):
    name: str
    model: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult: ...

    def context_limit(self) -> int: ...


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks that reasoning models leave in the content field."""
    return _THINK_BLOCK.sub("", text).strip()


def parse_json_object(text: str, what: str = "model") -> dict[str, Any]:
    """Best-effort JSON extraction from a chat response.

    Even in JSON mode, models wrap output in fences or prepend a sentence often
    enough that failing hard here would abort a summary that is 95% usable.
    """
    cleaned = strip_reasoning(text)
    candidates = [cleaned]
    if match := _FENCE.search(cleaned):
        candidates.insert(0, match.group(1))
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    raise BackendError(
        f"{what} did not return valid JSON.",
        hint=f"First 200 characters were: {cleaned[:200]!r}",
    )
