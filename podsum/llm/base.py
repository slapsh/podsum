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


_DANGLING_KEY = re.compile(r',?\s*"[^"]*"\s*:\s*$')


def close_truncated_json(text: str) -> str | None:
    """Repair JSON that was cut off when the model hit its token limit.

    Discards the partial value at the end and closes every container still
    open. A summary missing its last chapter beats losing the whole run, which
    for a long episode is minutes of local inference.
    """
    start = text.find("{")
    if start == -1:
        return None
    body = text[start:]

    stack: list[str] = []
    in_string = False
    escaped = False
    string_start = -1
    for index, char in enumerate(body):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            string_start = index
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack:
                stack.pop()

    if in_string and string_start != -1:
        body = body[:string_start]
    if not stack:
        return None

    body = body.rstrip().rstrip(",")
    body = _DANGLING_KEY.sub("", body).rstrip().rstrip(",")
    return body + "".join(reversed(stack))


def parse_json_object(text: str, what: str = "model") -> dict[str, Any]:
    """Best-effort JSON extraction from a chat response.

    Even in JSON mode, models wrap output in fences, prepend a sentence, or run
    out of tokens mid-object often enough that failing hard here would abort a
    summary that is mostly usable.
    """
    cleaned = strip_reasoning(text)
    candidates = [cleaned]
    if match := _FENCE.search(cleaned):
        candidates.insert(0, match.group(1))
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start : end + 1])
    if repaired := close_truncated_json(cleaned):
        candidates.append(repaired)
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    raise BackendError(
        f"{what} did not return valid JSON.",
        hint=(
            "The answer may have been cut off — raise the output allowance with a "
            "larger --num-ctx, or use a model with a bigger context window. "
            f"First 200 characters were: {cleaned[:200]!r}"
        ),
    )
