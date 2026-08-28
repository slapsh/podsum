"""Offline chat backend so the pipeline can be exercised without a model."""

from __future__ import annotations

import json

from ..config import Config
from .base import ChatResult


class StubChat:
    name = "stub"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model = "stub"
        self.calls: list[tuple[str, str, bool]] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.calls.append((system, user, json_mode))
        if json_mode:
            payload = {
                "topics": [
                    {
                        "title": "Заглушка",
                        "timestamp": "00:00:00",
                        "points": ["Сводка сгенерирована офлайн-заглушкой."],
                    }
                ],
                "tech": [],
                "quotes": [],
                "takeaways": [],
                "open_questions": [],
            }
            return ChatResult(text=json.dumps(payload, ensure_ascii=False), model="stub")
        return ChatResult(text="Заглушка: сводка не генерировалась.", model="stub")

    def context_limit(self) -> int:
        return self.cfg.context_budget("stub")
