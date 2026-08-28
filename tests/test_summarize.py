"""Windowing, strategy selection, and response normalization."""

from __future__ import annotations

import json

import pytest

from podsum.asr.base import Segment
from podsum.llm.base import ChatResult
from podsum.summarize.chunker import batch_by_tokens, estimate_tokens, window_segments
from podsum.summarize.pipeline import normalize_summary, summarize

FULL_SUMMARY = {
    "title": "Переезд с монолита на микросервисы",
    "tldr": ["Начали с инфраструктуры и получили распределённый монолит."],
    "topics": [
        {
            "title": "Порядок работ",
            "timestamp": "00:00:11",
            "points": ["Сначала границы доменов"],
            "detail": "Команда подняла Kubernetes раньше, чем нарезала домены.",
        }
    ],
    "tech": [
        {"name": "Kubernetes", "what": "оркестратор", "context": "подняли слишком рано"},
        {"name": "kubernetes", "what": "дубль", "context": "должен схлопнуться"},
    ],
    "takeaways": ["Контрактные тесты развязывают релизы"],
    "debates": [{"question": "Платформа сразу?", "positions": ["Да", "Нет"]}],
    "quotes": [{"text": "Kubernetes сам по себе ничего не чинит", "timestamp": "00:01:28"}],
    "chapters": [
        {"timestamp": "00:01:52", "title": "Итоги"},
        {"timestamp": "00:00:00", "title": "Вступление"},
    ],
}


class ScriptedChat:
    """A chat client that replays canned JSON and records what it was asked."""

    name = "scripted"
    model = "scripted-model"

    def __init__(self, responses, limit=200_000):
        self.responses = list(responses)
        self.limit = limit
        self.prompts: list[str] = []

    def complete(self, system, user, *, json_mode=False, max_tokens=None):
        self.prompts.append(user)
        payload = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return ChatResult(
            text=json.dumps(payload, ensure_ascii=False),
            model=self.model,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.001},
        )

    def context_limit(self):
        return self.limit


class TestTokenEstimation:
    def test_cyrillic_costs_more_tokens_than_latin(self):
        assert estimate_tokens("абвгдеёжзи") > estimate_tokens("abcdefghij")

    def test_empty_text_is_free(self):
        assert estimate_tokens("") == 0

    def test_estimate_is_in_a_plausible_range_for_russian(self):
        text = "Мы переехали с монолита на микросервисы за полгода. " * 20
        tokens = estimate_tokens(text)
        assert len(text) / 3.5 < tokens < len(text) / 1.5


class TestWindowing:
    def test_windows_respect_the_budget_and_carry_timestamps(self, long_transcript):
        windows = window_segments(long_transcript.segments, max_tokens=600)
        assert len(windows) > 1
        assert all(estimate_tokens(w) <= 700 for w in windows)
        assert windows[0].startswith("[00:00:00]")

    def test_windows_overlap_so_a_split_thought_is_visible_twice(self, long_transcript):
        windows = window_segments(long_transcript.segments, max_tokens=600, overlap_segments=2)
        tail = windows[0].splitlines()[-1]
        assert tail in windows[1]

    def test_a_short_transcript_is_a_single_window(self, transcript):
        assert len(window_segments(transcript.segments, max_tokens=100_000)) == 1

    def test_a_single_oversized_segment_still_yields_a_window(self):
        segments = [Segment(start=0.0, end=60.0, text="слово " * 5000)]
        assert len(window_segments(segments, max_tokens=500)) == 1

    def test_batching_groups_notes_under_the_budget(self):
        items = ["а" * 400 for _ in range(10)]
        batches = batch_by_tokens(items, max_tokens=500)
        assert len(batches) > 1
        assert sum(len(b) for b in batches) == 10


class TestNormalization:
    def test_fills_in_missing_keys(self):
        result = normalize_summary({"tldr": ["один"]})
        assert result["topics"] == []
        assert result["tech"] == []
        assert result["chapters"] == []
        assert result["title"] == ""

    def test_accepts_a_string_where_a_list_was_asked_for(self):
        assert normalize_summary({"tldr": "один тезис"})["tldr"] == ["один тезис"]

    def test_deduplicates_tools_case_insensitively(self):
        assert [t["name"] for t in normalize_summary(FULL_SUMMARY)["tech"]] == ["Kubernetes"]

    def test_chapters_are_sorted_chronologically(self):
        chapters = normalize_summary(FULL_SUMMARY)["chapters"]
        assert [c["timestamp"] for c in chapters] == ["00:00:00", "00:01:52"]

    def test_drops_entries_with_no_content(self):
        data = {"topics": [{"title": "", "points": ["x"]}], "quotes": [{"text": ""}]}
        result = normalize_summary(data)
        assert result["topics"] == []
        assert result["quotes"] == []


class TestStrategySelection:
    def test_a_short_transcript_takes_one_pass(self, transcript, cloud_config):
        client = ScriptedChat([FULL_SUMMARY], limit=200_000)
        result = summarize(transcript, client, cloud_config)
        assert result.strategy == "single"
        assert result.windows == 1
        assert len(client.prompts) == 1
        assert "Расшифровка:" in client.prompts[0]

    def test_a_long_transcript_maps_then_reduces(self, long_transcript, cloud_config):
        cloud_config.max_output_tokens = 1_000
        client = ScriptedChat([FULL_SUMMARY], limit=12_000)
        result = summarize(long_transcript, client, cloud_config)
        assert result.strategy == "mapreduce"
        assert result.windows > 1
        assert len(client.prompts) == result.windows + 1
        assert "Фрагмент:" in client.prompts[0]
        assert "Заметки:" in client.prompts[-1]

    def test_strategy_can_be_forced(self, transcript, cloud_config):
        cloud_config.strategy = "mapreduce"
        client = ScriptedChat([FULL_SUMMARY], limit=200_000)
        assert summarize(transcript, client, cloud_config).strategy == "mapreduce"

    def test_notes_too_large_for_one_reduce_are_folded_in_stages(self, long_transcript, cloud_config):
        cloud_config.max_output_tokens = 500
        client = ScriptedChat([FULL_SUMMARY], limit=3_000)
        result = summarize(long_transcript, client, cloud_config)
        reduce_prompts = [p for p in client.prompts if "Заметки:" in p]
        assert len(reduce_prompts) > 1  # batched, then combined
        assert result.data["title"] == FULL_SUMMARY["title"]

    def test_reduce_terminates_when_every_note_exceeds_the_budget(self, cloud_config):
        """Batching cannot shrink notes that are individually too large."""
        from podsum.summarize.pipeline import _reduce

        notes = [json.dumps({"topics": [{"title": "т" * 4000}]}, ensure_ascii=False) for _ in range(4)]
        client = ScriptedChat([FULL_SUMMARY], limit=2_000)
        data, usage = _reduce(notes, client, "system", budget=100)
        assert data["title"] == FULL_SUMMARY["title"]
        assert len(usage) < 10  # folded in pairs rather than spinning forever

    def test_cost_and_tokens_are_accumulated(self, long_transcript, cloud_config):
        client = ScriptedChat([FULL_SUMMARY], limit=4_000)
        result = summarize(long_transcript, client, cloud_config)
        assert result.cost == pytest.approx(0.001 * len(client.prompts))
        assert result.tokens["prompt"] == 100 * len(client.prompts)

    def test_progress_is_reported_for_every_window(self, long_transcript, cloud_config):
        client = ScriptedChat([FULL_SUMMARY], limit=4_000)
        seen: list[tuple[str, int, int]] = []
        summarize(long_transcript, client, cloud_config, on_progress=lambda *a: seen.append(a))
        assert seen
        assert any(stage.startswith("Разбор") for stage, _, _ in seen)
        assert any(stage.startswith("Сборка") for stage, _, _ in seen)
