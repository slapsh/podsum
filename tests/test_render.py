"""Markdown rendering."""

from __future__ import annotations

import re

from podsum.render import render_markdown, write_outputs

SUMMARY = {
    "title": "Переезд с монолита на микросервисы",
    "tldr": ["Сначала домены, потом инфраструктура", "Kafka развязала релизы"],
    "topics": [
        {
            "title": "Распределённый монолит",
            "timestamp": "00:00:27",
            "points": ["Сервисы отдельные, деплой общий"],
            "detail": "Полгода команда жила с сервисами, которые нельзя выкатить по отдельности.",
        },
        {"title": "Без деталей", "timestamp": "", "points": ["Только тезис"], "detail": ""},
    ],
    "tech": [
        {"name": "Kafka", "what": "брокер | сообщений", "context": "развязала команды"},
    ],
    "takeaways": ["Начните с контрактных тестов"],
    "debates": [{"question": "Платформа сразу?", "positions": ["Да", "Нет"]}],
    "quotes": [{"text": "Kubernetes сам по себе ничего не чинит", "timestamp": "00:01:28"}],
    "chapters": [{"timestamp": "00:00:00", "title": "Вступление"}],
}


def test_document_has_every_section_in_order(transcript):
    markdown = render_markdown(SUMMARY, transcript, llm_model="google/gemini-3.7-flash")
    headings = [line for line in markdown.splitlines() if line.startswith("## ")]
    assert headings == [
        "## О выпуске",
        "## Кратко",
        "## Ключевые тезисы",
        "## Разбор по темам",
        "## Технологии и инструменты",
        "## Практические выводы",
        "## Спорные моменты и открытые вопросы",
        "## Цитаты",
        "## Тайм-коды",
    ]
    assert markdown.startswith("# Переезд с монолита на микросервисы")


def test_metadata_names_both_models_and_the_source(transcript):
    markdown = render_markdown(
        SUMMARY, transcript, llm_model="qwen3.8:27b", llm_backend="ollama", strategy="single"
    )
    assert "`episode.mp3`" in markdown
    assert "`fixture` (stub)" in markdown
    assert "`qwen3.8:27b` (ollama, single)" in markdown
    assert "2м 32с" in markdown


def test_cost_is_shown_only_when_there_is_one(transcript):
    assert "Стоимость" not in render_markdown(SUMMARY, transcript)
    assert "$0.0042" in render_markdown(SUMMARY, transcript, cost=0.0042)


def test_pipes_inside_table_cells_do_not_break_the_table(transcript):
    markdown = render_markdown(SUMMARY, transcript)
    row = next(line for line in markdown.splitlines() if line.startswith("| Kafka"))
    unescaped = re.findall(r"(?<!\\)\|", row)
    assert len(unescaped) == 4  # three cells, so four delimiters
    assert "брокер \\| сообщений" in row


def test_topics_without_detail_are_skipped_in_the_deep_section(transcript):
    markdown = render_markdown(SUMMARY, transcript)
    deep = markdown.split("## Разбор по темам")[1].split("## Технологии")[0]
    assert "Распределённый монолит" in deep
    assert "Без деталей" not in deep


def test_empty_sections_are_omitted_but_tldr_always_appears(transcript):
    markdown = render_markdown({"tldr": []}, transcript)
    assert "## Кратко" in markdown
    assert "## Технологии и инструменты" not in markdown
    assert "Модель не выделила" in markdown


def test_english_output_uses_english_headings(transcript):
    markdown = render_markdown(SUMMARY, transcript, lang="en")
    assert "## TL;DR" in markdown
    assert "## Chapters" in markdown


def test_write_outputs_produces_three_files(tmp_path, transcript):
    paths = write_outputs(tmp_path, "episode", transcript, markdown="# Заголовок\n")
    assert paths["transcript_json"].is_file()
    assert paths["summary"].read_text(encoding="utf-8") == "# Заголовок\n"
    timed = paths["transcript_txt"].read_text(encoding="utf-8")
    assert timed.startswith("[00:00:00] Всем привет")


def test_transcript_round_trips_through_json(tmp_path, transcript):
    from podsum.asr.base import Transcript

    path = transcript.save(tmp_path / "e.transcript.json")
    reloaded = Transcript.load(path)
    assert reloaded.text == transcript.text
    assert reloaded.segments[3].start == transcript.segments[3].start
    assert reloaded.duration == transcript.duration
