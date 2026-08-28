"""Render the structured summary as a readable Markdown document."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .asr.base import Transcript
from .audio import format_duration

LABELS = {
    "ru": {
        "tldr": "Кратко",
        "theses": "Ключевые тезисы",
        "topics": "Разбор по темам",
        "tech": "Технологии и инструменты",
        "takeaways": "Практические выводы",
        "debates": "Спорные моменты и открытые вопросы",
        "quotes": "Цитаты",
        "chapters": "Тайм-коды",
        "meta": "О выпуске",
        "duration": "Длительность",
        "words": "Слов в расшифровке",
        "asr": "Распознавание",
        "llm": "Сводка",
        "source": "Источник",
        "generated": "Сгенерировано",
        "cost": "Стоимость",
        "name": "Название",
        "what": "Что это",
        "context": "Контекст",
        "fallback_title": "Сводка выпуска",
        "empty": "_Модель не выделила ничего в этом разделе._",
    },
    "en": {
        "tldr": "TL;DR",
        "theses": "Key points",
        "topics": "Topic breakdown",
        "tech": "Technologies and tools",
        "takeaways": "Practical takeaways",
        "debates": "Disagreements and open questions",
        "quotes": "Quotes",
        "chapters": "Chapters",
        "meta": "About this episode",
        "duration": "Duration",
        "words": "Transcript words",
        "asr": "Transcription",
        "llm": "Summary",
        "source": "Source",
        "generated": "Generated",
        "cost": "Cost",
        "name": "Name",
        "what": "What it is",
        "context": "Context",
        "fallback_title": "Episode summary",
        "empty": "_The model found nothing for this section._",
    },
}


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items]


def render_markdown(
    summary: dict[str, Any],
    transcript: Transcript,
    *,
    lang: str = "ru",
    llm_model: str = "",
    llm_backend: str = "",
    strategy: str = "",
    cost: float = 0.0,
) -> str:
    labels = LABELS.get(lang, LABELS["ru"])
    lines: list[str] = []

    title = summary.get("title") or labels["fallback_title"]
    lines.append(f"# {title}")
    lines.append("")

    meta = [
        f"- **{labels['source']}:** `{Path(transcript.source).name or '—'}`",
        f"- **{labels['duration']}:** {format_duration(transcript.duration)}",
        f"- **{labels['words']}:** {transcript.word_count}",
        f"- **{labels['asr']}:** `{transcript.model}` ({transcript.backend})",
        f"- **{labels['llm']}:** `{llm_model}` ({llm_backend}, {strategy})",
        f"- **{labels['generated']}:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    if cost:
        meta.append(f"- **{labels['cost']}:** ${cost:.4f}")
    lines.append(f"## {labels['meta']}")
    lines.append("")
    lines.extend(meta)
    lines.append("")

    lines.append(f"## {labels['tldr']}")
    lines.append("")
    tldr = summary.get("tldr") or []
    lines.extend(_bullets(tldr) if tldr else [labels["empty"]])
    lines.append("")

    topics = summary.get("topics") or []
    if topics:
        lines.append(f"## {labels['theses']}")
        lines.append("")
        for topic in topics:
            stamp = topic.get("timestamp") or ""
            heading = f"**{topic['title']}**" + (f" — `{stamp}`" if stamp else "")
            lines.append(heading)
            lines.append("")
            lines.extend(_bullets(topic.get("points") or []))
            lines.append("")

        detailed = [t for t in topics if t.get("detail")]
        if detailed:
            lines.append(f"## {labels['topics']}")
            lines.append("")
            for topic in detailed:
                stamp = topic.get("timestamp") or ""
                lines.append(f"### {topic['title']}" + (f" `{stamp}`" if stamp else ""))
                lines.append("")
                lines.append(topic["detail"])
                lines.append("")

    tech = summary.get("tech") or []
    if tech:
        lines.append(f"## {labels['tech']}")
        lines.append("")
        lines.append(f"| {labels['name']} | {labels['what']} | {labels['context']} |")
        lines.append("| --- | --- | --- |")
        for item in tech:
            lines.append(
                f"| {_escape_cell(item['name'])} "
                f"| {_escape_cell(item.get('what', ''))} "
                f"| {_escape_cell(item.get('context', ''))} |"
            )
        lines.append("")

    takeaways = summary.get("takeaways") or []
    if takeaways:
        lines.append(f"## {labels['takeaways']}")
        lines.append("")
        lines.extend(_bullets(takeaways))
        lines.append("")

    debates = summary.get("debates") or []
    if debates:
        lines.append(f"## {labels['debates']}")
        lines.append("")
        for item in debates:
            lines.append(f"**{item['question']}**")
            lines.append("")
            lines.extend(_bullets(item.get("positions") or []))
            lines.append("")

    quotes = summary.get("quotes") or []
    if quotes:
        lines.append(f"## {labels['quotes']}")
        lines.append("")
        for quote in quotes:
            stamp = quote.get("timestamp") or ""
            speaker = quote.get("speaker")
            attribution = " — ".join(x for x in (speaker, f"`{stamp}`" if stamp else "") if x)
            lines.append(f"> {quote['text']}")
            if attribution:
                lines.append(">")
                lines.append(f"> — {attribution}")
            lines.append("")

    chapters = summary.get("chapters") or []
    if chapters:
        lines.append(f"## {labels['chapters']}")
        lines.append("")
        for chapter in chapters:
            stamp = chapter.get("timestamp") or "—"
            lines.append(f"- `{stamp}` {chapter['title']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    out_dir: Path,
    stem: str,
    transcript: Transcript,
    markdown: str | None = None,
) -> dict[str, Path]:
    """Persist transcript JSON, timed text, and (when present) the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "transcript_json": out_dir / f"{stem}.transcript.json",
        "transcript_txt": out_dir / f"{stem}.transcript.txt",
    }
    transcript.save(paths["transcript_json"])
    paths["transcript_txt"].write_text(transcript.timed_text() + "\n", encoding="utf-8")
    if markdown is not None:
        paths["summary"] = out_dir / f"{stem}.summary.md"
        paths["summary"].write_text(markdown, encoding="utf-8")
    return paths
