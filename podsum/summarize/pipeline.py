"""Single-pass and map-reduce summarization over a transcript."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..asr.base import Transcript
from ..config import Config
from ..llm.base import ChatClient, parse_json_object
from . import prompts
from .chunker import batch_by_tokens, estimate_tokens, window_segments

# Room reserved for the instructions themselves, which the transcript budget
# must not eat into.
PROMPT_OVERHEAD_TOKENS = 1_200

ProgressCB = Callable[[str, int, int], None]

SUMMARY_KEYS = ("title", "tldr", "topics", "tech", "takeaways", "debates", "quotes", "chapters")


@dataclass
class SummaryResult:
    data: dict[str, Any]
    strategy: str
    windows: int = 1
    model: str = ""
    backend: str = ""
    usage: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cost(self) -> float:
        total = 0.0
        for entry in self.usage:
            value = entry.get("cost")
            if isinstance(value, (int, float)):
                total += float(value)
        return total

    @property
    def tokens(self) -> dict[str, int]:
        prompt = completion = 0
        for entry in self.usage:
            prompt += int(entry.get("prompt_tokens") or 0)
            completion += int(entry.get("completion_tokens") or 0)
        return {"prompt": prompt, "completion": completion}


def _input_budget(client: ChatClient, cfg: Config) -> int:
    limit = client.context_limit()
    usable = int(limit * cfg.context_fill_ratio) - cfg.max_output_tokens - PROMPT_OVERHEAD_TOKENS
    return max(1_000, usable)


def normalize_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce a model response into the shape the renderer expects.

    Models drop keys, return strings where a list belongs, or repeat a tool
    across chunks; fixing that here keeps the renderer free of defensive code.
    """
    result: dict[str, Any] = {key: data.get(key) for key in SUMMARY_KEYS}
    result["title"] = (result.get("title") or "").strip() if isinstance(result.get("title"), str) else ""

    for key in ("tldr", "takeaways"):
        result[key] = _as_str_list(result.get(key))

    result["topics"] = [
        {
            "title": _as_text(topic.get("title")),
            "timestamp": _as_text(topic.get("timestamp")),
            "points": _as_str_list(topic.get("points")),
            "detail": _as_text(topic.get("detail")),
        }
        for topic in _as_dict_list(result.get("topics"))
        if _as_text(topic.get("title"))
    ]

    seen: set[str] = set()
    tech: list[dict[str, str]] = []
    for item in _as_dict_list(result.get("tech")):
        name = _as_text(item.get("name"))
        key = name.lower()
        if not name or key in seen:
            continue
        seen.add(key)
        tech.append(
            {
                "name": name,
                "what": _as_text(item.get("what")),
                "context": _as_text(item.get("context")),
            }
        )
    result["tech"] = tech

    result["debates"] = [
        {
            "question": _as_text(item.get("question")),
            "positions": _as_str_list(item.get("positions")),
        }
        for item in _as_dict_list(result.get("debates"))
        if _as_text(item.get("question"))
    ]

    result["quotes"] = [
        {
            "text": _as_text(item.get("text")),
            "timestamp": _as_text(item.get("timestamp")),
            "speaker": _as_text(item.get("speaker")) or None,
        }
        for item in _as_dict_list(result.get("quotes"))
        if _as_text(item.get("text"))
    ]

    chapters = [
        {"timestamp": _as_text(item.get("timestamp")), "title": _as_text(item.get("title"))}
        for item in _as_dict_list(result.get("chapters"))
        if _as_text(item.get("title"))
    ]
    result["chapters"] = sorted(chapters, key=lambda c: _timestamp_key(c["timestamp"]))
    return result


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                text = _as_text(item.get("text") or item.get("point") or item.get("title"))
                if text:
                    out.append(text)
        return out
    return []


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _timestamp_key(value: str) -> tuple[int, ...]:
    parts = [p for p in value.replace("[", "").replace("]", "").split(":") if p.strip().isdigit()]
    if not parts:
        return (10**9,)
    nums = [int(p) for p in parts]
    while len(nums) < 3:
        nums.insert(0, 0)
    return (nums[0] * 3600 + nums[1] * 60 + nums[2],)


def summarize(
    transcript: Transcript,
    client: ChatClient,
    cfg: Config,
    on_progress: ProgressCB | None = None,
) -> SummaryResult:
    """Produce the structured summary, choosing single-pass when it fits."""
    system = prompts.system_prompt(cfg.output_lang)
    budget = _input_budget(client, cfg)
    timed = transcript.timed_text()
    fits = estimate_tokens(timed) <= budget

    strategy = cfg.strategy
    if strategy == "auto":
        strategy = "single" if fits else "mapreduce"

    usage: list[dict[str, Any]] = []
    if strategy == "single":
        if on_progress:
            on_progress("Сводка одним проходом", 0, 1)
        result = client.complete(
            system, prompts.single_pass_prompt(timed), json_mode=True
        )
        usage.append(result.usage)
        data = normalize_summary(parse_json_object(result.text, "Summarizer"))
        if on_progress:
            on_progress("Сводка одним проходом", 1, 1)
        return SummaryResult(
            data=data,
            strategy="single",
            windows=1,
            model=result.model or client.model,
            backend=client.name,
            usage=usage,
        )

    windows = window_segments(transcript.segments, budget)
    notes: list[str] = []
    for index, window in enumerate(windows, start=1):
        if on_progress:
            on_progress("Разбор фрагментов", index - 1, len(windows))
        result = client.complete(
            system, prompts.map_prompt(window, index, len(windows)), json_mode=True
        )
        usage.append(result.usage)
        parsed = parse_json_object(result.text, f"Summarizer (фрагмент {index})")
        notes.append(json.dumps(parsed, ensure_ascii=False))
    if on_progress:
        on_progress("Разбор фрагментов", len(windows), len(windows))

    data, reduce_usage = _reduce(notes, client, system, budget, on_progress)
    usage.extend(reduce_usage)
    return SummaryResult(
        data=data,
        strategy="mapreduce",
        windows=len(windows),
        model=client.model,
        backend=client.name,
        usage=usage,
    )


def _reduce(
    notes: list[str],
    client: ChatClient,
    system: str,
    budget: int,
    on_progress: ProgressCB | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fold per-window notes into one summary, in stages if they do not fit.

    A three-hour episode against a 32k local context produces more notes than
    one reduce call can hold, so batches are folded first and then combined.
    """
    usage: list[dict[str, Any]] = []
    batches = batch_by_tokens(notes, budget)
    if len(batches) == len(notes) > 1:
        # Every note is individually over budget, so batching made no progress.
        # Pair them up regardless: the real context limit is larger than our
        # deliberately conservative budget, and halving each round terminates.
        batches = [notes[i : i + 2] for i in range(0, len(notes), 2)]
    if on_progress:
        on_progress("Сборка сводки", 0, len(batches))
    partials: list[str] = []
    for index, batch in enumerate(batches, start=1):
        result = client.complete(system, prompts.reduce_prompt("\n".join(batch)), json_mode=True)
        usage.append(result.usage)
        partials.append(
            json.dumps(parse_json_object(result.text, "Summarizer (сборка)"), ensure_ascii=False)
        )
        if on_progress:
            on_progress("Сборка сводки", index, len(batches))
    if len(partials) == 1:
        return normalize_summary(json.loads(partials[0])), usage
    data, extra = _reduce(partials, client, system, budget, on_progress)
    usage.extend(extra)
    return data, usage
