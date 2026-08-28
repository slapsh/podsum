"""Token estimation and transcript windowing for map-reduce."""

from __future__ import annotations

import math
from typing import Iterable

from ..asr.base import Segment

# Rough characters-per-token for a Cyrillic-heavy transcript. `tiktoken` would
# be precise only for OpenAI tokenizers and misleading for everything else, so
# we estimate from script mix instead and keep a safety margin.
_CYRILLIC_WEIGHT = 1 / 2.2
_OTHER_WEIGHT = 1 / 3.6


def estimate_tokens(text: str) -> int:
    cyrillic = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    other = len(text) - cyrillic
    return int(math.ceil(cyrillic * _CYRILLIC_WEIGHT + other * _OTHER_WEIGHT))


def window_segments(
    segments: Iterable[Segment],
    max_tokens: int,
    overlap_segments: int = 2,
) -> list[str]:
    """Split a transcript into timed-text windows of at most `max_tokens`.

    Windows overlap by a couple of segments so a thought split across the
    boundary is visible to both halves.
    """
    max_tokens = max(500, max_tokens)
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        line = f"[{segment.timestamp}] {text}"
        line_tokens = estimate_tokens(line)
        if current and current_tokens + line_tokens > max_tokens:
            windows.append("\n".join(current))
            current = current[-overlap_segments:] if overlap_segments else []
            current_tokens = sum(estimate_tokens(x) for x in current)
        current.append(line)
        current_tokens += line_tokens
    if current:
        windows.append("\n".join(current))
    return windows


def batch_by_tokens(items: list[str], max_tokens: int) -> list[list[str]]:
    """Group already-rendered strings into batches that fit a context budget."""
    batches: list[list[str]] = []
    batch: list[str] = []
    total = 0
    for item in items:
        size = estimate_tokens(item)
        if batch and total + size > max_tokens:
            batches.append(batch)
            batch, total = [], 0
        batch.append(item)
        total += size
    if batch:
        batches.append(batch)
    return batches
