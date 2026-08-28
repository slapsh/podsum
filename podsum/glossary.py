"""Domain vocabulary handed to the recognizer as a decoding hint."""

from __future__ import annotations

from pathlib import Path

DEFAULT_GLOSSARY = Path(__file__).resolve().parent.parent / "glossary" / "it_ru.txt"

# Whisper reads at most ~224 prompt tokens and silently drops the rest, so we
# trim here rather than letting the tail be discarded unpredictably.
MAX_PROMPT_CHARS = 850


def load_terms(path: Path | None = None) -> list[str]:
    path = path or DEFAULT_GLOSSARY
    if not Path(path).is_file():
        return []
    terms = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def build_prompt(path: Path | None = None, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """Render the glossary as a sentence-shaped hint the decoder can condition on."""
    terms = load_terms(path)
    if not terms:
        return ""
    prefix = "Подкаст об IT и разработке. Термины: "
    body = ""
    for term in terms:
        candidate = f"{body}{term}, " if body else f"{term}, "
        if len(prefix) + len(candidate) > max_chars:
            break
        body = candidate
    return (prefix + body).rstrip(", ") + "."
