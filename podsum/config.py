"""Runtime configuration: profiles, environment variables, and `.env` loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from .errors import ConfigError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

ASR_BACKENDS = ("openrouter", "local", "gigaam", "stub")
LLM_BACKENDS = ("openrouter", "ollama", "stub")
STRATEGIES = ("auto", "single", "mapreduce")

# Defaults per profile. Slugs were checked against the live OpenRouter catalog
# and the Ollama library; `podsum models` re-checks them at runtime.
PROFILES: dict[str, dict[str, str]] = {
    "cloud": {
        "asr_backend": "openrouter",
        # Whisper large-v3 is the Russian option with published WER numbers and
        # documented prompt biasing; `podsum models --asr` lists the rest.
        "asr_model": "openai/whisper-large-v3",
        "llm_backend": "openrouter",
        "llm_model": "google/gemini-3.7-flash",
    },
    "local": {
        "asr_backend": "local",
        # Quality first: turbo is roughly twice as fast but measurably worse on
        # Russian, so speed is an explicit choice rather than the default.
        "asr_model": "large-v3",
        "llm_backend": "ollama",
        "llm_model": "qwen3.8:27b",
    },
}

# Fallback context budgets used only when the backend cannot report its own.
DEFAULT_CONTEXT_TOKENS = {"openrouter": 128_000, "ollama": 32_768, "stub": 32_768}


def load_dotenv(path: Path | None = None) -> None:
    """Populate os.environ from a `.env` file without adding a dependency.

    Existing environment variables always win, so an exported key is not
    silently overridden by a stale file.
    """
    path = path or Path.cwd() / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


@dataclass
class Config:
    """Everything the pipeline needs to run one episode."""

    asr_backend: str = "openrouter"
    asr_model: str = PROFILES["cloud"]["asr_model"]
    llm_backend: str = "openrouter"
    llm_model: str = PROFILES["cloud"]["llm_model"]

    language: str = "ru"
    output_lang: str = "ru"

    openrouter_api_key: str | None = None
    openrouter_base_url: str = OPENROUTER_BASE_URL
    ollama_base_url: str = OLLAMA_BASE_URL

    # Whisper-family knobs (ignored by cloud backends that do not expose them).
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5

    # Audio handling.
    chunk_seconds: int = 600
    chunk_overlap_seconds: float = 0.0
    cloud_audio_format: str = "opus"

    # Summarization.
    strategy: str = "auto"
    context_tokens: int | None = None
    num_ctx: int = 32_768
    max_output_tokens: int = 8_000
    temperature: float = 0.2
    context_fill_ratio: float = 0.55

    glossary_path: Path | None = None
    cache: bool = True
    timeout_seconds: float = 900.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_profile(cls, profile: str = "cloud", **overrides: object) -> "Config":
        if profile not in PROFILES:
            raise ConfigError(
                f"Unknown profile {profile!r}.",
                hint=f"Available profiles: {', '.join(PROFILES)}.",
            )
        cfg = cls(**PROFILES[profile])  # type: ignore[arg-type]
        cfg.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY") or None
        cfg.openrouter_base_url = os.environ.get(
            "OPENROUTER_BASE_URL", OPENROUTER_BASE_URL
        )
        cfg.ollama_base_url = os.environ.get("OLLAMA_HOST", OLLAMA_BASE_URL)
        if not cfg.ollama_base_url.startswith("http"):
            cfg.ollama_base_url = f"http://{cfg.ollama_base_url}"
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(cfg, **clean)  # type: ignore[arg-type]

    def validate(self) -> None:
        if self.asr_backend not in ASR_BACKENDS:
            raise ConfigError(
                f"Unknown ASR backend {self.asr_backend!r}.",
                hint=f"Choose one of: {', '.join(ASR_BACKENDS)}.",
            )
        if self.llm_backend not in LLM_BACKENDS:
            raise ConfigError(
                f"Unknown LLM backend {self.llm_backend!r}.",
                hint=f"Choose one of: {', '.join(LLM_BACKENDS)}.",
            )
        if self.strategy not in STRATEGIES:
            raise ConfigError(
                f"Unknown strategy {self.strategy!r}.",
                hint=f"Choose one of: {', '.join(STRATEGIES)}.",
            )
        needs_key = "openrouter" in (self.asr_backend, self.llm_backend)
        if needs_key and not self.openrouter_api_key:
            raise ConfigError(
                "OPENROUTER_API_KEY is not set but an OpenRouter backend was selected.",
                hint=(
                    "Export the key, put it in a .env file, or switch to the "
                    "on-premise path with --profile local."
                ),
            )

    def context_budget(self, backend: str) -> int:
        if self.context_tokens:
            return self.context_tokens
        if backend == "ollama":
            return self.num_ctx
        return DEFAULT_CONTEXT_TOKENS.get(backend, 128_000)
