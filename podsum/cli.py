"""Command line interface."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import __version__
from .asr import build_asr
from .asr.base import Transcript
from .audio import format_duration, normalize, probe
from .config import PROFILES, Config, load_dotenv
from .errors import PodsumError
from .llm import build_llm
from .render import render_markdown, write_outputs
from .summarize import summarize as run_summarize

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Transcribe Russian IT podcasts and summarize them at two levels of detail.",
)
console = Console()

# Backends that want a pre-decoded 16 kHz mono WAV rather than the raw podcast file.
_WANTS_WAV = {"local", "gigaam"}


def _fail(exc: PodsumError) -> None:
    console.print(f"[bold red]Ошибка:[/] {exc.message}")
    if exc.hint:
        console.print(f"[dim]{exc.hint}[/]")
    raise typer.Exit(code=1)


def _build_config(
    profile: str,
    asr: Optional[str],
    asr_model: Optional[str],
    llm: Optional[str],
    llm_model: Optional[str],
    output_lang: Optional[str],
    strategy: Optional[str],
    num_ctx: Optional[int],
    chunk_seconds: Optional[int],
    glossary: Optional[Path],
    device: Optional[str],
    no_cache: bool = False,
) -> Config:
    load_dotenv()
    cfg = Config.from_profile(
        profile,
        asr_backend=asr,
        asr_model=asr_model,
        llm_backend=llm,
        llm_model=llm_model,
        output_lang=output_lang,
        strategy=strategy,
        num_ctx=num_ctx,
        chunk_seconds=chunk_seconds,
        glossary_path=glossary,
        device=device,
    )
    if no_cache:
        cfg.cache = False
    return cfg


def _transcribe(cfg: Config, audio: Path) -> Transcript:
    backend = build_asr(cfg)
    info = probe(audio)
    console.print(
        f"[dim]{audio.name}: {format_duration(info.duration)}, "
        f"{info.codec} {info.sample_rate} Hz, {info.size_bytes / 1e6:.1f} MB[/]"
    )
    tmpdir: str | None = None
    target = audio
    try:
        if backend.name in _WANTS_WAV:
            tmpdir = tempfile.mkdtemp(prefix="podsum-audio-")
            console.print("[dim]Готовлю аудио: 16 кГц моно WAV…[/]")
            target = normalize(audio, Path(tmpdir) / "episode.wav", fmt="wav")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Распознавание речи", total=max(info.duration, 1.0))

            def on_progress(done: float, total: float) -> None:
                progress.update(task, completed=min(done, total), total=max(total, 1.0))

            transcript = backend.transcribe(target, on_progress)
        transcript.source = str(audio)
        transcript.duration = transcript.duration or info.duration
        return transcript
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _summarize(cfg: Config, transcript: Transcript):
    client = build_llm(cfg)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed:>3.0f}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Сводка", total=1)

        def on_progress(stage: str, done: int, total: int) -> None:
            progress.update(task, description=stage, completed=done, total=max(total, 1))

        result = run_summarize(transcript, client, cfg, on_progress)
    close = getattr(client, "close", None)
    if callable(close):
        close()
    return result


def _report(paths: dict[str, Path], result=None) -> None:
    console.print()
    console.print("[bold green]Готово.[/]")
    for label, path in paths.items():
        console.print(f"  [dim]{label}:[/] {path}")
    if result is not None:
        tokens = result.tokens
        details = [f"стратегия: {result.strategy}", f"фрагментов: {result.windows}"]
        if tokens["prompt"] or tokens["completion"]:
            details.append(f"токенов: {tokens['prompt']} → {tokens['completion']}")
        if result.cost:
            details.append(f"стоимость: ${result.cost:.4f}")
        console.print(f"  [dim]{', '.join(details)}[/]")


@app.command()
def run(
    audio: Path = typer.Argument(..., help="Podcast file: mp3, m4a, opus, wav, flac, or video."),
    profile: str = typer.Option("cloud", "--profile", "-p", help=f"One of: {', '.join(PROFILES)}."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    asr: Optional[str] = typer.Option(None, help="ASR backend: openrouter, local, gigaam, stub."),
    asr_model: Optional[str] = typer.Option(None, help="ASR model slug, size, or path."),
    llm: Optional[str] = typer.Option(None, help="LLM backend: openrouter, ollama, stub."),
    llm_model: Optional[str] = typer.Option(None, help="Chat model slug or Ollama tag."),
    output_lang: Optional[str] = typer.Option(None, help="Summary language: ru or en."),
    strategy: Optional[str] = typer.Option(None, help="auto, single, or mapreduce."),
    num_ctx: Optional[int] = typer.Option(None, help="Ollama context window in tokens."),
    chunk_seconds: Optional[int] = typer.Option(None, help="Cloud upload chunk length."),
    glossary: Optional[Path] = typer.Option(None, help="Term list biasing the recognizer."),
    device: Optional[str] = typer.Option(None, help="cpu, cuda, or auto (local backend)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Re-transcribe even if a transcript exists."),
) -> None:
    """Transcribe an episode and write the summary."""
    try:
        cfg = _build_config(
            profile, asr, asr_model, llm, llm_model, output_lang, strategy,
            num_ctx, chunk_seconds, glossary, device, no_cache,
        )
        cfg.validate()
        stem = audio.stem
        cached = out / f"{stem}.transcript.json"
        if cfg.cache and cached.is_file():
            console.print(f"[dim]Использую готовую расшифровку: {cached}[/]")
            transcript = Transcript.load(cached)
        else:
            transcript = _transcribe(cfg, audio)
        result = _summarize(cfg, transcript)
        markdown = render_markdown(
            result.data,
            transcript,
            lang=cfg.output_lang,
            llm_model=result.model or cfg.llm_model,
            llm_backend=result.backend,
            strategy=result.strategy,
            cost=result.cost,
        )
        paths = write_outputs(out, stem, transcript, markdown)
        _report(paths, result)
    except PodsumError as exc:
        _fail(exc)


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., help="Podcast file to transcribe."),
    profile: str = typer.Option("cloud", "--profile", "-p"),
    out: Path = typer.Option(Path("out"), "--out", "-o"),
    asr: Optional[str] = typer.Option(None),
    asr_model: Optional[str] = typer.Option(None),
    chunk_seconds: Optional[int] = typer.Option(None),
    glossary: Optional[Path] = typer.Option(None),
    device: Optional[str] = typer.Option(None),
) -> None:
    """Transcribe only; write the transcript JSON and timed text."""
    try:
        cfg = _build_config(
            profile, asr, None, "stub", None, None, None, None,
            chunk_seconds, glossary, device,
        )
        if asr_model:
            cfg.asr_model = asr_model
        if cfg.asr_backend == "openrouter" and not cfg.openrouter_api_key:
            cfg.llm_backend = "stub"
        cfg.validate()
        transcript = _transcribe(cfg, audio)
        paths = write_outputs(out, audio.stem, transcript)
        _report(paths)
    except PodsumError as exc:
        _fail(exc)


@app.command()
def summarize(
    transcript_path: Path = typer.Argument(..., help="A .transcript.json produced by `transcribe`."),
    profile: str = typer.Option("cloud", "--profile", "-p"),
    out: Path = typer.Option(Path("out"), "--out", "-o"),
    llm: Optional[str] = typer.Option(None),
    llm_model: Optional[str] = typer.Option(None),
    output_lang: Optional[str] = typer.Option(None),
    strategy: Optional[str] = typer.Option(None),
    num_ctx: Optional[int] = typer.Option(None),
) -> None:
    """Summarize an existing transcript without re-running speech recognition."""
    try:
        cfg = _build_config(
            profile, "stub", None, llm, llm_model, output_lang, strategy,
            num_ctx, None, None, None,
        )
        cfg.validate()
        transcript = Transcript.load(transcript_path)
        result = _summarize(cfg, transcript)
        markdown = render_markdown(
            result.data,
            transcript,
            lang=cfg.output_lang,
            llm_model=result.model or cfg.llm_model,
            llm_backend=result.backend,
            strategy=result.strategy,
            cost=result.cost,
        )
        stem = transcript_path.name.replace(".transcript.json", "") or transcript_path.stem
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / f"{stem}.summary.md"
        summary_path.write_text(markdown, encoding="utf-8")
        _report({"summary": summary_path}, result)
    except PodsumError as exc:
        _fail(exc)


@app.command()
def models(
    asr: bool = typer.Option(False, "--asr", help="List speech-to-text models instead of chat models."),
    min_context: int = typer.Option(200_000, help="Hide chat models below this context length."),
    limit: int = typer.Option(25, help="Rows to show."),
) -> None:
    """List what OpenRouter actually serves today, with prices."""
    from .llm.openrouter import list_models

    load_dotenv()
    cfg = Config.from_profile("cloud")
    try:
        entries = list_models(cfg, transcription=asr)
    except Exception as exc:  # network failure should not raise a traceback
        console.print(f"[bold red]Не удалось получить каталог OpenRouter:[/] {exc}")
        raise typer.Exit(code=1)

    rows = []
    if asr:
        table = Table(title="OpenRouter: транскрибация")
        table.add_column("slug", style="cyan", no_wrap=True)
        table.add_column("цена за единицу", justify="right")
        table.add_column("modality")
        for entry in entries:
            pricing = entry.get("pricing") or {}
            architecture = entry.get("architecture") or {}
            rows.append(
                (
                    entry.get("id", "?"),
                    _unit_price(pricing.get("prompt")),
                    architecture.get("modality", "—"),
                )
            )
    else:
        table = Table(title="OpenRouter: суммаризация")
        table.add_column("slug", style="cyan", no_wrap=True)
        table.add_column("context", justify="right")
        table.add_column("$/M in", justify="right")
        table.add_column("$/M out", justify="right")
        for entry in entries:
            context = int(entry.get("context_length") or 0)
            if context < min_context:
                continue
            pricing = entry.get("pricing") or {}
            rows.append(
                (
                    entry.get("id", "?"),
                    f"{context:,}",
                    _price_per_million(pricing.get("prompt")),
                    _price_per_million(pricing.get("completion")),
                )
            )
    rows.sort(key=lambda r: r[0])
    for row in rows[:limit]:
        table.add_row(*row)
    console.print(table)
    console.print(f"[dim]Показано {min(len(rows), limit)} из {len(rows)}.[/]")
    if asr:
        console.print(
            "[dim]Единица зависит от модели: Whisper-класс тарифицируется за секунду "
            "аудио, новые STT — за токен. Точную единицу смотрите на странице модели.[/]"
        )


def _price_per_million(value: object) -> str:
    try:
        per_million = float(value) * 1_000_000  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if per_million == 0:
        return "0"
    return f"{per_million:,.2f}"


def _unit_price(value: object) -> str:
    try:
        price = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    return "0" if price == 0 else f"${price:.8f}".rstrip("0")


@app.command()
def doctor(
    profile: str = typer.Option("cloud", "--profile", "-p"),
    llm_model: Optional[str] = typer.Option(None),
) -> None:
    """Check ffmpeg, credentials, local runtimes, and GPU availability."""
    load_dotenv()
    cfg = Config.from_profile(profile, llm_model=llm_model)
    table = Table(title="podsum doctor")
    table.add_column("Проверка")
    table.add_column("Статус")
    table.add_column("Детали")

    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        table.add_row(binary, "OK" if path else "нет", path or "apt install ffmpeg")

    key = cfg.openrouter_api_key
    table.add_row(
        "OPENROUTER_API_KEY",
        "OK" if key else "нет",
        f"…{key[-4:]}" if key else "нужен для --profile cloud",
    )

    try:
        from faster_whisper import __version__ as fw_version  # noqa: F401

        from .asr.local_whisper import resolve_compute_type, resolve_device

        device = resolve_device("auto")
        table.add_row(
            "faster-whisper",
            "OK",
            f"device={device}, compute={resolve_compute_type('auto', device)}",
        )
    except ImportError:
        table.add_row("faster-whisper", "нет", escape("pip install 'podsum[local]'"))

    from .llm.ollama import OllamaChat

    ollama_ok, message = OllamaChat(cfg).is_available()
    table.add_row("Ollama", "OK" if ollama_ok else "нет", message)

    console.print(table)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"podsum {__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
