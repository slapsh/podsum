# podsum

Turn a Russian IT podcast episode into a transcript with timestamps and a two-level summary: a TL;DR you can read in half a minute, and a deep structured breakdown with topics, tools, disagreements, quotes, and chapters.

Two interchangeable paths:

- **Cloud** — everything through one OpenRouter API key.
- **On-premise** — `faster-whisper` (or GigaAM) for speech recognition, Ollama for summarization. Nothing leaves the machine.

## Install

`ffmpeg` is required; everything else is Python.

```bash
# Debian/Ubuntu: apt install ffmpeg   |   macOS: brew install ffmpeg
git clone <this repo> && cd podsum

uv venv && uv pip install -e .            # cloud path only
uv pip install -e ".[local]"              # + faster-whisper for on-premise ASR
uv pip install -e ".[gigaam]"             # + GigaAM for CPU-only Russian ASR
```

Plain pip works too: `python -m venv .venv && .venv/bin/pip install -e .`

Set the key for the cloud path, in the environment or a `.env` file next to the project:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

Check the setup at any time:

```bash
podsum doctor            # ffmpeg, API key, faster-whisper, GPU, Ollama
```

## Use

```bash
# Cloud: transcribe and summarize in one go
podsum run episode.mp3

# On-premise: local Whisper + local Ollama
podsum run episode.mp3 --profile local

# Split the stages — the transcript is cached, so summaries are cheap to redo
podsum transcribe episode.mp3 --profile local
podsum summarize out/episode.transcript.json --llm-model anthropic/claude-sonnet-5

# What is actually available on OpenRouter today, with prices
podsum models --asr
podsum models
```

Outputs land in `out/`:

| File | What it is |
| --- | --- |
| `<name>.transcript.json` | Segments with absolute timestamps, plus which models produced them |
| `<name>.transcript.txt` | The same transcript as `[HH:MM:SS] text` lines |
| `<name>.summary.md` | The summary document |

Because the transcript is cached, re-running `podsum run` on the same episode skips speech recognition entirely. Use `--no-cache` to force a re-transcribe.

### Input formats

Anything `ffmpeg` decodes: `mp3`, `m4a`/`aac`, `opus`, `ogg`, `wav`, `flac`, and video containers like `mp4`/`mkv`. MP3 is a fine format to keep your episodes in — there is no accuracy benefit to a lossless source at speech bandwidth.

Internally the audio is converted once to 16 kHz mono, which is what every speech model resamples to anyway. For cloud transcription each chunk is encoded as 24 kbps Opus, roughly a tenth the bytes of the original MP3, with no measurable accuracy cost.

## Choosing models

### Cloud transcription (OpenRouter `/audio/transcriptions`)

| Slug | Why you would pick it |
| --- | --- |
| `openai/whisper-large-v3` | Default. The Russian baseline with published numbers — roughly 6-9% WER on conversational Russian — and it honours the biasing prompt. |
| `openai/whisper-large-v3-turbo` | Faster and cheaper distillation. Noticeably weaker on Russian than large-v3 (13.25% versus single digits on published aggregates), so trade deliberately. |
| `qwen/qwen3-asr-flash-2026-02-10` | Recent multilingual ASR, token-priced and inexpensive. Worth benchmarking against Whisper on your own audio. |
| `openai/gpt-4o-transcribe` | Best at English product names inside Russian speech — the exact failure mode of IT podcasts. Costs more. |
| `deepgram/nova-3` | Fast, good punctuation, Russian supported. |
| `google/chirp-3`, `mistralai/voxtral-small-24b-2507-stt` | Further alternatives to benchmark. |

Run `podsum models --asr` for the live list; the catalog moves. Be careful comparing the prices it prints: Whisper-class models are billed per second of audio while newer speech-to-text models are billed per token, so the per-unit numbers are not comparable across rows.

### Cloud summarization (OpenRouter `/chat/completions`)

| Slug | Why you would pick it |
| --- | --- |
| `google/gemini-3.7-flash` | Default. ~1M context, so a two-hour episode fits in one pass, and cheap enough to re-run freely. |
| `anthropic/claude-sonnet-5`, `openai/gpt-5.6-sol` | Better at the analytical layer: disagreements, subtext, what was actually decided. |
| `x-ai/grok-4.6` | Strong alternative with a large context. |
| `deepseek/deepseek-v4-flash`, `qwen/qwen3.8-flash` | Near-free. Good for bulk-processing a back catalog. |

### On-premise transcription

Ollama does not do speech recognition, so this is a separate runtime.

| Model | Notes |
| --- | --- |
| `large-v3` (default) | `faster-whisper` on GPU with `int8_float16`. Around 6-9% WER on conversational Russian. |
| `large-v3-turbo` | Roughly twice as fast, measurably worse on Russian. Pick it when throughput matters more than accuracy. |
| `coriollon/whisper-large-v3-turbo-russian` | Russian fine-tune with CTranslate2 weights; roughly 9.6% aggregate WER across six Russian test sets versus 13.25% for vanilla turbo. Turbo speed, closer to large-v3 quality. |
| `bond005/podlodka-turbo` | Fine-tuned on Podlodka, a Russian IT podcast — the closest domain match you can get. |
| GigaAM v3 RNN-T (`--asr gigaam`) | For CPU-only machines. About 10x faster than Whisper on CPU at comparable Russian accuracy, ~225 MB. Whisper turbo on CPU does **not** keep up with real time. |

Any Hugging Face CTranslate2 repo id works as `--asr-model`. GigaAM weights are side-loaded, so point `--asr-model` (or `GIGASTT_MODEL_DIR`) at the model directory.

### On-premise summarization (Ollama)

| Model | Fits in |
| --- | --- |
| `qwen3.8:27b` (default) | ~24 GB. The Qwen line is the strongest open family on Russian. |
| `qwen3.6:35b-a3b` | ~32 GB, MoE, fast for its quality. |
| `gemma4:12b` | ~16 GB. |
| `qwen3.6:27b-q4_K_M` | ~24 GB GPU at 4-bit. |

```bash
ollama serve
ollama pull qwen3.8:27b
podsum run episode.mp3 --profile local --num-ctx 65536
```

**Set `--num-ctx` deliberately.** Ollama defaults to a few thousand tokens and silently truncates anything longer, which would summarize the first ten minutes of an episode and quietly drop the rest. `podsum` always sends an explicit `num_ctx` (32k by default); raise it if your hardware allows, or the pipeline will fall back to map-reduce over more windows than necessary.

The answer allowance scales with the window (a quarter of it, capped by `max_output_tokens`), because asking for an 8k answer inside a 4k window just gets the JSON cut off mid-sentence. If a model does run out of tokens anyway, the truncated object is repaired and what it produced is kept rather than discarding the whole run.

## How it works

```
episode.mp3
  └─ ffmpeg: 16 kHz mono
      ├─ cloud: silence-aware chunks → /audio/transcriptions → merge
      └─ local: faster-whisper / GigaAM over the whole file
          └─ transcript.json (absolute timestamps, cached)
              ├─ fits in context → one summarization call
              └─ does not fit  → map over windows, then reduce
                  └─ summary.md
```

Chunk boundaries snap to the nearest silence within a tolerance window, and each chunk's timestamps are shifted back to absolute episode time, so anchors in the summary point at the right moment in the audio.

Whether to use one pass or map-reduce is decided from an estimate of transcript size against the model's real context length (queried from the OpenRouter catalog, or your `--num-ctx` for Ollama). Force it with `--strategy single|mapreduce`.

### Russian-specific handling

- The language is pinned to `ru` rather than auto-detected. Auto-detection drifts on code-switched speech, and an IT podcast switches to English every other sentence.
- A glossary (`glossary/it_ru.txt`) is fed to the recognizer as a decoding hint so "Kubernetes", "Kafka" and "CI/CD" stay in Latin script instead of being transliterated phonetically. Point `--glossary` at your own list — the terms your show actually uses matter more than the length of the list.
- `condition_on_previous_text=False` and VAD filtering are on for local Whisper. Whisper's repetition loop is the single most common way an hour-long Russian transcript gets corrupted.
- Prompts are written in Russian, which keeps models from drifting into English partway through the summary. `--output-lang en` produces an English summary of Russian audio.

## Summary structure

`summary.md` contains, in order: episode metadata, **Кратко** (the TL;DR), **Ключевые тезисы** (points grouped by topic, each with a timestamp), **Разбор по темам** (the deep layer — context, positions taken, conclusions), **Технологии и инструменты** (a table of everything named), **Практические выводы**, **Спорные моменты и открытые вопросы**, **Цитаты**, and **Тайм-коды**.

A rendered example is in [examples/sample.summary.md](examples/sample.summary.md).

## Cost and speed

A one-hour Russian podcast is roughly 9,000-11,000 words, about 25-30k tokens.

- Cloud transcription: cents per episode at Whisper-turbo or Qwen ASR rates.
- Cloud summarization with `google/gemini-3.7-flash`: well under a cent per episode in one pass.
- Local on a modern GPU: several minutes for transcription with `large-v3` (about half that with `large-v3-turbo`), plus summarization time that depends entirely on the model size.
- Local on CPU: use `--asr gigaam` for transcription — Whisper on CPU is slower than listening to the episode. Summarization on CPU is the real bottleneck: on four cores, `qwen3:1.7b` took about two minutes to summarize a 150-word transcript in one pass, and four minutes for a two-window map-reduce. Summarizing a full episode without a GPU is an overnight job, not an interactive one.

## Development

```bash
uv pip install -e ".[dev]"
pytest                    # unit tests, no network, no models
pytest -m slow            # additionally downloads Whisper tiny and runs real inference
```

Plain `pytest` is fully offline: it drives the pipeline through a stub ASR backend and mocked HTTP transports. `pytest -m slow` additionally downloads Whisper `tiny` and transcribes a short clip of synthesized Russian speech (`espeak-ng` required).

**Verification status**, so you know what to re-check yourself:

- Verified by the offline suite: chunk planning, transcript merging and timestamp offsets, both HTTP clients against recorded response shapes, strategy selection, JSON repair, rendering, and the CLI.
- Verified against real inference: local `faster-whisper` transcription of Russian audio, and on-premise summarization against a live Ollama (`qwen3:1.7b`) in both single-pass and map-reduce modes.
- **Not** verified live: the OpenRouter transcription and chat endpoints. No API key was available where this was built, so those paths are exercised only against recorded shapes taken from the documented request and response formats. Run `podsum doctor` and one short episode before pointing the cloud path at a batch.

## License

MIT.
