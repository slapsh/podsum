# Example output

What `podsum` writes for one episode:

- `sample.transcript.json` — the machine-readable transcript, with absolute timestamps and the models that produced it. This is the file `podsum summarize` reads.
- `sample.transcript.txt` — the same content as `[HH:MM:SS] text` lines, for reading and grepping.
- `sample.summary.md` — the summary document.

A caveat about how these were produced: the transcript is a short hand-written stand-in for a real episode, and the summary body was written by hand to show what a good model returns for it. No API key was available in the environment where this repository was built, so nothing here is the output of a live model. The Markdown itself is real — it comes from `podsum.render.render_markdown`, the same code path the CLI uses — so the structure, headings, and formatting are exactly what you will get.

Regenerate the summary from the transcript with a real model:

```bash
podsum summarize examples/sample.transcript.json --out /tmp/podsum-example
```
