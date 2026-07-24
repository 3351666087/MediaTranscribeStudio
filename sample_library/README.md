# Multilingual Short Sample Library

`manifest.v1.json` defines short, reproducible samples for iterative quality
checks. The binary media is intentionally generated into
`.runtime_cache/sample-library` and is not committed to Git.

The library is deliberately varied along two axes:

- language: Chinese, English, Japanese, Spanish, French, German, Portuguese,
  and Russian;
- capture conditions: clean speech, phone-band noise, room reverb, two-speaker
  turns, overlap, a three-speaker meeting, video scene cuts, and one downloaded
  upstream reference clip.

Build it on macOS with:

```bash
zsh -ic 'mts_activate && python tools/build_sample_library.py'
```

The builder records `sample-library.resolved.v1.json` with the actual duration,
format, byte size, and SHA-256 for every generated artifact. Every case is
hard-limited to 30 seconds. Rebuilding with `--force` refreshes only generated
media and the resolved manifest; prior `results/` evidence is preserved.

Run the quality evaluator against completed worker outputs with:

```bash
zsh -ic 'mts_activate && python tools/evaluate_sample_library.py \
  --manifest .runtime_cache/sample-library/sample-library.resolved.v1.json \
  --results-root .runtime_cache/sample-library/results'
```

To exercise subtitle formatting on a persisted transcript after review:

```bash
zsh -ic 'mts_activate && python tools/export_sample_subtitles.py \
  --transcript .runtime_cache/outputs/sample-library/en_clean_single-run2/transcript-document.v2.json \
  --output-root .runtime_cache/outputs/sample-library/en_clean_single-run2/subtitles'
```

The downloaded reference clip is fetched only at build time. Its URL,
upstream attribution, and expected hash remain in the manifest; the binary is
not checked into this repository.
