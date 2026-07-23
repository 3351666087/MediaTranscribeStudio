# High-Quality Subtitle Domain Slice

## Scope

This slice provides a deterministic, pure-Python subtitle domain layer. It
accepts transcript segments containing `start`/`end`/`text`/`speaker` (seconds)
or `startMs`/`endMs`/`text`/`speakerId` (milliseconds), composes readable cues,
runs hard QA, exports subtitle text, and creates reviewable FFmpeg plans.

It does **not** execute FFmpeg or FFprobe, inspect media streams, write files,
or claim that every FFmpeg build supports every container, codec, subtitle
codec, font, or filter.

## Non-negotiable invariants

- Source media is immutable.
- Every output is a new sidecar or derived-media file.
- `sourcePath == outputPath` is rejected at runtime.
- Cue times are monotonic, non-overlapping, and separated by the configured
  minimum gap.
- Cue text is split without dropping or reordering source characters.
- Line count, line length, reading speed, and minimum/maximum duration are
  hard QA checks.
- Input order remains authoritative. Overlapping source segments are shifted
  forward and the repair is recorded.
- This module never launches an external process.

## Output modes

| Mode | Result | FFmpeg | Derived content reversibility |
| --- | --- | --- | --- |
| `sidecar` | New `.srt`, `.vtt`, or `.ass` text file | Not required | Fully reversible |
| `soft-mux` | New media copy with a selectable subtitle stream | Required later | Reversible; subtitle stream can be disabled or removed |
| `burn-in` | New media copy with subtitles rendered into video pixels | Required later | Burned pixels are not reversible, but the original remains untouched |

All three modes are workflow-reversible because the source remains available
and the derived artifact can be discarded.

## Formats and themes

Text export supports:

- SubRip (`srt`)
- WebVTT (`webvtt`)
- Advanced SubStation Alpha (`ass`)

The domain defines these selectable themes:

- YouTube Clean
- YouTube Bold
- Minimal Glass
- Karaoke Highlight
- Speaker Color
- Documentary
- News Lower Third
- Custom

Preset styles carry font family and fallbacks, size, weight, colors, outline,
shadow, background opacity, safe-area margins, and alignment. `Custom` requires
an explicit style. Karaoke-grade active-word animation additionally requires
word-level timings; segment-only input keeps the visual theme but must not
fabricate word timing.

## Cue composition and QA

`arrange_cues()`:

1. validates and normalizes segment timing;
2. reserves layout capacity for optional speaker labels;
3. chooses punctuation-first or whitespace-first breakpoints before hard
   Unicode code-point breaks;
4. splits text to satisfy line, cue, and reading-speed capacity;
5. allocates available source duration up to the configured cue maximum;
6. repairs overlaps by shifting later cues;
7. reconstructs every source segment from cue provenance and fails if any
   character was changed or lost.

`audit_cues()` can also validate externally assembled cues. Reading speed is
measured as non-whitespace Unicode code points per second. This deterministic
metric is language-neutral, but production profiles may later select
locale-sensitive thresholds.

## FFmpeg plan boundary

`build_subtitle_output_plan()` returns data only. Soft mux and burn-in plans
are deliberately marked as requiring content probing and not permitted for
execution.

- Soft mux plans copy existing source streams and reserve subtitle codec
  selection for the capability layer.
- Burn-in plans use the subtitle filter, require video re-encoding, and prefer
  audio stream copy only after the target container has been validated.
- Both plans use FFmpeg `-n` semantics so an existing derived output is not
  overwritten.

## Required integration work

Before production execution, a separate trusted adapter must:

1. run FFprobe content inspection for actual streams, durations, containers,
   codecs, attachments, rotation, HDR/color metadata, and existing subtitles;
2. query the installed FFmpeg build for available demuxers, muxers, encoders,
   subtitle codecs, filters, and libass support;
3. select a compatible subtitle codec for the target container;
4. select encoding, quality, pixel-format, color-metadata, and hardware policy
   for burn-in without degrading the source unexpectedly;
5. resolve and package font files, including CJK and other Unicode fallbacks;
6. execute only after explicit output-path and overwrite checks;
7. verify sidecar syntax, probe the derived media, and compare duration/stream
   invariants;
8. render representative subtitle frames and run visual QA for clipping,
   safe areas, contrast, line breaks, speaker colors, light/dark scenes, and
   multilingual glyph coverage.

No extension registry alone is sufficient. Capability decisions must be based
on media content and the exact local FFprobe/FFmpeg build.
