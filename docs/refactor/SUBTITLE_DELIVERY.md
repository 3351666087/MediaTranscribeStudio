# Source-Immutable Subtitle Delivery

## Purpose

`backend/subtitle_delivery.py` is the trusted execution boundary between the
pure subtitle planning layer and the local FFmpeg/FFprobe toolchain. It
delivers:

- UTF-8 SRT, WebVTT, or ASS sidecars;
- soft subtitle streams in a newly derived media file;
- burned-in subtitles in a newly encoded media file.

It never edits, truncates, renames, replaces, or opens the source media for
writing. It never overwrites an existing destination.

## Execution pipeline

1. Validate the immutable-source `SubtitleOutputPlan`.
2. Canonicalize the source, subtitle input, output parent, and alias
   relationships.
3. Reject any existing output, including a dangling symlink.
4. Capture source size, nanosecond mtime, and full SHA-256 evidence.
5. Probe the source through the trusted `MediaProbe` content boundary.
6. Validate subtitle syntax and UTF-8 encoding.
7. Reject ASS karaoke timing tags unless their exact subtitle SHA-256 is
   covered by verified word-level timing evidence.
8. Query the exact FFmpeg build for encoders, filters, and muxers.
9. Select only an approved subtitle codec or explicit burn-in video strategy.
10. Execute FFmpeg as an argument vector with `shell=False`, no stdin, a hard
    timeout, and bounded stdout/stderr.
11. Write to an unpredictable temporary path in the final output directory.
12. FFprobe and QA the temporary media before publication.
13. Atomically publish with a same-directory hard-link create. Hard-link
    creation is no-clobber: if the final name appears concurrently, delivery
    fails instead of replacing it.
14. FFprobe the published media again.
15. Recheck source size, mtime, and SHA-256.
16. Remove the newly published artifact if any post-publication hard gate
    fails.

The output contract is `contracts/subtitle-delivery.schema.json`.

## Sidecar delivery

Sidecar mode accepts explicit UTF-8 text or bytes. It validates the declared
format:

- SRT requires a valid SubRip timestamp arrow;
- WebVTT must begin with `WEBVTT`;
- ASS requires Script Info, Events, and at least one Dialogue line.

The temporary sidecar is written with exclusive-create semantics, flushed,
`fsync`-ed, hashed, and atomically published. FFprobe is not applied to the
sidecar because it is not an audio/video container. The source media is still
probed before delivery and its full integrity evidence is checked afterward.

## Soft-mux policy

Soft muxing preserves every source stream with stream copy and adds exactly
one subtitle stream. The new subtitle stream is addressed by its subtitle
type index so existing subtitle streams are not accidentally transcoded.

Approved target mappings are deliberately narrow:

| Target | SRT | WebVTT | ASS |
| --- | --- | --- | --- |
| MP4/M4V/MOV | `mov_text` | `mov_text` | `mov_text` |
| Matroska | `subrip` | `webvtt`, then `subrip` fallback | `ass` |
| WebM | `webvtt` | `webvtt` | `webvtt` |

Any unlisted target fails closed. A listed codec must also appear in the
installed FFmpeg encoder inventory. Conversion to `mov_text` or WebVTT can
discard ASS styling; choose Matroska for reversible styled ASS or use burn-in
when pixel appearance is authoritative.

Post-delivery QA requires:

- measurable source and output duration within the configured absolute or
  relative tolerance;
- all source audio and non-attached video streams;
- unchanged copied audio/video codec evidence;
- all existing subtitle streams in their original order;
- exactly one newly added subtitle stream with the selected codec;
- unchanged HDR and rotation evidence;
- no loss of chapters;
- non-empty output;
- a successful post-publication trusted probe.

## Burn-in policy

Burn-in always creates a new media file. It maps all non-attached video
streams and all audio streams. The subtitle filter is applied only to the
first mapped video stream; secondary video streams remain stream-copied.
Audio remains stream-copied. Format metadata and chapters are mapped from the
source.

The caller must select an explicit named strategy:

- H.264 high quality (`libx264`, slow, CRF 18);
- H.265 high quality (`libx265`, slow, CRF 20);
- VP9 high quality;
- AV1 high quality;
- ProRes 422 HQ;
- FFV1 lossless.

Strategies are constrained by the target container and must exist in the
exact FFmpeg encoder inventory. The `subtitles` filter and target muxer must
also exist.

HDR burn-in currently fails closed. Rendering styled SDR subtitles into HDR
pixels without an explicit color-managed composition and visual-QA pipeline
can silently alter transfer characteristics, peak luminance, subtitle
brightness, and metadata. Soft mux or sidecar delivery remains available for
HDR sources.

Burn-in post-QA requires duration tolerance, all source audio/video stream
counts, unchanged audio codecs, chapters, non-empty output, and successful
pre- and post-publication FFprobe checks.

## No fake karaoke

This layer never invents per-word timing. Segment-only "karaoke" themes remain
static visual styles. Actual ASS `\\k`, `\\K`, `\\kf`, or `\\ko` timing tags
are rejected unless a `KaraokeTimingEvidence` object:

- names a forced aligner, native word-timestamp source, or human authoring;
- is marked verified;
- has a positive word count;
- contains the exact SHA-256 of the delivered subtitle bytes.

## Failure and concurrency behavior

- FFmpeg receives `-n`; it cannot overwrite the temporary path.
- The final destination is never passed to FFmpeg.
- Final publication uses atomic hard-link creation rather than `replace`.
- A concurrently created final destination causes `output-exists`.
- Temporary artifacts are removed on every handled failure.
- A post-publication failure rolls back only the artifact whose size, mtime,
  and SHA-256 still match the artifact created by this delivery.
- If the filesystem cannot provide same-directory hard-link publication, the
  operation fails closed.

## Dependency injection and tests

`SubtitleDeliveryExecutor` accepts:

- an injected `DeliveryRunner`;
- an injected trusted media probe;
- configurable execution and capability limits;
- an alternate FFmpeg argv prefix.

`tests/test_subtitle_delivery.py` uses only fake runners and fake probe
evidence. It does not require FFmpeg or FFprobe to be installed and does not
encode real media.

## Remaining system-level gates

This execution boundary does not replace:

- representative-frame subtitle visual QA;
- packaged and license-verified font fallback;
- device-specific hardware encoder qualification;
- color-managed HDR subtitle rendering;
- end-to-end real-media acceptance on each supported platform and packaged
  FFmpeg build.

Those gates must remain separate and may reject an artifact even after this
delivery receipt passes.
