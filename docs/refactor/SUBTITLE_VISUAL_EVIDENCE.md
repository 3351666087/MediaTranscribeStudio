# Subtitle Visual Evidence

`backend/subtitle_visual_evidence.py` collects real, auditable representative-frame evidence for `backend/subtitle_visual_qa.py`. Its output is governed by `contracts/subtitle-render-evidence.schema.json`; `SubtitleRenderEvidenceResult.qa_request` is the exact request object intended for the visual-QA evaluator.

## Trust boundary

- The request supplies separate, absolute paths for the immutable source media and the already subtitle-rendered media. They must not alias.
- FFmpeg and FFprobe are explicit, absolute, injected executable paths. Processes receive argument vectors, run with `shell=False`, have stdin disabled, and are constrained by strict time and stdout/stderr limits.
- The collector snapshots and hashes both media inputs before collection and verifies them again afterward. Any mutation fails closed.
- Extracted frames and process diagnostics live in a temporary directory beside the rendered media. The directory is removed after success or failure.
- Representative timestamps are selected deterministically from each cue. FFprobe resolves real frame times; the collector does not invent frame timing.

## Pixel and font evidence

The default `PillowFrameAnalyzer` compares source and rendered PNG frames and records frame SHA-256, dimensions, resolved time, subtitle box, visible-ink measurements, dark/light background samples, and contrast observations. If Pillow or image analysis is unavailable or incomplete, collection fails closed.

Font evidence is supplied through an injected `FontEvidenceProvider`. A positive installation, resolution, embedding, or glyph claim is accepted only when it points to real local evidence and font files that the collector can hash itself. `NoFontEvidenceProvider` makes no positive font claim. The collector never installs fonts and never treats an unverified name or caller-provided digest as proof.

Word timing is pass-through evidence only. Accepted timings must declare a real source such as a forced aligner, native word timestamps, or human authorship. Interpolated and synthetic evenly split timings are rejected; missing timings are never fabricated.

## Integrity chain

Canonical JSON and SHA-256 bindings cover:

1. the normalized collection request;
2. deterministic frame selection;
3. source/rendered media and extracted frames;
4. cue, speaker, style, frame, analysis-component, and font-evidence relationships;
5. the exact visual-QA request;
6. the final evidence envelope.

Use `verify_evidence_artifact_hash()` before accepting a persisted envelope. Submit `result.qa_request` to `evaluate_subtitle_visual_qa`; do not reconstruct it from partial evidence.

## Testing without FFmpeg

`SubtitleVisualEvidenceCollector` accepts injected runner, frame analyzer, and font-evidence provider implementations. The focused pytest suite uses deterministic fakes, so contract, safety, binding, cleanup, and fail-closed behavior are testable without invoking a real FFmpeg installation. Separate tests cover `BoundedEvidenceRunner` process semantics.
