"""Export hash-bound SRT, WebVTT, and ASS previews from a sample transcript."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import replace as replace_dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.output_orchestration import transcript_subtitle_segments
from backend.persistence import (
    atomic_write_json,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)
from backend.subtitles import (
    CuePolicy,
    SubtitleFormat,
    SubtitleSegment,
    arrange_cues,
    audit_cues,
    export_subtitles,
)


def _sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return normalized


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return number


def _assert_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def export_sample_subtitles(
    transcript_path: Path,
    output_root: Path,
    *,
    include_speaker_labels: bool = False,
    review_queue_path: Path | None = None,
    expected_transcript_sha256: str | None = None,
    expected_review_queue_sha256: str | None = None,
    expected_open_review_items: int | None = None,
    replace: bool = False,
) -> list[Path]:
    transcript_path = transcript_path.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    transcript_hash = sha256_file(transcript_path)
    if expected_transcript_sha256 is not None:
        transcript_hash = _assert_hash(
            transcript_path, expected_transcript_sha256, "transcript"
        )
    document = read_json_strict(transcript_path)
    open_review_items: int | None = None
    review_hash: str | None = None
    resolved_review_path: Path | None = None
    if review_queue_path is not None:
        resolved_review_path = review_queue_path.expanduser().resolve(strict=True)
        review_hash = sha256_file(resolved_review_path)
        if expected_review_queue_sha256 is not None:
            review_hash = _assert_hash(
                resolved_review_path,
                expected_review_queue_sha256,
                "review queue",
            )
        review_queue = read_json_strict(resolved_review_path)
        if review_queue.get("jobId") != document.get("jobId"):
            raise RuntimeError("review queue jobId does not match transcript")
        open_review_items = sum(
            1
            for item in review_queue.get("items", [])
            if isinstance(item, Mapping) and item.get("status") == "open"
        )
        if (
            expected_open_review_items is not None
            and open_review_items != expected_open_review_items
        ):
            raise RuntimeError(
                "open review count mismatch: "
                f"expected {expected_open_review_items}, observed {open_review_items}"
            )
    segments = tuple(
        SubtitleSegment.from_value(segment)
        for segment in transcript_subtitle_segments(document)
    )
    duration_ms = document.get("source", {}).get("durationMs")
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms < 1
    ):
        raise RuntimeError("transcript source duration is invalid")
    configured_policy = CuePolicy(include_speaker_labels=include_speaker_labels)
    source_segments_are_monotonic = all(
        right.start_ms >= left.end_ms
        for left, right in zip(segments, segments[1:], strict=False)
    )
    configured_arrangement = arrange_cues(
        segments,
        policy=configured_policy,
    )
    source_bound_fallback_applied = (
        source_segments_are_monotonic
        and configured_arrangement.cues[-1].end_ms > duration_ms
    )
    effective_policy = (
        replace_dataclass(
            configured_policy,
            gap_ms=0,
            max_reading_speed=100.0,
        )
        if source_bound_fallback_applied
        else configured_policy
    )
    arrangement = (
        arrange_cues(segments, policy=effective_policy)
        if source_bound_fallback_applied
        else configured_arrangement
    )
    configured_policy_qa = audit_cues(
        arrangement.cues,
        policy=configured_policy,
        source_segments=segments,
        repairs=arrangement.qa.repairs,
    )
    if not arrangement.qa.passed or not arrangement.qa.source_text_preserved:
        raise RuntimeError("subtitle cue QA failed")
    if any(
        right.start_ms < left.end_ms
        for left, right in zip(arrangement.cues, arrangement.cues[1:], strict=False)
    ):
        raise RuntimeError("subtitle cue times are not monotonic")
    if arrangement.cues[-1].end_ms > duration_ms:
        raise RuntimeError("subtitle cues exceed the persisted source duration")

    output_root.mkdir(parents=True, exist_ok=True)
    stem = str(document.get("jobId") or transcript_path.stem)
    managed_paths = (
        output_root / f"{stem}.srt",
        output_root / f"{stem}.vtt",
        output_root / f"{stem}.ass",
        output_root / "unapproved-subtitle-preview-acceptance.v1.json",
    )
    if not replace and any(path.exists() for path in managed_paths):
        raise RuntimeError("output exists; pass --replace to regenerate it")
    paths: list[Path] = []
    artifacts: list[dict[str, object]] = []
    for subtitle_format in (
        SubtitleFormat.SRT,
        SubtitleFormat.WEBVTT,
        SubtitleFormat.ASS,
    ):
        suffix = ".vtt" if subtitle_format is SubtitleFormat.WEBVTT else (
            f".{subtitle_format.value}"
        )
        path = output_root / f"{stem}{suffix}"
        path.write_text(
            export_subtitles(arrangement, subtitle_format),
            encoding="utf-8",
        )
        path.read_text(encoding="utf-8", errors="strict")
        paths.append(path)
        artifacts.append(
            {
                "format": subtitle_format.value,
                "path": str(path),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "utf8": True,
            }
        )

    manifest = {
        "schemaVersion": "1.0.0",
        "artifactType": "unapproved-subtitle-preview-acceptance",
        "applicationPolicy": "suggestion-only",
        "requiresHumanApproval": True,
        "releaseApproved": False,
        "source": {
            "jobId": document.get("jobId"),
            "transcriptPath": str(transcript_path),
            "transcriptSha256": transcript_hash,
            "transcriptCanonicalSha256": canonical_json_sha256(document),
            "reviewQueuePath": (
                str(resolved_review_path) if resolved_review_path is not None else None
            ),
            "reviewQueueSha256": review_hash,
            "openReviewItems": open_review_items,
            "durationMs": duration_ms,
        },
        "cueQa": {
            "passed": True,
            "cueCount": arrangement.qa.cue_count,
            "sourceSegmentCount": len(segments),
            "sourceTextPreserved": True,
            "monotonicNonoverlapping": True,
            "withinSourceDuration": True,
            "firstCueStartMs": arrangement.cues[0].start_ms,
            "lastCueEndMs": arrangement.cues[-1].end_ms,
            "maximumObservedReadingSpeed": (
                arrangement.qa.maximum_observed_reading_speed
            ),
            "repairs": list(arrangement.qa.repairs),
            "speakerLabelsIncluded": include_speaker_labels,
            "timingPolicy": {
                "configuredGapMs": configured_policy.gap_ms,
                "effectiveGapMs": effective_policy.gap_ms,
                "configuredMaxReadingSpeed": (
                    configured_policy.max_reading_speed
                ),
                "effectiveMaxReadingSpeed": effective_policy.max_reading_speed,
                "sourceSegmentsMonotonicNonoverlapping": (
                    source_segments_are_monotonic
                ),
                "zeroGapApplied": effective_policy.gap_ms == 0,
                "sourceBoundFallbackApplied": source_bound_fallback_applied,
                "configuredPolicyQaPassed": configured_policy_qa.passed,
                "configuredPolicyIssues": [
                    {
                        "code": issue.code,
                        "message": issue.message,
                        "cueNumber": issue.cue_number,
                    }
                    for issue in configured_policy_qa.issues
                ],
            },
        },
        "artifacts": artifacts,
        "qualityBoundary": {
            "technicalSubtitlePreviewPassed": True,
            "visualQaPassed": False,
            "linguisticQualityPassed": False,
            "speakerQualityPassed": False,
            "releaseApproved": False,
            "reason": "PREVIEW_ONLY_AND_SOURCE_REQUIRES_REVIEW",
        },
    }
    manifest_path = output_root / "unapproved-subtitle-preview-acceptance.v1.json"
    atomic_write_json(manifest_path, manifest)
    paths.append(manifest_path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-speaker-labels", action="store_true")
    parser.add_argument("--review-queue", type=Path)
    parser.add_argument("--expected-transcript-sha256", type=_sha256)
    parser.add_argument("--expected-review-queue-sha256", type=_sha256)
    parser.add_argument("--expected-open-review-items", type=_nonnegative_int)
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = export_sample_subtitles(
        args.transcript,
        args.output_root,
        include_speaker_labels=args.include_speaker_labels,
        review_queue_path=args.review_queue,
        expected_transcript_sha256=args.expected_transcript_sha256,
        expected_review_queue_sha256=args.expected_review_queue_sha256,
        expected_open_review_items=args.expected_open_review_items,
        replace=args.replace,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
