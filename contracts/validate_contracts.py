"""Validate contract JSON syntax and deterministic cross-field invariants.

This validator deliberately uses only the Python standard library so that it
can run in the offline media worker before optional dependencies are installed.
Java and Python runtime consumers still perform their own full schema binding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "1.0.0"
EXPECTED_FACETS = [
    "AESTHETIC-COHERENCE",
    "AESTHETIC-DISTINCTION",
    "AESTHETIC-REFINEMENT",
    "AESTHETIC-PROPORTION",
    "AESTHETIC-HIERARCHY",
    "AESTHETIC-TYPOGRAPHY",
    "AESTHETIC-COLOR-RELATIONSHIPS",
    "AESTHETIC-RHYTHM",
    "AESTHETIC-DENSITY",
    "AESTHETIC-RESTRAINT",
    "AESTHETIC-REAL-CONTENT-STRESS",
    "AESTHETIC-FONT-FAILURE",
    "AESTHETIC-IMAGE-FAILURE",
    "AESTHETIC-SCRIPT-FAILURE",
]


class ContractError(ValueError):
    """Raised when a portable contract invariant is violated."""


def canonical_speaker_ids(count: int) -> list[str]:
    if count < 1:
        raise ContractError("speaker count must be at least 1")
    return [f"speaker-{index}" for index in range(1, count + 1)]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"{path}: root must be an object")
    return value


def validate_schema_documents() -> None:
    schema_paths = sorted(ROOT.glob("*.schema.json"))
    if len(schema_paths) != 7:
        raise ContractError(f"expected 7 schemas, found {len(schema_paths)}")
    for path in schema_paths:
        schema = load_json(path)
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise ContractError(f"{path.name}: unexpected meta-schema")
        if not str(schema.get("$id", "")).endswith(f"/{SCHEMA_VERSION}"):
            raise ContractError(f"{path.name}: $id must end in /{SCHEMA_VERSION}")
        if schema.get("type") != "object":
            raise ContractError(f"{path.name}: root type must be object")


def validate_report_document(document: dict[str, Any]) -> None:
    if document.get("schemaVersion") != SCHEMA_VERSION:
        raise ContractError("unsupported ReportDocument schemaVersion")
    policy = document.get("speakerPolicy") or {}
    mode = str(policy.get("mode") or "")
    if mode not in {"auto", "manual", "hybrid"}:
        raise ContractError("speakerPolicy.mode must be auto, manual, or hybrid")
    resolved_count = int(policy.get("resolvedCount") or 0)
    expected_speakers = canonical_speaker_ids(resolved_count)
    speaker_ids = list(policy.get("speakerIds") or [])
    if speaker_ids != expected_speakers:
        raise ContractError(
            "speakerPolicy.speakerIds must be the contiguous ordered "
            "speaker-1..speaker-N set"
        )
    requested_count = policy.get("requestedCount")
    if mode == "manual":
        if requested_count is None or int(requested_count) != resolved_count:
            raise ContractError(
                "manual speakerPolicy must request and resolve the same count"
            )
    minimum_count = policy.get("minimumCount")
    maximum_count = policy.get("maximumCount")
    if minimum_count is not None and resolved_count < int(minimum_count):
        raise ContractError("resolved speaker count is below minimumCount")
    if maximum_count is not None and resolved_count > int(maximum_count):
        raise ContractError("resolved speaker count exceeds maximumCount")
    if (
        minimum_count is not None
        and maximum_count is not None
        and int(minimum_count) > int(maximum_count)
    ):
        raise ContractError("minimumCount cannot exceed maximumCount")
    detection = policy.get("detection")
    if mode in {"auto", "hybrid"}:
        if not isinstance(detection, dict):
            raise ContractError(f"{mode} speakerPolicy requires detection evidence")
        if int(detection.get("estimatedCount") or 0) != resolved_count:
            raise ContractError(
                "speaker count detection must match the resolved count"
            )
    speakers = document.get("speakers") or []
    actual_speaker_ids = [
        item.get("id") for item in speakers if isinstance(item, dict)
    ]
    if actual_speaker_ids != expected_speakers:
        raise ContractError(
            "speakers must preserve the contiguous canonical speaker order"
        )
    if [item.get("order") for item in speakers] != list(
        range(1, resolved_count + 1)
    ):
        raise ContractError("speaker order must be contiguous from 1")
    duration_ms = int((document.get("source") or {}).get("durationMs") or 0)
    if duration_ms <= 0:
        raise ContractError("source.durationMs must be positive")

    seen: set[str] = set()
    previous_start = -1
    for index, segment in enumerate(document.get("segments") or []):
        if not isinstance(segment, dict):
            raise ContractError(f"segments[{index}] must be an object")
        segment_id = str(segment.get("id") or "")
        if not segment_id or segment_id in seen:
            raise ContractError(f"segments[{index}] has missing or duplicate id")
        seen.add(segment_id)
        start_ms = int(segment.get("startMs") or 0)
        end_ms = int(segment.get("endMs") or 0)
        if start_ms < previous_start:
            raise ContractError(f"{segment_id}: startMs is not monotonic")
        if end_ms <= start_ms or end_ms > duration_ms:
            raise ContractError(f"{segment_id}: invalid or out-of-range boundary")
        if segment.get("speakerId") not in expected_speakers:
            raise ContractError(f"{segment_id}: invalid speakerId")
        for field in ("rawText", "normalizedText", "displayText"):
            if not str(segment.get(field) or "").strip():
                raise ContractError(f"{segment_id}: {field} must not be empty")
        speaker_evidence = ((segment.get("evidence") or {}).get("speaker") or {})
        if speaker_evidence.get("assignment") != segment.get("speakerId"):
            raise ContractError(f"{segment_id}: acoustic assignment is not traceable")
        scores = speaker_evidence.get("scores") or []
        score_ids = [
            score.get("speakerId") for score in scores if isinstance(score, dict)
        ]
        if score_ids != expected_speakers:
            raise ContractError(
                f"{segment_id}: speaker evidence must score every resolved "
                "speaker in canonical order"
            )
        if segment["rawText"] != segment["normalizedText"] and not segment.get("revisions"):
            raise ContractError(f"{segment_id}: normalized text changed without revision")
        previous_start = start_ms


def validate_render_request(request: dict[str, Any]) -> None:
    quality = request.get("qualityPolicy") or {}
    if quality.get("facetIds") != EXPECTED_FACETS:
        raise ContractError("qualityPolicy.facetIds must preserve stable Design Pack ordering")
    score = float(quality.get("minimumScore") or 0)
    if score < 85:
        raise ContractError("minimum PDF quality score cannot be below 85")
    rounds = int(quality.get("maxRounds") or 0)
    if rounds < 1 or rounds > 5:
        raise ContractError("maxRounds must be between 1 and 5")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--document",
        type=Path,
        default=ROOT / "examples" / "synthetic-report-document.json",
    )
    parser.add_argument("--render-request", type=Path)
    args = parser.parse_args()

    validate_schema_documents()
    validate_report_document(load_json(args.document))
    if args.render_request:
        validate_render_request(load_json(args.render_request))
    print(f"validated 7 schemas and {args.document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
