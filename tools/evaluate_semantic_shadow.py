"""Score validated semantic suggestions on an isolated transcript copy."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import (
    atomic_write_json,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)
from backend.semantic_processing import validate_semantic_suggestions_artifact
from tools.evaluate_sample_library import (
    _content_integrity_quality,
    _diarization_quality,
    _factual_integrity_quality,
    _final_language_quality,
    _joint_transcription_quality,
    _scoring_unit,
)
from tools.sample_library import word_error_rate


_DELTA_METRICS = (
    ("speakerCount.absoluteError", ("speakerCount", "speakerCountAbsoluteError"), "min"),
    ("diarization.der", ("diarization", "der"), "min"),
    ("diarization.jer", ("diarization", "jer"), "min"),
    (
        "diarization.speakerConfusion",
        ("diarization", "speakerConfusion"),
        "min",
    ),
    ("diarization.overlapF1", ("diarization", "overlapF1"), "max"),
    (
        "boundary.meanAbsoluteErrorMs",
        ("boundary", "meanAbsoluteErrorMs"),
        "min",
    ),
    (
        "boundary.p95AbsoluteErrorMs",
        ("boundary", "p95AbsoluteErrorMs"),
        "min",
    ),
    ("finalText.werOrCer", ("finalText", "werOrCer"), "min"),
    (
        "jointTranscription.cpWer",
        ("jointTranscription", "cpWer", "errorRate"),
        "min",
    ),
    (
        "jointTranscription.tcpWer",
        ("jointTranscription", "tcpWer", "errorRate"),
        "min",
    ),
    (
        "jointTranscription.speakerAttributedWer",
        ("jointTranscription", "speakerAttributedWer", "errorRate"),
        "min",
    ),
    (
        "language.durationWeightedAccuracy",
        ("language", "durationWeighted", "accuracy"),
        "max",
    ),
    (
        "language.lexicalTokenWeightedAccuracy",
        ("language", "lexicalTokenWeighted", "accuracy"),
        "max",
    ),
    (
        "codeSwitch.switchPointMeanAbsoluteErrorMs",
        ("codeSwitch", "switchPointMeanAbsoluteErrorMs"),
        "min",
    ),
    (
        "contentIntegrity.tokenErrorRate",
        ("contentIntegrity", "tokenErrorRate"),
        "min",
    ),
    (
        "contentIntegrity.hallucinationRate",
        ("contentIntegrity", "hallucinationRate"),
        "min",
    ),
    (
        "contentIntegrity.deletionRate",
        ("contentIntegrity", "deletionRate"),
        "min",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--semantic-artifact", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser


def _load_case(manifest: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest cases must be an array")
    matches = [
        dict(case)
        for case in cases
        if isinstance(case, Mapping) and case.get("id") == case_id
    ]
    if len(matches) != 1:
        raise ValueError("case ID must resolve exactly once in the manifest")
    return matches[0]


def _segment_map(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_segments = document.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("transcript must contain segments")
    segments: dict[str, dict[str, Any]] = {}
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            raise ValueError("transcript segments must be objects")
        segment = copy.deepcopy(dict(raw))
        segment_id = segment.get("id")
        if (
            not isinstance(segment_id, str)
            or not segment_id
            or segment_id in segments
        ):
            raise ValueError("transcript segment IDs must be non-empty and unique")
        segments[segment_id] = segment
    return segments


def apply_shadow_suggestions(
    document: Mapping[str, Any],
    semantic_artifact: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply only validator-approved suggestions without mutating production state."""

    transcript_hash = canonical_json_sha256(document)
    job_id = document.get("jobId")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("transcript jobId must be non-empty")
    artifact = validate_semantic_suggestions_artifact(
        semantic_artifact,
        expected_job_id=job_id,
        expected_transcript_sha256=transcript_hash,
    )
    metrics = artifact.get("metrics")
    if (
        artifact.get("status") != "completed"
        or not isinstance(metrics, Mapping)
        or any(
            metrics.get(field) != 0
            for field in (
                "rejectionCount",
                "failureCount",
                "unresolvedSegmentCount",
                "autoAppliedCount",
            )
        )
    ):
        raise ValueError(
            "semantic shadow evaluation requires a complete, unresolved-free "
            "suggestion artifact"
        )

    baseline = _segment_map(document)
    shadow = copy.deepcopy(baseline)
    policy = document.get("speakerPolicy")
    speaker_ids = (
        policy.get("speakerIds") if isinstance(policy, Mapping) else None
    )
    if (
        not isinstance(speaker_ids, list)
        or not speaker_ids
        or any(not isinstance(item, str) or not item for item in speaker_ids)
    ):
        raise ValueError("transcript canonical speaker namespace is invalid")
    allowed_speakers = set(speaker_ids)
    applied: list[dict[str, Any]] = []
    seen_segments: set[str] = set()
    suggestions = artifact.get("suggestions")
    if not isinstance(suggestions, list):
        raise ValueError("semantic suggestions must be an array")
    for suggestion in suggestions:
        if not isinstance(suggestion, Mapping):
            raise ValueError("semantic suggestions must be objects")
        segment_id = suggestion.get("segmentId")
        changes = suggestion.get("changes")
        proposal = suggestion.get("proposal")
        validation = suggestion.get("validation")
        if (
            not isinstance(segment_id, str)
            or segment_id not in shadow
            or segment_id in seen_segments
            or not isinstance(changes, list)
            or not changes
            or not isinstance(proposal, Mapping)
            or not isinstance(validation, Mapping)
            or validation.get("deterministicDecision") != "suggest"
        ):
            raise ValueError("semantic suggestion is not shadow-applicable")
        seen_segments.add(segment_id)
        segment = shadow[segment_id]
        before = baseline[segment_id]
        if segment.get("humanLocked") is True:
            raise ValueError("semantic shadow cannot override a human lock")

        applied_changes: list[str] = []
        if "speaker" in changes:
            target = proposal.get("targetSpeakerId")
            if not isinstance(target, str) or target not in allowed_speakers:
                raise ValueError("semantic target speaker is outside the namespace")
            segment["speakerId"] = target
            applied_changes.append("speaker")
        if "text" in changes:
            text_patch = suggestion.get("textPatch")
            after = (
                text_patch.get("after")
                if isinstance(text_patch, Mapping)
                else None
            )
            if not isinstance(after, str) or not after.strip():
                raise ValueError("semantic text suggestion is invalid")
            segment["normalizedText"] = after.strip()
            segment["displayText"] = after.strip()
            applied_changes.append("text")

        for immutable in ("id", "startMs", "endMs", "rawText", "language"):
            if segment.get(immutable) != before.get(immutable):
                raise ValueError(f"semantic shadow changed immutable {immutable}")
        applied.append(
            {
                "suggestionId": suggestion.get("id"),
                "segmentId": segment_id,
                "changes": applied_changes,
                "sourceSpeakerId": before.get("speakerId"),
                "targetSpeakerId": segment.get("speakerId"),
                "textChanged": (
                    segment.get("normalizedText")
                    != before.get("normalizedText")
                ),
            }
        )

    ordered_ids = [
        str(segment["id"])
        for segment in document["segments"]
        if isinstance(segment, Mapping)
    ]
    return [shadow[segment_id] for segment_id in ordered_ids], applied


def _redact_factual_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"requiredLiterals", "forbiddenLiterals"}
    }


def score_shadow_state(
    *,
    case: Mapping[str, Any],
    document: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    review_open_count: int | None,
) -> dict[str, Any]:
    normalized = [dict(segment) for segment in segments]
    source = document.get("source")
    policy = document.get("speakerPolicy")
    if not isinstance(source, Mapping) or not isinstance(policy, Mapping):
        raise ValueError("transcript source and speaker policy are required")
    duration_ms = source.get("durationMs")
    resolved_count = policy.get("resolvedCount")
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms < 1
        or isinstance(resolved_count, bool)
        or not isinstance(resolved_count, int)
        or resolved_count < 1
    ):
        raise ValueError("transcript duration or resolved speaker count is invalid")

    eligibility = case.get("truthEligibility")
    expected_count = case.get("expectedSpeakerCount")
    count_eligible = (
        isinstance(expected_count, int)
        and not isinstance(expected_count, bool)
        and expected_count > 0
        and (
            not isinstance(eligibility, Mapping)
            or eligibility.get("speakerCount") is not False
        )
    )
    assigned = {str(segment.get("speakerId")) for segment in normalized}
    reference = str(
        case.get("scoringTranscript")
        or case.get("expectedTranscript")
        or ""
    )
    text_eligible = bool(reference.strip()) and (
        not isinstance(eligibility, Mapping)
        or eligibility.get("asr") is not False
    )
    final_text = " ".join(
        str(segment.get("normalizedText") or "").strip()
        for segment in normalized
    ).strip()
    final_segments = [
        {
            **segment,
            "finalText": str(segment.get("normalizedText") or "").strip(),
        }
        for segment in normalized
    ]
    diarization, boundary = _diarization_quality(
        dict(case),
        normalized,
        speaker_timeline=None,
    )
    joint = _joint_transcription_quality(
        case=case,
        transcript={"source": dict(source)},
        segments=normalized,
        speaker_timeline=None,
        text_field="normalizedText",
    )
    language, code_switch = _final_language_quality(
        case=case,
        segments=final_segments,
        duration_ms=duration_ms,
    )
    factual = _redact_factual_metrics(
        _factual_integrity_quality(case=case, final_text=final_text)
    )
    return {
        "speakerCount": {
            "eligible": count_eligible,
            "expectedSpeakerCount": expected_count,
            "resolvedSpeakerCount": resolved_count,
            "distinctAssignedSpeakerCount": len(assigned),
            "speakerCountAbsoluteError": (
                abs(resolved_count - expected_count)
                if count_eligible
                else None
            ),
            "speakerCountMatch": (
                resolved_count == expected_count
                and len(assigned) == expected_count
                if count_eligible
                else None
            ),
        },
        "diarization": diarization,
        "boundary": boundary,
        "finalText": {
            "referenceAvailable": text_eligible,
            "werOrCer": (
                word_error_rate(reference, final_text)
                if text_eligible
                else None
            ),
            "scoringUnit": (
                _scoring_unit([reference]) if text_eligible else None
            ),
            "hypothesisAuthority": "normalizedText-shadow",
        },
        "jointTranscription": joint,
        "language": language,
        "codeSwitch": code_switch,
        "factualIntegrity": factual,
        "contentIntegrity": _content_integrity_quality(
            reference_text=reference,
            hypothesis_text=final_text,
            eligible=text_eligible,
        ),
        "review": {
            "openCount": review_open_count,
            "shadowResolutionsApplied": 0,
            "humanApprovalSimulated": False,
        },
    }


def _path_value(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for field in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(field)
    return current


def compare_shadow_metrics(
    baseline: Mapping[str, Any],
    shadow: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    comparisons: list[dict[str, Any]] = []
    for metric_id, path, objective in _DELTA_METRICS:
        before = _path_value(baseline, path)
        after = _path_value(shadow, path)
        if (
            isinstance(before, bool)
            or not isinstance(before, (int, float))
            or isinstance(after, bool)
            or not isinstance(after, (int, float))
            or not math.isfinite(float(before))
            or not math.isfinite(float(after))
        ):
            outcome = "not-scored"
            delta = None
        else:
            delta = float(after) - float(before)
            if math.isclose(delta, 0.0, abs_tol=1e-12):
                outcome = "unchanged"
            elif (objective == "min" and delta < 0) or (
                objective == "max" and delta > 0
            ):
                outcome = "improved"
            else:
                outcome = "regressed"
        comparisons.append(
            {
                "metric": metric_id,
                "objective": objective,
                "baseline": before,
                "shadow": after,
                "delta": delta,
                "outcome": outcome,
            }
        )
    outcomes = {item["outcome"] for item in comparisons}
    overall = (
        "regressed"
        if "regressed" in outcomes
        else "improved"
        if "improved" in outcomes
        else "unchanged"
    )
    return comparisons, overall


def build_shadow_report(
    *,
    case: Mapping[str, Any],
    document: Mapping[str, Any],
    semantic_artifact: Mapping[str, Any],
    review_open_count: int | None = None,
) -> dict[str, Any]:
    baseline_segments = list(_segment_map(document).values())
    shadow_segments, applied = apply_shadow_suggestions(
        document,
        semantic_artifact,
    )
    baseline = score_shadow_state(
        case=case,
        document=document,
        segments=baseline_segments,
        review_open_count=review_open_count,
    )
    shadow = score_shadow_state(
        case=case,
        document=document,
        segments=shadow_segments,
        review_open_count=review_open_count,
    )
    comparisons, overall = compare_shadow_metrics(baseline, shadow)
    semantic_metrics = semantic_artifact.get("metrics")
    if not isinstance(semantic_metrics, Mapping):
        raise ValueError("semantic artifact metrics are missing")
    blockers = [
        "shadow-evaluation-is-not-production-adjudication",
        "human-review-not-simulated",
        "frozen-production-thresholds-not-evaluated",
    ]
    if overall == "unchanged":
        blockers.append("no-measured-semantic-gain")
    elif overall == "regressed":
        blockers.append("semantic-hard-domain-regression")
    return {
        "schemaVersion": "1.0.0",
        "evaluationType": "semantic-shadow-apply",
        "productionAdjudication": False,
        "releaseApproved": False,
        "case": {
            "id": case.get("id"),
            "evaluationSplit": case.get("evaluationSplit"),
            "language": case.get("language"),
            "expectedSpeakerCount": case.get("expectedSpeakerCount"),
        },
        "policy": {
            "truthVisibleToSemanticModel": False,
            "applyAllDeterministicallyValidatedSuggestions": True,
            "simulateHumanApproval": False,
            "mutateProductionTranscript": False,
            "nonCompensatingDomains": True,
        },
        "semantic": {
            "model": semantic_artifact.get("model"),
            "promptVersion": semantic_artifact.get("promptVersion"),
            "status": semantic_artifact.get("status"),
            "metrics": dict(semantic_metrics),
            "appliedSuggestionCount": len(applied),
            "appliedSuggestions": applied,
        },
        "baselineMetrics": baseline,
        "shadowMetrics": shadow,
        "metricComparisons": comparisons,
        "overallOutcome": overall,
        "promotionEligible": False,
        "blockingReasons": blockers,
        "transcriptTextPersisted": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.replace:
        raise RuntimeError("refusing to overwrite semantic shadow evidence")
    manifest_path = args.manifest.expanduser().resolve()
    transcript_path = args.transcript.expanduser().resolve()
    semantic_path = args.semantic_artifact.expanduser().resolve()
    manifest = read_json_strict(manifest_path)
    document = read_json_strict(transcript_path)
    semantic = read_json_strict(semantic_path)
    if not all(
        isinstance(value, Mapping)
        for value in (manifest, document, semantic)
    ):
        raise ValueError("manifest, transcript, and semantic artifact must be objects")
    review_open_count: int | None = None
    review_path: Path | None = None
    if args.review_queue is not None:
        review_path = args.review_queue.expanduser().resolve()
        review = read_json_strict(review_path)
        raw_open_count = review.get("openCount") if isinstance(review, Mapping) else None
        if (
            isinstance(raw_open_count, bool)
            or not isinstance(raw_open_count, int)
            or raw_open_count < 0
        ):
            raise ValueError("review queue openCount is invalid")
        review_open_count = raw_open_count
    case = _load_case(manifest, args.case_id)
    report = build_shadow_report(
        case=case,
        document=document,
        semantic_artifact=semantic,
        review_open_count=review_open_count,
    )
    report["evidence"] = {
        "manifest": {
            "path": str(manifest_path),
            "fileSha256": sha256_file(manifest_path),
            "canonicalSha256": canonical_json_sha256(manifest),
        },
        "transcript": {
            "path": str(transcript_path),
            "fileSha256": sha256_file(transcript_path),
            "canonicalSha256": canonical_json_sha256(document),
        },
        "semanticArtifact": {
            "path": str(semantic_path),
            "fileSha256": sha256_file(semantic_path),
            "canonicalSha256": canonical_json_sha256(semantic),
        },
        "reviewQueue": (
            {
                "path": str(review_path),
                "fileSha256": sha256_file(review_path),
            }
            if review_path is not None
            else None
        ),
    }
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
