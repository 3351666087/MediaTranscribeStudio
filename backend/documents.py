"""Dynamic-cardinality transcript validation, review derivation, and assembly."""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts.pyannote_evidence import is_verified_pyannote_speaker_revision

from .errors import WorkerError
from .language import MULTIPLE_LANGUAGES, UNDETERMINED_LANGUAGE, normalize_language_tag
from .models import (
    SpeakerCountEstimate,
    SpeakerCountMode,
    SpeakerCountPolicy,
    TranscriptSegment,
    TranscriptionResult,
    canonical_speaker_ids,
)
from .persistence import sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_speaker_count(
    policy: SpeakerCountPolicy,
    result: TranscriptionResult,
) -> tuple[int, SpeakerCountEstimate]:
    if policy.mode is SpeakerCountMode.MANUAL:
        assert policy.manual_count is not None
        adapter_estimate = result.speaker_count_estimate
        if (
            adapter_estimate is not None
            and adapter_estimate.estimated_count != policy.manual_count
        ):
            raise WorkerError(
                "MANUAL_SPEAKER_COUNT_MISMATCH",
                "adapter estimate conflicts with the manually fixed speaker count",
            )
        return policy.manual_count, SpeakerCountEstimate.manual(policy.manual_count)

    estimate = result.speaker_count_estimate
    if estimate is None:
        raise WorkerError(
            "SPEAKER_COUNT_ESTIMATE_MISSING",
            "auto and hybrid modes require estimatedCount, confidence, and candidateRange",
        )
    if policy.mode is SpeakerCountMode.HYBRID:
        assert policy.minimum is not None and policy.maximum is not None
        if not policy.minimum <= estimate.estimated_count <= policy.maximum:
            raise WorkerError(
                "HYBRID_SPEAKER_COUNT_OUT_OF_BOUNDS",
                "estimated speaker count violates the human bounds",
            )
        if (
            estimate.candidate_min < policy.minimum
            or estimate.candidate_max > policy.maximum
        ):
            raise WorkerError(
                "HYBRID_CANDIDATE_RANGE_OUT_OF_BOUNDS",
                "speaker candidate range violates the human bounds",
            )
    return estimate.estimated_count, estimate


def _speaker_revisions(segment: TranscriptSegment) -> list[Any]:
    return [
        revision
        for revision in segment.revisions
        if revision.revision_type == "speaker"
    ]


def validate_segments(
    segments: tuple[TranscriptSegment, ...],
    *,
    speaker_count: int,
    duration_ms: int,
    high_margin_threshold: float,
) -> None:
    canonical = canonical_speaker_ids(speaker_count)
    canonical_set = set(canonical)
    observed: set[str] = set()
    segment_ids: set[str] = set()
    revision_ids: set[str] = set()
    previous_start = -1
    for segment in segments:
        if segment.segment_id in segment_ids:
            raise WorkerError(
                "TRANSCRIPT_INVARIANT_VIOLATION",
                f"duplicate segment id: {segment.segment_id}",
            )
        segment_ids.add(segment.segment_id)
        if segment.start_ms < previous_start or segment.end_ms > duration_ms:
            raise WorkerError(
                "TRANSCRIPT_INVARIANT_VIOLATION",
                f"{segment.segment_id} has invalid or non-monotonic boundaries",
            )
        previous_start = segment.start_ms
        if segment.speaker_id not in canonical_set:
            raise WorkerError(
                "TRANSCRIPT_INVARIANT_VIOLATION",
                f"{segment.segment_id} speaker is outside the canonical namespace",
            )
        observed.add(segment.speaker_id)
        score_ids = [score.speaker_id for score in segment.speaker_scores]
        if len(score_ids) != speaker_count or set(score_ids) != canonical_set:
            raise WorkerError(
                "INCOMPLETE_SPEAKER_EVIDENCE",
                f"{segment.segment_id} scores must cover every canonical speaker exactly once",
            )
        if len(score_ids) != len(set(score_ids)):
            raise WorkerError(
                "INCOMPLETE_SPEAKER_EVIDENCE",
                f"{segment.segment_id} contains duplicate speaker scores",
            )
        ranked = sorted(
            segment.speaker_scores, key=lambda item: item.score, reverse=True
        )
        calculated_margin = ranked[0].score - ranked[1].score if len(ranked) > 1 else 2.0
        if abs(calculated_margin - segment.speaker_margin) > 1e-6:
            raise WorkerError(
                "SPEAKER_MARGIN_MISMATCH",
                f"{segment.segment_id} speakerMargin does not match acoustic scores",
            )
        revisions = _speaker_revisions(segment)
        verified_pyannote_revisions = {
            revision.revision_id
            for revision in revisions
            if is_verified_pyannote_speaker_revision(
                revision,
                segment.evidence,
                canonical_set,
            )
        }
        for revision in segment.revisions:
            if revision.source == "llm":
                raise WorkerError(
                    "LLM_AUTO_APPLY_FORBIDDEN",
                    f"{segment.segment_id} contains an LLM-authored revision",
                )
            if not revision.evidence_refs:
                raise WorkerError(
                    "REVISION_EVIDENCE_REQUIRED",
                    f"{segment.segment_id} revisions require evidence references",
                )
            if not 0.0 <= revision.confidence <= 1.0:
                raise WorkerError(
                    "REVISION_CONFIDENCE_INVALID",
                    f"{segment.segment_id} revision confidence is invalid",
                )
            if (
                revision.source != "manual"
                and revision.revision_type in {"boundary", "split", "merge"}
            ):
                raise WorkerError(
                    "AUTOMATIC_TURN_MUTATION_FORBIDDEN",
                    f"{segment.segment_id} turn and boundary structure is protected",
                )
            if (
                segment.overlapping
                and revision.source != "manual"
                and revision.revision_type
                in {"speaker", "boundary", "split", "merge"}
                and revision.revision_id not in verified_pyannote_revisions
            ):
                raise WorkerError(
                    "OVERLAP_OVERRIDE_FORBIDDEN",
                    f"{segment.segment_id} overlap/串话 structure requires manual review",
                    details={
                        "segmentId": segment.segment_id,
                        "revisionId": revision.revision_id,
                        "revisionSource": revision.source,
                        "revisionType": revision.revision_type,
                        "reasonCode": revision.reason_code,
                        "verifiedPyannoteRevisionIds": sorted(
                            verified_pyannote_revisions
                        ),
                    },
                )
        non_manual_revisions = [
            item for item in revisions if item.source != "manual"
        ]
        unsafe_non_manual_revisions = [
            item
            for item in non_manual_revisions
            if item.revision_id not in verified_pyannote_revisions
        ]
        if segment.human_locked and non_manual_revisions:
            raise WorkerError(
                "HUMAN_LOCK_OVERRIDE_FORBIDDEN",
                f"{segment.segment_id} is human-locked and cannot be changed automatically",
            )
        if (
            unsafe_non_manual_revisions
            and segment.speaker_margin >= high_margin_threshold
        ):
            raise WorkerError(
                "HIGH_MARGIN_OVERRIDE_FORBIDDEN",
                f"{segment.segment_id} has high-margin voiceprint evidence",
            )
        top_two = {
            ranked[0].speaker_id,
            ranked[1].speaker_id if len(ranked) > 1 else ranked[0].speaker_id,
        }
        effective_speaker = ranked[0].speaker_id
        for revision in revisions:
            if (
                not isinstance(revision.before, str)
                or revision.before not in canonical_set
                or not isinstance(revision.after, str)
                or revision.after not in canonical_set
            ):
                raise WorkerError(
                    "SPEAKER_REVISION_INVALID",
                    f"{segment.segment_id} speaker revisions must use canonical IDs",
                )
            if revision.before != effective_speaker:
                raise WorkerError(
                    "SPEAKER_REVISION_CHAIN_INVALID",
                    f"{segment.segment_id} speaker revision history is not contiguous",
                )
            if revision.after == revision.before:
                raise WorkerError(
                    "SPEAKER_REVISION_INVALID",
                    f"{segment.segment_id} speaker revisions must record a change",
                )
            if (
                revision.source != "manual"
                and revision.after not in top_two
                and revision.revision_id not in verified_pyannote_revisions
            ):
                raise WorkerError(
                    "SEMANTIC_ARBITRATION_OUTSIDE_TOP2",
                    f"{segment.segment_id} automatic arbitration must stay within acoustic top-2",
                )
            effective_speaker = revision.after
        if revisions and effective_speaker != segment.speaker_id:
            raise WorkerError(
                "SPEAKER_REVISION_RESULT_MISMATCH",
                f"{segment.segment_id} final speaker revision must match speakerId",
            )
        if segment.speaker_id != ranked[0].speaker_id and (
            not revisions or effective_speaker != segment.speaker_id
        ):
            raise WorkerError(
                "SPEAKER_ASSIGNMENT_WITHOUT_REVISION",
                f"{segment.segment_id} differs from acoustic top-1 without evidence",
            )
        if (
            segment.normalized_text != segment.raw_text
            and not any(
                revision.revision_type == "text"
                and revision.before == segment.raw_text
                and revision.after in {
                    segment.normalized_text,
                    segment.display_text,
                }
                for revision in segment.revisions
            )
        ):
            raise WorkerError(
                "TEXT_CHANGE_WITHOUT_REVISION",
                f"{segment.segment_id} normalized text changed without a revision",
            )
        if (
            segment.display_text != segment.normalized_text
            and not any(
                revision.revision_type == "text"
                and revision.before
                in {segment.raw_text, segment.normalized_text}
                and revision.after == segment.display_text
                for revision in segment.revisions
            )
        ):
            raise WorkerError(
                "TEXT_CHANGE_WITHOUT_REVISION",
                f"{segment.segment_id} display text changed without a revision",
            )
        text_revisions = [
            revision
            for revision in segment.revisions
            if revision.revision_type == "text"
        ]
        text_cursor = segment.raw_text
        for revision in text_revisions:
            if revision.before != text_cursor:
                raise WorkerError(
                    "TEXT_REVISION_CHAIN_INVALID",
                    f"{segment.segment_id} text revision history is not contiguous",
                )
            if not isinstance(revision.after, str) or not revision.after.strip():
                raise WorkerError(
                    "TEXT_REVISION_INVALID",
                    f"{segment.segment_id} text revisions require non-empty text",
                )
            text_cursor = revision.after
        expected_text = (
            segment.display_text
            if segment.display_text != segment.normalized_text
            else segment.normalized_text
        )
        if text_revisions and text_cursor != expected_text:
            raise WorkerError(
                "TEXT_REVISION_RESULT_MISMATCH",
                f"{segment.segment_id} final text revision does not match display text",
            )
        for revision in segment.revisions:
            if revision.revision_id in revision_ids:
                raise WorkerError(
                    "TRANSCRIPT_INVARIANT_VIOLATION",
                    f"duplicate revision id: {revision.revision_id}",
                )
            revision_ids.add(revision.revision_id)

    if observed != canonical_set:
        raise WorkerError(
            "CANONICAL_SPEAKER_SET_INCOMPLETE",
            "resolved speaker count must equal the fully observed canonical speaker set",
            details={
                "expected": list(canonical),
                "observed": sorted(observed),
            },
        )


def _actual_overlap_segment_ids(
    segments: tuple[TranscriptSegment, ...],
) -> set[str]:
    active_heap: list[tuple[int, str]] = []
    active_ids: set[str] = set()
    overlapping_ids: set[str] = set()
    for segment in segments:
        while active_heap and active_heap[0][0] <= segment.start_ms:
            _, segment_id = heapq.heappop(active_heap)
            active_ids.discard(segment_id)
        if active_ids:
            overlapping_ids.add(segment.segment_id)
            overlapping_ids.update(active_ids)
        heapq.heappush(active_heap, (segment.end_ms, segment.segment_id))
        active_ids.add(segment.segment_id)
    return overlapping_ids


def build_review_queue(
    *,
    job_id: str,
    policy: SpeakerCountPolicy,
    estimate: SpeakerCountEstimate,
    segments: tuple[TranscriptSegment, ...],
    count_confidence_threshold: float,
    segment_confidence_threshold: float,
    speaker_margin_threshold: float,
    range_width_threshold: int,
) -> dict[str, Any]:
    created_at = utc_now()
    items: list[dict[str, Any]] = []
    actual_overlap_ids = _actual_overlap_segment_ids(segments)
    if (
        policy.mode is not SpeakerCountMode.MANUAL
        and estimate.confidence < count_confidence_threshold
    ):
        items.append(
            {
                "id": "speaker-count-confidence",
                "scope": "job",
                "reasonCode": "SPEAKER_COUNT_LOW_CONFIDENCE",
                "status": "open",
                "speakerCountEstimate": estimate.as_dict(),
            }
        )
    if (
        policy.mode is not SpeakerCountMode.MANUAL
        and estimate.candidate_max - estimate.candidate_min > range_width_threshold
    ):
        items.append(
            {
                "id": "speaker-count-range",
                "scope": "job",
                "reasonCode": "SPEAKER_COUNT_RANGE_WIDE",
                "status": "open",
                "speakerCountEstimate": estimate.as_dict(),
            }
        )
    for segment in segments:
        reasons: list[str] = []

        def add_reason(reason: str) -> None:
            if reason not in reasons:
                reasons.append(reason)

        if segment.confidence < segment_confidence_threshold:
            add_reason("SEGMENT_LOW_CONFIDENCE")
        if segment.speaker_margin < speaker_margin_threshold:
            add_reason("SPEAKER_MARGIN_LOW")
        if segment.overlapping or segment.segment_id in actual_overlap_ids:
            add_reason("OVERLAP_REVIEW_REQUIRED")
        overlap = segment.evidence.get("overlap")
        if (
            isinstance(overlap, Mapping)
            and overlap.get("detectorStatus") == "UNAVAILABLE"
            and overlap.get("overlapDetectorRun") is False
            and overlap.get("reviewStatus") == "REVIEW_REQUIRED"
            and overlap.get("reasonCode") == "OVERLAP_DETECTOR_UNAVAILABLE"
        ):
            add_reason("OVERLAP_DETECTOR_UNAVAILABLE")
        semantic = segment.evidence.get("semantic")
        if isinstance(semantic, Mapping) and semantic.get("decision") in {
            "review-required",
            "reject",
        }:
            add_reason("SEMANTIC_REVIEW_REQUIRED")
        boundary = segment.evidence.get("boundary")
        if isinstance(boundary, Mapping) and boundary.get("conflict") is True:
            add_reason("BOUNDARY_CONFLICT")
        selective_review = segment.evidence.get("selectiveReview")
        selective_refs: set[str] = set()
        if isinstance(selective_review, Mapping):
            exit_reason = selective_review.get("exitReason")
            unresolved_reason_codes = {
                "HUMAN_REVIEW_REQUIRED": "OVERLAP_REVIEW_REQUIRED",
                "OVERLAP_DETECTOR_UNAVAILABLE": (
                    "OVERLAP_DETECTOR_UNAVAILABLE"
                ),
                "PROTECTED_REQUIRES_HUMAN_REVIEW": (
                    "PROTECTED_SPEAKER_REVIEW_REQUIRED"
                ),
                "SECONDARY_BUDGET_EXHAUSTED": "SECONDARY_BUDGET_EXHAUSTED",
                "SECONDARY_ADAPTER_DISABLED": "SECONDARY_VERIFIER_UNAVAILABLE",
                "NO_REFERENCE": "SECONDARY_NO_REFERENCE",
                "SINGLE_REFERENCE_INSUFFICIENT": (
                    "SECONDARY_REFERENCE_INSUFFICIENT"
                ),
                "LOW_MARGIN_UNRESOLVED": "SECONDARY_LOW_MARGIN_UNRESOLVED",
            }
            if exit_reason in unresolved_reason_codes:
                add_reason(unresolved_reason_codes[exit_reason])
            elif (
                exit_reason is not None
                and exit_reason
                not in {"VERIFIED_NO_CHANGE", "VERIFIED_SPEAKER_CHANGE"}
            ):
                add_reason("SECONDARY_VERIFIER_UNRESOLVED")
            raw_refs = selective_review.get("evidenceRefs", ())
            if isinstance(raw_refs, (list, tuple, set)):
                selective_refs.update(
                    str(reference)
                    for reference in raw_refs
                    if str(reference).strip()
                )
        for reason in reasons:
            items.append(
                {
                    "id": f"{segment.segment_id}:{reason}",
                    "scope": "segment",
                    "segmentId": segment.segment_id,
                    "reasonCode": reason,
                    "status": "open",
                    "timeRange": {
                        "startMs": segment.start_ms,
                        "endMs": segment.end_ms,
                    },
                    "speakerId": segment.speaker_id,
                    "speakerCandidates": [
                        score.as_dict() for score in segment.speaker_scores
                    ],
                    "text": {
                        "rawText": segment.raw_text,
                        "normalizedText": segment.normalized_text,
                        "displayText": segment.display_text,
                    },
                    "evidenceRefs": sorted(
                        {
                            reference
                            for revision in segment.revisions
                            for reference in revision.evidence_refs
                        }
                        | selective_refs
                    ),
                }
            )
    return {
        "schemaVersion": "2.0.0",
        "jobId": job_id,
        "speakerCountMode": policy.mode.value,
        "speakerCountEstimate": estimate.as_dict(),
        "createdAt": created_at,
        "updatedAt": created_at,
        "items": items,
    }


def assemble_transcript_document(
    *,
    job_id: str,
    source_path: Path,
    policy: SpeakerCountPolicy,
    estimate: SpeakerCountEstimate,
    speaker_count: int,
    result: TranscriptionResult,
    title: str | None,
    language: str,
    adapter_id: str,
    adapter_version: str,
) -> dict[str, Any]:
    try:
        resolved_language = normalize_language_tag(language, allow_auto=False)
    except ValueError as exc:
        raise WorkerError(
            "ADAPTER_RESULT_INVALID",
            "transcript language must be a persisted BCP-47 tag, und, or mul",
        ) from exc

    normalized_segment_languages: list[str | None] = []
    detected_languages: set[str] = set()
    for segment in result.segments:
        if segment.language is None:
            normalized_segment_languages.append(None)
            continue
        try:
            segment_language = normalize_language_tag(
                segment.language,
                allow_auto=False,
            )
        except ValueError as exc:
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                f"{segment.segment_id}.language must be a valid persisted BCP-47 language tag",
            ) from exc
        normalized_segment_languages.append(segment_language)
        if segment_language not in {
            UNDETERMINED_LANGUAGE,
            MULTIPLE_LANGUAGES,
        }:
            detected_languages.add(segment_language)

    if len(detected_languages) > 1:
        resolved_language = MULTIPLE_LANGUAGES

    segments: list[dict[str, Any]] = []
    for segment, segment_language in zip(
        result.segments,
        normalized_segment_languages,
    ):
        entry = segment.as_dict()
        entry["language"] = segment_language or resolved_language
        segments.append(entry)

    canonical = canonical_speaker_ids(speaker_count)
    source_hash = sha256_file(source_path)
    document_seed = f"{source_hash}:{job_id}:{speaker_count}".encode("utf-8")
    document_id = "doc-" + hashlib.sha256(document_seed).hexdigest()[:24]
    speaker_entries = []
    for index, speaker_id in enumerate(canonical):
        entry: dict[str, Any] = {"id": speaker_id}
        if policy.roles:
            entry["role"] = policy.roles[index]
        speaker_entries.append(entry)
    speaker_policy = {
        **policy.as_dict(),
        "resolvedCount": speaker_count,
        "speakerIds": list(canonical),
        "requireExactSet": True,
        "unknownSpeakerAllowed": False,
        "speakerChangeRequiresEvidence": True,
        "estimate": estimate.as_dict(),
    }
    document: dict[str, Any] = {
        "schemaVersion": "2.0.0",
        "documentId": document_id,
        "jobId": job_id,
        "generatedAt": utc_now(),
        "language": resolved_language,
        "source": {
            "fileName": source_path.name,
            "sha256": source_hash,
            "durationMs": result.duration_ms,
        },
        "speakerPolicy": speaker_policy,
        "speakers": speaker_entries,
        "segments": segments,
        "provenance": {
            "offline": True,
            "workerVersion": "2.0.0",
            "transcriptionAdapter": {
                "id": adapter_id,
                "version": adapter_version,
            },
            "models": [dict(item) for item in result.models],
        },
    }
    if result.pipeline_metrics is not None:
        document["provenance"]["pipelineMetricsSchemaVersion"] = str(
            result.pipeline_metrics.get("schemaVersion") or ""
        )
    if result.speaker_timeline is not None:
        document["speakerTimeline"] = dict(result.speaker_timeline)
    if title:
        document["title"] = title
    return document
