"""Manual-review validation and transcript/queue mutations.

This module deliberately contains no filesystem or executor logic.  Callers
must persist the returned document and queue in one transaction while holding
the owning job lock.
"""

from __future__ import annotations

import copy
import math
import uuid
from collections.abc import Mapping
from typing import Any

from .documents import utc_now, validate_segments
from .errors import WorkerError, invalid_request
from .models import TranscriptSegment, canonical_speaker_ids
from .persistence import validate_strict_json


_MANUAL_FIELDS = frozenset(
    {"reason", "evidence", "confidence", "audit", "decisionId"}
)
MANUAL_REVIEW_AUDIT_SOURCES = frozenset({"human", "codex-agent"})
_RESOLVED_ITEM_STATUSES = frozenset({"accepted", "rejected"})


def validate_manual_decision(
    payload: Mapping[str, Any],
    *,
    command: str,
) -> dict[str, Any]:
    """Validate and normalize the mandatory manual-decision audit envelope."""

    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise WorkerError(
            "REVIEW_REASON_REQUIRED",
            "manual decisions require a non-empty reason",
        )
    evidence = payload.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(item, str) or not item.strip() for item in evidence)
    ):
        raise WorkerError(
            "REVIEW_EVIDENCE_REQUIRED",
            "manual decisions require a non-empty evidence array",
        )
    normalized_evidence = [item.strip() for item in evidence]
    if len(set(normalized_evidence)) != len(normalized_evidence):
        raise WorkerError(
            "REVIEW_EVIDENCE_INVALID",
            "manual-decision evidence references must be unique",
        )

    confidence = payload.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise WorkerError(
            "REVIEW_CONFIDENCE_INVALID",
            "manual-decision confidence must be a finite number between 0 and 1",
        )

    audit = payload.get("audit")
    if not isinstance(audit, Mapping):
        raise WorkerError(
            "REVIEW_AUDIT_REQUIRED",
            "manual decisions require an audit object",
        )
    actor = audit.get("actor")
    if not isinstance(actor, str) or not actor.strip():
        raise WorkerError(
            "REVIEW_AUDIT_INVALID",
            "manual-decision audit.actor must be a non-empty string",
        )
    normalized_audit = copy.deepcopy(dict(audit))
    normalized_audit["actor"] = actor.strip()
    if normalized_audit.get("source") not in MANUAL_REVIEW_AUDIT_SOURCES:
        raise WorkerError(
            "REVIEW_AUDIT_INVALID",
            "manual-decision audit.source must be human or codex-agent",
        )
    timestamp = normalized_audit.get("timestamp")
    if timestamp is not None and (
        not isinstance(timestamp, str) or not timestamp.strip()
    ):
        raise WorkerError(
            "REVIEW_AUDIT_INVALID",
            "manual-decision audit.timestamp must be non-empty text when provided",
        )
    try:
        validate_strict_json(normalized_audit)
    except ValueError as exc:
        raise WorkerError(
            "REVIEW_AUDIT_INVALID",
            "manual-decision audit must contain strict finite JSON values",
            details={"reason": str(exc)},
        ) from exc

    decision_id_raw = payload.get("decisionId")
    if not isinstance(decision_id_raw, str) or not decision_id_raw.strip():
        raise WorkerError(
            "REVIEW_DECISION_ID_INVALID",
            "human decisions require a non-empty decisionId",
        )
    decision_id = decision_id_raw.strip()
    if len(decision_id) > 160:
        raise WorkerError(
            "REVIEW_DECISION_ID_INVALID",
            "decisionId exceeds 160 characters",
        )

    return {
        "decisionId": decision_id,
        "command": command,
        "reason": reason.strip(),
        "evidence": normalized_evidence,
        "confidence": float(confidence),
        "audit": normalized_audit,
        "recordedAt": utc_now(),
    }


def validate_review_state(
    document: Mapping[str, Any],
    queue: Mapping[str, Any],
    *,
    expected_job_id: str,
    high_margin_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate durable review state, including raw-text snapshots."""

    document_copy = copy.deepcopy(dict(document))
    queue_copy = copy.deepcopy(dict(queue))
    if document_copy.get("schemaVersion") != "2.0.0":
        raise WorkerError(
            "REVIEW_DOCUMENT_INVALID",
            "transcript document schemaVersion must be 2.0.0",
        )
    if document_copy.get("jobId") != expected_job_id:
        raise WorkerError(
            "REVIEW_DOCUMENT_INVALID",
            "transcript document jobId does not match the requested job",
        )
    if queue_copy.get("schemaVersion") != "2.0.0":
        raise WorkerError(
            "REVIEW_QUEUE_INVALID",
            "review queue schemaVersion must be 2.0.0",
        )
    if queue_copy.get("jobId") != expected_job_id:
        raise WorkerError(
            "REVIEW_QUEUE_INVALID",
            "review queue jobId does not match the requested job",
        )
    raw_segments = document_copy.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise WorkerError(
            "REVIEW_DOCUMENT_INVALID",
            "transcript document must contain segments",
        )
    segments = tuple(
        TranscriptSegment.from_mapping(item, index)
        for index, item in enumerate(raw_segments)
    )
    speaker_policy = document_copy.get("speakerPolicy")
    if not isinstance(speaker_policy, Mapping):
        raise WorkerError(
            "REVIEW_DOCUMENT_INVALID",
            "transcript document speakerPolicy must be an object",
        )
    speaker_count = speaker_policy.get("resolvedCount")
    if (
        isinstance(speaker_count, bool)
        or not isinstance(speaker_count, int)
        or speaker_count < 1
    ):
        raise WorkerError(
            "REVIEW_DOCUMENT_INVALID",
            "speakerPolicy.resolvedCount must be a positive integer",
        )
    expected_ids = canonical_speaker_ids(speaker_count)
    if speaker_policy.get("speakerIds") != list(expected_ids):
        raise WorkerError(
            "REVIEW_DOCUMENT_INVALID",
            "speakerPolicy.speakerIds must match the canonical namespace",
        )
    speakers = document_copy.get("speakers")
    if not isinstance(speakers, list) or [
        item.get("id") if isinstance(item, Mapping) else None for item in speakers
    ] != list(expected_ids):
        raise WorkerError(
            "REVIEW_DOCUMENT_INVALID",
            "speakers must exactly match the canonical namespace",
        )
    source = document_copy.get("source")
    duration_ms = source.get("durationMs") if isinstance(source, Mapping) else None
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms < 1
    ):
        raise WorkerError(
            "REVIEW_DOCUMENT_INVALID",
            "source.durationMs must be a positive integer",
        )
    validate_segments(
        segments,
        speaker_count=speaker_count,
        duration_ms=duration_ms,
        high_margin_threshold=high_margin_threshold,
    )

    items = queue_copy.get("items")
    if not isinstance(items, list):
        raise WorkerError("REVIEW_QUEUE_INVALID", "review queue items must be an array")
    decisions = queue_copy.get("decisions", [])
    if not isinstance(decisions, list):
        raise WorkerError(
            "REVIEW_QUEUE_INVALID",
            "review queue decisions must be an array",
        )
    queue_copy["decisions"] = decisions
    segment_by_id = {segment.segment_id: segment for segment in segments}
    raw_by_id = {
        segment_id: segment.raw_text for segment_id, segment in segment_by_id.items()
    }
    item_ids: set[str] = set()
    decision_ids: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise WorkerError(
                "REVIEW_QUEUE_INVALID",
                "review queue decisions must be objects",
            )
        decision_id = decision.get("decisionId")
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise WorkerError(
                "REVIEW_QUEUE_INVALID",
                "persisted decisions require decisionId",
            )
        if decision_id in decision_ids:
            raise WorkerError(
                "REVIEW_QUEUE_INVALID",
                "persisted decisionId values must be unique",
            )
        decision_ids.add(decision_id)
    for item in items:
        if not isinstance(item, Mapping):
            raise WorkerError(
                "REVIEW_QUEUE_INVALID",
                "review queue items must be objects",
            )
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip() or item_id in item_ids:
            raise WorkerError(
                "REVIEW_QUEUE_INVALID",
                "review queue item IDs must be non-empty and unique",
            )
        item_ids.add(item_id)
        status = item.get("status")
        if status not in {"open", *_RESOLVED_ITEM_STATUSES}:
            raise WorkerError(
                "REVIEW_QUEUE_INVALID",
                f"review item {item_id} has an invalid status",
            )
        segment_id = item.get("segmentId")
        if segment_id is not None:
            if segment_id not in raw_by_id:
                raise WorkerError(
                    "REVIEW_QUEUE_INVALID",
                    f"review item {item_id} references an unknown segment",
                )
            segment = segment_by_id[segment_id]
            text = item.get("text")
            if (
                not isinstance(text, Mapping)
                or text.get("rawText") != raw_by_id[segment_id]
                or text.get("normalizedText") != segment.normalized_text
                or text.get("displayText") != segment.display_text
            ):
                raise WorkerError(
                    "RAW_TEXT_IMMUTABLE",
                    f"review item {item_id} text snapshot does not match the transcript",
                )
            if item.get("speakerId") != segment.speaker_id:
                raise WorkerError(
                    "REVIEW_QUEUE_INVALID",
                    f"review item {item_id} speakerId does not match the transcript",
                )
            candidates = item.get("speakerCandidates")
            if candidates is not None:
                if (
                    not isinstance(candidates, list)
                    or [
                        candidate.get("speakerId")
                        if isinstance(candidate, Mapping)
                        else None
                        for candidate in candidates
                    ]
                    != list(expected_ids)
                ):
                    raise WorkerError(
                        "REVIEW_QUEUE_INVALID",
                        f"review item {item_id} speakerCandidates must match the canonical namespace",
                    )
        if status != "open":
            decision = item.get("decision")
            if not isinstance(decision, Mapping):
                raise WorkerError(
                    "REVIEW_QUEUE_INVALID",
                    f"resolved review item {item_id} requires a decision",
                )
    queue_copy["openCount"] = open_count(queue_copy)
    return document_copy, queue_copy


def open_count(queue: Mapping[str, Any]) -> int:
    items = queue.get("items", [])
    if not isinstance(items, list):
        return 0
    return sum(
        1
        for item in items
        if isinstance(item, Mapping) and item.get("status") == "open"
    )


def resolve_review_item(
    document: Mapping[str, Any],
    queue: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    command: str,
    action: str,
    suggestion_defaults: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Accept/reject one queue item and optionally apply manual edits."""

    decision = validate_manual_decision(payload, command=command)
    document_copy = copy.deepcopy(dict(document))
    queue_copy = copy.deepcopy(dict(queue))
    _reject_duplicate_decision(queue_copy, decision["decisionId"])
    item_id = payload.get("itemId")
    if not isinstance(item_id, str) or not item_id.strip():
        raise invalid_request("itemId must be a non-empty string")
    item = _find_item(queue_copy, item_id.strip())
    if item.get("status") != "open":
        raise WorkerError(
            "DUPLICATE_REVIEW_DECISION",
            "the review item already has a human decision",
            details={"itemId": item_id},
        )
    if action not in _RESOLVED_ITEM_STATUSES:
        raise invalid_request("review action must be accept or reject")

    defaults = dict(suggestion_defaults or {})
    segment_id = item.get("segmentId")
    if action == "accepted" and segment_id is not None:
        segment = _find_segment(document_copy, segment_id)
        _apply_segment_edits(
            document_copy,
            segment,
            payload,
            defaults=defaults,
            decision=decision,
        )
        _refresh_segment_queue_snapshots(queue_copy, segment)

    item["status"] = action
    item["decision"] = copy.deepcopy(decision)
    if isinstance(segment_id, str):
        _mark_segment_human_reviewed_if_complete(
            document_copy,
            queue_copy,
            segment_id=segment_id,
            decision=decision,
        )
    _append_decision(queue_copy, decision, item_id=item_id.strip(), status=action)
    _touch_queue(queue_copy)
    return document_copy, queue_copy, decision


def rename_speaker(
    document: Mapping[str, Any],
    queue: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    decision = validate_manual_decision(payload, command="speaker.rename")
    document_copy = copy.deepcopy(dict(document))
    queue_copy = copy.deepcopy(dict(queue))
    _reject_duplicate_decision(queue_copy, decision["decisionId"])
    speaker_id = _require_speaker(document_copy, payload.get("speakerId"), "speakerId")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise invalid_request("name must be a non-empty string")
    normalized_name = name.strip()
    if len(normalized_name) > 160:
        raise invalid_request("name exceeds 160 characters")
    speakers = document_copy["speakers"]
    assert isinstance(speakers, list)
    for entry in speakers:
        if entry.get("id") != speaker_id and normalized_name in {
            entry.get("name"),
            entry.get("role"),
        }:
            raise WorkerError(
                "DUPLICATE_SPEAKER_NAME",
                "speaker names must remain unique",
            )
    target = next(entry for entry in speakers if entry.get("id") == speaker_id)
    target["name"] = normalized_name
    target["role"] = normalized_name
    _append_decision(
        queue_copy,
        decision,
        speaker_id=speaker_id,
        after={"name": normalized_name},
    )
    _touch_queue(queue_copy)
    return document_copy, queue_copy, decision


def merge_speakers(
    document: Mapping[str, Any],
    queue: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Merge source into target and compact the canonical namespace."""

    decision = validate_manual_decision(payload, command="speaker.merge")
    document_copy = copy.deepcopy(dict(document))
    queue_copy = copy.deepcopy(dict(queue))
    _reject_duplicate_decision(queue_copy, decision["decisionId"])
    source_id = _require_speaker(
        document_copy, payload.get("sourceSpeakerId"), "sourceSpeakerId"
    )
    target_id = _require_speaker(
        document_copy, payload.get("targetSpeakerId"), "targetSpeakerId"
    )
    if source_id == target_id:
        raise WorkerError(
            "SPEAKER_MERGE_INVALID",
            "sourceSpeakerId and targetSpeakerId must differ",
        )

    old_speakers = document_copy["speakers"]
    assert isinstance(old_speakers, list)
    if len(old_speakers) <= 1:
        raise WorkerError(
            "SPEAKER_MERGE_INVALID",
            "a single-speaker transcript cannot be merged",
        )
    remaining_ids = [
        entry["id"] for entry in old_speakers if entry.get("id") != source_id
    ]
    namespace_map = {
        old_id: f"speaker-{index}"
        for index, old_id in enumerate(remaining_ids, start=1)
    }
    namespace_map[source_id] = namespace_map[target_id]

    raw_segments = document_copy["segments"]
    assert isinstance(raw_segments, list)
    for segment in raw_segments:
        old_assignment = segment["speakerId"]
        merged_assignment = target_id if old_assignment == source_id else old_assignment
        new_assignment = namespace_map[merged_assignment]
        assignment_changed = (
            old_assignment == source_id or new_assignment != old_assignment
        )
        if assignment_changed and segment.get("humanLocked") is True:
            raise WorkerError(
                "HUMAN_LOCK_CONFLICT",
                "speaker merge would override a human-locked assignment",
                details={"segmentId": segment.get("id")},
            )

    new_speakers: list[dict[str, Any]] = []
    for entry in old_speakers:
        old_id = entry.get("id")
        if old_id == source_id:
            continue
        new_entry = copy.deepcopy(entry)
        new_entry["id"] = namespace_map[old_id]
        new_speakers.append(new_entry)
    document_copy["speakers"] = new_speakers

    for segment in raw_segments:
        old_assignment = segment["speakerId"]
        merged_assignment = target_id if old_assignment == source_id else old_assignment
        new_assignment = namespace_map[merged_assignment]
        assignment_changed = (
            old_assignment == source_id or new_assignment != old_assignment
        )
        combined: dict[str, float] = {}
        for score in segment["speakerScores"]:
            mapped_id = namespace_map[score["speakerId"]]
            combined[mapped_id] = max(
                combined.get(mapped_id, -2.0),
                float(score["score"]),
            )
        segment["speakerScores"] = [
            {"speakerId": speaker_id, "score": combined[speaker_id]}
            for speaker_id in canonical_speaker_ids(len(new_speakers))
        ]
        segment["speakerMargin"] = _speaker_margin(segment["speakerScores"])
        segment["speakerId"] = new_assignment
        non_speaker_revisions = [
            revision
            for revision in segment.get("revisions", [])
            if revision.get("type") != "speaker"
        ]
        acoustic_top = max(
            segment["speakerScores"], key=lambda item: item["score"]
        )["speakerId"]
        if acoustic_top != new_assignment:
            non_speaker_revisions.append(
                _revision(
                    "speaker",
                    acoustic_top,
                    new_assignment,
                    decision,
                    reason_code="MANUAL_SPEAKER_MERGE",
                )
            )
        segment["revisions"] = non_speaker_revisions
        if assignment_changed:
            segment["humanLocked"] = True
        _refresh_segment_queue_snapshots(queue_copy, segment)

    _update_policy_cardinality(
        document_copy,
        len(new_speakers),
        method="manual-merge",
    )
    _sync_queue_policy(queue_copy, document_copy)
    _append_decision(
        queue_copy,
        decision,
        before={"sourceSpeakerId": source_id, "targetSpeakerId": target_id},
        after={"speakerIds": [entry["id"] for entry in new_speakers]},
    )
    _touch_queue(queue_copy)
    return document_copy, queue_copy, decision


def split_speaker(
    document: Mapping[str, Any],
    queue: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Move selected source-speaker segments into one new canonical speaker."""

    decision = validate_manual_decision(payload, command="speaker.split")
    document_copy = copy.deepcopy(dict(document))
    queue_copy = copy.deepcopy(dict(queue))
    _reject_duplicate_decision(queue_copy, decision["decisionId"])
    source_id = _require_speaker(
        document_copy, payload.get("sourceSpeakerId"), "sourceSpeakerId"
    )
    segment_ids = payload.get("segmentIds")
    if (
        not isinstance(segment_ids, list)
        or not segment_ids
        or any(not isinstance(item, str) or not item.strip() for item in segment_ids)
    ):
        raise invalid_request("segmentIds must be a non-empty array of segment IDs")
    normalized_ids = [item.strip() for item in segment_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise invalid_request("segmentIds must not contain duplicates")

    raw_segments = document_copy["segments"]
    assert isinstance(raw_segments, list)
    segment_by_id = {segment["id"]: segment for segment in raw_segments}
    if any(item not in segment_by_id for item in normalized_ids):
        raise WorkerError(
            "SEGMENT_NOT_FOUND",
            "speaker split references an unknown segment",
        )
    selected = [segment_by_id[item] for item in normalized_ids]
    if any(segment.get("speakerId") != source_id for segment in selected):
        raise WorkerError(
            "SPEAKER_SPLIT_INVALID",
            "every split segment must currently belong to sourceSpeakerId",
        )
    source_segments = [
        segment for segment in raw_segments if segment.get("speakerId") == source_id
    ]
    if len(selected) >= len(source_segments):
        raise WorkerError(
            "SPEAKER_SPLIT_INVALID",
            "speaker split must leave at least one segment with the source speaker",
        )
    locked = next(
        (segment for segment in selected if segment.get("humanLocked") is True),
        None,
    )
    if locked is not None:
        raise WorkerError(
            "HUMAN_LOCK_CONFLICT",
            "speaker split would override a human-locked assignment",
            details={"segmentId": locked.get("id")},
        )

    speakers = document_copy["speakers"]
    assert isinstance(speakers, list)
    new_id = f"speaker-{len(speakers) + 1}"
    new_name = payload.get("newSpeakerName")
    new_entry: dict[str, Any] = {"id": new_id}
    if new_name is not None:
        if not isinstance(new_name, str) or not new_name.strip():
            raise invalid_request("newSpeakerName must be non-empty text")
        new_entry["name"] = new_name.strip()
        new_entry["role"] = new_name.strip()
    speakers.append(new_entry)

    selected_ids = set(normalized_ids)
    for segment in raw_segments:
        scores = segment["speakerScores"]
        minimum = min(float(item["score"]) for item in scores)
        synthetic_score = max(-2.0, minimum - 0.001)
        scores.append({"speakerId": new_id, "score": synthetic_score})
        segment["speakerMargin"] = _speaker_margin(scores)
        if segment["id"] in selected_ids:
            before = segment["speakerId"]
            segment["speakerId"] = new_id
            segment["humanLocked"] = True
            segment.setdefault("revisions", []).append(
                _revision(
                    "speaker",
                    before,
                    new_id,
                    decision,
                    reason_code="MANUAL_SPEAKER_SPLIT",
                )
            )
        _refresh_segment_queue_snapshots(queue_copy, segment)

    _update_policy_cardinality(
        document_copy,
        len(speakers),
        method="manual-split",
    )
    _sync_queue_policy(queue_copy, document_copy)
    _append_decision(
        queue_copy,
        decision,
        before={"sourceSpeakerId": source_id},
        after={"newSpeakerId": new_id, "segmentIds": normalized_ids},
    )
    _touch_queue(queue_copy)
    return document_copy, queue_copy, decision


def assert_raw_text_unchanged(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    before_segments = before.get("segments", [])
    after_segments = after.get("segments", [])
    before_raw = {
        item.get("id"): item.get("rawText")
        for item in before_segments
        if isinstance(item, Mapping)
    }
    after_raw = {
        item.get("id"): item.get("rawText")
        for item in after_segments
        if isinstance(item, Mapping)
    }
    if before_raw != after_raw:
        raise WorkerError(
            "RAW_TEXT_IMMUTABLE",
            "rawText is immutable and cannot be changed by review",
        )


def manual_fields() -> frozenset[str]:
    return _MANUAL_FIELDS


def _find_item(queue: dict[str, Any], item_id: str) -> dict[str, Any]:
    for item in queue.get("items", []):
        if isinstance(item, dict) and item.get("id") == item_id:
            return item
    raise WorkerError(
        "REVIEW_ITEM_NOT_FOUND",
        "review item does not exist",
        details={"itemId": item_id},
    )


def _find_segment(document: dict[str, Any], segment_id: str) -> dict[str, Any]:
    for segment in document.get("segments", []):
        if isinstance(segment, dict) and segment.get("id") == segment_id:
            return segment
    raise WorkerError(
        "SEGMENT_NOT_FOUND",
        "segment does not exist",
        details={"segmentId": segment_id},
    )


def _mark_segment_human_reviewed_if_complete(
    document: dict[str, Any],
    queue: Mapping[str, Any],
    *,
    segment_id: str,
    decision: Mapping[str, Any],
) -> None:
    items = queue.get("items")
    if not isinstance(items, list):
        return
    segment_items = [
        item
        for item in items
        if isinstance(item, Mapping) and item.get("segmentId") == segment_id
    ]
    if not segment_items or any(item.get("status") == "open" for item in segment_items):
        return

    segment = _find_segment(document, segment_id)
    raw_evidence = segment.get("evidence")
    evidence = dict(raw_evidence) if isinstance(raw_evidence, Mapping) else {}
    audit = decision.get("audit")
    reviewer = audit.get("actor") if isinstance(audit, Mapping) else None
    notes = decision.get("reason")
    audio_review: dict[str, Any] = {"status": "human-reviewed"}
    if isinstance(reviewer, str) and reviewer.strip():
        audio_review["reviewer"] = reviewer.strip()
    if isinstance(notes, str) and notes.strip():
        audio_review["notes"] = notes.strip()
    evidence["audioReview"] = audio_review
    segment["evidence"] = evidence


def _canonical_ids(document: Mapping[str, Any]) -> tuple[str, ...]:
    policy = document.get("speakerPolicy")
    count = policy.get("resolvedCount") if isinstance(policy, Mapping) else None
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
    ):
        raise WorkerError(
            "REVIEW_DOCUMENT_INVALID",
            "speakerPolicy.resolvedCount must be a positive integer",
        )
    return canonical_speaker_ids(count)


def _require_speaker(
    document: Mapping[str, Any],
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise invalid_request(f"{field_name} must be a non-empty speaker ID")
    speaker_id = value.strip()
    if speaker_id not in set(_canonical_ids(document)):
        raise WorkerError(
            "INVALID_SPEAKER",
            f"{field_name} is outside the current canonical speaker namespace",
            details={field_name: speaker_id},
        )
    return speaker_id


def _apply_segment_edits(
    document: dict[str, Any],
    segment: dict[str, Any],
    payload: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    if "rawText" in payload and payload.get("rawText") != segment.get("rawText"):
        raise WorkerError(
            "RAW_TEXT_IMMUTABLE",
            "rawText is immutable and cannot be changed by review",
        )
    if "rawText" in defaults and defaults.get("rawText") != segment.get("rawText"):
        raise WorkerError(
            "RAW_TEXT_IMMUTABLE",
            "a suggestion cannot replace rawText",
        )

    target_speaker = payload.get(
        "targetSpeakerId",
        defaults.get("targetSpeakerId", defaults.get("speakerId")),
    )
    if target_speaker is not None:
        target_speaker = _require_speaker(
            document, target_speaker, "targetSpeakerId"
        )
        current_speaker = segment["speakerId"]
        if target_speaker != current_speaker:
            if segment.get("humanLocked") is True:
                raise WorkerError(
                    "HUMAN_LOCK_CONFLICT",
                    "review would override a human-locked speaker assignment",
                    details={"segmentId": segment.get("id")},
                )
            segment.setdefault("revisions", []).append(
                _revision(
                    "speaker",
                    current_speaker,
                    target_speaker,
                    decision,
                    reason_code="MANUAL_SPEAKER_REVIEW",
                )
            )
            segment["speakerId"] = target_speaker
            segment["humanLocked"] = True

    normalized_supplied = "normalizedText" in payload or "normalizedText" in defaults
    display_supplied = "displayText" in payload or "displayText" in defaults
    if normalized_supplied or display_supplied:
        current_normalized = segment["normalizedText"]
        current_display = segment["displayText"]
        current_effective = (
            current_display
            if current_display != current_normalized
            else current_normalized
        )
        normalized = payload.get(
            "normalizedText",
            defaults.get("normalizedText", current_normalized),
        )
        if normalized_supplied and (
            not isinstance(normalized, str) or not normalized.strip()
        ):
            raise WorkerError(
                "REVIEW_TEXT_INVALID",
                "normalizedText must be non-empty source-language text",
            )
        if display_supplied:
            display = payload.get(
                "displayText",
                defaults.get("displayText", current_display),
            )
        elif normalized_supplied:
            display = normalized
        else:
            display = current_display
        if not isinstance(display, str) or not display.strip():
            raise WorkerError(
                "REVIEW_TEXT_INVALID",
                "displayText must be non-empty source-language text",
            )
        normalized = normalized.strip() if isinstance(normalized, str) else normalized
        display = display.strip()
        new_effective = display if display != normalized else normalized
        if new_effective != current_effective:
            segment.setdefault("revisions", []).append(
                _revision(
                    "text",
                    current_effective,
                    new_effective,
                    decision,
                    reason_code="MANUAL_TEXT_REVIEW",
                )
            )
        segment["normalizedText"] = normalized
        segment["displayText"] = display


def _revision(
    revision_type: str,
    before: Any,
    after: Any,
    decision: Mapping[str, Any],
    *,
    reason_code: str,
) -> dict[str, Any]:
    audit = decision["audit"]
    return {
        "id": f"revision-{uuid.uuid4().hex}",
        "type": revision_type,
        "source": "manual",
        "actor": audit["actor"],
        "occurredAt": decision["recordedAt"],
        "before": before,
        "after": after,
        "reasonCode": reason_code,
        "confidence": decision["confidence"],
        "evidenceRefs": list(decision["evidence"]),
    }


def _refresh_segment_queue_snapshots(
    queue: dict[str, Any],
    segment: Mapping[str, Any],
) -> None:
    for item in queue.get("items", []):
        if not isinstance(item, dict) or item.get("segmentId") != segment.get("id"):
            continue
        item["speakerId"] = segment.get("speakerId")
        item["speakerCandidates"] = copy.deepcopy(segment.get("speakerScores", []))
        item["text"] = {
            "rawText": segment.get("rawText"),
            "normalizedText": segment.get("normalizedText"),
            "displayText": segment.get("displayText"),
        }


def _append_decision(
    queue: dict[str, Any],
    decision: Mapping[str, Any],
    **context: Any,
) -> None:
    entry = copy.deepcopy(dict(decision))
    entry.update(copy.deepcopy(context))
    queue.setdefault("decisions", []).append(entry)


def _reject_duplicate_decision(queue: Mapping[str, Any], decision_id: str) -> None:
    for item in queue.get("decisions", []):
        if isinstance(item, Mapping) and item.get("decisionId") == decision_id:
            raise WorkerError(
                "DUPLICATE_REVIEW_DECISION",
                "decisionId has already been persisted",
                details={"decisionId": decision_id},
            )


def _touch_queue(queue: dict[str, Any]) -> None:
    queue["updatedAt"] = utc_now()
    queue["openCount"] = open_count(queue)


def _sync_queue_policy(
    queue: dict[str, Any],
    document: Mapping[str, Any],
) -> None:
    policy = document.get("speakerPolicy")
    if not isinstance(policy, Mapping):
        raise WorkerError(
            "REVIEW_DOCUMENT_INVALID",
            "transcript document speakerPolicy must be an object",
        )
    estimate = policy.get("estimate")
    if not isinstance(estimate, Mapping):
        raise WorkerError(
            "REVIEW_DOCUMENT_INVALID",
            "transcript document speakerPolicy.estimate must be an object",
        )
    queue["speakerCountMode"] = policy.get("mode")
    queue["speakerCountEstimate"] = copy.deepcopy(dict(estimate))
    for item in queue.get("items", []):
        if (
            isinstance(item, dict)
            and item.get("scope") == "job"
            and "speakerCountEstimate" in item
        ):
            item["speakerCountEstimate"] = copy.deepcopy(dict(estimate))


def _speaker_margin(scores: list[Mapping[str, Any]]) -> float:
    ranked = sorted((float(item["score"]) for item in scores), reverse=True)
    return ranked[0] - ranked[1] if len(ranked) > 1 else 2.0


def _update_policy_cardinality(
    document: dict[str, Any],
    count: int,
    *,
    method: str,
) -> None:
    ids = list(canonical_speaker_ids(count))
    policy = document["speakerPolicy"]
    policy["resolvedCount"] = count
    policy["speakerIds"] = ids
    if policy.get("mode") == "manual":
        policy["manualCount"] = count
        roles = [
            entry.get("role") or entry.get("name")
            for entry in document.get("speakers", [])
        ]
        if all(isinstance(role, str) and role.strip() for role in roles):
            policy["roles"] = roles
        else:
            policy.pop("roles", None)
    estimate = policy.get("estimate")
    if not isinstance(estimate, dict):
        estimate = {}
        policy["estimate"] = estimate
    estimate.update(
        {
            "estimatedCount": count,
            "confidence": 1.0,
            "candidateRange": {"min": count, "max": count},
            "method": method,
        }
    )
