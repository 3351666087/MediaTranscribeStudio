"""Versioned scoring subject after semantic processing and human review."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from .documents import utc_now
from .errors import WorkerError
from .language import normalize_language_tag
from .persistence import canonical_json_sha256, validate_strict_json
from .semantic_processing import (
    SEMANTIC_APPLICATION_POLICY,
    validate_semantic_suggestions_artifact,
)
from .semantic_candidate_lattice import validate_semantic_candidate_lattice
from .semantic_composition import (
    validate_semantic_composition,
    validate_semantic_job_arbitration,
)
from .voice_activity import validate_voice_activity


FINAL_ADJUDICATED_TRANSCRIPT_SCHEMA_VERSION = "1.1.0"
FINAL_COMPOSED_TRANSCRIPT_SCHEMA_VERSION = "1.2.0"
FINAL_ADJUDICATED_TRANSCRIPT_ARTIFACT_TYPE = "final-adjudicated-transcript"


def _fail(code: str, message: str, **details: Any) -> WorkerError:
    return WorkerError(code, message, details=details or None)


def _resolved_review_counts(queue: Mapping[str, Any]) -> tuple[int, int, int]:
    items = queue.get("items")
    decisions = queue.get("decisions", [])
    if not isinstance(items, list) or not isinstance(decisions, list):
        raise _fail(
            "FINAL_ADJUDICATION_REVIEW_INVALID",
            "final adjudication requires review items and decisions arrays",
        )
    accepted = 0
    rejected = 0
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise _fail(
                "FINAL_ADJUDICATION_REVIEW_INVALID",
                "review items must be objects",
                itemIndex=index,
            )
        status = item.get("status")
        if status == "accepted":
            accepted += 1
        elif status == "rejected":
            rejected += 1
        elif status == "open":
            raise _fail(
                "FINAL_ADJUDICATION_REVIEW_INCOMPLETE",
                "final adjudication refuses open review items",
                itemId=item.get("id"),
            )
        else:
            raise _fail(
                "FINAL_ADJUDICATION_REVIEW_INVALID",
                "review item status is invalid",
                itemId=item.get("id"),
            )
    open_count = queue.get("openCount")
    if (
        isinstance(open_count, bool)
        or not isinstance(open_count, int)
        or open_count != 0
        or accepted + rejected != len(items)
    ):
        raise _fail(
            "FINAL_ADJUDICATION_REVIEW_INCOMPLETE",
            "review queue must have zero open items",
            openCount=open_count,
        )
    return accepted, rejected, len(decisions)


def _semantic_input(
    semantic_artifact: Mapping[str, Any],
    *,
    job_id: str,
) -> tuple[dict[str, Any], str]:
    raw_input = semantic_artifact.get("input")
    semantic_input_hash = (
        raw_input.get("transcriptSha256")
        if isinstance(raw_input, Mapping)
        else None
    )
    if not isinstance(semantic_input_hash, str):
        raise _fail(
            "FINAL_ADJUDICATION_SEMANTIC_INVALID",
            "semantic artifact is missing its transcript hash",
        )
    validated = validate_semantic_suggestions_artifact(
        semantic_artifact,
        expected_job_id=job_id,
        expected_transcript_sha256=semantic_input_hash,
    )
    metrics = validated.get("metrics")
    if not isinstance(metrics, Mapping):
        raise _fail(
            "FINAL_ADJUDICATION_SEMANTIC_INVALID",
            "semantic artifact metrics are missing",
        )
    blockers = {
        field: metrics.get(field)
        for field in (
            "rejectionCount",
            "failureCount",
            "unresolvedSegmentCount",
            "autoAppliedCount",
        )
    }
    if (
        validated.get("status") != "completed"
        or validated.get("applicationPolicy")
        != SEMANTIC_APPLICATION_POLICY
        or validated.get("requiresHumanApproval") is not True
        or any(value != 0 for value in blockers.values())
    ):
        raise _fail(
            "FINAL_ADJUDICATION_SEMANTIC_INCOMPLETE",
            "semantic processing is not complete and approval-safe",
            status=validated.get("status"),
            **blockers,
        )
    return validated, semantic_input_hash


def _final_segments(
    document: Mapping[str, Any],
    *,
    speaker_ids: list[str],
    duration_ms: int,
) -> list[dict[str, Any]]:
    raw_segments = document.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise _fail(
            "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
            "final transcript must contain segments",
        )
    allowed_speakers = set(speaker_ids)
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_start = -1
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, Mapping):
            raise _fail(
                "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
                "transcript segments must be objects",
                segmentIndex=index,
            )
        segment_id = segment.get("id")
        start_ms = segment.get("startMs")
        end_ms = segment.get("endMs")
        speaker_id = segment.get("speakerId")
        raw_text = segment.get("rawText")
        final_text = segment.get("normalizedText")
        if (
            not isinstance(segment_id, str)
            or not segment_id
            or segment_id in seen_ids
            or isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or start_ms < previous_start
            or start_ms < 0
            or end_ms <= start_ms
            or end_ms > duration_ms
            or speaker_id not in allowed_speakers
            or not isinstance(raw_text, str)
            or not raw_text.strip()
            or not isinstance(final_text, str)
            or not final_text.strip()
        ):
            raise _fail(
                "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
                "final transcript segment identity, timing, speaker, or text is invalid",
                segmentIndex=index,
            )
        language = segment.get("language")
        try:
            language = normalize_language_tag(language, allow_auto=False)
        except ValueError as exc:
            raise _fail(
                "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
                "final transcript segment language is invalid",
                segmentId=segment_id,
            ) from exc
        revisions = segment.get("revisions", [])
        if not isinstance(revisions, list):
            raise _fail(
                "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
                "final transcript segment revisions must be an array",
                segmentId=segment_id,
            )
        overlapping = segment.get("overlapping", False)
        human_locked = segment.get("humanLocked", False)
        if not isinstance(overlapping, bool) or not isinstance(human_locked, bool):
            raise _fail(
                "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
                "final transcript segment flags must be booleans",
                segmentId=segment_id,
            )
        seen_ids.add(segment_id)
        previous_start = start_ms
        output.append(
            {
                "id": segment_id,
                "startMs": start_ms,
                "endMs": end_ms,
                "speakerId": speaker_id,
                "language": language,
                "finalText": final_text.strip(),
                "rawTextSha256": hashlib.sha256(
                    raw_text.encode("utf-8")
                ).hexdigest(),
                "overlapping": overlapping,
                "humanLocked": human_locked,
                "revisionCount": len(revisions),
            }
        )
    return output


def build_final_adjudicated_transcript(
    document: Mapping[str, Any],
    review_queue: Mapping[str, Any],
    semantic_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the only transcript representation eligible for final scoring."""

    for value, label in (
        (document, "transcript"),
        (review_queue, "review queue"),
        (semantic_artifact, "semantic artifact"),
    ):
        try:
            validate_strict_json(dict(value))
        except ValueError as exc:
            raise _fail(
                "FINAL_ADJUDICATION_INPUT_INVALID",
                f"{label} must contain strict finite JSON",
                reason=str(exc),
            ) from exc
    job_id = document.get("jobId")
    document_id = document.get("documentId")
    if (
        document.get("schemaVersion") != "2.0.0"
        or not isinstance(job_id, str)
        or not job_id
        or not isinstance(document_id, str)
        or not document_id
        or review_queue.get("jobId") != job_id
    ):
        raise _fail(
            "FINAL_ADJUDICATION_INPUT_INVALID",
            "transcript and review queue identity is invalid",
        )
    accepted, rejected, decision_count = _resolved_review_counts(review_queue)
    semantic, semantic_input_hash = _semantic_input(
        semantic_artifact,
        job_id=job_id,
    )

    source = document.get("source")
    policy = document.get("speakerPolicy")
    if not isinstance(source, Mapping) or not isinstance(policy, Mapping):
        raise _fail(
            "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
            "transcript source and speaker policy are required",
        )
    source_sha256 = source.get("sha256")
    duration_ms = source.get("durationMs")
    speaker_ids = policy.get("speakerIds")
    resolved_count = policy.get("resolvedCount")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms < 1
        or not isinstance(speaker_ids, list)
        or any(not isinstance(item, str) or not item for item in speaker_ids)
        or isinstance(resolved_count, bool)
        or not isinstance(resolved_count, int)
        or resolved_count != len(speaker_ids)
        or resolved_count < 1
    ):
        raise _fail(
            "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
            "transcript source or canonical speaker namespace is invalid",
        )
    segments = _final_segments(
        document,
        speaker_ids=speaker_ids,
        duration_ms=duration_ms,
    )
    semantic_hash = canonical_json_sha256(semantic)
    transcript_hash = canonical_json_sha256(document)
    review_hash = canonical_json_sha256(review_queue)
    artifact = {
        "schemaVersion": FINAL_ADJUDICATED_TRANSCRIPT_SCHEMA_VERSION,
        "artifactType": FINAL_ADJUDICATED_TRANSCRIPT_ARTIFACT_TYPE,
        "artifactId": f"final-{document_id}",
        "jobId": job_id,
        "documentId": document_id,
        "generatedAt": utc_now(),
        "status": "adjudication-complete",
        "disposition": "transcribable-speech",
        "acceptanceSubject": "speaker-language-time-final-text",
        "input": {
            "sourceMediaSha256": source_sha256,
            "transcriptDocumentSha256": transcript_hash,
            "semanticInputTranscriptSha256": semantic_input_hash,
            "semanticArtifactSha256": semantic_hash,
            "reviewQueueSha256": review_hash,
        },
        "semantic": {
            "status": semantic["status"],
            "model": semantic["model"],
            "promptVersion": semantic["promptVersion"],
            "applicationPolicy": semantic["applicationPolicy"],
            "requiresHumanApproval": semantic["requiresHumanApproval"],
            "suggestionCount": semantic["metrics"]["suggestionCount"],
            "autoAppliedCount": semantic["metrics"]["autoAppliedCount"],
        },
        "review": {
            "openCount": 0,
            "itemCount": accepted + rejected,
            "acceptedCount": accepted,
            "rejectedCount": rejected,
            "decisionCount": decision_count,
        },
        "source": {
            "durationMs": duration_ms,
        },
        "speakerPolicy": {
            "resolvedCount": resolved_count,
            "speakerIds": list(speaker_ids),
        },
        "finalTextAuthority": "normalizedText",
        "segments": segments,
    }
    return validate_final_adjudicated_transcript(
        artifact,
        expected_document=document,
        expected_review_queue=review_queue,
        expected_semantic_artifact=semantic,
    )


def validate_final_adjudicated_transcript(
    artifact: Mapping[str, Any],
    *,
    expected_document: Mapping[str, Any],
    expected_review_queue: Mapping[str, Any],
    expected_semantic_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the final scoring subject and every source binding."""

    value = dict(artifact)
    try:
        validate_strict_json(value)
    except ValueError as exc:
        raise _fail(
            "FINAL_ADJUDICATION_ARTIFACT_INVALID",
            "final adjudication artifact must contain strict finite JSON",
            reason=str(exc),
        ) from exc
    required = {
        "schemaVersion",
        "artifactType",
        "artifactId",
        "jobId",
        "documentId",
        "generatedAt",
        "status",
        "disposition",
        "acceptanceSubject",
        "input",
        "semantic",
        "review",
        "source",
        "speakerPolicy",
        "finalTextAuthority",
        "segments",
    }
    if set(value) != required:
        raise _fail(
            "FINAL_ADJUDICATION_ARTIFACT_INVALID",
            "final adjudication fields do not match schema 1.1.0",
        )
    expected_job_id = expected_document.get("jobId")
    expected_document_id = expected_document.get("documentId")
    if (
        value.get("schemaVersion")
        != FINAL_ADJUDICATED_TRANSCRIPT_SCHEMA_VERSION
        or value.get("artifactType")
        != FINAL_ADJUDICATED_TRANSCRIPT_ARTIFACT_TYPE
        or value.get("artifactId") != f"final-{expected_document_id}"
        or value.get("jobId") != expected_job_id
        or value.get("documentId") != expected_document_id
        or not isinstance(value.get("generatedAt"), str)
        or not value["generatedAt"]
        or value.get("status") != "adjudication-complete"
        or value.get("disposition") != "transcribable-speech"
        or value.get("acceptanceSubject")
        != "speaker-language-time-final-text"
        or value.get("finalTextAuthority") != "normalizedText"
    ):
        raise _fail(
            "FINAL_ADJUDICATION_ARTIFACT_INVALID",
            "final adjudication identity or authority is invalid",
        )
    expected_semantic, semantic_input_hash = _semantic_input(
        expected_semantic_artifact,
        job_id=str(expected_job_id or ""),
    )
    accepted, rejected, decisions = _resolved_review_counts(
        expected_review_queue
    )
    expected_input = {
        "sourceMediaSha256": expected_document.get("source", {}).get("sha256"),
        "transcriptDocumentSha256": canonical_json_sha256(expected_document),
        "semanticInputTranscriptSha256": semantic_input_hash,
        "semanticArtifactSha256": canonical_json_sha256(expected_semantic),
        "reviewQueueSha256": canonical_json_sha256(expected_review_queue),
    }
    if value.get("input") != expected_input:
        raise _fail(
            "FINAL_ADJUDICATION_BINDING_INVALID",
            "final adjudication source hashes do not match current evidence",
        )
    expected_semantic_summary = {
        "status": expected_semantic["status"],
        "model": expected_semantic["model"],
        "promptVersion": expected_semantic["promptVersion"],
        "applicationPolicy": expected_semantic["applicationPolicy"],
        "requiresHumanApproval": expected_semantic["requiresHumanApproval"],
        "suggestionCount": expected_semantic["metrics"]["suggestionCount"],
        "autoAppliedCount": expected_semantic["metrics"]["autoAppliedCount"],
    }
    if value.get("semantic") != expected_semantic_summary:
        raise _fail(
            "FINAL_ADJUDICATION_ARTIFACT_INVALID",
            "final adjudication semantic summary is invalid",
        )
    expected_review = {
        "openCount": 0,
        "itemCount": accepted + rejected,
        "acceptedCount": accepted,
        "rejectedCount": rejected,
        "decisionCount": decisions,
    }
    if value.get("review") != expected_review:
        raise _fail(
            "FINAL_ADJUDICATION_ARTIFACT_INVALID",
            "final adjudication review summary is invalid",
        )
    source = expected_document.get("source")
    policy = expected_document.get("speakerPolicy")
    if not isinstance(source, Mapping) or not isinstance(policy, Mapping):
        raise _fail(
            "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
            "expected transcript source or speaker policy is invalid",
        )
    expected_source = {"durationMs": source.get("durationMs")}
    expected_policy = {
        "resolvedCount": policy.get("resolvedCount"),
        "speakerIds": policy.get("speakerIds"),
    }
    if (
        value.get("source") != expected_source
        or value.get("speakerPolicy") != expected_policy
    ):
        raise _fail(
            "FINAL_ADJUDICATION_ARTIFACT_INVALID",
            "final adjudication source or speaker policy is invalid",
        )
    expected_segments = _final_segments(
        expected_document,
        speaker_ids=list(expected_policy["speakerIds"]),
        duration_ms=int(expected_source["durationMs"]),
    )
    if value.get("segments") != expected_segments:
        raise _fail(
            "FINAL_ADJUDICATION_BINDING_INVALID",
            "final adjudication segments do not match the reviewed transcript",
        )
    return value


def _composition_final_segments(
    document: Mapping[str, Any],
    composition: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_segments = document.get("segments")
    composed_segments = composition.get("segments")
    if not isinstance(raw_segments, list) or not isinstance(
        composed_segments,
        list,
    ):
        raise _fail(
            "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
            "composed final transcript requires source and composed segments",
        )
    source_by_id = {
        str(segment.get("id")): segment
        for segment in raw_segments
        if isinstance(segment, Mapping)
    }
    if len(source_by_id) != len(raw_segments):
        raise _fail(
            "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
            "source transcript segments must have unique identities",
        )
    output: list[dict[str, Any]] = []
    for index, segment in enumerate(composed_segments):
        if not isinstance(segment, Mapping):
            raise _fail(
                "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
                "semantic composition segments must be objects",
                segmentIndex=index,
            )
        segment_id = str(segment.get("id") or "")
        source = source_by_id.get(segment_id)
        revisions = source.get("revisions", []) if isinstance(source, Mapping) else None
        if source is None or not isinstance(revisions, list):
            raise _fail(
                "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
                "semantic composition is rebound to another transcript segment",
                segmentId=segment_id,
            )
        output.append(
            {
                "id": segment_id,
                "startMs": segment["startMs"],
                "endMs": segment["endMs"],
                "speakerId": segment["speakerId"],
                "language": segment["language"],
                "finalText": segment["finalText"],
                "rawTextSha256": segment["rawTextSha256"],
                "overlapping": segment["overlapping"],
                "humanLocked": segment["humanLocked"],
                "revisionCount": len(revisions),
            }
        )
    if len(output) != len(raw_segments):
        raise _fail(
            "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
            "semantic composition must preserve every source segment",
        )
    return output


def _changed_selection_count(
    lattice: Mapping[str, Any],
    arbitration: Mapping[str, Any],
) -> int:
    current_by_group = {
        str(group["groupId"]): str(group["currentCandidateId"])
        for domain in lattice["domains"]
        for group in domain["groups"]
    }
    return sum(
        str(selection["selectedCandidateId"])
        != current_by_group[str(selection["groupId"])]
        for selection in arbitration["selections"]
    )


def build_final_composed_transcript(
    document: Mapping[str, Any],
    review_queue: Mapping[str, Any],
    composition_artifact: Mapping[str, Any],
    arbitration_artifact: Mapping[str, Any],
    input_lattice: Mapping[str, Any],
    *,
    generated_at: str | None = None,
    _validate_result: bool = True,
) -> dict[str, Any]:
    """Build the final scoring subject from mandatory candidate composition."""

    for value, label in (
        (document, "transcript"),
        (review_queue, "review queue"),
        (composition_artifact, "semantic composition"),
        (arbitration_artifact, "semantic arbitration"),
        (input_lattice, "semantic candidate lattice"),
    ):
        try:
            validate_strict_json(dict(value))
        except ValueError as exc:
            raise _fail(
                "FINAL_ADJUDICATION_INPUT_INVALID",
                f"{label} must contain strict finite JSON",
                reason=str(exc),
            ) from exc
    job_id = document.get("jobId")
    document_id = document.get("documentId")
    if (
        document.get("schemaVersion") != "2.0.0"
        or not isinstance(job_id, str)
        or not job_id
        or not isinstance(document_id, str)
        or not document_id
        or review_queue.get("jobId") != job_id
    ):
        raise _fail(
            "FINAL_ADJUDICATION_INPUT_INVALID",
            "transcript and review queue identity is invalid",
        )
    accepted, rejected, decision_count = _resolved_review_counts(review_queue)
    transcript_hash = canonical_json_sha256(document)
    lattice = validate_semantic_candidate_lattice(
        input_lattice,
        expected_source_media_sha256=document.get("source", {}).get("sha256"),
        expected_transcript_sha256=transcript_hash,
    )
    arbitration = validate_semantic_job_arbitration(
        arbitration_artifact,
        expected_job_id=job_id,
        expected_lattice=lattice,
    )
    composition = validate_semantic_composition(
        composition_artifact,
        expected_document=document,
        expected_lattice=lattice,
        expected_arbitration=arbitration,
    )
    if (
        arbitration["status"] != "ready-to-compose"
        or composition["status"] != "composition-complete"
        or composition["disposition"] != "transcribable-speech"
    ):
        raise _fail(
            "FINAL_ADJUDICATION_SEMANTIC_INCOMPLETE",
            "final composed transcript requires completed speech composition",
        )
    source = document.get("source")
    if not isinstance(source, Mapping):
        raise _fail(
            "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
            "transcript source is required",
        )
    source_sha256 = source.get("sha256")
    duration_ms = source.get("durationMs")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms < 1
    ):
        raise _fail(
            "FINAL_ADJUDICATION_TRANSCRIPT_INVALID",
            "transcript source identity or duration is invalid",
        )
    segments = _composition_final_segments(document, composition)
    generated = generated_at or utc_now()
    if not isinstance(generated, str) or not generated or len(generated) > 64:
        raise _fail(
            "FINAL_ADJUDICATION_INPUT_INVALID",
            "final composed generatedAt is invalid",
        )
    artifact = {
        "schemaVersion": FINAL_COMPOSED_TRANSCRIPT_SCHEMA_VERSION,
        "artifactType": FINAL_ADJUDICATED_TRANSCRIPT_ARTIFACT_TYPE,
        "artifactId": f"final-composed-{document_id}",
        "jobId": job_id,
        "documentId": document_id,
        "generatedAt": generated,
        "status": "adjudication-complete",
        "disposition": "transcribable-speech",
        "acceptanceSubject": "speech-speaker-timeline-language-final-text",
        "input": {
            "sourceMediaSha256": source_sha256,
            "transcriptDocumentSha256": transcript_hash,
            "candidateLatticeSha256": lattice["latticeSha256"],
            "arbitrationArtifactSha256": canonical_json_sha256(arbitration),
            "compositionArtifactSha256": canonical_json_sha256(composition),
            "reviewQueueSha256": canonical_json_sha256(review_queue),
        },
        "semantic": {
            "status": "composition-complete",
            "model": arbitration["model"],
            "promptVersion": arbitration["promptVersion"],
            "applicationPolicy": "mandatory-candidate-selection",
            "requiresHumanApproval": False,
            "selectedCandidateCount": len(arbitration["selections"]),
            "changedCandidateCount": _changed_selection_count(
                lattice,
                arbitration,
            ),
        },
        "review": {
            "openCount": 0,
            "itemCount": accepted + rejected,
            "acceptedCount": accepted,
            "rejectedCount": rejected,
            "decisionCount": decision_count,
        },
        "source": {
            "durationMs": duration_ms,
        },
        "speakerPolicy": dict(composition["speakerPolicy"]),
        "timeline": dict(composition["timeline"]),
        "finalTextAuthority": "semantic-composition-selected-asr-text",
        "timelineAuthority": "semantic-composition-selected-timeline",
        "segments": segments,
    }
    if _validate_result:
        return validate_final_composed_transcript(
            artifact,
            expected_document=document,
            expected_review_queue=review_queue,
            expected_composition_artifact=composition,
            expected_arbitration_artifact=arbitration,
            expected_input_lattice=lattice,
        )
    return artifact


def validate_final_composed_transcript(
    artifact: Mapping[str, Any],
    *,
    expected_document: Mapping[str, Any],
    expected_review_queue: Mapping[str, Any],
    expected_composition_artifact: Mapping[str, Any],
    expected_arbitration_artifact: Mapping[str, Any],
    expected_input_lattice: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild and compare the composition-authoritative final transcript."""

    value = dict(artifact)
    try:
        validate_strict_json(value)
    except ValueError as exc:
        raise _fail(
            "FINAL_ADJUDICATION_ARTIFACT_INVALID",
            "composed final adjudication must contain strict finite JSON",
            reason=str(exc),
        ) from exc
    required = {
        "schemaVersion",
        "artifactType",
        "artifactId",
        "jobId",
        "documentId",
        "generatedAt",
        "status",
        "disposition",
        "acceptanceSubject",
        "input",
        "semantic",
        "review",
        "source",
        "speakerPolicy",
        "timeline",
        "finalTextAuthority",
        "timelineAuthority",
        "segments",
    }
    if set(value) != required:
        raise _fail(
            "FINAL_ADJUDICATION_ARTIFACT_INVALID",
            "composed final adjudication fields do not match schema 1.2.0",
        )
    rebuilt = build_final_composed_transcript(
        expected_document,
        expected_review_queue,
        expected_composition_artifact,
        expected_arbitration_artifact,
        expected_input_lattice,
        generated_at=value.get("generatedAt"),
        _validate_result=False,
    )
    if value != rebuilt:
        raise _fail(
            "FINAL_ADJUDICATION_BINDING_INVALID",
            "composed final adjudication does not match current evidence",
        )
    return value


def _no_speech_voice_summary(
    voice_activity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "classification": voice_activity["classification"],
        "hasSpeechCandidates": voice_activity["hasSpeechCandidates"],
        "hasTranscribableSpeech": voice_activity[
            "hasTranscribableSpeech"
        ],
        "speechWindowCount": voice_activity["speechWindowCount"],
        "speechDurationMs": voice_activity["speechDurationMs"],
    }


def build_final_no_speech_adjudication(
    voice_activity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the final scoring subject for verified speech absence."""

    try:
        validate_strict_json(dict(voice_activity))
    except ValueError as exc:
        raise _fail(
            "FINAL_ADJUDICATION_INPUT_INVALID",
            "voice activity must contain strict finite JSON",
            reason=str(exc),
        ) from exc
    validated_voice = validate_voice_activity(voice_activity)
    if validated_voice["hasTranscribableSpeech"] is not False:
        raise _fail(
            "FINAL_ADJUDICATION_VOICE_ACTIVITY_INVALID",
            "no-speech adjudication requires a non-transcribable disposition",
            classification=validated_voice["classification"],
        )
    job_id = validated_voice["jobId"]
    artifact = {
        "schemaVersion": FINAL_ADJUDICATED_TRANSCRIPT_SCHEMA_VERSION,
        "artifactType": FINAL_ADJUDICATED_TRANSCRIPT_ARTIFACT_TYPE,
        "artifactId": f"final-no-speech-{job_id}",
        "jobId": job_id,
        "generatedAt": utc_now(),
        "status": "adjudication-complete",
        "disposition": "no-transcribable-speech",
        "acceptanceSubject": "lexical-speech-presence",
        "input": {
            "sourceMediaSha256": validated_voice["sourceSha256"],
            "voiceActivitySha256": canonical_json_sha256(validated_voice),
        },
        "source": {
            "durationMs": validated_voice["mediaDurationMs"],
        },
        "voiceActivity": _no_speech_voice_summary(validated_voice),
        "segments": [],
    }
    return validate_final_no_speech_adjudication(
        artifact,
        expected_voice_activity=validated_voice,
    )


def validate_final_no_speech_adjudication(
    artifact: Mapping[str, Any],
    *,
    expected_voice_activity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a no-speech final subject and its voice-evidence binding."""

    value = dict(artifact)
    try:
        validate_strict_json(value)
        validate_strict_json(dict(expected_voice_activity))
    except ValueError as exc:
        raise _fail(
            "FINAL_ADJUDICATION_ARTIFACT_INVALID",
            "no-speech final adjudication must contain strict finite JSON",
            reason=str(exc),
        ) from exc
    voice = validate_voice_activity(expected_voice_activity)
    if voice["hasTranscribableSpeech"] is not False:
        raise _fail(
            "FINAL_ADJUDICATION_VOICE_ACTIVITY_INVALID",
            "expected voice activity does not prove speech absence",
            classification=voice["classification"],
        )
    required = {
        "schemaVersion",
        "artifactType",
        "artifactId",
        "jobId",
        "generatedAt",
        "status",
        "disposition",
        "acceptanceSubject",
        "input",
        "source",
        "voiceActivity",
        "segments",
    }
    if set(value) != required:
        raise _fail(
            "FINAL_ADJUDICATION_ARTIFACT_INVALID",
            "no-speech final adjudication fields do not match schema 1.1.0",
        )
    job_id = voice["jobId"]
    if (
        value.get("schemaVersion")
        != FINAL_ADJUDICATED_TRANSCRIPT_SCHEMA_VERSION
        or value.get("artifactType")
        != FINAL_ADJUDICATED_TRANSCRIPT_ARTIFACT_TYPE
        or value.get("artifactId") != f"final-no-speech-{job_id}"
        or value.get("jobId") != job_id
        or not isinstance(value.get("generatedAt"), str)
        or not value["generatedAt"]
        or value.get("status") != "adjudication-complete"
        or value.get("disposition") != "no-transcribable-speech"
        or value.get("acceptanceSubject")
        != "lexical-speech-presence"
        or value.get("segments") != []
    ):
        raise _fail(
            "FINAL_ADJUDICATION_ARTIFACT_INVALID",
            "no-speech final adjudication identity or disposition is invalid",
        )
    expected_input = {
        "sourceMediaSha256": voice["sourceSha256"],
        "voiceActivitySha256": canonical_json_sha256(voice),
    }
    if value.get("input") != expected_input:
        raise _fail(
            "FINAL_ADJUDICATION_BINDING_INVALID",
            "no-speech final adjudication source hashes do not match voice evidence",
        )
    if value.get("source") != {"durationMs": voice["mediaDurationMs"]}:
        raise _fail(
            "FINAL_ADJUDICATION_ARTIFACT_INVALID",
            "no-speech final adjudication source duration is invalid",
        )
    if value.get("voiceActivity") != _no_speech_voice_summary(voice):
        raise _fail(
            "FINAL_ADJUDICATION_BINDING_INVALID",
            "no-speech final adjudication summary does not match voice evidence",
        )
    return value


__all__ = [
    "FINAL_ADJUDICATED_TRANSCRIPT_ARTIFACT_TYPE",
    "FINAL_ADJUDICATED_TRANSCRIPT_SCHEMA_VERSION",
    "FINAL_COMPOSED_TRANSCRIPT_SCHEMA_VERSION",
    "build_final_composed_transcript",
    "build_final_no_speech_adjudication",
    "build_final_adjudicated_transcript",
    "validate_final_composed_transcript",
    "validate_final_no_speech_adjudication",
    "validate_final_adjudicated_transcript",
]
